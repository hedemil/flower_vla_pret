import os
import gc
import logging
import copy
from collections import OrderedDict
import traceback

import hydra
import torch
import wandb
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from omegaconf import DictConfig
from flower_vla.agents.ddp_wrapper import Mode
from flower_vla.agents.utils.ema import ExponentialMovingAverage
from flower_vla.agents.utils.optimizer_hook import wandb_optimizer_hook
from accelerate import load_checkpoint_and_dispatch, Accelerator
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List

from flower_vla.agents.utils.dataset_metrics import DatasetMetricsTracker
from flower_vla.dataset.utils.dataset_index import DATASET_INDEX_MAPPING
from flower_vla.utils.model_loading import load_model_weights, load_file
from diffusers.training_utils import EMAModel
# from accelerate.utils import EMA

log = logging.getLogger(__name__)


class AccelerateTrainer:
    def __init__(
        self,
        agent: DictConfig,
        accelerator: Accelerator,
        datamodule,
        single_loader: bool,
        target_modality: str,
        obs_modalities: str,
        goal_modalities: str,
        img_modalities: list,
        lang_modalities: list,
        max_train_steps: int,
        max_eval_steps: int,
        eval_every_n_steps: int,
        use_ema: bool,
        dit_lr_scheduler: DictConfig,
        vlm_lr_scheduler: DictConfig,
        weight_decay: float,
        decay: float,
        rampup_ratio: float,
        update_ema_every_n_steps: int,
        save_every_n_steps: int,
        batch_size: int,
        learning_rate_dit: float,
        learning_rate_vlm: float,
        single_optimizer: bool = False,  # New parameter
        beta_dit_1: float = 0.9,
        beta_dit_2: float = 0.95,
        beta_vlm_1: float = 0.9,
        beta_vlm_2: float = 0.999,
        use_torch_compile: bool = False,
        use_lr_scheduler: bool = True,
        early_stopping_patience: int = 0
    ):
        # Store configuration
        self.max_train_steps = int(max_train_steps)
        self.max_eval_steps = int(max_eval_steps)
        self.eval_every_n_steps = eval_every_n_steps
        self.use_ema = use_ema
        self.weight_decay = weight_decay
        self.decay = decay
        self.rampup_ratio = rampup_ratio
        self.use_torch_compile = use_torch_compile
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.update_ema_every_n_steps = update_ema_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        self.batch_size = batch_size
        self.use_simpler_env = False
        self.single_optimizer = single_optimizer
        self.use_lr_scheduler = use_lr_scheduler
        self.agent_config = agent
        self.previous_stats = {}
        self.setup_core_components(
            accelerator, datamodule, agent, single_loader, 
            target_modality, obs_modalities, goal_modalities,
            img_modalities, lang_modalities
        )
        # Set up save/load hooks immediately after having the accelerator
        self.setup_optimization(dit_lr_scheduler, vlm_lr_scheduler, learning_rate_dit, learning_rate_vlm,
                                  betas_dit=(beta_dit_1, beta_dit_2), betas_vlm=(beta_vlm_1, beta_vlm_2))
        self.log_model_stats()
        self.metrics_tracker = DatasetMetricsTracker(DATASET_INDEX_MAPPING, self.accelerator)

    def setup_core_components(
        self, accelerator, datamodule, agent, single_loader,
        target_modality, obs_modalities, goal_modalities,
        img_modalities, lang_modalities
    ):
        """Initialize core training components"""
        self.accelerator = accelerator
        self.device = accelerator.device
        self.datamodule = datamodule
        self.target_modality = target_modality
        self.obs_modalities = obs_modalities
        self.goal_modalities = goal_modalities
        self.img_modalities = img_modalities
        self.lang_modalities = lang_modalities
        
        # Setup dataloaders
        main_process = accelerator.is_main_process if single_loader else False
        self.train_dataloader = self.datamodule.create_train_dataloader(main_process=main_process)
        self.val_loader = self.datamodule.create_val_dataloader(main_process=main_process)
        
        # Setup agent
        print('now instantiating agent')
        self.agent = hydra.utils.instantiate(
            agent, 
            device=self.device,
            process_id=self.accelerator.process_index,
            accelerator=accelerator
        ).to(self.accelerator.device)
        self.setup_ema()

        # move agent to bfloat16 
        self.agent = self.agent.to(torch.bfloat16)

        self.global_step = 0
        self.working_dir = os.getcwd()

    def setup_optimization(self, dit_lr_scheduler: DictConfig, vlm_lr_scheduler: DictConfig, 
                        learning_rate_dit: float, learning_rate_vlm: float,
                        betas_dit: tuple = (0.9, 0.95), betas_vlm: tuple = (0.9, 0.999)):
        if self.single_optimizer:
            # Combine all parameters into a single optimizer
            dit_optim_groups = self.agent.get_optim_groups(self.weight_decay)
            vlm_params = [p for p in self.agent.agent.vlm.parameters() if p.requires_grad]
            
            # Add VLM parameters to the appropriate group based on weight decay
            for param in vlm_params:
                if any(nd in name.lower() for nd in ['bias', 'layernorm', 'ln', 'norm'] 
                        for name, _ in self.agent.agent.vlm.named_parameters()):
                    dit_optim_groups[1]['params'].append(param)  # no decay group
                else:
                    dit_optim_groups[0]['params'].append(param)  # decay group
            
            # Single optimizer for all parameters
            self.optimizer = torch.optim.AdamW(
                dit_optim_groups,
                lr=learning_rate_dit,
                betas=betas_dit
            )
            
            # Use DiT scheduler for everything
            if self.use_lr_scheduler:
                self.lr_scheduler = hydra.utils.instantiate(
                    dit_lr_scheduler,
                    optimizer=self.optimizer
                )
            else:
                self.lr_scheduler = None
            
            if self.accelerator.is_main_process:
                self.optimizer.register_step_pre_hook(wandb_optimizer_hook)
            
            # Prepare with accelerator
            if self.val_loader is not None:
                if self.use_ema and self.ema is not None:
                    (
                        self.agent, self.optimizer, self.lr_scheduler,
                        self.train_dataloader, self.val_loader, self.ema
                    ) = self.accelerator.prepare(
                        self.agent, self.optimizer, self.lr_scheduler,
                        self.train_dataloader, self.val_loader, self.ema
                    )
                else:
                    (
                        self.agent, self.optimizer, self.lr_scheduler,
                        self.train_dataloader, self.val_loader
                    ) = self.accelerator.prepare(
                        self.agent, self.optimizer, self.lr_scheduler,
                        self.train_dataloader, self.val_loader
                    )
            else:
                if self.use_ema and self.ema is not None:
                    (
                        self.agent, self.optimizer, self.lr_scheduler,
                        self.train_dataloader, self.ema
                    ) = self.accelerator.prepare(
                        self.agent, self.optimizer, self.lr_scheduler,
                        self.train_dataloader, self.ema
                    )
                else:
                    (
                        self.agent, self.optimizer, self.lr_scheduler,
                        self.train_dataloader
                    ) = self.accelerator.prepare(
                        self.agent, self.optimizer, self.lr_scheduler,
                        self.train_dataloader
                    )
                    
        else:
            # Original two-optimizer setup
            dit_optim_groups = self.agent.get_optim_groups(self.weight_decay)
            self.dit_optimizer = torch.optim.AdamW(
                dit_optim_groups, 
                lr=learning_rate_dit,
                betas=betas_dit
            )
            if self.use_lr_scheduler:
                self.dit_lr_scheduler = hydra.utils.instantiate(
                    dit_lr_scheduler,
                    optimizer=self.dit_optimizer
                )
            else:
                self.dit_lr_scheduler = None

            vlm_params = [p for p in self.agent.agent.vlm.parameters() if p.requires_grad]
            self.vlm_optimizer = torch.optim.AdamW(
                vlm_params,
                lr=learning_rate_vlm, 
                betas=betas_vlm
            )
            if self.use_lr_scheduler:
                self.vlm_lr_scheduler = hydra.utils.instantiate(
                    vlm_lr_scheduler, 
                    optimizer=self.vlm_optimizer
                )
            else:
                self.vlm_lr_scheduler = None

            if self.accelerator.is_main_process:
                self.dit_optimizer.register_step_pre_hook(wandb_optimizer_hook)
                self.vlm_optimizer.register_step_pre_hook(wandb_optimizer_hook)
            
            print('### Optimizer parameter count ###')
            print('Total optimizer parameters:', sum(p.numel() for p in self.agent.parameters()))
            print('DIT optimizer parameters:', sum(p.numel() for p in dit_optim_groups[0]['params']))
            print('DIT optimizer parameters for no decay:', sum(p.numel() for p in dit_optim_groups[1]['params']))
            print('VLM optimizer parameters:', sum(p.numel() for p in vlm_params))
            print('### Optimizer parameter count ###')
            
            # Prepare with accelerator
            if self.val_loader is not None:
                if self.use_ema and self.ema is not None:
                    (
                        self.agent, self.dit_optimizer, self.vlm_optimizer,
                        self.dit_lr_scheduler, self.vlm_lr_scheduler,
                        self.train_dataloader, self.val_loader, self.ema
                    ) = self.accelerator.prepare(
                        self.agent, self.dit_optimizer, self.vlm_optimizer,
                        self.dit_lr_scheduler, self.vlm_lr_scheduler,
                        self.train_dataloader, self.val_loader, self.ema
                    )
                else:
                    (
                        self.agent, self.dit_optimizer, self.vlm_optimizer,
                        self.dit_lr_scheduler, self.vlm_lr_scheduler,
                        self.train_dataloader, self.val_loader
                    ) = self.accelerator.prepare(
                        self.agent, self.dit_optimizer, self.vlm_optimizer,
                        self.dit_lr_scheduler, self.vlm_lr_scheduler,
                        self.train_dataloader, self.val_loader
                    )
            else:
                if self.use_ema and self.ema is not None:
                    (
                        self.agent, self.dit_optimizer, self.vlm_optimizer,
                        self.dit_lr_scheduler, self.vlm_lr_scheduler,
                        self.train_dataloader, self.ema
                    ) = self.accelerator.prepare(
                        self.agent, self.dit_optimizer, self.vlm_optimizer,
                        self.dit_lr_scheduler, self.vlm_lr_scheduler,
                        self.train_dataloader, self.ema
                    )
                else:
                    (
                        self.agent, self.dit_optimizer, self.vlm_optimizer,
                        self.dit_lr_scheduler, self.vlm_lr_scheduler,
                        self.train_dataloader
                    ) = self.accelerator.prepare(
                        self.agent, self.dit_optimizer, self.vlm_optimizer,
                        self.dit_lr_scheduler, self.vlm_lr_scheduler,
                        self.train_dataloader
                    )

    def setup_ema(self):
        if self.use_ema:
            # Use the simple copy approach from code 1
            self.ema = copy.deepcopy(self.agent).to(self.device)
            for param in self.ema.parameters():
                param.requires_grad = False
        else:
            self.ema = None

    def train_agent(self):
        """Main training loop with improved gradient and loss handling"""
        # load EMA if chosen
        if self.use_ema and self.ema is not None:
            update_ema(self.ema, self.agent.module if hasattr(self.agent, "module") else self.agent, decay=0)
            self.ema.eval()

        best_test_mse = float('inf')
        train_generator = iter(self.train_dataloader)
        val_generator = iter(self.val_loader) if self.val_loader is not None else None
        avg_test_mse = 0
        try:
            print('starting training')
            for step in tqdm(range(self.max_train_steps), 
                        initial=self.global_step,
                        disable=(not self.accelerator.is_main_process)):

                # Checkpoint saving
                if step != 0 and step % self.save_every_n_steps == 0:
                    self.accelerator.wait_for_everyone()
                    self.store_model_weights(self.working_dir, f"{self.global_step}_")
                    if self.accelerator.is_main_process:
                        log.info(f'{self.global_step} Steps reached. Stored weights updated!')

                if step % 5000 == 0 and hasattr(self.train_dataloader.dataset, 'tracker') and getattr(self.train_dataloader.dataset.tracker, 'enable_tracking', False):
                    self.train_dataloader.dataset.tracker.replace_overused_samples()

                # Evaluation
                if step != 0 and step % self.eval_every_n_steps == 0:
                    avg_test_mse, best_test_mse = self.evaluate(
                        val_generator if val_generator is not None else train_generator,
                        best_test_mse
                    )
                    self.accelerator.wait_for_everyone()

                    # Early stopping check
                    if self.early_stopping_patience > 0 and self.patience_counter >= self.early_stopping_patience:
                        if self.accelerator.is_main_process:
                            log.info(f"Early stopping at step {self.global_step}: no improvement for {self.patience_counter} evals")
                        break

                # Training step
                try:
                    batch = next(train_generator)
                except StopIteration:
                    train_generator = iter(self.train_dataloader)
                    batch = next(train_generator)

                batch_loss = self.train_step(batch)

                # Logging
                if step % 1000 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                if step % 1000 == 0 and self.accelerator.is_main_process:
                    losses_info = self.agent.module.agent.losses_dict
                    raw_mse = losses_info.get("raw_mse", float("nan"))
                    cos_v_vtgt = losses_info.get("cos_v_vtgt", float("nan"))
                    vtheta_norm = losses_info.get("vtheta_norm", float("nan"))
                    log.info(f"Step {self.global_step}: raw_mse={raw_mse:.6f}, cos_v_vtgt={cos_v_vtgt:.4f}, vtheta_norm={vtheta_norm:.4f}")

                if step % 100 == 0 and wandb.run is not None and self.accelerator.is_main_process:
                    # Get metrics from agent
                    losses_dict = self.agent.module.agent.losses_dict
                    # Create metrics dict with base metrics
                    metrics = {
                        **losses_dict,
                        "loss": batch_loss,
                        "test_loss": avg_test_mse,
                    }

                    # Add tracker metrics if enabled
                    if hasattr(self.train_dataloader.dataset, 'tracker') and getattr(self.train_dataloader.dataset.tracker, 'enable_tracking', False):
                        metrics["dataset_coverage"] = self.train_dataloader.dataset.tracker.get_coverage_stats()
                    
                    # Add learning rate metrics based on optimizer configuration
                    if self.single_optimizer:
                        if self.use_lr_scheduler and self.lr_scheduler is not None:
                            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                        else:
                            metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                    else:
                        if self.use_lr_scheduler and self.dit_lr_scheduler is not None and self.vlm_lr_scheduler is not None:
                            metrics.update({
                                "dit_learning_rate": self.dit_lr_scheduler.get_last_lr()[0],
                                "vlm_learning_rate": self.vlm_lr_scheduler.get_last_lr()[0]
                            })
                        else:
                            metrics["dit_learning_rate"] = self.dit_optimizer.param_groups[0]["lr"]
                            metrics["vlm_learning_rate"] = self.vlm_optimizer.param_groups[0]["lr"]
                    
                    
                    # Log to wandb
                    self.accelerator.log(metrics)

        except Exception as e:
            log.error(f"Training interrupted: {str(e)}")
            self.store_model_weights(self.working_dir, f"{self.global_step}_")
            raise
        finally:
            self.store_model_weights(self.working_dir, f"{self.global_step}_")
            log.info("Training completed!")

    def train_step(self, batch: dict) -> float:
        """Single training step with EMA update on unwrapped parameters."""
        with self.accelerator.accumulate(self.agent):
            self.agent.train()

            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = self.agent(batch, Mode.TRAINING)
                if isinstance(loss, dict):
                    loss = loss['loss']
                elif not isinstance(loss, torch.Tensor):
                    raise ValueError(f"Expected tensor loss, got {type(loss)}")

                if torch.isnan(loss) or torch.isinf(loss):
                    log.error(f"NaN/Inf in loss at step {self.global_step}")
                    self.store_model_weights(self.working_dir, f"nan_error_{self.global_step}_")
                    raise RuntimeError("Training stopped due to NaN/Inf loss")

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.agent.module.agent.dit.parameters(), 1.0)
                self.accelerator.clip_grad_norm_(self.agent.module.agent.vlm.parameters(), 1.0)

            if self.single_optimizer:
                self.optimizer.step()
                # Update EMA only on sync_gradients steps
                if self.use_ema and self.ema is not None and self.accelerator.sync_gradients:
                    update_ema(self.ema, self.agent.module if hasattr(self.agent, "module") else self.agent, decay=self.decay)

                if self.use_lr_scheduler and self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.dit_optimizer.step()
                # Update EMA only on sync_gradients steps
                if self.use_ema and self.ema is not None and self.accelerator.sync_gradients:
                    update_ema(self.ema, self.agent.module if hasattr(self.agent, "module") else self.agent, decay=self.decay)

                if self.use_lr_scheduler and self.dit_lr_scheduler is not None:
                    self.dit_lr_scheduler.step()
                self.vlm_optimizer.step()
                if self.use_lr_scheduler and self.vlm_lr_scheduler is not None:
                    self.vlm_lr_scheduler.step()

                self.dit_optimizer.zero_grad(set_to_none=True)
                self.vlm_optimizer.zero_grad(set_to_none=True)

        self.global_step += 1
        return loss.item()

    def evaluate_step(self, batch: dict):
        # Store original model for later
        original_model = None

        if self.use_ema and self.ema is not None:
            # Use EMA weights for evaluation
            original_model = self.agent
            self.agent = self.ema.to(torch.bfloat16)
        
        self.agent.eval()
        
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = self.agent(batch, Mode.EVALUATION)
        
        # Switch back to original model if we used EMA
        if original_model is not None:
            self.agent = original_model
        
        return output

    def _should_save_checkpoint(self, step: int) -> bool:
        return step != 0 and step % self.save_every_n_steps == 0

    def _should_evaluate(self, step: int) -> bool:
        return step != 0 and step % self.eval_every_n_steps == 0

    def _get_next_batch(self, generator) -> Dict[str, Any]:
        """Get next batch with generator reset handling"""
        try:
            return next(generator)
        except StopIteration:
            return next(iter(self.train_dataloader))

    def _save_checkpoint(self):
        """Save model checkpoint"""
        self.accelerator.wait_for_everyone()
        self.store_model_weights(self.working_dir, str(self.global_step) + "_")
        if self.accelerator.is_main_process:
            logging.info(f'{self.global_step} Steps reached. Stored weights have been updated!')

    def log_model_stats(self):
        """Log model statistics"""
        total_params = sum(p.numel() for p in self.agent.parameters())
        trainable_params = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

    def store_model_weights(self, store_path: str, additional_name: str = ""):
        """
        Saves model weights and EMA weights separately.
        """
        try:
            # Ensure the directory exists
            os.makedirs(store_path, exist_ok=True)
            
            # Synchronize all processes
            self.accelerator.wait_for_everyone()
            
            # Only save from main process to avoid conflicts
            if self.accelerator.is_main_process:
                # Get unwrapped model
                model = self.accelerator.unwrap_model(self.agent)
                
                # Save default model weights to CPU
                default_weights_path = os.path.join(
                    store_path, f"{self.global_step}_default_weights.pt"
                )
                model_state = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
                torch.save(model_state, default_weights_path)
                
                # Save EMA weights if available
                if self.use_ema and self.ema is not None:
                    ema_weights_path = os.path.join(
                        store_path, f"{self.global_step}_ema_weights.pt"
                    )
                    ema_model = self.accelerator.unwrap_model(self.ema)
                    ema_state = {
                        k: v.detach().cpu() for k, v in ema_model.state_dict().items()
                    }
                    torch.save(ema_state, ema_weights_path)
            
            # Synchronize again
            self.accelerator.wait_for_everyone()
            
            # Let Accelerate handle full checkpoint
            full_state_path = os.path.join(store_path, f"checkpoint_{self.global_step}")
            self.accelerator.save_state(full_state_path)
                
        except Exception as e:
            log.error(f"Error in store_model_weights: {e}")
            import traceback
            traceback.print_exc()

            
    def evaluate(self, generator, best_test_mse):
        test_losses = []
        
        for step in range(self.max_eval_steps):
            try:
                batch = next(generator)
            except StopIteration:
                generator = iter(self.val_loader if self.val_loader else self.train_dataloader)
                batch = next(generator)

            eval_dict = self.evaluate_step(batch)
            eval_loss = eval_dict['loss'].mean().item() if torch.is_tensor(eval_dict['loss']) else eval_dict['loss']
            test_losses.append(eval_loss)

            self.metrics_tracker.update(
                losses=eval_dict['loss'],
                dataset_indices=eval_dict['dataset_index']
            )

        # Gather and average losses
        gathered_losses = self.accelerator.gather(torch.tensor(test_losses, device=self.accelerator.device))
        avg_test_mse = float(gathered_losses.mean().item())
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            log.info(f"Step {self.global_step}: Mean test MSE is {avg_test_mse}")
            if avg_test_mse < best_test_mse:
                best_test_mse = avg_test_mse
                log.info('New best test loss!')

        # Best model saving + early stopping tracking
        if avg_test_mse < self.best_val_loss:
            self.best_val_loss = avg_test_mse
            self.patience_counter = 0
            self.store_model_weights(self.working_dir, "best_")
            if self.accelerator.is_main_process:
                log.info(f"Saved best model at step {self.global_step} with val_loss={avg_test_mse:.6f}")
        else:
            self.patience_counter += 1
            if self.accelerator.is_main_process:
                log.info(f"No improvement for {self.patience_counter} eval(s) (best={self.best_val_loss:.6f})")

        self.compute_and_log_metrics()

        # Free GPU memory after evaluation
        torch.cuda.empty_cache()

        return avg_test_mse, best_test_mse

    def load_pretrained_model(self, weights_path: str, ema_name: str = None) -> None:
        """
        Method to load a pretrained model weights with enhanced EMA support
        """
        # Load model weights
        load_checkpoint_and_dispatch(self.agent, weights_path)
        
        # Setup and load EMA if needed
        if self.use_ema and ema_name is not None:
            ema_path = os.path.join(weights_path, ema_name)
            self.load_ema_weights(ema_path)  # Use the simple loading method
            
        log.info('Loaded pre-trained agent parameters')

    def continue_training(self, weights_path: str, ema_name=None, step=0) -> None:
        """
        Method to continue training with enhanced EMA support
        """
        print('Loading state from', weights_path)
        
        # Load the main model state
        self.accelerator.load_state(weights_path)
        
        # Specifically handle EMA if needed
        if self.use_ema and ema_name is not None:
            ema_path = os.path.join(weights_path, ema_name)
            self.load_ema_weights(ema_path)  # Use the simple loading method
        
        print('State loaded')
        self.global_step = step

    # Remove or simplify the complex EMA helper methods and replace with:
    def load_ema_weights(self, ema_path: str):
        """Simple EMA weight loading"""
        if self.use_ema and self.ema is not None and os.path.exists(ema_path):
            ema_state = torch.load(ema_path, map_location='cpu')
            ema_model = self.accelerator.unwrap_model(self.ema)
            ema_model.load_state_dict(ema_state)
            log.info(f"Loaded EMA weights from {ema_path}")

    def _log_gradient_norms(self, error_mode: bool = False):
        """Helper method to log gradient norms for both VLM and DiT components with detailed breakdown
        Args:
            error_mode: If True, log with error level and more detail for debugging
        """
        log_fn = log.error if error_mode else log.info
        
        # Unwrap model from DDP wrapper
        if hasattr(self.agent, 'module'):
            vlm = self.agent.module.agent.vlm
            dit = self.agent.module.agent.dit
        else:
            vlm = self.agent.agent.vlm
            dit = self.agent.agent.dit
        
        # Calculate VLM norms by layer group
        vlm_layer_norms = {}
        vlm_total_norm = 0.
        for name, param in vlm.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2).item()
                vlm_total_norm += param_norm ** 2
                
                # Group by main layer components
                layer_name = name.split('.')[0]  # Get top-level component name
                if layer_name not in vlm_layer_norms:
                    vlm_layer_norms[layer_name] = 0
                vlm_layer_norms[layer_name] += param_norm ** 2
        
        vlm_total_norm = vlm_total_norm ** 0.5 if vlm_total_norm > 0 else 0.0
        
        # Calculate DiT norms by layer group
        dit_layer_norms = {}
        dit_total_norm = 0.
        for name, param in dit.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2).item()
                dit_total_norm += param_norm ** 2
                
                # Group by main layer components
                layer_name = name.split('.')[0]  # Get top-level component name
                if layer_name not in dit_layer_norms:
                    dit_layer_norms[layer_name] = 0
                dit_layer_norms[layer_name] += param_norm ** 2
        
        dit_total_norm = dit_total_norm ** 0.5 if dit_total_norm > 0 else 0.0
        
        # Always log detailed breakdown        
        if wandb.run is not None and self.accelerator.is_main_process:
            wandb_metrics = {
                "gradient_norms/vlm_total": vlm_total_norm,
                "gradient_norms/dit_total": dit_total_norm,
            }
            
            # Only add ratio if both norms are non-zero
            if dit_total_norm > 0 and vlm_total_norm > 0:
                wandb_metrics["gradient_norms/vlm_dit_ratio"] = vlm_total_norm/dit_total_norm
            
            # Add individual layer norms
            for layer_name, norm in vlm_layer_norms.items():
                wandb_metrics[f"gradient_norms/vlm_{layer_name}"] = norm ** 0.5
            for layer_name, norm in dit_layer_norms.items():
                wandb_metrics[f"gradient_norms/dit_{layer_name}"] = norm ** 0.5

            # add total norm for logging of the complete model
            wandb_metrics["total_norm"] = torch.norm(
                torch.cat([p.grad.flatten() for p in self.agent.parameters() if p.grad is not None])
            ).item()
            
            self.accelerator.log(wandb_metrics, # step=self.global_step+1
                                 )
    
    def compute_and_log_metrics(self):
        """
        Computes metrics and handles wandb logging in a synchronized way.
        """
        self.accelerator.wait_for_everyone()
        # Compute metrics - this will handle gathering across processes
        individual_metrics = self.metrics_tracker.compute_metrics()
        print('done computing metrics')
        # Only log on main process
        if wandb.run is not None and self.accelerator.is_main_process:
            self.accelerator.log(
                individual_metrics,
                # step=self.global_step + 1
            )
            print("Logged metrics:", individual_metrics)
        
        # Make sure all processes wait before continuing
        self.accelerator.wait_for_everyone()

    def log_norm_layer_weights(self, agent):
        """
        Log detailed statistics about various model weights and track changes
        
        Args:
            agent: The model agent
            previous_stats (dict, optional): Previously stored weight statistics
        
        Returns:
            dict: Current weight statistics for comparison
        """
        # Initialize storage for current weight statistics
        current_stats = {}
        weight_layers = []
        
        try:
            for idx, layer in enumerate(agent.module.agent.dit):
                if idx > 1:
                    continue

                # Norm layer weights
                if hasattr(layer, 'action_norms1'):
                    if getattr(agent.module.agent, 'use_shared_norm', False):
                        weight_layers.append(('shared_norm1', layer.action_norms1.weight))
                    else:
                        weight_layers.extend([
                            (f'action_norm1_{idx}_{i}', norm.weight) 
                            for i, norm in enumerate(layer.action_norms1)
                        ])
                
                if hasattr(layer, 'action_norms2'):
                    if getattr(agent.module.agent, 'use_shared_norm', False):
                        weight_layers.append(('shared_norm2', layer.action_norms2.weight))
                    else:
                        weight_layers.extend([
                            (f'action_norm2_{idx}_{i}', norm.weight) 
                            for i, norm in enumerate(layer.action_norms2)
                        ])

                if hasattr(layer, 'action_norms3'):
                    if getattr(agent.module.agent, 'use_shared_norm', False):
                        weight_layers.append(('shared_norm3', layer.action_norms3.weight))
                    else:
                        weight_layers.extend([
                            (f'action_norm3_{idx}_{i}', norm.weight) 
                            for i, norm in enumerate(layer.action_norms3)
                        ])
                
                # Add QKV weight logging
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'qkv'):
                    weight_layers.append((f'dit_layer_{idx}_self_attn_qkv', layer.self_attn.qkv.weight))
            
            # Log statistics
            for name, weight in weight_layers:
                try:
                    # Ensure we're working with the tensor, handling different possible types
                    if not isinstance(weight, torch.Tensor):
                        continue
                    
                    # Ensure tensor is detached and on CPU
                    weight_data = weight.detach().cpu()
                    
                    # Compute statistics
                    stats = {
                        'mean': weight_data.mean().item(),
                        'std': weight_data.std().item(),
                        'min': weight_data.min().item(),
                        'max': weight_data.max().item(),
                        'size': weight_data.numel()
                    }
                    current_stats[name] = stats

                    if self.accelerator.is_main_process:
                        log.info(f"{name} Weight Statistics:")
                        for stat_name, stat_value in stats.items():
                            log.info(f"  {stat_name.capitalize()}: {stat_value}")
                        
                        # Compare with previous statistics if available
                        if self.previous_stats and name in self.previous_stats:
                            prev = self.previous_stats[name]
                            log.info("  Changes:")
                            for stat_name in ['mean', 'std', 'min', 'max']:
                                change = stats[stat_name] - prev[stat_name]
                                change_pct = (change / (prev[stat_name] + 1e-8)) * 100
                                log.info(f"    {stat_name.capitalize()} change: {change:.6f} ({change_pct:.2f}%)")
                        
                        # Optional: Log to wandb if available
                        if wandb.run is not None and self.accelerator.is_main_process:
                            # Log current stats
                            wandb_logs = {
                                f"model_weights/{name}_{stat_name}": stat_value
                                for stat_name, stat_value in stats.items()
                            }
                            
                            # Log changes if previous stats exist
                            if self.previous_stats and name in self.previous_stats:
                                prev = self.previous_stats[name]
                                change_logs = {
                                    f"model_weights/{name}_{stat_name}_change": 
                                    stats[stat_name] - prev[stat_name]
                                    for stat_name in ['mean', 'std', 'min', 'max']
                                }
                                wandb_logs.update(change_logs)
                            
                            wandb.log(wandb_logs)

                            # update previous stats
                        
                        self.previous_stats[name] = stats

                except Exception as layer_err:
                    log.error(f"Error processing {name} weights:")
                    log.error(str(layer_err))
                    traceback.print_exc()

        except Exception as e:
            log.error("Error in log_model_weights:")
            log.error(str(e))
            traceback.print_exc()
        
        return current_stats

    def load_pretrained_weights(
        self,
        weights_path: str,
        strict: bool = False,
        exclude_keys: Optional[list] = None,
        map_weights: bool = False  # Adding the missing parameter
    ) -> None:
        """
        Load pretrained weights for finetuning.
        
        Args:
            weights_path: Path to weights file (.safetensors or .pt)
            strict: Whether to strictly enforce all keys match
            exclude_keys: List of key patterns to exclude from loading
            map_weights: If True, attempt to map weights even if key names don't match exactly
        """
        try:
            log.info(f"Loading pretrained weights from {weights_path}")
            
            # Load state dict based on file type
            if weights_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location='cpu')
                # Handle different state dict formats
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

            # Filter excluded keys if specified
            if exclude_keys:
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    if not any(ex in k for ex in exclude_keys):
                        filtered_state_dict[k] = v
                state_dict = filtered_state_dict
            
            

            # Get base model without DDP wrapper
            if hasattr(self.agent, 'module'):
                base_model = self.agent.module.agent
            else:
                base_model = self.agent.agent
            
            # Handle key mapping if needed
            if map_weights:
                log.info("Attempting to map weights by matching key patterns")
                new_state_dict = {}
                model_keys = dict(base_model.named_parameters())
                model_keys.update(dict(base_model.named_buffers()))
                
                # Try to match keys by removing prefixes or finding similar patterns
                for k, v in state_dict.items():
                    # Try different variations to match keys
                    clean_k = k.replace('agent.', '').replace('module.', '')
                    if clean_k in model_keys:
                        new_state_dict[clean_k] = v
                    elif 'agent.' + clean_k in model_keys:
                        new_state_dict['agent.' + clean_k] = v
                
                if len(new_state_dict) > 0:
                    log.info(f"Successfully mapped {len(new_state_dict)}/{len(state_dict)} keys")
                    state_dict = new_state_dict
                else:
                    log.warning("Could not map any keys, falling back to original state dict")

            # Load weights
            missing, unexpected = base_model.load_state_dict(state_dict, strict=strict)
            
            # Move to correct device and dtype
            base_model = base_model.to(device=self.device, dtype=torch.bfloat16)
            
            # Log results
            log.info(f"Successfully loaded pretrained weights from {weights_path}")
            if len(missing) > 0:
                log.info(f"Missing keys: {len(missing)}")
            if len(unexpected) > 0:
                log.info(f"Unexpected keys: {len(unexpected)}")
                
        except Exception as e:
            log.error(f"Failed to load pretrained weights: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise

    def load_for_finetuning(
        self,
        path: str,
        mode: str = "weights_only",  # Options: "weights_only", "full_state"
        strict: bool = False,
        exclude_keys: Optional[list] = None,
        map_weights: bool = False,
        ema_name: Optional[str] = None,
        step: int = 0
    ) -> None:
        """
        Flexible method for loading pretrained models for finetuning - either just weights or full state.
        
        Args:
            path: Path to weights file or directory containing checkpoint
            mode: Loading mode - "weights_only" or "full_state"
            strict: Whether to strictly enforce all keys match (for weights_only mode)
            exclude_keys: List of key patterns to exclude from loading (for weights_only mode)
            map_weights: If True, attempt to map weights even if key names don't match (for weights_only mode)
            ema_name: Name of EMA state file to load if available (for full_state mode)
            step: Step to start from when loading full state (for full_state mode)
        """
        log.info(f"Loading pretrained model with mode: {mode}")
        
        if mode == "weights_only":
            # Use enhanced load_pretrained_weights to load just the model weights
            self.load_pretrained_weights(
                weights_path=path,
                strict=strict,
                exclude_keys=exclude_keys,
                map_weights=map_weights
            )
            log.info("Successfully loaded pretrained weights for finetuning")
            
        elif mode == "full_state":
            self.accelerator.wait_for_everyone()
            log.info(f"Loading full accelerate state from {path}")
            
            try:
                # Load the full accelerate state
                self.accelerator.load_state(path)
                
                # Specifically handle EMA if needed
                if self.use_ema and ema_name is not None:
                    ema_path = os.path.join(path, ema_name)
                    self.load_ema_weights(ema_path)  # Use simple method
                    
                # Set global step
                self.global_step = step
                log.info(f"Full state loaded successfully, continuing from step {self.global_step}")
                
            except Exception as e:
                log.error(f"Error loading full state: {e}")
                import traceback
                log.error(traceback.format_exc())
                raise
                    
        else:
            raise ValueError(f"Unknown loading mode: {mode}. Choose 'weights_only' or 'full_state'")

@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)