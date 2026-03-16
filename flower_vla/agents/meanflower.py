import logging
import math
import os
from typing import Optional, Dict, Tuple, Union, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import wandb
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoModelForCausalLM
from timm.layers.mlp import Mlp
from torchdiffeq import odeint

# Import helper modules from your codebase.
from flower_vla.agents.utils.action_index import ActionIndex
from flower_vla.agents.networks.meanflower_transformers import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    ZeroEncoder,
    FlowBlock,
    DMFTransformer,
    stateless_norm
)
from flower_vla.dataset.utils.act_seq_mapping import DATASET_ACT_SEQ_MAP

logger = logging.getLogger(__name__)


def logvar_timestep_embedding(t, dim=128, max_period=10000):
    """Sinusoidal timestep embedding for log-variance head (no learnable params)."""
    half = dim // 2
    freqs = 1000 * torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def log_lv_loss(x, y, lv, valid_dims_count=None, eps=0.01):
    """
    Log-variance weighted Gaussian NLL loss from DMF.
    Matches official repo: https://github.com/kyungmnlee/dmf/blob/main/loss.py

    Args:
        x: prediction [B, T, D]
        y: target [B, T, D]
        lv: log-variance φ [B, 1, 1] (predicted by model)
        valid_dims_count: [B] number of valid (non-padded) elements per sample.
        eps: small constant for numerical stability.

    Returns:
        mse_loss: per-sample sum of squared errors [B]
        log_loss: per-sample log-variance weighted loss [B]
    """
    err = (x - y) ** 2
    mse_loss = torch.sum(err, dim=list(range(1, len(x.shape))))
    if valid_dims_count is not None:
        mean_loss = mse_loss / valid_dims_count.clamp(min=1)
    else:
        mean_loss = torch.mean(err, dim=list(range(1, len(x.shape))))
    # Gaussian NLL: log(exp(-φ) * mean_loss + eps) + φ
    log_loss = torch.log(torch.exp(-lv) * mean_loss + eps) + lv
    # Squeeze to [B] in case lv broadcasts extra dims
    log_loss = log_loss.view(x.shape[0])
    return mse_loss, log_loss


class MeanFlowerVLA(nn.Module):
    def __init__(
        self,
        # Core configuration
        device,
        process_id: int,
        accelerator,
        # Modality configuration 
        target_modality: str,
        obs_modalities: str,
        goal_modalities: str,
        img_modalities: List[str],
        lang_modalities: List[str],
        # Vision-Language Model configuration
        vlm_path: str = "microsoft/Florence-2-large",
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        freeze_embeddings_only: bool = False,  # New parameter
        vlm_prompt_style: str = 'default',
        token_dropout: float = 0.2,
        cfg_dropout: float = 0.1,
        # Action and observation dimensions
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,
        multistep: int = 10,
        num_sampling_steps: int = 5,
        # Model architecture flags
        use_second_view: bool = False,
        second_view_key: str = 'image_wrist',
        action_type_adaln: bool = False,
        use_causal_attention: bool = True,
        use_cross_attn: bool = True,
        use_adaln_cond: bool = False,
        use_readout_token: bool = False,
        use_proprio: bool = False,
        return_act_chunk: bool = False,
        # DiT configuration 
        sampling_type: str = 'ln',
        dit_dim: int = 512,
        n_heads: int = 16,
        n_layers: int = 12,
        # Dropout rates
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        # RoPE configuration
        use_rope: bool = False,
        use_nope: bool = False,
        query_seq_len: int = 128,
        rope_theta: float = 32.0,
        # Mean Flow configuration
        noise_dist: str = 'logit_normal',
        P_mean: float = -0.4,
        P_std: float = 1.0,
        ratio: float = 0.75,
        # DMF separate distribution parameters (paper defaults for 1-step generation)
        P_mean_t: float = 0.4,
        P_std_t: float = 1.0,
        P_mean_r: float = -1.2,
        P_std_r: float = 1.0,
    ):
        """
        Initializes the MeanFlowerVLA agent that combines a pretrained vision–language model
        with a flow-based DiT architecture for learning a generalist policy.

        Args:
            device: Target device for computations.
            process_id: Process ID (for distributed setups).
            accelerator: An accelerator instance.
            target_modality, obs_modalities, goal_modalities, img_modalities, lang_modalities:
                Modality configurations.
            vlm_path: Path or identifier for the pretrained VLM.
            freeze_florence, freeze_vision_tower: Flags for freezing parts of the VLM.
            vlm_prompt_style: Prompt style configuration.
            token_dropout: Dropout probability for VLM tokens.
            lowdim_obs_dim, action_dim: Dimensions for observations and actions.
            act_window_size, multistep, num_sampling_steps: Parameters for action sequence prediction.
            use_second_view: Whether to use an additional image view.
            second_view_key: Key for the second view.
            action_type_adaln, use_causal_attention, use_cross_attn, use_adaln_cond, use_readout_token:
                Flags controlling attention and conditioning.
            use_proprio: Whether to incorporate proprioception.
            return_act_chunk: Whether to return the whole action chunk.
            sampling_type: Sampling type (e.g., 'ln', 'pi_zero', etc.).
            dit_dim, n_heads, n_layers: DiT architecture parameters.
            attn_pdrop, resid_pdrop, mlp_pdrop: Dropout rates.
            use_rope, use_nope, query_seq_len, rope_theta: Positional encoding parameters.
        """
        super().__init__()
        self.device = device
        self.process_id = process_id
        self.accelerator = accelerator

        # Initialize configuration groups.
        self._init_modalities(target_modality, obs_modalities, goal_modalities, img_modalities, lang_modalities)
        self._init_dimensions(dit_dim, n_heads, lowdim_obs_dim, action_dim, act_window_size, multistep, num_sampling_steps)
        self._init_flags(use_second_view, use_causal_attention, use_cross_attn, use_adaln_cond,
                         use_readout_token, use_rope, use_nope, vlm_prompt_style, token_dropout,
                         action_type_adaln, sampling_type, use_proprio, return_act_chunk, 
                         second_view_key, cfg_dropout)
        logger.info("Configuration (modalities, dimensions, flags) initialized.")

        # Initialize action space index.
        self.action_space_index = ActionIndex()

        # Setup model components.
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence, freeze_embeddings_only)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.use_dopri5 = False

        
        self._setup_dit_components_meanflow(
            dit_dim, n_heads, n_layers, action_dim, act_window_size, hidden_dim,
            attn_pdrop, resid_pdrop, mlp_pdrop, use_cross_attn,
            use_rope, use_nope, query_seq_len, rope_theta,
        )
        # Mean Flow specific parameters
        self.noise_dist = noise_dist
        self.ratio = ratio
        self.register_buffer(
            "P_mean",
            torch.tensor(P_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "P_std",
            torch.tensor(P_std, dtype=torch.float32)
        )
        # DMF separate distribution parameters for t and r
        self.register_buffer(
            "P_mean_t",
            torch.tensor(P_mean_t, dtype=torch.float32)
        )
        self.register_buffer(
            "P_std_t",
            torch.tensor(P_std_t, dtype=torch.float32)
        )
        self.register_buffer(
            "P_mean_r",
            torch.tensor(P_mean_r, dtype=torch.float32)
        )
        self.register_buffer(
            "P_std_r",
            torch.tensor(P_std_r, dtype=torch.float32)
        )

        logger.info("VLM and DiT components set up.")

        # Initialize rollout state.
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self._train_step = 0

        # Ensure that all parameters and buffers are on the correct device.
        self.ensure_device_consistency()

    # === Initialization Helpers ===
    def _init_modalities(self, target_modality: str, obs_modalities: str, goal_modalities: str,
                           img_modalities: List[str], lang_modalities: List[str]) -> None:
        """Initializes modality-related attributes."""
        self.target_modality = target_modality
        self.obs_modalities = obs_modalities
        self.goal_modalities = goal_modalities
        self.img_modalities = img_modalities
        self.lang_modalities = lang_modalities

    def _init_dimensions(self, dit_dim: int, n_heads: int, lowdim_obs_dim: int, action_dim: int,
                           act_window_size: int, multistep: int, num_sampling_steps: int) -> None:
        """Initializes dimension-related attributes and checks consistency."""
        if dit_dim % n_heads != 0:
            raise ValueError(f"dit_dim ({dit_dim}) must be divisible by n_heads ({n_heads})")
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim = action_dim
        self.act_window_size = act_window_size
        self.multistep = multistep
        self.num_sampling_steps = num_sampling_steps
        self.dit_dim = dit_dim

    def _init_flags(self, use_second_view: bool, use_causal_attention: bool, use_cross_attn: bool,
                    use_adaln_cond: bool, use_readout_token: bool, use_rope: bool, use_nope: bool,
                    vlm_prompt_style: str, token_dropout: float, action_type_adaln: bool,
                    sampling_type: str, use_proprio: bool, return_act_chunk: bool, second_view_key: str,
                    cfg_dropout: float) -> None:
        """Initializes boolean flags and related parameters."""
        if vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
        if sampling_type not in ['ln', 'pi_zero', 'loglogistic', 'uniform', 'stratified']:
            raise ValueError(f"Invalid sampling type: {sampling_type}")
        self.use_second_view = use_second_view
        self.use_causal_attention = use_causal_attention
        self.use_cross_attn = use_cross_attn
        self.use_adaln_cond = use_adaln_cond
        self.use_readout_token = use_readout_token
        self.use_rope = use_rope
        self.use_nope = use_nope
        self.use_proprio = use_proprio
        self.return_act_chunk = return_act_chunk
        self.vlm_prompt_style = vlm_prompt_style
        self.token_dropout = token_dropout
        self.action_type_adaln = action_type_adaln
        self.sampling_type = sampling_type
        self.second_view_key = second_view_key
        self.cfg_dropout = cfg_dropout
        self.cfg_lambda = 1.0

    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool, freeze_embeddings_only: bool) -> None:
        """
        Loads the pretrained VLM, sets up the processor/tokenizer, adds a prompt token,
        and optionally freezes parameters.
        """
        logger.info(f"Loading VLM from {vlm_path}")
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)
        self.train_vlm = not freeze_florence
        
        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif freeze_embeddings_only:
            embedding_layer = self.vlm.get_input_embeddings()
            for param in embedding_layer.parameters():
                param.requires_grad = False
            if hasattr(self.vlm.language_model, 'shared'):
                for param in self.vlm.language_model.shared.parameters():
                    param.requires_grad = False
        
        if not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True
        
        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.prompt_embeds = self._create_prompt_embed("<Flow>").to(self.device)
        del self.vlm.language_model.model.decoder, self.vlm.language_model.lm_head
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    def _setup_dit_components_meanflow(
        self,
        dit_dim: int,
        n_heads: int,
        n_layers: int,
        action_dim: int,
        act_window_size: int,
        hidden_dim: int,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        use_cross_attn: bool,
        use_rope: bool,
        use_nope: bool,
        query_seq_len: int,
        rope_theta: float
    ) -> None:
        """
        Sets up DiT components for Decoupled MeanFlow (DMF).
        Uses encoder-decoder DMFTransformer with separate t/r conditioning.
        """
        # Initialize module dictionaries
        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        if self.action_type_adaln:
            self.adaln = nn.ModuleDict()

        # Set up action-specific components
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)

            # Action encoder (same as rectified flow)
            self.action_encoders[action_name] = Mlp(
                in_features=input_dim,
                hidden_features=dit_dim,
                out_features=dit_dim,
                bias=True
            )

            # DMF: plain linear decoder — r-conditioning is in the decoder blocks
            self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim).to(self.device)

            # Action-specific AdaLN
            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    dit_dim,
                    global_conddim=dit_dim,
                    use_cross_attn=use_cross_attn
                )

            # Proprioceptive encoders
            if self.use_proprio:
                if action_name == 'bimanual_nav':
                    self.proprio_encoders[action_name] = Mlp(
                        input_dim,
                        dit_dim,
                        out_features=dit_dim,
                        drop=0.2
                    ).to(self.device)
                else:
                    self.proprio_encoders[action_name] = ZeroEncoder(
                        self.dit_dim,
                        device=self.device
                    )

        # Set up shared conditioning components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)     # for encoder (t)
        self.r_embedder = TimestepEmbedder(dit_dim)     # for decoder (r) — NEW
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            dit_dim,
            max_actions=len(self.action_space_index.action_spaces)
        )

        # Set up positional encoding if neither RoPE nor NoPE is used
        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, act_window_size, dit_dim) * 0.1
            )

        # Replace flat DiT list with encoder-decoder split
        n_encoder_layers = n_layers * 2 // 3  # e.g., 8 of 12
        n_decoder_layers = n_layers - n_encoder_layers  # e.g., 4 of 12

        # Set up DiT blocks
        self.dit = DMFTransformer(
            dim=dit_dim,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta
        )

        # Separate AdaLN controllers for encoder and decoder
        if not self.action_type_adaln:
            self.adaln_t = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)
            self.adaln_r = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)

        # Log-variance prediction head for DMF loss weighting
        # Uses sinusoidal embeddings of (t, r) → concat → linear → scalar lv
        self.logvar_linear = nn.Linear(256, 1)  # 128 (t embed) + 128 (r embed) → 1
        nn.init.constant_(self.logvar_linear.weight, 0)
        nn.init.constant_(self.logvar_linear.bias, 0)

    def _verify_device_consistency(self) -> None:
        """Verifies that all parameters and buffers are on the expected device."""
        expected = self.device
        inconsistent = []
        for name, param in self.named_parameters():
            if param.device != expected:
                inconsistent.append(f"{name}: {param.device} (expected {expected})")
        for name, buf in self.named_buffers():
            if buf.device != expected:
                inconsistent.append(f"{name} (buffer): {buf.device} (expected {expected})")
        if inconsistent:
            logger.warning("Device consistency issues: " + "; ".join(inconsistent))

    # === Encoding and Decoding Methods ===
    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Encodes primary (and optional second view) image observations and text goals.
        Returns a dictionary with:
            - 'features': Encoder outputs.
            - 'frequency_embeds': Frequency embeddings.
            - 'action_space_embeds': Action space embeddings.
            - 'action_type': Action type indices.
            - 'proprio': Proprioception data (if available).
            - 'attention_mask': Attention mask.
        """
        device = self.device
        default_dtype = next(self.parameters()).dtype
        image_tensor = batch[self.obs_modalities]['image_primary']
        B, T, C, H, W = image_tensor.shape
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_dtype)
        )
        image_features = image_features.view(B, T * image_features.shape[1], -1)


        if self.use_second_view and self.second_view_key in batch[self.obs_modalities]:
            image2_tensor = batch[self.obs_modalities][self.second_view_key]
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_dtype)
            )
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        text_embeds = self.vlm.get_input_embeddings()(
            batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'].to(device)
        ).to(device).squeeze(1)

        # get the flow prompt for florence
        task_prompt = self.prompt_embeds.expand(B, -1, -1)
        merged_embeds = torch.cat([task_prompt.to(image_features.device), image_features, text_embeds.to(image_features.device)], dim=1)

        # get attention mask from txt
        lang_attention_mask = batch[self.goal_modalities][self.lang_modalities[0]]['attention_mask'].to(device).squeeze(1)
        # define attention mask for image — use bool dtype throughout for cross-attention compatibility
        vis_attention_mask = torch.ones(image_features.shape[:2], dtype=torch.bool, device=image_features.device)
        prompt_mask = torch.zeros(B, 1, dtype=torch.bool, device=image_features.device)
        attention_mask = torch.cat([prompt_mask, vis_attention_mask, lang_attention_mask.bool()], dim=1)

        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds, 
            attention_mask=attention_mask,
        ).last_hidden_state

        features = self.vlm_token_dropout(features)

        # add optinal cfg dropout
        if self.cfg_dropout > 0 and self.training:
            prompt_length = task_prompt.shape[1]
            image_length = image_features.shape[1]
            text_length = text_embeds.shape[1]  # assumed fixed length per example
            text_start = prompt_length + image_length
            text_end = text_start + text_length  # text features occupy features[:, text_start:text_end, :]
            # Create a dropout mask for the entire batch (per example)
            drop_mask = (torch.rand(B, device=device) < self.cfg_dropout).to(dtype=default_dtype).view(B, 1, 1)
            # Apply the mask only to the text portion of the features.
            features[:, text_start:text_end, :] = features[:, text_start:text_end, :] * (1 - drop_mask)

        return {
            'features': features,
            'frequency_embeds': self.frequency_embedder(batch[self.goal_modalities]['frequency'].to(device).to(default_dtype)),
            'action_space_embeds': self.action_space_embedder(batch[self.goal_modalities]['action_space_index'].to(device)),
            'action_type': batch[self.goal_modalities]['action_space_index'],
            'proprio': batch[self.obs_modalities]['proprio'].to(device).to(default_dtype) if self.use_proprio and 'proprio' in batch[self.obs_modalities] else None,
            'attention_mask': attention_mask,
        }

    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes actions for each sample based on its action type.
        Returns:
            - Encoded actions (latent representations).
            - A valid dimensions mask.
        """
        action_type = action_type.to(self.device)
        B = z.shape[0]
        encoded = torch.zeros(B, z.shape[1], self.dit_dim, device=self.device, dtype=z.dtype)
        valid_dims = torch.zeros_like(z)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_dims[mask, :, :adim] = 1
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim]).to(encoded.dtype)
        return encoded, valid_dims

    def decode_actions_meanflow(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent representations into actions using plain linear projection.
        In DMF, r-conditioning is handled by the decoder transformer blocks,
        so the output head is just nn.Linear.

        Args:
            z: DiT latent features [B, T, dit_dim]
            action_type: Action type indices [B] or broadcastable.
            valid_dims: Valid dimensions mask [B, T, action_dim].
        """
        default_dtype = next(self.parameters()).dtype
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=self.device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                pred = self.action_decoders[action_name](z[mask])
                decoded[mask, :, :adim] = (pred[..., :adim] * valid_dims[mask, :, :adim]).to(default_dtype)
        return decoded

    # === Loss Functions ===
    def meanflow_loss(self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the Decoupled MeanFlow (DMF) dual loss.

        Each branch gets independent timesteps and noise (matches DMF reference):
          FM branch: t_fm from P_mean/P_std, r=t_fm, no JVP
          MF branch: t_mf/r_mf from P_mean_t/P_std_t/P_mean_r/P_std_r, JVP

        Reference: https://github.com/kyungmnlee/dmf
        """
        default_dtype = next(self.parameters()).dtype
        action_type = cond['action_type']
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(dtype=default_dtype)

        # Sample independent timesteps per branch (matches DMF reference)
        t_fm, t_mf, r_mf = self.sample_times(b)

        # Sample independent noise per branch, per action space
        e_fm = torch.zeros_like(actions)
        e_mf = torch.zeros_like(actions)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                n = mask.sum()
                seq_len = actions.size(1)
                e_fm[mask, :, :adim] = torch.randn(n, seq_len, adim, dtype=actions.dtype, device=device)
                e_mf[mask, :, :adim] = torch.randn(n, seq_len, adim, dtype=actions.dtype, device=device)

        v_fm_target = e_fm - actions  # FM target velocity
        v_mf_target = e_mf - actions  # MF target velocity

        # Build valid mask for action spaces
        valid_mask = torch.zeros_like(actions, dtype=torch.bool)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                mask_expanded = mask.view(-1, 1, 1).expand(-1, actions.size(1), adim).to(device)
                valid_mask[mask, :, :adim] = mask_expanded[mask]

        valid_mask_f = valid_mask.to(dtype=torch.float32)
        valid_dims_count = valid_mask_f.sum(dim=list(range(1, valid_mask_f.ndim)))  # [B]

        # Construct independent z_t per branch
        t_fm_exp = t_fm.to(dtype=default_dtype)
        t_mf_exp = t_mf.to(dtype=default_dtype)
        z_t_fm = ((1 - t_fm_exp) * actions + t_fm_exp * e_fm).float()
        z_t_mf = ((1 - t_mf_exp) * actions + t_mf_exp * e_mf).float()
        v_fm_f32 = v_fm_target.float()
        v_mf_f32 = v_mf_target.float()
        t_fm_flat = t_fm.view(-1).float()
        t_mf_flat = t_mf.view(-1).float()
        r_mf_flat = r_mf.view(-1).float()
        r_mf_f32 = r_mf.float()

        # Monkey-patch nn.Linear and RmsNorm for dtype safety (bf16 weights + f32 inputs)
        _orig_linear_forward = nn.Linear.forward
        _orig_rmsnorm_forward = RmsNorm.forward

        def _jvp_safe_linear_forward(self, input):
            return F.linear(
                input,
                self.weight.to(input.dtype),
                self.bias.to(input.dtype) if self.bias is not None else None,
            )

        def _jvp_safe_rmsnorm_forward(self, x):
            return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)

        with torch.amp.autocast("cuda", enabled=False):
            nn.Linear.forward = _jvp_safe_linear_forward
            RmsNorm.forward = _jvp_safe_rmsnorm_forward
            try:
                # ========== Prepare conditioning (outside JVP — doesn't depend on z) ==========
                working_dtype = z_t_fm.dtype  # float32
                vlm_features = self.cond_linear(self.cond_norm(cond['features'].to(working_dtype)))
                freq_embeds = cond['frequency_embeds'].squeeze(1).to(working_dtype)
                action_type_dev = cond['action_type'].to(self.device)
                proprio = cond.get('proprio', torch.zeros_like(freq_embeds)).to(working_dtype) if self.use_proprio else torch.zeros_like(freq_embeds)
                proprio_embeds = self.encode_proprio(proprio, action_type_dev, freq_embeds.shape).to(working_dtype)

                # Apply CFG dropout on freq_embeds and proprio_embeds
                if self.training and self.cfg_dropout > 0:
                    drop_mask = (torch.rand(freq_embeds.size(0), device=freq_embeds.device) < self.cfg_dropout).to(dtype=working_dtype).unsqueeze(1)
                    freq_embeds = freq_embeds * (1 - drop_mask)
                    proprio_embeds = proprio_embeds * (1 - drop_mask)

                shared_signals = sum(map(stateless_norm, [freq_embeds, proprio_embeds]))
                context = vlm_features if self.use_cross_attn else None
                cross_attn_mask = cond['attention_mask']
                if self.use_adaln_cond:
                    global_cond_base = vlm_features[:, 0, :] if self.use_readout_token else vlm_features.mean(dim=1)

                # ========== FM branch: independent encoder + decoder, no JVP ==========
                t_fm_emb = stateless_norm(self.t_embedder(t_fm_flat.detach())) + shared_signals
                t_fm_cond = (global_cond_base + t_fm_emb) if self.use_adaln_cond else t_fm_emb
                t_fm_global_adaln = self.action_specific_adaln(t_fm_cond, action_type_dev) if self.action_type_adaln else self.adaln_t(t_fm_cond)

                r_fm_emb = stateless_norm(self.r_embedder(t_fm_flat.detach())) + shared_signals  # r=t for FM
                r_fm_cond = (global_cond_base + r_fm_emb) if self.use_adaln_cond else r_fm_emb
                r_fm_global_adaln = self.action_specific_adaln(r_fm_cond, action_type_dev) if self.action_type_adaln else self.adaln_r(r_fm_cond)

                z_fm_enc, valid_dims_fm = self.encode_actions(z_t_fm, action_type_dev)
                if not (self.use_rope or self.use_nope):
                    z_fm_enc = z_fm_enc + self.positional_encoding
                h_fm = self.dit.encode(
                    z_fm_enc, t_fm_cond,
                    context=context,
                    custom_attn_mask=None,
                    custom_cross_attn_mask=cross_attn_mask,
                    is_causal=True,
                    t_global_adaln=t_fm_global_adaln
                )
                z_fm_dec = self.dit.decode(
                    h_fm, r_fm_cond,
                    context=context,
                    custom_attn_mask=None,
                    custom_cross_attn_mask=cross_attn_mask,
                    is_causal=True,
                    r_global_adaln=r_fm_global_adaln
                )
                v_pred = self.decode_actions_meanflow(z_fm_dec, action_type_dev, valid_dims_fm)

                # FM logvar
                lv_fm = self.logvar_linear(
                    torch.cat([logvar_timestep_embedding(t_fm_flat), logvar_timestep_embedding(t_fm_flat)], dim=1)
                )
                lv_fm = lv_fm.view(-1, *[1] * (v_pred.ndim - 1))

                # ========== MF branch: encoder JVP + decoder JVP ==========
                t_mf_emb = stateless_norm(self.t_embedder(t_mf_flat.detach())) + shared_signals
                t_mf_cond = (global_cond_base + t_mf_emb) if self.use_adaln_cond else t_mf_emb
                t_mf_global_adaln = self.action_specific_adaln(t_mf_cond, action_type_dev) if self.action_type_adaln else self.adaln_t(t_mf_cond)

                r_mf_emb = stateless_norm(self.r_embedder(r_mf_flat.detach())) + shared_signals
                r_mf_cond = (global_cond_base + r_mf_emb) if self.use_adaln_cond else r_mf_emb
                r_mf_global_adaln = self.action_specific_adaln(r_mf_cond, action_type_dev) if self.action_type_adaln else self.adaln_r(r_mf_cond)

                _, valid_dims_mf = self.encode_actions(z_t_mf, action_type_dev)

                def enc_fn(z_input):
                    z_enc, _ = self.encode_actions(z_input, action_type_dev)
                    if not (self.use_rope or self.use_nope):
                        z_enc = z_enc + self.positional_encoding
                    h = self.dit.encode(
                        z_enc, t_mf_cond,
                        context=context,
                        custom_attn_mask=None,
                        custom_cross_attn_mask=cross_attn_mask,
                        is_causal=True,
                        t_global_adaln=t_mf_global_adaln
                    )
                    return h

                (h_mf, dh) = torch.func.jvp(enc_fn, (z_t_mf,), (v_mf_f32,))

                def dec_fn(h_input):
                    z_dec = self.dit.decode(
                        h_input, r_mf_cond,
                        context=context,
                        custom_attn_mask=None,
                        custom_cross_attn_mask=cross_attn_mask,
                        is_causal=True,
                        r_global_adaln=r_mf_global_adaln
                    )
                    return self.decode_actions_meanflow(z_dec, action_type_dev, valid_dims_mf)

                (u_pred, dudt) = torch.func.jvp(dec_fn, (h_mf,), (dh,))

                # MF logvar
                lv_mf = self.logvar_linear(
                    torch.cat([logvar_timestep_embedding(t_mf_flat), logvar_timestep_embedding(r_mf_flat)], dim=1)
                )
                lv_mf = lv_mf.view(-1, *[1] * (u_pred.ndim - 1))

            finally:
                nn.Linear.forward = _orig_linear_forward
                RmsNorm.forward = _orig_rmsnorm_forward

            # ========== Compute losses ==========
            v_pred_valid = v_pred * valid_mask_f
            v_target_valid = v_fm_f32 * valid_mask_f
            fm_mse, fm_log = log_lv_loss(v_pred_valid, v_target_valid, lv_fm, valid_dims_count=valid_dims_count)

            # u_tgt = v + (r - t) * du/dt  (DMF sign convention)
            gap = (r_mf_f32 - t_mf.float())  # negative since r <= t
            u_tgt = (v_mf_f32 + gap * dudt).detach()

            u_pred_valid = u_pred * valid_mask_f
            u_tgt_valid = u_tgt * valid_mask_f
            mf_mse, mf_log = log_lv_loss(u_pred_valid, u_tgt_valid, lv_mf, valid_dims_count=valid_dims_count)

        # ========== Combine losses (per-sample [B], matching DMF reference) ==========
        loss = 0.5 * (fm_log + mf_log)

        # Check for NaN/Inf (per-sample)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("NaN/Inf detected in loss! Clipping to prevent crash.")
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

        # ========== Monitor metrics ==========
        with torch.no_grad():
            fm_mse_mean = fm_mse.mean()
            mf_mse_mean = mf_mse.mean()
            lv_fm_mean = lv_fm.mean()
            lv_mf_mean = lv_mf.mean()
            dudt_norm = dudt[valid_mask].norm(dim=0).mean() if valid_mask.any() else torch.tensor(0.0)
            # Cosine similarity: u_pred vs v (single-step convergence, using MF branch's v)
            valid_u = u_pred[valid_mask]
            valid_v = v_mf_f32[valid_mask]
            cos_u_v = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_v.unsqueeze(0), dim=-1
            ).mean() if valid_mask.any() else torch.tensor(0.0)
            # Cosine similarity: u_pred vs u_tgt (MF branch target)
            valid_u_tgt = u_tgt[valid_mask]
            cos_u_utgt = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_u_tgt.unsqueeze(0), dim=-1
            ).mean() if valid_mask.any() else torch.tensor(0.0)

        losses_dict = {
            "loss": loss.mean().item(),
            "fm_mse": fm_mse_mean.item(),
            "mf_mse": mf_mse_mean.item(),
            "fm_log_loss": fm_log.mean().item(),
            "mf_log_loss": mf_log.mean().item(),
            "lv_fm": lv_fm_mean.item(),
            "lv_mf": lv_mf_mean.item(),
            "dudt_norm": dudt_norm.item(),
            "cos_u_v": cos_u_v.item(),
            # Aliases expected by flower_trainer logging
            "raw_mse": (fm_mse_mean + mf_mse_mean).item(),
            "v_loss": loss.mean().item(),
            "cos_u_utgt": cos_u_utgt.item(),
        }

        if hasattr(self, 'accelerator') and self.accelerator is not None and wandb.run is not None:
            if self.accelerator.is_main_process:
                wandb.log(losses_dict)

        return loss, losses_dict

    def _logit_normal_sample(self, bz: int, P_mean: torch.Tensor, P_std: torch.Tensor) -> torch.Tensor:
        """Sample from logit-normal distribution with given mean and std."""
        rnd_normal = torch.randn(
            bz, 1, 1,
            device=P_mean.device,
            dtype=torch.float32
        )
        out = torch.sigmoid(rnd_normal * P_std.float() + P_mean.float())
        return out.to(next(self.parameters()).dtype)

    def sample_times(self, b: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DMF time sampling with independent timesteps per branch.

        FM branch: t_fm sampled from P_mean/P_std, uses r=t_fm.
        MF branch: two logit-normal draws sorted so t_mf >= r_mf.

        Returns:
            t_fm: [B, 1, 1] — FM branch timestep
            t_mf: [B, 1, 1] — MF branch timestep (larger of two samples)
            r_mf: [B, 1, 1] — MF branch query time (smaller of two samples)
        """
        dtype = next(self.parameters()).dtype

        t_fm = self._logit_normal_sample(b, self.P_mean, self.P_std).to(device=self.device, dtype=dtype)

        ln_1 = self._logit_normal_sample(b, self.P_mean_t, self.P_std_t).to(device=self.device, dtype=dtype)
        ln_2 = self._logit_normal_sample(b, self.P_mean_r, self.P_std_r).to(device=self.device, dtype=dtype)
        t_mf = torch.maximum(ln_1, ln_2)
        r_mf = torch.minimum(ln_1, ln_2)

        return t_fm, t_mf, r_mf

    # === Sampling Methods ===
    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """
        Samples actions from the DiT model.
        Chooses between an adaptive ODE solver and fixed-step Euler integration.
        """
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        action_type = cond['action_type']
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                z[mask, :, adim:] = 0.0
        
        return self._sample_with_fixed_steps(z, cond, inference)

    def _sample_with_fixed_steps(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """
        Samples actions using fixed-step Euler integration (rectified flow)
        or single-step sampling (mean flow).
        """
        b = z.size(0)
        device = z.device
        action_type = cond['action_type']
        
        # MeanFlow: single-step sampling
        # z_0 = z_1 - u(z_1, t=1, r=0)
        dtype = next(self.parameters()).dtype
        z = z.to(dtype=dtype)
        t_tensor = torch.ones(b, device=device, dtype=dtype)
        r_tensor = torch.zeros(b, device=device, dtype=dtype)  # r=0 for single-step
        u = self.dit_forward_meanflow(z, t_tensor, r_tensor, cond)
        z = z - u

        return z.clamp(-1, 1)

    # === Forward Pass and Rollout Methods ===
    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Inference forward pass.
        Given observation and goal dictionaries, it encodes them and samples an action sequence.
        """
        batch = {'observation': obs, 'task': goal}
        features = self.encode_observations(batch)
        noise = torch.randn(len(features['features']), self.act_window_size, self.action_dim,
                              device=features['features'].device)
        return self.sample_actions(noise, features, inference=True)

    def reset(self) -> None:
        """
        Resets the rollout state.
        """
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()

    def step(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Returns the current action (or full chunk) based on the rollout step and updates the state.
        """
        if self.rollout_step_counter % self.multistep == 0:
            self.pred_action_seq = self(obs, goal)
        if not self.return_act_chunk:
            current_action = self.pred_action_seq[0, self.rollout_step_counter]
            if len(current_action.shape) == 2:
                current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        else:
            current_action = self.pred_action_seq
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        return current_action

    # === Additional Utility Methods ===
    def training_step(self, batch: Dict[str, Dict]) -> torch.Tensor:
        """
        A single training step.
        Encodes observations, computes the appropriate flow loss, and returns the loss tensor.
        """
        self.train()
        self._train_step += 1
        obs_features = self.encode_observations(batch)

        action_loss, losses_dict = self.meanflow_loss(
            obs_features, batch[self.target_modality], batch['task']['dataset_index']
        )

        # Store debugging losses if needed.
        self.losses_dict = losses_dict
        return action_loss.mean()  # batch reduction here, not inside meanflow_loss

    def validation_step(self, batch: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """
        A single evaluation step.
        Returns a dictionary containing loss and predictions.
        """
        self.eval()
        with torch.no_grad():
            target_actions = batch[self.target_modality]
            if len(target_actions.shape) == 4:
                target_actions = target_actions.squeeze(1)
            obs_features = self.encode_observations(batch)
            action_type = obs_features['action_type']
            noise_actions = torch.zeros_like(target_actions)
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                mask = (action_type == action_idx)
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    rand_slice = torch.randn((mask.sum(), target_actions.size(1), adim),
                                             device=self.device, dtype=target_actions.dtype)
                    noise_actions[mask, :, :adim] = rand_slice
            action_pred = self.sample_actions(noise_actions, obs_features, inference=True)
            # Per-sample MSE so DatasetMetricsTracker can group by dataset
            B = target_actions.shape[0]
            per_sample_loss = torch.zeros(B, device=self.device)
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                mask = (action_type == action_idx)
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    per_sample_loss[mask] = F.mse_loss(
                        action_pred[mask, :, :adim],
                        target_actions[mask, :, :adim],
                        reduction='none'
                    ).mean(dim=(1, 2))
            return {
                "loss": per_sample_loss.detach(),
                "losses": {},
                "dataset_index": batch['task'].get('dataset_index', torch.zeros(B, device=self.device)).detach()
            }

    def meanflow_eval_loss_step(self, batch: dict) -> dict:
        """Wrapper for meanflow eval loss: encodes observations then computes v_loss."""
        self.eval()
        with torch.no_grad():
            target_actions = batch[self.target_modality]
            if len(target_actions.shape) == 4:
                target_actions = target_actions.squeeze(1)
            obs_features = self.encode_observations(batch)
            result = self.meanflow_eval_loss(obs_features, target_actions)
            B = target_actions.shape[0]
            result['dataset_index'] = batch['task'].get(
                'dataset_index', torch.zeros(B, device=self.device)
            ).detach()
            return result

    @torch.no_grad()
    def meanflow_eval_loss(self, cond: dict, actions: torch.Tensor) -> dict:
        """
        Compute ||u(z_t, t, r) - v||² during eval (no JVP needed).
        Uses sample_times() and passes r directly.
        """
        default_dtype = next(self.parameters()).dtype
        action_type = cond['action_type']
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(dtype=default_dtype)

        # Sample t and r using DMF sampling (only MF branch times needed for eval)
        _, t_mf, r_mf = self.sample_times(b)

        texp = t_mf.to(dtype=default_dtype)

        # Sample noise per action space
        e = torch.zeros_like(actions)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn(
                    (mask.sum(), actions.size(1), adim),
                    dtype=actions.dtype, device=device
                )
                e[mask, :, :adim] = noise_slice

        z = (1 - texp) * actions + texp * e
        v = e - actions  # target velocity

        t_flat = t_mf.view(-1)
        r_flat = r_mf.view(-1)

        # Single forward pass
        with torch.autocast('cuda', dtype=torch.bfloat16):
            u_pred = self.dit_forward_meanflow(z, t_flat, r_flat, cond)

        # Per-sample ||u - v||² over valid action dims
        per_sample_loss = torch.zeros(b, device=device)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                per_sample_loss[mask] = F.mse_loss(
                    u_pred[mask, :, :adim].float(),
                    v[mask, :, :adim].float(),
                    reduction='none'
                ).mean(dim=(1, 2))

        return {"loss": per_sample_loss.detach()}

    def dit_forward_meanflow(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor,
                             cond_dict: dict, return_logvar: bool = False):
        """
        Forward pass through the DMF DiT (encoder conditioned on t, decoder on r).

        Args:
            z: Latent actions [B, T, action_dim]
            t: Encoder timestep [B]
            r: Decoder query timestep [B]
            cond_dict: Conditioning dictionary
            return_logvar: If True, also return log-variance scalar for loss weighting

        Returns:
            decoded: Action predictions [B, T, action_dim]
            logvar (optional): Log-variance [B, 1, 1] (only if return_logvar=True)
        """
        B, t_seq, d = z.shape
        working_dtype = z.dtype  # float32 during JVP, bf16 during inference
        # Extract and process conditioning inputs — cast to working_dtype
        cond = self.cond_linear(self.cond_norm(cond_dict['features'].to(working_dtype)))
        freq_embeds = cond_dict['frequency_embeds'].squeeze(1).to(working_dtype)
        action_type = cond_dict['action_type'].to(self.device)
        proprio = cond_dict.get('proprio', torch.zeros_like(freq_embeds)).to(working_dtype) if self.use_proprio else torch.zeros_like(freq_embeds)
        proprio_embeds = self.encode_proprio(proprio, action_type, freq_embeds.shape).to(working_dtype)

        # Encode actions and positional information
        z, valid_dims = self.encode_actions(z, action_type)
        if not (self.use_rope or self.use_nope):
            z += self.positional_encoding

        # Apply CFG dropout on freq_embeds and proprio_embeds only
        if self.training and self.cfg_dropout > 0:
            drop_mask = (torch.rand(freq_embeds.size(0), device=freq_embeds.device) < self.cfg_dropout).to(dtype=working_dtype).unsqueeze(1)
            freq_embeds = freq_embeds * (1 - drop_mask)
            proprio_embeds = proprio_embeds * (1 - drop_mask)

        # Shared additive signals (frequency, proprio, action space)
        shared_signals = sum(map(stateless_norm, [freq_embeds, proprio_embeds]))

        # Encoder conditioning: t
        t_emb = stateless_norm(self.t_embedder(t)) + shared_signals
        # Decoder conditioning: r
        r_emb = stateless_norm(self.r_embedder(r)) + shared_signals

        # Optionally add VLM global conditioning
        if self.use_adaln_cond:
            global_cond_base = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            t_cond = global_cond_base + t_emb
            r_cond = global_cond_base + r_emb
        else:
            t_cond = t_emb
            r_cond = r_emb

        context = cond if self.use_cross_attn else None

        # Separate AdaLN modulation for encoder (t) and decoder (r)
        if self.action_type_adaln:
            t_global_adaln = self.action_specific_adaln(t_cond, action_type)
            r_global_adaln = self.action_specific_adaln(r_cond, action_type)
        else:
            t_global_adaln = self.adaln_t(t_cond)
            r_global_adaln = self.adaln_r(r_cond)

        # DMFTransformer: encoder blocks use t_cond, decoder blocks use r_cond
        z = self.dit(
            z, t_cond, r_cond,
            context=context,
            custom_attn_mask=None,
            custom_cross_attn_mask=cond_dict['attention_mask'],
            is_causal=True,
            t_global_adaln=t_global_adaln,
            r_global_adaln=r_global_adaln
        )

        decoded = self.decode_actions_meanflow(z, action_type, valid_dims)

        if return_logvar:
            logvar = self.logvar_linear(
                torch.cat([logvar_timestep_embedding(t), logvar_timestep_embedding(r)], dim=1)
            )
            logvar = logvar.view(-1, *[1] * (decoded.ndim - 1))  # [B, 1, 1]
            return decoded, logvar
        return decoded

    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """
        Encodes proprioceptive data based on action type.
        Returns a tensor with shape [batch, dit_dim].
        """
        batch_size, _ = output_shape

        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=proprio.dtype)

        encoded = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=proprio.dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded[mask] = self.proprio_encoders[action_name](proprio[mask]).squeeze(1)
        
        return encoded

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """
        Computes action-specific AdaLN modulation signals.
        Returns a list of modulation tensors.
        """
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6

        mod_signals = [torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=global_cond.dtype) for _ in range(num_chunks)]
        
        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond[mask])
                for i, signal in enumerate(action_mod):
                    mod_signals[i][mask] = signal
        
        return mod_signals

    # === Optimizer Configuration ===
    def configure_optimizers(self, optimizer_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Configures optimizer parameter groups for non-VLM parameters.
        Returns a list of parameter groups.
        """
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decoder_params_set = set()
        for decoder in self.action_decoders.values():
            decoder_params_set.update(p for p in decoder.parameters())

        decoder_weight_decay = optimizer_config.get("decoder_weight_decay", optimizer_config["transformer_weight_decay"])

        decay_group = []
        no_decay_group = []
        decoder_decay_group = []
        decoder_no_decay_group = []
        vlm_params = set(p for p in self.vlm.parameters())
        for name, param in self.named_parameters():
            if param.requires_grad and param.is_leaf and param not in vlm_params:
                is_no_decay = any(nd in name.lower() for nd in no_decay)
                if param in decoder_params_set:
                    if is_no_decay:
                        decoder_no_decay_group.append(param)
                    else:
                        decoder_decay_group.append(param)
                else:
                    if is_no_decay:
                        no_decay_group.append(param)
                    else:
                        decay_group.append(param)
        optim_groups = [
            {"params": decay_group, "weight_decay": optimizer_config["transformer_weight_decay"]},
            {"params": no_decay_group, "weight_decay": 0.0},
            {"params": decoder_decay_group, "weight_decay": decoder_weight_decay},
            {"params": decoder_no_decay_group, "weight_decay": 0.0},
        ]
        return optim_groups

    def print_encoded_texts(self, batch: Dict, device) -> None:
        """
        Prints original and encoded texts for debugging.
        """
        text_embeds = self.vlm.get_input_embeddings()(
            batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'].to(self.device)
        ).to(device).squeeze(1)
        input_ids = batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'][0].squeeze(0).to(self.device)
        decoded_text = self.processor.tokenizer.decode(input_ids.cpu(), skip_special_tokens=False)
        print("Original text:", decoded_text)
        decoded_texts = self.processor.tokenizer.batch_decode(text_embeds.cpu(), skip_special_tokens=True)
        print("Encoded texts:")
        for i, text in enumerate(decoded_texts):
            print(f"Sequence {i+1}: {text}")

    def _create_prompt_embed(self, prompt_text: str) -> nn.Parameter:
        """
        Creates a prompt embedding. Adds the prompt token to the tokenizer
        and returns its embedding (frozen).
        """
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)
    
    # === Device Consistency Methods ===
    def ensure_device_consistency(self) -> None:
        """Moves the entire model (and buffers) to the designated device."""
        self.to(self.device)
        self.vlm.to(self.device)
        if not self.use_rope and hasattr(self, 'positional_encoding'):
            self.positional_encoding = self.positional_encoding.to(self.device)
        if self.use_readout_token and hasattr(self, 'register_token'):
            self.register_token = self.register_token.to(self.device)
        self._verify_device_consistency()