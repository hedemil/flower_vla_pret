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
    MeanFlowDecoder,
    VelocityDecoder,
    stateless_norm
)
from flower_vla.dataset.utils.act_seq_mapping import DATASET_ACT_SEQ_MAP

logger = logging.getLogger(__name__)


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
        freeze_embeddings_only: bool = False,
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
        # Mean Flow Configuration
        noise_dist: str = 'logit_normal',
        P_mean: float = -0.4,
        P_std: float = 1.0,
        ratio: float = 0.75,
        norm_eps: float = 1e-2,
        norm_p: float = 0.5,
        # iMF Configuration
        use_imf: bool = False,
        imf_v_weight: float = 1.0,
        imf_head_depth: int = 8,
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
        self.use_imf = use_imf
        self.imf_head_depth = imf_head_depth


        # Setup DiT components (Mean Flow version with MeanFlowDecoder)
        self._setup_dit_components_meanflow(
            dit_dim=dit_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            action_dim=action_dim,
            act_window_size=act_window_size,
            hidden_dim=hidden_dim,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            use_nope=use_nope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )

        # Mean Flow specific parameters
        self.noise_dist = noise_dist
        self.ratio = ratio
        self.register_buffer("P_mean", torch.tensor(P_mean, dtype=torch.float32))
        self.register_buffer("P_std", torch.tensor(P_std, dtype=torch.float32))
        self.norm_eps = norm_eps
        self.norm_p = norm_p
        self.imf_v_weight = imf_v_weight

        logger.info("VLM and DiT components set up.")

        # Initialize rollout state.
        self.rollout_step_counter = 0
        self.pred_action_seq = None

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
        Sets up DiT components for Mean Flow. Identical to _setup_dit_components
        except action_decoders use MeanFlowDecoder (h-conditioned) instead of nn.Linear.
        """
        # Initialize module dictionaries
        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        # Set up shared conditioning components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.h_embedder = TimestepEmbedder(dit_dim)
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

        # Set up DiT blocks
        block_kwargs = dict(
            dim=dit_dim,
            heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )

        if self.use_imf:
            # Split into shared backbone + separate u/v heads (matching official iMF)
            shared_depth = n_layers - self.imf_head_depth
            self.shared_blocks = nn.ModuleList([
                FlowBlock(**block_kwargs) for _ in range(shared_depth)
            ])
            self.u_head_blocks = nn.ModuleList([
                FlowBlock(**block_kwargs) for _ in range(self.imf_head_depth)
            ])
            self.v_head_blocks = nn.ModuleList([
                FlowBlock(**block_kwargs) for _ in range(self.imf_head_depth)
            ])
        else:
            self.dit = nn.ModuleList([
                FlowBlock(**block_kwargs) for _ in range(n_layers)
            ])

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
            # MeanFlowDecoder replaces nn.Linear
            self.action_decoders[action_name] = MeanFlowDecoder(
                dit_dim=dit_dim,
                action_dim=input_dim,
                hidden_dim=dit_dim * 2
            ).to(self.device)

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

        # Set up velocity decoders for iMF
        if self.use_imf:
            self.velocity_decoders = nn.ModuleDict()
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                input_dim = self.action_space_index.get_action_dim(action_idx)
                self.velocity_decoders[action_name] = VelocityDecoder(
                    dit_dim=dit_dim,
                    action_dim=input_dim,
                ).to(self.device)

        # Set up shared AdaLN if not using action-specific AdaLN
        if not self.action_type_adaln:
            self.adaln = SharedAdaLNController(
                dit_dim,
                global_conddim=dit_dim,
                use_cross_attn=use_cross_attn
            )

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
        attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)
        lang_attention_mask = batch[self.goal_modalities][self.lang_modalities[0]]['attention_mask'].to(device).squeeze(1)
        # define attention mask for image
        vis_attention_mask = torch.ones(image_features.shape[:2], device=image_features.device)
        prompt_mask = torch.zeros(B, 1, dtype=torch.bool, device=image_features.device)
        attention_mask = torch.cat([prompt_mask, vis_attention_mask, lang_attention_mask], dim=1)

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
            drop_mask = (torch.rand(B, device=device) < self.cfg_dropout).float().view(B, 1, 1)
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

    @property
    def dit_blocks(self) -> nn.ModuleList:
        """Returns all DiT blocks as a single ModuleList for gradient clipping / iteration.
        Works for both iMF (shared + u-head + v-head) and non-iMF (single dit) modes."""
        if self.use_imf:
            all_blocks = nn.ModuleList(
                list(self.shared_blocks) + list(self.u_head_blocks) + list(self.v_head_blocks)
            )
            return all_blocks
        return self.dit

    def dit_parameters(self):
        """Yields all DiT block parameters for gradient clipping."""
        if self.use_imf:
            yield from self.shared_blocks.parameters()
            yield from self.u_head_blocks.parameters()
            yield from self.v_head_blocks.parameters()
        else:
            yield from self.dit.parameters()

    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes actions for each sample based on its action type.
        Returns:
            - Encoded actions (latent representations).
            - A valid dimensions mask.
        """
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(self.device)
        B = z.shape[0]
        encoded = torch.zeros(B, z.shape[1], self.dit_dim, device=self.device, dtype=default_dtype)
        valid_dims = torch.zeros_like(z, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_dims[mask, :, :adim] = 1
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim])
        return encoded, valid_dims

    def decode_actions_meanflow(
        self, z: torch.Tensor, h: torch.Tensor,
        action_type: torch.Tensor, valid_dims: torch.Tensor
    ) -> torch.Tensor:
        """
        Decodes latent representations into actions using MeanFlowDecoder.
        The decoder is conditioned on h = t - r.
        """
        default_dtype = next(self.parameters()).dtype
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=z.device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                decoded[mask, :, :adim] = self.action_decoders[action_name](z[mask], h[mask])
        return decoded

    def decode_velocity(
        self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor
    ) -> torch.Tensor:
        """Decodes latent representations into instantaneous velocity using VelocityDecoder (no h)."""
        default_dtype = next(self.parameters()).dtype
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=z.device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                decoded[mask, :, :adim] = self.velocity_decoders[action_name](z[mask])
        return decoded

    # === Loss Functions ===
    def meanflow_loss(self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the Mean Flow loss using JVP (Jacobian Vector Product).
        Based on: https://github.com/Gsunshine/py-meanflow
        """
        default_dtype = next(self.parameters()).dtype
        action_type = cond['action_type']
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(dtype=default_dtype)

        # Sample t and r with constraint t >= r
        t, r = self.sample_tr(b)

        # Interpolate: z_t = (1 - t) * x + t * e
        texp = t.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)
        rexp = r.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)

        # Sample noise only over valid action dimensions (padding stays zero)
        e = torch.zeros_like(actions)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn((mask.sum(), actions.size(1), adim), dtype=default_dtype, device=device)
                e[mask, :, :adim] = noise_slice

        z = (1 - texp) * actions + texp * e
        v = e - actions  # target velocity

        # Define network function for JVP.
        # t and h are NOT detached — the full du/dt includes ∂u/∂t (through
        # t_embedder/adaLN) and ∂u/∂h (through MeanFlowDecoder's h_embedder).
        def u_func(z_input, t_input, r_input):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_input = t_input - r_input
                t_flat = t_input.view(-1)
                h_flat = h_input.view(-1)
                return self.dit_forward_meanflow(z_input, t_flat, h_flat, cond)

        # Tangent vectors: dz/dt = v, dt/dt = 1, dr/dt = 0
        dtdt = torch.ones_like(texp)
        drdt = torch.zeros_like(rexp)

        with torch.amp.autocast("cuda", enabled=False):
           
            u_pred, dudt = torch.func.jvp(
                u_func,
                (z, texp, rexp),
                (v, dtdt, drdt)
            )

            # u_tgt = v - h * du/dt
            h = (texp - rexp).clamp(min=0.0, max=1.0)
            u_tgt = (v - h * dudt).detach()


            # Build valid mask over action dimensions
            valid_mask = torch.zeros_like(actions, dtype=torch.bool)
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                amask = (action_type == action_idx)
                if amask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    mask_expanded = amask.view(-1, 1, 1).expand(-1, actions.size(1), adim).to(device)
                    valid_mask[amask, :, :adim] = mask_expanded[amask]

            # Compute loss only over valid dimensions
            diff = u_pred - u_tgt
            diff = diff * valid_mask.to(dtype=default_dtype)
            loss_per_sample = (diff ** 2).sum(dim=(1, 2))
            raw_mse_per_sample = loss_per_sample.detach()

            # Adaptive weighting: normalizes loss to ~1.0 per sample.
            # This is critical for MeanFlow stability — without it, the
            # self-referential target u_tgt = v - h*du/dt creates a positive
            # feedback loop where large du/dt → large loss → large gradients
            # → even larger du/dt, causing divergence.
            adp_wt = (loss_per_sample.detach() + self.norm_eps) ** self.norm_p
            loss_per_sample = loss_per_sample / adp_wt

            loss = loss_per_sample.mean()

        # Monitor metrics (only over valid dimensions)
        with torch.no_grad():
            valid_u = u_pred[valid_mask]
            valid_v = v[valid_mask]
            valid_utgt = u_tgt[valid_mask]
            v_loss = ((valid_u - valid_v) ** 2).mean()
            # Raw MSE before adaptive normalization — the real convergence signal
            raw_mse = raw_mse_per_sample.mean()
            # Track du/dt magnitude — if this vanishes, the model degenerates
            # to standard flow and single-step sampling will fail.
            dudt_norm = dudt.norm(dim=0).mean()
            # Prediction/target norms
            u_pred_norm = valid_u.norm(dim=0).mean()
            u_tgt_norm = valid_utgt.norm(dim=0).mean()
            # Cosine similarity: u_pred vs u_tgt (training alignment)
            cos_u_utgt = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_utgt.unsqueeze(0), dim=-1
            ).mean()
            # Cosine similarity: u_pred vs v (single-step convergence)
            cos_u_v = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_v.unsqueeze(0), dim=-1
            ).mean()

        # Check for NaN/Inf in outputs
        if torch.isnan(u_pred).any() or torch.isinf(u_pred).any():
            logger.warning(f"NaN/Inf detected in u_pred! "
                           f"u_pred stats: min={u_pred.min().item():.4f}, max={u_pred.max().item():.4f}, "
                           f"z stats: min={z.min().item():.4f}, max={z.max().item():.4f}, "
                           f"h stats: min={h.min().item():.4f}, max={h.max().item():.4f}")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("NaN/Inf detected in loss! Clipping to prevent crash.")
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

        # Verify loss has gradient function
        if loss.grad_fn is None and loss.requires_grad:
            logger.warning("Loss requires_grad=True but has no grad_fn! "
                           "This indicates a gradient tracking issue.")
        elif not loss.requires_grad:
            logger.error("Loss does not require gradients! Setting requires_grad=True")
            loss.requires_grad_(True)

        losses_dict = {
            "loss": loss.item() if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else 1e6,
            "raw_mse": raw_mse.item(),
            "v_loss": v_loss.item() if not (torch.isnan(v_loss).any() or torch.isinf(v_loss).any()) else 1e6,
            "dudt_norm": dudt_norm.item(),
            "u_pred_norm": u_pred_norm.item(),
            "u_tgt_norm": u_tgt_norm.item(),
            "cos_u_utgt": cos_u_utgt.item(),
            "cos_u_v": cos_u_v.item(),
            "h_mean": h.mean().item(),
        }

        return loss, losses_dict

    def imf_loss(self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Improved MeanFlow (iMF) loss. Non-self-referential compound velocity target.
        Based on: https://github.com/Lyy-iiis/imeanflow

        Key differences from meanflow_loss:
        1. JVP tangent uses predicted velocity v_c (from velocity head at h=0)
        2. Loss target: V = u + h * sg(du/dt) trained against v = e - x
        3. Auxiliary v-loss trains the velocity head
        """
        default_dtype = next(self.parameters()).dtype
        action_type = cond['action_type']
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(dtype=default_dtype)

        # Sample t and r with constraint t >= r
        t, r = self.sample_tr(b)

        texp = t.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)
        rexp = r.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)

        # Sample noise only over valid action dimensions
        e = torch.zeros_like(actions)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn((mask.sum(), actions.size(1), adim), dtype=default_dtype, device=device)
                e[mask, :, :adim] = noise_slice

        z = (1 - texp) * actions + texp * e
        v = e - actions  # data velocity (target)

        # Step 1: Compute v_c (velocity prediction at h=0) for JVP tangent.
        # h=0 gives instantaneous velocity (not mean flow).
        # No gradients needed — v_c is only used as tangent direction.
        h_zero = torch.zeros_like(t)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, v_c_tangent = self._forward_imf(z, t, h_zero, cond)

        # Step 2: JVP with has_aux=True — u_func returns (u, v_pred) where
        # v_pred is auxiliary (not differentiated). Matches official iMF pattern.
        # Single forward pass: shared blocks -> branch -> u-head + v-head.
        def u_func(z_input, t_input, r_input):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_input = t_input - r_input
                t_flat = t_input.view(-1)
                h_flat = h_input.view(-1)
                u, v_pred = self._forward_imf(z_input, t_flat, h_flat, cond)
            return u, v_pred

        dtdt = torch.ones_like(texp)
        drdt = torch.zeros_like(rexp)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, du_dt, v_pred = torch.func.jvp(
                u_func,
                (z, texp, rexp),
                (v_c_tangent, dtdt, drdt),
                has_aux=True,
            )

            # Step 3: Compound velocity V = u + h * sg(du/dt)
            h = (texp - rexp).clamp(min=0.0, max=1.0)
            V = u_pred + h * du_dt.detach()

            # Stop-gradient on target
            v_g = v.detach()

            # Build valid mask over action dimensions
            valid_mask = torch.zeros_like(actions, dtype=torch.bool)
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                amask = (action_type == action_idx)
                if amask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    mask_expanded = amask.view(-1, 1, 1).expand(-1, actions.size(1), adim).to(device)
                    valid_mask[amask, :, :adim] = mask_expanded[amask]

            valid_float = valid_mask.to(dtype=default_dtype)

            # Step 4: Compound velocity loss (V vs v)
            diff_V = (V - v_g) * valid_float
            loss_V_per_sample = (diff_V ** 2).sum(dim=(1, 2))
            adp_wt_V = (loss_V_per_sample.detach() + self.norm_eps) ** self.norm_p
            loss_V = (loss_V_per_sample / adp_wt_V).mean()

            # Step 5: Auxiliary velocity loss (v_pred from JVP primal vs v)
            diff_vc = (v_pred - v_g) * valid_float
            loss_vc_per_sample = (diff_vc ** 2).sum(dim=(1, 2))
            adp_wt_vc = (loss_vc_per_sample.detach() + self.norm_eps) ** self.norm_p
            loss_vc = (loss_vc_per_sample / adp_wt_vc).mean()

            # Total loss: compound velocity + auxiliary velocity (matches iMF paper)
            loss = loss_V + loss_vc

        # Monitor metrics
        with torch.no_grad():
            valid_V = V[valid_mask]
            valid_v = v[valid_mask]
            valid_u = u_pred[valid_mask]
            raw_mse_V = loss_V_per_sample.detach().mean()
            raw_mse_vc = loss_vc_per_sample.detach().mean()
            dudt_norm = du_dt.norm(dim=0).mean()
            cos_V_v = F.cosine_similarity(
                valid_V.unsqueeze(0), valid_v.unsqueeze(0), dim=-1
            ).mean()
            cos_u_v = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_v.unsqueeze(0), dim=-1
            ).mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("NaN/Inf detected in iMF loss! Clipping to prevent crash.")
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

        losses_dict = {
            "loss": loss.item() if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else 1e6,
            "loss_V": loss_V.item(),
            "loss_vc": loss_vc.item(),
            "raw_mse_V": raw_mse_V.item(),
            "raw_mse_vc": raw_mse_vc.item(),
            "dudt_norm": dudt_norm.item(),
            "cos_V_v": cos_V_v.item(),
            "cos_u_v": cos_u_v.item(),
            "h_mean": h.mean().item(),
        }

        return loss, losses_dict

    # === Noise Distribution & Sampling for Mean Flow ===
    def noise_distribution(self):
        """Returns the noise distribution function based on config."""
        if self.noise_dist == 'logit_normal':
            return self._logit_normal_dist
        elif self.noise_dist == 'uniform':
            return self._uniform_dist
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def _logit_normal_dist(self, bz: int) -> torch.Tensor:
        """Sample from logit-normal distribution. Math in float32 for stability."""
        rnd_normal = torch.randn(
            bz,
            device=self.P_mean.device,
            dtype=torch.float32
        )
        out = torch.sigmoid(
            rnd_normal * self.P_std.float() + self.P_mean.float()
        )
        return out.to(next(self.parameters()).dtype)

    def _uniform_dist(self, bz: int) -> torch.Tensor:
        """Sample from uniform distribution."""
        return torch.rand(
            bz,
            device=self.P_mean.device,
            dtype=next(self.parameters()).dtype
        )

    def sample_tr(self, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample timesteps t and r with constraint t >= r.
        `ratio` fraction of samples keep r != t (integral/mean-flow samples).
        The remaining (1 - ratio) fraction get r = t (instantaneous velocity).

        Returns:
            t: Sampled timesteps [B]
            r: Sampled timesteps [B]
        """
        dtype = next(self.parameters()).dtype

        t = self.noise_distribution()(b).to(device=self.device, dtype=dtype)
        r = self.noise_distribution()(b).to(device=self.device, dtype=dtype)

        # Ensure t >= r element-wise
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        # With probability (1 - ratio), collapse to velocity (r = t)
        prob = torch.rand(b, device=self.device)
        velocity_mask = prob < (1 - self.ratio)
        r = torch.where(velocity_mask, t, r)

        return t, r

    # === Sampling Methods ===
    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """
        Mean Flow single-step sampling: z_0 = z_1 - u(z_1, t=1, r=0)
        (h = t - r = 1)
        """
        b = z.size(0)
        device = z.device
        dtype = next(self.parameters()).dtype
        z = z.to(dtype=dtype)
        action_type = cond['action_type']
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                z[mask, :, adim:] = 0.0
        t_tensor = torch.ones(b, device=device, dtype=dtype)
        h_tensor = torch.ones(b, device=device, dtype=dtype)  # h = t - r = 1 - 0 = 1
        u = self.dit_forward_meanflow(z, t_tensor, h_tensor, cond)
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
        obs_features = self.encode_observations(batch)

        if self.use_imf:
            action_loss, losses_dict = self.imf_loss(
                obs_features, batch[self.target_modality], batch['task']['dataset_index']
            )
        else:
            action_loss, losses_dict = self.meanflow_loss(
                obs_features, batch[self.target_modality], batch['task']['dataset_index']
            )

        # Store debugging losses if needed.
        self.losses_dict = losses_dict
        return action_loss

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

    def _dit_backbone(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond_dict: dict):
        """Shared DiT backbone: encode actions, build conditioning, run blocks.
        When use_imf=True, runs only shared_blocks; otherwise runs all dit blocks.
        Returns (features, action_type, valid_dims, cond_kwargs)."""
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape

        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(self.device)

        if self.use_proprio and cond_dict['proprio'] is not None:
            proprio = cond_dict['proprio'].to(default_dtype)
            proprio_embeds = self.encode_proprio(proprio, action_type, frequency_embeds.shape)
        else:
            proprio_embeds = torch.zeros_like(frequency_embeds)

        z, valid_dims = self.encode_actions(z, action_type)

        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding

        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(self.h_embedder(h)) + \
                stateless_norm(frequency_embeds).squeeze(1) + \
                stateless_norm(proprio_embeds).squeeze(1)

        cond = self.cond_linear(self.cond_norm(cond))

        if self.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb

        cx = z
        context = cond if self.use_cross_attn else None

        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)

        # Run shared blocks (iMF) or all blocks (non-iMF)
        blocks = self.shared_blocks if self.use_imf else self.dit
        for layer in blocks:
            cx = layer(cx, global_cond, context=context, is_causal=True, global_adaln=global_adaln)

        cond_kwargs = dict(global_cond=global_cond, context=context, global_adaln=global_adaln)
        return cx, action_type, valid_dims, cond_kwargs

    def _forward_imf(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond_dict: dict):
        """Single forward pass for iMF: shared blocks -> branch -> u-head + v-head.
        Returns (u, v) from one pass, matching official iMF __call__."""
        cx, action_type, valid_dims, cond_kwargs = self._dit_backbone(z, t, h, cond_dict)

        # Branch from shared output
        cx_u = cx
        cx_v = cx

        for block in self.u_head_blocks:
            cx_u = block(cx_u, cond_kwargs['global_cond'], context=cond_kwargs['context'],
                         is_causal=True, global_adaln=cond_kwargs['global_adaln'])

        for block in self.v_head_blocks:
            cx_v = block(cx_v, cond_kwargs['global_cond'], context=cond_kwargs['context'],
                         is_causal=True, global_adaln=cond_kwargs['global_adaln'])

        u = self.decode_actions_meanflow(cx_u, h, action_type, valid_dims)
        v = self.decode_velocity(cx_v, action_type, valid_dims)

        return u, v

    def dit_forward_meanflow(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """Forward pass for inference: backbone + u-head + MeanFlowDecoder.
        Only uses shared + u-head blocks (no v-head needed at inference)."""
        cx, action_type, valid_dims, cond_kwargs = self._dit_backbone(z, t, h, cond_dict)
        if self.use_imf:
            for block in self.u_head_blocks:
                cx = block(cx, cond_kwargs['global_cond'], context=cond_kwargs['context'],
                           is_causal=True, global_adaln=cond_kwargs['global_adaln'])
        return self.decode_actions_meanflow(cx, h, action_type, valid_dims)

    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """
        Encodes proprioceptive data based on action type.
        Returns a tensor with shape [batch, dit_dim].
        """
        batch_size, _ = output_shape
        dtype = next(self.parameters()).dtype
        
        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device)
        
        encoded = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded[mask] = self.proprio_encoders[action_name](proprio[mask]).squeeze(1).to(dtype)
        
        return encoded

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """
        Computes action-specific AdaLN modulation signals.
        Returns a list of modulation tensors.
        """
        dtype = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        
        mod_signals = [torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=dtype) for _ in range(num_chunks)]
        
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
        decay_group = []
        no_decay_group = []
        vlm_params = set(p for p in self.vlm.parameters())
        for name, param in self.named_parameters():
            if param.requires_grad and param.is_leaf and param not in vlm_params:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)
        optim_groups = [
            {"params": decay_group, "weight_decay": optimizer_config["transformer_weight_decay"]},
            {"params": no_decay_group, "weight_decay": 0.0}
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