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
        data_proportion: float = 0.5, # 1.0,
        # iMF v-head configuration
        aux_head_depth: int = 8,
        # dudt clipping
        max_dudt_norm: float = 50.0,
        # u-head vector gates (official iMF style, replaces adaLN in u-head)
        u_head_vector_gates: bool = False,
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

        
        self.max_dudt_norm = max_dudt_norm
        self.u_head_vector_gates = u_head_vector_gates
        self._setup_dit_components_meanflow(
            dit_dim, n_heads, n_layers, action_dim, act_window_size, hidden_dim,
            attn_pdrop, resid_pdrop, mlp_pdrop, use_cross_attn,
            use_rope, use_nope, query_seq_len, rope_theta,
            aux_head_depth=aux_head_depth,
            u_head_vector_gates=u_head_vector_gates,
        )
        # Mean Flow specific parameters
        self.noise_dist = noise_dist
        self.data_proportion = data_proportion
        self.register_buffer(
            "P_mean",
            torch.tensor(P_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "P_std",
            torch.tensor(P_std, dtype=torch.float32)
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
        rope_theta: float,
        aux_head_depth: int = 8,
        u_head_vector_gates: bool = False,
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

        # Set up shared AdaLN if not using action-specific AdaLN
        if not self.action_type_adaln:
            self.adaln = SharedAdaLNController(
                dit_dim,
                global_conddim=dit_dim,
                use_cross_attn=use_cross_attn
            )

        # Set up shared conditioning components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
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

        # Set up DiT blocks: shared backbone + separate u-head and v-head
        shared_depth = n_layers - aux_head_depth
        assert shared_depth >= 0, f"aux_head_depth ({aux_head_depth}) must be <= n_layers ({n_layers})"
        self.aux_head_depth = aux_head_depth

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

        self.shared_blocks = nn.ModuleList([
            FlowBlock(**block_kwargs) for _ in range(shared_depth)
        ])
        self.u_head_blocks = nn.ModuleList([
            FlowBlock(**block_kwargs, use_vector_gates=u_head_vector_gates)
            for _ in range(aux_head_depth)
        ])
        self.v_head_blocks = nn.ModuleList([
            FlowBlock(**block_kwargs) for _ in range(aux_head_depth)
        ])

        # Token embedders for vector gates mode (tokenized conditioning)
        if u_head_vector_gates:
            self.cond_t_token_embedder = TimestepEmbedder(hidden_size=dit_dim)
            self.cond_h_token_embedder = TimestepEmbedder(hidden_size=dit_dim)

        # v-head decoders (one per action space, like u-head)
        self.v_action_decoders = nn.ModuleDict()
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            self.v_action_decoders[action_name] = MeanFlowDecoder(
                dit_dim=dit_dim,
                action_dim=input_dim,
                hidden_dim=dit_dim * 2
            ).to(self.device)

        # self.dit: all DiT blocks (shared + u-head + v-head) for grad clipping/EMA
        self.dit = nn.ModuleList(
            list(self.shared_blocks) + list(self.u_head_blocks) + list(self.v_head_blocks)
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
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim])
        return encoded, valid_dims

    def decode_actions_meanflow(
        self, z: torch.Tensor, h: torch.Tensor,
        action_type: torch.Tensor, valid_dims: torch.Tensor
    ) -> torch.Tensor:
        """
        Decodes latent representations into actions using MeanFlowDecoder.
        The decoder is conditioned on h = t - r.

        Args:
            z: DiT latent features [B, T, dit_dim]
            h: Timestep difference (t - r), broadcastable shape.
            action_type: Action type indices [B] or broadcastable.
            valid_dims: Valid dimensions mask [B, T, action_dim].
        """
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=self.device, dtype=z.dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                # MeanFlowDecoder takes (z, h)
                if mask.all():
                    pred = self.action_decoders[action_name](z, h)
                else:
                    # Slice h for masked batch elements
                    h_masked = h[mask] if h.dim() >= 1 and h.shape[0] == B else h
                    pred = self.action_decoders[action_name](z[mask], h_masked)
                decoded[mask, :, :adim] = pred[..., :adim] * valid_dims[mask, :, :adim]
        return decoded

    def decode_v(
        self, z: torch.Tensor, h: torch.Tensor,
        action_type: torch.Tensor, valid_dims: torch.Tensor
    ) -> torch.Tensor:
        """Decodes v-head latents into action-space predictions."""
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=self.device, dtype=z.dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                if mask.all():
                    pred = self.v_action_decoders[action_name](z, h)
                else:
                    h_masked = h[mask] if h.dim() >= 1 and h.shape[0] == B else h
                    pred = self.v_action_decoders[action_name](z[mask], h_masked)
                decoded[mask, :, :adim] = pred[..., :adim] * valid_dims[mask, :, :adim]
        return decoded

    # === Loss Functions ===
    def meanflow_loss(self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the Improved Mean Flow (iMF) loss using JVP.
        Based on: Geng, Lu et al. (2025) "Improved Mean Flows" (arXiv:2512.02012)

        Key difference from original MF: the training target is (e - x),
        which is network-independent, eliminating bootstrap instability.
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

        # Cast to float32 for JVP — dual tensors must have matching dtype
        # throughout. RmsNorm's JVP promotes tangents to float32, so if primals
        # are bf16 we get mixed-dtype duals that crash F.linear.
        z = z.float()
        v = v.float()
        texp = texp.float()
        rexp = rexp.float()

        # Define network function for JVP — returns u only (v computed separately)
        def u_func(z_input, t_input, r_input):
            h_input = t_input - r_input
            t_flat = t_input.view(-1)
            h_flat = h_input.view(-1)
            return self.dit_forward_meanflow(z_input, t_flat, h_flat, cond, return_v=False,
                                            detach_time_cond=True)

        # v_cond_fn: get v-head prediction with h=0 (for JVP tangent)
        def v_cond_fn(z_input, t_input):
            t_flat = t_input.view(-1)
            h_zero = torch.zeros_like(t_flat)
            # v_only=True skips u-head, only runs shared + v-head
            return self.dit_forward_meanflow(z_input, t_flat, h_zero, cond, v_only=True)

        # Tangent vectors for JVP (float32 to match)
        dtdt = torch.ones_like(texp) * 1e-3
        drdt = torch.zeros_like(rexp)

        # Compute u and du/dt using JVP
        # Monkey-patch nn.Linear and RmsNorm to cast weights to input dtype
        # during JVP, because torch.func.jvp dual tensors' .to(dtype) only
        # casts the primal, not the tangent — so we cast weights instead.
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
                # Get v-head prediction at h=0 for JVP tangent
                # no_grad: we only need the value for the tangent direction,
                # no need to retain the computation graph (saves ~12 blocks of activations)
                with torch.no_grad():
                    v_c = v_cond_fn(z, texp)

                # JVP only through shared + u-head (NOT v-head)
                # v_head blocks are excluded from JVP to save memory (~8 fewer
                # blocks with dual tensor overhead)
                u_pred, dudt = torch.func.jvp(
                    u_func,
                    (z, texp, rexp),
                    (v_c, dtdt, drdt),
                )

                # Compute v_pred separately outside JVP (with gradient for v-head loss)
                # v_only=True skips u-head, only runs shared + v-head blocks
                t_flat = texp.view(-1)
                h_flat = (texp - rexp).view(-1)
                v_pred = self.dit_forward_meanflow(z, t_flat, h_flat, cond, v_only=True)
            finally:
                nn.Linear.forward = _orig_linear_forward
                RmsNorm.forward = _orig_rmsnorm_forward

            # Clip dudt per-sample norm to prevent V domination
            # (analogous to gradient clipping — bounds the correction term)
            dudt_raw = dudt.detach()  # save raw for logging
            dudt_norms = dudt.flatten(1).norm(dim=1)  # [B]
            clip_scale = (self.max_dudt_norm / dudt_norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]
            dudt = dudt * clip_scale.view(-1, *([1] * (dudt.dim() - 1)))

            # Compound function V (iMF Eq. 9)
            h = (texp - rexp).clamp(min=0.0, max=1.0)
            V = u_pred + h * dudt.detach()

            # Stop gradient on target
            v_target = v.detach()

            # Build valid mask
            valid_mask = torch.zeros_like(actions, dtype=torch.bool)
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                mask = (action_type == action_idx)
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    mask_expanded = mask.view(-1, 1, 1).expand(-1, actions.size(1), adim).to(device)
                    valid_mask[mask, :, :adim] = mask_expanded[mask]

            # iMF u-loss: ||V - (e - x)||^2
            diff_u = V - v_target
            diff_u = diff_u * valid_mask.to(dtype=default_dtype)
            loss_u = (diff_u ** 2).sum(dim=(1, 2))

            # Auxiliary v-head loss: ||v_pred - (e - x)||^2
            diff_v = v_pred - v_target
            diff_v = diff_v * valid_mask.to(dtype=default_dtype)
            loss_v = (diff_v ** 2).sum(dim=(1, 2))

            # Adaptive weighting: normalizes loss to ~1.0 per sample.
            norm_eps = 0.01
            norm_p = 1.0
            adp_wt_u = (loss_u.detach() + norm_eps) ** norm_p
            loss_u = loss_u / adp_wt_u
            adp_wt_v = (loss_v.detach() + norm_eps) ** norm_p
            loss_v = loss_v / adp_wt_v

            loss = (loss_u + loss_v).mean()

        # Monitor metrics
        with torch.no_grad():
            valid_V = V[valid_mask]
            valid_v = v_target[valid_mask]
            valid_vpred = v_pred[valid_mask]
            v_loss = ((valid_V - valid_v) ** 2).mean()  # ||V - (e-x)||^2
            v_aux_loss = ((valid_vpred - valid_v) ** 2).mean()  # ||v_pred - (e-x)||^2
            # Track du/dt magnitude (raw, pre-clip) — if this vanishes, the model
            # degenerates to standard flow and single-step sampling will fail.
            dudt_norm = dudt_raw.flatten(1).norm(dim=1).mean()
            dudt_clip_frac = (dudt_norms > self.max_dudt_norm).float().mean()

        # Check for NaN/Inf in outputs
        if torch.isnan(V).any() or torch.isinf(V).any():
            logger.warning(f"NaN/Inf detected in V! "
                           f"V stats: min={V.min().item():.4f}, max={V.max().item():.4f}, "
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
            "v_loss": v_loss.item() if not (torch.isnan(v_loss).any() or torch.isinf(v_loss).any()) else 1e6,
            "v_aux_loss": v_aux_loss.item() if not (torch.isnan(v_aux_loss).any() or torch.isinf(v_aux_loss).any()) else 1e6,
            "dudt_norm": dudt_norm.item(),
            "dudt_clip_frac": dudt_clip_frac.item(),
            "h_mean": h.mean().item(),
        }

        if hasattr(self, 'accelerator') and self.accelerator is not None and wandb.run is not None:
            if self.accelerator.is_main_process:
                wandb.log(losses_dict)

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
            bz, 1, 1, 1,
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
            bz, 1, 1, 1,
            device=self.P_mean.device,
            dtype=next(self.parameters()).dtype
        )

    def sample_tr(self, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample timesteps t and r with constraint t >= r.
        For data_proportion of samples, set r = t (instantaneous velocity).

        Returns:
            t: Sampled timesteps [B, 1, 1]
            r: Sampled timesteps [B, 1, 1]
        """
        dtype = next(self.parameters()).dtype # Get model dtype (bf16)

        # Ensure these are cast
        t = self.noise_distribution()(b).to(device=self.device, dtype=dtype)
        r = self.noise_distribution()(b).to(device=self.device, dtype=dtype)
        
        # Ensure t >= r element-wise
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        data_size = int(b * self.data_proportion)
        zero_mask = torch.arange(b, device=t.device) < data_size
        zero_mask = zero_mask.view(b, 1, 1, 1)

        r = torch.where(zero_mask, t, r)
        return t, r

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
        # z_0 = z_1 - u(z_1, t=1, h=1)
        dtype = next(self.parameters()).dtype
        z = z.to(dtype=dtype)
        t_tensor = torch.ones(b, device=device, dtype=dtype)
        h_tensor = torch.ones(b, device=device, dtype=dtype)
        u = self.dit_forward_meanflow(z, t_tensor, h_tensor, cond, return_v=False)
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
            losses_dict = {}
            total_loss = torch.tensor(0.0, device=self.device)
            num_action_types = 0
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                mask = (action_type == action_idx)
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    space_loss = F.mse_loss(
                        action_pred[mask, :, :adim],
                        target_actions[mask, :, :adim],
                        reduction='mean'
                    )
                    losses_dict[f"val_loss_{action_name}"] = space_loss.item()
                    total_loss += space_loss
                    num_action_types += 1
            avg_loss = total_loss
            return {
                "loss": avg_loss.detach(),
                "losses": losses_dict,
                "dataset_index": batch['task'].get('dataset_index', torch.zeros(action_pred.shape[0], device=self.device)).detach()
            }

    def dit_forward_meanflow(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor,
                              cond_dict: dict, return_v: bool = True,
                              v_only: bool = False, detach_time_cond: bool = False):
        """
        Forward pass through the DiT blocks using MeanFlowDecoder.

        Modes:
          - return_v=False, v_only=False: returns u only (used inside JVP)
          - return_v=True, v_only=False: returns (u, v_pred)
          - v_only=True: returns v_pred only, skipping u-head (memory efficient)

        Args:
            z: Latent actions [B, T, action_dim]
            t: Current timestep [B]
            h: Timestep difference (t - r) [B]
            cond_dict: Conditioning dictionary
            return_v: If True, also compute v-head output (for iMF training)
            v_only: If True, only compute v-head output, skip u-head entirely
            detach_time_cond: If True, detach t before t_embedder so JVP tangent
                doesn't include ∂u/∂t through the deep adaLN chain. The official
                iMF (Geng et al.) uses token-based conditioning with vector gates
                (no adaLN), so ∂u/∂t doesn't propagate through blocks. This flag
                achieves the same effect with our adaLN architecture.
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

        # Compute temporal embedding
        # When detach_time_cond=True (during JVP), detach t so the JVP tangent
        # doesn't include ∂u/∂t through the adaLN chain. This prevents dudt
        # explosion from high-frequency sinusoidal embeddings amplified through
        # all DiT blocks. The primal value is unchanged; only the tangent is zeroed.
        t_for_emb = t.detach() if detach_time_cond else t
        t_emb = sum(map(stateless_norm, [self.t_embedder(t_for_emb), freq_embeds, proprio_embeds]))

        # Compute global conditioning
        if self.use_adaln_cond:
            global_cond = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond += t_emb
        else:
            global_cond = t_emb

        context = cond if self.use_cross_attn else None

        # Compute AdaLN modulation
        global_adaln = self.adaln(global_cond) if not self.action_type_adaln else self.action_specific_adaln(global_cond, action_type)

        # v-head kwargs: standard adaLN + causal (no tokens)
        v_attn_kwargs = dict(context=context, custom_attn_mask=None,
                             custom_cross_attn_mask=cond_dict['attention_mask'],
                             is_causal=True, global_adaln=global_adaln)

        if self.u_head_vector_gates:
            # Prepend t/h tokens BEFORE shared blocks
            t_token = self.cond_t_token_embedder(t).unsqueeze(1)   # [B, 1, dim]
            h_token = self.cond_h_token_embedder(h).unsqueeze(1)   # [B, 1, dim]
            n_cond = 2
            z = torch.cat([t_token, h_token, z], dim=1)            # [B, T+2, dim]
            T_action = z.shape[1] - n_cond

            # RoPE position_ids: cond tokens [0,1], action tokens [0..T-1]
            position_ids = torch.cat([
                torch.arange(n_cond, device=z.device),
                torch.arange(T_action, device=z.device),
            ])  # [T+2]

            # Custom attention mask (True=attend, False=block)
            T_total = z.shape[1]
            action_size = T_total - n_cond
            mask = torch.zeros(T_total, T_total, dtype=torch.bool, device=z.device)
            mask[:, :n_cond] = True                  # all positions attend to cond tokens
            mask[:n_cond, :] = True                  # cond tokens attend to everything
            mask[n_cond:, n_cond:] = ~torch.triu(    # action tokens: causal among themselves
                torch.ones(action_size, action_size, dtype=torch.bool, device=z.device),
                diagonal=1
            )
            mask = mask.unsqueeze(0)  # [1, T+2, T+2]

            tok_attn_kwargs = dict(context=context, custom_attn_mask=mask,
                                   custom_cross_attn_mask=cond_dict['attention_mask'],
                                   is_causal=False, global_adaln=global_adaln,
                                   position_ids=position_ids)

            # Shared backbone (with tokens)
            for layer in self.shared_blocks:
                z = layer(z, global_cond, **tok_attn_kwargs)

            if v_only:
                z_v = z[:, n_cond:, :]  # strip tokens
                for layer in self.v_head_blocks:
                    z_v = layer(z_v, global_cond, **v_attn_kwargs)
                return self.decode_v(z_v, h, action_type, valid_dims)

            # u-head (continues with tokens)
            z_u = z
            for layer in self.u_head_blocks:
                z_u = layer(z_u, global_cond, **tok_attn_kwargs)
            z_u = z_u[:, n_cond:, :]  # strip tokens
            u = self.decode_actions_meanflow(z_u, h, action_type, valid_dims)

            if not return_v:
                return u

            # v-head (strip tokens, standard adaLN)
            z_v = z[:, n_cond:, :]
            for layer in self.v_head_blocks:
                z_v = layer(z_v, global_cond, **v_attn_kwargs)
            v_pred = self.decode_v(z_v, h, action_type, valid_dims)
            return u, v_pred

        else:
            # Standard adaLN path (no vector gates)
            attn_kwargs = v_attn_kwargs
            for layer in self.shared_blocks:
                z = layer(z, global_cond, **attn_kwargs)

            if v_only:
                z_v = z
                for layer in self.v_head_blocks:
                    z_v = layer(z_v, global_cond, **attn_kwargs)
                return self.decode_v(z_v, h, action_type, valid_dims)

            z_u = z
            for layer in self.u_head_blocks:
                z_u = layer(z_u, global_cond, **attn_kwargs)
            u = self.decode_actions_meanflow(z_u, h, action_type, valid_dims)

            if not return_v:
                return u

            z_v = z
            for layer in self.v_head_blocks:
                z_v = layer(z_v, global_cond, **attn_kwargs)
            v_pred = self.decode_v(z_v, h, action_type, valid_dims)
            return u, v_pred

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