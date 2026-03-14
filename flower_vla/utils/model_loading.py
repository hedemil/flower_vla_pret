import os
import logging
import torch
from safetensors.torch import load_file
from typing import Tuple, Dict, Union, Optional
from accelerate import Accelerator

log = logging.getLogger(__name__)

def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load model weights from a safetensors file."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No safetensors file found at {path}")
        return load_file(path)
    except Exception as e:
        log.error(f"Error loading safetensors file: {e}")
        raise

def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load model weights from a PyTorch state dict."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No state dict found at {path}")
        return torch.load(path, map_location='cpu')
    except Exception as e:
        log.error(f"Error loading state dict: {e}")
        raise

def load_model_weights(
    model: torch.nn.Module,
    weights_path: str,
    strict: bool = False,
    exclude_keys: Optional[list] = None
) -> Tuple[set, set]:
    """
    Load pre-trained model weights for finetuning.
    
    Args:
        model: Target model to load weights into
        weights_path: Path to weights file (.safetensors or .pt)
        strict: Whether to strictly enforce all keys match
        exclude_keys: List of key patterns to exclude from loading
    
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    try:
        # Determine file type and load weights
        if weights_path.endswith('.safetensors'):
            state_dict = load_safetensors(weights_path)
        else:
            state_dict = load_state_dict(weights_path)
            
        # Filter out excluded keys if specified
        if exclude_keys:
            state_dict = {k: v for k, v in state_dict.items() 
                         if not any(ex in k for ex in exclude_keys)}
        
        # Load weights
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        
        # Log results
        if missing:
            log.info(f"Missing keys: {missing}")
        if unexpected:
            log.info(f"Unexpected keys: {unexpected}")
            
        return missing, unexpected
        
    except Exception as e:
        log.error(f"Failed to load model weights: {e}")
        raise

def adjust_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    remove_prefix: str = None,
    add_prefix: str = None
) -> Dict[str, torch.Tensor]:
    """
    Adjust state dict keys by adding/removing prefixes.
    Useful when loading weights from different model versions.
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        if remove_prefix and key.startswith(remove_prefix):
            new_key = key[len(remove_prefix):]
            
        if add_prefix:
            new_key = f"{add_prefix}{new_key}"
            
        new_state_dict[new_key] = value
        
    return new_state_dict


def map_flower_to_meanflower(state_dict: Dict[str, torch.Tensor],
                              n_encoder_layers: int = 8) -> Dict[str, torch.Tensor]:
    """
    Map FlowerVLA (rectified flow) state_dict keys to MeanFlowerVLA (DMF) format.

    Key transformations:
      - dit.{i}.* → dit.encoder_blocks.{i}.* (if i < n_encoder_layers)
                   → dit.decoder_blocks.{i - n_enc}.* (otherwise)
      - adaln.modCX.* (shared, action_type_adaln=False)
                   → adaln_t.modCX.* + adaln_r.modCX.* (duplicated)
      - adaln.{action_name}.* (per-action, action_type_adaln=True)
                   → kept as-is (MeanFlowerVLA uses same adaln ModuleDict)
      - t_embedder.* → kept + copied to r_embedder.*
      - logvar_linear.* is zero-init'd by the model constructor, no mapping needed

    Args:
        state_dict: FlowerVLA state dict
        n_encoder_layers: Number of layers to assign to encoder (rest go to decoder)

    Returns:
        New state dict with DMF key naming
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('dit.'):
            parts = k.split('.')
            try:
                layer_idx = int(parts[1])
            except (ValueError, IndexError):
                # Not a numbered layer (e.g. dit.some_other_attr), keep as-is
                new_sd[k] = v
                continue
            rest = '.'.join(parts[2:])
            # Rename MLP keys: FLOWER uses c_fc1/c_fc2/c_proj, DMF uses fc1/fc2/proj
            rest = rest.replace('mlp.c_fc1', 'mlp.fc1') \
                       .replace('mlp.c_fc2', 'mlp.fc2') \
                       .replace('mlp.c_proj', 'mlp.proj')
            if layer_idx < n_encoder_layers:
                new_sd[f'dit.encoder_blocks.{layer_idx}.{rest}'] = v
            else:
                new_sd[f'dit.decoder_blocks.{layer_idx - n_encoder_layers}.{rest}'] = v
        elif k.startswith('adaln.'):
            # Detect shared vs per-action AdaLN:
            # Shared (action_type_adaln=False): adaln.modCX.{i}.weight
            # Per-action (action_type_adaln=True): adaln.{action_name}.modCX.{i}.weight
            parts_after = k[len('adaln.'):].split('.')
            if parts_after[0] == 'modCX':
                # Shared AdaLN → duplicate to adaln_t and adaln_r
                suffix = k[len('adaln.'):]
                new_sd[f'adaln_t.{suffix}'] = v.clone()
                new_sd[f'adaln_r.{suffix}'] = v.clone()
            else:
                # Per-action AdaLN → keep as-is (MeanFlowerVLA has same adaln ModuleDict)
                new_sd[k] = v
        elif k.startswith('t_embedder.'):
            new_sd[k] = v
            new_sd[k.replace('t_embedder.', 'r_embedder.')] = v.clone()
        else:
            new_sd[k] = v

    log.info(f"Mapped FlowerVLA → MeanFlowerVLA: {len(state_dict)} → {len(new_sd)} keys "
             f"(encoder={n_encoder_layers} layers)")
    return new_sd


def load_pretrained_weights(
    model: torch.nn.Module,
    weights_path: str,
    map_type: Optional[str] = None,
    n_encoder_layers: int = 8,
    strict: bool = False,
    exclude_keys: Optional[list] = None,
) -> Tuple[set, set]:
    """
    Load pre-trained weights with optional key mapping.

    Args:
        model: Target model
        weights_path: Path to weights file (.safetensors or .pt)
        map_type: Key mapping to apply before loading.
                  'flower_to_dmf' — maps FlowerVLA keys to DMF format.
                  None — load as-is.
        n_encoder_layers: Encoder layer count (used by flower_to_dmf mapping)
        strict: Strict loading
        exclude_keys: Key patterns to exclude
    """
    if weights_path.endswith('.safetensors'):
        state_dict = load_safetensors(weights_path)
    else:
        state_dict = load_state_dict(weights_path)

    if exclude_keys:
        state_dict = {k: v for k, v in state_dict.items()
                      if not any(ex in k for ex in exclude_keys)}

    if map_type == 'flower_to_dmf':
        state_dict = map_flower_to_meanflower(state_dict, n_encoder_layers)
    elif map_type is not None:
        raise ValueError(f"Unknown map_type: {map_type}")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        log.info(f"Missing keys: {missing}")
    if unexpected:
        log.info(f"Unexpected keys: {unexpected}")
    return missing, unexpected