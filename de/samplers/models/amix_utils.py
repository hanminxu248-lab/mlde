"""Utility functions for loading AMix models."""
import sys
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import torch
import torch.nn.functional as F
import os


def load_amix_config(ckpt_path: str):
    """Load AMix model configuration from checkpoint path.
    
    Args:
        ckpt_path (str): Path to AMix checkpoint file.
        
    Returns:
        tuple: (cfg, root_path) - Configuration object and root path
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If checkpoint path is invalid
    """
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    
    root_path = Path(ckpt_path).parents[1]
    
    # Validate that root_path is a reasonable location before adding to sys.path
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Invalid checkpoint directory structure: {root_path}")
    
    # Only add to sys.path if not already present
    root_path_str = str(root_path)
    if root_path_str not in sys.path:
        sys.path.append(root_path_str)
    
    cfg_path = Path(root_path, ".hydra", "config.yaml")
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    
    ckpt_cfg = OmegaConf.load(cfg_path)
    ckpt_cfg.model.bfn.net.config._attn_implementation = 'sdpa'
    
    return ckpt_cfg, root_path


def load_amix_model(ckpt_path: str, device='cpu'):
    """Load AMix model from checkpoint.
    
    Args:
        ckpt_path (str): Path to AMix checkpoint file.
        device (str): Device to load model on.
        
    Returns:
        model: Loaded AMix model
        
    Raises:
        FileNotFoundError: If checkpoint or config file doesn't exist
        ValueError: If checkpoint path is invalid
    """
    ckpt_cfg, _ = load_amix_config(ckpt_path)
    
    model = hydra.utils.instantiate(ckpt_cfg.model)
    state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    del state_dict
    
    return model


def prepare_amix_inputs(input_ids: torch.Tensor, tokenizer, attention_mask=None):
    """Prepare inputs for AMix model by converting to one-hot embeddings.
    
    Args:
        input_ids (torch.Tensor): Token IDs of shape [batch, seq_len]
        tokenizer: Tokenizer with vocabulary
        attention_mask (torch.Tensor, optional): Attention mask
        
    Returns:
        tuple: (inputs_embeds, timestep, attention_mask) ready for AMix forward pass
        
    Note:
        This creates one-hot embeddings which are memory-intensive for large
        vocabularies. The AMix BFN architecture requires this format for
        discrete diffusion modeling.
    """
    # Create one-hot encoding for BFN input
    # Note: This is required by AMix's Bayesian Flow Network architecture
    inputs_embeds = F.one_hot(input_ids, num_classes=len(tokenizer)).float()
    
    # Set timestep to 1.0 for inference (fully denoised)
    if attention_mask is None:
        attention_mask = (input_ids != tokenizer.pad_token_id)
    t = torch.ones_like(attention_mask).float()
    
    return inputs_embeds, t, attention_mask
