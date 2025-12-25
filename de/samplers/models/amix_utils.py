"""Utility functions for loading AMix models."""
import sys
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import torch


def load_amix_config(ckpt_path: str):
    """Load AMix model configuration from checkpoint path.
    
    Args:
        ckpt_path (str): Path to AMix checkpoint file.
        
    Returns:
        tuple: (cfg, root_path) - Configuration object and root path
    """
    root_path = Path(ckpt_path).parents[1]
    sys.path.append(str(root_path))
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
    """
    ckpt_cfg, _ = load_amix_config(ckpt_path)
    
    model = hydra.utils.instantiate(ckpt_cfg.model)
    state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    del state_dict
    
    return model
