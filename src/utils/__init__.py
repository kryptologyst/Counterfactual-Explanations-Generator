"""Core utilities for deterministic seeding and device management."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device with fallback.
    
    Returns:
        torch.device: CUDA, MPS (Apple Silicon), or CPU device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    """Get human-readable device name.
    
    Returns:
        str: Device name for display purposes.
    """
    device = get_device()
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name()})"
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return "CPU"
