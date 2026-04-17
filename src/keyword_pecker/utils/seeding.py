"""Reproducibility utilities for setting random seeds."""

import random

import numpy as np
import torch


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Seeds the standard library ``random`` module, NumPy, and PyTorch (both
    CPU and all CUDA devices) and configures cuDNN for deterministic operation.

    Args:
        seed: Integer seed value. Must be non-negative.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
