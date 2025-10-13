"""Utilities for deterministic seeding across libraries."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def set_global_seeds(seed: int, numpy_seed: Optional[int] = None, torch_seed: Optional[int] = None) -> None:
    """Set python, numpy and torch seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(numpy_seed or seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(torch_seed or seed)
        torch.cuda.manual_seed_all(torch_seed or seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
