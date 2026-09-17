from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Use CUDA when explicitly available, otherwise remain CPU-compatible."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
