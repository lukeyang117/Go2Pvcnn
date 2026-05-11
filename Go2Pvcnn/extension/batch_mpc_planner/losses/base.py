"""Shared loss helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def safe_mean(value: Tensor, *, dim=None) -> Tensor:
    if value.numel() == 0:
        return torch.zeros((), dtype=value.dtype, device=value.device)
    return value.mean(dim=dim)


def l2_norm(value: Tensor, *, dim: int = -1) -> Tensor:
    return torch.linalg.norm(value, dim=dim)


def zero_per_env(batch: int, *, like: Tensor) -> Tensor:
    return torch.zeros(batch, dtype=like.dtype, device=like.device)


__all__ = ["l2_norm", "safe_mean", "zero_per_env"]
