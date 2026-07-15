"""Finite relaxed logarithmic barriers for inequality margins g(x) >= 0."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def relaxed_barrier(margin: Tensor, *, relaxation: float) -> Tensor:
    """Return a C1 continuation of ``-log(margin)`` below ``relaxation``."""
    value = torch.as_tensor(margin)
    delta = value.new_tensor(float(relaxation))
    if float(relaxation) <= 0.0:
        raise ValueError("relaxation must be positive")
    safe = -torch.log(value.clamp_min(delta))
    normalized = (value - 2.0 * delta) / delta
    continuation = 0.5 * normalized * normalized - 0.5 - math.log(float(relaxation))
    return torch.where(value >= delta, safe, continuation)


def masked_mean(value: Tensor, mask: Tensor | None = None, *, dims: tuple[int, ...]) -> Tensor:
    tensor = torch.as_tensor(value)
    if mask is None:
        return tensor.mean(dim=dims)
    weight = torch.as_tensor(mask, dtype=tensor.dtype, device=tensor.device)
    numerator = (tensor * weight).sum(dim=dims)
    denominator = weight.sum(dim=dims).clamp_min(1.0)
    return numerator / denominator


__all__ = ["masked_mean", "relaxed_barrier"]
