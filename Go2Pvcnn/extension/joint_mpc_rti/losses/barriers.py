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


def relaxed_barrier_derivative(margin: Tensor, *, relaxation: float) -> Tensor:
    """Analytic derivative of :func:`relaxed_barrier` with respect to its margin."""
    value = torch.as_tensor(margin)
    delta = value.new_tensor(float(relaxation))
    if float(relaxation) <= 0.0:
        raise ValueError("relaxation must be positive")
    logarithmic = -torch.reciprocal(value.clamp_min(delta))
    continuation = (value - 2.0 * delta) / (delta * delta)
    return torch.where(value >= delta, logarithmic, continuation)


def localized_relaxed_barrier(
    margin: Tensor,
    *,
    activation_margin: float,
    relaxation: float,
) -> Tensor:
    """C1 penalty that is zero outside a finite relaxed-barrier influence band."""
    value = torch.as_tensor(margin)
    cutoff = relaxed_barrier(value.new_tensor(float(activation_margin)), relaxation=relaxation)
    gap = torch.relu(relaxed_barrier(value, relaxation=relaxation) - cutoff)
    return gap * gap


def localized_relaxed_barrier_derivative(
    margin: Tensor,
    *,
    activation_margin: float,
    relaxation: float,
) -> Tensor:
    value = torch.as_tensor(margin)
    cutoff = relaxed_barrier(value.new_tensor(float(activation_margin)), relaxation=relaxation)
    gap = torch.relu(relaxed_barrier(value, relaxation=relaxation) - cutoff)
    return 2.0 * gap * relaxed_barrier_derivative(value, relaxation=relaxation)


def masked_mean(value: Tensor, mask: Tensor | None = None, *, dims: tuple[int, ...]) -> Tensor:
    tensor = torch.as_tensor(value)
    if mask is None:
        return tensor.mean(dim=dims)
    weight = torch.as_tensor(mask, dtype=tensor.dtype, device=tensor.device)
    numerator = (tensor * weight).sum(dim=dims)
    denominator = weight.sum(dim=dims).clamp_min(1.0)
    return numerator / denominator


__all__ = [
    "localized_relaxed_barrier",
    "localized_relaxed_barrier_derivative",
    "masked_mean",
    "relaxed_barrier",
    "relaxed_barrier_derivative",
]
