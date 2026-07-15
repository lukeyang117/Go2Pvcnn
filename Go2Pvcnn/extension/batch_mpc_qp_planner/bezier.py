"""Fixed-shape Bezier helpers for the continuous MPC-QP backend."""

from __future__ import annotations

import torch
from torch import Tensor


def cubic_bezier_basis(sample_count: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    count = max(2, int(sample_count))
    s = torch.linspace(0.0, 1.0, count, dtype=dtype, device=device)
    one = 1.0 - s
    return torch.stack((one * one * one, 3.0 * one * one * s, 3.0 * one * s * s, s * s * s), dim=-1)


def sample_cubic_bezier(control_points: Tensor, basis: Tensor) -> Tensor:
    if control_points.shape[-2:] != (4, 3):
        raise ValueError(f"control_points must end with [4,3], got {tuple(control_points.shape)}")
    if basis.ndim != 2 or int(basis.shape[-1]) != 4:
        raise ValueError(f"basis must have shape [S,4], got {tuple(basis.shape)}")
    return torch.einsum(
        "sc,...cd->...sd",
        basis.to(dtype=control_points.dtype, device=control_points.device),
        control_points,
    )


def trajectory_frame_deltas(samples: Tensor) -> tuple[Tensor, Tensor]:
    first = samples[..., 1:, :] - samples[..., :-1, :]
    second = first[..., 1:, :] - first[..., :-1, :]
    return first, second


__all__ = ["cubic_bezier_basis", "sample_cubic_bezier", "trajectory_frame_deltas"]
