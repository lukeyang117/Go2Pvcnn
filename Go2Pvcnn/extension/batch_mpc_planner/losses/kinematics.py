"""Kinematics regularization losses."""

from __future__ import annotations

import torch
from torch import Tensor

_JOINT_LIMITS = torch.tensor(
    (
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
    ),
    dtype=torch.float32,
)

def joint_limit_loss(
    joint_angles: Tensor,
    *,
    joint_limit_rad: float,
    joint_limit_margin_rad: float,
) -> Tensor:
    # Legacy global abs-limit guardrail.
    over_abs = torch.relu(torch.abs(joint_angles) - float(joint_limit_rad))

    # Prefer staying away from true per-joint hardware limits by a small margin.
    limits = _JOINT_LIMITS.to(device=joint_angles.device, dtype=joint_angles.dtype)
    lower = limits[:, 0].view(1, 1, -1) + float(joint_limit_margin_rad)
    upper = limits[:, 1].view(1, 1, -1) - float(joint_limit_margin_rad)
    over_lower = torch.relu(lower - joint_angles)
    over_upper = torch.relu(joint_angles - upper)
    over_limits = over_lower + over_upper
    return (over_abs + over_limits).mean(dim=(1, 2))


__all__ = ["joint_limit_loss"]
