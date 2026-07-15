"""Height-field penetration and body/leg clearance barriers."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.losses.barriers import relaxed_barrier


def _clearance_cost(position_w: Tensor, height_w: Tensor, margin: float, relaxation: float) -> Tensor:
    position = torch.as_tensor(position_w)
    height = torch.as_tensor(height_w, dtype=position.dtype, device=position.device)
    clearance = position[..., 2] - height - float(margin)
    return relaxed_barrier(clearance, relaxation=relaxation).reshape(position.shape[0], -1).mean(dim=1)


def clearance_losses(
    *,
    foot_pos_w: Tensor,
    foot_height_w: Tensor,
    knee_pos_w: Tensor,
    knee_height_w: Tensor,
    shank_pos_w: Tensor,
    shank_height_w: Tensor,
    body_pos_w: Tensor,
    body_height_w: Tensor,
    swing_mask: Tensor,
    foot_penetration_margin: float = 0.0,
    knee_margin: float = 0.02,
    shank_margin: float = 0.015,
    body_margin: float = 0.04,
    barrier_relaxation: float = 0.01,
) -> dict[str, Tensor]:
    del swing_mask
    return {
        "foot_ground_penetration": _clearance_cost(
            foot_pos_w, foot_height_w, foot_penetration_margin, barrier_relaxation
        ),
        "knee_ground_clearance": _clearance_cost(knee_pos_w, knee_height_w, knee_margin, barrier_relaxation),
        "shank_ground_clearance": _clearance_cost(
            shank_pos_w, shank_height_w, shank_margin, barrier_relaxation
        ),
        "body_ground_clearance": _clearance_cost(body_pos_w, body_height_w, body_margin, barrier_relaxation),
    }


__all__ = ["clearance_losses"]
