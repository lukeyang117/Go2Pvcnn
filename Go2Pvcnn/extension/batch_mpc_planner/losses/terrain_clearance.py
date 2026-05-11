"""Terrain/clearance placeholder losses for batch MPC."""

from __future__ import annotations

import torch
from torch import Tensor


def swing_clearance_loss(foot_pos: Tensor, contact_prob: Tensor, min_clearance_m: float) -> Tensor:
    swing_weight = 1.0 - contact_prob
    z = foot_pos[..., 2]
    deficit = torch.relu(float(min_clearance_m) - z)
    return (deficit * swing_weight).mean(dim=(1, 2))


def terrain_clearance_loss(foot_pos: Tensor, min_clearance_m: float) -> Tensor:
    z = foot_pos[..., 2]
    return torch.relu(float(min_clearance_m) - z).mean(dim=(1, 2))


def obstacle_margin_loss(foot_pos: Tensor, body_margin_m: float, foot_margin_m: float) -> Tensor:
    z = foot_pos[..., 2]
    foot_penalty = torch.relu(float(foot_margin_m) - z)
    body_penalty = torch.relu(float(body_margin_m) - z.mean(dim=2))
    return foot_penalty.mean(dim=(1, 2)) + body_penalty.mean(dim=1)


__all__ = ["obstacle_margin_loss", "swing_clearance_loss", "terrain_clearance_loss"]
