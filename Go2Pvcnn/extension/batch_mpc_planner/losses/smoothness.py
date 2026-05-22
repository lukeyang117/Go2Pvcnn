"""Smoothness losses for root and feet."""

from __future__ import annotations

import torch
from torch import Tensor


def root_smoothness_loss(root_pos: Tensor, root_rpy: Tensor) -> Tensor:
    if int(root_pos.shape[1]) < 2:
        return torch.zeros(root_pos.shape[0], dtype=root_pos.dtype, device=root_pos.device)
    dpos = root_pos[:, 1:] - root_pos[:, :-1]
    drpy = root_rpy[:, 1:] - root_rpy[:, :-1]
    return torch.linalg.norm(dpos, dim=-1).mean(dim=-1) + torch.linalg.norm(drpy, dim=-1).mean(dim=-1)


def foot_smoothness_loss(foot_pos: Tensor) -> Tensor:
    if int(foot_pos.shape[1]) < 2:
        return torch.zeros(foot_pos.shape[0], dtype=foot_pos.dtype, device=foot_pos.device)
    dfoot = foot_pos[:, 1:] - foot_pos[:, :-1]
    return torch.linalg.norm(dfoot, dim=-1).mean(dim=(1, 2))


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    weight = weight.to(dtype=value.dtype, device=value.device)
    return (value * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)


def foot_boundary_smoothness_loss(foot_pos: Tensor, swing_prob: Tensor) -> Tensor:
    if int(foot_pos.shape[1]) < 2:
        return torch.zeros(foot_pos.shape[0], dtype=foot_pos.dtype, device=foot_pos.device)
    step = torch.linalg.norm(foot_pos[:, 1:] - foot_pos[:, :-1], dim=-1)
    boundary_weight = torch.abs(swing_prob[:, 1:] - swing_prob[:, :-1])
    return _weighted_mean(step, boundary_weight)


def foot_acceleration_smoothness_loss(foot_pos: Tensor, swing_prob: Tensor) -> Tensor:
    if int(foot_pos.shape[1]) < 3:
        return torch.zeros(foot_pos.shape[0], dtype=foot_pos.dtype, device=foot_pos.device)
    accel = foot_pos[:, 2:] - 2.0 * foot_pos[:, 1:-1] + foot_pos[:, :-2]
    accel_norm = torch.linalg.norm(accel, dim=-1)
    swing_weight = swing_prob[:, 1:-1].clamp_min(0.0)
    return _weighted_mean(accel_norm, swing_weight)


__all__ = [
    "foot_acceleration_smoothness_loss",
    "foot_boundary_smoothness_loss",
    "foot_smoothness_loss",
    "root_smoothness_loss",
]
