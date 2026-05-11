"""Tracking and progress losses."""

from __future__ import annotations

import torch
from torch import Tensor


def command_tracking_loss(root_pos: Tensor, root_rpy: Tensor, command: Tensor, dt: float) -> Tensor:
    cmd = command[:, :3]
    est_vx = (root_pos[:, -1, 0] - root_pos[:, 0, 0]) / (float(root_pos.shape[1] - 1) * dt)
    est_vy = (root_pos[:, -1, 1] - root_pos[:, 0, 1]) / (float(root_pos.shape[1] - 1) * dt)
    est_yaw_rate = (root_rpy[:, -1, 2] - root_rpy[:, 0, 2]) / (float(root_rpy.shape[1] - 1) * dt)
    err = torch.stack((est_vx - cmd[:, 0], est_vy - cmd[:, 1], est_yaw_rate - cmd[:, 2]), dim=-1)
    return torch.linalg.norm(err, dim=-1)


def progress_direction_loss(root_pos: Tensor, command: Tensor, min_progress_m: float) -> Tensor:
    delta_x = root_pos[:, -1, 0] - root_pos[:, 0, 0]
    desired = torch.where(command[:, 0] > 0.0, torch.full_like(delta_x, float(min_progress_m)), torch.zeros_like(delta_x))
    return torch.relu(desired - delta_x)


__all__ = ["command_tracking_loss", "progress_direction_loss"]
