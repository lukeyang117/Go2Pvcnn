"""Tracking and progress losses."""

from __future__ import annotations

import torch
from torch import Tensor


def _rotate_world_xy_to_body(xy: Tensor, yaw: Tensor) -> Tensor:
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    return torch.stack((cy * xy[..., 0] + sy * xy[..., 1], -sy * xy[..., 0] + cy * xy[..., 1]), dim=-1)


def command_tracking_loss(
    root_pos: Tensor,
    root_rpy: Tensor,
    command: Tensor,
    dt: float,
    *,
    vel_weight: float = 1.0,
    yaw_weight: float = 1.0,
    linear_scale: Tensor | float | None = None,
    yaw_scale: Tensor | float | None = None,
) -> Tensor:
    if int(root_pos.shape[1]) < 2:
        return torch.zeros(root_pos.shape[0], dtype=root_pos.dtype, device=root_pos.device)
    cmd = torch.as_tensor(command, dtype=root_pos.dtype, device=root_pos.device)[:, :3]
    dxy_w = root_pos[:, 1:, :2] - root_pos[:, :-1, :2]
    yaw = root_rpy[:, :-1, 2]
    vel_b = _rotate_world_xy_to_body(dxy_w, yaw) / float(dt)
    yaw_rate = (root_rpy[:, 1:, 2] - root_rpy[:, :-1, 2]) / float(dt)
    xy_err = vel_b - cmd[:, None, :2]
    yaw_err = yaw_rate - cmd[:, None, 2]
    lin_term = torch.linalg.vector_norm(xy_err, dim=-1).mean(dim=1)
    yaw_term = torch.abs(yaw_err).mean(dim=1)
    if linear_scale is not None:
        lin_term = lin_term * torch.as_tensor(linear_scale, dtype=root_pos.dtype, device=root_pos.device).reshape(-1)
    if yaw_scale is not None:
        yaw_term = yaw_term * torch.as_tensor(yaw_scale, dtype=root_pos.dtype, device=root_pos.device).reshape(-1)
    return float(vel_weight) * lin_term + float(yaw_weight) * yaw_term


def progress_direction_loss(root_pos: Tensor, root_rpy: Tensor, command: Tensor, min_progress_m: float) -> Tensor:
    if int(root_pos.shape[1]) < 2:
        return torch.zeros(root_pos.shape[0], dtype=root_pos.dtype, device=root_pos.device)
    cmd = torch.as_tensor(command, dtype=root_pos.dtype, device=root_pos.device)[:, :2]
    desired = torch.linalg.vector_norm(cmd, dim=-1) > 1.0e-4
    dxy_w = root_pos[:, -1, :2] - root_pos[:, 0, :2]
    dxy_b = _rotate_world_xy_to_body(dxy_w, root_rpy[:, 0, 2])
    cmd_norm = torch.linalg.vector_norm(cmd, dim=-1).clamp_min(1.0e-6)
    heading = cmd / cmd_norm.unsqueeze(-1)
    progress = (dxy_b * heading).sum(dim=-1)
    return torch.where(desired, torch.relu(float(min_progress_m) - progress), torch.zeros_like(progress))


__all__ = ["command_tracking_loss", "progress_direction_loss"]
