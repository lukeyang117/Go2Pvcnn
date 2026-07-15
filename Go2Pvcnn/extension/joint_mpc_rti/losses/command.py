"""Command tracking and horizon progress residuals."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.integration.command import body_linear_velocity_to_world


def command_losses(
    root_pos_w: Tensor,
    root_rpy_w: Tensor,
    control: Tensor,
    command_body: Tensor,
    *,
    dt: float,
) -> dict[str, Tensor]:
    root_pos = torch.as_tensor(root_pos_w)
    root_rpy = torch.as_tensor(root_rpy_w, dtype=root_pos.dtype, device=root_pos.device)
    controls = torch.as_tensor(control, dtype=root_pos.dtype, device=root_pos.device)
    command = torch.as_tensor(command_body, dtype=root_pos.dtype, device=root_pos.device)
    linear_error = controls[..., :2] - command[:, None, :2]
    yaw_error = controls[..., 5] - command[:, None, 2]
    command_velocity = (linear_error * linear_error).mean(dim=(1, 2))
    command_yaw = (yaw_error * yaw_error).mean(dim=1)
    horizon_duration = float(dt) * float(controls.shape[1])
    desired_world_velocity = body_linear_velocity_to_world(command[:, :2], root_rpy[:, 0, 2])
    desired_progress = horizon_duration * desired_world_velocity
    actual_progress = root_pos[:, -1, :2] - root_pos[:, 0, :2]
    progress_error = actual_progress - desired_progress
    progress_loss = (progress_error * progress_error).sum(dim=-1)
    command_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    actual_norm = torch.linalg.vector_norm(actual_progress, dim=-1)
    direction_cosine = (actual_progress * desired_world_velocity).sum(dim=-1) / (
        actual_norm * command_speed + 1.0e-6
    )
    active = command_speed > 1.0e-4
    direction_loss = torch.where(active, (1.0 - direction_cosine.clamp(-1.0, 1.0)) ** 2, torch.zeros_like(command_speed))
    return {
        "command_linear_velocity": command_velocity,
        "command_yaw_rate": command_yaw,
        "command_progress": progress_loss,
        "command_direction": direction_loss,
    }


__all__ = ["command_losses"]
