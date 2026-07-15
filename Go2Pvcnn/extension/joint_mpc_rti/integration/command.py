"""Command-frame conversions for the joint MPC RTI boundary."""

from __future__ import annotations

import torch
from torch import Tensor


def body_linear_velocity_to_world(linear_velocity_b: Tensor, yaw_w: Tensor) -> Tensor:
    """Rotate batched body-frame XY velocities into the world frame exactly once."""
    velocity = torch.as_tensor(linear_velocity_b)
    yaw = torch.as_tensor(yaw_w, dtype=velocity.dtype, device=velocity.device)
    if velocity.ndim != 2 or int(velocity.shape[-1]) != 2:
        raise ValueError("linear_velocity_b must have shape [B,2]")
    if yaw.ndim != 1 or int(yaw.shape[0]) != int(velocity.shape[0]):
        raise ValueError("yaw_w must have shape [B]")
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    vx = velocity[:, 0]
    vy = velocity[:, 1]
    return torch.stack((cosine * vx - sine * vy, sine * vx + cosine * vy), dim=-1)


__all__ = ["body_linear_velocity_to_world"]
