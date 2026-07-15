"""Discrete root-and-joint kinematics used by the RTI shooting model."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.integration.command import body_linear_velocity_to_world


def _body_angular_velocity_to_rpy_rate(rpy_w: Tensor, angular_velocity_b: Tensor) -> Tensor:
    roll = rpy_w[:, 0]
    pitch = rpy_w[:, 1]
    wx = angular_velocity_b[:, 0]
    wy = angular_velocity_b[:, 1]
    wz = angular_velocity_b[:, 2]
    sin_roll = torch.sin(roll)
    cos_roll = torch.cos(roll)
    cos_pitch = torch.cos(pitch).clamp_min(1.0e-4)
    tan_pitch = torch.sin(pitch) / cos_pitch
    return torch.stack(
        (
            wx + sin_roll * tan_pitch * wy + cos_roll * tan_pitch * wz,
            cos_roll * wy - sin_roll * wz,
            sin_roll / cos_pitch * wy + cos_roll / cos_pitch * wz,
        ),
        dim=-1,
    )


def kinematic_step(state: Tensor, control: Tensor, *, dt: float) -> Tensor:
    """Integrate one fixed-size kinematic shooting interval."""
    state_tensor = torch.as_tensor(state)
    control_tensor = torch.as_tensor(control, dtype=state_tensor.dtype, device=state_tensor.device)
    if state_tensor.ndim != 2 or int(state_tensor.shape[-1]) != 18:
        raise ValueError("state must have shape [B,18]")
    if control_tensor.shape != state_tensor.shape:
        raise ValueError("control must have shape [B,18]")
    root_pos = state_tensor[:, :3]
    root_rpy = state_tensor[:, 3:6]
    joint_pos = state_tensor[:, 6:]
    linear_velocity_b = control_tensor[:, :3]
    angular_velocity_b = control_tensor[:, 3:6]
    joint_velocity = control_tensor[:, 6:]
    linear_velocity_w_xy = body_linear_velocity_to_world(linear_velocity_b[:, :2], root_rpy[:, 2])
    linear_velocity_w = torch.cat((linear_velocity_w_xy, linear_velocity_b[:, 2:3]), dim=-1)
    rpy_rate = _body_angular_velocity_to_rpy_rate(root_rpy, angular_velocity_b)
    step = state_tensor.new_tensor(float(dt))
    return torch.cat(
        (
            root_pos + step * linear_velocity_w,
            root_rpy + step * rpy_rate,
            joint_pos + step * joint_velocity,
        ),
        dim=-1,
    )


__all__ = ["kinematic_step"]
