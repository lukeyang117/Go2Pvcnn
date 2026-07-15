"""Analytic local linearization of the kinematic shooting dynamics."""

from __future__ import annotations

import torch
from torch import Tensor


def dynamics_jacobians(state: Tensor, control: Tensor, *, dt: float) -> tuple[Tensor, Tensor]:
    """Return exact Jacobians of ``kinematic_step`` with respect to state and control."""
    state_tensor = torch.as_tensor(state)
    control_tensor = torch.as_tensor(control, dtype=state_tensor.dtype, device=state_tensor.device)
    if state_tensor.ndim != 2 or int(state_tensor.shape[-1]) != 18 or control_tensor.shape != state_tensor.shape:
        raise ValueError("state and control must have shape [B,18]")
    batch = int(state_tensor.shape[0])
    matrix_a = torch.eye(18, dtype=state_tensor.dtype, device=state_tensor.device).unsqueeze(0).expand(batch, -1, -1).clone()
    matrix_b = torch.zeros(batch, 18, 18, dtype=state_tensor.dtype, device=state_tensor.device)
    step = float(dt)
    roll = state_tensor[:, 3]
    pitch = state_tensor[:, 4]
    yaw = state_tensor[:, 5]
    vx = control_tensor[:, 0]
    vy = control_tensor[:, 1]
    wy = control_tensor[:, 4]
    wz = control_tensor[:, 5]
    sine_yaw = torch.sin(yaw)
    cosine_yaw = torch.cos(yaw)
    matrix_a[:, 0, 5] = step * (-sine_yaw * vx - cosine_yaw * vy)
    matrix_a[:, 1, 5] = step * (cosine_yaw * vx - sine_yaw * vy)
    matrix_b[:, 0, 0] = step * cosine_yaw
    matrix_b[:, 0, 1] = -step * sine_yaw
    matrix_b[:, 1, 0] = step * sine_yaw
    matrix_b[:, 1, 1] = step * cosine_yaw
    matrix_b[:, 2, 2] = step
    sine_roll = torch.sin(roll)
    cosine_roll = torch.cos(roll)
    sine_pitch = torch.sin(pitch)
    cosine_pitch = torch.cos(pitch).clamp_min(1.0e-4)
    tangent_pitch = sine_pitch / cosine_pitch
    secant_sq = 1.0 / (cosine_pitch * cosine_pitch)
    matrix_a[:, 3, 3] += step * (cosine_roll * tangent_pitch * wy - sine_roll * tangent_pitch * wz)
    matrix_a[:, 3, 4] = step * (sine_roll * wy + cosine_roll * wz) * secant_sq
    matrix_a[:, 4, 3] = step * (-sine_roll * wy - cosine_roll * wz)
    matrix_a[:, 5, 3] = step * (cosine_roll * wy - sine_roll * wz) / cosine_pitch
    matrix_a[:, 5, 4] = step * (sine_roll * wy + cosine_roll * wz) * sine_pitch * secant_sq
    matrix_b[:, 3, 3] = step
    matrix_b[:, 3, 4] = step * sine_roll * tangent_pitch
    matrix_b[:, 3, 5] = step * cosine_roll * tangent_pitch
    matrix_b[:, 4, 4] = step * cosine_roll
    matrix_b[:, 4, 5] = -step * sine_roll
    matrix_b[:, 5, 4] = step * sine_roll / cosine_pitch
    matrix_b[:, 5, 5] = step * cosine_roll / cosine_pitch
    joint_identity = torch.eye(12, dtype=state_tensor.dtype, device=state_tensor.device).unsqueeze(0).expand(batch, -1, -1)
    matrix_b[:, 6:, 6:] = step * joint_identity
    return matrix_a, matrix_b


__all__ = ["dynamics_jacobians"]
