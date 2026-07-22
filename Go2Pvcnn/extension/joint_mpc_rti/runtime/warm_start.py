"""Tensor-only shifted trajectory helpers for rolling RTI."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def measurement_decay(
    nodes: int,
    decay_nodes: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if decay_nodes < 1:
        raise ValueError("decay_nodes must be positive")
    node = torch.arange(nodes, dtype=dtype, device=device)
    return (1.0 - node / float(decay_nodes)).clamp(0.0, 1.0)


def shift_rebase_trajectory(previous: Tensor, measured: Tensor, *, decay_nodes: int) -> Tensor:
    """Shift accepted Z, rebase root in SE(2), and decay measured joint mismatch."""
    trajectory = torch.as_tensor(previous)
    measured_state = torch.as_tensor(measured, dtype=trajectory.dtype, device=trajectory.device)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (31, 18):
        raise ValueError("previous must have shape [B,31,18]")
    if measured_state.shape != trajectory[:, 0].shape:
        raise ValueError("measured must have shape [B,18]")

    terminal_root = trajectory[:, -1, :6] + (trajectory[:, -1, :6] - trajectory[:, -2, :6])
    terminal = torch.cat((terminal_root, trajectory[:, -1, 6:]), dim=-1)
    shifted = torch.cat((trajectory[:, 1:], terminal[:, None]), dim=1)

    yaw_coordinate_delta = measured_state[:, 5] - shifted[:, 0, 5]
    delta_yaw = wrap_angle(yaw_coordinate_delta)
    relative_pos = shifted[..., :3] - shifted[:, :1, :3]
    cosine = torch.cos(delta_yaw)[:, None]
    sine = torch.sin(delta_yaw)[:, None]
    rebased_xy = measured_state[:, None, :2] + torch.stack(
        (
            cosine * relative_pos[..., 0] - sine * relative_pos[..., 1],
            sine * relative_pos[..., 0] + cosine * relative_pos[..., 1],
        ),
        dim=-1,
    )
    rebased_z = measured_state[:, None, 2] + relative_pos[..., 2]
    rebased_pos = torch.cat((rebased_xy, rebased_z[..., None]), dim=-1)
    rebased_rpy = torch.cat(
        (shifted[..., 3:5], (shifted[..., 5] + yaw_coordinate_delta[:, None])[..., None]),
        dim=-1,
    )

    beta = measurement_decay(
        31,
        decay_nodes,
        dtype=trajectory.dtype,
        device=trajectory.device,
    ).view(1, 31, 1)
    joint_mismatch = measured_state[:, 6:] - shifted[:, 0, 6:]
    rebased_joint = shifted[..., 6:] + beta * joint_mismatch[:, None]
    rebased = torch.cat((rebased_pos, rebased_rpy, rebased_joint), dim=-1)
    return torch.cat((measured_state[:, None], rebased[:, 1:]), dim=1)


__all__ = ["measurement_decay", "shift_rebase_trajectory", "wrap_angle"]
