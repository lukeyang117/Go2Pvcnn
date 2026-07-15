"""Root posture, joint nominal, and joint-limit terms."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.losses.barriers import relaxed_barrier


def posture_losses(
    *,
    root_pos_w: Tensor,
    root_rpy_w: Tensor,
    joint_pos: Tensor,
    joint_velocity: Tensor,
    support_height: Tensor,
    nominal_root_clearance: float,
    nominal_joint_pos: Tensor,
    joint_lower: Tensor,
    joint_upper: Tensor,
    joint_velocity_limit: Tensor,
    root_linear_velocity_b: Tensor | None = None,
    root_angular_velocity_b: Tensor | None = None,
    barrier_relaxation: float = 0.01,
) -> dict[str, Tensor]:
    root_pos = torch.as_tensor(root_pos_w)
    root_rpy = torch.as_tensor(root_rpy_w, dtype=root_pos.dtype, device=root_pos.device)
    joint = torch.as_tensor(joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    joint_vel = torch.as_tensor(joint_velocity, dtype=root_pos.dtype, device=root_pos.device)
    support = torch.as_tensor(support_height, dtype=root_pos.dtype, device=root_pos.device)
    nominal_joint = torch.as_tensor(nominal_joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    lower = torch.as_tensor(joint_lower, dtype=root_pos.dtype, device=root_pos.device)
    upper = torch.as_tensor(joint_upper, dtype=root_pos.dtype, device=root_pos.device)
    velocity_limit = torch.as_tensor(joint_velocity_limit, dtype=root_pos.dtype, device=root_pos.device)
    height_error = root_pos[..., 2] - support - float(nominal_root_clearance)
    joint_error = joint - nominal_joint
    lower_margin = joint - lower
    upper_margin = upper - joint
    velocity_margin = velocity_limit - torch.abs(joint_vel)
    batch = int(root_pos.shape[0])
    zero = root_pos.new_zeros((batch,))
    vertical_velocity = zero
    if root_linear_velocity_b is not None:
        linear = torch.as_tensor(root_linear_velocity_b, dtype=root_pos.dtype, device=root_pos.device)
        vertical_velocity = (linear[..., 2] ** 2).mean(dim=1)
    roll_pitch_rate = zero
    if root_angular_velocity_b is not None:
        angular = torch.as_tensor(root_angular_velocity_b, dtype=root_pos.dtype, device=root_pos.device)
        roll_pitch_rate = (angular[..., :2] ** 2).mean(dim=(1, 2))
    return {
        "root_support_height": (height_error * height_error).mean(dim=1),
        "root_roll_pitch": (root_rpy[..., :2] ** 2).mean(dim=(1, 2)),
        "root_vertical_velocity": vertical_velocity,
        "root_roll_pitch_rate": roll_pitch_rate,
        "joint_nominal_posture": (joint_error * joint_error).mean(dim=(1, 2)),
        "joint_position_limit_barrier": (
            relaxed_barrier(lower_margin, relaxation=barrier_relaxation)
            + relaxed_barrier(upper_margin, relaxation=barrier_relaxation)
        ).mean(dim=(1, 2)),
        "joint_velocity_limit_barrier": relaxed_barrier(
            velocity_margin, relaxation=barrier_relaxation
        ).mean(dim=(1, 2)),
    }


__all__ = ["posture_losses"]
