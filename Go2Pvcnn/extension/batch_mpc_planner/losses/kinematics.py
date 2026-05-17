"""Kinematics regularization losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..kinematics import MpcLegPoints, fk_feet_from_joint_angles, fk_leg_points_from_joint_angles, solve_joint_angles_from_trajectory

_JOINT_LIMITS = torch.tensor(
    (
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
    ),
    dtype=torch.float32,
)


@dataclass(frozen=True)
class MpcKinematicsForLoss:
    joint_angles: Tensor
    leg_points: MpcLegPoints


def solve_ik_for_loss(root_pos: Tensor, root_rpy: Tensor, foot_pos: Tensor) -> Tensor:
    return solve_joint_angles_from_trajectory(root_pos, root_rpy, foot_pos, clamp_to_limits=False)


def evaluate_kinematics_for_loss(
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    *,
    clamp_to_limits: bool,
    shank_sample_count: int,
) -> MpcKinematicsForLoss:
    joint_angles = solve_joint_angles_from_trajectory(
        root_pos,
        root_rpy,
        foot_pos,
        clamp_to_limits=bool(clamp_to_limits),
    )
    leg_points = fk_leg_points_from_joint_angles(
        root_pos,
        root_rpy,
        joint_angles,
        shank_sample_count=int(shank_sample_count),
    )
    return MpcKinematicsForLoss(joint_angles=joint_angles, leg_points=leg_points)


def joint_limit_loss_from_root_foot(
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    *,
    joint_limit_margin_rad: float,
) -> Tensor:
    joint_angles = solve_ik_for_loss(root_pos, root_rpy, foot_pos)
    limits = _JOINT_LIMITS.to(device=joint_angles.device, dtype=joint_angles.dtype)
    lower = limits[:, 0].view(1, 1, -1) + float(joint_limit_margin_rad)
    upper = limits[:, 1].view(1, 1, -1) - float(joint_limit_margin_rad)
    over_lower = torch.relu(lower - joint_angles)
    over_upper = torch.relu(joint_angles - upper)
    return (over_lower.square() + over_upper.square()).mean(dim=(1, 2))


def ik_fk_residual_loss(
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    contact_prob: Tensor,
    *,
    contact_weight: float,
) -> Tensor:
    """Penalize foot targets that cannot be reproduced after IK + FK."""
    solved = solve_joint_angles_from_trajectory(root_pos, root_rpy, foot_pos, clamp_to_limits=True)
    fk_foot = fk_feet_from_joint_angles(root_pos, root_rpy, solved)
    residual = torch.linalg.vector_norm(fk_foot - foot_pos, dim=-1)
    base = residual.mean(dim=(1, 2))
    contact_w = contact_prob.to(dtype=residual.dtype)
    contact_mass = torch.clamp(contact_w.sum(dim=(1, 2)), min=1.0)
    contact = (contact_w * residual).sum(dim=(1, 2)) / contact_mass
    return base + float(contact_weight) * contact


__all__ = [
    "MpcKinematicsForLoss",
    "evaluate_kinematics_for_loss",
    "ik_fk_residual_loss",
    "joint_limit_loss_from_root_foot",
    "solve_ik_for_loss",
]
