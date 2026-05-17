"""GPU batched IK helper for dense MPC planner outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.batched_planner.types import CALF_LENGTH, HIP_OFFSETS_ARRAY, HIP_OFFSET_Y, THIGH_LENGTH

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
_LEG_SIDE_SIGNS = torch.tensor((1.0, -1.0, 1.0, -1.0), dtype=torch.float32)


@dataclass(frozen=True)
class MpcLegPoints:
    foot_pos_world: Tensor
    knee_pos_world: Tensor
    shank_sample_world: Tensor


def _rpy_to_rot_matrix(root_rpy: Tensor) -> Tensor:
    roll = root_rpy[..., 0]
    pitch = root_rpy[..., 1]
    yaw = root_rpy[..., 2]
    cr = torch.cos(0.5 * roll)
    sr = torch.sin(0.5 * roll)
    cp = torch.cos(0.5 * pitch)
    sp = torch.sin(0.5 * pitch)
    cy = torch.cos(0.5 * yaw)
    sy = torch.sin(0.5 * yaw)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    row0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def solve_joint_angles_from_trajectory(
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos_w: Tensor,
    *,
    clamp_to_limits: bool = True,
) -> Tensor:
    """Solve per-frame Go2 leg IK from world-frame root pose and foot targets."""
    if root_pos.ndim != 3 or int(root_pos.shape[-1]) != 3:
        raise ValueError("root_pos must have shape [B, T, 3]")
    if tuple(root_rpy.shape) != tuple(root_pos.shape):
        raise ValueError("root_rpy must match root_pos shape")
    if foot_pos_w.ndim != 4 or tuple(foot_pos_w.shape[-2:]) != (4, 3):
        raise ValueError("foot_pos_w must have shape [B, T, 4, 3]")

    device = root_pos.device
    dtype = root_pos.dtype
    rot_body_to_world = _rpy_to_rot_matrix(root_rpy)
    rot_world_to_body = rot_body_to_world.transpose(-1, -2)
    foot_delta_w = foot_pos_w - root_pos.unsqueeze(2)
    foot_body = torch.einsum("btij,btkj->btki", rot_world_to_body, foot_delta_w)

    hip_offsets = HIP_OFFSETS_ARRAY.to(device=device, dtype=dtype).view(1, 1, 4, 3)
    foot_hip = foot_body - hip_offsets
    px = foot_hip[..., 0]
    py = foot_hip[..., 1]
    pz = foot_hip[..., 2]

    side_signs = _LEG_SIDE_SIGNS.to(device=device, dtype=dtype).view(1, 1, 4)
    d = torch.as_tensor(HIP_OFFSET_Y, device=device, dtype=dtype) * side_signs
    yz_sq = py * py + pz * pz
    lateral = torch.sqrt(torch.clamp(yz_sq - d * d, min=0.0))
    hip_angle = torch.atan2(py, -pz) - torch.atan2(d, lateral)

    pz_eff = -lateral
    reach_sq = px * px + pz_eff * pz_eff
    thigh = torch.as_tensor(THIGH_LENGTH, device=device, dtype=dtype)
    calf = torch.as_tensor(CALF_LENGTH, device=device, dtype=dtype)
    cos_calf = (reach_sq - thigh * thigh - calf * calf) / (2.0 * thigh * calf)
    calf_angle = -torch.arccos(cos_calf.clamp(-1.0, 1.0))
    alpha = torch.atan2(-px, -pz_eff)
    beta = torch.atan2(calf * torch.sin(calf_angle), thigh + calf * torch.cos(calf_angle))
    thigh_angle = alpha - beta

    joint_raw = torch.stack((hip_angle, thigh_angle, calf_angle), dim=-1).reshape(root_pos.shape[0], root_pos.shape[1], 12)
    if not bool(clamp_to_limits):
        return joint_raw
    limits = _JOINT_LIMITS.to(device=device, dtype=dtype)
    lower = limits[:, 0].view(1, 1, 12)
    upper = limits[:, 1].view(1, 1, 12)
    return joint_raw.clamp(min=lower, max=upper)


def fk_leg_points_from_joint_angles(
    root_pos: Tensor,
    root_rpy: Tensor,
    joint_angles: Tensor,
    *,
    shank_sample_count: int = 2,
) -> MpcLegPoints:
    """Forward-kinematics foot, knee, and shank samples from planner-order joints."""
    if int(shank_sample_count) < 0:
        raise ValueError("shank_sample_count must be non-negative")
    leg_angles = joint_angles.reshape(root_pos.shape[0], root_pos.shape[1], 4, 3)
    h = leg_angles[..., 0]
    theta_t = leg_angles[..., 1]
    theta_c = leg_angles[..., 2]
    side_signs = _LEG_SIDE_SIGNS.to(device=root_pos.device, dtype=root_pos.dtype).view(1, 1, 4)
    d = torch.as_tensor(HIP_OFFSET_Y, device=root_pos.device, dtype=root_pos.dtype) * side_signs
    thigh = torch.as_tensor(THIGH_LENGTH, device=root_pos.device, dtype=root_pos.dtype)
    calf = torch.as_tensor(CALF_LENGTH, device=root_pos.device, dtype=root_pos.dtype)

    knee_x = -thigh * torch.sin(theta_t)
    knee_z = -thigh * torch.cos(theta_t)
    calf_abs = theta_t + theta_c
    foot_x = knee_x - calf * torch.sin(calf_abs)
    foot_z = knee_z - calf * torch.cos(calf_abs)
    cos_h = torch.cos(h)
    sin_h = torch.sin(h)
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=root_pos.device, dtype=root_pos.dtype).view(1, 1, 4, 3)
    knee_body = torch.stack(
        (
            hip_offsets[..., 0] + knee_x,
            hip_offsets[..., 1] + cos_h * d - sin_h * knee_z,
            hip_offsets[..., 2] + sin_h * d + cos_h * knee_z,
        ),
        dim=-1,
    )
    foot_body = torch.stack(
        (
            hip_offsets[..., 0] + foot_x,
            hip_offsets[..., 1] + cos_h * d - sin_h * foot_z,
            hip_offsets[..., 2] + sin_h * d + cos_h * foot_z,
        ),
        dim=-1,
    )
    alpha = torch.linspace(
        0.0,
        1.0,
        steps=int(shank_sample_count) + 2,
        dtype=root_pos.dtype,
        device=root_pos.device,
    )[1:-1]
    shank_body = (
        knee_body.unsqueeze(-2) * (1.0 - alpha.view(1, 1, 1, -1, 1))
        + foot_body.unsqueeze(-2) * alpha.view(1, 1, 1, -1, 1)
    )
    rot_body_to_world = _rpy_to_rot_matrix(root_rpy)
    knee_world = torch.einsum("btij,btkj->btki", rot_body_to_world, knee_body) + root_pos.unsqueeze(2)
    foot_world = torch.einsum("btij,btkj->btki", rot_body_to_world, foot_body) + root_pos.unsqueeze(2)
    shank_world = torch.einsum("btij,btkqj->btkqi", rot_body_to_world, shank_body) + root_pos.unsqueeze(2).unsqueeze(-2)
    return MpcLegPoints(
        foot_pos_world=foot_world,
        knee_pos_world=knee_world,
        shank_sample_world=shank_world,
    )


def fk_feet_from_joint_angles(root_pos: Tensor, root_rpy: Tensor, joint_angles: Tensor) -> Tensor:
    """Forward-kinematics foot positions from world root pose and planner-order joints."""
    return fk_leg_points_from_joint_angles(root_pos, root_rpy, joint_angles, shank_sample_count=0).foot_pos_world


__all__ = ["MpcLegPoints", "fk_feet_from_joint_angles", "fk_leg_points_from_joint_angles", "solve_joint_angles_from_trajectory"]
