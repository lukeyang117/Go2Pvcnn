"""Vectorized Go2 kinematics diagnostics for together planner outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .types import CALF_LENGTH, HIP_OFFSETS_ARRAY, HIP_OFFSET_Y, THIGH_LENGTH

JOINT_LIMITS = torch.tensor(
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
LEG_SIDE_SIGNS = torch.tensor((1.0, -1.0, 1.0, -1.0), dtype=torch.float32)


@dataclass(frozen=True)
class TogetherKinematicsResult:
    joint_angles: Tensor
    joint_limit_violation: Tensor
    workspace_margin: Tensor
    hip_world: Tensor
    knee_world: Tensor
    foot_world: Tensor


def rpy_to_rot_matrix(root_rpy: Tensor) -> Tensor:
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


def evaluate_kinematics(root_pos: Tensor, root_rpy: Tensor, foot_pos: Tensor) -> TogetherKinematicsResult:
    if root_pos.ndim != 3 or root_pos.shape[-1] != 3:
        raise ValueError("root_pos must have shape [B, T, 3]")
    if root_rpy.shape != root_pos.shape:
        raise ValueError("root_rpy must match root_pos")
    if foot_pos.ndim != 4 or foot_pos.shape[-2:] != (4, 3):
        raise ValueError("foot_pos must have shape [B, T, 4, 3]")
    device = root_pos.device
    dtype = root_pos.dtype
    rot_body_to_world = rpy_to_rot_matrix(root_rpy)
    rot_world_to_body = rot_body_to_world.transpose(-1, -2)
    delta_world = foot_pos - root_pos.unsqueeze(2)
    foot_body = torch.einsum("btij,btkj->btki", rot_world_to_body, delta_world)
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=device, dtype=dtype).view(1, 1, 4, 3)
    foot_hip = foot_body - hip_offsets
    px = foot_hip[..., 0]
    py = foot_hip[..., 1]
    pz = foot_hip[..., 2]
    side_signs = LEG_SIDE_SIGNS.to(device=device, dtype=dtype).view(1, 1, 4)
    hip_y = torch.as_tensor(HIP_OFFSET_Y, device=device, dtype=dtype)
    d = hip_y * side_signs
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
    limits = JOINT_LIMITS.to(device=device, dtype=dtype)
    lower = limits[:, 0].view(1, 1, 12)
    upper = limits[:, 1].view(1, 1, 12)
    joint_limit_violation = torch.maximum(
        torch.clamp(lower - joint_raw, min=0.0),
        torch.clamp(joint_raw - upper, min=0.0),
    )
    joint_angles = joint_raw.clamp(min=lower, max=upper)
    radius_yz = torch.sqrt(py * py + pz * pz)
    side_clearance = radius_yz - torch.abs(d)
    plane_reach = torch.sqrt(px * px + lateral * lateral)
    reach_margin = thigh + calf - plane_reach
    workspace_margin = torch.minimum(side_clearance, reach_margin)
    leg_angles = joint_angles.reshape(root_pos.shape[0], root_pos.shape[1], 4, 3)
    hip_world, knee_world, foot_world = _forward_leg_keypoints_world(root_pos, rot_body_to_world, leg_angles, dtype=dtype)
    return TogetherKinematicsResult(
        joint_angles=joint_angles,
        joint_limit_violation=joint_limit_violation,
        workspace_margin=workspace_margin,
        hip_world=hip_world,
        knee_world=knee_world,
        foot_world=foot_world,
    )


def _forward_leg_keypoints_world(root_pos: Tensor, rot_body_to_world: Tensor, leg_angles: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    device = root_pos.device
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=device, dtype=dtype).view(1, 1, 4, 3).expand_as(leg_angles[..., :1].expand(-1, -1, -1, 3))
    side_signs = LEG_SIDE_SIGNS.to(device=device, dtype=dtype).view(1, 1, 4)
    hip_angle = leg_angles[..., 0]
    thigh_angle = leg_angles[..., 1]
    calf_angle = leg_angles[..., 2]
    d = torch.as_tensor(HIP_OFFSET_Y, device=device, dtype=dtype) * side_signs
    cos_h = torch.cos(hip_angle)
    sin_h = torch.sin(hip_angle)
    thigh = torch.as_tensor(THIGH_LENGTH, device=device, dtype=dtype)
    calf = torch.as_tensor(CALF_LENGTH, device=device, dtype=dtype)
    knee_x = -thigh * torch.sin(thigh_angle)
    knee_z = -thigh * torch.cos(thigh_angle)
    calf_abs = thigh_angle + calf_angle
    foot_x = knee_x - calf * torch.sin(calf_abs)
    foot_z = knee_z - calf * torch.cos(calf_abs)
    hip_y = d.expand_as(knee_x)
    knee_body = torch.stack(
        (
            hip_offsets[..., 0] + knee_x,
            hip_offsets[..., 1] + cos_h * hip_y - sin_h * knee_z,
            hip_offsets[..., 2] + sin_h * hip_y + cos_h * knee_z,
        ),
        dim=-1,
    )
    foot_body = torch.stack(
        (
            hip_offsets[..., 0] + foot_x,
            hip_offsets[..., 1] + cos_h * hip_y - sin_h * foot_z,
            hip_offsets[..., 2] + sin_h * hip_y + cos_h * foot_z,
        ),
        dim=-1,
    )
    hip_world = torch.einsum("btij,btkj->btki", rot_body_to_world, hip_offsets) + root_pos.unsqueeze(2)
    knee_world = torch.einsum("btij,btkj->btki", rot_body_to_world, knee_body) + root_pos.unsqueeze(2)
    foot_world = torch.einsum("btij,btkj->btki", rot_body_to_world, foot_body) + root_pos.unsqueeze(2)
    return hip_world, knee_world, foot_world


__all__ = ["JOINT_LIMITS", "TogetherKinematicsResult", "evaluate_kinematics", "rpy_to_rot_matrix"]
