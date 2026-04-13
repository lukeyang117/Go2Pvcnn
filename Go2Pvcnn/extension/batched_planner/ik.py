"""Torch batched inverse and forward kinematics helpers for the Go2 planner."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import (
    CALF_LENGTH,
    HIP_OFFSETS_ARRAY,
    HIP_OFFSET_Y,
    LEG_SIDE_SIGN,
    THIGH_LENGTH,
)

JOINT_LIMITS = torch.tensor(
    [
        [-1.0472, 1.0472],
        [-1.5708, 3.4907],
        [-2.7227, -0.8378],
        [-1.0472, 1.0472],
        [-1.5708, 3.4907],
        [-2.7227, -0.8378],
        [-1.0472, 1.0472],
        [-0.5236, 4.5379],
        [-2.7227, -0.8378],
        [-1.0472, 1.0472],
        [-0.5236, 4.5379],
        [-2.7227, -0.8378],
    ],
    dtype=torch.float64,
)

_LEG_SIDE_SIGNS = torch.tensor(
    [LEG_SIDE_SIGN["FL"], LEG_SIDE_SIGN["FR"], LEG_SIDE_SIGN["RL"], LEG_SIDE_SIGN["RR"]],
    dtype=torch.float64,
)


def _resolve_input_device(*values) -> torch.device:
    devices = [value.device for value in values if isinstance(value, Tensor)]
    if not devices:
        return torch.device("cpu")

    first = devices[0]
    for device in devices[1:]:
        if device != first:
            raise ValueError("batched IK helpers do not accept tensor inputs on multiple devices")
    return first


def _coerce_tensor(value, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64, device=device)


def _expect_shape(name: str, tensor: Tensor, shape_suffix: tuple[int, ...]) -> Tensor:
    if tensor.ndim != len(shape_suffix) + 1 or tuple(tensor.shape[1:]) != shape_suffix:
        raise ValueError(f"{name} must have shape (N, {', '.join(str(dim) for dim in shape_suffix)}); got {tuple(tensor.shape)}")
    return tensor


def _as_root_pos(root_pos, *, device: torch.device) -> Tensor:
    return _expect_shape("root_pos", _coerce_tensor(root_pos, device=device), (3,))


def _as_root_quat(root_quat, *, device: torch.device) -> Tensor:
    return _expect_shape("root_quat", _coerce_tensor(root_quat, device=device), (4,))


def _as_foot_targets(foot_targets, *, device: torch.device) -> Tensor:
    return _expect_shape("foot_targets", _coerce_tensor(foot_targets, device=device), (4, 3))


def _as_joint_angles(joint_angles, *, device: torch.device) -> Tensor:
    return _expect_shape("joint_angles", _coerce_tensor(joint_angles, device=device), (12,))


def _quat_to_rot_batch(quat: Tensor) -> Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _solve_leg_ik_batch(
    foot_pos_hip: Tensor,
    side_sign: Tensor,
    thigh_length: float = THIGH_LENGTH,
    calf_length: float = CALF_LENGTH,
    hip_offset_y: float = HIP_OFFSET_Y,
) -> Tensor:
    px = foot_pos_hip[..., 0]
    py = foot_pos_hip[..., 1]
    pz = foot_pos_hip[..., 2]
    d = hip_offset_y * side_sign

    yz_sq = py * py + pz * pz
    lateral = torch.sqrt(torch.clamp_min(yz_sq - d * d, 0.0))
    hip_angle = torch.atan2(py, -pz) - torch.atan2(d, lateral)

    pz_eff = -lateral
    reach_sq = px * px + pz_eff * pz_eff
    cos_calf = (reach_sq - thigh_length**2 - calf_length**2) / (2.0 * thigh_length * calf_length)
    cos_calf = torch.clamp(cos_calf, -1.0, 1.0)
    calf_angle = -torch.arccos(cos_calf)

    alpha = torch.atan2(-px, -pz_eff)
    beta = torch.atan2(
        calf_length * torch.sin(calf_angle),
        thigh_length + calf_length * torch.cos(calf_angle),
    )
    thigh_angle = alpha - beta
    return torch.stack([hip_angle, thigh_angle, calf_angle], dim=-1)


def _forward_kinematics_leg_batch(
    angles: Tensor,
    side_sign: Tensor,
    hip_offset: Tensor,
    thigh_length: float = THIGH_LENGTH,
    calf_length: float = CALF_LENGTH,
    hip_offset_y: float = HIP_OFFSET_Y,
) -> tuple[Tensor, Tensor, Tensor]:
    h = angles[..., 0]
    theta_t = angles[..., 1]
    theta_c = angles[..., 2]
    d = hip_offset_y * side_sign

    cos_h = torch.cos(h)
    sin_h = torch.sin(h)

    knee_x = -thigh_length * torch.sin(theta_t)
    knee_z = -thigh_length * torch.cos(theta_t)

    calf_abs = theta_t + theta_c
    foot_x = knee_x - calf_length * torch.sin(calf_abs)
    foot_z = knee_z - calf_length * torch.cos(calf_abs)

    hip_y = d.expand_as(knee_x)
    knee_body = torch.stack(
        [
            hip_offset[..., 0] + knee_x,
            hip_offset[..., 1] + cos_h * hip_y - sin_h * knee_z,
            hip_offset[..., 2] + sin_h * hip_y + cos_h * knee_z,
        ],
        dim=-1,
    )
    foot_body = torch.stack(
        [
            hip_offset[..., 0] + foot_x,
            hip_offset[..., 1] + cos_h * hip_y - sin_h * foot_z,
            hip_offset[..., 2] + sin_h * hip_y + cos_h * foot_z,
        ],
        dim=-1,
    )
    return hip_offset, knee_body, foot_body


def batch_inverse_kinematics(root_pos, root_quat, foot_targets) -> Tensor:
    device = _resolve_input_device(root_pos, root_quat, foot_targets)
    root_pos_t = _as_root_pos(root_pos, device=device)
    root_quat_t = _as_root_quat(root_quat, device=device)
    foot_targets_t = _as_foot_targets(foot_targets, device=device)

    n_frames = int(root_pos_t.shape[0])
    if root_quat_t.shape[0] != n_frames or foot_targets_t.shape[0] != n_frames:
        raise ValueError("root_pos, root_quat, and foot_targets must share the same leading dimension")

    rot = _quat_to_rot_batch(root_quat_t)
    foot_body = torch.einsum("nji,nmj->nmi", rot, foot_targets_t - root_pos_t[:, None, :])

    hip_offsets = HIP_OFFSETS_ARRAY.to(device=device, dtype=torch.float64).unsqueeze(0).expand(n_frames, -1, -1)
    side_signs = _LEG_SIDE_SIGNS.to(device=device).view(1, 4).expand(n_frames, -1)
    angles = _solve_leg_ik_batch(foot_body - hip_offsets, side_signs)
    joints = angles.reshape(n_frames, 12)
    lower = JOINT_LIMITS[:, 0].to(device=device)
    upper = JOINT_LIMITS[:, 1].to(device=device)
    return torch.clamp(joints, min=lower, max=upper)


def batch_forward_kinematics(root_pos, root_quat, joint_angles) -> Tensor:
    device = _resolve_input_device(root_pos, root_quat, joint_angles)
    root_pos_t = _as_root_pos(root_pos, device=device)
    root_quat_t = _as_root_quat(root_quat, device=device)
    joint_angles_t = _as_joint_angles(joint_angles, device=device)

    n_frames = int(root_pos_t.shape[0])
    if root_quat_t.shape[0] != n_frames or joint_angles_t.shape[0] != n_frames:
        raise ValueError("root_pos, root_quat, and joint_angles must share the same leading dimension")

    rot = _quat_to_rot_batch(root_quat_t)
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=device, dtype=torch.float64).unsqueeze(0).expand(n_frames, -1, -1)
    side_signs = _LEG_SIDE_SIGNS.to(device=device).view(1, 4).expand(n_frames, -1)
    leg_angles = joint_angles_t.reshape(n_frames, 4, 3)
    hip_body, knee_body, foot_body = _forward_kinematics_leg_batch(leg_angles, side_signs, hip_offsets)
    body_links = torch.cat([hip_body, knee_body, foot_body], dim=1)
    return torch.einsum("nij,nmj->nmi", rot, body_links) + root_pos_t[:, None, :]


def batch_body_pos_root_relative(root_pos, root_quat, body_pos_w) -> Tensor:
    device = _resolve_input_device(root_pos, root_quat, body_pos_w)
    root_pos_t = _as_root_pos(root_pos, device=device)
    root_quat_t = _as_root_quat(root_quat, device=device)
    body_pos_w_t = _expect_shape("body_pos_w", _coerce_tensor(body_pos_w, device=device), (12, 3))

    n_frames = int(root_pos_t.shape[0])
    if root_quat_t.shape[0] != n_frames or body_pos_w_t.shape[0] != n_frames:
        raise ValueError("root_pos, root_quat, and body_pos_w must share the same leading dimension")

    rot = _quat_to_rot_batch(root_quat_t)
    delta = body_pos_w_t - root_pos_t[:, None, :]
    return torch.einsum("nji,nmj->nmi", rot, delta)


__all__ = [
    "JOINT_LIMITS",
    "batch_body_pos_root_relative",
    "batch_forward_kinematics",
    "batch_inverse_kinematics",
]
