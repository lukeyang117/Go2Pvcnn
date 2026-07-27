from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


HIP_OFFSETS = (
    (0.1934, 0.0465, 0.0),
    (0.1934, -0.0465, 0.0),
    (-0.1934, 0.0465, 0.0),
    (-0.1934, -0.0465, 0.0),
)
LEG_SIDE_SIGNS = (1.0, -1.0, 1.0, -1.0)
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_OFFSET_Y = 0.0955
JOINT_LOWER = (-1.0472, -0.6632, -2.721)
JOINT_UPPER = (1.0472, 2.966, -0.837)


@dataclass(frozen=True)
class Go2ParallelGeometry:
    hip_pos_w: Tensor
    foot_pos_w: Tensor
    knee_pos_w: Tensor
    calf_samples_w: Tensor
    thigh_samples_w: Tensor


def _constant(reference: Tensor, values: tuple[tuple[float, ...], ...] | tuple[float, ...]) -> Tensor:
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def rpy_to_rotation_matrix(root_rpy_w: Tensor) -> Tensor:
    rpy = torch.as_tensor(root_rpy_w)
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    row0 = torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=-1)
    row1 = torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=-1)
    row2 = torch.stack((-sp, cp * sr, cp * cr), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _leg_points_body(joint_pos: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    joint = torch.as_tensor(joint_pos)
    leading = joint.shape[:-1]
    angles = joint.reshape(*leading, 4, 3)
    abad = angles[..., 0]
    thigh_angle = angles[..., 1]
    calf_angle = angles[..., 2]
    side = _constant(joint, LEG_SIDE_SIGNS).view(*((1,) * len(leading)), 4).expand(*leading, 4)
    lateral = float(HIP_OFFSET_Y) * side
    hip = _constant(joint, HIP_OFFSETS).view(*((1,) * len(leading)), 4, 3).expand(*leading, 4, 3)
    knee_x = -float(THIGH_LENGTH) * torch.sin(thigh_angle)
    knee_z = -float(THIGH_LENGTH) * torch.cos(thigh_angle)
    calf_absolute = thigh_angle + calf_angle
    foot_x = knee_x - float(CALF_LENGTH) * torch.sin(calf_absolute)
    foot_z = knee_z - float(CALF_LENGTH) * torch.cos(calf_absolute)
    cosine = torch.cos(abad)
    sine = torch.sin(abad)
    upper_body = torch.stack(
        (hip[..., 0], hip[..., 1] + cosine * lateral, hip[..., 2] + sine * lateral),
        dim=-1,
    )
    knee_body = torch.stack(
        (
            hip[..., 0] + knee_x,
            hip[..., 1] + cosine * lateral - sine * knee_z,
            hip[..., 2] + sine * lateral + cosine * knee_z,
        ),
        dim=-1,
    )
    foot_body = torch.stack(
        (
            hip[..., 0] + foot_x,
            hip[..., 1] + cosine * lateral - sine * foot_z,
            hip[..., 2] + sine * lateral + cosine * foot_z,
        ),
        dim=-1,
    )
    return hip.expand(*leading, 4, 3), upper_body, knee_body, foot_body


def fk_go2(root_pos_w: Tensor, root_rpy_w: Tensor, joint_pos: Tensor, *, capsule_samples: int = 5) -> Go2ParallelGeometry:
    root_pos = torch.as_tensor(root_pos_w)
    root_rpy = torch.as_tensor(root_rpy_w, dtype=root_pos.dtype, device=root_pos.device)
    joint = torch.as_tensor(joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    if root_pos.shape[-1] != 3 or root_rpy.shape != root_pos.shape or joint.shape[:-1] != root_pos.shape[:-1] or joint.shape[-1] != 12:
        raise ValueError("root_pos_w/root_rpy_w/joint_pos must have shapes [...,3], [...,3], [...,12]")
    hip_body, upper_body, knee_body, foot_body = _leg_points_body(joint)
    rotation = rpy_to_rotation_matrix(root_rpy)
    hip_world = torch.einsum("...ij,...lj->...li", rotation, hip_body) + root_pos.unsqueeze(-2)
    knee_world = torch.einsum("...ij,...lj->...li", rotation, knee_body) + root_pos.unsqueeze(-2)
    foot_world = torch.einsum("...ij,...lj->...li", rotation, foot_body) + root_pos.unsqueeze(-2)
    sample_alpha = torch.linspace(
        0.0,
        1.0,
        int(capsule_samples),
        dtype=root_pos.dtype,
        device=root_pos.device,
    ).view(*((1,) * (foot_world.ndim - 1)), int(capsule_samples), 1)
    calf_samples = knee_world.unsqueeze(-2) * (1.0 - sample_alpha) + foot_world.unsqueeze(-2) * sample_alpha
    thigh_samples = hip_world.unsqueeze(-2) * (1.0 - sample_alpha) + knee_world.unsqueeze(-2) * sample_alpha
    return Go2ParallelGeometry(
        hip_pos_w=hip_world,
        foot_pos_w=foot_world,
        knee_pos_w=knee_world,
        calf_samples_w=calf_samples,
        thigh_samples_w=thigh_samples,
    )
