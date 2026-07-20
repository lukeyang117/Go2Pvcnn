"""Vectorized closed-form inverse kinematics for all four Go2 legs."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.model.go2_kinematics import (
    CALF_LENGTH,
    HIP_OFFSETS,
    HIP_OFFSET_Y,
    LEG_SIDE_SIGNS,
    THIGH_LENGTH,
    rpy_to_rotation_matrix,
)
from extension.joint_mpc_rti.tensor_constants import constant_like


def go2_analytic_ik(
    root_pos_w: Tensor,
    root_rpy_w: Tensor,
    foot_target_w: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return unconstrained leg angles and a per-target geometric reachability mask."""
    root_pos = torch.as_tensor(root_pos_w)
    root_rpy = torch.as_tensor(root_rpy_w, dtype=root_pos.dtype, device=root_pos.device)
    foot_target = torch.as_tensor(foot_target_w, dtype=root_pos.dtype, device=root_pos.device)
    if root_pos.ndim < 2 or root_pos.shape[-1] != 3:
        raise ValueError("root_pos_w must have shape [...,3]")
    if root_rpy.shape != root_pos.shape:
        raise ValueError("root_rpy_w must match root_pos_w")
    if foot_target.shape != root_pos.shape[:-1] + (4, 3):
        raise ValueError("foot_target_w must have shape [...,4,3]")

    rotation_wb = rpy_to_rotation_matrix(root_rpy)
    target_delta_w = foot_target - root_pos.unsqueeze(-2)
    target_b = torch.einsum("...ji,...li->...lj", rotation_wb.transpose(-1, -2), target_delta_w)
    hip = constant_like(root_pos, "hip_offsets", HIP_OFFSETS).view(*((1,) * (root_pos.ndim - 1)), 4, 3)
    hip_local = target_b - hip
    px, py, pz = hip_local.unbind(dim=-1)

    side = constant_like(root_pos, "leg_side_signs", LEG_SIDE_SIGNS).view(
        *((1,) * (root_pos.ndim - 1)), 4
    )
    lateral_offset = float(HIP_OFFSET_Y) * side
    lateral_sq = py.square() + pz.square() - lateral_offset.square()
    lateral = lateral_sq.clamp_min(0.0).sqrt()
    abad = torch.atan2(py, -pz) - torch.atan2(lateral_offset, lateral)

    effective_z = -lateral
    reach_sq = px.square() + effective_z.square()
    thigh = float(THIGH_LENGTH)
    calf = float(CALF_LENGTH)
    cosine_calf = (reach_sq - thigh * thigh - calf * calf) / (2.0 * thigh * calf)
    calf_angle = -torch.acos(cosine_calf.clamp(-1.0, 1.0))
    alpha = torch.atan2(-px, -effective_z)
    beta = torch.atan2(calf * torch.sin(calf_angle), thigh + calf * torch.cos(calf_angle))
    thigh_angle = alpha - beta
    joint = torch.stack((abad, thigh_angle, calf_angle), dim=-1)

    tolerance = 1.0e-6
    reach = reach_sq.clamp_min(0.0).sqrt()
    reachable = (
        (lateral_sq >= -tolerance)
        & (reach >= abs(thigh - calf) - tolerance)
        & (reach <= thigh + calf + tolerance)
        & torch.isfinite(hip_local).all(dim=-1)
        & torch.isfinite(joint).all(dim=-1)
    )
    return joint, reachable


__all__ = ["go2_analytic_ik"]
