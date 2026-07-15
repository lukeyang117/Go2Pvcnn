"""Fixed-shape continuous trajectory diagnostics for MPC-QP."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
from extension.batch_mpc_planner.terrain import height_at, semantic_at
from extension.batch_mpc_planner.types import MpcPlannerResult, MpcPlannerTerrain


_CLEARANCE_PENETRATION_TOL_M = 1.0e-4

def _row_count(mask: Tensor) -> Tensor:
    return torch.count_nonzero(mask.reshape(mask.shape[0], -1), dim=1).to(dtype=torch.float32)


def _footprint_offsets(*, radius_m: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    r = float(radius_m)
    return torch.tensor(
        (
            (0.0, 0.0),
            (r, 0.0),
            (-r, 0.0),
            (0.0, r),
            (0.0, -r),
            (r, r),
            (r, -r),
            (-r, r),
            (-r, -r),
        ),
        dtype=dtype,
        device=device,
    )


def continuous_loss_diagnostics(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    *,
    footprint_radius_m: float,
    low_small_clearance_m: float = 0.0,
) -> dict[str, Tensor]:
    foot = torch.as_tensor(result.foot_pos)
    first = foot[:, 1:] - foot[:, :-1]
    second = first[:, 1:] - first[:, :-1]
    foot_jump = torch.linalg.vector_norm(first, dim=-1).reshape(foot.shape[0], -1).amax(dim=1)
    if int(second.shape[1]) == 0:
        foot_acc = torch.zeros((foot.shape[0],), dtype=foot.dtype, device=foot.device)
    else:
        foot_acc = torch.linalg.vector_norm(second, dim=-1).reshape(foot.shape[0], -1).amax(dim=1)

    touchdown_xy = torch.as_tensor(result.planned_touchdown_w[:, 0, :, :2], dtype=foot.dtype, device=foot.device)
    touchdown_semantic = semantic_at(terrain, touchdown_xy)
    touchdown_bad = _row_count(touchdown_semantic != 0).to(dtype=foot.dtype, device=foot.device)

    offsets = _footprint_offsets(radius_m=footprint_radius_m, dtype=foot.dtype, device=foot.device)
    footprint_xy = touchdown_xy[..., None, :] + offsets.view(1, 1, -1, 2)
    heights = height_at(terrain, footprint_xy.reshape(foot.shape[0], -1, 2)).reshape(*footprint_xy.shape[:-1])
    height_var = (heights.amax(dim=-1) - heights.amin(dim=-1)).reshape(foot.shape[0], -1).amax(dim=1)

    foot_semantic = semantic_at(terrain, foot[..., :2])
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
    prev_contact = torch.nn.functional.pad(contact[:, :-1], (0, 0, 1, 0), value=True)
    next_contact = torch.nn.functional.pad(contact[:, 1:], (0, 0, 0, 1), value=True)
    mid_swing = torch.logical_and(
        torch.logical_not(contact),
        torch.logical_and(torch.logical_not(prev_contact), torch.logical_not(next_contact)),
    )
    low_small_swing = torch.logical_and(foot_semantic == 1, mid_swing)
    terrain_z = height_at(terrain, foot[..., :2]).to(dtype=foot.dtype, device=foot.device)
    planned_clearance = foot[..., 2] - terrain_z
    clearance_deficit = torch.clamp(terrain_z + float(low_small_clearance_m) - foot[..., 2], min=0.0)
    masked_deficit = torch.where(low_small_swing, clearance_deficit, torch.zeros_like(clearance_deficit))
    low_small_deficit_max = masked_deficit.reshape(foot.shape[0], -1).amax(dim=1)

    fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
    fk_terrain_z = height_at(terrain, fk_foot[..., :2]).to(dtype=fk_foot.dtype, device=fk_foot.device)
    fk_clearance = fk_foot[..., 2] - fk_terrain_z
    readback_error = torch.linalg.vector_norm(fk_foot - foot, dim=-1)
    fk_readback_max = readback_error.reshape(foot.shape[0], -1).amax(dim=1)
    fk_readback_mean = readback_error.reshape(foot.shape[0], -1).mean(dim=1)
    mid_idx = max(0, min(int(readback_error.shape[1]) - 1, int(readback_error.shape[1]) // 2))
    fk_readback_start_max = readback_error[:, 0].reshape(foot.shape[0], -1).amax(dim=1)
    fk_readback_mid_max = readback_error[:, mid_idx].reshape(foot.shape[0], -1).amax(dim=1)
    fk_readback_end_max = readback_error[:, -1].reshape(foot.shape[0], -1).amax(dim=1)

    joint = torch.as_tensor(result.joint_angles, dtype=foot.dtype, device=foot.device)
    if int(joint.shape[1]) < 2:
        joint_jump = torch.zeros((foot.shape[0],), dtype=foot.dtype, device=foot.device)
    else:
        joint_jump = torch.abs(joint[:, 1:] - joint[:, :-1]).reshape(foot.shape[0], -1).amax(dim=1)

    swing = torch.logical_not(contact)
    planned_swing_over = torch.where(swing, planned_clearance, torch.zeros_like(planned_clearance))
    low_small_swing_over = torch.where(low_small_swing, planned_clearance, torch.zeros_like(planned_clearance))
    fk_swing_over = torch.where(swing, fk_clearance, torch.zeros_like(fk_clearance))
    fk_low_small_swing = torch.logical_and(semantic_at(terrain, fk_foot[..., :2]) == 1, swing)
    fk_low_small_swing_over = torch.where(fk_low_small_swing, fk_clearance, torch.zeros_like(fk_clearance))

    return {
        "qp_continuous_foot_frame_jump_max": foot_jump,
        "qp_continuous_foot_acceleration_max": foot_acc,
        "qp_continuous_foothold_height_variation_max": height_var.to(dtype=foot.dtype, device=foot.device),
        "qp_continuous_touchdown_semantic_bad_count": touchdown_bad,
        "qp_continuous_low_small_clearance_deficit_max": low_small_deficit_max,
        "qp_continuous_planned_foot_terrain_clearance_min": planned_clearance.reshape(foot.shape[0], -1).amin(dim=1),
        "qp_continuous_fk_foot_terrain_clearance_min": fk_clearance.reshape(foot.shape[0], -1).amin(dim=1),
        "qp_continuous_planned_foot_terrain_penetration_count": _row_count(planned_clearance < -_CLEARANCE_PENETRATION_TOL_M).to(
            dtype=foot.dtype,
            device=foot.device,
        ),
        "qp_continuous_fk_foot_terrain_penetration_count": _row_count(fk_clearance < -_CLEARANCE_PENETRATION_TOL_M).to(
            dtype=foot.dtype,
            device=foot.device,
        ),
        "qp_continuous_swing_height_over_terrain_max": planned_swing_over.reshape(foot.shape[0], -1).amax(dim=1),
        "qp_continuous_low_small_swing_height_over_terrain_max": low_small_swing_over.reshape(
            foot.shape[0],
            -1,
        ).amax(dim=1),
        "qp_continuous_fk_swing_height_over_terrain_max": fk_swing_over.reshape(fk_foot.shape[0], -1).amax(dim=1).to(
            dtype=foot.dtype,
            device=foot.device,
        ),
        "qp_continuous_fk_low_small_swing_height_over_terrain_max": fk_low_small_swing_over.reshape(
            fk_foot.shape[0],
            -1,
        ).amax(dim=1).to(dtype=foot.dtype, device=foot.device),
        "qp_continuous_fk_readback_error_max": fk_readback_max,
        "qp_continuous_fk_readback_error_mean": fk_readback_mean,
        "qp_continuous_fk_readback_start_error_max": fk_readback_start_max,
        "qp_continuous_fk_readback_mid_error_max": fk_readback_mid_max,
        "qp_continuous_fk_readback_end_error_max": fk_readback_end_max,
        "qp_continuous_joint_frame_jump_max": joint_jump,
    }


__all__ = ["continuous_loss_diagnostics"]
