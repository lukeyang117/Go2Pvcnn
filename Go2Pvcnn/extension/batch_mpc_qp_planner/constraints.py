"""Sparse safety diagnostics and repairs for the MPC-QP backend."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.batch_mpc_planner.kinematics import fk_leg_points_from_joint_angles
from extension.batch_mpc_planner.terrain import height_at, semantic_at
from extension.batch_mpc_planner.types import MpcPlannerResult, MpcPlannerStatus, MpcPlannerTerrain, MpcRobotState


def _row_counts(mask: Tensor) -> Tensor:
    return torch.count_nonzero(mask.reshape(mask.shape[0], -1), dim=1).to(dtype=torch.float32)


def touchdown_semantic_violation_count(terrain: MpcPlannerTerrain, touchdown_w: Tensor) -> Tensor:
    semantic = semantic_at(terrain, touchdown_w[..., :2])
    return _row_counts(semantic != 0)


def height_violation_max(terrain: MpcPlannerTerrain, points_w: Tensor, *, margin_m: float) -> Tensor:
    terrain_z = height_at(terrain, points_w[..., :2]).to(dtype=points_w.dtype, device=points_w.device)
    deficit = torch.relu(terrain_z + float(margin_m) - points_w[..., 2])
    return deficit.reshape(points_w.shape[0], -1).amax(dim=1)


def _underbody_points(root_pos: Tensor, *, sample_count: int = 5) -> Tensor:
    batch, horizon = int(root_pos.shape[0]), int(root_pos.shape[1])
    offsets = torch.tensor(
        (
            (0.0, 0.0, -0.16),
            (0.18, 0.10, -0.16),
            (0.18, -0.10, -0.16),
            (-0.18, 0.10, -0.16),
            (-0.18, -0.10, -0.16),
        ),
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    count = max(1, min(int(sample_count), int(offsets.shape[0])))
    return root_pos[:, :, None, :] + offsets[:count].view(1, 1, count, 3).expand(batch, horizon, -1, -1)


def _semantic_collision_count(terrain: MpcPlannerTerrain, points_w: Tensor, *, height_margin_m: float = 0.0) -> Tensor:
    semantic = semantic_at(terrain, points_w[..., :2])
    terrain_z = height_at(terrain, points_w[..., :2]).to(dtype=points_w.dtype, device=points_w.device)
    colliding = torch.logical_and(semantic != 0, points_w[..., 2] < terrain_z + float(height_margin_m) - 1.0e-5)
    return _row_counts(colliding)


def _height_violation(terrain: MpcPlannerTerrain, points_w: Tensor, *, margin_m: float) -> Tensor:
    terrain_z = height_at(terrain, points_w[..., :2]).to(dtype=points_w.dtype, device=points_w.device)
    return torch.relu(terrain_z + float(margin_m) - points_w[..., 2])


def repair_touchdown_semantic_keepout(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
) -> MpcPlannerResult:
    """Move semantically invalid touchdown legs back to current foot anchors."""

    touchdown_semantic = semantic_at(terrain, result.planned_touchdown_w[:, 0, :, :2])
    bad_leg = touchdown_semantic != 0
    if not bool(torch.any(bad_leg).item()):
        return result

    foot_anchor = torch.as_tensor(state.foot_pos, dtype=result.foot_pos.dtype, device=result.foot_pos.device)
    bad_foot = bad_leg[:, None, :, None]
    repaired_foot = torch.where(bad_foot, foot_anchor[:, None, :, :], result.foot_pos)
    repaired_touchdown = torch.where(bad_foot, foot_anchor[:, None, :, :], result.planned_touchdown_w)
    event_bad = bad_leg[:, :, None, None]
    repaired_touchdown_seq = torch.where(event_bad, foot_anchor[:, :, None, :], result.touchdown_seq)
    repaired_contact = torch.where(
        bad_leg[:, None, :],
        torch.ones_like(result.contact_state, dtype=torch.bool),
        result.contact_state,
    )
    feasible = result.feasible.clone()
    safe_fallback = result.safe_fallback.clone()
    status = result.status.clone()
    finite_ok = (
        torch.isfinite(repaired_foot).flatten(1).all(dim=1)
        & torch.isfinite(repaired_touchdown).flatten(1).all(dim=1)
        & torch.isfinite(repaired_touchdown_seq).flatten(1).all(dim=1)
    )
    feasible = torch.logical_and(feasible, finite_ok)
    safe_fallback = torch.logical_or(safe_fallback, torch.logical_not(finite_ok))
    status = torch.where(finite_ok, status, torch.full_like(status, int(MpcPlannerStatus.ALL_INFEASIBLE)))
    return MpcPlannerResult(
        root_pos=result.root_pos,
        root_rpy=result.root_rpy,
        foot_pos=repaired_foot,
        joint_angles=result.joint_angles,
        contact_state=repaired_contact,
        touchdown_seq=repaired_touchdown_seq,
        planned_touchdown_w=repaired_touchdown,
        cost_total=result.cost_total,
        cost_breakdown=result.cost_breakdown,
        status=status,
        feasible=feasible,
        safe_fallback=safe_fallback,
        loss_breakdown=result.loss_breakdown,
        hard_reason_mask=result.hard_reason_mask,
    )


def safety_diagnostics(result: MpcPlannerResult, terrain: MpcPlannerTerrain) -> dict[str, Tensor]:
    touchdown_semantic = semantic_at(terrain, result.planned_touchdown_w[:, 0, :, :2])
    touchdown_count = _row_counts(touchdown_semantic != 0)
    touchdown_on_small = _row_counts(touchdown_semantic == 1)
    foot_height = height_violation_max(terrain, result.foot_pos, margin_m=0.0)
    root_height = height_violation_max(terrain, result.root_pos, margin_m=-0.30)
    height_max = torch.maximum(foot_height, root_height)
    semantic_violation = (touchdown_count > 0).to(dtype=result.root_pos.dtype)
    foot_semantic = semantic_at(terrain, result.foot_pos[..., :2])
    foot_terrain_z = height_at(terrain, result.foot_pos[..., :2]).to(
        dtype=result.foot_pos.dtype,
        device=result.foot_pos.device,
    )
    over_small = foot_semantic == 1
    crossing_leg_mask = torch.any(over_small, dim=1)
    crossing_leg_count = _row_counts(crossing_leg_mask)
    clearance = result.foot_pos[..., 2] - foot_terrain_z
    collision = torch.logical_and(over_small, clearance < -1.0e-5)
    collision_count = _row_counts(collision)
    over_small_count = _row_counts(over_small)
    collision_rate = collision_count / over_small_count.clamp_min(1.0)
    masked_clearance = torch.where(over_small, clearance, torch.full_like(clearance, 1.0e6))
    min_clearance = masked_clearance.reshape(masked_clearance.shape[0], -1).amin(dim=1)
    min_clearance = torch.where(over_small_count > 0, min_clearance, torch.zeros_like(min_clearance))
    leg_points = fk_leg_points_from_joint_angles(
        result.root_pos,
        result.root_rpy,
        result.joint_angles,
        shank_sample_count=2,
    )
    knee_semantic_count = _semantic_collision_count(terrain, leg_points.knee_pos_world, height_margin_m=0.01)
    shank_semantic_count = _semantic_collision_count(terrain, leg_points.shank_sample_world, height_margin_m=0.01)
    underbody = _underbody_points(result.root_pos, sample_count=5)
    underbody_semantic_count = _semantic_collision_count(terrain, underbody, height_margin_m=0.015)
    knee_height = _height_violation(terrain, leg_points.knee_pos_world, margin_m=0.01)
    shank_height = _height_violation(terrain, leg_points.shank_sample_world, margin_m=0.01)
    underbody_height = _height_violation(terrain, underbody, margin_m=0.015)
    fk_body_height_max = torch.maximum(
        knee_height.reshape(knee_height.shape[0], -1).amax(dim=1),
        shank_height.reshape(shank_height.shape[0], -1).amax(dim=1),
    )
    fk_body_height_max = torch.maximum(
        fk_body_height_max,
        underbody_height.reshape(underbody_height.shape[0], -1).amax(dim=1),
    )
    root_underbody_collision_count = underbody_semantic_count + _row_counts(underbody_height > 1.0e-5)
    body_leg_collision_count = knee_semantic_count + shank_semantic_count + root_underbody_collision_count
    return {
        "qp_touchdown_semantic_violation_count": touchdown_count,
        "qp_touchdown_on_small_count": touchdown_on_small,
        "qp_height_violation_max": height_max,
        "qp_max_semantic_constraint_violation": semantic_violation,
        "qp_max_height_constraint_violation": height_max,
        "qp_crossing_leg_count": crossing_leg_count,
        "qp_fk_semantic_collision_count": collision_count,
        "qp_fk_semantic_collision_rate": collision_rate,
        "qp_fk_semantic_min_clearance_over_semantic_m": min_clearance,
        "qp_fk_body_leg_collision_count": body_leg_collision_count,
        "qp_root_underbody_collision_count": root_underbody_collision_count,
        "qp_fk_knee_semantic_collision_count": knee_semantic_count,
        "qp_fk_shank_semantic_collision_count": shank_semantic_count,
        "qp_underbody_semantic_collision_count": underbody_semantic_count,
        "qp_fk_body_leg_height_violation_max": fk_body_height_max,
    }


__all__ = [
    "height_violation_max",
    "repair_touchdown_semantic_keepout",
    "safety_diagnostics",
    "touchdown_semantic_violation_count",
]
