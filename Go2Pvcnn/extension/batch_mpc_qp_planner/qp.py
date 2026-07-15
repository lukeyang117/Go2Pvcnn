"""Small fixed-shape safety QP step for the experimental MPC-QP backend."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.batch_mpc_planner.kinematics import (
    fk_feet_from_joint_angles,
    fk_leg_points_from_joint_angles,
    solve_joint_angles_from_trajectory,
)
from extension.batch_mpc_planner.parametric import command_frame_axes
from extension.batch_mpc_planner.terrain import height_at, semantic_at
from extension.batch_mpc_planner.types import MpcPlannerResult, MpcPlannerTerrain, MpcRobotState

from .config import MpcQpPlannerCfg
from .distance_field import fixed_repair_offsets


def _row_count(mask: Tensor) -> Tensor:
    return torch.count_nonzero(mask.reshape(mask.shape[0], -1), dim=1).to(dtype=torch.float32)


def _replace_touchdown(result: MpcPlannerResult, touchdown_w: Tensor, contact_state: Tensor | None = None) -> MpcPlannerResult:
    batch = int(result.root_pos.shape[0])
    event_cap = int(result.touchdown_seq.shape[2])
    horizon = int(result.root_pos.shape[1])
    touchdown_seq = touchdown_w.unsqueeze(2).expand(batch, 4, event_cap, 3).contiguous()
    planned_touchdown_w = touchdown_w.unsqueeze(1).expand(batch, horizon, 4, 3).contiguous()
    old_touchdown = result.planned_touchdown_w[:, 0]
    delta = touchdown_w - old_touchdown
    phase = torch.linspace(0.0, 1.0, horizon, dtype=result.foot_pos.dtype, device=result.foot_pos.device)
    foot_pos = result.foot_pos.clone()
    swing_weight = torch.where(result.contact_state, torch.zeros_like(result.contact_state, dtype=foot_pos.dtype), phase.view(1, -1, 1))
    foot_pos = foot_pos + delta[:, None, :, :] * swing_weight[..., None]
    terrain_z = planned_touchdown_w[..., 2]
    foot_pos[..., 2] = torch.maximum(foot_pos[..., 2], terrain_z + 0.025)
    joint_angles = solve_joint_angles_from_trajectory(result.root_pos, result.root_rpy, foot_pos)
    if contact_state is None:
        contact_state = result.contact_state
    return MpcPlannerResult(
        root_pos=result.root_pos,
        root_rpy=result.root_rpy,
        foot_pos=foot_pos,
        joint_angles=joint_angles,
        contact_state=contact_state,
        touchdown_seq=touchdown_seq,
        planned_touchdown_w=planned_touchdown_w,
        cost_total=result.cost_total,
        cost_breakdown=result.cost_breakdown,
        status=result.status,
        feasible=result.feasible,
        safe_fallback=result.safe_fallback,
        loss_breakdown=result.loss_breakdown,
        hard_reason_mask=result.hard_reason_mask,
    )


def _project_touchdown_semantic(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    cfg: MpcQpPlannerCfg,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    touchdown = result.planned_touchdown_w[:, 0].clone()
    semantic0 = semantic_at(terrain, touchdown[..., :2])
    bad = semantic0 != 0
    if not bool(torch.any(bad).item()):
        zero = torch.zeros((touchdown.shape[0],), dtype=touchdown.dtype, device=touchdown.device)
        return result, {
            "qp_semantic_repaired_touchdown_count": zero,
            "qp_touchdown_semantic_fallback_count": zero,
        }

    offsets = fixed_repair_offsets(
        radius_m=cfg.runtime.semantic_repair_radius_m,
        step_m=cfg.runtime.semantic_repair_ring_step_m,
        dtype=touchdown.dtype,
        device=touchdown.device,
    )
    candidates_xy = touchdown[..., None, :2] + offsets.view(1, 1, -1, 2)
    candidate_semantic = semantic_at(terrain, candidates_xy.reshape(touchdown.shape[0], -1, 2)).reshape(
        touchdown.shape[0], 4, -1
    )
    safe = candidate_semantic == 0
    offset_norm = torch.linalg.vector_norm(offsets, dim=-1).view(1, 1, -1)
    anchor_xy = torch.as_tensor(state.foot_pos, dtype=touchdown.dtype, device=touchdown.device)[..., :2]
    anchor_dist = torch.linalg.vector_norm(candidates_xy - anchor_xy[..., None, :], dim=-1)
    score = offset_norm + 0.05 * anchor_dist
    score = torch.where(safe, score, torch.full_like(score, 1.0e6))
    best_score, best_idx = score.min(dim=-1)
    has_safe = best_score < 1.0e5
    best_xy = candidates_xy.gather(2, best_idx[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    fallback_xy = anchor_xy
    repaired_xy = torch.where(has_safe[..., None], best_xy, fallback_xy)
    repaired_xy = torch.where(bad[..., None], repaired_xy, touchdown[..., :2])
    repaired_z = height_at(terrain, repaired_xy).to(dtype=touchdown.dtype, device=touchdown.device)
    repaired_touchdown = torch.cat((repaired_xy, repaired_z.unsqueeze(-1)), dim=-1)
    repaired = _replace_touchdown(result, repaired_touchdown)
    fallback = torch.logical_and(bad, torch.logical_not(has_safe))
    return repaired, {
        "qp_semantic_repaired_touchdown_count": _row_count(torch.logical_and(bad, has_safe)),
        "qp_touchdown_semantic_fallback_count": _row_count(fallback),
    }


def _terrain_path_variation(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    terminal_xy: Tensor,
    *,
    sample_count: int,
) -> Tensor:
    batch = int(terminal_xy.shape[0])
    phase = torch.linspace(0.0, 1.0, max(2, int(sample_count)), dtype=terminal_xy.dtype, device=terminal_xy.device)
    start = torch.as_tensor(state.root_pos, dtype=terminal_xy.dtype, device=terminal_xy.device)[:, :2]
    path = start[:, None, :] + (terminal_xy - start)[:, None, :] * phase.view(1, -1, 1)
    height = height_at(terrain, path.reshape(batch, -1, 2)).reshape(batch, -1)
    return height.amax(dim=1) - height.amin(dim=1)


def _apply_terrain_step_cap(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    cfg: MpcQpPlannerCfg,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    root0 = torch.as_tensor(state.root_pos, dtype=result.root_pos.dtype, device=result.root_pos.device)
    terminal_xy = result.root_pos[:, -1, :2]
    delta = terminal_xy - root0[:, :2]
    length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-8)
    variation = _terrain_path_variation(
        terrain,
        state,
        terminal_xy,
        sample_count=cfg.runtime.terrain_step_cap_sample_count,
    ).to(dtype=result.root_pos.dtype, device=result.root_pos.device)
    risk = (variation / max(float(cfg.runtime.terrain_height_variation_threshold_m), 1.0e-6)).clamp(0.0, 1.0)
    cap = float(cfg.runtime.terrain_step_cap_base_m) * (
        1.0 - (1.0 - float(cfg.runtime.terrain_step_cap_min_scale)) * risk
    )
    target_length = torch.minimum(length, cap)
    scale = (target_length / length).clamp(0.0, 1.0)
    if not bool(torch.any(scale < 0.999).item()):
        zero = torch.zeros((result.root_pos.shape[0],), dtype=result.root_pos.dtype, device=result.root_pos.device)
        return result, {
            "qp_step_cap_violation_count": zero,
            "qp_terrain_risk_reduces_target_progress": zero,
            "qp_height_variation_max": variation,
        }

    phase = torch.linspace(0.0, 1.0, result.root_pos.shape[1], dtype=result.root_pos.dtype, device=result.root_pos.device)
    capped_root = result.root_pos.clone()
    capped_terminal = root0[:, :2] + delta * scale.unsqueeze(-1)
    capped_root[..., :2] = root0[:, None, :2] + (capped_terminal - root0[:, :2])[:, None, :] * phase.view(1, -1, 1)
    root_height = height_at(terrain, capped_root[..., :2].reshape(result.root_pos.shape[0], -1, 2)).reshape(
        result.root_pos.shape[0], result.root_pos.shape[1]
    )
    capped_root[..., 2] = torch.maximum(capped_root[..., 2], root_height + 0.32)
    capped_root[:, 0, :] = result.root_pos[:, 0, :]

    capped_touchdown = result.planned_touchdown_w[:, 0].clone()
    foot0 = torch.as_tensor(state.foot_pos, dtype=result.foot_pos.dtype, device=result.foot_pos.device)
    foot_delta = capped_touchdown[..., :2] - foot0[..., :2]
    capped_touchdown[..., :2] = foot0[..., :2] + foot_delta * scale[:, None, None]
    capped_touchdown[..., 2] = height_at(terrain, capped_touchdown[..., :2]).to(
        dtype=capped_touchdown.dtype,
        device=capped_touchdown.device,
    )
    capped = _replace_touchdown(
        MpcPlannerResult(
            root_pos=capped_root,
            root_rpy=result.root_rpy,
            foot_pos=result.foot_pos,
            joint_angles=result.joint_angles,
            contact_state=result.contact_state,
            touchdown_seq=result.touchdown_seq,
            planned_touchdown_w=result.planned_touchdown_w,
            cost_total=result.cost_total,
            cost_breakdown=result.cost_breakdown,
            status=result.status,
            feasible=result.feasible,
            safe_fallback=result.safe_fallback,
            loss_breakdown=result.loss_breakdown,
            hard_reason_mask=result.hard_reason_mask,
        ),
        capped_touchdown,
    )
    reduced = scale < 0.999
    zero = torch.zeros_like(variation)
    return capped, {
        "qp_step_cap_violation_count": zero,
        "qp_terrain_risk_reduces_target_progress": reduced.to(dtype=result.root_pos.dtype),
        "qp_height_variation_max": variation,
    }


def _replace_foot_pos(result: MpcPlannerResult, foot_pos: Tensor) -> MpcPlannerResult:
    joint_angles = solve_joint_angles_from_trajectory(result.root_pos, result.root_rpy, foot_pos)
    return MpcPlannerResult(
        root_pos=result.root_pos,
        root_rpy=result.root_rpy,
        foot_pos=foot_pos,
        joint_angles=joint_angles,
        contact_state=result.contact_state,
        touchdown_seq=result.touchdown_seq,
        planned_touchdown_w=result.planned_touchdown_w,
        cost_total=result.cost_total,
        cost_breakdown=result.cost_breakdown,
        status=result.status,
        feasible=result.feasible,
        safe_fallback=result.safe_fallback,
        loss_breakdown=result.loss_breakdown,
        hard_reason_mask=result.hard_reason_mask,
    )


def _safe_point_delta(points_xy: Tensor, terrain: MpcPlannerTerrain, cfg: MpcQpPlannerCfg) -> tuple[Tensor, Tensor]:
    offsets = fixed_repair_offsets(
        radius_m=cfg.runtime.body_leg_xy_repair_radius_m,
        step_m=cfg.runtime.body_leg_xy_repair_step_m,
        dtype=points_xy.dtype,
        device=points_xy.device,
    )
    candidates = points_xy[..., None, :] + offsets.view(*((1,) * (points_xy.ndim - 1)), -1, 2)
    cand_semantic = semantic_at(terrain, candidates.reshape(points_xy.shape[0], -1, 2)).reshape(*points_xy.shape[:-1], -1)
    safe = cand_semantic == 0
    offset_norm = torch.linalg.vector_norm(offsets, dim=-1).view(*((1,) * (points_xy.ndim - 1)), -1)
    score = torch.where(safe, offset_norm, torch.full_like(offset_norm.expand_as(safe).to(dtype=points_xy.dtype), 1.0e6))
    best_score, best_idx = score.min(dim=-1)
    best_offset = offsets.index_select(0, best_idx.reshape(-1)).reshape(*points_xy.shape[:-1], 2)
    has_safe = best_score < 1.0e5
    return best_offset, has_safe


def _small_footprint_mask(
    terrain: MpcPlannerTerrain,
    points_xy: Tensor,
    *,
    radius_m: float,
    step_m: float,
) -> Tensor:
    offsets = fixed_repair_offsets(
        radius_m=float(radius_m),
        step_m=float(step_m),
        dtype=points_xy.dtype,
        device=points_xy.device,
    )
    candidates = points_xy[..., None, :] + offsets.view(*((1,) * (points_xy.ndim - 1)), -1, 2)
    semantic = semantic_at(terrain, candidates.reshape(points_xy.shape[0], -1, 2)).reshape(*points_xy.shape[:-1], -1)
    return torch.any(semantic == 1, dim=-1)


def _semantic_required_height(
    terrain: MpcPlannerTerrain,
    points_xy: Tensor,
    semantic: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    cfg: MpcQpPlannerCfg,
    margin_m: float,
) -> Tensor:
    terrain_height = height_at(terrain, points_xy).to(dtype=dtype, device=device)
    semantic_floor = float(cfg.runtime.body_leg_semantic_clearance_m) + float(margin_m)
    semantic_height = torch.where(semantic != 0, semantic_floor, float(margin_m))
    return torch.maximum(terrain_height + float(margin_m), semantic_height.to(dtype=dtype, device=device))


def _body_leg_collision_mask(
    terrain: MpcPlannerTerrain,
    knee: Tensor,
    shank: Tensor,
    cfg: MpcQpPlannerCfg,
) -> tuple[Tensor, Tensor, Tensor]:
    knee_semantic = semantic_at(terrain, knee[..., :2])
    knee_required = _semantic_required_height(
        terrain,
        knee[..., :2],
        knee_semantic,
        dtype=knee.dtype,
        device=knee.device,
        cfg=cfg,
        margin_m=0.01,
    )
    knee_hit = torch.logical_and(knee_semantic != 0, knee[..., 2] <= knee_required)
    shank_semantic = semantic_at(terrain, shank[..., :2])
    shank_required = _semantic_required_height(
        terrain,
        shank[..., :2],
        shank_semantic,
        dtype=shank.dtype,
        device=shank.device,
        cfg=cfg,
        margin_m=0.01,
    )
    shank_hit = torch.logical_and(shank_semantic != 0, shank[..., 2] <= shank_required)
    per_leg = torch.logical_or(knee_hit, shank_hit.any(dim=-1))
    return knee_hit, shank_hit, per_leg


def _body_leg_clearance_deficit(
    terrain: MpcPlannerTerrain,
    knee: Tensor,
    shank: Tensor,
    cfg: MpcQpPlannerCfg,
    *,
    margin_m: float,
) -> Tensor:
    knee_semantic = semantic_at(terrain, knee[..., :2])
    knee_required = _semantic_required_height(
        terrain,
        knee[..., :2],
        knee_semantic,
        dtype=knee.dtype,
        device=knee.device,
        cfg=cfg,
        margin_m=margin_m,
    )
    knee_deficit = torch.where(
        knee_semantic != 0,
        torch.relu(knee_required - knee[..., 2]),
        torch.zeros_like(knee[..., 2]),
    )
    shank_semantic = semantic_at(terrain, shank[..., :2])
    shank_required = _semantic_required_height(
        terrain,
        shank[..., :2],
        shank_semantic,
        dtype=shank.dtype,
        device=shank.device,
        cfg=cfg,
        margin_m=margin_m,
    )
    shank_deficit = torch.where(
        shank_semantic != 0,
        torch.relu(shank_required - shank[..., 2]),
        torch.zeros_like(shank[..., 2]),
    )
    return torch.maximum(knee_deficit.amax(dim=-1), shank_deficit.amax(dim=(-1, -2)))


def apply_fk_body_leg_root_lift(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    leg_points = fk_leg_points_from_joint_angles(
        result.root_pos,
        result.root_rpy,
        result.joint_angles,
        shank_sample_count=2,
    )
    deficit = _body_leg_clearance_deficit(
        terrain,
        leg_points.knee_pos_world,
        leg_points.shank_sample_world,
        cfg,
        margin_m=cfg.runtime.body_leg_root_lift_margin_m,
    ).clamp(max=float(cfg.runtime.body_leg_root_lift_max_m))
    if not bool(torch.any(deficit > 1.0e-6).item()):
        zero = torch.zeros((result.root_pos.shape[0],), dtype=result.root_pos.dtype, device=result.root_pos.device)
        return result, {"qp_fk_body_leg_root_lift_count": zero}

    left = torch.nn.functional.pad(deficit[:, :-1], (1, 0))
    right = torch.nn.functional.pad(deficit[:, 1:], (0, 1))
    smooth_lift = torch.maximum(deficit, torch.maximum(left, right) * 0.5)
    root_pos = result.root_pos.clone()
    root_pos[..., 2] = root_pos[..., 2] + smooth_lift
    joint_angles = solve_joint_angles_from_trajectory(root_pos, result.root_rpy, result.foot_pos)
    lifted = MpcPlannerResult(
        root_pos=root_pos,
        root_rpy=result.root_rpy,
        foot_pos=result.foot_pos,
        joint_angles=joint_angles,
        contact_state=result.contact_state,
        touchdown_seq=result.touchdown_seq,
        planned_touchdown_w=result.planned_touchdown_w,
        cost_total=result.cost_total,
        cost_breakdown=result.cost_breakdown,
        status=result.status,
        feasible=result.feasible,
        safe_fallback=result.safe_fallback,
        loss_breakdown=result.loss_breakdown,
        hard_reason_mask=result.hard_reason_mask,
    )
    return lifted, {"qp_fk_body_leg_root_lift_count": _row_count(smooth_lift > 1.0e-6)}


def apply_low_small_crossing_root_lift(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    foot_semantic = semantic_at(terrain, result.foot_pos[..., :2])
    over_small = foot_semantic == 1
    if not bool(torch.any(over_small).item()):
        zero = torch.zeros((result.root_pos.shape[0],), dtype=result.root_pos.dtype, device=result.root_pos.device)
        return result, {"qp_low_small_crossing_root_lift_count": zero}

    horizon_crossing = over_small.any(dim=(1, 2))
    smooth_lift = horizon_crossing[:, None].to(dtype=result.root_pos.dtype) * float(
        cfg.runtime.low_small_crossing_root_lift_m
    )
    smooth_lift = smooth_lift.expand(-1, result.root_pos.shape[1])
    root_pos = result.root_pos.clone()
    root_pos[..., 2] = root_pos[..., 2] + smooth_lift
    joint_angles = solve_joint_angles_from_trajectory(root_pos, result.root_rpy, result.foot_pos)
    lifted = MpcPlannerResult(
        root_pos=root_pos,
        root_rpy=result.root_rpy,
        foot_pos=result.foot_pos,
        joint_angles=joint_angles,
        contact_state=result.contact_state,
        touchdown_seq=result.touchdown_seq,
        planned_touchdown_w=result.planned_touchdown_w,
        cost_total=result.cost_total,
        cost_breakdown=result.cost_breakdown,
        status=result.status,
        feasible=result.feasible,
        safe_fallback=result.safe_fallback,
        loss_breakdown=result.loss_breakdown,
        hard_reason_mask=result.hard_reason_mask,
    )
    return lifted, {"qp_low_small_crossing_root_lift_count": _row_count(smooth_lift > 1.0e-6)}


def apply_fk_body_leg_xy_repair(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    repaired = result
    cumulative_mask = torch.zeros(
        repaired.foot_pos.shape[:-1],
        dtype=torch.bool,
        device=repaired.foot_pos.device,
    )
    pass_count = max(1, int(cfg.runtime.body_leg_xy_repair_passes))
    for _ in range(pass_count):
        leg_points = fk_leg_points_from_joint_angles(
            repaired.root_pos,
            repaired.root_rpy,
            repaired.joint_angles,
            shank_sample_count=2,
        )
        _, _, active = _body_leg_collision_mask(terrain, leg_points.knee_pos_world, leg_points.shank_sample_world, cfg)
        if not bool(torch.any(active).item()):
            break

        offsets = fixed_repair_offsets(
            radius_m=cfg.runtime.body_leg_xy_repair_radius_m,
            step_m=cfg.runtime.body_leg_xy_repair_step_m,
            dtype=repaired.foot_pos.dtype,
            device=repaired.foot_pos.device,
        )
        best_score = torch.full(
            repaired.foot_pos.shape[:-1],
            1.0e9,
            dtype=repaired.foot_pos.dtype,
            device=repaired.foot_pos.device,
        )
        best_foot_pos = repaired.foot_pos
        for offset in offsets:
            candidate_foot = repaired.foot_pos.clone()
            candidate_foot[..., :2] = torch.where(
                active[..., None],
                candidate_foot[..., :2] + offset.view(1, 1, 1, 2) * float(cfg.runtime.body_leg_xy_repair_gain),
                candidate_foot[..., :2],
            )
            terrain_z = height_at(
                terrain,
                candidate_foot[..., :2].reshape(candidate_foot.shape[0], -1, 2),
            ).reshape(candidate_foot.shape[:-1]).to(dtype=candidate_foot.dtype, device=candidate_foot.device)
            candidate_z = torch.maximum(
                candidate_foot[..., 2],
                terrain_z + float(cfg.runtime.body_leg_xy_repair_height_margin_m),
            )
            candidate_foot[..., 2] = torch.where(active, candidate_z, candidate_foot[..., 2])
            candidate_joint = solve_joint_angles_from_trajectory(repaired.root_pos, repaired.root_rpy, candidate_foot)
            candidate_points = fk_leg_points_from_joint_angles(
                repaired.root_pos,
                repaired.root_rpy,
                candidate_joint,
                shank_sample_count=2,
            )
            candidate_knee_hit, candidate_shank_hit, _ = _body_leg_collision_mask(
                terrain,
                candidate_points.knee_pos_world,
                candidate_points.shank_sample_world,
                cfg,
            )
            collision_score = candidate_knee_hit.to(dtype=candidate_foot.dtype) + candidate_shank_hit.to(
                dtype=candidate_foot.dtype
            ).sum(dim=-1)
            offset_cost = torch.linalg.vector_norm(offset).to(dtype=candidate_foot.dtype) * 0.01
            score = collision_score * 1000.0 + offset_cost
            improve = torch.logical_and(active, score < best_score)
            best_score = torch.where(improve, score, best_score)
            best_foot_pos = torch.where(improve[..., None], candidate_foot, best_foot_pos)

        foot_pos = best_foot_pos
        cumulative_mask = torch.logical_or(cumulative_mask, active)
        repaired = _replace_foot_pos(repaired, foot_pos)

    return repaired, {"qp_fk_body_leg_xy_repair_count": _row_count(cumulative_mask)}


def apply_fk_shank_clearance_lift(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    leg_points = fk_leg_points_from_joint_angles(
        result.root_pos,
        result.root_rpy,
        result.joint_angles,
        shank_sample_count=2,
    )
    shank = leg_points.shank_sample_world
    knee = leg_points.knee_pos_world
    shank_semantic = semantic_at(terrain, shank[..., :2])
    shank_terrain_z = height_at(terrain, shank[..., :2]).to(dtype=shank.dtype, device=shank.device)
    shank_deficit = torch.relu(shank_terrain_z + 0.015 - shank[..., 2])
    shank_semantic_hit = shank_semantic != 0
    shank_semantic_deficit = torch.where(shank_semantic_hit, shank_deficit + 0.05, shank_deficit)
    shank_per_leg_frame = shank_semantic_deficit.amax(dim=-1)
    shank_semantic_frame = shank_semantic_hit.any(dim=-1)

    knee_semantic = semantic_at(terrain, knee[..., :2])
    knee_terrain_z = height_at(terrain, knee[..., :2]).to(dtype=knee.dtype, device=knee.device)
    knee_deficit = torch.relu(knee_terrain_z + 0.015 - knee[..., 2])
    knee_semantic_hit = knee_semantic != 0
    knee_semantic_deficit = torch.where(knee_semantic_hit, knee_deficit + 0.05, knee_deficit)
    per_leg_frame = torch.maximum(shank_per_leg_frame, knee_semantic_deficit)
    semantic_frame = torch.logical_or(shank_semantic_frame, knee_semantic_hit)
    swing_mask = torch.logical_not(result.contact_state).to(dtype=result.foot_pos.dtype)
    lift = torch.where(semantic_frame, per_leg_frame, per_leg_frame * swing_mask)
    if not bool(torch.any(lift > 1.0e-6).item()):
        zero = torch.zeros((result.root_pos.shape[0],), dtype=result.root_pos.dtype, device=result.root_pos.device)
        return result, {"qp_fk_shank_clearance_lift_count": zero}

    foot_pos = result.foot_pos.clone()
    foot_pos[..., 2] = foot_pos[..., 2] + lift
    lifted = _replace_foot_pos(result, foot_pos)
    return lifted, {"qp_fk_shank_clearance_lift_count": _row_count(lift > 1.0e-6)}


def suppress_fk_low_small_contact_state(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    fk_foot = fk_feet_from_joint_angles(
        result.root_pos,
        result.root_rpy,
        result.joint_angles,
    )
    fk_foot_semantic = semantic_at(terrain, fk_foot[..., :2])
    suppress = torch.logical_and(result.contact_state, fk_foot_semantic == 1)
    if not bool(torch.any(suppress).item()):
        zero = torch.zeros((result.root_pos.shape[0],), dtype=result.root_pos.dtype, device=result.root_pos.device)
        return result, {"qp_fk_low_small_contact_suppressed_count": zero}
    contact_state = torch.where(suppress, torch.zeros_like(result.contact_state, dtype=torch.bool), result.contact_state)
    cleaned = MpcPlannerResult(
        root_pos=result.root_pos,
        root_rpy=result.root_rpy,
        foot_pos=result.foot_pos,
        joint_angles=result.joint_angles,
        contact_state=contact_state,
        touchdown_seq=result.touchdown_seq,
        planned_touchdown_w=result.planned_touchdown_w,
        cost_total=result.cost_total,
        cost_breakdown=result.cost_breakdown,
        status=result.status,
        feasible=result.feasible,
        safe_fallback=result.safe_fallback,
        loss_breakdown=result.loss_breakdown,
        hard_reason_mask=result.hard_reason_mask,
    )
    return cleaned, {"qp_fk_low_small_contact_suppressed_count": _row_count(suppress)}


def _apply_low_small_swing_over_repair(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    foot_xy = result.foot_pos[..., :2]
    batch, horizon, legs = int(foot_xy.shape[0]), int(foot_xy.shape[1]), int(foot_xy.shape[2])
    offsets = fixed_repair_offsets(
        radius_m=cfg.runtime.low_small_swing_repair_radius_m,
        step_m=cfg.runtime.low_small_swing_repair_step_m,
        dtype=foot_xy.dtype,
        device=foot_xy.device,
    )
    candidates = foot_xy[..., None, :] + offsets.view(1, 1, 1, -1, 2)
    cand_semantic = semantic_at(terrain, candidates.reshape(batch, -1, 2)).reshape(batch, horizon, legs, -1)
    small = cand_semantic == 1
    swing = torch.logical_not(result.contact_state)
    needs_repair = torch.logical_and(small.any(dim=-1), swing)
    if not bool(torch.any(needs_repair).item()):
        zero = torch.zeros((batch,), dtype=result.root_pos.dtype, device=result.root_pos.device)
        return result, {"qp_low_small_swing_over_repair_count": zero}

    offset_norm = torch.linalg.vector_norm(offsets, dim=-1).view(1, 1, 1, -1)
    score = torch.where(small, offset_norm, torch.full_like(offset_norm.expand_as(small).to(dtype=foot_xy.dtype), 1.0e6))
    best_idx = score.argmin(dim=-1)
    target_xy = candidates.gather(3, best_idx[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)
    terrain_z = height_at(terrain, target_xy.reshape(batch, -1, 2)).reshape(batch, horizon, legs).to(
        dtype=result.foot_pos.dtype,
        device=result.foot_pos.device,
    )
    blend = float(cfg.runtime.low_small_swing_xy_blend)
    repaired_xy = foot_xy * (1.0 - blend) + target_xy * blend
    repaired_z = torch.maximum(result.foot_pos[..., 2], terrain_z + float(cfg.runtime.low_small_swing_clearance_m))
    foot_pos = result.foot_pos.clone()
    foot_pos[..., :2] = torch.where(needs_repair[..., None], repaired_xy, foot_pos[..., :2])
    foot_pos[..., 2] = torch.where(needs_repair, repaired_z, foot_pos[..., 2])
    repaired = _replace_foot_pos(result, foot_pos)
    return repaired, {"qp_low_small_swing_over_repair_count": _row_count(needs_repair)}


def _apply_low_small_contact_over_repair(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    command: Tensor,
    cfg: MpcQpPlannerCfg,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    footprint_radius = max(float(cfg.runtime.low_small_swing_repair_radius_m), 0.06)
    footprint_step = min(float(cfg.runtime.low_small_swing_repair_step_m), 0.03)
    foot_over_small = _small_footprint_mask(
        terrain,
        result.foot_pos[..., :2],
        radius_m=footprint_radius,
        step_m=footprint_step,
    )
    fk_foot = fk_leg_points_from_joint_angles(
        result.root_pos,
        result.root_rpy,
        result.joint_angles,
        shank_sample_count=1,
    ).foot_pos_world
    fk_foot_over_small = _small_footprint_mask(
        terrain,
        fk_foot[..., :2],
        radius_m=footprint_radius,
        step_m=footprint_step,
    )
    contact_over_small = torch.logical_and(
        torch.logical_or(foot_over_small, fk_foot_over_small),
        result.contact_state,
    )
    if not bool(torch.any(contact_over_small).item()):
        zero = torch.zeros((result.root_pos.shape[0],), dtype=result.root_pos.dtype, device=result.root_pos.device)
        return result, {"qp_low_small_contact_over_repair_count": zero}

    heading, _left, _active = command_frame_axes(
        command.to(dtype=result.foot_pos.dtype, device=result.foot_pos.device),
        result.root_rpy[:, 0, 2],
        linear_eps=1.0e-6,
    )
    reland_xy = result.foot_pos[..., :2] + heading[:, None, None, :] * float(
        cfg.runtime.low_small_contact_reland_forward_m
    )
    reland_z = height_at(terrain, reland_xy.reshape(result.foot_pos.shape[0], -1, 2)).reshape(
        result.foot_pos.shape[:-1]
    ).to(dtype=result.foot_pos.dtype, device=result.foot_pos.device)
    repaired_z = torch.maximum(result.foot_pos[..., 2], reland_z + float(cfg.runtime.low_small_swing_clearance_m))
    foot_pos = result.foot_pos.clone()
    foot_pos[..., :2] = torch.where(contact_over_small[..., None], reland_xy, foot_pos[..., :2])
    foot_pos[..., 2] = torch.where(contact_over_small, repaired_z, foot_pos[..., 2])
    repaired = _replace_foot_pos(result, foot_pos)
    return repaired, {"qp_low_small_contact_over_repair_count": _row_count(contact_over_small)}


def apply_safety_qp_step(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    cfg: MpcQpPlannerCfg,
) -> tuple[MpcPlannerResult, dict[str, Tensor]]:
    """Apply one projected safety QP iteration.

    The current implementation solves the first compact active set explicitly:
    touchdown semantic keepout and terrain-risk step caps. It is intentionally
    fixed-shape and GPU-resident, so it can be iterated as RTI/SQP refinement.
    """
    projected, sem_diag = _project_touchdown_semantic(result, terrain, state, cfg)
    capped, cap_diag = _apply_terrain_step_cap(projected, terrain, state, cfg)
    over_repaired, over_diag = _apply_low_small_swing_over_repair(capped, terrain, cfg)
    contact_repaired, contact_diag = _apply_low_small_contact_over_repair(over_repaired, terrain, command, cfg)
    crossing_lifted, crossing_lift_diag = apply_low_small_crossing_root_lift(contact_repaired, terrain, cfg)
    xy_repaired, xy_diag = apply_fk_body_leg_xy_repair(crossing_lifted, terrain, cfg)
    root_lifted, root_diag = apply_fk_body_leg_root_lift(xy_repaired, terrain, cfg)
    lifted, fk_diag = apply_fk_shank_clearance_lift(root_lifted, terrain)
    diagnostics = {}
    diagnostics.update(sem_diag)
    diagnostics.update(cap_diag)
    diagnostics.update(over_diag)
    diagnostics.update(contact_diag)
    diagnostics.update(crossing_lift_diag)
    diagnostics.update(xy_diag)
    diagnostics.update(root_diag)
    diagnostics.update(fk_diag)
    return lifted, diagnostics


__all__ = [
    "apply_fk_body_leg_root_lift",
    "apply_fk_body_leg_xy_repair",
    "apply_fk_shank_clearance_lift",
    "apply_low_small_crossing_root_lift",
    "apply_safety_qp_step",
    "suppress_fk_low_small_contact_state",
]
