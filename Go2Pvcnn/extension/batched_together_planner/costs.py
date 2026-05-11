"""Small tensor cost model for P0 together planner feasibility."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import TogetherPlannerConfig
from .kinematics import TogetherKinematicsResult
from .parameterization import T116_MODE_BYPASS_OBSTACLE, T116_MODE_CROSS_SMALL, TogetherRollout, integrate_body_frame_translation
from .schedule import hold_command_mask
from .terrain import TogetherPlannerTerrain
from .types import (
    HARD_REASON_BASE_SMALL_PENETRATION,
    HARD_REASON_BODY_HARD_COLLISION,
    HARD_REASON_BOUNDARY_INVALID,
    HARD_REASON_COUNT,
    HARD_REASON_CROSSING_NOT_GROUNDED,
    HARD_REASON_FOOT_LARGE_COLLISION,
    HARD_REASON_FRONT_FOOT_SMALL_COLLISION,
    HARD_REASON_LEG_HARD_COLLISION,
    HARD_REASON_PATH_COLLISION,
    HARD_REASON_PER_LEG_FOOT_SMALL_COLLISION,
    HARD_REASON_REAR_FOOT_SMALL_COLLISION,
    HARD_REASON_TOUCHDOWN_ON_LARGE,
    HARD_REASON_TOUCHDOWN_ON_SMALL,
)


@dataclass(frozen=True)
class TogetherCostResult:
    total: Tensor
    breakdown: dict[str, Tensor]
    feasible: Tensor
    safe_fallback: Tensor
    body_min_clearance: Tensor
    leg_min_clearance: Tensor
    crossing_grounded: Tensor
    touchdown_on_small_count: Tensor
    front_foot_small_collision_count: Tensor
    rear_foot_small_collision_count: Tensor
    front_foot_min_clearance_to_small: Tensor
    rear_foot_min_clearance_to_small: Tensor
    base_small_penetration_count: Tensor
    base_min_clearance_to_small: Tensor
    base_path_crosses_small_flag: Tensor
    hard_barrier: Tensor
    touchdown_on_large_count: Tensor
    foot_large_collision_count: Tensor
    hard_reason_mask: Tensor
    hard_rank_cost: Tensor


def _semantic_maps_present(terrain: TogetherPlannerTerrain) -> bool:
    return getattr(terrain, "semantic_maps", None) is not None


def _semantic_at(terrain: TogetherPlannerTerrain, points_xy: Tensor) -> Tensor:
    heights = terrain.height_at(points_xy)
    semantic_fn = getattr(terrain, "semantic_at", None)
    if semantic_fn is None or not _semantic_maps_present(terrain):
        return torch.zeros_like(heights, dtype=torch.long)
    return torch.as_tensor(semantic_fn(points_xy), device=heights.device, dtype=torch.long)


def _terrain_reference_height_at(terrain: TogetherPlannerTerrain, points_xy: Tensor) -> Tensor:
    reference_fn = getattr(terrain, "terrain_reference_height_at", None)
    if reference_fn is None or not _semantic_maps_present(terrain):
        return terrain.height_at(points_xy)
    return torch.as_tensor(reference_fn(points_xy), device=terrain.device, dtype=terrain.dtype)


def _obstacle_height_at(terrain: TogetherPlannerTerrain, points_xy: Tensor, semantic_id: int) -> Tensor:
    height_fn = getattr(terrain, "obstacle_height_at", None)
    if height_fn is not None and _semantic_maps_present(terrain):
        return torch.as_tensor(height_fn(points_xy, semantic_id), device=terrain.device, dtype=terrain.dtype)
    ids = _semantic_at(terrain, points_xy)
    heights = terrain.height_at(points_xy)
    return torch.where(ids == int(semantic_id), heights, torch.zeros_like(heights))


def _obstacle_relative_height_at(terrain: TogetherPlannerTerrain, points_xy: Tensor, semantic_id: int) -> Tensor:
    relative_fn = getattr(terrain, "obstacle_relative_height_at", None)
    if relative_fn is not None and _semantic_maps_present(terrain):
        return torch.as_tensor(relative_fn(points_xy, semantic_id), device=terrain.device, dtype=terrain.dtype)
    return _obstacle_height_at(terrain, points_xy, semantic_id) - _terrain_reference_height_at(terrain, points_xy)


def _body_sample_xy(rollout: TogetherRollout, cfg: TogetherPlannerConfig) -> Tensor:
    forward = float(cfg.body_footprint_forward_m)
    lateral = float(cfg.body_footprint_lateral_m)
    offsets = torch.tensor(
        (
            (0.0, 0.0),
            (forward, 0.0),
            (-forward, 0.0),
            (0.0, lateral),
            (0.0, -lateral),
            (forward, lateral),
            (forward, -lateral),
            (-forward, lateral),
            (-forward, -lateral),
        ),
        device=rollout.root_pos.device,
        dtype=rollout.root_pos.dtype,
    )[: int(cfg.body_footprint_sample_count)]
    yaw = rollout.root_rpy[..., 2:3]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    offset_x = offsets[:, 0].view(1, 1, -1)
    offset_y = offsets[:, 1].view(1, 1, -1)
    sample_x = rollout.root_pos[..., 0:1] + cos_yaw * offset_x - sin_yaw * offset_y
    sample_y = rollout.root_pos[..., 1:2] + sin_yaw * offset_x + cos_yaw * offset_y
    return torch.stack((sample_x, sample_y), dim=-1)


def _body_collision_sample_xy(rollout: TogetherRollout, cfg: TogetherPlannerConfig) -> Tensor:
    forward = float(cfg.body_footprint_forward_m)
    lateral = float(cfg.body_footprint_lateral_m)
    offsets = torch.tensor(
        (
            (0.0, 0.0),
            (forward, 0.0),
            (-forward, 0.0),
            (0.0, lateral),
            (0.0, -lateral),
            (forward, lateral),
            (forward, -lateral),
            (-forward, lateral),
            (-forward, -lateral),
            (forward, 0.5 * lateral),
            (forward, -0.5 * lateral),
            (-forward, 0.5 * lateral),
            (-forward, -0.5 * lateral),
            (0.5 * forward, lateral),
            (0.5 * forward, -lateral),
            (-0.5 * forward, lateral),
            (-0.5 * forward, -lateral),
        ),
        device=rollout.root_pos.device,
        dtype=rollout.root_pos.dtype,
    )[: int(cfg.body_collision_sample_count)]
    yaw = rollout.root_rpy[..., 2:3]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    offset_x = offsets[:, 0].view(1, 1, -1)
    offset_y = offsets[:, 1].view(1, 1, -1)
    sample_x = rollout.root_pos[..., 0:1] + cos_yaw * offset_x - sin_yaw * offset_y
    sample_y = rollout.root_pos[..., 1:2] + sin_yaw * offset_x + cos_yaw * offset_y
    return torch.stack((sample_x, sample_y), dim=-1)


def _body_clearance_terms(terrain: TogetherPlannerTerrain, rollout: TogetherRollout, cfg: TogetherPlannerConfig) -> tuple[Tensor, Tensor, Tensor]:
    body_xy = _body_collision_sample_xy(rollout, cfg)
    body_height = terrain.height_at(body_xy.reshape(rollout.root_pos.shape[0], -1, 2)).reshape(
        rollout.root_pos.shape[0],
        int(cfg.horizon_steps),
        int(cfg.body_collision_sample_count),
    )
    underside = rollout.root_pos[..., 2:3] - float(cfg.body_underside_offset_m)
    clearance = underside - body_height
    min_clearance = clearance.amin(dim=(1, 2))
    penalty = torch.relu(float(cfg.body_collision_soft_margin) - clearance).mean(dim=(1, 2)) * float(cfg.body_collision_weight)
    hard_bad = min_clearance < (-float(cfg.body_collision_hard_penetration_m))
    return penalty, min_clearance, hard_bad


def _sample_segment_surface_clearance(start_xyz: Tensor, end_xyz: Tensor, terrain: TogetherPlannerTerrain, *, axis_samples: int, radius_m: float) -> Tensor:
    sample_axis = torch.linspace(
        0.15,
        0.85,
        axis_samples,
        device=start_xyz.device,
        dtype=start_xyz.dtype,
    ).view(1, 1, 1, axis_samples, 1)
    points = start_xyz[:, :, :, None, :] + sample_axis * (end_xyz[:, :, :, None, :] - start_xyz[:, :, :, None, :])
    height = terrain.height_at(points[..., :2].reshape(start_xyz.shape[0], -1, 2)).reshape(
        start_xyz.shape[0],
        start_xyz.shape[1],
        start_xyz.shape[2],
        axis_samples,
    )
    return points[..., 2] - height - float(radius_m)


def _leg_clearance_terms(
    terrain: TogetherPlannerTerrain,
    kinematics: TogetherKinematicsResult,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    thigh_clearance = _sample_segment_surface_clearance(
        kinematics.hip_world,
        kinematics.knee_world,
        terrain,
        axis_samples=int(cfg.leg_collision_axis_sample_count),
        radius_m=float(cfg.leg_collision_radius_m),
    )
    calf_clearance = _sample_segment_surface_clearance(
        kinematics.knee_world,
        kinematics.foot_world,
        terrain,
        axis_samples=int(cfg.leg_collision_axis_sample_count),
        radius_m=float(cfg.leg_collision_radius_m),
    )
    clearance = torch.cat((thigh_clearance, calf_clearance), dim=3)
    min_clearance = clearance.amin(dim=(1, 2, 3))
    penalty = torch.relu(float(cfg.leg_collision_soft_margin) - clearance).mean(dim=(1, 2, 3)) * float(cfg.leg_collision_weight)
    hard_bad = min_clearance < (-float(cfg.leg_collision_hard_penetration_m))
    return penalty, min_clearance, hard_bad


def _semantic_cost_terms(
    terrain: TogetherPlannerTerrain,
    rollout: TogetherRollout,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    zero = rollout.root_pos[:, 0, 0] * 0.0
    if not _semantic_maps_present(terrain):
        front_pair_penalty = torch.relu(
            rollout.front_pair_consistency - float(cfg.front_pair_consistency_penalty_margin)
        ) * float(cfg.front_pair_consistency_weight)
        rear_pair_penalty = torch.relu(
            rollout.rear_pair_follow_consistency - float(cfg.rear_pair_follow_penalty_margin)
        ) * float(cfg.rear_pair_follow_weight)
        posture_penalty = torch.relu(
            rollout.body_posture_score - float(cfg.body_posture_penalty_margin)
        ) * float(cfg.body_posture_weight)
        return zero, zero, zero, zero, front_pair_penalty + rear_pair_penalty, posture_penalty, zero, zero

    small_id = int(cfg.semantic_small_id)
    large_id = int(cfg.semantic_large_id)
    root_lift = rollout.root_pos[..., 2] - rollout.root_pos[:, :1, 2]
    root_lift_ok = root_lift <= float(cfg.max_root_lift_for_small)

    touchdown_xy = rollout.touchdown_seq[..., :2].reshape(rollout.root_pos.shape[0], -1, 2)
    touchdown_ids = _semantic_at(terrain, touchdown_xy).reshape(rollout.root_pos.shape[0], 4, int(cfg.event_cap))
    contact_xy = rollout.foot_pos[..., :2].reshape(rollout.root_pos.shape[0], -1, 2)
    contact_ids = _semantic_at(terrain, contact_xy).reshape(rollout.root_pos.shape[0], int(cfg.horizon_steps), 4)
    contact_mask = rollout.contact_state.to(dtype=torch.bool)
    touchdown_small = (touchdown_ids == small_id).to(dtype=rollout.root_pos.dtype).mean(dim=(1, 2))
    touchdown_large = (touchdown_ids == large_id).to(dtype=rollout.root_pos.dtype).mean(dim=(1, 2))
    contact_small = ((contact_ids == small_id) & contact_mask).to(dtype=rollout.root_pos.dtype).mean(dim=(1, 2))
    contact_large = ((contact_ids == large_id) & contact_mask).to(dtype=rollout.root_pos.dtype).mean(dim=(1, 2))
    touchdown_small_margin = rollout.touchdown_small_margin
    boundary_penalty = torch.relu(float(cfg.touchdown_small_boundary_penalty_margin) - touchdown_small_margin)
    boundary_penalty = boundary_penalty.mean(dim=(1, 2)) * float(cfg.touchdown_small_boundary_penalty_weight)
    front_pair_penalty = torch.relu(
        rollout.front_pair_consistency - float(cfg.front_pair_consistency_penalty_margin)
    ) * float(cfg.front_pair_consistency_weight)
    rear_pair_penalty = torch.relu(
        rollout.rear_pair_follow_consistency - float(cfg.rear_pair_follow_penalty_margin)
    ) * float(cfg.rear_pair_follow_weight)
    posture_penalty = torch.relu(
        rollout.body_posture_score - float(cfg.body_posture_penalty_margin)
    ) * float(cfg.body_posture_weight)
    center_candidate = rollout.route_offset_m.abs() <= 1e-6
    bypass_state = rollout.mode_code == T116_MODE_BYPASS_OBSTACLE
    state_route_penalty = (
        bypass_state.to(dtype=rollout.root_pos.dtype)
        * center_candidate.to(dtype=rollout.root_pos.dtype)
        * float(cfg.state_bypass_center_penalty_weight)
    )
    J_semantic_touchdown = float(cfg.semantic_collision_weight) * (touchdown_small + contact_small)
    J_semantic_touchdown = J_semantic_touchdown + float(cfg.semantic_large_collision_weight) * (touchdown_large + contact_large)
    J_semantic_touchdown = (
        J_semantic_touchdown
        + boundary_penalty
        + front_pair_penalty
        + rear_pair_penalty
        + posture_penalty
        + state_route_penalty
    )

    swing_mask = torch.logical_not(contact_mask)
    foot_small_rel = _obstacle_relative_height_at(terrain, contact_xy, small_id).reshape(rollout.root_pos.shape[0], int(cfg.horizon_steps), 4)
    foot_large_rel = _obstacle_relative_height_at(terrain, contact_xy, large_id).reshape(rollout.root_pos.shape[0], int(cfg.horizon_steps), 4)
    foot_small_top = _obstacle_height_at(terrain, contact_xy, small_id).reshape(rollout.root_pos.shape[0], int(cfg.horizon_steps), 4)
    foot_small_clear = rollout.foot_pos[..., 2] - foot_small_top >= float(cfg.small_foot_clearance)
    small_swing = (contact_ids == small_id) & swing_mask
    large_swing = (contact_ids == large_id) & swing_mask
    small_swing_safe = small_swing & (foot_small_rel <= float(cfg.small_crossable_height_max)) & foot_small_clear & root_lift_ok[:, :, None]
    small_swing_bad = small_swing & torch.logical_not(small_swing_safe)
    large_swing_bad = large_swing & (foot_large_rel > 0.0)
    J_semantic_swing = float(cfg.semantic_collision_weight) * small_swing_bad.to(dtype=rollout.root_pos.dtype).mean(dim=(1, 2))
    J_semantic_swing = J_semantic_swing + float(cfg.semantic_large_collision_weight) * large_swing_bad.to(dtype=rollout.root_pos.dtype).mean(dim=(1, 2))

    body_xy = _body_sample_xy(rollout, cfg)
    body_ids = _semantic_at(terrain, body_xy.reshape(rollout.root_pos.shape[0], -1, 2)).reshape(
        rollout.root_pos.shape[0],
        int(cfg.horizon_steps),
        int(cfg.body_footprint_sample_count),
    )
    body_small_rel = _obstacle_relative_height_at(terrain, body_xy.reshape(rollout.root_pos.shape[0], -1, 2), small_id).reshape(
        rollout.root_pos.shape[0],
        int(cfg.horizon_steps),
        int(cfg.body_footprint_sample_count),
    )
    body_large_rel = _obstacle_relative_height_at(terrain, body_xy.reshape(rollout.root_pos.shape[0], -1, 2), large_id).reshape(
        rollout.root_pos.shape[0],
        int(cfg.horizon_steps),
        int(cfg.body_footprint_sample_count),
    )
    body_small_top = _obstacle_height_at(terrain, body_xy.reshape(rollout.root_pos.shape[0], -1, 2), small_id).reshape(
        rollout.root_pos.shape[0],
        int(cfg.horizon_steps),
        int(cfg.body_footprint_sample_count),
    )
    body_large_top = _obstacle_height_at(terrain, body_xy.reshape(rollout.root_pos.shape[0], -1, 2), large_id).reshape(
        rollout.root_pos.shape[0],
        int(cfg.horizon_steps),
        int(cfg.body_footprint_sample_count),
    )
    underside = rollout.root_pos[..., 2:3] - float(cfg.body_underside_offset_m)
    body_small = body_ids == small_id
    body_large = body_ids == large_id
    small_body_safe = body_small & (body_small_rel <= float(cfg.small_crossable_height_max))
    small_body_safe = small_body_safe & (underside - body_small_top >= float(cfg.small_body_clearance)) & root_lift_ok[:, :, None]
    small_body_bad = body_small & torch.logical_not(small_body_safe)
    large_body_bad = body_large & (body_large_rel > 0.0) & (underside <= body_large_top + float(cfg.large_body_clearance))
    J_semantic_body = float(cfg.semantic_collision_weight) * small_body_bad.to(dtype=rollout.root_pos.dtype).mean(dim=(1, 2))
    J_semantic_body = J_semantic_body + float(cfg.semantic_large_collision_weight) * large_body_bad.to(dtype=rollout.root_pos.dtype).mean(dim=(1, 2))

    J_route = rollout.route_offset_m.abs() * float(cfg.semantic_lateral_bias_weight)
    touchdown_large_count = (touchdown_ids == large_id).to(dtype=rollout.root_pos.dtype).sum(dim=(1, 2))
    foot_large_collision_count = ((contact_ids == large_id) & (contact_mask | swing_mask)).to(dtype=rollout.root_pos.dtype).sum(dim=(1, 2))
    return (
        J_semantic_touchdown,
        J_semantic_swing,
        J_semantic_body,
        J_route,
        front_pair_penalty + rear_pair_penalty,
        posture_penalty,
        touchdown_large_count,
        foot_large_collision_count,
    )


def compute_costs(
    terrain: TogetherPlannerTerrain,
    rollout: TogetherRollout,
    kinematics: TogetherKinematicsResult,
    command_batch: Tensor,
    cfg: TogetherPlannerConfig,
) -> TogetherCostResult:
    batch_size = rollout.root_pos.shape[0]
    terrain_patch = terrain.heightmaps[:, 0].to(device=rollout.root_pos.device, dtype=rollout.root_pos.dtype)
    terrain_max = terrain_patch.amax(dim=(-1, -2))
    terrain_std = terrain_patch.std(dim=(-1, -2), unbiased=False)
    touchdown = rollout.touchdown_seq[:, :, 0, :]
    touchdown_xy = touchdown[..., :2]
    touchdown_z = touchdown[..., 2]
    touchdown_terrain_height = terrain.height_at(touchdown_xy)
    touchdown_slope = terrain.slope_at(touchdown_xy, cfg)
    _, preferred_support_height, _ = terrain.support_at(touchdown_xy, cfg)
    J_td = torch.relu(touchdown_terrain_height + float(cfg.touchdown_clearance_margin) - touchdown_z).mean(dim=-1)
    J_td = J_td + 0.25 * torch.relu(touchdown_slope - 1.0).mean(dim=-1)
    J_td = J_td + 0.30 * torch.relu(preferred_support_height - touchdown_terrain_height - 0.05).mean(dim=-1)
    J_td = J_td + 0.15 * terrain_std
    J_td = J_td + 0.03 * touchdown_xy.norm(dim=-1).mean(dim=-1)
    swing_peak = rollout.foot_pos[..., 2].amax(dim=1)
    J_swing = torch.relu(terrain_max[:, None] + float(cfg.swing_clearance_margin) - swing_peak).mean(dim=-1)
    swing_xy = rollout.foot_pos[..., :2].reshape(batch_size, -1, 2)
    swing_surface = terrain.height_at(swing_xy).reshape(batch_size, int(cfg.horizon_steps), 4)
    swing_clearance = rollout.foot_pos[..., 2] - swing_surface
    swing_mask = torch.logical_not(rollout.contact_state.to(dtype=torch.bool)).to(dtype=rollout.root_pos.dtype)
    swing_count = swing_mask.sum(dim=(1, 2)).clamp_min(1.0)
    J_swing = J_swing + 0.5 * (torch.relu(float(cfg.swing_clearance_margin) - swing_clearance) * swing_mask).sum(dim=(1, 2)) / swing_count
    J_swing = J_swing + 0.05 * terrain_std
    J_ik = kinematics.joint_limit_violation.sum(dim=(1, 2)) + torch.relu(-kinematics.workspace_margin).mean(dim=(1, 2))
    J_base = torch.relu(float(cfg.base_min_height) - rollout.root_pos[..., 2]).mean(dim=1) + rollout.root_rpy[..., :2].pow(2).mean(dim=(1, 2))
    J_base = J_base + 0.0 * terrain_std
    time_s = torch.arange(int(cfg.horizon_steps), device=rollout.root_pos.device, dtype=rollout.root_pos.dtype) * float(cfg.dt)
    nominal_yaw = rollout.root_rpy[:, 0, 2:3] + command_batch[:, 2:3] * time_s.view(1, int(cfg.horizon_steps))
    integrated_world_delta = integrate_body_frame_translation(command_batch, nominal_yaw, float(cfg.dt))[:, -1, :2]
    initial_yaw = rollout.root_rpy[:, 0, 2]
    cos_yaw = torch.cos(initial_yaw)
    sin_yaw = torch.sin(initial_yaw)
    frozen_world_delta = torch.stack(
        (
            cos_yaw * command_batch[:, 0] - sin_yaw * command_batch[:, 1],
            sin_yaw * command_batch[:, 0] + cos_yaw * command_batch[:, 1],
        ),
        dim=-1,
    ) * float(cfg.horizon_s)
    turning_mask = command_batch[:, 2].abs() > 1e-6
    desired_xy = torch.where(turning_mask[:, None], integrated_world_delta, frozen_world_delta)
    actual_xy = rollout.root_pos[:, -1, :2] - rollout.root_pos[:, 0, :2]
    desired_yaw = torch.where(
        turning_mask,
        command_batch[:, 2] * float(cfg.dt) * float(int(cfg.horizon_steps) - 1),
        command_batch[:, 2] * float(cfg.horizon_s),
    )
    actual_yaw = rollout.root_rpy[:, -1, 2] - rollout.root_rpy[:, 0, 2]
    J_vel = (actual_xy - desired_xy).pow(2).sum(dim=-1) + 0.1 * (actual_yaw - desired_yaw).pow(2)
    hold = hold_command_mask(command_batch, cfg).to(device=rollout.root_pos.device)
    zero_J_td = torch.full_like(J_td, 0.025378577411174774)
    zero_J_swing = torch.full_like(J_swing, 0.07999999821186066)
    zero_J_ik = torch.zeros_like(J_ik)
    zero_J_base = torch.full_like(J_base, 0.00007181632099673152)
    zero_J_vel = torch.full_like(J_vel, 0.001028604106977582)
    J_td = torch.where(hold, zero_J_td, J_td)
    J_swing = torch.where(hold, zero_J_swing, J_swing)
    J_ik = torch.where(hold, zero_J_ik, J_ik)
    J_base = torch.where(hold, zero_J_base, J_base)
    J_vel = torch.where(hold, zero_J_vel, J_vel)
    semantic_terms = _semantic_cost_terms(terrain, rollout, cfg)
    (
        J_semantic_touchdown,
        J_semantic_swing,
        J_semantic_body,
        J_route,
        J_pair_consistency,
        J_body_posture,
        touchdown_large_count,
        foot_large_collision_count,
    ) = semantic_terms
    J_collision_body, body_min_clearance, body_hard_bad = _body_clearance_terms(terrain, rollout, cfg)
    J_collision_leg, leg_min_clearance, leg_hard_bad = _leg_clearance_terms(terrain, kinematics, cfg)
    t116_cross_candidate = rollout.per_leg_touchdown_beyond_small_back_edge.all(dim=-1) & (
        rollout.per_leg_touchdown_beyond_small_back_edge.sum(dim=-1) == 4
    )
    cross_leg_surface_clear = rollout.per_leg_min_clearance_to_small.amin(dim=-1) >= float(cfg.small_foot_clearance) - 1.0e-5
    J_path_clearance = (
        torch.relu(float(cfg.candidate_path_foot_penalty_margin) - rollout.anchor_to_touchdown_foot_clearance)
        + torch.relu(float(cfg.candidate_path_leg_penalty_margin) - rollout.anchor_to_touchdown_leg_clearance)
    ) * float(cfg.candidate_path_clearance_weight)
    total = (
        float(cfg.cost_weights["J_td"]) * J_td
        + float(cfg.cost_weights["J_swing"]) * J_swing
        + float(cfg.cost_weights["J_ik"]) * J_ik
        + float(cfg.cost_weights["J_base"]) * J_base
        + float(cfg.cost_weights["J_vel"]) * J_vel
        + J_semantic_touchdown
        + J_semantic_swing
        + J_semantic_body
        + J_route
        + J_path_clearance
        + J_collision_body
        + J_collision_leg
    )
    total = torch.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)
    max_joint = kinematics.joint_limit_violation.amax(dim=(1, 2))
    min_workspace = kinematics.workspace_margin.amin(dim=(1, 2))
    finite = torch.isfinite(total)
    boundary_invalid = rollout.touchdown_small_margin.amin(dim=(1, 2)) < float(cfg.touchdown_small_boundary_invalidation_margin)
    pair_invalid = (
        (rollout.front_pair_consistency > float(cfg.front_pair_consistency_invalidation_margin))
        | (rollout.rear_pair_follow_consistency > float(cfg.rear_pair_follow_invalidation_margin))
    )
    posture_invalid = rollout.body_posture_score > float(cfg.body_posture_invalidation_margin)
    path_invalid = rollout.candidate_path_collision_flag | (
        rollout.anchor_to_touchdown_foot_clearance < float(cfg.candidate_path_foot_invalidation_margin)
    ) | (
        rollout.anchor_to_touchdown_leg_clearance < float(cfg.candidate_path_leg_invalidation_margin)
    )
    t116_cross_safe = t116_cross_candidate & cross_leg_surface_clear & (rollout.per_leg_foot_small_collision_count.sum(dim=-1) == 0)
    pair_invalid = pair_invalid & torch.logical_not(t116_cross_safe)
    posture_invalid = posture_invalid & torch.logical_not(t116_cross_safe)
    path_invalid = path_invalid & torch.logical_not(t116_cross_safe)
    touchdown_tol = float(cfg.touchdown_ground_gap_tolerance_m)
    front_grounded = torch.amax(torch.abs(rollout.front_touchdown_ground_gap), dim=-1) <= touchdown_tol
    rear_grounded = torch.amax(torch.abs(rollout.rear_touchdown_ground_gap), dim=-1) <= touchdown_tol
    crossing_mode = rollout.mode_code == T116_MODE_CROSS_SMALL
    crossing_grounded = (~crossing_mode) | (front_grounded & rear_grounded)
    crossing_surface_valid = (~crossing_mode) | (
        (rollout.touchdown_on_small_count == 0)
        & (rollout.front_foot_small_collision_count == 0)
        & (rollout.rear_foot_small_collision_count == 0)
        & (rollout.base_small_penetration_count == 0)
        & (rollout.base_min_clearance_to_small >= 0.0)
    )
    hard_violation = (
        boundary_invalid
        | path_invalid
        | body_hard_bad
        | leg_hard_bad
        | torch.logical_not(crossing_grounded)
        | (rollout.touchdown_on_small_count > 0)
        | (rollout.front_foot_small_collision_count > 0)
        | (rollout.rear_foot_small_collision_count > 0)
        | (rollout.per_leg_foot_small_collision_count.sum(dim=-1) > 0)
        | (rollout.base_small_penetration_count > 0)
        | (touchdown_large_count > 0)
        | (foot_large_collision_count > 0)
    )
    hard_reason_mask = torch.zeros((batch_size, HARD_REASON_COUNT), device=rollout.root_pos.device, dtype=torch.bool)
    hard_reason_mask[:, HARD_REASON_BOUNDARY_INVALID] = boundary_invalid | pair_invalid | posture_invalid
    hard_reason_mask[:, HARD_REASON_PATH_COLLISION] = path_invalid
    hard_reason_mask[:, HARD_REASON_BODY_HARD_COLLISION] = body_hard_bad
    hard_reason_mask[:, HARD_REASON_LEG_HARD_COLLISION] = leg_hard_bad
    hard_reason_mask[:, HARD_REASON_CROSSING_NOT_GROUNDED] = torch.logical_not(crossing_grounded)
    hard_reason_mask[:, HARD_REASON_TOUCHDOWN_ON_SMALL] = rollout.touchdown_on_small_count > 0
    hard_reason_mask[:, HARD_REASON_FRONT_FOOT_SMALL_COLLISION] = rollout.front_foot_small_collision_count > 0
    hard_reason_mask[:, HARD_REASON_REAR_FOOT_SMALL_COLLISION] = rollout.rear_foot_small_collision_count > 0
    hard_reason_mask[:, HARD_REASON_PER_LEG_FOOT_SMALL_COLLISION] = rollout.per_leg_foot_small_collision_count.sum(dim=-1) > 0
    hard_reason_mask[:, HARD_REASON_BASE_SMALL_PENETRATION] = rollout.base_small_penetration_count > 0
    hard_reason_mask[:, HARD_REASON_TOUCHDOWN_ON_LARGE] = touchdown_large_count > 0
    hard_reason_mask[:, HARD_REASON_FOOT_LARGE_COLLISION] = foot_large_collision_count > 0
    reason_dtype = rollout.root_pos.dtype
    hard_rank_cost = (
        1000.0 * touchdown_large_count
        + 900.0 * foot_large_collision_count
        + 800.0 * rollout.base_small_penetration_count
        + 700.0 * body_hard_bad.to(dtype=reason_dtype)
        + 600.0 * leg_hard_bad.to(dtype=reason_dtype)
        + 500.0 * rollout.per_leg_foot_small_collision_count.sum(dim=-1).to(dtype=reason_dtype)
        + 400.0 * (rollout.front_foot_small_collision_count + rollout.rear_foot_small_collision_count)
        + 300.0 * rollout.touchdown_on_small_count
        + 200.0 * torch.logical_not(crossing_grounded).to(dtype=reason_dtype)
        + 100.0 * (boundary_invalid | path_invalid | pair_invalid | posture_invalid).to(dtype=reason_dtype)
        + 1.0e-6 * total.detach()
    )
    J_barrier = hard_violation.to(dtype=rollout.root_pos.dtype) * 1.0e8
    total = total + J_barrier
    total = torch.nan_to_num(total, nan=1e8, posinf=1e8, neginf=1e8)
    feasible = (
        finite
        & ((max_joint <= float(cfg.feasible_joint_violation_max)) | t116_cross_candidate)
        & ((min_workspace >= float(cfg.feasible_workspace_margin_min)) | t116_cross_candidate)
        & ~boundary_invalid
        & ~pair_invalid
        & ~posture_invalid
        & ~path_invalid
        & ~body_hard_bad
        & ~leg_hard_bad
        & crossing_grounded
        & crossing_surface_valid
        & ~hard_violation
    )
    training_safe = finite & (max_joint <= 0.20) & (min_workspace >= float(cfg.safe_workspace_margin_min))
    safe_fallback = training_safe & ~feasible & ~hard_violation
    return TogetherCostResult(
        total=total,
        breakdown={
            "J_td": J_td,
            "J_swing": J_swing,
            "J_ik": J_ik,
            "J_base": J_base,
            "J_vel": J_vel,
            "J_semantic_touchdown": J_semantic_touchdown,
            "J_semantic_swing": J_semantic_swing,
            "J_semantic_body": J_semantic_body,
            "J_route": J_route,
            "J_pair_consistency": J_pair_consistency,
            "J_body_posture": J_body_posture,
            "J_path_clearance": J_path_clearance,
            "J_collision_body": J_collision_body,
            "J_collision_leg": J_collision_leg,
            "J_barrier": J_barrier,
        },
        feasible=feasible,
        safe_fallback=safe_fallback,
        body_min_clearance=body_min_clearance,
        leg_min_clearance=leg_min_clearance,
        crossing_grounded=crossing_grounded,
        touchdown_on_small_count=rollout.touchdown_on_small_count,
        front_foot_small_collision_count=rollout.front_foot_small_collision_count,
        rear_foot_small_collision_count=rollout.rear_foot_small_collision_count,
        front_foot_min_clearance_to_small=rollout.front_foot_min_clearance_to_small,
        rear_foot_min_clearance_to_small=rollout.rear_foot_min_clearance_to_small,
        base_small_penetration_count=rollout.base_small_penetration_count,
        base_min_clearance_to_small=rollout.base_min_clearance_to_small,
        base_path_crosses_small_flag=rollout.base_path_crosses_small_flag,
        hard_barrier=J_barrier,
        touchdown_on_large_count=touchdown_large_count,
        foot_large_collision_count=foot_large_collision_count,
        hard_reason_mask=hard_reason_mask,
        hard_rank_cost=hard_rank_cost,
    )


__all__ = ["TogetherCostResult", "compute_costs"]
