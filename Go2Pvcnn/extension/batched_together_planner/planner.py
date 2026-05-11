"""Native torch-only P0 together planner core."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from .config import TogetherPlannerConfig, validate_config
from .costs import compute_costs
from .kinematics import evaluate_kinematics
from .parameterization import (
    T116_MODE_BYPASS_OBSTACLE,
    T116_MODE_CROSS_SMALL,
    classify_mode_and_geometry,
    expand_segment,
)
from .schedule import build_cross_small_schedule, build_fixed_schedule
from .terrain import TogetherPlannerTerrain, build_together_terrain_from_scanner
from .types import (
    HARD_REASON_COUNT,
    HARD_REASON_DIRECTION_VIOLATION,
    T116_CANDIDATE_COUNT,
    TogetherContactSchedule,
    TogetherPlannerResult,
    TogetherPlannerStatus,
    TogetherRobotState,
)


def _as_batch_tensor(value: Tensor, *, device: torch.device, dtype: torch.dtype, suffix: tuple[int, ...], name: str) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == len(suffix):
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != len(suffix) + 1 or tuple(tensor.shape[1:]) != suffix:
        raise ValueError(f"{name} must have shape [B, ...] with suffix {suffix}")
    return tensor


def _coerce_state(state: TogetherRobotState, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    root_pos = _as_batch_tensor(state.root_pos, device=device, dtype=dtype, suffix=(3,), name="root_pos")
    root_rpy = _as_batch_tensor(state.root_rpy, device=device, dtype=dtype, suffix=(3,), name="root_rpy")
    foot_pos = _as_batch_tensor(state.foot_pos, device=device, dtype=dtype, suffix=(4, 3), name="foot_pos")
    if state.joint_angles is None:
        joint_angles = torch.zeros((root_pos.shape[0], 12), device=device, dtype=dtype)
    else:
        joint_angles = _as_batch_tensor(state.joint_angles, device=device, dtype=dtype, suffix=(12,), name="joint_angles")
    if root_rpy.shape[0] != root_pos.shape[0] or foot_pos.shape[0] != root_pos.shape[0] or joint_angles.shape[0] != root_pos.shape[0]:
        raise ValueError("state tensors must share batch dimension")
    return root_pos, root_rpy, foot_pos, joint_angles


def _coerce_command(command_batch: Tensor, *, device: torch.device, dtype: torch.dtype, batch_size: int) -> Tensor:
    command = torch.as_tensor(command_batch, device=device, dtype=dtype)
    if command.ndim == 1:
        command = command.unsqueeze(0)
    if command.shape != (batch_size, 3):
        raise ValueError("command_batch must have shape [B, 3]")
    return command


ROUTE_CENTER = 0
ROUTE_LEFT = 1
ROUTE_RIGHT = 2
DIRECTION_FORWARD = 0
DIRECTION_BACKWARD = 1
DIRECTION_LATERAL_LEFT = 2
DIRECTION_LATERAL_RIGHT = 3
DIRECTION_IDLE = 4


def _t116_candidate_tables(mode_code: Tensor, *, device: torch.device, dtype: torch.dtype, cfg: TogetherPlannerConfig) -> tuple[Tensor, Tensor, Tensor]:
    if int(cfg.candidate_count) != T116_CANDIDATE_COUNT:
        raise ValueError("candidate_count must equal the fixed K=5 contract")
    beta_table = torch.tensor(
        (
            (1.00, 0.80, 0.60, 0.40, 0.20),
            (0.80, 0.65, 0.50, 0.35, 0.20),
            (0.60, 0.50, 0.40, 0.30, 0.20),
            (0.60, 0.40, 0.60, 0.40, 0.20),
        ),
        device=device,
        dtype=dtype,
    )
    route_table = torch.tensor(
        (
            (ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER),
            (ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER),
            (ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER, ROUTE_CENTER),
            (ROUTE_LEFT, ROUTE_LEFT, ROUTE_RIGHT, ROUTE_RIGHT, ROUTE_CENTER),
        ),
        device=device,
        dtype=torch.long,
    )
    route_sign_table = torch.tensor(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, -1.0, -1.0, 0.0),
        ),
        device=device,
        dtype=dtype,
    )
    return beta_table[mode_code], route_table[mode_code], route_sign_table[mode_code]


def _candidate_commands(command: Tensor, root_rpy: Tensor, candidate_betas: Tensor, route_signs: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    body_xy = command[:, :2]
    body_norm = torch.linalg.vector_norm(body_xy, dim=-1, keepdim=True)
    fallback = torch.ones_like(body_xy)
    fallback[:, 1] = 0.0
    body_dir = torch.where(body_norm > float(cfg.idle_command_eps), body_xy / body_norm.clamp_min(1e-6), fallback)
    body_normal = torch.stack((-body_dir[:, 1], body_dir[:, 0]), dim=-1)
    route_speed = float(cfg.semantic_lateral_offset_m) / max(float(cfg.horizon_s), 1e-6)
    scaled = command[:, None, :] * candidate_betas[..., None]
    route_xy = route_signs[..., None] * body_normal[:, None, :] * route_speed
    candidate_command = scaled.clone()
    candidate_command[..., :2] = candidate_command[..., :2] + route_xy
    return candidate_command


def _command_direction_id(command: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    command_xy = command[:, :2]
    abs_xy = command_xy.abs()
    longitudinal = abs_xy[:, 0] >= abs_xy[:, 1]
    forward = torch.where(
        command_xy[:, 0] >= 0.0,
        torch.full((command.shape[0],), DIRECTION_FORWARD, device=command.device, dtype=torch.long),
        torch.full((command.shape[0],), DIRECTION_BACKWARD, device=command.device, dtype=torch.long),
    )
    lateral = torch.where(
        command_xy[:, 1] >= 0.0,
        torch.full((command.shape[0],), DIRECTION_LATERAL_LEFT, device=command.device, dtype=torch.long),
        torch.full((command.shape[0],), DIRECTION_LATERAL_RIGHT, device=command.device, dtype=torch.long),
    )
    direction_id = torch.where(longitudinal, forward, lateral)
    idle = torch.linalg.vector_norm(command_xy, dim=-1) <= float(cfg.idle_command_eps)
    return torch.where(
        idle,
        torch.full((command.shape[0],), DIRECTION_IDLE, device=command.device, dtype=torch.long),
        direction_id,
    )


def _repeat_rows(tensor: Tensor, count: int) -> Tensor:
    shape = tensor.shape
    return tensor.unsqueeze(1).expand(shape[0], count, *shape[1:]).reshape(shape[0] * count, *shape[1:])


def _gather_candidate(tensor: Tensor, best_idx: Tensor, candidate_count: int) -> Tensor:
    batch_size = best_idx.shape[0]
    reshaped = tensor.reshape(batch_size, candidate_count, *tensor.shape[1:])
    gather_shape = (batch_size, 1) + (1,) * (reshaped.ndim - 2)
    index = best_idx.reshape(gather_shape).expand(batch_size, 1, *reshaped.shape[2:])
    return reshaped.gather(1, index).squeeze(1)


def _select_best_candidate(total: Tensor, feasible: Tensor, safe_fallback: Tensor, hard_rank_cost: Tensor) -> Tensor:
    batch_size = total.shape[0]
    candidate_count = total.shape[1]
    inf_fill = torch.full_like(total, 1e9)
    feasible_cost = torch.where(feasible, total, inf_fill)
    feasible_exists = feasible.any(dim=1)
    feasible_idx = torch.argmin(feasible_cost, dim=1)
    fallback_cost = torch.where(safe_fallback, total, inf_fill)
    fallback_exists = safe_fallback.any(dim=1)
    fallback_idx = torch.argmin(fallback_cost, dim=1)
    hard_idx = torch.argmin(hard_rank_cost, dim=1)
    base_idx = torch.where(feasible_exists, feasible_idx, torch.where(fallback_exists, fallback_idx, hard_idx))
    return base_idx.reshape(batch_size)


def _select_t116_candidate(
    total: Tensor,
    feasible: Tensor,
    safe_fallback: Tensor,
    mode_code: Tensor,
    route_ids: Tensor,
    candidate_betas: Tensor,
    hard_rank_cost: Tensor,
) -> Tensor:
    base_idx = _select_best_candidate(total, feasible, safe_fallback, hard_rank_cost)
    bypass_mode = mode_code == 3
    cross_mode = mode_code == 2
    non_center = route_ids != ROUTE_CENTER
    selectable = feasible | safe_fallback
    non_center_selectable = selectable & non_center
    inf_fill = torch.full_like(total, 1e9)
    non_center_cost = torch.where(non_center_selectable, total, inf_fill)
    non_center_exists = non_center_selectable.any(dim=1)
    non_center_idx = torch.argmin(non_center_cost, dim=1)
    nonzero_cross_selectable = feasible & (candidate_betas > 1.0e-6)
    nonzero_cross_cost = torch.where(nonzero_cross_selectable, total, inf_fill)
    nonzero_cross_exists = nonzero_cross_selectable.any(dim=1)
    nonzero_cross_idx = torch.argmin(nonzero_cross_cost, dim=1)
    selected_idx = torch.where(bypass_mode & non_center_exists, non_center_idx, base_idx)
    return torch.where(cross_mode & nonzero_cross_exists, nonzero_cross_idx, selected_idx)


def _gather_breakdown(breakdown: dict[str, Tensor], best_idx: Tensor, candidate_count: int) -> dict[str, Tensor]:
    j_td = _gather_candidate(breakdown["J_td"].unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    j_swing = _gather_candidate(breakdown["J_swing"].unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    j_ik = _gather_candidate(breakdown["J_ik"].unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    j_base = _gather_candidate(breakdown["J_base"].unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    j_vel = _gather_candidate(breakdown["J_vel"].unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    j_semantic_touchdown = _gather_candidate(
        breakdown["J_semantic_touchdown"].unsqueeze(-1), best_idx, candidate_count
    ).squeeze(-1)
    j_semantic_swing = _gather_candidate(
        breakdown["J_semantic_swing"].unsqueeze(-1), best_idx, candidate_count
    ).squeeze(-1)
    j_semantic_body = _gather_candidate(
        breakdown["J_semantic_body"].unsqueeze(-1), best_idx, candidate_count
    ).squeeze(-1)
    j_route = _gather_candidate(breakdown["J_route"].unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    j_pair_consistency = _gather_candidate(
        breakdown["J_pair_consistency"].unsqueeze(-1), best_idx, candidate_count
    ).squeeze(-1)
    j_body_posture = _gather_candidate(
        breakdown["J_body_posture"].unsqueeze(-1), best_idx, candidate_count
    ).squeeze(-1)
    j_path_clearance = _gather_candidate(
        breakdown["J_path_clearance"].unsqueeze(-1), best_idx, candidate_count
    ).squeeze(-1)
    j_collision_body = _gather_candidate(
        breakdown["J_collision_body"].unsqueeze(-1), best_idx, candidate_count
    ).squeeze(-1)
    j_collision_leg = _gather_candidate(
        breakdown["J_collision_leg"].unsqueeze(-1), best_idx, candidate_count
    ).squeeze(-1)
    j_barrier = _gather_candidate(breakdown["J_barrier"].unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    return {
        "J_td": j_td,
        "J_swing": j_swing,
        "J_ik": j_ik,
        "J_base": j_base,
        "J_vel": j_vel,
        "J_semantic_touchdown": j_semantic_touchdown,
        "J_semantic_swing": j_semantic_swing,
        "J_semantic_body": j_semantic_body,
        "J_route": j_route,
        "J_pair_consistency": j_pair_consistency,
        "J_body_posture": j_body_posture,
        "J_path_clearance": j_path_clearance,
        "J_collision_body": j_collision_body,
        "J_collision_leg": j_collision_leg,
        "J_barrier": j_barrier,
    }


def _repeat_terrain(terrain: TogetherPlannerTerrain, candidate_count: int) -> TogetherPlannerTerrain:
    if candidate_count == 1:
        return terrain
    semantic_maps = getattr(terrain, "semantic_maps", None)
    kwargs = {"heightmaps": _repeat_rows(terrain.heightmaps, candidate_count)}
    fields = getattr(type(terrain), "__dataclass_fields__", {})
    if semantic_maps is not None and "semantic_maps" in fields:
        kwargs["semantic_maps"] = _repeat_rows(semantic_maps, candidate_count)
    return replace(terrain, **kwargs)


def plan_segment(
    terrain: TogetherPlannerTerrain,
    state: TogetherRobotState,
    command_batch: Tensor,
    cfg: TogetherPlannerConfig | None = None,
) -> TogetherPlannerResult:
    planner_cfg = cfg or TogetherPlannerConfig()
    validate_config(planner_cfg)
    if not isinstance(terrain, TogetherPlannerTerrain):
        raise TypeError("terrain must be a TogetherPlannerTerrain")
    device = terrain.device
    dtype = torch.float32 if terrain.dtype not in (torch.float32, torch.float64) else terrain.dtype
    root_pos, root_rpy, foot_pos, _ = _coerce_state(state, device=device, dtype=dtype)
    batch_size = root_pos.shape[0]
    if terrain.batch_size != batch_size:
        raise ValueError("terrain and state must share batch dimension")
    command = _coerce_command(command_batch, device=device, dtype=dtype, batch_size=batch_size)
    candidate_count = int(planner_cfg.candidate_count)
    mode_geometry = classify_mode_and_geometry(terrain, root_pos, root_rpy, foot_pos, command, planner_cfg)
    candidate_betas, candidate_route_ids, candidate_route_signs = _t116_candidate_tables(
        mode_geometry.mode_code,
        device=device,
        dtype=dtype,
        cfg=planner_cfg,
    )
    route_offsets = candidate_route_signs * float(planner_cfg.semantic_lateral_offset_m)
    flat_route_offsets = route_offsets.reshape(batch_size * candidate_count)
    candidate_command = _candidate_commands(command, root_rpy, candidate_betas, candidate_route_signs, planner_cfg)
    repeated_root_pos = _repeat_rows(root_pos, candidate_count)
    repeated_root_rpy = _repeat_rows(root_rpy, candidate_count)
    repeated_foot_pos = _repeat_rows(foot_pos, candidate_count)
    repeated_command = candidate_command.reshape(batch_size * candidate_count, 3)
    terrain_repeated = _repeat_terrain(terrain, candidate_count)
    schedule = build_fixed_schedule(
        batch_size * candidate_count,
        int(planner_cfg.horizon_steps),
        float(planner_cfg.dt),
        device,
        dtype,
        repeated_command,
        planner_cfg,
    )
    cross_schedule = build_cross_small_schedule(
        int(planner_cfg.horizon_steps),
        float(planner_cfg.dt),
        device,
        dtype,
        _repeat_rows(command, candidate_count),
        repeated_root_pos,
        repeated_root_rpy,
        repeated_foot_pos,
        planner_cfg,
    )
    repeated_mode_code = _repeat_rows(mode_geometry.mode_code[:, None], candidate_count).reshape(batch_size * candidate_count)
    cross_mode = repeated_mode_code == T116_MODE_CROSS_SMALL
    schedule = TogetherContactSchedule(
        contact_state=torch.where(cross_mode[:, None, None], cross_schedule.contact_state, schedule.contact_state),
        touchdown_mask=torch.where(cross_mode[:, None, None], cross_schedule.touchdown_mask, schedule.touchdown_mask),
        touchdown_frames=torch.where(cross_mode[:, None, None], cross_schedule.touchdown_frames, schedule.touchdown_frames),
        horizon_steps=schedule.horizon_steps,
        dt=schedule.dt,
        event_cap=schedule.event_cap,
    )
    rollout = expand_segment(
        terrain_repeated,
        repeated_root_pos,
        repeated_root_rpy,
        repeated_foot_pos,
        repeated_command,
        schedule,
        planner_cfg,
        flat_route_offsets,
        repeated_mode_code,
        _repeat_rows(mode_geometry.small_back_s[:, None], candidate_count).reshape(batch_size * candidate_count),
        _repeat_rows(mode_geometry.small_top_z[:, None], candidate_count).reshape(batch_size * candidate_count),
        _repeat_rows(mode_geometry.small_center_xy, candidate_count),
        _repeat_rows(command, candidate_count),
    )
    kinematics = evaluate_kinematics(rollout.root_pos, rollout.root_rpy, rollout.foot_pos)
    costs = compute_costs(terrain_repeated, rollout, kinematics, repeated_command, planner_cfg)
    candidate_costs = costs.total.reshape(batch_size, candidate_count)
    candidate_feasible = costs.feasible.reshape(batch_size, candidate_count)
    candidate_safe_fallback = costs.safe_fallback.reshape(batch_size, candidate_count)
    candidate_hard_reason_mask = costs.hard_reason_mask.reshape(batch_size, candidate_count, HARD_REASON_COUNT)
    candidate_hard_rank_cost = costs.hard_rank_cost.reshape(batch_size, candidate_count)
    command_xy = command[:, :2]
    command_norm = torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True)
    retention_modes = (mode_geometry.mode_code == 0) | (mode_geometry.mode_code == 1)
    moving_command = command_norm > float(planner_cfg.idle_command_eps)
    command_retention_cost = (
        command_norm
        * (1.0 - candidate_betas).pow(2)
        * float(planner_cfg.command_retention_weight)
        * (retention_modes & moving_command.squeeze(-1)).to(dtype=dtype)[:, None]
    )
    candidate_costs = candidate_costs + command_retention_cost
    fallback_body_dir = torch.zeros_like(command_xy)
    fallback_body_dir[:, 0] = 1.0
    command_body_dir = torch.where(command_norm > float(planner_cfg.idle_command_eps), command_xy / command_norm.clamp_min(1e-6), fallback_body_dir)
    yaw0 = root_rpy[:, 2]
    command_world_dir = torch.stack(
        (
            torch.cos(yaw0) * command_body_dir[:, 0] - torch.sin(yaw0) * command_body_dir[:, 1],
            torch.sin(yaw0) * command_body_dir[:, 0] + torch.cos(yaw0) * command_body_dir[:, 1],
        ),
        dim=-1,
    )
    candidate_root = rollout.root_pos.reshape(batch_size, candidate_count, int(planner_cfg.horizon_steps), 3)
    candidate_root_delta = candidate_root[:, :, -1, :2] - root_pos[:, None, :2]
    candidate_root_progress = (candidate_root_delta * command_world_dir[:, None, :]).sum(dim=-1)
    candidate_command_progress = (candidate_command[..., :2] * command_body_dir[:, None, :]).sum(dim=-1)
    candidate_direction_violation = (candidate_root_progress < -1.0e-5) | (candidate_command_progress < -1.0e-5)
    direction_barrier = candidate_direction_violation.to(dtype=dtype) * 1.0e8
    candidate_costs = candidate_costs + direction_barrier
    candidate_feasible = candidate_feasible & torch.logical_not(candidate_direction_violation)
    candidate_safe_fallback = candidate_safe_fallback & torch.logical_not(candidate_direction_violation)
    direction_reason = candidate_direction_violation.unsqueeze(-1)
    reason_index = torch.arange(HARD_REASON_COUNT, device=device).view(1, 1, HARD_REASON_COUNT)
    candidate_hard_reason_mask = candidate_hard_reason_mask | (
        direction_reason & (reason_index == int(HARD_REASON_DIRECTION_VIOLATION))
    )
    candidate_hard_rank_cost = candidate_hard_rank_cost + candidate_direction_violation.to(dtype=dtype) * 1200.0
    candidate_hard_rank_cost = candidate_hard_rank_cost + 1.0e-6 * candidate_costs.detach()
    best_idx = _select_t116_candidate(
        candidate_costs,
        candidate_feasible,
        candidate_safe_fallback,
        mode_geometry.mode_code,
        candidate_route_ids,
        candidate_betas,
        candidate_hard_rank_cost,
    )
    best_total = _gather_candidate(costs.total.unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    best_feasible = candidate_feasible.gather(1, best_idx.view(batch_size, 1)).squeeze(1)
    best_safe_fallback = candidate_safe_fallback.gather(1, best_idx.view(batch_size, 1)).squeeze(1)
    best_breakdown = _gather_breakdown(costs.breakdown, best_idx, candidate_count)
    best_direction_barrier = direction_barrier.gather(1, best_idx.view(batch_size, 1)).squeeze(1)
    best_retention_cost = command_retention_cost.gather(1, best_idx.view(batch_size, 1)).squeeze(1)
    best_total = best_total + best_direction_barrier + best_retention_cost
    best_breakdown["J_barrier"] = best_breakdown["J_barrier"] + best_direction_barrier
    selected_beta = candidate_betas.gather(1, best_idx.view(batch_size, 1)).squeeze(1)
    selected_route_id = candidate_route_ids.gather(1, best_idx.view(batch_size, 1)).squeeze(1)
    selected_route_sign = candidate_route_signs.gather(1, best_idx.view(batch_size, 1)).squeeze(1)
    selected_hard_reason_mask = candidate_hard_reason_mask.gather(
        1,
        best_idx.view(batch_size, 1, 1).expand(batch_size, 1, HARD_REASON_COUNT),
    ).squeeze(1)
    selected_hard_rank_cost = candidate_hard_rank_cost.gather(1, best_idx.view(batch_size, 1)).squeeze(1)
    selected_candidate_command = _gather_candidate(repeated_command, best_idx, candidate_count)
    selected_delta = _gather_candidate(rollout.root_pos, best_idx, candidate_count)[:, -1, :2] - root_pos[:, :2]
    selected_progress = (selected_delta * command_world_dir).sum(dim=-1)
    selected_command_progress = (selected_candidate_command[:, :2] * command_body_dir).sum(dim=-1)
    command_direction_violation = (selected_progress < -1.0e-5) | (selected_command_progress < -1.0e-5)
    zero_long = torch.zeros((batch_size,), device=device, dtype=torch.long)
    zero_bool = torch.zeros((batch_size,), device=device, dtype=torch.bool)
    zero_float = torch.zeros((batch_size,), device=device, dtype=dtype)
    zero_pair = torch.zeros((batch_size, 2), device=device, dtype=dtype)
    no_small_s = torch.full((batch_size,), float("inf"), device=device, dtype=dtype)
    selected_per_leg_touchdown_on_small = _gather_candidate(rollout.per_leg_touchdown_on_small_count, best_idx, candidate_count)
    selected_per_leg_foot_small_collision = _gather_candidate(rollout.per_leg_foot_small_collision_count, best_idx, candidate_count)
    selected_per_leg_min_clearance = _gather_candidate(rollout.per_leg_min_clearance_to_small, best_idx, candidate_count)
    selected_touchdown_beyond = _gather_candidate(rollout.per_leg_touchdown_beyond_small_back_edge, best_idx, candidate_count)
    selected_schedule_ok = _gather_candidate(
        rollout.command_leading_before_trailing_schedule_ok.unsqueeze(-1),
        best_idx,
        candidate_count,
    ).squeeze(-1)
    selected_front_ground_gap = _gather_candidate(rollout.front_touchdown_ground_gap, best_idx, candidate_count)
    selected_rear_ground_gap = _gather_candidate(rollout.rear_touchdown_ground_gap, best_idx, candidate_count)
    touchdown_tol = float(planner_cfg.touchdown_ground_gap_tolerance_m)
    selected_grounded = torch.cat((selected_front_ground_gap, selected_rear_ground_gap), dim=-1).abs() <= touchdown_tol
    selected_base_penetration = _gather_candidate(
        costs.base_small_penetration_count.unsqueeze(-1),
        best_idx,
        candidate_count,
    ).squeeze(-1)
    selected_base_crosses = _gather_candidate(
        costs.base_path_crosses_small_flag.unsqueeze(-1),
        best_idx,
        candidate_count,
    ).squeeze(-1).to(dtype=torch.bool)
    selected_body_clearance = _gather_candidate(costs.body_min_clearance.unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    selected_leg_clearance = _gather_candidate(costs.leg_min_clearance.unsqueeze(-1), best_idx, candidate_count).squeeze(-1)
    selected_root = _gather_candidate(rollout.root_pos, best_idx, candidate_count)
    final_root_s = ((selected_root[:, -1, :2] - root_pos[:, :2]) * command_world_dir).sum(dim=-1)
    final_body_rear_s = final_root_s - float(planner_cfg.body_footprint_forward_m)
    selected_mode_cross = mode_geometry.mode_code == T116_MODE_CROSS_SMALL
    selected_touchdown_on_small_count = selected_per_leg_touchdown_on_small.sum(dim=-1).to(dtype=dtype)
    selected_foot_path_small_count = selected_per_leg_foot_small_collision.sum(dim=-1).to(dtype=dtype)
    cross_small_success = (
        selected_mode_cross
        & selected_touchdown_beyond.all(dim=-1)
        & selected_grounded.all(dim=-1)
        & (selected_per_leg_touchdown_on_small.sum(dim=-1) == 0)
        & (selected_per_leg_foot_small_collision.sum(dim=-1) == 0)
        & (selected_base_penetration == 0)
        & (selected_body_clearance >= -float(planner_cfg.body_collision_hard_penetration_m))
        & (selected_leg_clearance >= -float(planner_cfg.leg_collision_hard_penetration_m))
        & (final_root_s > mode_geometry.small_back_s + float(planner_cfg.support_search_radius))
        & (final_body_rear_s > mode_geometry.small_back_s + float(planner_cfg.small_body_clearance))
        & selected_schedule_ok
    )
    selected_bypass_mode = mode_geometry.mode_code == T116_MODE_BYPASS_OBSTACLE
    selected_route_offset = selected_route_sign * float(planner_cfg.semantic_lateral_offset_m)
    ok_status = torch.full((batch_size,), int(TogetherPlannerStatus.OK), device=device, dtype=torch.int64)
    infeasible_status = torch.full((batch_size,), int(TogetherPlannerStatus.ALL_INFEASIBLE), device=device, dtype=torch.int64)
    status = torch.where(best_feasible, ok_status, infeasible_status)
    result = TogetherPlannerResult(
        root_pos=_gather_candidate(rollout.root_pos, best_idx, candidate_count),
        root_rpy=_gather_candidate(rollout.root_rpy, best_idx, candidate_count),
        foot_pos=_gather_candidate(rollout.foot_pos, best_idx, candidate_count),
        joint_angles=_gather_candidate(kinematics.joint_angles, best_idx, candidate_count),
        contact_state=_gather_candidate(rollout.contact_state, best_idx, candidate_count),
        touchdown_seq=_gather_candidate(rollout.touchdown_seq, best_idx, candidate_count),
        touchdown_mask=_gather_candidate(rollout.touchdown_mask, best_idx, candidate_count),
        cost_total=best_total,
        cost_breakdown=best_breakdown,
        status=status,
        feasible=best_feasible,
        safe_fallback=best_safe_fallback,
        joint_limit_violation=_gather_candidate(kinematics.joint_limit_violation, best_idx, candidate_count),
        workspace_margin=_gather_candidate(kinematics.workspace_margin, best_idx, candidate_count),
        support_xy=_gather_candidate(rollout.support_xy, best_idx, candidate_count),
        support_height=_gather_candidate(rollout.support_height, best_idx, candidate_count),
        support_slope=_gather_candidate(rollout.support_slope, best_idx, candidate_count),
        state_mode=mode_geometry.mode_code,
        small_strategy_outcome=mode_geometry.mode_code,
        selected_route_offset=selected_route_offset,
        semantic_candidate_costs=candidate_costs,
        body_min_clearance=selected_body_clearance,
        leg_min_clearance=selected_leg_clearance,
        mode=mode_geometry.mode_code,
        selected_beta=selected_beta,
        selected_route=selected_route_id,
        direction_id=_command_direction_id(command, planner_cfg),
        small_front_s=torch.where(mode_geometry.small_present_mask, mode_geometry.small_front_s, no_small_s),
        small_back_s=torch.where(mode_geometry.small_present_mask, mode_geometry.small_back_s, no_small_s),
        small_top_z=torch.where(mode_geometry.small_present_mask, mode_geometry.small_top_z, zero_float),
        command_direction_violation=command_direction_violation,
        cross_small_success=cross_small_success,
        front_touchdown_ground_gap=selected_front_ground_gap,
        rear_touchdown_ground_gap=selected_rear_ground_gap,
        touchdown_on_small_count=selected_touchdown_on_small_count,
        front_foot_small_collision_count=selected_per_leg_foot_small_collision[:, :2].sum(dim=-1).to(dtype=dtype),
        rear_foot_small_collision_count=selected_per_leg_foot_small_collision[:, 2:].sum(dim=-1).to(dtype=dtype),
        front_foot_min_clearance_to_small=selected_per_leg_min_clearance[:, :2].amin(dim=-1),
        rear_foot_min_clearance_to_small=selected_per_leg_min_clearance[:, 2:].amin(dim=-1),
        base_small_penetration_count=selected_base_penetration,
        base_min_clearance_to_small=_gather_candidate(
            costs.base_min_clearance_to_small.unsqueeze(-1), best_idx, candidate_count
        ).squeeze(-1),
        base_path_crosses_small_flag=selected_base_crosses,
        per_leg_touchdown_on_small_count=selected_per_leg_touchdown_on_small.to(dtype=dtype),
        per_leg_foot_small_collision_count=selected_per_leg_foot_small_collision.to(dtype=dtype),
        per_leg_min_clearance_to_small=selected_per_leg_min_clearance,
        per_leg_touchdown_beyond_small_back_edge=selected_touchdown_beyond,
        touchdown_ground_gap_by_leg=_gather_candidate(rollout.touchdown_ground_gap_by_leg, best_idx, candidate_count),
        touchdown_semantic_by_leg=_gather_candidate(rollout.touchdown_semantic_by_leg, best_idx, candidate_count),
        touchdown_frame_by_leg=_gather_candidate(rollout.touchdown_frame_by_leg, best_idx, candidate_count),
        command_leading_before_trailing_schedule_ok=selected_schedule_ok,
        candidate_hard_reason_mask=candidate_hard_reason_mask,
        selected_hard_reason_mask=selected_hard_reason_mask,
        candidate_hard_rank_cost=candidate_hard_rank_cost,
        selected_hard_rank_cost=selected_hard_rank_cost,
        selected_candidate_index=best_idx,
    )
    return result


__all__ = ["plan_segment", "build_together_terrain_from_scanner"]
