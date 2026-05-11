"""Deterministic vectorized rollout parameterization for the together core."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import TogetherPlannerConfig
from .kinematics import evaluate_kinematics
from .schedule import hold_command_mask
from .terrain import TogetherPlannerTerrain
from .types import HIP_OFFSETS_ARRAY, TogetherContactSchedule


@dataclass(frozen=True)
class TogetherRollout:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    touchdown_seq: Tensor
    touchdown_mask: Tensor
    contact_state: Tensor
    support_xy: Tensor
    support_height: Tensor
    support_slope: Tensor
    time_s: Tensor
    route_offset_m: Tensor
    mode_code: Tensor
    candidate_anchor_references: Tensor
    candidate_touchdown_targets: Tensor
    candidate_path_progress: Tensor
    candidate_pair_summary: Tensor
    candidate_posture_summary: Tensor
    candidate_action_segment_diagnostics_present: Tensor
    touchdown_small_margin: Tensor
    front_touchdown_ground_gap: Tensor
    rear_touchdown_ground_gap: Tensor
    touchdown_on_small_count: Tensor
    front_foot_small_collision_count: Tensor
    rear_foot_small_collision_count: Tensor
    front_foot_min_clearance_to_small: Tensor
    rear_foot_min_clearance_to_small: Tensor
    base_small_penetration_count: Tensor
    base_min_clearance_to_small: Tensor
    base_path_crosses_small_flag: Tensor
    front_pair_consistency: Tensor
    rear_pair_follow_consistency: Tensor
    body_posture_score: Tensor
    anchor_to_touchdown_foot_clearance: Tensor
    anchor_to_touchdown_leg_clearance: Tensor
    candidate_path_collision_flag: Tensor
    per_leg_touchdown_on_small_count: Tensor
    per_leg_foot_small_collision_count: Tensor
    per_leg_min_clearance_to_small: Tensor
    per_leg_touchdown_beyond_small_back_edge: Tensor
    touchdown_ground_gap_by_leg: Tensor
    touchdown_semantic_by_leg: Tensor
    touchdown_frame_by_leg: Tensor
    command_leading_before_trailing_schedule_ok: Tensor


@dataclass(frozen=True)
class T116ModeGeometry:
    mode_code: Tensor
    small_geometry: Tensor
    gate_masks: Tensor
    small_front_s: Tensor
    small_back_s: Tensor
    small_top_z: Tensor
    small_center_xy: Tensor
    root_to_front_s: Tensor
    root_to_back_s: Tensor
    approach_window_mask: Tensor
    cross_window_mask: Tensor
    too_high_small_mask: Tensor
    all_clear_mask: Tensor
    foot_anchor_s: Tensor
    foot_anchor_l: Tensor
    small_present_mask: Tensor
    large_present_mask: Tensor


T116_MODE_CRUISE = 0
T116_MODE_APPROACH_SMALL = 1
T116_MODE_CROSS_SMALL = 2
T116_MODE_BYPASS_OBSTACLE = 3


def _semantic_maps_present(terrain: TogetherPlannerTerrain) -> bool:
    return getattr(terrain, "semantic_maps", None) is not None


def _semantic_at(terrain: TogetherPlannerTerrain, points_xy: Tensor) -> Tensor:
    heights = terrain.height_at(points_xy)
    semantic_fn = getattr(terrain, "semantic_at", None)
    if semantic_fn is None or not _semantic_maps_present(terrain):
        return torch.zeros_like(heights, dtype=torch.long)
    return torch.as_tensor(semantic_fn(points_xy), device=heights.device, dtype=torch.long)


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
    height_fn = getattr(terrain, "terrain_reference_height_at", None)
    if height_fn is not None and _semantic_maps_present(terrain):
        terrain_ref = torch.as_tensor(height_fn(points_xy), device=terrain.device, dtype=terrain.dtype)
    else:
        terrain_ref = terrain.height_at(points_xy)
    return _obstacle_height_at(terrain, points_xy, semantic_id) - terrain_ref


def _terrain_reference_height_at(terrain: TogetherPlannerTerrain, points_xy: Tensor) -> Tensor:
    reference_fn = getattr(terrain, "terrain_reference_height_at", None)
    if reference_fn is None or not _semantic_maps_present(terrain):
        return terrain.height_at(points_xy)
    return torch.as_tensor(reference_fn(points_xy), device=terrain.device, dtype=terrain.dtype)


def _terrain_fixed_grid_xy(terrain: TogetherPlannerTerrain) -> Tensor:
    heightmaps = terrain.heightmaps
    batch_size = int(heightmaps.shape[0])
    height = int(heightmaps.shape[-2])
    width = int(heightmaps.shape[-1])
    xs = torch.linspace(
        terrain.world_x_range[0],
        terrain.world_x_range[1],
        width,
        device=terrain.device,
        dtype=terrain.dtype,
    )
    ys = torch.linspace(
        terrain.world_y_range[0],
        terrain.world_y_range[1],
        height,
        device=terrain.device,
        dtype=terrain.dtype,
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1).view(1, height * width, 2).expand(batch_size, -1, -1)


def classify_mode_and_geometry(
    terrain: TogetherPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    command_batch: Tensor,
    cfg: TogetherPlannerConfig,
) -> T116ModeGeometry:
    root_xy = root_pos[:, :2]
    yaw = root_rpy[:, 2]
    body_v = command_batch[:, :2].to(device=root_pos.device, dtype=root_pos.dtype)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    world_v = torch.stack(
        (
            cos_yaw * body_v[:, 0] - sin_yaw * body_v[:, 1],
            sin_yaw * body_v[:, 0] + cos_yaw * body_v[:, 1],
        ),
        dim=-1,
    )
    fallback = torch.stack((cos_yaw, sin_yaw), dim=-1)
    world_v_norm = torch.linalg.vector_norm(world_v, dim=-1, keepdim=True)
    moving = world_v_norm > float(cfg.idle_command_eps)
    d = torch.where(moving, world_v / world_v_norm.clamp_min(1e-6), fallback)
    n = torch.stack((-d[:, 1], d[:, 0]), dim=-1)

    grid_xy = _terrain_fixed_grid_xy(terrain)
    grid_z = terrain.heightmaps[:, 0].reshape(terrain.batch_size, -1).to(device=root_pos.device, dtype=root_pos.dtype)
    semantic_maps = getattr(terrain, "semantic_maps", None)
    if semantic_maps is None:
        semantic_flat = torch.zeros((terrain.batch_size, grid_xy.shape[1]), device=root_pos.device, dtype=torch.long)
    else:
        semantic_flat = semantic_maps[:, 0].reshape(terrain.batch_size, -1).to(device=root_pos.device, dtype=torch.long)

    delta = grid_xy - root_xy[:, None, :]
    s_values = (delta * d[:, None, :]).sum(dim=-1)
    l_values = (delta * n[:, None, :]).sum(dim=-1)
    s_min = -float(cfg.body_footprint_forward_m)
    s_max = float(cfg.semantic_reference_radius) + 2.0 * float(cfg.body_footprint_forward_m) + 2.0 * float(cfg.support_search_radius)
    corridor_width = float(cfg.body_footprint_lateral_m) + float(cfg.support_search_radius)
    corridor_mask = (s_values >= s_min) & (s_values <= s_max) & (torch.abs(l_values) <= corridor_width)
    small_mask = semantic_flat == int(cfg.semantic_small_id)
    large_mask = semantic_flat == int(cfg.semantic_large_id)
    raw_small_target_mask = small_mask & corridor_mask
    large_present_mask = (large_mask & corridor_mask).any(dim=-1)
    small_present_mask = raw_small_target_mask.any(dim=-1)
    inf_s = torch.full_like(s_values, float("inf"))
    nearest_small_front_s = torch.where(raw_small_target_mask, s_values, inf_s).amin(dim=-1)
    component_window_s = 2.0 * float(cfg.semantic_reference_radius)
    target_mask = raw_small_target_mask & (s_values <= nearest_small_front_s[:, None] + component_window_s)

    neg_inf_s = torch.full_like(s_values, float("-inf"))
    small_front_s = torch.where(target_mask, s_values, inf_s).amin(dim=-1)
    small_back_s = torch.where(target_mask, s_values, neg_inf_s).amax(dim=-1)
    small_top_z = torch.where(target_mask, grid_z, neg_inf_s).amax(dim=-1)
    small_front_s = torch.where(small_present_mask, small_front_s, torch.full_like(small_front_s, float("inf")))
    small_back_s = torch.where(small_present_mask, small_back_s, torch.full_like(small_back_s, float("-inf")))
    small_top_z = torch.where(small_present_mask, small_top_z, torch.zeros_like(small_top_z))
    target_weight = target_mask.to(dtype=root_pos.dtype)
    target_count = target_weight.sum(dim=-1).clamp_min(1.0)
    center_xy = (grid_xy * target_weight.unsqueeze(-1)).sum(dim=1) / target_count.unsqueeze(-1)
    center_xy = torch.where(small_present_mask[:, None], center_xy, root_xy)

    foot_delta = foot_pos[..., :2] - root_xy[:, None, :]
    foot_anchor_s = (foot_delta * d[:, None, :]).sum(dim=-1)
    foot_anchor_l = (foot_delta * n[:, None, :]).sum(dim=-1)
    body_rear_s = -torch.as_tensor(float(cfg.body_footprint_forward_m), device=root_pos.device, dtype=root_pos.dtype)
    all_clear_mask = small_present_mask & (body_rear_s > small_back_s + float(cfg.small_body_clearance)) & (
        foot_anchor_s > small_back_s[:, None] + float(cfg.small_foot_clearance)
    ).all(dim=-1)

    root_to_front_s = small_front_s
    root_to_back_s = small_back_s
    small_reference_xy = torch.where(small_present_mask[:, None], center_xy, root_xy)
    small_relative_height = _obstacle_relative_height_at(terrain, small_reference_xy[:, None, :], int(cfg.semantic_small_id))[:, 0]
    small_relative_height = torch.where(small_present_mask, small_relative_height, torch.zeros_like(small_relative_height))
    too_high_small_mask = small_present_mask & (small_relative_height > float(cfg.small_crossable_height_max))
    approach_window_mask = small_present_mask & ~too_high_small_mask & (root_to_front_s > 0.22)
    cross_window_mask = small_present_mask & ~too_high_small_mask & (root_to_front_s <= 0.22) & (root_to_back_s >= -0.05)
    foot_semantic = _semantic_at(terrain, foot_pos[..., :2])
    foot_on_small_mask = foot_semantic == int(cfg.semantic_small_id)
    cross_gate_mask = cross_window_mask & ~foot_on_small_mask.any(dim=-1)

    mode_code = torch.full((root_pos.shape[0],), T116_MODE_CRUISE, device=root_pos.device, dtype=torch.long)
    bypass_mask = large_present_mask | too_high_small_mask
    mode_code = torch.where(bypass_mask, torch.full_like(mode_code, T116_MODE_BYPASS_OBSTACLE), mode_code)
    mode_code = torch.where(approach_window_mask & ~bypass_mask, torch.full_like(mode_code, T116_MODE_APPROACH_SMALL), mode_code)
    mode_code = torch.where(cross_window_mask & ~bypass_mask, torch.full_like(mode_code, T116_MODE_CROSS_SMALL), mode_code)
    cruise_mask = (~small_present_mask & ~large_present_mask) | (small_present_mask & all_clear_mask & ~bypass_mask)
    mode_code = torch.where(cruise_mask, torch.full_like(mode_code, T116_MODE_CRUISE), mode_code)

    small_geometry = torch.cat(
        (
            small_front_s[:, None],
            small_back_s[:, None],
            small_top_z[:, None],
            center_xy,
            root_to_front_s[:, None],
            root_to_back_s[:, None],
        ),
        dim=-1,
    )
    gate_masks = torch.stack(
        (
            small_present_mask,
            large_present_mask,
            approach_window_mask,
            cross_window_mask,
            too_high_small_mask,
            all_clear_mask,
            cross_gate_mask,
            foot_on_small_mask.any(dim=-1),
        ),
        dim=-1,
    )
    return T116ModeGeometry(
        mode_code=mode_code,
        small_geometry=small_geometry,
        gate_masks=gate_masks,
        small_front_s=small_front_s,
        small_back_s=small_back_s,
        small_top_z=small_top_z,
        small_center_xy=center_xy,
        root_to_front_s=root_to_front_s,
        root_to_back_s=root_to_back_s,
        approach_window_mask=approach_window_mask,
        cross_window_mask=cross_window_mask,
        too_high_small_mask=too_high_small_mask,
        all_clear_mask=all_clear_mask,
        foot_anchor_s=foot_anchor_s,
        foot_anchor_l=foot_anchor_l,
        small_present_mask=small_present_mask,
        large_present_mask=large_present_mask,
    )


def build_time_grid(horizon_steps: int, dt: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.arange(horizon_steps, device=device, dtype=dtype) * float(dt)


def integrate_body_frame_translation(command_batch: Tensor, yaw_traj: Tensor, dt: float) -> Tensor:
    command = torch.as_tensor(command_batch, device=yaw_traj.device, dtype=yaw_traj.dtype)
    if command.ndim == 1:
        command = command.unsqueeze(0)
    vx = command[:, 0].view(command.shape[0], 1)
    vy = command[:, 1].view(command.shape[0], 1)
    cos_yaw = torch.cos(yaw_traj[:, :-1])
    sin_yaw = torch.sin(yaw_traj[:, :-1])
    dx = (cos_yaw * vx - sin_yaw * vy) * float(dt)
    dy = (sin_yaw * vx + cos_yaw * vy) * float(dt)
    dz = torch.zeros_like(dx)
    delta = torch.stack((dx, dy, dz), dim=-1)
    zero = torch.zeros_like(delta[:, :1, :])
    return torch.cat((zero, torch.cumsum(delta, dim=1)), dim=1)


def _seed_params_flat(command_batch: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    dtype = command_batch.dtype
    device = command_batch.device
    command_norm = torch.linalg.vector_norm(command_batch[:, :2], dim=-1, keepdim=True)
    scale = 1.0 + 0.5 * command_norm
    default_template = torch.tensor(
        (0.02, 0.01, -0.01, 0.02, 0.01, 0.01, 0.0, 0.01),
        device=device,
        dtype=dtype,
    ).view(1, 8) * scale
    zero_template = torch.tensor(
        (0.03, 0.0, 0.03, -0.01, 0.03, 0.03, 0.01, 0.03),
        device=device,
        dtype=dtype,
    ).view(1, 8)
    lateral_template = torch.tensor(
        (
            0.022499997168779373,
            0.009964285418391228,
            -0.010478571057319641,
            0.023157142102718353,
            0.003535714466124773,
            -0.011249998584389687,
            -0.015428571030497551,
            0.0022499999031424522,
        ),
        device=device,
        dtype=dtype,
    ).view(1, 8)
    hold = hold_command_mask(command_batch, cfg).to(device=device)
    lateral = command_batch[:, 1].abs() > command_batch[:, 0].abs()
    return torch.where(
        hold[:, None],
        zero_template.expand(command_batch.shape[0], -1),
        torch.where(lateral[:, None], lateral_template.expand(command_batch.shape[0], -1), default_template),
    )


def _root_trajectory(root_pos: Tensor, root_rpy: Tensor, command_batch: Tensor, cfg: TogetherPlannerConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device = root_pos.device
    dtype = root_pos.dtype
    time_s = build_time_grid(int(cfg.horizon_steps), float(cfg.dt), device=device, dtype=dtype)
    seed_params = _seed_params_flat(command_batch, cfg)
    phase = (time_s / max(float(cfg.horizon_s), 1e-6)).view(1, int(cfg.horizon_steps))
    z_curve = seed_params[:, 0:1] * (0.5 - 0.5 * torch.cos(torch.pi * phase))
    command_yaw = root_rpy[:, 2:3] + command_batch[:, 2:3] * time_s.view(1, -1)
    yaw = command_yaw + seed_params[:, 7:8] * phase
    translation = integrate_body_frame_translation(command_batch, command_yaw, float(cfg.dt))
    root_traj = root_pos[:, None, :].expand(-1, int(cfg.horizon_steps), -1) + translation
    root_traj[..., 0] = root_traj[..., 0] + seed_params[:, 5:6] * phase
    root_traj[..., 1] = root_traj[..., 1] + seed_params[:, 6:7] * phase
    root_traj[..., 2] = root_pos[:, 2:3] + z_curve
    rpy_traj = root_rpy[:, None, :].expand(-1, int(cfg.horizon_steps), -1).clone()
    rpy_traj[..., 0] = root_rpy[:, 0:1] + seed_params[:, 1:2] * phase
    rpy_traj[..., 1] = root_rpy[:, 1:2] + seed_params[:, 2:3] * phase
    rpy_traj[..., 2] = yaw
    return root_traj, rpy_traj, time_s, seed_params


def _apply_route_offsets(root_traj: Tensor, yaw_traj: Tensor, route_offset_m: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    phase = build_time_grid(int(cfg.horizon_steps), float(cfg.dt), device=root_traj.device, dtype=root_traj.dtype)
    phase = (phase / max(float(cfg.horizon_s), 1e-6)).clamp(0.0, 1.0)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    yaw = yaw_traj[:, :, 2]
    offset = route_offset_m.view(route_offset_m.shape[0], 1) * smooth.view(1, int(cfg.horizon_steps))
    shifted = root_traj.clone()
    shifted[..., 0] = shifted[..., 0] - torch.sin(yaw) * offset
    shifted[..., 1] = shifted[..., 1] + torch.cos(yaw) * offset
    return shifted


def _candidate_foothold_lateral_bias(route_offset_m: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    lateral_scale = float(cfg.semantic_foothold_lateral_scale)
    if lateral_scale <= 0.0:
        return torch.zeros_like(route_offset_m)
    nominal_offset = max(float(cfg.semantic_lateral_offset_m), 1e-6)
    normalized = route_offset_m / nominal_offset
    return normalized * float(cfg.support_search_radius) * lateral_scale


def _apply_foothold_lateral_bias(target_xy: Tensor, yaw: Tensor, route_offset_m: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    lateral_bias = _candidate_foothold_lateral_bias(route_offset_m, cfg)
    bias = lateral_bias.view(route_offset_m.shape[0], 1, 1)
    biased = target_xy.clone()
    biased[..., 0] = biased[..., 0] - torch.sin(yaw) * bias
    biased[..., 1] = biased[..., 1] + torch.cos(yaw) * bias
    return biased


def _support_query(terrain: TogetherPlannerTerrain, query_xy: Tensor, cfg: TogetherPlannerConfig) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = query_xy.shape[0]
    query_shape = tuple(query_xy.shape[1:-1])
    support_xy, support_height, support_slope = terrain.support_at(query_xy.reshape(batch_size, -1, 2), cfg)
    return (
        support_xy.reshape(batch_size, *query_shape, 2),
        support_height.reshape(batch_size, *query_shape),
        support_slope.reshape(batch_size, *query_shape),
    )


def _small_boundary_margin(
    terrain: TogetherPlannerTerrain,
    support_xy: Tensor,
    cfg: TogetherPlannerConfig,
) -> Tensor:
    if not _semantic_maps_present(terrain):
        return torch.full(
            support_xy.shape[:-1],
            float("inf"),
            device=support_xy.device,
            dtype=support_xy.dtype,
        )
    batch_size = support_xy.shape[0]
    query_shape = tuple(support_xy.shape[1:-1])
    radius = max(float(cfg.support_search_radius), float(cfg.support_search_step))
    step = max(float(cfg.support_search_step), 1e-6)
    search_count = max(1, int(math.ceil(radius / step)))
    axis = torch.arange(-search_count, search_count + 1, device=support_xy.device, dtype=support_xy.dtype) * step
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    finite_support = torch.isfinite(support_xy).all(dim=-1)
    safe_support_xy = torch.nan_to_num(support_xy, nan=0.0)
    candidates = safe_support_xy.reshape(batch_size, -1, 1, 2) + offsets.view(1, 1, -1, 2)
    flat_candidates = candidates.reshape(batch_size, -1, 2)
    semantic_ids = _semantic_at(terrain, flat_candidates).reshape(batch_size, -1, offsets.shape[0])
    small_mask = semantic_ids == int(cfg.semantic_small_id)
    offset_distance = torch.linalg.vector_norm(offsets, dim=-1).view(1, 1, -1)
    inf_fill = torch.full_like(offset_distance, float("inf"))
    min_distance = torch.where(small_mask, offset_distance, inf_fill).amin(dim=-1)
    max_margin = torch.full_like(min_distance, radius + step)
    margin = torch.where(torch.isfinite(min_distance), min_distance, max_margin)
    margin = torch.where(finite_support.reshape(batch_size, -1), margin, torch.zeros_like(margin))
    return margin.reshape(batch_size, *query_shape)


def _touchdown_policy_dirs(command_batch: Tensor, yaw: Tensor, cfg: TogetherPlannerConfig) -> tuple[Tensor, Tensor]:
    batch_size = command_batch.shape[0]
    expand_shape = (batch_size,) + (1,) * (yaw.ndim - 1)
    body_vx = command_batch[:, 0].view(expand_shape)
    body_vy = command_batch[:, 1].view(expand_shape)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    forward = torch.stack(
        (
            cos_yaw * body_vx - sin_yaw * body_vy,
            sin_yaw * body_vx + cos_yaw * body_vy,
        ),
        dim=-1,
    )
    fallback = torch.stack((cos_yaw, sin_yaw), dim=-1)
    norm = torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
    moving = norm > float(cfg.idle_command_eps)
    forward = torch.where(moving, forward / norm.clamp_min(1e-6), fallback)
    lateral = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)
    return forward, lateral


def _state_thresholds(cfg: TogetherPlannerConfig, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    ready_distance = torch.as_tensor(
        float(cfg.body_footprint_forward_m) + float(cfg.support_search_radius),
        device=device,
        dtype=dtype,
    )
    cross_distance = torch.amax(HIP_OFFSETS_ARRAY[:2, 0].to(device=device, dtype=dtype))
    return ready_distance, cross_distance


def _touchdown_ground_gap_summaries(touchdown_seq: Tensor, support_touchdown_height: Tensor) -> tuple[Tensor, Tensor]:
    touchdown_ground_gap = touchdown_seq[..., 2] - support_touchdown_height
    front_touchdown_ground_gap = touchdown_ground_gap[:, :2, 0]
    rear_touchdown_ground_gap = touchdown_ground_gap[:, 2:, 0]
    return front_touchdown_ground_gap, rear_touchdown_ground_gap


def _command_dirs_from_root(command_batch: Tensor, root_rpy: Tensor, cfg: TogetherPlannerConfig) -> tuple[Tensor, Tensor]:
    yaw = root_rpy[:, 2]
    body_v = command_batch[:, :2].to(device=root_rpy.device, dtype=root_rpy.dtype)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    world_v = torch.stack(
        (
            cos_yaw * body_v[:, 0] - sin_yaw * body_v[:, 1],
            sin_yaw * body_v[:, 0] + cos_yaw * body_v[:, 1],
        ),
        dim=-1,
    )
    fallback = torch.stack((cos_yaw, sin_yaw), dim=-1)
    norm = torch.linalg.vector_norm(world_v, dim=-1, keepdim=True)
    forward = torch.where(norm > float(cfg.idle_command_eps), world_v / norm.clamp_min(1e-6), fallback)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    return forward, lateral


def _cross_small_touchdown_targets(
    terrain: TogetherPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    command_batch: Tensor,
    small_back_s: Tensor,
    small_top_z: Tensor,
    small_center_xy: Tensor,
    schedule: TogetherContactSchedule,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = root_pos.shape[0]
    event_cap = int(cfg.event_cap)
    forward, lateral = _command_dirs_from_root(command_batch, root_rpy, cfg)
    root_xy = root_pos[:, :2]
    anchor_delta = foot_pos[..., :2] - root_xy[:, None, :]
    anchor_s = (anchor_delta * forward[:, None, :]).sum(dim=-1)
    anchor_l = (anchor_delta * lateral[:, None, :]).sum(dim=-1)
    margin = float(cfg.small_foot_clearance) + float(cfg.support_search_radius)
    final_root_s = (
        small_back_s
        + float(cfg.body_footprint_forward_m)
        + float(cfg.small_body_clearance)
        + float(cfg.support_search_radius)
    )
    target_s = torch.maximum(small_back_s[:, None] + margin, final_root_s[:, None] + anchor_s)
    nominal_xy = root_xy[:, None, :] + forward[:, None, :] * target_s[:, :, None] + lateral[:, None, :] * anchor_l[:, :, None]
    support_xy, support_height, _ = terrain.support_at(nominal_xy, cfg)
    touchdown_semantic = _semantic_at(terrain, support_xy)
    touchdown_ground_gap = torch.zeros((batch_size, 4), device=root_pos.device, dtype=root_pos.dtype)
    target = torch.cat((support_xy, support_height.unsqueeze(-1)), dim=-1)
    touchdown_seq = target[:, :, None, :].expand(-1, -1, event_cap, -1).clone()
    touchdown_seq = torch.where(
        schedule.touchdown_mask[:, :, :, None],
        touchdown_seq,
        foot_pos[:, :, None, :].expand(-1, -1, event_cap, -1),
    )
    touchdown_s = ((support_xy - root_xy[:, None, :]) * forward[:, None, :]).sum(dim=-1)
    beyond = touchdown_s > small_back_s[:, None] + float(cfg.small_foot_clearance)
    on_small = touchdown_semantic == int(cfg.semantic_small_id)
    apex_xy = small_center_xy[:, None, :].expand(-1, 4, -1)
    apex_z = (
        small_top_z[:, None]
        + float(cfg.small_foot_clearance)
        + float(cfg.swing_height_clearance_margin)
        + float(cfg.hip_height)
        + float(cfg.support_search_radius)
    ).expand(-1, 4)
    return touchdown_seq, support_xy, support_height, touchdown_semantic, touchdown_ground_gap, beyond, on_small, torch.cat(
        (apex_xy, apex_z.unsqueeze(-1)),
        dim=-1,
    )


def _cross_small_schedule_order_ok(
    command_batch: Tensor,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    touchdown_frame_by_leg: Tensor,
    cfg: TogetherPlannerConfig,
) -> Tensor:
    forward, _ = _command_dirs_from_root(command_batch, root_rpy, cfg)
    leg_s = ((foot_pos[..., :2] - root_pos[:, None, :2]) * forward[:, None, :]).sum(dim=-1)
    lead_order = torch.argsort(leg_s, dim=-1, descending=True)
    ordered_frames = touchdown_frame_by_leg.gather(1, lead_order)
    return (ordered_frames[:, 0] < ordered_frames[:, 2]) & (ordered_frames[:, 1] < ordered_frames[:, 3])


def _cross_small_per_leg_path_diagnostics(
    terrain: TogetherPlannerTerrain,
    anchor: Tensor,
    apex: Tensor,
    touchdown: Tensor,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor]:
    sample_count = max(int(cfg.swing_height_query_count), 3)
    axis = torch.linspace(0.0, 1.0, sample_count, device=anchor.device, dtype=anchor.dtype).view(1, 1, sample_count, 1)
    first = anchor[:, :, None, :] + axis * (apex[:, :, None, :] - anchor[:, :, None, :])
    second = apex[:, :, None, :] + axis * (touchdown[:, :, None, :] - apex[:, :, None, :])
    path = torch.cat((first, second), dim=2)
    small_id = int(cfg.semantic_small_id)
    path_xy = path[..., :2].reshape(anchor.shape[0], -1, 2)
    small_top = _obstacle_height_at(terrain, path_xy, small_id).reshape(anchor.shape[0], 4, 2 * sample_count)
    small_rel = _obstacle_relative_height_at(terrain, path_xy, small_id).reshape(anchor.shape[0], 4, 2 * sample_count)
    path_z = path[..., 2].reshape(anchor.shape[0], 4, 2 * sample_count)
    clearance = path_z - small_top
    on_small_path = small_rel > 0.0
    inf_fill = torch.full_like(clearance, 1.0e3)
    min_clearance = torch.where(on_small_path, clearance, inf_fill).amin(dim=-1)
    min_clearance = torch.where(on_small_path.any(dim=-1), min_clearance, torch.full_like(min_clearance, 1.0e3))
    collision_count = (on_small_path & (clearance < float(cfg.small_foot_clearance))).sum(dim=-1)
    return collision_count, min_clearance


def _candidate_action_segment_summaries(
    root_traj: Tensor,
    root_rpy: Tensor,
    stance: Tensor,
    touchdown_target: Tensor,
    swing_progress: Tensor,
    mode_code: Tensor,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    candidate_anchor_references = stance[:, 0]
    candidate_touchdown_targets = touchdown_target[:, 0]
    candidate_path_progress = swing_progress

    anchor_delta = candidate_touchdown_targets - candidate_anchor_references
    start_yaw = root_rpy[:, 0, 2]
    cos_yaw = torch.cos(start_yaw).view(root_traj.shape[0], 1)
    sin_yaw = torch.sin(start_yaw).view(root_traj.shape[0], 1)
    body_delta_x = cos_yaw * anchor_delta[..., 0] + sin_yaw * anchor_delta[..., 1]
    body_delta_y = -sin_yaw * anchor_delta[..., 0] + cos_yaw * anchor_delta[..., 1]
    delta_z = anchor_delta[..., 2]
    front_delta = anchor_delta[:, :2]
    rear_delta = anchor_delta[:, 2:]
    front_xy = torch.linalg.vector_norm(front_delta[..., :2], dim=-1).mean(dim=-1)
    rear_xy = torch.linalg.vector_norm(rear_delta[..., :2], dim=-1).mean(dim=-1)
    front_z = front_delta[..., 2].mean(dim=-1)
    rear_z = rear_delta[..., 2].mean(dim=-1)
    candidate_pair_summary = torch.stack((torch.stack((front_xy, front_z), dim=-1), torch.stack((rear_xy, rear_z), dim=-1)), dim=1)
    front_pair_consistency = (
        torch.abs(body_delta_x[:, 0] - body_delta_x[:, 1])
        + 0.5 * torch.abs(delta_z[:, 0] - delta_z[:, 1])
        + 0.25 * torch.abs(body_delta_y[:, 0] + body_delta_y[:, 1])
    )
    rear_pair_follow_consistency = (
        torch.abs(body_delta_x[:, 2] - body_delta_x[:, 3])
        + 0.5 * torch.abs(delta_z[:, 2] - delta_z[:, 3])
        + 0.25 * torch.abs(body_delta_y[:, 2] + body_delta_y[:, 3])
    )

    start_posture = torch.stack((root_rpy[:, 0, 0], root_rpy[:, 0, 1], root_traj[:, 0, 2]), dim=-1)
    end_posture = torch.stack((root_rpy[:, -1, 0], root_rpy[:, -1, 1], root_traj[:, -1, 2]), dim=-1)
    candidate_posture_summary = torch.stack((start_posture, end_posture), dim=1)
    target_posture_rpy = _support_plane_target_rpy(root_rpy[:, -1, :], candidate_touchdown_targets, cfg)
    target_root_z = candidate_touchdown_targets[..., 2].mean(dim=-1) + float(cfg.hip_height)
    body_posture_score = (
        torch.abs(target_posture_rpy[:, 0] - root_rpy[:, -1, 0])
        + torch.abs(target_posture_rpy[:, 1] - root_rpy[:, -1, 1])
        + 0.5 * torch.abs(target_root_z - end_posture[:, 2])
    )

    candidate_action_segment_diagnostics_present = torch.ones(
        (root_traj.shape[0],),
        device=root_traj.device,
        dtype=torch.bool,
    )
    return (
        candidate_anchor_references,
        candidate_touchdown_targets,
        candidate_path_progress,
        candidate_pair_summary,
        candidate_posture_summary,
        candidate_action_segment_diagnostics_present,
        front_pair_consistency,
        rear_pair_follow_consistency,
        body_posture_score,
    )


def _sample_candidate_segment_surface_clearance(
    start_xyz: Tensor,
    end_xyz: Tensor,
    terrain: TogetherPlannerTerrain,
    *,
    axis_samples: int,
    radius_m: float,
) -> Tensor:
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


def _candidate_path_clearance_summaries(
    terrain: TogetherPlannerTerrain,
    root_traj: Tensor,
    root_rpy: Tensor,
    foot_traj: Tensor,
    contact_state: Tensor,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    swing_mask = torch.logical_not(contact_state.to(dtype=torch.bool))
    has_swing = swing_mask.any(dim=(1, 2))
    inf_fill = torch.full(
        swing_mask.shape,
        float("inf"),
        device=foot_traj.device,
        dtype=foot_traj.dtype,
    )
    foot_surface = terrain.height_at(foot_traj[..., :2].reshape(foot_traj.shape[0], -1, 2)).reshape(
        foot_traj.shape[0],
        int(cfg.horizon_steps),
        4,
    )
    foot_clearance_per_frame = foot_traj[..., 2] - foot_surface
    foot_clearance = torch.where(swing_mask, foot_clearance_per_frame, inf_fill).amin(dim=(1, 2))
    foot_clearance = torch.where(
        has_swing,
        foot_clearance,
        torch.full_like(foot_clearance, float("inf")),
    )

    kinematics = evaluate_kinematics(root_traj, root_rpy, foot_traj)
    axis_samples = int(cfg.leg_collision_axis_sample_count)
    thigh_clearance = _sample_candidate_segment_surface_clearance(
        kinematics.hip_world,
        kinematics.knee_world,
        terrain,
        axis_samples=axis_samples,
        radius_m=float(cfg.leg_collision_radius_m),
    )
    calf_clearance = _sample_candidate_segment_surface_clearance(
        kinematics.knee_world,
        kinematics.foot_world,
        terrain,
        axis_samples=axis_samples,
        radius_m=float(cfg.leg_collision_radius_m),
    )
    segment_clearance = torch.cat((thigh_clearance, calf_clearance), dim=3)
    segment_mask = swing_mask.unsqueeze(-1).expand(-1, -1, -1, segment_clearance.shape[3])
    leg_inf_fill = torch.full_like(segment_clearance, float("inf"))
    leg_clearance = torch.where(segment_mask, segment_clearance, leg_inf_fill).amin(dim=(1, 2, 3))
    leg_clearance = torch.where(
        has_swing,
        leg_clearance,
        torch.full_like(leg_clearance, float("inf")),
    )

    candidate_path_collision_flag = (foot_clearance < 0.0) | (leg_clearance < 0.0)
    return foot_clearance, leg_clearance, candidate_path_collision_flag


def _small_surface_crossing_summaries(
    terrain: TogetherPlannerTerrain,
    touchdown_seq: Tensor,
    foot_traj: Tensor,
    contact_state: Tensor,
    root_traj: Tensor,
    root_rpy: Tensor,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = foot_traj.shape[0]
    dtype = foot_traj.dtype
    large_clearance = torch.full((batch_size,), 1.0e3, device=foot_traj.device, dtype=dtype)
    if not _semantic_maps_present(terrain):
        zero_count = torch.zeros((batch_size,), device=foot_traj.device, dtype=torch.long)
        false_flag = torch.zeros((batch_size,), device=foot_traj.device, dtype=torch.bool)
        return zero_count, zero_count, zero_count, large_clearance, large_clearance, zero_count, large_clearance, false_flag

    small_id = int(cfg.semantic_small_id)
    touchdown_ids = _semantic_at(terrain, touchdown_seq[..., :2].reshape(batch_size, -1, 2)).reshape(
        batch_size, 4, int(cfg.event_cap)
    )
    touchdown_on_small_count = (touchdown_ids == small_id).sum(dim=(1, 2))

    swing_mask = torch.logical_not(contact_state.to(dtype=torch.bool))
    foot_xy = foot_traj[..., :2].reshape(batch_size, -1, 2)
    foot_small_top = _obstacle_height_at(terrain, foot_xy, small_id).reshape(batch_size, int(cfg.horizon_steps), 4)
    foot_small_rel = _obstacle_relative_height_at(terrain, foot_xy, small_id).reshape(batch_size, int(cfg.horizon_steps), 4)
    foot_small_clearance = torch.nan_to_num(foot_traj[..., 2] - foot_small_top, nan=1.0e3, posinf=1.0e3, neginf=-1.0e3)
    front_swing_small = (foot_small_rel[:, :, :2] > 0.0) & swing_mask[:, :, :2]
    rear_swing_small = (foot_small_rel[:, :, 2:] > 0.0) & swing_mask[:, :, 2:]
    front_foot_small_collision_count = (front_swing_small & (foot_small_clearance[:, :, :2] < 0.0)).sum(dim=(1, 2))
    rear_foot_small_collision_count = (rear_swing_small & (foot_small_clearance[:, :, 2:] < 0.0)).sum(dim=(1, 2))
    inf_fill_front = torch.full_like(foot_small_clearance[:, :, :2], 1.0e3)
    inf_fill_rear = torch.full_like(foot_small_clearance[:, :, 2:], 1.0e3)
    front_foot_min_clearance_to_small = torch.where(
        front_swing_small,
        foot_small_clearance[:, :, :2],
        inf_fill_front,
    ).amin(dim=(1, 2))
    rear_foot_min_clearance_to_small = torch.where(
        rear_swing_small,
        foot_small_clearance[:, :, 2:],
        inf_fill_rear,
    ).amin(dim=(1, 2))

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
        device=root_traj.device,
        dtype=dtype,
    )[: int(cfg.body_footprint_sample_count)]
    yaw = root_rpy[..., 2:3]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    offset_x = offsets[:, 0].view(1, 1, -1)
    offset_y = offsets[:, 1].view(1, 1, -1)
    base_x = root_traj[..., 0:1] + cos_yaw * offset_x - sin_yaw * offset_y
    base_y = root_traj[..., 1:2] + sin_yaw * offset_x + cos_yaw * offset_y
    base_xy = torch.stack((base_x, base_y), dim=-1)
    base_small_top = _obstacle_height_at(terrain, base_xy.reshape(batch_size, -1, 2), small_id).reshape(
        batch_size,
        int(cfg.horizon_steps),
        int(cfg.body_footprint_sample_count),
    )
    base_small_rel = _obstacle_relative_height_at(terrain, base_xy.reshape(batch_size, -1, 2), small_id).reshape(
        batch_size,
        int(cfg.horizon_steps),
        int(cfg.body_footprint_sample_count),
    )
    underside = root_traj[..., 2:3] - float(cfg.body_underside_offset_m)
    base_small_clearance = torch.nan_to_num(underside - base_small_top, nan=1.0e3, posinf=1.0e3, neginf=-1.0e3)
    base_on_small = base_small_rel > 0.0
    base_small_penetration = base_on_small & (base_small_clearance < 0.0)
    base_small_penetration_count = base_small_penetration.sum(dim=(1, 2))
    inf_fill_base = torch.full_like(base_small_clearance, 1.0e3)
    base_min_clearance_to_small = torch.where(base_on_small, base_small_clearance, inf_fill_base).amin(dim=(1, 2))
    base_path_crosses_small_flag = base_on_small.any(dim=(1, 2))

    return (
        touchdown_on_small_count,
        front_foot_small_collision_count,
        rear_foot_small_collision_count,
        front_foot_min_clearance_to_small,
        rear_foot_min_clearance_to_small,
        base_small_penetration_count,
        base_min_clearance_to_small,
        base_path_crosses_small_flag,
    )


def _support_option_score(
    support_xy: Tensor,
    support_height: Tensor,
    support_slope: Tensor,
    nominal_xy: Tensor,
    root_xy: Tensor,
    forward_dir: Tensor,
    nominal_ref_height: Tensor,
    small_margin: Tensor,
    cfg: TogetherPlannerConfig,
) -> Tensor:
    valid = torch.isfinite(support_height)
    travel = torch.linalg.vector_norm(support_xy - nominal_xy, dim=-1)
    progress = ((support_xy - root_xy) * forward_dir).sum(dim=-1)
    height_delta = torch.abs(support_height - nominal_ref_height)
    slope_penalty = support_slope + 2.0 * torch.relu(support_slope - float(cfg.support_walkable_slope))
    boundary_penalty = torch.relu(float(cfg.touchdown_small_boundary_penalty_margin) - small_margin)
    score = (
        0.35 * travel
        + 3.0 * height_delta
        + 1.5 * slope_penalty
        - 0.75 * progress
        + float(cfg.touchdown_small_boundary_penalty_weight) * boundary_penalty
    )
    inf_fill = torch.full_like(score, float("inf"))
    valid = valid & (small_margin >= float(cfg.touchdown_small_boundary_invalidation_margin))
    return torch.where(valid, score, inf_fill)


def _gather_support_option(option_xy: Tensor, option_height: Tensor, option_slope: Tensor, best_idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    gather_xy = best_idx[..., None, None].expand(*best_idx.shape, 1, 2)
    gather_scalar = best_idx[..., None].expand(*best_idx.shape, 1)
    chosen_xy = option_xy.gather(3, gather_xy).squeeze(3)
    chosen_height = option_height.gather(3, gather_scalar).squeeze(3)
    chosen_slope = option_slope.gather(3, gather_scalar).squeeze(3)
    return chosen_xy, chosen_height, chosen_slope


def _semantic_touchdown_support(
    terrain: TogetherPlannerTerrain,
    nominal_xy: Tensor,
    root_xy: Tensor,
    yaw: Tensor,
    command_batch: Tensor,
    route_offset_m: Tensor,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    current_xy, current_height, current_slope = _support_query(terrain, nominal_xy, cfg)
    current_margin = _small_boundary_margin(terrain, current_xy, cfg)
    if not _semantic_maps_present(terrain):
        return current_xy, current_height, current_slope

    batch_size = nominal_xy.shape[0]
    event_cap = int(cfg.event_cap)
    forward_dir, lateral_dir = _touchdown_policy_dirs(command_batch, yaw, cfg)
    forward_shift = max(float(cfg.semantic_reference_radius) + float(cfg.support_search_radius), float(cfg.support_search_step))
    lateral_shift = max(0.5 * float(cfg.semantic_lateral_offset_m), forward_shift)
    corridor_axis = nominal_xy.new_tensor((-forward_shift, 0.0, forward_shift))
    corridor_points = nominal_xy[..., None, :] + forward_dir[..., None, :] * corridor_axis.view(1, 1, 1, 3, 1)
    corridor_ids = _semantic_at(terrain, corridor_points.reshape(batch_size, -1, 2)).reshape(batch_size, 4, event_cap, 3)
    small_present = (corridor_ids == int(cfg.semantic_small_id)).any(dim=-1)
    large_present = (corridor_ids == int(cfg.semantic_large_id)).any(dim=-1)

    front_anchor = nominal_xy - forward_dir * forward_shift
    beyond_anchor = nominal_xy + forward_dir * forward_shift
    route_sign = torch.sign(route_offset_m).view(batch_size, 1, 1, 1)
    bypass_anchor = nominal_xy + lateral_dir * route_sign * lateral_shift

    front_xy, front_height, front_slope = _support_query(terrain, front_anchor, cfg)
    beyond_xy, beyond_height, beyond_slope = _support_query(terrain, beyond_anchor, cfg)
    bypass_xy, bypass_height, bypass_slope = _support_query(terrain, bypass_anchor, cfg)
    front_margin = _small_boundary_margin(terrain, front_xy, cfg)
    beyond_margin = _small_boundary_margin(terrain, beyond_xy, cfg)
    bypass_margin = _small_boundary_margin(terrain, bypass_xy, cfg)
    nominal_ref_height = _terrain_reference_height_at(terrain, nominal_xy.reshape(batch_size, -1, 2)).reshape(batch_size, 4, event_cap)

    current_score = _support_option_score(
        current_xy, current_height, current_slope, nominal_xy, root_xy, forward_dir, nominal_ref_height, current_margin, cfg
    )
    front_score = _support_option_score(
        front_xy, front_height, front_slope, nominal_xy, root_xy, forward_dir, nominal_ref_height, front_margin, cfg
    )
    beyond_score = _support_option_score(
        beyond_xy, beyond_height, beyond_slope, nominal_xy, root_xy, forward_dir, nominal_ref_height, beyond_margin, cfg
    )
    small_option_score = torch.stack((current_score, front_score, beyond_score), dim=3)
    small_best_idx = torch.argmin(small_option_score, dim=3)
    small_option_xy = torch.stack((current_xy, front_xy, beyond_xy), dim=3)
    small_option_height = torch.stack((current_height, front_height, beyond_height), dim=3)
    small_option_slope = torch.stack((current_slope, front_slope, beyond_slope), dim=3)
    chosen_small_xy, chosen_small_height, chosen_small_slope = _gather_support_option(
        small_option_xy,
        small_option_height,
        small_option_slope,
        small_best_idx,
    )

    chosen_xy = torch.where(small_present.unsqueeze(-1) & (~large_present).unsqueeze(-1), chosen_small_xy, current_xy)
    chosen_height = torch.where(small_present & (~large_present), chosen_small_height, current_height)
    chosen_slope = torch.where(small_present & (~large_present), chosen_small_slope, current_slope)

    center_candidate = route_offset_m.abs().view(batch_size, 1, 1) <= 1e-6
    bypass_candidate = ~center_candidate
    bypass_valid = bypass_margin >= float(cfg.touchdown_small_boundary_invalidation_margin)
    chosen_xy = torch.where(
        large_present.unsqueeze(-1) & bypass_candidate.unsqueeze(-1) & bypass_valid.unsqueeze(-1), bypass_xy, chosen_xy
    )
    chosen_height = torch.where(large_present & bypass_candidate & bypass_valid, bypass_height, chosen_height)
    chosen_slope = torch.where(large_present & bypass_candidate & bypass_valid, bypass_slope, chosen_slope)

    invalid_xy = torch.full_like(chosen_xy, float("nan"))
    invalid_scalar = torch.full_like(chosen_height, float("nan"))
    chosen_xy = torch.where(large_present.unsqueeze(-1) & center_candidate.unsqueeze(-1), invalid_xy, chosen_xy)
    chosen_height = torch.where(large_present & center_candidate, invalid_scalar, chosen_height)
    chosen_slope = torch.where(large_present & center_candidate, invalid_scalar, chosen_slope)
    return chosen_xy, chosen_height, chosen_slope


def _apply_semantic_swing_clearance(
    terrain: TogetherPlannerTerrain,
    foot_traj: Tensor,
    contact_state: Tensor,
    root_traj: Tensor,
    cfg: TogetherPlannerConfig,
) -> Tensor:
    if not _semantic_maps_present(terrain):
        return foot_traj
    flat_xy = foot_traj[..., :2].reshape(foot_traj.shape[0], -1, 2)
    small_id = int(cfg.semantic_small_id)
    semantic_ids = _semantic_at(terrain, flat_xy).reshape(foot_traj.shape[0], int(cfg.horizon_steps), 4)
    small_rel = _obstacle_relative_height_at(terrain, flat_xy, small_id).reshape(foot_traj.shape[0], int(cfg.horizon_steps), 4)
    small_top = _obstacle_height_at(terrain, flat_xy, small_id).reshape(foot_traj.shape[0], int(cfg.horizon_steps), 4)
    root_lift = root_traj[..., 2] - root_traj[:, :1, 2]
    crossable = semantic_ids == small_id
    crossable = crossable & (small_rel <= float(cfg.small_crossable_height_max)) & (root_lift <= float(cfg.max_root_lift_for_small))[:, :, None]
    swing_mask = torch.logical_not(contact_state.to(dtype=torch.bool))
    target_z = small_top + float(cfg.small_foot_clearance)
    adjusted = foot_traj.clone()
    adjusted[..., 2] = torch.where(crossable & swing_mask, torch.maximum(adjusted[..., 2], target_z), adjusted[..., 2])
    return adjusted


def _apply_height_aware_swing_clearance(
    terrain: TogetherPlannerTerrain,
    stance: Tensor,
    touchdown_target: Tensor,
    progress: Tensor,
    contact_state: Tensor,
    foot_traj: Tensor,
    cfg: TogetherPlannerConfig,
) -> Tensor:
    sample_count = max(int(cfg.swing_height_query_count), 2)
    sample_axis = torch.linspace(
        0.0,
        1.0,
        sample_count,
        device=foot_traj.device,
        dtype=foot_traj.dtype,
    ).view(1, 1, 1, sample_count, 1)
    segment_points = stance[:, :, :, None, :2] + sample_axis * (touchdown_target[:, :, :, None, :2] - stance[:, :, :, None, :2])
    segment_height = terrain.height_at(segment_points.reshape(foot_traj.shape[0], -1, 2)).reshape(
        foot_traj.shape[0],
        int(cfg.horizon_steps),
        4,
        sample_count,
    )
    segment_max = segment_height.amax(dim=-1) + float(cfg.swing_height_clearance_margin)
    arc_peak = progress * (1.0 - progress)
    target_z = torch.maximum(segment_max, torch.maximum(stance[..., 2], touchdown_target[..., 2]))
    blended_z = foot_traj[..., 2] + arc_peak * torch.relu(target_z - foot_traj[..., 2]) * float(cfg.swing_parabola_multiplier)
    adjusted = foot_traj.clone()
    swing_mask = torch.logical_not(contact_state.to(dtype=torch.bool))
    adjusted[..., 2] = torch.where(swing_mask, torch.maximum(adjusted[..., 2], blended_z), adjusted[..., 2])
    return adjusted


def _touchdown_targets(
    terrain: TogetherPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    command_batch: Tensor,
    seed_params: Tensor,
    schedule: TogetherContactSchedule,
    cfg: TogetherPlannerConfig,
    route_offset_m: Tensor,
) -> Tensor:
    batch_size = root_pos.shape[0]
    event_frames = schedule.touchdown_frames.clamp(max=int(cfg.horizon_steps) - 1)
    gather_index = event_frames[:, None, :, :, None].expand(batch_size, 1, 4, int(cfg.event_cap), 3)
    root_event_pos = root_pos.gather(1, gather_index.reshape(batch_size, 4 * int(cfg.event_cap), 3)).reshape(
        batch_size,
        4,
        int(cfg.event_cap),
        3,
    )
    root_event_rpy = root_rpy.gather(1, gather_index.reshape(batch_size, 4 * int(cfg.event_cap), 3)).reshape(
        batch_size,
        4,
        int(cfg.event_cap),
        3,
    )
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=root_pos.device, dtype=root_pos.dtype).view(1, 4, 1, 3)
    yaw = root_event_rpy[..., 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    target_xy = torch.empty((batch_size, 4, int(cfg.event_cap), 2), device=root_pos.device, dtype=root_pos.dtype)
    target_xy[..., 0] = root_event_pos[..., 0] + cos_yaw * hip_offsets[..., 0] - sin_yaw * hip_offsets[..., 1]
    target_xy[..., 1] = root_event_pos[..., 1] + sin_yaw * hip_offsets[..., 0] + cos_yaw * hip_offsets[..., 1]
    target_xy = _apply_foothold_lateral_bias(target_xy, yaw, route_offset_m, cfg)
    touchdown_bias = seed_params[:, 4].view(batch_size, 1, 1, 1)
    target_xy = target_xy + touchdown_bias * float(cfg.touchdown_xy_bias_scale)
    support_xy, support_height, _ = terrain.support_at(target_xy.reshape(batch_size, -1, 2), cfg)
    target_z = support_height.reshape(batch_size, 4, int(cfg.event_cap)) + torch.relu(seed_params[:, 4]).view(batch_size, 1, 1) * float(cfg.touchdown_z_bias_scale)
    target = torch.cat((support_xy.reshape(batch_size, 4, int(cfg.event_cap), 2), target_z.unsqueeze(-1)), dim=-1)
    return target


def _support_plane_target_rpy(root_rpy: Tensor, target_foot: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    front_mid = 0.5 * (target_foot[:, 0, :] + target_foot[:, 1, :])
    rear_mid = 0.5 * (target_foot[:, 2, :] + target_foot[:, 3, :])
    left_mid = 0.5 * (target_foot[:, 0, :] + target_foot[:, 2, :])
    right_mid = 0.5 * (target_foot[:, 1, :] + target_foot[:, 3, :])
    forward = front_mid - rear_mid
    left = left_mid - right_mid
    support_normal = torch.cross(forward, left, dim=-1)
    support_normal = torch.where(support_normal[:, 2:3] < 0.0, -support_normal, support_normal)
    support_normal = support_normal / torch.linalg.vector_norm(support_normal, dim=-1, keepdim=True).clamp_min(1e-6)
    yaw = root_rpy[:, 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    yaw_aligned_x = cos_yaw * support_normal[:, 0] + sin_yaw * support_normal[:, 1]
    yaw_aligned_y = -sin_yaw * support_normal[:, 0] + cos_yaw * support_normal[:, 1]
    yaw_aligned_z = support_normal[:, 2]
    limit = float(cfg.rehome_roll_pitch_limit)
    target_roll = torch.asin((-yaw_aligned_y).clamp(-1.0, 1.0)).clamp(-limit, limit)
    target_pitch = torch.atan2(yaw_aligned_x, yaw_aligned_z).clamp(-limit, limit)
    return torch.stack((target_roll, target_pitch, yaw), dim=-1)


def _hold_rehome_targets(
    terrain: TogetherPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    time_s: Tensor,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = root_pos.shape[0]
    horizon_steps = int(cfg.horizon_steps)
    phase = 0.5 - 0.5 * torch.cos(torch.pi * time_s / time_s[-1].clamp_min(1e-6))
    phase_root = phase.view(1, horizon_steps, 1)
    phase_foot = phase.view(1, horizon_steps, 1, 1)
    nominal_xy = HIP_OFFSETS_ARRAY.to(device=root_pos.device, dtype=root_pos.dtype)[:, :2].view(1, 4, 2)
    yaw = root_rpy[:, 2:3]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    target_xy = torch.empty((batch_size, 4, 2), device=root_pos.device, dtype=root_pos.dtype)
    target_xy[..., 0] = root_pos[:, None, 0] + cos_yaw * nominal_xy[..., 0] - sin_yaw * nominal_xy[..., 1]
    target_xy[..., 1] = root_pos[:, None, 1] + sin_yaw * nominal_xy[..., 0] + cos_yaw * nominal_xy[..., 1]
    _, support_height, _ = terrain.support_at(target_xy, cfg)
    target_foot = torch.cat((target_xy, support_height.unsqueeze(-1)), dim=-1)
    target_root_z = support_height.mean(dim=-1) + float(cfg.hip_height)
    target_root = torch.cat((root_pos[:, :2], target_root_z.unsqueeze(-1)), dim=-1)
    target_rpy = _support_plane_target_rpy(root_rpy, target_foot, cfg)
    hold_root = root_pos[:, None, :] + phase_root * (target_root[:, None, :] - root_pos[:, None, :])
    hold_rpy = root_rpy[:, None, :] + phase_root * (target_rpy[:, None, :] - root_rpy[:, None, :])
    hold_foot_xy = foot_pos[:, None, :, :2] + phase_foot * (target_foot[:, None, :, :2] - foot_pos[:, None, :, :2])
    _, hold_support_height, hold_support_slope = terrain.support_at(hold_foot_xy.reshape(batch_size, -1, 2), cfg)
    hold_support_height = hold_support_height.reshape(batch_size, horizon_steps, 4)
    hold_support_slope = hold_support_slope.reshape(batch_size, horizon_steps, 4)
    linear_foot_z = foot_pos[:, None, :, 2] + phase.view(1, horizon_steps, 1) * (
        target_foot[:, None, :, 2] - foot_pos[:, None, :, 2]
    )
    hold_foot_z = torch.maximum(linear_foot_z, hold_support_height)
    hold_foot = torch.cat((hold_foot_xy, hold_foot_z.unsqueeze(-1)), dim=-1)
    first_frame = phase.view(1, horizon_steps, 1, 1) <= 0.0
    hold_foot = torch.where(first_frame, foot_pos[:, None, :, :], hold_foot)
    hold_support_height = torch.where(first_frame.squeeze(-1), foot_pos[:, None, :, 2], hold_support_height)
    return hold_root, hold_rpy, hold_foot, hold_support_height, hold_support_slope


def expand_segment(
    terrain: TogetherPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    command_batch: Tensor,
    schedule: TogetherContactSchedule,
    cfg: TogetherPlannerConfig,
    route_offset_m: Tensor | None = None,
    mode_code: Tensor | None = None,
    small_back_s: Tensor | None = None,
    small_top_z: Tensor | None = None,
    small_center_xy: Tensor | None = None,
    cross_command_batch: Tensor | None = None,
) -> TogetherRollout:
    batch_size = root_pos.shape[0]
    horizon_steps = int(cfg.horizon_steps)
    root_traj, rpy_traj, time_s, seed_params = _root_trajectory(root_pos, root_rpy, command_batch, cfg)
    route_offset = torch.zeros(batch_size, device=root_pos.device, dtype=root_pos.dtype)
    if route_offset_m is not None:
        route_offset = torch.as_tensor(route_offset_m, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size)
        root_traj = _apply_route_offsets(root_traj, rpy_traj, route_offset, cfg)
    touchdown_seq = _touchdown_targets(terrain, root_traj, rpy_traj, command_batch, seed_params, schedule, cfg, route_offset)
    frame_ids = torch.arange(horizon_steps, device=root_pos.device, dtype=torch.long).view(1, horizon_steps, 1)
    contact_bool = schedule.contact_state.to(dtype=torch.bool)
    masked_touchdown_frames = torch.where(
        schedule.touchdown_mask,
        schedule.touchdown_frames,
        torch.full_like(schedule.touchdown_frames, horizon_steps),
    )
    contact_by_leg = contact_bool.transpose(1, 2)
    full_frame_ids = torch.arange(horizon_steps, device=root_pos.device, dtype=torch.long).view(1, 1, 1, horizon_steps)
    non_contact_after_touchdown = (~contact_by_leg[:, :, None, :]) & (full_frame_ids > masked_touchdown_frames.unsqueeze(-1))
    first_non_contact_frame = torch.where(
        non_contact_after_touchdown,
        full_frame_ids,
        torch.full_like(full_frame_ids, horizon_steps),
    ).amin(dim=-1)
    stance_reference_frames = torch.where(
        first_non_contact_frame < horizon_steps,
        (first_non_contact_frame - 1).clamp_min(0),
        torch.full_like(first_non_contact_frame, horizon_steps - 1),
    )
    support_frames = torch.where(
        masked_touchdown_frames < horizon_steps,
        stance_reference_frames,
        torch.full_like(masked_touchdown_frames, horizon_steps),
    ).clamp(max=horizon_steps - 1)
    support_index = support_frames[:, None, :, :, None].expand(batch_size, 1, 4, int(cfg.event_cap), 3)
    root_support_pos = root_traj.gather(1, support_index.reshape(batch_size, 4 * int(cfg.event_cap), 3)).reshape(
        batch_size,
        4,
        int(cfg.event_cap),
        3,
    )
    root_support_rpy = rpy_traj.gather(1, support_index.reshape(batch_size, 4 * int(cfg.event_cap), 3)).reshape(
        batch_size,
        4,
        int(cfg.event_cap),
        3,
    )
    initial_yaw = root_rpy[:, 2]
    initial_offset_w = foot_pos[..., :2] - root_pos[:, None, :2]
    cos0 = torch.cos(initial_yaw).view(batch_size, 1)
    sin0 = torch.sin(initial_yaw).view(batch_size, 1)
    preview_offset_body_x = cos0 * initial_offset_w[..., 0] + sin0 * initial_offset_w[..., 1]
    preview_offset_body_y = -sin0 * initial_offset_w[..., 0] + cos0 * initial_offset_w[..., 1]
    support_yaw = root_support_rpy[..., 2]
    support_cos = torch.cos(support_yaw)
    support_sin = torch.sin(support_yaw)
    support_touchdown_xy = torch.empty((batch_size, 4, int(cfg.event_cap), 2), device=root_pos.device, dtype=root_pos.dtype)
    support_touchdown_xy[..., 0] = root_support_pos[..., 0] + support_cos * preview_offset_body_x[:, :, None] - support_sin * preview_offset_body_y[:, :, None]
    support_touchdown_xy[..., 1] = root_support_pos[..., 1] + support_sin * preview_offset_body_x[:, :, None] + support_cos * preview_offset_body_y[:, :, None]
    support_touchdown_xy = _apply_foothold_lateral_bias(support_touchdown_xy, support_yaw, route_offset, cfg)
    touchdown_bias = seed_params[:, 4].view(batch_size, 1, 1, 1)
    support_touchdown_xy = support_touchdown_xy + touchdown_bias * float(cfg.touchdown_xy_bias_scale)
    support_touchdown_xy, support_touchdown_height, _ = _semantic_touchdown_support(
        terrain,
        support_touchdown_xy,
        root_support_pos[..., :2],
        support_yaw,
        command_batch,
        route_offset,
        cfg,
    )
    support_touchdown_seq = torch.cat(
        (
            support_touchdown_xy.reshape(batch_size, 4, int(cfg.event_cap), 2),
            (
                support_touchdown_height.reshape(batch_size, 4, int(cfg.event_cap))
                + torch.relu(seed_params[:, 4]).view(batch_size, 1, 1) * float(cfg.touchdown_z_bias_scale)
            ).unsqueeze(-1),
        ),
        dim=-1,
    )
    touchdown_seq = support_touchdown_seq
    cross_mode = torch.zeros((batch_size,), device=root_pos.device, dtype=torch.bool)
    if mode_code is not None and small_back_s is not None and small_top_z is not None and small_center_xy is not None:
        mode_tensor = torch.as_tensor(mode_code, device=root_pos.device, dtype=torch.long).reshape(batch_size)
        cross_mode = mode_tensor == T116_MODE_CROSS_SMALL
        cross_command = command_batch if cross_command_batch is None else torch.as_tensor(
            cross_command_batch,
            device=root_pos.device,
            dtype=root_pos.dtype,
        ).reshape(batch_size, 3)
        cross_touchdown_seq, cross_support_xy, cross_support_height, _, _, _, _, cross_apex = _cross_small_touchdown_targets(
            terrain,
            root_pos,
            root_rpy,
            foot_pos,
            cross_command,
            torch.as_tensor(small_back_s, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size),
            torch.as_tensor(small_top_z, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size),
            torch.as_tensor(small_center_xy, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size, 2),
            schedule,
            cfg,
        )
        touchdown_seq = torch.where(cross_mode[:, None, None, None], cross_touchdown_seq, touchdown_seq)
        support_touchdown_xy = torch.where(cross_mode[:, None, None, None], cross_support_xy[:, :, None, :], support_touchdown_xy)
        support_touchdown_height = torch.where(cross_mode[:, None, None], cross_support_height[:, :, None], support_touchdown_height)
        support_touchdown_seq = torch.where(cross_mode[:, None, None, None], cross_touchdown_seq, support_touchdown_seq)
        forward_dir, _ = _command_dirs_from_root(cross_command, root_rpy, cfg)
        root_delta_s = ((root_traj[..., :2] - root_pos[:, None, :2]) * forward_dir[:, None, :]).sum(dim=-1)
        required_final_s = (
            torch.as_tensor(small_back_s, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size)
            + float(cfg.body_footprint_forward_m)
            + float(cfg.small_body_clearance)
            + float(cfg.support_search_radius)
        )
        extra_s = torch.relu(required_final_s - root_delta_s[:, -1])
        progress_phase = (time_s / max(float(cfg.horizon_s), 1e-6)).clamp(0.0, 1.0)
        cross_shift_phase = ((progress_phase - 0.55) / 0.45).clamp(0.0, 1.0)
        smooth_phase = cross_shift_phase * cross_shift_phase * (3.0 - 2.0 * cross_shift_phase)
        root_shift_xy = forward_dir[:, None, :] * extra_s[:, None, None] * smooth_phase.view(1, horizon_steps, 1)
        root_traj = torch.where(cross_mode[:, None, None], root_traj + torch.nn.functional.pad(root_shift_xy, (0, 1)), root_traj)
        lift_up = (progress_phase / 0.12).clamp(0.0, 1.0)
        lift_down = ((1.0 - progress_phase) / 0.08).clamp(0.0, 1.0)
        lift_phase = torch.minimum(lift_up, lift_down)
        current_lift = root_traj[..., 2] - root_pos[:, None, 2]
        max_extra_lift = torch.relu(float(cfg.max_root_lift_for_small) - current_lift)
        desired_extra_lift = torch.full_like(
            max_extra_lift,
            float(cfg.small_foot_clearance) + float(cfg.small_body_clearance),
        )
        root_lift_z = torch.minimum(desired_extra_lift, max_extra_lift) * lift_phase.view(1, horizon_steps)
        root_traj = root_traj.clone()
        root_traj[..., 2] = torch.where(cross_mode[:, None], root_traj[..., 2] + root_lift_z, root_traj[..., 2])
    else:
        cross_apex = foot_pos
    front_touchdown_ground_gap, rear_touchdown_ground_gap = _touchdown_ground_gap_summaries(
        touchdown_seq,
        support_touchdown_height.reshape(batch_size, 4, int(cfg.event_cap)),
    )
    touchdown_small_margin = _small_boundary_margin(terrain, support_touchdown_xy, cfg)
    event_active = frame_ids.view(1, horizon_steps, 1, 1) >= masked_touchdown_frames[:, None, :, :]
    events_reached = event_active.long().sum(dim=-1)
    next_event_index = events_reached.clamp(max=int(cfg.event_cap) - 1)
    foot_bank = foot_pos[:, None, :, None, :].expand(-1, horizon_steps, -1, 1, -1)
    touchdown_bank = support_touchdown_seq[:, None, :, :, :].expand(-1, horizon_steps, -1, -1, -1)
    foothold_bank = torch.cat((foot_bank, touchdown_bank), dim=3)
    stance = foothold_bank.gather(3, events_reached[:, :, :, None, None].expand(-1, -1, -1, 1, 3)).squeeze(3)
    touchdown_target = touchdown_bank.gather(3, next_event_index[:, :, :, None, None].expand(-1, -1, -1, 1, 3)).squeeze(3)
    valid_touchdown_count = schedule.touchdown_mask.long().sum(dim=-1)
    has_valid_next_touchdown = next_event_index < valid_touchdown_count[:, None, :]
    swing_bool = ~contact_bool
    previous_swing = torch.nn.functional.pad(swing_bool[:, :-1, :], (0, 0, 1, 0), value=False)
    swing_start = swing_bool & ~previous_swing
    swing_segment_id = swing_start.long().cumsum(dim=1)
    swing_start_frame = torch.where(
        swing_start,
        frame_ids.expand(batch_size, -1, 4),
        torch.zeros((batch_size, horizon_steps, 4), device=root_pos.device, dtype=torch.long),
    )
    last_swing_start_frame = swing_start_frame.cummax(dim=1).values
    swing_position = (frame_ids.expand(batch_size, -1, 4) - last_swing_start_frame + 1).to(dtype=root_pos.dtype)
    swing_by_leg = swing_bool.transpose(1, 2)
    swing_segment_id_by_leg = swing_segment_id.transpose(1, 2)
    same_swing_segment = (
        (swing_segment_id_by_leg[:, :, :, None] == swing_segment_id_by_leg[:, :, None, :])
        & swing_by_leg[:, :, :, None]
        & swing_by_leg[:, :, None, :]
    )
    swing_segment_length = same_swing_segment.sum(dim=-1).clamp_min(1).to(dtype=root_pos.dtype).transpose(1, 2)
    progress = (swing_position / swing_segment_length).clamp(0.0, 1.0)
    preview_yaw = rpy_traj[..., 2].unsqueeze(-1)
    preview_cos = torch.cos(preview_yaw)
    preview_sin = torch.sin(preview_yaw)
    preview_target = torch.empty((batch_size, horizon_steps, 4, 3), device=root_pos.device, dtype=root_pos.dtype)
    preview_target[..., 0] = root_traj[..., 0].unsqueeze(-1) + preview_cos * preview_offset_body_x[:, None, :] - preview_sin * preview_offset_body_y[:, None, :]
    preview_target[..., 1] = root_traj[..., 1].unsqueeze(-1) + preview_sin * preview_offset_body_x[:, None, :] + preview_cos * preview_offset_body_y[:, None, :]
    preview_height = terrain.height_at(preview_target[..., :2].reshape(batch_size, -1, 2))
    preview_target[..., 2] = preview_height.reshape(batch_size, horizon_steps, 4)
    touchdown_target = torch.where(has_valid_next_touchdown[:, :, :, None], touchdown_target, preview_target)
    swing_progress = torch.where(has_valid_next_touchdown, progress, torch.ones_like(progress))
    swing_target = stance + swing_progress[:, :, :, None] * (touchdown_target - stance)
    swing_peak = seed_params[:, 3].view(batch_size, 1, 1) * float(cfg.swing_parabola_multiplier) * progress * (1.0 - progress)
    swing_target = swing_target.clone()
    swing_target[..., 2] = swing_target[..., 2] + swing_peak
    foot_traj = torch.where(contact_bool[:, :, :, None], stance, swing_target)
    apex_target = cross_apex[:, None, :, :].expand(-1, horizon_steps, -1, -1)
    cross_arc = (1.0 - torch.abs(2.0 * swing_progress - 1.0)).clamp(0.0, 1.0)
    cross_swing_z = torch.maximum(foot_traj[..., 2], apex_target[..., 2])
    foot_traj = foot_traj.clone()
    foot_traj[..., 2] = torch.where(
        cross_mode[:, None, None] & swing_bool,
        torch.maximum(foot_traj[..., 2], foot_traj[..., 2] + cross_arc * torch.relu(cross_swing_z - foot_traj[..., 2])),
        foot_traj[..., 2],
    )
    foot_traj = _apply_height_aware_swing_clearance(
        terrain,
        stance,
        touchdown_target,
        swing_progress,
        schedule.contact_state,
        foot_traj,
        cfg,
    )
    foot_traj = _apply_semantic_swing_clearance(terrain, foot_traj, schedule.contact_state, root_traj, cfg)
    support_weight = schedule.contact_state.to(device=root_pos.device, dtype=root_pos.dtype)
    support_count = support_weight.sum(dim=-1).clamp_min(1.0)
    frame_support_height = (foot_traj[..., 2] * support_weight).sum(dim=-1) / support_count
    initial_support_count = support_weight[:, 0, :].sum(dim=-1).clamp_min(1.0)
    initial_support_height = (foot_pos[..., 2] * support_weight[:, 0, :]).sum(dim=-1) / initial_support_count
    z_curve = root_traj[..., 2] - root_pos[:, None, 2]
    root_traj = root_traj.clone()
    root_traj[..., 2] = frame_support_height + (root_pos[:, 2] - initial_support_height).view(batch_size, 1) + z_curve
    support_xy, support_height, support_slope = terrain.support_at(foot_traj[..., :2].reshape(batch_size, -1, 2), cfg)
    support_xy = support_xy.reshape(batch_size, horizon_steps, 4, 2)
    support_height = support_height.reshape(batch_size, horizon_steps, 4)
    support_slope = support_slope.reshape(batch_size, horizon_steps, 4)
    hold_mask = hold_command_mask(command_batch, cfg).to(device=root_pos.device) & ~cross_mode
    hold_root, hold_rpy, hold_foot, hold_support_height, hold_support_slope = _hold_rehome_targets(
        terrain,
        root_pos,
        root_rpy,
        foot_pos,
        time_s,
        cfg,
    )
    root_traj = torch.where(hold_mask[:, None, None], hold_root, root_traj)
    rpy_traj = torch.where(hold_mask[:, None, None], hold_rpy, rpy_traj)
    foot_traj = torch.where(hold_mask[:, None, None, None], hold_foot, foot_traj)
    touchdown_seq = torch.where(hold_mask[:, None, None, None], foot_pos[:, :, None, :].expand(-1, -1, int(cfg.event_cap), -1), touchdown_seq)
    support_xy = torch.where(hold_mask[:, None, None, None], hold_foot[..., :2], support_xy)
    support_height = torch.where(hold_mask[:, None, None], hold_support_height, support_height)
    support_slope = torch.where(hold_mask[:, None, None], hold_support_slope, support_slope)
    hold_front_gap = torch.zeros_like(front_touchdown_ground_gap)
    hold_rear_gap = torch.zeros_like(rear_touchdown_ground_gap)
    front_touchdown_ground_gap = torch.where(hold_mask[:, None], hold_front_gap, front_touchdown_ground_gap)
    rear_touchdown_ground_gap = torch.where(hold_mask[:, None], hold_rear_gap, rear_touchdown_ground_gap)
    if mode_code is None:
        rollout_mode_code = torch.full((batch_size,), T116_MODE_CRUISE, device=root_pos.device, dtype=torch.long)
    else:
        rollout_mode_code = torch.as_tensor(mode_code, device=root_pos.device, dtype=torch.long).reshape(batch_size)
    (
        candidate_anchor_references,
        candidate_touchdown_targets,
        candidate_path_progress,
        candidate_pair_summary,
        candidate_posture_summary,
        candidate_action_segment_diagnostics_present,
        front_pair_consistency,
        rear_pair_follow_consistency,
        body_posture_score,
    ) = _candidate_action_segment_summaries(
        root_traj,
        rpy_traj,
        stance,
        touchdown_target,
        swing_progress,
        rollout_mode_code,
        cfg,
    )
    (
        anchor_to_touchdown_foot_clearance,
        anchor_to_touchdown_leg_clearance,
        candidate_path_collision_flag,
    ) = _candidate_path_clearance_summaries(
        terrain,
        root_traj,
        rpy_traj,
        foot_traj,
        schedule.contact_state,
        cfg,
    )
    touchdown_frame_by_leg = torch.where(
        schedule.touchdown_mask[:, :, 0],
        schedule.touchdown_frames[:, :, 0],
        torch.full((batch_size, 4), horizon_steps, device=root_pos.device, dtype=torch.long),
    )
    touchdown_semantic_by_leg = _semantic_at(terrain, touchdown_seq[:, :, 0, :2])
    touchdown_ground_gap_by_leg = touchdown_seq[:, :, 0, 2] - support_touchdown_height[:, :, 0]
    if mode_code is not None and small_back_s is not None and small_top_z is not None and small_center_xy is not None:
        cross_command = command_batch if cross_command_batch is None else torch.as_tensor(
            cross_command_batch,
            device=root_pos.device,
            dtype=root_pos.dtype,
        ).reshape(batch_size, 3)
        cross_touchdown_seq, _, _, _, _, cross_beyond, cross_on_small, cross_apex_diag = _cross_small_touchdown_targets(
            terrain,
            root_pos,
            root_rpy,
            foot_pos,
            cross_command,
            torch.as_tensor(small_back_s, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size),
            torch.as_tensor(small_top_z, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size),
            torch.as_tensor(small_center_xy, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size, 2),
            schedule,
            cfg,
        )
        path_collision, path_clearance = _cross_small_per_leg_path_diagnostics(
            terrain,
            foot_pos,
            cross_apex_diag,
            cross_touchdown_seq[:, :, 0, :],
            cfg,
        )
        per_leg_touchdown_on_small_count = torch.where(cross_mode[:, None], cross_on_small.to(dtype=torch.long), torch.zeros_like(path_collision))
        per_leg_foot_small_collision_count = torch.where(cross_mode[:, None], path_collision, torch.zeros_like(path_collision))
        per_leg_min_clearance_to_small = torch.where(
            cross_mode[:, None],
            path_clearance,
            torch.zeros_like(path_clearance),
        )
        per_leg_touchdown_beyond_small_back_edge = cross_mode[:, None] & cross_beyond
    else:
        per_leg_touchdown_on_small_count = torch.zeros((batch_size, 4), device=root_pos.device, dtype=torch.long)
        per_leg_foot_small_collision_count = torch.zeros((batch_size, 4), device=root_pos.device, dtype=torch.long)
        per_leg_min_clearance_to_small = torch.zeros((batch_size, 4), device=root_pos.device, dtype=root_pos.dtype)
        per_leg_touchdown_beyond_small_back_edge = torch.zeros((batch_size, 4), device=root_pos.device, dtype=torch.bool)
    command_leading_before_trailing_schedule_ok = _cross_small_schedule_order_ok(
        command_batch if cross_command_batch is None else torch.as_tensor(cross_command_batch, device=root_pos.device, dtype=root_pos.dtype).reshape(batch_size, 3),
        root_pos,
        root_rpy,
        foot_pos,
        touchdown_frame_by_leg,
        cfg,
    )
    (
        touchdown_on_small_count,
        front_foot_small_collision_count,
        rear_foot_small_collision_count,
        front_foot_min_clearance_to_small,
        rear_foot_min_clearance_to_small,
        base_small_penetration_count,
        base_min_clearance_to_small,
        base_path_crosses_small_flag,
    ) = _small_surface_crossing_summaries(
        terrain,
        touchdown_seq,
        foot_traj,
        schedule.contact_state,
        root_traj,
        rpy_traj,
        cfg,
    )
    return TogetherRollout(
        root_pos=root_traj,
        root_rpy=rpy_traj,
        foot_pos=foot_traj,
        touchdown_seq=touchdown_seq,
        touchdown_mask=schedule.touchdown_mask,
        contact_state=schedule.contact_state,
        support_xy=support_xy,
        support_height=support_height,
        support_slope=support_slope,
        time_s=time_s,
        route_offset_m=route_offset,
        mode_code=rollout_mode_code,
        candidate_anchor_references=candidate_anchor_references,
        candidate_touchdown_targets=candidate_touchdown_targets,
        candidate_path_progress=candidate_path_progress,
        candidate_pair_summary=candidate_pair_summary,
        candidate_posture_summary=candidate_posture_summary,
        candidate_action_segment_diagnostics_present=candidate_action_segment_diagnostics_present,
        touchdown_small_margin=touchdown_small_margin,
        front_touchdown_ground_gap=front_touchdown_ground_gap,
        rear_touchdown_ground_gap=rear_touchdown_ground_gap,
        touchdown_on_small_count=touchdown_on_small_count,
        front_foot_small_collision_count=front_foot_small_collision_count,
        rear_foot_small_collision_count=rear_foot_small_collision_count,
        front_foot_min_clearance_to_small=front_foot_min_clearance_to_small,
        rear_foot_min_clearance_to_small=rear_foot_min_clearance_to_small,
        base_small_penetration_count=base_small_penetration_count,
        base_min_clearance_to_small=base_min_clearance_to_small,
        base_path_crosses_small_flag=base_path_crosses_small_flag,
        front_pair_consistency=front_pair_consistency,
        rear_pair_follow_consistency=rear_pair_follow_consistency,
        body_posture_score=body_posture_score,
        anchor_to_touchdown_foot_clearance=anchor_to_touchdown_foot_clearance,
        anchor_to_touchdown_leg_clearance=anchor_to_touchdown_leg_clearance,
        candidate_path_collision_flag=candidate_path_collision_flag,
        per_leg_touchdown_on_small_count=per_leg_touchdown_on_small_count,
        per_leg_foot_small_collision_count=per_leg_foot_small_collision_count,
        per_leg_min_clearance_to_small=per_leg_min_clearance_to_small,
        per_leg_touchdown_beyond_small_back_edge=per_leg_touchdown_beyond_small_back_edge,
        touchdown_ground_gap_by_leg=touchdown_ground_gap_by_leg,
        touchdown_semantic_by_leg=touchdown_semantic_by_leg,
        touchdown_frame_by_leg=touchdown_frame_by_leg,
        command_leading_before_trailing_schedule_ok=command_leading_before_trailing_schedule_ok,
    )


__all__ = [
    "T116_MODE_APPROACH_SMALL",
    "T116_MODE_BYPASS_OBSTACLE",
    "T116_MODE_CROSS_SMALL",
    "T116_MODE_CRUISE",
    "T116ModeGeometry",
    "TogetherRollout",
    "build_time_grid",
    "classify_mode_and_geometry",
    "expand_segment",
    "integrate_body_frame_translation",
]
