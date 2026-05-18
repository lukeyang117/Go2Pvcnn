"""Scanner-driven terrain and semantic losses for batch MPC."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..terrain import TerrainQueryCache, _world_to_grid, height_at, semantic_at, slope_at, support_at
from ..types import MpcPlannerTerrain


def _safe_norm(value: Tensor, *, dim: int, eps: float = 1.0e-12) -> Tensor:
    return torch.sqrt(torch.sum(value.square(), dim=dim) + float(eps))


@dataclass(frozen=True)
class ObstacleRiskScales:
    linear_scale: Tensor
    yaw_scale: Tensor
    linear_trigger_count: Tensor
    yaw_trigger_count: Tensor
    trigger_horizon_index: Tensor
    trigger_semantic_class: Tensor


def _smooth_l1_small(value: Tensor, target: Tensor, *, beta: float = 0.02) -> Tensor:
    err = torch.abs(value - target)
    beta_t = torch.as_tensor(float(beta), dtype=err.dtype, device=err.device)
    return torch.where(err < beta_t, 0.5 * err.square() / beta_t, err - 0.5 * beta_t)


def _semantic_id_mask(semantic: Tensor, ids: tuple[int, ...]) -> Tensor:
    if len(ids) == 0:
        return torch.zeros_like(semantic, dtype=torch.bool)
    id_tensor = torch.as_tensor(ids, dtype=semantic.dtype, device=semantic.device)
    return (semantic.unsqueeze(-1) == id_tensor.view(*([1] * semantic.ndim), -1)).any(dim=-1)


def _semantic_obstacle_field(
    terrain: MpcPlannerTerrain,
    *,
    dtype: torch.dtype,
    device: torch.device,
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    small_weight: float,
    large_weight: float,
    soft_margin_m: float,
    small_mask_override: Tensor | None = None,
    large_mask_override: Tensor | None = None,
) -> Tensor | None:
    if terrain.semantic_map is None:
        return None
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if small_mask_override is None:
        small_mask = _semantic_id_mask(semantic, small_ids)
    else:
        small_mask = torch.as_tensor(small_mask_override, dtype=torch.bool, device=device)
    if large_mask_override is None:
        large_mask = _semantic_id_mask(semantic, large_ids)
    else:
        large_mask = torch.as_tensor(large_mask_override, dtype=torch.bool, device=device)
    small = small_mask.to(dtype=dtype, device=device)
    large = large_mask.to(dtype=dtype, device=device)
    obstacle = float(small_weight) * small + float(large_weight) * large
    margin = float(soft_margin_m)
    if margin <= 0.0:
        return obstacle
    rows = max(int(obstacle.shape[-2]), 1)
    cols = max(int(obstacle.shape[-1]), 1)
    cell_x = abs(float(terrain.world_x_range[1]) - float(terrain.world_x_range[0])) / max(cols - 1, 1)
    cell_y = abs(float(terrain.world_y_range[1]) - float(terrain.world_y_range[0])) / max(rows - 1, 1)
    radius = max(1, int(torch.ceil(torch.as_tensor(margin / max(min(cell_x, cell_y), 1.0e-6))).item()))
    pad = radius
    kernel_size = 2 * radius + 1
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=dtype, device=device),
        torch.arange(-radius, radius + 1, dtype=dtype, device=device),
        indexing="ij",
    )
    dist = torch.sqrt((xx * float(cell_x)).square() + (yy * float(cell_y)).square() + 1.0e-12)
    kernel = torch.clamp(1.0 - dist / max(margin, 1.0e-6), min=0.0)
    weighted = F.conv2d(
        obstacle.unsqueeze(1),
        kernel.view(1, 1, kernel_size, kernel_size),
        padding=pad,
    ).squeeze(1)
    return weighted.clamp_min(0.0)


def _sample_obstacle_field(terrain: MpcPlannerTerrain, field: Tensor, points_xy: Tensor) -> Tensor:
    batch = int(field.shape[0])
    points = torch.as_tensor(points_xy, dtype=field.dtype, device=field.device)
    if points.ndim < 3:
        points = points.unsqueeze(0)
    original_shape = tuple(points.shape[:-1])
    point_batch = int(points.shape[0])
    if batch == 1 and point_batch > 1:
        field = field.expand(point_batch, -1, -1)
        batch = point_batch
    if point_batch == 1 and batch > 1:
        points = points.expand(batch, *points.shape[1:])
        original_shape = tuple(points.shape[:-1])
    if int(points.shape[0]) != batch:
        raise ValueError(f"points batch {int(points.shape[0])} must match field batch {batch}")
    flat = points.reshape(batch, -1, 2)
    grid = _world_to_grid(terrain, flat).unsqueeze(2)
    sampled = F.grid_sample(
        field.unsqueeze(1),
        grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return sampled[:, 0, :, 0].reshape(original_shape)


def stance_ground_loss(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    contact_prob: Tensor,
    *,
    min_contact_prob: float = 0.0,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    foot_xy = foot_pos[..., :2]
    terrain_z = height_at(terrain, foot_xy, cache=query_cache).to(dtype=foot_pos.dtype, device=foot_pos.device)
    err = _smooth_l1_small(foot_pos[..., 2], terrain_z)
    weight = contact_prob.to(dtype=foot_pos.dtype)
    if float(min_contact_prob) > 0.0:
        weight = torch.where(weight >= float(min_contact_prob), weight, torch.zeros_like(weight))
    return (weight * err).sum(dim=(1, 2)) / torch.clamp(weight.sum(dim=(1, 2)), min=1.0)


def swing_clearance_terrain_loss(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    swing_prob: Tensor,
    *,
    min_clearance_m: float,
    worst_deficit_weight: float = 0.0,
    min_swing_prob: float = 0.0,
    hard_active_weight: bool = False,
    boundary_min_swing_prob: float = 0.0,
    boundary_weight: float = 0.0,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    foot_xy = foot_pos[..., :2]
    terrain_z = height_at(terrain, foot_xy, cache=query_cache).to(dtype=foot_pos.dtype, device=foot_pos.device)
    deficit = torch.relu(terrain_z + float(min_clearance_m) - foot_pos[..., 2])
    weight = swing_prob.to(dtype=foot_pos.dtype)
    if float(min_swing_prob) > 0.0:
        active = weight >= float(min_swing_prob)
        if bool(hard_active_weight):
            weight = active.to(dtype=foot_pos.dtype)
        else:
            weight = torch.where(active, weight, torch.zeros_like(weight))
    mean_loss = (weight * deficit.square()).sum(dim=(1, 2)) / torch.clamp(weight.sum(dim=(1, 2)), min=1.0)
    if float(boundary_weight) > 0.0 and float(min_swing_prob) > float(boundary_min_swing_prob):
        raw_prob = swing_prob.to(dtype=foot_pos.dtype)
        lower = float(boundary_min_swing_prob)
        upper = float(min_swing_prob)
        boundary = torch.clamp((raw_prob - lower) / max(upper - lower, 1.0e-6), 0.0, 1.0)
        boundary = torch.where(raw_prob < upper, boundary, torch.zeros_like(boundary))
        mean_loss = mean_loss + float(boundary_weight) * (boundary * deficit.square()).sum(dim=(1, 2)) / torch.clamp(
            boundary.sum(dim=(1, 2)),
            min=1.0,
        )
    if float(worst_deficit_weight) <= 0.0:
        return mean_loss
    active_deficit = torch.where(weight > 1.0e-4, deficit, torch.zeros_like(deficit))
    worst_loss = active_deficit.amax(dim=(1, 2)).square()
    return mean_loss + float(worst_deficit_weight) * worst_loss


def body_heightfield_collision_loss(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    *,
    bottom_offset_z: float,
    margin_m: float,
    stencil_xy: tuple[tuple[float, float], ...],
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    """Penalize root-bottom samples that penetrate the height field."""
    offsets = torch.as_tensor(stencil_xy, dtype=root_pos.dtype, device=root_pos.device)
    if offsets.ndim != 2 or int(offsets.shape[-1]) != 2:
        raise ValueError("stencil_xy must contain 2D xy offsets")
    yaw = root_rpy[..., 2]
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    ox = offsets[:, 0].view(1, 1, -1)
    oy = offsets[:, 1].view(1, 1, -1)
    sample_xy = torch.stack((cy * ox - sy * oy, sy * ox + cy * oy), dim=-1) + root_pos[..., None, :2]
    sample_z = root_pos[..., None, 2] + float(bottom_offset_z)
    terrain_z = height_at(terrain, sample_xy, cache=query_cache).to(dtype=root_pos.dtype, device=root_pos.device)
    deficit = torch.relu(terrain_z + float(margin_m) - sample_z)
    return deficit.square().mean(dim=(1, 2))


def knee_shank_heightfield_collision_loss(
    terrain: MpcPlannerTerrain,
    knee_pos_world: Tensor,
    shank_sample_world: Tensor,
    *,
    knee_margin_m: float,
    shank_margin_m: float,
    worst_deficit_weight: float = 0.0,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    """Penalize planned knee and shank sample points that collide with the height field."""
    knee_xy = knee_pos_world[..., :2]
    shank_xy = shank_sample_world[..., :2]
    knee_h = height_at(terrain, knee_xy, cache=query_cache).to(dtype=knee_pos_world.dtype, device=knee_pos_world.device)
    knee_deficit = torch.relu(knee_h + float(knee_margin_m) - knee_pos_world[..., 2])
    shank_h = height_at(terrain, shank_xy, cache=query_cache).to(
        dtype=shank_sample_world.dtype,
        device=shank_sample_world.device,
    )
    shank_deficit = torch.relu(shank_h + float(shank_margin_m) - shank_sample_world[..., 2])
    mean_loss = knee_deficit.square().mean(dim=(1, 2)) + shank_deficit.square().mean(dim=(1, 2, 3))
    if float(worst_deficit_weight) <= 0.0:
        return mean_loss
    worst_loss = torch.maximum(knee_deficit.amax(dim=(1, 2)), shank_deficit.amax(dim=(1, 2, 3))).square()
    return mean_loss + float(worst_deficit_weight) * worst_loss


def _terrain_grid_world_xy(terrain: MpcPlannerTerrain, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    batch, height_count, width_count = int(height.shape[0]), int(height.shape[1]), int(height.shape[2])
    x = torch.linspace(float(terrain.world_x_range[0]), float(terrain.world_x_range[1]), width_count, dtype=dtype, device=device)
    y = torch.linspace(float(terrain.world_y_range[0]), float(terrain.world_y_range[1]), height_count, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    local_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    local_xy = local_xy.view(1, height_count * width_count, 2).expand(batch, -1, -1)
    if terrain.sensor_pos_w is None:
        return local_xy
    sensor_pos = torch.as_tensor(terrain.sensor_pos_w, dtype=dtype, device=device)
    if sensor_pos.ndim == 1:
        sensor_pos = sensor_pos.view(1, -1).expand(batch, -1)
    if terrain.sensor_yaw is None:
        yaw = torch.zeros((batch,), dtype=dtype, device=device)
    else:
        yaw = torch.as_tensor(terrain.sensor_yaw, dtype=dtype, device=device).reshape(-1)
    cy = torch.cos(yaw).view(batch, 1)
    sy = torch.sin(yaw).view(batch, 1)
    world_xy = torch.stack(
        (cy * local_xy[..., 0] - sy * local_xy[..., 1], sy * local_xy[..., 0] + cy * local_xy[..., 1]),
        dim=-1,
    )
    return world_xy + sensor_pos[:, None, :2]


def _nearby_height_for_sparse_semantic(
    terrain: MpcPlannerTerrain,
    height: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    neighborhood_m: float = 0.18,
) -> Tensor:
    """Return a local max-height field so sparse semantic hits inherit nearby object height."""
    rows = max(int(height.shape[-2]), 1)
    cols = max(int(height.shape[-1]), 1)
    cell_x = abs(float(terrain.world_x_range[1]) - float(terrain.world_x_range[0])) / max(cols - 1, 1)
    cell_y = abs(float(terrain.world_y_range[1]) - float(terrain.world_y_range[0])) / max(rows - 1, 1)
    radius = max(0, int(math.ceil(float(neighborhood_m) / max(min(cell_x, cell_y), 1.0e-6))))
    if radius <= 0:
        return height.reshape(int(height.shape[0]), -1)
    padded = F.pad(height.unsqueeze(1), (radius, radius, radius, radius), mode="constant", value=-1.0e6)
    pooled = F.max_pool2d(padded, kernel_size=2 * radius + 1, stride=1).squeeze(1)
    return pooled.to(dtype=dtype, device=device).reshape(int(height.shape[0]), -1)


def obstacle_risk_scales(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    command: Tensor,
    *,
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    high_small_relative_height_m: float,
    linear_corridor_width_m: float,
    linear_forward_distance_m: float,
    yaw_swept_radius_m: float,
    linear_scale_when_blocked: float,
    yaw_scale_when_blocked: float,
    linear_speed_eps: float,
    yaw_speed_eps: float,
    query_cache: TerrainQueryCache | None = None,
) -> ObstacleRiskScales:
    batch = int(root_pos.shape[0])
    dtype = root_pos.dtype
    device = root_pos.device
    ones = torch.ones((batch,), dtype=dtype, device=device)
    zero_counts = torch.zeros((batch,), dtype=torch.long, device=device)
    no_trigger_index = torch.full((batch,), -1, dtype=torch.long, device=device)
    no_trigger_class = torch.zeros((batch,), dtype=torch.long, device=device)
    if terrain.semantic_map is None:
        return ObstacleRiskScales(
            linear_scale=ones,
            yaw_scale=ones,
            linear_trigger_count=zero_counts,
            yaw_trigger_count=zero_counts,
            trigger_horizon_index=no_trigger_index,
            trigger_semantic_class=no_trigger_class,
        )

    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0).expand(batch, -1, -1)
    grid_xy = _terrain_grid_world_xy(terrain, dtype=dtype, device=device)
    grid_z = height.reshape(batch, -1)
    nearby_grid_z = _nearby_height_for_sparse_semantic(terrain, height, dtype=dtype, device=device)
    grid_sem = semantic.reshape(batch, -1)
    root0 = root_pos[:, 0]
    rpy0 = root_rpy[:, 0]
    root0_xy = root0[:, None, :2]
    root_ground_z = height_at(terrain, root0_xy, cache=query_cache).reshape(batch).to(dtype=dtype, device=device)
    small = _semantic_id_mask(grid_sem, small_ids)
    large = _semantic_id_mask(grid_sem, large_ids)
    high_small = torch.logical_and(small, (nearby_grid_z - root_ground_z[:, None]) > float(high_small_relative_height_m))
    risky = torch.logical_or(large, high_small)

    delta = grid_xy - root0[:, None, :2]
    yaw = rpy0[:, 2]
    cy = torch.cos(yaw).view(batch, 1)
    sy = torch.sin(yaw).view(batch, 1)
    body_delta = torch.stack((cy * delta[..., 0] + sy * delta[..., 1], -sy * delta[..., 0] + cy * delta[..., 1]), dim=-1)
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    cmd_xy = cmd[:, :2]
    cmd_speed = _safe_norm(cmd_xy, dim=-1)
    heading = cmd_xy / cmd_speed.clamp_min(1.0e-6).unsqueeze(-1)
    forward = (body_delta * heading[:, None, :]).sum(dim=-1)
    lateral = body_delta[..., 0] * (-heading[:, 1]).view(batch, 1) + body_delta[..., 1] * heading[:, 0].view(batch, 1)
    linear_active = cmd_speed > float(linear_speed_eps)
    linear_risk = torch.logical_and(
        risky,
        torch.logical_and(
            torch.logical_and(forward >= 0.0, forward <= float(linear_forward_distance_m)),
            torch.abs(lateral) <= float(linear_corridor_width_m),
        ),
    )
    root_path_delta = grid_xy[:, None, :, :] - root_pos[..., None, :2]
    root_path_dist = _safe_norm(root_path_delta, dim=-1).amin(dim=1)
    path_risk = torch.logical_and(risky, root_path_dist <= float(linear_corridor_width_m))
    linear_risk = torch.logical_and(torch.logical_or(linear_risk, path_risk), linear_active[:, None])
    yaw_active = torch.abs(cmd[:, 2]) > float(yaw_speed_eps)
    radial = _safe_norm(body_delta, dim=-1)
    yaw_risk = torch.logical_and(risky, radial <= float(yaw_swept_radius_m))
    yaw_risk = torch.logical_and(yaw_risk, yaw_active[:, None])
    linear_count = linear_risk.sum(dim=-1).to(dtype=torch.long)
    yaw_count = yaw_risk.sum(dim=-1).to(dtype=torch.long)
    linear_scale = torch.where(linear_count > 0, torch.full_like(ones, float(linear_scale_when_blocked)), ones)
    yaw_scale = torch.where(yaw_count > 0, torch.full_like(ones, float(yaw_scale_when_blocked)), ones)
    any_risk = torch.logical_or(linear_risk, yaw_risk)
    safe_index = torch.arange(grid_sem.shape[1], dtype=torch.long, device=device).view(1, -1).expand(batch, -1)
    first_index = torch.where(any_risk, safe_index, torch.full_like(safe_index, grid_sem.shape[1])).amin(dim=-1)
    trigger_index = torch.where(first_index < grid_sem.shape[1], first_index, no_trigger_index)
    clamped_index = first_index.clamp(max=grid_sem.shape[1] - 1)
    trigger_semantic = grid_sem.gather(1, clamped_index.unsqueeze(-1)).squeeze(-1)
    trigger_semantic = torch.where(first_index < grid_sem.shape[1], trigger_semantic, no_trigger_class)
    return ObstacleRiskScales(
        linear_scale=linear_scale,
        yaw_scale=yaw_scale,
        linear_trigger_count=linear_count,
        yaw_trigger_count=yaw_count,
        trigger_horizon_index=trigger_index,
        trigger_semantic_class=trigger_semantic,
    )


def low_small_crossing_progress_loss(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    command: Tensor,
    *,
    small_ids: tuple[int, ...],
    high_small_relative_height_m: float,
    corridor_width_m: float,
    forward_distance_m: float,
    pass_margin_m: float,
    obstacle_depth_m: float = 0.0,
    linear_speed_eps: float,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    """Encourage crossing low small obstacles that lie in the commanded path."""
    batch = int(root_pos.shape[0])
    dtype = root_pos.dtype
    device = root_pos.device
    zero = torch.zeros((batch,), dtype=dtype, device=device)
    if terrain.semantic_map is None:
        return zero

    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0).expand(batch, -1, -1)
    grid_xy = _terrain_grid_world_xy(terrain, dtype=dtype, device=device)
    grid_z = height.reshape(batch, -1)
    nearby_grid_z = _nearby_height_for_sparse_semantic(terrain, height, dtype=dtype, device=device)
    grid_sem = semantic.reshape(batch, -1)

    root0 = root_pos[:, 0]
    root_end = root_pos[:, -1]
    root0_xy = root0[:, None, :2]
    root_ground_z = height_at(terrain, root0_xy, cache=query_cache).reshape(batch).to(dtype=dtype, device=device)
    low_small = torch.logical_and(
        _semantic_id_mask(grid_sem, small_ids),
        (nearby_grid_z - root_ground_z[:, None]) <= float(high_small_relative_height_m),
    )

    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    cmd_xy = cmd[:, :2]
    cmd_speed = _safe_norm(cmd_xy, dim=-1)
    active = cmd_speed > float(linear_speed_eps)
    heading = cmd_xy / cmd_speed.clamp_min(1.0e-6).unsqueeze(-1)

    delta = grid_xy - root0[:, None, :2]
    forward = (delta * heading[:, None, :]).sum(dim=-1)
    lateral = delta[..., 0] * (-heading[:, 1]).view(batch, 1) + delta[..., 1] * heading[:, 0].view(batch, 1)
    candidate = torch.logical_and(
        low_small,
        torch.logical_and(
            torch.logical_and(forward >= 0.0, forward <= float(forward_distance_m)),
            torch.abs(lateral) <= float(corridor_width_m),
        ),
    )
    candidate = torch.logical_and(candidate, active[:, None])
    desired_pass = torch.where(
        candidate,
        forward + max(float(obstacle_depth_m), 0.0) + float(pass_margin_m),
        torch.zeros_like(forward),
    )
    required_progress = desired_pass.amax(dim=-1)

    dxy_w = root_end[:, :2] - root0[:, :2]
    progress = (dxy_w * heading).sum(dim=-1)
    return torch.where(required_progress > 0.0, torch.relu(required_progress - progress).square(), zero)


def high_obstacle_avoidance_loss(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    command: Tensor,
    *,
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    high_small_relative_height_m: float,
    corridor_width_m: float,
    forward_distance_m: float,
    lateral_clearance_m: float,
    longitudinal_influence_m: float,
    linear_speed_eps: float,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    """Push the root path laterally around large and too-high small obstacles in the command corridor."""
    batch = int(root_pos.shape[0])
    dtype = root_pos.dtype
    device = root_pos.device
    zero = torch.zeros((batch,), dtype=dtype, device=device)
    if terrain.semantic_map is None:
        return zero

    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0).expand(batch, -1, -1)
    grid_xy = _terrain_grid_world_xy(terrain, dtype=dtype, device=device)
    grid_z = height.reshape(batch, -1)
    nearby_grid_z = _nearby_height_for_sparse_semantic(terrain, height, dtype=dtype, device=device)
    grid_sem = semantic.reshape(batch, -1)

    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    cmd_xy = cmd[:, :2]
    cmd_speed = _safe_norm(cmd_xy, dim=-1)
    active = cmd_speed > float(linear_speed_eps)
    heading_body = cmd_xy / cmd_speed.clamp_min(1.0e-6).unsqueeze(-1)

    root0 = root_pos[:, 0]
    yaw0 = root_rpy[:, 0, 2]
    cy = torch.cos(yaw0)
    sy = torch.sin(yaw0)
    heading = torch.stack(
        (
            cy * heading_body[:, 0] - sy * heading_body[:, 1],
            sy * heading_body[:, 0] + cy * heading_body[:, 1],
        ),
        dim=-1,
    )
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)

    root0_xy = root0[:, None, :2]
    root_ground_z = height_at(terrain, root0_xy, cache=query_cache).reshape(batch).to(dtype=dtype, device=device)
    small = _semantic_id_mask(grid_sem, small_ids)
    large = _semantic_id_mask(grid_sem, large_ids)
    high_small = torch.logical_and(small, (nearby_grid_z - root_ground_z[:, None]) > float(high_small_relative_height_m))
    risky = torch.logical_or(large, high_small)

    delta0 = grid_xy - root0[:, None, :2]
    forward0 = (delta0 * heading[:, None, :]).sum(dim=-1)
    lateral0 = (delta0 * left[:, None, :]).sum(dim=-1)
    in_corridor = torch.logical_and(
        torch.logical_and(forward0 >= 0.0, forward0 <= float(forward_distance_m)),
        torch.abs(lateral0) <= float(corridor_width_m),
    )
    root_path_delta = grid_xy[:, None, :, :] - root_pos[..., None, :2]
    root_path_dist = _safe_norm(root_path_delta, dim=-1).amin(dim=1)
    path_near = root_path_dist <= float(corridor_width_m)
    candidate = torch.logical_and(torch.logical_and(risky, torch.logical_or(in_corridor, path_near)), active[:, None])
    weight = candidate.to(dtype=dtype, device=device)
    count = weight.sum(dim=-1)

    obstacle_lateral = (lateral0 * weight).sum(dim=-1) / count.clamp_min(1.0)
    desired_side = torch.where(obstacle_lateral > 0.0, -torch.ones_like(obstacle_lateral), torch.ones_like(obstacle_lateral))

    rel_path = root_pos[..., None, :2] - grid_xy[:, None, :, :]
    path_forward = (rel_path * heading[:, None, None, :]).sum(dim=-1)
    path_lateral = (rel_path * left[:, None, None, :]).sum(dim=-1)
    influence = torch.relu(1.0 - torch.abs(path_forward) / max(float(longitudinal_influence_m), 1.0e-6))
    signed_clearance = desired_side[:, None, None] * path_lateral
    deficit = torch.relu(float(lateral_clearance_m) - signed_clearance)
    cell_cost = weight[:, None, :] * influence * deficit.square()
    mean_cost = cell_cost.sum(dim=(1, 2)) / torch.clamp((weight[:, None, :] * influence).sum(dim=(1, 2)), min=1.0)
    worst_cost = cell_cost.amax(dim=(1, 2))
    return torch.where(count > 0.0, mean_cost + worst_cost, zero)


def finite_horizon_touchdown_phase(swing_center: Tensor, swing_width: Tensor) -> Tensor:
    """Return touchdown endpoint phase in the current finite horizon."""
    return torch.clamp(swing_center + 0.5 * swing_width, min=0.0, max=1.0)


def sample_time(values: Tensor, phase: Tensor, *, cyclic: bool = True) -> Tensor:
    """Linearly sample [B,T,...] values at cyclic phase [B,4]."""
    batch, horizon, legs, *tail = values.shape
    if cyclic:
        pos = torch.remainder(phase, 1.0) * float(horizon)
        i0 = torch.floor(pos).to(dtype=torch.long) % horizon
        i1 = (i0 + 1) % horizon
    else:
        pos = torch.clamp(phase, 0.0, 1.0) * float(max(horizon - 1, 1))
        i0 = torch.floor(pos).to(dtype=torch.long).clamp(0, horizon - 1)
        i1 = (i0 + 1).clamp(0, horizon - 1)
    alpha = (pos - torch.floor(pos)).to(dtype=values.dtype)
    b = torch.arange(batch, device=values.device).view(batch, 1).expand(batch, legs)
    l = torch.arange(legs, device=values.device).view(1, legs).expand(batch, legs)
    v0 = values[b, i0, l]
    v1 = values[b, i1, l]
    return torch.lerp(v0, v1, alpha.view(batch, legs, *([1] * len(tail))))


def touchdown_surface_loss(
    terrain: MpcPlannerTerrain,
    touchdown_w: Tensor,
    *,
    slope_sample_step: float,
    support_search_radius: float,
    support_search_step: float,
    max_slope: float,
    max_support_slope: float,
    support_height_tolerance: float,
    ground_weight: float,
    slope_weight: float,
    support_distance_weight: float,
    support_height_weight: float,
    support_slope_weight: float,
    invalid_support_weight: float,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    touchdown_xy = touchdown_w[..., :2]
    touchdown_z = touchdown_w[..., 2]
    terrain_z = height_at(terrain, touchdown_xy, cache=query_cache).to(dtype=touchdown_w.dtype, device=touchdown_w.device)
    slope = slope_at(terrain, touchdown_xy, sample_step=float(slope_sample_step), cache=query_cache).to(dtype=touchdown_w.dtype, device=touchdown_w.device)
    support_xy, support_z, support_slope, invalid = support_at(
        terrain,
        touchdown_xy,
        search_radius=float(support_search_radius),
        search_step=float(support_search_step),
        max_support_slope=float(max_support_slope),
        cache=query_cache,
    )
    support_xy = support_xy.to(dtype=touchdown_w.dtype, device=touchdown_w.device)
    support_z = support_z.to(dtype=touchdown_w.dtype, device=touchdown_w.device)
    support_slope = support_slope.to(dtype=touchdown_w.dtype, device=touchdown_w.device)
    invalid_f = invalid.to(dtype=touchdown_w.dtype, device=touchdown_w.device)
    ground = _smooth_l1_small(touchdown_z, terrain_z)
    slope_pen = torch.relu(slope - float(max_slope)).square()
    support_dist = _safe_norm(touchdown_xy - support_xy, dim=-1)
    support_height = torch.relu(torch.abs(touchdown_z - support_z) - float(support_height_tolerance)).square()
    support_slope_pen = torch.relu(support_slope - float(max_support_slope)).square()
    total = (
        float(ground_weight) * ground
        + float(slope_weight) * slope_pen
        + float(support_distance_weight) * support_dist
        + float(support_height_weight) * support_height
        + float(support_slope_weight) * support_slope_pen
        + float(invalid_support_weight) * invalid_f
    )
    return total.mean(dim=-1)


def touchdown_semantic_loss(
    terrain: MpcPlannerTerrain,
    touchdown_xy: Tensor,
    touchdown_z: Tensor | None = None,
    *,
    small_weight: float,
    large_weight: float,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    semantic = semantic_at(terrain, touchdown_xy, cache=query_cache)
    small = (semantic == 1).to(dtype=torch.float32, device=touchdown_xy.device)
    large = (semantic >= 2).to(dtype=torch.float32, device=touchdown_xy.device)
    return (float(small_weight) * small + float(large_weight) * large).mean(dim=-1)


def stance_semantic_obstacle_loss(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    contact_prob: Tensor,
    *,
    ground_ids: tuple[int, ...],
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    small_weight: float,
    large_weight: float,
    min_contact_prob: float = 0.0,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    foot_xy = foot_pos[..., :2]
    semantic = semantic_at(terrain, foot_xy, cache=query_cache)
    ground = _semantic_id_mask(semantic, ground_ids)
    small = torch.logical_and(_semantic_id_mask(semantic, small_ids), torch.logical_not(ground))
    large = torch.logical_and(_semantic_id_mask(semantic, large_ids), torch.logical_not(ground))
    penalty = (
        float(small_weight) * small.to(dtype=foot_pos.dtype, device=foot_pos.device)
        + float(large_weight) * large.to(dtype=foot_pos.dtype, device=foot_pos.device)
    )
    weight = contact_prob.to(dtype=foot_pos.dtype, device=foot_pos.device)
    if float(min_contact_prob) > 0.0:
        weight = torch.where(weight >= float(min_contact_prob), weight, torch.zeros_like(weight))
    return (weight * penalty).sum(dim=(1, 2)) / torch.clamp(weight.sum(dim=(1, 2)), min=1.0)


def semantic_contact_avoidance_loss(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    contact_prob: Tensor,
    *,
    ground_ids: tuple[int, ...],
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    small_weight: float,
    large_weight: float,
    activation_margin: float,
    worst_contact_weight: float = 0.0,
    soft_margin_m: float = 0.0,
    soft_field_weight: float = 0.0,
    soft_worst_field_weight: float = 0.0,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    foot_xy = foot_pos[..., :2]
    terrain_z = height_at(terrain, foot_xy, cache=query_cache).to(dtype=foot_pos.dtype, device=foot_pos.device)
    near_terrain = foot_pos[..., 2] <= terrain_z + float(activation_margin)
    semantic = semantic_at(terrain, foot_xy, cache=query_cache)
    ground = _semantic_id_mask(semantic, ground_ids)
    small = torch.logical_and(_semantic_id_mask(semantic, small_ids), torch.logical_not(ground))
    large = torch.logical_and(_semantic_id_mask(semantic, large_ids), torch.logical_not(ground))
    obstacle_weight = (
        float(small_weight) * small.to(dtype=foot_pos.dtype, device=foot_pos.device)
        + float(large_weight) * large.to(dtype=foot_pos.dtype, device=foot_pos.device)
    )
    active = near_terrain.to(dtype=foot_pos.dtype, device=foot_pos.device) * obstacle_weight
    contact_sq = contact_prob.to(dtype=foot_pos.dtype, device=foot_pos.device).square()
    mean_loss = (active * contact_sq).sum(dim=(1, 2)) / torch.clamp(
        active.sum(dim=(1, 2)),
        min=1.0,
    )
    if float(worst_contact_weight) <= 0.0:
        out = mean_loss
    else:
        worst_contact = torch.where(active > 0.0, contact_sq, torch.zeros_like(contact_sq)).amax(dim=(1, 2))
        out = mean_loss + float(worst_contact_weight) * worst_contact
    if float(soft_field_weight) <= 0.0 and float(soft_worst_field_weight) <= 0.0:
        return out
    field = _semantic_obstacle_field(
        terrain,
        dtype=foot_pos.dtype,
        device=foot_pos.device,
        small_ids=small_ids,
        large_ids=large_ids,
        small_weight=small_weight,
        large_weight=large_weight,
        soft_margin_m=soft_margin_m,
    )
    if field is None:
        return out
    sampled = _sample_obstacle_field(terrain, field, foot_pos[..., :2]).to(dtype=foot_pos.dtype, device=foot_pos.device)
    near_weight = near_terrain.to(dtype=foot_pos.dtype, device=foot_pos.device)
    soft_cost = near_weight * sampled * contact_sq
    soft_mean = soft_cost.sum(dim=(1, 2)) / torch.clamp(near_weight.sum(dim=(1, 2)), min=1.0)
    soft_worst = torch.where(near_weight > 0.0, sampled * contact_sq, torch.zeros_like(sampled)).amax(dim=(1, 2))
    return out + float(soft_field_weight) * soft_mean + float(soft_worst_field_weight) * soft_worst


def semantic_obstacle_loss(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    contact_prob: Tensor,
    swing_prob: Tensor,
    *,
    small_weight: float,
    large_weight: float,
    body_weight: float,
    foot_weight: float,
    body_stencil_radius_m: float,
    soft_margin_m: float = 0.0,
    body_soft_field_weight: float = 0.0,
    body_soft_worst_field_weight: float = 0.0,
    foot_soft_field_weight: float = 0.0,
    foot_soft_worst_field_weight: float = 0.0,
    high_small_relative_height_m: float | None = None,
    query_cache: TerrainQueryCache | None = None,
) -> Tensor:
    root_xy = root_pos[..., :2]
    foot_xy = foot_pos[..., :2]
    root_ground = height_at(terrain, root_xy, cache=query_cache).to(dtype=root_pos.dtype, device=root_pos.device)
    foot_sem = semantic_at(terrain, foot_xy, cache=query_cache)
    terrain_z = height_at(terrain, foot_xy, cache=query_cache).to(dtype=foot_pos.dtype, device=foot_pos.device)
    foot_small_mask = foot_sem == 1
    if high_small_relative_height_m is not None:
        foot_small_mask = torch.logical_and(
            foot_small_mask,
            (terrain_z - root_ground[..., None].to(dtype=foot_pos.dtype, device=foot_pos.device))
            > float(high_small_relative_height_m),
        )
    foot_small = foot_small_mask.to(dtype=foot_pos.dtype, device=foot_pos.device)
    foot_large = (foot_sem >= 2).to(dtype=foot_pos.dtype, device=foot_pos.device)
    obstacle = float(small_weight) * foot_small + float(large_weight) * foot_large
    clearance = torch.relu(terrain_z + 0.04 - foot_pos[..., 2])
    contact_pen = contact_prob.to(dtype=foot_pos.dtype) * obstacle
    swing_pen = swing_prob.to(dtype=foot_pos.dtype) * obstacle * clearance.square()
    foot_pen = contact_pen + swing_pen

    radius = float(body_stencil_radius_m)
    offsets = torch.tensor(
        [[0.0, 0.0], [radius, 0.0], [-radius, 0.0], [0.0, radius], [0.0, -radius]],
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    yaw = root_rpy[..., 2]
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    ox = offsets[:, 0].view(1, 1, -1)
    oy = offsets[:, 1].view(1, 1, -1)
    body_xy = torch.stack((cy * ox - sy * oy, sy * ox + cy * oy), dim=-1) + root_pos[..., None, :2]
    body_sem = semantic_at(terrain, body_xy, cache=query_cache)
    body_small_mask = body_sem == 1
    if high_small_relative_height_m is not None:
        body_height = height_at(terrain, body_xy, cache=query_cache).to(dtype=root_pos.dtype, device=root_pos.device)
        body_small_mask = torch.logical_and(
            body_small_mask,
            (body_height - root_ground[..., None]) > float(high_small_relative_height_m),
        )
    body_small = body_small_mask.to(dtype=root_pos.dtype, device=root_pos.device)
    body_large = (body_sem >= 2).to(dtype=root_pos.dtype, device=root_pos.device)
    body_pen = float(small_weight) * body_small + float(large_weight) * body_large
    out = float(foot_weight) * foot_pen.mean(dim=(1, 2)) + float(body_weight) * body_pen.mean(dim=(1, 2))

    if (
        float(body_soft_field_weight) <= 0.0
        and float(body_soft_worst_field_weight) <= 0.0
        and float(foot_soft_field_weight) <= 0.0
        and float(foot_soft_worst_field_weight) <= 0.0
    ):
        return out
    small_mask_override = None
    if high_small_relative_height_m is not None and terrain.semantic_map is not None:
        semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=root_pos.device)
        if semantic.ndim == 2:
            semantic = semantic.unsqueeze(0)
        height = torch.as_tensor(terrain.height_map, dtype=root_pos.dtype, device=root_pos.device)
        nearby_height = _nearby_height_for_sparse_semantic(
            terrain,
            height,
            dtype=root_pos.dtype,
            device=root_pos.device,
        ).reshape(int(height.shape[0]), int(height.shape[1]), int(height.shape[2]))
        root0_xy = root_pos[:, :1, :2]
        root0_ground = height_at(terrain, root0_xy, cache=query_cache).reshape(root_pos.shape[0], 1, 1).to(
            dtype=root_pos.dtype,
            device=root_pos.device,
        )
        small_mask_override = torch.logical_and(
            _semantic_id_mask(semantic, (1,)),
            (nearby_height - root0_ground) > float(high_small_relative_height_m),
        )
    field = _semantic_obstacle_field(
        terrain,
        dtype=root_pos.dtype,
        device=root_pos.device,
        small_ids=(1,),
        large_ids=(2,),
        small_weight=small_weight,
        large_weight=large_weight,
        soft_margin_m=soft_margin_m,
        small_mask_override=small_mask_override,
    )
    if field is None:
        return out
    if float(body_soft_field_weight) > 0.0 or float(body_soft_worst_field_weight) > 0.0:
        body_soft = _sample_obstacle_field(terrain, field, body_xy).to(dtype=root_pos.dtype, device=root_pos.device)
        out = out + float(body_soft_field_weight) * body_soft.mean(dim=(1, 2))
        out = out + float(body_soft_worst_field_weight) * body_soft.amax(dim=(1, 2))
    if float(foot_soft_field_weight) > 0.0 or float(foot_soft_worst_field_weight) > 0.0:
        foot_soft = _sample_obstacle_field(terrain, field, foot_pos[..., :2]).to(dtype=foot_pos.dtype, device=foot_pos.device)
        stance_weight = contact_prob.to(dtype=foot_pos.dtype, device=foot_pos.device)
        swing_near = (swing_prob.to(dtype=foot_pos.dtype, device=foot_pos.device) * clearance.square()).detach()
        foot_soft_cost = foot_soft * (stance_weight.square() + swing_near)
        out = out + float(foot_soft_field_weight) * foot_soft_cost.mean(dim=(1, 2))
        out = out + float(foot_soft_worst_field_weight) * foot_soft_cost.amax(dim=(1, 2))
    return out


__all__ = [
    "body_heightfield_collision_loss",
    "finite_horizon_touchdown_phase",
    "high_obstacle_avoidance_loss",
    "knee_shank_heightfield_collision_loss",
    "low_small_crossing_progress_loss",
    "ObstacleRiskScales",
    "obstacle_risk_scales",
    "sample_time",
    "semantic_contact_avoidance_loss",
    "semantic_obstacle_loss",
    "stance_ground_loss",
    "stance_semantic_obstacle_loss",
    "swing_clearance_terrain_loss",
    "touchdown_semantic_loss",
    "touchdown_surface_loss",
]
