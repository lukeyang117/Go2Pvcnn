"""Near-field semantic body-part clearance reward."""

from __future__ import annotations

import torch


_CIRCLE_OFFSET_CACHE: dict[tuple[str, str, float, float, int], torch.Tensor] = {}


def _as_part_value(values, key: str, default: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if values is None:
        value = default
    elif isinstance(values, dict):
        value = values.get(key, default)
    else:
        value = getattr(values, key, default)
    return torch.as_tensor(float(value), dtype=dtype, device=device)


def _semantic_id_mask(semantic_values: torch.Tensor, small_semantic_ids) -> torch.Tensor:
    ids = torch.as_tensor(tuple(small_semantic_ids), dtype=semantic_values.dtype, device=semantic_values.device)
    if int(ids.numel()) == 0:
        return torch.zeros_like(semantic_values, dtype=torch.bool)
    return (semantic_values[..., None] == ids.view(*((1,) * semantic_values.ndim), -1)).any(dim=-1)


def _part_points_3d(raw_points, *, part_name: str, num_envs: int, dtype: torch.dtype, device: torch.device):
    points = torch.as_tensor(raw_points, dtype=dtype, device=device)
    if points.ndim == 4:
        points = points.reshape(num_envs, -1, 3)
    elif points.ndim != 3:
        raise ValueError(f"{part_name} points must be [N,P,3] or [N,L,S,3], got {tuple(points.shape)}")
    if int(points.shape[0]) != num_envs or int(points.shape[-1]) != 3:
        raise ValueError(f"{part_name} points must share map env dimension and end in 3")
    return points


def _terrain_resolution_xy(terrain) -> tuple[float, float]:
    height_map = torch.as_tensor(terrain.height_map)
    if height_map.ndim != 3:
        raise ValueError("terrain height map must be [N,H,W]")
    _, height, width = height_map.shape
    x0, x1 = terrain.world_x_range
    y0, y1 = terrain.world_y_range
    x_resolution = abs(float(x1) - float(x0)) / max(int(width) - 1, 1)
    y_resolution = abs(float(y1) - float(y0)) / max(int(height) - 1, 1)
    return x_resolution, y_resolution


def _cached_circle_offsets(
    *,
    radius_m: float,
    resolution_m: float,
    device: torch.device,
    dtype: torch.dtype,
    max_offsets: int | None = None,
) -> torch.Tensor:
    """Return fixed, cached xy offsets inside a query disk."""

    radius = float(radius_m)
    resolution = max(float(resolution_m), 1.0e-6)
    max_count = None if max_offsets is None else int(max_offsets)
    key = (str(device), str(dtype), round(radius, 6), round(resolution, 6), -1 if max_count is None else max_count)
    cached = _CIRCLE_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached

    steps = max(1, int(torch.ceil(torch.tensor(radius / resolution)).item()))
    coords = torch.arange(-steps, steps + 1, dtype=dtype, device=device) * resolution
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    offsets = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    keep = torch.linalg.vector_norm(offsets, dim=-1) <= radius + 1.0e-6
    offsets = offsets[keep]
    distance = torch.linalg.vector_norm(offsets, dim=-1)
    angle = torch.atan2(offsets[:, 1], offsets[:, 0])
    order = torch.argsort(distance + angle.abs() * 1.0e-6)
    offsets = offsets.index_select(0, order)
    if max_count is not None and int(offsets.shape[0]) > max_count:
        sample_ids = torch.linspace(0, int(offsets.shape[0]) - 1, max_count, dtype=torch.float32, device=device).round().long()
        offsets = offsets.index_select(0, sample_ids)
    _CIRCLE_OFFSET_CACHE[key] = offsets.contiguous()
    return _CIRCLE_OFFSET_CACHE[key]


def _expand_centers_with_offsets(
    centers: torch.Tensor,
    *,
    surface_z: torch.Tensor,
    offsets_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_centers = centers.reshape(centers.shape[0], -1, 3)
    flat_surface_z = surface_z.reshape(surface_z.shape[0], -1)
    query_xy = flat_centers[..., :2].unsqueeze(2) + offsets_xy.view(1, 1, -1, 2)
    expanded_surface_z = flat_surface_z.unsqueeze(-1).expand(-1, -1, int(offsets_xy.shape[0]))
    return query_xy.reshape(centers.shape[0], -1, 2), expanded_surface_z.reshape(centers.shape[0], -1)


def _base_footprint_centers(
    *,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    half_extents_m: tuple[float, float, float],
    footprint_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    from extension.convention import extract_yaw_batch

    half_x, half_y, half_z = [float(v) for v in half_extents_m]
    grid_x, grid_y = [int(v) for v in footprint_grid]
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError(f"base footprint grid must be positive, got {(grid_x, grid_y)}")
    xs = torch.linspace(-half_x, half_x, grid_x, dtype=root_pos_w.dtype, device=root_pos_w.device)
    ys = torch.linspace(-half_y, half_y, grid_y, dtype=root_pos_w.dtype, device=root_pos_w.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    local = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    yaw = extract_yaw_batch(root_quat_w)
    cos_yaw = torch.cos(yaw).view(-1, 1)
    sin_yaw = torch.sin(yaw).view(-1, 1)
    local_x = local[:, 0].view(1, -1)
    local_y = local[:, 1].view(1, -1)
    world_x = root_pos_w[:, 0:1] + local_x * cos_yaw - local_y * sin_yaw
    world_y = root_pos_w[:, 1:2] + local_x * sin_yaw + local_y * cos_yaw
    world_z = root_pos_w[:, 2:3].expand_as(world_x)
    centers = torch.stack((world_x, world_y, world_z), dim=-1)
    surface_z = centers[..., 2] - half_z
    return centers, surface_z


def _body_geometry_query_points(
    *,
    centers: torch.Tensor,
    surface_z: torch.Tensor,
    query_radius_m: float,
    terrain,
    max_offsets: int | None = 81,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand primitive centers into scanner-map query xy and surface z tensors."""

    x_resolution, y_resolution = _terrain_resolution_xy(terrain)
    offsets = _cached_circle_offsets(
        radius_m=float(query_radius_m),
        resolution_m=min(x_resolution, y_resolution),
        device=centers.device,
        dtype=centers.dtype,
        max_offsets=int(max_offsets),
    )
    return _expand_centers_with_offsets(centers, surface_z=surface_z, offsets_xy=offsets)


def _geometry_group_penalty(
    *,
    terrain,
    centers: torch.Tensor,
    surface_z: torch.Tensor,
    query_radius_m: float,
    margin_m: float,
    small_semantic_ids,
    cache,
) -> torch.Tensor:
    from extension.batch_mpc_planner.terrain import height_at, semantic_at

    query_xy, query_surface_z = _body_geometry_query_points(
        centers=centers,
        surface_z=surface_z,
        query_radius_m=float(query_radius_m),
        terrain=terrain,
    )
    terrain_z = height_at(terrain, query_xy, cache=cache).to(dtype=centers.dtype, device=centers.device)
    semantic_id = semantic_at(terrain, query_xy, cache=cache)
    small_mask = _semantic_id_mask(semantic_id.to(dtype=torch.long), small_semantic_ids).to(dtype=centers.dtype)
    deficit = torch.relu(terrain_z + float(margin_m) - query_surface_z)
    return (small_mask * deficit.square()).mean(dim=1)


def _semantic_geometry_clearance_penalty(
    *,
    terrain,
    centers_by_part,
    root_pos_w,
    root_quat_w,
    small_semantic_ids=(1,),
    foot_radius_m=0.022,
    foot_query_radius_m=0.035,
    foot_margin_m=0.015,
    foot_weight=0.5,
    calf_radius_m=0.040,
    calf_query_radius_m=0.045,
    calf_margin_m=0.040,
    calf_weight=2.0,
    thigh_radius_m=0.040,
    thigh_query_radius_m=0.045,
    thigh_margin_m=0.040,
    thigh_weight=1.5,
    include_base=False,
    base_half_extents_m=(0.20, 0.06, 0.07),
    base_footprint_grid=(5, 3),
    base_query_radius_m=0.030,
    base_margin_m=0.020,
    base_weight=1.0,
    penalty_clip=1.0,
):
    """Return negative per-env reward from fitted body-geometry map neighborhoods."""

    from extension.batch_mpc_planner.terrain import TerrainQueryCache

    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    if height_map.ndim != 3:
        raise ValueError("terrain height map must be [N,H,W]")
    device = height_map.device
    dtype = height_map.dtype
    num_envs = int(height_map.shape[0])
    total = torch.zeros(num_envs, dtype=dtype, device=device)
    cache = TerrainQueryCache()

    if "foot" in centers_by_part:
        centers = _part_points_3d(centers_by_part["foot"], part_name="foot", num_envs=num_envs, dtype=dtype, device=device)
        surface_z = centers[..., 2] - float(foot_radius_m)
        total = total + float(foot_weight) * _geometry_group_penalty(
            terrain=terrain,
            centers=centers,
            surface_z=surface_z,
            query_radius_m=foot_query_radius_m,
            margin_m=foot_margin_m,
            small_semantic_ids=small_semantic_ids,
            cache=cache,
        )

    if "calf" in centers_by_part or "shank" in centers_by_part:
        calf_points = centers_by_part["calf"] if "calf" in centers_by_part else centers_by_part["shank"]
        centers = _part_points_3d(calf_points, part_name="calf", num_envs=num_envs, dtype=dtype, device=device)
        surface_z = centers[..., 2] - float(calf_radius_m)
        total = total + float(calf_weight) * _geometry_group_penalty(
            terrain=terrain,
            centers=centers,
            surface_z=surface_z,
            query_radius_m=calf_query_radius_m,
            margin_m=calf_margin_m,
            small_semantic_ids=small_semantic_ids,
            cache=cache,
        )

    if "thigh" in centers_by_part:
        centers = _part_points_3d(centers_by_part["thigh"], part_name="thigh", num_envs=num_envs, dtype=dtype, device=device)
        surface_z = centers[..., 2] - float(thigh_radius_m)
        total = total + float(thigh_weight) * _geometry_group_penalty(
            terrain=terrain,
            centers=centers,
            surface_z=surface_z,
            query_radius_m=thigh_query_radius_m,
            margin_m=thigh_margin_m,
            small_semantic_ids=small_semantic_ids,
            cache=cache,
        )

    if include_base:
        root_pos = torch.as_tensor(root_pos_w, dtype=dtype, device=device)
        root_quat = torch.as_tensor(root_quat_w, dtype=dtype, device=device)
        centers, surface_z = _base_footprint_centers(
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            half_extents_m=tuple(base_half_extents_m),
            footprint_grid=tuple(base_footprint_grid),
        )
        total = total + float(base_weight) * _geometry_group_penalty(
            terrain=terrain,
            centers=centers,
            surface_z=surface_z,
            query_radius_m=base_query_radius_m,
            margin_m=base_margin_m,
            small_semantic_ids=small_semantic_ids,
            cache=cache,
        )

    return -torch.clamp(total, min=0.0, max=float(penalty_clip))


def _semantic_clearance_penalty_from_points(
    *,
    terrain,
    points_by_part,
    small_semantic_ids=(1,),
    margins=None,
    weights=None,
    penalty_clip=1.0,
    foot_contact_mask=None,
    stance_foot_weight=0.5,
    swing_foot_weight=1.0,
):
    """Return negative per-env clearance reward for current body-part sample points."""

    from extension.batch_mpc_planner.terrain import TerrainQueryCache, height_at, semantic_at

    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    if height_map.ndim != 3:
        raise ValueError("terrain height map must be [N,H,W]")
    device = height_map.device
    dtype = height_map.dtype
    num_envs = int(height_map.shape[0])
    point_blocks = []
    margin_blocks = []
    reduction_weight_blocks = []
    foot_contact_scale = None
    point_offset = 0
    for part_name, raw_points in points_by_part.items():
        points = _part_points_3d(raw_points, part_name=part_name, num_envs=num_envs, dtype=dtype, device=device)
        point_count = int(points.shape[1])
        if point_count == 0:
            continue
        point_blocks.append(points)
        margin = _as_part_value(margins, part_name, 0.0, device=device, dtype=dtype)
        weight = _as_part_value(weights, part_name, 1.0, device=device, dtype=dtype)
        margin_blocks.append(margin.expand(point_count))
        reduction_weight_blocks.append((weight / float(point_count)).expand(point_count))
        if part_name == "foot" and foot_contact_mask is not None:
            contact = torch.as_tensor(foot_contact_mask, dtype=torch.bool, device=device).reshape(num_envs, -1)
            if int(contact.shape[1]) == point_count:
                scale = torch.ones((num_envs, sum(int(block.shape[1]) for block in point_blocks)), dtype=dtype, device=device)
                if foot_contact_scale is not None:
                    scale[:, : foot_contact_scale.shape[1]] = foot_contact_scale
                scale[:, point_offset : point_offset + point_count] = torch.where(
                    contact,
                    torch.as_tensor(float(stance_foot_weight), dtype=dtype, device=device),
                    torch.as_tensor(float(swing_foot_weight), dtype=dtype, device=device),
                )
                foot_contact_scale = scale
        point_offset += point_count

    if not point_blocks:
        return torch.zeros(num_envs, dtype=dtype, device=device)

    points = torch.cat(point_blocks, dim=1)
    margins_per_point = torch.cat(margin_blocks, dim=0).view(1, -1)
    reduction_weights = torch.cat(reduction_weight_blocks, dim=0).view(1, -1)
    if foot_contact_scale is not None and int(foot_contact_scale.shape[1]) != int(points.shape[1]):
        expanded = torch.ones((num_envs, int(points.shape[1])), dtype=dtype, device=device)
        expanded[:, : foot_contact_scale.shape[1]] = foot_contact_scale
        foot_contact_scale = expanded

    query_cache = TerrainQueryCache()
    terrain_z = height_at(terrain, points[..., :2], cache=query_cache).to(dtype=dtype, device=device)
    semantic_id = semantic_at(terrain, points[..., :2], cache=query_cache)
    small_mask = _semantic_id_mask(semantic_id.to(dtype=torch.long), small_semantic_ids).to(dtype=dtype)
    deficit = torch.relu(terrain_z + margins_per_point - points[..., 2])
    point_cost = small_mask * deficit.square()
    if foot_contact_scale is not None:
        point_cost = point_cost * foot_contact_scale
    total = (point_cost * reduction_weights).sum(dim=1)

    return -torch.clamp(total, min=0.0, max=float(penalty_clip))


def semantic_foot_over_clearance_bonus_from_tensors(
    *,
    terrain,
    foot_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    command: torch.Tensor,
    small_semantic_ids=(1,),
    corridor_width_m=0.42,
    lookahead_m=1.6,
    low_small_max_height_m=0.30,
    obstacle_half_extent_m=0.18,
    clearance_margin_m=0.05,
    bonus_clip=1.0,
) -> torch.Tensor:
    """Reward feet that pass clearly above low-small semantic cells on the commanded path."""

    from extension.batch_mpc_planner.terrain import height_at, semantic_at
    from extension.convention import extract_yaw_batch

    foot_pos = torch.as_tensor(foot_pos_w, dtype=torch.float32)
    device = foot_pos.device
    root_pos = torch.as_tensor(root_pos_w, dtype=torch.float32, device=device)
    root_quat = torch.as_tensor(root_quat_w, dtype=torch.float32, device=device)
    command_t = torch.as_tensor(command, dtype=torch.float32, device=device)
    if foot_pos.ndim != 3 or int(foot_pos.shape[-1]) != 3:
        raise ValueError(f"foot_pos_w must be [N,F,3], got {tuple(foot_pos.shape)}")
    if command_t.ndim != 2 or int(command_t.shape[0]) != int(foot_pos.shape[0]) or int(command_t.shape[1]) < 2:
        raise ValueError("command must be [N,>=2] and match foot env dimension")

    heading = command_t[:, :2]
    speed = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    yaw = extract_yaw_batch(root_quat)
    yaw_heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    heading = torch.where((speed > 1.0e-4).expand_as(heading), heading / speed.clamp_min(1.0e-6), yaw_heading)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)

    foot_delta = foot_pos[..., :2] - root_pos[:, None, :2]
    foot_along = (foot_delta * heading[:, None, :]).sum(dim=-1)
    foot_lateral = (foot_delta * left[:, None, :]).sum(dim=-1)
    foot_semantic = semantic_at(terrain, foot_pos[..., :2]).to(device=device)
    foot_ground = height_at(terrain, foot_pos[..., :2]).to(dtype=foot_pos.dtype, device=device)
    root_ground = height_at(terrain, root_pos[:, None, :2]).reshape(-1, 1).to(dtype=foot_pos.dtype, device=device)
    small_mask = _semantic_id_mask(foot_semantic.to(dtype=torch.long), small_semantic_ids)
    obstacle_height = foot_ground - root_ground
    in_path = (
        small_mask
        & (foot_along > 0.0)
        & (foot_along < float(lookahead_m))
        & (torch.abs(foot_lateral) < 0.5 * float(corridor_width_m) + float(obstacle_half_extent_m))
        & (obstacle_height > 0.015)
        & (obstacle_height <= float(low_small_max_height_m))
    )
    clearance = foot_pos[..., 2] - foot_ground
    over_score = torch.relu(clearance - float(clearance_margin_m))
    return torch.where(in_path, over_score, torch.zeros_like(over_score)).amax(dim=1).clamp(0.0, float(bonus_clip))


def _body_id_tensor(body_ids, *, key: str, device: torch.device) -> torch.Tensor:
    if key not in body_ids:
        raise KeyError(f"body_ids missing required key: {key}")
    ids = torch.as_tensor(body_ids[key], dtype=torch.long, device=device).reshape(-1)
    if int(ids.numel()) != 4:
        raise ValueError(f"body_ids[{key!r}] must contain 4 ids, got {int(ids.numel())}")
    return ids


def _segment_samples(start: torch.Tensor, end: torch.Tensor, sample_count: int) -> torch.Tensor:
    count = int(sample_count)
    if count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    alpha = torch.linspace(0.0, 1.0, count + 2, dtype=start.dtype, device=start.device)[1:-1]
    shape = (1,) * (start.ndim - 1) + (count, 1)
    alpha = alpha.view(shape)
    return start.unsqueeze(-2) * (1.0 - alpha) + end.unsqueeze(-2) * alpha


def _current_body_part_sample_points(
    robot,
    *,
    body_ids,
    calf_sections=7,
    thigh_sections=7,
    shank_sample_count=None,
    thigh_sample_count=None,
):
    """Build fixed-shape foot, calf, and thigh section centers from current body poses."""

    body_pos_w = torch.as_tensor(robot.data.body_pos_w)
    if body_pos_w.ndim != 3 or int(body_pos_w.shape[-1]) != 3:
        raise ValueError(f"robot.data.body_pos_w must be [N,B,3], got {tuple(body_pos_w.shape)}")
    device = body_pos_w.device
    foot_ids = _body_id_tensor(body_ids, key="foot", device=device)
    calf_ids = _body_id_tensor(body_ids, key="calf", device=device)
    thigh_ids = _body_id_tensor(body_ids, key="thigh", device=device)
    if shank_sample_count is not None:
        calf_sections = int(shank_sample_count)
    if thigh_sample_count is not None:
        thigh_sections = int(thigh_sample_count)

    foot = body_pos_w.index_select(1, foot_ids)
    calf = body_pos_w.index_select(1, calf_ids)
    thigh = body_pos_w.index_select(1, thigh_ids)
    calf_samples = _segment_samples(calf, foot, int(calf_sections))
    return {
        "foot": foot.unsqueeze(2),
        "calf": calf_samples,
        "shank": calf_samples,
        "thigh": _segment_samples(thigh, calf, int(thigh_sections)),
    }


def _scanner_value(scanner, name: str):
    data = getattr(scanner, "_data", None)
    if data is not None and hasattr(data, name):
        return getattr(data, name)
    if hasattr(scanner, name):
        return getattr(scanner, name)
    data = getattr(scanner, "data", None)
    if data is not None and hasattr(data, name):
        return getattr(data, name)
    return None


def _scanner_map(scanner, *names: str) -> torch.Tensor:
    for name in names:
        value = _scanner_value(scanner, name)
        if value is not None:
            out = torch.as_tensor(value)
            if out.ndim == 2:
                out = out.unsqueeze(0)
            if out.ndim == 3:
                return out
    raise AttributeError(f"scanner does not expose any of: {', '.join(names)}")


def _current_scanner_terrain(scanner, *, device):
    from extension.batch_mpc_planner.terrain import MpcPlannerTerrain
    from extension.convention import extract_yaw_batch

    elevation = _scanner_map(scanner, "elevation_map", "height_map").to(dtype=torch.float32, device=device)
    semantic = _scanner_map(scanner, "semantic_map").to(dtype=torch.long, device=device)
    if tuple(semantic.shape) != tuple(elevation.shape):
        raise ValueError(
            "scanner elevation/semantic maps must share shape, "
            f"got {tuple(elevation.shape)} and {tuple(semantic.shape)}"
        )
    pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
    size = getattr(pattern_cfg, "size", (1.5, 1.5))
    half_x = 0.5 * float(size[0])
    half_y = 0.5 * float(size[1])
    scanner_data = getattr(scanner, "data", None)
    sensor_pos = getattr(scanner_data, "pos_w", None)
    sensor_quat = getattr(scanner_data, "quat_w", None)
    return MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-half_x, half_x),
        world_y_range=(-half_y, half_y),
        sensor_pos_w=None
        if sensor_pos is None
        else torch.as_tensor(sensor_pos, dtype=torch.float32, device=device).contiguous(),
        sensor_yaw=None
        if sensor_quat is None
        else extract_yaw_batch(torch.as_tensor(sensor_quat, dtype=torch.float32, device=device)).contiguous(),
    )


def semantic_foot_over_clearance_bonus(
    env,
    *,
    asset_cfg,
    scanner_cfg,
    command_name="base_velocity",
    small_semantic_ids=(1,),
    corridor_width_m=0.42,
    lookahead_m=1.6,
    low_small_max_height_m=0.30,
    obstacle_half_extent_m=0.18,
    clearance_margin_m=0.05,
    bonus_clip=1.0,
    bonus_scale=1.0,
):
    """Return positive reward when feet clear low-small cells on the commanded path."""

    robot = env.scene[asset_cfg.name]
    scanner = env.scene[scanner_cfg.name]
    root = getattr(env, "unwrapped", env)
    foot_ids = getattr(root, "_semantic_foot_over_body_ids", None)
    if foot_ids is None:
        foot_ids, _ = robot.find_bodies(".*_foot")
        root._semantic_foot_over_body_ids = foot_ids
    ids = torch.as_tensor(foot_ids, dtype=torch.long, device=torch.as_tensor(robot.data.body_pos_w).device)
    foot_pos = torch.as_tensor(robot.data.body_pos_w, dtype=torch.float32, device=ids.device).index_select(1, ids)
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        command = torch.zeros((foot_pos.shape[0], 3), dtype=foot_pos.dtype, device=foot_pos.device)
    else:
        command = torch.as_tensor(command_manager.get_command(command_name), dtype=foot_pos.dtype, device=foot_pos.device)
    terrain = _current_scanner_terrain(scanner, device=foot_pos.device)
    return 1000*float(bonus_scale) * semantic_foot_over_clearance_bonus_from_tensors(
        terrain=terrain,
        foot_pos_w=foot_pos,
        root_pos_w=robot.data.root_pos_w,
        root_quat_w=robot.data.root_quat_w,
        command=command,
        small_semantic_ids=small_semantic_ids,
        corridor_width_m=corridor_width_m,
        lookahead_m=lookahead_m,
        low_small_max_height_m=low_small_max_height_m,
        obstacle_half_extent_m=obstacle_half_extent_m,
        clearance_margin_m=clearance_margin_m,
        bonus_clip=bonus_clip,
    )


def semantic_body_part_clearance_reward(
    env,
    *,
    asset_cfg,
    scanner_cfg,
    contact_sensor_cfg=None,
    small_semantic_ids=(1,),
    foot_margin_m=0.015,
    calf_margin_m=0.04,
    thigh_margin_m=0.04,
    base_margin_m=0.02,
    foot_weight=0.5,
    calf_weight=2.0,
    thigh_weight=1.5,
    base_weight=1.0,
    foot_sphere_radius_m=0.022,
    foot_query_radius_m=0.035,
    calf_capsule_radius_m=0.040,
    calf_query_radius_m=0.045,
    calf_sections=7,
    thigh_capsule_radius_m=0.040,
    thigh_query_radius_m=0.045,
    thigh_sections=7,
    base_half_extents_m=(0.20, 0.06, 0.07),
    base_footprint_grid=(5, 3),
    base_query_radius_m=0.030,
    include_base=True,
    stance_foot_weight=0.5,
    swing_foot_weight=1.0,
    contact_force_threshold=1.0,
    shank_margin_m=None,
    shank_weight=None,
    shank_sample_count=None,
    thigh_sample_count=None,
    penalty_clip=1.0,
    clearance_scale=1.0,
):
    """Return a per-env clearance reward."""

    robot = env.scene[asset_cfg.name]
    scanner = env.scene[scanner_cfg.name]
    root = getattr(env, "unwrapped", env)
    body_ids = getattr(root, "_semantic_body_part_clearance_body_ids", None)
    if body_ids is None:
        foot_ids, _ = robot.find_bodies(".*_foot")
        calf_ids, _ = robot.find_bodies(".*_calf")
        thigh_ids, _ = robot.find_bodies(".*_thigh")
        body_ids = {"foot": foot_ids, "calf": calf_ids, "thigh": thigh_ids}
        root._semantic_body_part_clearance_body_ids = body_ids

    terrain = _current_scanner_terrain(scanner, device=torch.as_tensor(robot.data.body_pos_w).device)
    points = _current_body_part_sample_points(
        robot,
        body_ids=body_ids,
        calf_sections=calf_sections,
        thigh_sections=thigh_sections,
        shank_sample_count=shank_sample_count,
        thigh_sample_count=thigh_sample_count,
    )
    _foot_contact_mask = None
    if contact_sensor_cfg is not None:
        contact_sensor = env.scene[contact_sensor_cfg.name]
        forces = getattr(getattr(contact_sensor, "data", None), "net_forces_w", None)
        if forces is not None:
            force_norm = torch.linalg.vector_norm(torch.as_tensor(forces), dim=-1)
            _foot_contact_mask = force_norm > float(contact_force_threshold)

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part=points,
        root_pos_w=robot.data.root_pos_w,
        root_quat_w=robot.data.root_quat_w,
        small_semantic_ids=small_semantic_ids,
        foot_radius_m=foot_sphere_radius_m,
        foot_query_radius_m=foot_query_radius_m,
        foot_margin_m=foot_margin_m,
        foot_weight=foot_weight,
        calf_radius_m=calf_capsule_radius_m,
        calf_query_radius_m=calf_query_radius_m,
        calf_margin_m=calf_margin_m if shank_margin_m is None else shank_margin_m,
        calf_weight=calf_weight if shank_weight is None else shank_weight,
        thigh_radius_m=thigh_capsule_radius_m,
        thigh_query_radius_m=thigh_query_radius_m,
        thigh_margin_m=thigh_margin_m,
        thigh_weight=thigh_weight,
        include_base=include_base,
        base_half_extents_m=base_half_extents_m,
        base_footprint_grid=base_footprint_grid,
        base_query_radius_m=base_query_radius_m,
        base_margin_m=base_margin_m,
        base_weight=base_weight,
        penalty_clip=penalty_clip,
    )
    return reward * float(clearance_scale)
