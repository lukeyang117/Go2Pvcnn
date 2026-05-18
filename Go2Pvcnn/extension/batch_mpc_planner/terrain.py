"""Terrain helpers for batch MPC planner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import MpcPlannerTerrain


@dataclass
class TerrainQueryCache:
    """Per-loss-call cache for repeated height/semantic grid samples."""

    height: dict[tuple[int, str], Tensor] = field(default_factory=dict)
    semantic: dict[int, Tensor] = field(default_factory=dict)


def _query_cache_key(points_xy: Tensor, mode: str | None = None) -> tuple:
    points = torch.as_tensor(points_xy)
    base = (
        int(points.data_ptr()),
        tuple(points.shape),
        tuple(points.stride()),
        str(points.device),
        str(points.dtype),
    )
    return (*base, str(mode)) if mode is not None else base


def _safe_norm(value: Tensor, *, dim: int, eps: float = 1.0e-12) -> Tensor:
    return torch.sqrt(torch.sum(value.square(), dim=dim) + float(eps))


def _reshape_ray_hits(ray_hits_w: Tensor) -> Tensor:
    hits = torch.as_tensor(ray_hits_w)
    if hits.ndim == 2 and int(hits.shape[-1]) == 3:
        hits = hits.unsqueeze(0)
    if hits.ndim == 4 and int(hits.shape[-1]) == 3:
        return hits
    if hits.ndim == 3 and int(hits.shape[-1]) == 3:
        # [H, W, 3] (single env grid)
        if int(hits.shape[0]) > 1 and int(hits.shape[0]) == int(hits.shape[1]):
            return hits.unsqueeze(0)
        # [B, H*W, 3] (flattened scanner grid per env)
        ray_count = int(hits.shape[1])
        side = int(round(math.sqrt(ray_count)))
        if side * side != ray_count:
            raise ValueError(
                "ray_hits_w with shape [B, N, 3] requires square N=H*W, "
                f"got N={ray_count} for shape {tuple(hits.shape)}"
            )
        return hits.reshape(int(hits.shape[0]), side, side, 3)
    raise ValueError(
        "ray_hits_w must be one of [B,H,W,3], [B,H*W,3], [H,W,3], or [H*W,3], "
        f"got {tuple(hits.shape)}"
    )


def _reshape_semantic_map(
    semantic_map: Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor | None:
    if semantic_map is None:
        return None
    sem = torch.as_tensor(semantic_map, device=device)
    if sem.ndim == 1:
        if batch != 1 or int(sem.numel()) != height * width:
            raise ValueError(
                "semantic_map [H*W] requires single-env input and matching H*W; "
                f"got batch={batch}, semantic_map={tuple(sem.shape)}, target={(height, width)}"
            )
        sem = sem.reshape(1, height, width)
    elif sem.ndim == 2:
        if tuple(sem.shape) == (height, width):
            sem = sem.unsqueeze(0).expand(batch, -1, -1)
        elif tuple(sem.shape) == (batch, height * width):
            sem = sem.reshape(batch, height, width)
        else:
            raise ValueError(
                "semantic_map [2D] must be [H,W] or [B,H*W]; "
                f"got {tuple(sem.shape)} for target batch/grid {(batch, height, width)}"
            )
    elif sem.ndim == 3:
        if tuple(sem.shape) == (1, height, width) and batch > 1:
            sem = sem.expand(batch, -1, -1)
        elif tuple(sem.shape) != (batch, height, width):
            raise ValueError(
                "semantic_map [3D] must match [B,H,W]; "
                f"got {tuple(sem.shape)} for target {(batch, height, width)}"
            )
    else:
        raise ValueError(
            "semantic_map must be one of [B,H,W], [H,W], [B,H*W], or [H*W], "
            f"got {tuple(sem.shape)}"
        )
    if torch.is_floating_point(sem):
        sem = torch.nan_to_num(sem, nan=0.0, posinf=0.0, neginf=0.0)
    return sem.to(dtype=torch.long).contiguous()


def build_mpc_terrain_from_scanner(
    ray_hits_w: Tensor,
    *,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    semantic_map: Tensor | None = None,
    sensor_pos_w: Tensor | None = None,
    sensor_yaw: Tensor | None = None,
) -> MpcPlannerTerrain:
    hits = torch.nan_to_num(_reshape_ray_hits(ray_hits_w), nan=0.0, posinf=0.0, neginf=0.0)
    height_map = hits[..., 2].to(dtype=torch.float32).contiguous()
    sem = _reshape_semantic_map(
        semantic_map,
        batch=int(height_map.shape[0]),
        height=int(height_map.shape[1]),
        width=int(height_map.shape[2]),
        device=height_map.device,
    )
    return MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=sem,
        world_x_range=world_x_range,
        world_y_range=world_y_range,
        sensor_pos_w=None if sensor_pos_w is None else torch.as_tensor(sensor_pos_w, dtype=torch.float32, device=height_map.device).contiguous(),
        sensor_yaw=None if sensor_yaw is None else torch.as_tensor(sensor_yaw, dtype=torch.float32, device=height_map.device).contiguous(),
    )


def subset_mpc_terrain(terrain: MpcPlannerTerrain, env_ids: Tensor) -> MpcPlannerTerrain:
    if terrain.height_map.ndim != 3:
        raise ValueError(f"terrain.height_map must be [B,H,W], got {tuple(terrain.height_map.shape)}")
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=terrain.height_map.device).reshape(-1)
    batch = int(terrain.height_map.shape[0])
    if int(ids.numel()) > 0:
        valid = torch.logical_and(ids >= 0, ids < batch)
        if not bool(torch.all(valid)):
            bad = ids[torch.logical_not(valid)]
            raise IndexError(
                f"env_ids out of bounds for terrain batch={batch}; "
                f"first bad ids={bad[:8].tolist()}"
            )
    height = terrain.height_map.index_select(0, ids)
    sem = terrain.semantic_map.index_select(0, ids) if terrain.semantic_map is not None else None
    return MpcPlannerTerrain(
        height_map=height,
        semantic_map=sem,
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=terrain.sensor_pos_w.index_select(0, ids) if terrain.sensor_pos_w is not None else None,
        sensor_yaw=terrain.sensor_yaw.index_select(0, ids) if terrain.sensor_yaw is not None else None,
    )


def _batched_query_xy(terrain: MpcPlannerTerrain, points_xy: Tensor) -> tuple[Tensor, tuple[int, ...], bool]:
    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    device = height_map.device
    points = torch.as_tensor(points_xy, dtype=torch.float32, device=device)
    if points.shape[-1] != 2:
        raise ValueError(f"points_xy must end in 2, got {tuple(points.shape)}")
    single_batch = False
    if points.ndim == 2:
        points = points.unsqueeze(0)
        single_batch = True
    original_shape = tuple(points.shape[:-1])
    batch = int(height_map.shape[0]) if height_map.ndim == 3 else 1
    if int(points.shape[0]) == 1 and batch > 1:
        points = points.expand(batch, *points.shape[1:])
        original_shape = tuple(points.shape[:-1])
        single_batch = False
    if int(points.shape[0]) != batch:
        raise ValueError(f"points batch {int(points.shape[0])} must match terrain batch {batch}")
    return points.reshape(batch, -1, 2), original_shape, single_batch


def _world_to_grid(terrain: MpcPlannerTerrain, points_xy: Tensor) -> Tensor:
    x0, x1 = terrain.world_x_range
    y0, y1 = terrain.world_y_range
    if terrain.sensor_pos_w is not None:
        sensor_pos = torch.as_tensor(terrain.sensor_pos_w, dtype=points_xy.dtype, device=points_xy.device)
        if sensor_pos.ndim == 1:
            sensor_pos = sensor_pos.view(1, -1)
        sensor_xy = sensor_pos[:, None, :2]
        delta = points_xy - sensor_xy
        if terrain.sensor_yaw is None:
            yaw = torch.zeros((points_xy.shape[0],), dtype=points_xy.dtype, device=points_xy.device)
        else:
            yaw = torch.as_tensor(terrain.sensor_yaw, dtype=points_xy.dtype, device=points_xy.device).reshape(-1)
        cy = torch.cos(yaw).view(-1, 1)
        sy = torch.sin(yaw).view(-1, 1)
        query_xy = torch.stack(
            (cy * delta[..., 0] + sy * delta[..., 1], -sy * delta[..., 0] + cy * delta[..., 1]),
            dim=-1,
        )
    else:
        query_xy = points_xy
    xs = query_xy[..., 0].clamp(float(x0), float(x1))
    ys = query_xy[..., 1].clamp(float(y0), float(y1))
    x_norm = (xs - float(x0)) / max(float(x1) - float(x0), 1.0e-6) * 2.0 - 1.0
    y_norm = (ys - float(y0)) / max(float(y1) - float(y0), 1.0e-6) * 2.0 - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def height_at(
    terrain: MpcPlannerTerrain,
    points_xy: Tensor,
    mode: str = "bilinear",
    *,
    cache: TerrainQueryCache | None = None,
) -> Tensor:
    """Sample terrain height at world-frame xy points."""
    if cache is not None:
        key = _query_cache_key(points_xy, mode)
        cached = cache.height.get(key)
        if cached is not None:
            return cached
    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    if height_map.ndim == 2:
        height_map = height_map.unsqueeze(0)
    points, original_shape, single_batch = _batched_query_xy(terrain, points_xy)
    grid = _world_to_grid(terrain, points).unsqueeze(2)
    sampled = F.grid_sample(
        height_map.unsqueeze(1),
        grid,
        mode=str(mode),
        align_corners=True,
        padding_mode="border",
    )
    out = sampled[:, 0, :, 0].reshape(original_shape)
    out = out.squeeze(0) if single_batch else out
    if cache is not None:
        cache.height[_query_cache_key(points_xy, mode)] = out
    return out


def semantic_at(
    terrain: MpcPlannerTerrain,
    points_xy: Tensor,
    *,
    cache: TerrainQueryCache | None = None,
) -> Tensor:
    """Nearest-neighbor sample semantic ids at world-frame xy points."""
    if cache is not None:
        key = _query_cache_key(points_xy)
        cached = cache.semantic.get(key)
        if cached is not None:
            return cached
    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    if height_map.ndim == 2:
        height_map = height_map.unsqueeze(0)
    points, original_shape, single_batch = _batched_query_xy(terrain, points_xy)
    if terrain.semantic_map is None:
        out = torch.zeros(original_shape, dtype=torch.long, device=points.device)
        out = out.squeeze(0) if single_batch else out
        if cache is not None:
            cache.semantic[_query_cache_key(points_xy)] = out
        return out
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.float32, device=points.device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    grid = _world_to_grid(terrain, points).unsqueeze(2)
    sampled = F.grid_sample(
        semantic.unsqueeze(1),
        grid,
        mode="nearest",
        align_corners=True,
        padding_mode="border",
    )
    out = sampled[:, 0, :, 0].round().to(dtype=torch.long).reshape(original_shape)
    out = out.squeeze(0) if single_batch else out
    if cache is not None:
        cache.semantic[_query_cache_key(points_xy)] = out
    return out


def slope_at(
    terrain: MpcPlannerTerrain,
    points_xy: Tensor,
    sample_step: float = 0.03,
    *,
    cache: TerrainQueryCache | None = None,
) -> Tensor:
    """Estimate terrain slope magnitude by finite differences."""
    points = torch.as_tensor(points_xy, dtype=torch.float32, device=terrain.height_map.device)
    step = float(sample_step)
    dx = torch.tensor([step, 0.0], dtype=points.dtype, device=points.device)
    dy = torch.tensor([0.0, step], dtype=points.dtype, device=points.device)
    hx0 = height_at(terrain, points - dx, cache=cache)
    hx1 = height_at(terrain, points + dx, cache=cache)
    hy0 = height_at(terrain, points - dy, cache=cache)
    hy1 = height_at(terrain, points + dy, cache=cache)
    dzdx = (hx1 - hx0) / max(2.0 * step, 1.0e-6)
    dzdy = (hy1 - hy0) / max(2.0 * step, 1.0e-6)
    return torch.sqrt(dzdx.square() + dzdy.square() + 1.0e-12)


def support_at(
    terrain: MpcPlannerTerrain,
    points_xy: Tensor,
    search_radius: float,
    search_step: float,
    max_support_slope: float,
    *,
    cache: TerrainQueryCache | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Find nearby semantic-terrain support points with finite fallback."""
    points, original_shape, single_batch = _batched_query_xy(terrain, points_xy)
    device = points.device
    dtype = points.dtype
    radius = float(search_radius)
    step = max(float(search_step), 1.0e-6)
    offsets_1d = torch.arange(-radius, radius + 0.5 * step, step, dtype=dtype, device=device)
    oy, ox = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
    offsets = torch.stack((ox.reshape(-1), oy.reshape(-1)), dim=-1)
    candidates = points.unsqueeze(2) + offsets.view(1, 1, -1, 2)
    candidate_shape = candidates.shape
    flat_candidates = candidates.reshape(candidate_shape[0], -1, 2)
    cand_z = height_at(terrain, flat_candidates, cache=cache).reshape(candidate_shape[:3])
    cand_slope = slope_at(terrain, flat_candidates, sample_step=step, cache=cache).reshape(candidate_shape[:3])
    cand_semantic = semantic_at(terrain, flat_candidates, cache=cache).reshape(candidate_shape[:3])
    legal = torch.logical_and(cand_semantic == 0, cand_slope <= float(max_support_slope))
    dist = _safe_norm(offsets, dim=-1).view(1, 1, -1)
    score = dist + cand_slope + torch.where(legal, torch.zeros_like(cand_slope), torch.full_like(cand_slope, 1.0e6))
    idx = score.argmin(dim=-1)
    invalid = torch.logical_not(legal.any(dim=-1))
    gather_xy = idx[..., None, None].expand(-1, -1, 1, 2)
    support_xy = candidates.gather(2, gather_xy).squeeze(2)
    support_z = cand_z.gather(2, idx.unsqueeze(-1)).squeeze(-1)
    support_slope = cand_slope.gather(2, idx.unsqueeze(-1)).squeeze(-1)
    query_z = height_at(terrain, points.reshape(points.shape[0], -1, 2), cache=cache).reshape(points.shape[:2])
    support_xy = torch.where(invalid.unsqueeze(-1), points, support_xy)
    support_z = torch.where(invalid, query_z, support_z)
    query_slope = slope_at(terrain, points.reshape(points.shape[0], -1, 2), sample_step=step, cache=cache).reshape(points.shape[:2])
    support_slope = torch.where(invalid, query_slope, support_slope)
    support_xy = support_xy.reshape(*original_shape, 2)
    support_z = support_z.reshape(original_shape)
    support_slope = support_slope.reshape(original_shape)
    invalid = invalid.reshape(original_shape)
    if single_batch:
        return support_xy.squeeze(0), support_z.squeeze(0), support_slope.squeeze(0), invalid.squeeze(0)
    return support_xy, support_z, support_slope, invalid


__all__ = [
    "build_mpc_terrain_from_scanner",
    "height_at",
    "semantic_at",
    "slope_at",
    "subset_mpc_terrain",
    "support_at",
    "TerrainQueryCache",
]
