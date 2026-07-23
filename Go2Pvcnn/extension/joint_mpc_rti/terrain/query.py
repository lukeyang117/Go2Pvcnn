"""Bilinear world-coordinate queries against immutable field batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.types import JointMpcPerceptiveField, JointMpcTerrainField


@dataclass(frozen=True)
class JointMpcTerrainQuery:
    height_w: Tensor
    height_gradient_w: Tensor
    small_distance_m: Tensor
    large_distance_m: Tensor
    small_gradient_w: Tensor
    large_gradient_w: Tensor
    valid: Tensor
    small_occupancy: Tensor
    large_occupancy: Tensor
    small_propagated_height: Tensor
    large_propagated_height: Tensor
    small_occupancy_gradient_w: Tensor
    large_occupancy_gradient_w: Tensor
    semantic_id: Tensor


@dataclass(frozen=True)
class JointMpcPerceptiveQuery:
    height_w: Tensor
    semantic_id: Tensor
    valid: Tensor
    small_mask: Tensor
    large_mask: Tensor
    unknown_mask: Tensor
    inflated_height_w: Tensor
    landing_safe: Tensor
    slope_xy: Tensor
    slope_rad: Tensor
    roughness: Tensor
    semantic_edge_mask: Tensor
    boundary_distance_m: Tensor


def _gather_grid(grid: Tensor, flat_index: Tensor) -> Tensor:
    field_batch = int(grid.shape[0])
    query_batch = int(flat_index.shape[0])
    trailing = grid.shape[3:] if grid.ndim > 3 else ()
    if query_batch != field_batch:
        if query_batch % field_batch != 0:
            raise ValueError("query batch must be an integer repeat of field batch")
        repeats = query_batch // field_batch
        row_index = torch.arange(query_batch, device=flat_index.device, dtype=torch.long) // repeats
        cell_count = int(grid.shape[1] * grid.shape[2])
        global_index = row_index[:, None] * cell_count + flat_index
        if trailing:
            flattened = grid.reshape(field_batch * cell_count, *trailing)
            gathered = torch.index_select(flattened, 0, global_index.reshape(-1))
            return gathered.reshape(query_batch, flat_index.shape[1], *trailing)
        return torch.take(grid.reshape(-1), global_index)
    if trailing:
        flattened = grid.reshape(field_batch, -1, *trailing)
        index = flat_index.view(field_batch, -1, *([1] * len(trailing))).expand(
            field_batch, flat_index.shape[1], *trailing
        )
        return torch.gather(flattened, 1, index)
    return torch.gather(grid.reshape(field_batch, -1), 1, flat_index)


def _bilinear(grid: Tensor, index_x: Tensor, index_y: Tensor) -> Tensor:
    nx = int(grid.shape[1])
    ny = int(grid.shape[2])
    x0 = torch.floor(index_x).to(dtype=torch.long).clamp(0, nx - 1)
    y0 = torch.floor(index_y).to(dtype=torch.long).clamp(0, ny - 1)
    x1 = (x0 + 1).clamp(max=nx - 1)
    y1 = (y0 + 1).clamp(max=ny - 1)
    wx = (index_x - x0.to(index_x.dtype)).clamp(0.0, 1.0)
    wy = (index_y - y0.to(index_y.dtype)).clamp(0.0, 1.0)
    f00 = _gather_grid(grid, x0 * ny + y0)
    f10 = _gather_grid(grid, x1 * ny + y0)
    f01 = _gather_grid(grid, x0 * ny + y1)
    f11 = _gather_grid(grid, x1 * ny + y1)
    while wx.ndim < f00.ndim:
        wx = wx.unsqueeze(-1)
        wy = wy.unsqueeze(-1)
    return (1.0 - wx) * (1.0 - wy) * f00 + wx * (1.0 - wy) * f10 + (1.0 - wx) * wy * f01 + wx * wy * f11


def _bilinear_scalar_with_gradient(
    grid: Tensor,
    index_x: Tensor,
    index_y: Tensor,
    *,
    resolution: float,
) -> tuple[Tensor, Tensor]:
    nx = int(grid.shape[1])
    ny = int(grid.shape[2])
    x0 = torch.floor(index_x).to(dtype=torch.long).clamp(0, nx - 1)
    y0 = torch.floor(index_y).to(dtype=torch.long).clamp(0, ny - 1)
    x1 = (x0 + 1).clamp(max=nx - 1)
    y1 = (y0 + 1).clamp(max=ny - 1)
    wx = (index_x - x0.to(index_x.dtype)).clamp(0.0, 1.0)
    wy = (index_y - y0.to(index_y.dtype)).clamp(0.0, 1.0)
    f00 = _gather_grid(grid, x0 * ny + y0)
    f10 = _gather_grid(grid, x1 * ny + y0)
    f01 = _gather_grid(grid, x0 * ny + y1)
    f11 = _gather_grid(grid, x1 * ny + y1)
    value = (1.0 - wx) * (1.0 - wy) * f00 + wx * (1.0 - wy) * f10 + (1.0 - wx) * wy * f01 + wx * wy * f11
    inverse_resolution = 1.0 / float(resolution)
    gradient_x = inverse_resolution * ((1.0 - wy) * (f10 - f00) + wy * (f11 - f01))
    gradient_y = inverse_resolution * ((1.0 - wx) * (f01 - f00) + wx * (f11 - f10))
    return value, torch.stack((gradient_x, gradient_y), dim=-1)


def query_world(field: JointMpcTerrainField, points_w: Tensor) -> JointMpcTerrainQuery:
    points = torch.as_tensor(points_w, dtype=field.height_w.dtype, device=field.height_w.device)
    if points.ndim != 3 or int(points.shape[-1]) not in (2, 3):
        raise ValueError("points_w must have shape [B,N,2 or 3]")
    field_batch, nx, ny = map(int, field.height_w.shape)
    query_batch = int(points.shape[0])
    if query_batch % field_batch != 0:
        raise ValueError("points_w batch must be an integer repeat of field batch")
    repeats = query_batch // field_batch
    origin_w = field.origin_w.repeat_interleave(repeats, dim=0)
    yaw_w = field.yaw_w.repeat_interleave(repeats, dim=0)
    delta = points[..., :2] - origin_w[:, None, :2]
    cosine = torch.cos(yaw_w)[:, None]
    sine = torch.sin(yaw_w)[:, None]
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    index_x = local_x / float(field.resolution) + 0.5 * float(nx - 1)
    index_y = local_y / float(field.resolution) + 0.5 * float(ny - 1)
    inside = torch.logical_and(
        torch.logical_and(index_x >= 0.0, index_x <= float(nx - 1)),
        torch.logical_and(index_y >= 0.0, index_y <= float(ny - 1)),
    )
    sampled_valid = _bilinear(field.valid_mask.to(dtype=field.height_w.dtype), index_x, index_y) > 0.999
    valid = torch.logical_and(inside, sampled_valid)
    small_distance, small_gradient_local = _bilinear_scalar_with_gradient(
        field.small_distance_m, index_x, index_y, resolution=field.resolution
    )
    large_distance, large_gradient_local = _bilinear_scalar_with_gradient(
        field.large_distance_m, index_x, index_y, resolution=field.resolution
    )

    def rotate_gradient(local: Tensor) -> Tensor:
        return torch.stack(
            (
                cosine * local[..., 0] - sine * local[..., 1],
                sine * local[..., 0] + cosine * local[..., 1],
            ),
            dim=-1,
        )

    zero_scalar = torch.zeros_like(field.height_w)
    zero_vector = torch.zeros(*field.height_w.shape, 2, dtype=field.height_w.dtype, device=field.height_w.device)
    small_occupancy_grid = zero_scalar if field.small_occupancy is None else field.small_occupancy
    large_occupancy_grid = zero_scalar if field.large_occupancy is None else field.large_occupancy
    small_height_grid = field.height_w if field.small_propagated_height is None else field.small_propagated_height
    large_height_grid = field.height_w if field.large_propagated_height is None else field.large_propagated_height
    small_soft_gradient = zero_vector if field.small_occupancy_gradient_xy is None else field.small_occupancy_gradient_xy
    large_soft_gradient = zero_vector if field.large_occupancy_gradient_xy is None else field.large_occupancy_gradient_xy
    nearest_x = torch.round(index_x).to(dtype=torch.long).clamp(0, nx - 1)
    nearest_y = torch.round(index_y).to(dtype=torch.long).clamp(0, ny - 1)
    height_w, height_gradient_local = _bilinear_scalar_with_gradient(
        field.height_w,
        index_x,
        index_y,
        resolution=field.resolution,
    )

    return JointMpcTerrainQuery(
        height_w=height_w,
        height_gradient_w=rotate_gradient(height_gradient_local),
        small_distance_m=small_distance,
        large_distance_m=large_distance,
        small_gradient_w=rotate_gradient(small_gradient_local),
        large_gradient_w=rotate_gradient(large_gradient_local),
        valid=valid,
        small_occupancy=_bilinear(small_occupancy_grid, index_x, index_y),
        large_occupancy=_bilinear(large_occupancy_grid, index_x, index_y),
        small_propagated_height=_bilinear(small_height_grid, index_x, index_y),
        large_propagated_height=_bilinear(large_height_grid, index_x, index_y),
        small_occupancy_gradient_w=rotate_gradient(_bilinear(small_soft_gradient, index_x, index_y)),
        large_occupancy_gradient_w=rotate_gradient(_bilinear(large_soft_gradient, index_x, index_y)),
        semantic_id=_gather_grid(field.semantic_id, nearest_x * ny + nearest_y),
    )


def query_perceptive_world(
    field: JointMpcPerceptiveField,
    points_w: Tensor,
) -> JointMpcPerceptiveQuery:
    points = torch.as_tensor(points_w, dtype=field.height_w.dtype, device=field.height_w.device)
    if points.ndim != 3 or int(points.shape[-1]) not in (2, 3):
        raise ValueError("points_w must have shape [B,N,2 or 3]")
    field_batch, nx, ny = map(int, field.height_w.shape)
    query_batch = int(points.shape[0])
    if query_batch % field_batch != 0:
        raise ValueError("points_w batch must be an integer repeat of field batch")
    repeats = query_batch // field_batch
    origin_w = field.origin_w.repeat_interleave(repeats, dim=0)
    yaw_w = field.yaw_w.repeat_interleave(repeats, dim=0)
    delta = points[..., :2] - origin_w[:, None, :2]
    cosine = torch.cos(yaw_w)[:, None]
    sine = torch.sin(yaw_w)[:, None]
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    index_x = local_x / float(field.resolution) + 0.5 * float(nx - 1)
    index_y = local_y / float(field.resolution) + 0.5 * float(ny - 1)
    boundary_distance = torch.minimum(
        torch.minimum(index_x, index_y),
        torch.minimum(float(nx - 1) - index_x, float(ny - 1) - index_y),
    ) * float(field.resolution)
    inside = boundary_distance >= 0.0
    sampled_valid = _bilinear(
        field.valid_mask.to(dtype=field.height_w.dtype), index_x, index_y
    ) > 0.999
    valid = inside & sampled_valid
    nearest_x = torch.round(index_x).to(dtype=torch.long).clamp(0, nx - 1)
    nearest_y = torch.round(index_y).to(dtype=torch.long).clamp(0, ny - 1)
    nearest_index = nearest_x * ny + nearest_y

    def nearest_mask(mask: Tensor) -> Tensor:
        return _gather_grid(mask, nearest_index).to(dtype=torch.bool)

    inflated = field.inflated_height_w.permute(0, 2, 3, 1)
    return JointMpcPerceptiveQuery(
        height_w=_bilinear(field.height_w, index_x, index_y),
        semantic_id=_gather_grid(field.semantic_id, nearest_index),
        valid=valid,
        small_mask=nearest_mask(field.small_mask),
        large_mask=nearest_mask(field.large_mask),
        unknown_mask=(~valid) | nearest_mask(field.unknown_mask),
        inflated_height_w=_bilinear(inflated, index_x, index_y),
        landing_safe=inside
        & (_bilinear(field.landing_safe.to(field.height_w.dtype), index_x, index_y) > 0.999),
        slope_xy=_bilinear(field.slope_xy, index_x, index_y),
        slope_rad=_bilinear(field.slope_rad, index_x, index_y),
        roughness=_bilinear(field.roughness, index_x, index_y),
        semantic_edge_mask=nearest_mask(field.semantic_edge_mask),
        boundary_distance_m=boundary_distance,
    )


_COMPILED_QUERY_WORLD = torch.compile(
    query_world,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)


def query_world_maybe_compiled(
    field: JointMpcTerrainField,
    points_w: Tensor,
    *,
    enabled: bool,
) -> JointMpcTerrainQuery:
    if enabled and torch.as_tensor(points_w).is_cuda:
        return _COMPILED_QUERY_WORLD(field, points_w)
    return query_world(field, points_w)


__all__ = [
    "JointMpcPerceptiveQuery",
    "JointMpcTerrainQuery",
    "query_perceptive_world",
    "query_world",
    "query_world_maybe_compiled",
]
