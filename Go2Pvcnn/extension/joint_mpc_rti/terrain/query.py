"""Bilinear world-coordinate queries against immutable field batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.types import JointMpcTerrainField


@dataclass(frozen=True)
class JointMpcTerrainQuery:
    height_w: Tensor
    small_distance_m: Tensor
    large_distance_m: Tensor
    small_gradient_w: Tensor
    large_gradient_w: Tensor
    valid: Tensor


def _gather_grid(grid: Tensor, flat_index: Tensor) -> Tensor:
    batch = int(grid.shape[0])
    trailing = grid.shape[3:] if grid.ndim > 3 else ()
    if trailing:
        flattened = grid.reshape(batch, -1, *trailing)
        index = flat_index.view(batch, -1, *([1] * len(trailing))).expand(batch, flat_index.shape[1], *trailing)
        return torch.gather(flattened, 1, index)
    return torch.gather(grid.reshape(batch, -1), 1, flat_index)


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


def query_world(field: JointMpcTerrainField, points_w: Tensor) -> JointMpcTerrainQuery:
    points = torch.as_tensor(points_w, dtype=field.height_w.dtype, device=field.height_w.device)
    if points.ndim != 3 or int(points.shape[-1]) not in (2, 3):
        raise ValueError("points_w must have shape [B,N,2 or 3]")
    batch, nx, ny = map(int, field.height_w.shape)
    if int(points.shape[0]) != batch:
        raise ValueError("points_w batch must match field batch")
    delta = points[..., :2] - field.origin_w[:, None, :2]
    cosine = torch.cos(field.yaw_w)[:, None]
    sine = torch.sin(field.yaw_w)[:, None]
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
    small_gradient_local = _bilinear(field.small_gradient_xy, index_x, index_y)
    large_gradient_local = _bilinear(field.large_gradient_xy, index_x, index_y)

    def rotate_gradient(local: Tensor) -> Tensor:
        return torch.stack(
            (
                cosine * local[..., 0] - sine * local[..., 1],
                sine * local[..., 0] + cosine * local[..., 1],
            ),
            dim=-1,
        )

    return JointMpcTerrainQuery(
        height_w=_bilinear(field.height_w, index_x, index_y),
        small_distance_m=_bilinear(field.small_distance_m, index_x, index_y),
        large_distance_m=_bilinear(field.large_distance_m, index_x, index_y),
        small_gradient_w=rotate_gradient(small_gradient_local),
        large_gradient_w=rotate_gradient(large_gradient_local),
        valid=valid,
    )


__all__ = ["JointMpcTerrainQuery", "query_world"]
