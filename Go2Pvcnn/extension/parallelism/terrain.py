from __future__ import annotations

import torch
from torch import Tensor

from extension.parallelism.types import ParallelismTerrain, TerrainQueryResult


def _world_to_grid(terrain: ParallelismTerrain, points_w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    points = torch.as_tensor(
        points_w, dtype=terrain.height_w.dtype, device=terrain.height_w.device
    )
    origin_xy = terrain.origin_w[:, None, :2].to(dtype=points.dtype, device=points.device)
    yaw = terrain.yaw_w[:, None].to(dtype=points.dtype, device=points.device)
    delta = points - origin_xy
    cosine = torch.cos(-yaw)
    sine = torch.sin(-yaw)
    gx_m = cosine * delta[..., 0] - sine * delta[..., 1]
    gy_m = sine * delta[..., 0] + cosine * delta[..., 1]
    col = torch.round(gx_m / float(terrain.resolution)).to(torch.long)
    row = torch.round(gy_m / float(terrain.resolution)).to(torch.long)
    return row, col, points


def query_height_semantic_valid(
    terrain: ParallelismTerrain, points_w: Tensor
) -> TerrainQueryResult:
    row, col, points = _world_to_grid(terrain, points_w)
    batch, height_count, width_count = terrain.height_w.shape
    batch_index = torch.arange(batch, device=points.device)[:, None].expand_as(row)
    inside = (row >= 0) & (row < height_count) & (col >= 0) & (col < width_count)
    safe_row = row.clamp(0, height_count - 1)
    safe_col = col.clamp(0, width_count - 1)
    height = terrain.height_w[batch_index, safe_row, safe_col]
    semantic = terrain.semantic_id[batch_index, safe_row, safe_col]
    valid = inside & terrain.valid_mask[batch_index, safe_row, safe_col]
    return TerrainQueryResult(height=height, semantic=semantic, valid=valid)
