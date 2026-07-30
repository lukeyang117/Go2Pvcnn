from __future__ import annotations

import torch
from torch import Tensor

from extension.parallelism.terrain import query_height_semantic_valid
from extension.parallelism.types import ParallelismTerrain


def swing_curve(start_w: Tensor, touchdown_w: Tensor, *, frames: int, height_m: float) -> Tensor:
    start = torch.as_tensor(start_w)
    touchdown = torch.as_tensor(touchdown_w, dtype=start.dtype, device=start.device)
    tau = torch.linspace(0.0, 1.0, int(frames), dtype=start.dtype, device=start.device)
    tau_view = tau.view(*((1,) * (start.ndim - 1)), int(frames), 1)
    curve = (1.0 - tau_view) * start[..., None, :] + tau_view * touchdown[..., None, :]
    curve = curve.clone()
    curve[..., 2] = curve[..., 2] + float(height_m) * 4.0 * tau * (1.0 - tau)
    return curve


def terrain_aware_swing_curve(
    start_w: Tensor,
    touchdown_w: Tensor,
    terrain: ParallelismTerrain,
    *,
    frames: int,
    clearance_m: float,
    min_apex_m: float,
) -> Tensor:
    start = torch.as_tensor(start_w)
    touchdown = torch.as_tensor(touchdown_w, dtype=start.dtype, device=start.device)
    tau = torch.linspace(0.0, 1.0, int(frames), dtype=start.dtype, device=start.device)
    tau_view = tau.view(*((1,) * (start.ndim - 1)), int(frames), 1)
    curve = (1.0 - tau_view) * start[..., None, :] + tau_view * touchdown[..., None, :]
    shape = 4.0 * tau * (1.0 - tau)
    xy = curve[..., :2]
    batch = int(start.shape[0])
    query = query_height_semantic_valid(terrain, xy.reshape(batch, -1, 2))
    terrain_z = query.height.reshape(*xy.shape[:-1])
    base_z = curve[..., 2]
    safe_z = terrain_z + float(clearance_m)
    shape_view = shape.view(*((1,) * (base_z.ndim - 1)), int(frames))
    interior = shape_view > 1.0e-6
    required = torch.where(
        interior,
        (safe_z - base_z) / shape_view.clamp_min(1.0e-6),
        torch.zeros_like(base_z),
    )
    apex = torch.amax(required, dim=-1).clamp_min(float(min_apex_m))
    curve = curve.clone()
    curve[..., 2] = base_z + apex[..., None] * shape_view
    curve[..., 0, 2] = start[..., 2]
    curve[..., -1, 2] = touchdown[..., 2]
    return curve
