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
    terrain_query_radius_m: float = 0.0,
    terrain_query_angle_count: int = 8,
) -> Tensor:
    start = torch.as_tensor(start_w)
    touchdown = torch.as_tensor(touchdown_w, dtype=start.dtype, device=start.device)
    tau = torch.linspace(0.0, 1.0, int(frames), dtype=start.dtype, device=start.device)
    tau_view = tau.view(*((1,) * (start.ndim - 1)), int(frames), 1)
    curve = (1.0 - tau_view) * start[..., None, :] + tau_view * touchdown[..., None, :]
    shape = 4.0 * tau * (1.0 - tau)
    xy = curve[..., :2]
    batch = int(start.shape[0])
    query_radius = float(terrain_query_radius_m)
    if query_radius > 0.0:
        angle_count = max(int(terrain_query_angle_count), 4)
        angles = torch.arange(angle_count, dtype=xy.dtype, device=xy.device) * (2.0 * torch.pi / angle_count)
        offsets = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) * query_radius
        query_points = xy.unsqueeze(-2) + offsets.view(*((1,) * (xy.ndim - 1)), angle_count, 2)
        query = query_height_semantic_valid(terrain, query_points.reshape(batch, -1, 2))
        heights = query.height.reshape(*xy.shape[:-1], angle_count)
        valid = query.valid.reshape(*xy.shape[:-1], angle_count)
        terrain_z = heights.masked_fill(~valid, -torch.inf).amax(dim=-1)
        terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, torch.zeros_like(terrain_z))
    else:
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


def minimum_clearance_swing_curve(
    start_w: Tensor,
    touchdown_w: Tensor,
    terrain: ParallelismTerrain,
    *,
    frames: int,
    clearance_m: float,
    min_apex_m: float,
    terrain_query_radius_m: float = 0.0,
    terrain_query_angle_count: int = 8,
) -> Tensor:
    """Build the lowest sampled swing curve that clears the queried terrain.

    Unlike the parabolic terrain profile, the terrain constraint is applied
    directly at each sample. This prevents an obstacle detected near a swing
    endpoint from inflating the whole arc through division by a small
    parabolic shape value.
    """
    start = torch.as_tensor(start_w)
    touchdown = torch.as_tensor(touchdown_w, dtype=start.dtype, device=start.device)
    tau = torch.linspace(0.0, 1.0, int(frames), dtype=start.dtype, device=start.device)
    tau_view = tau.view(*((1,) * (start.ndim - 1)), int(frames), 1)
    curve = (1.0 - tau_view) * start[..., None, :] + tau_view * touchdown[..., None, :]
    xy = curve[..., :2]
    batch = int(start.shape[0])
    query_radius = float(terrain_query_radius_m)
    if query_radius > 0.0:
        angle_count = max(int(terrain_query_angle_count), 4)
        angles = torch.arange(angle_count, dtype=xy.dtype, device=xy.device) * (2.0 * torch.pi / angle_count)
        offsets = torch.cat(
            (
                torch.zeros(1, 2, dtype=xy.dtype, device=xy.device),
                torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) * query_radius,
            ),
            dim=0,
        )
        query_points = xy.unsqueeze(-2) + offsets.view(*((1,) * (xy.ndim - 1)), offsets.shape[0], 2)
        query = query_height_semantic_valid(terrain, query_points.reshape(batch, -1, 2))
        heights = query.height.reshape(*xy.shape[:-1], offsets.shape[0])
        valid = query.valid.reshape(*xy.shape[:-1], offsets.shape[0])
        terrain_z = heights.masked_fill(~valid, -torch.inf).amax(dim=-1)
    else:
        query = query_height_semantic_valid(terrain, xy.reshape(batch, -1, 2))
        terrain_z = query.height.reshape(*xy.shape[:-1])
    base_z = curve[..., 2]
    terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, torch.zeros_like(terrain_z))
    obstacle_z = terrain_z + float(clearance_m)
    # Zero-height ground is not an obstacle and must not lift a wheel that is
    # already at its nominal contact height.
    required_z = torch.where(terrain_z > 1.0e-6, obstacle_z, base_z)
    shape = 4.0 * tau * (1.0 - tau)
    nominal_z = base_z + float(min_apex_m) * shape.view(*((1,) * (base_z.ndim - 1)), int(frames))
    curve = curve.clone()
    curve[..., 2] = torch.maximum(nominal_z, required_z)
    curve[..., 0, 2] = start[..., 2]
    curve[..., -1, 2] = touchdown[..., 2]
    return curve
