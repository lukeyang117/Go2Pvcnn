"""Loss helpers for parametric MPC trajectory variables."""

from __future__ import annotations

import torch
from torch import Tensor

from .semantic_geometry import low_small_component_circles
from .terrain import height_at, semantic_at
from .types import MpcPlannerTerrain


def parametric_swing_foot_clearance_loss(
    terrain: MpcPlannerTerrain,
    target_foot_pos: Tensor,
    swing_prob: Tensor,
    *,
    margin_m: float,
) -> Tensor:
    foot = torch.as_tensor(target_foot_pos)
    batch, horizon = int(foot.shape[0]), int(foot.shape[1])
    dtype = foot.dtype
    device = foot.device
    terrain_z = height_at(terrain, foot[..., :2].reshape(batch, horizon * 4, 2)).reshape(batch, horizon, 4)
    terrain_z = terrain_z.to(dtype=dtype, device=device)
    deficit = torch.relu(terrain_z + float(margin_m) - foot[..., 2])
    return (deficit.square() * torch.as_tensor(swing_prob, dtype=dtype, device=device)).mean(dim=(1, 2))


def parametric_touchdown_keepout_loss(
    terrain: MpcPlannerTerrain,
    touchdown_w: Tensor,
    *,
    radius_extra_m: float,
    max_components: int,
) -> Tensor:
    touchdown = torch.as_tensor(touchdown_w)
    batch = int(touchdown.shape[0])
    dtype = touchdown.dtype
    device = touchdown.device
    zero = torch.zeros((batch,), dtype=dtype, device=device)
    if terrain.semantic_map is None:
        return zero
    semantic = semantic_at(terrain, touchdown[..., :2]).to(device=device)
    trigger = semantic == 1
    if not bool(torch.any(trigger)):
        return zero
    circles = low_small_component_circles(
        torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        max_components=int(max_components),
    )
    dist = torch.linalg.vector_norm(touchdown[..., None, :2] - circles.center_xy[:, None, :, :].to(dtype=dtype), dim=-1)
    keepout_radius = circles.radius[:, None, :].to(dtype=dtype, device=device) + float(radius_extra_m)
    deficit = torch.relu(keepout_radius - dist)
    circle_cost = torch.where(circles.valid[:, None, :].to(device=device), deficit.square(), torch.zeros_like(deficit))
    per_leg = circle_cost.amax(dim=-1)
    return (per_leg * trigger.to(dtype=dtype)).mean(dim=1)


__all__ = ["parametric_swing_foot_clearance_loss", "parametric_touchdown_keepout_loss"]
