"""Loss helpers for parametric MPC trajectory variables."""

from __future__ import annotations

import torch
from torch import Tensor

from .semantic_geometry import low_small_component_circles
from .terrain import semantic_at
from .types import MpcPlannerTerrain


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


__all__ = ["parametric_touchdown_keepout_loss"]
