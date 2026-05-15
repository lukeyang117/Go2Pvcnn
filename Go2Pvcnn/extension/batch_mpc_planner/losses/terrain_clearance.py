"""Scanner-driven terrain and semantic losses for batch MPC."""

from __future__ import annotations

import torch
from torch import Tensor

from ..terrain import height_at, semantic_at, slope_at, support_at
from ..types import MpcPlannerTerrain


def _safe_norm(value: Tensor, *, dim: int, eps: float = 1.0e-12) -> Tensor:
    return torch.sqrt(torch.sum(value.square(), dim=dim) + float(eps))


def _smooth_l1_small(value: Tensor, target: Tensor, *, beta: float = 0.02) -> Tensor:
    err = torch.abs(value - target)
    beta_t = torch.as_tensor(float(beta), dtype=err.dtype, device=err.device)
    return torch.where(err < beta_t, 0.5 * err.square() / beta_t, err - 0.5 * beta_t)


def stance_ground_loss(terrain: MpcPlannerTerrain, foot_pos: Tensor, contact_prob: Tensor) -> Tensor:
    terrain_z = height_at(terrain, foot_pos[..., :2]).to(dtype=foot_pos.dtype, device=foot_pos.device)
    err = _smooth_l1_small(foot_pos[..., 2], terrain_z)
    weight = contact_prob.to(dtype=foot_pos.dtype)
    return (weight * err).sum(dim=(1, 2)) / torch.clamp(weight.sum(dim=(1, 2)), min=1.0)


def swing_clearance_terrain_loss(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    swing_prob: Tensor,
    *,
    min_clearance_m: float,
) -> Tensor:
    terrain_z = height_at(terrain, foot_pos[..., :2]).to(dtype=foot_pos.dtype, device=foot_pos.device)
    deficit = torch.relu(terrain_z + float(min_clearance_m) - foot_pos[..., 2])
    weight = swing_prob.to(dtype=foot_pos.dtype)
    return (weight * deficit.square()).sum(dim=(1, 2)) / torch.clamp(weight.sum(dim=(1, 2)), min=1.0)


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
) -> Tensor:
    touchdown_xy = touchdown_w[..., :2]
    touchdown_z = touchdown_w[..., 2]
    terrain_z = height_at(terrain, touchdown_xy).to(dtype=touchdown_w.dtype, device=touchdown_w.device)
    slope = slope_at(terrain, touchdown_xy, sample_step=float(slope_sample_step)).to(dtype=touchdown_w.dtype, device=touchdown_w.device)
    support_xy, support_z, support_slope, invalid = support_at(
        terrain,
        touchdown_xy,
        search_radius=float(support_search_radius),
        search_step=float(support_search_step),
        max_support_slope=float(max_support_slope),
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
) -> Tensor:
    semantic = semantic_at(terrain, touchdown_xy)
    small = (semantic == 1).to(dtype=torch.float32, device=touchdown_xy.device)
    large = (semantic >= 2).to(dtype=torch.float32, device=touchdown_xy.device)
    return (float(small_weight) * small + float(large_weight) * large).mean(dim=-1)


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
) -> Tensor:
    foot_sem = semantic_at(terrain, foot_pos[..., :2])
    foot_small = (foot_sem == 1).to(dtype=foot_pos.dtype, device=foot_pos.device)
    foot_large = (foot_sem >= 2).to(dtype=foot_pos.dtype, device=foot_pos.device)
    obstacle = float(small_weight) * foot_small + float(large_weight) * foot_large
    terrain_z = height_at(terrain, foot_pos[..., :2]).to(dtype=foot_pos.dtype, device=foot_pos.device)
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
    body_sem = semantic_at(terrain, body_xy)
    body_small = (body_sem == 1).to(dtype=root_pos.dtype, device=root_pos.device)
    body_large = (body_sem >= 2).to(dtype=root_pos.dtype, device=root_pos.device)
    body_pen = float(small_weight) * body_small + float(large_weight) * body_large
    return float(foot_weight) * foot_pen.mean(dim=(1, 2)) + float(body_weight) * body_pen.mean(dim=(1, 2))


__all__ = [
    "finite_horizon_touchdown_phase",
    "sample_time",
    "semantic_obstacle_loss",
    "stance_ground_loss",
    "swing_clearance_terrain_loss",
    "touchdown_semantic_loss",
    "touchdown_surface_loss",
]
