"""Loss helpers for parametric MPC trajectory variables."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .kinematics import MpcLegPoints
from .semantic_geometry import LowSmallCircles, low_small_component_circles
from .terrain import height_at, semantic_at
from .types import MpcPlannerTerrain


@dataclass(frozen=True)
class FkCollisionMargins:
    foot: float
    knee: float
    shank: float
    root: float
    underbody: float


def _terrain_collision_cost(terrain: MpcPlannerTerrain, points: Tensor, *, margin_m: float) -> Tensor:
    pts = torch.as_tensor(points)
    batch = int(pts.shape[0])
    terrain_z = height_at(terrain, pts[..., :2].reshape(batch, -1, 2)).reshape(pts.shape[:-1])
    terrain_z = terrain_z.to(dtype=pts.dtype, device=pts.device)
    return torch.relu(terrain_z + float(margin_m) - pts[..., 2]).square()


def _underbody_points(root_pos: Tensor, *, sample_count: int) -> Tensor:
    batch, horizon = int(root_pos.shape[0]), int(root_pos.shape[1])
    dtype = root_pos.dtype
    device = root_pos.device
    offsets = torch.tensor(
        (
            (0.0, 0.0, -0.16),
            (0.18, 0.10, -0.16),
            (0.18, -0.10, -0.16),
            (-0.18, 0.10, -0.16),
            (-0.18, -0.10, -0.16),
        ),
        dtype=dtype,
        device=device,
    )
    count = max(1, min(int(sample_count), int(offsets.shape[0])))
    return root_pos[:, :, None, :] + offsets[:count].view(1, 1, count, 3).expand(batch, horizon, -1, -1)


def parametric_fk_body_leg_collision_loss(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    leg_points: MpcLegPoints,
    *,
    margins: FkCollisionMargins,
    underbody_sample_count: int,
) -> Tensor:
    root = torch.as_tensor(root_pos)
    foot_raw = _terrain_collision_cost(terrain, leg_points.foot_pos_world, margin_m=float(margins.foot))
    knee_raw = _terrain_collision_cost(terrain, leg_points.knee_pos_world, margin_m=float(margins.knee))
    shank_raw = _terrain_collision_cost(terrain, leg_points.shank_sample_world, margin_m=float(margins.shank))
    root_raw = _terrain_collision_cost(terrain, root, margin_m=float(margins.root))
    underbody_raw = _terrain_collision_cost(
        terrain,
        _underbody_points(root, sample_count=int(underbody_sample_count)),
        margin_m=float(margins.underbody),
    )
    foot = foot_raw.mean(dim=(1, 2)) + foot_raw.amax(dim=(1, 2))
    knee = knee_raw.mean(dim=(1, 2)) + knee_raw.amax(dim=(1, 2))
    shank = shank_raw.mean(dim=(1, 2, 3)) + shank_raw.amax(dim=(1, 2, 3))
    root_cost = root_raw.mean(dim=1) + root_raw.amax(dim=1)
    underbody = underbody_raw.mean(dim=(1, 2)) + underbody_raw.amax(dim=(1, 2))
    return foot + knee + shank + root_cost + underbody


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


def _world_to_root_frame(root_pos: Tensor, root_rpy: Tensor, points_w: Tensor) -> Tensor:
    yaw = torch.as_tensor(root_rpy, dtype=points_w.dtype, device=points_w.device)[..., 2]
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    delta = points_w - root_pos[:, :, None, :]
    x = cy[:, :, None] * delta[..., 0] + sy[:, :, None] * delta[..., 1]
    y = -sy[:, :, None] * delta[..., 0] + cy[:, :, None] * delta[..., 1]
    return torch.stack((x, y, delta[..., 2]), dim=-1)


def parametric_trajectory_fk_consistency_loss(
    root_pos: Tensor,
    root_rpy: Tensor,
    target_foot_pos: Tensor,
    fk_foot_pos: Tensor,
) -> Tensor:
    target = torch.as_tensor(target_foot_pos)
    fk = torch.as_tensor(fk_foot_pos, dtype=target.dtype, device=target.device)
    root = torch.as_tensor(root_pos, dtype=target.dtype, device=target.device)
    rpy = torch.as_tensor(root_rpy, dtype=target.dtype, device=target.device)
    abs_cost = (target - fk).square().sum(dim=-1).mean(dim=(1, 2))
    opt_rel = _world_to_root_frame(root, rpy, target)
    fk_rel = _world_to_root_frame(root, rpy, fk)
    rel_cost = (opt_rel - fk_rel).square().sum(dim=-1).mean(dim=(1, 2))
    return abs_cost + rel_cost


def parametric_plane_root_z_target_loss(
    root_pos: Tensor,
    root0: Tensor,
    is_plane_terrain: Tensor | None,
    *,
    target_height_m: float | None,
) -> Tensor:
    root = torch.as_tensor(root_pos)
    batch = int(root.shape[0])
    dtype = root.dtype
    device = root.device
    if is_plane_terrain is None:
        return torch.zeros((batch,), dtype=dtype, device=device)
    state_root = torch.as_tensor(root0, dtype=dtype, device=device)
    if target_height_m is None:
        target = state_root[:, 2]
    else:
        target = torch.full((batch,), float(target_height_m), dtype=dtype, device=device)
    err = (root[..., 2] - target[:, None]).square().mean(dim=1)
    return torch.where(torch.as_tensor(is_plane_terrain, dtype=torch.bool, device=device), err, torch.zeros_like(err))


def parametric_touchdown_keepout_loss(
    terrain: MpcPlannerTerrain,
    touchdown_w: Tensor,
    *,
    radius_extra_m: float,
    max_components: int,
    low_small_circles: LowSmallCircles | None = None,
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
    circles = low_small_circles
    if circles is None:
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


__all__ = [
    "FkCollisionMargins",
    "parametric_fk_body_leg_collision_loss",
    "parametric_plane_root_z_target_loss",
    "parametric_swing_foot_clearance_loss",
    "parametric_touchdown_keepout_loss",
    "parametric_trajectory_fk_consistency_loss",
]
