"""Semantic obstacle policy helpers for the batch MPC backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from .config import MpcPlannerCfg
from .losses.terrain_clearance import _nearby_height_for_sparse_semantic, _safe_norm, _semantic_id_mask, _terrain_grid_world_xy
from .terrain import height_at, semantic_at
from .types import MpcPlannerTerrain, MpcRobotState


class SemanticObstacleMode(IntEnum):
    NONE = 0
    LOW_SMALL_FORWARD = 1
    LOW_SMALL_MIXED = 2
    HIGH_OR_LARGE_AVOID = 3


@dataclass(frozen=True)
class SemanticObstaclePolicy:
    mode: Tensor
    obstacle_xy: Tensor
    obstacle_forward: Tensor
    obstacle_lateral: Tensor
    has_obstacle: Tensor


@dataclass(frozen=True)
class NominalCommandShapeDiagnostics:
    command_shaped: Tensor
    shape_side: Tensor
    left_score: Tensor
    right_score: Tensor


def _padded_command(command: Tensor, *, batch: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    return cmd[:, :3]


def classify_semantic_obstacle_mode(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    cfg: MpcPlannerCfg,
) -> SemanticObstaclePolicy:
    """Classify the nearest semantic obstacle in the commanded corridor."""
    root0 = torch.as_tensor(state.root_pos)
    batch = int(root0.shape[0])
    dtype = root0.dtype
    device = root0.device
    mode = torch.zeros((batch,), dtype=torch.long, device=device)
    obstacle_xy = torch.zeros((batch, 2), dtype=dtype, device=device)
    obstacle_forward = torch.zeros((batch,), dtype=dtype, device=device)
    obstacle_lateral = torch.zeros((batch,), dtype=dtype, device=device)
    has_obstacle = torch.zeros((batch,), dtype=torch.bool, device=device)
    if terrain.semantic_map is None:
        return SemanticObstaclePolicy(mode, obstacle_xy, obstacle_forward, obstacle_lateral, has_obstacle)

    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    if int(height.shape[0]) == 1 and batch > 1:
        height = height.expand(batch, -1, -1)
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    if int(height.shape[0]) != batch or int(semantic.shape[0]) != batch:
        return SemanticObstaclePolicy(mode, obstacle_xy, obstacle_forward, obstacle_lateral, has_obstacle)

    losses = cfg.losses
    grid_xy = _terrain_grid_world_xy(terrain, dtype=dtype, device=device)
    nearby_z = _nearby_height_for_sparse_semantic(terrain, height, dtype=dtype, device=device)
    grid_sem = semantic.reshape(batch, -1)
    root_ground = height_at(terrain, root0[:, None, :2]).reshape(batch).to(dtype=dtype, device=device)
    small = _semantic_id_mask(grid_sem, losses.touchdown_semantic.small_ids)
    large = _semantic_id_mask(grid_sem, losses.touchdown_semantic.large_ids)
    high_small = torch.logical_and(
        small,
        (nearby_z - root_ground[:, None]) > float(losses.low_small_crossing.high_small_relative_height_m),
    )
    low_small = torch.logical_and(small, torch.logical_not(high_small))

    cmd = _padded_command(command, batch=batch, dtype=dtype, device=device)
    cmd_xy = cmd[:, :2]
    speed = _safe_norm(cmd_xy, dim=-1)
    linear_active = speed > float(losses.low_small_crossing.linear_speed_eps)
    heading_body = cmd_xy / speed.clamp_min(1.0e-6).unsqueeze(-1)
    yaw = torch.as_tensor(state.root_rpy, dtype=dtype, device=device)[:, 2]
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    heading = torch.stack(
        (
            cy * heading_body[:, 0] - sy * heading_body[:, 1],
            sy * heading_body[:, 0] + cy * heading_body[:, 1],
        ),
        dim=-1,
    )
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    delta = grid_xy - root0[:, None, :2]
    forward = (delta * heading[:, None, :]).sum(dim=-1)
    lateral = (delta * left[:, None, :]).sum(dim=-1)
    corridor = torch.logical_and(
        torch.logical_and(forward >= 0.0, forward <= float(losses.high_obstacle_avoidance.forward_distance_m)),
        torch.abs(lateral) <= float(losses.high_obstacle_avoidance.corridor_width_m),
    )
    candidates = torch.logical_and(torch.logical_or(low_small, torch.logical_or(high_small, large)), corridor)
    candidates = torch.logical_and(candidates, linear_active[:, None])
    score = torch.where(candidates, forward, torch.full_like(forward, 1.0e6))
    index = score.argmin(dim=-1)
    has_obstacle = score.gather(1, index[:, None]).squeeze(-1) < 1.0e5
    obstacle_xy = grid_xy.gather(1, index[:, None, None].expand(batch, 1, 2)).squeeze(1)
    obstacle_forward = forward.gather(1, index[:, None]).squeeze(-1)
    obstacle_lateral = lateral.gather(1, index[:, None]).squeeze(-1)
    is_low = low_small.gather(1, index[:, None]).squeeze(-1)
    is_high_or_large = torch.logical_or(high_small, large).gather(1, index[:, None]).squeeze(-1)
    mixed = torch.logical_or(torch.abs(cmd[:, 1]) > 1.0e-4, torch.abs(cmd[:, 2]) > 1.0e-4)
    mode = torch.where(
        torch.logical_and(has_obstacle, is_high_or_large),
        torch.full_like(mode, int(SemanticObstacleMode.HIGH_OR_LARGE_AVOID)),
        mode,
    )
    mode = torch.where(
        torch.logical_and(has_obstacle, torch.logical_and(is_low, mixed)),
        torch.full_like(mode, int(SemanticObstacleMode.LOW_SMALL_MIXED)),
        mode,
    )
    mode = torch.where(
        torch.logical_and(has_obstacle, torch.logical_and(is_low, torch.logical_not(mixed))),
        torch.full_like(mode, int(SemanticObstacleMode.LOW_SMALL_FORWARD)),
        mode,
    )
    return SemanticObstaclePolicy(mode, obstacle_xy, obstacle_forward, obstacle_lateral, has_obstacle)


def shape_nominal_command_for_semantic_obstacles(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    cfg: MpcPlannerCfg,
) -> tuple[Tensor, NominalCommandShapeDiagnostics]:
    """Shape only the nominal seed command for high-small/large obstacles."""
    root0 = torch.as_tensor(state.root_pos)
    batch = int(root0.shape[0])
    dtype = root0.dtype
    device = root0.device
    cmd = _padded_command(command, batch=batch, dtype=dtype, device=device)
    policy = classify_semantic_obstacle_mode(terrain, state, cmd, cfg)
    avoid = policy.mode == int(SemanticObstacleMode.HIGH_OR_LARGE_AVOID)
    shaped = cmd.clone()
    left_score = torch.zeros((batch,), dtype=dtype, device=device)
    right_score = torch.zeros((batch,), dtype=dtype, device=device)
    side = torch.ones((batch,), dtype=dtype, device=device)
    if not bool(torch.any(avoid)):
        return shaped, NominalCommandShapeDiagnostics(
            command_shaped=avoid,
            shape_side=side.to(dtype=torch.long),
            left_score=left_score,
            right_score=right_score,
        )

    cmd_xy = cmd[:, :2]
    speed = _safe_norm(cmd_xy, dim=-1)
    direction = cmd_xy / speed.clamp_min(1.0e-6).unsqueeze(-1)
    left = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    forward_offsets = torch.tensor((-0.10, 0.15, 0.40, 0.65), dtype=dtype, device=device)
    lateral_offsets = torch.tensor((0.24, 0.36, 0.50), dtype=dtype, device=device)
    for side_value, target in ((1.0, left_score), (-1.0, right_score)):
        points = (
            policy.obstacle_xy[:, None, None, :]
            + forward_offsets.view(1, -1, 1, 1) * direction[:, None, None, :]
            + float(side_value) * lateral_offsets.view(1, 1, -1, 1) * left[:, None, None, :]
        ).reshape(batch, -1, 2)
        semantic = semantic_at(terrain, points)
        occupied = (torch.as_tensor(semantic, device=device) > 0).to(dtype=dtype)
        weights = torch.linspace(1.0, 0.4, int(points.shape[1]), dtype=dtype, device=device).view(1, -1)
        target.copy_((occupied * weights).sum(dim=-1))
    side = torch.where(left_score > right_score, -torch.ones_like(side), torch.ones_like(side))
    shaped_vx = cmd[:, 0] * 0.58
    shaped_vy = side * torch.maximum(
        torch.maximum(torch.abs(cmd[:, 1]), torch.full_like(cmd[:, 0], 0.22)),
        torch.abs(cmd[:, 0]) * 0.55,
    )
    current_ratio = torch.abs(cmd[:, 1]) / torch.abs(cmd[:, 0]).clamp_min(1.0e-6)
    shaped_ratio = torch.abs(shaped_vy) / torch.abs(shaped_vx).clamp_min(1.0e-6)
    shaped_vy = torch.where(current_ratio > shaped_ratio, cmd[:, 1], shaped_vy)
    shaped[:, 0] = torch.where(avoid, shaped_vx, shaped[:, 0])
    shaped[:, 1] = torch.where(avoid, shaped_vy, shaped[:, 1])
    return shaped, NominalCommandShapeDiagnostics(
        command_shaped=avoid,
        shape_side=side.to(dtype=torch.long),
        left_score=left_score,
        right_score=right_score,
    )


__all__ = [
    "NominalCommandShapeDiagnostics",
    "SemanticObstacleMode",
    "SemanticObstaclePolicy",
    "classify_semantic_obstacle_mode",
    "shape_nominal_command_for_semantic_obstacles",
]
