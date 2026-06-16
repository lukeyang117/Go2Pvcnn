"""Parametric MPC planner core."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import MpcPlannerCfg, validate_mpc_config
from .diagnostics import evaluate_hard_reasons, status_from_hard_reasons
from .kinematics import fk_feet_from_joint_angles, fk_leg_points_from_joint_angles, solve_joint_angles_from_trajectory
from .losses.gait_coupling import swing_direction_loss
from .losses.kinematics import joint_limit_loss_from_root_foot
from .losses.terrain_clearance import finite_horizon_touchdown_phase, sample_time
from .parametric import command_frame_axes, decode_parametric_trajectory, init_parametric_variables
from .parametric_losses import (
    FkCollisionMargins,
    parametric_fk_body_leg_collision_loss,
    parametric_plane_root_z_target_loss,
    parametric_swing_foot_clearance_loss,
    parametric_touchdown_keepout_loss,
    parametric_trajectory_fk_consistency_loss,
)
from .profiling import MpcProfile, maybe_print_mpc_profile, should_profile_mpc
from .semantic_geometry import LowSmallCircles, low_small_component_circles
from .semantic_policy import build_parametric_nominal
from .terrain import height_at, semantic_at
from .types import MPC_HARD_REASON_COUNT, MpcPlannerResult, MpcPlannerStatus, MpcPlannerTerrain, MpcRobotState


def _normal_tensor(value: Tensor | None) -> Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value).detach().clone()


def _normal_state(state: MpcRobotState) -> MpcRobotState:
    return MpcRobotState(
        root_pos=_normal_tensor(state.root_pos),
        root_rpy=_normal_tensor(state.root_rpy),
        foot_pos=_normal_tensor(state.foot_pos),
        joint_angles=_normal_tensor(state.joint_angles),
        foot_vel=_normal_tensor(state.foot_vel),
    )


def _normal_terrain(terrain: MpcPlannerTerrain) -> MpcPlannerTerrain:
    return MpcPlannerTerrain(
        height_map=_normal_tensor(terrain.height_map),
        semantic_map=_normal_tensor(terrain.semantic_map),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=_normal_tensor(terrain.sensor_pos_w),
        sensor_yaw=_normal_tensor(terrain.sensor_yaw),
        is_plane_terrain=_normal_tensor(terrain.is_plane_terrain),
    )


def sample_touchdown_positions(foot_pos: Tensor, swing_center: Tensor, swing_width: Tensor) -> Tensor:
    touchdown_phase = finite_horizon_touchdown_phase(swing_center, swing_width)
    return sample_time(foot_pos, touchdown_phase, cyclic=False)


def _terrain_grid_world_xy(terrain: MpcPlannerTerrain, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    batch, height_count, width_count = int(height.shape[0]), int(height.shape[1]), int(height.shape[2])
    x = torch.linspace(float(terrain.world_x_range[0]), float(terrain.world_x_range[1]), width_count, dtype=dtype, device=device)
    y = torch.linspace(float(terrain.world_y_range[0]), float(terrain.world_y_range[1]), height_count, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    local_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1).view(1, height_count * width_count, 2).expand(batch, -1, -1)
    if terrain.sensor_pos_w is None:
        return local_xy
    sensor_pos = torch.as_tensor(terrain.sensor_pos_w, dtype=dtype, device=device)
    if sensor_pos.ndim == 1:
        sensor_pos = sensor_pos.view(1, -1).expand(batch, -1)
    yaw = torch.zeros((batch,), dtype=dtype, device=device) if terrain.sensor_yaw is None else torch.as_tensor(terrain.sensor_yaw, dtype=dtype, device=device).reshape(-1)
    if int(yaw.numel()) == 1 and batch > 1:
        yaw = yaw.expand(batch)
    cy = torch.cos(yaw).view(batch, 1)
    sy = torch.sin(yaw).view(batch, 1)
    world_xy = torch.stack((cy * local_xy[..., 0] - sy * local_xy[..., 1], sy * local_xy[..., 0] + cy * local_xy[..., 1]), dim=-1)
    return world_xy + sensor_pos[:, None, :2]


def _terrain_grid_local_xy(
    *,
    batch: int,
    height_count: int,
    width_count: int,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    x = torch.linspace(float(world_x_range[0]), float(world_x_range[1]), width_count, dtype=dtype, device=device)
    y = torch.linspace(float(world_y_range[0]), float(world_y_range[1]), height_count, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1).view(1, height_count * width_count, 2).expand(batch, -1, -1)


def _world_xy_to_field_local_xy(
    points_xy: Tensor,
    *,
    sensor_pos_w: Tensor | None,
    sensor_yaw: Tensor | None,
) -> Tensor:
    points = torch.as_tensor(points_xy)
    batch = int(points.shape[0])
    if sensor_pos_w is None:
        return points
    sensor_pos = torch.as_tensor(sensor_pos_w, dtype=points.dtype, device=points.device)
    if sensor_pos.ndim == 1:
        sensor_pos = sensor_pos.view(1, -1).expand(batch, -1)
    if int(sensor_pos.shape[0]) == 1 and batch > 1:
        sensor_pos = sensor_pos.expand(batch, -1)
    yaw = torch.zeros((batch,), dtype=points.dtype, device=points.device)
    if sensor_yaw is not None:
        yaw = torch.as_tensor(sensor_yaw, dtype=points.dtype, device=points.device).reshape(-1)
        if int(yaw.numel()) == 1 and batch > 1:
            yaw = yaw.expand(batch)
    delta = points - sensor_pos[:, :2].view(batch, *([1] * (points.ndim - 2)), 2)
    cy = torch.cos(yaw).view(batch, *([1] * (points.ndim - 2)))
    sy = torch.sin(yaw).view(batch, *([1] * (points.ndim - 2)))
    return torch.stack((cy * delta[..., 0] + sy * delta[..., 1], -sy * delta[..., 0] + cy * delta[..., 1]), dim=-1)


def _build_semantic_proximity_field(
    *,
    height_map: Tensor,
    semantic_map: Tensor,
    root_xy: Tensor,
    root_ground_z: Tensor,
    command: Tensor,
    root_yaw: Tensor,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    sensor_pos_w: Tensor | None = None,
    sensor_yaw: Tensor | None = None,
) -> Tensor:
    height = torch.as_tensor(height_map, dtype=torch.float32)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    device = height.device
    dtype = height.dtype
    batch, height_count, width_count = int(height.shape[0]), int(height.shape[1]), int(height.shape[2])
    semantic = torch.as_tensor(semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    if tuple(semantic.shape) != (batch, height_count, width_count):
        raise ValueError(
            "semantic_map must match height_map shape [B,H,W], "
            f"got semantic={tuple(semantic.shape)} height={tuple(height.shape)}"
        )
    root = torch.as_tensor(root_xy, dtype=dtype, device=device).reshape(batch, 2)
    root_ground = torch.as_tensor(root_ground_z, dtype=dtype, device=device).reshape(batch)
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    yaw = torch.as_tensor(root_yaw, dtype=dtype, device=device).reshape(-1)
    if int(yaw.numel()) == 1 and batch > 1:
        yaw = yaw.expand(batch)

    large_mask = semantic >= 2
    high_small = torch.logical_and(semantic == 1, (height - root_ground.view(batch, 1, 1)) > 0.30)
    risky = torch.logical_or(large_mask, high_small).reshape(batch, -1)

    grid_local = _terrain_grid_local_xy(
        batch=batch,
        height_count=height_count,
        width_count=width_count,
        world_x_range=world_x_range,
        world_y_range=world_y_range,
        dtype=dtype,
        device=device,
    )
    if sensor_pos_w is None:
        grid_xy = grid_local
    else:
        sensor_pos = torch.as_tensor(sensor_pos_w, dtype=dtype, device=device)
        if sensor_pos.ndim == 1:
            sensor_pos = sensor_pos.view(1, -1).expand(batch, -1)
        if int(sensor_pos.shape[0]) == 1 and batch > 1:
            sensor_pos = sensor_pos.expand(batch, -1)
        syaw = torch.zeros((batch,), dtype=dtype, device=device)
        if sensor_yaw is not None:
            syaw = torch.as_tensor(sensor_yaw, dtype=dtype, device=device).reshape(-1)
            if int(syaw.numel()) == 1 and batch > 1:
                syaw = syaw.expand(batch)
        cy = torch.cos(syaw).view(batch, 1)
        sy = torch.sin(syaw).view(batch, 1)
        grid_xy = torch.stack(
            (cy * grid_local[..., 0] - sy * grid_local[..., 1], sy * grid_local[..., 0] + cy * grid_local[..., 1]),
            dim=-1,
        ) + sensor_pos[:, None, :2]

    speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    yaw_active = torch.abs(cmd[:, 2]) > 1.0e-4
    active = torch.logical_or(speed > 1.0e-4, yaw_active)
    heading, left, _linear_active = command_frame_axes(cmd, yaw, linear_eps=1.0e-6)
    delta = grid_xy - root[:, None, :]
    along = (delta * heading[:, None, :]).sum(dim=-1)
    lateral = (delta * left[:, None, :]).sum(dim=-1)
    linear_candidate = torch.logical_and(
        torch.logical_and(along >= -0.10, along <= 1.50),
        torch.abs(lateral) <= 0.45,
    )
    yaw_candidate = torch.logical_and(yaw_active[:, None], torch.linalg.vector_norm(delta, dim=-1) <= 0.75)
    candidate = torch.logical_and(risky, torch.logical_or(linear_candidate, yaw_candidate))
    candidate = torch.logical_and(candidate, active[:, None]).view(batch, 1, height_count, width_count)

    base = candidate.to(dtype=dtype)
    near = F.avg_pool2d(F.max_pool2d(base, kernel_size=5, stride=1, padding=2), kernel_size=5, stride=1, padding=2)
    mid = F.avg_pool2d(F.max_pool2d(base, kernel_size=15, stride=1, padding=7), kernel_size=9, stride=1, padding=4)
    far = F.avg_pool2d(F.max_pool2d(base, kernel_size=31, stride=1, padding=15), kernel_size=15, stride=1, padding=7)
    return (near + 0.4 * mid + 0.1 * far).clamp(max=1.0)


def _sample_proximity_field_at_world_xy(
    field: Tensor,
    points_xy: Tensor,
    *,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    sensor_pos_w: Tensor | None = None,
    sensor_yaw: Tensor | None = None,
) -> Tensor:
    risk_field = torch.as_tensor(field)
    if risk_field.ndim != 4 or int(risk_field.shape[1]) != 1:
        raise ValueError(f"field must have shape [B,1,H,W], got {tuple(risk_field.shape)}")
    points = torch.as_tensor(points_xy, dtype=risk_field.dtype, device=risk_field.device)
    if points.shape[-1] != 2:
        raise ValueError(f"points_xy must end in 2, got {tuple(points.shape)}")
    batch = int(risk_field.shape[0])
    if int(points.shape[0]) == 1 and batch > 1:
        points = points.expand(batch, *points.shape[1:])
    if int(points.shape[0]) != batch:
        raise ValueError(f"points batch {int(points.shape[0])} must match field batch {batch}")
    original_shape = tuple(points.shape[:-1])
    local = _world_xy_to_field_local_xy(points, sensor_pos_w=sensor_pos_w, sensor_yaw=sensor_yaw)
    x0, x1 = float(world_x_range[0]), float(world_x_range[1])
    y0, y1 = float(world_y_range[0]), float(world_y_range[1])
    x_norm = (local[..., 0].clamp(x0, x1) - x0) / max(x1 - x0, 1.0e-6) * 2.0 - 1.0
    y_norm = (local[..., 1].clamp(y0, y1) - y0) / max(y1 - y0, 1.0e-6) * 2.0 - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).reshape(batch, -1, 1, 2)
    sampled = F.grid_sample(risk_field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, 0, :, 0].reshape(original_shape)


def _semantic_avoidance_context(
    terrain: MpcPlannerTerrain,
    root_xy: Tensor,
    command: Tensor,
    *,
    root_yaw: Tensor,
) -> dict[str, Tensor] | None:
    batch = int(root_xy.shape[0])
    dtype = root_xy.dtype
    device = root_xy.device
    if terrain.semantic_map is None:
        return None
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if int(height.shape[0]) == 1 and batch > 1:
        height = height.expand(batch, -1, -1)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    if int(semantic.shape[0]) != batch or int(height.shape[0]) != batch:
        return None
    root_ground = height_at(terrain, root_xy[:, None, :]).reshape(batch).to(dtype=dtype, device=device)
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    field = _build_semantic_proximity_field(
        height_map=height,
        semantic_map=semantic,
        root_xy=root_xy,
        root_ground_z=root_ground,
        command=cmd,
        root_yaw=root_yaw,
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=terrain.sensor_pos_w,
        sensor_yaw=terrain.sensor_yaw,
    ).to(dtype=dtype, device=device)
    return {
        "field": field,
        "active": field.flatten(1).amax(dim=1) > 0.0,
    }


def _low_small_circles_context(
    terrain: MpcPlannerTerrain,
    *,
    max_components: int,
    device: torch.device,
) -> LowSmallCircles | None:
    if terrain.semantic_map is None:
        return None
    return low_small_component_circles(
        torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        max_components=int(max_components),
    )


def _nearest_low_small_obstacle(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    heading: Tensor,
    left: Tensor,
    speed: Tensor,
    *,
    corridor_width_m: float = 0.24,
    forward_distance_m: float = 1.2,
    small_id: int = 1,
) -> tuple[Tensor, Tensor]:
    batch = int(root_pos.shape[0])
    dtype = root_pos.dtype
    device = root_pos.device
    if terrain.semantic_map is None:
        return torch.zeros((batch, 2), dtype=dtype, device=device), torch.zeros((batch,), dtype=torch.bool, device=device)
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    height_count, width_count = int(height.shape[1]), int(height.shape[2])
    if semantic.ndim == 1 and int(semantic.numel()) == height_count * width_count:
        semantic = semantic.reshape(1, height_count, width_count)
    if semantic.ndim == 2:
        if tuple(semantic.shape) == (height_count, width_count):
            semantic = semantic.unsqueeze(0)
        elif int(semantic.shape[-1]) == height_count * width_count:
            semantic = semantic.reshape(int(semantic.shape[0]), height_count, width_count)
        else:
            return torch.zeros((batch, 2), dtype=dtype, device=device), torch.zeros((batch,), dtype=torch.bool, device=device)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    if int(semantic.shape[0]) != batch:
        return torch.zeros((batch, 2), dtype=dtype, device=device), torch.zeros((batch,), dtype=torch.bool, device=device)
    grid_xy = _terrain_grid_world_xy(terrain, dtype=dtype, device=device)
    small = semantic.reshape(batch, -1) == int(small_id)
    delta = grid_xy - root_pos[:, None, :2]
    along = (delta * heading[:, None, :]).sum(dim=-1)
    lateral = (delta * left[:, None, :]).sum(dim=-1)
    candidate = torch.logical_and(
        small,
        torch.logical_and(
            torch.logical_and(along >= -0.20, along <= float(forward_distance_m)),
            torch.abs(lateral) <= float(corridor_width_m),
        ),
    )
    candidate = torch.logical_and(candidate, speed[:, None] > 1.0e-4)
    score = torch.where(candidate, torch.abs(along) + 0.25 * torch.abs(lateral), torch.full_like(along, 1.0e6))
    idx = score.argmin(dim=1)
    valid = score.gather(1, idx[:, None]).squeeze(1) < 1.0e5
    obstacle_xy = grid_xy.gather(1, idx[:, None, None].expand(batch, 1, 2)).squeeze(1)
    return obstacle_xy, valid


def _command_farthest_touchdown_positions(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    contact_state: Tensor,
    command: Tensor,
    *,
    root_yaw: Tensor | None = None,
) -> Tensor:
    batch, horizon, legs, _ = foot_pos.shape
    cmd = torch.as_tensor(command, dtype=foot_pos.dtype, device=foot_pos.device)
    if cmd.ndim != 2 or int(cmd.shape[0]) != batch:
        return foot_pos[:, -1]
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=cmd.dtype, device=cmd.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    yaw = torch.zeros((batch,), dtype=cmd.dtype, device=cmd.device) if root_yaw is None else torch.as_tensor(root_yaw, dtype=cmd.dtype, device=cmd.device).reshape(-1)
    heading, _left, _linear_active = command_frame_axes(cmd, yaw, linear_eps=1.0e-6)
    along = (foot_pos[..., :2] * heading[:, None, None, :]).sum(dim=-1)
    swing_mask = torch.logical_not(contact_state)
    along = torch.where(swing_mask, along, torch.full_like(along, -1.0e6))
    farthest_idx = along.argmax(dim=1)
    gather_idx = farthest_idx[:, None, :, None].expand(batch, 1, legs, 3)
    farthest = torch.gather(foot_pos, dim=1, index=gather_idx).squeeze(1)
    touchdown_z = height_at(terrain, farthest[..., :2]).to(dtype=foot_pos.dtype, device=foot_pos.device)
    farthest = torch.cat((farthest[..., :2], touchdown_z.unsqueeze(-1)), dim=-1)
    fallback = foot_pos[:, -1]
    active = torch.logical_and(speed > 1.0e-6, torch.any(swing_mask, dim=1))
    return torch.where(active.unsqueeze(-1), farthest, fallback)


def _structured_low_small_touchdown_positions(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    foot_pos: Tensor,
    contact_state: Tensor,
    command: Tensor,
    *,
    return_apply_mask: bool = False,
) -> Tensor:
    batch, horizon, legs, _ = foot_pos.shape
    dtype = foot_pos.dtype
    device = foot_pos.device
    state_rpy = torch.as_tensor(state.root_rpy, dtype=dtype, device=device)
    root_yaw = state_rpy[:, 2]
    fallback = _command_farthest_touchdown_positions(terrain, foot_pos, contact_state, command, root_yaw=root_yaw)
    empty_mask = torch.zeros((batch, legs), dtype=torch.bool, device=device)
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if cmd.ndim != 2 or int(cmd.shape[0]) != batch:
        return (fallback, empty_mask) if return_apply_mask else fallback
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    heading, left, _linear_active = command_frame_axes(cmd, root_yaw, linear_eps=1.0e-6)
    root0 = torch.as_tensor(state.root_pos, dtype=dtype, device=device)
    foot0 = torch.as_tensor(state.foot_pos, dtype=dtype, device=device)
    if foot0.ndim != 3 or int(foot0.shape[0]) != batch or int(foot0.shape[1]) != legs:
        return (fallback, empty_mask) if return_apply_mask else fallback
    obstacle_xy, active = _nearest_low_small_obstacle(terrain, root0, heading, left, speed)
    if not bool(active.any().item()):
        return (fallback, empty_mask) if return_apply_mask else fallback

    foot_delta = foot0[..., :2] - obstacle_xy[:, None, :]
    foot_along = (foot_delta * heading[:, None, :]).sum(dim=-1)
    foot_lateral = (foot_delta * left[:, None, :]).sum(dim=-1)
    lane = torch.abs(foot_lateral) <= 0.26
    in_front = foot_along < -0.035
    in_cross_window = torch.logical_and(foot_along >= -0.48, foot_along <= -0.06)
    already_after = foot_along > 0.08
    swing_mask = torch.logical_not(contact_state).any(dim=1)
    approaching = torch.logical_and(torch.logical_and(lane, in_front), foot_along >= -0.58)
    eligible = torch.logical_and(swing_mask, torch.logical_and(lane, in_cross_window))
    best_score = torch.where(eligible, torch.abs(foot_along + 0.18) + 0.35 * torch.abs(foot_lateral), torch.full_like(foot_along, 1.0e6))
    cross_idx = best_score.argmin(dim=1)
    has_cross = best_score.gather(1, cross_idx[:, None]).squeeze(1) < 1.0e5
    leg_ids = torch.arange(legs, device=device).view(1, legs).expand(batch, legs)
    cross_leg = torch.logical_and(has_cross[:, None], leg_ids == cross_idx[:, None])

    approach_along = -0.14
    cross_along = 0.18
    keep_after_along = torch.clamp(foot_along, min=0.12, max=0.34)
    target_along = torch.where(approaching, torch.full_like(foot_along, approach_along), foot_along)
    target_along = torch.where(cross_leg, torch.full_like(target_along, cross_along), target_along)
    target_along = torch.where(already_after, keep_after_along, target_along)
    approach_lateral = torch.clamp(foot_lateral, min=-0.16, max=0.16)
    cross_lateral = torch.clamp(foot_lateral, min=-0.045, max=0.045)
    target_lateral = torch.where(cross_leg, cross_lateral, approach_lateral)
    target_xy = obstacle_xy[:, None, :] + heading[:, None, :] * target_along[..., None] + left[:, None, :] * target_lateral[..., None]
    target_z = height_at(terrain, target_xy).to(dtype=dtype, device=device)
    structured = torch.cat((target_xy, target_z.unsqueeze(-1)), dim=-1)
    apply_leg = torch.logical_and(active[:, None], torch.logical_or(torch.logical_or(swing_mask, approaching), already_after))
    touchdown = torch.where(apply_leg[..., None], structured, fallback)
    return (touchdown, apply_leg) if return_apply_mask else touchdown


def _align_low_small_swing_to_touchdown(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    root_pos: Tensor,
    foot_pos: Tensor,
    contact_state: Tensor,
    touchdown_w: Tensor,
    apply_leg: Tensor,
) -> Tensor:
    if not bool(apply_leg.any().item()):
        return foot_pos
    batch, horizon, legs, _ = foot_pos.shape
    dtype = foot_pos.dtype
    device = foot_pos.device
    foot0 = torch.as_tensor(state.foot_pos, dtype=dtype, device=device)
    if foot0.ndim != 3 or int(foot0.shape[0]) != batch or int(foot0.shape[1]) != legs:
        return foot_pos
    swing = torch.logical_not(contact_state)
    active = torch.logical_and(swing, apply_leg[:, None, :])
    if not bool(active.any().item()):
        return foot_pos
    swing_f = swing.to(dtype=dtype)
    swing_count = swing_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    phase = torch.cumsum(swing_f, dim=1) / swing_count
    if horizon > 0:
        phase = phase.clone()
        phase[:, 0, :] = 0.0
    phase = phase.clamp(0.0, 1.0)
    target_xy = foot0[:, None, :, :2] + (touchdown_w[:, None, :, :2] - foot0[:, None, :, :2]) * phase[..., None]
    terrain_z = height_at(terrain, target_xy.reshape(batch, horizon * legs, 2)).reshape(batch, horizon, legs).to(dtype=dtype, device=device)
    arc = 4.0 * phase * (1.0 - phase) * 0.135
    desired_z = terrain_z + arc
    root_cap = root_pos[..., 2:3].expand(batch, horizon, legs) - 0.005
    min_clear = terrain_z + 0.025
    desired_z = torch.where(root_cap > min_clear, torch.minimum(desired_z, root_cap), min_clear)
    desired = torch.cat((target_xy, desired_z.unsqueeze(-1)), dim=-1)
    return torch.where(active[..., None], desired, foot_pos)


def _touchdown_export(
    foot_pos: Tensor,
    swing_center: Tensor,
    swing_width: Tensor,
    *,
    event_cap: int,
) -> tuple[Tensor, Tensor]:
    batch, horizon, legs, _ = foot_pos.shape
    touchdown_w = sample_touchdown_positions(foot_pos, swing_center, swing_width)
    touchdown_seq = touchdown_w.unsqueeze(2).expand(batch, legs, int(event_cap), 3).contiguous()
    planned_touchdown_w = touchdown_w.unsqueeze(1).expand(batch, horizon, legs, 3).contiguous()
    return touchdown_seq, planned_touchdown_w


def _zero_command_mask(command: Tensor, *, batch: int, device: torch.device) -> Tensor:
    cmd = torch.as_tensor(command, dtype=torch.float32, device=device)
    if cmd.ndim != 2:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    if int(cmd.shape[0]) != batch:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((int(cmd.shape[0]), 3 - int(cmd.shape[-1])), dtype=cmd.dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    return torch.linalg.vector_norm(cmd[:, :3], dim=-1) <= 1.0e-5


def _standstill_result_from_state(
    state: MpcRobotState,
    *,
    horizon: int,
    cfg: MpcPlannerCfg,
    terrain: MpcPlannerTerrain | None = None,
) -> MpcPlannerResult:
    root_pos0 = torch.as_tensor(state.root_pos)
    device = root_pos0.device
    dtype = root_pos0.dtype
    batch = int(root_pos0.shape[0])
    root_rpy0 = torch.as_tensor(state.root_rpy, dtype=dtype, device=device)
    foot0 = torch.as_tensor(state.foot_pos, dtype=dtype, device=device)
    joint0 = torch.as_tensor(state.joint_angles, dtype=dtype, device=device)
    foot_standstill = foot0
    if terrain is not None:
        terrain_z = height_at(terrain, foot0[..., :2]).to(dtype=dtype, device=device)
        foot_standstill = torch.cat((foot0[..., :2], terrain_z.unsqueeze(-1)), dim=-1)
    root_pos = root_pos0[:, None, :].expand(batch, int(horizon), 3).contiguous()
    root_rpy = root_rpy0[:, None, :].expand(batch, int(horizon), 3).contiguous()
    foot_pos = foot_standstill[:, None, :, :].expand(batch, int(horizon), 4, 3).contiguous()
    joint_angles = joint0[:, None, :].expand(batch, int(horizon), 12).contiguous()
    contact_state = torch.ones((batch, int(horizon), 4), dtype=torch.bool, device=device)
    touchdown_seq = foot_standstill.unsqueeze(2).expand(batch, 4, int(cfg.runtime.touchdown_event_cap), 3).contiguous()
    planned_touchdown_w = foot_standstill[:, None, :, :].expand(batch, int(horizon), 4, 3).contiguous()
    cost_total = torch.zeros((batch,), dtype=dtype, device=device)
    status = torch.full((batch,), int(MpcPlannerStatus.OK), dtype=torch.long, device=device)
    feasible = torch.ones((batch,), dtype=torch.bool, device=device)
    safe_fallback = torch.zeros((batch,), dtype=torch.bool, device=device)
    hard_reason_mask = torch.zeros((batch, MPC_HARD_REASON_COUNT), dtype=torch.bool, device=device)
    return MpcPlannerResult(
        root_pos=root_pos,
        root_rpy=root_rpy,
        foot_pos=foot_pos,
        joint_angles=joint_angles,
        contact_state=contact_state,
        touchdown_seq=touchdown_seq,
        planned_touchdown_w=planned_touchdown_w,
        cost_total=cost_total,
        cost_breakdown={"cost_total": cost_total},
        status=status,
        feasible=feasible,
        safe_fallback=safe_fallback,
        loss_breakdown={} if cfg.diagnostics.enabled else None,
        hard_reason_mask=hard_reason_mask if cfg.diagnostics.enabled else None,
    )


def _subset_state(state: MpcRobotState, ids: Tensor) -> MpcRobotState:
    return MpcRobotState(
        root_pos=state.root_pos.index_select(0, ids),
        root_rpy=state.root_rpy.index_select(0, ids),
        foot_pos=state.foot_pos.index_select(0, ids),
        joint_angles=state.joint_angles.index_select(0, ids),
        foot_vel=state.foot_vel.index_select(0, ids) if state.foot_vel is not None else None,
    )


def _subset_terrain(terrain: MpcPlannerTerrain, ids: Tensor) -> MpcPlannerTerrain:
    return MpcPlannerTerrain(
        height_map=terrain.height_map.index_select(0, ids),
        semantic_map=terrain.semantic_map.index_select(0, ids) if terrain.semantic_map is not None else None,
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=terrain.sensor_pos_w.index_select(0, ids) if terrain.sensor_pos_w is not None else None,
        sensor_yaw=terrain.sensor_yaw.index_select(0, ids) if terrain.sensor_yaw is not None else None,
        is_plane_terrain=terrain.is_plane_terrain.index_select(0, ids) if terrain.is_plane_terrain is not None else None,
    )


def _merge_subset_result(base: MpcPlannerResult, subset: MpcPlannerResult, ids: Tensor, *, diagnostics_enabled: bool) -> MpcPlannerResult:
    def _scatter(field: str):
        dst = getattr(base, field).clone()
        dst.index_copy_(0, ids, getattr(subset, field))
        return dst

    cost_breakdown = {name: value.clone() for name, value in base.cost_breakdown.items()}
    for name, value in subset.cost_breakdown.items():
        target = cost_breakdown.get(name)
        if target is None:
            target = torch.zeros_like(base.cost_total)
        else:
            target = target.clone()
        target.index_copy_(0, ids, value)
        cost_breakdown[name] = target

    loss_breakdown = None
    if diagnostics_enabled:
        loss_breakdown = {}
        base_loss = base.loss_breakdown or {}
        sub_loss = subset.loss_breakdown or {}
        for name in sorted(set(base_loss) | set(sub_loss)):
            if name in base_loss:
                target = base_loss[name].clone()
            else:
                target = torch.zeros_like(base.cost_total)
            if name in sub_loss:
                target.index_copy_(0, ids, sub_loss[name])
            loss_breakdown[name] = target

    hard_reason_mask = None
    if diagnostics_enabled:
        hard_reason_mask = base.hard_reason_mask.clone() if base.hard_reason_mask is not None else None
        if hard_reason_mask is not None and subset.hard_reason_mask is not None:
            hard_reason_mask.index_copy_(0, ids, subset.hard_reason_mask)

    return MpcPlannerResult(
        root_pos=_scatter("root_pos"),
        root_rpy=_scatter("root_rpy"),
        foot_pos=_scatter("foot_pos"),
        joint_angles=_scatter("joint_angles"),
        contact_state=_scatter("contact_state"),
        touchdown_seq=_scatter("touchdown_seq"),
        planned_touchdown_w=_scatter("planned_touchdown_w"),
        cost_total=_scatter("cost_total"),
        cost_breakdown=cost_breakdown,
        status=_scatter("status"),
        feasible=_scatter("feasible"),
        safe_fallback=_scatter("safe_fallback"),
        loss_breakdown=loss_breakdown,
        hard_reason_mask=hard_reason_mask,
    )


def _parametric_result_from_state(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    *,
    cfg: MpcPlannerCfg,
    profile: MpcProfile | None = None,
) -> MpcPlannerResult:
    horizon = int(cfg.runtime.horizon_steps)
    command_tensor = torch.as_tensor(command, dtype=torch.as_tensor(state.root_pos).dtype, device=torch.as_tensor(state.root_pos).device)
    nominal_t0 = profile.now() if profile is not None else 0.0
    nominal = build_parametric_nominal(state, terrain, command_tensor, cfg, horizon=horizon)
    if profile is not None:
        profile.add_stage("parametric.nominal", (profile.now() - nominal_t0) * 1000.0)
    planning_command = nominal.command
    init_t0 = profile.now() if profile is not None else 0.0
    variables = init_parametric_variables(state, planning_command, horizon=horizon)
    if profile is not None:
        profile.add_stage("parametric.init_vars", (profile.now() - init_t0) * 1000.0)
    decoded, loss_breakdown = _optimize_parametric_variables(
        terrain,
        state,
        planning_command,
        loss_command=command,
        variables=variables,
        nominal=nominal,
        horizon=horizon,
        cfg=cfg,
        profile=profile,
    )
    export_t0 = profile.now() if profile is not None else 0.0
    root_pos = decoded.root_pos.detach()
    root_rpy = decoded.root_rpy.detach()
    target_foot_pos = decoded.target_foot_pos.detach()
    batch = int(root_pos.shape[0])
    joint_seq = solve_joint_angles_from_trajectory(root_pos, root_rpy, target_foot_pos)
    if horizon > 0:
        state_joints = torch.as_tensor(state.joint_angles, dtype=joint_seq.dtype, device=joint_seq.device)
        joint_seq = joint_seq.clone()
        joint_seq[:, 0, :] = state_joints
    foot_pos = fk_feet_from_joint_angles(root_pos, root_rpy, joint_seq)
    if horizon > 0:
        state_foot = torch.as_tensor(state.foot_pos, dtype=foot_pos.dtype, device=foot_pos.device)
        foot_pos = foot_pos.clone()
        foot_pos[:, 0, :, :] = state_foot
    contact_state = decoded.contact_prob >= float(cfg.runtime.contact_threshold)
    touchdown_seq = decoded.touchdown_w.unsqueeze(2).expand(batch, 4, int(cfg.runtime.touchdown_event_cap), 3).contiguous()
    planned_touchdown_w = decoded.touchdown_w.unsqueeze(1).expand(batch, horizon, 4, 3).contiguous()
    loss_breakdown = {name: value.detach() for name, value in loss_breakdown.items()}
    cost_total = sum(loss_breakdown.values(), torch.zeros((batch,), dtype=root_pos.dtype, device=root_pos.device))
    cost_breakdown = {"cost_total": cost_total}
    cost_breakdown.update(loss_breakdown)
    status = torch.full((batch,), int(MpcPlannerStatus.OK), dtype=torch.long, device=root_pos.device)
    feasible = torch.ones_like(status, dtype=torch.bool)
    safe_fallback = torch.zeros_like(status, dtype=torch.bool)
    hard_reason_mask = torch.zeros((batch, MPC_HARD_REASON_COUNT), dtype=torch.bool, device=root_pos.device)
    if cfg.diagnostics.enabled:
        hard_reason_mask = evaluate_hard_reasons(
            root_pos=root_pos,
            foot_pos=foot_pos,
            joint_angles=joint_seq,
            contact_state=contact_state,
            command=torch.as_tensor(command, dtype=root_pos.dtype, device=root_pos.device),
        )
        status, feasible, safe_fallback = status_from_hard_reasons(hard_reason_mask)
    finite_ok = (
        torch.isfinite(root_pos).flatten(1).all(dim=1)
        & torch.isfinite(root_rpy).flatten(1).all(dim=1)
        & torch.isfinite(foot_pos).flatten(1).all(dim=1)
        & torch.isfinite(joint_seq).flatten(1).all(dim=1)
    )
    if bool(torch.any(torch.logical_not(finite_ok))):
        fallback = _standstill_result_from_state(state, horizon=horizon, cfg=cfg, terrain=terrain)
        bad_3 = torch.logical_not(finite_ok).view(batch, 1, 1)
        bad_4 = torch.logical_not(finite_ok).view(batch, 1, 1, 1)
        root_pos = torch.where(bad_3, fallback.root_pos.to(dtype=root_pos.dtype, device=root_pos.device), root_pos)
        root_rpy = torch.where(bad_3, fallback.root_rpy.to(dtype=root_rpy.dtype, device=root_rpy.device), root_rpy)
        foot_pos = torch.where(bad_4, fallback.foot_pos.to(dtype=foot_pos.dtype, device=foot_pos.device), foot_pos)
        joint_seq = torch.where(bad_3, fallback.joint_angles.to(dtype=joint_seq.dtype, device=joint_seq.device), joint_seq)
        contact_state = torch.where(bad_3, fallback.contact_state.to(device=contact_state.device), contact_state)
        touchdown_seq = torch.where(
            bad_4,
            fallback.touchdown_seq.to(dtype=touchdown_seq.dtype, device=touchdown_seq.device),
            touchdown_seq,
        )
        planned_touchdown_w = torch.where(
            bad_4,
            fallback.planned_touchdown_w.to(dtype=planned_touchdown_w.dtype, device=planned_touchdown_w.device),
            planned_touchdown_w,
        )
        cost_total = torch.where(finite_ok, cost_total, torch.full_like(cost_total, 1.0e6))
        cost_breakdown["cost_total"] = cost_total
    status = torch.where(finite_ok, status, torch.full_like(status, int(MpcPlannerStatus.ALL_INFEASIBLE)))
    feasible = torch.logical_and(feasible, finite_ok)
    safe_fallback = torch.logical_or(safe_fallback, torch.logical_not(finite_ok))
    if profile is not None:
        profile.add_stage("parametric.export", (profile.now() - export_t0) * 1000.0)
    return MpcPlannerResult(
        root_pos=root_pos,
        root_rpy=root_rpy,
        foot_pos=foot_pos,
        joint_angles=joint_seq,
        contact_state=contact_state,
        touchdown_seq=touchdown_seq,
        planned_touchdown_w=planned_touchdown_w,
        cost_total=cost_total,
        cost_breakdown=cost_breakdown,
        status=status,
        feasible=feasible,
        safe_fallback=safe_fallback,
        loss_breakdown=loss_breakdown if cfg.diagnostics.enabled else None,
        hard_reason_mask=hard_reason_mask if cfg.diagnostics.enabled else None,
    )


def _cyclic_phase_distance(a: Tensor, b: Tensor) -> Tensor:
    return torch.abs(torch.remainder(a - b + 0.5, 1.0) - 0.5)


def _project_parametric_high_large_root_corridor(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    command: Tensor,
    cfg: MpcPlannerCfg,
    *,
    root_yaw: Tensor | None = None,
) -> Tensor:
    batch, horizon = int(root_pos.shape[0]), int(root_pos.shape[1])
    dtype = root_pos.dtype
    device = root_pos.device
    if terrain.semantic_map is None or horizon <= 0:
        return root_pos
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    if int(height.shape[0]) == 1 and batch > 1:
        height = height.expand(batch, -1, -1)
    if int(semantic.shape[0]) != batch or int(height.shape[0]) != batch:
        return root_pos

    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    yaw_active = torch.abs(cmd[:, 2]) > 1.0e-4
    active = torch.logical_or(speed > 1.0e-4, yaw_active)
    yaw = torch.zeros((batch,), dtype=dtype, device=device) if root_yaw is None else torch.as_tensor(root_yaw, dtype=dtype, device=device).reshape(-1)
    heading, left, _linear_active = command_frame_axes(cmd, yaw, linear_eps=1.0e-4)

    grid_xy = _terrain_grid_world_xy(terrain, dtype=dtype, device=device)
    grid_z = height.reshape(batch, -1)
    grid_sem = semantic.reshape(batch, -1)
    root_ground0 = height_at(terrain, root_pos[:, :1, :2]).reshape(batch).to(dtype=dtype, device=device)
    small_ids = tuple(int(v) for v in cfg.losses.touchdown_semantic.small_ids)
    large_ids = tuple(int(v) for v in cfg.losses.touchdown_semantic.large_ids)
    small_mask = torch.zeros_like(grid_sem, dtype=torch.bool)
    for semantic_id in small_ids:
        small_mask = torch.logical_or(small_mask, grid_sem == int(semantic_id))
    large_mask = torch.zeros_like(grid_sem, dtype=torch.bool)
    for semantic_id in large_ids:
        large_mask = torch.logical_or(large_mask, grid_sem == int(semantic_id))
    high_small = torch.logical_and(
        small_mask,
        (grid_z - root_ground0[:, None]) > float(cfg.losses.low_small_crossing.high_small_relative_height_m),
    )
    risky = torch.logical_or(large_mask, high_small)
    delta0 = grid_xy - root_pos[:, :1, :2]
    along0 = (delta0 * heading[:, None, :]).sum(dim=-1)
    lateral0 = (delta0 * left[:, None, :]).sum(dim=-1)
    linear_candidate = torch.logical_and(
        torch.logical_and(along0 >= -0.20, along0 <= float(cfg.losses.high_obstacle_avoidance.forward_distance_m)),
        torch.abs(lateral0) <= float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
    )
    yaw_candidate = torch.logical_and(yaw_active[:, None], torch.linalg.vector_norm(delta0, dim=-1) <= 0.75)
    candidate = torch.logical_and(risky, torch.logical_or(linear_candidate, yaw_candidate))
    candidate = torch.logical_and(candidate, active[:, None])
    score = torch.where(candidate, along0.abs() + 0.25 * lateral0.abs(), torch.full_like(along0, 1.0e6))
    idx = score.argmin(dim=1)
    valid = score.gather(1, idx[:, None]).squeeze(1) < 1.0e5
    if not bool(valid.any().item()):
        return root_pos
    obstacle_xy = grid_xy.gather(1, idx[:, None, None].expand(batch, 1, 2)).squeeze(1)
    obstacle_lateral = lateral0.gather(1, idx[:, None]).squeeze(1)
    side = torch.where(obstacle_lateral > 0.0, -torch.ones_like(obstacle_lateral), torch.ones_like(obstacle_lateral))
    margin = 0.5 * float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m) + 0.20
    safe_lateral = torch.full((batch,), float(margin), dtype=dtype, device=device) * side
    rel = root_pos[..., :2] - obstacle_xy[:, None, :]
    along = (rel * heading[:, None, :]).sum(dim=-1)
    lateral = (rel * left[:, None, :]).sum(dim=-1)
    influence = torch.relu(1.0 - torch.abs(along) / 0.55)
    phase = torch.linspace(0.0, 1.0, horizon, dtype=dtype, device=device).view(1, horizon)
    ramp = phase * phase * (3.0 - 2.0 * phase)
    deficit = torch.relu(side[:, None] * (safe_lateral[:, None] - lateral))
    projected_lateral = lateral + side[:, None] * deficit * torch.maximum(influence, yaw_active[:, None].to(dtype=dtype)) * ramp
    projected_xy = obstacle_xy[:, None, :] + heading[:, None, :] * along[..., None] + left[:, None, :] * projected_lateral[..., None]
    projected_xy = torch.where(valid[:, None, None], projected_xy, root_pos[..., :2])
    out = root_pos.clone()
    out[..., :2] = projected_xy
    root_ground = height_at(terrain, projected_xy).to(dtype=dtype, device=device)
    height_offset = (root_pos[..., 2] - height_at(terrain, root_pos[..., :2]).to(dtype=dtype, device=device)).clamp(0.26, 0.42)
    out[..., 2] = root_ground + height_offset
    out[:, 0, :] = root_pos[:, 0, :]
    return out


def _optimize_parametric_variables(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    planning_command: Tensor,
    *,
    loss_command: Tensor | None = None,
    variables,
    nominal,
    horizon: int,
    cfg: MpcPlannerCfg,
    profile: MpcProfile | None = None,
):
    command = planning_command if loss_command is None else loss_command
    steps = int(cfg.runtime.optimize_steps)
    root_yaw0 = torch.as_tensor(state.root_rpy, dtype=torch.as_tensor(state.root_pos).dtype, device=torch.as_tensor(state.root_pos).device)[:, 2]
    semantic_context = _semantic_avoidance_context(
        terrain,
        torch.as_tensor(state.root_pos, dtype=torch.as_tensor(state.root_pos).dtype, device=torch.as_tensor(state.root_pos).device)[:, :2],
        command,
        root_yaw=root_yaw0,
    )
    low_small_circles = _low_small_circles_context(
        terrain,
        max_components=int(cfg.losses.touchdown_keepout.low_small_circle_max_components),
        device=torch.as_tensor(state.root_pos).device,
    )
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=horizon)
    loss_t0 = profile.now() if profile is not None else 0.0
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=decoded.target_foot_pos,
        target_foot_pos=decoded.target_foot_pos,
        decoded=decoded,
        cfg=cfg,
        semantic_avoidance_context=semantic_context,
        low_small_circles=low_small_circles,
        profile=profile,
    )
    if profile is not None:
        profile.add_loss("sampled_frame", (profile.now() - loss_t0) * 1000.0)
    if steps <= 0:
        return decoded, losses
    optimizer = torch.optim.Adam(variables.parameters(), lr=float(cfg.runtime.lr))
    if profile is not None:
        profile.optimize_iters = int(steps)
    for _ in range(steps):
        iter_t0 = profile.now() if profile is not None else 0.0
        optimizer.zero_grad(set_to_none=True)
        decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=horizon)
        loss_t0 = profile.now() if profile is not None else 0.0
        losses = _parametric_sampled_frame_losses(
            terrain,
            state,
            command,
            root_pos=decoded.root_pos,
            foot_pos=decoded.target_foot_pos,
            target_foot_pos=decoded.target_foot_pos,
            decoded=decoded,
            cfg=cfg,
            semantic_avoidance_context=semantic_context,
            low_small_circles=low_small_circles,
            profile=profile,
        )
        if profile is not None:
            profile.add_loss("sampled_frame", (profile.now() - loss_t0) * 1000.0)
        total = sum(losses.values(), torch.zeros_like(next(iter(losses.values())))).mean()
        total.backward()
        torch.nn.utils.clip_grad_norm_(variables.parameters(), float(cfg.runtime.grad_clip_norm))
        optimizer.step()
        if profile is not None:
            profile.add_stage("opt.iter_total", (profile.now() - iter_t0) * 1000.0)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=horizon)
    loss_t0 = profile.now() if profile is not None else 0.0
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=decoded.target_foot_pos,
        target_foot_pos=decoded.target_foot_pos,
        decoded=decoded,
        cfg=cfg,
        semantic_avoidance_context=semantic_context,
        low_small_circles=low_small_circles,
        profile=profile,
    )
    if profile is not None:
        profile.add_loss("sampled_frame", (profile.now() - loss_t0) * 1000.0)
        profile.add_loss("total", profile.loss_ms.get("sampled_frame", 0.0))
    return decoded, losses


def _parametric_sampled_frame_losses(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    *,
    root_pos: Tensor,
    foot_pos: Tensor,
    target_foot_pos: Tensor,
    decoded,
    cfg: MpcPlannerCfg,
    semantic_avoidance_context: dict[str, Tensor] | None = None,
    low_small_circles: LowSmallCircles | None = None,
    profile: MpcProfile | None = None,
) -> dict[str, Tensor]:
    batch, horizon = int(root_pos.shape[0]), int(root_pos.shape[1])
    dtype = root_pos.dtype
    device = root_pos.device
    term_t0 = profile.now() if profile is not None else 0.0
    fk_joint = solve_joint_angles_from_trajectory(root_pos, decoded.root_rpy, target_foot_pos)
    if horizon > 0:
        state_joints = torch.as_tensor(state.joint_angles, dtype=dtype, device=device)
        fk_joint = fk_joint.clone()
        fk_joint[:, 0, :] = state_joints
    fk_foot = fk_feet_from_joint_angles(root_pos, decoded.root_rpy, fk_joint)
    if horizon > 0:
        state_foot = torch.as_tensor(state.foot_pos, dtype=dtype, device=device)
        fk_foot = fk_foot.clone()
        fk_foot[:, 0, :, :] = state_foot
    if profile is not None:
        profile.add_loss("term.fk_base", (profile.now() - term_t0) * 1000.0)
    term_t0 = profile.now() if profile is not None else 0.0
    target_fk_error = torch.linalg.vector_norm(target_foot_pos - foot_pos, dim=-1).mean(dim=(1, 2))
    terrain_z = height_at(terrain, foot_pos[..., :2].reshape(batch, horizon * 4, 2)).reshape(batch, horizon, 4).to(dtype=dtype, device=device)
    clearance_deficit = torch.relu(terrain_z + 0.015 - foot_pos[..., 2])
    terrain_clearance = clearance_deficit.square().mean(dim=(1, 2))
    semantic = semantic_at(terrain, foot_pos[..., :2].reshape(batch, horizon * 4, 2)).reshape(batch, horizon, 4).to(device=device)
    semantic_contact = (semantic != 0).to(dtype=dtype).mul(decoded.contact_prob.to(dtype=dtype)).mean(dim=(1, 2))
    if profile is not None:
        profile.add_loss("term.terrain_contact", (profile.now() - term_t0) * 1000.0)
    term_t0 = profile.now() if profile is not None else 0.0
    root_yaw0 = torch.as_tensor(state.root_rpy, dtype=dtype, device=device)[:, 2]
    semantic_avoidance = _parametric_semantic_avoidance_loss(
        terrain,
        root_pos,
        foot_pos,
        decoded.touchdown_w,
        command,
        root_yaw=root_yaw0,
        context=semantic_avoidance_context,
    )
    if profile is not None:
        profile.add_loss("term.semantic_avoidance", (profile.now() - term_t0) * 1000.0)
    term_t0 = profile.now() if profile is not None else 0.0
    if bool(cfg.losses.touchdown_keepout.enabled):
        touchdown_keepout = float(cfg.losses.touchdown_keepout.weight) * parametric_touchdown_keepout_loss(
            terrain,
            decoded.touchdown_w,
            radius_extra_m=float(cfg.losses.touchdown_keepout.touchdown_keepout_radius_extra_m),
            max_components=int(cfg.losses.touchdown_keepout.low_small_circle_max_components),
            low_small_circles=low_small_circles,
        )
    else:
        touchdown_keepout = torch.zeros((batch,), dtype=dtype, device=device)
    if bool(cfg.losses.swing_foot_clearance.enabled):
        swing_foot_clearance = float(cfg.losses.swing_foot_clearance.weight) * parametric_swing_foot_clearance_loss(
            terrain,
            target_foot_pos,
            decoded.swing_prob,
            margin_m=float(cfg.losses.swing_foot_clearance.swing_foot_clearance_margin_m),
        )
    else:
        swing_foot_clearance = torch.zeros((batch,), dtype=dtype, device=device)
    if profile is not None:
        profile.add_loss("term.touchdown_clearance", (profile.now() - term_t0) * 1000.0)
    term_t0 = profile.now() if profile is not None else 0.0
    touchdown_endpoint = _parametric_touchdown_endpoint_loss(
        terrain,
        foot_pos,
        decoded.touchdown_w,
        decoded.swing_center,
        decoded.swing_width,
        command,
        root_yaw=root_yaw0,
    )
    foot_height_guard = _parametric_foot_height_guard_loss(root_pos, foot_pos, decoded.swing_prob)
    contact_weight = decoded.contact_prob.to(dtype=dtype, device=device).clamp_min(0.0)
    stance_mass = contact_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-4)
    stance_center_xy = (contact_weight.unsqueeze(-1) * foot_pos[..., :2]).sum(dim=2) / stance_mass
    root_foot_center = 2.0 * (root_pos[..., :2] - stance_center_xy).square().sum(dim=-1).mean(dim=1)
    if profile is not None:
        profile.add_loss("term.endpoint_center", (profile.now() - term_t0) * 1000.0)
    term_t0 = profile.now() if profile is not None else 0.0
    if bool(cfg.losses.plane_root_z_target.enabled):
        plane_root_z_target = float(cfg.losses.plane_root_z_target.weight) * parametric_plane_root_z_target_loss(
            root_pos,
            torch.as_tensor(state.root_pos, dtype=dtype, device=device),
            terrain.is_plane_terrain,
            target_height_m=cfg.losses.plane_root_z_target.root_z_target_height_m,
        )
    else:
        plane_root_z_target = torch.zeros((batch,), dtype=dtype, device=device)
    if bool(cfg.losses.fk_body_leg_collision.enabled):
        fk_points = fk_leg_points_from_joint_angles(
            root_pos,
            decoded.root_rpy,
            fk_joint,
            shank_sample_count=int(cfg.losses.fk_body_leg_collision.shank_sample_count),
        )
        fk_body_leg_collision = float(cfg.losses.fk_body_leg_collision.weight) * parametric_fk_body_leg_collision_loss(
            terrain,
            root_pos,
            fk_points,
            margins=FkCollisionMargins(
                foot=float(cfg.losses.fk_body_leg_collision.foot_margin_m),
                knee=float(cfg.losses.fk_body_leg_collision.knee_margin_m),
                shank=float(cfg.losses.fk_body_leg_collision.shank_margin_m),
                root=float(cfg.losses.fk_body_leg_collision.root_margin_m),
                underbody=float(cfg.losses.fk_body_leg_collision.underbody_margin_m),
            ),
            underbody_sample_count=int(cfg.losses.fk_body_leg_collision.underbody_sample_count),
        )
    else:
        fk_body_leg_collision = torch.zeros((batch,), dtype=dtype, device=device)
    if profile is not None:
        profile.add_loss("term.plane_fk_body", (profile.now() - term_t0) * 1000.0)
    term_t0 = profile.now() if profile is not None else 0.0
    trajectory_fk_consistency = float(cfg.losses.ik_fk_residual.weight) * parametric_trajectory_fk_consistency_loss(
        root_pos,
        decoded.root_rpy,
        target_foot_pos,
        fk_foot,
    )
    if bool(cfg.losses.kinematics.enabled):
        joint_limit = float(cfg.losses.kinematics.weight) * joint_limit_loss_from_root_foot(
            root_pos,
            decoded.root_rpy,
            target_foot_pos,
            joint_limit_margin_rad=float(cfg.losses.kinematics.joint_limit_margin_rad),
        )
    else:
        joint_limit = torch.zeros((batch,), dtype=dtype, device=device)
    if profile is not None:
        profile.add_loss("term.trajectory_joint", (profile.now() - term_t0) * 1000.0)
    term_t0 = profile.now() if profile is not None else 0.0
    pair_same = _cyclic_phase_distance(decoded.swing_center[:, 0], decoded.swing_center[:, 3])
    pair_same = pair_same + _cyclic_phase_distance(decoded.swing_center[:, 1], decoded.swing_center[:, 2])
    pair_half = torch.abs(_cyclic_phase_distance(decoded.swing_center[:, 0], decoded.swing_center[:, 1]) - 0.5)
    width_match = torch.abs(decoded.swing_width[:, 0] - decoded.swing_width[:, 3]) + torch.abs(decoded.swing_width[:, 1] - decoded.swing_width[:, 2])
    gait_regularization = pair_same + pair_half + 0.25 * width_match
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    empty_semantic = torch.ones((batch,), dtype=torch.bool, device=device)
    if terrain.semantic_map is not None:
        semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
        if semantic.ndim == 2:
            semantic = semantic.unsqueeze(0)
        if int(semantic.shape[0]) == 1 and batch > 1:
            semantic = semantic.expand(batch, -1, -1)
        if int(semantic.shape[0]) == batch:
            empty_semantic = torch.count_nonzero(semantic.reshape(batch, -1), dim=1) == 0
    empty_linear = torch.logical_and(empty_semantic, speed > 1.0e-4)
    if bool(cfg.losses.swing_direction.enabled):
        swing_direction = float(cfg.losses.swing_direction.weight) * swing_direction_loss(
            root_pos,
            decoded.root_rpy,
            foot_pos,
            decoded.swing_center,
            decoded.swing_width,
            command,
            cfg.runtime,
        )
        swing_direction = torch.where(empty_linear, swing_direction, torch.zeros_like(swing_direction))
    else:
        swing_direction = torch.zeros((batch,), dtype=dtype, device=device)
    root_yaw = torch.as_tensor(state.root_rpy, dtype=dtype, device=device)[:, 2]
    heading, left, _linear_active = command_frame_axes(cmd, root_yaw, linear_eps=1.0e-6)
    root_delta = root_pos[:, -1, :2] - torch.as_tensor(state.root_pos, dtype=dtype, device=device)[:, :2]
    progress = (root_delta * heading).sum(dim=-1)
    target_progress = torch.maximum(
        torch.clamp(speed * 0.50, max=0.35),
        torch.full_like(speed, float(cfg.losses.progress.min_progress_m)),
    )
    progress_error = torch.relu(target_progress - progress).square()
    lateral_progress = (root_delta * left).sum(dim=-1)
    direction_error = torch.where(empty_linear, lateral_progress.square(), torch.zeros_like(progress_error))
    if bool(cfg.losses.progress.enabled):
        command_progress = float(cfg.losses.progress.weight) * (progress_error + direction_error)
    else:
        command_progress = torch.zeros((batch,), dtype=dtype, device=device)
    if horizon >= 3:
        root_acc = root_pos[:, 2:] - 2.0 * root_pos[:, 1:-1] + root_pos[:, :-2]
        foot_acc = target_foot_pos[:, 2:] - 2.0 * target_foot_pos[:, 1:-1] + target_foot_pos[:, :-2]
        curve_regularization = root_acc.square().mean(dim=(1, 2)) + foot_acc.square().mean(dim=(1, 2, 3))
    else:
        curve_regularization = torch.zeros((batch,), dtype=dtype, device=device)
    if profile is not None:
        profile.add_loss("term.gait_progress_curve", (profile.now() - term_t0) * 1000.0)
    return {
        "parametric_reachability": target_fk_error,
        "parametric_terrain_clearance": terrain_clearance,
        "parametric_semantic_contact": semantic_contact,
        "parametric_semantic_avoidance": semantic_avoidance,
        "parametric_touchdown_keepout": touchdown_keepout,
        "parametric_swing_foot_clearance": swing_foot_clearance,
        "parametric_touchdown_endpoint": touchdown_endpoint,
        "parametric_foot_height_guard": foot_height_guard,
        "parametric_root_foot_center": root_foot_center,
        "parametric_plane_root_z_target": plane_root_z_target,
        "parametric_fk_body_leg_collision": fk_body_leg_collision,
        "parametric_trajectory_fk_consistency": trajectory_fk_consistency,
        "parametric_joint_limit": joint_limit,
        "parametric_gait_regularization": gait_regularization,
        "parametric_swing_direction": swing_direction,
        "parametric_command_progress": command_progress,
        "parametric_curve_regularization": curve_regularization,
    }


def _parametric_semantic_avoidance_loss(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    foot_pos: Tensor,
    touchdown_w: Tensor,
    command: Tensor,
    *,
    root_yaw: Tensor,
    context: dict[str, Tensor] | None = None,
) -> Tensor:
    batch = int(root_pos.shape[0])
    dtype = root_pos.dtype
    device = root_pos.device
    zero = torch.zeros((batch,), dtype=dtype, device=device)
    if context is not None:
        field = torch.as_tensor(context["field"], dtype=dtype, device=device)
        active = torch.as_tensor(context["active"], dtype=torch.bool, device=device)
    else:
        field = None
        active = None
    if terrain.semantic_map is None:
        return zero
    if field is None or active is None:
        built_context = _semantic_avoidance_context(terrain, root_pos[:, 0, :2], command, root_yaw=root_yaw)
        if built_context is None:
            return zero
        field = torch.as_tensor(built_context["field"], dtype=dtype, device=device)
        active = torch.as_tensor(built_context["active"], dtype=torch.bool, device=device)

    root_risk = _sample_proximity_field_at_world_xy(
        field,
        root_pos[..., :2],
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=terrain.sensor_pos_w,
        sensor_yaw=terrain.sensor_yaw,
    )
    foot_risk = _sample_proximity_field_at_world_xy(
        field,
        foot_pos[..., :2],
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=terrain.sensor_pos_w,
        sensor_yaw=terrain.sensor_yaw,
    )
    touchdown_risk = _sample_proximity_field_at_world_xy(
        field,
        touchdown_w[..., :2],
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=terrain.sensor_pos_w,
        sensor_yaw=terrain.sensor_yaw,
    )
    root_cost = root_risk.mean(dim=1)
    foot_cost = foot_risk.mean(dim=(1, 2))
    touchdown_cost = touchdown_risk.mean(dim=1)
    loss = 30.0 * root_cost + 20.0 * foot_cost + 25.0 * touchdown_cost
    return torch.where(active, loss, zero)


def _parametric_touchdown_endpoint_loss(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    touchdown_w: Tensor,
    swing_center: Tensor,
    swing_width: Tensor,
    command: Tensor,
    *,
    root_yaw: Tensor,
) -> Tensor:
    del terrain
    batch = int(foot_pos.shape[0])
    dtype = foot_pos.dtype
    device = foot_pos.device
    sampled = sample_touchdown_positions(foot_pos, swing_center, swing_width)
    endpoint_error = torch.linalg.vector_norm(sampled - touchdown_w, dim=-1).square().mean(dim=1)
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    heading, _left, _linear_active = command_frame_axes(cmd, root_yaw, linear_eps=1.0e-6)
    foot_along = ((foot_pos[..., :2] - touchdown_w[:, None, :, :2]) * heading[:, None, None, :]).sum(dim=-1)
    behind = torch.relu(foot_along.amax(dim=1) - 0.02).square().mean(dim=1)
    return torch.where(speed > 1.0e-4, 8.0 * endpoint_error + 10.0 * behind, torch.zeros((batch,), dtype=dtype, device=device))


def _parametric_foot_height_guard_loss(root_pos: Tensor, foot_pos: Tensor, swing_prob: Tensor) -> Tensor:
    root_limit = root_pos[..., 2, None] - 0.02
    over = torch.relu(foot_pos[..., 2] - root_limit)
    return 12.0 * (over.square() * swing_prob.to(dtype=foot_pos.dtype, device=foot_pos.device)).mean(dim=(1, 2))


def plan_segment(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    *,
    cfg: MpcPlannerCfg,
) -> MpcPlannerResult:
    """Plan one horizon for a batch of environments."""
    validate_mpc_config(cfg)
    profile = (
        MpcProfile(sync_cuda=bool(cfg.diagnostics.profile_cuda_sync))
        if should_profile_mpc(cfg)
        else None
    )
    plan_t0 = profile.now() if profile is not None else 0.0
    with torch.inference_mode(False):
        normalize_t0 = profile.now() if profile is not None else 0.0
        terrain = _normal_terrain(terrain)
        state = _normal_state(state)
        command = _normal_tensor(command)
        if profile is not None:
            profile.add_stage("plan.normalize", (profile.now() - normalize_t0) * 1000.0)
    batch_for_zero = int(state.root_pos.shape[0])
    horizon_for_zero = int(cfg.runtime.horizon_steps)
    zero_mask_pre = _zero_command_mask(command, batch=batch_for_zero, device=state.root_pos.device)
    if bool(torch.all(zero_mask_pre)):
        if profile is not None:
            profile.batch_size = batch_for_zero
            profile.horizon = horizon_for_zero
            profile.add_stage("plan.zero_command_standstill", (profile.now() - plan_t0) * 1000.0)
            profile.add_stage("plan.total", (profile.now() - plan_t0) * 1000.0)
            maybe_print_mpc_profile(profile, cfg=cfg)
        return _standstill_result_from_state(state, horizon=horizon_for_zero, cfg=cfg, terrain=terrain)
    if bool(torch.any(zero_mask_pre)):
        nonzero_ids = torch.nonzero(torch.logical_not(zero_mask_pre), as_tuple=False).squeeze(-1)
        base = _standstill_result_from_state(state, horizon=horizon_for_zero, cfg=cfg, terrain=terrain)
        subset = plan_segment(
            _subset_terrain(terrain, nonzero_ids),
            _subset_state(state, nonzero_ids),
            command.index_select(0, nonzero_ids),
            cfg=cfg,
        )
        merged = _merge_subset_result(base, subset, nonzero_ids, diagnostics_enabled=bool(cfg.diagnostics.enabled))
        if profile is not None:
            profile.batch_size = batch_for_zero
            profile.horizon = horizon_for_zero
            profile.add_stage("plan.mixed_zero_split", (profile.now() - plan_t0) * 1000.0)
            profile.add_stage("plan.total", (profile.now() - plan_t0) * 1000.0)
            maybe_print_mpc_profile(profile, cfg=cfg)
        return merged
    with torch.inference_mode(False), torch.enable_grad():
        result = _parametric_result_from_state(terrain, state, command, cfg=cfg, profile=profile)
    if profile is not None:
        profile.batch_size = batch_for_zero
        profile.horizon = horizon_for_zero
        profile.add_stage("plan.parametric", (profile.now() - plan_t0) * 1000.0)
        profile.add_stage("plan.total", (profile.now() - plan_t0) * 1000.0)
        maybe_print_mpc_profile(profile, cfg=cfg)
    return result


__all__ = ["plan_segment", "sample_touchdown_positions"]
