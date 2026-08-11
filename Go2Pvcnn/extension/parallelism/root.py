from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.parallelism.config import ParallelismCfg
from extension.parallelism.kinematics import fk_go2
from extension.parallelism.terrain import query_height_semantic_valid
from extension.parallelism.types import ParallelismState, ParallelismTerrain


@dataclass(frozen=True)
class RootRollout:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    clamped_command_body: Tensor


def clamp_command(command_body: Tensor, cfg: ParallelismCfg) -> Tensor:
    command = torch.as_tensor(command_body)
    limits = torch.tensor(
        [cfg.vx_limit, cfg.vy_limit, cfg.vyaw_limit],
        dtype=command.dtype,
        device=command.device,
    )
    return torch.maximum(torch.minimum(command, limits), -limits)


def soft_clamp_terrain_command(command_body: Tensor, cfg: ParallelismCfg) -> Tensor:
    command = clamp_command(command_body, cfg)
    limits = torch.tensor(
        [
            cfg.terrain_following_vx_soft_limit,
            cfg.terrain_following_vy_soft_limit,
            cfg.terrain_following_vyaw_soft_limit,
        ],
        dtype=command.dtype,
        device=command.device,
    ).clamp_min(0.0)
    scales = torch.tensor(
        [
            cfg.terrain_following_vx_excess_scale,
            cfg.terrain_following_vy_excess_scale,
            cfg.terrain_following_vyaw_excess_scale,
        ],
        dtype=command.dtype,
        device=command.device,
    ).clamp_min(0.0)
    magnitude = command.abs()
    softened = limits + (magnitude - limits).clamp_min(0.0) * scales
    magnitude = torch.where(magnitude > limits, softened, magnitude)
    return command.sign() * magnitude


def _terrain_grid_world_xy(terrain: ParallelismTerrain, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    batch, height_count, width_count = terrain.height_w.shape
    row = torch.arange(height_count, dtype=dtype, device=device)
    col = torch.arange(width_count, dtype=dtype, device=device)
    grid_row, grid_col = torch.meshgrid(row, col, indexing="ij")
    local_x = grid_col * float(terrain.resolution)
    local_y = grid_row * float(terrain.resolution)
    yaw = terrain.yaw_w.to(dtype=dtype, device=device).reshape(batch, 1, 1)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    world_x = cosine * local_x[None] - sine * local_y[None] + terrain.origin_w[:, None, None, 0].to(
        dtype=dtype,
        device=device,
    )
    world_y = sine * local_x[None] + cosine * local_y[None] + terrain.origin_w[:, None, None, 1].to(
        dtype=dtype,
        device=device,
    )
    return torch.stack((world_x, world_y), dim=-1)


def _large_obstacle_avoidance_command(
    state: ParallelismState,
    command_body: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
) -> Tensor:
    root0 = torch.as_tensor(state.root_pos_w)
    command = torch.as_tensor(command_body, dtype=root0.dtype, device=root0.device)
    if command.shape[-1] != 3:
        raise ValueError("command_body must have shape [B, 3]")
    batch = int(root0.shape[0])
    if int(terrain.semantic_id.shape[0]) != batch:
        raise ValueError("terrain batch size must match state root batch size")

    eps = torch.tensor(1.0e-6, dtype=command.dtype, device=command.device)
    rpy0 = torch.as_tensor(state.root_rpy_w, dtype=command.dtype, device=command.device)
    yaw = rpy0[:, 2]
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)

    vx_b = command[:, 0]
    vy_b = command[:, 1]
    vx_w = cosine * vx_b - sine * vy_b
    vy_w = sine * vx_b + cosine * vy_b
    speed_xy = torch.sqrt(vx_w.square() + vy_w.square())
    moving = speed_xy > eps
    forward = torch.stack((vx_w, vy_w), dim=-1) / speed_xy.clamp_min(eps)[:, None]
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)

    grid_xy = _terrain_grid_world_xy(terrain, dtype=command.dtype, device=command.device)
    delta = grid_xy - root0[:, None, None, :2].to(dtype=command.dtype, device=command.device)
    forward_distance = (delta * forward[:, None, None, :]).sum(dim=-1)
    lateral_position = (delta * left[:, None, None, :]).sum(dim=-1)

    rect_half_width = 0.5 * float(cfg.large_obstacle_rect_width_m)
    rect_length = max(float(cfg.large_obstacle_rect_length_m), 1.0e-6)
    semantic = terrain.semantic_id.to(device=command.device)
    valid = terrain.valid_mask.to(device=command.device)
    front_large = (
        (semantic == 2)
        & valid
        & moving[:, None, None]
        & (forward_distance >= 0.0)
        & (forward_distance <= rect_length)
        & (lateral_position.abs() <= rect_half_width)
    )
    has_front_large = front_large.any(dim=(-1, -2))

    sigma = max(rect_half_width, 1.0e-6)
    lateral_weight = torch.exp(-lateral_position.square() / (2.0 * sigma * sigma))
    mask_weight = front_large.to(dtype=command.dtype) * lateral_weight
    weight_sum = mask_weight.sum(dim=(-1, -2))
    mean_l = (mask_weight * lateral_position).sum(dim=(-1, -2)) / weight_sum.clamp_min(float(eps.item()))

    infinity = torch.full_like(forward_distance, torch.inf)
    nearest_s = torch.where(front_large, forward_distance, infinity).amin(dim=(-1, -2))
    proximity = (1.0 - nearest_s / rect_length).clamp(0.0, 1.0)
    lateral_speed = float(cfg.large_obstacle_lateral_speed_max_mps) * proximity
    lateral_speed = torch.where(has_front_large, lateral_speed, torch.zeros_like(lateral_speed))

    default_side = 1.0 if int(cfg.large_obstacle_default_side) >= 0 else -1.0
    side = torch.where(
        mean_l > eps,
        torch.full_like(mean_l, -1.0),
        torch.where(mean_l < -eps, torch.full_like(mean_l, 1.0), torch.full_like(mean_l, default_side)),
    )
    avoid_world = side[:, None] * lateral_speed[:, None] * left
    avoid_body_x = cosine * avoid_world[:, 0] + sine * avoid_world[:, 1]
    avoid_body_y = -sine * avoid_world[:, 0] + cosine * avoid_world[:, 1]

    corrected = command.clone()
    corrected[:, 0] = corrected[:, 0] + avoid_body_x
    corrected[:, 1] = corrected[:, 1] + avoid_body_y
    return clamp_command(corrected, cfg)


def _half_profile(cfg: ParallelismCfg, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    tau = torch.linspace(0.0, 1.0, int(cfg.half_cycle), dtype=dtype, device=device)
    return tau * tau * (3.0 - 2.0 * tau)


def _uniform_symmetric_samples(
    range_m: float,
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    count = max(int(count), 1)
    extent = max(float(range_m), 0.0)
    if count == 1:
        return torch.zeros(1, dtype=dtype, device=device)
    denom = torch.tensor(float(count - 1), dtype=dtype, device=device)
    index = torch.arange(count, dtype=dtype, device=device)
    return (index * 2.0 / denom - 1.0) * extent


def _fit_height_slope(heights: Tensor, offsets: Tensor) -> Tensor:
    heights = torch.as_tensor(heights)
    offsets = torch.as_tensor(offsets, dtype=heights.dtype, device=heights.device)
    if heights.ndim < 2 or offsets.ndim < 2:
        raise ValueError("heights and offsets must include a sample dimension")
    if heights.shape[-2] != offsets.shape[-2]:
        raise ValueError("heights and offsets must have the same sample count")
    x = offsets
    if x.shape[0] == 1 and heights.shape[0] != 1:
        x = x.expand(heights.shape[0], *x.shape[1:])
    x = x.expand(*heights.shape[:-2], x.shape[-2], x.shape[-1])
    y = heights
    x_mean = x.mean(dim=-2, keepdim=True)
    y_mean = y.mean(dim=-2, keepdim=True)
    centered_x = x - x_mean
    centered_y = y - y_mean
    denominator = (centered_x.square()).sum(dim=-2).clamp_min(1.0e-8)
    numerator = (centered_x * centered_y).sum(dim=-2)
    return numerator / denominator


def _smooth_rate_limit(
    target: Tensor,
    initial: Tensor,
    *,
    smoothing: float,
    rate_limit: float,
    deadband: float = 0.0,
) -> Tensor:
    alpha = float(max(0.0, min(float(smoothing), 1.0)))
    limit = float(max(float(rate_limit), 0.0))
    band = float(max(float(deadband), 0.0))
    out = target.clone()
    out[:, 0] = initial
    for frame in range(1, int(target.shape[1])):
        prev = out[:, frame - 1]
        desired = target[:, frame]
        if band > 0.0:
            desired = torch.where((desired - prev).abs() < band, prev, desired)
        blended = prev + (desired - prev) * alpha
        delta = (blended - prev).clamp(min=-limit, max=limit) if limit > 0.0 else blended - prev
        out[:, frame] = prev + delta
    return out


def _rollout_xy_yaw(root0: Tensor, rpy0: Tensor, command: Tensor, cfg: ParallelismCfg) -> tuple[Tensor, Tensor]:
    half = _half_profile(cfg, dtype=root0.dtype, device=root0.device)
    disp_half = command[:, :2] * (float(cfg.half_cycle) * float(cfg.dt))
    yaw_half = command[:, 2] * (float(cfg.half_cycle) * float(cfg.dt))
    first_xy_body = half[None, :, None] * disp_half[:, None, :]
    second_xy_body = disp_half[:, None, :] + half[None, :, None] * disp_half[:, None, :]
    xy_body = torch.cat((first_xy_body, second_xy_body), dim=1)
    first_yaw = half[None, :] * yaw_half[:, None]
    second_yaw = yaw_half[:, None] + half[None, :] * yaw_half[:, None]
    yaw_delta = torch.cat((first_yaw, second_yaw), dim=1)
    yaw = rpy0[:, 2:3] + yaw_delta
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    world_dx = cosine * xy_body[..., 0] - sine * xy_body[..., 1]
    world_dy = sine * xy_body[..., 0] + cosine * xy_body[..., 1]
    return root0[:, None, :2] + torch.stack((world_dx, world_dy), dim=-1), yaw


def _flat_root_z(state: ParallelismState, root0: Tensor, rpy0: Tensor, terrain: ParallelismTerrain, cfg: ParallelismCfg) -> Tensor:
    joint = torch.as_tensor(state.joint_pos, dtype=root0.dtype, device=root0.device)
    foot0 = (
        torch.as_tensor(state.foot_pos_w, dtype=root0.dtype, device=root0.device)
        if state.foot_pos_w is not None
        else fk_go2(root0, rpy0, joint).foot_pos_w
    )
    stance_first = query_height_semantic_valid(terrain, foot0[:, (1, 2), :2]).height.mean(dim=1)
    stance_second = query_height_semantic_valid(terrain, foot0[:, (0, 3), :2]).height.mean(dim=1)
    return torch.cat(
        (
            stance_first[:, None].expand(-1, int(cfg.half_cycle)),
            stance_second[:, None].expand(-1, int(cfg.half_cycle)),
        ),
        dim=1,
    ) + float(cfg.root_clearance_m)


def _level_root_rpy(rpy0: Tensor, yaw: Tensor, cfg: ParallelismCfg) -> Tensor:
    root_rpy = torch.zeros(yaw.shape[0], int(cfg.horizon), 3, dtype=rpy0.dtype, device=rpy0.device)
    leveling_frames = max(int(cfg.root_leveling_frames), 1)
    frame = torch.arange(int(cfg.horizon), dtype=rpy0.dtype, device=rpy0.device)
    u = torch.clamp(frame / float(leveling_frames), min=0.0, max=1.0)
    smoothstep = u * u * (3.0 - 2.0 * u)
    level_scale = (1.0 - smoothstep)[None, :]
    root_rpy[..., 0] = rpy0[:, None, 0] * level_scale
    root_rpy[..., 1] = rpy0[:, None, 1] * level_scale
    root_rpy[..., 2] = yaw
    return root_rpy


def _terrain_following_root_z(root0: Tensor, root_xy: Tensor, terrain: ParallelismTerrain, cfg: ParallelismCfg) -> Tensor:
    query = query_height_semantic_valid(terrain, root_xy.reshape(root_xy.shape[0], -1, 2))
    terrain_z = query.height.reshape(root_xy.shape[0], int(cfg.horizon))
    target = terrain_z + float(cfg.terrain_following_root_clearance_m)
    return _smooth_rate_limit(
        target,
        root0[:, 2],
        smoothing=float(cfg.terrain_following_root_z_smoothing),
        rate_limit=float(cfg.terrain_following_root_z_rate_limit_m),
        deadband=float(cfg.terrain_following_root_height_deadband_m),
    )


def _terrain_following_rpy(root_xy: Tensor, yaw: Tensor, rpy0: Tensor, terrain: ParallelismTerrain, cfg: ParallelismCfg) -> Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    forward = torch.stack((cosine, sine), dim=-1)
    left = torch.stack((-sine, cosine), dim=-1)
    pitch_offsets = _uniform_symmetric_samples(
        cfg.terrain_following_pitch_sample_range_m,
        cfg.terrain_following_pitch_sample_count,
        dtype=root_xy.dtype,
        device=root_xy.device,
    )
    roll_offsets = _uniform_symmetric_samples(
        cfg.terrain_following_roll_sample_range_m,
        cfg.terrain_following_roll_sample_count,
        dtype=root_xy.dtype,
        device=root_xy.device,
    )
    pitch_points = root_xy[:, None, :, :] + pitch_offsets[None, :, None, None] * forward[:, None, :, :]
    roll_points = root_xy[:, None, :, :] + roll_offsets[None, :, None, None] * left[:, None, :, :]
    batch = int(root_xy.shape[0])
    horizon = int(root_xy.shape[1])
    pitch_count = int(pitch_offsets.numel())
    roll_count = int(roll_offsets.numel())
    pitch_height = query_height_semantic_valid(
        terrain,
        pitch_points.reshape(batch, pitch_count * horizon, 2),
    ).height.reshape(batch, pitch_count, horizon)
    roll_height = query_height_semantic_valid(
        terrain,
        roll_points.reshape(batch, roll_count * horizon, 2),
    ).height.reshape(batch, roll_count, horizon)
    pitch_slope = _fit_height_slope(
        pitch_height,
        pitch_offsets.reshape(1, -1, 1),
    )
    roll_slope = _fit_height_slope(
        roll_height,
        roll_offsets.reshape(1, -1, 1),
    )
    pitch = -torch.atan(pitch_slope)
    roll = torch.atan(roll_slope)
    deadband = float(max(cfg.terrain_following_rpy_deadband_rad, 0.0))
    pitch = torch.where(pitch.abs() < deadband, torch.zeros_like(pitch), pitch)
    roll = torch.where(roll.abs() < deadband, torch.zeros_like(roll), roll)
    roll = roll.clamp(min=-float(cfg.terrain_following_roll_limit_rad), max=float(cfg.terrain_following_roll_limit_rad))
    pitch = pitch.clamp(min=-float(cfg.terrain_following_pitch_limit_rad), max=float(cfg.terrain_following_pitch_limit_rad))
    roll = _smooth_rate_limit(
        roll,
        rpy0[:, 0],
        smoothing=float(cfg.terrain_following_rpy_smoothing),
        rate_limit=float(cfg.terrain_following_rpy_rate_limit_rad),
    )
    pitch = _smooth_rate_limit(
        pitch,
        rpy0[:, 1],
        smoothing=float(cfg.terrain_following_rpy_smoothing),
        rate_limit=float(cfg.terrain_following_rpy_rate_limit_rad),
    )
    return torch.stack((roll, pitch, yaw), dim=-1)


def rollout_root(
    state: ParallelismState,
    command_body: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
    terrain_following_mask: Tensor | None = None,
) -> RootRollout:
    root0 = torch.as_tensor(state.root_pos_w)
    rpy0 = torch.as_tensor(state.root_rpy_w, dtype=root0.dtype, device=root0.device)
    command_input = torch.as_tensor(command_body, dtype=root0.dtype, device=root0.device)
    command_avoid = _large_obstacle_avoidance_command(state, command_input, terrain, cfg)
    flat_command = clamp_command(command_avoid, cfg)
    terrain_command = soft_clamp_terrain_command(command_avoid, cfg)
    if terrain_following_mask is None:
        mask = torch.zeros(root0.shape[0], dtype=torch.bool, device=root0.device)
    else:
        mask = torch.as_tensor(terrain_following_mask, dtype=torch.bool, device=root0.device).reshape(root0.shape[0])
    command = torch.where(mask[:, None], terrain_command, flat_command)
    root_xy, yaw = _rollout_xy_yaw(root0, rpy0, command, cfg)
    flat_z = _flat_root_z(state, root0, rpy0, terrain, cfg)
    terrain_z = _terrain_following_root_z(root0, root_xy, terrain, cfg)
    z = torch.where(mask[:, None], terrain_z, flat_z)
    root_pos = torch.cat((root_xy, z[..., None]), dim=-1)
    flat_rpy = _level_root_rpy(rpy0, yaw, cfg)
    terrain_rpy = _terrain_following_rpy(root_xy, yaw, rpy0, terrain, cfg)
    root_rpy = torch.where(mask[:, None, None], terrain_rpy, flat_rpy)
    return RootRollout(root_pos_w=root_pos, root_rpy_w=root_rpy, clamped_command_body=command)
