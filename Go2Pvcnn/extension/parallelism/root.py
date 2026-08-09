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


def _half_profile(cfg: ParallelismCfg, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    tau = torch.linspace(0.0, 1.0, int(cfg.half_cycle), dtype=dtype, device=device)
    return tau * tau * (3.0 - 2.0 * tau)


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
    pitch_offset = forward * float(cfg.terrain_following_pitch_sample_m)
    roll_offset = left * float(cfg.terrain_following_roll_sample_m)
    points = torch.cat(
        (
            root_xy + pitch_offset,
            root_xy - pitch_offset,
            root_xy + roll_offset,
            root_xy - roll_offset,
        ),
        dim=1,
    )
    height = query_height_semantic_valid(terrain, points).height.reshape(root_xy.shape[0], 4, int(cfg.horizon))
    h_front, h_back, h_left, h_right = height[:, 0], height[:, 1], height[:, 2], height[:, 3]
    pitch = -torch.atan2(
        h_front - h_back,
        torch.full_like(h_front, 2.0 * max(float(cfg.terrain_following_pitch_sample_m), 1.0e-6)),
    )
    roll = torch.atan2(
        h_left - h_right,
        torch.full_like(h_left, 2.0 * max(float(cfg.terrain_following_roll_sample_m), 1.0e-6)),
    )
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
    flat_command = clamp_command(command_input, cfg)
    terrain_command = soft_clamp_terrain_command(command_input, cfg)
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
