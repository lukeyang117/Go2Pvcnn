"""Nominal trajectory builders for the MPC optimizer."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import MpcRuntimeCfg
from .terrain import height_at
from .types import MpcPlannerTerrain, MpcRobotState


def _as_command(command: Tensor, *, device: torch.device) -> Tensor:
    cmd = torch.as_tensor(command, dtype=torch.float32, device=device)
    if cmd.ndim != 2:
        raise ValueError(f"command must have shape [B, C], got {tuple(cmd.shape)}")
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((cmd.shape[0], 3 - cmd.shape[-1]), dtype=cmd.dtype, device=cmd.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    return cmd[:, :3]


def _rotate_body_xy_to_world(xy: Tensor, yaw: Tensor) -> Tensor:
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    return torch.stack((cy * xy[..., 0] - sy * xy[..., 1], sy * xy[..., 0] + cy * xy[..., 1]), dim=-1)


def _rotate_world_xy_to_body(xy: Tensor, yaw: Tensor) -> Tensor:
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    return torch.stack((cy * xy[..., 0] + sy * xy[..., 1], -sy * xy[..., 0] + cy * xy[..., 1]), dim=-1)


def _gather_time(values: Tensor, frame_idx: Tensor) -> Tensor:
    """Gather [B,T,...] values with [B,4] frame ids into [B,4,...]."""
    batch, _, *tail = values.shape
    legs = int(frame_idx.shape[1])
    view_shape = (batch, legs, *tail)
    gather_idx = frame_idx.view(batch, legs, *([1] * len(tail))).expand(view_shape)
    return values.gather(1, gather_idx)


def _forward_phase_distance(start: Tensor, end: Tensor) -> Tensor:
    return torch.remainder(end - start, 1.0)


def _circular_progress(phase: Tensor, start: Tensor, width: Tensor) -> Tensor:
    return torch.clamp(_forward_phase_distance(start.unsqueeze(1), phase) / torch.clamp(width.unsqueeze(1), min=1.0e-6), 0.0, 1.0)


def _phase_in_forward_window(phase: Tensor, start: Tensor, width: Tensor) -> Tensor:
    progress = _forward_phase_distance(start.unsqueeze(1), phase)
    return progress < width.unsqueeze(1) - 1.0e-6


def build_nominal_trajectory(
    state: MpcRobotState,
    command: Tensor,
    terrain: MpcPlannerTerrain,
    runtime_cfg: MpcRuntimeCfg,
) -> dict[str, Tensor]:
    """Build a differentiable world-frame nominal seed for root/foot/windows."""
    root_pos0 = torch.as_tensor(state.root_pos, dtype=torch.float32)
    root_rpy0 = torch.as_tensor(state.root_rpy, dtype=torch.float32, device=root_pos0.device)
    foot_pos0 = torch.as_tensor(state.foot_pos, dtype=torch.float32, device=root_pos0.device)
    cmd = _as_command(command, device=root_pos0.device)
    batch = int(root_pos0.shape[0])
    horizon = int(runtime_cfg.horizon_steps)
    dt = float(runtime_cfg.dt)

    frame = torch.arange(horizon, dtype=root_pos0.dtype, device=root_pos0.device)
    if horizon > 1:
        interval = frame[:-1]
        yaw_interval = root_rpy0[:, 2:3] + interval.view(1, -1) * dt * cmd[:, 2:3]
        v_world = _rotate_body_xy_to_world(cmd[:, None, :2].expand(-1, horizon - 1, -1), yaw_interval)
        root_xy_tail = root_pos0[:, None, :2] + torch.cumsum(v_world * dt, dim=1)
        root_xy = torch.cat((root_pos0[:, None, :2], root_xy_tail), dim=1)
    else:
        root_xy = root_pos0[:, None, :2]
    root_pos = root_pos0[:, None, :].expand(batch, horizon, 3).clone()
    root_pos[..., :2] = root_xy
    root_pos[..., 2] = root_pos0[:, None, 2]
    root_rpy = root_rpy0[:, None, :].expand(batch, horizon, 3).clone()
    root_rpy[..., 2] = root_rpy0[:, None, 2] + frame.view(1, horizon) * dt * cmd[:, 2:3]

    offsets = torch.as_tensor(runtime_cfg.leg_phase_offsets, dtype=root_pos.dtype, device=root_pos.device).view(1, 4)
    if bool(runtime_cfg.randomize_replan_phase):
        phase_flip = torch.randint(0, 2, (batch, 1), device=root_pos.device).to(dtype=root_pos.dtype) * 0.5
    else:
        phase_flip = torch.zeros((batch, 1), dtype=root_pos.dtype, device=root_pos.device)
    prior_offset = torch.remainder(offsets + phase_flip, 1.0)
    duty = float(runtime_cfg.duty_factor)
    swing_width = torch.full((batch, 4), max(1.0e-3, 1.0 - duty), dtype=root_pos.dtype, device=root_pos.device)
    swing_center = torch.remainder(prior_offset + duty + 0.5 * swing_width, 1.0)
    swing_start = torch.remainder(swing_center - 0.5 * swing_width, 1.0)
    swing_end = torch.remainder(swing_center + 0.5 * swing_width, 1.0)
    touchdown_phase = torch.clamp(swing_center + 0.5 * swing_width, min=0.0, max=1.0)

    start_idx = torch.remainder(torch.round(swing_start * float(horizon)).to(dtype=torch.long), horizon)
    touchdown_idx = torch.round(touchdown_phase * float(max(horizon - 1, 1))).to(dtype=torch.long).clamp(0, horizon - 1)
    root_start = _gather_time(root_pos, start_idx)
    yaw_start = _gather_time(root_rpy[..., 2:3], start_idx).squeeze(-1)
    root_td = _gather_time(root_pos, touchdown_idx)
    yaw_td = _gather_time(root_rpy[..., 2:3], touchdown_idx).squeeze(-1)

    foot_start_w = foot_pos0
    foot_start_rel_xy_w = foot_start_w[..., :2] - root_start[..., :2]
    foot_start_body_xy = _rotate_world_xy_to_body(foot_start_rel_xy_w, yaw_start)
    step_bias = float(runtime_cfg.nominal_stride_scale) * (dt * float(horizon)) * cmd[:, None, :2]
    yaw_perp = torch.stack((-foot_start_body_xy[..., 1], foot_start_body_xy[..., 0]), dim=-1)
    yaw_bias = float(runtime_cfg.nominal_yaw_stride_scale) * (dt * float(horizon)) * cmd[:, None, 2:3] * yaw_perp
    target_body_xy = foot_start_body_xy + step_bias + yaw_bias
    target_world_xy = root_td[..., :2] + _rotate_body_xy_to_world(target_body_xy, yaw_td)
    target_z = height_at(terrain, target_world_xy).to(dtype=root_pos.dtype, device=root_pos.device)
    touchdown_target_w = torch.cat((target_world_xy, target_z.unsqueeze(-1)), dim=-1)

    phase = frame.view(1, horizon, 1) / float(horizon)
    alpha = _circular_progress(phase, swing_start, swing_width)
    alpha_s = alpha * alpha * (3.0 - 2.0 * alpha)
    foot_pos = foot_pos0[:, None, :, :].expand(batch, horizon, 4, 3).clone()
    foot_xy = (1.0 - alpha_s.unsqueeze(-1)) * foot_pos0[:, None, :, :2] + alpha_s.unsqueeze(-1) * target_world_xy[:, None, :, :]
    foot_z = (
        (1.0 - alpha_s) * foot_pos0[:, None, :, 2]
        + alpha_s * target_z[:, None, :]
        + float(runtime_cfg.nominal_swing_height_m) * 4.0 * alpha * (1.0 - alpha)
    )
    swing_mask = _phase_in_forward_window(phase, swing_start, swing_width)
    frame_idx = torch.arange(horizon, dtype=torch.long, device=root_pos.device).view(1, horizon, 1)
    touchdown_in_horizon = touchdown_phase >= swing_start
    post_touchdown_mask = touchdown_in_horizon.unsqueeze(1) & (frame_idx >= touchdown_idx.unsqueeze(1))
    foot_pos[..., :2] = torch.where(swing_mask.unsqueeze(-1), foot_xy, foot_pos[..., :2])
    foot_pos[..., 2] = torch.where(swing_mask, foot_z, foot_pos[..., 2])
    foot_pos[..., :2] = torch.where(post_touchdown_mask.unsqueeze(-1), target_world_xy[:, None, :, :], foot_pos[..., :2])
    foot_pos[..., 2] = torch.where(post_touchdown_mask, target_z[:, None, :], foot_pos[..., 2])
    contact_prior = torch.logical_not(swing_mask).to(dtype=root_pos.dtype)

    return {
        "root_pos": root_pos,
        "root_rpy": root_rpy,
        "foot_pos": foot_pos,
        "swing_center": swing_center,
        "swing_width": swing_width,
        "swing_start": swing_start,
        "swing_end": swing_end,
        "contact_prior": contact_prior,
        "touchdown_target_w": touchdown_target_w,
        "touchdown_phase": touchdown_phase,
    }


__all__ = ["build_nominal_trajectory"]
