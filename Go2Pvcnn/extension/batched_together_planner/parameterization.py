"""Deterministic vectorized rollout parameterization for the together core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import TogetherPlannerConfig
from .schedule import hold_command_mask
from .terrain import TogetherPlannerTerrain
from .types import HIP_OFFSETS_ARRAY, TogetherContactSchedule


@dataclass(frozen=True)
class TogetherRollout:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    touchdown_seq: Tensor
    touchdown_mask: Tensor
    contact_state: Tensor
    support_xy: Tensor
    support_height: Tensor
    support_slope: Tensor
    time_s: Tensor


def build_time_grid(horizon_steps: int, dt: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.arange(horizon_steps, device=device, dtype=dtype) * float(dt)


def integrate_body_frame_translation(command_batch: Tensor, yaw_traj: Tensor, dt: float) -> Tensor:
    command = torch.as_tensor(command_batch, device=yaw_traj.device, dtype=yaw_traj.dtype)
    if command.ndim == 1:
        command = command.unsqueeze(0)
    vx = command[:, 0].view(command.shape[0], 1)
    vy = command[:, 1].view(command.shape[0], 1)
    cos_yaw = torch.cos(yaw_traj[:, :-1])
    sin_yaw = torch.sin(yaw_traj[:, :-1])
    dx = (cos_yaw * vx - sin_yaw * vy) * float(dt)
    dy = (sin_yaw * vx + cos_yaw * vy) * float(dt)
    dz = torch.zeros_like(dx)
    delta = torch.stack((dx, dy, dz), dim=-1)
    zero = torch.zeros_like(delta[:, :1, :])
    return torch.cat((zero, torch.cumsum(delta, dim=1)), dim=1)


def _seed_params_flat(command_batch: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    dtype = command_batch.dtype
    device = command_batch.device
    command_norm = torch.linalg.vector_norm(command_batch[:, :2], dim=-1, keepdim=True)
    scale = 1.0 + 0.5 * command_norm
    default_template = torch.tensor(
        (0.02, 0.01, -0.01, 0.02, 0.01, 0.01, 0.0, 0.01),
        device=device,
        dtype=dtype,
    ).view(1, 8) * scale
    zero_template = torch.tensor(
        (0.03, 0.0, 0.03, -0.01, 0.03, 0.03, 0.01, 0.03),
        device=device,
        dtype=dtype,
    ).view(1, 8)
    lateral_template = torch.tensor(
        (
            0.022499997168779373,
            0.009964285418391228,
            -0.010478571057319641,
            0.023157142102718353,
            0.003535714466124773,
            -0.011249998584389687,
            -0.015428571030497551,
            0.0022499999031424522,
        ),
        device=device,
        dtype=dtype,
    ).view(1, 8)
    hold = hold_command_mask(command_batch, cfg).to(device=device)
    lateral = command_batch[:, 1].abs() > command_batch[:, 0].abs()
    return torch.where(
        hold[:, None],
        zero_template.expand(command_batch.shape[0], -1),
        torch.where(lateral[:, None], lateral_template.expand(command_batch.shape[0], -1), default_template),
    )


def _root_trajectory(root_pos: Tensor, root_rpy: Tensor, command_batch: Tensor, cfg: TogetherPlannerConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device = root_pos.device
    dtype = root_pos.dtype
    time_s = build_time_grid(int(cfg.horizon_steps), float(cfg.dt), device=device, dtype=dtype)
    seed_params = _seed_params_flat(command_batch, cfg)
    phase = (time_s / max(float(cfg.horizon_s), 1e-6)).view(1, int(cfg.horizon_steps))
    z_curve = seed_params[:, 0:1] * (0.5 - 0.5 * torch.cos(torch.pi * phase))
    command_yaw = root_rpy[:, 2:3] + command_batch[:, 2:3] * time_s.view(1, -1)
    yaw = command_yaw + seed_params[:, 7:8] * phase
    translation = integrate_body_frame_translation(command_batch, command_yaw, float(cfg.dt))
    root_traj = root_pos[:, None, :].expand(-1, int(cfg.horizon_steps), -1) + translation
    root_traj[..., 0] = root_traj[..., 0] + seed_params[:, 5:6] * phase
    root_traj[..., 1] = root_traj[..., 1] + seed_params[:, 6:7] * phase
    root_traj[..., 2] = root_pos[:, 2:3] + z_curve
    rpy_traj = root_rpy[:, None, :].expand(-1, int(cfg.horizon_steps), -1).clone()
    rpy_traj[..., 0] = root_rpy[:, 0:1] + seed_params[:, 1:2] * phase
    rpy_traj[..., 1] = root_rpy[:, 1:2] + seed_params[:, 2:3] * phase
    rpy_traj[..., 2] = yaw
    return root_traj, rpy_traj, time_s, seed_params


def _touchdown_targets(
    terrain: TogetherPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    command_batch: Tensor,
    seed_params: Tensor,
    schedule: TogetherContactSchedule,
    cfg: TogetherPlannerConfig,
) -> Tensor:
    batch_size = root_pos.shape[0]
    event_frames = schedule.touchdown_frames.clamp(max=int(cfg.horizon_steps) - 1)
    gather_index = event_frames[:, None, :, :, None].expand(batch_size, 1, 4, int(cfg.event_cap), 3)
    root_event_pos = root_pos.gather(1, gather_index.reshape(batch_size, 4 * int(cfg.event_cap), 3)).reshape(
        batch_size,
        4,
        int(cfg.event_cap),
        3,
    )
    root_event_rpy = root_rpy.gather(1, gather_index.reshape(batch_size, 4 * int(cfg.event_cap), 3)).reshape(
        batch_size,
        4,
        int(cfg.event_cap),
        3,
    )
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=root_pos.device, dtype=root_pos.dtype).view(1, 4, 1, 3)
    yaw = root_event_rpy[..., 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    target_xy = torch.empty((batch_size, 4, int(cfg.event_cap), 2), device=root_pos.device, dtype=root_pos.dtype)
    target_xy[..., 0] = root_event_pos[..., 0] + cos_yaw * hip_offsets[..., 0] - sin_yaw * hip_offsets[..., 1]
    target_xy[..., 1] = root_event_pos[..., 1] + sin_yaw * hip_offsets[..., 0] + cos_yaw * hip_offsets[..., 1]
    touchdown_bias = seed_params[:, 4].view(batch_size, 1, 1, 1)
    target_xy = target_xy + touchdown_bias * float(cfg.touchdown_xy_bias_scale)
    support_xy, support_height, _ = terrain.support_at(target_xy.reshape(batch_size, -1, 2), cfg)
    target_z = support_height.reshape(batch_size, 4, int(cfg.event_cap)) + torch.relu(seed_params[:, 4]).view(batch_size, 1, 1) * float(cfg.touchdown_z_bias_scale)
    target = torch.cat((support_xy.reshape(batch_size, 4, int(cfg.event_cap), 2), target_z.unsqueeze(-1)), dim=-1)
    return target


def _support_plane_target_rpy(root_rpy: Tensor, target_foot: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    front_mid = 0.5 * (target_foot[:, 0, :] + target_foot[:, 1, :])
    rear_mid = 0.5 * (target_foot[:, 2, :] + target_foot[:, 3, :])
    left_mid = 0.5 * (target_foot[:, 0, :] + target_foot[:, 2, :])
    right_mid = 0.5 * (target_foot[:, 1, :] + target_foot[:, 3, :])
    forward = front_mid - rear_mid
    left = left_mid - right_mid
    support_normal = torch.cross(forward, left, dim=-1)
    support_normal = torch.where(support_normal[:, 2:3] < 0.0, -support_normal, support_normal)
    support_normal = support_normal / torch.linalg.vector_norm(support_normal, dim=-1, keepdim=True).clamp_min(1e-6)
    yaw = root_rpy[:, 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    yaw_aligned_x = cos_yaw * support_normal[:, 0] + sin_yaw * support_normal[:, 1]
    yaw_aligned_y = -sin_yaw * support_normal[:, 0] + cos_yaw * support_normal[:, 1]
    yaw_aligned_z = support_normal[:, 2]
    limit = float(cfg.rehome_roll_pitch_limit)
    target_roll = torch.asin((-yaw_aligned_y).clamp(-1.0, 1.0)).clamp(-limit, limit)
    target_pitch = torch.atan2(yaw_aligned_x, yaw_aligned_z).clamp(-limit, limit)
    return torch.stack((target_roll, target_pitch, yaw), dim=-1)


def _hold_rehome_targets(
    terrain: TogetherPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    time_s: Tensor,
    cfg: TogetherPlannerConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = root_pos.shape[0]
    horizon_steps = int(cfg.horizon_steps)
    phase = 0.5 - 0.5 * torch.cos(torch.pi * time_s / time_s[-1].clamp_min(1e-6))
    phase_root = phase.view(1, horizon_steps, 1)
    phase_foot = phase.view(1, horizon_steps, 1, 1)
    nominal_xy = HIP_OFFSETS_ARRAY.to(device=root_pos.device, dtype=root_pos.dtype)[:, :2].view(1, 4, 2)
    yaw = root_rpy[:, 2:3]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    target_xy = torch.empty((batch_size, 4, 2), device=root_pos.device, dtype=root_pos.dtype)
    target_xy[..., 0] = root_pos[:, None, 0] + cos_yaw * nominal_xy[..., 0] - sin_yaw * nominal_xy[..., 1]
    target_xy[..., 1] = root_pos[:, None, 1] + sin_yaw * nominal_xy[..., 0] + cos_yaw * nominal_xy[..., 1]
    _, support_height, _ = terrain.support_at(target_xy, cfg)
    target_foot = torch.cat((target_xy, support_height.unsqueeze(-1)), dim=-1)
    target_root_z = support_height.mean(dim=-1) + float(cfg.hip_height)
    target_root = torch.cat((root_pos[:, :2], target_root_z.unsqueeze(-1)), dim=-1)
    target_rpy = _support_plane_target_rpy(root_rpy, target_foot, cfg)
    hold_root = root_pos[:, None, :] + phase_root * (target_root[:, None, :] - root_pos[:, None, :])
    hold_rpy = root_rpy[:, None, :] + phase_root * (target_rpy[:, None, :] - root_rpy[:, None, :])
    hold_foot_xy = foot_pos[:, None, :, :2] + phase_foot * (target_foot[:, None, :, :2] - foot_pos[:, None, :, :2])
    _, hold_support_height, hold_support_slope = terrain.support_at(hold_foot_xy.reshape(batch_size, -1, 2), cfg)
    hold_support_height = hold_support_height.reshape(batch_size, horizon_steps, 4)
    hold_support_slope = hold_support_slope.reshape(batch_size, horizon_steps, 4)
    linear_foot_z = foot_pos[:, None, :, 2] + phase.view(1, horizon_steps, 1) * (
        target_foot[:, None, :, 2] - foot_pos[:, None, :, 2]
    )
    hold_foot_z = torch.maximum(linear_foot_z, hold_support_height)
    hold_foot = torch.cat((hold_foot_xy, hold_foot_z.unsqueeze(-1)), dim=-1)
    first_frame = phase.view(1, horizon_steps, 1, 1) <= 0.0
    hold_foot = torch.where(first_frame, foot_pos[:, None, :, :], hold_foot)
    hold_support_height = torch.where(first_frame.squeeze(-1), foot_pos[:, None, :, 2], hold_support_height)
    return hold_root, hold_rpy, hold_foot, hold_support_height, hold_support_slope


def expand_segment(
    terrain: TogetherPlannerTerrain,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    command_batch: Tensor,
    schedule: TogetherContactSchedule,
    cfg: TogetherPlannerConfig,
) -> TogetherRollout:
    batch_size = root_pos.shape[0]
    horizon_steps = int(cfg.horizon_steps)
    root_traj, rpy_traj, time_s, seed_params = _root_trajectory(root_pos, root_rpy, command_batch, cfg)
    touchdown_seq = _touchdown_targets(terrain, root_traj, rpy_traj, command_batch, seed_params, schedule, cfg)
    frame_ids = torch.arange(horizon_steps, device=root_pos.device, dtype=torch.long).view(1, horizon_steps, 1)
    contact_bool = schedule.contact_state.to(dtype=torch.bool)
    masked_touchdown_frames = torch.where(
        schedule.touchdown_mask,
        schedule.touchdown_frames,
        torch.full_like(schedule.touchdown_frames, horizon_steps),
    )
    contact_by_leg = contact_bool.transpose(1, 2)
    full_frame_ids = torch.arange(horizon_steps, device=root_pos.device, dtype=torch.long).view(1, 1, 1, horizon_steps)
    non_contact_after_touchdown = (~contact_by_leg[:, :, None, :]) & (full_frame_ids > masked_touchdown_frames.unsqueeze(-1))
    first_non_contact_frame = torch.where(
        non_contact_after_touchdown,
        full_frame_ids,
        torch.full_like(full_frame_ids, horizon_steps),
    ).amin(dim=-1)
    stance_reference_frames = torch.where(
        first_non_contact_frame < horizon_steps,
        (first_non_contact_frame - 1).clamp_min(0),
        torch.full_like(first_non_contact_frame, horizon_steps - 1),
    )
    support_frames = torch.where(
        masked_touchdown_frames < horizon_steps,
        stance_reference_frames,
        torch.full_like(masked_touchdown_frames, horizon_steps),
    ).clamp(max=horizon_steps - 1)
    support_index = support_frames[:, None, :, :, None].expand(batch_size, 1, 4, int(cfg.event_cap), 3)
    root_support_pos = root_traj.gather(1, support_index.reshape(batch_size, 4 * int(cfg.event_cap), 3)).reshape(
        batch_size,
        4,
        int(cfg.event_cap),
        3,
    )
    root_support_rpy = rpy_traj.gather(1, support_index.reshape(batch_size, 4 * int(cfg.event_cap), 3)).reshape(
        batch_size,
        4,
        int(cfg.event_cap),
        3,
    )
    initial_yaw = root_rpy[:, 2]
    initial_offset_w = foot_pos[..., :2] - root_pos[:, None, :2]
    cos0 = torch.cos(initial_yaw).view(batch_size, 1)
    sin0 = torch.sin(initial_yaw).view(batch_size, 1)
    preview_offset_body_x = cos0 * initial_offset_w[..., 0] + sin0 * initial_offset_w[..., 1]
    preview_offset_body_y = -sin0 * initial_offset_w[..., 0] + cos0 * initial_offset_w[..., 1]
    support_yaw = root_support_rpy[..., 2]
    support_cos = torch.cos(support_yaw)
    support_sin = torch.sin(support_yaw)
    support_touchdown_xy = torch.empty((batch_size, 4, int(cfg.event_cap), 2), device=root_pos.device, dtype=root_pos.dtype)
    support_touchdown_xy[..., 0] = root_support_pos[..., 0] + support_cos * preview_offset_body_x[:, :, None] - support_sin * preview_offset_body_y[:, :, None]
    support_touchdown_xy[..., 1] = root_support_pos[..., 1] + support_sin * preview_offset_body_x[:, :, None] + support_cos * preview_offset_body_y[:, :, None]
    touchdown_bias = seed_params[:, 4].view(batch_size, 1, 1, 1)
    support_touchdown_xy = support_touchdown_xy + touchdown_bias * float(cfg.touchdown_xy_bias_scale)
    _, support_touchdown_height, _ = terrain.support_at(support_touchdown_xy.reshape(batch_size, -1, 2), cfg)
    support_touchdown_seq = torch.cat(
        (
            support_touchdown_xy,
            (
                support_touchdown_height.reshape(batch_size, 4, int(cfg.event_cap))
                + torch.relu(seed_params[:, 4]).view(batch_size, 1, 1) * float(cfg.touchdown_z_bias_scale)
            ).unsqueeze(-1),
        ),
        dim=-1,
    )
    touchdown_seq = support_touchdown_seq
    event_active = frame_ids.view(1, horizon_steps, 1, 1) >= masked_touchdown_frames[:, None, :, :]
    events_reached = event_active.long().sum(dim=-1)
    next_event_index = events_reached.clamp(max=int(cfg.event_cap) - 1)
    foot_bank = foot_pos[:, None, :, None, :].expand(-1, horizon_steps, -1, 1, -1)
    touchdown_bank = support_touchdown_seq[:, None, :, :, :].expand(-1, horizon_steps, -1, -1, -1)
    foothold_bank = torch.cat((foot_bank, touchdown_bank), dim=3)
    stance = foothold_bank.gather(3, events_reached[:, :, :, None, None].expand(-1, -1, -1, 1, 3)).squeeze(3)
    touchdown_target = touchdown_bank.gather(3, next_event_index[:, :, :, None, None].expand(-1, -1, -1, 1, 3)).squeeze(3)
    valid_touchdown_count = schedule.touchdown_mask.long().sum(dim=-1)
    has_valid_next_touchdown = next_event_index < valid_touchdown_count[:, None, :]
    swing_bool = ~contact_bool
    previous_swing = torch.nn.functional.pad(swing_bool[:, :-1, :], (0, 0, 1, 0), value=False)
    swing_start = swing_bool & ~previous_swing
    swing_segment_id = swing_start.long().cumsum(dim=1)
    swing_start_frame = torch.where(
        swing_start,
        frame_ids.expand(batch_size, -1, 4),
        torch.zeros((batch_size, horizon_steps, 4), device=root_pos.device, dtype=torch.long),
    )
    last_swing_start_frame = swing_start_frame.cummax(dim=1).values
    swing_position = (frame_ids.expand(batch_size, -1, 4) - last_swing_start_frame + 1).to(dtype=root_pos.dtype)
    swing_by_leg = swing_bool.transpose(1, 2)
    swing_segment_id_by_leg = swing_segment_id.transpose(1, 2)
    same_swing_segment = (
        (swing_segment_id_by_leg[:, :, :, None] == swing_segment_id_by_leg[:, :, None, :])
        & swing_by_leg[:, :, :, None]
        & swing_by_leg[:, :, None, :]
    )
    swing_segment_length = same_swing_segment.sum(dim=-1).clamp_min(1).to(dtype=root_pos.dtype).transpose(1, 2)
    progress = (swing_position / swing_segment_length).clamp(0.0, 1.0)
    preview_yaw = rpy_traj[..., 2].unsqueeze(-1)
    preview_cos = torch.cos(preview_yaw)
    preview_sin = torch.sin(preview_yaw)
    preview_target = torch.empty((batch_size, horizon_steps, 4, 3), device=root_pos.device, dtype=root_pos.dtype)
    preview_target[..., 0] = root_traj[..., 0].unsqueeze(-1) + preview_cos * preview_offset_body_x[:, None, :] - preview_sin * preview_offset_body_y[:, None, :]
    preview_target[..., 1] = root_traj[..., 1].unsqueeze(-1) + preview_sin * preview_offset_body_x[:, None, :] + preview_cos * preview_offset_body_y[:, None, :]
    preview_height = terrain.height_at(preview_target[..., :2].reshape(batch_size, -1, 2))
    preview_target[..., 2] = preview_height.reshape(batch_size, horizon_steps, 4)
    touchdown_target = torch.where(has_valid_next_touchdown[:, :, :, None], touchdown_target, preview_target)
    swing_progress = torch.where(has_valid_next_touchdown, progress, torch.ones_like(progress))
    swing_target = stance + swing_progress[:, :, :, None] * (touchdown_target - stance)
    swing_peak = seed_params[:, 3].view(batch_size, 1, 1) * float(cfg.swing_parabola_multiplier) * progress * (1.0 - progress)
    swing_target = swing_target.clone()
    swing_target[..., 2] = swing_target[..., 2] + swing_peak
    foot_traj = torch.where(contact_bool[:, :, :, None], stance, swing_target)
    support_weight = schedule.contact_state.to(device=root_pos.device, dtype=root_pos.dtype)
    support_count = support_weight.sum(dim=-1).clamp_min(1.0)
    frame_support_height = (foot_traj[..., 2] * support_weight).sum(dim=-1) / support_count
    initial_support_count = support_weight[:, 0, :].sum(dim=-1).clamp_min(1.0)
    initial_support_height = (foot_pos[..., 2] * support_weight[:, 0, :]).sum(dim=-1) / initial_support_count
    z_curve = root_traj[..., 2] - root_pos[:, None, 2]
    root_traj = root_traj.clone()
    root_traj[..., 2] = frame_support_height + (root_pos[:, 2] - initial_support_height).view(batch_size, 1) + z_curve
    support_xy, support_height, support_slope = terrain.support_at(foot_traj[..., :2].reshape(batch_size, -1, 2), cfg)
    support_xy = support_xy.reshape(batch_size, horizon_steps, 4, 2)
    support_height = support_height.reshape(batch_size, horizon_steps, 4)
    support_slope = support_slope.reshape(batch_size, horizon_steps, 4)
    hold_mask = hold_command_mask(command_batch, cfg).to(device=root_pos.device)
    hold_root, hold_rpy, hold_foot, hold_support_height, hold_support_slope = _hold_rehome_targets(
        terrain,
        root_pos,
        root_rpy,
        foot_pos,
        time_s,
        cfg,
    )
    root_traj = torch.where(hold_mask[:, None, None], hold_root, root_traj)
    rpy_traj = torch.where(hold_mask[:, None, None], hold_rpy, rpy_traj)
    foot_traj = torch.where(hold_mask[:, None, None, None], hold_foot, foot_traj)
    touchdown_seq = torch.where(hold_mask[:, None, None, None], foot_pos[:, :, None, :].expand(-1, -1, int(cfg.event_cap), -1), touchdown_seq)
    support_xy = torch.where(hold_mask[:, None, None, None], hold_foot[..., :2], support_xy)
    support_height = torch.where(hold_mask[:, None, None], hold_support_height, support_height)
    support_slope = torch.where(hold_mask[:, None, None], hold_support_slope, support_slope)
    return TogetherRollout(
        root_pos=root_traj,
        root_rpy=rpy_traj,
        foot_pos=foot_traj,
        touchdown_seq=touchdown_seq,
        touchdown_mask=schedule.touchdown_mask,
        contact_state=schedule.contact_state,
        support_xy=support_xy,
        support_height=support_height,
        support_slope=support_slope,
        time_s=time_s,
    )


__all__ = ["TogetherRollout", "build_time_grid", "expand_segment", "integrate_body_frame_translation"]
