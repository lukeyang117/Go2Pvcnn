"""Fixed raw-compatible contact schedule for the together backend."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import TogetherPlannerConfig
from .types import HIP_OFFSETS_ARRAY, TogetherContactSchedule


def hold_command_mask(command_batch: Tensor, cfg: TogetherPlannerConfig | None = None) -> Tensor:
    planner_cfg = cfg or TogetherPlannerConfig()
    command = torch.as_tensor(command_batch)
    if command.ndim == 1:
        command = command.unsqueeze(0)
    if command.ndim != 2 or command.shape[-1] != 3:
        raise ValueError("command_batch must have shape [B, 3]")
    return command.abs().amax(dim=-1) <= float(planner_cfg.idle_command_eps)


def build_fixed_schedule(
    batch_size: int,
    horizon_steps: int,
    dt: float,
    device: torch.device,
    dtype: torch.dtype,
    command_batch: Tensor | None = None,
    planner_cfg: TogetherPlannerConfig | None = None,
) -> TogetherContactSchedule:
    cfg = planner_cfg or TogetherPlannerConfig()
    if int(horizon_steps) != int(cfg.horizon_steps):
        raise ValueError("horizon_steps must match TogetherPlannerConfig")
    if abs(float(dt) - float(cfg.dt)) > 1e-12:
        raise ValueError("dt must match TogetherPlannerConfig")

    times = torch.arange(horizon_steps, device=device, dtype=torch.float32) * float(dt)
    offsets = torch.as_tensor(cfg.leg_phase_offsets, device=device, dtype=torch.float32)
    phases = torch.remainder(times[:, None] * float(cfg.step_freq) + offsets[None, :], 1.0)
    contact_single = (phases < float(cfg.duty_factor)).to(dtype=dtype)
    contact_state = contact_single.unsqueeze(0).expand(batch_size, -1, -1)
    contact_bool = contact_single.to(dtype=torch.bool)
    touchdown_events = contact_bool[1:] & ~contact_bool[:-1]
    event_frames = torch.arange(1, horizon_steps, device=device, dtype=torch.long).view(horizon_steps - 1, 1)
    fill_frames = torch.full((horizon_steps - 1, 4), horizon_steps, device=device, dtype=torch.long)
    touchdown_candidates = torch.where(touchdown_events, event_frames.expand(-1, 4), fill_frames)
    touchdown_frames_single = touchdown_candidates.transpose(0, 1).topk(
        k=int(cfg.event_cap),
        dim=-1,
        largest=False,
        sorted=True,
    ).values
    touchdown_mask_single = touchdown_frames_single < horizon_steps
    touchdown_frames = touchdown_frames_single.unsqueeze(0).expand(batch_size, -1, -1)
    touchdown_mask = touchdown_mask_single.unsqueeze(0).expand(batch_size, -1, -1)

    if command_batch is None:
        hold_mask = torch.zeros((batch_size,), device=device, dtype=torch.bool)
    else:
        command = torch.as_tensor(command_batch, device=device, dtype=torch.float32)
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape != (batch_size, 3):
            raise ValueError("command_batch must have shape [B, 3]")
        hold_mask = hold_command_mask(command, cfg).to(device=device)

    contact_state = torch.where(
        hold_mask[:, None, None],
        torch.ones((batch_size, horizon_steps, 4), device=device, dtype=dtype),
        contact_state,
    )
    touchdown_mask = torch.where(
        hold_mask[:, None, None],
        torch.zeros((batch_size, 4, int(cfg.event_cap)), device=device, dtype=torch.bool),
        touchdown_mask,
    )
    touchdown_frames = torch.where(
        hold_mask[:, None, None],
        torch.full((batch_size, 4, int(cfg.event_cap)), horizon_steps, device=device, dtype=torch.long),
        touchdown_frames,
    )
    return TogetherContactSchedule(
        contact_state=contact_state,
        touchdown_mask=touchdown_mask,
        touchdown_frames=touchdown_frames,
        horizon_steps=horizon_steps,
        dt=float(dt),
        event_cap=int(cfg.event_cap),
    )


def _command_dirs(command_batch: Tensor, root_rpy: Tensor, cfg: TogetherPlannerConfig) -> Tensor:
    yaw = root_rpy[:, 2]
    body_v = command_batch[:, :2].to(device=root_rpy.device, dtype=root_rpy.dtype)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    world_v = torch.stack(
        (
            cos_yaw * body_v[:, 0] - sin_yaw * body_v[:, 1],
            sin_yaw * body_v[:, 0] + cos_yaw * body_v[:, 1],
        ),
        dim=-1,
    )
    fallback = torch.stack((cos_yaw, sin_yaw), dim=-1)
    norm = torch.linalg.vector_norm(world_v, dim=-1, keepdim=True)
    return torch.where(norm > float(cfg.idle_command_eps), world_v / norm.clamp_min(1e-6), fallback)


def build_cross_small_schedule(
    horizon_steps: int,
    dt: float,
    device: torch.device,
    dtype: torch.dtype,
    command_batch: Tensor,
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    planner_cfg: TogetherPlannerConfig | None = None,
) -> TogetherContactSchedule:
    cfg = planner_cfg or TogetherPlannerConfig()
    if int(horizon_steps) != int(cfg.horizon_steps):
        raise ValueError("horizon_steps must match TogetherPlannerConfig")
    if abs(float(dt) - float(cfg.dt)) > 1e-12:
        raise ValueError("dt must match TogetherPlannerConfig")

    batch_size = command_batch.shape[0]
    command = torch.as_tensor(command_batch, device=device, dtype=torch.float32)
    root_xy = torch.as_tensor(root_pos, device=device, dtype=torch.float32)[:, :2]
    root_rpy_t = torch.as_tensor(root_rpy, device=device, dtype=torch.float32)
    foot_xy = torch.as_tensor(foot_pos, device=device, dtype=torch.float32)[..., :2]
    d = _command_dirs(command, root_rpy_t, cfg)
    hip_xy = HIP_OFFSETS_ARRAY.to(device=device, dtype=torch.float32)[:, :2].view(1, 4, 2)
    anchor_xy = torch.where(torch.isfinite(foot_xy).all(dim=-1, keepdim=True), foot_xy, root_xy[:, None, :] + hip_xy)
    leg_s = ((anchor_xy - root_xy[:, None, :]) * d[:, None, :]).sum(dim=-1)
    lead_order = torch.argsort(leg_s, dim=-1, descending=True)

    starts = torch.as_tensor((0.10, 0.18, 0.55, 0.63), device=device, dtype=torch.float32)
    ends = torch.as_tensor((0.38, 0.46, 0.83, 0.91), device=device, dtype=torch.float32)
    slot_start_frames = torch.clamp(torch.round(starts / float(dt)).to(dtype=torch.long), 0, int(horizon_steps) - 1)
    slot_end_frames = torch.clamp(torch.round(ends / float(dt)).to(dtype=torch.long), 1, int(horizon_steps))
    start_frames = torch.empty((batch_size, 4), device=device, dtype=torch.long)
    end_frames = torch.empty((batch_size, 4), device=device, dtype=torch.long)
    start_frames.scatter_(1, lead_order, slot_start_frames.view(1, 4).expand(batch_size, -1))
    end_frames.scatter_(1, lead_order, slot_end_frames.view(1, 4).expand(batch_size, -1))

    frame_ids = torch.arange(int(horizon_steps), device=device, dtype=torch.long).view(1, int(horizon_steps), 1)
    swing = (frame_ids >= start_frames[:, None, :]) & (frame_ids < end_frames[:, None, :])
    contact_state = (~swing).to(dtype=dtype)
    touchdown_frame = end_frames.clamp(max=int(horizon_steps) - 1)
    fill = torch.full((batch_size, 4, int(cfg.event_cap)), int(horizon_steps), device=device, dtype=torch.long)
    touchdown_frames = fill.clone()
    touchdown_frames[:, :, 0] = touchdown_frame
    touchdown_mask = torch.zeros((batch_size, 4, int(cfg.event_cap)), device=device, dtype=torch.bool)
    touchdown_mask[:, :, 0] = True

    hold_mask = hold_command_mask(command, cfg).to(device=device)
    contact_state = torch.where(hold_mask[:, None, None], torch.ones_like(contact_state), contact_state)
    touchdown_mask = torch.where(hold_mask[:, None, None], torch.zeros_like(touchdown_mask), touchdown_mask)
    touchdown_frames = torch.where(hold_mask[:, None, None], fill, touchdown_frames)
    return TogetherContactSchedule(
        contact_state=contact_state,
        touchdown_mask=touchdown_mask,
        touchdown_frames=touchdown_frames,
        horizon_steps=horizon_steps,
        dt=float(dt),
        event_cap=int(cfg.event_cap),
    )


__all__ = ["build_cross_small_schedule", "build_fixed_schedule", "hold_command_mask"]
