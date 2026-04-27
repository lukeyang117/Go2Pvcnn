"""Fixed raw-compatible contact schedule for the together backend."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import TogetherPlannerConfig
from .types import TogetherContactSchedule


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


__all__ = ["build_fixed_schedule", "hold_command_mask"]
