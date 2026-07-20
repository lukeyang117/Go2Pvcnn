"""Fixed tensor schedule for the 24-frame diagonal trot."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FixedTrotSchedule:
    phase: Tensor
    swing: Tensor
    stance: Tensor
    swing_tau: Tensor


def fixed_trot_schedule(phase0: Tensor, *, horizon_steps: int = 30) -> FixedTrotSchedule:
    """Broadcast a 12-swing/12-stance diagonal trot from each batch phase."""
    phase0 = torch.as_tensor(phase0, dtype=torch.long)
    if phase0.ndim == 0:
        phase0 = phase0.unsqueeze(0)
    if phase0.ndim != 1:
        raise ValueError("phase0 must have shape [B]")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")

    node = torch.arange(horizon_steps + 1, device=phase0.device)
    leg_offset = torch.tensor((0, 12, 12, 0), device=phase0.device)
    phase = (phase0[:, None, None] + node[None, :, None] + leg_offset[None, None, :]) % 24
    swing = phase < 12
    swing_tau = phase.to(torch.float32).div(11.0).clamp(0.0, 1.0)
    return FixedTrotSchedule(
        phase=phase,
        swing=swing,
        stance=~swing,
        swing_tau=swing_tau,
    )


__all__ = ["FixedTrotSchedule", "fixed_trot_schedule"]
