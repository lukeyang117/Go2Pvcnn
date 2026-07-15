"""Externally fixed diagonal trot schedule."""

from __future__ import annotations

import torch
from torch import Tensor


def fixed_trot_schedule(
    batch: int,
    horizon_steps: int,
    device: torch.device | str,
    *,
    half_cycle_steps: int = 4,
    phase_offset_steps: int = 0,
) -> Tensor:
    """Return contact states in planner leg order FL, FR, RL, RR."""
    if batch < 1:
        raise ValueError("batch must be positive")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    if half_cycle_steps < 1:
        raise ValueError("half_cycle_steps must be positive")
    frame = torch.arange(horizon_steps + 1, device=device)
    group_a_contact = torch.remainder((frame + int(phase_offset_steps)) // int(half_cycle_steps), 2) == 0
    group_b_contact = torch.logical_not(group_a_contact)
    contact = torch.stack((group_a_contact, group_b_contact, group_b_contact, group_a_contact), dim=-1)
    return contact.unsqueeze(0).expand(batch, -1, -1).clone()


__all__ = ["fixed_trot_schedule"]
