"""Externally fixed diagonal trot schedule."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ContactSchedulerAdvance:
    contact_state: Tensor
    phase_age: Tensor
    swing_extension_age: Tensor
    stance_age: Tensor
    recovery_state: Tensor
    liftoff_blocked: Tensor
    progress_scale: Tensor


def fixed_trot_schedule(
    batch: int,
    horizon_steps: int,
    device: torch.device | str,
    *,
    half_cycle_steps: int = 4,
    phase_offset_steps: int | Tensor = 0,
) -> Tensor:
    """Return contact states in planner leg order FL, FR, RL, RR."""
    if batch < 1:
        raise ValueError("batch must be positive")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    if half_cycle_steps < 1:
        raise ValueError("half_cycle_steps must be positive")
    frame = torch.arange(horizon_steps + 1, device=device).view(1, -1)
    offset = torch.as_tensor(phase_offset_steps, dtype=torch.long, device=device)
    if offset.ndim == 0:
        offset = offset.expand(batch)
    if offset.shape != (batch,):
        raise ValueError("phase_offset_steps must be scalar or have shape [B]")
    group_a_contact = torch.remainder(
        (frame + offset[:, None]) // int(half_cycle_steps),
        2,
    ) == 0
    group_b_contact = torch.logical_not(group_a_contact)
    contact = torch.stack((group_a_contact, group_b_contact, group_b_contact, group_a_contact), dim=-1)
    return contact.clone()


def adaptive_contact_schedule(
    *,
    contact_state: Tensor,
    phase_age: Tensor,
    touchdown_ready: Tensor,
    horizon_steps: int,
    half_cycle_steps: int,
) -> Tensor:
    """Build a fixed-shape per-leg schedule with one-step unsafe touchdown delay."""
    contact = torch.as_tensor(contact_state, dtype=torch.bool)
    age = torch.as_tensor(phase_age, dtype=torch.long, device=contact.device)
    ready = torch.as_tensor(touchdown_ready, dtype=torch.bool, device=contact.device)
    if contact.ndim != 2 or tuple(contact.shape[1:]) != (4,) or age.shape != contact.shape or ready.shape != contact.shape:
        raise ValueError("adaptive contact inputs must have shape [B,4]")
    if int(horizon_steps) < 1 or int(half_cycle_steps) < 1:
        raise ValueError("horizon_steps and half_cycle_steps must be positive")
    remaining = (int(half_cycle_steps) - age).clamp_min(1)
    unsafe_boundary = torch.logical_and(
        torch.logical_not(contact),
        torch.logical_and(age >= int(half_cycle_steps) - 1, torch.logical_not(ready)),
    )
    remaining = torch.where(unsafe_boundary, torch.maximum(remaining, remaining.new_full((), 2)), remaining)
    provisional_x1 = torch.where(remaining <= 1, torch.logical_not(contact), contact)
    insufficient_support = provisional_x1.sum(dim=1) < 2
    blocked_liftoff = torch.logical_and(
        torch.logical_and(contact, remaining <= 1),
        insufficient_support.unsqueeze(-1),
    )
    remaining = torch.where(blocked_liftoff, torch.maximum(remaining, remaining.new_full((), 2)), remaining)
    frame = torch.arange(int(horizon_steps) + 1, device=contact.device).view(1, -1, 1)
    remaining_node = remaining[:, None, :]
    before_transition = frame < remaining_node
    elapsed = (frame - remaining_node).clamp_min(0)
    transition_count = 1 + torch.div(elapsed, int(half_cycle_steps), rounding_mode="floor")
    toggled = torch.remainder(transition_count, 2) == 1
    after_transition = torch.logical_xor(contact[:, None, :], toggled)
    return torch.where(before_transition, contact[:, None, :], after_transition).clone()


def advance_contact_scheduler(
    *,
    contact_state: Tensor,
    phase_age: Tensor,
    swing_extension_age: Tensor,
    stance_age: Tensor,
    recovery_state: Tensor,
    touchdown_scheduled: Tensor,
    touchdown_ready: Tensor,
    liftoff_scheduled: Tensor,
    reliable_stance: Tensor,
    max_swing_extension_steps: int,
) -> ContactSchedulerAdvance:
    """Advance one tensorized per-leg contact step without forcing unsafe touchdown."""
    contact = torch.as_tensor(contact_state, dtype=torch.bool)
    device = contact.device
    phase = torch.as_tensor(phase_age, dtype=torch.long, device=device)
    extension = torch.as_tensor(swing_extension_age, dtype=torch.long, device=device)
    stance = torch.as_tensor(stance_age, dtype=torch.long, device=device)
    recovery = torch.as_tensor(recovery_state, dtype=torch.bool, device=device)
    touchdown = torch.as_tensor(touchdown_scheduled, dtype=torch.bool, device=device)
    ready = torch.as_tensor(touchdown_ready, dtype=torch.bool, device=device)
    liftoff = torch.as_tensor(liftoff_scheduled, dtype=torch.bool, device=device)
    reliable = torch.as_tensor(reliable_stance, dtype=torch.bool, device=device)
    tensors = (phase, extension, stance, recovery, touchdown, ready, liftoff, reliable)
    if contact.ndim != 2 or int(contact.shape[1]) != 4 or any(value.shape != contact.shape for value in tensors):
        raise ValueError("per-leg scheduler tensors must have shape [B,4]")
    if int(max_swing_extension_steps) < 1:
        raise ValueError("max_swing_extension_steps must be positive")

    confirmed = torch.logical_and(torch.logical_and(torch.logical_not(contact), touchdown), ready)
    unsafe_touchdown = torch.logical_and(
        torch.logical_and(torch.logical_not(contact), touchdown), torch.logical_not(ready)
    )
    liftoff_candidate = torch.logical_and(contact, liftoff)
    available_reliable = torch.logical_or(reliable, confirmed)
    reliable_count = available_reliable.sum(dim=1)
    unreliable_candidate = torch.logical_and(liftoff_candidate, torch.logical_not(reliable))
    release_unreliable = torch.logical_and(
        unreliable_candidate,
        (reliable_count >= 2).unsqueeze(-1),
    )
    reliable_candidate = torch.logical_and(liftoff_candidate, reliable)
    reliable_budget = (reliable_count - 2).clamp_min(0)
    reliable_rank = torch.cumsum(reliable_candidate.to(torch.long), dim=1)
    release_reliable = torch.logical_and(
        reliable_candidate,
        reliable_rank <= reliable_budget.unsqueeze(-1),
    )
    liftoff_allowed = torch.logical_or(release_unreliable, release_reliable)
    liftoff_blocked = torch.logical_and(liftoff_candidate, torch.logical_not(liftoff_allowed))

    next_contact = torch.logical_or(contact, confirmed)
    next_contact = torch.logical_and(next_contact, torch.logical_not(liftoff_allowed))
    next_extension = torch.where(
        confirmed,
        torch.zeros_like(extension),
        torch.where(
            unsafe_touchdown,
            (extension + 1).clamp_max(int(max_swing_extension_steps)),
            extension,
        ),
    )
    next_extension = torch.where(next_contact, torch.zeros_like(next_extension), next_extension)
    next_recovery = torch.logical_and(recovery, torch.logical_not(confirmed))
    next_recovery = torch.logical_or(
        next_recovery,
        torch.logical_and(torch.logical_not(next_contact), next_extension >= int(max_swing_extension_steps)),
    )
    transitioned = torch.logical_or(confirmed, liftoff_allowed)
    next_phase = torch.where(transitioned, torch.zeros_like(phase), phase + 1)
    next_stance = torch.where(
        confirmed,
        torch.zeros_like(stance),
        torch.where(next_contact, stance + 1, torch.zeros_like(stance)),
    )
    progress_scale = 1.0 - next_extension.amax(dim=1).to(torch.float32) / float(max_swing_extension_steps)
    progress_scale = progress_scale.clamp(0.0, 1.0)
    next_reliable_count = torch.logical_and(
        available_reliable,
        torch.logical_not(liftoff_allowed),
    ).sum(dim=1)
    support_scale = (next_reliable_count.to(torch.float32) / 2.0).clamp(0.0, 1.0)
    progress_scale = torch.maximum(progress_scale, support_scale)
    blocked_without_extension_clock = torch.logical_and(
        liftoff_blocked.any(dim=1),
        next_extension.amax(dim=1) == 0,
    )
    progress_scale = torch.where(
        blocked_without_extension_clock,
        torch.zeros_like(progress_scale),
        progress_scale,
    )
    return ContactSchedulerAdvance(
        contact_state=next_contact,
        phase_age=next_phase,
        swing_extension_age=next_extension,
        stance_age=next_stance,
        recovery_state=next_recovery,
        liftoff_blocked=liftoff_blocked,
        progress_scale=progress_scale,
    )


__all__ = [
    "ContactSchedulerAdvance",
    "adaptive_contact_schedule",
    "advance_contact_scheduler",
    "fixed_trot_schedule",
]
