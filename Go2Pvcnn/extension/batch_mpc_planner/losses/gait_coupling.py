"""Gait-coupling losses to tie root tracking to feasible foot motion."""

from __future__ import annotations

import torch
from torch import Tensor


def stance_slip_loss(contact_prob: Tensor, foot_pos: Tensor, *, slip_tolerance_m_per_step: float) -> Tensor:
    """Penalize support-phase foot motion above a small tolerance."""
    if int(foot_pos.shape[1]) < 2:
        return torch.zeros(foot_pos.shape[0], dtype=foot_pos.dtype, device=foot_pos.device)
    dfoot = foot_pos[:, 1:] - foot_pos[:, :-1]
    dnorm = torch.linalg.vector_norm(dfoot, dim=-1)  # [B, T-1, 4]
    support = 0.5 * (contact_prob[:, 1:] + contact_prob[:, :-1])
    excess = torch.relu(dnorm - float(slip_tolerance_m_per_step))
    return (support * excess).mean(dim=(1, 2))


def swing_stride_loss(
    contact_prob: Tensor,
    foot_pos: Tensor,
    command: Tensor,
    *,
    min_swing_span_m: float,
    command_speed_deadzone_mps: float,
) -> Tensor:
    """Require non-trivial swing span along command direction when moving."""
    command_xy = command[:, :2]
    cmd_speed = torch.linalg.vector_norm(command_xy, dim=-1)
    moving = (cmd_speed > float(command_speed_deadzone_mps)).to(dtype=foot_pos.dtype)
    safe_speed = torch.where(cmd_speed > 1.0e-6, cmd_speed, torch.ones_like(cmd_speed))
    heading = command_xy / safe_speed.unsqueeze(-1)  # [B,2]
    foot_xy = foot_pos[..., :2]  # [B,T,4,2]
    proj = (foot_xy * heading[:, None, None, :]).sum(dim=-1)  # [B,T,4]
    swing_mask = (1.0 - contact_prob).to(dtype=foot_pos.dtype)
    valid = swing_mask > 0.05
    max_fill = torch.full_like(proj, -1.0e6)
    min_fill = torch.full_like(proj, 1.0e6)
    proj_max = torch.where(valid, proj, max_fill).amax(dim=1)
    proj_min = torch.where(valid, proj, min_fill).amin(dim=1)
    swing_span = proj_max - proj_min  # [B,4]
    has_swing = valid.any(dim=1).to(dtype=foot_pos.dtype)
    span_penalty = torch.relu(float(min_swing_span_m) - swing_span) * has_swing
    per_env = span_penalty.mean(dim=-1)
    return moving * per_env


def root_frame_drift_loss(
    root_pos: Tensor,
    foot_pos: Tensor,
    *,
    min_rel_m: float,
    max_rel_m: float,
) -> Tensor:
    """Keep feet within a feasible root-relative workspace envelope."""
    rel = foot_pos - root_pos.unsqueeze(2)
    rel_norm = torch.linalg.vector_norm(rel, dim=-1)
    low = torch.relu(float(min_rel_m) - rel_norm)
    high = torch.relu(rel_norm - float(max_rel_m))
    return (low + high).mean(dim=(1, 2))


def root_frame_follow_loss(
    root_pos: Tensor,
    foot_pos: Tensor,
    *,
    rel_change_tolerance_m_per_step: float,
) -> Tensor:
    """Penalize abrupt foot motion in root frame across horizon steps."""
    if int(foot_pos.shape[1]) < 2:
        return torch.zeros(foot_pos.shape[0], dtype=foot_pos.dtype, device=foot_pos.device)
    rel = foot_pos - root_pos.unsqueeze(2)
    drel = rel[:, 1:] - rel[:, :-1]
    dnorm = torch.linalg.vector_norm(drel, dim=-1)
    excess = torch.relu(dnorm - float(rel_change_tolerance_m_per_step))
    return excess.mean(dim=(1, 2))


__all__ = [
    "root_frame_drift_loss",
    "root_frame_follow_loss",
    "stance_slip_loss",
    "swing_stride_loss",
]
