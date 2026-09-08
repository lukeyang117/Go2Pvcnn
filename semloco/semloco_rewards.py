"""Pure SemLoco reward terms."""

from __future__ import annotations

import torch
from torch import Tensor


def semantic_foothold_tracking_reward(
    foot_pos_w: Tensor, target_foothold_w: Tensor, swing_mask: Tensor, valid_mask: Tensor, sigma_foot: float = 0.10
) -> Tensor:
    error_sq = (foot_pos_w - target_foothold_w).square().sum(dim=-1)
    active = torch.as_tensor(swing_mask, dtype=torch.bool, device=error_sq.device) & torch.as_tensor(
        valid_mask, dtype=torch.bool, device=error_sq.device
    )
    score = torch.exp(-error_sq / max(float(sigma_foot), 1e-6)) * active
    return score.sum(-1) / active.sum(-1).clamp_min(1).to(score.dtype) * (active.any(-1)).to(score.dtype)


def semloco_clearance_penalty(
    foot_pos_w: Tensor, swing_phase: Tensor, swing_mask: Tensor, swing_height: float = 0.08, margin: float = 0.02
) -> Tensor:
    phase = torch.as_tensor(swing_phase, dtype=foot_pos_w.dtype, device=foot_pos_w.device).clamp(0.0, 1.0)
    mask = torch.as_tensor(swing_mask, dtype=torch.bool, device=foot_pos_w.device)
    z_ref = float(swing_height) * torch.sqrt(torch.sin(torch.pi * phase).clamp_min(0.0)) + float(margin)
    deficit = torch.relu(z_ref - foot_pos_w[..., 2])
    return -(deficit.square() * mask).sum(-1)


__all__ = ["semantic_foothold_tracking_reward", "semloco_clearance_penalty"]
