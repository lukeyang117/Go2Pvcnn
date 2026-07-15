"""Fixed gait masks for the MPC-QP backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class QpGaitMasks:
    swing_mask: Tensor
    stance_mask: Tensor


def alternating_diagonal_gait_masks(
    *,
    batch: int,
    horizon: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> QpGaitMasks:
    swing = torch.zeros((int(batch), int(horizon), 4), dtype=torch.bool, device=device)
    split = max(1, int(horizon) // 2)
    swing[:, :split, 1] = True
    swing[:, :split, 2] = True
    swing[:, split:, 0] = True
    swing[:, split:, 3] = True
    stance = torch.logical_not(swing)
    if dtype is not torch.bool:
        return QpGaitMasks(swing_mask=swing.to(dtype=dtype), stance_mask=stance.to(dtype=dtype))
    return QpGaitMasks(swing_mask=swing, stance_mask=stance)


__all__ = ["QpGaitMasks", "alternating_diagonal_gait_masks"]
