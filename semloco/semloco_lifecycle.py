"""Per-environment target locking for swing phases."""

from __future__ import annotations

import torch
from torch import Tensor


class SemlocoTargetCache:
    def __init__(self, num_envs: int, device: torch.device | str):
        self.target = torch.zeros(num_envs, 4, 3, device=device)
        self.valid = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
        self.swing = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)

    def reset(self, env_ids: Tensor | slice) -> None:
        ids = torch.arange(self.target.shape[0], device=self.target.device) if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.target.device)
        self.target[ids] = 0
        self.valid[ids] = False
        self.swing[ids] = False

    def update_swing_targets(self, output, swing_mask: Tensor, touchdown_mask: Tensor | None = None) -> None:
        swing = torch.as_tensor(swing_mask, dtype=torch.bool, device=self.target.device)
        touchdown = torch.zeros_like(swing) if touchdown_mask is None else torch.as_tensor(touchdown_mask, dtype=torch.bool, device=self.target.device)
        new_swing = swing & ~self.swing
        writable = new_swing | touchdown
        self.target = torch.where(writable[..., None], output.target_foothold_w, self.target)
        self.valid = torch.where(writable, output.valid, self.valid)
        self.swing = swing


__all__ = ["SemlocoTargetCache"]
