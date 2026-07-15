"""Pending next-step reference storage."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.types import JointMpcPendingReference


class PendingReferenceBuffer:
    def __init__(self, *, num_envs: int, device: torch.device | str) -> None:
        self.valid = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
        self.reference: JointMpcPendingReference | None = None

    def update(self, reference: JointMpcPendingReference) -> None:
        self.reference = reference
        self.valid.copy_(reference.valid)

    def reset_rows(self, env_mask: Tensor) -> None:
        mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self.valid.device)
        if mask.shape != self.valid.shape:
            raise ValueError("env_mask must have shape [B]")
        self.valid.logical_and_(torch.logical_not(mask))


__all__ = ["PendingReferenceBuffer"]
