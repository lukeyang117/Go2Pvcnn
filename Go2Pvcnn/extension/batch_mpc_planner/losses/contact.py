"""Contact-related losses."""

from __future__ import annotations

import torch
from torch import Tensor


def contact_binary_loss(contact_prob: Tensor) -> Tensor:
    return (contact_prob * (1.0 - contact_prob)).mean(dim=(1, 2))


def contact_transition_loss(contact_prob: Tensor) -> Tensor:
    if int(contact_prob.shape[1]) < 2:
        return torch.zeros(contact_prob.shape[0], dtype=contact_prob.dtype, device=contact_prob.device)
    return torch.abs(contact_prob[:, 1:] - contact_prob[:, :-1]).mean(dim=(1, 2))


def support_stability_loss(
    contact_prob: Tensor,
    *,
    min_support_legs: int,
    contact_threshold: float = 0.5,
) -> Tensor:
    legs = int(contact_prob.shape[-1])
    support_count = max(1, min(int(min_support_legs), legs))
    top_support = torch.topk(contact_prob, k=support_count, dim=-1).values
    deficit = torch.relu(float(contact_threshold) - top_support)
    return deficit.sum(dim=-1).mean(dim=-1)


__all__ = [
    "contact_binary_loss",
    "contact_transition_loss",
    "support_stability_loss",
]
