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


def support_stability_loss(contact_prob: Tensor, *, min_support_legs: int) -> Tensor:
    support = contact_prob.sum(dim=-1)
    return torch.relu(float(min_support_legs) - support).mean(dim=-1)


def contact_schedule_tracking_loss(
    contact_prob: Tensor,
    nominal_contact_prob: Tensor,
    *,
    min_support_prob: float,
) -> Tensor:
    fit = torch.square(contact_prob - nominal_contact_prob).mean(dim=(1, 2))
    support = torch.relu(float(min_support_prob) - contact_prob.sum(dim=-1)).mean(dim=-1)
    return fit + support


__all__ = [
    "contact_binary_loss",
    "contact_schedule_tracking_loss",
    "contact_transition_loss",
    "support_stability_loss",
]
