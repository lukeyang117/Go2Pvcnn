"""Parallel fixed-candidate line search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LineSearchResult:
    control: Tensor
    merit: Tensor
    alpha: Tensor


def parallel_line_search(
    base_control: Tensor,
    delta_control: Tensor,
    merit_fn: Callable[[Tensor], Tensor],
    *,
    alphas: tuple[float, ...],
) -> LineSearchResult:
    base = torch.as_tensor(base_control)
    delta = torch.as_tensor(delta_control, dtype=base.dtype, device=base.device)
    alpha_tensor = base.new_tensor(alphas)
    candidate = base[:, None] + alpha_tensor[None, :, None, None] * delta[:, None]
    batch, candidate_count = int(candidate.shape[0]), int(candidate.shape[1])
    candidate_merit = merit_fn(candidate.reshape(batch * candidate_count, *candidate.shape[2:])).reshape(
        batch, candidate_count
    )
    base_merit = merit_fn(base)
    finite = torch.isfinite(candidate_merit)
    improving = torch.logical_and(finite, candidate_merit < base_merit[:, None])
    selectable = torch.where(improving, candidate_merit, torch.full_like(candidate_merit, float("inf")))
    best_merit, best_index = selectable.min(dim=1)
    any_improving = improving.any(dim=1)
    gather_index = best_index[:, None, None, None].expand(batch, 1, *base.shape[1:])
    selected_control = torch.gather(candidate, 1, gather_index).squeeze(1)
    selected_alpha = alpha_tensor.index_select(0, best_index)
    return LineSearchResult(
        control=torch.where(any_improving[:, None, None], selected_control, base),
        merit=torch.where(any_improving, best_merit, base_merit),
        alpha=torch.where(any_improving, selected_alpha, torch.zeros_like(selected_alpha)),
    )


__all__ = ["LineSearchResult", "parallel_line_search"]
