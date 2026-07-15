"""Parallel fixed-candidate line search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.tensor_constants import constant_like


@dataclass(frozen=True)
class LineSearchResult:
    control: Tensor
    merit: Tensor
    base_merit: Tensor
    alpha: Tensor
    selected_index: Tensor
    used_base: Tensor


def parallel_line_search(
    base_control: Tensor,
    delta_control: Tensor,
    merit_fn: Callable[[Tensor], Tensor],
    *,
    alphas: tuple[float, ...],
    base_merit: Tensor | None = None,
) -> LineSearchResult:
    base = torch.as_tensor(base_control)
    delta = torch.as_tensor(delta_control, dtype=base.dtype, device=base.device)
    alpha_tensor = constant_like(base, "line_search_alphas_" + "_".join(map(str, alphas)), alphas)
    candidate_count = int(alpha_tensor.shape[0])
    if base_merit is None:
        all_alphas = torch.cat((alpha_tensor, alpha_tensor.new_zeros(1)))
        evaluated = base[:, None] + all_alphas[None, :, None, None] * delta[:, None]
        batch, evaluated_count = int(evaluated.shape[0]), int(evaluated.shape[1])
        all_merit = merit_fn(evaluated.reshape(batch * evaluated_count, *evaluated.shape[2:])).reshape(
            batch, evaluated_count
        )
        candidate = evaluated[:, :candidate_count]
        candidate_merit = all_merit[:, :candidate_count]
        base_value = all_merit[:, candidate_count]
    else:
        candidate = base[:, None] + alpha_tensor[None, :, None, None] * delta[:, None]
        batch = int(candidate.shape[0])
        candidate_merit = merit_fn(candidate.reshape(batch * candidate_count, *candidate.shape[2:])).reshape(
            batch, candidate_count
        )
        base_value = torch.as_tensor(base_merit, dtype=base.dtype, device=base.device)
    finite = torch.isfinite(candidate_merit)
    improving = torch.logical_and(finite, candidate_merit < base_value[:, None])
    selectable = torch.where(improving, candidate_merit, torch.full_like(candidate_merit, float("inf")))
    best_merit, best_index = selectable.min(dim=1)
    any_improving = improving.any(dim=1)
    gather_index = best_index[:, None, None, None].expand(batch, 1, *base.shape[1:])
    selected_control = torch.gather(candidate, 1, gather_index).squeeze(1)
    selected_alpha = alpha_tensor.index_select(0, best_index)
    return LineSearchResult(
        control=torch.where(any_improving[:, None, None], selected_control, base),
        merit=torch.where(any_improving, best_merit, base_value),
        base_merit=base_value,
        alpha=torch.where(any_improving, selected_alpha, torch.zeros_like(selected_alpha)),
        selected_index=best_index,
        used_base=torch.logical_not(any_improving),
    )


__all__ = ["LineSearchResult", "parallel_line_search"]
