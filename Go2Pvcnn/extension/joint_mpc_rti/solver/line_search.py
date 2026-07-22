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
    constraint_violation: Tensor
    base_constraint_violation: Tensor


def parallel_line_search(
    base_control: Tensor,
    delta_control: Tensor,
    merit_fn: Callable[[Tensor], Tensor | tuple[Tensor, Tensor]],
    *,
    alphas: tuple[float, ...],
    base_merit: Tensor | None = None,
    base_constraint_violation: Tensor | None = None,
    delta_limit: Tensor | None = None,
    constraint_tolerance: Tensor | None = None,
    required_control: Tensor | None = None,
) -> LineSearchResult:
    base = torch.as_tensor(base_control)
    delta = torch.as_tensor(delta_control, dtype=base.dtype, device=base.device)
    if delta_limit is not None:
        limit = torch.as_tensor(delta_limit, dtype=base.dtype, device=base.device)
        if limit.shape != (base.shape[-1],):
            raise ValueError("delta_limit must be positive with shape [control_dim]")
        capturing = limit.is_cuda and torch.cuda.is_current_stream_capturing()
        if not capturing and torch.any(limit <= 0.0):
            raise ValueError("delta_limit must be positive with shape [control_dim]")
        delta = torch.clamp(delta, min=-limit, max=limit)
    required = (
        torch.zeros_like(delta)
        if required_control is None
        else torch.as_tensor(required_control, dtype=base.dtype, device=base.device)
    )
    if required.shape != base.shape:
        raise ValueError("required_control must match base_control shape")
    alpha_tensor = constant_like(base, "line_search_alphas_" + "_".join(map(str, alphas)), alphas)
    candidate_count = int(alpha_tensor.shape[0])
    candidate = (
        base[:, None]
        + required[:, None]
        + alpha_tensor[None, :, None, None] * delta[:, None]
    )
    if base_merit is None:
        evaluated = torch.cat((candidate, base[:, None]), dim=1)
        batch, evaluated_count = int(evaluated.shape[0]), int(evaluated.shape[1])
        evaluated_result = merit_fn(evaluated.reshape(batch * evaluated_count, *evaluated.shape[2:]))
        if isinstance(evaluated_result, tuple):
            evaluated_merit, evaluated_violation = evaluated_result
        else:
            evaluated_merit = evaluated_result
            evaluated_violation = torch.zeros_like(evaluated_merit).unsqueeze(-1)
        all_merit = evaluated_merit.reshape(batch, evaluated_count)
        all_violation_components = evaluated_violation.reshape(batch, evaluated_count, -1)
        candidate_merit = all_merit[:, :candidate_count]
        candidate_violation_components = all_violation_components[:, :candidate_count]
        base_value = all_merit[:, candidate_count]
        base_violation_components = all_violation_components[:, candidate_count]
    else:
        batch = int(candidate.shape[0])
        evaluated_result = merit_fn(candidate.reshape(batch * candidate_count, *candidate.shape[2:]))
        if isinstance(evaluated_result, tuple):
            evaluated_merit, evaluated_violation = evaluated_result
        else:
            evaluated_merit = evaluated_result
            evaluated_violation = torch.zeros_like(evaluated_merit).unsqueeze(-1)
        candidate_merit = evaluated_merit.reshape(batch, candidate_count)
        candidate_violation_components = evaluated_violation.reshape(batch, candidate_count, -1)
        base_value = torch.as_tensor(base_merit, dtype=base.dtype, device=base.device)
        if base_constraint_violation is None:
            base_violation_components = torch.zeros_like(base_value).unsqueeze(-1)
        else:
            base_violation_components = torch.as_tensor(
                base_constraint_violation, dtype=base.dtype, device=base.device
            ).reshape(batch, -1)
    candidate_violation = candidate_violation_components.amax(dim=-1)
    base_violation = base_violation_components.amax(dim=-1)
    if constraint_tolerance is None:
        tolerance = base.new_full((candidate_violation_components.shape[-1],), 1.0e-9)
    else:
        tolerance = torch.as_tensor(
            constraint_tolerance, dtype=base.dtype, device=base.device
        )
        if tolerance.shape != (candidate_violation_components.shape[-1],):
            raise ValueError("constraint_tolerance must have shape [constraint_components]")
        capturing = tolerance.is_cuda and torch.cuda.is_current_stream_capturing()
        if not capturing and torch.any(tolerance < 0.0):
            raise ValueError("constraint_tolerance must be nonnegative")
    improvement_epsilon = 1.0e-9
    finite = torch.logical_and(
        torch.isfinite(candidate_merit),
        torch.isfinite(candidate_violation_components).all(dim=-1),
    )
    candidate_feasible = (candidate_violation_components <= tolerance).all(dim=-1)
    base_feasible = (base_violation_components <= tolerance).all(dim=-1)
    merit_improving = torch.logical_and(candidate_feasible, candidate_merit < base_value[:, None])
    no_component_worse = (
        candidate_violation_components
        <= base_violation_components[:, None, :] + tolerance[None, None, :]
    ).all(dim=-1)
    component_improves = (
        candidate_violation_components
        < base_violation_components[:, None, :] - improvement_epsilon
    ).any(dim=-1)
    restores = torch.logical_and(
        no_component_worse,
        component_improves,
    )
    improving = torch.logical_and(
        finite,
        torch.where(base_feasible[:, None], merit_improving, restores),
    )
    selection_score = torch.where(
        base_feasible[:, None],
        candidate_merit,
        candidate_violation,
    )
    selectable = torch.where(improving, selection_score, torch.full_like(selection_score, float("inf")))
    _, best_index = selectable.min(dim=1)
    any_improving = improving.any(dim=1)
    gather_index = best_index[:, None, None, None].expand(batch, 1, *base.shape[1:])
    selected_control = torch.gather(candidate, 1, gather_index).squeeze(1)
    selected_alpha = alpha_tensor.index_select(0, best_index)
    selected_merit = torch.gather(candidate_merit, 1, best_index[:, None]).squeeze(1)
    selected_violation = torch.gather(candidate_violation, 1, best_index[:, None]).squeeze(1)
    return LineSearchResult(
        control=torch.where(any_improving[:, None, None], selected_control, base),
        merit=torch.where(any_improving, selected_merit, base_value),
        base_merit=base_value,
        alpha=torch.where(any_improving, selected_alpha, torch.zeros_like(selected_alpha)),
        selected_index=best_index,
        used_base=torch.logical_not(any_improving),
        constraint_violation=torch.where(any_improving, selected_violation, base_violation),
        base_constraint_violation=base_violation,
    )


__all__ = ["LineSearchResult", "parallel_line_search"]
