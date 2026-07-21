"""Five-candidate loss-only line search for direct state trajectories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0)
FILTER_NAMES = ("finite", "joint_position", "joint_velocity")


@dataclass(frozen=True)
class LineSearchResult:
    state: Tensor
    candidates: Tensor
    alphas: Tensor
    candidate_loss: Tensor
    selected_loss: Tensor
    alpha: Tensor
    selected_index: Tensor
    valid: Tensor
    selected_feasible: Tensor
    used_nominal: Tensor


def parallel_line_search(
    nominal: Tensor,
    direction: Tensor,
    objective: Callable[[Tensor], Tensor],
    *,
    joint_lower: Tensor,
    joint_upper: Tensor,
    joint_velocity_limit: Tensor | float,
    dt: float,
    tie_tolerance: float = 1.0e-7,
) -> LineSearchResult:
    """Select the lowest seven-loss candidate after the three approved filters."""
    base = torch.as_tensor(nominal)
    delta = torch.as_tensor(direction, dtype=base.dtype, device=base.device)
    if base.ndim != 3 or base.shape[1:] != (31, 18) or delta.shape != base.shape:
        raise ValueError("nominal and direction must have shape [B,31,18]")
    alphas = base.new_tensor(ALPHAS)
    candidates = base[:, None] + alphas[None, :, None, None] * delta[:, None]
    batch = int(base.shape[0])
    candidate_loss = objective(candidates.reshape(batch * 5, 31, 18)).reshape(batch, 5)

    finite = torch.isfinite(candidates).all(dim=(2, 3)) & torch.isfinite(candidate_loss)
    lower = torch.as_tensor(joint_lower, dtype=base.dtype, device=base.device)
    upper = torch.as_tensor(joint_upper, dtype=base.dtype, device=base.device)
    joints = candidates[..., 6:]
    position_ok = ((joints >= lower) & (joints <= upper)).all(dim=(2, 3))
    velocity_limit = torch.as_tensor(joint_velocity_limit, dtype=base.dtype, device=base.device)
    joint_step = joints[:, :, 1:] - joints[:, :, :-1]
    velocity_ok = (joint_step.abs() <= velocity_limit * float(dt)).all(dim=(2, 3))
    valid = finite & position_ok & velocity_ok

    selectable = torch.where(valid, candidate_loss, torch.full_like(candidate_loss, float("inf")))
    minimum = selectable.amin(dim=1, keepdim=True)
    tie = selectable <= minimum + float(tie_tolerance)
    selected_index = tie.to(torch.int64).argmax(dim=1)
    any_valid = valid.any(dim=1)
    nominal_index = torch.full_like(selected_index, len(ALPHAS) - 1)
    selected_index = torch.where(any_valid, selected_index, nominal_index)
    row = torch.arange(batch, device=base.device)
    state = candidates[row, selected_index]
    selected_loss = candidate_loss[row, selected_index]
    alpha = alphas[selected_index]
    selected_feasible = valid[row, selected_index]
    return LineSearchResult(
        state=state,
        candidates=candidates,
        alphas=alphas,
        candidate_loss=candidate_loss,
        selected_loss=selected_loss,
        alpha=alpha,
        selected_index=selected_index,
        valid=valid,
        selected_feasible=selected_feasible,
        used_nominal=selected_index == len(ALPHAS) - 1,
    )


__all__ = ["ALPHAS", "FILTER_NAMES", "LineSearchResult", "parallel_line_search"]
