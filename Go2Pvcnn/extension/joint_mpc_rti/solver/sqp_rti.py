"""One-shot SQP real-time iteration update."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.solver.line_search import parallel_line_search
from extension.joint_mpc_rti.solver.primal_dual_ilqr import (
    LqProblem,
    LqSolution,
    solve_diagonal_lq_subproblem,
    solve_go2_block_lq_subproblem,
    solve_lq_subproblem,
)


@dataclass(frozen=True)
class SqpRtiUpdate:
    control: Tensor
    delta_control: Tensor
    alpha: Tensor
    merit_before: Tensor
    merit_after: Tensor
    lq_solution: LqSolution
    selected_index: Tensor
    used_base: Tensor
    constraint_violation: Tensor
    base_constraint_violation: Tensor


def sqp_rti_update(
    *,
    base_control: Tensor,
    lq_problem: LqProblem,
    merit_fn: Callable[[Tensor], Tensor | tuple[Tensor, Tensor]],
    regularization: float,
    alphas: tuple[float, ...],
    diagonal_state_riccati: bool = False,
    coupled_state_riccati: bool = False,
    base_merit: Tensor | None = None,
    base_constraint_violation: Tensor | None = None,
    recover_control_direction: Callable[
        [LqSolution], Tensor | tuple[Tensor, Tensor]
    ] | None = None,
    delta_limit: Tensor | None = None,
    constraint_tolerance: Tensor | None = None,
) -> SqpRtiUpdate:
    base = torch.as_tensor(base_control)
    if bool(coupled_state_riccati):
        lq_solution = solve_lq_subproblem(lq_problem, regularization=regularization)
    elif bool(diagonal_state_riccati):
        lq_solution = solve_diagonal_lq_subproblem(lq_problem, regularization=regularization)
    elif int(base.shape[-1]) == 18:
        lq_solution = solve_go2_block_lq_subproblem(lq_problem, regularization=regularization)
    else:
        lq_solution = solve_lq_subproblem(lq_problem, regularization=regularization)
    recovered = (
        lq_solution.delta_control
        if recover_control_direction is None
        else recover_control_direction(lq_solution)
    )
    if isinstance(recovered, tuple):
        required_control = torch.as_tensor(
            recovered[0], dtype=base.dtype, device=base.device
        )
        free_control = torch.as_tensor(
            recovered[1], dtype=base.dtype, device=base.device
        )
        delta_control = required_control + free_control
    else:
        required_control = None
        free_control = torch.as_tensor(recovered, dtype=base.dtype, device=base.device)
        delta_control = free_control
    if delta_control.shape != base.shape:
        raise ValueError("recovered control direction must match base_control shape")
    if required_control is not None and required_control.shape != base.shape:
        raise ValueError("required control correction must match base_control shape")
    search = parallel_line_search(
        base,
        free_control,
        merit_fn,
        alphas=alphas,
        base_merit=base_merit,
        base_constraint_violation=base_constraint_violation,
        delta_limit=delta_limit,
        constraint_tolerance=constraint_tolerance,
        required_control=required_control,
    )
    return SqpRtiUpdate(
        control=search.control,
        delta_control=delta_control,
        alpha=search.alpha,
        merit_before=search.base_merit,
        merit_after=search.merit,
        lq_solution=lq_solution,
        selected_index=search.selected_index,
        used_base=search.used_base,
        constraint_violation=search.constraint_violation,
        base_constraint_violation=search.base_constraint_violation,
    )


__all__ = ["SqpRtiUpdate", "sqp_rti_update"]
