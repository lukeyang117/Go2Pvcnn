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


def sqp_rti_update(
    *,
    base_control: Tensor,
    lq_problem: LqProblem,
    merit_fn: Callable[[Tensor], Tensor],
    regularization: float,
    alphas: tuple[float, ...],
    diagonal_state_riccati: bool = False,
    coupled_state_riccati: bool = False,
    base_merit: Tensor | None = None,
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
    search = parallel_line_search(
        base,
        lq_solution.delta_control,
        merit_fn,
        alphas=alphas,
        base_merit=base_merit,
    )
    return SqpRtiUpdate(
        control=search.control,
        delta_control=lq_solution.delta_control,
        alpha=search.alpha,
        merit_before=search.base_merit,
        merit_after=search.merit,
        lq_solution=lq_solution,
        selected_index=search.selected_index,
        used_base=search.used_base,
    )


__all__ = ["SqpRtiUpdate", "sqp_rti_update"]
