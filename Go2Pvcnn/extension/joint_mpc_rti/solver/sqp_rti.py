"""One-shot SQP real-time iteration update."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.solver.line_search import parallel_line_search
from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem, LqSolution, solve_lq_subproblem


@dataclass(frozen=True)
class SqpRtiUpdate:
    control: Tensor
    delta_control: Tensor
    alpha: Tensor
    merit_before: Tensor
    merit_after: Tensor
    lq_solution: LqSolution


def sqp_rti_update(
    *,
    base_control: Tensor,
    lq_problem: LqProblem,
    merit_fn: Callable[[Tensor], Tensor],
    regularization: float,
    alphas: tuple[float, ...],
) -> SqpRtiUpdate:
    base = torch.as_tensor(base_control)
    lq_solution = solve_lq_subproblem(lq_problem, regularization=regularization)
    search = parallel_line_search(base, lq_solution.delta_control, merit_fn, alphas=alphas)
    return SqpRtiUpdate(
        control=search.control,
        delta_control=lq_solution.delta_control,
        alpha=search.alpha,
        merit_before=merit_fn(base),
        merit_after=search.merit,
        lq_solution=lq_solution,
    )


__all__ = ["SqpRtiUpdate", "sqp_rti_update"]
