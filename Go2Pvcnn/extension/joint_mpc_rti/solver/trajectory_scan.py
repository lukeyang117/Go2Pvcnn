"""H30/32 fixed-tree constrained LQ solver contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.solver.associative_scan import (
    ConditionalValueFactor,
    combine_conditional_value_factors,
)
from extension.joint_mpc_rti.solver.lq_problem import LqProblem
from extension.joint_mpc_rti.solver.trajectory_qp import (
    QpSolution,
    TrajectoryQp,
    refine_active_set,
    solve_dense_active_kkt,
    solve_dense_qp,
)


INTERVALS = 30
PADDED_INTERVALS = 32


@dataclass(frozen=True)
class TrajectoryScanSolution:
    direction: Tensor
    kkt_primal_residual: Tensor
    kkt_dual_residual: Tensor
    slack_max: dict[str, Tensor]
    active_constraint_count: dict[str, Tensor]
    dense_parity_error: Tensor


def pad_h30_factors(
    factors: ConditionalValueFactor,
) -> tuple[ConditionalValueFactor, Tensor]:
    """Append two identity/no-cost intervals to exactly 30 real factors."""
    if any(int(value.shape[0]) != INTERVALS for value in factors):
        raise ValueError("trajectory factors must contain exactly 30 intervals")
    matrix_a, vector_c, matrix_c, vector_p, matrix_p = factors
    dimension = int(matrix_a.shape[-1])
    identity = torch.eye(
        dimension, dtype=matrix_a.dtype, device=matrix_a.device
    ).expand(2, *matrix_a.shape[1:-2], dimension, dimension)

    def append_zeros(value: Tensor) -> Tensor:
        return torch.cat((value, value.new_zeros(2, *value.shape[1:])), dim=0)

    padded = (
        torch.cat((matrix_a, identity), dim=0),
        append_zeros(vector_c),
        append_zeros(matrix_c),
        append_zeros(vector_p),
        append_zeros(matrix_p),
    )
    valid = torch.arange(PADDED_INTERVALS, device=matrix_a.device) < INTERVALS
    return padded, valid


def _combine_pairs(factors: ConditionalValueFactor) -> ConditionalValueFactor:
    left = tuple(value[0::2] for value in factors)
    right = tuple(value[1::2] for value in factors)
    return combine_conditional_value_factors(left, right)


def fixed_five_level_tree(
    factors: ConditionalValueFactor,
) -> tuple[ConditionalValueFactor, ...]:
    level1 = _combine_pairs(factors)
    level2 = _combine_pairs(level1)
    level3 = _combine_pairs(level2)
    level4 = _combine_pairs(level3)
    level5 = _combine_pairs(level4)
    return factors, level1, level2, level3, level4, level5


def factor_tree_shapes() -> tuple[int, ...]:
    return (32, 16, 8, 4, 2, 1)


def _from_qp_solution(
    solution: QpSolution, parity_reference: QpSolution
) -> TrajectoryScanSolution:
    parity = (solution.direction - parity_reference.direction).abs().amax(dim=(1, 2))
    return TrajectoryScanSolution(
        direction=solution.direction,
        kkt_primal_residual=solution.kkt_primal_residual,
        kkt_dual_residual=solution.kkt_dual_residual,
        slack_max=solution.slack_max,
        active_constraint_count=solution.active_constraint_count,
        dense_parity_error=parity,
    )


def solve_trajectory_qp_scan(problem: LqProblem | TrajectoryQp):
    """Solve one already-built LQ problem with two fixed active refinements.

    The final compiled factor recovery replaces this eager reference in the
    performance task; this boundary already consumes the final full-horizon
    problem and never rebuilds nonlinear residuals during refinement.
    """
    refinement_count = 2
    if isinstance(problem, TrajectoryQp):
        return refine_active_set(
            problem, solve_dense_active_kkt, refinements=refinement_count
        )
    solution = solve_dense_qp(problem, refinements=refinement_count)
    parity_reference = solution
    return _from_qp_solution(solution, parity_reference)


__all__ = [
    "TrajectoryScanSolution",
    "factor_tree_shapes",
    "fixed_five_level_tree",
    "pad_h30_factors",
    "solve_trajectory_qp_scan",
]
