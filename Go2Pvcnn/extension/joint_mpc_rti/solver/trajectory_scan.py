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
from extension.joint_mpc_rti.solver.fixed_general import fixed_general_solve
from extension.joint_mpc_rti.solver.fixed_spd import fixed_spd_solve
from extension.joint_mpc_rti.solver.trajectory_qp import (
    QpSolution,
    TrajectoryQp,
    refine_active_set,
    solve_dense_active_kkt,
)
from extension.joint_mpc_rti.tensor_constants import constant_like


INTERVALS = 30
PADDED_INTERVALS = 32
STATE_DIM = 18
SEPARATOR_DIM = 36


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
    identity = constant_like(
        matrix_a,
        f"trajectory_factor_identity_{dimension}",
        tuple(
            tuple(1.0 if row == column else 0.0 for column in range(dimension))
            for row in range(dimension)
        ),
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
    valid = constant_like(
        matrix_a,
        "trajectory_factor_valid_mask",
        (1.0,) * INTERVALS + (0.0,) * (PADDED_INTERVALS - INTERVALS),
    ).to(torch.bool)
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


def _trajectory_factors_from_blocks(
    diagonal: Tensor,
    first_offdiag: Tensor,
    second_offdiag: Tensor,
    gradient: Tensor,
) -> ConditionalValueFactor:
    """Map a direct-state pentadiagonal objective to 36D separator factors."""
    batch = int(gradient.shape[0])
    identity = constant_like(
        gradient,
        "trajectory_scan_state_identity",
        tuple(
            tuple(1.0 if row == column else 0.0 for column in range(STATE_DIM))
            for row in range(STATE_DIM)
        ),
    )
    dynamics_a = gradient.new_zeros(batch, INTERVALS, SEPARATOR_DIM, SEPARATOR_DIM)
    dynamics_a[..., :STATE_DIM, STATE_DIM:] = identity
    dynamics_b = gradient.new_zeros(batch, INTERVALS, SEPARATOR_DIM, STATE_DIM)
    dynamics_b[..., STATE_DIM:, :] = identity
    cross = gradient.new_zeros(batch, INTERVALS, SEPARATOR_DIM, STATE_DIM)
    cross[..., STATE_DIM:, :] = first_offdiag
    cross[:, 1:, :STATE_DIM, :] = second_offdiag

    control_hessian = diagonal[:, 1:]
    control_gradient = gradient[:, 1:]
    solve_b_and_gradient = fixed_spd_solve(
        control_hessian,
        torch.cat(
            (dynamics_b.transpose(-1, -2), control_gradient.unsqueeze(-1)), dim=-1
        ),
    )
    solve_cross = fixed_spd_solve(
        control_hessian, cross.transpose(-1, -2)
    )
    b_inverse = solve_b_and_gradient[..., :SEPARATOR_DIM].transpose(-1, -2)
    inverse_gradient = solve_b_and_gradient[..., SEPARATOR_DIM]
    cross_inverse = solve_cross.transpose(-1, -2)
    matrix_a = dynamics_a - b_inverse @ cross.transpose(-1, -2)
    vector_c = -(dynamics_b @ inverse_gradient.unsqueeze(-1)).squeeze(-1)
    matrix_c = b_inverse @ dynamics_b.transpose(-1, -2)
    vector_p = -(cross @ inverse_gradient.unsqueeze(-1)).squeeze(-1)
    matrix_p = -(cross_inverse @ cross.transpose(-1, -2))
    return tuple(
        value.movedim(1, 0)
        for value in (matrix_a, vector_c, matrix_c, vector_p, matrix_p)
    )


def _split_boundaries(
    left: ConditionalValueFactor,
    right: ConditionalValueFactor,
    state_left: Tensor,
    costate_right: Tensor,
) -> tuple[Tensor, Tensor]:
    a_left, c_left, c_matrix_left, _, _ = left
    a_right, _, _, p_right, p_matrix_right = right
    dimension = int(a_left.shape[-1])
    identity = constant_like(
        a_left,
        f"trajectory_split_identity_{dimension}",
        tuple(
            tuple(1.0 if row == column else 0.0 for column in range(dimension))
            for row in range(dimension)
        ),
    )
    right_value = p_right + (
        a_right.transpose(-1, -2) @ costate_right.unsqueeze(-1)
    ).squeeze(-1)
    rhs = (
        (a_left @ state_left.unsqueeze(-1)).squeeze(-1)
        + c_left
        - (c_matrix_left @ right_value.unsqueeze(-1)).squeeze(-1)
    )
    state_middle = fixed_general_solve(
        identity + c_matrix_left @ p_matrix_right, rhs.unsqueeze(-1)
    ).squeeze(-1)
    costate_middle = right_value + (
        p_matrix_right @ state_middle.unsqueeze(-1)
    ).squeeze(-1)
    return state_middle, costate_middle


def _expand_boundaries(
    child_factors: ConditionalValueFactor,
    state_left: Tensor,
    costate_right: Tensor,
) -> tuple[Tensor, Tensor]:
    left = tuple(value[0::2] for value in child_factors)
    right = tuple(value[1::2] for value in child_factors)
    state_middle, costate_middle = _split_boundaries(
        left, right, state_left, costate_right
    )
    state_children = torch.stack((state_left, state_middle), dim=1).flatten(0, 1)
    costate_children = torch.stack((costate_middle, costate_right), dim=1).flatten(0, 1)
    return state_children, costate_children


def _recover_direction(
    levels: tuple[ConditionalValueFactor, ...], batch: int
) -> Tensor:
    root = levels[-1]
    state_left = root[0].new_zeros(1, batch, SEPARATOR_DIM)
    costate_right = root[0].new_zeros(1, batch, SEPARATOR_DIM)
    for child_factors in reversed(levels[:-1]):
        state_left, costate_right = _expand_boundaries(
            child_factors, state_left, costate_right
        )
    final_state = (
        levels[0][0][-1] @ state_left[-1].unsqueeze(-1)
    ).squeeze(-1) + levels[0][1][-1] - (
        levels[0][2][-1] @ costate_right[-1].unsqueeze(-1)
    ).squeeze(-1)
    boundaries = torch.cat((state_left, final_state.unsqueeze(0)), dim=0)
    return boundaries[:31, :, STATE_DIM:].movedim(0, 1)


def _solve_augmented_associative(
    diagonal: Tensor,
    first_offdiag: Tensor,
    second_offdiag: Tensor,
    gradient: Tensor,
) -> Tensor:
    factors = _trajectory_factors_from_blocks(
        diagonal, first_offdiag, second_offdiag, gradient
    )
    padded, _ = pad_h30_factors(factors)
    return _recover_direction(fixed_five_level_tree(padded), int(gradient.shape[0]))


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


def _add_local_rows(
    diagonal: Tensor,
    gradient: Tensor,
    rows: Tensor,
    target: Tensor,
    active: Tensor,
    penalty: float,
) -> tuple[Tensor, Tensor]:
    batch, nodes = rows.shape[:2]
    flat_rows = rows.reshape(batch, nodes, -1, rows.shape[-1])
    flat_target = target.reshape(batch, nodes, -1)
    flat_active = active.reshape(batch, nodes, -1).to(rows.dtype)
    weighted = flat_rows * flat_active[..., None]
    diagonal = diagonal + float(penalty) * torch.einsum(
        "bnri,bnrj->bnij", weighted, flat_rows
    )
    gradient = gradient - float(penalty) * torch.einsum(
        "bnri,bnr->bni", weighted, flat_target
    )
    return diagonal, gradient


def _augmented_system(
    problem: LqProblem,
    direction: Tensor,
    active: dict[str, Tensor],
    *,
    penalty: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    diagonal = problem.diagonal.clone()
    first = problem.first_offdiag.clone()
    second = problem.second_offdiag.clone()
    gradient = problem.gradient.clone()

    stance_active = problem.stance_active[..., None].expand(-1, -1, -1, 3).clone()
    stance_active[:, 0] = False
    diagonal, gradient = _add_local_rows(
        diagonal,
        gradient,
        problem.stance_rows,
        problem.stance_target,
        stance_active,
        penalty,
    )
    diagonal, gradient = _add_local_rows(
        diagonal,
        gradient,
        problem.touchdown_plane_rows,
        problem.touchdown_plane_target,
        problem.touchdown_plane_active,
        penalty,
    )
    diagonal, gradient = _add_local_rows(
        diagonal,
        gradient,
        problem.touchdown_region_rows.transpose(-1, -2),
        problem.touchdown_region_target - float(problem.slack_caps["region"]),
        active["region"],
        penalty,
    )
    diagonal, gradient = _add_local_rows(
        diagonal,
        gradient,
        problem.clearance_rows,
        problem.clearance_target - float(problem.slack_caps["collision"]),
        active["clearance"],
        penalty,
    )

    box_low = active["box_low"].to(diagonal.dtype)
    box_high = active["box_high"].to(diagonal.dtype)
    box_weight = box_low + box_high
    box_target = box_low * problem.lower + box_high * problem.upper
    diagonal = diagonal + float(penalty) * torch.diag_embed(box_weight)
    gradient = gradient - float(penalty) * box_target

    rate_low = active["rate_low"].to(diagonal.dtype)
    rate_high = active["rate_high"].to(diagonal.dtype)
    rate_weight = rate_low + rate_high
    rate_target = rate_low * problem.rate_lower + rate_high * problem.rate_upper
    rate_coordinates = constant_like(
        diagonal, "scan_rate_coordinates", (2, 3, 4) + tuple(range(6, 18))
    ).to(torch.long)
    rate_diagonal = torch.zeros_like(first[..., 0])
    rate_diagonal.scatter_add_(2, rate_coordinates.view(1, 1, -1).expand_as(rate_weight), rate_weight)
    diagonal[:, :-1] = diagonal[:, :-1] + float(penalty) * torch.diag_embed(rate_diagonal)
    diagonal[:, 1:] = diagonal[:, 1:] + float(penalty) * torch.diag_embed(rate_diagonal)
    rate_first = torch.zeros_like(first[..., 0])
    rate_first.scatter_add_(2, rate_coordinates.view(1, 1, -1).expand_as(rate_weight), -rate_weight)
    first = first + float(penalty) * torch.diag_embed(rate_first)
    rate_gradient = torch.zeros_like(gradient)
    rate_gradient[:, :-1].scatter_add_(2, rate_coordinates.view(1, 1, -1).expand_as(rate_target), float(penalty) * rate_target)
    rate_gradient[:, 1:].scatter_add_(2, rate_coordinates.view(1, 1, -1).expand_as(rate_target), -float(penalty) * rate_target)
    gradient = gradient + rate_gradient

    identity = constant_like(
        diagonal,
        "scan_state_identity",
        tuple(tuple(1.0 if i == j else 0.0 for j in range(18)) for i in range(18)),
    )
    diagonal[:, 0] = identity
    gradient[:, 0] = 0.0
    first[:, 0] = 0.0
    second[:, 0] = 0.0
    return diagonal, first, second, gradient


def _constraint_activity(problem: LqProblem, direction: Tensor) -> dict[str, Tensor]:
    fixed = problem.lower == problem.upper
    box_low = (direction < problem.lower - 1.0e-7) & ~fixed
    box_high = (direction > problem.upper + 1.0e-7) & ~fixed & ~box_low
    rate_value = torch.cat(
        (
            direction[:, 1:, 2:5] - direction[:, :-1, 2:5],
            direction[:, 1:, 6:] - direction[:, :-1, 6:],
        ),
        dim=-1,
    )
    rate_low = rate_value < problem.rate_lower - 1.0e-7
    rate_high = (rate_value > problem.rate_upper + 1.0e-7) & ~rate_low
    region_value = torch.einsum(
        "bnlri,bnli->bnlr",
        problem.touchdown_region_rows.transpose(-1, -2),
        direction[:, :, None].expand(-1, -1, 4, -1),
    )
    region = problem.touchdown_region_active & (
        region_value < problem.touchdown_region_target - float(problem.slack_caps["region"])
    )
    clearance_value = torch.einsum(
        "bnri,bni->bnr", problem.clearance_rows, direction
    )
    clearance = problem.clearance_active & (
        clearance_value < problem.clearance_target - float(problem.slack_caps["collision"])
    )
    return {
        "box_low": box_low,
        "box_high": box_high,
        "rate_low": rate_low,
        "rate_high": rate_high,
        "region": region,
        "clearance": clearance,
    }


def _merge_activity(left: dict[str, Tensor], right: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: left[name] | right[name] for name in left}


def _solve_lq_problem(problem: LqProblem) -> TrajectoryScanSolution:
    empty = {
        "box_low": torch.zeros_like(problem.lower, dtype=torch.bool),
        "box_high": torch.zeros_like(problem.upper, dtype=torch.bool),
        "rate_low": torch.zeros_like(problem.rate_lower, dtype=torch.bool),
        "rate_high": torch.zeros_like(problem.rate_upper, dtype=torch.bool),
        "region": torch.zeros_like(problem.touchdown_region_active),
        "clearance": torch.zeros_like(problem.clearance_active),
    }
    penalty = 1.0e8 if problem.gradient.dtype == torch.float64 else 2.0e4
    direction = torch.zeros_like(problem.gradient)
    active = empty
    for _refinement in range(3):
        diagonal, first, second, gradient = _augmented_system(
            problem, direction, active, penalty=penalty
        )
        direction = _solve_augmented_associative(
            diagonal, first, second, gradient
        )
        direction[:, 0] = 0.0
        active = _merge_activity(active, _constraint_activity(problem, direction))

    stance_error = torch.einsum(
        "bnlri,bni->bnlr", problem.stance_rows, direction
    ) - problem.stance_target
    stance_error = torch.where(
        problem.stance_active[..., None], stance_error, torch.zeros_like(stance_error)
    )
    plane_error = torch.einsum(
        "bnli,bni->bnl", problem.touchdown_plane_rows, direction
    ) - problem.touchdown_plane_target
    plane_error = torch.where(
        problem.touchdown_plane_active, plane_error, torch.zeros_like(plane_error)
    )
    box_violation = torch.maximum(problem.lower - direction, direction - problem.upper).clamp_min(0.0)
    rate_value = torch.cat(
        (
            direction[:, 1:, 2:5] - direction[:, :-1, 2:5],
            direction[:, 1:, 6:] - direction[:, :-1, 6:],
        ),
        dim=-1,
    )
    rate_violation = torch.maximum(
        problem.rate_lower - rate_value, rate_value - problem.rate_upper
    ).clamp_min(0.0)
    region_value = torch.einsum(
        "bnlri,bnli->bnlr",
        problem.touchdown_region_rows.transpose(-1, -2),
        direction[:, :, None].expand(-1, -1, 4, -1),
    )
    region_slack = torch.where(
        problem.touchdown_region_active,
        (problem.touchdown_region_target - region_value).clamp_min(0.0),
        torch.zeros_like(region_value),
    )
    clearance_value = torch.einsum("bnri,bni->bnr", problem.clearance_rows, direction)
    collision_slack = torch.where(
        problem.clearance_active,
        (problem.clearance_target - clearance_value).clamp_min(0.0),
        torch.zeros_like(clearance_value),
    )
    primal = torch.stack(
        (
            box_violation.flatten(1).amax(1),
            rate_violation.flatten(1).amax(1),
            stance_error.abs().flatten(1).amax(1),
            plane_error.abs().flatten(1).amax(1),
            (region_slack - float(problem.slack_caps["region"])).clamp_min(0.0).flatten(1).amax(1),
            (collision_slack - float(problem.slack_caps["collision"])).clamp_min(0.0).flatten(1).amax(1),
        ),
        dim=1,
    ).amax(1)
    stationarity = (
        torch.matmul(diagonal, direction.unsqueeze(-1)).squeeze(-1) + gradient
    )
    stationarity[:, :-1] = stationarity[:, :-1] + (
        first @ direction[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    stationarity[:, 1:] = stationarity[:, 1:] + (
        first.transpose(-1, -2) @ direction[:, :-1].unsqueeze(-1)
    ).squeeze(-1)
    stationarity[:, :-2] = stationarity[:, :-2] + (
        second @ direction[:, 2:].unsqueeze(-1)
    ).squeeze(-1)
    stationarity[:, 2:] = stationarity[:, 2:] + (
        second.transpose(-1, -2) @ direction[:, :-2].unsqueeze(-1)
    ).squeeze(-1)
    dual = stationarity.abs().flatten(1).amax(1)
    parity_not_evaluated = torch.zeros_like(primal)
    return TrajectoryScanSolution(
        direction=direction,
        kkt_primal_residual=primal,
        kkt_dual_residual=dual,
        slack_max={
            "collision": collision_slack.flatten(1).amax(1),
            "region": region_slack.flatten(1).amax(1),
        },
        active_constraint_count={
            "box": (active["box_low"] | active["box_high"]).flatten(1).sum(1),
            "rate": (active["rate_low"] | active["rate_high"]).flatten(1).sum(1),
            "stance": problem.stance_active.flatten(1).sum(1) * 3,
            "touchdown_region": active["region"].flatten(1).sum(1),
            "touchdown_plane": problem.touchdown_plane_active.flatten(1).sum(1),
            "clearance": active["clearance"].flatten(1).sum(1),
        },
        dense_parity_error=parity_not_evaluated,
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
    return _solve_lq_problem(problem)


__all__ = [
    "TrajectoryScanSolution",
    "factor_tree_shapes",
    "fixed_five_level_tree",
    "pad_h30_factors",
    "solve_trajectory_qp_scan",
]
