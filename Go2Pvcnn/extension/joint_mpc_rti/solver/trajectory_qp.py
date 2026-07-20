"""Block-banded direct-state trajectory QP contracts and dense references."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


JOINT_LOWER = (-1.0472, -0.6632, -2.721) * 4
JOINT_UPPER = (1.0472, 2.966, -0.837) * 4


@dataclass(frozen=True)
class TrajectoryQp:
    diagonal: Tensor
    first_offdiag: Tensor
    second_offdiag: Tensor
    gradient: Tensor
    lower: Tensor
    upper: Tensor
    joint_difference_lower: Tensor
    joint_difference_upper: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.gradient.shape[0])

    @property
    def nodes(self) -> int:
        return int(self.gradient.shape[1])

    def to_dense(self) -> tuple[Tensor, Tensor]:
        """Materialize the test-only dense Hessian and flattened gradient."""
        batch, nodes, state_dim = self.gradient.shape
        blocks = self.gradient.new_zeros(batch, nodes, nodes, state_dim, state_dim)
        node = torch.arange(nodes, device=self.gradient.device)
        edge = torch.arange(nodes - 1, device=self.gradient.device)
        second = torch.arange(nodes - 2, device=self.gradient.device)
        blocks[:, node, node] = self.diagonal
        blocks[:, edge, edge + 1] = self.first_offdiag
        blocks[:, edge + 1, edge] = self.first_offdiag.transpose(-1, -2)
        blocks[:, second, second + 2] = self.second_offdiag
        blocks[:, second + 2, second] = self.second_offdiag.transpose(-1, -2)
        dense = blocks.permute(0, 1, 3, 2, 4).reshape(batch, nodes * state_dim, nodes * state_dim)
        return dense, self.gradient.flatten(1)


@dataclass(frozen=True)
class ActiveConstraints:
    box_low: Tensor
    box_high: Tensor
    velocity_low: Tensor
    velocity_high: Tensor

    @property
    def box_mask(self) -> Tensor:
        return self.box_low | self.box_high

    @property
    def velocity_mask(self) -> Tensor:
        return self.velocity_low | self.velocity_high

    @property
    def max_rows_per_interval(self) -> int:
        return 30

    @classmethod
    def empty(cls, qp: TrajectoryQp) -> "ActiveConstraints":
        return cls(
            box_low=torch.zeros_like(qp.lower, dtype=torch.bool),
            box_high=torch.zeros_like(qp.upper, dtype=torch.bool),
            velocity_low=torch.zeros_like(qp.joint_difference_lower, dtype=torch.bool),
            velocity_high=torch.zeros_like(qp.joint_difference_upper, dtype=torch.bool),
        )

    @staticmethod
    def validate_compile_budget(*, constraint_rows: int) -> dict[str, int]:
        from extension.joint_mpc_rti.solver.primal_dual_ilqr import joint_kkt_compile_budget

        return joint_kkt_compile_budget(constraint_rows=constraint_rows)


@dataclass(frozen=True)
class ActiveSetSolution:
    direction: Tensor
    active: ActiveConstraints


def select_active_constraints(
    qp: TrajectoryQp,
    direction: Tensor,
    *,
    tolerance: float = 1.0e-7,
) -> ActiveConstraints:
    value = torch.as_tensor(direction, dtype=qp.gradient.dtype, device=qp.gradient.device)
    box_low = value <= qp.lower + float(tolerance)
    box_high = (value >= qp.upper - float(tolerance)) & ~box_low
    difference = value[:, 1:, 6:] - value[:, :-1, 6:]
    velocity_low = difference <= qp.joint_difference_lower + float(tolerance)
    velocity_high = (difference >= qp.joint_difference_upper - float(tolerance)) & ~velocity_low
    return ActiveConstraints(box_low, box_high, velocity_low, velocity_high)


def _active_matrix_and_target(
    qp: TrajectoryQp,
    active: ActiveConstraints,
    batch_index: int,
) -> tuple[Tensor, Tensor]:
    nodes = qp.nodes
    state_dim = int(qp.gradient.shape[-1])
    box_low_index = torch.nonzero(active.box_low[batch_index], as_tuple=False)
    box_high_index = torch.nonzero(active.box_high[batch_index], as_tuple=False)
    velocity_low_index = torch.nonzero(active.velocity_low[batch_index], as_tuple=False)
    velocity_high_index = torch.nonzero(active.velocity_high[batch_index], as_tuple=False)
    box_mask = active.box_mask[batch_index]

    def remove_box_redundant_edges(index: Tensor) -> Tensor:
        if int(index.shape[0]) == 0:
            return index
        edge, joint = index.unbind(dim=1)
        both_box_fixed = box_mask[edge, 6 + joint] & box_mask[edge + 1, 6 + joint]
        return index[~both_box_fixed]

    velocity_low_index = remove_box_redundant_edges(velocity_low_index)
    velocity_high_index = remove_box_redundant_edges(velocity_high_index)
    row_count = sum(
        int(index.shape[0])
        for index in (box_low_index, box_high_index, velocity_low_index, velocity_high_index)
    )
    matrix = qp.gradient.new_zeros(row_count, nodes * state_dim)
    target = qp.gradient.new_zeros(row_count)
    cursor = 0
    for index, bound in (
        (box_low_index, qp.lower[batch_index]),
        (box_high_index, qp.upper[batch_index]),
    ):
        count = int(index.shape[0])
        if count:
            columns = index[:, 0] * state_dim + index[:, 1]
            rows = torch.arange(cursor, cursor + count, device=matrix.device)
            matrix[rows, columns] = 1.0
            target[cursor : cursor + count] = bound[index[:, 0], index[:, 1]]
            cursor += count
    for index, bound in (
        (velocity_low_index, qp.joint_difference_lower[batch_index]),
        (velocity_high_index, qp.joint_difference_upper[batch_index]),
    ):
        count = int(index.shape[0])
        if count:
            edge = index[:, 0]
            joint = index[:, 1]
            row = torch.arange(cursor, cursor + count, device=matrix.device)
            matrix[row, edge * state_dim + 6 + joint] = -1.0
            matrix[row, (edge + 1) * state_dim + 6 + joint] = 1.0
            target[cursor : cursor + count] = bound[edge, joint]
            cursor += count
    return matrix, target


def solve_dense_active_kkt(qp: TrajectoryQp, active: ActiveConstraints) -> Tensor:
    """Eager dense active-KKT reference used for parity and refinement tests."""
    ActiveConstraints.validate_compile_budget(constraint_rows=active.max_rows_per_interval)
    hessian, gradient = qp.to_dense()
    solutions: list[Tensor] = []
    for batch_index in range(qp.batch_size):
        matrix, target = _active_matrix_and_target(qp, active, batch_index)
        if int(matrix.shape[0]) == 0:
            solution = torch.linalg.solve(hessian[batch_index], -gradient[batch_index])
        else:
            zeros = hessian.new_zeros(matrix.shape[0], matrix.shape[0])
            kkt = torch.cat(
                (
                    torch.cat((hessian[batch_index], matrix.transpose(0, 1)), dim=1),
                    torch.cat((matrix, zeros), dim=1),
                ),
                dim=0,
            )
            rhs = torch.cat((-gradient[batch_index], target), dim=0)
            solution = torch.linalg.solve(kkt, rhs)[: gradient.shape[1]]
        solutions.append(solution.reshape(qp.nodes, -1))
    return torch.stack(solutions, dim=0)


def refine_active_set(
    qp: TrajectoryQp,
    solve_fn=solve_dense_active_kkt,
    *,
    refinements: int = 2,
) -> ActiveSetSolution:
    if int(refinements) != 2:
        raise ValueError("the production active-set refinement count is fixed at two")
    free = solve_fn(qp, ActiveConstraints.empty(qp))
    free_active = select_active_constraints(qp, free)
    first_active = ActiveConstraints(
        box_low=free_active.box_low,
        box_high=free_active.box_high,
        velocity_low=torch.zeros_like(free_active.velocity_low),
        velocity_high=torch.zeros_like(free_active.velocity_high),
    )
    first = solve_fn(qp, first_active)
    remaining = select_active_constraints(qp, first)
    merged_box_low = first_active.box_low | remaining.box_low
    merged_velocity_low = remaining.velocity_low
    second_active = ActiveConstraints(
        box_low=merged_box_low,
        box_high=(first_active.box_high | remaining.box_high) & ~merged_box_low,
        velocity_low=merged_velocity_low,
        velocity_high=remaining.velocity_high & ~merged_velocity_low,
    )
    second = solve_fn(qp, second_active)
    return ActiveSetSolution(direction=second, active=second_active)


def trajectory_bounds(nominal: Tensor, cfg) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state = torch.as_tensor(nominal)
    trust = state.new_tensor(
        (cfg.solver.root_position_trust,) * 3
        + (cfg.solver.root_orientation_trust,) * 3
        + (cfg.solver.joint_trust,) * 12
    ).view(1, 1, 18)
    lower = -trust.expand_as(state).clone()
    upper = trust.expand_as(state).clone()
    joint_lower = state.new_tensor(JOINT_LOWER).view(1, 1, 12) - state[..., 6:]
    joint_upper = state.new_tensor(JOINT_UPPER).view(1, 1, 12) - state[..., 6:]
    lower[..., 6:] = torch.maximum(lower[..., 6:], joint_lower)
    upper[..., 6:] = torch.minimum(upper[..., 6:], joint_upper)
    lower[:, 0] = 0.0
    upper[:, 0] = 0.0

    nominal_difference = state[:, 1:, 6:] - state[:, :-1, 6:]
    maximum_step = float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt)
    difference_lower = -maximum_step - nominal_difference
    difference_upper = maximum_step - nominal_difference
    return lower, upper, difference_lower, difference_upper


__all__ = [
    "ActiveConstraints",
    "ActiveSetSolution",
    "JOINT_LOWER",
    "JOINT_UPPER",
    "TrajectoryQp",
    "refine_active_set",
    "select_active_constraints",
    "solve_dense_active_kkt",
    "trajectory_bounds",
]
