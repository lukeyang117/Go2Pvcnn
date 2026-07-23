"""Block-banded direct-state trajectory QP contracts and dense references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from extension.joint_mpc_rti.solver.fixed_general import fixed_general_solve
from extension.joint_mpc_rti.tensor_constants import constant_like

if TYPE_CHECKING:
    from extension.joint_mpc_rti.solver.lq_problem import LqProblem


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
    support_jacobian: Tensor
    support_target: Tensor

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
        fixed = qp.lower == qp.upper
        return cls(
            box_low=fixed,
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


@dataclass(frozen=True)
class QpSolution:
    direction: Tensor
    kkt_primal_residual: Tensor
    kkt_dual_residual: Tensor
    slack_max: dict[str, Tensor]
    active_constraint_count: dict[str, Tensor]


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
    active_row_count = sum(
        int(index.shape[0])
        for index in (box_low_index, box_high_index, velocity_low_index, velocity_high_index)
    )
    support_rows = int(qp.support_jacobian.shape[1])
    row_count = active_row_count + support_rows
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
    support = qp.support_jacobian[batch_index]
    matrix[cursor : cursor + support_rows, state_dim : 2 * state_dim] = support
    target[cursor : cursor + support_rows] = qp.support_target[batch_index]
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


def _support_feasible_seed(qp: TrajectoryQp, active: ActiveConstraints) -> Tensor:
    """Construct the affine seed in the free subspace of fixed box variables."""
    fixed = active.box_mask
    box_target = torch.where(active.box_low, qp.lower, qp.upper)
    direction = torch.where(fixed, box_target, torch.zeros_like(qp.gradient))
    free_x1 = (~fixed[:, 1]).to(qp.gradient.dtype)
    reduced_support = qp.support_jacobian * free_x1[:, None]
    support_rhs = qp.support_target - torch.einsum(
        "bri,bi->br", qp.support_jacobian, direction[:, 1]
    )
    gram = torch.bmm(reduced_support, reduced_support.transpose(1, 2))
    multiplier = fixed_general_solve(gram, support_rhs.unsqueeze(-1))
    correction = torch.bmm(reduced_support.transpose(1, 2), multiplier).squeeze(-1)
    direction[:, 1] += free_x1 * correction
    return direction


def refine_active_set(
    qp: TrajectoryQp,
    solve_fn=solve_dense_active_kkt,
    *,
    refinements: int = 2,
) -> ActiveSetSolution:
    if int(refinements) != 2:
        raise ValueError("the production active-set refinement count is fixed at two")
    def feasible_step(
        current: Tensor, target: Tensor, active_constraints: ActiveConstraints
    ) -> Tensor:
        step = target - current
        infinity = torch.full_like(step, float("inf"))
        box_ratio = torch.where(
            step > 0.0,
            (qp.upper - current) / step,
            torch.where(step < 0.0, (qp.lower - current) / step, infinity),
        )
        box_ratio = torch.where(active_constraints.box_mask, infinity, box_ratio)
        current_difference = current[:, 1:, 6:] - current[:, :-1, 6:]
        target_difference = target[:, 1:, 6:] - target[:, :-1, 6:]
        difference_step = target_difference - current_difference
        difference_infinity = torch.full_like(difference_step, float("inf"))
        velocity_ratio = torch.where(
            difference_step > 0.0,
            (qp.joint_difference_upper - current_difference) / difference_step,
            torch.where(
                difference_step < 0.0,
                (qp.joint_difference_lower - current_difference) / difference_step,
                difference_infinity,
            ),
        )
        velocity_ratio = torch.where(
            active_constraints.velocity_mask, difference_infinity, velocity_ratio
        )
        fraction = torch.minimum(
            box_ratio.amin(dim=(1, 2)), velocity_ratio.amin(dim=(1, 2))
        ).clamp(0.0, 1.0)
        return current + fraction[:, None, None] * step

    def merge(left: ActiveConstraints, right: ActiveConstraints) -> ActiveConstraints:
        box_low = left.box_low | right.box_low
        velocity_low = left.velocity_low | right.velocity_low
        return ActiveConstraints(
            box_low=box_low,
            box_high=(left.box_high | right.box_high) & ~box_low,
            velocity_low=velocity_low,
            velocity_high=(left.velocity_high | right.velocity_high) & ~velocity_low,
        )

    active = ActiveConstraints.empty(qp)
    direction = _support_feasible_seed(qp, active)
    free = solve_fn(qp, active)
    direction = feasible_step(direction, free, active)
    active = merge(active, select_active_constraints(qp, direction, tolerance=1.0e-5))

    first = solve_fn(qp, active)
    direction = feasible_step(direction, first, active)
    active = merge(active, select_active_constraints(qp, direction, tolerance=1.0e-5))

    second = solve_fn(qp, active)
    direction = feasible_step(direction, second, active)
    active = merge(active, select_active_constraints(qp, direction, tolerance=1.0e-5))
    return ActiveSetSolution(direction=direction, active=active)


def trajectory_bounds(nominal: Tensor, cfg) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state = torch.as_tensor(nominal)
    trust_values = (
        (cfg.solver.root_position_trust,) * 3
        + (cfg.solver.root_roll_pitch_trust,) * 2
        + (cfg.solver.root_yaw_trust,)
        + (cfg.solver.joint_trust,) * 12
    )
    trust = constant_like(state, f"trajectory_trust_{trust_values}", trust_values).view(1, 1, 18)
    upper = trust.expand_as(state).clone()
    node = torch.arange(state.shape[1], dtype=state.dtype, device=state.device).view(1, -1, 1)
    upper[..., :3] = node * float(cfg.solver.root_position_trust)
    lower = -upper.clone()
    lower[:, 1, :2] = 0.0
    upper[:, 1, :2] = 0.0
    joint_lower = constant_like(state, "trajectory_joint_lower", JOINT_LOWER).view(1, 1, 12) - state[..., 6:]
    joint_upper = constant_like(state, "trajectory_joint_upper", JOINT_UPPER).view(1, 1, 12) - state[..., 6:]
    lower[..., 6:] = torch.maximum(lower[..., 6:], joint_lower)
    upper[..., 6:] = torch.minimum(upper[..., 6:], joint_upper)
    lower[:, 0] = 0.0
    upper[:, 0] = 0.0

    nominal_difference = state[:, 1:, 6:] - state[:, :-1, 6:]
    maximum_step = float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt)
    difference_lower = -maximum_step - nominal_difference
    difference_upper = maximum_step - nominal_difference
    return lower, upper, difference_lower, difference_upper


def _dense_local_rows(
    local_rows: Tensor,
    target: Tensor,
    active: Tensor,
    *,
    nodes: int,
    state_dim: int,
) -> tuple[Tensor, Tensor]:
    """Materialize fixed-shape node-local rows for one eager reference batch."""
    row_shape = local_rows.shape[:-1]
    node_index = torch.arange(nodes, device=local_rows.device)
    node_index = node_index.view(nodes, *((1,) * (len(row_shape) - 1))).expand(row_shape)
    selected_row = local_rows[active]
    selected_target = target[active]
    selected_node = node_index[active]
    matrix = local_rows.new_zeros(selected_row.shape[0], nodes * state_dim)
    if int(selected_row.shape[0]):
        column = selected_node[:, None] * state_dim + torch.arange(
            state_dim, device=local_rows.device
        )[None]
        matrix.scatter_(1, column, selected_row)
    return matrix, selected_target


def _dense_constraints_for_batch(
    problem: "LqProblem", batch_index: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
    nodes, state_dim = problem.gradient.shape[1:]
    dtype, device = problem.gradient.dtype, problem.gradient.device
    identity = torch.eye(nodes * state_dim, dtype=dtype, device=device)
    fixed = problem.lower[batch_index] == problem.upper[batch_index]
    fixed_flat = fixed.flatten()
    equality = [identity[fixed_flat]]
    equality_target = [problem.lower[batch_index].flatten()[fixed_flat]]

    stance_active = problem.stance_active[batch_index][..., None].expand(-1, -1, 3).clone()
    stance_active[0] = False
    stance_matrix, stance_target = _dense_local_rows(
        problem.stance_rows[batch_index],
        problem.stance_target[batch_index],
        stance_active,
        nodes=nodes,
        state_dim=state_dim,
    )
    equality.append(stance_matrix)
    equality_target.append(stance_target)

    box_free = ~fixed_flat
    inequality = [identity[box_free], -identity[box_free]]
    inequality_target = [
        problem.lower[batch_index].flatten()[box_free],
        -problem.upper[batch_index].flatten()[box_free],
    ]
    inequality_type = [
        torch.zeros(int(box_free.sum()) * 2, dtype=torch.long, device=device)
    ]

    rate_coordinates = torch.tensor(
        (2, 3, 4) + tuple(range(6, 18)), dtype=torch.long, device=device
    )
    edge = torch.arange(nodes - 1, device=device)[:, None]
    coordinate = rate_coordinates[None].expand(nodes - 1, -1)
    rate_matrix = problem.gradient.new_zeros((nodes - 1) * 15, nodes * state_dim)
    row = torch.arange((nodes - 1) * 15, device=device)
    rate_matrix[row, (edge * state_dim + coordinate).flatten()] = -1.0
    rate_matrix[row, ((edge + 1) * state_dim + coordinate).flatten()] = 1.0
    inequality.extend((rate_matrix, -rate_matrix))
    inequality_target.extend(
        (
            problem.rate_lower[batch_index].flatten(),
            -problem.rate_upper[batch_index].flatten(),
        )
    )
    inequality_type.append(
        torch.ones(2 * rate_matrix.shape[0], dtype=torch.long, device=device)
    )

    region_rows = problem.touchdown_region_rows[batch_index].transpose(-1, -2)
    region_matrix, region_target = _dense_local_rows(
        region_rows,
        problem.touchdown_region_target[batch_index],
        problem.touchdown_region_active[batch_index],
        nodes=nodes,
        state_dim=state_dim,
    )
    inequality.append(region_matrix)
    inequality_target.append(
        region_target - float(problem.slack_caps["region"])
    )
    inequality_type.append(
        torch.full((region_matrix.shape[0],), 2, dtype=torch.long, device=device)
    )

    clearance_matrix, clearance_target = _dense_local_rows(
        problem.clearance_rows[batch_index],
        problem.clearance_target[batch_index],
        problem.clearance_active[batch_index],
        nodes=nodes,
        state_dim=state_dim,
    )
    inequality.append(clearance_matrix)
    inequality_target.append(
        clearance_target - float(problem.slack_caps["collision"])
    )
    inequality_type.append(
        torch.full((clearance_matrix.shape[0],), 3, dtype=torch.long, device=device)
    )

    exact_targets = {
        "region": region_target,
        "collision": clearance_target,
        "region_matrix": region_matrix,
        "collision_matrix": clearance_matrix,
    }
    return (
        torch.cat(equality, dim=0),
        torch.cat(equality_target, dim=0),
        torch.cat(inequality, dim=0),
        torch.cat(inequality_target, dim=0),
        torch.cat(inequality_type, dim=0),
        exact_targets,
    )


def _solve_equality_kkt(
    hessian: Tensor,
    gradient: Tensor,
    matrix: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    if int(matrix.shape[0]) == 0:
        return torch.linalg.solve(hessian, -gradient), gradient.new_zeros(0)
    zeros = hessian.new_zeros(matrix.shape[0], matrix.shape[0])
    kkt = torch.cat(
        (
            torch.cat((hessian, matrix.transpose(0, 1)), dim=1),
            torch.cat((matrix, zeros), dim=1),
        ),
        dim=0,
    )
    rhs = torch.cat((-gradient, target), dim=0)
    solution = torch.linalg.lstsq(kkt, rhs.unsqueeze(-1)).solution[:, 0]
    return solution[: gradient.shape[0]], solution[gradient.shape[0] :]


def solve_dense_qp(problem: "LqProblem", *, refinements: int = 2) -> QpSolution:
    """Solve the fixed-shape constrained LQ as an eager dense test reference."""
    if int(refinements) != 2:
        raise ValueError("dense reference uses exactly two active refinements")
    hessian, gradient = problem.to_dense()
    directions: list[Tensor] = []
    primal: list[Tensor] = []
    dual: list[Tensor] = []
    collision_slack: list[Tensor] = []
    region_slack: list[Tensor] = []
    count_values = {
        name: []
        for name in (
            "box",
            "rate",
            "stance",
            "touchdown_region",
            "touchdown_plane",
            "clearance",
        )
    }
    for batch_index in range(problem.gradient.shape[0]):
        eq, eq_target, ineq, ineq_target, ineq_type, exact = _dense_constraints_for_batch(
            problem, batch_index
        )
        direction, multiplier = _solve_equality_kkt(
            hessian[batch_index], gradient[batch_index], eq, eq_target
        )
        active = torch.zeros(ineq.shape[0], dtype=torch.bool, device=ineq.device)
        active_multiplier = direction.new_zeros(0)
        for _ in range(refinements):
            violated = torch.einsum("ri,i->r", ineq, direction) < ineq_target - 1.0e-8
            active = active | violated
            combined = torch.cat((eq, ineq[active]), dim=0)
            combined_target = torch.cat((eq_target, ineq_target[active]), dim=0)
            direction, combined_multiplier = _solve_equality_kkt(
                hessian[batch_index], gradient[batch_index], combined, combined_target
            )
            multiplier = combined_multiplier[: eq.shape[0]]
            active_multiplier = combined_multiplier[eq.shape[0] :]

        equality_error = torch.einsum("ri,i->r", eq, direction) - eq_target
        hard_mask = ineq_type < 2
        hard_violation = (
            ineq_target[hard_mask]
            - torch.einsum("ri,i->r", ineq[hard_mask], direction)
        ).clamp_min(0.0)
        region_violation = (
            exact["region"]
            - torch.einsum("ri,i->r", exact["region_matrix"], direction)
        ).clamp_min(0.0)
        collision_violation = (
            exact["collision"]
            - torch.einsum("ri,i->r", exact["collision_matrix"], direction)
        ).clamp_min(0.0)
        plane_rows, plane_target = _dense_local_rows(
            problem.touchdown_plane_rows[batch_index],
            problem.touchdown_plane_target[batch_index],
            problem.touchdown_plane_active[batch_index],
            nodes=problem.gradient.shape[1],
            state_dim=problem.gradient.shape[2],
        )
        plane_error = torch.einsum("ri,i->r", plane_rows, direction) - plane_target
        residual_terms = (
            equality_error.abs(),
            hard_violation,
            (region_violation - float(problem.slack_caps["region"])).clamp_min(0.0),
            (collision_violation - float(problem.slack_caps["collision"])).clamp_min(0.0),
            plane_error.abs(),
        )
        primal.append(
            torch.cat(tuple(value.flatten() for value in residual_terms)).amax()
        )
        stationarity = hessian[batch_index] @ direction + gradient[batch_index]
        stationarity = stationarity + eq.transpose(0, 1) @ multiplier
        if int(active.sum()):
            stationarity = stationarity + ineq[active].transpose(0, 1) @ active_multiplier
        dual.append(stationarity.abs().amax())
        directions.append(direction.reshape(problem.gradient.shape[1:]))
        region_slack.append(
            region_violation.amax() if region_violation.numel() else direction.new_zeros(())
        )
        collision_slack.append(
            collision_violation.amax()
            if collision_violation.numel()
            else direction.new_zeros(())
        )
        count_values["box"].append((active & (ineq_type == 0)).sum())
        count_values["rate"].append((active & (ineq_type == 1)).sum())
        count_values["touchdown_region"].append((active & (ineq_type == 2)).sum())
        count_values["clearance"].append((active & (ineq_type == 3)).sum())
        count_values["stance"].append(problem.stance_active[batch_index].sum() * 3)
        count_values["touchdown_plane"].append(
            problem.touchdown_plane_active[batch_index].sum()
        )
    return QpSolution(
        direction=torch.stack(directions),
        kkt_primal_residual=torch.stack(primal),
        kkt_dual_residual=torch.stack(dual),
        slack_max={
            "collision": torch.stack(collision_slack),
            "region": torch.stack(region_slack),
        },
        active_constraint_count={
            name: torch.stack(values).to(torch.long)
            for name, values in count_values.items()
        },
    )


__all__ = [
    "ActiveConstraints",
    "ActiveSetSolution",
    "JOINT_LOWER",
    "JOINT_UPPER",
    "TrajectoryQp",
    "QpSolution",
    "refine_active_set",
    "select_active_constraints",
    "solve_dense_active_kkt",
    "solve_dense_qp",
    "trajectory_bounds",
]
