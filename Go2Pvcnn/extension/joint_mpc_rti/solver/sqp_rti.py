"""Exactly one direct-state SQP real-time iteration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import (
    LossContext,
    total_trajectory_loss,
    trajectory_loss_breakdown,
)
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.solver.line_search import parallel_line_search
from extension.joint_mpc_rti.solver.linearization import linearize_trajectory
from extension.joint_mpc_rti.solver.trajectory_qp import ActiveConstraints, JOINT_LOWER, JOINT_UPPER
from extension.joint_mpc_rti.solver.trajectory_scan import solve_trajectory_qp_scan


@dataclass(frozen=True)
class SqpRtiUpdate:
    state: Tensor
    direction: Tensor
    alpha: Tensor
    loss_before: Tensor
    selected_loss: Tensor
    selected_index: Tensor
<<<<<<< HEAD
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
=======
    used_nominal: Tensor
    status: Tensor
    active: ActiveConstraints
    loss_breakdown: dict[str, Tensor]


def _repeat_context(context: LossContext, repeats: int) -> LossContext:
    if repeats == 1:
        return context
    schedule = context.schedule
    return LossContext(
        command_body=context.command_body.repeat_interleave(repeats, dim=0),
        touchdown_reference_w=context.touchdown_reference_w.repeat_interleave(repeats, dim=0),
        schedule=FixedTrotSchedule(
            phase=schedule.phase.repeat_interleave(repeats, dim=0),
            swing=schedule.swing.repeat_interleave(repeats, dim=0),
            stance=schedule.stance.repeat_interleave(repeats, dim=0),
            swing_tau=schedule.swing_tau.repeat_interleave(repeats, dim=0),
        ),
        terrain=context.terrain,
        stance_anchor_w=context.stance_anchor_w.repeat_interleave(repeats, dim=0),
        support_height=context.support_height.repeat_interleave(repeats, dim=0),
    )


def sqp_rti_update(
    nominal: Tensor,
    context: LossContext,
    cfg: JointMpcRtiCfg,
) -> SqpRtiUpdate:
    """Linearize once, solve the approved active QP, and search five candidates."""
    state = torch.as_tensor(nominal)
    qp = linearize_trajectory(state, context, cfg)
    scan = solve_trajectory_qp_scan(qp)

    def objective(candidate: Tensor) -> Tensor:
        repeats = int(candidate.shape[0]) // int(state.shape[0])
        return total_trajectory_loss(candidate, _repeat_context(context, repeats), cfg)

    search = parallel_line_search(
        state,
        scan.direction,
        objective,
        joint_lower=state.new_tensor(JOINT_LOWER),
        joint_upper=state.new_tensor(JOINT_UPPER),
        joint_velocity_limit=state.new_full((12,), float(cfg.solver.joint_velocity_limit)),
        dt=float(cfg.runtime.dt),
        tie_tolerance=float(cfg.solver.line_search_tie_tolerance),
>>>>>>> 156a6c0 (refactor: route joint mpc through pure kinematic rti)
    )
    finite = torch.isfinite(search.state).all(dim=(1, 2)) & torch.isfinite(search.selected_loss)
    solved = finite & search.selected_feasible
    status = torch.where(solved, torch.zeros_like(solved, dtype=torch.long), torch.ones_like(solved, dtype=torch.long))
    return SqpRtiUpdate(
<<<<<<< HEAD
        control=search.control,
        delta_control=delta_control,
=======
        state=search.state,
        direction=scan.direction,
>>>>>>> 156a6c0 (refactor: route joint mpc through pure kinematic rti)
        alpha=search.alpha,
        loss_before=objective(state),
        selected_loss=search.selected_loss,
        selected_index=search.selected_index,
<<<<<<< HEAD
        used_base=search.used_base,
        constraint_violation=search.constraint_violation,
        base_constraint_violation=search.base_constraint_violation,
=======
        used_nominal=search.used_nominal,
        status=status,
        active=scan.active,
        loss_breakdown=trajectory_loss_breakdown(search.state, context, cfg),
>>>>>>> 156a6c0 (refactor: route joint mpc through pure kinematic rti)
    )


__all__ = ["SqpRtiUpdate", "sqp_rti_update"]
