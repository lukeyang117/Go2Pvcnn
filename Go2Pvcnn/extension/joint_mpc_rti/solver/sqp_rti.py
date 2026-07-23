"""Exactly one direct-state SQP real-time iteration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import (
    LossContext,
    total_trajectory_loss,
    trajectory_loss_diagnostics,
)
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.model.nominal import NominalTrajectory
from extension.joint_mpc_rti.solver.lq_problem import build_lq_problem, lq_residuals
from extension.joint_mpc_rti.solver.line_search import hard_safe_line_search
from extension.joint_mpc_rti.solver.line_search import parallel_line_search
from extension.joint_mpc_rti.solver.linearization import linearize_trajectory
from extension.joint_mpc_rti.solver.trajectory_qp import ActiveConstraints, JOINT_LOWER, JOINT_UPPER
from extension.joint_mpc_rti.solver.trajectory_scan import solve_trajectory_qp_scan
from extension.joint_mpc_rti.tensor_constants import constant_like


@dataclass(frozen=True)
class SqpRtiUpdate:
    state: Tensor
    direction: Tensor
    alpha: Tensor
    loss_before: Tensor
    selected_loss: Tensor
    selected_index: Tensor
    used_nominal: Tensor
    status: Tensor
    candidate_loss: Tensor
    candidate_filter_valid: Tensor
    candidate_swing_safe_z: Tensor
    support_target: Tensor
    active: ActiveConstraints | None
    loss_breakdown: dict[str, Tensor]
    node_loss_breakdown: dict[str, Tensor]
    kkt_primal_residual: Tensor | None = None
    kkt_dual_residual: Tensor | None = None
    slack_max: dict[str, Tensor] | None = None
    active_constraint_count: dict[str, Tensor] | None = None
    alpha_reject_bits: Tensor | None = None
    publish: Tensor | None = None
    stop: Tensor | None = None


def published_stance_filter_mask(schedule: FixedTrotSchedule) -> Tensor:
    """Return only feet whose stance persists across the published edge."""
    return schedule.stance[:, 0] & schedule.stance[:, 1]


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
        perceptive_field=context.perceptive_field,
    )


def perceptive_sqp_rti_update(
    nominal: NominalTrajectory,
    context: LossContext,
    cfg: JointMpcRtiCfg,
) -> SqpRtiUpdate:
    """Build one final LQ/QP and run one five-alpha exact line search."""
    problem = build_lq_problem(nominal, context, cfg)
    scan = solve_trajectory_qp_scan(problem)
    batch = int(nominal.state.shape[0])

    def objective(candidate: Tensor) -> Tensor:
        repeats = int(candidate.shape[0]) // batch
        repeated_context = _repeat_context(context, repeats)
        repeated_nominal = NominalTrajectory(
            **{
                name: (
                    value.repeat_interleave(repeats, dim=0)
                    if isinstance(value, Tensor) and value.shape[0] == batch
                    else value
                )
                for name, value in vars(nominal).items()
            }
        )
        residuals = lq_residuals(candidate, repeated_nominal, repeated_context, cfg)
        total = candidate.new_zeros(candidate.shape[0])
        for value in residuals.values():
            total = total + 0.5 * value.square().sum(dim=1)
        return total

    search = hard_safe_line_search(
        nominal, scan.direction, objective, context, problem, cfg
    )
    return SqpRtiUpdate(
        state=search.state,
        direction=scan.direction,
        alpha=search.alpha,
        loss_before=objective(nominal.state),
        selected_loss=search.selected_loss,
        selected_index=search.selected_index,
        used_nominal=search.selected_index == len(search.alphas) - 1,
        status=search.stop.to(torch.long),
        candidate_loss=search.candidate_loss,
        candidate_filter_valid=~search.alpha_reject_bits,
        candidate_swing_safe_z=nominal.state.new_zeros(batch, 5, 4),
        support_target=problem.stance_target[:, 1].reshape(batch, -1),
        active=None,
        loss_breakdown=problem.cost_breakdown,
        node_loss_breakdown={},
        kkt_primal_residual=scan.kkt_primal_residual,
        kkt_dual_residual=scan.kkt_dual_residual,
        slack_max=scan.slack_max,
        active_constraint_count=scan.active_constraint_count,
        alpha_reject_bits=search.alpha_reject_bits,
        publish=search.publish,
        stop=search.stop,
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
    onset = ~context.schedule.stance[:, 0] & context.schedule.stance[:, 1]
    published_stance_anchor = torch.where(
        onset[..., None],
        context.touchdown_reference_w[:, 1],
        context.stance_anchor_w[:, 1],
    )

    def objective(candidate: Tensor) -> Tensor:
        repeats = int(candidate.shape[0]) // int(state.shape[0])
        return total_trajectory_loss(candidate, _repeat_context(context, repeats), cfg)

    search = parallel_line_search(
        state,
        scan.direction,
        objective,
        joint_lower=constant_like(state, "line_search_joint_lower", JOINT_LOWER),
        joint_upper=constant_like(state, "line_search_joint_upper", JOINT_UPPER),
        joint_velocity_limit=state.new_full((12,), float(cfg.solver.joint_velocity_limit)),
        published_stance_anchor_w=published_stance_anchor,
        published_stance_mask=published_stance_filter_mask(context.schedule),
        published_stance_ground_mask=context.schedule.stance[:, 1],
        published_stance_tolerance=float(cfg.solver.published_stance_tolerance),
        published_swing_mask=context.schedule.swing[:, 1],
        published_terrain_field=context.terrain,
        published_foot_contact_offset=float(cfg.gait.foot_contact_offset),
        published_swing_clearance_buffer=float(
            cfg.solver.published_swing_clearance_buffer
        ),
        published_h_wall=float(cfg.terrain.h_wall),
        dt=float(cfg.runtime.dt),
        tie_tolerance=float(cfg.solver.line_search_tie_tolerance),
    )
    finite = torch.isfinite(search.state).all(dim=(1, 2)) & torch.isfinite(search.selected_loss)
    solved = finite & search.selected_feasible
    status = torch.where(solved, torch.zeros_like(solved, dtype=torch.long), torch.ones_like(solved, dtype=torch.long))
    loss_breakdown, node_loss_breakdown = trajectory_loss_diagnostics(search.state, context, cfg)
    return SqpRtiUpdate(
        state=search.state,
        direction=scan.direction,
        alpha=search.alpha,
        loss_before=objective(state),
        selected_loss=search.selected_loss,
        selected_index=search.selected_index,
        used_nominal=search.used_nominal,
        status=status,
        candidate_loss=search.candidate_loss,
        candidate_filter_valid=search.filter_valid,
        candidate_swing_safe_z=getattr(
            search,
            "published_swing_safe_z",
            state.new_zeros(state.shape[0], 5, 4),
        ),
        support_target=qp.support_target,
        active=scan.active,
        loss_breakdown=loss_breakdown,
        node_loss_breakdown=node_loss_breakdown,
    )


__all__ = [
    "SqpRtiUpdate",
    "perceptive_sqp_rti_update",
    "published_stance_filter_mask",
    "sqp_rti_update",
]
