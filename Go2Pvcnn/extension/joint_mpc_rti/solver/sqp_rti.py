"""Exactly one direct-state SQP real-time iteration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.solver.context import LossContext
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.model.nominal import NominalTrajectory
from extension.joint_mpc_rti.solver.lq_problem import build_lq_problem, lq_residuals
from extension.joint_mpc_rti.solver.line_search import hard_safe_line_search
from extension.joint_mpc_rti.solver.trajectory_qp import ActiveConstraints
from extension.joint_mpc_rti.solver.trajectory_scan import solve_trajectory_qp_scan


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
    alpha_min_clearance: Tensor | None = None
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
        perceptive_field=(
            _repeat_dataclass(context.perceptive_field, repeats)
            if context.perceptive_field is not None
            else None
        ),
    )


def _repeat_dataclass(value, repeats: int):
    if value is None or repeats == 1:
        return value
    return type(value)(
        **{
            name: (
                tensor.repeat_interleave(repeats, dim=0)
                if isinstance(tensor, Tensor) and tensor.ndim > 0
                else (
                    _repeat_dataclass(tensor, repeats)
                    if hasattr(tensor, "__dataclass_fields__")
                    else tensor
                )
            )
            for name, tensor in vars(value).items()
        }
    )


def perceptive_sqp_rti_update(
    nominal: NominalTrajectory,
    context: LossContext,
    cfg: JointMpcRtiCfg,
    *,
    stage_profiler=None,
) -> SqpRtiUpdate:
    """Build one final LQ/QP and run one five-alpha exact line search."""
    problem = build_lq_problem(nominal, context, cfg)
    if stage_profiler is not None:
        stage_profiler.record("linearization")
    scan = solve_trajectory_qp_scan(problem)
    if stage_profiler is not None:
        stage_profiler.record("scan_qp")
    batch = int(nominal.state.shape[0])

    def objective(candidate: Tensor) -> Tensor:
        repeats = int(candidate.shape[0]) // batch
        repeated_context = _repeat_context(context, repeats)
        repeated_nominal = _repeat_dataclass(nominal, repeats)
        residuals = lq_residuals(candidate, repeated_nominal, repeated_context, cfg)
        total = candidate.new_zeros(candidate.shape[0])
        for value in residuals.values():
            total = total + 0.5 * value.square().sum(dim=1)
        return total

    search = hard_safe_line_search(
        nominal, scan.direction, objective, context, problem, cfg
    )
    if stage_profiler is not None:
        stage_profiler.record("line_search_safety")
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
        alpha_min_clearance=search.minimum_clearance_by_part,
        publish=search.publish,
        stop=search.stop,
    )


__all__ = [
    "SqpRtiUpdate",
    "perceptive_sqp_rti_update",
    "published_stance_filter_mask",
]
