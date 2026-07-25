"""Pure-kinematic H30 joint MPC RTI orchestration."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.solver.context import LossContext
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.model.nominal import (
    NominalTrajectory,
    build_nominal,
    build_rebased_seed,
)
from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns
from extension.joint_mpc_rti.solver.sqp_rti import perceptive_sqp_rti_update
from extension.joint_mpc_rti.solver.trajectory_qp import JOINT_LOWER, JOINT_UPPER
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
from extension.joint_mpc_rti.terrain.query import query_ground_support_height_world
from extension.joint_mpc_rti.types import (
    JointMpcPendingReference,
    JointMpcFieldFrame,
    JointMpcPerceptiveField,
    JointMpcRtiSolverState,
    JointMpcRtiState,
    JointMpcRtiStepDiagnostics,
    JointMpcRtiStepResult,
    JointMpcRtiTrajectory,
    JointMpcTerrainField,
)


def build_loss_context(
    nominal: NominalTrajectory,
    command_body: Tensor,
    terrain_field: JointMpcTerrainField,
    gait_phase: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    perceptive_field: JointMpcPerceptiveField,
) -> LossContext:
    schedule = fixed_trot_schedule(gait_phase, horizon_steps=int(cfg.runtime.horizon_steps))
    support_height = query_ground_support_height_world(
        perceptive_field, nominal.state[..., :2]
    )
    stance_anchor_w = _stance_anchors_from_state(
        nominal.state,
        nominal.touchdown_reference_w,
        nominal.current_stance_anchor_w,
        schedule,
    ).clone()
    return LossContext(
        command_body=command_body,
        touchdown_reference_w=nominal.touchdown_reference_w,
        schedule=schedule,
        terrain=terrain_field,
        stance_anchor_w=stance_anchor_w,
        support_height=support_height,
        perceptive_field=perceptive_field,
    )


def _foot_positions(state: Tensor) -> Tensor:
    batch, nodes = state.shape[:2]
    geometry = go2_fk(
        state[..., :3].reshape(batch * nodes, 3),
        state[..., 3:6].reshape(batch * nodes, 3),
        state[..., 6:].reshape(batch * nodes, 12),
    )
    return geometry.foot_pos_w.reshape(batch, nodes, 4, 3)


def _selected_components(values: dict[str, Tensor], selected_index: Tensor) -> Tensor:
    stacked = torch.stack(tuple(values.values()), dim=-1)
    index = selected_index[..., None, None].expand(-1, -1, 1, stacked.shape[-1])
    return torch.gather(stacked, 2, index).squeeze(2)


def _stance_anchors_from_state(
    state: Tensor,
    touchdown_reference_w: Tensor,
    current_stance_anchor_w: Tensor,
    schedule,
) -> Tensor:
    anchor = torch.as_tensor(
        current_stance_anchor_w, dtype=state.dtype, device=state.device
    )
    touchdown = torch.as_tensor(
        touchdown_reference_w, dtype=state.dtype, device=state.device
    )
    future_stance = schedule.stance_node & (
        schedule.swing.to(torch.int64).cumsum(dim=1) > 0
    )
    return torch.where(future_stance[..., None], touchdown, anchor[:, None])


def step(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    terrain_field: JointMpcTerrainField,
    solver_state: JointMpcRtiSolverState | None,
    cfg: JointMpcRtiCfg,
    *,
    stage_profiler=None,
) -> JointMpcRtiStepResult:
    command = torch.as_tensor(command_body, dtype=measured_state.root_pos_w.dtype, device=measured_state.device)
    if command.shape != (measured_state.batch_size, 3):
        raise ValueError("command_body must have shape [B,3]")
    if int(terrain_field.height_w.shape[0]) != measured_state.batch_size:
        raise ValueError("terrain_field batch must match measured state")
    if solver_state is None:
        measured_foot_w = go2_fk(
            measured_state.root_pos_w,
            measured_state.root_rpy_w,
            measured_state.joint_pos,
        ).foot_pos_w
        previous = JointMpcRtiSolverState(
            trajectory=measured_state.as_vector()[:, None].expand(-1, 31, -1).clone(),
            gait_phase=torch.zeros(measured_state.batch_size, dtype=torch.long, device=measured_state.device),
            initialized=torch.zeros(measured_state.batch_size, dtype=torch.bool, device=measured_state.device),
            stance_anchor_w=measured_foot_w,
        )
    else:
        previous = solver_state
    schedule = fixed_trot_schedule(
        previous.gait_phase, horizon_steps=int(cfg.runtime.horizon_steps)
    )
    frame = JointMpcFieldFrame(
        origin_w=terrain_field.origin_w,
        yaw_w=terrain_field.yaw_w,
        timestamp=terrain_field.timestamp,
        refresh_id=terrain_field.version,
    )
    perceptive_field = build_perceptive_field(
        terrain_field.height_w,
        terrain_field.semantic_id,
        terrain_field.valid_mask,
        frame,
        cfg,
    )
    if stage_profiler is not None:
        stage_profiler.record("field")
    warm_nodes = build_rebased_seed(
        measured_state, command, previous.gait_phase, previous, cfg
    )
    perceptive_plan = select_touchdowns(
        measured_state,
        command,
        schedule,
        warm_nodes,
        perceptive_field,
        cfg,
        previous_target_w=previous.touchdown_target_w,
        previous_selected_index=previous.touchdown_selected_index,
        previous_crossing=previous.touchdown_crossing,
        previous_remaining_steps=previous.touchdown_remaining_steps,
        previous_swing_offset_w=previous.touchdown_swing_offset_w,
        previous_lift_w=previous.stance_anchor_w,
        terrain_field=terrain_field,
        stage_profiler=stage_profiler,
    )
    if stage_profiler is not None:
        stage_profiler.record("selector")
        stage_profiler.record("region")
    nominal = build_nominal(
        measured_state,
        command,
        perceptive_field,
        previous.gait_phase,
        perceptive_plan=perceptive_plan,
        previous=previous,
        cfg=cfg,
    )
    if stage_profiler is not None:
        stage_profiler.record("nominal_ik")
    context = build_loss_context(
        nominal,
        command,
        terrain_field,
        previous.gait_phase,
        cfg,
        perceptive_field=perceptive_field,
    )
    update = perceptive_sqp_rti_update(
        nominal, context, cfg, stage_profiler=stage_profiler
    )
    state = update.state
    foot_pos_w = _foot_positions(state)
    nominal_foot_w = _foot_positions(nominal.state)
    warm_foot_w = _foot_positions(warm_nodes)
    event_index = perceptive_plan.event_step[..., None, None].expand(-1, -1, 1, 3)
    warm_event_foot = torch.gather(warm_foot_w.transpose(1, 2), 2, event_index).squeeze(2)
    nominal_event_foot = torch.gather(
        nominal_foot_w.transpose(1, 2), 2, event_index
    ).squeeze(2)
    target_change = (
        perceptive_plan.target_w - warm_event_foot
    ).square().sum(dim=-1).sqrt()
    target_reason = state.new_zeros(state.shape[0], 4, 4, dtype=torch.bool)
    unsafe_retarget = nominal.candidate_retry_rank > 0
    target_reason[..., 3] = unsafe_retarget
    target_reason[..., 0] = (target_change > 1.0e-7) & ~unsafe_retarget
    stance_error = (
        nominal_foot_w - context.stance_anchor_w
    ).square().sum(dim=-1).sqrt()
    stance_error = torch.where(
        context.schedule.stance, stance_error, torch.zeros_like(stance_error)
    ).amax(dim=1)
    joint_lower = constant_like(state, "diagnostic_joint_lower", JOINT_LOWER).view(1, 1, 4, 3)
    joint_upper = constant_like(state, "diagnostic_joint_upper", JOINT_UPPER).view(1, 1, 4, 3)
    nominal_joint = nominal.state[..., 6:].reshape(state.shape[0], 31, 4, 3)
    joint_margin = torch.minimum(
        nominal_joint - joint_lower, joint_upper - nominal_joint
    ).amin(dim=(1, 3))
    derived_velocity = (state[:, 1:] - state[:, :-1]) / float(cfg.runtime.dt)
    finite = torch.isfinite(state).all(dim=(1, 2)) & torch.isfinite(foot_pos_w).all(dim=(1, 2, 3))
    publish = (
        update.publish
        if update.publish is not None
        else finite & (update.status == 0)
    )
    valid = nominal.nominal_safe & finite & publish & (update.status == 0)
    status = torch.where(valid, update.status, torch.ones_like(update.status))
    trajectory = JointMpcRtiTrajectory(
        state=state,
        derived_velocity=derived_velocity,
        foot_pos_w=foot_pos_w,
        contact_state=nominal.contact_state,
        valid=valid,
        fallback=update.used_nominal,
        status=status,
        line_search_alpha=update.alpha,
        loss_breakdown=update.loss_breakdown,
        cold_start=nominal.used_cold_start,
        warm_start=nominal.used_warm_start,
        warm_cache_invariant_fault=nominal.warm_cache_invariant_fault,
        publish=valid,
        stop=~valid,
    )
    pending = JointMpcPendingReference(
        root_pos_w=state[:, 1, :3],
        root_rpy_w=state[:, 1, 3:6],
        joint_angles=state[:, 1, 6:],
        foot_pos_w=foot_pos_w[:, 1],
        contact_state=nominal.contact_state[:, 1],
        valid=valid,
        target_step=1,
    )
    accepted_state = torch.where(valid[:, None, None], state, previous.trajectory)
    effective_plan = nominal.perceptive_plan
    assert effective_plan is not None
    effective_crossing = torch.gather(
        effective_plan.small_cross_required,
        2,
        effective_plan.selected_index[..., None],
    ).squeeze(-1)
    previous_target = (
        effective_plan.target_w
        if previous.touchdown_target_w is None
        else previous.touchdown_target_w
    )
    previous_index = (
        effective_plan.selected_index
        if previous.touchdown_selected_index is None
        else previous.touchdown_selected_index
    )
    previous_crossing = (
        torch.zeros_like(effective_crossing)
        if previous.touchdown_crossing is None
        else previous.touchdown_crossing
    )
    previous_remaining = (
        torch.zeros_like(effective_plan.event_step)
        if previous.touchdown_remaining_steps is None
        else previous.touchdown_remaining_steps
    )
    previous_swing_offset = (
        effective_plan.selected_swing_offset_w
        if previous.touchdown_swing_offset_w is None
        else previous.touchdown_swing_offset_w
    )
    active_previous_crossing = previous_crossing & (previous_remaining > 0)
    selected_remaining = (effective_plan.event_step - 1).clamp_min(0)
    crossing_remaining = torch.where(
        active_previous_crossing,
        (previous_remaining - 1).clamp_min(0),
        selected_remaining,
    )
    next_crossing = torch.where(
        valid[:, None], effective_crossing, active_previous_crossing
    )
    next_remaining = torch.where(
        valid[:, None] & effective_crossing,
        crossing_remaining,
        torch.where(
            ~valid[:, None] & active_previous_crossing,
            (previous_remaining - 1).clamp_min(0),
            torch.zeros_like(previous_remaining),
        ),
    )
    next_solver_state = JointMpcRtiSolverState(
        trajectory=accepted_state,
        gait_phase=previous.gait_phase + 1,
        initialized=previous.initialized | nominal.used_cold_start,
        stance_anchor_w=torch.where(
            (
                ~nominal.contact_state[:, 0]
                & nominal.contact_state[:, 1]
                & valid[:, None]
            )[..., None],
            foot_pos_w[:, 1],
            nominal.current_stance_anchor_w,
        ),
        preview_tail_state=(
            nominal.preview_tail_state
            if previous.preview_tail_state is None
            else torch.where(
                valid[:, None, None],
                nominal.preview_tail_state,
                previous.preview_tail_state,
            )
        ),
        touchdown_target_w=torch.where(
            valid[:, None, None], effective_plan.target_w, previous_target
        ),
        touchdown_selected_index=torch.where(
            valid[:, None], effective_plan.selected_index, previous_index
        ),
        touchdown_crossing=next_crossing,
        touchdown_remaining_steps=next_remaining,
        touchdown_swing_offset_w=torch.where(
            valid[:, None, None],
            effective_plan.selected_swing_offset_w,
            previous_swing_offset,
        ),
    )
    result = JointMpcRtiStepResult(
        full_trajectory=trajectory,
        pending_reference=pending,
        solver_state=next_solver_state,
        diagnostics=JointMpcRtiStepDiagnostics(
            nominal_state=nominal.state,
            qp_direction=update.direction,
            stance_anchor_w=context.stance_anchor_w[:, 0],
            touchdown_reference_w=context.touchdown_reference_w[:, :2],
            candidate_loss=getattr(
                update,
                "candidate_loss",
                state.new_zeros(state.shape[0], 5),
            ),
            candidate_filter_valid=getattr(
                update,
                "candidate_filter_valid",
                torch.ones(
                    state.shape[0], 5, 4, dtype=torch.bool, device=state.device
                ),
            ),
            candidate_swing_safe_z=getattr(
                update,
                "candidate_swing_safe_z",
                state.new_zeros(state.shape[0], 5, 4),
            ),
            support_target=getattr(
                update,
                "support_target",
                state.new_zeros(state.shape[0], 6),
            ),
            node_loss_breakdown=update.node_loss_breakdown,
            selector_candidate_valid_count=perceptive_plan.safe_mask.sum(dim=-1),
            selector_candidate_reject_reason_count=(
                ~torch.stack(tuple(perceptive_plan.valid_components.values()), dim=-1)
            ).sum(dim=-2),
            selector_candidate_behind_count=perceptive_plan.small_after_mask.sum(dim=-1),
            selector_candidate_sweep_safe_count=(
                perceptive_plan.valid_components.get(
                    "sweep_safe", torch.zeros_like(perceptive_plan.safe_mask)
                ).sum(dim=-1)
            ),
            selector_selected_index=perceptive_plan.selected_index,
            selector_selected_rank=torch.zeros_like(perceptive_plan.selected_index),
            selector_score_components=_selected_components(
                perceptive_plan.score_components, perceptive_plan.selected_index
            ),
            region_valid=perceptive_plan.region.valid,
            region_area=perceptive_plan.region.area,
            region_min_half_extent=perceptive_plan.region.half_extent.amin(dim=-1),
            region_plane_residual=perceptive_plan.region.plane_residual,
            region_distance_to_forbidden=perceptive_plan.region.distance_to_forbidden,
            warm_shift_rebase_error=(nominal.rebased_state[:, 0] - state[:, 0]).abs().amax(dim=-1),
            touchdown_target_change=target_change,
            touchdown_target_change_reason_bits=target_reason,
            retarget_trajectory_change=(nominal.state[:, 1:] - nominal.rebased_state[:, 1:]).square().mean(dim=(1, 2)).sqrt(),
            nominal_safe=nominal.nominal_safe,
            nominal_min_clearance=nominal.minimum_clearance_by_part,
            nominal_stance_anchor_error=stance_error,
            nominal_touchdown_error=(
                nominal_event_foot - perceptive_plan.target_w
            ).square().sum(dim=-1).sqrt(),
            nominal_joint_margin=joint_margin,
            unsafe_candidate_retry_count=nominal.candidate_retry_rank,
            kkt_primal_residual=update.kkt_primal_residual,
            kkt_dual_residual=update.kkt_dual_residual,
            delta_z_norm=update.direction[:, 1:].square().mean(dim=(1, 2)).sqrt(),
            slack_max=torch.stack(tuple(update.slack_max.values()), dim=-1)
            if update.slack_max
            else state.new_zeros(state.shape[0], 0),
            active_constraint_count=torch.stack(tuple(update.active_constraint_count.values()), dim=-1)
            if update.active_constraint_count
            else state.new_zeros(state.shape[0], 0),
            alpha_feasible=~update.alpha_reject_bits.any(dim=-1)
            if update.alpha_reject_bits is not None
            else state.new_zeros(state.shape[0], 5, dtype=torch.bool),
            alpha_cost=update.candidate_loss,
            alpha_reject_bits=update.alpha_reject_bits,
            alpha_min_clearance=update.alpha_min_clearance,
            selected_alpha=update.alpha,
            touchdown_candidate_w=perceptive_plan.candidate_w,
            touchdown_candidate_safe=perceptive_plan.safe_mask,
            touchdown_candidate_reject_bits=~torch.stack(
                tuple(perceptive_plan.valid_components.values()), dim=-1
            ),
            selected_target_w=perceptive_plan.target_w,
            previous_target_w=warm_event_foot,
            region_A=perceptive_plan.region.A,
            region_b=perceptive_plan.region.b,
            region_plane=perceptive_plan.region.plane,
            region_corners_w=perceptive_plan.region.corners_w,
        ),
    )
    if stage_profiler is not None:
        stage_profiler.record("cache_diagnostics")
    return result


__all__ = ["build_loss_context", "step"]
