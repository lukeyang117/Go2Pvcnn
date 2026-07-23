"""Pure-kinematic H30 joint MPC RTI orchestration."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import LossContext
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.model.nominal import (
    NominalTrajectory,
    build_nominal,
    build_rebased_seed,
)
from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns
from extension.joint_mpc_rti.solver.sqp_rti import perceptive_sqp_rti_update
from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
from extension.joint_mpc_rti.terrain.query import query_world
from extension.joint_mpc_rti.types import (
    JointMpcPendingReference,
    JointMpcFieldFrame,
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
    perceptive_field=None,
) -> LossContext:
    schedule = fixed_trot_schedule(gait_phase, horizon_steps=int(cfg.runtime.horizon_steps))
    support_height = query_world(terrain_field, nominal.state[..., :2]).height_w
    stance_anchor_w = _stance_anchors_from_state(
        nominal.state,
        nominal.touchdown_reference_w,
        nominal.current_stance_anchor_w,
        schedule,
    ).clone()
    anchor_shape = stance_anchor_w.shape
    anchor_height = query_world(
        terrain_field, stance_anchor_w.reshape(anchor_shape[0], -1, 3)
    ).height_w.reshape(anchor_shape[:-1])
    stance_anchor_w[..., 2] = anchor_height + float(cfg.gait.foot_contact_offset)
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


def _stance_anchors_from_state(
    state: Tensor,
    touchdown_reference_w: Tensor,
    current_stance_anchor_w: Tensor,
    schedule,
) -> Tensor:
    del touchdown_reference_w, schedule
    anchor = torch.as_tensor(
        current_stance_anchor_w, dtype=state.dtype, device=state.device
    )
    return anchor[:, None].expand(-1, state.shape[1], -1, -1)


def step(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    terrain_field: JointMpcTerrainField,
    solver_state: JointMpcRtiSolverState | None,
    cfg: JointMpcRtiCfg,
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
    )
    nominal = build_nominal(
        measured_state,
        command,
        perceptive_field,
        previous.gait_phase,
        perceptive_plan=perceptive_plan,
        previous=previous,
        cfg=cfg,
    )
    context = build_loss_context(
        nominal,
        command,
        terrain_field,
        previous.gait_phase,
        cfg,
        perceptive_field=perceptive_field,
    )
    update = perceptive_sqp_rti_update(nominal, context, cfg)
    state = update.state
    foot_pos_w = _foot_positions(state)
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
    next_solver_state = JointMpcRtiSolverState(
        trajectory=accepted_state,
        gait_phase=torch.remainder(previous.gait_phase + 1, int(cfg.gait.period_steps)),
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
    )
    return JointMpcRtiStepResult(
        full_trajectory=trajectory,
        pending_reference=pending,
        solver_state=next_solver_state,
        diagnostics=JointMpcRtiStepDiagnostics(
            nominal_state=nominal.state[:, :2],
            qp_direction=update.direction[:, :2],
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
        ),
    )


__all__ = ["build_loss_context", "step"]
