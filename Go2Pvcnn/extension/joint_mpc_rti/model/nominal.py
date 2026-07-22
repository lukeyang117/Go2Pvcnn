"""One-call cold and rolling nominal trajectory construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.analytic_ik import go2_analytic_ik
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule, fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.runtime.warm_start import shift_rebase_trajectory
from extension.joint_mpc_rti.solver.trajectory_qp import JOINT_LOWER, JOINT_UPPER
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.query import query_world
from extension.joint_mpc_rti.types import JointMpcRtiSolverState, JointMpcRtiState, JointMpcTerrainField


@dataclass(frozen=True)
class NominalTrajectory:
    state: Tensor
    foot_reference_w: Tensor
    touchdown_reference_w: Tensor
    contact_state: Tensor
    used_cold_start: Tensor
    used_warm_start: Tensor
    valid: Tensor
    current_stance_anchor_w: Tensor


class WarmStartInvariantError(RuntimeError):
    """Raised when an initialized environment loses its finite warm trajectory."""


@dataclass(frozen=True)
class _FootReferences:
    foot: Tensor
    touchdown: Tensor
    valid: Tensor
    measured_foot_w: Tensor


@dataclass(frozen=True)
class _WarmManifoldInitialization:
    state: Tensor
    valid: Tensor


def _integrate_root(measured: JointMpcRtiState, command: Tensor, cfg: JointMpcRtiCfg) -> tuple[Tensor, Tensor]:
    batch = measured.batch_size
    nodes = cfg.runtime.horizon_steps + 1
    node = torch.arange(nodes, dtype=measured.root_pos_w.dtype, device=measured.device)
    progress_node = (node - 1.0).clamp_min(0.0)
    scaled = command * float(cfg.nominal.command_scale)
    yaw = measured.root_rpy_w[:, 2:3] + scaled[:, 2:3] * float(cfg.runtime.dt) * progress_node[None]
    interval_yaw = yaw[:, :-1]
    vx = torch.cos(interval_yaw) * scaled[:, None, 0] - torch.sin(interval_yaw) * scaled[:, None, 1]
    vy = torch.sin(interval_yaw) * scaled[:, None, 0] + torch.cos(interval_yaw) * scaled[:, None, 1]
    increments = torch.stack((vx, vy), dim=-1) * float(cfg.runtime.dt)
    increments[:, 0] = 0.0
    displacement = torch.cat(
        (
            torch.zeros(batch, 1, 2, dtype=increments.dtype, device=increments.device),
            torch.cumsum(increments, dim=1),
        ),
        dim=1,
    )
    root_xy = measured.root_pos_w[:, None, :2] + displacement
    root_z = measured.root_pos_w[:, None, 2:3].expand(-1, nodes, -1)
    root_pos = torch.cat((root_xy, root_z), dim=-1)
    root_rpy = torch.cat(
        (
            measured.root_rpy_w[:, None, :2].expand(-1, nodes, -1),
            yaw[..., None],
        ),
        dim=-1,
    )
    return root_pos, root_rpy


def _gather_event(value: Tensor, index: Tensor) -> Tensor:
    expanded = value[:, :, None].expand(-1, -1, 4, -1)
    return torch.gather(expanded, 1, index[..., None].expand(-1, -1, -1, value.shape[-1]))


def _event_placement_xy(
    root_pos: Tensor,
    root_rpy: Tensor,
    footprint_xy: Tensor,
    command: Tensor,
    index: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    step_scale: float,
) -> Tensor:
    event_root = _gather_event(root_pos, index)
    event_rpy = _gather_event(root_rpy, index)
    lead_body = command[:, None, None, :2] * (
        float(cfg.runtime.dt) * float(cfg.gait.swing_steps) * float(cfg.nominal.step_reference_scale) * step_scale
    )
    local_xy = footprint_xy[:, None] + lead_body
    yaw = event_rpy[..., 2]
    world_offset = torch.stack(
        (
            torch.cos(yaw) * local_xy[..., 0] - torch.sin(yaw) * local_xy[..., 1],
            torch.sin(yaw) * local_xy[..., 0] + torch.cos(yaw) * local_xy[..., 1],
        ),
        dim=-1,
    )
    return event_root[..., :2] + world_offset


def _build_foot_references(
    measured: JointMpcRtiState,
    command: Tensor,
    terrain: JointMpcTerrainField,
    root_pos: Tensor,
    root_rpy: Tensor,
    schedule: FixedTrotSchedule,
    cfg: JointMpcRtiCfg,
    *,
    step_scale: float,
) -> _FootReferences:
    batch, nodes = root_pos.shape[:2]
    measured_foot = go2_fk(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos).foot_pos_w
    nominal_joint = constant_like(
        root_pos,
        "nominal_joint_pos",
        cfg.gait.nominal_joint_pos,
    ).expand(batch, -1)
    footprint_xy = go2_fk(
        torch.zeros_like(measured.root_pos_w),
        torch.zeros_like(measured.root_rpy_w),
        nominal_joint,
    ).foot_pos_w[..., :2]

    node = torch.arange(nodes, device=root_pos.device).view(1, nodes, 1)
    lift_raw = node - schedule.phase
    touchdown_raw = lift_raw + int(cfg.gait.swing_steps)
    stance_raw = node - (schedule.phase - int(cfg.gait.swing_steps))
    previous_touchdown_raw = lift_raw - int(cfg.gait.stance_steps)
    previous_touchdown_index = previous_touchdown_raw.clamp(0, nodes - 1)
    touchdown_index = touchdown_raw.clamp(0, nodes - 1)
    stance_index = stance_raw.clamp(0, nodes - 1)

    lift_xy_event = _event_placement_xy(
        root_pos,
        root_rpy,
        footprint_xy,
        command,
        previous_touchdown_index,
        cfg,
        step_scale=step_scale,
    )
    lift_xy_event = torch.where(
        (previous_touchdown_raw <= 0)[..., None],
        measured_foot[:, None, :, :2],
        lift_xy_event,
    )
    touchdown_xy = _event_placement_xy(
        root_pos, root_rpy, footprint_xy, command, touchdown_index, cfg, step_scale=step_scale
    )
    stance_xy_event = _event_placement_xy(
        root_pos, root_rpy, footprint_xy, command, stance_index, cfg, step_scale=step_scale
    )

    tau0 = schedule.swing_tau[:, :1]
    remaining_swing = 1.0 - tau0
    endpoint = remaining_swing <= 1.0e-4
    denominator = remaining_swing.clamp_min(1.0e-4)
    inferred_lift_xy = (measured_foot[:, None, :, :2] - tau0[..., None] * touchdown_xy[:, :1]) / denominator[..., None]
    inferred_lift_xy = torch.where(
        endpoint[..., None], measured_foot[:, None, :, :2], inferred_lift_xy
    )
    inferred_lift_xy = torch.where(
        schedule.swing[:, :1, :, None],
        inferred_lift_xy,
        measured_foot[:, None, :, :2],
    )
    lift_xy = torch.where(
        (lift_raw <= 0)[..., None], inferred_lift_xy.expand(-1, nodes, -1, -1), lift_xy_event
    )
    stance_xy = torch.where(
        ((stance_raw < 0) | (lift_raw < 0))[..., None],
        measured_foot[:, None, :, :2],
        stance_xy_event,
    )

    points_xy = torch.cat(
        (
            lift_xy.reshape(batch, -1, 2),
            touchdown_xy.reshape(batch, -1, 2),
            stance_xy.reshape(batch, -1, 2),
        ),
        dim=1,
    )
    query = query_world(terrain, points_xy)
    group = nodes * 4
    height = query.height_w.reshape(batch, 3, group).reshape(batch, 3, nodes, 4)
    query_valid = query.valid.reshape(batch, 3, group).reshape(batch, 3, nodes, 4)
    contact_offset = float(cfg.gait.foot_contact_offset)
    lift_z_event = height[:, 0] + contact_offset
    touchdown_z = height[:, 1] + contact_offset
    stance_z_event = height[:, 2] + contact_offset

    inferred_lift_z = (
        measured_foot[:, None, :, 2]
        - tau0 * touchdown_z[:, :1]
        - float(cfg.gait.h_swing) * 4.0 * tau0 * (1.0 - tau0)
    ) / denominator
    inferred_lift_z = torch.where(endpoint, measured_foot[:, None, :, 2], inferred_lift_z)
    inferred_lift_z = torch.where(schedule.swing[:, :1], inferred_lift_z, measured_foot[:, None, :, 2])
    lift_z_event = torch.where(
        previous_touchdown_raw <= 0,
        measured_foot[:, None, :, 2],
        lift_z_event,
    )
    lift_z = torch.where(
        lift_raw <= 0, inferred_lift_z.expand(-1, nodes, -1), lift_z_event
    )
    stance_z = torch.where(
        (stance_raw < 0) | (lift_raw < 0),
        measured_foot[:, None, :, 2],
        stance_z_event,
    )
    tau = schedule.swing_tau
    swing_xy = (1.0 - tau[..., None]) * lift_xy + tau[..., None] * touchdown_xy
    swing_z = (
        (1.0 - tau) * lift_z
        + tau * touchdown_z
        + float(cfg.gait.h_swing) * 4.0 * tau * (1.0 - tau)
    )
    swing_foot = torch.cat((swing_xy, swing_z[..., None]), dim=-1)
    stance_foot = torch.cat((stance_xy, stance_z[..., None]), dim=-1)
    foot = torch.where(schedule.swing[..., None], swing_foot, stance_foot)
    touchdown = torch.cat((touchdown_xy, touchdown_z[..., None]), dim=-1)
    valid = query_valid.all(dim=(1, 2, 3))
    return _FootReferences(
        foot=foot,
        touchdown=touchdown,
        valid=valid,
        measured_foot_w=measured_foot,
    )


def _build_cold_nominal(
    measured: JointMpcRtiState,
    command: Tensor,
    terrain: JointMpcTerrainField,
    schedule: FixedTrotSchedule,
    cfg: JointMpcRtiCfg,
) -> NominalTrajectory:
    root_pos, root_rpy = _integrate_root(measured, command, cfg)
    full = _build_foot_references(
        measured, command, terrain, root_pos, root_rpy, schedule, cfg, step_scale=1.0
    )
    full_joint, full_reachable = go2_analytic_ik(root_pos, root_rpy, full.foot)
    reduced = _build_foot_references(
        measured,
        command,
        terrain,
        root_pos,
        root_rpy,
        schedule,
        cfg,
        step_scale=float(cfg.nominal.unreachable_step_scale),
    )
    reduced_joint, reduced_reachable = go2_analytic_ik(root_pos, root_rpy, reduced.foot)
    use_reduced = ~full_reachable.all(dim=(1, 2))
    joint = torch.where(use_reduced[:, None, None, None], reduced_joint, full_joint).reshape(
        measured.batch_size, 31, 12
    )
    foot = torch.where(use_reduced[:, None, None, None], reduced.foot, full.foot)
    touchdown = torch.where(use_reduced[:, None, None, None], reduced.touchdown, full.touchdown)
    reachable = torch.where(use_reduced[:, None, None], reduced_reachable, full_reachable)
    state = torch.cat((root_pos, root_rpy, joint), dim=-1)
    state = torch.cat((measured.as_vector()[:, None], state[:, 1:]), dim=1)
    valid = reachable.all(dim=(1, 2)) & torch.where(use_reduced, reduced.valid, full.valid)
    return NominalTrajectory(
        state=state,
        foot_reference_w=foot,
        touchdown_reference_w=touchdown,
        contact_state=schedule.stance,
        used_cold_start=torch.ones_like(valid),
        used_warm_start=torch.zeros_like(valid),
        valid=valid,
        current_stance_anchor_w=full.measured_foot_w,
    )


def _build_warm_nominal(
    measured: JointMpcRtiState,
    command: Tensor,
    terrain: JointMpcTerrainField,
    schedule: FixedTrotSchedule,
    previous: JointMpcRtiSolverState,
    cfg: JointMpcRtiCfg,
) -> NominalTrajectory:
    state = shift_rebase_trajectory(
        previous.trajectory,
        measured.as_vector(),
        decay_nodes=int(cfg.nominal.measurement_decay_nodes),
    )
    manifold = _initialize_published_stance_manifold(
        state,
        schedule,
        previous.stance_anchor_w,
        previous.initialized,
        terrain,
        cfg,
    )
    state = manifold.state
    references = _build_foot_references(
        measured,
        command,
        terrain,
        state[..., :3],
        state[..., 3:6],
        schedule,
        cfg,
        step_scale=1.0,
    )
    foot = references.foot
    valid = references.valid & manifold.valid & torch.isfinite(state).all(dim=(1, 2))
    return NominalTrajectory(
        state=state,
        foot_reference_w=foot,
        touchdown_reference_w=references.touchdown,
        contact_state=schedule.stance,
        used_cold_start=torch.zeros_like(valid),
        used_warm_start=torch.ones_like(valid),
        valid=valid,
        current_stance_anchor_w=previous.stance_anchor_w,
    )


def _initialize_published_stance_manifold(
    state: Tensor,
    schedule: FixedTrotSchedule,
    stance_anchor_w: Tensor,
    initialized: Tensor,
    terrain: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
) -> _WarmManifoldInitialization:
    warm_rows = torch.as_tensor(initialized, dtype=torch.bool, device=state.device)
    continuing = schedule.stance[:, 0] & schedule.stance[:, 1] & warm_rows[:, None]
    published_stance = schedule.stance[:, 1] & warm_rows[:, None]
    joint_1 = state[:, 1, 6:].reshape(state.shape[0], 4, 3)
    foot_1 = go2_fk(state[:, 1, :3], state[:, 1, 3:6], state[:, 1, 6:]).foot_pos_w
    anchor = torch.as_tensor(stance_anchor_w, dtype=state.dtype, device=state.device)
    target_xy = torch.where(continuing[..., None], anchor[..., :2], foot_1[..., :2])
    ground = query_world(terrain, target_xy)
    target_z = ground.height_w + float(cfg.gait.foot_contact_offset)
    target_foot_1 = torch.cat((target_xy, target_z[..., None]), dim=-1)
    ik_joint_1, reachable = go2_analytic_ik(
        state[:, 1, :3], state[:, 1, 3:6], target_foot_1
    )
    corrected_joint_1 = torch.where(published_stance[..., None], ik_joint_1, joint_1)
    corrected_node_1 = torch.cat((state[:, 1, :6], corrected_joint_1.flatten(1)), dim=-1)
    corrected_state = torch.cat(
        (state[:, :1], corrected_node_1[:, None], state[:, 2:]), dim=1
    )

    finite_ik = torch.isfinite(ik_joint_1).all(dim=-1)
    stance_valid = torch.where(
        published_stance,
        reachable & finite_ik & ground.valid,
        torch.ones_like(published_stance),
    ).all(dim=-1)
    joint_lower = constant_like(state, "nominal_joint_lower", JOINT_LOWER)
    joint_upper = constant_like(state, "nominal_joint_upper", JOINT_UPPER)
    position_valid = (
        (corrected_state[:, 1, 6:] >= joint_lower)
        & (corrected_state[:, 1, 6:] <= joint_upper)
    ).all(dim=-1)
    maximum_step = float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt)
    velocity_valid = (
        (corrected_state[:, 1, 6:] - corrected_state[:, 0, 6:]).abs() <= maximum_step
    ).all(dim=-1)
    valid = (
        stance_valid
        & position_valid
        & velocity_valid
        & torch.isfinite(corrected_state).all(dim=(1, 2))
    )
    return _WarmManifoldInitialization(state=corrected_state, valid=valid)


def build_nominal(
    measured: JointMpcRtiState,
    command_body: Tensor,
    terrain_field: JointMpcTerrainField,
    gait_phase: Tensor,
    *,
    previous: JointMpcRtiSolverState,
    cfg: JointMpcRtiCfg,
) -> NominalTrajectory:
    """Build the complete cold/warm `[B,31,18]` nominal without trajectory loops."""
    command = torch.as_tensor(command_body, dtype=measured.root_pos_w.dtype, device=measured.device)
    phase = torch.as_tensor(gait_phase, dtype=torch.long, device=measured.device)
    if command.shape != (measured.batch_size, 3):
        raise ValueError("command_body must have shape [B,3]")
    if phase.shape != (measured.batch_size,):
        raise ValueError("gait_phase must have shape [B]")
    if previous.trajectory.shape != (measured.batch_size, 31, 18):
        raise WarmStartInvariantError("initialized warm cache must have shape [B,31,18]")
    initialized = torch.as_tensor(previous.initialized, dtype=torch.bool, device=measured.device)
    if initialized.shape != (measured.batch_size,):
        raise WarmStartInvariantError("initialized warm cache mask must have shape [B]")
    if previous.stance_anchor_w.shape != (measured.batch_size, 4, 3):
        raise WarmStartInvariantError("initialized stance anchor must have shape [B,4,3]")
    cache_finite = torch.isfinite(previous.trajectory).all(dim=(1, 2))
    anchor_finite = torch.isfinite(previous.stance_anchor_w).all(dim=(1, 2))
    if previous.trajectory.device.type == "cpu":
        if bool((initialized & ~cache_finite).any().item()):
            raise WarmStartInvariantError("initialized warm cache must be finite")
        if bool((initialized & ~anchor_finite).any().item()):
            raise WarmStartInvariantError("initialized stance anchor must be finite")
    schedule = fixed_trot_schedule(phase, horizon_steps=cfg.runtime.horizon_steps)
    cold = _build_cold_nominal(measured, command, terrain_field, schedule, cfg)
    warm = _build_warm_nominal(measured, command, terrain_field, schedule, previous, cfg)
    use_warm = initialized[:, None, None]
    state = torch.where(use_warm, warm.state, cold.state)
    foot = torch.where(use_warm[:, :, None], warm.foot_reference_w, cold.foot_reference_w)
    touchdown = torch.where(
        use_warm[:, :, None], warm.touchdown_reference_w, cold.touchdown_reference_w
    )
    valid = torch.where(initialized, warm.valid, cold.valid)
    current_stance_anchor = torch.where(
        initialized[:, None, None],
        warm.current_stance_anchor_w,
        cold.current_stance_anchor_w,
    )
    return NominalTrajectory(
        state=state,
        foot_reference_w=foot,
        touchdown_reference_w=touchdown,
        contact_state=schedule.stance,
        used_cold_start=~initialized,
        used_warm_start=initialized,
        valid=valid,
        current_stance_anchor_w=current_stance_anchor,
    )


__all__ = ["NominalTrajectory", "WarmStartInvariantError", "build_nominal"]
