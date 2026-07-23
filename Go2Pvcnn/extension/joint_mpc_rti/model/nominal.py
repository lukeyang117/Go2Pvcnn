"""Cold-once and warm-retargeted pure-kinematic nominal construction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.analytic_ik import go2_analytic_ik
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule, fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import (
    LEG_SIDE_SIGNS,
    go2_collision_geometry,
    go2_fk,
)
from extension.joint_mpc_rti.model.perceptive_plan import TouchdownPlan
from extension.joint_mpc_rti.runtime.warm_start import shift_rebase_trajectory
from extension.joint_mpc_rti.solver.trajectory_qp import JOINT_LOWER, JOINT_UPPER
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.query import query_perceptive_world
from extension.joint_mpc_rti.terrain.swept_safety import (
    PART_NAMES,
    evaluate_nodes,
    evaluate_swept_intervals,
)
from extension.joint_mpc_rti.types import (
    JointMpcPerceptiveField,
    JointMpcRtiSolverState,
    JointMpcRtiState,
    JointMpcTouchdownRegion,
)


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
    rebased_state: Tensor | None = None
    warm_cache_invariant_fault: Tensor | None = None
    nominal_safe: Tensor | None = None
    retarget_change: Tensor | None = None
    minimum_clearance_by_part: Tensor | None = None
    preview_touchdown_reference_w: Tensor | None = None
    swing_apex_w: Tensor | None = None
    preview_tail_state: Tensor | None = None
    candidate_retry_rank: Tensor | None = None
    preview_candidate_retry_rank: Tensor | None = None
    perceptive_plan: TouchdownPlan | None = None

    def __post_init__(self) -> None:
        batch = int(self.state.shape[0])
        if self.rebased_state is None:
            object.__setattr__(self, "rebased_state", self.state)
        if self.warm_cache_invariant_fault is None:
            object.__setattr__(
                self,
                "warm_cache_invariant_fault",
                torch.zeros(batch, dtype=torch.bool, device=self.state.device),
            )
        if self.nominal_safe is None:
            object.__setattr__(self, "nominal_safe", self.valid)
        if self.retarget_change is None:
            object.__setattr__(self, "retarget_change", self.state.new_zeros(batch))
        if self.minimum_clearance_by_part is None:
            object.__setattr__(
                self,
                "minimum_clearance_by_part",
                self.state.new_full((batch, len(PART_NAMES)), torch.inf),
            )
        if self.candidate_retry_rank is None:
            object.__setattr__(
                self,
                "candidate_retry_rank",
                torch.zeros(batch, 4, dtype=torch.long, device=self.state.device),
            )
        if self.preview_candidate_retry_rank is None:
            object.__setattr__(
                self,
                "preview_candidate_retry_rank",
                torch.zeros(batch, 4, dtype=torch.long, device=self.state.device),
            )


class WarmStartInvariantError(RuntimeError):
    """Raised for fixed-shape warm cache contract violations."""


@dataclass(frozen=True)
class _SwingSegment:
    foot_w: Tensor
    apex_w: Tensor
    valid: Tensor


@dataclass(frozen=True)
class _SelectedTouchdownPlan:
    target_w: Tensor
    event_step: Tensor
    preview_touchdown_step: Tensor
    valid: Tensor
    region: JointMpcTouchdownRegion
    preview_target_w: Tensor
    preview_valid: Tensor
    preview_region: JointMpcTouchdownRegion


def _quintic(value: Tensor) -> Tensor:
    return 10.0 * value.pow(3) - 15.0 * value.pow(4) + 6.0 * value.pow(5)


def _quintic_start_velocity_basis(value: Tensor) -> Tensor:
    return value - 6.0 * value.pow(3) + 8.0 * value.pow(4) - 3.0 * value.pow(5)


def _cold_root_trajectory(
    measured: JointMpcRtiState,
    command: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    nodes = int(cfg.runtime.state_nodes)
    node = constant_like(
        measured.root_pos_w,
        f"nominal_node_index_{nodes}",
        tuple(float(index) for index in range(nodes)),
    )
    scaled = command * float(cfg.nominal.command_scale)
    yaw = (
        measured.root_rpy_w[:, 2:3]
        + scaled[:, 2:3] * float(cfg.runtime.dt) * node[None]
    )
    edge_yaw = yaw[:, :-1]
    world_velocity = torch.stack(
        (
            torch.cos(edge_yaw) * scaled[:, None, 0]
            - torch.sin(edge_yaw) * scaled[:, None, 1],
            torch.sin(edge_yaw) * scaled[:, None, 0]
            + torch.cos(edge_yaw) * scaled[:, None, 1],
        ),
        dim=-1,
    )
    displacement = torch.cat(
        (
            torch.zeros_like(world_velocity[:, :1]),
            torch.cumsum(world_velocity * float(cfg.runtime.dt), dim=1),
        ),
        dim=1,
    )
    root_pos = torch.cat(
        (
            measured.root_pos_w[:, None, :2] + displacement,
            measured.root_pos_w[:, None, 2:3].expand(-1, nodes, -1),
        ),
        dim=-1,
    )
    root_rpy = torch.cat(
        (
            measured.root_rpy_w[:, None, :2].expand(-1, nodes, -1),
            yaw[..., None],
        ),
        dim=-1,
    )
    return root_pos, root_rpy


def _build_rebased_state(
    measured: JointMpcRtiState,
    command: Tensor,
    previous: JointMpcRtiSolverState,
    initialized: Tensor,
    cache_finite: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    measured_vector = measured.as_vector()
    measured_hold = measured_vector[:, None].expand(-1, int(cfg.runtime.state_nodes), -1)
    finite_cache = torch.where(
        cache_finite[:, None, None], previous.trajectory, measured_hold
    )
    terminal_state = None
    if (
        previous.preview_tail_state is not None
        and previous.preview_tail_state.shape
        == (measured.batch_size, int(cfg.gait.swing_steps + 1), 18)
    ):
        terminal_state = previous.preview_tail_state[:, 1]
    warm = shift_rebase_trajectory(
        finite_cache,
        measured_vector,
        decay_nodes=int(cfg.nominal.measurement_decay_nodes),
        command_body=command,
        dt=float(cfg.runtime.dt),
        terminal_command_scale=float(cfg.nominal.terminal_command_fill_scale),
        terminal_joint_step_limit=float(cfg.solver.joint_velocity_limit)
        * float(cfg.runtime.dt),
        terminal_state=terminal_state,
    )
    cold_root, cold_rpy = _cold_root_trajectory(measured, command, cfg)
    cold = torch.cat(
        (
            cold_root,
            cold_rpy,
            measured.joint_pos[:, None].expand(-1, int(cfg.runtime.state_nodes), -1),
        ),
        dim=-1,
    )
    rebased = torch.where(initialized[:, None, None], warm, cold)
    return torch.cat((measured_vector[:, None], rebased[:, 1:]), dim=1)


def _preview_touchdown(
    target_w: Tensor,
    command: Tensor,
    event_yaw: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    duration = float(cfg.gait.period_steps) * float(cfg.runtime.dt)
    scale = float(cfg.touchdown.command_prediction_scale)
    displacement_b = command[..., :2] * (duration * scale)
    displacement_w = torch.stack(
        (
            torch.cos(event_yaw) * displacement_b[:, None, 0]
            - torch.sin(event_yaw) * displacement_b[:, None, 1],
            torch.sin(event_yaw) * displacement_b[:, None, 0]
            + torch.cos(event_yaw) * displacement_b[:, None, 1],
        ),
        dim=-1,
    )
    preview_xy = target_w[..., :2] + displacement_w
    query = query_perceptive_world(field, preview_xy.reshape(target_w.shape[0], 4, 2))
    preview = torch.cat(
        (
            preview_xy,
            (query.height_w + float(cfg.gait.foot_contact_offset))[..., None],
        ),
        dim=-1,
    )
    return preview, query.valid & query.landing_safe


def _swing_segment(
    lift_w: Tensor,
    touchdown_w: Tensor,
    event_yaw: Tensor,
    tau: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    name: str,
) -> _SwingSegment:
    batch = int(lift_w.shape[0])
    samples = int(cfg.touchdown.swing_samples)
    sample_tau = constant_like(
        lift_w,
        f"nominal_{name}_apex_fraction_{samples}",
        tuple(index / float(samples - 1) for index in range(samples)),
    ).view(1, 1, samples, 1)
    side = constant_like(lift_w, "nominal_leg_side", LEG_SIDE_SIGNS).view(1, 4)
    outward = torch.stack(
        (-torch.sin(event_yaw) * side, torch.cos(event_yaw) * side), dim=-1
    )

    sample_smooth = _quintic(sample_tau)
    sample_bump = 64.0 * sample_tau.pow(3) * (1.0 - sample_tau).pow(3)
    sample_xy = (
        lift_w[:, :, None, :2]
        + sample_smooth * (touchdown_w[:, :, None, :2] - lift_w[:, :, None, :2])
        + sample_bump
        * float(cfg.nominal.swing_outward_offset_m)
        * outward[:, :, None]
    )
    query = query_perceptive_world(field, sample_xy.reshape(batch, 4 * samples, 2))
    height = query.inflated_height_w[..., 0].reshape(batch, 4, samples)
    query_valid = query.valid.reshape(batch, 4, samples)
    height = torch.where(query_valid, height, torch.full_like(height, -torch.inf))
    apex_z = (
        height.amax(dim=-1)
        + float(cfg.terrain.foot_radius_m)
        + float(cfg.nominal.swing_apex_margin_m)
    )
    apex_z = torch.maximum(
        apex_z, torch.maximum(lift_w[..., 2], touchdown_w[..., 2])
    )
    apex_xy = (
        0.5 * (lift_w[..., :2] + touchdown_w[..., :2])
        + float(cfg.nominal.swing_outward_offset_m) * outward
    )

    tau_value = tau[..., None].clamp(0.0, 1.0)
    smooth = _quintic(tau_value)
    bump = 64.0 * tau_value.pow(3) * (1.0 - tau_value).pow(3)
    foot_xy = (
        lift_w[:, None, :, :2]
        + smooth * (touchdown_w[:, None, :, :2] - lift_w[:, None, :, :2])
        + bump
        * float(cfg.nominal.swing_outward_offset_m)
        * outward[:, None]
    )
    first_tau = (2.0 * tau_value).clamp(0.0, 1.0)
    second_tau = (2.0 * tau_value - 1.0).clamp(0.0, 1.0)
    foot_z = torch.where(
        tau_value <= 0.5,
        lift_w[:, None, :, 2:3]
        + _quintic(first_tau)
        * (apex_z[:, None, :, None] - lift_w[:, None, :, 2:3]),
        apex_z[:, None, :, None]
        + _quintic(second_tau)
        * (touchdown_w[:, None, :, 2:3] - apex_z[:, None, :, None]),
    )
    return _SwingSegment(
        foot_w=torch.cat((foot_xy, foot_z), dim=-1),
        apex_w=torch.cat((apex_xy, apex_z[..., None]), dim=-1),
        valid=query_valid.all(dim=-1),
    )


def _foot_references(
    measured: JointMpcRtiState,
    command: Tensor,
    rebased: Tensor,
    schedule: FixedTrotSchedule,
    plan: _SelectedTouchdownPlan,
    current_anchor_w: Tensor,
    preserve_warm_boundary: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, nodes = map(int, rebased.shape[:2])
    node = constant_like(
        rebased,
        f"nominal_foot_node_index_{nodes}",
        tuple(float(index) for index in range(nodes)),
    ).view(1, nodes, 1)
    phase0 = schedule.phase_node[:, 0]
    first_touchdown = plan.event_step
    first_liftoff = torch.where(
        phase0 < int(cfg.gait.swing_steps),
        torch.zeros_like(first_touchdown),
        first_touchdown - int(cfg.gait.swing_steps),
    )
    second_liftoff = first_touchdown + int(cfg.gait.stance_steps)
    second_touchdown = first_touchdown + int(cfg.gait.period_steps)

    root_index = first_touchdown[:, None, :, None].expand(-1, 1, -1, 3)
    event_rpy = torch.gather(
        rebased[..., 3:6].unsqueeze(2).expand(-1, -1, 4, -1),
        1,
        root_index,
    ).squeeze(1)
    preview_target = plan.preview_target_w
    preview_valid = plan.preview_valid

    measured_foot = go2_collision_geometry(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_center_w
    current_swing = phase0 < int(cfg.gait.swing_steps)
    first_lift = torch.where(current_swing[..., None], measured_foot, current_anchor_w)
    first_duration = (first_touchdown - first_liftoff).clamp_min(1).to(rebased.dtype)
    first_tau = (node - first_liftoff[:, None].to(rebased.dtype)) / first_duration[:, None]
    first_segment = _swing_segment(
        first_lift,
        plan.target_w,
        event_rpy[..., 2],
        first_tau,
        field,
        cfg,
        name="first",
    )
    shifted_foot = go2_fk(
        rebased[..., :3], rebased[..., 3:6], rebased[..., 6:]
    ).foot_pos_w
    first_tau_clamped = first_tau.clamp(0.0, 1.0)
    first_step_tau = first_duration.reciprocal()
    velocity_basis = _quintic_start_velocity_basis(first_tau_clamped)
    first_step_basis = _quintic_start_velocity_basis(first_step_tau).clamp_min(1.0e-6)
    boundary_correction = shifted_foot[:, 1] - first_segment.foot_w[:, 1]
    correction_mask = preserve_warm_boundary[:, None] & current_swing
    corrected_first_foot = first_segment.foot_w + (
        velocity_basis / first_step_basis[:, None]
    )[..., None] * boundary_correction[:, None] * correction_mask[:, None, :, None]
    first_segment = _SwingSegment(
        foot_w=corrected_first_foot,
        apex_w=first_segment.apex_w,
        valid=first_segment.valid,
    )

    preview_yaw = event_rpy[..., 2] + command[:, None, 2] * (
        float(cfg.gait.period_steps) * float(cfg.runtime.dt)
    )
    second_tau = (
        node - second_liftoff[:, None].to(rebased.dtype)
    ) / float(cfg.gait.swing_steps)
    second_segment = _swing_segment(
        plan.target_w,
        preview_target,
        preview_yaw,
        second_tau,
        field,
        cfg,
        name="preview",
    )

    before_first_lift = node < first_liftoff[:, None]
    after_first_touchdown = node >= first_touchdown[:, None]
    after_second_liftoff = node >= second_liftoff[:, None]
    after_second_touchdown = node >= second_touchdown[:, None]
    foot = torch.where(
        before_first_lift[..., None],
        current_anchor_w[:, None],
        first_segment.foot_w,
    )
    foot = torch.where(
        after_first_touchdown[..., None], plan.target_w[:, None], foot
    )
    foot = torch.where(
        after_second_liftoff[..., None], second_segment.foot_w, foot
    )
    foot = torch.where(
        after_second_touchdown[..., None], preview_target[:, None], foot
    )
    touchdown = torch.where(
        after_second_liftoff[..., None], preview_target[:, None], plan.target_w[:, None]
    )
    preview_required = second_liftoff <= int(cfg.runtime.horizon_steps)
    path_valid = first_segment.valid & torch.where(
        preview_required,
        second_segment.valid,
        torch.ones_like(second_segment.valid),
    )
    preview_safe = torch.where(
        preview_required, preview_valid, torch.ones_like(preview_valid)
    )
    valid = path_valid.all(dim=-1) & preview_safe.all(dim=-1)
    apex = torch.stack((first_segment.apex_w, second_segment.apex_w), dim=2)
    return foot, touchdown, preview_target, apex, valid


def _gather_foot_at_event(foot_w: Tensor, event_step: Tensor) -> Tensor:
    index = event_step[:, None, :, None].expand(-1, 1, -1, 3)
    return torch.gather(foot_w, 1, index).squeeze(1)


def _extend_rebased_to_preview(
    rebased: Tensor,
    command: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    preview_nodes = int(cfg.gait.swing_steps)
    step = constant_like(
        rebased,
        f"nominal_preview_step_{preview_nodes}",
        tuple(float(index) for index in range(1, preview_nodes + 1)),
    )
    scaled = command * float(cfg.nominal.command_scale)
    yaw0 = rebased[:, -1, 5:6]
    yaw = yaw0 + scaled[:, 2:3] * float(cfg.runtime.dt) * step[None]
    edge_yaw = yaw0 + scaled[:, 2:3] * float(cfg.runtime.dt) * (
        step[None] - 1.0
    )
    world_velocity = torch.stack(
        (
            torch.cos(edge_yaw) * scaled[:, None, 0]
            - torch.sin(edge_yaw) * scaled[:, None, 1],
            torch.sin(edge_yaw) * scaled[:, None, 0]
            + torch.cos(edge_yaw) * scaled[:, None, 1],
        ),
        dim=-1,
    )
    displacement = torch.cumsum(world_velocity * float(cfg.runtime.dt), dim=1)
    root_xy = rebased[:, -1:, :2] + displacement
    root_trend = rebased[:, -1, 2:5] - rebased[:, -2, 2:5]
    root_z_rp = rebased[:, -1:, 2:5] + step[None, :, None] * root_trend[:, None]
    root = torch.cat(
        (
            root_xy,
            root_z_rp[..., :1],
            root_z_rp[..., 1:],
            yaw[..., None],
        ),
        dim=-1,
    )
    joint_trend = (rebased[:, -1, 6:] - rebased[:, -2, 6:]).clamp(
        -float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt),
        float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt),
    )
    joint = rebased[:, -1:, 6:] + step[None, :, None] * joint_trend[:, None]
    return torch.cat((rebased, torch.cat((root, joint), dim=-1)), dim=1)


def _build_preview_tail(
    measured: JointMpcRtiState,
    command: Tensor,
    rebased: Tensor,
    primary_state: Tensor,
    plan: _SelectedTouchdownPlan,
    current_anchor_w: Tensor,
    preserve_warm_boundary: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    gait_phase: Tensor,
) -> tuple[Tensor, FixedTrotSchedule, Tensor, Tensor]:
    extended = _extend_rebased_to_preview(rebased, command, cfg)
    schedule = fixed_trot_schedule(
        gait_phase,
        horizon_steps=int(cfg.runtime.horizon_steps + cfg.gait.swing_steps),
    )
    foot, _, _, _, path_valid = _foot_references(
        measured,
        command,
        extended,
        schedule,
        plan,
        current_anchor_w,
        preserve_warm_boundary,
        field,
        cfg,
    )
    joint, reachable = go2_analytic_ik(
        extended[..., :3], extended[..., 3:6], foot
    )
    full_state = torch.cat(
        (extended[..., :6], joint.reshape(int(extended.shape[0]), 43, 12)), dim=-1
    )
    tail = full_state[:, 30:]
    tail = torch.cat((primary_state[:, 30:31], tail[:, 1:]), dim=1)
    return tail, schedule, reachable[:, 30:], path_valid


def _preview_tail_safety(
    tail_state: Tensor,
    tail_schedule: FixedTrotSchedule,
    tail_reachable: Tensor,
    path_valid: Tensor,
    plan: _SelectedTouchdownPlan,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    node_safety = evaluate_nodes(
        tail_state,
        field,
        cfg,
        contact_state=tail_schedule.stance_node[:, 30:],
    )
    swept_safety = evaluate_swept_intervals(
        tail_state,
        field,
        cfg,
        contact_state=tail_schedule.stance_node[:, 30:],
    )
    preview_step = plan.preview_touchdown_step
    batch_limit = torch.where(
        preview_step > 30, preview_step, preview_step.new_full((), 30)
    ).amax(dim=-1)
    node = constant_like(
        tail_state,
        "nominal_preview_global_node",
        tuple(float(index) for index in range(30, 43)),
    ).view(1, 13)
    node_active = node <= batch_limit[:, None]
    edge_active = node[:, :-1] < batch_limit[:, None]
    reachable_safe = torch.where(
        node_active[..., None], tail_reachable, torch.ones_like(tail_reachable)
    ).all(dim=(1, 2))
    node_safe = torch.where(
        node_active, node_safety.safe, torch.ones_like(node_safety.safe)
    ).all(dim=1)
    swept_safe = torch.where(
        edge_active, swept_safety.safe, torch.ones_like(swept_safety.safe)
    ).all(dim=1)
    joint = tail_state[..., 6:]
    lower = constant_like(tail_state, "preview_joint_lower", JOINT_LOWER)
    upper = constant_like(tail_state, "preview_joint_upper", JOINT_UPPER)
    joint_position = torch.where(
        node_active[..., None],
        (joint >= lower) & (joint <= upper),
        torch.ones_like(joint, dtype=torch.bool),
    ).all(dim=(1, 2))
    maximum_step = float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt)
    joint_velocity = torch.where(
        edge_active[..., None],
        (joint[:, 1:] - joint[:, :-1]).abs() <= maximum_step,
        torch.ones_like(joint[:, 1:], dtype=torch.bool),
    ).all(dim=(1, 2))
    preview_event = plan.event_step + int(cfg.gait.period_steps)
    preview_required = (preview_event > 30) & (preview_event <= 42)
    preview_index = (preview_event - 30).clamp(0, 12)
    tail_foot = go2_fk(
        tail_state[..., :3], tail_state[..., 3:6], tail_state[..., 6:]
    ).foot_pos_w
    preview_foot = _gather_foot_at_event(tail_foot, preview_index)
    preview_region_margin = (
        torch.einsum(
            "blij,blj->bli", plan.preview_region.A, preview_foot[..., :2]
        )
        + plan.preview_region.b
    )
    preview_region_safe = (preview_region_margin >= -1.0e-5).all(dim=-1)
    preview_target_safe = (
        torch.linalg.vector_norm(preview_foot - plan.preview_target_w, dim=-1)
        <= float(cfg.solver.published_stance_tolerance)
    )
    preview_event_safe = torch.where(
        preview_required,
        preview_region_safe
        & preview_target_safe
        & plan.preview_valid
        & plan.preview_region.valid,
        torch.ones_like(preview_required),
    ).all(dim=-1)
    required = batch_limit > 30
    safe = torch.where(
        required,
        node_safe
        & swept_safe
        & reachable_safe
        & joint_position
        & joint_velocity
        & path_valid
        & preview_event_safe,
        torch.ones_like(required),
    )
    clearance = torch.stack(
        tuple(
            torch.minimum(
                torch.where(
                    node_active,
                    node_safety.minimum_clearance_by_part[name],
                    torch.full_like(
                        node_safety.minimum_clearance_by_part[name], torch.inf
                    ),
                ).amin(dim=1),
                torch.where(
                    edge_active,
                    swept_safety.minimum_clearance_by_part[name],
                    torch.full_like(
                        swept_safety.minimum_clearance_by_part[name], torch.inf
                    ),
                ).amin(dim=1),
            )
            for name in PART_NAMES
        ),
        dim=-1,
    )
    return safe, clearance


def _hard_safety(
    state: Tensor,
    foot_reference_w: Tensor,
    schedule: FixedTrotSchedule,
    plan: _SelectedTouchdownPlan,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    node_safety = evaluate_nodes(
        state, field, cfg, contact_state=schedule.stance_node
    )
    swept_safety = evaluate_swept_intervals(
        state, field, cfg, contact_state=schedule.stance_node
    )
    lower = constant_like(state, "nominal_joint_lower", JOINT_LOWER)
    upper = constant_like(state, "nominal_joint_upper", JOINT_UPPER)
    joint = state[..., 6:]
    position_safe = ((joint >= lower) & (joint <= upper)).all(dim=(1, 2))
    maximum_step = float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt)
    velocity_safe = (
        (joint[:, 1:] - joint[:, :-1]).abs() <= maximum_step
    ).all(dim=(1, 2))

    actual_foot = go2_fk(state[..., :3], state[..., 3:6], state[..., 6:]).foot_pos_w
    future = constant_like(
        state,
        "nominal_future_node_mask",
        tuple(False if index == 0 else True for index in range(31)),
    ).to(torch.bool).view(1, 31, 1)
    stance_mask = schedule.stance_node & future
    stance_error = torch.linalg.vector_norm(actual_foot - foot_reference_w, dim=-1)
    stance_safe = torch.where(
        stance_mask,
        stance_error <= float(cfg.solver.published_stance_tolerance),
        torch.ones_like(stance_mask),
    ).all(dim=(1, 2))

    event_foot = _gather_foot_at_event(actual_foot, plan.event_step)
    region_margin = (
        torch.einsum("blij,blj->bli", plan.region.A, event_foot[..., :2])
        + plan.region.b
    )
    region_safe = (region_margin >= -1.0e-5).all(dim=(1, 2))
    touchdown_safe = (
        torch.linalg.vector_norm(event_foot - plan.target_w, dim=-1)
        <= float(cfg.solver.published_stance_tolerance)
    ).all(dim=-1)
    preview_event = plan.event_step + int(cfg.gait.period_steps)
    preview_inside = preview_event <= int(cfg.runtime.horizon_steps)
    preview_foot = _gather_foot_at_event(
        actual_foot, preview_event.clamp_max(int(cfg.runtime.horizon_steps))
    )
    preview_region_margin = (
        torch.einsum(
            "blij,blj->bli", plan.preview_region.A, preview_foot[..., :2]
        )
        + plan.preview_region.b
    )
    preview_region_safe = (preview_region_margin >= -1.0e-5).all(dim=-1)
    preview_target_safe = (
        torch.linalg.vector_norm(preview_foot - plan.preview_target_w, dim=-1)
        <= float(cfg.solver.published_stance_tolerance)
    )
    preview_safe = torch.where(
        preview_inside,
        preview_region_safe
        & preview_target_safe
        & plan.preview_valid
        & plan.preview_region.valid,
        torch.ones_like(preview_inside),
    ).all(dim=-1)

    clearance = torch.stack(
        tuple(
            torch.minimum(
                node_safety.minimum_clearance_by_part[name].amin(dim=1),
                swept_safety.minimum_clearance_by_part[name].amin(dim=1),
            )
            for name in PART_NAMES
        ),
        dim=-1,
    )
    safe = (
        torch.isfinite(state).all(dim=(1, 2))
        & position_safe
        & velocity_safe
        & stance_safe
        & region_safe
        & touchdown_safe
        & preview_safe
        & plan.valid.all(dim=-1)
        & plan.region.valid.all(dim=-1)
        & node_safety.safe.all(dim=1)
        & swept_safety.safe.all(dim=1)
    )
    return safe, clearance


def _build_nominal_once(
    measured: JointMpcRtiState,
    command_body: Tensor,
    terrain_field: JointMpcPerceptiveField,
    gait_phase: Tensor,
    *,
    perceptive_plan: _SelectedTouchdownPlan,
    previous: JointMpcRtiSolverState,
    cfg: JointMpcRtiCfg,
) -> NominalTrajectory:
    """Retarget one fixed `[B,31,18]` nominal without environment or node loops."""
    batch = measured.batch_size
    command = torch.as_tensor(
        command_body, dtype=measured.root_pos_w.dtype, device=measured.device
    )
    phase = torch.as_tensor(gait_phase, dtype=torch.long, device=measured.device)
    if command.shape != (batch, 3):
        raise ValueError("command_body must have shape [B,3]")
    if phase.shape != (batch,):
        raise ValueError("gait_phase must have shape [B]")
    if previous.trajectory.shape != (batch, 31, 18):
        raise WarmStartInvariantError("warm cache must have shape [B,31,18]")
    if previous.initialized.shape != (batch,):
        raise WarmStartInvariantError("initialized must have shape [B]")
    if previous.stance_anchor_w.shape != (batch, 4, 3):
        raise WarmStartInvariantError("stance anchor must have shape [B,4,3]")

    initialized = torch.as_tensor(
        previous.initialized, dtype=torch.bool, device=measured.device
    )
    cache_finite = torch.isfinite(previous.trajectory).all(dim=(1, 2))
    anchor_finite = torch.isfinite(previous.stance_anchor_w).all(dim=(1, 2))
    warm_fault = initialized & ~(cache_finite & anchor_finite)
    schedule = fixed_trot_schedule(phase, horizon_steps=int(cfg.runtime.horizon_steps))
    rebased = _build_rebased_state(
        measured, command, previous, initialized, cache_finite, cfg
    )

    measured_foot = go2_collision_geometry(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_center_w
    finite_anchor = torch.where(
        anchor_finite[:, None, None], previous.stance_anchor_w, measured_foot
    )
    current_anchor = torch.where(
        initialized[:, None, None], finite_anchor, measured_foot
    )
    foot_reference, touchdown_reference, preview_target, apex, path_valid = _foot_references(
        measured,
        command,
        rebased,
        schedule,
        perceptive_plan,
        current_anchor,
        initialized & ~warm_fault,
        terrain_field,
        cfg,
    )

    ik_joint, reachable = go2_analytic_ik(
        rebased[..., :3], rebased[..., 3:6], foot_reference
    )
    node = constant_like(
        rebased,
        "nominal_ik_node_index",
        tuple(float(index) for index in range(31)),
    ).view(1, 31, 1)
    warm_swing_gamma = _quintic(schedule.swing_tau_node) * float(
        cfg.nominal.ik_blend_scale
    )
    current_swing = schedule.phase_node[:, 0] < int(cfg.gait.swing_steps)
    cold_current_tau = (
        node / perceptive_plan.event_step.clamp_min(1)[:, None].to(rebased.dtype)
    ).clamp(0.0, 1.0)
    cold_swing_gamma = torch.where(
        current_swing[:, None]
        & (node <= perceptive_plan.event_step[:, None]),
        _quintic(cold_current_tau),
        torch.ones_like(warm_swing_gamma),
    )
    swing_gamma = torch.where(
        initialized[:, None, None], warm_swing_gamma, cold_swing_gamma
    )
    swing_gamma = torch.where(
        initialized[:, None, None]
        & (schedule.phase_node == int(cfg.gait.swing_steps - 1)),
        torch.ones_like(swing_gamma),
        swing_gamma,
    )
    gamma = torch.where(
        schedule.stance_node,
        torch.ones_like(swing_gamma),
        swing_gamma,
    )
    gamma = torch.where(node == 0, torch.zeros_like(gamma), gamma)
    joint = rebased[..., 6:].reshape(batch, 31, 4, 3)
    future_liftoff = (schedule.phase_node == 0) & (node > 0)
    joint = torch.where(future_liftoff[..., None], ik_joint, joint)
    joint = joint + gamma[..., None] * (ik_joint - joint)
    state = torch.cat((rebased[..., :6], joint.reshape(batch, 31, 12)), dim=-1)
    state = torch.cat((measured.as_vector()[:, None], state[:, 1:]), dim=1)

    nominal_safe, clearance = _hard_safety(
        state,
        foot_reference,
        schedule,
        perceptive_plan,
        terrain_field,
        cfg,
    )
    tail_state, tail_schedule, tail_reachable, tail_path_valid = _build_preview_tail(
        measured,
        command,
        rebased,
        state,
        perceptive_plan,
        current_anchor,
        initialized & ~warm_fault,
        terrain_field,
        cfg,
        phase,
    )
    tail_safe, tail_clearance = _preview_tail_safety(
        tail_state,
        tail_schedule,
        tail_reachable,
        tail_path_valid,
        perceptive_plan,
        terrain_field,
        cfg,
    )
    clearance = torch.minimum(clearance, tail_clearance)
    nominal_safe = (
        nominal_safe
        & path_valid
        & reachable.all(dim=(1, 2))
        & tail_safe
        & ~warm_fault
    )
    retarget_change = torch.sqrt(
        (state[:, 1:] - rebased[:, 1:]).square().mean(dim=(1, 2))
    )
    return NominalTrajectory(
        state=state,
        rebased_state=rebased,
        foot_reference_w=foot_reference,
        touchdown_reference_w=touchdown_reference,
        contact_state=schedule.stance_node,
        used_cold_start=~initialized,
        used_warm_start=initialized,
        warm_cache_invariant_fault=warm_fault,
        nominal_safe=nominal_safe,
        valid=nominal_safe,
        current_stance_anchor_w=current_anchor,
        retarget_change=retarget_change,
        minimum_clearance_by_part=clearance,
        preview_touchdown_reference_w=preview_target,
        swing_apex_w=apex,
        preview_tail_state=tail_state,
    )


def _gather_ranked_candidates(value: Tensor, ranked_index: Tensor) -> Tensor:
    trailing = tuple(value.shape[3:])
    gather_index = ranked_index
    for _ in trailing:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(*ranked_index.shape, *trailing)
    gathered = torch.gather(value, 2, gather_index)
    order = (0, 2, 1, *range(3, gathered.ndim))
    return gathered.permute(order)


def _select_leg_retry(value: Tensor, index: Tensor) -> Tensor:
    order = (0, 2, 1, *range(3, value.ndim))
    per_leg = value.permute(order)
    trailing = tuple(per_leg.shape[3:])
    gather_index = index[..., None]
    for _ in trailing:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(*index.shape, 1, *trailing)
    return torch.gather(per_leg, 2, gather_index).squeeze(2)


def _ranked_regions(
    plan: TouchdownPlan, *, preview: bool = False
) -> JointMpcTouchdownRegion:
    ranked = plan.preview_ranked_index if preview else plan.ranked_index
    selected_index = (
        plan.preview_selected_index if preview else plan.selected_index
    )
    primary = (ranked == selected_index[..., None]).permute(0, 2, 1)

    def choose(candidate: Tensor, selected: Tensor) -> Tensor:
        ranked_value = _gather_ranked_candidates(candidate, ranked)
        selected_value = selected[:, None].expand(
            -1, int(ranked.shape[-1]), *selected.shape[1:]
        )
        condition = primary
        while condition.ndim < ranked_value.ndim:
            condition = condition.unsqueeze(-1)
        return torch.where(condition, selected_value, ranked_value)

    candidate = plan.preview_candidate_region if preview else plan.candidate_region
    selected = plan.preview_region if preview else plan.region
    return JointMpcTouchdownRegion(
        A=choose(candidate.A, selected.A),
        b=choose(candidate.b, selected.b),
        half_extent=choose(candidate.half_extent, selected.half_extent),
        corners_w=choose(candidate.corners_w, selected.corners_w),
        plane=choose(candidate.plane, selected.plane),
        normal_w=choose(candidate.normal_w, selected.normal_w),
        plane_residual=choose(candidate.plane_residual, selected.plane_residual),
        area=choose(candidate.area, selected.area),
        distance_to_forbidden=choose(
            candidate.distance_to_forbidden, selected.distance_to_forbidden
        ),
        valid=choose(candidate.valid, selected.valid),
    )


def _normalize_previous_cache(
    measured: JointMpcRtiState,
    gait_phase: Tensor,
    previous: JointMpcRtiSolverState,
) -> JointMpcRtiSolverState:
    batch = measured.batch_size
    measured_nodes = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    measured_anchor = go2_collision_geometry(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_center_w

    initialized_ok = (
        isinstance(previous.initialized, Tensor)
        and previous.initialized.shape == (batch,)
    )
    initialized = (
        previous.initialized.to(device=measured.device, dtype=torch.bool)
        if initialized_ok
        else torch.ones(batch, dtype=torch.bool, device=measured.device)
    )
    trajectory_ok = (
        isinstance(previous.trajectory, Tensor)
        and previous.trajectory.shape == (batch, 31, 18)
    )
    anchor_ok = (
        isinstance(previous.stance_anchor_w, Tensor)
        and previous.stance_anchor_w.shape == (batch, 4, 3)
    )
    shape_fault = not (initialized_ok and trajectory_ok and anchor_ok)
    fault = initialized & torch.full_like(initialized, shape_fault)

    trajectory = (
        previous.trajectory.to(device=measured.device, dtype=measured.root_pos_w.dtype)
        if trajectory_ok
        else measured_nodes
    )
    anchor = (
        previous.stance_anchor_w.to(
            device=measured.device, dtype=measured.root_pos_w.dtype
        )
        if anchor_ok
        else measured_anchor
    )
    trajectory = torch.where(
        fault[:, None, None], torch.full_like(trajectory, torch.nan), trajectory
    )
    anchor = torch.where(
        fault[:, None, None], torch.full_like(anchor, torch.nan), anchor
    )
    phase = torch.as_tensor(gait_phase, dtype=torch.long, device=measured.device)
    preview = previous.preview_tail_state
    if not (
        isinstance(preview, Tensor)
        and preview.shape == (batch, 13, 18)
    ):
        preview = None
    return JointMpcRtiSolverState(
        trajectory=trajectory,
        gait_phase=phase,
        initialized=initialized,
        stance_anchor_w=anchor,
        preview_tail_state=preview,
    )


def build_rebased_seed(
    measured: JointMpcRtiState,
    command_body: Tensor,
    gait_phase: Tensor,
    previous: JointMpcRtiSolverState,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    command = torch.as_tensor(
        command_body, dtype=measured.root_pos_w.dtype, device=measured.device
    )
    normalized = _normalize_previous_cache(measured, gait_phase, previous)
    initialized = normalized.initialized.to(dtype=torch.bool, device=measured.device)
    cache_finite = torch.isfinite(normalized.trajectory).all(dim=(1, 2))
    return _build_rebased_state(
        measured, command, normalized, initialized, cache_finite, cfg
    )


def build_nominal(
    measured: JointMpcRtiState,
    command_body: Tensor,
    terrain_field: JointMpcPerceptiveField,
    gait_phase: Tensor,
    *,
    perceptive_plan: TouchdownPlan,
    previous: JointMpcRtiSolverState,
    cfg: JointMpcRtiCfg,
) -> NominalTrajectory:
    """Build all ranked candidate layers and select the first hard-safe nominal."""
    batch = measured.batch_size
    retries = int(perceptive_plan.ranked_index.shape[-1])
    if perceptive_plan.ranked_index.shape != (batch, 4, 25) or retries != 25:
        raise ValueError("ranked touchdown candidates must have shape [B,4,25]")
    previous = _normalize_previous_cache(measured, gait_phase, previous)

    ranked_index = perceptive_plan.ranked_index
    ranked_candidate = _gather_ranked_candidates(
        perceptive_plan.candidate_w, ranked_index
    )
    ranked_region = _ranked_regions(perceptive_plan)
    ranked_target = torch.cat(
        (
            ranked_candidate[..., :2],
            ranked_region.plane[..., :1] + float(cfg.gait.foot_contact_offset),
        ),
        dim=-1,
    )
    selector_valid = _gather_ranked_candidates(
        perceptive_plan.safe_mask, ranked_index
    )
    region_margin = (
        torch.einsum(
            "brlci,brli->brlc", ranked_region.A, ranked_target[..., :2]
        )
        + ranked_region.b
    )
    ranked_valid = (
        selector_valid
        & ranked_region.valid
        & (region_margin >= -1.0e-6).all(dim=-1)
        & torch.isfinite(ranked_target).all(dim=-1)
    )
    any_feasible = ranked_valid.any(dim=1)
    first_feasible = ranked_valid.to(torch.int64).argmax(dim=1)
    chosen_index = _select_leg_retry(
        ranked_index.permute(0, 2, 1), first_feasible
    )
    chosen_target = _select_leg_retry(ranked_target, first_feasible)

    def choose_region(value: Tensor) -> Tensor:
        return _select_leg_retry(value, first_feasible)

    chosen_region = JointMpcTouchdownRegion(
        A=choose_region(ranked_region.A),
        b=choose_region(ranked_region.b),
        half_extent=choose_region(ranked_region.half_extent),
        corners_w=choose_region(ranked_region.corners_w),
        plane=choose_region(ranked_region.plane),
        normal_w=choose_region(ranked_region.normal_w),
        plane_residual=choose_region(ranked_region.plane_residual),
        area=choose_region(ranked_region.area),
        distance_to_forbidden=choose_region(ranked_region.distance_to_forbidden),
        valid=choose_region(ranked_region.valid) & any_feasible,
    )
    chosen_sweep = _select_leg_retry(
        _gather_ranked_candidates(
            perceptive_plan.valid_components["sweep"], ranked_index
        ),
        first_feasible,
    )

    preview_ranked_index = perceptive_plan.preview_ranked_index
    preview_ranked_candidate = _gather_ranked_candidates(
        perceptive_plan.preview_candidate_w, preview_ranked_index
    )
    preview_ranked_region = _ranked_regions(perceptive_plan, preview=True)
    preview_ranked_target = torch.cat(
        (
            preview_ranked_candidate[..., :2],
            preview_ranked_region.plane[..., :1]
            + float(cfg.gait.foot_contact_offset),
        ),
        dim=-1,
    )
    preview_selector_valid = _gather_ranked_candidates(
        perceptive_plan.preview_safe_mask, preview_ranked_index
    )
    preview_region_margin = (
        torch.einsum(
            "brlci,brli->brlc",
            preview_ranked_region.A,
            preview_ranked_target[..., :2],
        )
        + preview_ranked_region.b
    )
    preview_ranked_valid = (
        preview_selector_valid
        & preview_ranked_region.valid
        & (preview_region_margin >= -1.0e-6).all(dim=-1)
        & torch.isfinite(preview_ranked_target).all(dim=-1)
    )
    preview_any_feasible = preview_ranked_valid.any(dim=1)
    preview_first_feasible = preview_ranked_valid.to(torch.int64).argmax(dim=1)
    preview_chosen_index = _select_leg_retry(
        preview_ranked_index.permute(0, 2, 1), preview_first_feasible
    )
    preview_chosen_target = _select_leg_retry(
        preview_ranked_target, preview_first_feasible
    )

    def choose_preview_region(value: Tensor) -> Tensor:
        return _select_leg_retry(value, preview_first_feasible)

    preview_chosen_region = JointMpcTouchdownRegion(
        A=choose_preview_region(preview_ranked_region.A),
        b=choose_preview_region(preview_ranked_region.b),
        half_extent=choose_preview_region(preview_ranked_region.half_extent),
        corners_w=choose_preview_region(preview_ranked_region.corners_w),
        plane=choose_preview_region(preview_ranked_region.plane),
        normal_w=choose_preview_region(preview_ranked_region.normal_w),
        plane_residual=choose_preview_region(preview_ranked_region.plane_residual),
        area=choose_preview_region(preview_ranked_region.area),
        distance_to_forbidden=choose_preview_region(
            preview_ranked_region.distance_to_forbidden
        ),
        valid=choose_preview_region(preview_ranked_region.valid)
        & preview_any_feasible,
    )
    preview_chosen_sweep = _select_leg_retry(
        _gather_ranked_candidates(
            perceptive_plan.preview_valid_components["sweep"],
            preview_ranked_index,
        ),
        preview_first_feasible,
    )
    effective_plan = replace(
        perceptive_plan,
        selected_index=chosen_index,
        target_w=chosen_target,
        selected_sweep_safe=chosen_sweep,
        valid=any_feasible,
        region=chosen_region,
        preview_selected_index=preview_chosen_index,
        preview_target_w=preview_chosen_target,
        preview_selected_sweep_safe=preview_chosen_sweep,
        preview_valid=preview_any_feasible,
        preview_region=preview_chosen_region,
    )
    selected_plan = _SelectedTouchdownPlan(
        target_w=chosen_target,
        event_step=perceptive_plan.event_step,
        preview_touchdown_step=perceptive_plan.preview_touchdown_step,
        valid=any_feasible,
        region=chosen_region,
        preview_target_w=preview_chosen_target,
        preview_valid=preview_any_feasible,
        preview_region=preview_chosen_region,
    )
    trial = _build_nominal_once(
        measured,
        command_body,
        terrain_field,
        gait_phase,
        perceptive_plan=selected_plan,
        previous=previous,
        cfg=cfg,
    )
    retry_rank = torch.where(
        any_feasible, first_feasible, first_feasible.new_full((), -1)
    )
    preview_retry_rank = torch.where(
        preview_any_feasible,
        preview_first_feasible,
        preview_first_feasible.new_full((), -1),
    )
    return replace(
        trial,
        candidate_retry_rank=retry_rank,
        preview_candidate_retry_rank=preview_retry_rank,
        perceptive_plan=effective_plan,
    )


__all__ = [
    "NominalTrajectory",
    "WarmStartInvariantError",
    "build_nominal",
    "build_rebased_seed",
]
