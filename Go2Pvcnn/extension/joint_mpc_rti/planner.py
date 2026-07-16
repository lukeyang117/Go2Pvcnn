"""Pure-tensor one-step rolling planner API."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.barriers import (
    localized_relaxed_barrier_derivative,
    relaxed_barrier,
    relaxed_barrier_derivative,
)
from extension.joint_mpc_rti.losses.rollout_objective import rollout_loss_breakdown_maybe_compiled
from extension.joint_mpc_rti.model.dynamics import kinematic_step
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import foot_jacobian_leg, go2_foot_pos, link_sample_jacobians
from extension.joint_mpc_rti.model.rollout import JointMpcRollout, rollout_controls, rollout_state_sequence
from extension.joint_mpc_rti.solver.linearization import dynamics_jacobians
from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem
from extension.joint_mpc_rti.solver.sqp_rti import sqp_rti_update
from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery, query_world_maybe_compiled
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.types import (
    JointMpcPendingReference,
    JointMpcRtiSolverState,
    JointMpcRtiState,
    JointMpcRtiStepResult,
    JointMpcRtiTrajectory,
    JointMpcTerrainField,
)


def _query_world(field: JointMpcTerrainField, points_w: Tensor, cfg: JointMpcRtiCfg):
    return query_world_maybe_compiled(
        field,
        points_w,
        enabled=bool(cfg.solver.compile_kernels),
    )


@dataclass(frozen=True)
class _LinearizationQueries:
    body: JointMpcTerrainQuery
    foot: JointMpcTerrainQuery
    knee: JointMpcTerrainQuery
    shank: JointMpcTerrainQuery
    thigh: JointMpcTerrainQuery
    root: JointMpcTerrainQuery


def _query_linearization_geometry(
    rollout: JointMpcRollout,
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
) -> _LinearizationQueries:
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    shank = rollout.shank_samples_w.reshape(batch, nodes, 12, 3)
    thigh = rollout.thigh_samples_w.reshape(batch, nodes, 12, 3)
    root = rollout.state[..., :3].unsqueeze(2)
    packed = torch.cat(
        (rollout.body_samples_w, rollout.foot_pos_w, rollout.knee_pos_w, shank, thigh, root),
        dim=2,
    )
    points_per_node = int(packed.shape[2])
    queried = _query_world(terrain_field, packed.reshape(batch, nodes * points_per_node, 3), cfg)

    def section(start: int, stop: int) -> JointMpcTerrainQuery:
        def scalar(value: Tensor) -> Tensor:
            return value.reshape(batch, nodes, points_per_node)[:, :, start:stop]

        def vector(value: Tensor) -> Tensor:
            return value.reshape(batch, nodes, points_per_node, 2)[:, :, start:stop]

        return JointMpcTerrainQuery(
            height_w=scalar(queried.height_w),
            small_distance_m=scalar(queried.small_distance_m),
            large_distance_m=scalar(queried.large_distance_m),
            small_gradient_w=vector(queried.small_gradient_w),
            large_gradient_w=vector(queried.large_gradient_w),
            valid=scalar(queried.valid),
        )

    body_stop = int(rollout.body_samples_w.shape[2])
    foot_stop = body_stop + 4
    knee_stop = foot_stop + 4
    shank_stop = knee_stop + 12
    thigh_stop = shank_stop + 12
    return _LinearizationQueries(
        body=section(0, body_stop),
        foot=section(body_stop, foot_stop),
        knee=section(foot_stop, knee_stop),
        shank=section(knee_stop, shank_stop),
        thigh=section(shank_stop, thigh_stop),
        root=section(thigh_stop, thigh_stop + 1),
    )


def _repeat_state(state: JointMpcRtiState, repeats: int) -> JointMpcRtiState:
    def repeat(tensor: Tensor) -> Tensor:
        return tensor[:, None].expand(-1, repeats, *tensor.shape[1:]).reshape(
            tensor.shape[0] * repeats, *tensor.shape[1:]
        )

    return JointMpcRtiState(
        root_pos_w=repeat(state.root_pos_w),
        root_rpy_w=repeat(state.root_rpy_w),
        joint_pos=repeat(state.joint_pos),
        root_lin_vel_b=repeat(state.root_lin_vel_b),
        root_ang_vel_b=repeat(state.root_ang_vel_b),
        joint_vel=repeat(state.joint_vel),
    )


def _select_candidate_rollout(
    candidate_rollout: JointMpcRollout,
    base_rollout: JointMpcRollout,
    selected_index: Tensor,
    used_base: Tensor,
    candidate_count: int,
) -> JointMpcRollout:
    batch = int(base_rollout.state.shape[0])

    def select(candidate: Tensor, base: Tensor) -> Tensor:
        shaped = candidate.reshape(batch, candidate_count, *candidate.shape[1:])
        index = selected_index.view(batch, 1, *([1] * (candidate.ndim - 1))).expand(
            batch,
            1,
            *candidate.shape[1:],
        )
        chosen = torch.gather(shaped, 1, index).squeeze(1)
        mask = used_base.view(batch, *([1] * (base.ndim - 1)))
        return torch.where(mask, base, chosen)

    return JointMpcRollout(
        state=select(candidate_rollout.state, base_rollout.state),
        control=select(candidate_rollout.control, base_rollout.control),
        foot_pos_w=select(candidate_rollout.foot_pos_w, base_rollout.foot_pos_w),
        knee_pos_w=select(candidate_rollout.knee_pos_w, base_rollout.knee_pos_w),
        shank_samples_w=select(candidate_rollout.shank_samples_w, base_rollout.shank_samples_w),
        thigh_samples_w=select(candidate_rollout.thigh_samples_w, base_rollout.thigh_samples_w),
        body_samples_w=select(candidate_rollout.body_samples_w, base_rollout.body_samples_w),
    )


def _nominal_joint_target(
    contact: Tensor,
    phase_step: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    dtype: torch.dtype,
) -> Tensor:
    batch, nodes = int(contact.shape[0]), int(contact.shape[1])
    reference = contact.new_empty((), dtype=dtype)
    nominal = constant_like(reference, "nominal_joint_pos", cfg.gait.nominal_joint_pos).view(1, 1, 4, 3)
    target = nominal.expand(batch, nodes, 4, 3).clone()
    swing = torch.logical_not(contact)
    half_cycle = int(cfg.gait.half_cycle_steps)
    frame = torch.arange(nodes, device=contact.device).view(1, nodes)
    frame_in_half = torch.remainder(frame + phase_step[:, None], half_cycle).to(dtype=dtype)
    envelope = torch.sin(torch.pi * frame_in_half / float(half_cycle)).unsqueeze(-1)
    envelope = envelope * swing.to(dtype=dtype)
    target[..., 1] = target[..., 1] + envelope * (float(cfg.gait.swing_thigh_angle) - target[..., 1])
    target[..., 2] = target[..., 2] + envelope * (float(cfg.gait.swing_calf_angle) - target[..., 2])
    return target.reshape(batch, nodes, 12)


def _swing_phase_weight(contact: Tensor, phase_step: Tensor, cfg: JointMpcRtiCfg, *, dtype: torch.dtype) -> Tensor:
    nodes = int(contact.shape[1])
    half_cycle = int(cfg.gait.half_cycle_steps)
    frame = torch.arange(nodes, device=contact.device).view(1, nodes)
    frame_in_half = torch.remainder(frame + phase_step[:, None], half_cycle).to(dtype=dtype)
    return (
        torch.sin(torch.pi * frame_in_half / float(half_cycle)).unsqueeze(-1)
        * torch.logical_not(contact).to(dtype=dtype)
    )


def _small_swing_handoff_weights(
    contact: Tensor,
    phase_step: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    nodes = int(contact.shape[1])
    half_cycle = int(cfg.gait.half_cycle_steps)
    frame = torch.arange(nodes, device=contact.device).view(1, nodes)
    frame_in_half = torch.remainder(frame + phase_step[:, None], half_cycle).to(dtype=dtype)
    progress = frame_in_half / float(max(half_cycle - 1, 1))
    swing = torch.logical_not(contact).to(dtype=dtype)
    mid_swing = torch.sin(torch.pi * progress).clamp_min(0.0).pow(
        float(cfg.gait.small_foot_over_phase_exponent)
    )
    safe_landing = progress.pow(float(cfg.gait.small_safe_landing_phase_exponent))
    return mid_swing.unsqueeze(-1) * swing, safe_landing.unsqueeze(-1) * swing


def _stance_anchor_targets(
    nominal_foot_pos_w: Tensor,
    contact_state: Tensor,
    initial_anchor_w: Tensor | None = None,
) -> Tensor:
    """Hold one world-space foot anchor through each contiguous stance segment."""
    foot = torch.as_tensor(nominal_foot_pos_w)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    if foot.ndim != 4 or tuple(foot.shape[-2:]) != (4, 3):
        raise ValueError("nominal_foot_pos_w must have shape [B,T,4,3]")
    if contact.shape != foot.shape[:-1]:
        raise ValueError("contact_state must have shape [B,T,4]")
    if initial_anchor_w is None:
        current = foot[:, 0]
    else:
        initial = torch.as_tensor(initial_anchor_w, dtype=foot.dtype, device=foot.device)
        if initial.shape != foot[:, 0].shape:
            raise ValueError("initial_anchor_w must have shape [B,4,3]")
        current = torch.where(torch.isfinite(initial), initial, foot[:, 0])
    anchors: list[Tensor] = [current]
    for node in range(1, int(foot.shape[1])):
        touchdown = torch.logical_and(contact[:, node], torch.logical_not(contact[:, node - 1]))
        current = torch.where(touchdown.unsqueeze(-1), foot[:, node], current)
        anchors.append(torch.where(contact[:, node].unsqueeze(-1), current, foot[:, node]))
    return torch.stack(anchors, dim=1)


def _desired_control(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    contact: Tensor,
    phase_step: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    joint_target = _nominal_joint_target(contact, phase_step, cfg, dtype=measured_state.root_pos_w.dtype)
    return _control_from_joint_target(measured_state, command_body, joint_target, cfg), joint_target


def _control_from_joint_target(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    joint_target: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    batch = measured_state.batch_size
    horizon = int(cfg.runtime.horizon_steps)
    joint_velocity = (joint_target[:, 1:] - joint_target[:, :-1]) / float(cfg.runtime.dt)
    joint_velocity = joint_velocity.clamp(
        -float(cfg.gait.max_nominal_joint_velocity), float(cfg.gait.max_nominal_joint_velocity)
    )
    desired = torch.zeros(batch, horizon, 18, dtype=measured_state.root_pos_w.dtype, device=measured_state.device)
    desired[..., :2] = command_body[:, None, :2]
    desired[..., 5] = command_body[:, None, 2]
    desired[..., 6:] = joint_velocity
    return desired


def _initial_control(
    desired_control: Tensor,
    solver_state: JointMpcRtiSolverState | None,
) -> Tensor:
    if solver_state is None:
        return desired_control.clone()
    previous = torch.as_tensor(solver_state.control, dtype=desired_control.dtype, device=desired_control.device)
    if previous.shape != desired_control.shape:
        return desired_control.clone()
    shifted = torch.cat((previous[:, 1:], previous[:, -1:]), dim=1)
    return torch.cat((desired_control[..., :6], shifted[..., 6:]), dim=-1)


def _build_lq_problem(
    rollout: JointMpcRollout,
    desired_control: Tensor,
    joint_target: Tensor,
    measured_state: JointMpcRtiState,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    batch, horizon = int(rollout.control.shape[0]), int(rollout.control.shape[1])
    state_flat = rollout.state[:, :-1].reshape(batch * horizon, 18)
    control_flat = rollout.control.reshape(batch * horizon, 18)
    matrix_a, matrix_b = dynamics_jacobians(state_flat, control_flat, dt=float(cfg.runtime.dt))
    matrix_a = matrix_a.reshape(batch, horizon, 18, 18)
    matrix_b = matrix_b.reshape(batch, horizon, 18, 18)
    predicted_next = kinematic_step(state_flat, control_flat, dt=float(cfg.runtime.dt)).reshape(batch, horizon, 18)
    affine = predicted_next - rollout.state[:, 1:]
    state_weight = rollout.state.new_zeros((18,))
    state_weight[6:] = float(cfg.core_losses.joint_posture_weight)
    control_weight = rollout.state.new_full((18,), float(cfg.core_losses.joint_velocity_weight))
    control_weight[:6] = float(cfg.core_losses.command_control_weight)
    matrix_q = torch.diag(state_weight).view(1, 1, 18, 18).expand(batch, horizon, -1, -1).clone()
    matrix_r = torch.diag(control_weight).view(1, 1, 18, 18).expand(batch, horizon, -1, -1).clone()
    vector_q = matrix_q.diagonal(dim1=-2, dim2=-1) * (
        rollout.state[:, :-1] - torch.cat((rollout.state[:, :-1, :6], joint_target[:, :-1]), dim=-1)
    )
    vector_r = matrix_r.diagonal(dim1=-2, dim2=-1) * (rollout.control - desired_control)
    terminal_weight = rollout.state.new_zeros((18,))
    terminal_weight[6:] = float(cfg.core_losses.terminal_joint_posture_weight)
    terminal_q = torch.diag(terminal_weight).unsqueeze(0).expand(batch, -1, -1).clone()
    terminal_target = torch.cat((rollout.state[:, -1, :6], joint_target[:, -1]), dim=-1)
    terminal_vector = terminal_weight.unsqueeze(0) * (rollout.state[:, -1] - terminal_target)
    return LqProblem(
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        matrix_q=matrix_q,
        matrix_r=matrix_r,
        vector_q=vector_q,
        vector_r=vector_r,
        terminal_q=terminal_q,
        terminal_vector=terminal_vector,
        initial_state=torch.zeros(batch, 18, dtype=rollout.state.dtype, device=rollout.state.device),
        affine_dynamics=affine,
        matrix_s=torch.zeros(batch, horizon, 18, 18, dtype=rollout.state.dtype, device=rollout.state.device),
    )


def _add_large_obstacle_linearization(
    problem: LqProblem,
    rollout: JointMpcRollout,
    queries: _LinearizationQueries,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    matrix_q = problem.matrix_q.clone()
    vector_q = problem.vector_q.clone()
    terminal_q = problem.terminal_q.clone()
    terminal_vector = problem.terminal_vector.clone()
    relaxation = float(cfg.solver.barrier_relaxation)
    trust_scale = float(cfg.solver.root_xy_trust_scale)

    def add_geometry_gradient(query: JointMpcTerrainQuery, *, margin: float, weight: float) -> None:
        point_count = int(query.large_distance_m.shape[2])
        distance = query.large_distance_m.reshape(batch, nodes, point_count)
        gradient = query.large_gradient_w.reshape(batch, nodes, point_count, 2)
        derivative = relaxed_barrier_derivative(distance - float(margin), relaxation=relaxation)
        root_gradient = float(weight) * (derivative.unsqueeze(-1) * gradient).mean(dim=2) / float(nodes)
        vector_q[..., :2].add_(root_gradient[:, :-1])
        terminal_vector[..., :2].add_(root_gradient[:, -1])
        matrix_q[..., 0, 0].add_(root_gradient[:, :-1, 0].abs() / trust_scale)
        matrix_q[..., 1, 1].add_(root_gradient[:, :-1, 1].abs() / trust_scale)
        terminal_q[..., 0, 0].add_(root_gradient[:, -1, 0].abs() / trust_scale)
        terminal_q[..., 1, 1].add_(root_gradient[:, -1, 1].abs() / trust_scale)

    add_geometry_gradient(
        queries.body,
        margin=0.12,
        weight=float(cfg.losses.large_root_footprint_barrier),
    )
    add_geometry_gradient(
        queries.body,
        margin=0.08,
        weight=float(cfg.losses.large_body_collision),
    )
    add_geometry_gradient(
        queries.foot,
        margin=0.03,
        weight=float(cfg.losses.large_foot_collision),
    )
    link_query = JointMpcTerrainQuery(
        height_w=torch.cat((queries.knee.height_w, queries.shank.height_w), dim=2),
        small_distance_m=torch.cat((queries.knee.small_distance_m, queries.shank.small_distance_m), dim=2),
        large_distance_m=torch.cat((queries.knee.large_distance_m, queries.shank.large_distance_m), dim=2),
        small_gradient_w=torch.cat((queries.knee.small_gradient_w, queries.shank.small_gradient_w), dim=2),
        large_gradient_w=torch.cat((queries.knee.large_gradient_w, queries.shank.large_gradient_w), dim=2),
        valid=torch.cat((queries.knee.valid, queries.shank.valid), dim=2),
    )
    add_geometry_gradient(
        link_query,
        margin=0.04,
        weight=float(cfg.losses.large_knee_shank_collision),
    )
    root_large_distance = queries.root.large_distance_m.squeeze(-1)
    root_large_gradient = queries.root.large_gradient_w.squeeze(-2)
    terminal_derivative = relaxed_barrier_derivative(
        root_large_distance[:, -1] - 0.16,
        relaxation=relaxation,
    )
    terminal_gradient = (
        float(cfg.losses.large_terminal_risk)
        * terminal_derivative.unsqueeze(-1)
        * root_large_gradient[:, -1]
    )
    terminal_vector[:, :2].add_(terminal_gradient)
    terminal_q[..., 0, 0].add_(terminal_gradient[:, 0].abs() / trust_scale)
    terminal_q[..., 1, 1].add_(terminal_gradient[:, 1].abs() / trust_scale)
    return replace(
        problem,
        matrix_q=matrix_q,
        vector_q=vector_q,
        terminal_q=terminal_q,
        terminal_vector=terminal_vector,
    )


def _add_foot_terrain_linearization(
    problem: LqProblem,
    rollout: JointMpcRollout,
    contact_state: Tensor,
    swing_weight: Tensor,
    foot_over_weight: Tensor,
    safe_landing_weight: Tensor,
    stance_anchor_w: Tensor,
    foot_query: JointMpcTerrainQuery,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    foot = rollout.foot_pos_w
    height = foot_query.height_w.reshape(batch, nodes, 4)
    small_distance = foot_query.small_distance_m.reshape(batch, nodes, 4)
    small_gradient = foot_query.small_gradient_w.reshape(batch, nodes, 4, 2)
    large_distance = foot_query.large_distance_m.reshape(batch, nodes, 4)
    large_gradient = foot_query.large_gradient_w.reshape(batch, nodes, 4, 2)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    stance_anchor = torch.as_tensor(stance_anchor_w, dtype=foot.dtype, device=foot.device)
    if stance_anchor.shape != foot.shape:
        raise ValueError("stance_anchor_w must match rollout foot positions")
    swing = torch.logical_not(contact)
    swing_weight = torch.as_tensor(swing_weight, dtype=foot.dtype, device=foot.device)
    foot_over_weight = torch.as_tensor(foot_over_weight, dtype=foot.dtype, device=foot.device)
    safe_landing_weight = torch.as_tensor(safe_landing_weight, dtype=foot.dtype, device=foot.device)
    relaxation = float(cfg.solver.barrier_relaxation)
    foot_gradient = torch.zeros_like(foot)

    def normalized_mask(mask: Tensor) -> Tensor:
        return mask.to(foot.dtype) / mask.sum(dim=(1, 2), keepdim=True).clamp_min(1).to(foot.dtype)

    all_normalizer = 1.0 / float(nodes * 4)
    penetration_derivative = localized_relaxed_barrier_derivative(
        foot[..., 2] - height,
        activation_margin=0.005,
        relaxation=relaxation,
    )
    foot_gradient[..., 2].add_(
        float(cfg.losses.foot_ground_penetration) * all_normalizer * penetration_derivative
    )
    swing_derivative = localized_relaxed_barrier_derivative(
        foot[..., 2] - height - float(cfg.gait.nominal_swing_clearance),
        activation_margin=0.005,
        relaxation=relaxation,
    )
    foot_gradient[..., 2].add_(
        float(cfg.losses.terrain_swing_clearance) * normalized_mask(swing_weight) * swing_derivative
    )
    small_influence = torch.sigmoid(
        (float(cfg.gait.small_foot_over_influence_radius) - small_distance)
        / float(cfg.gait.small_collision_temperature)
    )
    small_effective_height = height + float(cfg.gait.small_semantic_height) * torch.sigmoid(
        small_distance / float(cfg.gait.small_collision_temperature)
    )
    small_over_derivative = localized_relaxed_barrier_derivative(
        foot[..., 2] - small_effective_height - float(cfg.gait.small_semantic_clearance),
        activation_margin=0.005,
        relaxation=relaxation,
    )
    foot_gradient[..., 2].add_(
        float(cfg.losses.small_object_foot_over)
        * normalized_mask(foot_over_weight)
        * small_influence
        * small_over_derivative
    )
    landing_safe_weight = safe_landing_weight * torch.sigmoid(
        (small_distance - float(cfg.gait.small_safe_landing_margin))
        / float(cfg.gait.small_collision_temperature)
    )
    landing_normalizer = normalized_mask(landing_safe_weight)
    landing_error = foot[..., 2] - height - float(cfg.gait.foot_contact_offset)
    foot_gradient[..., 2].add_(
        float(cfg.losses.small_object_safe_landing) * landing_normalizer * 2.0 * landing_error
    )
    stance_error = foot[..., 2] - height - float(cfg.gait.foot_contact_offset)
    stance_normalizer = normalized_mask(contact)
    stance_far_weight = torch.sigmoid(
        (
            small_distance.amin(dim=2)
            - float(cfg.gait.stance_ground_far_influence_radius)
        )
        / float(cfg.gait.stance_ground_far_temperature)
    )
    stance_ground_normalizer = stance_normalizer + float(cfg.losses.stance_ground_far_gain) * normalized_mask(
        contact.to(dtype=foot.dtype) * stance_far_weight.unsqueeze(-1)
    )
    foot_gradient[..., 2].add_(
        float(cfg.losses.stance_ground_contact) * stance_ground_normalizer * 2.0 * stance_error
    )
    support_epsilon = 1.0e-6
    support_safety = torch.sigmoid(
        (small_distance - float(cfg.gait.small_support_safety_margin))
        / float(cfg.gait.small_support_safety_temperature)
    ).pow(float(cfg.gait.small_support_safety_exponent))
    contact_float = contact.to(dtype=foot.dtype) * support_safety
    stance_error_sq = stance_error * stance_error
    support_count = contact_float.sum(dim=2, keepdim=True)
    support_inverse = (contact_float / (stance_error_sq + support_epsilon)).sum(dim=2, keepdim=True)
    support_error_derivative = (
        support_count
        * contact_float
        / support_inverse.clamp_min(1.0e-12).square()
        / (stance_error_sq + support_epsilon).square()
    )
    support_node_weight = float(cfg.losses.stance_support_viability) / float(nodes)
    foot_gradient[..., 2].add_(
        support_node_weight * support_error_derivative * 2.0 * stance_error
    )
    stance_xy_error = foot[..., :2] - stance_anchor[..., :2]
    foot_gradient[..., :2].add_(
        float(cfg.losses.stance_xy_lock) * stance_normalizer.unsqueeze(-1) * 2.0 * stance_xy_error
    )
    small_touchdown_derivative = relaxed_barrier_derivative(
        small_distance - float(cfg.gait.small_touchdown_margin),
        relaxation=relaxation,
    )
    small_xy_gradient = (
        float(cfg.losses.small_object_touchdown_avoidance)
        * normalized_mask(contact)
        * small_touchdown_derivative
    ).unsqueeze(-1) * small_gradient
    foot_gradient[..., :2].add_(small_xy_gradient)
    large_derivative = relaxed_barrier_derivative(large_distance - 0.03, relaxation=relaxation)
    foot_gradient[..., :2].add_(
        float(cfg.losses.large_foot_collision)
        * all_normalizer
        * large_derivative.unsqueeze(-1)
        * large_gradient
    )

    state_flat = rollout.state.reshape(batch * nodes, 18)
    jacobian = foot_jacobian_leg(
        state_flat[:, :3],
        state_flat[:, 3:6],
        state_flat[:, 6:],
    ).reshape(batch, nodes, 4, 3, 3)
    joint_gradient = (foot_gradient.unsqueeze(-1) * jacobian).sum(dim=-2).reshape(batch, nodes, 12)
    stance_axis_weight = torch.stack(
        (
            float(cfg.losses.stance_xy_lock) * stance_normalizer,
            float(cfg.losses.stance_xy_lock) * stance_normalizer,
            float(cfg.losses.stance_ground_contact) * stance_ground_normalizer
            + float(cfg.losses.small_object_safe_landing) * landing_normalizer
            + support_node_weight * support_error_derivative,
        ),
        dim=-1,
    )
    joint_curvature = (
        2.0 * (stance_axis_weight.unsqueeze(-1) * (jacobian * jacobian)).sum(dim=-2)
    ).reshape(batch, nodes, 12)
    matrix_q = problem.matrix_q.clone()
    vector_q = problem.vector_q.clone()
    terminal_q = problem.terminal_q.clone()
    terminal_vector = problem.terminal_vector.clone()
    trust_scale = float(cfg.solver.joint_trust_scale)
    vector_q[..., 6:].add_(joint_gradient[:, :-1])
    terminal_vector[..., 6:].add_(joint_gradient[:, -1])
    matrix_q[..., 6:, 6:].add_(torch.diag_embed(joint_gradient[:, :-1].abs() / trust_scale))
    matrix_q[..., 6:, 6:].add_(torch.diag_embed(joint_curvature[:, :-1]))
    terminal_q[..., 6:, 6:].add_(torch.diag_embed(joint_gradient[:, -1].abs() / trust_scale))
    terminal_q[..., 6:, 6:].add_(torch.diag_embed(joint_curvature[:, -1]))
    return replace(
        problem,
        matrix_q=matrix_q,
        vector_q=vector_q,
        terminal_q=terminal_q,
        terminal_vector=terminal_vector,
    )


def _add_small_obstacle_linearization(
    problem: LqProblem,
    rollout: JointMpcRollout,
    queries: _LinearizationQueries,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    """Add signed-distance foot/calf/thigh gradients to the RTI LQ direction."""
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    state_flat = rollout.state.reshape(batch * nodes, 18)
    foot_jacobian = foot_jacobian_leg(
        state_flat[:, :3],
        state_flat[:, 3:6],
        state_flat[:, 6:],
    ).reshape(batch, nodes, 4, 1, 3, 3)
    link_jacobian = link_sample_jacobians(
        state_flat[:, :3],
        state_flat[:, 3:6],
        state_flat[:, 6:],
    )
    calf_jacobian = link_jacobian.calf_samples.reshape(batch, nodes, 4, 3, 3, 3)
    thigh_jacobian = link_jacobian.thigh_samples.reshape(batch, nodes, 4, 3, 3, 3)
    matrix_q = problem.matrix_q.clone()
    vector_q = problem.vector_q.clone()
    terminal_q = problem.terminal_q.clone()
    terminal_vector = problem.terminal_vector.clone()
    relaxation = float(cfg.solver.barrier_relaxation)
    temperature = float(cfg.gait.small_collision_temperature)
    influence_radius = float(cfg.gait.small_collision_influence_radius)
    margin_xy = float(cfg.gait.small_collision_margin_xy)
    margin_z = float(cfg.gait.small_collision_margin_z)
    joint_trust = float(cfg.solver.joint_trust_scale)
    root_trust = float(cfg.solver.root_xy_trust_scale)

    def add_part(
        positions: Tensor,
        query: JointMpcTerrainQuery,
        jacobian: Tensor,
        *,
        radius: float,
        weight: float,
    ) -> None:
        if float(weight) == 0.0:
            return
        position = positions.reshape(batch, nodes, 4, -1, 3)
        sample_count = int(position.shape[3])
        distance = query.small_distance_m.reshape(batch, nodes, 4, sample_count)
        height = query.height_w.reshape(batch, nodes, 4, sample_count)
        sdf_gradient = query.small_gradient_w.reshape(batch, nodes, 4, sample_count, 2)
        proximity = torch.sigmoid((influence_radius - distance) / temperature)
        normalizer = proximity.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        effective_height = height + float(cfg.gait.small_semantic_height) * torch.sigmoid(
            distance / temperature
        )
        vertical = torch.sigmoid((effective_height + margin_z - position[..., 2]) / temperature)
        clearance = distance - float(radius) - margin_xy
        barrier = relaxed_barrier(clearance, relaxation=relaxation)
        derivative = relaxed_barrier_derivative(clearance, relaxation=relaxation)
        factor = float(weight) * proximity / normalizer
        gradient_xy = (
            float(cfg.gait.small_collision_link_xy_scale)
            * factor.unsqueeze(-1)
            * vertical.unsqueeze(-1)
            * derivative.unsqueeze(-1)
            * sdf_gradient
        )
        vertical_derivative = -vertical * (1.0 - vertical) / temperature
        gradient_z = factor * vertical_derivative * barrier
        point_gradient = torch.cat((gradient_xy, gradient_z.unsqueeze(-1)), dim=-1)
        joint_gradient = torch.einsum("bnlsd,bnlsdj->bnlj", point_gradient, jacobian).reshape(
            batch, nodes, 12
        )
        root_gradient = (
            float(cfg.gait.small_collision_root_xy_scale)
            * point_gradient[..., :2].sum(dim=(2, 3))
        )
        vector_q[..., :2].add_(root_gradient[:, :-1])
        vector_q[..., 6:].add_(joint_gradient[:, :-1])
        terminal_vector[..., :2].add_(root_gradient[:, -1])
        terminal_vector[..., 6:].add_(joint_gradient[:, -1])
        matrix_q[..., 0, 0].add_(root_gradient[:, :-1, 0].abs() / root_trust)
        matrix_q[..., 1, 1].add_(root_gradient[:, :-1, 1].abs() / root_trust)
        matrix_q[..., 6:, 6:].add_(torch.diag_embed(joint_gradient[:, :-1].abs() / joint_trust))
        terminal_q[..., 0, 0].add_(root_gradient[:, -1, 0].abs() / root_trust)
        terminal_q[..., 1, 1].add_(root_gradient[:, -1, 1].abs() / root_trust)
        terminal_q[..., 6:, 6:].add_(torch.diag_embed(joint_gradient[:, -1].abs() / joint_trust))

    add_part(
        rollout.foot_pos_w.unsqueeze(3),
        queries.foot,
        foot_jacobian,
        radius=cfg.gait.foot_collision_radius,
        weight=cfg.losses.small_object_foot_clearance,
    )
    add_part(
        rollout.shank_samples_w,
        queries.shank,
        calf_jacobian,
        radius=cfg.gait.calf_collision_radius,
        weight=cfg.losses.small_object_calf_clearance,
    )
    add_part(
        rollout.thigh_samples_w,
        queries.thigh,
        thigh_jacobian,
        radius=cfg.gait.thigh_collision_radius,
        weight=cfg.losses.small_object_thigh_clearance,
    )
    return replace(
        problem,
        matrix_q=matrix_q,
        vector_q=vector_q,
        terminal_q=terminal_q,
        terminal_vector=terminal_vector,
    )


def _add_root_support_linearization(
    problem: LqProblem,
    rollout: JointMpcRollout,
    contact_state: Tensor,
    foot_query: JointMpcTerrainQuery,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    foot_height = foot_query.height_w.reshape(batch, nodes, 4)
    contact = torch.as_tensor(contact_state, dtype=rollout.state.dtype, device=rollout.state.device)
    support_height = (foot_height * contact).sum(dim=2) / contact.sum(dim=2).clamp_min(1.0)
    error = rollout.state[..., 2] - support_height - 0.32
    weight = float(cfg.losses.root_support_height) / float(nodes)
    vector_q = problem.vector_q.clone()
    matrix_q = problem.matrix_q.clone()
    vector_q[..., 2].add_(2.0 * weight * error[:, :-1])
    matrix_q[..., 2, 2].add_(2.0 * weight)
    return replace(problem, matrix_q=matrix_q, vector_q=vector_q)


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
    phase_step = (
        torch.zeros(measured_state.batch_size, dtype=torch.long, device=measured_state.device)
        if solver_state is None or solver_state.gait_phase is None
        else torch.as_tensor(solver_state.gait_phase, dtype=torch.long, device=measured_state.device)
    )
    contact = fixed_trot_schedule(
        measured_state.batch_size,
        int(cfg.runtime.horizon_steps),
        measured_state.device,
        half_cycle_steps=int(cfg.gait.half_cycle_steps),
        phase_offset_steps=phase_step,
    )
    swing_weight = _swing_phase_weight(contact, phase_step, cfg, dtype=measured_state.root_pos_w.dtype)
    foot_over_weight, safe_landing_weight = _small_swing_handoff_weights(
        contact,
        phase_step,
        cfg,
        dtype=measured_state.root_pos_w.dtype,
    )
    desired_control, joint_target = _desired_control(measured_state, command, contact, phase_step, cfg)
    nominal_state = rollout_state_sequence(
        measured_state,
        desired_control,
        dt=float(cfg.runtime.dt),
        compile_kernels=bool(cfg.solver.compile_kernels),
    )
    batch, nodes = int(nominal_state.shape[0]), int(nominal_state.shape[1])
    nominal_foot_pos_w = go2_foot_pos(
        nominal_state[..., :3].reshape(batch * nodes, 3),
        nominal_state[..., 3:6].reshape(batch * nodes, 3),
        nominal_state[..., 6:].reshape(batch * nodes, 12),
    ).reshape(batch, nodes, 4, 3)
    measured_foot_w = go2_foot_pos(
        measured_state.root_pos_w,
        measured_state.root_rpy_w,
        measured_state.joint_pos,
    )
    previous_stance_anchor = (
        measured_foot_w
        if solver_state is None or solver_state.stance_anchor_w is None
        else torch.as_tensor(
            solver_state.stance_anchor_w,
            dtype=measured_foot_w.dtype,
            device=measured_foot_w.device,
        )
    )
    stance_anchor_w = _stance_anchor_targets(
        nominal_foot_pos_w,
        contact,
        initial_anchor_w=previous_stance_anchor,
    )
    base_control = _initial_control(desired_control, solver_state)
    base_rollout = rollout_controls(
        measured_state,
        base_control,
        dt=float(cfg.runtime.dt),
        compile_kernels=bool(cfg.solver.compile_kernels),
    )
    lq_problem = _build_lq_problem(base_rollout, desired_control, joint_target, measured_state, cfg)
    linearization_queries = _query_linearization_geometry(base_rollout, terrain_field, cfg)
    lq_problem = _add_large_obstacle_linearization(lq_problem, base_rollout, linearization_queries, cfg)
    lq_problem = _add_small_obstacle_linearization(lq_problem, base_rollout, linearization_queries, cfg)
    lq_problem = _add_foot_terrain_linearization(
        lq_problem,
        base_rollout,
        contact,
        swing_weight,
        foot_over_weight,
        safe_landing_weight,
        stance_anchor_w,
        linearization_queries.foot,
        cfg,
    )
    lq_problem = _add_root_support_linearization(lq_problem, base_rollout, contact, linearization_queries.foot, cfg)
    previous_control = (
        measured_state.joint_vel.new_zeros((measured_state.batch_size, 18))
        if solver_state is None
        else solver_state.previous_control
    )

    candidate_rollout_cache: list[JointMpcRollout] = []

    def evaluate_rollout(candidate_rollout: JointMpcRollout, repeats: int) -> Tensor:
        repeated_command = command[:, None].expand(-1, repeats, -1).reshape(-1, 3)
        repeated_nominal_foot = nominal_foot_pos_w[:, None].expand(-1, repeats, -1, -1, -1).reshape(
            measured_state.batch_size * repeats,
            nominal_foot_pos_w.shape[1],
            4,
            3,
        )
        repeated_target = joint_target[:, None].expand(-1, repeats, -1, -1).reshape(
            measured_state.batch_size * repeats, joint_target.shape[1], 12
        )
        repeated_stance_anchor = stance_anchor_w[:, None].expand(-1, repeats, -1, -1, -1).reshape(
            measured_state.batch_size * repeats,
            stance_anchor_w.shape[1],
            4,
            3,
        )
        _, total = rollout_loss_breakdown_maybe_compiled(
            rollout=candidate_rollout,
            nominal_foot_pos_w=repeated_nominal_foot,
            stance_anchor_w=repeated_stance_anchor,
            contact_state=contact[:, None].expand(-1, repeats, -1, -1).reshape(
                measured_state.batch_size * repeats, *contact.shape[1:]
            ),
            swing_weight=swing_weight[:, None].expand(-1, repeats, -1, -1).reshape(
                measured_state.batch_size * repeats, *swing_weight.shape[1:]
            ),
            foot_over_weight=foot_over_weight[:, None].expand(-1, repeats, -1, -1).reshape(
                measured_state.batch_size * repeats, *foot_over_weight.shape[1:]
            ),
            safe_landing_weight=safe_landing_weight[:, None].expand(-1, repeats, -1, -1).reshape(
                measured_state.batch_size * repeats, *safe_landing_weight.shape[1:]
            ),
            terrain_field=terrain_field,
            command_body=repeated_command,
            joint_target=repeated_target,
            previous_control=previous_control[:, None].expand(-1, repeats, -1).reshape(-1, 18),
            cfg=cfg,
        )
        return total

    base_merit = evaluate_rollout(base_rollout, 1)

    def merit_fn(candidate_control: Tensor) -> Tensor:
        repeats = int(candidate_control.shape[0]) // measured_state.batch_size
        repeated_state = _repeat_state(measured_state, repeats)
        candidate_rollout = rollout_controls(
            repeated_state,
            candidate_control,
            dt=float(cfg.runtime.dt),
            compile_kernels=bool(cfg.solver.compile_kernels),
        )
        candidate_rollout_cache.append(candidate_rollout)
        return evaluate_rollout(candidate_rollout, repeats)

    update = sqp_rti_update(
        base_control=base_control,
        lq_problem=lq_problem,
        merit_fn=merit_fn,
        regularization=float(cfg.solver.regularization),
        alphas=tuple(cfg.solver.line_search_alphas),
        diagonal_state_riccati=bool(cfg.solver.diagonal_state_riccati),
        base_merit=base_merit,
    )
    rollout = _select_candidate_rollout(
        candidate_rollout_cache[0],
        base_rollout,
        update.selected_index,
        update.used_base,
        len(cfg.solver.line_search_alphas),
    )
    final_losses = {}
    if bool(cfg.solver.emit_loss_breakdown):
        final_losses, _ = rollout_loss_breakdown_maybe_compiled(
            rollout=rollout,
            nominal_foot_pos_w=nominal_foot_pos_w,
            stance_anchor_w=stance_anchor_w,
            contact_state=contact,
            swing_weight=swing_weight,
            foot_over_weight=foot_over_weight,
            safe_landing_weight=safe_landing_weight,
            terrain_field=terrain_field,
            command_body=command,
            joint_target=joint_target,
            previous_control=previous_control,
            cfg=cfg,
        )
    finite = torch.isfinite(rollout.state).all(dim=(1, 2)) & torch.isfinite(rollout.control).all(dim=(1, 2))
    status = torch.where(finite, torch.zeros_like(finite, dtype=torch.long), torch.ones_like(finite, dtype=torch.long))
    trajectory = JointMpcRtiTrajectory(
        state=rollout.state,
        control=rollout.control,
        foot_pos_w=rollout.foot_pos_w,
        contact_state=contact,
        valid=finite,
        fallback=torch.logical_not(finite),
        status=status,
        loss_breakdown={
            **final_losses,
            "merit_before": update.merit_before,
            "merit_after": update.merit_after,
            "line_search_alpha": update.alpha,
        },
    )
    pending = JointMpcPendingReference(
        root_pos_w=rollout.state[:, 1, :3],
        root_rpy_w=rollout.state[:, 1, 3:6],
        joint_angles=rollout.state[:, 1, 6:],
        foot_pos_w=rollout.foot_pos_w[:, 1],
        contact_state=contact[:, 1],
        valid=finite,
        target_step=1,
    )
    next_solver_state = JointMpcRtiSolverState(
        state=rollout.state,
        control=rollout.control,
        dual=update.lq_solution.dual,
        previous_control=rollout.control[:, 0],
        gait_phase=torch.remainder(phase_step + 1, 2 * int(cfg.gait.half_cycle_steps)),
        stance_anchor_w=stance_anchor_w[:, 1],
    )
    return JointMpcRtiStepResult(
        full_trajectory=trajectory,
        pending_reference=pending,
        solver_state=next_solver_state,
    )


__all__ = ["step"]
