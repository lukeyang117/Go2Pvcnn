"""Pure-tensor one-step rolling planner API."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.barriers import (
    localized_relaxed_barrier_derivative,
    relaxed_barrier_derivative,
)
from extension.joint_mpc_rti.losses.rollout_objective import rollout_loss_breakdown
from extension.joint_mpc_rti.model.dynamics import kinematic_step
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import foot_jacobian_joint
from extension.joint_mpc_rti.model.rollout import JointMpcRollout, rollout_controls
from extension.joint_mpc_rti.solver.linearization import dynamics_jacobians
from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem
from extension.joint_mpc_rti.solver.sqp_rti import sqp_rti_update
from extension.joint_mpc_rti.terrain.query import query_world
from extension.joint_mpc_rti.types import (
    JointMpcPendingReference,
    JointMpcRtiSolverState,
    JointMpcRtiState,
    JointMpcRtiStepResult,
    JointMpcRtiTrajectory,
    JointMpcTerrainField,
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


def _repeat_field(field: JointMpcTerrainField, repeats: int) -> JointMpcTerrainField:
    def repeat(tensor: Tensor) -> Tensor:
        return tensor[:, None].expand(-1, repeats, *tensor.shape[1:]).reshape(
            tensor.shape[0] * repeats, *tensor.shape[1:]
        )

    return JointMpcTerrainField(
        height_w=repeat(field.height_w),
        semantic_id=repeat(field.semantic_id),
        small_distance_m=repeat(field.small_distance_m),
        large_distance_m=repeat(field.large_distance_m),
        small_gradient_xy=repeat(field.small_gradient_xy),
        large_gradient_xy=repeat(field.large_gradient_xy),
        valid_mask=repeat(field.valid_mask),
        origin_w=repeat(field.origin_w),
        yaw_w=repeat(field.yaw_w),
        timestamp=repeat(field.timestamp),
        version=repeat(field.version),
        resolution=field.resolution,
    )


def _repeat_rollout(rollout: JointMpcRollout, repeats: int) -> JointMpcRollout:
    def repeat(tensor: Tensor) -> Tensor:
        return tensor[:, None].expand(-1, repeats, *tensor.shape[1:]).reshape(
            tensor.shape[0] * repeats, *tensor.shape[1:]
        )

    return JointMpcRollout(
        state=repeat(rollout.state),
        control=repeat(rollout.control),
        foot_pos_w=repeat(rollout.foot_pos_w),
        knee_pos_w=repeat(rollout.knee_pos_w),
        shank_samples_w=repeat(rollout.shank_samples_w),
        body_samples_w=repeat(rollout.body_samples_w),
    )


def _nominal_joint_target(
    contact: Tensor,
    phase_step: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    dtype: torch.dtype,
) -> Tensor:
    batch, nodes = int(contact.shape[0]), int(contact.shape[1])
    nominal = torch.tensor(cfg.gait.nominal_joint_pos, dtype=dtype, device=contact.device).view(1, 1, 4, 3)
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


def _desired_control(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    contact: Tensor,
    phase_step: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    batch = measured_state.batch_size
    horizon = int(cfg.runtime.horizon_steps)
    joint_target = _nominal_joint_target(contact, phase_step, cfg, dtype=measured_state.root_pos_w.dtype)
    joint_velocity = (joint_target[:, 1:] - joint_target[:, :-1]) / float(cfg.runtime.dt)
    joint_velocity = joint_velocity.clamp(
        -float(cfg.gait.max_nominal_joint_velocity), float(cfg.gait.max_nominal_joint_velocity)
    )
    desired = torch.zeros(batch, horizon, 18, dtype=measured_state.root_pos_w.dtype, device=measured_state.device)
    desired[..., :2] = command_body[:, None, :2]
    desired[..., 5] = command_body[:, None, 2]
    desired[..., 6:] = joint_velocity
    return desired, joint_target


def _initial_control(
    desired_control: Tensor,
    solver_state: JointMpcRtiSolverState | None,
) -> Tensor:
    if solver_state is None:
        return desired_control.clone()
    previous = torch.as_tensor(solver_state.control, dtype=desired_control.dtype, device=desired_control.device)
    if previous.shape != desired_control.shape:
        return desired_control.clone()
    return torch.cat((previous[:, 1:], previous[:, -1:]), dim=1)


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
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    matrix_q = problem.matrix_q.clone()
    vector_q = problem.vector_q.clone()
    terminal_q = problem.terminal_q.clone()
    terminal_vector = problem.terminal_vector.clone()
    relaxation = float(cfg.solver.barrier_relaxation)
    trust_scale = float(cfg.solver.root_xy_trust_scale)

    def add_geometry_gradient(position_w: Tensor, *, margin: float, weight: float) -> None:
        query = query_world(terrain_field, position_w.reshape(batch, -1, 3))
        point_count = int(position_w.reshape(batch, nodes, -1, 3).shape[2])
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
        rollout.body_samples_w,
        margin=0.12,
        weight=float(cfg.losses.large_root_footprint_barrier),
    )
    add_geometry_gradient(
        rollout.body_samples_w,
        margin=0.08,
        weight=float(cfg.losses.large_body_collision),
    )
    add_geometry_gradient(
        rollout.foot_pos_w,
        margin=0.03,
        weight=float(cfg.losses.large_foot_collision),
    )
    link_position = torch.cat(
        (rollout.knee_pos_w, rollout.shank_samples_w.reshape(batch, nodes, -1, 3)),
        dim=2,
    )
    add_geometry_gradient(
        link_position,
        margin=0.04,
        weight=float(cfg.losses.large_knee_shank_collision),
    )
    root_query = query_world(terrain_field, rollout.state[..., :3])
    terminal_derivative = relaxed_barrier_derivative(
        root_query.large_distance_m[:, -1] - 0.16,
        relaxation=relaxation,
    )
    terminal_gradient = (
        float(cfg.losses.large_terminal_risk)
        * terminal_derivative.unsqueeze(-1)
        * root_query.large_gradient_w[:, -1]
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
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    foot = rollout.foot_pos_w
    query = query_world(terrain_field, foot.reshape(batch, -1, 3))
    height = query.height_w.reshape(batch, nodes, 4)
    small_distance = query.small_distance_m.reshape(batch, nodes, 4)
    small_gradient = query.small_gradient_w.reshape(batch, nodes, 4, 2)
    large_distance = query.large_distance_m.reshape(batch, nodes, 4)
    large_gradient = query.large_gradient_w.reshape(batch, nodes, 4, 2)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    swing = torch.logical_not(contact)
    swing_weight = torch.as_tensor(swing_weight, dtype=foot.dtype, device=foot.device)
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
    small_influence = torch.sigmoid((0.08 - small_distance) / 0.02)
    small_over_derivative = localized_relaxed_barrier_derivative(
        foot[..., 2] - height - float(cfg.gait.small_semantic_clearance),
        activation_margin=0.005,
        relaxation=relaxation,
    )
    foot_gradient[..., 2].add_(
        float(cfg.losses.small_object_foot_over)
        * normalized_mask(swing_weight)
        * small_influence
        * small_over_derivative
    )
    stance_error = foot[..., 2] - height
    foot_gradient[..., 2].add_(
        float(cfg.losses.stance_ground_contact) * normalized_mask(contact) * 2.0 * stance_error
    )
    small_touchdown_derivative = relaxed_barrier_derivative(
        small_distance - 0.02,
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
    jacobian = foot_jacobian_joint(
        state_flat[:, :3],
        state_flat[:, 3:6],
        state_flat[:, 6:],
    ).reshape(batch, nodes, 4, 3, 12)
    joint_gradient = torch.einsum("btli,btliq->btq", foot_gradient, jacobian)
    matrix_q = problem.matrix_q.clone()
    vector_q = problem.vector_q.clone()
    terminal_q = problem.terminal_q.clone()
    terminal_vector = problem.terminal_vector.clone()
    trust_scale = float(cfg.solver.joint_trust_scale)
    vector_q[..., 6:].add_(joint_gradient[:, :-1])
    terminal_vector[..., 6:].add_(joint_gradient[:, -1])
    matrix_q[..., 6:, 6:].add_(torch.diag_embed(joint_gradient[:, :-1].abs() / trust_scale))
    terminal_q[..., 6:, 6:].add_(torch.diag_embed(joint_gradient[:, -1].abs() / trust_scale))
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
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    foot_query = query_world(terrain_field, rollout.foot_pos_w.reshape(batch, -1, 3))
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
    desired_control, joint_target = _desired_control(measured_state, command, contact, phase_step, cfg)
    base_control = _initial_control(desired_control, solver_state)
    base_rollout = rollout_controls(measured_state, base_control, dt=float(cfg.runtime.dt))
    nominal_rollout = rollout_controls(measured_state, desired_control, dt=float(cfg.runtime.dt))
    lq_problem = _build_lq_problem(base_rollout, desired_control, joint_target, measured_state, cfg)
    lq_problem = _add_large_obstacle_linearization(lq_problem, base_rollout, terrain_field, cfg)
    lq_problem = _add_foot_terrain_linearization(
        lq_problem,
        base_rollout,
        contact,
        swing_weight,
        terrain_field,
        cfg,
    )
    lq_problem = _add_root_support_linearization(lq_problem, base_rollout, contact, terrain_field, cfg)
    previous_control = (
        measured_state.joint_vel.new_zeros((measured_state.batch_size, 18))
        if solver_state is None
        else solver_state.previous_control
    )

    def merit_fn(candidate_control: Tensor) -> Tensor:
        repeats = int(candidate_control.shape[0]) // measured_state.batch_size
        repeated_state = _repeat_state(measured_state, repeats)
        repeated_command = command[:, None].expand(-1, repeats, -1).reshape(-1, 3)
        repeated_field = _repeat_field(terrain_field, repeats)
        repeated_nominal = _repeat_rollout(nominal_rollout, repeats)
        repeated_target = joint_target[:, None].expand(-1, repeats, -1, -1).reshape(
            measured_state.batch_size * repeats, joint_target.shape[1], 12
        )
        candidate_rollout = rollout_controls(repeated_state, candidate_control, dt=float(cfg.runtime.dt))
        _, total = rollout_loss_breakdown(
            rollout=candidate_rollout,
            nominal_rollout=repeated_nominal,
            contact_state=contact[:, None].expand(-1, repeats, -1, -1).reshape(
                measured_state.batch_size * repeats, *contact.shape[1:]
            ),
            swing_weight=swing_weight[:, None].expand(-1, repeats, -1, -1).reshape(
                measured_state.batch_size * repeats, *swing_weight.shape[1:]
            ),
            terrain_field=repeated_field,
            command_body=repeated_command,
            joint_target=repeated_target,
            previous_control=previous_control[:, None].expand(-1, repeats, -1).reshape(-1, 18),
            cfg=cfg,
        )
        return total

    update = sqp_rti_update(
        base_control=base_control,
        lq_problem=lq_problem,
        merit_fn=merit_fn,
        regularization=float(cfg.solver.regularization),
        alphas=tuple(cfg.solver.line_search_alphas),
    )
    rollout = rollout_controls(measured_state, update.control, dt=float(cfg.runtime.dt))
    final_losses, _ = rollout_loss_breakdown(
        rollout=rollout,
        nominal_rollout=nominal_rollout,
        contact_state=contact,
        swing_weight=swing_weight,
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
    )
    return JointMpcRtiStepResult(
        full_trajectory=trajectory,
        pending_reference=pending,
        solver_state=next_solver_state,
    )


__all__ = ["step"]
