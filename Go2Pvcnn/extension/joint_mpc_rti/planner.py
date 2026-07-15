"""Pure-tensor one-step rolling planner API."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.command import command_losses
from extension.joint_mpc_rti.losses.smoothness import smoothness_losses
from extension.joint_mpc_rti.model.dynamics import kinematic_step
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.rollout import JointMpcRollout, rollout_controls
from extension.joint_mpc_rti.solver.linearization import dynamics_jacobians
from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem
from extension.joint_mpc_rti.solver.sqp_rti import sqp_rti_update
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


def _nominal_joint_target(contact: Tensor, cfg: JointMpcRtiCfg, *, dtype: torch.dtype) -> Tensor:
    batch, nodes = int(contact.shape[0]), int(contact.shape[1])
    nominal = torch.tensor(cfg.gait.nominal_joint_pos, dtype=dtype, device=contact.device).view(1, 1, 4, 3)
    target = nominal.expand(batch, nodes, 4, 3).clone()
    swing = torch.logical_not(contact)
    target[..., 1] = torch.where(swing, target.new_tensor(cfg.gait.swing_thigh_angle), target[..., 1])
    target[..., 2] = torch.where(swing, target.new_tensor(cfg.gait.swing_calf_angle), target[..., 2])
    return target.reshape(batch, nodes, 12)


def _desired_control(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    contact: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    batch = measured_state.batch_size
    horizon = int(cfg.runtime.horizon_steps)
    joint_target = _nominal_joint_target(contact, cfg, dtype=measured_state.root_pos_w.dtype)
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
    contact = fixed_trot_schedule(
        measured_state.batch_size,
        int(cfg.runtime.horizon_steps),
        measured_state.device,
        half_cycle_steps=int(cfg.gait.half_cycle_steps),
    )
    desired_control, joint_target = _desired_control(measured_state, command, contact, cfg)
    base_control = _initial_control(desired_control, solver_state)
    base_rollout = rollout_controls(measured_state, base_control, dt=float(cfg.runtime.dt))
    lq_problem = _build_lq_problem(base_rollout, desired_control, joint_target, measured_state, cfg)
    alpha_count = len(cfg.solver.line_search_alphas)

    def merit_fn(candidate_control: Tensor) -> Tensor:
        repeats = int(candidate_control.shape[0]) // measured_state.batch_size
        repeated_state = _repeat_state(measured_state, repeats)
        repeated_command = command[:, None].expand(-1, repeats, -1).reshape(-1, 3)
        repeated_target = joint_target[:, None].expand(-1, repeats, -1, -1).reshape(
            measured_state.batch_size * repeats, joint_target.shape[1], 12
        )
        candidate_rollout = rollout_controls(repeated_state, candidate_control, dt=float(cfg.runtime.dt))
        command_terms = command_losses(
            candidate_rollout.state[..., :3],
            candidate_rollout.state[..., 3:6],
            candidate_control,
            repeated_command,
            dt=float(cfg.runtime.dt),
        )
        posture = ((candidate_rollout.state[..., 6:] - repeated_target) ** 2).mean(dim=(1, 2))
        smooth = smoothness_losses(
            candidate_control,
            previous_control=(
                repeated_state.joint_vel.new_zeros((candidate_control.shape[0], 18))
                if solver_state is None
                else solver_state.previous_control[:, None]
                .expand(-1, repeats, -1)
                .reshape(candidate_control.shape[0], 18)
            ),
            dt=float(cfg.runtime.dt),
        )
        return (
            command_terms["command_linear_velocity"]
            + command_terms["command_yaw_rate"]
            + command_terms["command_progress"]
            + float(cfg.core_losses.joint_posture_weight) * posture
            + float(cfg.core_losses.smoothness_weight) * smooth["control_rate"]
        )

    del alpha_count
    update = sqp_rti_update(
        base_control=base_control,
        lq_problem=lq_problem,
        merit_fn=merit_fn,
        regularization=float(cfg.solver.regularization),
        alphas=tuple(cfg.solver.line_search_alphas),
    )
    rollout = rollout_controls(measured_state, update.control, dt=float(cfg.runtime.dt))
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
    )
    return JointMpcRtiStepResult(
        full_trajectory=trajectory,
        pending_reference=pending,
        solver_state=next_solver_state,
    )


__all__ = ["step"]
