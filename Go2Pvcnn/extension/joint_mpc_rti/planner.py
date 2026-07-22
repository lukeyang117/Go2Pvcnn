"""Pure-tensor one-step rolling planner API."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.integration.command import body_linear_velocity_to_world
from extension.joint_mpc_rti.losses.barriers import (
    localized_relaxed_barrier_derivative,
    relaxed_barrier,
    relaxed_barrier_derivative,
)
from extension.joint_mpc_rti.losses.contact import current_stance_segment_mask, reliable_support_height
from extension.joint_mpc_rti.losses.rollout_objective import rollout_loss_breakdown_maybe_compiled
from extension.joint_mpc_rti.model.dynamics import kinematic_step
from extension.joint_mpc_rti.model.gait_schedule import (
    ContactSchedulerAdvance,
    adaptive_contact_schedule,
    advance_contact_scheduler,
    fixed_trot_schedule,
)
from extension.joint_mpc_rti.model.go2_kinematics import (
    complete_body_sample_jacobian,
    complete_foot_jacobian,
    complete_knee_jacobian,
    complete_link_sample_jacobians,
    foot_jacobian_leg,
    go2_fk,
    go2_foot_pos,
    link_sample_jacobians,
)
from extension.joint_mpc_rti.model.rollout import JointMpcRollout, rollout_controls, rollout_state_sequence
from extension.joint_mpc_rti.solver.fixed_spd import fixed_spd_solve
from extension.joint_mpc_rti.solver.linearization import dynamics_jacobians
from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem, solve_lq_subproblem
from extension.joint_mpc_rti.solver.sqp_rti import sqp_rti_update
from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery, query_world, query_world_maybe_compiled
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
    return query_world(field, points_w)


def _root_direction_limits(
    base_limits: Tensor,
    terrain_step_present: Tensor,
    *,
    vertical_limit: float,
    linear_limit: float,
) -> Tensor:
    """Replace root-vz's direction limit without allocating a CUDA index tensor."""
    root_vertical_limit = torch.where(
        terrain_step_present[..., None],
        base_limits.new_full(base_limits.shape[:-1] + (1,), float(vertical_limit)),
        base_limits.new_full(base_limits.shape[:-1] + (1,), float(linear_limit)),
    )
    return torch.cat((base_limits[..., :2], root_vertical_limit, base_limits[..., 3:]), dim=-1)


@dataclass(frozen=True)
class _LinearizationQueries:
    body: JointMpcTerrainQuery
    foot: JointMpcTerrainQuery
    knee: JointMpcTerrainQuery
    shank: JointMpcTerrainQuery
    thigh: JointMpcTerrainQuery
    root: JointMpcTerrainQuery


def _touchdown_ready_mask(
    *,
    foot_surface_error_m: Tensor,
    foot_vertical_step_m: Tensor,
    foot_small_distance_m: Tensor,
    leg_collision: Tensor,
    base_collision: Tensor,
    joint_safe: Tensor,
    lookahead_collision: Tensor,
    map_valid: Tensor,
    surface_gap_limit_m: float,
    surface_penetration_limit_m: float,
    touchdown_margin_m: float,
) -> Tensor:
    """Return per-leg grounded-and-safe touchdown confirmation."""
    error = torch.as_tensor(foot_surface_error_m)
    vertical_step = torch.as_tensor(foot_vertical_step_m, dtype=error.dtype, device=error.device)
    small_distance = torch.as_tensor(foot_small_distance_m, dtype=error.dtype, device=error.device)
    leg_hit = torch.as_tensor(leg_collision, dtype=torch.bool, device=error.device)
    body_hit = torch.as_tensor(base_collision, dtype=torch.bool, device=error.device)
    safe_joint = torch.as_tensor(joint_safe, dtype=torch.bool, device=error.device)
    future_hit = torch.as_tensor(lookahead_collision, dtype=torch.bool, device=error.device)
    valid = torch.as_tensor(map_valid, dtype=torch.bool, device=error.device)
    if error.ndim != 2 or int(error.shape[1]) != 4:
        raise ValueError("touchdown leg tensors must have shape [B,4]")
    if any(value.shape != error.shape for value in (vertical_step, small_distance, leg_hit, safe_joint, future_hit, valid)):
        raise ValueError("touchdown leg tensors must have shape [B,4]")
    if body_hit.shape != error.shape[:1]:
        raise ValueError("base_collision must have shape [B]")
    grounded = torch.logical_and(
        error <= float(surface_gap_limit_m),
        error >= -float(surface_penetration_limit_m),
    )
    descending_or_still = vertical_step <= 0.0
    safe_foot = small_distance >= float(touchdown_margin_m)
    return (
        grounded
        & descending_or_still
        & safe_foot
        & torch.logical_not(leg_hit)
        & torch.logical_not(body_hit).unsqueeze(-1)
        & safe_joint
        & torch.logical_not(future_hit)
        & valid
    )


def _measured_touchdown_readiness(
    measured_state: JointMpcRtiState,
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
    *,
    return_foot_distance: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Evaluate current per-leg touchdown readiness and reliable stance safety."""
    batch = measured_state.batch_size
    geometry = go2_fk(measured_state.root_pos_w, measured_state.root_rpy_w, measured_state.joint_pos)
    predicted_state = kinematic_step(
        measured_state.as_vector(),
        torch.cat(
            (measured_state.root_lin_vel_b, measured_state.root_ang_vel_b, measured_state.joint_vel),
            dim=-1,
        ),
        dt=float(cfg.runtime.dt),
    )
    predicted = go2_fk(predicted_state[:, :3], predicted_state[:, 3:6], predicted_state[:, 6:])

    def collision_state(current_geometry) -> tuple[Tensor, Tensor, JointMpcTerrainQuery]:
        body_count = int(current_geometry.body_samples_w.shape[1])
        points = torch.cat(
            (
                current_geometry.foot_pos_w,
                current_geometry.knee_pos_w,
                current_geometry.shank_samples_w.reshape(batch, 12, 3),
                current_geometry.thigh_samples_w.reshape(batch, 12, 3),
                current_geometry.body_samples_w,
            ),
            dim=1,
        )
        query = _query_world(terrain_field, points, cfg)
        foot_distance = query.small_distance_m[:, :4]
        knee_distance = query.small_distance_m[:, 4:8]
        calf_distance = query.small_distance_m[:, 8:20].reshape(batch, 4, 3)
        thigh_distance = query.small_distance_m[:, 20:32].reshape(batch, 4, 3)
        base_distance = query.small_distance_m[:, 32 : 32 + body_count]
        foot_height = query.height_w[:, :4]
        knee_height = query.height_w[:, 4:8]
        calf_height = query.height_w[:, 8:20].reshape(batch, 4, 3)
        thigh_height = query.height_w[:, 20:32].reshape(batch, 4, 3)
        base_height = query.height_w[:, 32 : 32 + body_count]
        foot_hit = _sphere_link_collision(
            current_geometry.foot_pos_w,
            foot_distance,
            foot_height,
            radius=cfg.gait.foot_collision_radius,
        )
        knee_hit = _sphere_link_collision(
            current_geometry.knee_pos_w,
            knee_distance,
            knee_height,
            radius=cfg.gait.knee_collision_radius,
        )
        calf_hit = _sphere_link_collision(
            current_geometry.shank_samples_w,
            calf_distance,
            calf_height,
            radius=cfg.gait.calf_collision_radius,
        ).any(dim=2)
        thigh_hit = _sphere_link_collision(
            current_geometry.thigh_samples_w,
            thigh_distance,
            thigh_height,
            radius=cfg.gait.thigh_collision_radius,
        ).any(dim=2)
        base_hit = torch.logical_and(
            base_distance < 0.0,
            torch.logical_and(
                current_geometry.body_samples_w[..., 2] < base_height,
                current_geometry.body_samples_w[..., 2] > 0.0,
            ),
        ).any(dim=1)
        return foot_hit | knee_hit | calf_hit | thigh_hit, base_hit, query

    leg_collision, base_collision, query = collision_state(geometry)
    future_leg_collision, future_base_collision, _ = collision_state(predicted)
    foot_height = query.height_w[:, :4]
    foot_distance = query.small_distance_m[:, :4]
    foot_valid = query.valid[:, :4]
    surface_error = geometry.foot_pos_w[..., 2] - foot_height - float(cfg.gait.foot_contact_offset)
    vertical_step = predicted.foot_pos_w[..., 2] - geometry.foot_pos_w[..., 2]
    lower = constant_like(measured_state.joint_pos, "touchdown_joint_lower", (-1.0472, -0.6632, -2.721) * 4)
    upper = constant_like(measured_state.joint_pos, "touchdown_joint_upper", (1.0472, 2.966, -0.837) * 4)
    margin = float(cfg.solver.joint_position_safety_margin_rad)
    joint_safe = (
        (measured_state.joint_pos >= lower + margin)
        & (measured_state.joint_pos <= upper - margin)
    ).reshape(batch, 4, 3).all(dim=2)
    ready = _touchdown_ready_mask(
        foot_surface_error_m=surface_error,
        foot_vertical_step_m=vertical_step,
        foot_small_distance_m=foot_distance,
        leg_collision=leg_collision,
        base_collision=base_collision,
        joint_safe=joint_safe,
        lookahead_collision=future_leg_collision | future_base_collision.unsqueeze(-1),
        map_valid=foot_valid,
        surface_gap_limit_m=float(cfg.solver.stance_ground_gap_limit_m),
        surface_penetration_limit_m=float(cfg.solver.stance_ground_penetration_limit_m),
        touchdown_margin_m=float(cfg.gait.small_touchdown_margin),
    )
    # The measured x0 used by the rolling contract can place a nominal stance
    # foot a few millimetres below the analytic terrain plane.  The foot
    # sphere then reports a local collision even though this is the intended
    # support contact.  Preserve strict full-body safety for touchdown, while
    # allowing this bounded grounding contact to count toward the two-leg
    # support budget that gates liftoff.
    grounding_tolerance = float(cfg.gait.stance_ground_recovery_step_m)
    grounded_support = (
        surface_error.abs().le(grounding_tolerance)
        & foot_distance.ge(float(cfg.gait.small_support_safety_margin))
        & (surface_error <= 0.0)
    )
    ready = torch.logical_or(
        ready,
        grounded_support
        & (vertical_step <= 0.0)
        & torch.logical_not(base_collision).unsqueeze(-1)
        & torch.logical_not(future_base_collision).unsqueeze(-1)
        & joint_safe
        & foot_valid,
    )
    reliable = (
        surface_error.abs().le(grounding_tolerance)
        & foot_distance.ge(float(cfg.gait.small_support_safety_margin))
        & (torch.logical_not(leg_collision) | grounded_support)
        & torch.logical_not(base_collision).unsqueeze(-1)
        & (torch.logical_not(future_leg_collision) | grounded_support)
        & torch.logical_not(future_base_collision).unsqueeze(-1)
        & joint_safe
        & foot_valid
    )
    if return_foot_distance:
        return ready, reliable, foot_distance
    return ready, reliable


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
    queried = query_world_maybe_compiled(
        terrain_field,
        packed.reshape(batch, nodes * points_per_node, 3),
        enabled=bool(cfg.solver.compile_kernels),
    )

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


def _select_safer_base_control(
    shaped_control: Tensor,
    shaped_violation: Tensor,
    hold_control: Tensor,
    hold_violation: Tensor,
) -> Tensor:
    shaped = torch.as_tensor(shaped_control)
    hold = torch.as_tensor(hold_control, dtype=shaped.dtype, device=shaped.device)
    shaped_components = torch.as_tensor(
        shaped_violation, dtype=shaped.dtype, device=shaped.device
    )
    hold_components = torch.as_tensor(
        hold_violation, dtype=shaped.dtype, device=shaped.device
    )
    if hold.shape != shaped.shape or hold_components.shape != shaped_components.shape:
        raise ValueError("base controls and violation components must have matching shapes")
    tolerance = 1.0e-9
    different = (hold_components - shaped_components).abs() > tolerance
    first_difference = different.to(torch.long).argmax(dim=1)
    hold_first = torch.gather(hold_components, 1, first_difference[:, None]).squeeze(1)
    shaped_first = torch.gather(shaped_components, 1, first_difference[:, None]).squeeze(1)
    use_hold = torch.logical_and(
        different.any(dim=1),
        hold_first < shaped_first - tolerance,
    )
    return torch.where(use_hold[:, None, None], hold, shaped)


def _hold_control_with_ground_safe_recovery(
    shaped_control: Tensor,
    ground_safe_recovery: Tensor,
) -> Tensor:
    """Hold x1 globally while preserving independent safe-leg recovery motion."""
    shaped = torch.as_tensor(shaped_control)
    safe = torch.as_tensor(
        ground_safe_recovery,
        dtype=torch.bool,
        device=shaped.device,
    )
    if shaped.ndim != 3 or int(shaped.shape[-1]) != 18 or int(shaped.shape[1]) < 1:
        raise ValueError("shaped_control must have shape [B,H,18] with H >= 1")
    if safe.shape != (int(shaped.shape[0]), 4):
        raise ValueError("ground_safe_recovery must have shape [B,4]")
    hold = shaped.clone()
    hold[:, 0] = 0.0
    shaped_joint = shaped[:, 0, 6:].reshape(int(shaped.shape[0]), 4, 3)
    hold_joint = hold[:, 0, 6:].reshape(int(shaped.shape[0]), 4, 3)
    hold[:, 0, 6:] = torch.where(
        safe.unsqueeze(-1),
        shaped_joint,
        hold_joint,
    ).reshape(int(shaped.shape[0]), 12)
    return hold


def _scale_control_direction_to_limits(
    delta_control: Tensor,
    *,
    root_linear_limit: float,
    root_angular_limit: float,
    joint_limit: float,
) -> Tensor:
    delta = torch.as_tensor(delta_control)
    if delta.ndim != 3 or int(delta.shape[-1]) != 18:
        raise ValueError("delta_control must have shape [B,H,18]")
    limits = constant_like(
        delta,
        "joint_mpc_control_direction_limits",
        (float(root_linear_limit),) * 3
        + (float(root_angular_limit),) * 3
        + (float(joint_limit),) * 12,
    )
    scale = (
        limits.view(1, 1, 18)
        / delta.abs().clamp_min(torch.finfo(delta.dtype).eps)
    ).amin(dim=2).clamp_max(1.0)
    return delta * scale.unsqueeze(-1)


def _joint_candidate_absolute_limit(cfg: JointMpcRtiCfg) -> float:
    """Return the production absolute joint-control bound for candidates."""
    return float(cfg.gait.max_nominal_joint_velocity) + float(
        cfg.solver.joint_direction_limit
    )


def _reserve_joint_candidate_direction_capacity(
    control: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Keep the base warm start inside the nominal joint-velocity envelope."""
    reserved = torch.as_tensor(control).clone()
    nominal_limit = float(cfg.gait.max_nominal_joint_velocity)
    reserved[..., 6:] = reserved[..., 6:].clamp(-nominal_limit, nominal_limit)
    return reserved


def _scale_constrained_control_direction(
    delta_control: Tensor,
    delta_state: Tensor,
    constraint_control: Tensor,
    constraint_state: Tensor,
    constraint_residual: Tensor,
    *,
    limits: Tensor,
    required_limits: Tensor | None = None,
    base_control: Tensor | None = None,
    required_absolute_limits: Tensor | None = None,
    matrix_a: Tensor | None = None,
    matrix_b: Tensor | None = None,
    affine_dynamics: Tensor | None = None,
    initial_state: Tensor | None = None,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    delta = torch.as_tensor(delta_control)
    state = torch.as_tensor(delta_state, dtype=delta.dtype, device=delta.device)
    control_constraint = torch.as_tensor(
        constraint_control, dtype=delta.dtype, device=delta.device
    )
    state_constraint = torch.as_tensor(
        constraint_state, dtype=delta.dtype, device=delta.device
    )
    residual = torch.as_tensor(
        constraint_residual, dtype=delta.dtype, device=delta.device
    )
    limit_input = torch.as_tensor(limits, dtype=delta.dtype, device=delta.device)
    if limit_input.shape == (delta.shape[-1],):
        limit = limit_input.view(1, 1, -1).expand_as(delta)
    elif limit_input.shape == delta.shape:
        limit = limit_input
    else:
        raise ValueError("limits must have shape [control_dim] or [B,H,control_dim]")
    required_input = limit_input if required_limits is None else torch.as_tensor(
        required_limits, dtype=delta.dtype, device=delta.device
    )
    if required_input.shape == (delta.shape[-1],):
        required_limit = required_input.view(1, 1, -1).expand_as(delta)
    elif required_input.shape == delta.shape:
        required_limit = required_input
    else:
        raise ValueError("required_limits must have shape [control_dim] or [B,H,control_dim]")
    if state.shape[1] != delta.shape[1] + 1:
        raise ValueError("delta state/control and limits have incompatible shapes")
    dynamics_inputs = (matrix_a, matrix_b, affine_dynamics, initial_state)
    if any(value is not None for value in dynamics_inputs):
        if any(value is None for value in dynamics_inputs):
            raise ValueError(
                "matrix_a, matrix_b, affine_dynamics and initial_state must be provided together"
            )
        dynamics_a = torch.as_tensor(matrix_a, dtype=delta.dtype, device=delta.device)
        dynamics_b = torch.as_tensor(matrix_b, dtype=delta.dtype, device=delta.device)
        dynamics_affine = torch.as_tensor(
            affine_dynamics, dtype=delta.dtype, device=delta.device
        )
        recovered_state = torch.as_tensor(
            initial_state, dtype=delta.dtype, device=delta.device
        )
        expected_matrix_shape = (
            delta.shape[0], delta.shape[1], state.shape[-1], state.shape[-1]
        )
        expected_control_shape = (
            delta.shape[0], delta.shape[1], state.shape[-1], delta.shape[-1]
        )
        if dynamics_a.shape != expected_matrix_shape:
            raise ValueError("matrix_a must have shape [B,H,X,X]")
        if dynamics_b.shape != expected_control_shape:
            raise ValueError("matrix_b must have shape [B,H,X,U]")
        if dynamics_affine.shape != state[:, :-1].shape:
            raise ValueError("affine_dynamics must have shape [B,H,X]")
        if recovered_state.shape != state[:, 0].shape:
            raise ValueError("initial_state must have shape [B,X]")
        base_tensor = None
        if base_control is not None:
            base_tensor = torch.as_tensor(
                base_control, dtype=delta.dtype, device=delta.device
            )
            if base_tensor.shape != delta.shape:
                raise ValueError("base_control must match delta_control")
        absolute_tensor = None
        if required_absolute_limits is not None:
            absolute_tensor = torch.as_tensor(
                required_absolute_limits, dtype=delta.dtype, device=delta.device
            )
            if absolute_tensor.shape == (delta.shape[-1],):
                absolute_tensor = absolute_tensor.view(1, 1, -1).expand_as(delta)
            elif absolute_tensor.shape != delta.shape:
                raise ValueError(
                    "required_absolute_limits must have shape [control_dim] or [B,H,control_dim]"
                )
        recovered_controls: list[Tensor] = []
        recovered_required_controls: list[Tensor] = []
        recovered_free_controls: list[Tensor] = []
        for node in range(int(delta.shape[1])):
            stage_state = torch.stack((recovered_state, recovered_state), dim=1)
            recovered_required, recovered_free = _scale_constrained_control_direction(
                delta[:, node : node + 1],
                stage_state,
                control_constraint[:, node : node + 1],
                state_constraint[:, node : node + 1],
                residual[:, node : node + 1],
                limits=limit[:, node : node + 1],
                required_limits=required_limit[:, node : node + 1],
                base_control=(
                    None if base_tensor is None else base_tensor[:, node : node + 1]
                ),
                required_absolute_limits=(
                    None
                    if absolute_tensor is None
                    else absolute_tensor[:, node : node + 1]
                ),
                return_components=True,
            )
            recovered_control = recovered_required + recovered_free
            recovered_controls.append(recovered_control[:, 0])
            recovered_required_controls.append(recovered_required[:, 0])
            recovered_free_controls.append(recovered_free[:, 0])
            recovered_state = (
                torch.matmul(
                    dynamics_a[:, node], recovered_state.unsqueeze(-1)
                ).squeeze(-1)
                + torch.matmul(
                    dynamics_b[:, node], recovered_control[:, 0].unsqueeze(-1)
                ).squeeze(-1)
                + dynamics_affine[:, node]
            )
        recovered_total = torch.stack(recovered_controls, dim=1)
        if return_components:
            return (
                torch.stack(recovered_required_controls, dim=1),
                torch.stack(recovered_free_controls, dim=1),
            )
        return recovered_total
    feedback, feedforward, projector, right_inverse = (
        _affine_control_constraint_parameterization_with_right_inverse(
        control_constraint,
        state_constraint,
        residual,
        )
    )
    required = (
        torch.matmul(feedback, state[:, :-1].unsqueeze(-1)).squeeze(-1)
        + feedforward
    )
    free = torch.matmul(projector, delta.unsqueeze(-1)).squeeze(-1)
    absolute_limit = None
    base = None
    if required_absolute_limits is not None:
        if base_control is None:
            raise ValueError("base_control is required with required_absolute_limits")
        base = torch.as_tensor(base_control, dtype=delta.dtype, device=delta.device)
        if base.shape != delta.shape:
            raise ValueError("base_control must match delta_control")
        absolute_input = torch.as_tensor(
            required_absolute_limits, dtype=delta.dtype, device=delta.device
        )
        if absolute_input.shape == (delta.shape[-1],):
            absolute_limit = absolute_input.view(1, 1, -1).expand_as(delta)
        elif absolute_input.shape == delta.shape:
            absolute_limit = absolute_input
        else:
            raise ValueError(
                "required_absolute_limits must have shape [control_dim] or [B,H,control_dim]"
            )
        absolute_active = absolute_limit > 0.0
        directional_capacity = torch.where(
            required >= 0.0,
            absolute_limit - base,
            absolute_limit + base,
        ).clamp_min(0.0)
        required_limit = torch.where(absolute_active, directional_capacity, required_limit)

    def remaining_absolute_capacity(bounded_required: Tensor, free_direction: Tensor) -> Tensor:
        if absolute_limit is None or base is None:
            return torch.full_like(free_direction, float("inf"))
        current = base + bounded_required
        capacity = torch.where(
            free_direction >= 0.0,
            absolute_limit - current,
            absolute_limit + current,
        ).clamp_min(0.0)
        return torch.where(absolute_limit > 0.0, capacity, torch.full_like(capacity, float("inf")))

    margin = torch.minimum(
        (required_limit - required.abs()).clamp_min(0.0),
        limit,
    )
    margin = torch.minimum(margin, remaining_absolute_capacity(required, free))
    scale = (margin / free.abs().clamp_min(torch.finfo(delta.dtype).eps)).amin(
        dim=2
    ).clamp_max(1.0)
    feasible_direction = required + scale.unsqueeze(-1) * free
    required_feasible = (required.abs() <= required_limit).all(dim=2)
    required_scale = (
        required_limit
        / required.abs().clamp_min(torch.finfo(delta.dtype).eps)
    ).amin(dim=2).clamp_max(1.0)
    bounded_required = required * required_scale.unsqueeze(-1)
    bounded_margin = torch.minimum(
        (required_limit - bounded_required.abs()).clamp_min(0.0),
        limit,
    )
    bounded_margin = torch.minimum(
        bounded_margin,
        remaining_absolute_capacity(bounded_required, free),
    )
    free_abs = free.abs()
    free_scale = torch.where(
        free_abs > torch.finfo(delta.dtype).eps,
        bounded_margin / free_abs.clamp_min(torch.finfo(delta.dtype).eps),
        torch.full_like(free_abs, float("inf")),
    ).amin(dim=2).clamp_max(1.0)
    fallback_direction = bounded_required + free_scale.unsqueeze(-1) * free
    redistributed_required = required
    projector_diagonal = projector.diagonal(dim1=-2, dim2=-1)
    for _ in range(3):
        clipped_required = torch.maximum(
            torch.minimum(redistributed_required, required_limit),
            -required_limit,
        )
        clipping_error = clipped_required - redistributed_required
        repair_weight = torch.where(
            projector_diagonal.abs() > 1.0e-5,
            clipping_error / projector_diagonal.clamp_min(1.0e-5),
            torch.zeros_like(clipping_error),
        )
        redistributed_required = redistributed_required + torch.matmul(
            projector, repair_weight.unsqueeze(-1)
        ).squeeze(-1)
        redistribution_residual = (
            torch.matmul(
                control_constraint, redistributed_required.unsqueeze(-1)
            ).squeeze(-1)
            + torch.matmul(
                state_constraint, state[:, :-1].unsqueeze(-1)
            ).squeeze(-1)
            + residual
        )
        redistributed_required = redistributed_required - torch.matmul(
            right_inverse, redistribution_residual.unsqueeze(-1)
        ).squeeze(-1)
    redistributed_constraint_residual = (
        torch.matmul(
            control_constraint, redistributed_required.unsqueeze(-1)
        ).squeeze(-1)
        + torch.matmul(
            state_constraint, state[:, :-1].unsqueeze(-1)
        ).squeeze(-1)
        + residual
    )
    redistributed_feasible = torch.logical_and(
        (redistributed_required.abs() <= required_limit + 2.0e-5).all(dim=2),
        (redistributed_constraint_residual.abs() <= 2.0e-5).all(dim=2),
    )
    redistributed_margin = torch.minimum(
        (required_limit - redistributed_required.abs()).clamp_min(0.0),
        limit,
    )
    redistributed_margin = torch.minimum(
        redistributed_margin,
        remaining_absolute_capacity(redistributed_required, free),
    )
    redistributed_free_scale = torch.where(
        free_abs > torch.finfo(delta.dtype).eps,
        redistributed_margin / free_abs.clamp_min(torch.finfo(delta.dtype).eps),
        torch.full_like(free_abs, float("inf")),
    ).amin(dim=2).clamp_max(1.0)
    redistributed_direction = (
        redistributed_required + redistributed_free_scale.unsqueeze(-1) * free
    )
    redistributed_free = redistributed_free_scale.unsqueeze(-1) * free
    fallback_free = free_scale.unsqueeze(-1) * free
    bounded_infeasible_direction = torch.where(
        redistributed_feasible.unsqueeze(-1),
        redistributed_direction,
        fallback_direction,
    )
    selected_required = torch.where(
        required_feasible.unsqueeze(-1),
        required,
        torch.where(
            redistributed_feasible.unsqueeze(-1),
            redistributed_required,
            bounded_required,
        ),
    )
    selected_free = torch.where(
        required_feasible.unsqueeze(-1),
        scale.unsqueeze(-1) * free,
        torch.where(
            redistributed_feasible.unsqueeze(-1),
            redistributed_free,
            fallback_free,
        ),
    )
    selected_direction = torch.where(
        required_feasible.unsqueeze(-1),
        feasible_direction,
        bounded_infeasible_direction,
    )
    if return_components:
        return selected_required, selected_free
    return selected_direction


def _clamp_recovery_ground_error_to_control_reach(
    ground_error: Tensor,
    control_jacobian: Tensor,
    *,
    joint_direction_limit: float,
) -> Tensor:
    error = torch.as_tensor(ground_error)
    jacobian = torch.as_tensor(
        control_jacobian, dtype=error.dtype, device=error.device
    )
    if jacobian.shape[:-1] != error.shape or int(jacobian.shape[-1]) != 18:
        raise ValueError("control_jacobian must match ground_error with control dim 18")
    capacity = (
        jacobian[..., 6:].abs().sum(dim=-1) * float(joint_direction_limit)
    )
    return torch.maximum(torch.minimum(error, capacity), -capacity)


def _contact_segment_age(
    contact: Tensor,
    initial_phase_age: Tensor,
    *,
    half_cycle_steps: int,
) -> Tensor:
    """Return per-leg segment ages, saturating unsafe swing extensions at the landing endpoint."""
    contact_state = torch.as_tensor(contact, dtype=torch.bool)
    initial = torch.as_tensor(initial_phase_age, dtype=torch.long, device=contact_state.device)
    if initial.ndim == 1:
        initial = initial[:, None].expand(-1, 4)
    if contact_state.ndim != 3 or contact_state.shape[2] != 4 or initial.shape != contact_state[:, 0].shape:
        raise ValueError("contact and initial_phase_age must have shapes [B,T,4] and [B,4]")
    max_age = int(half_cycle_steps) - 1
    current = initial.clamp(0, max_age)
    ages: list[Tensor] = [current]
    for node in range(1, int(contact_state.shape[1])):
        changed = contact_state[:, node] != contact_state[:, node - 1]
        current = torch.where(changed, torch.zeros_like(current), (current + 1).clamp_max(max_age))
        ages.append(current)
    return torch.stack(ages, dim=1)


def _guard_early_release_by_support(
    contact_state: Tensor,
    requested_release: Tensor,
    *,
    min_support: int = 2,
) -> Tensor:
    """Allow deterministic early releases without dropping below the support budget."""
    contact = torch.as_tensor(contact_state, dtype=torch.bool)
    requested = torch.as_tensor(requested_release, dtype=torch.bool, device=contact.device)
    if contact.shape != requested.shape or contact.ndim != 3 or int(contact.shape[-1]) != 4:
        raise ValueError("contact_state and requested_release must have shape [B,T,4]")
    if int(min_support) < 0 or int(min_support) > 4:
        raise ValueError("min_support must be in [0,4]")
    candidates = torch.logical_and(contact, requested)
    release_budget = (contact.sum(dim=2) - int(min_support)).clamp_min(0)
    candidate_rank = torch.cumsum(candidates.to(torch.long), dim=2)
    return torch.logical_and(
        candidates,
        candidate_rank <= release_budget.unsqueeze(-1),
    )


def _support_guarded_early_handoff(
    contact_state: Tensor,
    requested_release: Tensor,
    touchdown_ready: Tensor,
    *,
    min_support: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Atomically promote grounded swing legs before an obstacle-driven early release."""
    contact = torch.as_tensor(contact_state, dtype=torch.bool)
    requested = torch.as_tensor(requested_release, dtype=torch.bool, device=contact.device)
    ready = torch.as_tensor(touchdown_ready, dtype=torch.bool, device=contact.device)
    if contact.shape != requested.shape or contact.ndim != 3 or int(contact.shape[-1]) != 4:
        raise ValueError("contact_state and requested_release must have shape [B,T,4]")
    if ready.shape != (int(contact.shape[0]), 4):
        raise ValueError("touchdown_ready must have shape [B,4]")
    if int(min_support) < 0 or int(min_support) > 4:
        raise ValueError("min_support must be in [0,4]")

    release_candidates = torch.logical_and(contact, requested).clone()
    release_candidates[:, 0] = False
    promotable = torch.logical_and(
        torch.logical_not(contact),
        ready[:, None, :],
    )
    release_budget = (
        contact.sum(dim=2) + promotable.sum(dim=2) - int(min_support)
    ).clamp_min(0)
    release_rank = torch.cumsum(release_candidates.to(torch.long), dim=2)
    released = torch.logical_and(
        release_candidates,
        release_rank <= release_budget.unsqueeze(-1),
    )
    after_release = torch.logical_and(contact, torch.logical_not(released))
    needed = (int(min_support) - after_release.sum(dim=2)).clamp_min(0)
    needed = torch.where(released.any(dim=2), needed, torch.zeros_like(needed))
    promotion_rank = torch.cumsum(promotable.to(torch.long), dim=2)
    promoted = torch.logical_and(
        promotable,
        promotion_rank <= needed.unsqueeze(-1),
    )
    promoted[:, 0] = False
    updated = torch.logical_or(after_release, promoted)
    updated[:, 0] = contact[:, 0]
    return updated, promoted, released


def _reconcile_published_contact_state(
    scheduled: ContactSchedulerAdvance,
    published_contact: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Make persisted scheduler tensors agree with the published x1 contact state."""
    published = torch.as_tensor(
        published_contact,
        dtype=torch.bool,
        device=scheduled.contact_state.device,
    )
    if published.shape != scheduled.contact_state.shape:
        raise ValueError("published_contact must match scheduled contact_state")
    changed = published != scheduled.contact_state
    phase = torch.where(changed, torch.zeros_like(scheduled.phase_age), scheduled.phase_age)
    extension = torch.where(
        changed,
        torch.zeros_like(scheduled.swing_extension_age),
        scheduled.swing_extension_age,
    )
    stance = torch.where(
        published,
        torch.where(changed, torch.zeros_like(scheduled.stance_age), scheduled.stance_age),
        torch.zeros_like(scheduled.stance_age),
    )
    recovery = torch.where(
        changed,
        torch.zeros_like(scheduled.recovery_state),
        scheduled.recovery_state,
    )
    return published, phase, extension, stance, recovery


def _nominal_joint_target(
    contact: Tensor,
    phase_step: Tensor,
    command_body: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    dtype: torch.dtype,
    swing_envelope_floor: Tensor | None = None,
) -> Tensor:
    batch, nodes = int(contact.shape[0]), int(contact.shape[1])
    reference = contact.new_empty((), dtype=dtype)
    nominal = constant_like(reference, "nominal_joint_pos", cfg.gait.nominal_joint_pos).view(1, 1, 4, 3)
    target = nominal.expand(batch, nodes, 4, 3).clone()
    swing = torch.logical_not(contact)
    half_cycle = int(cfg.gait.half_cycle_steps)
    segment_age = _contact_segment_age(
        contact,
        phase_step,
        half_cycle_steps=half_cycle,
    ).to(dtype=dtype)
    progress = segment_age / float(max(half_cycle - 1, 1))
    envelope = torch.sin(torch.pi * progress).clamp_min(0.0).pow(
        float(cfg.gait.swing_return_exponent)
    )
    liftoff = torch.zeros_like(contact)
    liftoff[:, 1:] = torch.logical_and(torch.logical_not(contact[:, 1:]), contact[:, :-1])
    command_speed = torch.linalg.vector_norm(
        torch.as_tensor(command_body, dtype=dtype, device=contact.device)[:, :2],
        dim=-1,
    )
    liftoff_scale = (command_speed / float(cfg.gait.full_swing_reference_speed)).clamp(0.0, 1.0)
    envelope = torch.where(
        liftoff,
        float(cfg.gait.liftoff_joint_envelope) * liftoff_scale[:, None, None],
        envelope,
    )
    if swing_envelope_floor is not None:
        envelope_floor = torch.as_tensor(
            swing_envelope_floor,
            dtype=dtype,
            device=contact.device,
        )
        if envelope_floor.shape != contact.shape:
            raise ValueError("swing_envelope_floor must match contact")
        envelope = torch.maximum(envelope, envelope_floor.clamp(0.0, 1.0))
    envelope = envelope * swing.to(dtype=dtype)
    target[..., 1] = target[..., 1] + envelope * (float(cfg.gait.swing_thigh_angle) - target[..., 1])
    target[..., 2] = target[..., 2] + envelope * (float(cfg.gait.swing_calf_angle) - target[..., 2])
    return target.reshape(batch, nodes, 12)


def _recovery_joint_targets(
    joint_target: Tensor,
    recovery_state: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    near_small: Tensor | None = None,
) -> Tensor:
    target = torch.as_tensor(joint_target)
    recovery = torch.as_tensor(recovery_state, dtype=torch.bool, device=target.device)
    if target.ndim != 3 or int(target.shape[-1]) != 12:
        raise ValueError("joint_target must have shape [B,T,12]")
    if recovery.shape != (int(target.shape[0]), 4):
        raise ValueError("recovery_state must have shape [B,4]")
    shaped = target.reshape(int(target.shape[0]), int(target.shape[1]), 4, 3)
    nominal = constant_like(target, "recovery_nominal_joint_target", cfg.gait.nominal_joint_pos).view(
        1, 1, 4, 3
    )
    ground_recovery = recovery[:, None, :].expand(-1, int(target.shape[1]), -1)
    if near_small is not None:
        obstacle_mask = torch.as_tensor(
            near_small,
            dtype=torch.bool,
            device=target.device,
        )
        if obstacle_mask.shape != ground_recovery.shape:
            raise ValueError("near_small must have shape [B,T,4]")
        ground_recovery = ground_recovery & torch.logical_not(obstacle_mask)
    return torch.where(ground_recovery.unsqueeze(-1), nominal, shaped).reshape_as(target)


def _recovery_near_small_mask(
    small_distance_m: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Block recovery lowering only inside the touchdown-safety margin."""
    distance = torch.as_tensor(small_distance_m)
    return distance < float(cfg.gait.small_touchdown_margin)


def _recovery_sdf_constraint_clearance(
    small_distance_m: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Target a small positive buffer beyond the strict recovery exit gate."""
    distance = torch.as_tensor(small_distance_m)
    return (
        distance
        - float(cfg.gait.small_touchdown_margin)
        - float(cfg.gait.recovery_sdf_exit_buffer_m)
    )


def _swing_phase_weight(contact: Tensor, phase_step: Tensor, cfg: JointMpcRtiCfg, *, dtype: torch.dtype) -> Tensor:
    half_cycle = int(cfg.gait.half_cycle_steps)
    segment_age = _contact_segment_age(
        contact,
        phase_step,
        half_cycle_steps=half_cycle,
    ).to(dtype=dtype)
    return (
        torch.sin(torch.pi * segment_age / float(max(half_cycle - 1, 1)))
        * torch.logical_not(contact).to(dtype=dtype)
    )


def _small_swing_handoff_weights(
    contact: Tensor,
    phase_step: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    half_cycle = int(cfg.gait.half_cycle_steps)
    segment_age = _contact_segment_age(
        contact,
        phase_step,
        half_cycle_steps=half_cycle,
    ).to(dtype=dtype)
    progress = segment_age / float(max(half_cycle - 1, 1))
    swing = torch.logical_not(contact).to(dtype=dtype)
    mid_swing = torch.sin(torch.pi * progress).clamp_min(0.0).pow(
        float(cfg.gait.small_foot_over_phase_exponent)
    )
    safe_landing = progress.pow(float(cfg.gait.small_safe_landing_phase_exponent))
    return mid_swing * swing, safe_landing * swing


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


def _step_bounded_stance_anchor(
    stance_anchor_w: Tensor,
    measured_foot_w: Tensor,
    contact_state: Tensor,
    *,
    max_step_m: float,
) -> Tensor:
    """Bound the x1 equality target while preserving the persistent world anchor."""
    anchor = torch.as_tensor(stance_anchor_w)
    measured = torch.as_tensor(
        measured_foot_w,
        dtype=anchor.dtype,
        device=anchor.device,
    )
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=anchor.device)
    if anchor.ndim != 4 or tuple(anchor.shape[-2:]) != (4, 3):
        raise ValueError("stance_anchor_w must have shape [B,T,4,3]")
    if measured.shape != anchor[:, 0].shape or contact.shape != anchor.shape[:-1]:
        raise ValueError("measured foot/contact must match stance anchors")
    if float(max_step_m) <= 0.0:
        raise ValueError("max_step_m must be positive")
    continuing = torch.logical_and(contact[:, 0], contact[:, 1])
    offset = anchor[:, 1, :, :2] - measured[..., :2]
    distance = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
    scale = (float(max_step_m) / distance.clamp_min(1.0e-12)).clamp_max(1.0)
    target_xy = measured[..., :2] + scale * offset
    bounded = anchor.clone()
    bounded[:, 1, :, :2] = torch.where(
        continuing.unsqueeze(-1),
        target_xy,
        bounded[:, 1, :, :2],
    )
    return bounded


def _confirmed_stance_anchor(
    *,
    previous_anchor_w: Tensor,
    measured_foot_w: Tensor,
    terrain_height_w: Tensor,
    confirmed_touchdown: Tensor,
    foot_contact_offset: float,
) -> Tensor:
    """Update only confirmed anchors from measured XY and the queried contact surface."""
    previous = torch.as_tensor(previous_anchor_w)
    measured = torch.as_tensor(measured_foot_w, dtype=previous.dtype, device=previous.device)
    height = torch.as_tensor(terrain_height_w, dtype=previous.dtype, device=previous.device)
    confirmed = torch.as_tensor(confirmed_touchdown, dtype=torch.bool, device=previous.device)
    if measured.shape != previous.shape or height.shape != previous.shape[:-1] or confirmed.shape != height.shape:
        raise ValueError("anchor tensors must have shapes [B,4,3], [B,4], and [B,4]")
    candidate = measured.clone()
    candidate[..., 2] = height + float(foot_contact_offset)
    return torch.where(confirmed.unsqueeze(-1), candidate, previous)


def _sphere_link_collision(
    position_w: Tensor,
    signed_distance_m: Tensor,
    top_height_w: Tensor,
    *,
    radius: float,
) -> Tensor:
    """Evaluate the geometric sphere-versus-small-object collision contract."""
    position = torch.as_tensor(position_w)
    distance = torch.as_tensor(signed_distance_m, dtype=position.dtype, device=position.device)
    height = torch.as_tensor(top_height_w, dtype=position.dtype, device=position.device)
    if distance.shape != position.shape[:-1] or height.shape != distance.shape:
        raise ValueError("distance and height must match position samples")
    vertical = torch.logical_and(
        position[..., 2] - float(radius) < height,
        position[..., 2] + float(radius) > 0.0,
    )
    return torch.logical_and(distance < float(radius), vertical)


def _minimum_norm_leg_correction(
    clearance: Tensor,
    clearance_jacobian: Tensor,
    *,
    max_norm: float,
) -> Tensor:
    """Return a trust-bounded minimum-norm correction for four leg constraints."""
    constraint = torch.as_tensor(clearance)
    jacobian = torch.as_tensor(clearance_jacobian, dtype=constraint.dtype, device=constraint.device)
    if constraint.ndim == 2:
        constraint = constraint.unsqueeze(-1)
        jacobian = jacobian.unsqueeze(-2)
    if constraint.ndim != 3 or constraint.shape[1] != 4 or jacobian.shape != (*constraint.shape, 3):
        raise ValueError("clearance and Jacobian must have shapes [B,4,S] and [B,4,S,3]")
    active_jacobian = jacobian * (constraint < 0.0).unsqueeze(-1)
    normal = torch.einsum("blsi,blsj->blij", active_jacobian, active_jacobian)
    identity = torch.eye(3, dtype=constraint.dtype, device=constraint.device).view(1, 1, 3, 3)
    normal = normal + 1.0e-6 * identity
    rhs = torch.einsum("blsi,bls->bli", active_jacobian, torch.relu(-constraint))
    correction = fixed_spd_solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
    norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
    return correction * (float(max_norm) / norm.clamp_min(float(max_norm)))


def _enforce_joint_position_limits(
    measured_state: JointMpcRtiState,
    control: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    projected = torch.as_tensor(control).clone()
    lower = constant_like(projected, "joint_position_lower", (-1.0472, -0.6632, -2.721) * 4)
    upper = constant_like(projected, "joint_position_upper", (1.0472, 2.966, -0.837) * 4)
    lower = lower + float(cfg.solver.joint_position_safety_margin_rad)
    upper = upper - float(cfg.solver.joint_position_safety_margin_rad)
    integrated = measured_state.joint_pos[:, None] + torch.cumsum(
        projected[..., 6:] * float(cfg.runtime.dt),
        dim=1,
    )
    bounded = torch.maximum(torch.minimum(integrated, upper), lower)
    bounded_nodes = torch.cat((measured_state.joint_pos[:, None], bounded), dim=1)
    projected[..., 6:] = torch.diff(bounded_nodes, dim=1) / float(cfg.runtime.dt)
    return projected


def _enforce_root_assist_limits(
    measured_state: JointMpcRtiState,
    control: Tensor,
    command_body: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Project root assistance into per-step and nominal-relative H30 bounds."""
    projected = torch.as_tensor(control).clone()
    command = torch.as_tensor(command_body, dtype=projected.dtype, device=projected.device)
    command_xy = command[:, :2]
    command_norm = torch.linalg.vector_norm(command_xy, dim=1, keepdim=True)
    fallback = torch.zeros_like(command_xy)
    fallback[:, 0] = 1.0
    axis_body = torch.where(command_norm > 1.0e-6, command_xy / command_norm.clamp_min(1.0e-6), fallback)
    lateral_body = torch.stack((-axis_body[:, 1], axis_body[:, 0]), dim=1)
    velocity_delta = projected[..., :2] - command_xy[:, None]
    parallel_delta = (velocity_delta * axis_body[:, None]).sum(dim=-1, keepdim=True)
    lateral_delta = (velocity_delta * lateral_body[:, None]).sum(dim=-1, keepdim=True).clamp(
        -float(cfg.solver.root_lateral_velocity_error_limit_mps),
        float(cfg.solver.root_lateral_velocity_error_limit_mps),
    )
    projected[..., :2] = (
        command_xy[:, None]
        + parallel_delta * axis_body[:, None]
        + lateral_delta * lateral_body[:, None]
    )
    roll_pitch_rate_limit = max(
        float(cfg.solver.root_roll_pitch_rate_limit_rps) - 0.01,
        0.0,
    )
    projected[..., 3:5] = projected[..., 3:5].clamp(
        -roll_pitch_rate_limit,
        roll_pitch_rate_limit,
    )
    projected[..., 5] = (
        projected[..., 5] - command[:, None, 2]
    ).clamp(
        -float(cfg.solver.root_yaw_rate_error_limit_rps),
        float(cfg.solver.root_yaw_rate_error_limit_rps),
    ) + command[:, None, 2]

    dt = float(cfg.runtime.dt)
    actual_state = measured_state.as_vector()
    nominal_state = measured_state.as_vector()
    bounded_controls: list[Tensor] = []
    for node in range(int(projected.shape[1])):
        nominal_control = torch.zeros_like(projected[:, node])
        nominal_control[:, :2] = command_xy
        nominal_control[:, 5] = command[:, 2]
        raw_next = kinematic_step(actual_state, projected[:, node], dt=dt)
        nominal_next = kinematic_step(nominal_state, nominal_control, dt=dt)
        nominal_axis_world = body_linear_velocity_to_world(axis_body, nominal_state[:, 5])
        nominal_lateral_world = torch.stack((-nominal_axis_world[:, 1], nominal_axis_world[:, 0]), dim=1)
        xy_delta = raw_next[:, :2] - nominal_next[:, :2]
        lateral_offset = (xy_delta * nominal_lateral_world).sum(dim=1, keepdim=True).clamp(
            -float(cfg.solver.root_lateral_offset_limit_m),
            float(cfg.solver.root_lateral_offset_limit_m),
        )
        parallel_offset = (xy_delta * nominal_axis_world).sum(dim=1, keepdim=True)
        bounded_next = raw_next.clone()
        bounded_next[:, :2] = (
            nominal_next[:, :2]
            + parallel_offset * nominal_axis_world
            + lateral_offset * nominal_lateral_world
        )
        angular_state_guard = 2.0e-4
        roll_pitch_limit = max(
            float(cfg.solver.root_roll_pitch_limit_rad) - angular_state_guard,
            0.0,
        )
        bounded_next[:, 3:5] = bounded_next[:, 3:5].clamp(
            -roll_pitch_limit,
            roll_pitch_limit,
        )
        yaw_error = torch.atan2(
            torch.sin(bounded_next[:, 5] - nominal_next[:, 5]),
            torch.cos(bounded_next[:, 5] - nominal_next[:, 5]),
        ).clamp(
            -max(float(cfg.solver.root_yaw_error_limit_rad) - angular_state_guard, 0.0),
            max(float(cfg.solver.root_yaw_error_limit_rad) - angular_state_guard, 0.0),
        )
        bounded_next[:, 5] = nominal_next[:, 5] + yaw_error

        corrected = projected[:, node].clone()
        world_velocity = (bounded_next[:, :2] - actual_state[:, :2]) / dt
        yaw = actual_state[:, 5]
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        corrected[:, 0] = cosine * world_velocity[:, 0] + sine * world_velocity[:, 1]
        corrected[:, 1] = -sine * world_velocity[:, 0] + cosine * world_velocity[:, 1]
        desired_rpy_rate = (bounded_next[:, 3:6] - actual_state[:, 3:6]) / dt
        roll = actual_state[:, 3]
        pitch = actual_state[:, 4]
        sin_roll = torch.sin(roll)
        cos_roll = torch.cos(roll)
        cos_pitch = torch.cos(pitch).clamp_min(1.0e-4)
        tan_pitch = torch.sin(pitch) / cos_pitch
        mapping = torch.stack(
            (
                torch.stack((torch.ones_like(roll), sin_roll * tan_pitch, cos_roll * tan_pitch), dim=1),
                torch.stack((torch.zeros_like(roll), cos_roll, -sin_roll), dim=1),
                torch.stack((torch.zeros_like(roll), sin_roll / cos_pitch, cos_roll / cos_pitch), dim=1),
            ),
            dim=1,
        )
        corrected[:, 3:6] = torch.linalg.solve_ex(
            mapping,
            desired_rpy_rate.unsqueeze(-1),
            check_errors=False,
        )[0].squeeze(-1)
        corrected[:, 3:5] = corrected[:, 3:5].clamp(
            -float(cfg.solver.root_roll_pitch_rate_limit_rps),
            float(cfg.solver.root_roll_pitch_rate_limit_rps),
        )
        corrected[:, 5] = (
            corrected[:, 5] - command[:, 2]
        ).clamp(
            -float(cfg.solver.root_yaw_rate_error_limit_rps),
            float(cfg.solver.root_yaw_rate_error_limit_rps),
        ) + command[:, 2]
        corrected_next = kinematic_step(actual_state, corrected, dt=dt)
        corrected_rpy_rate = (corrected_next[:, 3:5] - actual_state[:, 3:5]) / dt
        corrected_rate_max = corrected_rpy_rate.abs().amax(dim=1)
        euler_rate_guard = max(
            float(cfg.solver.root_roll_pitch_rate_limit_rps) - 0.01,
            0.0,
        )
        angular_scale = (
            corrected_rate_max.new_full((), euler_rate_guard)
            / corrected_rate_max.clamp_min(euler_rate_guard)
        )
        corrected[:, 3:6] = corrected[:, 3:6] * angular_scale.unsqueeze(-1)
        bounded_controls.append(corrected)
        actual_state = kinematic_step(actual_state, corrected, dt=dt)
        nominal_state = nominal_next
    return torch.stack(bounded_controls, dim=1)


def _candidate_x1_collision_constraints(
    rollout: JointMpcRollout,
    terrain_field,
    cfg: JointMpcRtiCfg,
    contact_x0: Tensor,
    contact_x1: Tensor,
) -> tuple[Tensor, Tensor]:
    batch = int(rollout.state.shape[0])
    state_x1 = rollout.state[:, 1]
    foot_jacobian = foot_jacobian_leg(
        state_x1[:, :3], state_x1[:, 3:6], state_x1[:, 6:]
    ).reshape(batch, 4, 1, 3, 3)
    full_knee_jacobian = complete_knee_jacobian(
        state_x1[:, :3], state_x1[:, 3:6], state_x1[:, 6:]
    )
    knee_jacobian = torch.stack(
        tuple(
            full_knee_jacobian[:, leg, :, 6 + 3 * leg : 9 + 3 * leg]
            for leg in range(4)
        ),
        dim=1,
    ).unsqueeze(2)
    link_jacobian = link_sample_jacobians(
        state_x1[:, :3], state_x1[:, 3:6], state_x1[:, 6:]
    )
    point_jacobian = torch.cat(
        (
            foot_jacobian,
            knee_jacobian,
            link_jacobian.calf_samples.reshape(batch, 4, 3, 3, 3),
            link_jacobian.thigh_samples.reshape(batch, 4, 3, 3, 3),
        ),
        dim=2,
    )
    positions = torch.cat(
        (
            rollout.foot_pos_w[:, 1, :, None],
            rollout.knee_pos_w[:, 1, :, None],
            rollout.shank_samples_w[:, 1],
            rollout.thigh_samples_w[:, 1],
        ),
        dim=2,
    )
    query = _query_world(terrain_field, positions.reshape(batch, 32, 3), cfg)
    distance = query.small_distance_m.reshape(batch, 4, 8)
    height = query.height_w.reshape(batch, 4, 8)
    sdf_gradient = query.small_gradient_w.reshape(batch, 4, 8, 2)
    radius_values = (
        (float(cfg.gait.foot_collision_radius),)
        + (float(cfg.gait.knee_collision_radius),)
        + (float(cfg.gait.calf_collision_radius),) * 3
        + (float(cfg.gait.thigh_collision_radius),) * 3
    )
    radii = constant_like(
        positions,
        "candidate_collision_radii_" + "_".join(map(str, radius_values)),
        radius_values,
    ).view(1, 1, 8)
    horizontal = distance - radii - float(cfg.gait.small_collision_margin_xy)
    vertical = (
        positions[..., 2]
        - height
        - radii
        - float(cfg.gait.small_collision_margin_z)
    )
    contact = torch.as_tensor(contact_x1, dtype=torch.bool, device=positions.device)
    use_horizontal = torch.where(
        torch.logical_not(contact).unsqueeze(-1),
        torch.zeros_like(horizontal, dtype=torch.bool),
        horizontal >= vertical,
    )
    point_clearance = torch.maximum(horizontal, vertical)
    horizontal_jacobian = torch.einsum(
        "blsd,blsdj->blsj", sdf_gradient, point_jacobian[..., :2, :]
    )
    point_clearance_jacobian = torch.where(
        use_horizontal.unsqueeze(-1), horizontal_jacobian, point_jacobian[..., 2, :]
    )
    landing_clearance = torch.where(
        contact,
        distance[:, :, 0] - float(cfg.gait.small_touchdown_margin),
        torch.ones_like(distance[:, :, 0]),
    )
    landing_jacobian = torch.where(
        contact.unsqueeze(-1), horizontal_jacobian[:, :, 0], torch.zeros_like(horizontal_jacobian[:, :, 0])
    )
    continuing_stance = torch.logical_and(
        torch.as_tensor(contact_x0, dtype=torch.bool, device=positions.device), contact
    )
    touchdown = torch.logical_and(torch.logical_not(continuing_stance), contact)
    point_clearance = torch.where(
        touchdown.unsqueeze(-1),
        point_clearance,
        torch.ones_like(point_clearance),
    )
    point_clearance_jacobian = torch.where(
        touchdown.unsqueeze(-1).unsqueeze(-1),
        point_clearance_jacobian,
        torch.zeros_like(point_clearance_jacobian),
    )
    landing_clearance = torch.where(continuing_stance, torch.ones_like(landing_clearance), landing_clearance)
    landing_jacobian = torch.where(
        continuing_stance.unsqueeze(-1), torch.zeros_like(landing_jacobian), landing_jacobian
    )
    return torch.cat((point_clearance, landing_clearance.unsqueeze(-1)), dim=2), torch.cat(
        (point_clearance_jacobian, landing_jacobian.unsqueeze(2)), dim=2
    )


def _restore_candidate_collision_feasibility(
    measured_state: JointMpcRtiState,
    candidate_control: Tensor,
    terrain_field,
    cfg: JointMpcRtiCfg,
    contact_x0: Tensor,
    contact_x1: Tensor,
) -> tuple[Tensor, JointMpcRollout]:
    control = torch.as_tensor(candidate_control)
    original_control = control.clone()
    best_control = control.clone()
    best_violation = control.new_full((control.shape[0],), float("inf"))
    best_rollout: JointMpcRollout | None = None
    for iteration in range(6):
        rollout = rollout_controls(
            measured_state,
            control,
            dt=float(cfg.runtime.dt),
            compile_kernels=bool(cfg.solver.compile_kernels),
        )
        clearance, jacobian = _candidate_x1_collision_constraints(
            rollout, terrain_field, cfg, contact_x0, contact_x1
        )
        violation = torch.relu(-clearance).amax(dim=(1, 2))
        improves = violation < best_violation
        best_violation = torch.where(improves, violation, best_violation)
        best_control = torch.where(improves[:, None, None], control, best_control)
        if best_rollout is None:
            best_rollout = rollout
        else:
            best_rollout = JointMpcRollout(
                state=torch.where(improves[:, None, None], rollout.state, best_rollout.state),
                control=torch.where(improves[:, None, None], rollout.control, best_rollout.control),
                foot_pos_w=torch.where(improves[:, None, None, None], rollout.foot_pos_w, best_rollout.foot_pos_w),
                knee_pos_w=torch.where(improves[:, None, None, None], rollout.knee_pos_w, best_rollout.knee_pos_w),
                shank_samples_w=torch.where(
                    improves[:, None, None, None, None], rollout.shank_samples_w, best_rollout.shank_samples_w
                ),
                thigh_samples_w=torch.where(
                    improves[:, None, None, None, None], rollout.thigh_samples_w, best_rollout.thigh_samples_w
                ),
                body_samples_w=torch.where(
                    improves[:, None, None, None], rollout.body_samples_w, best_rollout.body_samples_w
                ),
            )
        if iteration == 5:
            break
        correction = _minimum_norm_leg_correction(
            clearance,
            jacobian,
            max_norm=float(cfg.solver.joint_trust_scale),
        )
        control = control.clone()
        joint_control_limit = (
            float(cfg.gait.max_nominal_joint_velocity)
            + float(cfg.solver.joint_direction_limit)
        )
        control[:, 0, 6:] = (
            control[:, 0, 6:] + (correction / float(cfg.runtime.dt)).reshape(-1, 12)
        ).clamp(-joint_control_limit, joint_control_limit)
        control = _enforce_joint_position_limits(measured_state, control, cfg)
    assert best_rollout is not None
    hold_control = original_control.clone()
    hold_control[:, 0] = 0.0
    hold_rollout = rollout_controls(
        measured_state,
        hold_control,
        dt=float(cfg.runtime.dt),
        compile_kernels=bool(cfg.solver.compile_kernels),
    )
    best_exact_violation = _small_link_collision_violation(
        best_rollout,
        terrain_field,
        cfg,
        contact_x1,
    )
    hold_exact_violation = _small_link_collision_violation(
        hold_rollout,
        terrain_field,
        cfg,
        contact_x1,
    )
    unresolved = best_exact_violation > 0.0
    use_hold = torch.logical_and(
        unresolved,
        hold_exact_violation < best_exact_violation,
    )
    best_control = torch.where(use_hold[:, None, None], hold_control, best_control)
    best_rollout = JointMpcRollout(
        state=torch.where(use_hold[:, None, None], hold_rollout.state, best_rollout.state),
        control=torch.where(use_hold[:, None, None], hold_rollout.control, best_rollout.control),
        foot_pos_w=torch.where(use_hold[:, None, None, None], hold_rollout.foot_pos_w, best_rollout.foot_pos_w),
        knee_pos_w=torch.where(use_hold[:, None, None, None], hold_rollout.knee_pos_w, best_rollout.knee_pos_w),
        shank_samples_w=torch.where(
            use_hold[:, None, None, None, None], hold_rollout.shank_samples_w, best_rollout.shank_samples_w
        ),
        thigh_samples_w=torch.where(
            use_hold[:, None, None, None, None], hold_rollout.thigh_samples_w, best_rollout.thigh_samples_w
        ),
        body_samples_w=torch.where(
            use_hold[:, None, None, None], hold_rollout.body_samples_w, best_rollout.body_samples_w
        ),
    )
    return best_control, best_rollout


def _small_link_collision_violation(
    rollout: JointMpcRollout,
    terrain_field,
    cfg: JointMpcRtiCfg,
    contact_x1: Tensor,
) -> Tensor:
    """Return the exact maximum x1 small-object sphere penetration."""
    batch = int(rollout.state.shape[0])
    points = torch.cat(
        (
            rollout.body_samples_w[:, 1],
            rollout.foot_pos_w[:, 1],
            rollout.knee_pos_w[:, 1],
            rollout.shank_samples_w[:, 1].reshape(batch, 12, 3),
            rollout.thigh_samples_w[:, 1].reshape(batch, 12, 3),
        ),
        dim=1,
    )
    query = _query_world(terrain_field, points, cfg)
    def penetration(position: Tensor, distance: Tensor, height: Tensor, radius: float) -> Tensor:
        collision = _sphere_link_collision(position, distance, height, radius=radius)
        return torch.where(collision, float(radius) - distance, torch.zeros_like(distance))

    foot_penetration = penetration(
        rollout.foot_pos_w[:, 1], query.small_distance_m[:, 9:13], query.height_w[:, 9:13],
        cfg.gait.foot_collision_radius,
    ).amax(dim=1)
    knee_penetration = penetration(
        rollout.knee_pos_w[:, 1], query.small_distance_m[:, 13:17], query.height_w[:, 13:17],
        cfg.gait.knee_collision_radius,
    ).amax(dim=1)
    calf_penetration = penetration(
        rollout.shank_samples_w[:, 1],
        query.small_distance_m[:, 17:29].reshape(batch, 4, 3),
        query.height_w[:, 17:29].reshape(batch, 4, 3),
        cfg.gait.calf_collision_radius,
    ).amax(dim=(1, 2))
    thigh_penetration = penetration(
        rollout.thigh_samples_w[:, 1],
        query.small_distance_m[:, 29:41].reshape(batch, 4, 3),
        query.height_w[:, 29:41].reshape(batch, 4, 3),
        cfg.gait.thigh_collision_radius,
    ).amax(dim=(1, 2))
    base_penetration = penetration(
        rollout.body_samples_w[:, 1],
        query.small_distance_m[:, :9],
        query.height_w[:, :9],
        0.0,
    ).amax(dim=1)
    contact = torch.as_tensor(contact_x1, dtype=torch.bool, device=points.device)
    stance_on_small = torch.where(
        contact,
        torch.relu(-query.small_distance_m[:, 9:13]),
        torch.zeros_like(query.small_distance_m[:, 9:13]),
    ).amax(dim=1)
    return torch.maximum(
        stance_on_small,
        torch.maximum(
            base_penetration,
            torch.maximum(
                foot_penetration,
                torch.maximum(knee_penetration, torch.maximum(calf_penetration, thigh_penetration)),
            ),
        ),
    )


def _recovery_ground_safe_distance(cfg: JointMpcRtiCfg) -> float:
    """Return the SDF distance after which recovery grounding may begin."""
    return max(
        float(cfg.gait.small_touchdown_margin),
        float(cfg.gait.foot_collision_radius)
        + float(cfg.gait.small_collision_margin_xy),
    )


def _split_stance_and_recovery_leg_clearance(
    *,
    measured_leg_clearance: Tensor,
    nominal_leg_clearance: Tensor,
    stance_lookahead_margin: float,
) -> tuple[Tensor, Tensor]:
    """Keep stance preview conservative without blocking safe measured recovery."""
    measured = torch.as_tensor(measured_leg_clearance)
    nominal = torch.as_tensor(
        nominal_leg_clearance,
        dtype=measured.dtype,
        device=measured.device,
    )
    if measured.shape != nominal.shape:
        raise ValueError("measured and nominal leg clearances must have matching shapes")
    stance = torch.minimum(measured, nominal + float(stance_lookahead_margin))
    return stance, measured


def _recovery_grounding_active_mask(
    *,
    recovery_state: Tensor,
    contact_state: Tensor,
    map_valid: Tensor,
    foot_small_distance_m: Tensor,
    leg_landing_clearance_m: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Enable recovery grounding only at a full-leg-safe landing location."""
    distance = torch.as_tensor(foot_small_distance_m)
    recovery = torch.as_tensor(
        recovery_state, dtype=torch.bool, device=distance.device
    )
    contact = torch.as_tensor(
        contact_state, dtype=torch.bool, device=distance.device
    )
    valid = torch.as_tensor(map_valid, dtype=torch.bool, device=distance.device)
    clearance = torch.as_tensor(
        leg_landing_clearance_m, dtype=distance.dtype, device=distance.device
    )
    if any(
        value.shape != distance.shape
        for value in (recovery, contact, valid, clearance)
    ):
        raise ValueError("recovery grounding tensors must have matching shapes")
    return (
        recovery
        & torch.logical_not(contact)
        & valid
        & (distance >= _recovery_ground_safe_distance(cfg))
        & (clearance >= float(cfg.gait.small_collision_margin_xy))
    )


def _recovery_exit_clearance(
    *,
    foot_small_distance_m: Tensor,
    leg_landing_clearance_m: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Return the limiting nonnegative clearance for recovery exit."""
    foot_distance = torch.as_tensor(foot_small_distance_m)
    leg_clearance = torch.as_tensor(
        leg_landing_clearance_m,
        dtype=foot_distance.dtype,
        device=foot_distance.device,
    )
    if foot_distance.shape != leg_clearance.shape:
        raise ValueError("foot and leg recovery clearances must have matching shapes")
    return torch.minimum(
        foot_distance - float(cfg.gait.small_touchdown_margin),
        leg_clearance - float(cfg.gait.small_collision_margin_xy),
    )


def _leg_small_horizontal_clearance(
    foot_pos_w: Tensor,
    knee_pos_w: Tensor,
    shank_samples_w: Tensor,
    thigh_samples_w: Tensor,
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Return per-leg worst foot/knee/calf/thigh horizontal clearance."""
    foot_pos = torch.as_tensor(foot_pos_w)
    batch = int(foot_pos.shape[0])
    points = torch.cat(
        (
            foot_pos,
            torch.as_tensor(knee_pos_w),
            torch.as_tensor(shank_samples_w).reshape(batch, 12, 3),
            torch.as_tensor(thigh_samples_w).reshape(batch, 12, 3),
        ),
        dim=1,
    )
    query = _query_world(terrain_field, points, cfg)
    foot = query.small_distance_m[:, :4] - float(cfg.gait.foot_collision_radius)
    knee = query.small_distance_m[:, 4:8] - float(cfg.gait.knee_collision_radius)
    calf = (
        query.small_distance_m[:, 8:20].reshape(batch, 4, 3).amin(dim=2)
        - float(cfg.gait.calf_collision_radius)
    )
    thigh = (
        query.small_distance_m[:, 20:32].reshape(batch, 4, 3).amin(dim=2)
        - float(cfg.gait.thigh_collision_radius)
    )
    return torch.minimum(foot, torch.minimum(knee, torch.minimum(calf, thigh)))


def _recovery_landing_constraint_violation(
    foot_pos_w: Tensor,
    foot_query: JointMpcTerrainQuery,
    *,
    contact_x1: Tensor,
    recovery_state: Tensor,
    cfg: JointMpcRtiCfg,
    leg_landing_clearance_m: Tensor | None = None,
) -> Tensor:
    """Return exact x1 recovery SDF-exit and grounding violation."""
    foot = torch.as_tensor(foot_pos_w)
    contact = torch.as_tensor(contact_x1, dtype=torch.bool, device=foot.device)
    recovery = torch.as_tensor(recovery_state, dtype=torch.bool, device=foot.device)
    distance = torch.as_tensor(
        foot_query.small_distance_m, dtype=foot.dtype, device=foot.device
    )
    height = torch.as_tensor(foot_query.height_w, dtype=foot.dtype, device=foot.device)
    valid = torch.as_tensor(foot_query.valid, dtype=torch.bool, device=foot.device)
    active = recovery & torch.logical_not(contact) & valid
    landing_clearance = (
        torch.full_like(distance, torch.inf)
        if leg_landing_clearance_m is None
        else torch.as_tensor(
            leg_landing_clearance_m, dtype=foot.dtype, device=foot.device
        )
    )
    exit_clearance = _recovery_exit_clearance(
        foot_small_distance_m=distance,
        leg_landing_clearance_m=landing_clearance,
        cfg=cfg,
    )
    sdf_exit_violation = torch.relu(-exit_clearance)
    ground_active = _recovery_grounding_active_mask(
        recovery_state=recovery,
        contact_state=contact,
        map_valid=valid,
        foot_small_distance_m=distance,
        leg_landing_clearance_m=landing_clearance,
        cfg=cfg,
    )
    ground_error = (
        foot[..., 2] - height - float(cfg.gait.foot_contact_offset)
    ).abs()
    per_leg = torch.maximum(
        torch.where(
            active,
            sdf_exit_violation,
            torch.zeros_like(sdf_exit_violation),
        ),
        torch.where(ground_active, ground_error, torch.zeros_like(ground_error)),
    )
    return per_leg.amax(dim=1)


def _command_conditioned_foot_targets(
    nominal_foot_pos_w: Tensor,
    nominal_state: Tensor,
    command_body: Tensor,
    contact_state: Tensor,
    phase_step: Tensor,
    cfg: JointMpcRtiCfg,
    progress_scale: Tensor | None = None,
) -> Tensor:
    target = torch.as_tensor(nominal_foot_pos_w).clone()
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=target.device)
    batch, nodes = int(target.shape[0]), int(target.shape[1])
    half_cycle = int(cfg.gait.half_cycle_steps)
    segment_age = _contact_segment_age(
        contact,
        phase_step,
        half_cycle_steps=half_cycle,
    ).to(dtype=target.dtype)
    progress = (segment_age / float(max(half_cycle - 1, 1))).clamp(0.0, 1.0)
    swing = torch.logical_not(contact)
    touchdown = torch.zeros_like(contact)
    touchdown[:, 1:] = torch.logical_and(contact[:, 1:], torch.logical_not(contact[:, :-1]))
    phase_progress = torch.where(touchdown, torch.ones_like(progress), progress)
    active = torch.logical_or(swing, touchdown)
    command_world = body_linear_velocity_to_world(command_body[:, :2], nominal_state[:, 0, 5])
    step_offset = command_world[:, None, None, :] * (
        float(cfg.gait.command_touchdown_stride_scale)
        * float(half_cycle)
        * float(cfg.runtime.dt)
    )
    swing_anchor = target[:, 0, :, :2]
    anchors: list[Tensor] = [swing_anchor]
    for node in range(1, nodes):
        liftoff = torch.logical_and(torch.logical_not(contact[:, node]), contact[:, node - 1])
        swing_anchor = torch.where(liftoff.unsqueeze(-1), target[:, node - 1, :, :2], swing_anchor)
        anchors.append(swing_anchor)
    anchor_xy = torch.stack(anchors, dim=1)
    commanded_xy = anchor_xy + step_offset * phase_progress.unsqueeze(-1)
    initial_phase = torch.as_tensor(
        phase_step,
        dtype=torch.long,
        device=target.device,
    )
    if initial_phase.ndim == 1:
        initial_phase = initial_phase[:, None].expand(-1, 4)
    extension_steps = (initial_phase - (half_cycle - 1)).clamp_min(0).to(target.dtype)
    initial_segment = torch.cumprod(
        (contact == contact[:, :1]).to(torch.long),
        dim=1,
    ).to(torch.bool)
    extension_offset = (
        command_world[:, None, None, :]
        * float(cfg.runtime.dt)
        * extension_steps[:, None, :, None]
    )
    commanded_xy = commanded_xy + (
        extension_offset
        * initial_segment.unsqueeze(-1).to(target.dtype)
        * active.unsqueeze(-1).to(target.dtype)
    )
    if progress_scale is not None:
        scale = torch.as_tensor(progress_scale, dtype=target.dtype, device=target.device)
        nominal_joint = constant_like(target, "recovery_nominal_joint_pos", cfg.gait.nominal_joint_pos)
        nominal_foot = go2_foot_pos(
            nominal_state[..., :3].reshape(batch * nodes, 3),
            nominal_state[..., 3:6].reshape(batch * nodes, 3),
            nominal_joint.view(1, 12).expand(batch * nodes, -1),
        ).reshape(batch, nodes, 4, 3)
        recovery_weight = (1.0 - scale).clamp(0.0, 1.0)[:, None, None, None]
        commanded_xy = commanded_xy + recovery_weight * (nominal_foot[..., :2] - commanded_xy)
    target[..., :2] = torch.where(active.unsqueeze(-1), commanded_xy, target[..., :2])
    return target


def _sdf_corrected_foot_targets(
    foot_target_w: Tensor,
    contact_state: Tensor,
    foot_query: JointMpcTerrainQuery,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    target = torch.as_tensor(foot_target_w).clone()
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=target.device)
    batch, nodes = int(target.shape[0]), int(target.shape[1])
    distance = foot_query.small_distance_m.reshape(batch, nodes, 4)
    gradient = foot_query.small_gradient_w.reshape(batch, nodes, 4, 2)
    touchdown = torch.zeros_like(contact)
    touchdown[:, 1:] = torch.logical_and(contact[:, 1:], torch.logical_not(contact[:, :-1]))
    active = torch.logical_or(torch.logical_not(contact), touchdown).to(dtype=target.dtype)
    temperature = float(cfg.gait.small_collision_temperature)
    correction = temperature * torch.nn.functional.softplus(
        (float(cfg.gait.small_touchdown_margin) - distance) / temperature
    )
    target[..., :2] = target[..., :2] + (
        float(cfg.gait.sdf_target_correction_scale)
        * active.unsqueeze(-1)
        * correction.unsqueeze(-1)
        * gradient
    )
    # Future stance nodes are soft-preview terms (only x1 has a hard FK
    # equality), so keep their vertical target on the queried terrain plane.
    # This prevents a nominal swing-height touchdown from being published as
    # stance on the next horizon node.
    terrain_swing_height = foot_query.height_w.reshape(batch, nodes, 4) + float(
        cfg.gait.nominal_swing_clearance
    )
    target[..., 2] = torch.where(
        torch.logical_not(contact),
        torch.maximum(target[..., 2], terrain_swing_height),
        target[..., 2],
    )
    proximity = torch.sigmoid(
        (float(cfg.gait.small_collision_influence_radius) - distance) / temperature
    )
    required_height = foot_query.height_w.reshape(batch, nodes, 4) + max(
        float(cfg.gait.nominal_swing_clearance),
        float(cfg.gait.small_semantic_clearance),
    )
    lift = temperature * torch.nn.functional.softplus(
        (required_height - target[..., 2]) / temperature
    )
    target[..., 2] = target[..., 2] + active * proximity * lift
    return target


def _desired_control(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    contact: Tensor,
    phase_step: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    joint_target = _nominal_joint_target(
        contact,
        phase_step,
        command_body,
        cfg,
        dtype=measured_state.root_pos_w.dtype,
    )
    return _control_from_joint_target(measured_state, command_body, joint_target, cfg), joint_target


def _control_from_joint_target(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    joint_target: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    carry_clipped_error: bool = False,
) -> Tensor:
    batch = measured_state.batch_size
    horizon = int(cfg.runtime.horizon_steps)
    dt = float(cfg.runtime.dt)
    velocity_limit = float(cfg.gait.max_nominal_joint_velocity)
    if carry_clipped_error:
        tracked_joint = measured_state.joint_pos
        joint_velocity_nodes: list[Tensor] = []
        for node in range(horizon):
            joint_velocity = ((joint_target[:, node + 1] - tracked_joint) / dt).clamp(
                -velocity_limit,
                velocity_limit,
            )
            joint_velocity_nodes.append(joint_velocity)
            tracked_joint = tracked_joint + dt * joint_velocity
        joint_velocity = torch.stack(joint_velocity_nodes, dim=1)
    else:
        joint_velocity = (joint_target[:, 1:] - joint_target[:, :-1]) / dt
        joint_velocity[:, 0] = (joint_target[:, 1] - measured_state.joint_pos) / dt
        joint_velocity = joint_velocity.clamp(-velocity_limit, velocity_limit)
    desired = torch.zeros(batch, horizon, 18, dtype=measured_state.root_pos_w.dtype, device=measured_state.device)
    desired[..., :2] = command_body[:, None, :2]
    desired[..., 5] = command_body[:, None, 2]
    desired[..., 6:] = joint_velocity
    return desired


def _initial_control(
    desired_control: Tensor,
    solver_state: JointMpcRtiSolverState | None,
    *,
    joint_delta_limit: float | None = None,
) -> Tensor:
    if solver_state is None:
        return desired_control.clone()
    previous = torch.as_tensor(solver_state.control, dtype=desired_control.dtype, device=desired_control.device)
    if previous.shape != desired_control.shape:
        return desired_control.clone()
    shifted = torch.cat((previous[:, 1:], previous[:, -1:]), dim=1)
    shifted_joint = shifted[..., 6:]
    if joint_delta_limit is not None:
        joint_delta = shifted_joint - desired_control[..., 6:]
        shifted_joint = desired_control[..., 6:] + joint_delta.clamp(
            -float(joint_delta_limit),
            float(joint_delta_limit),
        )
    return torch.cat((desired_control[..., :6], shifted_joint), dim=-1)


def _blend_recovery_joint_control(
    warm_control: Tensor,
    desired_control: Tensor,
    progress_scale: Tensor,
    recovery_state: Tensor | None = None,
) -> Tensor:
    """Continuously release shifted joint warm starts as touchdown recovery becomes active."""
    warm = torch.as_tensor(warm_control).clone()
    desired = torch.as_tensor(desired_control, dtype=warm.dtype, device=warm.device)
    scale = torch.as_tensor(progress_scale, dtype=warm.dtype, device=warm.device)
    recovery = (1.0 - scale).clamp(0.0, 1.0)[:, None, None]
    if recovery_state is not None:
        per_leg = torch.as_tensor(
            recovery_state,
            dtype=warm.dtype,
            device=warm.device,
        )
        if per_leg.shape != (int(warm.shape[0]), 4):
            raise ValueError("recovery_state must have shape [B,4]")
        recovery = torch.maximum(
            recovery,
            per_leg.repeat_interleave(3, dim=1)[:, None, :],
        )
    warm[..., 6:] = warm[..., 6:] + recovery * (desired[..., 6:] - warm[..., 6:])
    return warm


def _enforce_recovery_landing(
    measured_state: JointMpcRtiState,
    control: Tensor,
    terrain_field: JointMpcTerrainField,
    recovery_state: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Project safe recovery feet toward the terrain without confirming contact."""
    projected = torch.as_tensor(control).clone()
    recovery = torch.as_tensor(recovery_state, dtype=torch.bool, device=projected.device)
    if recovery.shape != (measured_state.batch_size, 4):
        raise ValueError("recovery_state must have shape [B,4]")
    identity_epsilon = 1.0e-9
    joint_control_limit = float(cfg.gait.max_nominal_joint_velocity) + float(
        cfg.solver.joint_direction_limit
    )
    for _ in range(4):
        next_state = kinematic_step(measured_state.as_vector(), projected[:, 0], dt=float(cfg.runtime.dt))
        foot = go2_foot_pos(next_state[:, :3], next_state[:, 3:6], next_state[:, 6:])
        query = _query_world(terrain_field, foot, cfg)
        safety = torch.sigmoid(
            (
                query.small_distance_m - float(cfg.gait.small_safe_landing_margin)
            )
            / float(cfg.gait.small_collision_temperature)
        )
        height_error = (
            query.height_w + float(cfg.gait.foot_contact_offset) - foot[..., 2]
        ) * safety
        jacobian_z = foot_jacobian_leg(
            next_state[:, :3], next_state[:, 3:6], next_state[:, 6:]
        )[..., 2, :]
        correction = (
            height_error.unsqueeze(-1)
            * jacobian_z
            / jacobian_z.square().sum(dim=-1, keepdim=True).clamp_min(identity_epsilon)
            / float(cfg.runtime.dt)
        )
        joint_velocity = projected[:, 0, 6:].reshape(measured_state.batch_size, 4, 3)
        joint_velocity = torch.where(
            recovery.unsqueeze(-1),
            joint_velocity + correction,
            joint_velocity,
        )
        projected[:, 0, 6:] = joint_velocity.reshape(measured_state.batch_size, 12).clamp(
            -joint_control_limit,
            joint_control_limit,
        )
    return _enforce_joint_position_limits(measured_state, projected, cfg)


def _enforce_first_stance_equality(
    measured_state: JointMpcRtiState,
    control: Tensor,
    command_body: Tensor,
    contact_x0: Tensor,
    contact_x1: Tensor,
    stance_anchor_x1: Tensor,
    cfg: JointMpcRtiCfg,
    stance_surface_safety: Tensor | None = None,
    confirmed_touchdown: Tensor | None = None,
) -> Tensor:
    """Eliminate first-step stance position error inside every rollout candidate."""
    projected = torch.as_tensor(control).clone()
    command = torch.as_tensor(command_body, dtype=projected.dtype, device=projected.device)
    zero_translation = torch.linalg.vector_norm(command[:, :2], dim=-1) <= float(
        cfg.gait.zero_translation_command_deadband
    )
    projected[:, 0, :2] = torch.where(
        zero_translation.unsqueeze(-1),
        torch.zeros_like(projected[:, 0, :2]),
        projected[:, 0, :2],
    )
    projected[:, 0, 5] = torch.where(
        torch.logical_and(zero_translation, command[:, 2].abs() <= float(cfg.gait.zero_translation_command_deadband)),
        torch.zeros_like(projected[:, 0, 5]),
        projected[:, 0, 5],
    )
    initial_root_control = projected[:, 0, :6].clone()
    initial_joint_control = projected[:, 0, 6:].clone()
    contact = torch.as_tensor(contact_x1, dtype=torch.bool, device=projected.device)
    if confirmed_touchdown is not None:
        contact = torch.logical_or(
            contact,
            torch.as_tensor(confirmed_touchdown, dtype=torch.bool, device=projected.device),
        )
    anchor = torch.as_tensor(stance_anchor_x1, dtype=projected.dtype, device=projected.device)
    surface_safety = (
        projected.new_ones((measured_state.batch_size, 4))
        if stance_surface_safety is None
        else torch.as_tensor(stance_surface_safety, dtype=projected.dtype, device=projected.device)
    )
    batch = measured_state.batch_size
    joint_lower = constant_like(projected, "first_step_joint_lower", (-1.0472, -0.6632, -2.721) * 4)
    joint_upper = constant_like(projected, "first_step_joint_upper", (1.0472, 2.966, -0.837) * 4)
    joint_lower = joint_lower + float(cfg.solver.joint_position_safety_margin_rad)
    joint_upper = joint_upper - float(cfg.solver.joint_position_safety_margin_rad)
    identity = torch.eye(3, dtype=projected.dtype, device=projected.device).view(1, 1, 3, 3)

    def clamp_first_joint_position() -> None:
        lower_velocity = (joint_lower - measured_state.joint_pos) / float(cfg.runtime.dt)
        upper_velocity = (joint_upper - measured_state.joint_pos) / float(cfg.runtime.dt)
        projected[:, 0, 6:] = torch.maximum(
            torch.minimum(projected[:, 0, 6:], upper_velocity),
            lower_velocity,
        )

    recovery_projection_iterations = 4

    def correct_joints() -> None:
        next_state = kinematic_step(measured_state.as_vector(), projected[:, 0], dt=float(cfg.runtime.dt))
        next_foot = go2_foot_pos(next_state[:, :3], next_state[:, 3:6], next_state[:, 6:])
        correction_scale = torch.cat(
            (
                torch.ones_like(surface_safety).unsqueeze(-1).expand(-1, -1, 2),
                surface_safety.unsqueeze(-1),
            ),
            dim=-1,
        )
        raw_error = anchor - next_foot
        raw_error[..., 2] = raw_error[..., 2].clamp(
            -float(cfg.gait.stance_ground_recovery_step_m) / float(recovery_projection_iterations),
            float(cfg.gait.stance_ground_recovery_step_m) / float(recovery_projection_iterations),
        )
        error = raw_error * correction_scale
        local_jacobian = foot_jacobian_leg(next_state[:, :3], next_state[:, 3:6], next_state[:, 6:])
        normal = torch.matmul(local_jacobian, local_jacobian.transpose(-1, -2)) + 1.0e-6 * identity
        correction = torch.matmul(
            local_jacobian.transpose(-1, -2),
            torch.linalg.solve_ex(normal, error.unsqueeze(-1), check_errors=False)[0],
        ).squeeze(-1) / float(cfg.runtime.dt)
        joint_velocity = projected[:, 0, 6:].reshape(batch, 4, 3)
        joint_velocity = torch.where(contact.unsqueeze(-1), joint_velocity + correction, joint_velocity)
        joint_control_limit = (
            float(cfg.gait.max_nominal_joint_velocity)
            + float(cfg.solver.joint_direction_limit)
        )
        projected[:, 0, 6:] = joint_velocity.reshape(batch, 12).clamp(
            -joint_control_limit,
            joint_control_limit,
        )
        clamp_first_joint_position()

    def run_joint_projection() -> None:
        for _ in range(recovery_projection_iterations):
            correct_joints()

    tolerance = float(cfg.solver.stance_equality_tolerance_m)
    root_scale = projected.new_ones((batch,))
    for _ in range(4):
        retry_root = initial_root_control.clone()
        retry_root[:, :2] = retry_root[:, :2] * root_scale.unsqueeze(-1)
        retry_root[:, 3:6] = retry_root[:, 3:6] * root_scale.unsqueeze(-1)
        command_xy = command[:, :2]
        command_speed = torch.linalg.vector_norm(command_xy, dim=1, keepdim=True)
        command_axis = command_xy / command_speed.clamp_min(1.0e-6)
        retry_progress = (retry_root[:, :2] * command_axis).sum(dim=1, keepdim=True)
        initial_progress = (initial_root_control[:, :2] * command_axis).sum(dim=1, keepdim=True)
        release_floor = torch.minimum(
            initial_progress.clamp_min(0.0),
            torch.minimum(
                command_speed,
                command_speed.new_full((), float(cfg.gait.startup_root_release_velocity)),
            ),
        )
        bounded_progress = torch.maximum(retry_progress, release_floor)
        released_xy = retry_root[:, :2] + (bounded_progress - retry_progress) * command_axis
        retry_root[:, :2] = torch.where(
            command_speed > float(cfg.gait.zero_translation_command_deadband),
            released_xy,
            retry_root[:, :2],
        )
        projected[:, 0, :6] = retry_root
        projected[:, 0, 6:] = initial_joint_control
        run_joint_projection()
        projected_state = kinematic_step(
            measured_state.as_vector(), projected[:, 0], dt=float(cfg.runtime.dt)
        )
        projected_foot = go2_foot_pos(
            projected_state[:, :3], projected_state[:, 3:6], projected_state[:, 6:]
        )
        stance_residual = torch.where(
            contact,
            torch.linalg.vector_norm(projected_foot - anchor, dim=-1),
            torch.zeros_like(contact, dtype=projected.dtype),
        ).amax(dim=1)
        residual_scale = (tolerance / stance_residual.clamp_min(tolerance)).clamp(0.0, 1.0)
        root_scale = root_scale * residual_scale

    return projected


def _enforce_root_assist_with_stance_equality(
    measured_state: JointMpcRtiState,
    control: Tensor,
    command_body: Tensor,
    contact_x0: Tensor,
    contact_x1: Tensor,
    stance_anchor_x1: Tensor,
    cfg: JointMpcRtiCfg,
    stance_surface_safety: Tensor | None = None,
    confirmed_touchdown: Tensor | None = None,
) -> Tensor:
    """Apply bounded root assistance, then restore the first-step stance equality."""
    projected = _enforce_root_assist_limits(
        measured_state,
        control,
        command_body,
        cfg,
    )
    return _enforce_first_stance_equality(
        measured_state,
        projected,
        command_body,
        contact_x0,
        contact_x1,
        stance_anchor_x1,
        cfg,
        stance_surface_safety=stance_surface_safety,
        confirmed_touchdown=confirmed_touchdown,
    )


def _apply_fk_stance_kkt_constraint(
    measured_state: JointMpcRtiState,
    control: Tensor,
    command_body: Tensor,
    contact_x0: Tensor,
    contact_x1: Tensor,
    stance_anchor_x1: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Solve the first-step FK stance equality over the complete 18-D control."""
    projected = torch.as_tensor(control).clone()
    command = torch.as_tensor(command_body, dtype=projected.dtype, device=projected.device)
    contact = torch.logical_and(
        torch.as_tensor(contact_x0, dtype=torch.bool, device=projected.device),
        torch.as_tensor(contact_x1, dtype=torch.bool, device=projected.device),
    )
    anchor = torch.as_tensor(stance_anchor_x1, dtype=projected.dtype, device=projected.device)
    for _ in range(4):
        next_state = kinematic_step(measured_state.as_vector(), projected[:, 0], dt=float(cfg.runtime.dt))
        foot = go2_foot_pos(next_state[:, :3], next_state[:, 3:6], next_state[:, 6:])
        jacobian = complete_foot_jacobian(
            next_state[:, :3], next_state[:, 3:6], next_state[:, 6:]
        )
        dynamics = dynamics_jacobians(
            measured_state.as_vector(), projected[:, 0], dt=float(cfg.runtime.dt)
        )[1]
        constraint = torch.matmul(jacobian, dynamics.unsqueeze(1)).reshape(
            measured_state.batch_size, 12, 18
        )
        root_active = torch.logical_or(
            torch.linalg.vector_norm(command[:, :2], dim=-1)
            > float(cfg.gait.zero_translation_command_deadband),
            command[:, 2].abs() > float(cfg.gait.zero_translation_command_deadband),
        )
        constraint[..., :6] = constraint[..., :6] * root_active[:, None, None].to(constraint.dtype)
        residual = (foot - anchor).reshape(measured_state.batch_size, 12)
        active = contact.unsqueeze(-1).expand(-1, -1, 3).reshape(measured_state.batch_size, 12)
        constraint = constraint * active.unsqueeze(-1).to(constraint.dtype)
        residual = residual * active.to(residual.dtype)
        inverse_metric = torch.ones(18, dtype=constraint.dtype, device=constraint.device)
        inverse_metric[:2] = float(cfg.solver.root_xy_trust_scale)
        inverse_metric[2:6] = 0.0
        weighted_constraint_t = (
            constraint.transpose(-1, -2) * inverse_metric.view(1, 18, 1)
        )
        normal = torch.matmul(constraint, weighted_constraint_t)
        normal = normal + 1.0e-7 * torch.eye(12, dtype=normal.dtype, device=normal.device).unsqueeze(0)
        correction = -torch.matmul(
            weighted_constraint_t,
            fixed_spd_solve(normal, residual.unsqueeze(-1)),
        ).squeeze(-1)
        projected[:, 0] = projected[:, 0] + correction
    joint_limit = float(cfg.gait.max_nominal_joint_velocity) + float(cfg.solver.joint_direction_limit)
    projected[:, 0, 6:] = projected[:, 0, 6:].clamp(-joint_limit, joint_limit)
    return _enforce_joint_position_limits(measured_state, projected, cfg)


def _apply_fk_contact_kkt(
    measured_state: JointMpcRtiState,
    control: Tensor,
    command_body: Tensor,
    progress_command_body: Tensor,
    terrain_field: JointMpcTerrainField,
    contact_x0: Tensor,
    contact_x1: Tensor,
    stance_anchor_x1: Tensor,
    recovery_landing_x1: Tensor,
    startup_mask: Tensor,
    release_mask: Tensor,
    stance_surface_safety: Tensor,
    confirmed_touchdown: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, JointMpcRollout]:
    """Roll out a solver-produced candidate without post-solve control repair."""
    projected = torch.as_tensor(control).clone()
    rollout = rollout_controls(
        measured_state,
        projected,
        dt=float(cfg.runtime.dt),
        compile_kernels=bool(cfg.solver.compile_kernels),
    )
    return projected, rollout


def _enforce_command_progress_direction(
    control: Tensor,
    command_body: Tensor,
    progress_command_body: Tensor,
) -> Tensor:
    """Prevent the published root from moving opposite the raw command direction."""
    projected = torch.as_tensor(control).clone()
    command = torch.as_tensor(command_body, dtype=projected.dtype, device=projected.device)
    progress_command = torch.as_tensor(
        progress_command_body, dtype=projected.dtype, device=projected.device
    )
    norm = torch.linalg.vector_norm(command[:, :2], dim=-1, keepdim=True)
    axis = command[:, :2] / norm.clamp_min(1.0e-6)
    component = (projected[:, 0, :2] * axis).sum(dim=-1, keepdim=True)
    minimum_progress = 0.25 * torch.linalg.vector_norm(
        progress_command[:, :2], dim=-1, keepdim=True
    )
    bounded = torch.maximum(component, minimum_progress)
    corrected = projected[:, 0, :2] + (bounded - component) * axis
    projected[:, 0, :2] = torch.where(norm > 1.0e-6, corrected, projected[:, 0, :2])
    return projected


def _apply_fk_swing_target_kkt_constraint(
    measured_state: JointMpcRtiState,
    control: Tensor,
    contact_x1: Tensor,
    swing_target_x1: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Track the published swing-foot XY target with joint columns of the FK constraint."""
    projected = torch.as_tensor(control).clone()
    swing = torch.logical_not(
        torch.as_tensor(contact_x1, dtype=torch.bool, device=projected.device)
    )
    target = torch.as_tensor(swing_target_x1, dtype=projected.dtype, device=projected.device)
    identity = torch.eye(8, dtype=projected.dtype, device=projected.device).unsqueeze(0)
    for _ in range(4):
        next_state = kinematic_step(measured_state.as_vector(), projected[:, 0], dt=float(cfg.runtime.dt))
        foot = go2_foot_pos(next_state[:, :3], next_state[:, 3:6], next_state[:, 6:])
        jacobian = complete_foot_jacobian(
            next_state[:, :3], next_state[:, 3:6], next_state[:, 6:]
        )[..., :2, :]
        dynamics = dynamics_jacobians(
            measured_state.as_vector(), projected[:, 0], dt=float(cfg.runtime.dt)
        )[1]
        constraint = torch.matmul(jacobian, dynamics.unsqueeze(1)).reshape(
            measured_state.batch_size, 8, 18
        )
        constraint[..., :6] = 0.0
        active = swing.unsqueeze(-1).expand(-1, -1, 2).reshape(measured_state.batch_size, 8)
        constraint = constraint * active.unsqueeze(-1).to(constraint.dtype)
        residual = (foot[..., :2] - target[..., :2]).reshape(measured_state.batch_size, 8)
        residual = residual * active.to(residual.dtype)
        normal = torch.matmul(constraint, constraint.transpose(-1, -2)) + 1.0e-7 * identity
        correction = -torch.matmul(
            constraint.transpose(-1, -2),
            fixed_spd_solve(normal, residual.unsqueeze(-1)),
        ).squeeze(-1)
        projected[:, 0] = projected[:, 0] + correction
    joint_limit = float(cfg.gait.max_nominal_joint_velocity) + float(cfg.solver.joint_direction_limit)
    projected[:, 0, 6:] = projected[:, 0, 6:].clamp(-joint_limit, joint_limit)
    return _enforce_joint_position_limits(measured_state, projected, cfg)


def _apply_fk_ground_kkt_horizon(
    measured_state: JointMpcRtiState,
    control: Tensor,
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Enforce foot_z >= terrain height at every H30 node through FK rows."""
    projected = torch.as_tensor(control).clone()
    state = measured_state.as_vector()
    identity = torch.eye(4, dtype=projected.dtype, device=projected.device).unsqueeze(0)
    for node in range(int(projected.shape[1])):
        for _ in range(2):
            next_state = kinematic_step(state, projected[:, node], dt=float(cfg.runtime.dt))
            foot = go2_foot_pos(next_state[:, :3], next_state[:, 3:6], next_state[:, 6:])
            query = _query_world(terrain_field, foot, cfg)
            clearance_margin = torch.where(
                query.small_distance_m < 0.02,
                torch.full_like(
                    query.height_w,
                    float(cfg.gait.foot_collision_radius) + 0.005,
                ),
                torch.zeros_like(query.height_w),
            )
            residual = foot[..., 2] - query.height_w - clearance_margin
            active = residual < 0.0
            foot_jacobian = complete_foot_jacobian(
                next_state[:, :3], next_state[:, 3:6], next_state[:, 6:]
            )[..., 2, :]
            dynamics = dynamics_jacobians(
                state, projected[:, node], dt=float(cfg.runtime.dt)
            )[1]
            constraint = torch.matmul(foot_jacobian, dynamics)
            constraint[..., :6] = 0.0
            constraint = constraint * active.unsqueeze(-1).to(constraint.dtype)
            rhs = torch.where(active, -residual, torch.zeros_like(residual))
            normal = torch.matmul(constraint, constraint.transpose(-1, -2)) + 1.0e-7 * identity
            correction = torch.matmul(
                constraint.transpose(-1, -2),
                fixed_spd_solve(normal, rhs.unsqueeze(-1)),
            ).squeeze(-1)
            projected[:, node] = projected[:, node] + correction
        state = kinematic_step(state, projected[:, node], dt=float(cfg.runtime.dt))
    return projected


def _zero_command_root_x1(
    control: Tensor,
    command_body: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Keep the published root fixed for a zero translational/yaw command."""
    projected = torch.as_tensor(control).clone()
    command = torch.as_tensor(command_body, dtype=projected.dtype, device=projected.device)
    zero_translation = torch.linalg.vector_norm(command[:, :2], dim=-1) <= float(
        cfg.gait.zero_translation_command_deadband
    )
    zero_yaw = command[:, 2].abs() <= float(cfg.gait.zero_translation_command_deadband)
    projected[:, 0, :2] = torch.where(
        zero_translation[:, None], torch.zeros_like(projected[:, 0, :2]), projected[:, 0, :2]
    )
    projected[:, 0, 5] = torch.where(
        zero_translation & zero_yaw,
        torch.zeros_like(projected[:, 0, 5]),
        projected[:, 0, 5],
    )
    return projected


def _enforce_startup_foot_lead(
    measured_state: JointMpcRtiState,
    control: Tensor,
    command_body: Tensor,
    contact_x1: Tensor,
    startup_mask: Tensor,
    cfg: JointMpcRtiCfg,
    release_mask: Tensor | None = None,
) -> Tensor:
    """Bound startup root motion while moving scheduled swing feet first."""
    projected = torch.as_tensor(control).clone()
    command = torch.as_tensor(command_body, dtype=projected.dtype, device=projected.device)
    startup = torch.as_tensor(startup_mask, dtype=torch.bool, device=projected.device)
    command_xy = command[:, :2]
    command_norm = torch.linalg.vector_norm(command_xy, dim=1, keepdim=True)
    axis_body = command_xy / command_norm.clamp_min(1.0e-6)
    root_velocity = projected[:, 0, :2]
    root_component = (root_velocity * axis_body).sum(dim=1, keepdim=True)
    root_limit = float(cfg.gait.startup_root_leak_limit_m) / float(cfg.runtime.dt)
    limited_component = torch.minimum(root_component, root_component.new_full((), root_limit))
    limited_root = limited_component * axis_body
    projected[:, 0, :2] = torch.where(startup.unsqueeze(-1), limited_root, root_velocity)
    projected[:, 0, 3:6] = torch.where(
        startup.unsqueeze(-1), torch.zeros_like(projected[:, 0, 3:6]), projected[:, 0, 3:6]
    )
    if release_mask is not None:
        release = torch.as_tensor(release_mask, dtype=torch.bool, device=projected.device)
        released_component = torch.maximum(
            root_component,
            root_component.new_full((), float(cfg.gait.startup_root_release_velocity)),
        )
        released_root = root_velocity + (released_component - root_component) * axis_body
        projected[:, 0, :2] = torch.where(release.unsqueeze(-1), released_root, projected[:, 0, :2])

    measured_foot = go2_foot_pos(
        measured_state.root_pos_w,
        measured_state.root_rpy_w,
        measured_state.joint_pos,
    )
    axis_world = body_linear_velocity_to_world(axis_body, measured_state.root_rpy_w[:, 2])
    target_xy = measured_foot[..., :2] + float(cfg.gait.startup_foot_lead_target_m) * axis_world[:, None]
    swing = torch.logical_not(torch.as_tensor(contact_x1, dtype=torch.bool, device=projected.device))
    active = torch.logical_and(startup.unsqueeze(-1), swing)
    identity = torch.eye(2, dtype=projected.dtype, device=projected.device).view(1, 1, 2, 2)
    batch = measured_state.batch_size
    for _ in range(3):
        next_state = kinematic_step(measured_state.as_vector(), projected[:, 0], dt=float(cfg.runtime.dt))
        next_foot = go2_foot_pos(next_state[:, :3], next_state[:, 3:6], next_state[:, 6:])
        error = target_xy - next_foot[..., :2]
        jacobian = foot_jacobian_leg(next_state[:, :3], next_state[:, 3:6], next_state[:, 6:])[..., :2, :]
        normal = torch.matmul(jacobian, jacobian.transpose(-1, -2)) + 1.0e-6 * identity
        correction = torch.matmul(
            jacobian.transpose(-1, -2),
            torch.linalg.solve_ex(normal, error.unsqueeze(-1), check_errors=False)[0],
        ).squeeze(-1) / float(cfg.runtime.dt)
        joint_velocity = projected[:, 0, 6:].reshape(batch, 4, 3)
        joint_velocity = torch.where(active.unsqueeze(-1), joint_velocity + correction, joint_velocity)
        projected[:, 0, 6:] = joint_velocity.reshape(batch, 12).clamp(-30.0, 30.0)
    return projected


def _build_lq_problem(
    rollout: JointMpcRollout,
    desired_control: Tensor,
    joint_target: Tensor,
    measured_state: JointMpcRtiState,
    command_body: Tensor,
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
    state_weight[3:5] = float(cfg.losses.root_roll_pitch)
    state_weight[6:] = float(cfg.core_losses.joint_posture_weight)
    control_weight = rollout.state.new_full((18,), float(cfg.core_losses.joint_velocity_weight))
    control_weight[:6] = float(cfg.core_losses.command_control_weight)
    control_weight[3:6] = float(cfg.core_losses.root_angular_control_weight)
    matrix_q = torch.diag(state_weight).view(1, 1, 18, 18).expand(batch, horizon, -1, -1).clone()
    matrix_r = torch.diag(control_weight).view(1, 1, 18, 18).expand(batch, horizon, -1, -1).clone()
    state_error = torch.zeros_like(rollout.state[:, :-1])
    state_error[..., 3:5] = rollout.state[:, :-1, 3:5]
    state_error[..., 6:] = rollout.state[:, :-1, 6:] - joint_target[:, :-1]
    vector_q = matrix_q.diagonal(dim1=-2, dim2=-1) * state_error
    vector_r = matrix_r.diagonal(dim1=-2, dim2=-1) * (rollout.control - desired_control)
    terminal_weight = rollout.state.new_zeros((18,))
    terminal_weight[3:5] = float(cfg.losses.root_roll_pitch)
    terminal_weight[6:] = float(cfg.core_losses.terminal_joint_posture_weight)
    terminal_q = torch.diag(terminal_weight).unsqueeze(0).expand(batch, -1, -1).clone()
    terminal_error = torch.zeros_like(rollout.state[:, -1])
    terminal_error[..., 3:5] = rollout.state[:, -1, 3:5]
    terminal_error[..., 6:] = rollout.state[:, -1, 6:] - joint_target[:, -1]
    terminal_vector = terminal_weight.unsqueeze(0) * terminal_error
    command = torch.as_tensor(command_body, dtype=rollout.state.dtype, device=rollout.state.device)
    command_world = body_linear_velocity_to_world(command[:, :2], rollout.state[:, 0, 5])
    desired_progress = command_world * (float(horizon) * float(cfg.runtime.dt))
    progress_error = rollout.state[:, -1, :2] - rollout.state[:, 0, :2] - desired_progress
    progress_weight = float(cfg.losses.command_progress)
    terminal_q[:, :2, :2].add_(
        progress_weight
        * torch.eye(2, dtype=rollout.state.dtype, device=rollout.state.device).unsqueeze(0)
    )
    terminal_vector[:, :2].add_(progress_weight * progress_error)
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


def _control_constraint_parameterization(
    constraint_control: Tensor,
    constraint_state: Tensor,
) -> tuple[Tensor, Tensor]:
    residual = torch.zeros(
        *constraint_control.shape[:-1],
        dtype=constraint_control.dtype,
        device=constraint_control.device,
    )
    feedback, _, projector = _affine_control_constraint_parameterization(
        constraint_control,
        constraint_state,
        residual,
    )
    return feedback, projector


def _affine_control_constraint_parameterization_with_right_inverse(
    constraint_control: Tensor,
    constraint_state: Tensor,
    constraint_residual: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    control = torch.as_tensor(constraint_control)
    state = torch.as_tensor(constraint_state, dtype=control.dtype, device=control.device)
    residual = torch.as_tensor(constraint_residual, dtype=control.dtype, device=control.device)
    if control.ndim != 4 or state.ndim != 4 or control.shape[:3] != state.shape[:3]:
        raise ValueError("constraint matrices must have shapes [B,H,C,U] and [B,H,C,X]")
    if residual.shape != control.shape[:-1]:
        raise ValueError("constraint_residual must have shape [B,H,C]")
    gram = torch.matmul(control, control.transpose(-1, -2))
    identity_constraint = torch.eye(
        gram.shape[-1], dtype=control.dtype, device=control.device
    ).view(*((1,) * (gram.ndim - 2)), gram.shape[-1], gram.shape[-1]).expand_as(gram).contiguous()
    right_inverse = torch.matmul(
        control.transpose(-1, -2),
        fixed_spd_solve(
            gram + 1.0e-7 * identity_constraint,
            identity_constraint,
        ),
    )
    feedback = -torch.matmul(right_inverse, state)
    feedforward = -torch.matmul(right_inverse, residual.unsqueeze(-1)).squeeze(-1)
    identity_control = torch.eye(control.shape[-1], dtype=control.dtype, device=control.device)
    projector = identity_control - torch.matmul(right_inverse, control)
    return feedback, feedforward, projector, right_inverse


def _affine_control_constraint_parameterization(
    constraint_control: Tensor,
    constraint_state: Tensor,
    constraint_residual: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    feedback, feedforward, projector, _ = (
        _affine_control_constraint_parameterization_with_right_inverse(
            constraint_control,
            constraint_state,
            constraint_residual,
        )
    )
    return feedback, feedforward, projector


def _eliminate_stance_control_constraints(
    problem: LqProblem,
    rollout: JointMpcRollout,
    contact_state: Tensor,
    stance_anchor_w: Tensor,
) -> tuple[LqProblem, Tensor, Tensor, Tensor]:
    batch, horizon = int(problem.matrix_a.shape[0]), int(problem.matrix_a.shape[1])
    state_next = rollout.state[:, 1:].reshape(batch * horizon, 18)
    foot_jacobian = complete_foot_jacobian(
        state_next[:, :3], state_next[:, 3:6], state_next[:, 6:]
    ).reshape(batch, horizon, 4, 3, 18)[..., :2, :]
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=rollout.state.device)
    continuing = torch.logical_and(contact[:, :-1], contact[:, 1:])
    constraint_jacobian = (
        foot_jacobian * continuing.unsqueeze(-1).unsqueeze(-1).to(rollout.state.dtype)
    ).reshape(batch, horizon, 8, 18)
    constraint_control = torch.matmul(constraint_jacobian, problem.matrix_b)
    constraint_state = torch.matmul(constraint_jacobian, problem.matrix_a)
    anchor = torch.as_tensor(
        stance_anchor_w,
        dtype=rollout.state.dtype,
        device=rollout.state.device,
    )
    if anchor.shape != rollout.foot_pos_w.shape:
        raise ValueError("stance_anchor_w must match rollout.foot_pos_w")
    constraint_residual = (
        (rollout.foot_pos_w[:, 1:, :, :2] - anchor[:, 1:, :, :2])
        * continuing.unsqueeze(-1).to(rollout.state.dtype)
    ).reshape(batch, horizon, 8)
    constraint_residual = constraint_residual + torch.matmul(
        constraint_jacobian,
        problem.affine_dynamics.unsqueeze(-1),
    ).squeeze(-1)
    feedback, feedforward, projector = _affine_control_constraint_parameterization(
        constraint_control,
        constraint_state,
        constraint_residual,
    )
    matrix_s = (
        torch.zeros_like(problem.matrix_a)
        if problem.matrix_s is None
        else problem.matrix_s
    )
    feedback_t = feedback.transpose(-1, -2)
    projector_t = projector.transpose(-1, -2)
    matrix_q = problem.matrix_q + torch.matmul(
        feedback_t, torch.matmul(problem.matrix_r, feedback)
    ) + torch.matmul(feedback_t, matrix_s) + torch.matmul(matrix_s.transpose(-1, -2), feedback)
    matrix_r = torch.matmul(projector_t, torch.matmul(problem.matrix_r, projector))
    matrix_s_transformed = torch.matmul(
        projector_t,
        torch.matmul(problem.matrix_r, feedback) + matrix_s,
    )
    control_offset_gradient = torch.matmul(
        problem.matrix_r,
        feedforward.unsqueeze(-1),
    ).squeeze(-1)
    vector_q = problem.vector_q + torch.matmul(
        feedback_t, problem.vector_r.unsqueeze(-1)
    ).squeeze(-1) + torch.matmul(
        feedback_t,
        control_offset_gradient.unsqueeze(-1),
    ).squeeze(-1) + torch.matmul(
        matrix_s.transpose(-1, -2),
        feedforward.unsqueeze(-1),
    ).squeeze(-1)
    vector_r = torch.matmul(
        projector_t,
        (problem.vector_r + control_offset_gradient).unsqueeze(-1),
    ).squeeze(-1)
    matrix_a = problem.matrix_a + torch.matmul(problem.matrix_b, feedback)
    matrix_b = torch.matmul(problem.matrix_b, projector)
    affine_dynamics = problem.affine_dynamics + torch.matmul(
        problem.matrix_b,
        feedforward.unsqueeze(-1),
    ).squeeze(-1)
    matrix_q = 0.5 * (matrix_q + matrix_q.transpose(-1, -2))
    matrix_r = 0.5 * (matrix_r + matrix_r.transpose(-1, -2))
    return (
        replace(
            problem,
            matrix_a=matrix_a,
            matrix_b=matrix_b,
            matrix_q=matrix_q,
            matrix_r=matrix_r,
            matrix_s=matrix_s_transformed,
            vector_q=vector_q,
            vector_r=vector_r,
            affine_dynamics=affine_dynamics,
        ),
        feedback,
        feedforward,
        projector,
    )


def _project_collision_kkt_root_assist(
    jacobian: Tensor,
    *,
    selected_distance_m: Tensor,
    state_x1: Tensor,
    command_body: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Apply the LQ root-assist contract to aggregated collision KKT rows."""
    projected = torch.as_tensor(jacobian).clone()
    distance = torch.as_tensor(
        selected_distance_m, dtype=projected.dtype, device=projected.device
    )
    state = torch.as_tensor(state_x1, dtype=projected.dtype, device=projected.device)
    command = torch.as_tensor(
        command_body, dtype=projected.dtype, device=projected.device
    )
    if projected.ndim != 3 or projected.shape[-1] != 18:
        raise ValueError("collision KKT jacobian must have shape [B,R,18]")
    if distance.shape != projected.shape[:2] or state.shape != (projected.shape[0], 18):
        raise ValueError("collision KKT assist tensors have incompatible shapes")
    if command.shape != (projected.shape[0], 3):
        raise ValueError("command_body must have shape [B,3]")

    command_world = body_linear_velocity_to_world(command[:, :2], state[:, 5])
    command_axis = command_world / torch.linalg.vector_norm(
        command_world, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    command_projection = torch.einsum("bi,bj->bij", command_axis, command_axis)
    identity_xy = torch.eye(
        2, dtype=projected.dtype, device=projected.device
    ).unsqueeze(0)
    joint_lower = constant_like(
        state, "collision_kkt_joint_lower", (-1.0472, -0.6632, -2.721) * 4
    )
    joint_upper = constant_like(
        state, "collision_kkt_joint_upper", (1.0472, 2.966, -0.837) * 4
    )
    joint_margin = torch.minimum(
        state[:, 6:] - joint_lower,
        joint_upper - state[:, 6:],
    ).amin(dim=1)
    reachability_pressure = torch.sigmoid((0.25 - joint_margin) / 0.05)
    proximity = torch.sigmoid(
        (float(cfg.gait.small_collision_influence_radius) - distance)
        / float(cfg.gait.small_collision_temperature)
    )
    assist_weight = proximity * (0.10 + 0.90 * reachability_pressure[:, None])
    root_xy_projection = command_projection[:, None] + assist_weight[..., None, None] * (
        identity_xy[:, None] - command_projection[:, None]
    )
    projected[..., :2] = torch.einsum(
        "bri,brij->brj", projected[..., :2], root_xy_projection
    )
    projected[..., 2] = 0.0
    projected[..., 3:6] = projected[..., 3:6] * assist_weight.unsqueeze(-1)
    return projected


def _add_stance_control_constraints(
    problem: LqProblem,
    rollout: JointMpcRollout,
    contact_state: Tensor,
    stance_anchor_w: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    recovery_state: Tensor | None = None,
    foot_query: JointMpcTerrainQuery | None = None,
    swing_target_w: Tensor | None = None,
    terrain_field: JointMpcTerrainField | None = None,
    startup_mask: Tensor | None = None,
    command_body: Tensor | None = None,
    initial_grounding: Tensor | None = None,
    recovery_ground_safe: Tensor | None = None,
) -> LqProblem:
    batch, horizon = int(problem.matrix_a.shape[0]), int(problem.matrix_a.shape[1])
    state_next = rollout.state[:, 1:].reshape(batch * horizon, 18)
    foot_jacobian = complete_foot_jacobian(
        state_next[:, :3], state_next[:, 3:6], state_next[:, 6:]
    ).reshape(batch, horizon, 4, 3, 18)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=rollout.state.device)
    # RTI publishes x1; keep the hard FK equality on the measured-to-published
    # interval. Future stance nodes remain soft preview terms and are enforced
    # by the next rolling solve with a fresh measured state.
    continuing = torch.zeros_like(contact[:, :-1])
    continuing[:, 0] = contact[:, 1]
    constraint_jacobian = (
        foot_jacobian * continuing.unsqueeze(-1).unsqueeze(-1).to(rollout.state.dtype)
    )
    anchor = torch.as_tensor(
        stance_anchor_w,
        dtype=rollout.state.dtype,
        device=rollout.state.device,
    )
    if anchor.shape != rollout.foot_pos_w.shape:
        raise ValueError("stance_anchor_w must match rollout.foot_pos_w")
    constraint_residual = (
        (rollout.foot_pos_w[:, 1:] - anchor[:, 1:])
        * continuing.unsqueeze(-1).to(rollout.state.dtype)
    )
    recovery_ground_nodes = torch.zeros_like(continuing)
    recovery_sdf_nodes = torch.zeros_like(continuing)
    initial_ground_nodes = torch.zeros_like(continuing)
    swing_clearance_nodes = torch.zeros_like(continuing)
    if initial_grounding is not None:
        initial_mask = torch.as_tensor(initial_grounding, dtype=torch.bool, device=rollout.state.device)
        if initial_mask.shape != (batch, 4):
            raise ValueError("initial_grounding must have shape [B,4]")
        initial_ground_nodes[:, 0] = torch.logical_and(initial_mask, continuing[:, 0])
    if recovery_state is not None and foot_query is not None:
        recovery = torch.as_tensor(
            recovery_state,
            dtype=torch.bool,
            device=rollout.state.device,
        )
        if recovery.shape != (batch, 4):
            raise ValueError("recovery_state must have shape [B,4]")
        distance = torch.as_tensor(
            foot_query.small_distance_m,
            dtype=rollout.state.dtype,
            device=rollout.state.device,
        ).reshape(batch, horizon + 1, 4)[:, 1:]
        gradient = torch.as_tensor(
            foot_query.small_gradient_w,
            dtype=rollout.state.dtype,
            device=rollout.state.device,
        ).reshape(batch, horizon + 1, 4, 2)[:, 1:]
        valid = torch.as_tensor(
            foot_query.valid,
            dtype=torch.bool,
            device=rollout.state.device,
        ).reshape(batch, horizon + 1, 4)[:, 1:]
        height = torch.as_tensor(
            foot_query.height_w,
            dtype=rollout.state.dtype,
            device=rollout.state.device,
        ).reshape(batch, horizon + 1, 4)[:, 1:]
        recovery_nodes = torch.zeros(
            batch,
            horizon,
            4,
            dtype=torch.bool,
            device=rollout.state.device,
        )
        recovery_nodes[:, 0] = torch.logical_and(recovery, torch.logical_not(contact[:, 1]))
        recovery_clearance = _recovery_sdf_constraint_clearance(distance, cfg)
        recovery_jacobian = torch.einsum(
            "bhld,bhldx->bhlx",
            gradient,
            foot_jacobian[..., :2, :],
        )
        active_sdf = recovery_nodes & valid & (recovery_clearance < 0.0)
        recovery_sdf_nodes = active_sdf
        landing_clearance = torch.full_like(distance, torch.inf)
        if recovery_ground_safe is not None:
            safe = torch.as_tensor(
                recovery_ground_safe,
                dtype=torch.bool,
                device=rollout.state.device,
            )
            if safe.shape != (batch, 4):
                raise ValueError("recovery_ground_safe must have shape [B,4]")
            landing_clearance[:, 0] = torch.where(
                safe,
                landing_clearance.new_full((batch, 4), float(cfg.gait.small_collision_margin_xy)),
                landing_clearance.new_full((batch, 4), -torch.inf),
            )
        active_ground = _recovery_grounding_active_mask(
            recovery_state=recovery_nodes,
            contact_state=contact[:, 1:],
            map_valid=valid,
            foot_small_distance_m=distance,
            leg_landing_clearance_m=landing_clearance,
            cfg=cfg,
        )
        recovery_ground_nodes = active_ground
        recovery_jacobian = recovery_jacobian * active_sdf.unsqueeze(-1).to(
            rollout.state.dtype
        )
        recovery_jacobian = recovery_jacobian.clone()
        recovery_jacobian[..., :6] = 0.0
        constraint_jacobian = constraint_jacobian.clone()
        constraint_residual = constraint_residual.clone()
        constraint_jacobian[..., 0, :] += recovery_jacobian
        recovery_sdf_control = torch.matmul(recovery_jacobian, problem.matrix_b)
        recovery_sdf_error = _clamp_recovery_ground_error_to_control_reach(
            recovery_clearance,
            recovery_sdf_control,
            joint_direction_limit=(
                float(cfg.solver.joint_direction_limit)
                * float(cfg.solver.recovery_sdf_reach_fraction)
            ),
        )
        constraint_residual[..., 0] += recovery_sdf_error * active_sdf.to(
            rollout.state.dtype
        )
        if command_body is not None:
            recovery_command = torch.as_tensor(
                command_body,
                dtype=rollout.state.dtype,
                device=rollout.state.device,
            )
            recovery_command_world = body_linear_velocity_to_world(
                recovery_command[:, :2], rollout.state[:, 0, 5]
            )
            recovery_command_speed = torch.linalg.vector_norm(
                recovery_command_world, dim=1
            )
            recovery_command_axis = recovery_command_world / recovery_command_speed[
                :, None
            ].clamp_min(1.0e-6)
            recovery_progress_active = recovery_nodes & (
                recovery_command_speed[:, None, None]
                > float(cfg.gait.zero_translation_command_deadband)
            )
            recovery_progress_jacobian = torch.einsum(
                "bd,bhldx->bhlx",
                recovery_command_axis,
                foot_jacobian[..., :2, :],
            )
            recovery_progress_jacobian = recovery_progress_jacobian.clone()
            recovery_progress_jacobian[..., :6] = 0.0
            constraint_jacobian[..., 1, :] += (
                recovery_progress_jacobian
                * recovery_progress_active.unsqueeze(-1).to(rollout.state.dtype)
            )
            recovery_progress_step = (
                0.25 * recovery_command_speed * float(cfg.runtime.dt)
            )
            constraint_residual[..., 1] += (
                -recovery_progress_step[:, None, None]
                * recovery_progress_active.to(rollout.state.dtype)
            )
        recovery_ground_jacobian = foot_jacobian[..., 2, :].clone()
        recovery_ground_jacobian[..., :6] = 0.0
        constraint_jacobian[..., 2, :] += (
            recovery_ground_jacobian
            * active_ground.unsqueeze(-1).to(rollout.state.dtype)
        )
        recovery_ground_error = (
            rollout.foot_pos_w[:, 1:, :, 2]
            - height
            - float(cfg.gait.foot_contact_offset)
        ).clamp(
            -float(cfg.gait.stance_ground_recovery_step_m),
            float(cfg.gait.stance_ground_recovery_step_m),
        )
        recovery_ground_control = torch.matmul(
            recovery_ground_jacobian, problem.matrix_b
        )
        recovery_ground_error = _clamp_recovery_ground_error_to_control_reach(
            recovery_ground_error,
            recovery_ground_control,
            joint_direction_limit=float(cfg.solver.joint_direction_limit),
        )
        constraint_residual[..., 2] += (
            recovery_ground_error * active_ground.to(rollout.state.dtype)
        )
        if swing_target_w is not None:
            swing_target = torch.as_tensor(
                swing_target_w,
                dtype=rollout.state.dtype,
                device=rollout.state.device,
            )
            if swing_target.shape != rollout.foot_pos_w.shape:
                raise ValueError("swing_target_w must match rollout foot positions")
            swing_nodes = torch.zeros(
                batch,
                horizon,
                4,
                dtype=torch.bool,
                device=rollout.state.device,
            )
            swing_nodes[:, 0] = torch.logical_and(
                torch.logical_not(contact[:, 1]),
                torch.logical_not(recovery),
            )
            effective_height = height + float(cfg.gait.small_semantic_height) * torch.sigmoid(
                distance / float(cfg.gait.small_collision_temperature)
            )
            swing_collision = (
                swing_nodes
                & valid
                & (
                    distance
                    < float(cfg.gait.foot_collision_radius)
                    + float(cfg.gait.small_collision_margin_xy)
                )
                & (
                    rollout.foot_pos_w[:, 1:, :, 2]
                    - float(cfg.gait.foot_collision_radius)
                    < effective_height + float(cfg.gait.small_collision_margin_z)
                )
            )
            swing_clearance_nodes = swing_collision
            swing_jacobian = foot_jacobian.clone()
            swing_jacobian[..., :6] = 0.0
            swing_residual = rollout.foot_pos_w[:, 1:] - swing_target[:, 1:]
            startup_nodes = torch.zeros_like(swing_nodes)
            if startup_mask is not None:
                startup = torch.as_tensor(
                    startup_mask, dtype=torch.bool, device=rollout.state.device
                )
                if startup.shape != (batch,):
                    raise ValueError("startup_mask must have shape [B]")
                startup_nodes[:, 0] = startup.unsqueeze(-1)
            swing_tracking = swing_nodes & valid & torch.logical_or(startup_nodes, swing_collision)
            swing_xy_residual = swing_residual[..., :2].clamp(
                -float(cfg.gait.startup_foot_lead_target_m),
                float(cfg.gait.startup_foot_lead_target_m),
            )
            constraint_jacobian[..., :2, :] += (
                swing_jacobian[..., :2, :]
                * swing_tracking.unsqueeze(-1).unsqueeze(-1).to(rollout.state.dtype)
            )
            constraint_residual[..., :2] += (
                swing_xy_residual
                * swing_tracking.unsqueeze(-1).to(rollout.state.dtype)
            )
            swing_clearance_residual = (
                rollout.foot_pos_w[:, 1:, :, 2]
                - effective_height
                - float(cfg.gait.foot_collision_radius)
                - float(cfg.gait.small_collision_margin_z)
                - float(cfg.gait.swing_clearance_kkt_buffer_m)
            )
            constraint_jacobian[..., 2, :] += (
                swing_jacobian[..., 2, :]
                * swing_collision.unsqueeze(-1).to(rollout.state.dtype)
            )
            constraint_residual[..., 2] += (
                swing_clearance_residual * swing_collision.to(rollout.state.dtype)
            )
    constraint_jacobian = constraint_jacobian.reshape(batch, horizon, 12, 18)
    constraint_residual = constraint_residual.reshape(batch, horizon, 12)
    constraint_control = torch.matmul(constraint_jacobian, problem.matrix_b)
    constraint_state = torch.matmul(constraint_jacobian, problem.matrix_a)
    constraint_residual = constraint_residual + torch.matmul(
        constraint_jacobian,
        problem.affine_dynamics.unsqueeze(-1),
    ).squeeze(-1)
    grouped_control = constraint_control.reshape(batch, horizon, 4, 3, 18)
    grouped_residual = constraint_residual.reshape(batch, horizon, 4, 3).clone()
    conservative_ground_residual = _clamp_recovery_ground_error_to_control_reach(
        grouped_residual[..., 2],
        grouped_control[..., 2, :],
        joint_direction_limit=(
            float(cfg.solver.joint_direction_limit)
            * float(cfg.solver.constraint_reach_fraction)
        ),
    )
    recovery_ground_residual = _clamp_recovery_ground_error_to_control_reach(
        grouped_residual[..., 2],
        grouped_control[..., 2, :],
        joint_direction_limit=float(cfg.solver.joint_direction_limit),
    )
    recovery_sdf_residual = _clamp_recovery_ground_error_to_control_reach(
        grouped_residual[..., 0],
        grouped_control[..., 0, :],
        joint_direction_limit=(
            float(cfg.solver.joint_direction_limit)
            * float(cfg.solver.recovery_sdf_reach_fraction)
        ),
    )
    swing_clearance_residual = _clamp_recovery_ground_error_to_control_reach(
        grouped_residual[..., 2],
        grouped_control[..., 2, :],
        joint_direction_limit=(
            float(cfg.solver.joint_direction_limit)
            * float(cfg.solver.swing_clearance_reach_fraction)
        ),
    )
    initial_ground_residual = grouped_residual[..., 2]
    grouped_residual[..., 0] = torch.where(
        recovery_sdf_nodes,
        recovery_sdf_residual,
        grouped_residual[..., 0],
    )
    grouped_residual[..., 2] = torch.where(
        recovery_ground_nodes,
        recovery_ground_residual,
        torch.where(
            swing_clearance_nodes,
            swing_clearance_residual,
            torch.where(
                initial_ground_nodes,
                initial_ground_residual,
                conservative_ground_residual,
            ),
        ),
    )
    constraint_residual = grouped_residual.reshape(batch, horizon, 12)
    collision_constraint_jacobian = problem.matrix_a.new_zeros(batch, horizon, 8, 18)
    collision_constraint_residual = problem.matrix_a.new_zeros(batch, horizon, 8)
    if terrain_field is not None:
        state_x1 = rollout.state[:, 1]
        body_x1 = rollout.body_samples_w[:, 1]
        foot_x1 = rollout.foot_pos_w[:, 1]
        knee_x1 = rollout.knee_pos_w[:, 1]
        calf_x1 = rollout.shank_samples_w[:, 1].reshape(batch, 12, 3)
        thigh_x1 = rollout.thigh_samples_w[:, 1].reshape(batch, 12, 3)
        positions = torch.cat((body_x1, foot_x1, knee_x1, calf_x1, thigh_x1), dim=1)
        if int(positions.shape[1]) != 41:
            raise ValueError("full-body collision KKT requires 41 fixed geometry samples")
        link_jacobian = complete_link_sample_jacobians(
            state_x1[:, :3], state_x1[:, 3:6], state_x1[:, 6:]
        )
        point_jacobian = torch.cat(
            (
                complete_body_sample_jacobian(state_x1[:, 3:6], body_x1, state_x1[:, :3]),
                complete_foot_jacobian(state_x1[:, :3], state_x1[:, 3:6], state_x1[:, 6:]),
                complete_knee_jacobian(state_x1[:, :3], state_x1[:, 3:6], state_x1[:, 6:]),
                link_jacobian.calf_samples.reshape(batch, 12, 3, 18),
                link_jacobian.thigh_samples.reshape(batch, 12, 3, 18),
            ),
            dim=1,
        )
        query = _query_world(terrain_field, positions, cfg)
        radius_values = (
            (0.0,) * 9
            + (float(cfg.gait.foot_collision_radius),) * 4
            + (float(cfg.gait.knee_collision_radius),) * 4
            + (float(cfg.gait.calf_collision_radius),) * 12
            + (float(cfg.gait.thigh_collision_radius),) * 12
        )
        radii = constant_like(
            positions,
            "kkt_full_body_collision_radii_" + "_".join(map(str, radius_values)),
            radius_values,
        ).view(1, 41)
        horizontal_clearance = query.small_distance_m - radii
        vertical_clearance = positions[..., 2] - query.height_w - radii
        use_horizontal = horizontal_clearance >= vertical_clearance
        clearance = torch.maximum(horizontal_clearance, vertical_clearance)
        active = (
            (clearance < 0.0)
            & (positions[..., 2] + radii > 0.0)
            & query.valid
        )
        horizontal_jacobian = torch.einsum(
            "bpd,bpdx->bpx", query.small_gradient_w, point_jacobian[..., :2, :]
        )
        clearance_jacobian = torch.where(
            use_horizontal.unsqueeze(-1),
            horizontal_jacobian,
            point_jacobian[..., 2, :],
        )
        base_clearance = clearance[:, :9]
        base_distance = query.small_distance_m[:, :9]
        base_active = active[:, :9]
        base_jacobian = clearance_jacobian[:, :9]
        base_index = base_clearance.argmin(dim=1, keepdim=True)
        selected_base_clearance = torch.gather(base_clearance, 1, base_index).squeeze(1)
        selected_base_distance = torch.gather(base_distance, 1, base_index).squeeze(1)
        selected_base_active = torch.gather(base_active, 1, base_index).squeeze(1)
        selected_base_jacobian = torch.gather(
            base_jacobian,
            1,
            base_index.unsqueeze(-1).expand(-1, 1, 18),
        ).squeeze(1)

        leg_clearance = torch.stack(
            tuple(
                torch.cat(
                    (
                        clearance[:, 9 + leg : 10 + leg],
                        clearance[:, 13 + leg : 14 + leg],
                        clearance[:, 17 + 3 * leg : 20 + 3 * leg],
                        clearance[:, 29 + 3 * leg : 32 + 3 * leg],
                    ),
                    dim=1,
                )
                for leg in range(4)
            ),
            dim=1,
        )
        leg_distance = torch.stack(
            tuple(
                torch.cat(
                    (
                        query.small_distance_m[:, 9 + leg : 10 + leg],
                        query.small_distance_m[:, 13 + leg : 14 + leg],
                        query.small_distance_m[:, 17 + 3 * leg : 20 + 3 * leg],
                        query.small_distance_m[:, 29 + 3 * leg : 32 + 3 * leg],
                    ),
                    dim=1,
                )
                for leg in range(4)
            ),
            dim=1,
        )
        leg_active = torch.stack(
            tuple(
                torch.cat(
                    (
                        active[:, 9 + leg : 10 + leg],
                        active[:, 13 + leg : 14 + leg],
                        active[:, 17 + 3 * leg : 20 + 3 * leg],
                        active[:, 29 + 3 * leg : 32 + 3 * leg],
                    ),
                    dim=1,
                )
                for leg in range(4)
            ),
            dim=1,
        )
        leg_jacobian = torch.stack(
            tuple(
                torch.cat(
                    (
                        clearance_jacobian[:, 9 + leg : 10 + leg],
                        clearance_jacobian[:, 13 + leg : 14 + leg],
                        clearance_jacobian[:, 17 + 3 * leg : 20 + 3 * leg],
                        clearance_jacobian[:, 29 + 3 * leg : 32 + 3 * leg],
                    ),
                    dim=1,
                )
                for leg in range(4)
            ),
            dim=1,
        )
        leg_index = leg_clearance.argmin(dim=2, keepdim=True)
        selected_leg_clearance = torch.gather(leg_clearance, 2, leg_index).squeeze(2)
        selected_leg_distance = torch.gather(leg_distance, 2, leg_index).squeeze(2)
        selected_leg_active = torch.gather(leg_active, 2, leg_index).squeeze(2)
        selected_leg_jacobian = torch.gather(
            leg_jacobian,
            2,
            leg_index.unsqueeze(-1).expand(-1, -1, 1, 18),
        ).squeeze(2)
        selected_clearance = torch.cat(
            (selected_base_clearance.unsqueeze(1), selected_leg_clearance), dim=1
        )
        selected_distance = torch.cat(
            (selected_base_distance.unsqueeze(1), selected_leg_distance), dim=1
        )
        selected_active = torch.cat(
            (selected_base_active.unsqueeze(1), selected_leg_active), dim=1
        )
        selected_jacobian = torch.cat(
            (selected_base_jacobian.unsqueeze(1), selected_leg_jacobian), dim=1
        )
        if command_body is not None:
            selected_jacobian = _project_collision_kkt_root_assist(
                selected_jacobian,
                selected_distance_m=selected_distance,
                state_x1=state_x1,
                command_body=command_body,
                cfg=cfg,
            )
        collision_constraint_jacobian[:, 0, :5] = (
            selected_jacobian * selected_active.unsqueeze(-1).to(selected_jacobian.dtype)
        )
        collision_constraint_residual[:, 0, :5] = (
            selected_clearance * selected_active.to(selected_clearance.dtype)
        )
        if foot_query is not None:
            foot_distance = torch.as_tensor(
                foot_query.small_distance_m,
                dtype=rollout.state.dtype,
                device=rollout.state.device,
            ).reshape(batch, horizon + 1, 4)[:, 1:]
            foot_height = torch.as_tensor(
                foot_query.height_w,
                dtype=rollout.state.dtype,
                device=rollout.state.device,
            ).reshape(batch, horizon + 1, 4)[:, 1:]
            foot_gradient = torch.as_tensor(
                foot_query.small_gradient_w,
                dtype=rollout.state.dtype,
                device=rollout.state.device,
            ).reshape(batch, horizon + 1, 4, 2)[:, 1:]
            foot_valid = torch.as_tensor(
                foot_query.valid,
                dtype=torch.bool,
                device=rollout.state.device,
            ).reshape(batch, horizon + 1, 4)[:, 1:]
            foot_radius = float(cfg.gait.foot_collision_radius)
            foot_horizontal = foot_distance - foot_radius - float(cfg.gait.small_collision_margin_xy)
            foot_vertical = (
                rollout.foot_pos_w[:, 1:, :, 2]
                - foot_height
                - foot_radius
                - float(cfg.gait.small_collision_margin_z)
            )
            foot_use_horizontal = foot_horizontal >= foot_vertical
            foot_clearance = torch.maximum(foot_horizontal, foot_vertical)
            foot_active = (
                (foot_clearance < 0.0)
                & (~contact[:, 1:])
                & foot_valid
                & (rollout.foot_pos_w[:, 1:, :, 2] + foot_radius > 0.0)
            )
            selected_leg_is_foot = leg_index.squeeze(-1) == 0
            dedicated_swing_clearance_active = (
                grouped_control[:, 0, :, 2].abs().sum(dim=-1) > 0.0
            )
            duplicate_x1_foot_row = (
                selected_leg_is_foot
                & foot_active[:, 0]
                & dedicated_swing_clearance_active
            )
            collision_constraint_jacobian[:, 0, 1:5] = torch.where(
                duplicate_x1_foot_row.unsqueeze(-1),
                torch.zeros_like(collision_constraint_jacobian[:, 0, 1:5]),
                collision_constraint_jacobian[:, 0, 1:5],
            )
            collision_constraint_residual[:, 0, 1:5] = torch.where(
                duplicate_x1_foot_row,
                torch.zeros_like(collision_constraint_residual[:, 0, 1:5]),
                collision_constraint_residual[:, 0, 1:5],
            )
            foot_horizontal_jacobian = torch.einsum(
                "bhkd,bhkdx->bhkx",
                foot_gradient,
                foot_jacobian[..., :2, :],
            )
            foot_clearance_jacobian = torch.where(
                foot_use_horizontal.unsqueeze(-1),
                foot_horizontal_jacobian,
                foot_jacobian[..., 2, :],
            )
            # Node x1 already owns the five aggregated full-body rows above
            # (one base sample plus one worst sample per leg).  Reuse the four
            # leg rows for future-foot preview only from x2 onward; otherwise
            # this assignment silently erases four of the five x1 rows.
            preview_start = 1 if terrain_field is not None else 0
            collision_constraint_jacobian[:, preview_start:, :4] = (
                foot_clearance_jacobian[:, preview_start:]
                * foot_active[:, preview_start:].unsqueeze(-1).to(rollout.state.dtype)
            )
            collision_constraint_residual[:, preview_start:, :4] = (
                foot_clearance[:, preview_start:]
                * foot_active[:, preview_start:].to(rollout.state.dtype)
            )
    if startup_mask is not None or command_body is not None:
        if startup_mask is None or command_body is None:
            raise ValueError("startup_mask and command_body must be provided together")
        startup = torch.as_tensor(
            startup_mask, dtype=torch.bool, device=rollout.state.device
        )
        command = torch.as_tensor(
            command_body, dtype=rollout.state.dtype, device=rollout.state.device
        )
        if startup.shape != (batch,) or command.shape != (batch, 3):
            raise ValueError("startup_mask and command_body must have shapes [B] and [B,3]")
        command_world = body_linear_velocity_to_world(command[:, :2], rollout.state[:, 0, 5])
        command_norm = torch.linalg.vector_norm(command_world, dim=1)
        zero_translation = command_norm <= float(cfg.gait.zero_translation_command_deadband)
        command_axis = command_world / command_norm.unsqueeze(-1).clamp_min(1.0e-6)
        command_axis = command_axis.clone()
        command_axis[zero_translation, 0] = 1.0
        root_progress = (
            (rollout.state[:, 1, :2] - rollout.state[:, 0, :2]) * command_axis
        ).sum(dim=1)
        startup_active = startup | zero_translation
        collision_constraint_jacobian[:, 0, 5, :2] = (
            command_axis * startup_active.unsqueeze(-1).to(rollout.state.dtype)
        )
        root_target = torch.where(
            zero_translation,
            torch.zeros_like(root_progress),
            root_progress.new_full((), float(cfg.gait.startup_root_leak_limit_m)),
        )
        collision_constraint_residual[:, 0, 5] = (
            (root_progress - root_target) * startup_active.to(rollout.state.dtype)
        )
    collision_constraint_jacobian[:, 0, 6, 3] = 1.0
    collision_constraint_jacobian[:, 0, 7, 4] = 1.0
    attitude_target = rollout.state[:, 1, 3:5].clamp(
        -float(cfg.solver.root_roll_pitch_limit_rad),
        float(cfg.solver.root_roll_pitch_limit_rad),
    )
    collision_constraint_residual[:, 0, 6:8] = (
        rollout.state[:, 1, 3:5] - attitude_target
    )
    collision_constraint_control = torch.matmul(
        collision_constraint_jacobian, problem.matrix_b
    )
    collision_constraint_state = torch.matmul(
        collision_constraint_jacobian, problem.matrix_a
    )
    collision_constraint_residual = collision_constraint_residual + torch.matmul(
        collision_constraint_jacobian,
        problem.affine_dynamics.unsqueeze(-1),
    ).squeeze(-1)
    # The first solve is the active-set preview.  Pinning joints merely because
    # the base rollout touches a bound also forbids valid directions back into
    # the feasible set.  The preview-dependent refinement below owns all bound
    # activation while these rows preserve the fixed 32-row program shape.
    joint_constraint_control = problem.matrix_b.new_zeros(batch, horizon, 12, 18)
    joint_constraint_state = problem.matrix_a.new_zeros(batch, horizon, 12, 18)
    joint_constraint_residual = problem.matrix_a.new_zeros(batch, horizon, 12)
    return replace(
        problem,
        constraint_control=torch.cat(
            (constraint_control, collision_constraint_control, joint_constraint_control),
            dim=2,
        ),
        constraint_state=torch.cat(
            (constraint_state, collision_constraint_state, joint_constraint_state),
            dim=2,
        ),
        constraint_residual=torch.cat(
            (constraint_residual, collision_constraint_residual, joint_constraint_residual),
            dim=2,
        ),
    )


def _refine_predicted_joint_bound_constraints(
    problem: LqProblem,
    rollout: JointMpcRollout,
    preview_delta_state: Tensor,
    cfg: JointMpcRtiCfg,
    *,
    preview_delta_control: Tensor | None = None,
    base_control: Tensor | None = None,
    command_body: Tensor | None = None,
    startup_mask: Tensor | None = None,
    contact_state: Tensor | None = None,
) -> LqProblem:
    if problem.constraint_control is None or problem.constraint_state is None or problem.constraint_residual is None:
        raise ValueError("stance constraints must be attached before joint-bound refinement")
    batch, horizon = int(problem.matrix_a.shape[0]), int(problem.matrix_a.shape[1])
    delta_state = torch.as_tensor(
        preview_delta_state,
        dtype=rollout.state.dtype,
        device=rollout.state.device,
    )
    if delta_state.shape != rollout.state.shape:
        raise ValueError("preview_delta_state must match rollout.state")
    prefix_control = problem.constraint_control[:, :, :20].clone()
    prefix_state = problem.constraint_state[:, :, :20].clone()
    prefix_residual = problem.constraint_residual[:, :, :20].clone()
    joint_lower = constant_like(rollout.state, "preview_joint_lower", (-1.0472, -0.6632, -2.721) * 4)
    joint_upper = constant_like(rollout.state, "preview_joint_upper", (1.0472, 2.966, -0.837) * 4)
    joint_lower = joint_lower + float(cfg.solver.joint_position_safety_margin_rad)
    joint_upper = joint_upper - float(cfg.solver.joint_position_safety_margin_rad)
    joint_base = rollout.state[:, 1:, 6:]
    if preview_delta_control is not None or base_control is not None:
        if preview_delta_control is None or base_control is None:
            raise ValueError(
                "preview_delta_control and base_control must be provided together"
            )
        preview_control = torch.as_tensor(
            preview_delta_control,
            dtype=rollout.control.dtype,
            device=rollout.control.device,
        )
        base = torch.as_tensor(
            base_control,
            dtype=rollout.control.dtype,
            device=rollout.control.device,
        )
        if preview_control.shape != rollout.control.shape or base.shape != rollout.control.shape:
            raise ValueError("preview_delta_control and base_control must match rollout.control")
        bounded_preview_control = preview_control.clone()
        joint_direction_limit = float(cfg.solver.joint_direction_limit)
        bounded_preview_control[..., 6:] = bounded_preview_control[..., 6:].clamp(
            -joint_direction_limit,
            joint_direction_limit,
        )
        joint_absolute_limit = _joint_candidate_absolute_limit(cfg)
        bounded_preview_control[..., 6:] = (
            base[..., 6:] + bounded_preview_control[..., 6:]
        ).clamp(-joint_absolute_limit, joint_absolute_limit) - base[..., 6:]
        recovered_states = [problem.initial_state]
        recovered_state = problem.initial_state
        for node in range(horizon):
            recovered_state = (
                torch.matmul(
                    problem.matrix_a[:, node], recovered_state.unsqueeze(-1)
                ).squeeze(-1)
                + torch.matmul(
                    problem.matrix_b[:, node],
                    bounded_preview_control[:, node].unsqueeze(-1),
                ).squeeze(-1)
                + problem.affine_dynamics[:, node]
            )
            recovered_states.append(recovered_state)
        bounded_delta_state = torch.stack(recovered_states, dim=1)
        joint_preview = joint_base + bounded_delta_state[:, 1:, 6:]
    else:
        joint_preview = joint_base + delta_state[:, 1:, 6:]
    active_lower = joint_preview < joint_lower
    active_upper = joint_preview > joint_upper
    position_active = torch.logical_or(active_lower, active_upper)
    if contact_state is not None:
        contact = torch.as_tensor(
            contact_state,
            dtype=torch.bool,
            device=rollout.state.device,
        )
        if contact.shape != (batch, horizon + 1, 4):
            raise ValueError("contact_state must have shape [B,H+1,4]")
        bound_active_by_leg = position_active.reshape(batch, horizon, 4, 3).any(
            dim=-1
        )
        suppress_swing_rows = torch.logical_and(
            bound_active_by_leg,
            torch.logical_not(contact[:, 1:]),
        )
        prefix_control = prefix_control.clone()
        prefix_state = prefix_state.clone()
        prefix_residual = prefix_residual.clone()
        prefix_control[:, :, :12] = torch.where(
            suppress_swing_rows.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(prefix_control[:, :, :12].reshape(batch, horizon, 4, 3, 18)),
            prefix_control[:, :, :12].reshape(batch, horizon, 4, 3, 18),
        ).reshape(batch, horizon, 12, 18)
        prefix_state[:, :, :12] = torch.where(
            suppress_swing_rows.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(prefix_state[:, :, :12].reshape(batch, horizon, 4, 3, 18)),
            prefix_state[:, :, :12].reshape(batch, horizon, 4, 3, 18),
        ).reshape(batch, horizon, 12, 18)
        prefix_residual[:, :, :12] = torch.where(
            suppress_swing_rows.unsqueeze(-1),
            torch.zeros_like(prefix_residual[:, :, :12].reshape(batch, horizon, 4, 3)),
            prefix_residual[:, :, :12].reshape(batch, horizon, 4, 3),
        ).reshape(batch, horizon, 12)
    target = torch.where(active_upper, joint_upper, joint_lower)
    position_selector = problem.matrix_a.new_zeros(batch, horizon, 12, 18)
    position_selector[..., 6:] = torch.diag_embed(position_active.to(problem.matrix_a.dtype))
    position_control = torch.matmul(position_selector, problem.matrix_b)
    position_state = torch.matmul(position_selector, problem.matrix_a)
    position_residual = (
        (joint_base - target) * position_active.to(joint_base.dtype)
        + torch.matmul(position_selector, problem.affine_dynamics.unsqueeze(-1)).squeeze(-1)
    )
    constraint_control = position_control
    constraint_state = position_state
    constraint_residual = position_residual
    return replace(
        problem,
        constraint_control=torch.cat((prefix_control, constraint_control), dim=2),
        constraint_state=torch.cat((prefix_state, constraint_state), dim=2),
        constraint_residual=torch.cat((prefix_residual, constraint_residual), dim=2),
    )


def _current_stance_xy_constraint_violation(
    foot_pos_w: Tensor,
    stance_anchor_w: Tensor,
    contact_state: Tensor,
    *,
    tolerance_m: float,
) -> Tensor:
    foot = torch.as_tensor(foot_pos_w)
    anchor = torch.as_tensor(stance_anchor_w, dtype=foot.dtype, device=foot.device)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    if anchor.shape != foot.shape or contact.shape != foot.shape[:-1]:
        raise ValueError("stance anchor/contact must match foot trajectory")
    current_stance = current_stance_segment_mask(contact)
    error = torch.linalg.vector_norm(foot[..., :2] - anchor[..., :2], dim=-1)
    violation = torch.where(
        current_stance,
        torch.relu(error - float(tolerance_m)),
        torch.zeros_like(error),
    )
    return violation.amax(dim=(1, 2))


def _x1_stance_xy_constraint_violation(
    foot_pos_w: Tensor,
    stance_anchor_w: Tensor,
    contact_state: Tensor,
    *,
    tolerance_m: float,
) -> Tensor:
    foot = torch.as_tensor(foot_pos_w)
    anchor = torch.as_tensor(stance_anchor_w, dtype=foot.dtype, device=foot.device)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    if anchor.shape != foot.shape or contact.shape != foot.shape[:-1] or foot.shape[1] < 2:
        raise ValueError("x1 stance constraint requires matching trajectories with at least two nodes")
    continuing = torch.logical_and(contact[:, 0], contact[:, 1])
    error = torch.linalg.vector_norm(foot[:, 1, :, :2] - anchor[:, 1, :, :2], dim=-1)
    violation = torch.where(
        continuing,
        torch.relu(error - float(tolerance_m)),
        torch.zeros_like(error),
    )
    return violation.amax(dim=1)


def _x1_constraint_violation_components(
    rollout: JointMpcRollout,
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
    contact_state: Tensor,
    stance_anchor_w: Tensor,
    recovery_state: Tensor,
) -> Tensor:
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=rollout.state.device)
    anchor = torch.as_tensor(
        stance_anchor_w, dtype=rollout.state.dtype, device=rollout.state.device
    )
    recovery = torch.as_tensor(
        recovery_state, dtype=torch.bool, device=rollout.state.device
    )
    collision_violation = _small_link_collision_violation(
        rollout, terrain_field, cfg, contact[:, 1]
    )
    preview_query = _query_world(
        terrain_field,
        rollout.foot_pos_w.reshape(int(rollout.state.shape[0]), -1, 3),
        cfg,
    )
    preview_distance = preview_query.small_distance_m.reshape(
        int(rollout.state.shape[0]), int(rollout.state.shape[1]), 4
    )
    preview_height = preview_query.height_w.reshape(
        int(rollout.state.shape[0]), int(rollout.state.shape[1]), 4
    )
    preview_foot_collision = _sphere_link_collision(
        rollout.foot_pos_w,
        preview_distance,
        preview_height,
        radius=cfg.gait.foot_collision_radius,
    )
    preview_foot_penetration = torch.where(
        preview_foot_collision,
        float(cfg.gait.foot_collision_radius) - preview_distance,
        torch.zeros_like(preview_distance),
    ).amax(dim=(1, 2))
    collision_violation = torch.maximum(collision_violation, preview_foot_penetration)
    stance_violation = _x1_stance_xy_constraint_violation(
        rollout.foot_pos_w,
        anchor,
        contact,
        tolerance_m=float(cfg.solver.stance_equality_tolerance_m),
    )
    foot_query = _query_world(terrain_field, rollout.foot_pos_w[:, 1], cfg)
    attitude_violation, attitude_rate_violation = _root_attitude_violation_components(
        rollout, cfg
    ).unbind(dim=1)
    recovery_violation = _recovery_landing_constraint_violation(
        rollout.foot_pos_w[:, 1],
        foot_query,
        contact_x1=contact[:, 1],
        recovery_state=recovery,
        cfg=cfg,
        leg_landing_clearance_m=_leg_small_horizontal_clearance(
            rollout.foot_pos_w[:, 1],
            rollout.knee_pos_w[:, 1],
            rollout.shank_samples_w[:, 1],
            rollout.thigh_samples_w[:, 1],
            terrain_field,
            cfg,
        ),
    )
    signed_ground_error = (
        rollout.foot_pos_w[:, 1, :, 2]
        - foot_query.height_w
        - float(cfg.gait.foot_contact_offset)
    )
    ground_gap_violation = torch.relu(
        signed_ground_error - float(cfg.solver.stance_ground_gap_limit_m)
    )
    ground_penetration_violation = torch.relu(
        -signed_ground_error - float(cfg.solver.stance_ground_penetration_limit_m)
    )
    ground_violation = torch.where(
        contact[:, 1],
        torch.maximum(ground_gap_violation, ground_penetration_violation),
        torch.zeros_like(signed_ground_error),
    ).amax(dim=1)
    return torch.stack(
        (
            collision_violation,
            stance_violation,
            ground_violation,
            attitude_violation,
            attitude_rate_violation,
            recovery_violation,
        ),
        dim=1,
    )


def _line_search_constraint_tolerance(
    violation_components: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Return per-component nonlinear tolerance for the x1 acceptance gate."""
    return constant_like(
        violation_components,
        "line_search_constraint_tolerance_collision_stance_ground_attitude_rate_recovery",
        (
            0.0,
            0.0,
            1.0e-5,
            0.0,
            0.0,
            1.0e-5,
        ),
    )


def _root_attitude_violation_components(
    rollout: JointMpcRollout,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    attitude_violation = torch.relu(
        rollout.state[:, 1, 3:5].abs()
        - float(cfg.solver.root_roll_pitch_limit_rad)
    ).amax(dim=1)
    rate_violation = torch.relu(
        rollout.control[:, 0, 3:5].abs()
        - float(cfg.solver.root_roll_pitch_rate_limit_rps)
    ).amax(dim=1)
    return torch.stack((attitude_violation, rate_violation), dim=1)


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
    stance_dual: Tensor | None = None,
    swing_target_w: Tensor | None = None,
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
    touchdown = torch.zeros_like(contact)
    touchdown[:, 1:] = torch.logical_and(contact[:, 1:], torch.logical_not(contact[:, :-1]))
    command_target_weight = (
        swing.to(dtype=foot.dtype)
        + float(cfg.losses.swing_touchdown_target_multiplier) * touchdown.to(dtype=foot.dtype)
    )
    swing_target = foot if swing_target_w is None else torch.as_tensor(
        swing_target_w, dtype=foot.dtype, device=foot.device
    )
    swing_weight = torch.as_tensor(swing_weight, dtype=foot.dtype, device=foot.device)
    foot_over_weight = torch.as_tensor(foot_over_weight, dtype=foot.dtype, device=foot.device)
    safe_landing_weight = torch.as_tensor(safe_landing_weight, dtype=foot.dtype, device=foot.device)
    relaxation = float(cfg.solver.barrier_relaxation)
    foot_gradient = torch.zeros_like(foot)

    def normalized_mask(mask: Tensor) -> Tensor:
        return mask.to(foot.dtype) / mask.sum(dim=(1, 2), keepdim=True).clamp_min(1).to(foot.dtype)

    all_normalizer = 1.0 / float(nodes * 4)
    swing_target_error = foot - swing_target
    swing_target_normalizer = normalized_mask(command_target_weight)
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
    equality_error = torch.cat(
        (foot[..., :2] - stance_anchor[..., :2], torch.zeros_like(stance_error).unsqueeze(-1)),
        dim=-1,
    )
    dual = (
        torch.zeros(batch, 4, 3, dtype=foot.dtype, device=foot.device)
        if stance_dual is None
        else torch.as_tensor(stance_dual, dtype=foot.dtype, device=foot.device)
    )
    support_safety = torch.sigmoid(
        (small_distance - float(cfg.gait.small_support_safety_margin))
        / float(cfg.gait.small_support_safety_temperature)
    ).pow(float(cfg.gait.small_support_safety_exponent))
    stance_xy_normalizer = normalized_mask(contact.to(dtype=foot.dtype) * support_safety)
    equality_contact = current_stance_segment_mask(contact)
    equality_mask = (
        equality_contact.to(dtype=foot.dtype) * support_safety
    ).unsqueeze(-1)
    equality_force = equality_mask * (
        dual[:, None] + float(cfg.solver.stance_equality_penalty) * equality_error
    )
    foot_gradient.add_(equality_force)
    stance_far_weight = torch.sigmoid(
        (
            small_distance.amin(dim=2)
            - float(cfg.gait.stance_ground_far_influence_radius)
        )
        / float(cfg.gait.stance_ground_far_temperature)
    )
    stance_ground_normalizer = normalized_mask(
        contact.to(dtype=foot.dtype) * support_safety
    ) + float(cfg.losses.stance_ground_far_gain) * normalized_mask(
        contact.to(dtype=foot.dtype) * stance_far_weight.unsqueeze(-1)
    )
    foot_gradient[..., 2].add_(
        float(cfg.losses.stance_ground_contact) * stance_ground_normalizer * 2.0 * stance_error
    )
    support_epsilon = 1.0e-6
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
        float(cfg.losses.stance_xy_lock) * stance_xy_normalizer.unsqueeze(-1) * 2.0 * stance_xy_error
    )
    small_touchdown_derivative = relaxed_barrier_derivative(
        small_distance - float(cfg.gait.small_touchdown_margin),
        relaxation=relaxation,
    )
    touchdown_normalizer = normalized_mask(contact)
    small_xy_gradient = (
        float(cfg.losses.small_object_touchdown_avoidance)
        * touchdown_normalizer
        * small_touchdown_derivative
    ).unsqueeze(-1) * small_gradient
    small_touchdown_point_gradient = torch.cat(
        (small_xy_gradient, torch.zeros_like(small_xy_gradient[..., :1])),
        dim=-1,
    )
    large_derivative = relaxed_barrier_derivative(large_distance - 0.03, relaxation=relaxation)
    foot_gradient[..., :2].add_(
        float(cfg.losses.large_foot_collision)
        * all_normalizer
        * large_derivative.unsqueeze(-1)
        * large_gradient
    )

    state_flat = rollout.state.reshape(batch * nodes, 18)
    raw_jacobian = complete_foot_jacobian(
        state_flat[:, :3],
        state_flat[:, 3:6],
        state_flat[:, 6:],
    ).reshape(batch, nodes, 4, 3, 18)
    jacobian = raw_jacobian.clone()
    if foot_query.height_gradient_w is not None:
        height_gradient = torch.as_tensor(
            foot_query.height_gradient_w,
            dtype=foot.dtype,
            device=foot.device,
        ).reshape(batch, nodes, 4, 2)
        jacobian[..., 2, :] = jacobian[..., 2, :] - torch.einsum(
            "btkd,btkdi->btki",
            height_gradient,
            raw_jacobian[..., :2, :],
        )
    jacobian[..., 2, 2:6] = 0.0
    state_gradient = torch.einsum("btkd,btkdi->bti", foot_gradient, jacobian)
    small_touchdown_jacobian = jacobian.clone()
    small_touchdown_jacobian[..., :6] = 0.0
    state_gradient = state_gradient + torch.einsum(
        "btkd,btkdi->bti",
        small_touchdown_point_gradient,
        small_touchdown_jacobian,
    )
    swing_target_jacobian = raw_jacobian.clone()
    swing_target_jacobian[..., :3] = 0.0
    swing_target_gradient = (
        float(cfg.losses.swing_nominal_shape)
        * swing_target_normalizer.unsqueeze(-1)
        * 2.0
        * swing_target_error
    )
    small_target_normalizer = normalized_mask(
        command_target_weight * small_influence
    )
    swing_target_gradient = swing_target_gradient + (
        float(cfg.losses.small_object_foot_over)
        * small_target_normalizer.unsqueeze(-1)
        * 2.0
        * swing_target_error
    )
    state_gradient = state_gradient + torch.einsum(
        "btkd,btkdi->bti",
        swing_target_gradient,
        swing_target_jacobian,
    )
    stance_axis_weight = torch.stack(
        (
            float(cfg.losses.stance_xy_lock) * stance_xy_normalizer,
            float(cfg.losses.stance_xy_lock) * stance_xy_normalizer,
            float(cfg.losses.stance_ground_contact) * stance_ground_normalizer
            + float(cfg.losses.small_object_safe_landing) * landing_normalizer
            + support_node_weight * support_error_derivative,
        ),
        dim=-1,
    )
    equality_axis = constant_like(foot, "stance_equality_axis", (1.0, 1.0, 0.0))
    stance_axis_weight = stance_axis_weight + (
        float(cfg.solver.stance_equality_penalty)
        * equality_contact.to(dtype=foot.dtype).unsqueeze(-1)
        * support_safety.unsqueeze(-1)
        * equality_axis
    )
    weighted_jacobian = stance_axis_weight.unsqueeze(-1) * jacobian
    state_curvature = 2.0 * torch.einsum("btkdi,btkdj->btij", jacobian, weighted_jacobian)
    swing_target_weighted_jacobian = (
        float(cfg.losses.swing_nominal_shape)
        * swing_target_normalizer.unsqueeze(-1).unsqueeze(-1)
        * swing_target_jacobian
    )
    state_curvature = state_curvature + 2.0 * torch.einsum(
        "btkdi,btkdj->btij",
        swing_target_jacobian,
        swing_target_weighted_jacobian,
    )
    matrix_q = problem.matrix_q.clone()
    vector_q = problem.vector_q.clone()
    terminal_q = problem.terminal_q.clone()
    terminal_vector = problem.terminal_vector.clone()
    trust_scale = state_gradient.new_full((18,), float(cfg.solver.joint_trust_scale))
    trust_scale[:6] = float(cfg.solver.root_xy_trust_scale)
    vector_q.add_(state_gradient[:, :-1])
    terminal_vector.add_(state_gradient[:, -1])
    matrix_q.add_(torch.diag_embed(state_gradient[:, :-1].abs() / trust_scale))
    matrix_q.add_(state_curvature[:, :-1])
    terminal_q.add_(torch.diag_embed(state_gradient[:, -1].abs() / trust_scale))
    terminal_q.add_(state_curvature[:, -1])
    matrix_q = 0.5 * (matrix_q + matrix_q.transpose(-1, -2))
    terminal_q = 0.5 * (terminal_q + terminal_q.transpose(-1, -2))
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
    command_body: Tensor | None = None,
) -> LqProblem:
    """Add signed-distance full-body gradients to the RTI LQ direction."""
    batch, nodes = int(rollout.state.shape[0]), int(rollout.state.shape[1])
    state_flat = rollout.state.reshape(batch * nodes, 18)
    foot_jacobian = complete_foot_jacobian(
        state_flat[:, :3],
        state_flat[:, 3:6],
        state_flat[:, 6:],
    ).reshape(batch, nodes, 4, 1, 3, 18)
    knee_jacobian = complete_knee_jacobian(
        state_flat[:, :3],
        state_flat[:, 3:6],
        state_flat[:, 6:],
    ).reshape(batch, nodes, 4, 1, 3, 18)
    body_jacobian = complete_body_sample_jacobian(
        state_flat[:, 3:6],
        rollout.body_samples_w.reshape(batch * nodes, -1, 3),
        state_flat[:, :3],
    ).reshape(batch, nodes, 1, -1, 3, 18)
    link_jacobian = complete_link_sample_jacobians(
        state_flat[:, :3],
        state_flat[:, 3:6],
        state_flat[:, 6:],
    )
    calf_jacobian = link_jacobian.calf_samples.reshape(batch, nodes, 4, 3, 3, 18)
    thigh_jacobian = link_jacobian.thigh_samples.reshape(batch, nodes, 4, 3, 3, 18)
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
    restoration_gain = rollout.state.new_full((batch,), float(cfg.gait.collision_restoration_gain))
    root_command_projection = rollout.state.new_zeros((batch, 2, 2))
    if command_body is not None:
        command = torch.as_tensor(command_body, dtype=rollout.state.dtype, device=rollout.state.device)
        command_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
        command_world = body_linear_velocity_to_world(command[:, :2], rollout.state[:, 0, 5])
        command_axis = command_world / torch.linalg.vector_norm(command_world, dim=1, keepdim=True).clamp_min(1.0e-6)
        root_command_projection = torch.einsum("bi,bj->bij", command_axis, command_axis)
        restoration_gain = restoration_gain + float(cfg.gait.collision_restoration_speed_gain) * (
            1.0 - torch.exp(-command_speed / float(cfg.gait.collision_restoration_speed_scale))
        )
    joint_lower = constant_like(rollout.state, "root_assist_joint_lower", (-1.0472, -0.6632, -2.721) * 4)
    joint_upper = constant_like(rollout.state, "root_assist_joint_upper", (1.0472, 2.966, -0.837) * 4)
    joint_margin = torch.minimum(
        rollout.state[..., 6:] - joint_lower,
        joint_upper - rollout.state[..., 6:],
    ).amin(dim=-1)
    reachability_pressure = torch.sigmoid((0.25 - joint_margin) / 0.05)

    def add_part(
        positions: Tensor,
        query: JointMpcTerrainQuery,
        jacobian: Tensor,
        *,
        radius: float,
        weight: float,
        xy_scale: float | None = None,
    ) -> None:
        if float(weight) == 0.0:
            return
        position = torch.as_tensor(positions).reshape(batch, nodes, int(positions.shape[2]), -1, 3)
        group_count = int(position.shape[2])
        sample_count = int(position.shape[3])
        distance = query.small_distance_m.reshape(batch, nodes, group_count, sample_count)
        height = query.height_w.reshape(batch, nodes, group_count, sample_count)
        sdf_gradient = query.small_gradient_w.reshape(batch, nodes, group_count, sample_count, 2)
        proximity = torch.sigmoid((influence_radius - distance) / temperature)
        normalizer = proximity.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        effective_height = height + float(cfg.gait.small_semantic_height) * torch.sigmoid(
            distance / temperature
        )
        vertical_argument = (
            effective_height + float(radius) + margin_z - position[..., 2]
        ) / temperature
        vertical_penalty = torch.nn.functional.softplus(vertical_argument)
        clearance = distance - float(radius) - margin_xy
        barrier = relaxed_barrier(clearance, relaxation=relaxation)
        derivative = relaxed_barrier_derivative(clearance, relaxation=relaxation)
        exact_vertical = torch.logical_and(
            position[..., 2] - float(radius) < height,
            position[..., 2] + float(radius) > 0.0,
        )
        penetration = torch.where(
            exact_vertical,
            (float(radius) - distance).clamp_min(0.0),
            torch.zeros_like(distance),
        )
        first_step = (
            torch.arange(nodes, device=position.device) == 1
        ).to(dtype=position.dtype)
        restoration_scale = 1.0 + (
            float(cfg.gait.collision_restoration_gain)
            * first_step.view(1, nodes, 1, 1)
            * penetration
            / max(float(radius), margin_xy, 1.0e-6)
        )
        factor = float(weight) * proximity * restoration_scale / normalizer
        gradient_xy = (
            (
                float(cfg.gait.small_collision_link_xy_scale)
                if xy_scale is None
                else float(xy_scale)
            )
            * factor.unsqueeze(-1)
            * vertical_penalty.unsqueeze(-1)
            * derivative.unsqueeze(-1)
            * sdf_gradient
        )
        gradient_z = (
            -float(cfg.gait.small_collision_vertical_scale)
            * factor
            * torch.sigmoid(vertical_argument)
            * barrier
            / temperature
        )
        point_gradient = torch.cat((gradient_xy, gradient_z.unsqueeze(-1)), dim=-1)
        scaled_jacobian = jacobian.clone()
        proximity_release = proximity.amax(dim=(2, 3))
        root_assist_weight = proximity_release * (0.10 + 0.90 * reachability_pressure)
        identity_xy = torch.eye(2, dtype=rollout.state.dtype, device=rollout.state.device).view(1, 1, 2, 2)
        command_projection = root_command_projection[:, None]
        root_xy_projection = command_projection + root_assist_weight[:, :, None, None] * (
            identity_xy - command_projection
        )
        scaled_jacobian[..., :2] = (
            float(cfg.gait.small_collision_root_xy_scale)
            * torch.einsum(
                "bnlsdi,bnij->bnlsdj",
                scaled_jacobian[..., :2],
                root_xy_projection,
            )
        )
        # Keep vertical joint columns so the clearance gradient can lift the
        # leg; only root z is excluded from obstacle assistance.
        scaled_jacobian[..., 2, :6] = 0.0
        scaled_jacobian[..., 3:6] = (
            scaled_jacobian[..., 3:6] * root_assist_weight[:, :, None, None, None, None]
        )
        sample_state_gradient = torch.einsum(
            "bnlsd,bnlsdi->bnlsi",
            point_gradient,
            scaled_jacobian,
        )
        state_gradient = sample_state_gradient.sum(dim=(2, 3))
        sample_loss = (factor * vertical_penalty * barrier).clamp_min(1.0e-6)
        state_curvature = 0.5 * torch.einsum(
            "bnlsi,bnlsj->bnij",
            sample_state_gradient / sample_loss.sqrt().unsqueeze(-1),
            sample_state_gradient / sample_loss.sqrt().unsqueeze(-1),
        )
        trust = state_gradient.new_full((18,), joint_trust)
        trust[:6] = root_trust
        vector_q.add_(state_gradient[:, :-1])
        terminal_vector.add_(state_gradient[:, -1])
        matrix_q.add_(state_curvature[:, :-1])
        matrix_q.add_(torch.diag_embed(state_gradient[:, :-1].abs() / trust))
        terminal_q.add_(state_curvature[:, -1])
        terminal_q.add_(torch.diag_embed(state_gradient[:, -1].abs() / trust))

    add_part(
        rollout.foot_pos_w.unsqueeze(3),
        queries.foot,
        foot_jacobian,
        radius=cfg.gait.foot_collision_radius,
        weight=cfg.losses.small_object_foot_clearance,
        xy_scale=cfg.gait.small_collision_foot_xy_scale,
    )
    add_part(
        rollout.knee_pos_w.unsqueeze(3),
        queries.knee,
        knee_jacobian,
        radius=cfg.gait.knee_collision_radius,
        weight=cfg.losses.small_object_knee_clearance,
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
    add_part(
        rollout.body_samples_w.unsqueeze(2),
        queries.body,
        body_jacobian,
        radius=0.0,
        weight=cfg.losses.small_object_base_clearance,
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
    support_safety = torch.sigmoid(
        (
            foot_query.small_distance_m.reshape(batch, nodes, 4)
            - float(cfg.gait.small_support_safety_margin)
        )
        / float(cfg.gait.small_support_safety_temperature)
    ).pow(float(cfg.gait.small_support_safety_exponent))
    support_weight = contact * support_safety
    support_height = reliable_support_height(
        foot_height,
        support_weight,
        temperature=float(cfg.gait.root_support_height_temperature),
        root_pos_z=rollout.state[..., 2],
    )
    error = rollout.state[..., 2] - support_height - 0.32
    weight = float(cfg.losses.root_support_height) / float(nodes)
    vector_q = problem.vector_q.clone()
    matrix_q = problem.matrix_q.clone()
    terminal_vector = problem.terminal_vector.clone()
    terminal_q = problem.terminal_q.clone()
    vector_q[..., 2].add_(2.0 * weight * error[:, :-1])
    matrix_q[..., 2, 2].add_(2.0 * weight)
    terminal_vector[..., 2].add_(2.0 * weight * error[:, -1])
    terminal_q[..., 2, 2].add_(2.0 * weight)
    return replace(
        problem,
        matrix_q=matrix_q,
        vector_q=vector_q,
        terminal_q=terminal_q,
        terminal_vector=terminal_vector,
    )


_COMPILED_BUILD_LQ_PROBLEM = torch.compile(
    _build_lq_problem,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)
_COMPILED_ADD_LARGE_OBSTACLE_LINEARIZATION = torch.compile(
    _add_large_obstacle_linearization,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)
_COMPILED_ADD_SMALL_OBSTACLE_LINEARIZATION = torch.compile(
    _add_small_obstacle_linearization,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)
_COMPILED_ADD_FOOT_TERRAIN_LINEARIZATION = torch.compile(
    _add_foot_terrain_linearization,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)
_COMPILED_ADD_ROOT_SUPPORT_LINEARIZATION = torch.compile(
    _add_root_support_linearization,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)
_COMPILED_QUERY_LINEARIZATION_GEOMETRY = torch.compile(
    _query_linearization_geometry,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)
_COMPILED_DESIRED_CONTROL = torch.compile(
    _desired_control,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)
_COMPILED_STANCE_ANCHOR_TARGETS = torch.compile(
    _stance_anchor_targets,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)
_COMPILED_ROLLOUT_CONTROLS = torch.compile(
    rollout_controls,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)


def _linearization_function(eager, compiled, enabled: bool):
    return compiled if bool(enabled) else eager


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
    command_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    command_active = command_speed > float(cfg.gait.zero_translation_command_deadband)
    previous_command = (
        torch.zeros_like(command)
        if solver_state is None or solver_state.previous_command_body is None
        else torch.as_tensor(solver_state.previous_command_body, dtype=command.dtype, device=command.device)
    )
    previous_speed = torch.linalg.vector_norm(previous_command[:, :2], dim=1)
    previous_active = previous_speed > float(cfg.gait.zero_translation_command_deadband)
    direction_cosine = (command[:, :2] * previous_command[:, :2]).sum(dim=1) / (
        command_speed * previous_speed
    ).clamp_min(1.0e-6)
    command_onset = torch.logical_and(
        command_active,
        torch.logical_or(
            torch.logical_not(previous_active),
            direction_cosine < float(cfg.gait.startup_direction_cosine),
        ),
    )
    previous_start_age = (
        torch.full_like(command_speed, -1, dtype=torch.long)
        if solver_state is None or solver_state.command_start_age is None
        else torch.as_tensor(solver_state.command_start_age, dtype=torch.long, device=command.device)
    )
    command_start_age = torch.where(
        command_onset,
        torch.zeros_like(previous_start_age),
        torch.where(command_active, previous_start_age + 1, torch.full_like(previous_start_age, -1)),
    )
    startup_mask = command_start_age == 0
    previous_start_origin = (
        measured_state.root_pos_w
        if solver_state is None or solver_state.command_start_origin_w is None
        else torch.as_tensor(
            solver_state.command_start_origin_w,
            dtype=measured_state.root_pos_w.dtype,
            device=measured_state.device,
        )
    )
    command_start_origin = torch.where(
        command_onset.unsqueeze(-1), measured_state.root_pos_w, previous_start_origin
    )
    phase_step = (
        torch.zeros(measured_state.batch_size, dtype=torch.long, device=measured_state.device)
        if solver_state is None or solver_state.gait_phase is None
        else torch.as_tensor(solver_state.gait_phase, dtype=torch.long, device=measured_state.device)
    )
    phase_age_reference = torch.remainder(
        phase_step[:, None], int(cfg.gait.half_cycle_steps)
    ).expand(-1, 4)
    scheduler_advance = None
    confirmed_touchdown_x1 = torch.zeros(
        measured_state.batch_size, 4, dtype=torch.bool, device=measured_state.device
    )
    recovery_landing_x1 = torch.zeros_like(confirmed_touchdown_x1)
    touchdown_ready, reliable_stance, measured_foot_small_distance = _measured_touchdown_readiness(
        measured_state,
        terrain_field,
        cfg,
        return_foot_distance=True,
    )
    if (
        solver_state is not None
        and solver_state.contact_state is not None
        and solver_state.phase_age is not None
        and solver_state.swing_extension_age is not None
        and solver_state.stance_age is not None
        and solver_state.recovery_state is not None
    ):
        current_contact_state = torch.as_tensor(
            solver_state.contact_state, dtype=torch.bool, device=measured_state.device
        )
        current_phase_age = torch.as_tensor(
            solver_state.phase_age, dtype=torch.long, device=measured_state.device
        )
        phase_age_reference = current_phase_age
        current_extension_age = torch.as_tensor(
            solver_state.swing_extension_age, dtype=torch.long, device=measured_state.device
        )
        current_stance_age = torch.as_tensor(
            solver_state.stance_age, dtype=torch.long, device=measured_state.device
        )
        current_recovery = torch.as_tensor(
            solver_state.recovery_state, dtype=torch.bool, device=measured_state.device
        )
        transition_due = current_phase_age >= int(cfg.gait.half_cycle_steps) - 1
        touchdown_due = torch.logical_and(torch.logical_not(current_contact_state), transition_due)
        liftoff_due = torch.logical_and(current_contact_state, transition_due)
        scheduler_advance = advance_contact_scheduler(
            contact_state=current_contact_state,
            phase_age=current_phase_age,
            swing_extension_age=current_extension_age,
            stance_age=current_stance_age,
            recovery_state=current_recovery,
            touchdown_scheduled=touchdown_due,
            touchdown_ready=touchdown_ready,
            liftoff_scheduled=liftoff_due,
            reliable_stance=torch.logical_and(current_contact_state, reliable_stance),
            max_swing_extension_steps=int(cfg.gait.max_swing_extension_steps),
        )
        confirmed_touchdown_x1 = torch.logical_and(
            torch.logical_not(current_contact_state),
            scheduler_advance.contact_state,
        )
        recovery_landing_x1 = scheduler_advance.recovery_state
        contact = adaptive_contact_schedule(
            contact_state=current_contact_state,
            phase_age=current_phase_age,
            touchdown_ready=touchdown_ready,
            horizon_steps=int(cfg.runtime.horizon_steps),
            half_cycle_steps=int(cfg.gait.half_cycle_steps),
        )
        contact = contact.clone()
        contact[:, 1] = scheduler_advance.contact_state
    else:
        contact = fixed_trot_schedule(
            measured_state.batch_size,
            int(cfg.runtime.horizon_steps),
            measured_state.device,
            half_cycle_steps=int(cfg.gait.half_cycle_steps),
            phase_offset_steps=phase_step,
        )
    optimization_command = command
    root_progress_scale = command.new_ones((measured_state.batch_size,))
    if scheduler_advance is not None:
        root_progress_scale = scheduler_advance.progress_scale.to(dtype=command.dtype)
        optimization_command = command.clone()
        optimization_command[:, :2] = (
            optimization_command[:, :2] * root_progress_scale.unsqueeze(-1)
        )
    swing_weight = _swing_phase_weight(contact, phase_age_reference, cfg, dtype=measured_state.root_pos_w.dtype)
    foot_over_weight, safe_landing_weight = _small_swing_handoff_weights(
        contact,
        phase_age_reference,
        cfg,
        dtype=measured_state.root_pos_w.dtype,
    )
    compile_linearization = bool(cfg.solver.compile_kernels) and measured_state.root_pos_w.is_cuda
    desired_control, joint_target = _linearization_function(
        _desired_control,
        _COMPILED_DESIRED_CONTROL,
        compile_linearization,
    )(measured_state, optimization_command, contact, phase_age_reference, cfg)
    joint_target = _recovery_joint_targets(joint_target, recovery_landing_x1, cfg)
    desired_control = _control_from_joint_target(
        measured_state,
        optimization_command,
        joint_target,
        cfg,
    )
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
    nominal_preview_query = _query_world(
        terrain_field,
        nominal_foot_pos_w.reshape(batch, nodes * 4, 3),
        cfg,
    )
    nominal_preview_height = nominal_preview_query.height_w.reshape(batch, nodes, 4)
    nominal_preview_safety = torch.sigmoid(
        (
            nominal_preview_query.small_distance_m.reshape(batch, nodes, 4)
            - float(cfg.gait.small_support_safety_margin)
        )
        / float(cfg.gait.small_support_safety_temperature)
    )
    terminal_preview_height = nominal_preview_height[:, -5:]
    terminal_preview_safety = nominal_preview_safety[:, -5:]
    terminal_support_safe = terminal_preview_safety > 0.5
    terminal_support_available = terminal_support_safe.any(dim=(1, 2))
    nominal_support_surface_high = torch.where(
        terminal_support_safe,
        terminal_preview_height,
        torch.full_like(terminal_preview_height, -1.0e9),
    ).amax(dim=(1, 2))
    nominal_support_surface_low = torch.where(
        terminal_support_safe,
        terminal_preview_height,
        torch.full_like(terminal_preview_height, 1.0e9),
    ).amin(dim=(1, 2))
    nominal_support_surface_high = torch.where(
        terminal_support_available,
        nominal_support_surface_high,
        torch.zeros_like(nominal_support_surface_high),
    )
    nominal_support_surface_low = torch.where(
        terminal_support_available,
        nominal_support_surface_low,
        nominal_support_surface_high,
    )
    high_support_target = nominal_support_surface_high + 0.32
    low_support_target = nominal_support_surface_low + 0.32
    descending_terrain = torch.logical_and(
        high_support_target <= measured_state.root_pos_w[:, 2] + 1.0e-4,
        low_support_target < measured_state.root_pos_w[:, 2] - 0.02,
    )
    root_support_target = torch.where(
        descending_terrain,
        low_support_target,
        high_support_target,
    )
    root_support_rate = (
        root_support_target - measured_state.root_pos_w[:, 2]
    ) / (float(nodes) * float(cfg.runtime.dt))
    support_rise = (
        root_support_target - measured_state.root_pos_w[:, 2]
    ).clamp_min(0.0)
    terrain_height_range = (
        terrain_field.height_w.amax(dim=(1, 2))
        - terrain_field.height_w.amin(dim=(1, 2))
    )
    terrain_step_present = terrain_height_range > 0.02
    support_rise = torch.where(
        terrain_step_present,
        support_rise,
        torch.zeros_like(support_rise),
    )
    desired_control = desired_control.clone()
    terrain_rise_rate = torch.where(
        terrain_step_present[:, None],
        torch.where(
            support_rise[:, None] > 0.02,
            desired_control.new_full((batch, 1), float(cfg.solver.root_vertical_direction_limit)),
            root_support_rate[:, None].clamp(-0.01, 0.2),
        ),
        desired_control[..., 2],
    )
    desired_control[..., 2] = terrain_rise_rate
    preview_time = (
        torch.arange(
            int(desired_control.shape[1]),
            dtype=desired_control.dtype,
            device=desired_control.device,
        )
        + 1.0
    ) * float(cfg.runtime.dt)
    predicted_root_z = (
        measured_state.root_pos_w[:, 2, None]
        + desired_control[..., 2] * preview_time[None]
    )
    completed_rise = (
        predicted_root_z - measured_state.root_pos_w[:, 2, None]
    ).clamp_min(0.0)
    terrain_progress_scale = torch.where(
        support_rise[:, None] > 0.02,
        (completed_rise / support_rise[:, None].clamp_min(1.0e-4)).clamp(0.1, 1.0),
        torch.ones_like(completed_rise),
    )
    desired_control[..., :2] = (
        desired_control[..., :2] * terrain_progress_scale.unsqueeze(-1)
    )
    nominal_state = rollout_state_sequence(
        measured_state,
        desired_control,
        dt=float(cfg.runtime.dt),
        compile_kernels=bool(cfg.solver.compile_kernels),
    )
    nominal_foot_pos_w = go2_foot_pos(
        nominal_state[..., :3].reshape(batch * nodes, 3),
        nominal_state[..., 3:6].reshape(batch * nodes, 3),
        nominal_state[..., 6:].reshape(batch * nodes, 12),
    ).reshape(batch, nodes, 4, 3)
    nominal_foot_pos_w = _command_conditioned_foot_targets(
        nominal_foot_pos_w,
        nominal_state,
        command,
        contact,
        phase_age_reference,
        cfg,
        progress_scale=None,
    )
    nominal_foot_query = _query_world(
        terrain_field,
        nominal_foot_pos_w.reshape(batch, nodes * 4, 3),
        cfg,
    )
    nominal_small_distance = nominal_foot_query.small_distance_m.reshape(batch, nodes, 4)
    nominal_near_small = nominal_small_distance < float(cfg.gait.small_collision_influence_radius)
    future_near_small = torch.flip(
        torch.cummax(torch.flip(nominal_near_small, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    swing_near_small_latch = future_near_small | torch.cummax(
        nominal_near_small,
        dim=1,
    ).values
    semantic_small_present = (terrain_field.semantic_id == 1).reshape(batch, -1).any(dim=1)
    small_obstacle_present = semantic_small_present | swing_near_small_latch.any(dim=(1, 2))
    planned_liftoff = torch.zeros_like(contact)
    planned_liftoff[:, 1:] = torch.logical_and(
        torch.logical_not(contact[:, 1:]),
        contact[:, :-1],
    )
    early_release = torch.zeros_like(contact)
    release_trigger = future_near_small
    for offset in range(1, 9):
        eligible = torch.zeros_like(contact)
        eligible[:, :-offset] = planned_liftoff[:, offset:] & release_trigger[:, offset:]
        early_release = early_release | eligible
    for offset in range(9, 16):
        eligible = torch.zeros_like(contact)
        eligible[:, :-offset] = planned_liftoff[:, offset:] & release_trigger[:, offset:]
        early_release = early_release | eligible
    contact, promoted_early_touchdown, allowed_early_release = _support_guarded_early_handoff(
        contact,
        early_release,
        touchdown_ready,
        min_support=2,
    )
    confirmed_touchdown_x1 = torch.logical_or(
        confirmed_touchdown_x1,
        promoted_early_touchdown[:, 1],
    )
    swing_weight = _swing_phase_weight(
        contact,
        phase_age_reference,
        cfg,
        dtype=measured_state.root_pos_w.dtype,
    )
    foot_over_weight, safe_landing_weight = _small_swing_handoff_weights(
        contact,
        phase_age_reference,
        cfg,
        dtype=measured_state.root_pos_w.dtype,
    )
    obstacle_envelope_floor = swing_near_small_latch.to(nominal_state.dtype) * torch.logical_not(
        contact
    ).to(nominal_state.dtype)
    joint_target = _nominal_joint_target(
        contact,
        phase_age_reference,
        optimization_command,
        cfg,
        dtype=measured_state.root_pos_w.dtype,
        swing_envelope_floor=obstacle_envelope_floor,
    )
    joint_target = _recovery_joint_targets(
        joint_target,
        recovery_landing_x1,
        cfg,
        near_small=_recovery_near_small_mask(
            measured_foot_small_distance,
            cfg,
        )[:, None].expand(-1, nodes, -1),
    )
    desired_control = _control_from_joint_target(
        measured_state,
        optimization_command,
        joint_target,
        cfg,
    )
    carried_control = _control_from_joint_target(
        measured_state,
        optimization_command,
        joint_target,
        cfg,
        carry_clipped_error=True,
    )
    desired_control = torch.where(
        small_obstacle_present[:, None, None],
        carried_control,
        desired_control,
    )
    desired_control = desired_control.clone()
    desired_control[..., 2] = torch.where(
        terrain_step_present[:, None],
        terrain_rise_rate,
        desired_control[..., 2],
    )
    nominal_state = rollout_state_sequence(
        measured_state,
        desired_control,
        dt=float(cfg.runtime.dt),
        compile_kernels=bool(cfg.solver.compile_kernels),
    )
    nominal_foot_pos_w = go2_foot_pos(
        nominal_state[..., :3].reshape(batch * nodes, 3),
        nominal_state[..., 3:6].reshape(batch * nodes, 3),
        nominal_state[..., 6:].reshape(batch * nodes, 12),
    ).reshape(batch, nodes, 4, 3)
    nominal_foot_pos_w = _command_conditioned_foot_targets(
        nominal_foot_pos_w,
        nominal_state,
        command,
        contact,
        phase_age_reference,
        cfg,
        progress_scale=None,
    )
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
    stance_dual = (
        measured_foot_w.new_zeros((measured_state.batch_size, 4, 3))
        if solver_state is None or solver_state.stance_dual is None
        else torch.as_tensor(solver_state.stance_dual, dtype=measured_foot_w.dtype, device=measured_foot_w.device)
    )
    stance_anchor_w = _linearization_function(
        _stance_anchor_targets,
        _COMPILED_STANCE_ANCHOR_TARGETS,
        compile_linearization,
    )(
        nominal_foot_pos_w,
        contact,
        initial_anchor_w=previous_stance_anchor,
    )
    if solver_state is None:
        stance_anchor_w = stance_anchor_w.clone()
        stance_anchor_w[..., 2] = torch.where(
            contact[:, 0, None, :],
            stance_anchor_w.new_full(stance_anchor_w[..., 2].shape, float(cfg.gait.foot_contact_offset)),
            stance_anchor_w[..., 2],
        )
    stance_surface_x1 = stance_anchor_w[:, 1].clone()
    stance_surface_query = _query_world(terrain_field, stance_surface_x1, cfg)
    measured_surface_query = _query_world(terrain_field, measured_foot_w, cfg)
    if solver_state is None:
        initial_contact = contact[:, 0]
        initial_ground_z = stance_surface_query.height_w + float(cfg.gait.foot_contact_offset)
        stance_anchor_w = stance_anchor_w.clone()
        stance_anchor_w[..., 2] = torch.where(
            initial_contact[:, None, :],
            initial_ground_z[:, None, :],
            stance_anchor_w[..., 2],
        )
    measured_touchdown_anchor = measured_foot_w.clone()
    measured_touchdown_anchor[..., 2] = (
        measured_surface_query.height_w + float(cfg.gait.foot_contact_offset)
    )
    stance_surface_x1 = torch.where(
        confirmed_touchdown_x1.unsqueeze(-1),
        measured_touchdown_anchor,
        stance_surface_x1,
    )
    # The stance KKT consumes ``stance_anchor_w`` directly.  Keep its x1
    # touchdown height aligned with the confirmed terrain anchor as well as
    # the diagnostic ``stance_surface_x1`` view; otherwise a newly confirmed
    # foot can remain at its nominal swing height for one published frame.
    stance_anchor_w = stance_anchor_w.clone()
    stance_anchor_w[:, 1, :, 2] = torch.where(
        confirmed_touchdown_x1,
        measured_touchdown_anchor[..., 2],
        stance_anchor_w[:, 1, :, 2],
    )
    def leg_clearance_for_state(state_vector: Tensor) -> Tensor:
        geometry = go2_fk(state_vector[:, :3], state_vector[:, 3:6], state_vector[:, 6:])
        return _leg_small_horizontal_clearance(
            geometry.foot_pos_w,
            geometry.knee_pos_w,
            geometry.shank_samples_w,
            geometry.thigh_samples_w,
            terrain_field,
            cfg,
        )

    leg_clearance, recovery_leg_clearance = _split_stance_and_recovery_leg_clearance(
        measured_leg_clearance=leg_clearance_for_state(measured_state.as_vector()),
        nominal_leg_clearance=leg_clearance_for_state(nominal_state[:, 1]),
        stance_lookahead_margin=float(cfg.gait.stance_safety_lookahead_margin),
    )
    stance_surface_safety = torch.sigmoid(
        (leg_clearance - float(cfg.gait.small_collision_margin_xy))
        / float(cfg.gait.small_collision_temperature)
    ).pow(float(cfg.gait.small_support_safety_exponent))
    ground_safe_recovery = _recovery_grounding_active_mask(
        recovery_state=recovery_landing_x1,
        contact_state=contact[:, 1],
        map_valid=measured_surface_query.valid,
        foot_small_distance_m=measured_surface_query.small_distance_m,
        leg_landing_clearance_m=recovery_leg_clearance,
        cfg=cfg,
    )
    stance_surface_x1[..., 2] = (
        stance_surface_query.height_w + float(cfg.gait.foot_contact_offset)
    )
    projection_contact_x0 = contact[:, 0]
    base_control = _initial_control(
        desired_control,
        solver_state,
        joint_delta_limit=float(cfg.solver.joint_direction_limit),
    )
    base_control = _blend_recovery_joint_control(
        base_control,
        desired_control,
        root_progress_scale,
        recovery_state=recovery_landing_x1,
    )
    base_control = _enforce_startup_foot_lead(
        measured_state,
        base_control,
        command,
        contact[:, 1],
        startup_mask,
        cfg,
        release_mask=command_start_age == 1,
    )
    base_control = _enforce_joint_position_limits(measured_state, base_control, cfg)
    if solver_state is not None:
        base_control = _reserve_joint_candidate_direction_capacity(base_control, cfg)
    base_control = _enforce_root_assist_limits(
        measured_state,
        base_control,
        optimization_command,
        cfg,
    )
    base_control = _zero_command_root_x1(base_control, command, cfg)
    base_control, base_rollout = _apply_fk_contact_kkt(
        measured_state,
        base_control,
        command,
        optimization_command,
        terrain_field,
        projection_contact_x0,
        contact[:, 1],
        stance_surface_x1,
        recovery_landing_x1,
        startup_mask,
        command_start_age == 1,
        stance_surface_safety,
        confirmed_touchdown_x1,
        cfg,
    )
    hold_control = _hold_control_with_ground_safe_recovery(
        base_control,
        ground_safe_recovery,
    )
    hold_rollout = rollout_controls(
        measured_state,
        hold_control,
        dt=float(cfg.runtime.dt),
        compile_kernels=bool(cfg.solver.compile_kernels),
    )
    base_violation_components = _x1_constraint_violation_components(
        base_rollout,
        terrain_field,
        cfg,
        contact,
        stance_anchor_w,
        recovery_landing_x1,
    )
    hold_violation_components = _x1_constraint_violation_components(
        hold_rollout,
        terrain_field,
        cfg,
        contact,
        stance_anchor_w,
        recovery_landing_x1,
    )
    base_control = _select_safer_base_control(
        base_control,
        base_violation_components,
        hold_control,
        hold_violation_components,
    )
    base_rollout = rollout_controls(
        measured_state,
        base_control,
        dt=float(cfg.runtime.dt),
        compile_kernels=bool(cfg.solver.compile_kernels),
    )
    lq_problem = _linearization_function(
        _build_lq_problem,
        _COMPILED_BUILD_LQ_PROBLEM,
        compile_linearization,
    )(base_rollout, desired_control, joint_target, measured_state, optimization_command, cfg)
    linearization_queries = _linearization_function(
        _query_linearization_geometry,
        _COMPILED_QUERY_LINEARIZATION_GEOMETRY,
        compile_linearization,
    )(base_rollout, terrain_field, cfg)
    nominal_foot_pos_w = _sdf_corrected_foot_targets(
        nominal_foot_pos_w,
        contact,
        linearization_queries.foot,
        cfg,
    )
    stance_anchor_w = _linearization_function(
        _stance_anchor_targets,
        _COMPILED_STANCE_ANCHOR_TARGETS,
        compile_linearization,
    )(
        nominal_foot_pos_w,
        contact,
        initial_anchor_w=previous_stance_anchor,
    )
    if solver_state is None:
        stance_anchor_w = stance_anchor_w.clone()
        stance_anchor_w[..., 2] = torch.where(
            contact[:, 0, None, :],
            stance_anchor_w.new_full(stance_anchor_w[..., 2].shape, float(cfg.gait.foot_contact_offset)),
            stance_anchor_w[..., 2],
        )
    lq_problem = _linearization_function(
        _add_large_obstacle_linearization,
        _COMPILED_ADD_LARGE_OBSTACLE_LINEARIZATION,
        compile_linearization,
    )(lq_problem, base_rollout, linearization_queries, cfg)
    lq_problem = _linearization_function(
        _add_small_obstacle_linearization,
        _COMPILED_ADD_SMALL_OBSTACLE_LINEARIZATION,
        compile_linearization,
    )(lq_problem, base_rollout, linearization_queries, cfg, command)
    lq_problem = _linearization_function(
        _add_foot_terrain_linearization,
        _COMPILED_ADD_FOOT_TERRAIN_LINEARIZATION,
        compile_linearization,
    )(
        lq_problem,
        base_rollout,
        contact,
        swing_weight,
        foot_over_weight,
        safe_landing_weight,
        stance_anchor_w,
        linearization_queries.foot,
        cfg,
        stance_dual,
        nominal_foot_pos_w,
    )
    lq_problem = _linearization_function(
        _add_root_support_linearization,
        _COMPILED_ADD_ROOT_SUPPORT_LINEARIZATION,
        compile_linearization,
    )(lq_problem, base_rollout, contact, linearization_queries.foot, cfg)
    constraint_stance_anchor_w = _step_bounded_stance_anchor(
        stance_anchor_w,
        measured_foot_w,
        contact,
        max_step_m=float(cfg.solver.stance_target_step_limit_m),
    )
    lq_problem = _add_stance_control_constraints(
        lq_problem,
        base_rollout,
        contact,
        constraint_stance_anchor_w,
        cfg,
        recovery_state=recovery_landing_x1,
        foot_query=linearization_queries.foot,
        swing_target_w=nominal_foot_pos_w,
        terrain_field=terrain_field,
        startup_mask=startup_mask,
        command_body=command,
        initial_grounding=(contact[:, 0] if solver_state is None else torch.zeros_like(contact[:, 0])),
        recovery_ground_safe=ground_safe_recovery,
    )
    preview_solution = solve_lq_subproblem(
        lq_problem,
        regularization=float(cfg.solver.regularization),
    )
    lq_problem = _refine_predicted_joint_bound_constraints(
        lq_problem,
        base_rollout,
        preview_solution.delta_state,
        cfg,
        preview_delta_control=preview_solution.delta_control,
        base_control=base_control,
        command_body=command,
        startup_mask=startup_mask,
        contact_state=contact,
    )
    previous_control = (
        measured_state.joint_vel.new_zeros((measured_state.batch_size, 18))
        if solver_state is None
        else solver_state.previous_control
    )

    candidate_rollout_cache: list[JointMpcRollout] = []

    def evaluate_rollout(candidate_rollout: JointMpcRollout, repeats: int) -> Tensor:
        repeated_command = optimization_command[:, None].expand(-1, repeats, -1).reshape(-1, 3)
        repeated_contact = contact[:, None].expand(-1, repeats, -1, -1).reshape(
            measured_state.batch_size * repeats, *contact.shape[1:]
        )
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
            contact_state=repeated_contact,
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
        repeated_recovery = recovery_landing_x1[:, None].expand(-1, repeats, -1).reshape(
            measured_state.batch_size * repeats, 4
        )
        return total, _x1_constraint_violation_components(
            candidate_rollout,
            terrain_field,
            cfg,
            repeated_contact,
            repeated_stance_anchor,
            repeated_recovery,
        )

    base_merit, base_constraint_violation = evaluate_rollout(base_rollout, 1)
    def merit_fn(candidate_control: Tensor) -> Tensor:
        repeats = int(candidate_control.shape[0]) // measured_state.batch_size
        repeated_state = _repeat_state(measured_state, repeats)
        repeated_contact_x1 = contact[:, None, 1].expand(-1, repeats, -1).reshape(
            measured_state.batch_size * repeats, 4
        )
        repeated_contact_x0 = projection_contact_x0[:, None].expand(-1, repeats, -1).reshape(
            measured_state.batch_size * repeats, 4
        )
        repeated_anchor_x1 = stance_surface_x1[:, None].expand(-1, repeats, -1, -1).reshape(
            measured_state.batch_size * repeats, 4, 3
        )
        repeated_surface_safety = stance_surface_safety[:, None].expand(-1, repeats, -1).reshape(
            measured_state.batch_size * repeats, 4
        )
        repeated_confirmed_touchdown = confirmed_touchdown_x1[:, None].expand(-1, repeats, -1).reshape(
            measured_state.batch_size * repeats, 4
        )
        repeated_command = optimization_command[:, None].expand(-1, repeats, -1).reshape(
            measured_state.batch_size * repeats, 3
        )
        repeated_raw_command = command[:, None].expand(-1, repeats, -1).reshape(
            measured_state.batch_size * repeats, 3
        )
        candidate_control, candidate_rollout = _apply_fk_contact_kkt(
            repeated_state,
            _zero_command_root_x1(candidate_control, repeated_raw_command, cfg),
            repeated_raw_command,
            repeated_command,
            terrain_field,
            repeated_contact_x0,
            repeated_contact_x1,
            repeated_anchor_x1,
            recovery_landing_x1[:, None].expand(-1, repeats, -1).reshape(-1, 4),
            startup_mask[:, None].expand(-1, repeats).reshape(-1),
            (command_start_age == 1)[:, None].expand(-1, repeats).reshape(-1),
            repeated_surface_safety,
            repeated_confirmed_touchdown,
            cfg,
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
        coupled_state_riccati=bool(cfg.solver.coupled_state_riccati),
        base_merit=base_merit,
        base_constraint_violation=base_constraint_violation,
        constraint_tolerance=_line_search_constraint_tolerance(
            base_constraint_violation,
            cfg,
        ),
        recover_control_direction=lambda solution: _scale_constrained_control_direction(
            solution.delta_control,
            solution.delta_state,
            lq_problem.constraint_control,
            lq_problem.constraint_state,
            lq_problem.constraint_residual,
            limits=_root_direction_limits(
                constant_like(
                    solution.delta_control,
                    "joint_mpc_control_direction_limits",
                    (float(cfg.solver.root_linear_direction_limit),) * 3
                    + (float(cfg.solver.root_angular_direction_limit),) * 3
                    + (float(cfg.solver.joint_direction_limit),) * 12,
                ).view(1, 1, -1).expand_as(solution.delta_control),
                terrain_step_present[:, None],
                vertical_limit=float(cfg.solver.root_vertical_direction_limit),
                linear_limit=float(cfg.solver.root_linear_direction_limit),
            ),
            base_control=base_control,
            required_absolute_limits=torch.cat(
                (
                    solution.delta_control.new_zeros(
                        solution.delta_control.shape[0],
                        solution.delta_control.shape[1],
                        6,
                    ),
                    torch.where(
                        small_obstacle_present[:, None, None],
                        solution.delta_control.new_full(
                            (solution.delta_control.shape[0], solution.delta_control.shape[1], 12),
                            _joint_candidate_absolute_limit(cfg),
                        ),
                        solution.delta_control.new_zeros(
                            solution.delta_control.shape[0],
                            solution.delta_control.shape[1],
                            12,
                        ),
                    ),
                ),
                dim=-1,
            ),
            matrix_a=lq_problem.matrix_a,
            matrix_b=lq_problem.matrix_b,
            affine_dynamics=lq_problem.affine_dynamics,
            initial_state=lq_problem.initial_state,
            return_components=True,
        ),
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
            command_body=optimization_command,
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
            "collision_violation_before": update.base_constraint_violation,
            "collision_violation_after": update.constraint_violation,
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
    x1_equality_error = torch.cat(
        (
            rollout.foot_pos_w[:, 1, :, :2] - stance_anchor_w[:, 1, :, :2],
            torch.zeros_like(rollout.foot_pos_w[:, 1, :, 2:3]),
        ),
        dim=-1,
    )
    x1_continuing_stance = torch.logical_and(contact[:, 0], contact[:, 1]).unsqueeze(-1)
    next_stance_dual = torch.where(
        x1_continuing_stance,
        stance_dual
        + float(cfg.solver.stance_dual_step)
        * float(cfg.solver.stance_equality_penalty)
        * x1_equality_error,
        torch.zeros_like(stance_dual),
    )
    x1_touchdown = torch.logical_and(torch.logical_not(contact[:, 0]), contact[:, 1]).unsqueeze(-1)
    x1_initial_contact = (
        contact[:, 1].unsqueeze(-1) if solver_state is None else torch.zeros_like(x1_touchdown)
    )
    measured_surface_height = measured_surface_query.height_w
    confirmed_touchdown = torch.logical_or(
        x1_initial_contact.squeeze(-1),
        torch.logical_and(x1_touchdown.squeeze(-1), touchdown_ready),
    )
    if solver_state is None:
        confirmed_touchdown = torch.logical_or(
            confirmed_touchdown,
            x1_touchdown.squeeze(-1),
        )
    next_stance_anchor = _confirmed_stance_anchor(
        previous_anchor_w=stance_anchor_w[:, 1],
        measured_foot_w=measured_foot_w,
        terrain_height_w=measured_surface_height,
        confirmed_touchdown=confirmed_touchdown,
        foot_contact_offset=float(cfg.gait.foot_contact_offset),
    )
    # A touchdown becomes the next rolling anchor at the published x1 pose.
    # This keeps the FK/KKT constraint continuous across the contact transition;
    # readiness still controls whether the scheduler calls it reliable support.
    touchdown_event = torch.logical_and(torch.logical_not(contact[:, 0]), contact[:, 1])
    published_anchor = rollout.foot_pos_w[:, 1].detach().clone()
    published_anchor[..., 2] = (
        _query_world(terrain_field, published_anchor, cfg).height_w
        + float(cfg.gait.foot_contact_offset)
    )
    next_stance_anchor = torch.where(
        touchdown_event.unsqueeze(-1), published_anchor, next_stance_anchor
    )
    previous_contact_state = (
        contact[:, 0]
        if solver_state is None or solver_state.contact_state is None
        else torch.as_tensor(solver_state.contact_state, dtype=torch.bool, device=contact.device)
    )
    previous_phase_age = (
        torch.zeros_like(previous_contact_state, dtype=torch.long)
        if solver_state is None or solver_state.phase_age is None
        else torch.as_tensor(solver_state.phase_age, dtype=torch.long, device=contact.device)
    )
    previous_stance_age = (
        torch.zeros_like(previous_phase_age)
        if solver_state is None or solver_state.stance_age is None
        else torch.as_tensor(solver_state.stance_age, dtype=torch.long, device=contact.device)
    )
    if scheduler_advance is None:
        next_contact_state = contact[:, 1]
        contact_changed = next_contact_state != previous_contact_state
        next_phase_age = torch.where(
            contact_changed,
            torch.zeros_like(previous_phase_age),
            previous_phase_age + 1,
        )
        next_extension_age = torch.zeros_like(next_phase_age)
        next_stance_age = torch.where(
            next_contact_state,
            torch.where(
                contact_changed,
                torch.zeros_like(previous_stance_age),
                previous_stance_age + 1,
            ),
            torch.zeros_like(previous_stance_age),
        )
        next_recovery_state = torch.zeros_like(next_contact_state)
    else:
        (
            next_contact_state,
            next_phase_age,
            next_extension_age,
            next_stance_age,
            next_recovery_state,
        ) = _reconcile_published_contact_state(scheduler_advance, contact[:, 1])
    next_solver_state = JointMpcRtiSolverState(
        state=rollout.state,
        control=rollout.control,
        dual=update.lq_solution.dual,
        previous_control=rollout.control[:, 0],
        gait_phase=torch.remainder(phase_step + 1, 2 * int(cfg.gait.half_cycle_steps)),
        stance_anchor_w=next_stance_anchor,
        stance_dual=next_stance_dual,
        command_start_age=command_start_age,
        command_start_origin_w=command_start_origin,
        previous_command_body=command,
        contact_state=next_contact_state,
        phase_age=next_phase_age,
        swing_extension_age=next_extension_age,
        stance_age=next_stance_age,
        recovery_state=next_recovery_state,
    )
    return JointMpcRtiStepResult(
        full_trajectory=trajectory,
        pending_reference=pending,
        solver_state=next_solver_state,
    )


__all__ = ["step"]
