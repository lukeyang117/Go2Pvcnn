from __future__ import annotations

import torch

from .helpers import make_command, make_flat_field, make_state


def test_early_release_preserves_two_support_legs_per_horizon_node() -> None:
    from extension.joint_mpc_rti.planner import _guard_early_release_by_support

    contact = torch.tensor(
        [
            [[True, False, False, True]],
            [[True, True, True, False]],
        ]
    )
    requested = torch.tensor(
        [
            [[True, False, False, False]],
            [[False, True, True, False]],
        ]
    )

    allowed = _guard_early_release_by_support(contact, requested, min_support=2)

    assert torch.equal(allowed[0], torch.zeros(1, 4, dtype=torch.bool))
    assert allowed[1].sum() == 1
    assert (contact & torch.logical_not(allowed)).sum(dim=2).amin() >= 2


def test_early_release_promotes_grounded_safe_swing_leg_before_release() -> None:
    from extension.joint_mpc_rti.planner import _support_guarded_early_handoff

    contact = torch.tensor(
        [[[True, False, False, True], [True, False, False, True]]]
    )
    requested = torch.tensor(
        [[[False, False, False, False], [True, False, False, False]]]
    )
    touchdown_ready = torch.tensor([[False, True, True, True]])

    updated, promoted, released = _support_guarded_early_handoff(
        contact,
        requested,
        touchdown_ready,
        min_support=2,
    )

    assert torch.equal(updated[:, 0], contact[:, 0])
    assert torch.equal(released[0, 1], torch.tensor([True, False, False, False]))
    assert promoted[0, 1].sum() == 1
    assert updated[:, 1].sum() == 2


def test_published_early_handoff_is_saved_atomically_in_scheduler_state() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import ContactSchedulerAdvance
    from extension.joint_mpc_rti.planner import _reconcile_published_contact_state

    scheduled = ContactSchedulerAdvance(
        contact_state=torch.tensor([[False, True, False, True]]),
        phase_age=torch.tensor([[30, 5, 30, 5]]),
        swing_extension_age=torch.tensor([[10, 0, 10, 0]]),
        stance_age=torch.tensor([[0, 5, 0, 5]]),
        recovery_state=torch.tensor([[True, False, True, False]]),
        liftoff_blocked=torch.zeros(1, 4, dtype=torch.bool),
        progress_scale=torch.ones(1),
    )
    published = torch.tensor([[True, False, False, True]])

    contact, phase, extension, stance, recovery = _reconcile_published_contact_state(
        scheduled,
        published,
    )

    assert torch.equal(contact, published)
    assert torch.equal(phase, torch.tensor([[0, 0, 30, 5]]))
    assert torch.equal(extension, torch.tensor([[0, 0, 10, 0]]))
    assert torch.equal(stance, torch.tensor([[0, 0, 0, 5]]))
    assert torch.equal(recovery, torch.tensor([[False, False, True, False]]))


def test_x1_stance_target_moves_at_most_one_slip_limit_toward_persistent_anchor() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import _step_bounded_stance_anchor

    cfg = JointMpcRtiCfg()
    assert cfg.solver.stance_target_step_limit_m < cfg.solver.stance_equality_tolerance_m
    anchor = torch.zeros(1, 3, 4, 3)
    anchor[:, 1, 0, 0] = 0.001
    anchor[:, 1, 1, 0] = 0.001
    measured_foot = torch.zeros(1, 4, 3)
    contact = torch.tensor(
        [[[True, True, False, False], [True, True, False, True], [True, True, False, True]]]
    )

    bounded = _step_bounded_stance_anchor(
        anchor,
        measured_foot,
        contact,
        max_step_m=cfg.solver.stance_target_step_limit_m,
    )

    torch.testing.assert_close(bounded[0, 1, 0, 0], torch.tensor(0.00045))
    torch.testing.assert_close(bounded[0, 1, 1, 0], torch.tensor(0.00045))
    # A new touchdown is not a continuing stance row and keeps its published anchor.
    torch.testing.assert_close(bounded[0, 1, 3], anchor[0, 1, 3])
    # The persistent trajectory is not modified outside the published x1 target.
    torch.testing.assert_close(bounded[:, 0], anchor[:, 0])
    torch.testing.assert_close(bounded[:, 2], anchor[:, 2])


def test_stance_linearization_populates_root_leg_cross_hessian() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    command = make_command(1, vx=0.2)
    phase = torch.zeros(1, dtype=torch.long)
    contact = fixed_trot_schedule(1, cfg.runtime.horizon_steps, "cpu", half_cycle_steps=cfg.gait.half_cycle_steps)
    desired, joint_target = planner._desired_control(state, command, contact, phase, cfg)
    rollout = rollout_controls(state, desired, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(rollout, desired, joint_target, state, command, cfg)
    queries = planner._query_linearization_geometry(rollout, make_flat_field(1), cfg)
    anchor = planner._stance_anchor_targets(
        rollout.foot_pos_w,
        contact,
        initial_anchor_w=go2_foot_pos(state.root_pos_w, state.root_rpy_w, state.joint_pos),
    )
    swing = planner._swing_phase_weight(contact, phase, cfg, dtype=rollout.state.dtype)
    foot_over, landing = planner._small_swing_handoff_weights(contact, phase, cfg, dtype=rollout.state.dtype)

    coupled = planner._add_foot_terrain_linearization(
        problem,
        rollout,
        contact,
        swing,
        foot_over,
        landing,
        anchor,
        queries.foot,
        cfg,
    )

    root_leg_cross = coupled.matrix_q[0, 0, :6, 6:]
    assert torch.count_nonzero(root_leg_cross) > 0
    torch.testing.assert_close(coupled.matrix_q, coupled.matrix_q.transpose(-1, -2))


def test_lq_command_progress_is_a_terminal_root_position_task() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    command = make_command(1, vx=0.2)
    phase = torch.zeros(1, dtype=torch.long)
    contact = fixed_trot_schedule(1, cfg.runtime.horizon_steps, "cpu", half_cycle_steps=cfg.gait.half_cycle_steps)
    desired, joint_target = planner._desired_control(state, command, contact, phase, cfg)
    stopped_rollout = rollout_controls(state, torch.zeros_like(desired), dt=cfg.runtime.dt)

    problem = planner._build_lq_problem(stopped_rollout, desired, joint_target, state, command, cfg)

    assert problem.terminal_q[0, 0, 0] > 0.0
    assert problem.terminal_vector[0, 0] < 0.0


def test_command_foot_target_starts_from_liftoff_anchor_not_backward_nominal_fk() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    nominal_foot = torch.zeros(1, 4, 4, 3)
    nominal_foot[:, 1:, :, 0] = -0.1
    nominal_state = torch.zeros(1, 4, 18)
    contact = torch.tensor([[[True, False, False, True]]] * 4).reshape(1, 4, 4)

    target = planner._command_conditioned_foot_targets(
        nominal_foot,
        nominal_state,
        make_command(1, vx=0.2),
        contact,
        torch.zeros(1, dtype=torch.long),
        cfg,
    )

    assert torch.all(target[0, 1, 1:3, 0] > 0.0)
    torch.testing.assert_close(target[0, 1, (0, 3), 0], nominal_foot[0, 1, (0, 3), 0])


def test_stance_constraint_violation_only_covers_measured_stance_segment() -> None:
    from extension.joint_mpc_rti import planner

    contact = torch.tensor(
        [[[True, False], [True, False], [False, True], [True, True], [True, True]]]
    )
    anchor = torch.zeros(1, 5, 2, 3)
    foot = anchor.clone()
    foot[:, 1, 0, 0] = 0.004
    foot[:, 4, 1, 0] = 0.100

    violation = planner._current_stance_xy_constraint_violation(
        foot,
        anchor,
        contact,
        tolerance_m=0.0005,
    )

    torch.testing.assert_close(violation, torch.tensor([0.0035]))


def test_x1_stance_violation_ignores_future_horizon_error() -> None:
    from extension.joint_mpc_rti import planner

    contact = torch.tensor([[[True], [True], [True]]])
    anchor = torch.zeros(1, 3, 1, 3)
    foot = anchor.clone()
    foot[:, 1, 0, 0] = 0.004
    foot[:, 2, 0, 0] = 0.100

    violation = planner._x1_stance_xy_constraint_violation(
        foot,
        anchor,
        contact,
        tolerance_m=0.0005,
    )

    torch.testing.assert_close(violation, torch.tensor([0.0035]))


def test_first_stance_xy_lock_survives_zero_surface_safety() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.dynamics import kinematic_step
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    anchor = go2_foot_pos(state.root_pos_w, state.root_rpy_w, state.joint_pos)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[:, 0, 0] = 0.2
    contact = torch.tensor([[True, False, False, True]])

    projected = planner._enforce_first_stance_equality(
        state,
        control,
        make_command(1, vx=0.2),
        contact,
        contact,
        anchor,
        cfg,
        stance_surface_safety=torch.zeros(1, 4),
    )
    rollout = rollout_controls(state, projected, dt=cfg.runtime.dt)
    error = torch.linalg.vector_norm(
        rollout.foot_pos_w[:, 1, :, :2] - anchor[..., :2],
        dim=-1,
    )

    assert torch.all(error[contact] <= cfg.solver.stance_equality_tolerance_m)


def test_root_assist_is_reprojected_through_stance_equality() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.dynamics import kinematic_step
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=-1.0, vy=0.5, yaw=-1.0)
    measured_foot = go2_foot_pos(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    )
    anchor = measured_foot.clone()
    anchor[..., 2] = 0.022
    contact = torch.ones(1, 4, dtype=torch.bool)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[:, 0, :3] = torch.tensor([-1.0, 0.5, 0.0])
    control[:, 0, 5] = 0.0

    projected = planner._enforce_first_stance_equality(
        measured,
        control,
        command,
        contact,
        contact,
        anchor,
        cfg,
    )
    assisted = planner._enforce_root_assist_with_stance_equality(
        measured,
        projected,
        command,
        contact,
        contact,
        anchor,
        cfg,
    )
    next_state = kinematic_step(measured.as_vector(), assisted[:, 0], dt=cfg.runtime.dt)
    next_foot = go2_foot_pos(
        next_state[:, :3], next_state[:, 3:6], next_state[:, 6:]
    )
    residual = torch.linalg.vector_norm(next_foot[..., :2] - anchor[..., :2], dim=-1)
    assert residual.max() <= 5.0e-4


def test_first_stance_projection_respects_joint_control_trust_bound() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    anchor = go2_foot_pos(state.root_pos_w, state.root_rpy_w, state.joint_pos)
    anchor[..., 0] += 1.0
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    contact = torch.tensor([[True, False, False, True]])

    projected = planner._enforce_first_stance_equality(
        state,
        control,
        make_command(1, vx=0.2),
        contact,
        contact,
        anchor,
        cfg,
    )

    limit = cfg.gait.max_nominal_joint_velocity + cfg.solver.joint_direction_limit
    assert float(projected[:, 0, 6:].abs().max()) <= limit


def test_stance_ground_recovery_descends_without_one_step_snap() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    current = go2_foot_pos(state.root_pos_w, state.root_rpy_w, state.joint_pos)
    anchor = current.clone()
    anchor[..., 2] -= 0.10
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    contact = torch.tensor([[True, False, False, True]])

    projected = planner._enforce_first_stance_equality(
        state,
        control,
        make_command(1, vx=0.2),
        contact,
        contact,
        anchor,
        cfg,
    )
    next_foot = rollout_controls(state, projected, dt=cfg.runtime.dt).foot_pos_w[:, 1]
    descent = current[..., 2] - next_foot[..., 2]

    assert torch.all(descent[contact] > 0.0)
    assert torch.all(descent[contact] <= 0.021)


def test_lq_root_roll_pitch_loss_produces_an_upright_direction() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    state.root_rpy_w[:, :2] = torch.tensor([[0.2, -0.3]])
    command = make_command(1)
    phase = torch.zeros(1, dtype=torch.long)
    contact = fixed_trot_schedule(1, cfg.runtime.horizon_steps, "cpu", half_cycle_steps=cfg.gait.half_cycle_steps)
    desired, joint_target = planner._desired_control(state, command, contact, phase, cfg)
    rollout = rollout_controls(state, torch.zeros_like(desired), dt=cfg.runtime.dt)

    problem = planner._build_lq_problem(rollout, desired, joint_target, state, command, cfg)

    assert torch.all(problem.matrix_q[0, :, 3, 3] > 0.0)
    assert torch.all(problem.matrix_q[0, :, 4, 4] > 0.0)
    assert torch.all(problem.vector_q[0, :, 3] > 0.0)
    assert torch.all(problem.vector_q[0, :, 4] < 0.0)


def test_rolling_stance_is_world_anchored_instead_of_root_carried() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    cfg.solver.emit_loss_breakdown = False
    measured = make_state(1)
    command = make_command(1, vx=0.2)
    field = make_flat_field(1)
    solver_state = None
    previous_foot = None
    previous_contact = None
    max_slip = torch.tensor(0.0)
    slip_sum = torch.tensor(0.0)
    root_sum = torch.tensor(0.0)

    for _ in range(32):
        result = step(measured, command, field, solver_state, cfg)
        trajectory = result.full_trajectory
        foot = trajectory.foot_pos_w[:, 1]
        contact = trajectory.contact_state[:, 1]
        if previous_foot is not None and previous_contact is not None:
            stance = torch.logical_and(contact, previous_contact)
            foot_step = foot[..., :2] - previous_foot[..., :2]
            root_step = trajectory.state[:, 1, :2] - measured.root_pos_w[:, :2]
            slip = torch.linalg.vector_norm(foot_step, dim=-1)
            max_slip = torch.maximum(max_slip, torch.where(stance, slip, torch.zeros_like(slip)).max())
            slip_sum += torch.where(stance, torch.abs(foot_step[..., 0]), torch.zeros_like(slip)).sum()
            root_sum += (root_step[..., 0].abs() * stance.sum(dim=1)).sum()
        control = trajectory.control[:, 0]
        state = trajectory.state[:, 1]
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=control[:, :3],
            root_ang_vel_b=control[:, 3:6],
            joint_vel=control[:, 6:],
        )
        solver_state = result.solver_state
        previous_foot = foot
        previous_contact = contact

    assert max_slip <= 0.0005
    assert slip_sum / root_sum.clamp_min(1.0e-8) <= 0.10


def test_first_stance_equality_locks_confirmed_touchdown_legs() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.dynamics import kinematic_step
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[:, 0, 0] = 0.2
    anchor = go2_foot_pos(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos)
    anchor = anchor.clone()
    anchor[:, 1, 0] += 0.05
    anchor[:, 2, 0] += 0.05
    anchor[:, 1:3, 2] = 0.022
    contact_x0 = torch.tensor([[True, False, False, True]])
    contact_x1 = torch.tensor([[False, True, True, False]])

    projected = planner._enforce_first_stance_equality(
        measured,
        control,
        make_command(1, vx=0.2),
        contact_x0,
        contact_x1,
        anchor,
        cfg,
        confirmed_touchdown=torch.tensor([[False, True, True, False]]),
    )

    assert torch.count_nonzero(projected[:, 0, 6:] - control[:, 0, 6:]) > 0
    x1 = kinematic_step(measured.as_vector(), projected[:, 0], dt=cfg.runtime.dt)
    foot_x1 = go2_foot_pos(x1[:, :3], x1[:, 3:6], x1[:, 6:])
    assert torch.abs(foot_x1[:, 1:3, 2] - anchor[:, 1:3, 2]).max() <= 0.002


def test_small_link_collision_feasibility_matches_sphere_geometry() -> None:
    from extension.joint_mpc_rti import planner

    position = torch.tensor(
        [[[[0.0, 0.0, 0.05]], [[0.0, 0.0, 0.25]], [[0.0, 0.0, 0.05]], [[0.0, 0.0, 0.05]]]]
    )
    distance = torch.tensor([[[0.01], [0.01], [0.05], [0.01]]])
    height = torch.full((1, 4, 1), 0.16)

    collision = planner._sphere_link_collision(position, distance, height, radius=0.022)

    assert collision.squeeze(-1).tolist() == [[True, False, False, True]]


def test_minimum_norm_collision_correction_reduces_each_violated_leg() -> None:
    from extension.joint_mpc_rti import planner

    clearance = torch.tensor([[-0.02, 0.03, -0.01, 0.04]])
    jacobian = torch.zeros(1, 4, 3)
    jacobian[0, 0, 0] = 2.0
    jacobian[0, 1, 1] = 1.0
    jacobian[0, 2, 2] = -1.0
    jacobian[0, 3, 0] = 1.0

    correction = planner._minimum_norm_leg_correction(clearance, jacobian, max_norm=0.25)
    corrected = clearance + (jacobian * correction).sum(dim=-1)

    assert torch.all(corrected >= -1.0e-6)
    torch.testing.assert_close(correction[:, 1], torch.zeros_like(correction[:, 1]))
    torch.testing.assert_close(correction[:, 3], torch.zeros_like(correction[:, 3]))


def test_startup_projection_moves_swing_foot_before_root() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos
    from extension.joint_mpc_rti.model.dynamics import kinematic_step

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[:, 0, 0] = 0.2
    projected = planner._enforce_startup_foot_lead(
        measured,
        control,
        make_command(1, vx=0.2),
        torch.tensor([[True, False, False, True]]),
        torch.ones(1, dtype=torch.bool),
        cfg,
    )
    x1 = kinematic_step(measured.as_vector(), projected[:, 0], dt=cfg.runtime.dt)
    foot0 = go2_foot_pos(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos)
    foot1 = go2_foot_pos(x1[:, :3], x1[:, 3:6], x1[:, 6:])

    assert x1[0, 0] - measured.root_pos_w[0, 0] <= 0.0005
    assert torch.all(foot1[0, 1:3, 0] - foot0[0, 1:3, 0] >= 0.001)


def test_forward_swing_advances_relative_to_root() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    cfg.solver.emit_loss_breakdown = False
    measured = make_state(1)
    command = make_command(1, vx=0.2)
    field = make_flat_field(1)
    solver_state = None
    roots = [measured.root_pos_w.clone()]
    feet = [go2_foot_pos(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos)]
    contacts = [torch.tensor([[True, False, False, True]])]

    for _ in range(32):
        result = step(measured, command, field, solver_state, cfg)
        trajectory = result.full_trajectory
        roots.append(trajectory.state[:, 1, :3])
        feet.append(trajectory.foot_pos_w[:, 1])
        contacts.append(trajectory.contact_state[:, 1])
        control = trajectory.control[:, 0]
        state = trajectory.state[:, 1]
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=control[:, :3],
            root_ang_vel_b=control[:, 3:6],
            joint_vel=control[:, 6:],
        )
        solver_state = result.solver_state

    root = torch.stack(roots, dim=1)[0, :, 0]
    foot = torch.stack(feet, dim=1)[0, :, :, 0]
    contact = torch.stack(contacts, dim=1)[0]
    ratios: list[float] = []
    for leg in range(4):
        liftoff = torch.where(torch.logical_and(contact[:-1, leg], torch.logical_not(contact[1:, leg])))[0] + 1
        touchdown = torch.where(torch.logical_and(torch.logical_not(contact[:-1, leg]), contact[1:, leg]))[0] + 1
        for start in liftoff.tolist():
            end = touchdown[touchdown > start]
            if end.numel() == 0:
                continue
            stop = int(end[0])
            root_progress = root[stop] - root[start - 1]
            relative_progress = (foot[stop, leg] - root[stop]) - (foot[start - 1, leg] - root[start - 1])
            ratios.append(float(relative_progress / root_progress.clamp_min(1.0e-6)))

    assert ratios
    assert min(ratios) > 0.0
    assert sum(ratios) / len(ratios) >= 0.50
