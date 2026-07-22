from __future__ import annotations

from itertools import product
from dataclasses import replace

import torch

from .helpers import make_command, make_flat_field, make_state


def test_measured_nominal_flat_feet_are_grounded_and_reliable_for_scheduler() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import _measured_touchdown_readiness

    ready, reliable = _measured_touchdown_readiness(
        make_state(1),
        make_flat_field(1),
        JointMpcRtiCfg(),
    )

    assert torch.all(ready)
    assert torch.all(reliable)


def test_measured_touchdown_readiness_can_reuse_actual_foot_distance() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import _measured_touchdown_readiness

    ready, reliable, foot_distance = _measured_touchdown_readiness(
        make_state(1),
        make_flat_field(1),
        JointMpcRtiCfg(),
        return_foot_distance=True,
    )

    assert ready.shape == reliable.shape == foot_distance.shape == (1, 4)
    assert torch.all(foot_distance > 0.0)


def test_adaptive_contact_confirms_each_touchdown_leg_independently() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import advance_contact_scheduler

    result = advance_contact_scheduler(
        contact_state=torch.tensor([[False, True, True, False]]),
        phase_age=torch.tensor([[15, 15, 15, 15]]),
        swing_extension_age=torch.zeros(1, 4, dtype=torch.long),
        stance_age=torch.full((1, 4), 15, dtype=torch.long),
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
        touchdown_scheduled=torch.tensor([[True, False, False, True]]),
        touchdown_ready=torch.tensor([[True, False, False, False]]),
        liftoff_scheduled=torch.zeros(1, 4, dtype=torch.bool),
        reliable_stance=torch.tensor([[False, True, True, False]]),
        max_swing_extension_steps=10,
    )

    assert torch.equal(result.contact_state, torch.tensor([[True, True, True, False]]))
    assert torch.equal(result.swing_extension_age, torch.tensor([[0, 0, 0, 1]]))
    assert torch.equal(result.stance_age, torch.tensor([[0, 16, 16, 0]]))
    assert not bool(result.recovery_state[0, 3])


def test_safe_diagonal_touchdown_atomically_releases_previous_support_pair() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import advance_contact_scheduler

    result = advance_contact_scheduler(
        contact_state=torch.tensor([[True, False, False, True]]),
        phase_age=torch.full((1, 4), 14, dtype=torch.long),
        swing_extension_age=torch.zeros(1, 4, dtype=torch.long),
        stance_age=torch.full((1, 4), 14, dtype=torch.long),
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
        touchdown_scheduled=torch.tensor([[False, True, True, False]]),
        touchdown_ready=torch.tensor([[False, True, True, False]]),
        liftoff_scheduled=torch.tensor([[True, False, False, True]]),
        reliable_stance=torch.tensor([[True, False, False, True]]),
        max_swing_extension_steps=10,
    )

    assert torch.equal(result.contact_state, torch.tensor([[False, True, True, False]]))
    assert not torch.any(result.liftoff_blocked)
    torch.testing.assert_close(result.progress_scale, torch.ones(1))


def test_adaptive_schedule_delays_only_the_unready_touchdown_leg() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import adaptive_contact_schedule

    contact = adaptive_contact_schedule(
        contact_state=torch.tensor([[False, True, True, False]]),
        phase_age=torch.tensor([[15, 0, 0, 15]]),
        touchdown_ready=torch.tensor([[True, False, False, False]]),
        horizon_steps=30,
        half_cycle_steps=15,
    )

    assert contact.shape == (1, 31, 4)
    assert torch.equal(contact[0, :3, 0], torch.tensor([False, True, True]))
    assert torch.equal(contact[0, :3, 3], torch.tensor([False, False, True]))
    assert torch.all(contact[0, :3, 1:3])


def test_adaptive_contact_enters_recovery_without_forcing_stance() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import advance_contact_scheduler

    result = advance_contact_scheduler(
        contact_state=torch.tensor([[False, True, True, False]]),
        phase_age=torch.tensor([[24, 15, 15, 24]]),
        swing_extension_age=torch.tensor([[9, 0, 0, 9]]),
        stance_age=torch.tensor([[0, 15, 15, 0]]),
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
        touchdown_scheduled=torch.tensor([[True, False, False, True]]),
        touchdown_ready=torch.zeros(1, 4, dtype=torch.bool),
        liftoff_scheduled=torch.zeros(1, 4, dtype=torch.bool),
        reliable_stance=torch.tensor([[False, True, True, False]]),
        max_swing_extension_steps=10,
    )

    assert not torch.any(result.contact_state[:, (0, 3)])
    assert torch.equal(result.swing_extension_age[:, (0, 3)], torch.full((1, 2), 10))
    assert torch.all(result.recovery_state[:, (0, 3)])
    torch.testing.assert_close(result.progress_scale, torch.ones(1))


def test_liftoff_guard_stops_root_when_all_contact_deadlock_has_no_extension_clock() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import advance_contact_scheduler

    result = advance_contact_scheduler(
        contact_state=torch.ones(1, 4, dtype=torch.bool),
        phase_age=torch.full((1, 4), 15, dtype=torch.long),
        swing_extension_age=torch.zeros(1, 4, dtype=torch.long),
        stance_age=torch.full((1, 4), 15, dtype=torch.long),
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
        touchdown_scheduled=torch.zeros(1, 4, dtype=torch.bool),
        touchdown_ready=torch.zeros(1, 4, dtype=torch.bool),
        liftoff_scheduled=torch.tensor([[True, False, False, True]]),
        reliable_stance=torch.tensor([[False, True, False, False]]),
        max_swing_extension_steps=10,
    )

    assert torch.any(result.liftoff_blocked)
    torch.testing.assert_close(result.progress_scale, torch.zeros(1))


def test_liftoff_guard_releases_only_one_leg_from_pair_to_keep_two_supports() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import advance_contact_scheduler

    result = advance_contact_scheduler(
        contact_state=torch.tensor([[True, True, True, False]]),
        phase_age=torch.tensor([[1, 15, 15, 20]]),
        swing_extension_age=torch.tensor([[0, 0, 0, 5]]),
        stance_age=torch.tensor([[1, 15, 15, 0]]),
        recovery_state=torch.tensor([[False, False, False, True]]),
        touchdown_scheduled=torch.zeros(1, 4, dtype=torch.bool),
        touchdown_ready=torch.zeros(1, 4, dtype=torch.bool),
        liftoff_scheduled=torch.tensor([[False, True, True, False]]),
        reliable_stance=torch.tensor([[True, True, True, False]]),
        max_swing_extension_steps=10,
    )

    assert torch.equal(result.contact_state, torch.tensor([[True, False, True, False]]))
    assert torch.equal(result.liftoff_blocked, torch.tensor([[False, False, True, False]]))
    torch.testing.assert_close(result.progress_scale, torch.ones(1))


def test_liftoff_guard_releases_only_unsafe_leg_when_two_safe_supports_remain() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import advance_contact_scheduler

    result = advance_contact_scheduler(
        contact_state=torch.tensor([[True, False, True, True]]),
        phase_age=torch.full((1, 4), 15, dtype=torch.long),
        swing_extension_age=torch.zeros(1, 4, dtype=torch.long),
        stance_age=torch.full((1, 4), 15, dtype=torch.long),
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
        touchdown_scheduled=torch.zeros(1, 4, dtype=torch.bool),
        touchdown_ready=torch.zeros(1, 4, dtype=torch.bool),
        liftoff_scheduled=torch.tensor([[True, False, False, True]]),
        reliable_stance=torch.tensor([[False, False, True, True]]),
        max_swing_extension_steps=10,
    )

    assert torch.equal(result.contact_state, torch.tensor([[False, False, True, True]]))
    assert torch.equal(result.liftoff_blocked, torch.tensor([[False, False, False, True]]))


def test_stage_a_command_matrix_contains_all_275_signed_combinations() -> None:
    from .scenario_matrix import stage_a_commands

    commands = stage_a_commands()
    expected = tuple(
        product(
            (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8, -1.0, 1.0),
            (0.0, -0.3, 0.3, -0.5, 0.5),
            (0.0, -0.5, 0.5, -1.0, 1.0),
        )
    )

    assert commands == expected
    assert len(commands) == 275
    assert len(set(commands)) == 275


def test_planner_returns_fixed_shape_per_leg_scheduler_state() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step

    result = step(make_state(2), make_command(2, vx=0.2), make_flat_field(2), None, JointMpcRtiCfg())
    state = result.solver_state

    assert state.contact_state.shape == (2, 4)
    assert state.phase_age.shape == (2, 4)
    assert state.swing_extension_age.shape == (2, 4)
    assert state.stance_age.shape == (2, 4)
    assert state.recovery_state.shape == (2, 4)


def test_touchdown_readiness_requires_ground_and_all_body_safety() -> None:
    from extension.joint_mpc_rti.planner import _touchdown_ready_mask

    ready = _touchdown_ready_mask(
        foot_surface_error_m=torch.tensor([[0.005, 0.005, 0.005, 0.005]]),
        foot_vertical_step_m=torch.tensor([[-0.001, -0.001, -0.001, -0.001]]),
        foot_small_distance_m=torch.tensor([[0.06, 0.06, 0.06, 0.06]]),
        leg_collision=torch.tensor([[False, True, False, False]]),
        base_collision=torch.tensor([False]),
        joint_safe=torch.tensor([[True, True, False, True]]),
        lookahead_collision=torch.tensor([[False, False, False, True]]),
        map_valid=torch.ones(1, 4, dtype=torch.bool),
        surface_gap_limit_m=0.012,
        surface_penetration_limit_m=0.001,
        touchdown_margin_m=0.052,
    )

    assert torch.equal(ready, torch.tensor([[True, False, False, False]]))


def test_touchdown_readiness_rejects_airborne_upward_or_base_collision() -> None:
    from extension.joint_mpc_rti.planner import _touchdown_ready_mask

    ready = _touchdown_ready_mask(
        foot_surface_error_m=torch.tensor([[0.011, 0.005, 0.005, 0.005]]),
        foot_vertical_step_m=torch.tensor([[-0.001, 0.001, -0.001, -0.001]]),
        foot_small_distance_m=torch.full((1, 4), 0.06),
        leg_collision=torch.zeros(1, 4, dtype=torch.bool),
        base_collision=torch.tensor([True]),
        joint_safe=torch.ones(1, 4, dtype=torch.bool),
        lookahead_collision=torch.zeros(1, 4, dtype=torch.bool),
        map_valid=torch.ones(1, 4, dtype=torch.bool),
        surface_gap_limit_m=0.012,
        surface_penetration_limit_m=0.001,
        touchdown_margin_m=0.052,
    )

    assert not torch.any(ready)


def test_touchdown_readiness_uses_signed_stage_a_ground_limits() -> None:
    from extension.joint_mpc_rti.planner import _touchdown_ready_mask

    ready = _touchdown_ready_mask(
        foot_surface_error_m=torch.tensor([[0.012, 0.0121, -0.001, -0.0011]]),
        foot_vertical_step_m=torch.full((1, 4), -0.001),
        foot_small_distance_m=torch.full((1, 4), 0.06),
        leg_collision=torch.zeros(1, 4, dtype=torch.bool),
        base_collision=torch.tensor([False]),
        joint_safe=torch.ones(1, 4, dtype=torch.bool),
        lookahead_collision=torch.zeros(1, 4, dtype=torch.bool),
        map_valid=torch.ones(1, 4, dtype=torch.bool),
        surface_gap_limit_m=0.012,
        surface_penetration_limit_m=0.001,
        touchdown_margin_m=0.052,
    )

    assert torch.equal(ready, torch.tensor([[True, False, True, False]]))


def test_planner_holds_unready_touchdown_and_next_liftoff() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    measured = replace(
        measured,
        root_lin_vel_b=torch.tensor([[0.0, 0.0, 0.1]]),
    )
    initial = step(measured, make_command(1, vx=0.2), make_flat_field(1), None, cfg)
    solver_state = replace(
        initial.solver_state,
        gait_phase=torch.tensor([29]),
        contact_state=torch.tensor([[False, True, True, False]]),
        phase_age=torch.full((1, 4), 15, dtype=torch.long),
        swing_extension_age=torch.zeros(1, 4, dtype=torch.long),
        stance_age=torch.tensor([[0, 15, 15, 0]], dtype=torch.long),
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
    )

    result = step(measured, make_command(1, vx=0.2), make_flat_field(1), solver_state, cfg)

    assert torch.equal(result.full_trajectory.contact_state[:, 0], torch.tensor([[False, True, True, False]]))
    assert torch.equal(result.pending_reference.contact_state, torch.tensor([[False, True, True, False]]))
    assert torch.equal(result.solver_state.swing_extension_age, torch.tensor([[1, 0, 0, 1]]))


def test_root_assist_limits_bound_rate_and_nominal_relative_state() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.2, yaw=0.5)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[..., 0] = 0.2
    control[..., 1] = 1.0
    control[..., 3:5] = 2.0
    control[..., 5] = 2.0

    bounded = planner._enforce_root_assist_limits(measured, control, command, cfg)
    rollout = rollout_controls(measured, bounded, dt=cfg.runtime.dt)
    nominal_control = torch.zeros_like(control)
    nominal_control[..., 0] = 0.2
    nominal_control[..., 5] = 0.5
    nominal = rollout_controls(measured, nominal_control, dt=cfg.runtime.dt)

    assert float((bounded[..., 1] - command[:, None, 1]).abs().max()) <= 0.200001
    assert float(bounded[..., 3:5].abs().max()) <= 0.600001
    assert float((bounded[..., 5] - command[:, None, 2]).abs().max()) <= 0.800001
    assert float((rollout.state[..., 1] - nominal.state[..., 1]).abs().max()) <= 0.060001
    assert float(rollout.state[..., 3:5].abs().max()) <= cfg.solver.root_roll_pitch_limit_rad + 1.0e-6
    assert float((torch.diff(rollout.state[..., 3:5], dim=1) / cfg.runtime.dt).abs().max()) <= (
        cfg.solver.root_roll_pitch_rate_limit_rps + 1.0e-6
    )
    yaw_error = torch.atan2(
        torch.sin(rollout.state[..., 5] - nominal.state[..., 5]),
        torch.cos(rollout.state[..., 5] - nominal.state[..., 5]),
    )
    assert float(yaw_error.abs().max()) <= cfg.solver.root_yaw_error_limit_rad + 1.0e-6


def test_root_bounds_shape_base_without_mutating_line_search_candidates(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    observed_batches: list[int] = []
    original = planner._enforce_root_assist_limits

    def record_projection(measured_state, control, command_body, projection_cfg):
        observed_batches.append(int(control.shape[0]))
        return original(measured_state, control, command_body, projection_cfg)

    monkeypatch.setattr(planner, "_enforce_root_assist_limits", record_projection)
    planner.step(make_state(2), make_command(2, vx=0.2, yaw=0.5), make_flat_field(2), None, cfg)

    assert observed_batches == [2]


def test_confirmed_touchdown_anchor_uses_measured_xy_and_terrain_surface() -> None:
    from extension.joint_mpc_rti.planner import _confirmed_stance_anchor

    previous = torch.full((1, 4, 3), -1.0)
    measured = torch.tensor(
        [[[0.10, 0.20, 0.30], [0.40, 0.50, 0.60], [0.70, 0.80, 0.90], [1.0, 1.1, 1.2]]]
    )
    surface = torch.tensor([[0.01, 0.02, 0.03, 0.04]])
    touchdown = torch.tensor([[True, False, False, False]])

    anchor = _confirmed_stance_anchor(
        previous_anchor_w=previous,
        measured_foot_w=measured,
        terrain_height_w=surface,
        confirmed_touchdown=touchdown,
        foot_contact_offset=0.022,
    )

    torch.testing.assert_close(anchor[0, 0, :2], measured[0, 0, :2])
    torch.testing.assert_close(anchor[0, 0, 2], torch.tensor(0.032))
    torch.testing.assert_close(anchor[0, 1:], previous[0, 1:])


def test_extended_swing_phase_stays_at_landing_endpoint_instead_of_restarting() -> None:
    from extension.joint_mpc_rti.planner import _contact_segment_age

    contact = torch.zeros(1, 6, 4, dtype=torch.bool)
    age = _contact_segment_age(
        contact,
        torch.full((1, 4), 14, dtype=torch.long),
        half_cycle_steps=15,
    )

    assert torch.equal(age, torch.full((1, 6, 4), 14, dtype=torch.long))


def test_first_joint_control_tracks_measured_state_toward_landing_target() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import _control_from_joint_target

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    target = measured.joint_pos[:, None].expand(-1, cfg.runtime.horizon_steps + 1, -1).clone()
    target[:, :, 1] += 0.10

    control = _control_from_joint_target(measured, make_command(1), target, cfg)

    assert control[0, 0, 7] > 0.0


def test_recovery_foot_target_retracts_from_unreachable_swing_anchor() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    state = measured.as_vector()[:, None].expand(-1, 3, -1).clone()
    target = torch.zeros(1, 3, 4, 3)
    target[..., 0] = 0.60
    contact = torch.tensor([[[False, True, True, False]]] * 3).reshape(1, 3, 4)

    recovered = planner._command_conditioned_foot_targets(
        target,
        state,
        make_command(1),
        contact,
        torch.full((1, 4), 14, dtype=torch.long),
        cfg,
        progress_scale=torch.zeros(1),
    )

    assert torch.all(recovered[0, :, (0, 3), 0] < 0.40)


def test_extended_swing_touchdown_target_advances_with_command_progress() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    state = measured.as_vector()[:, None].expand(-1, 3, -1).clone()
    target = torch.zeros(1, 3, 4, 3)
    contact = torch.zeros(1, 3, 4, dtype=torch.bool)
    command = make_command(1, vx=0.2)

    nominal = planner._command_conditioned_foot_targets(
        target,
        state,
        command,
        contact,
        torch.full((1, 4), 14, dtype=torch.long),
        cfg,
    )
    extended = planner._command_conditioned_foot_targets(
        target,
        state,
        command,
        contact,
        torch.full((1, 4), 24, dtype=torch.long),
        cfg,
    )

    expected_extra = 10 * 0.2 * cfg.runtime.dt
    torch.testing.assert_close(
        extended[:, :, :, 0] - nominal[:, :, :, 0],
        torch.full((1, 3, 4), expected_extra),
    )


def test_recovery_progress_continuously_releases_joint_warm_start() -> None:
    from extension.joint_mpc_rti.planner import _blend_recovery_joint_control

    warm = torch.ones(2, 3, 18)
    desired = torch.zeros_like(warm)
    blended = _blend_recovery_joint_control(
        warm,
        desired,
        torch.tensor([0.0, 0.5]),
    )

    torch.testing.assert_close(blended[0, :, 6:], torch.zeros_like(blended[0, :, 6:]))
    torch.testing.assert_close(blended[1, :, 6:], torch.full_like(blended[1, :, 6:], 0.5))
    torch.testing.assert_close(blended[..., :6], warm[..., :6])


def test_recovery_leg_forces_desired_joint_control_even_with_full_root_progress() -> None:
    from extension.joint_mpc_rti.planner import _blend_recovery_joint_control

    warm = torch.ones(1, 2, 18)
    desired = torch.zeros_like(warm)
    blended = _blend_recovery_joint_control(
        warm,
        desired,
        torch.ones(1),
        recovery_state=torch.tensor([[False, False, True, False]]),
    )

    shaped = blended[..., 6:].reshape(1, 2, 4, 3)
    torch.testing.assert_close(shaped[:, :, 2], torch.zeros_like(shaped[:, :, 2]))
    torch.testing.assert_close(shaped[:, :, (0, 1, 3), :], torch.ones_like(shaped[:, :, (0, 1, 3), :]))
    torch.testing.assert_close(blended[..., :6], warm[..., :6])


def test_recovery_landing_projection_reduces_safe_foot_ground_error() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    measured.root_rpy_w[:, 0] = 0.10
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    before = rollout_controls(measured, control, dt=cfg.runtime.dt).foot_pos_w[:, 1]
    projected = planner._enforce_recovery_landing(
        measured,
        control,
        make_flat_field(1),
        torch.ones(1, 4, dtype=torch.bool),
        cfg,
    )
    after_state = rollout_controls(measured, projected, dt=cfg.runtime.dt).state[:, 1]
    after = go2_foot_pos(after_state[:, :3], after_state[:, 3:6], after_state[:, 6:])

    before_error = (before[..., 2] - 0.022).abs().mean()
    after_error = (after[..., 2] - 0.022).abs().mean()
    assert after_error < before_error
