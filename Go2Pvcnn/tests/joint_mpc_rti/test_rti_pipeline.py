from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from .helpers import make_command, make_flat_field, make_state


def test_planner_runs_each_final_stage_exactly_once(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.solver import sqp_rti

    calls = {name: 0 for name in ("selector", "nominal", "lq", "qp", "search")}

    def wrap(module, name, key):
        original = getattr(module, name)

        def spy(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    wrap(planner, "select_touchdowns", "selector")
    wrap(planner, "build_nominal", "nominal")
    wrap(sqp_rti, "build_lq_problem", "lq")
    wrap(sqp_rti, "solve_trajectory_qp_scan", "qp")
    wrap(sqp_rti, "hard_safe_line_search", "search")

    result = planner.step(
        make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg()
    )

    assert calls == {"selector": 1, "nominal": 1, "lq": 1, "qp": 1, "search": 1}
    assert result.full_trajectory.state_nodes.shape == (1, 31, 18)
    assert result.full_trajectory.future_state.shape == (1, 30, 18)
    torch.testing.assert_close(
        result.pending_reference.joint_angles,
        result.full_trajectory.state_nodes[:, 1, 6:],
    )


def test_nominal_only_mode_skips_rti_and_returns_alpha_zero(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner

    cfg = JointMpcRtiCfg()
    cfg.runtime.nominal_only = True

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("nominal-only mode must skip LQ/QP/line search")

    monkeypatch.setattr(planner, "perceptive_sqp_rti_update", fail_if_called)
    result = planner.step(
        make_state(1), make_command(1), make_flat_field(1), None, cfg
    )

    torch.testing.assert_close(
        result.full_trajectory.state_nodes, result.diagnostics.nominal_state
    )
    torch.testing.assert_close(
        result.diagnostics.qp_direction,
        torch.zeros_like(result.diagnostics.qp_direction),
    )
    torch.testing.assert_close(
        result.diagnostics.selected_alpha, torch.zeros_like(result.diagnostics.selected_alpha)
    )
    assert result.full_trajectory.fallback.all()


def test_first_call_is_cold_and_every_later_call_is_warm() -> None:
    from extension.joint_mpc_rti import planner

    state = make_state(1)
    command = make_command(1)
    field = make_flat_field(1)
    first = planner.step(state, command, field, None, JointMpcRtiCfg())
    second = planner.step(state, command, field, first.solver_state, JointMpcRtiCfg())

    assert first.full_trajectory.cold_start.all()
    assert not first.full_trajectory.warm_start.any()
    assert second.full_trajectory.warm_start.all()
    assert not second.full_trajectory.cold_start.any()


def test_no_feasible_candidate_stops_and_preserves_last_finite_cache(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.solver import sqp_rti

    first = planner.step(
        make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg()
    )
    cached = first.solver_state.trajectory.clone()
    original = sqp_rti.hard_safe_line_search

    def reject(*args, **kwargs):
        result = original(*args, **kwargs)
        return type(result)(
            **{
                **vars(result),
                "alpha_feasible": torch.zeros_like(result.alpha_feasible),
                "publish": torch.zeros_like(result.publish),
                "stop": torch.ones_like(result.stop),
            }
        )

    monkeypatch.setattr(sqp_rti, "hard_safe_line_search", reject)
    second = planner.step(
        make_state(1),
        make_command(1),
        make_flat_field(1),
        first.solver_state,
        JointMpcRtiCfg(),
    )

    assert second.full_trajectory.stop.all()
    assert not second.full_trajectory.publish.any()
    torch.testing.assert_close(second.solver_state.trajectory, cached)
    assert second.solver_state.initialized.all()
    assert second.full_trajectory.warm_start.all()


def test_published_x1_and_measured_x0_are_exact() -> None:
    from extension.joint_mpc_rti import planner

    measured = make_state(2)
    result = planner.step(
        measured, make_command(2), make_flat_field(2), None, JointMpcRtiCfg()
    )

    torch.testing.assert_close(result.full_trajectory.state_nodes[:, 0], measured.as_vector())
    torch.testing.assert_close(
        result.pending_reference.root_pos_w,
        result.full_trajectory.state_nodes[:, 1, :3],
    )


def test_lq_context_does_not_move_current_nominal_stance_anchor() -> None:
    from extension.joint_mpc_rti import planner

    result = planner.step(
        make_state(1),
        make_command(1, vx=0.8),
        make_flat_field(1),
        None,
        JointMpcRtiCfg(),
    )

    assert result.diagnostics.nominal_stance_anchor_error.max() <= 1.0e-4


def test_current_field_is_forwarded_to_lq_and_line_search(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.solver import sqp_rti

    seen = []
    original = sqp_rti.perceptive_sqp_rti_update

    def spy(nominal, context, cfg, **kwargs):
        seen.append(context.perceptive_field.refresh_id.clone())
        return original(nominal, context, cfg, **kwargs)

    monkeypatch.setattr(sqp_rti, "perceptive_sqp_rti_update", spy)
    monkeypatch.setattr(planner, "perceptive_sqp_rti_update", spy)
    field = make_flat_field(1)
    planner.step(make_state(1), make_command(1), field, None, JointMpcRtiCfg())

    assert len(seen) == 1
    torch.testing.assert_close(seen[0], field.version)


def test_zero_command_first_eight_refreshes_remain_publishable() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.0)
    field = make_flat_field(1)
    solver_state = None
    published = []

    for _ in range(8):
        result = planner.step(measured, command, field, solver_state, cfg)
        trajectory = result.full_trajectory
        state = trajectory.state_nodes[:, 1]
        velocity = trajectory.derived_velocity[:, 0]
        published.append(trajectory.publish)
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        solver_state = result.solver_state

    assert torch.stack(published, dim=1).all()


def test_fast_forward_one_gait_remains_publishable() -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    cfg.nominal.command_scale = 0.85
    cfg.nominal.step_reference_scale = 0.0
    measured = make_state(1)
    command = make_command(1, vx=1.0)
    field = make_flat_field(1)
    solver_state = None
    published = []

    for _ in range(24):
        origin = field.origin_w.clone()
        origin[:, :2] = measured.root_pos_w[:, :2]
        result = planner.step(
            measured,
            command,
            replace(field, origin_w=origin),
            solver_state,
            cfg,
        )
        trajectory = result.full_trajectory
        state = trajectory.state_nodes[:, 1]
        velocity = trajectory.derived_velocity[:, 0]
        published.append(trajectory.publish)
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        solver_state = result.solver_state

    assert torch.stack(published, dim=1).all()


def test_startup_root_hold_does_not_repeat_after_one_gait() -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.2)
    field = make_flat_field(1)
    solver_state = None
    phase_24_step = None

    for refresh in range(25):
        origin = field.origin_w.clone()
        origin[:, :2] = measured.root_pos_w[:, :2]
        result = planner.step(
            measured,
            command,
            replace(field, origin_w=origin),
            solver_state,
            cfg,
        )
        trajectory = result.full_trajectory
        if refresh == 24:
            phase_24_step = trajectory.state_nodes[:, 1, :2] - measured.root_pos_w[:, :2]
        state = trajectory.state_nodes[:, 1]
        velocity = trajectory.derived_velocity[:, 0]
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        solver_state = result.solver_state

    assert solver_state.gait_phase.item() == 25
    assert phase_24_step is not None and phase_24_step[0, 0] > 0.0


def test_cold_lq_preserves_the_three_edge_root_translation_lead_window() -> None:
    from extension.joint_mpc_rti import planner

    result = planner.step(
        make_state(1),
        make_command(1, vx=0.2),
        make_flat_field(1),
        None,
        JointMpcRtiCfg(),
    )

    torch.testing.assert_close(
        result.diagnostics.qp_direction[:, 1:4, :2],
        torch.zeros_like(result.diagnostics.qp_direction[:, 1:4, :2]),
        atol=1.0e-7,
        rtol=0.0,
    )


def test_published_root_does_not_leak_before_swing_foot_progress() -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.2)
    base_field = make_flat_field(1)
    initial_root = measured.root_pos_w[:, :2].clone()
    initial_foot = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w[..., :2]
    initial_relative = initial_foot - initial_root[:, None]
    solver_state = None
    foot_progressed = False

    for _ in range(12):
        origin = base_field.origin_w.clone()
        origin[:, :2] = measured.root_pos_w[:, :2]
        result = planner.step(
            measured,
            command,
            replace(base_field, origin_w=origin),
            solver_state,
            cfg,
        )
        trajectory = result.full_trajectory
        state = trajectory.state_nodes[:, 1]
        foot = trajectory.foot_pos_w[:, 1, :, :2]
        relative_progress = (
            foot[..., 0] - state[:, None, 0] - initial_relative[..., 0]
        )
        swing = ~trajectory.contact_state[:, 1]
        foot_progressed = bool((swing & (relative_progress >= 0.001)).any())
        if foot_progressed:
            break
        assert torch.linalg.vector_norm(state[:, :2] - initial_root, dim=-1).max() <= 0.0005

        velocity = trajectory.derived_velocity[:, 0]
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        solver_state = result.solver_state

    assert foot_progressed


def test_zero_xy_commands_keep_published_root_xy_exact() -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    measured = make_state(2)
    command = torch.tensor(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    base_field = make_flat_field(2)
    initial_xy = measured.root_pos_w[:, :2].clone()
    solver_state = None

    for _ in range(8):
        origin = base_field.origin_w.clone()
        origin[:, :2] = measured.root_pos_w[:, :2]
        result = planner.step(
            measured,
            command,
            replace(base_field, origin_w=origin),
            solver_state,
            cfg,
        )
        trajectory = result.full_trajectory
        state = trajectory.state_nodes[:, 1]
        torch.testing.assert_close(state[:, :2], initial_xy, atol=1.0e-7, rtol=0.0)
        velocity = trajectory.derived_velocity[:, 0]
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        solver_state = result.solver_state


def test_zero_command_one_gait_respects_published_joint_step_limit() -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.0)
    base_field = make_flat_field(1)
    solver_state = None
    previous_joint = measured.joint_pos
    maximum_step = measured.joint_pos.new_zeros(())

    for _ in range(24):
        origin = base_field.origin_w.clone()
        origin[:, :2] = measured.root_pos_w[:, :2]
        result = planner.step(
            measured,
            command,
            replace(base_field, origin_w=origin),
            solver_state,
            cfg,
        )
        trajectory = result.full_trajectory
        state = trajectory.state_nodes[:, 1]
        velocity = trajectory.derived_velocity[:, 0]
        maximum_step = torch.maximum(
            maximum_step, (state[:, 6:] - previous_joint).abs().amax()
        )
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        previous_joint = state[:, 6:]
        solver_state = result.solver_state

    assert maximum_step <= 0.35
