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
