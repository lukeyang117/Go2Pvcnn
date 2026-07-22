from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.solver.trajectory_qp import ActiveConstraints, ActiveSetSolution

from .helpers import make_command, make_flat_field, make_state


def test_planner_runs_one_nominal_linearize_scan_search_and_publishes_x1(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.solver import sqp_rti

    calls: list[str] = []
    original_nominal = planner.build_nominal

    def nominal_spy(*args, **kwargs):
        calls.append("nominal")
        return original_nominal(*args, **kwargs)

    def linearize_spy(state, context, cfg):
        calls.append("linearize")
        return SimpleNamespace(
            gradient=torch.zeros_like(state),
            support_target=state.new_zeros(state.shape[0], 6),
        )

    def scan_spy(qp):
        calls.append("scan")
        direction = torch.zeros_like(qp.gradient)
        active = ActiveConstraints(
            box_low=torch.zeros_like(direction, dtype=torch.bool),
            box_high=torch.zeros_like(direction, dtype=torch.bool),
            velocity_low=torch.zeros(direction.shape[0], 30, 12, dtype=torch.bool),
            velocity_high=torch.zeros(direction.shape[0], 30, 12, dtype=torch.bool),
        )
        return ActiveSetSolution(direction=direction, active=active)

    def search_spy(nominal, direction, objective, **kwargs):
        calls.append("line_search")
        loss = objective(nominal)
        return SimpleNamespace(
            state=nominal,
            alpha=nominal.new_zeros(nominal.shape[0]),
            candidate_loss=loss[:, None].expand(-1, 5),
            filter_valid=torch.ones(
                nominal.shape[0], 5, 4, dtype=torch.bool, device=nominal.device
            ),
            selected_loss=loss,
            selected_index=torch.full((nominal.shape[0],), 4, dtype=torch.long),
            used_nominal=torch.ones(nominal.shape[0], dtype=torch.bool),
            selected_feasible=torch.ones(nominal.shape[0], dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "build_nominal", nominal_spy)
    monkeypatch.setattr(sqp_rti, "linearize_trajectory", linearize_spy)
    monkeypatch.setattr(sqp_rti, "solve_trajectory_qp_scan", scan_spy)
    monkeypatch.setattr(sqp_rti, "parallel_line_search", search_spy)
    result = planner.step(make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg())

    assert calls == ["nominal", "linearize", "scan", "line_search"]


def test_real_rti_preserves_nominal_published_root_xy() -> None:
    from extension.joint_mpc_rti import planner

    result = planner.step(
        make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg()
    )

    assert result.diagnostics is not None
    torch.testing.assert_close(
        result.diagnostics.qp_direction[:, 1, :2],
        torch.zeros_like(result.diagnostics.qp_direction[:, 1, :2]),
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.full_trajectory.state[:, 1, :2],
        result.diagnostics.nominal_state[:, 1, :2],
        atol=1.0e-7,
        rtol=0.0,
    )


def test_planner_exposes_same_iteration_edge_diagnostics(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner

    calls = 0

    def update_stub(nominal, context, cfg):
        nonlocal calls
        calls += 1
        direction = torch.zeros_like(nominal)
        direction[:, 1, 0] = 0.003
        return SimpleNamespace(
            state=nominal + direction,
            direction=direction,
            alpha=nominal.new_ones(nominal.shape[0]),
            selected_loss=nominal.new_zeros(nominal.shape[0]),
            selected_index=torch.zeros(nominal.shape[0], dtype=torch.long),
            used_nominal=torch.zeros(nominal.shape[0], dtype=torch.bool),
            status=torch.zeros(nominal.shape[0], dtype=torch.long),
            candidate_loss=nominal.new_zeros(nominal.shape[0], 5),
            candidate_filter_valid=torch.ones(
                nominal.shape[0], 5, 4, dtype=torch.bool, device=nominal.device
            ),
            candidate_swing_safe_z=nominal.new_zeros(nominal.shape[0], 5, 4),
            loss_breakdown={},
            node_loss_breakdown={
                name: nominal.new_zeros(nominal.shape[0], nominal.shape[1])
                for name in ("step", "terrain", "smooth")
            },
        )

    monkeypatch.setattr(planner, "sqp_rti_update", update_stub)
    measured = make_state(1)

    result = planner.step(measured, make_command(1), make_flat_field(1), None, JointMpcRtiCfg())

    assert calls == 1
    assert result.diagnostics is not None
    assert result.diagnostics.nominal_state.shape == (1, 2, 18)
    assert result.diagnostics.qp_direction.shape == (1, 2, 18)
    assert result.diagnostics.stance_anchor_w.shape == (1, 4, 3)
    assert result.diagnostics.touchdown_reference_w.shape == (1, 2, 4, 3)
    assert result.diagnostics.candidate_loss.shape == (1, 5)
    assert result.diagnostics.candidate_filter_valid.shape == (1, 5, 4)
    assert result.diagnostics.candidate_swing_safe_z.shape == (1, 5, 4)
    assert result.diagnostics.support_target.shape == (1, 6)
    assert tuple(result.diagnostics.node_loss_breakdown) == ("step", "terrain", "smooth")
    assert all(
        value.shape == (1, 31)
        for value in result.diagnostics.node_loss_breakdown.values()
    )
    torch.testing.assert_close(result.diagnostics.nominal_state[:, 0], measured.as_vector())
    torch.testing.assert_close(result.diagnostics.qp_direction[:, 1, 0], torch.tensor([0.003]))
    assert result.full_trajectory.state.shape == (1, 31, 18)
    assert result.pending_reference.target_step == 1
    torch.testing.assert_close(result.pending_reference.root_pos_w, result.full_trajectory.state[:, 1, :3])


def test_planner_rejects_finite_nominal_fallback_when_all_candidates_are_filtered(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.solver import sqp_rti

    def linearize_stub(state, context, cfg):
        return SimpleNamespace(
            gradient=torch.zeros_like(state),
            support_target=state.new_zeros(state.shape[0], 6),
        )

    def scan_stub(qp):
        direction = torch.zeros_like(qp.gradient)
        active = ActiveConstraints(
            box_low=torch.zeros_like(direction, dtype=torch.bool),
            box_high=torch.zeros_like(direction, dtype=torch.bool),
            velocity_low=torch.zeros(direction.shape[0], 30, 12, dtype=torch.bool),
            velocity_high=torch.zeros(direction.shape[0], 30, 12, dtype=torch.bool),
        )
        return ActiveSetSolution(direction=direction, active=active)

    def all_filtered(nominal, direction, objective, **kwargs):
        loss = objective(nominal)
        return SimpleNamespace(
            state=nominal,
            alpha=nominal.new_zeros(nominal.shape[0]),
            candidate_loss=loss[:, None].expand(-1, 5),
            filter_valid=torch.zeros(
                nominal.shape[0], 5, 4, dtype=torch.bool, device=nominal.device
            ),
            selected_loss=loss,
            selected_index=torch.full((nominal.shape[0],), 4, dtype=torch.long),
            used_nominal=torch.ones(nominal.shape[0], dtype=torch.bool),
            selected_feasible=torch.zeros(nominal.shape[0], dtype=torch.bool),
        )

    monkeypatch.setattr(sqp_rti, "linearize_trajectory", linearize_stub)
    monkeypatch.setattr(sqp_rti, "solve_trajectory_qp_scan", scan_stub)
    monkeypatch.setattr(sqp_rti, "parallel_line_search", all_filtered)

    result = planner.step(make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg())

    assert result.full_trajectory.status.item() != 0
    assert not result.full_trajectory.valid.item()
    assert result.solver_state.initialized.item()

    nominal_sources: list[tuple[bool, bool]] = []
    original_nominal = planner.build_nominal

    def nominal_spy(*args, **kwargs):
        nominal = original_nominal(*args, **kwargs)
        nominal_sources.append(
            (bool(nominal.used_cold_start.item()), bool(nominal.used_warm_start.item()))
        )
        return nominal

    monkeypatch.setattr(planner, "build_nominal", nominal_spy)
    planner.step(
        make_state(1),
        make_command(1),
        make_flat_field(1),
        result.solver_state,
        JointMpcRtiCfg(),
    )

    assert nominal_sources == [(False, True)]


def test_planner_supports_b40_and_keeps_exact_measured_z0(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner

    def unchanged(nominal, context, cfg):
        return SimpleNamespace(
            state=nominal,
            direction=torch.zeros_like(nominal),
            alpha=nominal.new_zeros(nominal.shape[0]),
            selected_loss=nominal.new_zeros(nominal.shape[0]),
            selected_index=torch.full((nominal.shape[0],), 4, dtype=torch.long),
            used_nominal=torch.ones(nominal.shape[0], dtype=torch.bool),
            status=torch.zeros(nominal.shape[0], dtype=torch.long),
            loss_breakdown={},
            node_loss_breakdown={
                name: nominal.new_zeros(nominal.shape[0], nominal.shape[1])
                for name in ("step", "terrain", "smooth")
            },
        )

    monkeypatch.setattr(planner, "sqp_rti_update", unchanged)
    measured = make_state(40)
    measured.root_pos_w[:, 0] = torch.linspace(-0.2, 0.2, 40)
    result = planner.step(measured, make_command(40), make_flat_field(40), None, JointMpcRtiCfg())

    assert result.full_trajectory.state.shape == (40, 31, 18)
    torch.testing.assert_close(result.full_trajectory.state[:, 0], measured.as_vector())


def test_planner_validity_comes_from_accepted_finite_state_not_nominal_reachability(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner

    original_nominal = planner.build_nominal

    def unreachable_nominal(*args, **kwargs):
        nominal = original_nominal(*args, **kwargs)
        return replace(nominal, valid=torch.zeros_like(nominal.valid))

    def accepted_state(nominal, context, cfg):
        return SimpleNamespace(
            state=nominal,
            direction=torch.zeros_like(nominal),
            alpha=nominal.new_zeros(nominal.shape[0]),
            selected_loss=nominal.new_zeros(nominal.shape[0]),
            selected_index=torch.full((nominal.shape[0],), 4, dtype=torch.long),
            used_nominal=torch.zeros(nominal.shape[0], dtype=torch.bool),
            status=torch.zeros(nominal.shape[0], dtype=torch.long),
            loss_breakdown={},
            node_loss_breakdown={
                name: nominal.new_zeros(nominal.shape[0], nominal.shape[1])
                for name in ("step", "terrain", "smooth")
            },
        )

    monkeypatch.setattr(planner, "build_nominal", unreachable_nominal)
    monkeypatch.setattr(planner, "sqp_rti_update", accepted_state)
    result = planner.step(make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg())

    assert result.full_trajectory.valid.all()


def test_invalid_warm_manifold_row_remains_initialized_without_cold_restart(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.types import JointMpcRtiSolverState

    def unchanged(nominal, context, cfg):
        return SimpleNamespace(
            state=nominal,
            direction=torch.zeros_like(nominal),
            alpha=nominal.new_zeros(nominal.shape[0]),
            selected_loss=nominal.new_zeros(nominal.shape[0]),
            selected_index=torch.full((nominal.shape[0],), 4, dtype=torch.long),
            used_nominal=torch.ones(nominal.shape[0], dtype=torch.bool),
            status=torch.zeros(nominal.shape[0], dtype=torch.long),
            loss_breakdown={},
            node_loss_breakdown={
                name: nominal.new_zeros(nominal.shape[0], nominal.shape[1])
                for name in ("step", "terrain", "smooth")
            },
        )

    monkeypatch.setattr(planner, "sqp_rti_update", unchanged)
    measured = make_state(1)
    previous_trajectory = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    previous_trajectory[:, 2:, 0] += 0.30
    previous = JointMpcRtiSolverState(
        trajectory=previous_trajectory,
        gait_phase=torch.tensor((12,), dtype=torch.long),
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=go2_fk(
            measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
        ).foot_pos_w,
    )

    result = planner.step(
        measured,
        torch.zeros(1, 3),
        make_flat_field(1),
        previous,
        JointMpcRtiCfg(),
    )

    assert not result.full_trajectory.valid.any()
    assert result.full_trajectory.warm_start.all()
    assert not result.full_trajectory.cold_start.any()
    assert result.solver_state.initialized.all()


def test_planner_source_has_no_old_repair_or_control_rollout_calls() -> None:
    from extension.joint_mpc_rti import planner

    source = inspect.getsource(planner)
    forbidden = (
        "recovery",
        "startup_root",
        "restore_candidate",
        "minimum_norm",
        "enforce_first_stance",
        "adaptive_contact",
        "constraint_violation",
        "rollout_controls",
        "base_control",
    )
    assert not any(name in source for name in forbidden)


def test_solver_state_contains_only_warm_lifecycle_state() -> None:
    from extension.joint_mpc_rti.types import JointMpcRtiSolverState

    assert tuple(JointMpcRtiSolverState.__dataclass_fields__) == (
        "trajectory",
        "gait_phase",
        "initialized",
        "stance_anchor_w",
    )


def test_touchdown_updates_only_new_stance_anchors(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.types import JointMpcRtiSolverState

    def unchanged(nominal, context, cfg):
        return SimpleNamespace(
            state=nominal,
            direction=torch.zeros_like(nominal),
            alpha=nominal.new_zeros(nominal.shape[0]),
            selected_loss=nominal.new_zeros(nominal.shape[0]),
            selected_index=torch.full((nominal.shape[0],), 4, dtype=torch.long),
            used_nominal=torch.ones(nominal.shape[0], dtype=torch.bool),
            status=torch.zeros(nominal.shape[0], dtype=torch.long),
            loss_breakdown={},
            node_loss_breakdown={
                name: nominal.new_zeros(nominal.shape[0], nominal.shape[1])
                for name in ("step", "terrain", "smooth")
            },
        )

    monkeypatch.setattr(planner, "sqp_rti_update", unchanged)
    measured = make_state(1)
    anchor = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w
    anchor = anchor + torch.tensor((0.003, -0.002, 0.001))
    previous = JointMpcRtiSolverState(
        trajectory=measured.as_vector()[:, None].expand(-1, 31, -1).clone(),
        gait_phase=torch.tensor([11]),
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=anchor,
    )

    result = planner.step(
        measured, make_command(1), make_flat_field(1), previous, JointMpcRtiCfg()
    )

    torch.testing.assert_close(result.solver_state.stance_anchor_w[:, 1:3], anchor[:, 1:3])
    torch.testing.assert_close(
        result.solver_state.stance_anchor_w[:, (0, 3)],
        result.full_trajectory.foot_pos_w[:, 1, (0, 3)],
    )


def test_nonfinite_touchdown_candidate_preserves_finite_warm_anchor(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.types import JointMpcRtiSolverState

    def nonfinite(nominal, context, cfg):
        state = nominal.clone()
        state[:, 1, 6] = torch.nan
        return SimpleNamespace(
            state=state,
            direction=torch.zeros_like(nominal),
            alpha=nominal.new_ones(nominal.shape[0]),
            selected_loss=nominal.new_zeros(nominal.shape[0]),
            selected_index=torch.zeros(nominal.shape[0], dtype=torch.long),
            used_nominal=torch.zeros(nominal.shape[0], dtype=torch.bool),
            status=torch.zeros(nominal.shape[0], dtype=torch.long),
            loss_breakdown={},
            node_loss_breakdown={
                name: nominal.new_zeros(nominal.shape[0], nominal.shape[1])
                for name in ("step", "terrain", "smooth")
            },
        )

    monkeypatch.setattr(planner, "sqp_rti_update", nonfinite)
    measured = make_state(1)
    anchor = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w
    previous = JointMpcRtiSolverState(
        trajectory=measured.as_vector()[:, None].expand(-1, 31, -1).clone(),
        gait_phase=torch.tensor([11]),
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=anchor,
    )

    result = planner.step(
        measured, make_command(1), make_flat_field(1), previous, JointMpcRtiCfg()
    )

    assert result.solver_state.initialized.all()
    assert torch.isfinite(result.solver_state.trajectory).all()
    torch.testing.assert_close(result.solver_state.stance_anchor_w, anchor)
