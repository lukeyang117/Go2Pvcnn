from __future__ import annotations

import inspect
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
        return SimpleNamespace(gradient=torch.zeros_like(state))

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
            selected_loss=loss,
            selected_index=torch.full((nominal.shape[0],), 4, dtype=torch.long),
            used_nominal=torch.ones(nominal.shape[0], dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "build_nominal", nominal_spy)
    monkeypatch.setattr(sqp_rti, "linearize_trajectory", linearize_spy)
    monkeypatch.setattr(sqp_rti, "solve_trajectory_qp_scan", scan_spy)
    monkeypatch.setattr(sqp_rti, "parallel_line_search", search_spy)
    result = planner.step(make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg())

    assert calls == ["nominal", "linearize", "scan", "line_search"]
    assert result.full_trajectory.state.shape == (1, 31, 18)
    assert result.pending_reference.target_step == 1
    torch.testing.assert_close(result.pending_reference.root_pos_w, result.full_trajectory.state[:, 1, :3])


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
        )

    monkeypatch.setattr(planner, "sqp_rti_update", unchanged)
    measured = make_state(40)
    measured.root_pos_w[:, 0] = torch.linspace(-0.2, 0.2, 40)
    result = planner.step(measured, make_command(40), make_flat_field(40), None, JointMpcRtiCfg())

    assert result.full_trajectory.state.shape == (40, 31, 18)
    torch.testing.assert_close(result.full_trajectory.state[:, 0], measured.as_vector())


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


def test_solver_state_contains_only_trajectory_phase_and_valid() -> None:
    from extension.joint_mpc_rti.types import JointMpcRtiSolverState

    assert tuple(JointMpcRtiSolverState.__dataclass_fields__) == ("trajectory", "gait_phase", "valid")
