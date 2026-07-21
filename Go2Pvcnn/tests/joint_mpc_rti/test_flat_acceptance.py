from __future__ import annotations

import pytest


def _metric(name: str, passed: bool | None, *, applicable: bool = True):
    from .joint_metrics import MetricResult

    return MetricResult(
        name=name,
        value=0.0,
        numerator=0.0,
        denominator=1,
        valid_count=1,
        applicable=applicable,
        na_reason=None if applicable else "no small obstacle in flat scenario",
        threshold=0.0,
        passed=passed,
        worst_case_key=("flat",),
    )


def _report(*, failed: tuple[str, ...] = (), small_na: bool = True):
    from .run_joint_acceptance import AcceptanceCell, AcceptanceReport, CellReport

    metrics = {
        "joint_position_violation": _metric("joint_position_violation", "joint_position_violation" not in failed),
        "stance_ground_gap": _metric("stance_ground_gap", "stance_ground_gap" not in failed),
    }
    if small_na:
        metrics["strict_cross_success"] = _metric(
            "strict_cross_success", None, applicable=False
        )
    cell = AcceptanceCell(scenario="flat", command=(0.2, 0.0, 0.0))
    cell_report = CellReport(cell=cell, metrics=metrics, passed=not failed)
    return AcceptanceReport(stage="flat", code_ref="abc123", cells=(cell_report,))


def test_flat_gate_rejects_any_failed_applicable_metric() -> None:
    from .run_joint_acceptance import require_flat_gate

    result = require_flat_gate(_report(failed=("joint_position_violation", "stance_ground_gap")))

    assert not result.passed
    assert result.failures == ("flat/0.2/0/0/na/na/na/0:joint_position_violation", "flat/0.2/0/0/na/na/na/0:stance_ground_gap")


def test_flat_gate_does_not_require_small_only_metrics() -> None:
    from .run_joint_acceptance import require_flat_gate

    assert require_flat_gate(_report(small_na=True)).passed


def test_flat_simulator_returns_x0_to_x1_trace_without_control_variable() -> None:
    import torch

    from .run_joint_acceptance import simulate_flat_trace

    trace = simulate_flat_trace(torch.tensor([[0.2, 0.0, 0.0]]), steps=2)

    assert trace.root_pos_w.shape == (1, 3, 3)
    assert trace.foot_pos_w.shape == (1, 3, 4, 3)
    assert trace.contact_state.shape == (1, 3, 4)
    assert trace.x0_injection_error is not None
    assert float(trace.x0_injection_error.max()) <= 1.0e-6


def test_acceptance_initial_state_matches_configured_flat_contact_geometry() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    from .helpers import make_state

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    foot = go2_fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w

    assert float(state.root_pos_w[0, 2]) == pytest.approx(cfg.loss_terms.posture_root_clearance)
    assert float(foot[..., 2].min()) >= cfg.gait.foot_contact_offset
