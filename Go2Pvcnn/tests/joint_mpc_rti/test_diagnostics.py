from __future__ import annotations

import torch
import inspect
import pytest

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from .helpers import make_command, make_flat_field, make_state


def test_refresh_diagnostics_have_final_fixed_shapes_and_finite_values() -> None:
    from extension.joint_mpc_rti.planner import step

    batch = 2
    result = step(
        make_state(batch), make_command(batch), make_flat_field(batch), None, JointMpcRtiCfg()
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.selector_candidate_valid_count.shape == (batch, 4)
    assert diagnostics.selector_candidate_reject_reason_count.shape[:2] == (batch, 4)
    assert diagnostics.selector_selected_index.shape == (batch, 4)
    assert diagnostics.selector_selected_rank.shape == (batch, 4)
    assert diagnostics.selector_score_components.shape[:2] == (batch, 4)
    assert diagnostics.selector_score_components.ndim == 3
    assert diagnostics.region_area.shape == (batch, 4)
    assert diagnostics.touchdown_target_change_reason_bits.shape == (batch, 4, 4)
    assert diagnostics.nominal_min_clearance.shape == (batch, 5)
    assert diagnostics.kkt_primal_residual.shape == (batch,)
    assert diagnostics.kkt_dual_residual.shape == (batch,)
    assert diagnostics.slack_max.shape == (batch, 2)
    assert diagnostics.active_constraint_count.shape == (batch, 6)
    assert diagnostics.alpha_feasible.shape == (batch, 5)
    assert diagnostics.alpha_cost.shape == (batch, 5)
    assert diagnostics.alpha_reject_bits.shape == (batch, 5, 11)
    assert diagnostics.alpha_min_clearance.shape == (batch, 5, 5)
    assert diagnostics.selected_alpha.shape == (batch,)

    finite_fields = (
        diagnostics.region_area,
        diagnostics.warm_shift_rebase_error,
        diagnostics.retarget_trajectory_change,
        diagnostics.kkt_primal_residual,
        diagnostics.kkt_dual_residual,
        diagnostics.delta_z_norm,
        diagnostics.slack_max,
        diagnostics.alpha_cost,
        diagnostics.alpha_min_clearance,
    )
    assert all(torch.isfinite(value).all() for value in finite_fields)


def test_diagnostics_reuse_the_single_planner_refresh(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner

    calls = 0
    original = planner.perceptive_sqp_rti_update

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, "perceptive_sqp_rti_update", spy)
    planner.step(make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg())
    assert calls == 1


def test_profiler_declares_exact_eight_non_synchronizing_stages() -> None:
    from extension.joint_mpc_rti.diagnostics import profiler

    assert profiler.STAGE_NAMES == (
        "field",
        "selector",
        "region",
        "nominal_ik",
        "linearization",
        "scan_qp",
        "line_search_safety",
        "cache_diagnostics",
    )
    source = inspect.getsource(profiler.RefreshStageProfiler.record)
    assert "synchronize" not in source
    assert ".item(" not in source


def test_profiler_rejects_unknown_or_out_of_order_stage() -> None:
    from extension.joint_mpc_rti.diagnostics.profiler import RefreshStageProfiler

    profiler = RefreshStageProfiler(device="cpu")
    profiler.record("field")
    try:
        profiler.record("region")
    except ValueError as error:
        assert "selector" in str(error)
    else:
        raise AssertionError("out-of-order stage was accepted")


def test_one_refresh_records_all_eight_stage_durations() -> None:
    from extension.joint_mpc_rti.diagnostics.profiler import (
        RefreshStageProfiler,
        STAGE_NAMES,
    )
    from extension.joint_mpc_rti.planner import step

    profiler = RefreshStageProfiler(device="cpu")
    step(
        make_state(1),
        make_command(1),
        make_flat_field(1),
        None,
        JointMpcRtiCfg(),
        stage_profiler=profiler,
    )
    elapsed = profiler.elapsed_ms()
    assert tuple(elapsed) == STAGE_NAMES
    assert all(torch.isfinite(torch.tensor(value)) and value >= 0.0 for value in elapsed.values())
    assert elapsed["region"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_refresh_profiler_records_finite_stage_events() -> None:
    from extension.joint_mpc_rti.diagnostics.profiler import RefreshStageProfiler
    from extension.joint_mpc_rti.planner import step

    profiler = RefreshStageProfiler(device="cuda")
    step(
        make_state(1, device="cuda"),
        make_command(1, device="cuda"),
        make_flat_field(1, device="cuda"),
        None,
        JointMpcRtiCfg(),
        stage_profiler=profiler,
    )
    torch.cuda.synchronize()
    elapsed = profiler.elapsed_ms()
    assert all(torch.isfinite(torch.tensor(value)) and value >= 0.0 for value in elapsed.values())
    assert elapsed["region"] > 0.0
