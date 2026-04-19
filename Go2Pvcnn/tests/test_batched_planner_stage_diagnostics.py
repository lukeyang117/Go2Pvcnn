from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from tests.fixtures import viewer_runtime_diagnostics as viewer_diag


REQUIRED_STAGE_NAMES = {
    "input",
    "standstill",
    "gait",
    "footholds",
    "touchdown_eval",
    "swing_targets",
    "base_approx",
    "terrain_est",
    "base_solve",
    "ik",
    "fk",
    "mix",
    "result",
}


def _make_real_runtime_fixture(**kwargs):
    assert hasattr(viewer_diag, "make_real_runtime_fixture")
    return viewer_diag.make_real_runtime_fixture(**kwargs)


@pytest.fixture()
def real_runtime():
    runtime = _make_real_runtime_fixture(num_envs=2)
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture()
def real_runtime_32():
    runtime = _make_real_runtime_fixture(num_envs=32)
    try:
        yield runtime
    finally:
        runtime.close()


def test_planner_stage_outputs_respond_to_forward_command(real_runtime):
    standstill = real_runtime.plan_case_with_stage_diagnostics("standstill")
    forward = real_runtime.plan_case_with_stage_diagnostics("forward")

    input_summary = forward.stage_summaries["input"]
    base_summary = forward.stage_summaries["base_approx"]
    foothold_summary = forward.stage_summaries["footholds"]
    solve_summary = forward.stage_summaries["base_solve"]
    result_summary = forward.stage_summaries["result"]

    assert input_summary["command_vx_mean"] > 0.25
    assert base_summary["path_dx_mean"] > 0.05
    assert base_summary["path_dx_mean"] > abs(base_summary["path_dy_mean"]) + 0.03
    assert foothold_summary["touchdown_dx_mean"] > 0.02
    assert solve_summary["path_dx_mean"] > 0.05
    assert solve_summary["path_dx_mean"] > abs(solve_summary["path_dy_mean"]) + 0.03
    assert result_summary["standstill_ratio"] == 0.0
    assert result_summary["path_dx_mean"] > 0.05
    assert standstill.stage_summaries["result"]["standstill_ratio"] == 1.0


def test_planner_stage_outputs_respond_to_yaw_command(real_runtime):
    yaw_left = real_runtime.plan_case_with_stage_diagnostics("yaw_left")

    input_summary = yaw_left.stage_summaries["input"]
    foothold_summary = yaw_left.stage_summaries["footholds"]
    base_summary = yaw_left.stage_summaries["base_approx"]
    solve_summary = yaw_left.stage_summaries["base_solve"]
    result_summary = yaw_left.stage_summaries["result"]

    assert input_summary["command_yaw_mean"] > 0.25
    assert base_summary["yaw_delta_mean"] > 0.05
    assert solve_summary["yaw_delta_mean"] > 0.05
    assert result_summary["yaw_delta_mean"] > 0.05
    assert foothold_summary["left_touchdown_mean_y"] < -0.01
    assert foothold_summary["right_touchdown_mean_y"] > 0.01


def test_planner_standstill_stage_outputs_remain_symmetric(real_runtime):
    standstill = real_runtime.plan_case_with_stage_diagnostics("standstill")

    standstill_summary = standstill.stage_summaries["standstill"]
    result_summary = standstill.stage_summaries["result"]
    result_stage = standstill.stages["result"]

    assert standstill_summary["touchdown_delta_norm_max"] < 1e-5
    assert standstill_summary["touchdown_delta_norm_span"] < 1e-5
    assert abs(standstill_summary["left_touchdown_mean_y"]) < 1e-6
    assert abs(standstill_summary["right_touchdown_mean_y"]) < 1e-6
    assert result_summary["standstill_ratio"] == 1.0
    assert result_summary["contact_mean"] == pytest.approx(1.0)
    assert torch.allclose(
        result_stage.tensors["root_pos_w"],
        result_stage.tensors["root_pos_w"][:, :1].expand_as(result_stage.tensors["root_pos_w"]),
    )


def test_planner_output_vs_playback_divergence_report(real_runtime):
    report = real_runtime.planner_output_vs_playback_divergence("forward")

    assert report.frame_idx >= 0
    assert report.root_pos_max_abs < 1e-4
    assert report.joint_pos_max_abs < 1e-4
    assert report.root_pos_mean_abs < 1e-5
    assert report.joint_pos_mean_abs < 1e-5
    assert report.plan.stage_summaries["result"]["path_dx_mean"] > 0.05


def test_planner_output_vs_playback_emit_report(real_runtime):
    report = real_runtime.planner_output_vs_playback_divergence("forward")
    text = viewer_diag.format_playback_divergence_report(report)
    print(text)

    assert "[playback-diag]" in text
    assert "root_pos_max_abs=" in text
    assert "joint_pos_max_abs=" in text


def test_planner_stage_diagnostics_emit_summary(real_runtime):
    reports = [
        real_runtime.plan_case_with_stage_diagnostics("standstill"),
        real_runtime.plan_case_with_stage_diagnostics("forward"),
        real_runtime.plan_case_with_stage_diagnostics("yaw_left"),
    ]

    for report in reports:
        text = viewer_diag.format_stage_summary_report(
            report.plan.name,
            report.stage_summaries,
            stage_order=report.stage_order,
        )
        print(text)
        assert "[planner-diag]" in text
        assert "result:" in text


def test_planner_stage_diagnostics_batched_smoke_preserves_tensor_path(real_runtime_32):
    case_cycle = [
        "standstill",
        "forward",
        "yaw_left",
        "lateral_left",
        "backward",
        "yaw_right",
        "lateral_right",
        "forward",
    ]
    case_names = [case_cycle[idx % len(case_cycle)] for idx in range(real_runtime_32.num_envs)]

    diagnostics = real_runtime_32.plan_batched_cases_with_stage_diagnostics(case_names)

    assert REQUIRED_STAGE_NAMES.issubset(diagnostics.stages.keys())
    assert diagnostics.stage_summaries["mix"]["standstill_ratio"] > 0.0
    assert diagnostics.stage_summaries["result"]["batch_size"] == 32.0
    assert diagnostics.stage_summaries["result"]["path_dx_mean"] > 0.01
    assert diagnostics.stage_summaries["result"]["yaw_delta_abs_mean"] > 0.01
    assert diagnostics.stage_summaries["footholds"]["touchdown_delta_norm_max"] > 0.02

    for stage_name in REQUIRED_STAGE_NAMES:
        stage = diagnostics.stages[stage_name]
        primary = stage.primary_tensor
        assert isinstance(primary, torch.Tensor)
        assert primary.shape[0] == 32
