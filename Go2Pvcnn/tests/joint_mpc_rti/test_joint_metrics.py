from __future__ import annotations

from dataclasses import replace

import pytest
import torch


def _root_carried_trace():
    from .joint_metrics import JointMetricTrace

    root_x = torch.tensor([[0.000, 0.004, 0.008, 0.012]], dtype=torch.float32)
    root_pos = torch.zeros(1, 4, 3)
    root_pos[..., 0] = root_x
    root_pos[..., 2] = 0.32
    foot = torch.zeros(1, 4, 4, 3)
    footprint = torch.tensor(
        [[0.20, 0.10, 0.022], [0.20, -0.10, 0.022], [-0.20, 0.10, 0.022], [-0.20, -0.10, 0.022]]
    )
    foot[:] = footprint
    foot[..., 0] += root_x.unsqueeze(-1)
    contact = torch.tensor([[[True, False, False, True]]] * 4).reshape(1, 4, 4)
    command = torch.tensor([[[0.2, 0.0, 0.0]]] * 4).reshape(1, 4, 3)
    zeros = torch.zeros(1, 4, 4)
    joint = torch.tensor((0.0, 0.8, -1.5) * 4).view(1, 1, 12).expand(1, 4, 12).clone()
    return JointMetricTrace(
        root_pos_w=root_pos,
        root_rpy_w=torch.zeros(1, 4, 3),
        joint_pos=joint,
        foot_pos_w=foot,
        contact_state=contact,
        command_body=command,
        gait_phase=torch.arange(4).view(1, 4),
        foot_height_w=zeros,
        foot_small_distance_m=torch.ones(1, 4, 4),
        part_collision={part: torch.zeros(1, 4, dtype=torch.bool) for part in ("foot", "knee", "calf", "thigh", "base")},
        line_alpha=torch.ones(1, 4),
        nominal_root_pos_w=root_pos.clone(),
        nominal_root_rpy_w=torch.zeros(1, 4, 3),
        valid=torch.ones(1, 4, dtype=torch.bool),
        map_valid=torch.ones(1, 4, dtype=torch.bool),
        timestamps=0.02 * torch.arange(4).view(1, 4),
        dt=0.02,
    )


def test_joint_metrics_detect_root_carried_stance_and_missing_active_swing() -> None:
    from .joint_metrics import accumulate_joint_metrics

    metrics = accumulate_joint_metrics(_root_carried_trace())

    assert metrics["stance_root_carry_ratio_abs"] > 0.90
    assert metrics["stance_stationary_ratio"] < 1.0
    assert metrics["swing_active_motion_ratio"] < 0.10
    assert metrics["foot_root_lead_time_min_ms"] <= 0.0


def test_swing_progress_rotates_body_command_into_world_frame() -> None:
    from .joint_metrics import accumulate_joint_metrics

    trace = _root_carried_trace()
    time = torch.arange(4, dtype=torch.float32)
    root = trace.root_pos_w.clone()
    root[..., 0] = 0.0
    root[..., 1] = 0.004 * time
    rpy = trace.root_rpy_w.clone()
    rpy[..., 2] = torch.pi / 2.0
    foot = trace.foot_pos_w[:, :1].expand(-1, 4, -1, -1).clone()
    foot[:, :, (1, 2), 1] += (0.008 * time).view(1, 4, 1)

    metrics = accumulate_joint_metrics(
        replace(
            trace,
            root_pos_w=root,
            root_rpy_w=rpy,
            foot_pos_w=foot,
            nominal_root_pos_w=root.clone(),
            nominal_root_rpy_w=rpy.clone(),
        )
    )

    assert metrics["swing_active_motion_ratio"] > 0.90


def test_stance_carry_ignores_submillimeter_stationary_noise() -> None:
    from .joint_metrics import accumulate_joint_metrics

    trace = _root_carried_trace()
    root = trace.root_pos_w[:, :1].expand_as(trace.root_pos_w).clone()
    foot = trace.foot_pos_w[:, :1].expand_as(trace.foot_pos_w).clone()
    foot[:, 1:, :, 0] += 5.0e-8

    metrics = accumulate_joint_metrics(
        replace(trace, root_pos_w=root, foot_pos_w=foot)
    )

    assert metrics["stance_root_carry_ratio_abs"] <= 0.10


def test_root_direction_excludes_startup_edges_below_v_min() -> None:
    from .joint_metrics import accumulate_joint_metrics

    trace = _root_carried_trace()
    root = trace.root_pos_w.clone()
    root[..., 0] = torch.tensor((0.0, 0.0, 0.004, 0.008))

    metrics = accumulate_joint_metrics(replace(trace, root_pos_w=root))

    assert metrics["root_direction_error"] <= 1.0e-6


def test_universal_stance_failure_rejects_crossing_cell() -> None:
    from .acceptance_thresholds import evaluate_metric_cell

    values = {
        "stance_xy_slip_max_m": 0.002,
        "stance_xy_slip_mean_m": 0.001,
        "stance_stationary_ratio": 0.5,
        "stance_root_carry_ratio_abs": 0.5,
        "swing_active_motion_ratio": 0.8,
        "foot_root_lead_time_min_ms": 20.0,
        "foot_root_lead_time_max_ms": 20.0,
        "root_leak_before_foot_m": 0.0,
        "cross_success_rate": 1.0,
        "foot_collision_frame_rate": 0.0,
        "calf_collision_frame_rate": 0.0,
        "thigh_collision_frame_rate": 0.0,
        "base_collision_frame_rate": 0.0,
    }

    result = evaluate_metric_cell(("small", "sphere", "0.2", "phase0"), values)

    assert not result.passed
    assert "stance_xy_slip_max_m" in result.failures


def test_flat_marks_only_small_specific_metrics_not_applicable() -> None:
    from .joint_metrics import evaluate_trace

    report = evaluate_trace(_root_carried_trace(), scenario="flat")

    assert report.metric("joint_position_violation").applicable
    assert report.metric("stance_ground_gap").applicable
    assert not report.metric("strict_cross_success").applicable
    assert report.metric("strict_cross_success").na_reason == "no small obstacle in flat scenario"


def test_small_includes_every_flat_metric_plus_small_metrics() -> None:
    from .joint_metrics import applicable_metrics

    assert applicable_metrics("flat") < applicable_metrics("small")


def test_small_cross_rejects_forbidden_semantic_stance_and_touchdown() -> None:
    from .joint_metrics import evaluate_trace

    trace = _root_carried_trace()
    forbidden = torch.zeros(1, 4, dtype=torch.bool)
    forbidden[:, 1] = True
    report = evaluate_trace(
        replace(
            trace,
            stance_on_forbidden_semantic=forbidden,
            touchdown_on_forbidden_semantic=forbidden,
        ),
        scenario="small",
        key=("small", "cuboid", "0.2", "phase0"),
    )

    assert report.metric("stance_on_forbidden_semantic").value > 0.0
    assert not report.metric("stance_on_forbidden_semantic").passed
    assert report.metric("touchdown_on_forbidden_semantic").value > 0.0
    assert not report.metric("touchdown_on_forbidden_semantic").passed


def test_small_cross_rejects_stance_gap_and_penetration_even_after_cross() -> None:
    from .joint_metrics import evaluate_trace

    trace = _root_carried_trace()
    foot = trace.foot_pos_w.clone()
    foot[:, 1:, 0, 2] = 0.042
    foot[:, 1:, 3, 2] = 0.020
    report = evaluate_trace(
        replace(
            trace,
            foot_pos_w=foot,
            strict_cross_success=torch.ones(1),
        ),
        scenario="small",
        key=("small", "cuboid", "0.2", "phase0"),
    )

    assert report.metric("strict_cross_success").passed
    assert report.metric("stance_ground_gap").value == pytest.approx(0.020)
    assert report.metric("stance_ground_gap").passed is False
    assert report.metric("stance_ground_penetration").value == pytest.approx(0.002)
    assert report.metric("stance_ground_penetration").passed is False
    assert not report.passed


def test_lifecycle_metrics_require_one_cold_then_only_warm() -> None:
    from .joint_metrics import accumulate_joint_metrics

    trace = replace(
        _root_carried_trace(),
        cold_start=torch.tensor([[False, True, False, False]]),
        warm_start=torch.tensor([[False, False, True, True]]),
        warm_cache_invariant_fault=torch.zeros(1, 4, dtype=torch.bool),
    )

    metrics = accumulate_joint_metrics(trace)

    assert metrics["cold_start_count"] == 1
    assert metrics["warm_start_count"] == 2
    assert metrics["unexpected_cold_restart_count"] == 0
    assert metrics["warm_cache_invariant_fault_count"] == 0


def test_lifecycle_metrics_detect_second_cold_without_reset() -> None:
    from .joint_metrics import accumulate_joint_metrics

    trace = replace(
        _root_carried_trace(),
        cold_start=torch.tensor([[False, True, False, True]]),
        warm_start=torch.tensor([[False, False, True, False]]),
    )

    assert accumulate_joint_metrics(trace)["unexpected_cold_restart_count"] == 1


def test_nonzero_translation_does_not_apply_zero_drift_metric() -> None:
    from .joint_metrics import applicable_metrics

    metrics = applicable_metrics("flat", (0.2, 0.0, 0.0))

    assert "root_zero_drift_one_gait_m" not in metrics
    assert "root_zero_drift_10_gaits_m" not in metrics
    assert "root_zero_drift_1000_refresh_m" not in metrics


def test_zero_translation_keeps_stance_metrics_including_carry_diagnostic() -> None:
    from .joint_metrics import applicable_metrics

    metrics = applicable_metrics("flat", (0.0, 0.0, 0.0))

    assert "stance_root_carry_ratio_abs" in metrics
    assert "stance_xy_slip_max_m" in metrics
    assert "stance_xy_slip_mean_m" in metrics
    assert "stance_stationary_ratio" in metrics
    assert "stance_ground_gap" in metrics


def test_any_invalid_trajectory_node_fails_universal_validity_metric() -> None:
    from .joint_metrics import evaluate_trace

    trace = _root_carried_trace()
    valid = trace.valid.clone()
    valid[:, -1] = False
    report = evaluate_trace(replace(trace, valid=valid), scenario="flat")

    metric = report.metric("trajectory_valid_ratio")
    assert metric.applicable
    assert metric.value == 0.75
    assert metric.threshold == 1.0
    assert metric.passed is False
    assert not report.passed


def test_all_true_validity_ratios_are_exactly_one_for_long_float32_trace() -> None:
    from .joint_metrics import accumulate_joint_metrics

    trace = _root_carried_trace()
    nodes = 97
    expanded = replace(
        trace,
        root_pos_w=trace.root_pos_w[:, :1].expand(-1, nodes, -1).clone(),
        root_rpy_w=trace.root_rpy_w[:, :1].expand(-1, nodes, -1).clone(),
        joint_pos=trace.joint_pos[:, :1].expand(-1, nodes, -1).clone(),
        foot_pos_w=trace.foot_pos_w[:, :1].expand(-1, nodes, -1, -1).clone(),
        contact_state=trace.contact_state[:, :1].expand(-1, nodes, -1).clone(),
        command_body=trace.command_body[:, :1].expand(-1, nodes, -1).clone(),
        gait_phase=torch.arange(nodes).view(1, nodes) % 24,
        foot_height_w=trace.foot_height_w[:, :1].expand(-1, nodes, -1).clone(),
        foot_small_distance_m=trace.foot_small_distance_m[:, :1].expand(-1, nodes, -1).clone(),
        line_alpha=torch.ones(1, nodes),
        nominal_root_pos_w=trace.nominal_root_pos_w[:, :1].expand(-1, nodes, -1).clone(),
        nominal_root_rpy_w=trace.nominal_root_rpy_w[:, :1].expand(-1, nodes, -1).clone(),
        valid=torch.ones(1, nodes, dtype=torch.bool),
        map_valid=torch.ones(1, nodes, dtype=torch.bool),
        timestamps=0.02 * torch.arange(nodes).view(1, nodes),
    )

    metrics = accumulate_joint_metrics(expanded)

    assert metrics["trajectory_valid_ratio"] == 1.0
    assert metrics["map_valid_ratio"] == 1.0

    invalid = expanded.valid.clone()
    invalid[:, -1] = False
    partially_valid = replace(expanded, valid=invalid)
    partial_metrics = accumulate_joint_metrics(partially_valid)
    assert partial_metrics["trajectory_valid_ratio"] == 96.0 / 97.0


def test_all_stationary_stance_edges_use_an_exact_boolean_ratio(monkeypatch) -> None:
    from . import joint_metrics

    trace = _root_carried_trace()
    stationary = replace(
        trace,
        root_pos_w=trace.root_pos_w[:, :1].expand_as(trace.root_pos_w).clone(),
        foot_pos_w=trace.foot_pos_w[:, :1].expand_as(trace.foot_pos_w).clone(),
        nominal_root_pos_w=trace.nominal_root_pos_w[:, :1]
        .expand_as(trace.nominal_root_pos_w)
        .clone(),
    )
    original_masked_mean = joint_metrics._masked_mean

    def inexact_float_mean(value, mask, default=0.0):
        if value.dtype.is_floating_point and torch.all(value[mask] == 1.0):
            return 0.99999994
        return original_masked_mean(value, mask, default)

    monkeypatch.setattr(joint_metrics, "_masked_mean", inexact_float_mean)

    assert joint_metrics.accumulate_joint_metrics(stationary)["stance_stationary_ratio"] == 1.0


def test_final_metric_schema_contains_every_frozen_design_id() -> None:
    from .joint_metrics import FINAL_METRIC_IDS

    required = {
        "joint_acceleration_mean_rps2",
        "joint_acceleration_p95_rps2",
        "joint_jerk_max_rps3",
        "root_zero_drift_one_gait_m",
        "root_zero_drift_10_gaits_m",
        "root_zero_drift_1000_refresh_m",
        "alpha0_selected_ratio",
        "alpha0_selected_max_run",
        "line_search_no_feasible_ratio",
        "publish_ratio",
        "stop_ratio",
        "warm_shift_rebase_error",
        "retarget_trajectory_change_norm",
        "map_age_frames_max",
        "map_state_frame_mismatch_count",
        "world_query_transform_error",
        "target_change_normal_m",
        "target_change_due_to_command_m",
        "target_change_due_to_map_m",
        "target_change_due_to_unsafe_invalidation_m",
        "latched_target_drift_m",
        "cross_direction_margin_mps",
        "kkt_primal_residual",
        "kkt_dual_residual",
        "nominal_safe_after_retarget",
        "nominal_min_clearance_after_retarget",
    }

    assert required <= FINAL_METRIC_IDS


def test_zero_translation_marks_directional_swing_lead_metrics_not_applicable() -> None:
    from .joint_metrics import applicable_metrics

    metrics = applicable_metrics("small", (0.0, 0.0, 0.0))

    assert "root_direction_error" not in metrics
    assert "strict_cross_success" not in metrics
    assert "cross_direction_margin_mps" not in metrics
    assert "swing_active_motion_ratio" not in metrics
    assert "foot_root_lead_time_min_ms" not in metrics
    assert "foot_root_lead_time_max_ms" not in metrics
    assert "root_leak_before_foot_m" not in metrics
    assert "stance_root_carry_ratio_abs" in metrics


def test_foot_root_lead_is_report_only_while_root_leak_remains_a_gate() -> None:
    from .acceptance_thresholds import THRESHOLDS

    assert "foot_root_lead_time_min_ms" not in THRESHOLDS
    assert "foot_root_lead_time_max_ms" not in THRESHOLDS
    assert THRESHOLDS["root_leak_before_foot_m"] == ("le", 0.0005)


def test_every_metric_report_records_its_evidence_source() -> None:
    from .joint_metrics import evaluate_trace

    report = evaluate_trace(_root_carried_trace(), scenario="small")

    assert report.metrics
    assert all(metric.source for metric in report.metrics.values())
    assert report.metric("base_collision_frame_rate").source == "P+A+M"
    assert report.metric("map_age_frames_max").source == "M"


def test_actual_world_collision_fails_even_when_planned_geometry_is_safe() -> None:
    from .joint_metrics import evaluate_trace

    trace = _root_carried_trace()
    planned = {
        part: torch.zeros((1, 4), dtype=torch.bool)
        for part in ("foot", "knee", "calf", "thigh", "base")
    }
    actual = {part: value.clone() for part, value in planned.items()}
    actual["calf"][0, 2] = True

    report = evaluate_trace(
        replace(
            trace,
            planned_part_collision=planned,
            actual_part_collision=actual,
        ),
        scenario="small",
    )

    metric = report.metric("calf_collision_frame_rate")
    assert metric.value == 0.25
    assert metric.passed is False
    assert metric.source == "P+A+M"
