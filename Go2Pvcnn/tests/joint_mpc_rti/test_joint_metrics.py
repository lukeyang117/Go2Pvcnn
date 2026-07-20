from __future__ import annotations

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


def test_nonzero_translation_does_not_apply_zero_drift_metric() -> None:
    from .joint_metrics import applicable_metrics

    metrics = applicable_metrics("flat", (0.2, 0.0, 0.0))

    assert "root_zero_drift_m" not in metrics
