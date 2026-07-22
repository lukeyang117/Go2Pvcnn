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
    return JointMetricTrace(
        root_pos_w=root_pos,
        root_rpy_w=torch.zeros(1, 4, 3),
        foot_pos_w=foot,
        contact_state=contact,
        command_body=command,
        foot_height_w=zeros,
        foot_small_distance_m=torch.ones(1, 4, 4),
        part_collision={part: torch.zeros(1, 4, dtype=torch.bool) for part in ("foot", "calf", "thigh", "base")},
        valid=torch.ones(1, 4, dtype=torch.bool),
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


def test_joint_metrics_reports_root_contact_collision_and_line_search_contract() -> None:
    from dataclasses import replace

    from .joint_metrics import accumulate_joint_metrics

    trace = _root_carried_trace()
    root_rpy = trace.root_rpy_w.clone()
    root_rpy[0, :, 0] = torch.tensor([0.0, 0.02, 0.04, 0.06])
    root_rpy[0, :, 1] = torch.tensor([0.0, -0.01, -0.02, -0.03])
    contact = trace.contact_state.clone()
    contact[0, :, 0] = torch.tensor([False, False, True, True])
    extension = torch.zeros(1, 4, 4, dtype=torch.long)
    extension[0, 1, 0] = 1
    reliable = contact.clone()
    reliable[0, 2, 0] = False
    part_collision = dict(trace.part_collision)
    part_collision["knee"] = torch.tensor([[False, True, False, False]])

    metrics = accumulate_joint_metrics(
        replace(
            trace,
            root_rpy_w=root_rpy,
            contact_state=contact,
            part_collision=part_collision,
            reliable_stance=reliable,
            swing_extension_age=extension,
            recovery_state=torch.zeros_like(contact),
            liftoff_blocked=torch.zeros_like(contact),
            line_search_alpha=torch.tensor([[1.0, 0.5, 0.0, 0.1]]),
        )
    )

    assert metrics["root_roll_error_abs_max_deg"] > 3.0
    assert metrics["root_pitch_error_abs_max_deg"] > 1.0
    assert metrics["root_roll_pitch_rate_max_rps"] > 0.0
    assert metrics["confirmed_touchdown_count"] == 1
    assert metrics["unsafe_stance_anchor_count"] == 1
    assert metrics["swing_extension_frames_max"] == 1
    assert metrics["knee_collision_frame_rate"] == 0.25
    assert metrics["line_search_alpha_0_count"] == 1
    assert metrics["line_search_alpha_0_max_run"] == 1


def test_h30_root_touchdown_knee_and_line_search_thresholds_are_universal() -> None:
    from .acceptance_thresholds import evaluate_metric_cell

    result = evaluate_metric_cell(
        ("small", "sphere", "vx=0.2", "vy=0.3", "yaw=0.5"),
        {
            "root_roll_error_abs_max_deg": 6.1,
            "root_pitch_error_abs_max_deg": 5.0,
            "airborne_touchdown_count": 1,
            "unsafe_stance_anchor_count": 0,
            "knee_collision_frame_rate": 0.01,
            "line_search_alpha_0_rate": 0.11,
            "line_search_alpha_0_max_run": 3,
        },
    )

    assert not result.passed
    assert set(result.failures) == {
        "root_roll_error_abs_max_deg",
        "airborne_touchdown_count",
        "knee_collision_frame_rate",
        "line_search_alpha_0_rate",
        "line_search_alpha_0_max_run",
    }


def test_joint_metrics_reports_root_lateral_and_yaw_nominal_deviation() -> None:
    from dataclasses import replace

    from .joint_metrics import accumulate_joint_metrics

    trace = _root_carried_trace()
    nominal = trace.root_pos_w.clone()
    nominal[..., 1] -= torch.tensor([0.0, 0.005, 0.010, 0.015])
    nominal_rpy = torch.zeros_like(trace.root_rpy_w)
    metrics = accumulate_joint_metrics(
        replace(trace, root_nominal_pos_w=nominal, root_nominal_rpy_w=nominal_rpy)
    )

    assert metrics["root_lateral_offset_from_nominal_m"] > 0.0
    assert metrics["root_lateral_velocity_error_mps"] > 0.0
    assert metrics["root_yaw_error_from_nominal_deg"] == 0.0
    assert metrics["root_yaw_rate_assist_error_rps"] == 0.0


def test_joint_metrics_distinguishes_blocked_liftoff_from_guard_violation() -> None:
    from dataclasses import replace

    from .joint_metrics import accumulate_joint_metrics

    trace = _root_carried_trace()
    contact = trace.contact_state.clone()
    contact[0, :, 0] = torch.tensor([True, True, False, False])
    blocked = torch.zeros_like(contact)
    blocked[0, 1, 0] = True
    blocked[0, 2, 0] = True
    metrics = accumulate_joint_metrics(
        replace(trace, contact_state=contact, liftoff_blocked=blocked)
    )

    assert metrics["liftoff_blocked_count"] == 2
    assert metrics["liftoff_guard_violation_count"] == 1
