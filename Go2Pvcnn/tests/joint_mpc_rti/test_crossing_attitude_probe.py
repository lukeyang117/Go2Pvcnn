from __future__ import annotations

import torch


def test_attitude_metrics_detect_airborne_scheduled_touchdown() -> None:
    from .small_obstacle_attitude_probe import summarize_attitude_trace

    root_rpy = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.05, -0.10, 0.0], [0.10, -0.20, 0.0]]]
    )
    control = torch.zeros(1, 3, 18)
    control[:, 1:, 3] = 0.5
    foot = torch.zeros(1, 3, 4, 3)
    foot[..., 2] = 0.022
    foot[:, 2, 0, 2] = 0.052
    height = torch.zeros(1, 3, 4)
    contact = torch.tensor([[[False, True, True, True], [False, True, True, True], [True, True, True, True]]])
    joint = torch.zeros(1, 3, 12)
    alpha = torch.ones(1, 3)

    metrics = summarize_attitude_trace(
        root_rpy_w=root_rpy,
        control=control,
        foot_pos_w=foot,
        foot_height_w=height,
        contact_state=contact,
        joint_pos=joint,
        line_search_alpha=alpha,
        foot_contact_offset=0.022,
        dt=0.02,
    )

    assert metrics["touchdown_count"].item() == 1
    assert metrics["airborne_touchdown_20mm_count"].item() == 1
    torch.testing.assert_close(metrics["touchdown_surface_error_max_m"], torch.tensor([0.03]))
    assert metrics["roll_abs_max_deg"].item() > 5.0
    assert metrics["pitch_abs_max_deg"].item() > 10.0
