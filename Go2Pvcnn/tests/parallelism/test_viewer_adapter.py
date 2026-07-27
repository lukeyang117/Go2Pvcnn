from __future__ import annotations

import torch


def test_viewer_adapter_contract():
    from extension.parallelism import ParallelismDiagnostics, ParallelismTrajectory
    from extension.parallelism.viewer_adapter import parallelism_trajectory_to_viewer_result

    traj = ParallelismTrajectory(
        root_pos_w=torch.zeros(1, 24, 3),
        root_rpy_w=torch.zeros(1, 24, 3),
        joint_pos=torch.zeros(1, 24, 12),
        foot_pos_w=torch.zeros(1, 24, 4, 3),
        contact_state=torch.zeros(1, 24, 4, dtype=torch.bool),
        valid=torch.ones(1, dtype=torch.bool),
        selected_foothold_w=torch.zeros(1, 4, 3),
        selected_score=torch.zeros(1, 4),
        diagnostics=ParallelismDiagnostics(
            candidate_center_w=torch.zeros(1, 4, 3),
            candidate_w=torch.zeros(1, 4, 50, 3),
            candidate_score=torch.zeros(1, 4, 50),
            candidate_valid=torch.ones(1, 4, 50, dtype=torch.bool),
            candidate_reject_bits=torch.zeros(1, 4, 50, 6, dtype=torch.bool),
            candidate_collision_bits=torch.zeros(1, 4, 50, 10, dtype=torch.bool),
            collision_ellipsoid_names=tuple(f"ellipsoid_{idx}" for idx in range(10)),
            collision_probe_count=5,
            candidate_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            fk_touchdown_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            selected_index=torch.zeros(1, 4, dtype=torch.long),
        ),
    )
    result = parallelism_trajectory_to_viewer_result(traj)

    assert result.num_frames == 24
    assert result.root_pos_w.shape == (1, 24, 3)
    assert result.foot_pos_w.shape == (1, 24, 4, 3)
    assert result.planned_touchdown_w.shape == (1, 4, 3)
    assert result.parallelism_candidate_center_w.shape == (1, 4, 3)
    assert result.parallelism_candidate_radius_m == 0.24


def test_viewer_adapter_exposes_overlay_optional_fields():
    from extension.parallelism import ParallelismDiagnostics, ParallelismTrajectory
    from extension.parallelism.viewer_adapter import parallelism_trajectory_to_viewer_result

    traj = ParallelismTrajectory(
        root_pos_w=torch.zeros(1, 24, 3),
        root_rpy_w=torch.zeros(1, 24, 3),
        joint_pos=torch.zeros(1, 24, 12),
        foot_pos_w=torch.zeros(1, 24, 4, 3),
        contact_state=torch.zeros(1, 24, 4, dtype=torch.bool),
        valid=torch.ones(1, dtype=torch.bool),
        selected_foothold_w=torch.zeros(1, 4, 3),
        selected_score=torch.zeros(1, 4),
        diagnostics=ParallelismDiagnostics(
            candidate_center_w=torch.zeros(1, 4, 3),
            candidate_w=torch.zeros(1, 4, 50, 3),
            candidate_score=torch.zeros(1, 4, 50),
            candidate_valid=torch.ones(1, 4, 50, dtype=torch.bool),
            candidate_reject_bits=torch.zeros(1, 4, 50, 6, dtype=torch.bool),
            candidate_collision_bits=torch.zeros(1, 4, 50, 10, dtype=torch.bool),
            collision_ellipsoid_names=tuple(f"ellipsoid_{idx}" for idx in range(10)),
            collision_probe_count=5,
            candidate_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            fk_touchdown_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            selected_index=torch.zeros(1, 4, dtype=torch.long),
        ),
    )
    result = parallelism_trajectory_to_viewer_result(traj)

    assert result.joint_mpc_diagnostics is None
    assert result.nominal_state is None
    assert result.alpha_candidate_state is None


def test_viewer_parallelism_reject_uses_ellipsoid_names():
    from types import SimpleNamespace
    from extension.viz.go2_foostep_planner import _format_parallelism_reject_diagnostics

    diagnostics = SimpleNamespace(
        candidate_reject_bits=torch.zeros(1, 4, 50, 6, dtype=torch.bool),
        candidate_valid=torch.ones(1, 4, 50, dtype=torch.bool),
        candidate_collision_bits=torch.zeros(1, 4, 50, 2, dtype=torch.bool),
        collision_ellipsoid_names=("calf_mid_bar", "foot_pad"),
    )
    diagnostics.candidate_collision_bits[..., 0] = True

    text = _format_parallelism_reject_diagnostics(SimpleNamespace(parallelism_diagnostics=diagnostics))

    assert "collision_detail(calf_mid_bar=200 foot_pad=0)" in text
