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
            candidate_collision_bits=torch.zeros(1, 4, 50, 5, dtype=torch.bool),
            collision_shape_names=tuple(f"shape_{idx}" for idx in range(5)),
            collision_surface_point_count=6,
            candidate_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            fk_touchdown_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            selected_index=torch.zeros(1, 4, dtype=torch.long),
            candidate_radius_m=0.42,
        ),
    )
    result = parallelism_trajectory_to_viewer_result(traj)

    assert result.num_frames == 24
    assert result.root_pos_w.shape == (1, 24, 3)
    assert result.foot_pos_w.shape == (1, 24, 4, 3)
    assert result.planned_touchdown_w.shape == (1, 4, 3)
    assert result.parallelism_candidate_center_w.shape == (1, 4, 3)
    assert result.parallelism_candidate_radius_m == 0.42


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
            candidate_collision_bits=torch.zeros(1, 4, 50, 5, dtype=torch.bool),
            collision_shape_names=tuple(f"shape_{idx}" for idx in range(5)),
            collision_surface_point_count=6,
            candidate_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            fk_touchdown_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            selected_index=torch.zeros(1, 4, dtype=torch.long),
        ),
    )
    result = parallelism_trajectory_to_viewer_result(traj)

    assert result.joint_mpc_diagnostics is None
    assert result.nominal_state is None
    assert result.alpha_candidate_state is None


def test_viewer_parallelism_reject_uses_collision_shape_names():
    from types import SimpleNamespace
    from extension.viz.go2_foostep_planner import _format_parallelism_reject_diagnostics

    diagnostics = SimpleNamespace(
        candidate_reject_bits=torch.zeros(1, 4, 50, 6, dtype=torch.bool),
        candidate_valid=torch.ones(1, 4, 50, dtype=torch.bool),
        candidate_collision_bits=torch.zeros(1, 4, 50, 2, dtype=torch.bool),
        collision_shape_names=("calf_mid_cylinder", "foot_sphere"),
    )
    diagnostics.candidate_collision_bits[..., 0] = True

    text = _format_parallelism_reject_diagnostics(SimpleNamespace(parallelism_diagnostics=diagnostics))

    assert "collision_detail(calf_mid_cylinder=200 foot_sphere=0)" in text


def test_viewer_test_terminal_state_scales_command():
    from extension.viz.go2_foostep_planner import ViewerTestTerminalState, _apply_test_terminal_command

    state = ViewerTestTerminalState(vx=0.5, vy=0.25, vyaw=0.75, swing_clearance_m=0.12, enabled=True)
    command = _apply_test_terminal_command(torch.zeros(1, 3), state)

    assert torch.allclose(command, torch.tensor([[0.5, 0.25, 0.75]]))


def test_test_terminal_command_supports_signed_velocity():
    from extension.viz.go2_foostep_planner import ViewerTestTerminalState, _apply_test_terminal_command

    state = ViewerTestTerminalState(vx=-0.4, vy=-0.2, vyaw=-0.7, enabled=True)
    command = _apply_test_terminal_command(torch.zeros(2, 3), state)

    assert torch.allclose(command, torch.tensor([[-0.4, -0.2, -0.7], [-0.4, -0.2, -0.7]]))


def test_parallelism_cfg_from_viewer_uses_debug_panel_values():
    from argparse import Namespace
    from extension.viz.go2_foostep_planner import ViewerTestTerminalState, _parallelism_cfg_from_viewer_args

    cfg = _parallelism_cfg_from_viewer_args(
        Namespace(plan_dt=0.02),
        ViewerTestTerminalState(
            swing_clearance_m=0.11,
            semantic_touchdown_margin_m=0.04,
            candidate_radius_m=0.42,
            standstill_fallback_enabled=False,
        ),
    )

    assert cfg.swing_clearance_m == 0.11
    assert cfg.min_swing_apex_m == 0.08
    assert cfg.semantic_touchdown_margin_m == 0.04
    assert cfg.candidate_radius_m == 0.42
    assert cfg.standstill_fallback_enabled is False


def test_parallelism_visualization_flags_preserve_values():
    from extension.viz.go2_foostep_planner import _parallelism_visualization_flags

    flags = _parallelism_visualization_flags(
        show_mesh=True,
        show_collision_body=False,
        show_contact_points=True,
    )

    assert flags.show_mesh is True
    assert flags.show_collision_body is False
    assert flags.show_contact_points is True
