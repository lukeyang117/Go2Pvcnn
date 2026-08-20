from __future__ import annotations

from types import SimpleNamespace

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


def test_parallelism_viewer_adapter_preserves_root_roll_and_pitch():
    from extension.convention import euler_to_quat_batch
    from extension.parallelism import ParallelismCfg, ParallelismDiagnostics, ParallelismTrajectory
    from extension.parallelism.viewer_adapter import parallelism_trajectory_to_viewer_result

    root_rpy = torch.tensor([[[0.20, -0.30, 0.40]]], dtype=torch.float32).expand(1, 24, 3).clone()
    trajectory = ParallelismTrajectory(
        root_pos_w=torch.zeros(1, 24, 3),
        root_rpy_w=root_rpy,
        joint_pos=torch.zeros(1, 24, 12),
        foot_pos_w=torch.zeros(1, 24, 4, 3),
        contact_state=torch.ones(1, 24, 4, dtype=torch.bool),
        valid=torch.ones(1, dtype=torch.bool),
        selected_foothold_w=torch.zeros(1, 4, 3),
        selected_score=torch.zeros(1, 4),
        diagnostics=ParallelismDiagnostics(
            candidate_center_w=torch.zeros(1, 4, 3),
            candidate_w=torch.zeros(1, 4, 50, 3),
            candidate_score=torch.zeros(1, 4, 50),
            candidate_valid=torch.ones(1, 4, 50, dtype=torch.bool),
            candidate_reject_bits=torch.zeros(1, 4, 50, 6, dtype=torch.bool),
            candidate_collision_bits=torch.zeros(1, 4, 50, len(ParallelismCfg().official_collision_shapes), dtype=torch.bool),
            collision_shape_names=tuple(spec.name for spec in ParallelismCfg().official_collision_shapes),
            collision_surface_point_count=6,
            candidate_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            fk_touchdown_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            selected_index=torch.zeros(1, 4, dtype=torch.long),
        ),
    )

    result = parallelism_trajectory_to_viewer_result(trajectory)
    expected = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2])

    torch.testing.assert_close(result.root_quat_w, expected)


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


def test_parallelism_cfg_from_viewer_uses_terrain_root_sampling_values():
    from argparse import Namespace
    from extension.viz.go2_foostep_planner import ViewerTestTerminalState, _parallelism_cfg_from_viewer_args

    cfg = _parallelism_cfg_from_viewer_args(
        Namespace(plan_dt=0.02),
        ViewerTestTerminalState(
            terrain_following_pitch_sample_range_m=0.30,
            terrain_following_pitch_sample_count=9,
            terrain_following_roll_sample_range_m=0.25,
            terrain_following_roll_sample_count=7,
            terrain_following_rpy_deadband_rad=0.03,
        ),
    )

    assert cfg.terrain_following_pitch_sample_range_m == 0.30
    assert cfg.terrain_following_pitch_sample_count == 9
    assert cfg.terrain_following_roll_sample_range_m == 0.25
    assert cfg.terrain_following_roll_sample_count == 7
    assert cfg.terrain_following_rpy_deadband_rad == 0.03


def test_platform_reset_request_is_independent_from_keyboard_reset():
    from extension.viz.go2_foostep_planner import (
        TeleopCommand,
        ViewerTestTerminalState,
        _consume_platform_reset_request,
    )

    state = ViewerTestTerminalState(platform_reset_requested=True)
    teleop_cmd = TeleopCommand(values=torch.zeros(1, 3), reset_requested=False)

    assert _consume_platform_reset_request(state) is True
    assert state.platform_reset_requested is False
    assert teleop_cmd.reset_requested is False


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


def test_viewer_terrain_following_mask_uses_selected_column_proportions():
    from extension.viz.go2_foostep_planner import _viewer_terrain_following_mask_from_selection

    sub_terrains = {
        "flat_dense_small_obstacles": SimpleNamespace(proportion=0.05),
        "flat": SimpleNamespace(proportion=0.05),
        "random_rough": SimpleNamespace(proportion=0.1),
        "hf_pyramid_slope": SimpleNamespace(proportion=0.1),
        "hf_pyramid_slope_inv": SimpleNamespace(proportion=0.1),
        "boxes": SimpleNamespace(proportion=0.2),
        "pyramid_stairs": SimpleNamespace(proportion=0.2),
        "pyramid_stairs_inv": SimpleNamespace(proportion=0.2),
    }
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                num_cols=20,
                sub_terrains=sub_terrains,
            )
        )
    )
    scene = SimpleNamespace(terrain=terrain)

    assert _viewer_terrain_following_mask_from_selection(scene, terrain_col=0, device="cpu").item() is False
    assert _viewer_terrain_following_mask_from_selection(scene, terrain_col=1, device="cpu").item() is False
    assert _viewer_terrain_following_mask_from_selection(scene, terrain_col=2, device="cpu").item() is True
    assert _viewer_terrain_following_mask_from_selection(scene, terrain_col=12, device="cpu").item() is True
    assert _viewer_terrain_following_mask_from_selection(scene, terrain_col=16, device="cpu").item() is True
