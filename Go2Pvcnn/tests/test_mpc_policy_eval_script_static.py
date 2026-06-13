from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "Go2Pvcnn/scripts/mpc_policy_eval.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_mpc_policy_eval_script_exists_and_has_required_cli() -> None:
    source = _source()
    for flag in (
        "--mode",
        "--num-rounds",
        "--max-steps",
        "--run-dir",
        "--checkpoint",
        "--command-mode",
        "--small-count-per-tile",
        "--collision-force-threshold",
        "--output-dir",
    ):
        assert flag in source
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert "controlled_crossing" in source
    assert "--crossing-speeds" in source
    assert "--crossing-lateral-offsets" in source
    assert "--crossing-obstacles-per-env" in source


def test_mpc_policy_eval_script_has_no_shell_wrapper_dependency() -> None:
    source = _source()
    assert "mpc_policy_eval.sh" not in source
    assert "shell=True" not in source
    assert "subprocess" not in source


def test_mpc_policy_eval_script_defines_round_and_command_helpers() -> None:
    module = ast.parse(_source())
    function_names = {node.name for node in ast.walk(module) if isinstance(node, ast.FunctionDef)}
    assert "build_arg_parser" in function_names
    assert "validate_eval_args" in function_names
    assert "command_for_step" in function_names
    assert "run_eval" in function_names
    assert "main" in function_names


def test_mpc_policy_eval_writes_required_output_files() -> None:
    source = _source()
    assert "metrics.jsonl" in source
    assert "rounds.jsonl" in source
    assert "summary.json" in source
    assert "config.json" in source
    assert "write_jsonl" in source
    assert "write_summary" in source


def test_mpc_policy_eval_loads_policy_and_uses_eval_cfgs() -> None:
    source = _source()
    assert "OnPolicyRunner" in source
    assert "runner.load" in source
    assert "TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg" in source
    assert "TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg" in source
    assert "TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg" in source


def test_mpc_policy_eval_has_controlled_crossing_metric_helpers() -> None:
    source = _source()
    for name in (
        "ControlledCrossingAccumulator",
        "build_controlled_crossing_commands",
        "controlled_crossing_step_metrics",
        "success_by_speed",
        "success_by_lateral_offset",
        "small_overpass_success_rate_over_opportunities",
    ):
        assert name in source


def test_mpc_policy_eval_collects_tracking_reference_from_runtime_manager() -> None:
    source = _source()
    assert "TrackingRoundAccumulator" in source
    assert "tracking_metrics_for_env_step" in source
    assert "current_reference" in source
    assert "\"foot_pos_w\"" in source
    assert "_trajectory_reference_cache" in source
    assert "current_frame_ids" in source
    assert "reference_valid_ratio" in source
    assert "body_pos_w" in source


def test_mpc_policy_eval_collects_small_collision_env_rate_from_semantic_sensor() -> None:
    source = _source()
    assert "semantic_small_force_matrix_w" in source
    assert "semantic_contact_small" in source
    assert "force_matrix_w" in source
    assert "SmallCollisionRoundAccumulator" in source
    assert "aggregate_small_collision_rounds" in source
    assert "small_collision_env_rate_per_round" in source
    assert "aggregate_small_collision_env_rate" in source


def test_mpc_policy_eval_livestream_syncs_command_and_markers() -> None:
    source = _source()
    assert "sync_command_to_policy" in source
    assert "sync_command_to_mpc" in source
    assert "update_mpc_foot_markers" in source
    assert "VisualizationMarkers" in source
    assert "_trajectory_reference_cache" in source


def test_mpc_policy_eval_livestream_draws_full_foot_trajectories_and_follows_robot() -> None:
    source = _source()
    assert "build_mpc_foot_trajectory_markers" in source
    assert "update_mpc_foot_trajectory_markers" in source
    assert "_reference_foot_trajectory_w" in source
    assert "for leg_idx" in source
    assert "foot_traj[leg_idx].visualize" in source
    assert "set_camera_view" in source
    assert "update_follow_camera" in source
    assert "int(args.num_envs) == 1" in source
    assert "base.sim.render()" in source


def test_mpc_policy_eval_follow_camera_debug_logs_viewport_camera_state() -> None:
    source = _source()
    assert "--debug-follow-camera" in source
    assert "follow_camera_debug.jsonl" in source
    assert "_collect_follow_camera_debug" in source
    assert "get_active_viewport_and_window" in source
    assert "get_active_viewport_camera_string" in source
    assert "get_viewport_window_camera_string" in source
    assert "active_viewport_camera_path" in source
    assert "default_camera_world_position" in source
    assert "active_camera_world_position" in source


def test_mpc_policy_eval_preserves_livestream_flag_before_applauncher_mutates_args() -> None:
    source = _source()
    assert "livestream_enabled = int(getattr(args, \"livestream\", -1)) in (1, 2)" in source
    assert source.index("livestream_enabled = int(getattr(args, \"livestream\", -1)) in (1, 2)") < source.index(
        "app_launcher = AppLauncher(args)"
    )
    assert "render_mode = \"rgb_array\" if livestream_enabled else None" in source
    assert "mpc_foot_markers = build_mpc_foot_markers() if livestream_enabled else None" in source
    assert "if livestream_enabled and int(args.num_envs) == 1:" in source


def test_eval_records_body_command_source_diagnostics() -> None:
    source = _source()
    for field in (
        "requested_command_body",
        "policy_command_body",
        "mpc_input_command_body",
        "command_body_match_max_abs_error",
    ):
        assert field in source
    assert "command_body_source_diagnostics" in source
    assert "_commands_from_env" in source


def test_eval_records_flat_planned_direction_metrics() -> None:
    source = _source()
    for field in (
        "planned_root_direction_cosine",
        "planned_root_lateral_ratio",
        "planned_per_leg_direction_cosine_xy",
        "planned_per_leg_lateral_ratio_xy",
        "planned_insufficient_motion",
        "planned_insufficient_leg_motion",
        "semantic_nonzero_count",
    ):
        assert field in source
    assert "planned_direction_metrics_from_reference_cache" in source
