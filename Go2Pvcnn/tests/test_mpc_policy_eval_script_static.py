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
    assert "choices=[\"tracking\", \"small_collision\"]" in source


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
