from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "Go2Pvcnn/scripts/train.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_train_single_env_livestream_installs_follow_camera_only_for_env0() -> None:
    source = _source()
    assert "_SingleEnvLivestreamFollowCamera" in source
    assert "_attach_single_env_livestream_follow_camera" in source
    assert "livestream in (1, 2)" in source
    assert "num_envs != 1" in source
    assert "rank != 0" in source
    assert "scene[\"robot\"].data.root_pos_w[0]" in source
    assert "sim.set_camera_view" in source
    assert "wrap_env_step" in source


def test_train_preserves_livestream_flag_before_applauncher_mutates_args() -> None:
    source = _source()
    assert "requested_livestream = int(getattr(args_cli, \"livestream\", 0))" in source
    assert source.index("requested_livestream = int(getattr(args_cli, \"livestream\", 0))") < source.index(
        "app_launcher, simulation_app = _launch_app(args_cli)"
    )
    assert "livestream=requested_livestream" in source


def test_train_exposes_mpc_num_envs_cli_override() -> None:
    source = _source()

    assert '"--mpc_num_envs"' in source
    assert "default=None" in source
    assert "args_cli.mpc_num_envs" in source
    assert "env_cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size" in source
    assert source.index("env_cfg.scene.num_envs = args_cli.num_envs") < source.index(
        "env_cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size = int(args_cli.mpc_num_envs)"
    )


def test_train_versions_run_directory_by_git_short_hash_without_tag_file() -> None:
    source = _source()
    assert "get_git_short_hash" in source
    assert "date_dir" in source
    assert "git_hash" in source
    assert "os.path.join(log_root_path, date_dir, git_hash)" in source
    assert '"tag.txt"' not in source


def test_train_registers_parallelism_small_obstacles_experiment() -> None:
    source = _source()
    assert '"parallelism_tracking_small_obstacles"' in source
    assert "ParallelismTrackingSmallObstaclesEnvCfg" in source
    assert "Isaac-Go2-Parallelism-Tracking-Small-Obstacles-v0" in source
