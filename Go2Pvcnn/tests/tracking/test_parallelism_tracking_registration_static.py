from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_parallelism_tracking_task_id_is_registered() -> None:
    source = (ROOT / "tracking/register_envs.py").read_text()
    assert "Isaac-Go2-Parallelism-Tracking-Flat-v0" in source
    assert "ParallelismTrackingFlatEnvCfg" in source


def test_main_task_registration_imports_tracking_registration() -> None:
    source = (ROOT / "go2_pvcnn/tasks/register_envs.py").read_text()
    assert "tracking.register_envs" in source


def test_play_entrypoint_maps_parallelism_to_the_tracking_play_cfg() -> None:
    source = (ROOT / "scripts/play.py").read_text()

    assert "ParallelismTrackingFlatEnvCfg_PLAY" in source
    assert '"parallelism_tracking_flat": (' in source
    assert '"Isaac-Go2-Parallelism-Tracking-Flat-v0"' in source


def test_parallelism_tracking_env_logs_episode_reference_metrics() -> None:
    source = (ROOT / "tracking/env.py").read_text()
    registration = (ROOT / "tracking/register_envs.py").read_text()

    assert "class ParallelismTrackingEnv" in source
    assert 'Episode_Tracking/episode_joint_mean_error' in source
    assert 'Episode_Tracking/episode_joint_max_error' in source
    assert 'Episode_Tracking/episode_reference_root_lin_vel_error' in source
    assert 'Episode_Tracking/episode_reference_root_ang_vel_error' in source
    assert 'entry_point="tracking.env:ParallelismTrackingEnv"' in registration
