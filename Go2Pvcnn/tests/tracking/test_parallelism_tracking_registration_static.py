from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_parallelism_tracking_task_id_is_registered() -> None:
    source = (ROOT / "tracking/register_envs.py").read_text()
    assert "Isaac-Go2-Parallelism-Tracking-Flat-v0" in source
    assert "ParallelismTrackingFlatEnvCfg" in source


def test_main_task_registration_imports_tracking_registration() -> None:
    source = (ROOT / "go2_pvcnn/tasks/register_envs.py").read_text()
    assert "tracking.register_envs" in source
