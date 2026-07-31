from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_parallelism_tracking_env_cfg_contains_required_terms() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    assert "resampling_time_range=(0.48, 0.48)" in source
    assert "downsampled_elevation_semantic_scan" in source
    assert "parallelism_ref_joint_pos_too_far" in source
    assert "ParallelismTrackingFlatEnvCfg" in source


def test_train_script_contains_parallelism_tracking_experiment() -> None:
    source = (ROOT / "scripts/train.py").read_text()
    assert "parallelism_tracking_flat" in source
    assert "Isaac-Go2-Parallelism-Tracking-Flat-v0" in source
