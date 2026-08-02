from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_parallelism_tracking_env_cfg_contains_required_terms() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    assert "resampling_time_range=(0.48, 0.48)" in source
    assert "downsampled_elevation_semantic_scan" in source
    assert "parallelism_ref_joint_pos_too_far" in source
    assert "ParallelismTrackingFlatEnvCfg" in source
    assert "parallelism_plan_batch_size: int = 64" in source
    assert "self.scene.robot.init_state.pos = (0.0, 0.0, 0.3)" in source


def test_train_script_contains_parallelism_tracking_experiment() -> None:
    source = (ROOT / "scripts/train.py").read_text()
    assert "parallelism_tracking_flat" in source
    assert "Isaac-Go2-Parallelism-Tracking-Flat-v0" in source


def test_train_cfg_accepts_parallelism_tracking_experiment() -> None:
    source = (ROOT / "agent/train_cfg.py").read_text()
    assert '"parallelism_tracking_flat"' in source


def test_tracking_play_cfg_spawns_a_non_colliding_reference_go2() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()

    assert "ParallelismTrackingPlaySceneCfg" in source
    assert "reference_robot" in source
    assert "ParallelismReferenceGo2" in source
    assert "collision_enabled=False" in source


def test_tracking_play_cfg_does_not_override_instanced_reference_material() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    reference_source = source[
        source.index("reference_robot: ArticulationCfg")
        : source.index("\n\n\n@configclass", source.index("reference_robot: ArticulationCfg"))
    ]

    assert "visual_material=" not in reference_source


def test_tracking_play_cfg_disables_timeout_for_unbounded_play() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    play_source = source[source.index("class ParallelismTrackingFlatEnvCfg_PLAY") :]

    assert "self.terminations.time_out = None" in play_source
