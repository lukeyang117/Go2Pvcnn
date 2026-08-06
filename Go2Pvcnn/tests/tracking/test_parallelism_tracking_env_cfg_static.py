from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_parallelism_tracking_env_cfg_contains_required_terms() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    assert "resampling_time_range=(0.48, 0.48)" in source
    assert "downsampled_elevation_semantic_scan" in source
    assert "parallelism_ref_joint_pos_too_far" in source
    assert "reference_foot_pos = RewTerm" in source
    assert "reference_active_swing_foot_max = RewTerm" in source
    assert "reference_joint_max = RewTerm" in source
    assert "self.rewards.reference_foot_pos = None" not in source
    assert "ParallelismTrackingFlatEnvCfg" in source
    assert "parallelism_plan_batch_size: int = 1024" in source
    assert "self.scene.robot.init_state.pos = (0.0, 0.0, 0.3)" in source
    assert '"threshold": 1.0' in source
    assert '"threshold": 0.25' in source
    assert '"threshold": 0.8' in source
    assert '"threshold": 0.8,' in source
    assert '"root_pos_threshold": 0.12' in source
    assert '"root_rot_threshold": 0.30' in source
    assert '"joint_max_threshold": 0.8' in source
    assert "weight=1.5" in source
    assert '"std": 0.10' in source
    assert "weight=0.75" in source
    assert '"std": 0.30' in source
    assert 'self.rewards.joint_pos.weight = -0.2' in source
    assert 'self.rewards.feet_air_time.params["threshold"] = 0.20' in source
    assert 'self.rewards.air_time_variance.weight = -0.1' in source
    assert 'self.rewards.action_rate.weight = -0.03' in source


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


def test_tracking_play_cfg_keeps_reference_gravity_disabled() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    reference_source = source[
        source.index("reference_robot: ArticulationCfg")
        : source.index("\n\n\n@configclass", source.index("reference_robot: ArticulationCfg"))
    ]

    assert "disable_gravity=True" in reference_source


def test_tracking_play_cfg_uses_a_distinct_pale_blue_reference_material() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    reference_source = source[
        source.index("reference_robot: ArticulationCfg")
        : source.index("\n\n\n@configclass", source.index("reference_robot: ArticulationCfg"))
    ]
    assert "visual_material=sim_utils.PreviewSurfaceCfg" in reference_source
    assert "diffuse_color=(0.35, 0.72, 1.0)" in reference_source
    assert "func=_spawn_pale_reference_go2" in reference_source
    assert '"/visuals" in child.GetPath().pathString' in source
    assert "child.SetInstanceable(False)" in source
    assert '"/visuals" in path' in source


def test_tracking_play_cfg_disables_timeout_for_unbounded_play() -> None:
    source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    play_source = source[source.index("class ParallelismTrackingFlatEnvCfg_PLAY") :]

    assert "self.terminations.time_out = None" in play_source
