from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_cross_large_complex_config_declares_mixed_terrain_and_counts() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "ParallelismTrackingSmallObstaclesEnvCfg" in source
    assert "parallelism_tracking_cross_large_complex" in source
    assert "flat_dense_small_obstacles" in source
    assert "SemanticObstacleCount(small=0, large=2)" in source
    assert "SemanticObstacleCount(small=40, large=0)" in source
    assert "SemanticObstacleCount(small=5, large=2)" in source
    assert "proportion=0.1" in source
    assert "proportion=0.2" in source


def test_cross_large_complex_config_uses_expected_terrain_proportions() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert source.count("proportion=0.1") >= 6
    assert source.count("proportion=0.2") >= 2


def test_cross_large_complex_config_disables_standstill_termination() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "parallelism_consecutive_standstill" in source
    assert "self.terminations.parallelism_consecutive_standstill = None" in source


def test_cross_large_complex_config_keeps_geometry_collision_reward() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "parallelism_geometry_collision" in source
    reward_source = (ROOT / "tracking/parallelism_small_obstacles_env_cfg.py").read_text()
    assert "parallelism_geometry_collision" in reward_source
    assert "obstacle_semantic_ids" in (ROOT / "extension/parallelism/config.py").read_text()


def test_cross_large_complex_config_excludes_only_dense_flat_from_terrain_curriculum() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "flat_dense_small_obstacles" in source
    assert "excluded_terrain_names" in source
    assert 'excluded_terrain_names": ("flat_dense_small_obstacles",)' in source


def test_cross_large_teacher_tracks_command_and_removes_foot_z_termination() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "ParallelismCrossLargeTeacherObservationsCfg" in source
    assert "isaac_mdp.generated_commands" in source
    assert '"command_name": "base_velocity"' in source
    assert "ParallelismCrossLargeTeacherRewardsCfg" in source
    assert "track_lin_vel_xy_exp" in source
    assert "track_ang_vel_z_exp" in source
    assert "weight=4.5" in source
    assert "weight=2.25" in source
    assert '"std": math.sqrt(0.25)' in source
    assert "self.terminations.parallelism_ref_foot_z_too_far = None" in source


def test_cross_large_teacher_training_script_is_fresh() -> None:
    source = (ROOT / "scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh").read_text()
    assert "parallelism_tracking_cross_large_complex" in source
    assert "--headless" in source
    assert "--max_iterations 2000" in source
    assert "--resume" not in source
    assert "--load_run" not in source
    assert "--load_checkpoint" not in source
    assert "--teacher_checkpoint" in source


def test_cross_large_resume_script_loads_latest_requested_checkpoint() -> None:
    source = (ROOT / "scripts/train_parallelism_large_obstacles_rl_headless_resume.sh").read_text()
    assert "--resume" in source
    assert "--keep_std" in source
    assert "--load_run 2026-08-18_20-30-59/6def073" in source
    assert "--load_checkpoint model_4999.pt" in source
