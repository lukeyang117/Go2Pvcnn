from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_small_obstacle_config_is_one_fixed_subterrain() -> None:
    source = (ROOT / "tracking/parallelism_small_obstacles_env_cfg.py").read_text()
    scene_source = (ROOT / "tracking/parallelism_small_obstacles_scene.py").read_text()
    assert "class ParallelismTrackingSmallObstaclesEnvCfg" in source
    assert "ParallelismTrackingFlatEnvCfg" in source
    assert "parallelism_tracking_small_obstacles" in source
    assert "small_obstacles_terrain_cfg" in source
    assert "num_rows=1" in scene_source
    assert "num_cols=1" in scene_source
    assert "small_obstacle_count: int = 24" in scene_source
    assert "reset_clear_radius_m: float = 0.25" in scene_source
    assert "obstacle_center_exclusion_radius_m: float = 0.30" in scene_source
    assert "semantic_contact_collision = None" not in source


def test_small_obstacle_config_keeps_velocity_curriculum_and_relaxes_thresholds() -> None:
    source = (ROOT / "tracking/parallelism_small_obstacles_env_cfg.py").read_text()
    flat_source = (ROOT / "tracking/parallelism_tracking_env_cfg.py").read_text()
    assert "resampling_time_range=(0.48, 0.48)" in flat_source
    assert "lin_vel_x=(-1.0, 1.0)" in flat_source
    assert "lin_vel_y=(-0.5, 0.5)" in flat_source
    assert "ang_vel_z=(-1.0, 1.0)" in flat_source
    assert 'params["root_pos_threshold"] = 0.18' in source
    assert 'params["root_rot_threshold"] = 0.45' in source
    assert 'params["joint_mean_threshold"] = 0.32' in source
    assert 'params["joint_max_threshold"] = 1.0' in source
