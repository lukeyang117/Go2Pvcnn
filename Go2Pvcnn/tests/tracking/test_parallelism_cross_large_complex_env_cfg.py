from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_cross_large_complex_config_declares_mixed_terrain_and_counts() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "ParallelismTrackingSmallObstaclesEnvCfg" in source
    assert "parallelism_tracking_cross_large_complex" in source
    assert "flat_dense_small_obstacles" in source
    assert "SemanticObstacleCount(small=40, large=0)" in source
    assert "SemanticObstacleCount(small=5, large=2)" in source
    assert "proportion=0.0625" in source


def test_cross_large_complex_config_keeps_standstill_termination() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_env_cfg.py").read_text()
    assert "parallelism_consecutive_standstill" in source
    assert 'params["threshold"] == 2' in source


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
