from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ladder_config_uses_teacher_terrain_and_flat_only_obstacles() -> None:
    source = (ROOT / "tracking/parallelism_ladder_env_cfg.py").read_text()

    assert "SEMANTIC_TERRAIN_CFG" in source
    assert "plane_terrain_names=(\"flat\",)" in source
    assert "SemanticObstacleCount(small=40, large=0)" in source
    assert "non_plane_counts=(SemanticObstacleCount(small=0, large=0),)" in source
    assert "small_obstacle_count: int = 40" in source
    assert "terrain_generator = SEMANTIC_TERRAIN_CFG" in source
    assert "SemanticCourseTerrainImporter" in source


def test_ladder_config_mentions_all_teacher_terrain_names() -> None:
    source = (ROOT / "tracking/parallelism_ladder_env_cfg.py").read_text()

    for name in (
        "flat",
        "random_rough",
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
        "boxes",
        "pyramid_stairs",
        "pyramid_stairs_inv",
    ):
        assert name in source


def test_ladder_entries_are_registered_for_train_and_play() -> None:
    register_source = (ROOT / "tracking/register_envs.py").read_text()
    train_source = (ROOT / "scripts/train.py").read_text()
    play_source = (ROOT / "scripts/play.py").read_text()
    agent_source = (ROOT / "agent/train_cfg.py").read_text()

    assert "ParallelismTrackingLadderEnvCfg" in register_source
    assert "Isaac-Go2-Parallelism-Tracking-Ladder-v0" in register_source
    assert "parallelism_tracking_ladder" in train_source
    assert "ParallelismTrackingLadderEnvCfg" in train_source
    assert "parallelism_tracking_ladder" in play_source
    assert "ParallelismTrackingLadderEnvCfg_PLAY" in play_source
    assert "parallelism_tracking_ladder" in agent_source
