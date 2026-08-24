from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_pure_ppo_config_does_not_inherit_parallelism():
    source = (ROOT / "tracking/cross_large_complex_ppo_env_cfg.py").read_text()

    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg" in source
    assert "class CrossLargeComplexPpoEnvCfg(TeacherElevationTrajectoryMpcSemanticEnvCfg)" in source
    assert "ParallelismTrackingCrossLargeComplexEnvCfg" not in source
    assert "ParallelismTrackingPlaySceneCfg" not in source
    assert "planner_owned_reference_cache: bool = False" in source
    assert "use_batched_reference_trajectory: bool = False" in source


def test_pure_ppo_observation_contract():
    source = (ROOT / "tracking/cross_large_complex_ppo_env_cfg.py").read_text()

    assert "TeacherElevationTrajectoryMpcSemanticObservationsCfg" in source
    assert "parallelism_ref_" not in source
    assert "parallelism_plan_valid" not in source
    assert "base_lin_vel = None" not in source


def test_pure_ppo_reward_and_termination_contract():
    source = (ROOT / "tracking/cross_large_complex_ppo_env_cfg.py").read_text()

    assert "func=tracking_mdp.policy_geometry_collision_penalty" in source
    assert "weight=-10.0" in source
    assert "active_swing_foot_on_small_obstacle = None" in source
    assert "reference_foot_pos = None" in source
    assert "undesired_contacts = None" in source
    assert "semantic_contact_collision = None" in source
    assert "TeacherElevationTrajectoryMpcSemanticTerminationsCfg" in source
    assert "parallelism_consecutive_standstill" not in source
    assert "parallelism_ref_" not in source


def test_pure_ppo_reuses_mixed_terrain_and_obstacle_counts():
    source = (ROOT / "tracking/cross_large_complex_ppo_env_cfg.py").read_text()

    assert "_cross_large_complex_terrain_cfg" in source
    assert "cross_large_complex_semantic_obstacle_curriculum_cfg" in source
    assert "excluded_terrain_names" in source
    assert '"flat_dense_small_obstacles"' in source
