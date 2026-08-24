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


def test_pure_ppo_experiment_is_registered_as_normal_manager_env():
    register = (ROOT / "tracking/register_envs.py").read_text()
    train = (ROOT / "scripts/train.py").read_text()

    assert 'id="Isaac-Go2-Cross-Large-Complex-PPO-v0"' in register
    assert 'entry_point="isaaclab.envs:ManagerBasedRLEnv"' in register
    assert '"cross_large_complex_ppo"' in train


def test_pure_ppo_play_entrypoint_has_no_parallelism_path():
    play = (ROOT / "scripts/play.py").read_text()

    assert '"cross_large_complex_ppo"' in play
    assert "CrossLargeComplexPpoEnvCfg_PLAY" in play
    assert '"Isaac-Go2-Cross-Large-Complex-PPO-v0"' in play

    parallelism_block = play.split("is_parallelism_play =", 1)[1].split("parallelism_panel_state", 1)[0]
    assert '"cross_large_complex_ppo"' not in parallelism_block


def test_pure_ppo_launcher_is_headless_and_planner_free():
    source = (ROOT / "scripts/train_cross_large_complex_ppo_headless.sh").read_text()

    assert "--experiment cross_large_complex_ppo" in source
    assert "--headless" in source
    assert "--max_iterations" in source
    assert "--teacher_checkpoint" not in source
    assert "planner" not in source.lower()


def test_pure_ppo_runner_has_no_distillation_fields():
    from agent.train_cfg import get_train_cfg

    cfg = get_train_cfg("cross_large_complex_ppo")

    assert cfg["algorithm"]["class_name"] == "PPO"
    assert cfg["policy"]["class_name"] == "ActorCriticCNN"
    assert cfg["algorithm"]["learning_rate"] == 3e-4
    assert cfg["algorithm"]["schedule"] == "fixed"
    assert cfg["algorithm"]["entropy_coef"] == 0.01
    assert cfg["policy"]["init_noise_std"] == 1.0
    assert cfg["obs_groups"] == {
        "policy": ["policy_elevation_semantic_map", "policy_state"],
        "critic": ["critic_elevation_semantic_map", "critic_state"],
    }
    serialized = repr(cfg)
    assert "teacher_coef" not in serialized
    assert "teacher_ratio" not in serialized
    assert "HybridDistillationPPO" not in serialized
