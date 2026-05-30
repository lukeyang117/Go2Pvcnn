from __future__ import annotations


def test_mpc_semantic_contact_sensors_real_isaaclab() -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
            SEMANTIC_CONTACT_BODY_NAMES,
            TeacherElevationTrajectoryMpcSemanticEnvCfg,
        )

        cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
        cfg.scene.num_envs = 4
        env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
        env.reset()
        root = env.unwrapped
        for body in SEMANTIC_CONTACT_BODY_NAMES:
            for suffix in ("small", "large"):
                sensor = root.scene.sensors[f"semantic_contact_{body}_{suffix}"]
                matrix = sensor.data.force_matrix_w
                assert matrix.shape[0] == 4
                assert matrix.shape[1] == 1
                assert matrix.shape[-1] == 3
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
