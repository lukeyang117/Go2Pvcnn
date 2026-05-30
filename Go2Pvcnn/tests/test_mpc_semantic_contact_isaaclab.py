from __future__ import annotations


def _semantic_leaf_count(root: str) -> int:
    import isaaclab.sim as sim_utils
    from go2_pvcnn.sensor.semantic_contacter.semantic_global_contact_sensor import filter_semantic_leaf_obstacle_paths

    paths = sim_utils.find_matching_prim_paths(f"{root}/.*/.*/.*")
    return len(filter_semantic_leaf_obstacle_paths(paths, root))


def _run_global_contact_sensor_shape_check(num_envs: int) -> None:
    import gymnasium as gym
    import go2_pvcnn.tasks  # noqa: F401
    from extension.semantic_course import SEMANTIC_COURSE_LARGE_ROOT, SEMANTIC_COURSE_SMALL_ROOT
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        SEMANTIC_CONTACT_BODY_NAMES,
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
    )

    env = None
    try:
        cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
        cfg.scene.num_envs = int(num_envs)
        env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
        env.reset()
        root = env.unwrapped
        expected_small = _semantic_leaf_count(SEMANTIC_COURSE_SMALL_ROOT)
        expected_large = _semantic_leaf_count(SEMANTIC_COURSE_LARGE_ROOT)
        body_count = len(SEMANTIC_CONTACT_BODY_NAMES)

        small = root.scene.sensors["semantic_contact_small"]
        large = root.scene.sensors["semantic_contact_large"]
        small_matrix = small.data.force_matrix_w
        large_matrix = large.data.force_matrix_w

        assert list(small.body_names) == list(SEMANTIC_CONTACT_BODY_NAMES)
        assert list(large.body_names) == list(SEMANTIC_CONTACT_BODY_NAMES)
        assert small_matrix.shape == (num_envs, body_count, expected_small, 3)
        assert large_matrix.shape == (num_envs, body_count, expected_large, 3)
        assert small.contact_physx_view.sensor_count == num_envs * body_count
        assert large.contact_physx_view.sensor_count == num_envs * body_count
        assert small.contact_physx_view.filter_count == expected_small
        assert large.contact_physx_view.filter_count == expected_large
    finally:
        if env is not None:
            env.close()


def test_mpc_semantic_global_contact_sensors_real_isaaclab_small() -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        _run_global_contact_sensor_shape_check(num_envs=4)
    finally:
        simulation_app.close()


def test_mpc_semantic_global_contact_sensors_quantity_alignment_1024() -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        _run_global_contact_sensor_shape_check(num_envs=1024)
    finally:
        simulation_app.close()
