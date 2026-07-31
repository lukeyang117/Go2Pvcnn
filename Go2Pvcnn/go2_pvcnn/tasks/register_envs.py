"""Register the active Go2 semantic MPC environments with Gymnasium."""

import gymnasium as gym

from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
    TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
    TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY,
    TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
)


gym.register(
    id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationTrajectoryMpcSemanticEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

print("[go2_pvcnn] Registered Go2 semantic MPC environments:")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-v0")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-Play-v0")

try:
    import tracking.register_envs  # noqa: F401
except Exception as exc:  # noqa: BLE001 - tracking registration is optional for static imports.
    print(f"[go2_pvcnn] Tracking env registration skipped: {exc}")
