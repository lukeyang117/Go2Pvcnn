"""Register the active Go2 semantic MPC environments with Gymnasium."""

import gymnasium as gym

from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
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

print("[go2_pvcnn] Registered Go2 semantic MPC environments:")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0")
