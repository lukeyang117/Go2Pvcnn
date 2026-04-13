"""Register Go2 PVCNN environment with Gymnasium."""

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from go2_pvcnn.tasks.go2_pvcnn_env_cfg import Go2PvcnnEnvCfg, Go2PvcnnEnvCfg_PLAY
from go2_pvcnn.tasks.go2_pvcnn_test_env_cfg import Go2PvcnnTestEnvCfg
from go2_pvcnn.tasks.go2_himloco_test_cfg import Go2HimlocoTestEnvCfg, Go2HimlocoTestEnvCfg_PLAY
from go2_pvcnn.tasks.go2_abs_test_cfg import Go2AbsTestEnvCfg, Go2AbsTestEnvCfg_PLAY
from go2_pvcnn.tasks.teacher_semantic_env_cfg import TeacherSemanticEnvCfg, TeacherSemanticEnvCfg_PLAY
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import (
    TeacherWithoutSemanticEnvCfg,
    TeacherWithoutSemanticEnvCfg_PLAY,
)
from go2_pvcnn.tasks.teacher_elevation_env_cfg import TeacherElevationEnvCfg, TeacherElevationEnvCfg_PLAY
from go2_pvcnn.tasks.teacher_elevation_semantic_map_env_cfg import (
    TeacherElevationSemanticMapEnvCfg,
    TeacherElevationSemanticMapEnvCfg_PLAY,
)
from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import (
    TeacherElevationTrajectoryEnvCfg,
    TeacherElevationTrajectoryEnvCfg_PLAY,
)

##
# Register Gym environments
##

gym.register(
    id="Go2PvcnnEnv",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2PvcnnEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Go2PvcnnEnv-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2PvcnnEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Go2PvcnnEnv-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2PvcnnTestEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Go2HimlocoEnv-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2HimlocoTestEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Go2AbsEnv-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2AbsTestEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

# Teacher Semantic training environment
gym.register(
    id="Isaac-Teacher-Semantic-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherSemanticEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Semantic-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherSemanticEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

# Teacher without semantic (state-only)
gym.register(
    id="Isaac-Teacher-Without-Semantic-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherWithoutSemanticEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Without-Semantic-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherWithoutSemanticEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

# Teacher elevation (elevation map + CNN)
gym.register(
    id="Isaac-Teacher-Elevation-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Semantic-Map-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationSemanticMapEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Semantic-Map-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationSemanticMapEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Trajectory-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationTrajectoryEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)

gym.register(
    id="Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeacherElevationTrajectoryEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)

print("[go2_pvcnn] Registered Go2 PVCNN environments:")
print("[go2_pvcnn]   - Go2PvcnnEnv (training)")
print("[go2_pvcnn]   - Go2PvcnnEnv-Play (evaluation)")
print("[go2_pvcnn]   - Go2PvcnnEnv-Test (collision testing)")
print("[go2_pvcnn]   - Go2HimlocoEnv-Test (HIMLoco testing)")
print("[go2_pvcnn]   - Go2AbsEnv-Test (ABS testing)")
print("[go2_pvcnn]   - Isaac-Teacher-Semantic-Go2-v0 (teacher training)")
print("[go2_pvcnn]   - Isaac-Teacher-Semantic-Go2-Play-v0 (teacher evaluation)")
print("[go2_pvcnn]   - Isaac-Teacher-Without-Semantic-Go2-v0 (state-only ablation)")
print("[go2_pvcnn]   - Isaac-Teacher-Without-Semantic-Go2-Play-v0 (state-only play)")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Go2-v0 (elevation map CNN)")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Go2-Play-v0 (elevation map play)")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Semantic-Map-Go2-v0 (elevation + semantic grid CNN)")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Semantic-Map-Go2-Play-v0 (elevation + semantic play)")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Trajectory-Go2-v0 (high-res elevation + trajectory reward)")
print("[go2_pvcnn]   - Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0 (trajectory-guided play)")
