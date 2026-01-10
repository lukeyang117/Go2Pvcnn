"""Register Go2 PVCNN environment with Gymnasium."""

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from go2_pvcnn.tasks.go2_pvcnn_env_cfg import Go2PvcnnEnvCfg, Go2PvcnnEnvCfg_PLAY
from go2_pvcnn.tasks.go2_pvcnn_test_env_cfg import Go2PvcnnTestEnvCfg
from go2_pvcnn.tasks.go2_himloco_test_cfg import Go2HimlocoTestEnvCfg, Go2HimlocoTestEnvCfg_PLAY
from go2_pvcnn.tasks.go2_abs_test_cfg import Go2AbsTestEnvCfg, Go2AbsTestEnvCfg_PLAY

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

print("[go2_pvcnn] Registered Go2 PVCNN environments:")
print("[go2_pvcnn]   - Go2PvcnnEnv (training)")
print("[go2_pvcnn]   - Go2PvcnnEnv-Play (evaluation)")
print("[go2_pvcnn]   - Go2PvcnnEnv-Test (collision testing)")
print("[go2_pvcnn]   - Go2HimlocoEnv-Test (HIMLoco testing)")
print("[go2_pvcnn]   - Go2AbsEnv-Test (ABS testing)")
