"""Gymnasium registration for parallelism tracking tasks."""

from __future__ import annotations

import gymnasium as gym

from tracking.env import ParallelismTrackingEnv
from tracking.parallelism_ladder_env_cfg import ParallelismTrackingLadderEnvCfg
from tracking.parallelism_cross_large_complex_env_cfg import ParallelismTrackingCrossLargeComplexEnvCfg
from tracking.parallelism_cross_large_complex_distillation_env_cfg import (
    ParallelismTrackingCrossLargeComplexDistillationEnvCfg,
)
from tracking.cross_large_complex_ppo_env_cfg import CrossLargeComplexPpoEnvCfg
from tracking.parallelism_small_obstacles_env_cfg import ParallelismTrackingSmallObstaclesEnvCfg
from tracking.parallelism_tracking_env_cfg import ParallelismTrackingFlatEnvCfg
from tracking.parallelism_amp_cross_large_complex_env_cfg import ParallelismAmpCrossLargeComplexEnvCfg


gym.register(
    id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AMP-v0",
    entry_point="tracking.amp_env:ParallelismAmpEnv",
    kwargs={
        "env_cfg_entry_point": ParallelismAmpCrossLargeComplexEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Go2-Parallelism-Tracking-Flat-v0",
    entry_point="tracking.env:ParallelismTrackingEnv",
    kwargs={
        "env_cfg_entry_point": ParallelismTrackingFlatEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-v0",
    entry_point="tracking.env:ParallelismTrackingEnv",
    kwargs={
        "env_cfg_entry_point": ParallelismTrackingCrossLargeComplexEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-Distillation-v0",
    entry_point="tracking.env:ParallelismTrackingEnv",
    kwargs={
        "env_cfg_entry_point": ParallelismTrackingCrossLargeComplexDistillationEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Go2-Cross-Large-Complex-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": CrossLargeComplexPpoEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Go2-Parallelism-Tracking-Small-Obstacles-v0",
    entry_point="tracking.env:ParallelismTrackingEnv",
    kwargs={
        "env_cfg_entry_point": ParallelismTrackingSmallObstaclesEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Go2-Parallelism-Tracking-Ladder-v0",
    entry_point="tracking.env:ParallelismTrackingEnv",
    kwargs={
        "env_cfg_entry_point": ParallelismTrackingLadderEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)

print("[tracking] Registered parallelism tracking environments:")
print("[tracking]   - Isaac-Go2-Parallelism-Tracking-Flat-v0")
print("[tracking]   - Isaac-Go2-Parallelism-Tracking-Small-Obstacles-v0")
print("[tracking]   - Isaac-Go2-Parallelism-Tracking-Ladder-v0")
print("[tracking]   - Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-v0")
print("[tracking]   - Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-Distillation-v0")
print("[tracking]   - Isaac-Go2-Cross-Large-Complex-PPO-v0")
print("[tracking]   - Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AMP-v0")
print("[tracking]   - parallelism_tracking_cross_large_complex_distillation")
print("[tracking]   - parallelism_tracking_cross_large_complex")
