"""Gymnasium registration for parallelism tracking tasks."""

from __future__ import annotations

import gymnasium as gym

from tracking.env import ParallelismTrackingEnv
from tracking.parallelism_tracking_env_cfg import ParallelismTrackingFlatEnvCfg


gym.register(
    id="Isaac-Go2-Parallelism-Tracking-Flat-v0",
    entry_point="tracking.env:ParallelismTrackingEnv",
    kwargs={
        "env_cfg_entry_point": ParallelismTrackingFlatEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
    disable_env_checker=True,
)

print("[tracking] Registered parallelism tracking environments:")
print("[tracking]   - Isaac-Go2-Parallelism-Tracking-Flat-v0")
