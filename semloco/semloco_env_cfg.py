"""Independent SemLoco environment configuration."""

from __future__ import annotations

from dataclasses import field

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from tracking.cross_large_complex_ppo_env_cfg import (
    CrossLargeComplexPpoEnvCfg,
    CrossLargeComplexPpoEnvCfg_PLAY,
    CrossLargeComplexPpoRewardsCfg,
)
from .semloco_rewards import semantic_foothold_tracking_reward, semloco_clearance_penalty


def _foothold_reward(env):
    cache = getattr(env, "_semloco_cache", None)
    robot = env.scene["robot"]
    if cache is None:
        return robot.data.root_pos_w[:, 0] * 0.0
    feet = robot.data.body_pos_w.index_select(1, env._semloco_foot_ids)
    return semantic_foothold_tracking_reward(feet, cache.target, cache.swing, cache.valid)


def _clearance_reward(env):
    cache = getattr(env, "_semloco_cache", None)
    robot = env.scene["robot"]
    foot_ids = getattr(
        env,
        "_semloco_foot_ids",
        torch.arange(robot.data.body_pos_w.shape[1] - 4, robot.data.body_pos_w.shape[1], device=robot.data.body_pos_w.device),
    )
    feet = robot.data.body_pos_w.index_select(1, foot_ids)
    phase = getattr(env, "_semloco_swing_phase", torch.zeros(feet.shape[0], 4, device=feet.device))
    mask = cache.swing if cache is not None else torch.zeros_like(phase, dtype=torch.bool)
    return semloco_clearance_penalty(feet, phase, mask)


@configclass
class SemlocoCrossLargeComplexRewardsCfg(CrossLargeComplexPpoRewardsCfg):
    semantic_foothold_tracking = RewTerm(func=_foothold_reward, weight=0.5)
    semloco_clearance = RewTerm(func=_clearance_reward, weight=1.0)


@configclass
class SemlocoCrossLargeComplexEnvCfg(CrossLargeComplexPpoEnvCfg):
    experiment_name: str = "cross_large_complex_semloco"
    rewards: SemlocoCrossLargeComplexRewardsCfg = SemlocoCrossLargeComplexRewardsCfg()
    semloco_search_size: int = 5
    semloco_search_resolution: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "cross_large_complex_semloco"


@configclass
class SemlocoCrossLargeComplexEnvCfg_PLAY(SemlocoCrossLargeComplexEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


__all__ = ["SemlocoCrossLargeComplexEnvCfg", "SemlocoCrossLargeComplexEnvCfg_PLAY"]
