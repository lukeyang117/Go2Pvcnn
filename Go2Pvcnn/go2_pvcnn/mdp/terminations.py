"""Termination functions for Go2 PVCNN locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot's orientation is too far from upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    # compute projected gravity
    projected_gravity = asset.data.projected_gravity_b
    # check if z-component is below threshold (robot is tilted)
    return torch.abs(projected_gravity[:, 2]) < torch.cos(torch.tensor(limit_angle))


def base_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot's base is too low."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height
