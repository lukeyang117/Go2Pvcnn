"""M1 curriculum adapters for the unchanged Go2 terrain schedule."""

from __future__ import annotations

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import go2_pvcnn.mdp as go2_mdp
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticCurriculumCfg,
)


def m1_terrain_levels_vel_semantic_plane_gate(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply the Go2 terrain-level distance rule with M1-compatible bodies."""

    device = torch.device(getattr(env, "device", "cpu"))
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(
        asset.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2],
        dim=1,
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down = torch.logical_and(move_down, torch.logical_not(move_up))
    terrain.update_env_origins(ids, move_up, move_down)
    return {"mean_terrain_level": torch.mean(terrain.terrain_levels.float())}


@configclass
class M1AmeCurriculumCfg(TeacherElevationTrajectoryMpcSemanticCurriculumCfg):
    """Keep Go2 curriculum terms except the Go2 foot-name semantic probe."""

    terrain_levels = CurrTerm(func=m1_terrain_levels_vel_semantic_plane_gate)
    lin_vel_cmd_levels = CurrTerm(func=go2_mdp.lin_vel_cmd_levels)


__all__ = ["M1AmeCurriculumCfg", "m1_terrain_levels_vel_semantic_plane_gate"]
