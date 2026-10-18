"""M1 curriculum adapters for the unchanged Go2 terrain schedule."""

from __future__ import annotations

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
    cfg_name: str = "semantic_obstacle_curriculum",
    excluded_terrain_names: tuple[str, ...] = (),
):
    """Apply Go2's semantic curriculum with M1 body-name selectors."""

    return go2_mdp.terrain_levels_vel_semantic_plane_gate(
        env,
        env_ids,
        asset_cfg=asset_cfg,
        cfg_name=cfg_name,
        excluded_terrain_names=excluded_terrain_names,
        body_name_patterns={
            "foot": ".*_FOOT_LINK",
            "calf": ".*_KNEE_LINK",
            "thigh": ".*_HIP_LINK",
        },
    )


@configclass
class M1AmeCurriculumCfg(TeacherElevationTrajectoryMpcSemanticCurriculumCfg):
    """Keep Go2 curriculum terms except the Go2 foot-name semantic probe."""

    terrain_levels = CurrTerm(
        func=m1_terrain_levels_vel_semantic_plane_gate,
        params={
            "cfg_name": "semantic_obstacle_curriculum",
            "excluded_terrain_names": ("flat_dense_small_obstacles",),
        },
    )
    lin_vel_cmd_levels = CurrTerm(func=go2_mdp.lin_vel_cmd_levels)


__all__ = ["M1AmeCurriculumCfg", "m1_terrain_levels_vel_semantic_plane_gate"]
