"""Planner-free PPO configuration for the mixed cross-large terrain task."""

from __future__ import annotations

from dataclasses import field

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import go2_pvcnn.mdp as go2_mdp
import tracking.mdp as tracking_mdp
from extension.semantic_course import SemanticCourseLayoutCfg, SemanticCourseTerrainImporter
from extension.semantic_curriculum import SemanticObstacleCurriculumCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticCurriculumCfg,
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
    TeacherElevationTrajectoryMpcSemanticObservationsCfg,
    TeacherElevationTrajectoryMpcSemanticRewardsCfg,
    TeacherElevationTrajectoryMpcSemanticSceneCfg,
    TeacherElevationTrajectoryMpcSemanticTerminationsCfg,
)
from tracking.parallelism_cross_large_complex_env_cfg import (
    _cross_large_complex_terrain_cfg,
    cross_large_complex_semantic_obstacle_curriculum_cfg,
)


@configclass
class CrossLargeComplexPpoRewardsCfg(TeacherElevationTrajectoryMpcSemanticRewardsCfg):
    """Locomotion rewards aligned with distillation plus live geometry collision."""

    reference_foot_pos = None
    undesired_contacts = None
    semantic_contact_collision = None
    active_swing_foot_on_small_obstacle = None
    parallelism_geometry_collision = RewTerm(
        func=tracking_mdp.policy_geometry_collision_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
        },
    )


@configclass
class CrossLargeComplexPpoCurriculumCfg(TeacherElevationTrajectoryMpcSemanticCurriculumCfg):
    """Existing terrain curriculum plus Go2Pvcnn linear command curriculum."""

    terrain_levels = CurrTerm(
        func=go2_mdp.terrain_levels_vel_semantic_plane_gate,
        params={
            "cfg_name": "semantic_obstacle_curriculum",
            "excluded_terrain_names": ("flat_dense_small_obstacles",),
        },
    )
    lin_vel_cmd_levels = CurrTerm(go2_mdp.lin_vel_cmd_levels)


@configclass
class CrossLargeComplexPpoEnvCfg(TeacherElevationTrajectoryMpcSemanticEnvCfg):
    """Ordinary ManagerBasedRLEnv config with no planner lifecycle."""

    experiment_name: str = "cross_large_complex_ppo"
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: TeacherElevationTrajectoryMpcSemanticObservationsCfg = (
        TeacherElevationTrajectoryMpcSemanticObservationsCfg()
    )
    rewards: CrossLargeComplexPpoRewardsCfg = CrossLargeComplexPpoRewardsCfg()
    terminations: TeacherElevationTrajectoryMpcSemanticTerminationsCfg = (
        TeacherElevationTrajectoryMpcSemanticTerminationsCfg()
    )
    curriculum: CrossLargeComplexPpoCurriculumCfg = CrossLargeComplexPpoCurriculumCfg()
    planner_owned_reference_cache: bool = False
    use_batched_reference_trajectory: bool = False
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=cross_large_complex_semantic_obstacle_curriculum_cfg
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "cross_large_complex_ppo"
        self.planner_owned_reference_cache = False
        self.use_batched_reference_trajectory = False

        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.3)
        self.events.push_robot = None
        self.scene.terrain.terrain_generator = _cross_large_complex_terrain_cfg()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.scene.terrain.semantic_course_layout_cfg = SemanticCourseLayoutCfg(
            tile_margin_m=0.50,
            center_safety_half_extent_m=0.25,
            center_safety_radius_m=0.30,
            min_spacing_clearance_m=0.08,
        )


@configclass
class CrossLargeComplexPpoEnvCfg_PLAY(CrossLargeComplexPpoEnvCfg):
    """Single-environment play config without the Parallelism reference robot."""

    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = None
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False


__all__ = [
    "CrossLargeComplexPpoEnvCfg",
    "CrossLargeComplexPpoEnvCfg_PLAY",
]
