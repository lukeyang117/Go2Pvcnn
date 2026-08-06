"""Parallelism RL configuration for one fixed small-obstacle subterrain."""

from __future__ import annotations

from dataclasses import replace
from dataclasses import field

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from extension.semantic_course import SemanticCourseLayoutCfg, SemanticCourseTerrainImporter
from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticSceneCfg,
)
from tracking.parallelism_small_obstacles_scene import (
    ParallelismSmallObstacleSceneCfg,
    build_small_obstacle_local_xy,
    small_obstacles_terrain_cfg,
)
from tracking.parallelism_tracking_env_cfg import (
    ParallelismTrackingFlatEnvCfg,
    ParallelismTrackingFlatEnvCfg_PLAY,
    ParallelismTrackingPlaySceneCfg,
    ParallelismTrackingRewardsCfg,
)
import tracking.mdp as tracking_mdp


@configclass
class ParallelismSmallObstaclesRewardsCfg(ParallelismTrackingRewardsCfg):
    parallelism_geometry_collision = RewTerm(
        func=tracking_mdp.parallelism_geometry_collision_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "scanner_cfg": SceneEntityCfg("semantic_height_scanner")},
    )
    active_swing_foot_on_small_obstacle = RewTerm(
        func=tracking_mdp.active_swing_foot_on_small_obstacle_reward,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), "scanner_cfg": SceneEntityCfg("semantic_height_scanner")},
    )


@configclass
class ParallelismTrackingSmallObstaclesEnvCfg(ParallelismTrackingFlatEnvCfg):
    """One fixed small-obstacle subterrain with random all-direction commands."""

    experiment_name: str = "parallelism_tracking_small_obstacles"
    small_obstacle_count: int = 40
    rewards: ParallelismSmallObstaclesRewardsCfg = ParallelismSmallObstaclesRewardsCfg()
    small_obstacle_scene: ParallelismSmallObstacleSceneCfg = field(
        default_factory=ParallelismSmallObstacleSceneCfg
    )
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=lambda: SemanticObstacleCurriculumCfg(
            enabled=True,
            plane_terrain_names=("small_obstacles",),
            plane_counts=(SemanticObstacleCount(small=24, large=0),),
            non_plane_counts=(SemanticObstacleCount(small=0, large=0),),
            center_safety_half_extent_m=(0.25,),
            min_spacing_clearance_m=(0.18,),
            tile_margin_m=(0.50,),
            collision_force_threshold=1.0,
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_small_obstacles"
        self.small_obstacle_scene = replace(
            self.small_obstacle_scene,
            small_obstacle_count=int(self.small_obstacle_count),
        )
        self.semantic_obstacle_curriculum.plane_counts = (
            SemanticObstacleCount(small=self.small_obstacle_count, large=0),
        )
        self.scene.terrain.terrain_generator = small_obstacles_terrain_cfg(self.small_obstacle_scene)
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.scene.terrain.semantic_course_layout_cfg = SemanticCourseLayoutCfg(
            tile_margin_m=0.50,
            center_safety_half_extent_m=self.small_obstacle_scene.reset_clear_radius_m,
            center_safety_radius_m=self.small_obstacle_scene.obstacle_center_exclusion_radius_m,
            fixed_small_obstacle_local_xy=build_small_obstacle_local_xy(self.small_obstacle_scene),
            min_spacing_clearance_m=self.small_obstacle_scene.small_obstacle_min_spacing_m,
        )
        self.curriculum.parallelism_velocity.params["root_pos_threshold"] = 0.18
        self.curriculum.parallelism_velocity.params["root_rot_threshold"] = 0.45
        self.curriculum.parallelism_velocity.params["joint_mean_threshold"] = 0.32
        self.curriculum.parallelism_velocity.params["joint_max_threshold"] = 1.0
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class ParallelismTrackingSmallObstaclesEnvCfg_PLAY(ParallelismTrackingSmallObstaclesEnvCfg):
    scene: ParallelismTrackingPlaySceneCfg = ParallelismTrackingPlaySceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = None
        self.curriculum.parallelism_velocity = None
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
