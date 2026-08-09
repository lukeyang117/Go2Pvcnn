"""Parallelism RL config for teacher terrain with flat-only small obstacles."""

from __future__ import annotations

from dataclasses import field, replace

from isaaclab.utils import configclass

from extension.semantic_course import SemanticCourseLayoutCfg, SemanticCourseTerrainImporter
from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import SEMANTIC_TERRAIN_CFG
from tracking.parallelism_small_obstacles_scene import build_small_obstacle_local_xy
from tracking.parallelism_small_obstacles_env_cfg import (
    ParallelismTrackingSmallObstaclesEnvCfg,
)
from tracking.parallelism_tracking_env_cfg import ParallelismTrackingPlaySceneCfg


_TEACHER_TERRAIN_NAMES = (
    "flat",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "boxes",
    "pyramid_stairs",
    "pyramid_stairs_inv",
)


@configclass
class ParallelismTrackingLadderEnvCfg(ParallelismTrackingSmallObstaclesEnvCfg):
    """Teacher terrain set with 40 semantic small obstacles only on flat tiles."""

    experiment_name: str = "parallelism_tracking_ladder"
    small_obstacle_count: int = 40
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=lambda: SemanticObstacleCurriculumCfg(
            enabled=True,
            plane_terrain_names=("flat",),
            plane_counts=(SemanticObstacleCount(small=40, large=0),),
            non_plane_counts=(SemanticObstacleCount(small=0, large=0),),
            center_safety_half_extent_m=(0.25,),
            min_spacing_clearance_m=(0.08,),
            tile_margin_m=(0.50,),
            collision_force_threshold=1.0,
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_ladder"
        self.small_obstacle_scene = replace(
            self.small_obstacle_scene,
            small_obstacle_count=int(self.small_obstacle_count),
            small_obstacle_min_spacing_m=float(self.inner_obstacle_min_spacing_m),
            inner_obstacle_min_spacing_m=float(self.inner_obstacle_min_spacing_m),
            outer_obstacle_min_spacing_m=float(self.outer_obstacle_min_spacing_m),
            small_obstacle_jitter_m=float(self.small_obstacle_jitter_m),
        )
        self.semantic_obstacle_curriculum.plane_terrain_names = ("flat",)
        self.semantic_obstacle_curriculum.plane_counts = (
            SemanticObstacleCount(small=int(self.small_obstacle_count), large=0),
        )
        self.semantic_obstacle_curriculum.non_plane_counts = (
            SemanticObstacleCount(small=0, large=0),
        )
        self.semantic_obstacle_curriculum.center_safety_half_extent_m = (float(self.reset_clear_radius_m),)
        self.semantic_obstacle_curriculum.min_spacing_clearance_m = (float(self.inner_obstacle_min_spacing_m),)
        self.scene.terrain.terrain_generator = SEMANTIC_TERRAIN_CFG
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.scene.terrain.semantic_course_layout_cfg = SemanticCourseLayoutCfg(
            tile_margin_m=0.50,
            center_safety_half_extent_m=self.small_obstacle_scene.reset_clear_radius_m,
            center_safety_radius_m=self.small_obstacle_scene.obstacle_center_exclusion_radius_m,
            fixed_small_obstacle_local_xy=build_small_obstacle_local_xy(self.small_obstacle_scene),
            min_spacing_clearance_m=self.small_obstacle_scene.small_obstacle_min_spacing_m,
        )


@configclass
class ParallelismTrackingLadderEnvCfg_PLAY(ParallelismTrackingLadderEnvCfg):
    """Play config for the terrain-aware Parallelism task."""

    scene: ParallelismTrackingPlaySceneCfg = ParallelismTrackingPlaySceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_ladder"
        self.terminations.time_out = None
        self.curriculum.parallelism_velocity = None
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
