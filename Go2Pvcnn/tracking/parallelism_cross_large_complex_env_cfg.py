"""Mixed-terrain Parallelism RL configuration for small-cross and large-avoidance."""

from __future__ import annotations

from dataclasses import field

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from extension.semantic_course import SemanticCourseLayoutCfg, SemanticCourseTerrainImporter
from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import SEMANTIC_TERRAIN_CFG
import go2_pvcnn.mdp as go2_mdp
from tracking.parallelism_small_obstacles_env_cfg import (
    ParallelismTrackingSmallObstaclesEnvCfg,
)
from tracking.parallelism_tracking_env_cfg import (
    ParallelismTrackingCurriculumCfg,
    ParallelismTrackingPlaySceneCfg,
)


def _cross_large_complex_terrain_cfg() -> terrain_gen.TerrainGeneratorCfg:
    """Build the shared mixed terrain set; obstacles are spawned by SemanticCourseTerrainImporter."""

    return terrain_gen.TerrainGeneratorCfg(
        size=SEMANTIC_TERRAIN_CFG.size,
        border_width=SEMANTIC_TERRAIN_CFG.border_width,
        num_rows=SEMANTIC_TERRAIN_CFG.num_rows,
        num_cols=SEMANTIC_TERRAIN_CFG.num_cols,
        horizontal_scale=SEMANTIC_TERRAIN_CFG.horizontal_scale,
        vertical_scale=SEMANTIC_TERRAIN_CFG.vertical_scale,
        slope_threshold=SEMANTIC_TERRAIN_CFG.slope_threshold,
        difficulty_range=SEMANTIC_TERRAIN_CFG.difficulty_range,
        curriculum=True,
        sub_terrains={
            "flat_dense_small_obstacles": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0625),
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0375),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.10,
                noise_range=(0.01, 0.06),
                noise_step=0.01,
                border_width=0.25,
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.0, 0.4),
                platform_width=1.0,
                border_width=0.25,
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.0, 0.4),
                platform_width=1.0,
                border_width=0.25,
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.20,
                grid_width=0.45,
                grid_height_range=(0.05, 0.2),
                platform_width=2.0,
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.20,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=1.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.20,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=1.0,
                border_width=1.0,
                holes=False,
            ),
        },
    )


@configclass
class ParallelismTrackingCrossLargeComplexCurriculumCfg(ParallelismTrackingCurriculumCfg):
    terrain_levels = CurrTerm(
        func=go2_mdp.terrain_levels_vel_semantic_plane_gate,
        params={
            "cfg_name": "semantic_obstacle_curriculum",
            "excluded_terrain_names": ("flat_dense_small_obstacles",),
        },
    )


@configclass
class ParallelismTrackingCrossLargeComplexEnvCfg(ParallelismTrackingSmallObstaclesEnvCfg):
    """Mixed terrain task with dense flat small-obstacle crossing."""

    experiment_name: str = "parallelism_tracking_cross_large_complex"
    dense_small_obstacle_count: int = 40
    normal_small_obstacle_count: int = 5
    normal_large_obstacle_count: int = 2
    curriculum: ParallelismTrackingCrossLargeComplexCurriculumCfg = (
        ParallelismTrackingCrossLargeComplexCurriculumCfg()
    )
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=lambda: SemanticObstacleCurriculumCfg(
            enabled=True,
            plane_terrain_names=("flat_dense_small_obstacles", "flat"),
            plane_counts=(SemanticObstacleCount(small=5, large=2),),
            non_plane_counts=(SemanticObstacleCount(small=5, large=2),),
            terrain_obstacle_count_overrides={
                "flat_dense_small_obstacles": SemanticObstacleCount(small=40, large=0),
            },
            center_safety_half_extent_m=(0.25,),
            min_spacing_clearance_m=(0.08,),
            tile_margin_m=(0.50,),
            collision_force_threshold=1.0,
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_cross_large_complex"
        self.scene.terrain.terrain_generator = _cross_large_complex_terrain_cfg()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.scene.terrain.semantic_course_layout_cfg = SemanticCourseLayoutCfg(
            tile_margin_m=0.50,
            center_safety_half_extent_m=self.reset_clear_radius_m,
            center_safety_radius_m=self.obstacle_center_exclusion_radius_m,
            min_spacing_clearance_m=self.inner_obstacle_min_spacing_m,
        )
        # These inherited terms are intentionally kept active for the new task.
        assert self.terminations.parallelism_consecutive_standstill is not None
        assert self.terminations.parallelism_consecutive_standstill.params["threshold"] == 2
        assert self.rewards.parallelism_geometry_collision is not None


@configclass
class ParallelismTrackingCrossLargeComplexEnvCfg_PLAY(ParallelismTrackingCrossLargeComplexEnvCfg):
    """Play configuration for the mixed-terrain Parallelism task."""

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
