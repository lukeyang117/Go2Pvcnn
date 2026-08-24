"""Mixed-terrain Parallelism RL configuration for small-cross and large-avoidance."""

from __future__ import annotations

from dataclasses import field
import math

import isaaclab.terrains as terrain_gen
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from extension.semantic_course import SemanticCourseLayoutCfg, SemanticCourseTerrainImporter
from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import SEMANTIC_TERRAIN_CFG
import go2_pvcnn.mdp as go2_mdp
import tracking.mdp as tracking_mdp
from tracking.parallelism_small_obstacles_env_cfg import (
    ParallelismSmallObstaclesRewardsCfg,
    ParallelismTrackingSmallObstaclesEnvCfg,
)
from tracking.parallelism_tracking_env_cfg import (
    ParallelismTrackingCurriculumCfg,
    ParallelismTrackingObservationsCfg,
    ParallelismTrackingPlaySceneCfg,
)


@configclass
class ParallelismCrossLargeTeacherObservationsCfg(ParallelismTrackingObservationsCfg):
    """Privileged teacher observations with the command made explicit."""

    @configclass
    class PolicyStateCfg(ParallelismTrackingObservationsCfg.PolicyStateCfg):
        velocity_commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        parallelism_plan_valid = ObsTerm(func=tracking_mdp.parallelism_plan_valid)

    @configclass
    class CriticStateCfg(ParallelismTrackingObservationsCfg.CriticStateCfg):
        velocity_commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        parallelism_plan_valid = ObsTerm(func=tracking_mdp.parallelism_plan_valid)

    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class ParallelismCrossLargeTeacherRewardsCfg(ParallelismSmallObstaclesRewardsCfg):
    """Large-terrain teacher rewards with explicit command tracking."""

    parallelism_geometry_collision = RewTerm(
        func=tracking_mdp.parallelism_geometry_collision_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
        },
    )
    active_swing_foot_on_small_obstacle = RewTerm(
        func=tracking_mdp.active_swing_foot_on_small_obstacle_reward,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
        },
    )

    track_lin_vel_xy = RewTerm(
        func=go2_mdp.track_lin_vel_xy_exp,
        weight=4.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=go2_mdp.track_ang_vel_z_exp,
        weight=2.25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
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
            # With num_cols=20, each 0.05 family occupies exactly one column.
            "flat_dense_small_obstacles": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1,
                noise_range=(0.01, 0.06),
                noise_step=0.01,
                border_width=0.25,
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2,
                grid_width=0.45,
                grid_height_range=(0.05, 0.2),
                platform_width=2.0,
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        },
    )


def cross_large_complex_semantic_obstacle_curriculum_cfg() -> SemanticObstacleCurriculumCfg:
    """Return the shared semantic obstacle layout for cross-large experiments."""

    return SemanticObstacleCurriculumCfg(
        enabled=True,
        plane_terrain_names=("flat_dense_small_obstacles", "flat"),
        plane_counts=(SemanticObstacleCount(small=0, large=2),),
        non_plane_counts=(SemanticObstacleCount(small=5, large=2),),
        terrain_obstacle_count_overrides={
            "flat": SemanticObstacleCount(small=0, large=2),
            "flat_dense_small_obstacles": SemanticObstacleCount(small=40, large=0),
        },
        center_safety_half_extent_m=(0.25,),
        min_spacing_clearance_m=(0.08,),
        tile_margin_m=(0.50,),
        collision_force_threshold=1.0,
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
    observations: ParallelismCrossLargeTeacherObservationsCfg = (
        ParallelismCrossLargeTeacherObservationsCfg()
    )
    rewards: ParallelismCrossLargeTeacherRewardsCfg = ParallelismCrossLargeTeacherRewardsCfg()
    dense_small_obstacle_count: int = 40
    normal_small_obstacle_count: int = 5
    normal_large_obstacle_count: int = 2
    curriculum: ParallelismTrackingCrossLargeComplexCurriculumCfg = (
        ParallelismTrackingCrossLargeComplexCurriculumCfg()
    )
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=cross_large_complex_semantic_obstacle_curriculum_cfg
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_cross_large_complex"
        # Keep the sampled command stable while Parallelism independently
        # replans on reset, command changes, or every 23 control steps.
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        # Foot-height tracking remains a reward/metric, but no longer ends
        # the large-terrain teacher episode by itself.
        self.terminations.parallelism_ref_foot_z_too_far = None
        self.terminations.parallelism_consecutive_standstill = None
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
        # Keep the geometry collision reward active while allowing a failed
        # planner cycle to continue without resetting the environment.
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
