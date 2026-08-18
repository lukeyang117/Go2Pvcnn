"""Teacher-student distillation configuration for the Parallelism mixed-terrain task."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import go2_pvcnn.mdp as go2_mdp
from tracking.parallelism_cross_large_complex_env_cfg import (
    ParallelismTrackingCrossLargeComplexCurriculumCfg,
    ParallelismTrackingCrossLargeComplexEnvCfg,
    ParallelismTrackingCrossLargeComplexEnvCfg_PLAY,
)
from tracking.parallelism_small_obstacles_env_cfg import ParallelismSmallObstaclesRewardsCfg
from tracking.parallelism_tracking_env_cfg import (
    ParallelismTrackingObservationsCfg,
    ParallelismTrackingPlaySceneCfg,
)


@configclass
class ParallelismTrackingDistillationObservationsCfg:
    @configclass
    class TeacherElevationSemanticMapCfg(ParallelismTrackingObservationsCfg.PolicyElevationSemanticMapCfg):
        pass

    @configclass
    class TeacherStateCfg(ParallelismTrackingObservationsCfg.PolicyStateCfg):
        pass

    @configclass
    class StudentElevationSemanticMapCfg(TeacherElevationSemanticMapCfg):
        pass

    @configclass
    class StudentStateCfg(TeacherStateCfg):
        base_lin_vel = None
        velocity_commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        parallelism_ref_joint_pos = None
        parallelism_ref_joint_vel = None
        parallelism_ref_root_pos = None
        parallelism_ref_root_rot = None

    teacher_elevation_semantic_map: TeacherElevationSemanticMapCfg = TeacherElevationSemanticMapCfg()
    teacher_state: TeacherStateCfg = TeacherStateCfg()
    student_elevation_semantic_map: StudentElevationSemanticMapCfg = StudentElevationSemanticMapCfg()
    student_state: StudentStateCfg = StudentStateCfg()


@configclass
class ParallelismDistillationRewardsCfg(ParallelismSmallObstaclesRewardsCfg):
    """Rewards used only by the PPO-distillation student task."""

    track_lin_vel_xy = RewTerm(
        func=go2_mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=go2_mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )


@configclass
class ParallelismTrackingCrossLargeComplexDistillationCurriculumCfg(
    ParallelismTrackingCrossLargeComplexCurriculumCfg
):
    """Go2Pvcnn-style command curriculum for the distillation task."""

    lin_vel_cmd_levels = CurrTerm(go2_mdp.lin_vel_cmd_levels)


@configclass
class ParallelismTrackingCrossLargeComplexDistillationEnvCfg(
    ParallelismTrackingCrossLargeComplexEnvCfg
):
    """Mixed-terrain environment with a reference-privileged teacher and reference-free student."""

    experiment_name: str = "parallelism_tracking_cross_large_complex_distillation"
    observations: ParallelismTrackingDistillationObservationsCfg = (
        ParallelismTrackingDistillationObservationsCfg()
    )
    rewards: ParallelismDistillationRewardsCfg = ParallelismDistillationRewardsCfg()
    curriculum: ParallelismTrackingCrossLargeComplexDistillationCurriculumCfg = (
        ParallelismTrackingCrossLargeComplexDistillationCurriculumCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_cross_large_complex_distillation"
        self.curriculum.parallelism_velocity = None
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)
        self.terminations.parallelism_ref_foot_z_too_far = None
        self.terminations.parallelism_consecutive_standstill = None
        assert self.rewards.parallelism_geometry_collision is not None


@configclass
class ParallelismTrackingCrossLargeComplexDistillationEnvCfg_PLAY(
    ParallelismTrackingCrossLargeComplexDistillationEnvCfg
):
    scene: ParallelismTrackingPlaySceneCfg = ParallelismTrackingPlaySceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = None
        self.curriculum.parallelism_velocity = None
        self.observations.teacher_elevation_semantic_map.enable_corruption = False
        self.observations.teacher_state.enable_corruption = False
        self.observations.student_elevation_semantic_map.enable_corruption = False
        self.observations.student_state.enable_corruption = False
