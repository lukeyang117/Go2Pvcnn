"""Teacher-student distillation configuration for the Parallelism mixed-terrain task."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from extension.mdp.observations import downsampled_elevation_semantic_scan
import tracking.mdp as tracking_mdp
from tracking.parallelism_cross_large_complex_env_cfg import (
    ParallelismTrackingCrossLargeComplexEnvCfg,
    ParallelismTrackingCrossLargeComplexEnvCfg_PLAY,
)
from tracking.parallelism_tracking_env_cfg import ParallelismTrackingPlaySceneCfg


@configclass
class ParallelismTrackingDistillationObservationsCfg:
    @configclass
    class TeacherElevationSemanticMapCfg(ObsGroup):
        elevation_semantic_map = ObsTerm(
            func=downsampled_elevation_semantic_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=isaac_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_commands = None
        actions = ObsTerm(func=isaac_mdp.last_action)
        parallelism_ref_joint_pos = ObsTerm(func=tracking_mdp.parallelism_ref_joint_pos_rel_t)
        parallelism_ref_joint_vel = ObsTerm(func=tracking_mdp.parallelism_ref_joint_vel_t)
        parallelism_ref_root_pos = ObsTerm(func=tracking_mdp.parallelism_ref_root_pos_b_t)
        parallelism_ref_root_rot = ObsTerm(func=tracking_mdp.parallelism_ref_root_rot_b_t)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StudentElevationSemanticMapCfg(TeacherElevationSemanticMapCfg):
        pass

    @configclass
    class StudentStateCfg(TeacherStateCfg):
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
class ParallelismTrackingCrossLargeComplexDistillationEnvCfg(
    ParallelismTrackingCrossLargeComplexEnvCfg
):
    """Mixed-terrain environment with a reference-privileged teacher and reference-free student."""

    experiment_name: str = "parallelism_tracking_cross_large_complex_distillation"
    observations: ParallelismTrackingDistillationObservationsCfg = (
        ParallelismTrackingDistillationObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parallelism_tracking_cross_large_complex_distillation"
        assert self.terminations.parallelism_consecutive_standstill is not None
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
