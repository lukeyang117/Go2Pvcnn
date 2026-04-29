"""Viewer-only semantic-course trajectory config."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from extension.mdp.observations import downsampled_height_scan
from extension.semantic_course import (
    DEFAULT_VIEWER_REPRESENTATIVE_STAGE,
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
    spawn_semantic_course_prestartup,
)
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import (
    SEMANTIC_TERRAIN_CFG,
    TeacherElevationTrajectoryEnvCfg_PLAY,
    TeacherElevationTrajectoryObservationsCfg,
    TeacherElevationTrajectoryRewardsCfg,
    TeacherElevationTrajectorySceneCfg,
)
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import EventCfg as BaseEventCfg


@configclass
class TeacherElevationTrajectorySemanticViewerSceneCfg(TeacherElevationTrajectorySceneCfg):
    replicate_physics = False
    height_scanner = None
    semantic_height_scanner = SemanticGridRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=SemanticGridRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground", SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT],
        mesh_semantic_ids={
            "/World/ground": 0,
            SEMANTIC_COURSE_SMALL_ROOT: 1,
            SEMANTIC_COURSE_LARGE_ROOT: 2,
        },
        height_scan_offset=0.5,
    )


@configclass
class TeacherElevationTrajectorySemanticViewerObservationsCfg(TeacherElevationTrajectoryObservationsCfg):
    @configclass
    class PolicyElevationMapCfg(ObsGroup):
        elevation_map = ObsTerm(
            func=downsampled_height_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyStateCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticElevationMapCfg(ObsGroup):
        elevation_map = ObsTerm(
            func=downsampled_height_scan,
            params={"sensor_cfg": SceneEntityCfg("semantic_height_scanner"), "target_size": 16},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_elevation_map: PolicyElevationMapCfg = PolicyElevationMapCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_elevation_map: CriticElevationMapCfg = CriticElevationMapCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class TeacherElevationTrajectorySemanticViewerEventCfg(BaseEventCfg):
    generate_semantic_course = EventTerm(
        func=spawn_semantic_course_prestartup,
        mode="prestartup",
        params={"default_stage": DEFAULT_VIEWER_REPRESENTATIVE_STAGE.value},
    )


@configclass
class TeacherElevationTrajectorySemanticViewerEnvCfg_PLAY(TeacherElevationTrajectoryEnvCfg_PLAY):
    scene: TeacherElevationTrajectorySemanticViewerSceneCfg = TeacherElevationTrajectorySemanticViewerSceneCfg(
        num_envs=32,
        env_spacing=2.5,
        replicate_physics=False,
    )
    observations: TeacherElevationTrajectorySemanticViewerObservationsCfg = (
        TeacherElevationTrajectorySemanticViewerObservationsCfg()
    )
    rewards: TeacherElevationTrajectoryRewardsCfg = TeacherElevationTrajectoryRewardsCfg()
    events: TeacherElevationTrajectorySemanticViewerEventCfg = TeacherElevationTrajectorySemanticViewerEventCfg()

    reference_height_scanner_name: str = "semantic_height_scanner"

    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            tg.num_rows = SEMANTIC_TERRAIN_CFG.num_rows
            tg.num_cols = SEMANTIC_TERRAIN_CFG.num_cols
            tg.curriculum = SEMANTIC_TERRAIN_CFG.curriculum
        self.scene.replicate_physics = False
        self.scene.height_scanner = None
        if self.scene.semantic_height_scanner is not None:
            self.scene.semantic_height_scanner.update_period = self.decimation * self.sim.dt
