"""Independent MPC semantic trajectory teacher config."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from extension.mdp.observations import downsampled_elevation_semantic_scan
from extension.mdp.rewards_reference import reference_foot_pos_reward
from extension.mdp.rewards_reference import swing_leg_collision_reward
from extension.semantic_course import (
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
    SemanticCourseTerrainImporter,
)
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import (
    SEMANTIC_TERRAIN_CFG,
    TeacherElevationTrajectoryEnvCfg,
    TeacherElevationTrajectoryEnvCfg_PLAY,
    TeacherElevationTrajectoryRewardsCfg,
    TeacherElevationTrajectorySceneCfg,
)
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import EventCfg as BaseEventCfg


@configclass
class TeacherElevationTrajectoryMpcSemanticSceneCfg(TeacherElevationTrajectorySceneCfg):
    """Semantic course scene with a single high-resolution scanner for MPC and CNN maps."""

    replicate_physics = True
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
        max_update_envs_per_call=512,
    )


@configclass
class TeacherElevationTrajectoryMpcSemanticObservationsCfg:
    """Dual-channel semantic grid observations from the high-resolution scanner."""

    @configclass
    class PolicyElevationSemanticMapCfg(ObsGroup):
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
    class CriticElevationSemanticMapCfg(ObsGroup):
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

    policy_elevation_semantic_map: PolicyElevationSemanticMapCfg = PolicyElevationSemanticMapCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_elevation_semantic_map: CriticElevationSemanticMapCfg = CriticElevationSemanticMapCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class TeacherElevationTrajectoryMpcSemanticRewardsCfg(TeacherElevationTrajectoryRewardsCfg):
    reference_root_pose = None
    reference_joint_pos = None
    reference_contact = None
    reference_touchdown = None
    reference_foot_pos = RewTerm(
        func=reference_foot_pos_reward,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )
    swing_leg_collision = RewTerm(
        func=swing_leg_collision_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
            "clearance": 0.04,
            "contact_force_threshold": 1.0,
            "stance_weight": 0.25,
            "swing_weight": 1.0,
            "terrain_weight": 1.0,
            "small_obstacle_weight": 2.0,
            "large_obstacle_weight": 5.0,
        },
    )


@configclass
class TeacherElevationTrajectoryMpcSemanticEventCfg(BaseEventCfg):
    generate_semantic_course = None


@configclass
class TeacherElevationTrajectoryMpcSemanticEnvCfg(TeacherElevationTrajectoryEnvCfg):
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: TeacherElevationTrajectoryMpcSemanticObservationsCfg = (
        TeacherElevationTrajectoryMpcSemanticObservationsCfg()
    )
    rewards: TeacherElevationTrajectoryMpcSemanticRewardsCfg = TeacherElevationTrajectoryMpcSemanticRewardsCfg()
    events: TeacherElevationTrajectoryMpcSemanticEventCfg = TeacherElevationTrajectoryMpcSemanticEventCfg()

    planner_owned_reference_cache: bool = True
    use_batched_reference_trajectory: bool = True
    planner_backend: str = "mpc"
    reference_height_scanner_name: str = "semantic_height_scanner"
    mpc_parallel_plan_batch_size: int = 128
    mpc_diagnostics_emit_runtime_counters: bool = True
    mpc_diagnostics_profile_cuda_sync: bool = True
    

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        
        


@configclass
class TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY(TeacherElevationTrajectoryEnvCfg_PLAY):
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=32,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: TeacherElevationTrajectoryMpcSemanticObservationsCfg = (
        TeacherElevationTrajectoryMpcSemanticObservationsCfg()
    )
    rewards: TeacherElevationTrajectoryMpcSemanticRewardsCfg = TeacherElevationTrajectoryMpcSemanticRewardsCfg()
    events: TeacherElevationTrajectoryMpcSemanticEventCfg = TeacherElevationTrajectoryMpcSemanticEventCfg()

    planner_owned_reference_cache: bool = True
    use_batched_reference_trajectory: bool = True
    planner_backend: str = "mpc"
    reference_height_scanner_name: str = "semantic_height_scanner"
    mpc_parallel_plan_batch_size: int = 4096
    mpc_diagnostics_emit_runtime_counters: bool = True
    mpc_diagnostics_profile_cuda_sync: bool = True
    semantic_scanner_update_period_s: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            tg.num_rows = SEMANTIC_TERRAIN_CFG.num_rows
            tg.num_cols = SEMANTIC_TERRAIN_CFG.num_cols
            tg.curriculum = SEMANTIC_TERRAIN_CFG.curriculum
        self.scene.height_scanner = None
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
        if self.scene.semantic_height_scanner is not None:
            self.scene.semantic_height_scanner.update_period = max(
                self.decimation * self.sim.dt,
                float(self.semantic_scanner_update_period_s),
            )
