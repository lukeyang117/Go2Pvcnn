"""Independent MPC semantic trajectory teacher config."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from extension.mdp.observations import downsampled_elevation_semantic_scan
from extension.mdp.rewards_reference import reference_foot_pos_reward
from extension.mdp.semantic_contact_rewards import semantic_global_contact_collision_reward
from extension.semantic_course import (
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
    SemanticCourseTerrainImporter,
)
from go2_pvcnn.sensor.semantic_contacter import SemanticGlobalContactSensor
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg
from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import (
    SEMANTIC_TERRAIN_CFG,
    TeacherElevationTrajectoryEnvCfg,
    TeacherElevationTrajectoryEnvCfg_PLAY,
    TeacherElevationTrajectoryRewardsCfg,
    TeacherElevationTrajectorySceneCfg,
)
from go2_pvcnn.tasks.teacher_without_semantic_env_cfg import EventCfg as BaseEventCfg


SEMANTIC_CONTACT_BODY_NAMES = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "base",
)
SEMANTIC_CONTACT_BODY_WEIGHTS = (
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    5.0,
)


def _semantic_global_contact_sensor(semantic_root: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        class_type=SemanticGlobalContactSensor,
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        debug_vis=False,
        filter_prim_paths_expr=[f"{semantic_root}/.*"],
    )


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
    semantic_contact_small = _semantic_global_contact_sensor(SEMANTIC_COURSE_SMALL_ROOT)
    semantic_contact_large = _semantic_global_contact_sensor(SEMANTIC_COURSE_LARGE_ROOT)


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
    semantic_contact_collision = RewTerm(
        func=semantic_global_contact_collision_reward,
        weight=1.0,
        params={
            "small_sensor_cfg": SceneEntityCfg("semantic_contact_small"),
            "large_sensor_cfg": SceneEntityCfg("semantic_contact_large"),
            "body_names": SEMANTIC_CONTACT_BODY_NAMES,
            "body_weights": SEMANTIC_CONTACT_BODY_WEIGHTS,
            "force_threshold": 1.0,
            "force_scale": 50.0,
            "force_clip": 1.0,
            "small_weight": 1.0,
            "large_weight": 2.0,
        },
    )
    swing_leg_collision = None


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
    reference_trajectory_horizon: int = 25
    reference_replan_interval_steps: int = 25
    mpc_parallel_plan_batch_size: int = 64
    mpc_diagnostics_emit_runtime_counters: bool = False
    mpc_diagnostics_profile_cuda_sync: bool = False
    

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.mpc_planner_cfg.losses.fk_body_leg_collision.weight = 120.0
        
        


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
    reference_trajectory_horizon: int = 25
    reference_replan_interval_steps: int = 25
    mpc_parallel_plan_batch_size: int = 4096
    mpc_diagnostics_emit_runtime_counters: bool = True
    mpc_diagnostics_profile_cuda_sync: bool = True
    semantic_scanner_update_period_s: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.mpc_planner_cfg.losses.fk_body_leg_collision.weight = 120.0
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
