"""Independent MPC semantic trajectory teacher config."""

from __future__ import annotations

import math
from dataclasses import field
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from extension.batch_mpc_planner.config import MpcPlannerCfg
from extension.batch_mpc_planner.participation import MpcTerrainDifficultyPair
from extension.mdp.observations import downsampled_elevation_semantic_scan
from extension.mdp.rewards_reference import reference_foot_pos_reward
from extension.mdp.semantic_contact_rewards import semantic_global_contact_collision_reward
from extension.semantic_curriculum import (
    SemanticObstacleCount,
    SemanticObstacleCurriculumCfg,
)
from extension.semantic_course import (
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
    SemanticCourseTerrainImporter,
)
from go2_pvcnn.assets import UNITREE_GO2_CFG
import go2_pvcnn.mdp as mdp
from go2_pvcnn.sensor.semantic_contacter import SemanticGlobalContactSensor
from go2_pvcnn.sensor.semantic_raycaster import SemanticGridRayCasterCfg


_TEACHER_OBJECTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "teacher_object"

SEMANTIC_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

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


def _reference_foot_pos_reward_term() -> RewTerm:
    return RewTerm(
        func=reference_foot_pos_reward,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )


def _semantic_contact_collision_reward_term() -> RewTerm:
    return RewTerm(
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


@configclass
class TeacherElevationTrajectoryMpcSemanticSceneCfg(InteractiveSceneCfg):
    """Semantic course scene with a single high-resolution scanner for MPC and CNN maps."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SEMANTIC_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=str(
                _TEACHER_OBJECTS_DIR
                / "Materials"
                / "TilesMarbleSpiderWhiteBrickBondHoned"
                / "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    replicate_physics = True
    height_scanner = None
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class TeacherElevationTrajectoryMpcSemanticEventCfg:
    """Events for the active semantic MPC task."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class TeacherElevationTrajectoryMpcSemanticCommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class TeacherElevationTrajectoryMpcSemanticActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
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
class TeacherElevationTrajectoryMpcSemanticRewardsCfg:
    """Rewards for the active semantic MPC task."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )
    reference_foot_pos = _reference_foot_pos_reward_term()
    semantic_contact_collision = _semantic_contact_collision_reward_term()


@configclass
class TeacherElevationTrajectoryMpcSemanticTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class TeacherElevationTrajectoryMpcSemanticCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_semantic_plane_gate,
        params={"cfg_name": "semantic_obstacle_curriculum"},
    )
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class TeacherElevationTrajectoryMpcSemanticEnvCfg(ManagerBasedRLEnvCfg):
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: TeacherElevationTrajectoryMpcSemanticObservationsCfg = (
        TeacherElevationTrajectoryMpcSemanticObservationsCfg()
    )
    actions: TeacherElevationTrajectoryMpcSemanticActionsCfg = TeacherElevationTrajectoryMpcSemanticActionsCfg()
    commands: TeacherElevationTrajectoryMpcSemanticCommandsCfg = TeacherElevationTrajectoryMpcSemanticCommandsCfg()
    rewards: TeacherElevationTrajectoryMpcSemanticRewardsCfg = TeacherElevationTrajectoryMpcSemanticRewardsCfg()
    events: TeacherElevationTrajectoryMpcSemanticEventCfg = TeacherElevationTrajectoryMpcSemanticEventCfg()
    terminations: TeacherElevationTrajectoryMpcSemanticTerminationsCfg = TeacherElevationTrajectoryMpcSemanticTerminationsCfg()
    curriculum: TeacherElevationTrajectoryMpcSemanticCurriculumCfg = TeacherElevationTrajectoryMpcSemanticCurriculumCfg()

    planner_owned_reference_cache: bool = True
    use_batched_reference_trajectory: bool = True
    planner_backend: str = "mpc"
    reference_height_scanner_name: str = "semantic_height_scanner"
    mpc_planner_cfg: MpcPlannerCfg = field(default_factory=MpcPlannerCfg)
    semantic_obstacle_curriculum: SemanticObstacleCurriculumCfg = field(
        default_factory=lambda: SemanticObstacleCurriculumCfg(
            enabled=True,
            plane_terrain_names=("flat",),
            plane_counts=(
                SemanticObstacleCount(small=1, large=0),
                SemanticObstacleCount(small=3, large=1),
                SemanticObstacleCount(small=4, large=1),
                SemanticObstacleCount(small=5, large=1),
                SemanticObstacleCount(small=6, large=1),
                SemanticObstacleCount(small=7, large=1),
                SemanticObstacleCount(small=8, large=1),
                SemanticObstacleCount(small=9, large=1),
                SemanticObstacleCount(small=10, large=2),
                SemanticObstacleCount(small=11, large=3),
            ),
            non_plane_counts=(
                SemanticObstacleCount(small=0, large=0),
                SemanticObstacleCount(small=0, large=0),
                SemanticObstacleCount(small=1, large=0),
                SemanticObstacleCount(small=1, large=0),
                SemanticObstacleCount(small=2, large=0),
                SemanticObstacleCount(small=2, large=0),
                SemanticObstacleCount(small=3, large=1),
                SemanticObstacleCount(small=3, large=1),
                SemanticObstacleCount(small=4, large=1),
                SemanticObstacleCount(small=4, large=1),
            ),
            center_safety_half_extent_m=(0.85,),
            min_spacing_clearance_m=(0.15,),
            tile_margin_m=(0.50,),
            collision_force_threshold=1.0,
            plane_collision_rate_threshold=0.03,
            consecutive_success_required=5,
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        elif self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.class_type = SemanticCourseTerrainImporter
        self.scene.terrain.semantic_obstacle_curriculum = self.semantic_obstacle_curriculum
        self.mpc_planner_cfg.runtime.horizon_steps = 25
        self.mpc_planner_cfg.runtime.replan_interval_steps = 25
        self.mpc_planner_cfg.runtime.dt = 0.02
        self.mpc_planner_cfg.runtime.parallel_plan_batch_size = 64
        self.mpc_planner_cfg.diagnostics.emit_runtime_counters = False
        self.mpc_planner_cfg.diagnostics.profile_cuda_sync = False
        self.mpc_planner_cfg.reference_participation.exclude_pairs = (
            MpcTerrainDifficultyPair(
                terrain_names=(
                    "pyramid_stairs",
                    "pyramid_stairs_inv",
                    "boxes",
                    "random_rough",
                ),
                terrain_rows=(5, 6, 7, 8, 9),
            ),
            )
        self.mpc_planner_cfg.losses.fk_body_leg_collision.weight = 120.0


@configclass
class TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY(TeacherElevationTrajectoryMpcSemanticEnvCfg):
    scene: TeacherElevationTrajectoryMpcSemanticSceneCfg = TeacherElevationTrajectoryMpcSemanticSceneCfg(
        num_envs=32,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: TeacherElevationTrajectoryMpcSemanticObservationsCfg = (
        TeacherElevationTrajectoryMpcSemanticObservationsCfg()
    )
    actions: TeacherElevationTrajectoryMpcSemanticActionsCfg = TeacherElevationTrajectoryMpcSemanticActionsCfg()
    commands: TeacherElevationTrajectoryMpcSemanticCommandsCfg = TeacherElevationTrajectoryMpcSemanticCommandsCfg()
    rewards: TeacherElevationTrajectoryMpcSemanticRewardsCfg = TeacherElevationTrajectoryMpcSemanticRewardsCfg()
    events: TeacherElevationTrajectoryMpcSemanticEventCfg = TeacherElevationTrajectoryMpcSemanticEventCfg()
    terminations: TeacherElevationTrajectoryMpcSemanticTerminationsCfg = TeacherElevationTrajectoryMpcSemanticTerminationsCfg()
    curriculum: TeacherElevationTrajectoryMpcSemanticCurriculumCfg = TeacherElevationTrajectoryMpcSemanticCurriculumCfg()

    planner_owned_reference_cache: bool = False
    use_batched_reference_trajectory: bool = False
    planner_backend: str = "mpc"
    reference_height_scanner_name: str = "semantic_height_scanner"
    semantic_scanner_update_period_s: float = 0.02
    mpc_planner_cfg: MpcPlannerCfg = field(default_factory=MpcPlannerCfg)

    def __post_init__(self):
        super().__post_init__()
        self.planner_owned_reference_cache = False
        self.use_batched_reference_trajectory = False
        self.rewards.reference_foot_pos = None
        self.rewards.semantic_contact_collision = None
        self.scene.semantic_contact_small = None
        self.scene.semantic_contact_large = None
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            tg.num_rows = SEMANTIC_TERRAIN_CFG.num_rows
            tg.num_cols = SEMANTIC_TERRAIN_CFG.num_cols
            tg.curriculum = SEMANTIC_TERRAIN_CFG.curriculum
        self.scene.height_scanner = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.events.push_robot = None
        self.observations.policy_elevation_semantic_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_elevation_semantic_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
        if self.scene.semantic_height_scanner is not None:
            self.scene.semantic_height_scanner.update_period = max(
                self.decimation * self.sim.dt,
                float(self.semantic_scanner_update_period_s),
            )


@configclass
class TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER(TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY):
    planner_owned_reference_cache: bool = True
    use_batched_reference_trajectory: bool = True
    planner_backend: str = "mpc"

    def __post_init__(self):
        super().__post_init__()
        self.planner_owned_reference_cache = True
        self.use_batched_reference_trajectory = True
        self.rewards.reference_foot_pos = _reference_foot_pos_reward_term()
        self.rewards.semantic_contact_collision = _semantic_contact_collision_reward_term()
        self.scene.semantic_contact_small = _semantic_global_contact_sensor(SEMANTIC_COURSE_SMALL_ROOT)
        self.scene.semantic_contact_large = _semantic_global_contact_sensor(SEMANTIC_COURSE_LARGE_ROOT)
        self.mpc_planner_cfg.runtime.parallel_plan_batch_size = 4096
        self.mpc_planner_cfg.diagnostics.emit_runtime_counters = True
        self.mpc_planner_cfg.diagnostics.profile_cuda_sync = True


@configclass
class TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg(TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER):
    """MPC-enabled policy evaluation config for terrain tracking metrics."""

    def __post_init__(self):
        super().__post_init__()
        self.planner_owned_reference_cache = True
        self.use_batched_reference_trajectory = True
        self.planner_backend = "mpc"
        self.mpc_planner_cfg.runtime.horizon_steps = 25
        self.mpc_planner_cfg.runtime.replan_interval_steps = 25
        self.mpc_planner_cfg.runtime.dt = 0.02
        self.mpc_planner_cfg.runtime.parallel_plan_batch_size = 64
        self.mpc_planner_cfg.diagnostics.emit_runtime_counters = False
        self.mpc_planner_cfg.diagnostics.profile_cuda_sync = False


@configclass
class TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg(
    TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg
):
    """MPC-enabled policy evaluation config for dense small-obstacle flat collision metrics."""

    small_collision_eval_small_count_per_tile: int = 80
    small_collision_eval_large_count_per_tile: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.small_collision_eval_small_count_per_tile = 80
        self.small_collision_eval_large_count_per_tile = 0
