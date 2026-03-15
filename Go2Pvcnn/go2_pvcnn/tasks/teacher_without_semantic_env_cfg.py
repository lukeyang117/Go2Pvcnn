"""Teacher training environment WITHOUT semantic/cost map/height map (state-only).

Aligned with Isaac Lab velocity_env_cfg.py (except terrain).
Uses pure proprioceptive state observations (no height_scan) for ablation.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.envs import mdp as isaac_mdp
import go2_pvcnn.mdp as custom_mdp

from .teacher_semantic_env_cfg import (
    TeacherSemanticEnvCfg,
    CommandsCfg_PLAY,
    COBBLESTONE_ROAD_CFG,
)


##
# Scene without lidar (saves training time)
##


@configclass
class TeacherSceneCfg_NoLidar(InteractiveSceneCfg):
    """Same as TeacherSceneCfg but without lidar - saves training time."""

    # Ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    # Contact Sensors (no lidar)
    contact_forces_small_objects_fl: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/cracker_box_0",
            "{ENV_REGEX_NS}/cracker_box_1",
            "{ENV_REGEX_NS}/cracker_box_2",
            "{ENV_REGEX_NS}/sugar_box_0",
            "{ENV_REGEX_NS}/sugar_box_1",
            "{ENV_REGEX_NS}/sugar_box_2",
            "{ENV_REGEX_NS}/tomato_soup_can_0",
            "{ENV_REGEX_NS}/tomato_soup_can_1",
            "{ENV_REGEX_NS}/tomato_soup_can_2",
        ],
    )
    contact_forces_small_objects_fr: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/cracker_box_0",
            "{ENV_REGEX_NS}/cracker_box_1",
            "{ENV_REGEX_NS}/cracker_box_2",
            "{ENV_REGEX_NS}/sugar_box_0",
            "{ENV_REGEX_NS}/sugar_box_1",
            "{ENV_REGEX_NS}/sugar_box_2",
            "{ENV_REGEX_NS}/tomato_soup_can_0",
            "{ENV_REGEX_NS}/tomato_soup_can_1",
            "{ENV_REGEX_NS}/tomato_soup_can_2",
        ],
    )
    contact_forces_small_objects_rl: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/cracker_box_0",
            "{ENV_REGEX_NS}/cracker_box_1",
            "{ENV_REGEX_NS}/cracker_box_2",
            "{ENV_REGEX_NS}/sugar_box_0",
            "{ENV_REGEX_NS}/sugar_box_1",
            "{ENV_REGEX_NS}/sugar_box_2",
            "{ENV_REGEX_NS}/tomato_soup_can_0",
            "{ENV_REGEX_NS}/tomato_soup_can_1",
            "{ENV_REGEX_NS}/tomato_soup_can_2",
        ],
    )
    contact_forces_small_objects_rr: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/cracker_box_0",
            "{ENV_REGEX_NS}/cracker_box_1",
            "{ENV_REGEX_NS}/cracker_box_2",
            "{ENV_REGEX_NS}/sugar_box_0",
            "{ENV_REGEX_NS}/sugar_box_1",
            "{ENV_REGEX_NS}/sugar_box_2",
            "{ENV_REGEX_NS}/tomato_soup_can_0",
            "{ENV_REGEX_NS}/tomato_soup_can_1",
            "{ENV_REGEX_NS}/tomato_soup_can_2",
        ],
    )
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Dynamic YCB Objects
    cracker_box_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    cracker_box_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    cracker_box_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    sugar_box_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    sugar_box_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    sugar_box_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    tomato_soup_can_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    tomato_soup_can_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    tomato_soup_can_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    # Furniture
    furniture_1 = AssetBaseCfg(
        prim_path="/World/SM_sofa_1",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Sofa.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.0, -2.0, 0.0)),
    )
    furniture_2 = AssetBaseCfg(
        prim_path="/World/SM_armchair_1",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Armchair.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -2.0, 0.0)),
    )
    furniture_3 = AssetBaseCfg(
        prim_path="/World/SM_table_1",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_TableA.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.0, 0.0)),
    )
    furniture_4 = AssetBaseCfg(
        prim_path="/World/SM_sofa_2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Sofa.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, 0.0, 0.0)),
    )
    furniture_5 = AssetBaseCfg(
        prim_path="/World/SM_armchair_2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Armchair.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.0, 0.0, 0.0)),
    )
    furniture_6 = AssetBaseCfg(
        prim_path="/World/SM_table_2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_TableA.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -4.0, 0.0)),
    )
    furniture_7 = AssetBaseCfg(
        prim_path="/World/SM_sofa_3",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Sofa.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, -2.0, 0.0)),
    )
    furniture_8 = AssetBaseCfg(
        prim_path="/World/SM_armchair_3",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Armchair.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, 2.0, 0.0)),
    )
    furniture_9 = AssetBaseCfg(
        prim_path="/World/SM_table_3",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_TableA.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-6.0, 2.0, 0.0)),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Isaac Lab velocity_aligned MDP (except terrain)
##


@configclass
class CommandsCfg_IsaacLab:
    """Command specs aligned with velocity_env_cfg."""

    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg_IsaacLab:
    """Action specs aligned with velocity_env_cfg."""

    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg_IsaacLab:
    """Observation specs aligned with velocity_env_cfg (state-only, no height)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=isaac_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations (same as policy for velocity env)."""

        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg_IsaacLab:
    """Reward specs aligned with velocity_env_cfg."""

    track_lin_vel_xy_exp = RewTerm(
        func=isaac_mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=isaac_mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z_l2 = RewTerm(func=isaac_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=isaac_mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=isaac_mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=isaac_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=custom_mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=isaac_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
            "threshold": 1.0,
        },
    )
    flat_orientation_l2 = RewTerm(func=isaac_mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=isaac_mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg_IsaacLab:
    """Termination specs aligned with velocity_env_cfg (no bad_orientation)."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


@configclass
class EventCfg_IsaacLab:
    """Event specs aligned with velocity_env_cfg."""

    physics_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=isaac_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=isaac_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CurriculumCfg_IsaacLab:
    """Curriculum: empty (terrain excluded per user request)."""

    pass


##
# Environment Configuration
##


@configclass
class TeacherWithoutSemanticEnvCfg(ManagerBasedRLEnvCfg):
    """Teacher environment aligned with Isaac Lab velocity_env_cfg (except terrain)."""

    scene: TeacherSceneCfg_NoLidar = TeacherSceneCfg_NoLidar(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg_IsaacLab = ObservationsCfg_IsaacLab()
    actions: ActionsCfg_IsaacLab = ActionsCfg_IsaacLab()
    commands: CommandsCfg_IsaacLab = CommandsCfg_IsaacLab()
    rewards: RewardsCfg_IsaacLab = RewardsCfg_IsaacLab()
    terminations: TerminationsCfg_IsaacLab = TerminationsCfg_IsaacLab()
    events: EventCfg_IsaacLab = EventCfg_IsaacLab()
    curriculum: CurriculumCfg_IsaacLab = CurriculumCfg_IsaacLab()

    def __post_init__(self):
        from go2_pvcnn.assets import UNITREE_GO2_CFG

        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        print("[TeacherWithoutSemanticEnvCfg] Aligned with Isaac Lab velocity_env_cfg (except terrain)")


@configclass
class TeacherWithoutSemanticEnvCfg_PLAY(TeacherWithoutSemanticEnvCfg):
    """Play/evaluation for state-only - forward motion only."""

    def __post_init__(self):
        super().__post_init__()
        self.commands = CommandsCfg_PLAY()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.push_robot = None
        self.episode_length_s = 30.0
        print("[TeacherWithoutSemanticEnvCfg_PLAY] Play mode (state-only)")
