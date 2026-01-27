"""Configuration for testing Go2 PVCNN with mixed terrain (flat + stairs) and collision detection."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Import Isaac Lab MDP functions
from isaaclab.envs import mdp as isaac_mdp

# Import custom PVCNN MDP components
import go2_pvcnn.mdp as custom_mdp
from go2_pvcnn.mdp import create_dynamic_objects_collection_cfg

# Import usd root
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Test Terrain Configuration - Mixed flat ground and stairs
##

TEST_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,  # Fewer rows for testing
    num_cols=10,  # Fewer cols for testing
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # 50% flat terrain for baseline testing
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        # 50% pyramid stairs for challenging scenarios
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


##
# Pre-defined configs
##
from go2_pvcnn.assets import UNITREE_GO2_CFG  # isort: skip

# Shared robot body filter list for collision detection
_ROBOT_BODY_FILTER = [
    "{ENV_REGEX_NS}/Robot/base",
    "{ENV_REGEX_NS}/Robot/FL_hip",
    "{ENV_REGEX_NS}/Robot/FL_thigh",
    "{ENV_REGEX_NS}/Robot/FL_calf",
    "{ENV_REGEX_NS}/Robot/FL_foot",
    "{ENV_REGEX_NS}/Robot/FR_hip",
    "{ENV_REGEX_NS}/Robot/FR_thigh",
    "{ENV_REGEX_NS}/Robot/FR_calf",
    "{ENV_REGEX_NS}/Robot/FR_foot",
    "{ENV_REGEX_NS}/Robot/RL_hip",
    "{ENV_REGEX_NS}/Robot/RL_thigh",
    "{ENV_REGEX_NS}/Robot/RL_calf",
    "{ENV_REGEX_NS}/Robot/RL_foot",
    "{ENV_REGEX_NS}/Robot/RR_hip",
    "{ENV_REGEX_NS}/Robot/RR_thigh",
    "{ENV_REGEX_NS}/Robot/RR_calf",
    "{ENV_REGEX_NS}/Robot/RR_foot",
    "{ENV_REGEX_NS}/Robot/Head_upper",
    "{ENV_REGEX_NS}/Robot/Head_lower",
]


@configclass
class Go2TestSceneCfg(InteractiveSceneCfg):
    """Test scene configuration with mixed terrain and collision detection."""

    # Test terrain with flat and stairs
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TEST_TERRAIN_CFG,
        max_init_terrain_level=0,  # Start from easiest level
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

    # Robot
    robot: ArticulationCfg = MISSING

    # Height scanner
    

    # Contact sensors for collision detection
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    
    # Body-ground collision sensor (excluding feet)
    # Monitors undesired body-ground contacts
    body_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",  # Base body collision
        history_length=3,
        track_air_time=False,
    )
    
    # Object contact sensors
    object_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=_ROBOT_BODY_FILTER,
    )
    
    object_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=_ROBOT_BODY_FILTER,
    )
    
    object_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_2",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=_ROBOT_BODY_FILTER,
    )
    
    # Camera for third-person view - captures robot, point cloud, and objects
    # Positioned to see complete scene from behind-right
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/third_person_cam",
        update_period=0.1,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-3.0, 0.0, 3.0),  # 1m behind, 3m above - elevated third person view
            rot=(0.9239, 0.0, 0.3827, 0.0),  # Looking downward 30 degrees
            convention="world",
        ),
    )

    # NOTE: LiDAR sensor configuration removed - use go2_lidar_env_cfg.py instead
    # lidar_sensor = ...

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
    # Dynamic objects (8 per environment - more obstacles for testing)
    dynamic_objects = create_dynamic_objects_collection_cfg(num_objects=3)


##
# MDP settings - Simplified for testing
##

@configclass
class TestCommandsCfg:
    """Goal-based navigation commands for testing."""
    # Goal position command - robot navigates to a fixed goal point
    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(100.0, 100.0),  # Long resampling time for goal navigation
        rel_standing_envs=0.0,  # Always moving during test
        debug_vis=False,  # Disable visualization
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.8),  # Forward movement to goal
            lin_vel_y=(-0.3, 0.3),  # Lateral movement for obstacle avoidance
            ang_vel_z=(-0.8, 0.8)   # Turning to align with goal
        ),
    )


@configclass
class TestActionsCfg:
    """Action specifications."""
    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class TestObservationsCfg:
    """Observation specifications for testing."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # Robot state
        base_lin_vel = ObsTerm(func=custom_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=custom_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=custom_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=custom_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=custom_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=custom_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=custom_mdp.last_action)
        
        # PVCNN features with cost map
        pvcnn_features = ObsTerm(
            func=custom_mdp.pvcnn_features_with_cost_map,
            params={
                "sensor_cfg": SceneEntityCfg("lidar_sensor"),
                "height_scanner_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TestEventCfg:
    """Configuration for randomization during testing."""
    
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
    
    # Reset objects to test collision scenarios
    reset_objects_position = EventTerm(
        func=custom_mdp.reset_dynamic_objects_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dynamic_objects"),
            "position_range": (-1.5, 1.5, -1.5, 1.5),  # 3m×3m range - match sensor coverage
            "height_offset": 1.0,  # Test high spawn
        },
    )


@configclass
class TestRewardsCfg:
    """Minimal rewards for testing - focus on tracking commands."""
    
    track_lin_vel_xy_exp = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    
    track_ang_vel_z_exp = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


@configclass
class TestTerminationsCfg:
    """Termination terms for testing."""
    
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class Go2PvcnnTestEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Go2 PVCNN testing environment."""

    # Scene settings
    scene: Go2TestSceneCfg = Go2TestSceneCfg(num_envs=16, env_spacing=4.0)  # Fewer envs for testing
    
    # Basic settings
    observations: TestObservationsCfg = TestObservationsCfg()
    actions: TestActionsCfg = TestActionsCfg()
    commands: TestCommandsCfg = TestCommandsCfg()
    
    # MDP settings
    rewards: TestRewardsCfg = TestRewardsCfg()
    terminations: TestTerminationsCfg = TestTerminationsCfg()
    events: TestEventCfg = TestEventCfg()
    # No curriculum for testing

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0  # Shorter episodes for testing
        
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # Update scene - create new robot config with invisible setting
        from copy import deepcopy
        robot_cfg = deepcopy(UNITREE_GO2_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.visible = False  # Make robot invisible in camera view
        self.scene.robot = robot_cfg
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # Enable rendering for visualization during testing
        self.sim.enable_scene_query_support = False
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
