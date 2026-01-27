"""Configuration for testing Go2 PVCNN with furniture objects (Sofa, Armchair, Table) for dynamic obstacle detection."""

from __future__ import annotations

import math
from dataclasses import MISSING
from copy import deepcopy

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

# Import usd root
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Test Terrain Configuration - Flat ground for furniture testing
##

FURNITURE_TEST_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # 100% flat terrain for furniture obstacle testing
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
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
class Go2FurnitureTestSceneCfg(InteractiveSceneCfg):
    """Test scene configuration with furniture objects for dynamic obstacle detection."""

    # Flat terrain for furniture testing
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FURNITURE_TEST_TERRAIN_CFG,
        max_init_terrain_level=0,
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

    # Robot - VISIBLE for testing
    robot: ArticulationCfg = MISSING

    # Height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Contact sensors for collision detection
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    
    # Body-ground collision sensor (excluding feet)
    body_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        history_length=3,
        track_air_time=False,
    )
    
    # Note: Furniture objects don't have contact sensors enabled
    # because the USD files may not have proper rigid body structure
    
    # Camera for third-person view - captures robot, point cloud, and furniture
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
            pos=(-3.0, 0.0, 3.0),  # 3m behind, 3m above
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
    
    # Furniture objects as static assets - positioned around robot spawn area
    furniture_sofa = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_Sofa",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Sofa.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.2, 0.0, 0.0),  # 1.2m前方
        ),
    )
    
    furniture_armchair = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_Armchair",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Armchair.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.8, 1.0, 0.0),  # 左后方
        ),
    )
    
    furniture_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Furniture_Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_TableA.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, -1.0, 0.0),  # 右侧
        ),
    )


##
# MDP settings
##

@configclass
class FurnitureTestCommandsCfg:
    """Simple velocity commands for furniture testing."""
    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(100.0, 100.0),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class FurnitureTestActionsCfg:
    """Action configuration for testing."""
    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True
    )


@configclass
class FurnitureTestObservationsCfg:
    """Observation configuration for furniture testing."""

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

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class FurnitureTestEventCfg:
    """Event configuration for furniture testing."""

    


@configclass
class FurnitureTestRewardsCfg:
    """Minimal rewards for furniture testing."""
    
    # Keep robot alive
    alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)
    
    # Track velocity commands
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


@configclass
class FurnitureTestTerminationsCfg:
    """Termination conditions for furniture testing."""
    
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("body_ground_contact"), "threshold": 1.0},
    )


##
# Environment configuration
##

@configclass
class Go2PvcnnFurnitureTestEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Go2 PVCNN furniture obstacle test environment."""

    # Scene settings
    scene: Go2FurnitureTestSceneCfg = Go2FurnitureTestSceneCfg(num_envs=1, env_spacing=4.0)
    
    # Basic settings
    observations: FurnitureTestObservationsCfg = FurnitureTestObservationsCfg()
    actions: FurnitureTestActionsCfg = FurnitureTestActionsCfg()
    commands: FurnitureTestCommandsCfg = FurnitureTestCommandsCfg()
    
    # MDP settings
    rewards: FurnitureTestRewardsCfg = FurnitureTestRewardsCfg()
    terminations: FurnitureTestTerminationsCfg = FurnitureTestTerminationsCfg()
    events: FurnitureTestEventCfg = FurnitureTestEventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # Update viewer settings
        self.viewer.eye = (7.5, 7.5, 7.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Robot configuration - VISIBLE
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
