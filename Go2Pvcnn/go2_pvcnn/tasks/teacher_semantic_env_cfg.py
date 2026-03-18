"""Teacher training environment configuration with ground truth semantic labels.

This environment uses real semantic labels from LiDAR sensor (no PVCNN inference).
It combines elements from:
- go2_lidar_env_cfg.py: LiDAR sensor configuration
- go2_contact_global_env_cfg.py: Contact sensor configuration  
- go2_pvcnn_env_cfg.py: Base environment structure

Key differences from PVCNN environment:
1. Uses SemanticLidarSensor for ground truth semantic labels
2. No PVCNN wrapper/inference - direct semantic to cost map
3. Cost map includes command alignment bonus
4. Furniture assets commented out (for future use)
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# Import Isaac Lab MDP functions
from isaaclab.envs import mdp as isaac_mdp

# Import custom MDP components
import go2_pvcnn.mdp as custom_mdp

# Import Go2 asset
from go2_pvcnn.assets import UNITREE_GO2_CFG

# Import semantic lidar
from go2_pvcnn.sensor.lidar import SemanticLidarCfg, LivoxPatternCfg

# 本地 teacher 物体目录（运行 assets/download_teacher_objects.py 下载）
_TEACHER_OBJECTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "teacher_object"

##
# Terrain Configuration
##

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
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
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
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


# Shared robot body filter list for contact sensors
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


##
# Scene Definition
##

@configclass
class TeacherSceneCfg(InteractiveSceneCfg):
    """Scene configuration for teacher training with semantic LiDAR."""

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
            mdl_path=str(_TEACHER_OBJECTS_DIR / "Materials" / "TilesMarbleSpiderWhiteBrickBondHoned" / "TilesMarbleSpiderWhiteBrickBondHoned.mdl"),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Robot
    robot: ArticulationCfg = MISSING

    # ========================================
    # Semantic LiDAR Sensor (includes height map generation)
    # ========================================
    lidar: SemanticLidarCfg = SemanticLidarCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=SemanticLidarCfg.OffsetCfg(
            pos=[0.3, 0.0, 0.2],  # Forward and up from base
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        ray_alignment="yaw",
        mesh_prim_paths=[
            "/World/ground",
            # Global furniture (static, no reset)
            "/World/SM_sofa_1",
            "/World/SM_armchair_1",
            "/World/SM_table_1",
            "/World/SM_sofa_2",
            "/World/SM_armchair_2",
            "/World/SM_table_2",
            "/World/SM_sofa_3",
            "/World/SM_armchair_3",
            "/World/SM_table_3",
            # Dynamic YCB objects (per-env, will be reset)
            "{ENV_REGEX_NS}/cracker_box_0/_03_cracker_box",
            "{ENV_REGEX_NS}/cracker_box_1/_03_cracker_box",
            "{ENV_REGEX_NS}/cracker_box_2/_03_cracker_box",
            "{ENV_REGEX_NS}/sugar_box_0/_04_sugar_box",
            "{ENV_REGEX_NS}/sugar_box_1/_04_sugar_box",
            "{ENV_REGEX_NS}/sugar_box_2/_04_sugar_box",
            "{ENV_REGEX_NS}/tomato_soup_can_0/_05_tomato_soup_can",
            "{ENV_REGEX_NS}/tomato_soup_can_1/_05_tomato_soup_can",
            "{ENV_REGEX_NS}/tomato_soup_can_2/_05_tomato_soup_can",
        ],
        semantic_class_mapping={
            "terrain": ["ground", "wall", "floor", "plane"],
            "dynamic_obstacle": [
                "cracker_box", "sugar_box", "tomato_soup_can",
                "_03_cracker_box", "_04_sugar_box", "_05_tomato_soup_can"
            ],
            "valuable": [
                "sofa", "armchair", "table", "chair",
                "SM_Sofa", "SM_Armchair", "SM_Table"
            ],
        },
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            use_simple_grid=True,
            horizontal_line_num=50,
            vertical_line_num=50,
            horizontal_fov_deg_min=-180.0,
            horizontal_fov_deg_max=180.0,
            vertical_fov_deg_min=-29.5,
            vertical_fov_deg_max=40.5,
        ),
        update_frequency=10.0,
        drift_range=(-0.0, 0.0),
        max_distance=1.5,
        min_range=0.1,
        return_pointcloud=True,
        pointcloud_in_world_frame=False,
        return_semantic_labels=True,  
        enable_sensor_noise=False,
        debug_vis=False,
        # Height map
        return_height_map=True,
        height_map_size=(1.5, 1.5),
        height_map_resolution=0.1,  
    )

    # ========================================
    # Contact Sensors
    # ========================================
    # Small objects contact (base only, with filter)
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
    
    # # Furniture contact (global furniture, static)
    # contact_forces_furniture: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     update_period=0.0,
    #     history_length=3,
    #     track_air_time=False,
    #     debug_vis=False,
    #     filter_prim_paths_expr=[
    #         "/World/SM_sofa_1",
    #         "/World/SM_armchair_1",
    #         "/World/SM_table_1",
    #         "/World/SM_sofa_2",
    #         "/World/SM_armchair_2",
    #         "/World/SM_table_2",
    #         "/World/SM_sofa_3",
    #         "/World/SM_armchair_3",
    #         "/World/SM_table_3",
    #     ],
    # )
    
    # Contact sensor for rewards (feet_air_time, undesired_contacts, base_contact)
    # Aligned with teacher_without_semantic: history_length=3 for feet_air_time
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # ========================================
    # Dynamic YCB Objects - 9 objects per environment
    # ========================================
    
    # CrackerBox 0
    cracker_box_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "003_cracker_box.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    
    # CrackerBox 1
    cracker_box_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "003_cracker_box.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    
    # CrackerBox 2
    cracker_box_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "003_cracker_box.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    
    # SugarBox 0
    sugar_box_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "004_sugar_box.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    
    # SugarBox 1
    sugar_box_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "004_sugar_box.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    
    # SugarBox 2
    sugar_box_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "004_sugar_box.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    
    # TomatoSoupCan 0
    tomato_soup_can_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "005_tomato_soup_can.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    
    # TomatoSoupCan 1
    tomato_soup_can_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "005_tomato_soup_can.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    
    # TomatoSoupCan 2
    tomato_soup_can_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TEACHER_OBJECTS_DIR / "005_tomato_soup_can.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    # ========================================
    # Furniture Objects - Global (static assets, no physics)
    # ========================================
    
    furniture_1 = AssetBaseCfg(
        prim_path="/World/SM_sofa_1",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_Sofa.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.0, -2.0, 0.0)),
    )
    
    furniture_2 = AssetBaseCfg(
        prim_path="/World/SM_armchair_1",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_Armchair.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -2.0, 0.0)),
    )
    
    furniture_3 = AssetBaseCfg(
        prim_path="/World/SM_table_1",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_TableA.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.0, 0.0)),
    )
    
    furniture_4 = AssetBaseCfg(
        prim_path="/World/SM_sofa_2",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_Sofa.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, 0.0, 0.0)),
    )
    
    furniture_5 = AssetBaseCfg(
        prim_path="/World/SM_armchair_2",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_Armchair.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.0, 0.0, 0.0)),
    )
    
    furniture_6 = AssetBaseCfg(
        prim_path="/World/SM_table_2",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_TableA.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -4.0, 0.0)),
    )
    
    furniture_7 = AssetBaseCfg(
        prim_path="/World/SM_sofa_3",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_Sofa.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, -2.0, 0.0)),
    )
    
    furniture_8 = AssetBaseCfg(
        prim_path="/World/SM_armchair_3",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_Armchair.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, 2.0, 0.0)),
    )
    
    furniture_9 = AssetBaseCfg(
        prim_path="/World/SM_table_3",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_TEACHER_OBJECTS_DIR / "SM_TableA.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-6.0, 2.0, 0.0)),
    )

    # Lighting（使用本地天空光纹理）
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=str(
                _TEACHER_OBJECTS_DIR
                / "Materials"
                / "Textures"
                / "Skies"
                / "PolyHaven"
                / "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


##
# MDP Settings
##

@configclass
class CommandsCfg:
    """Command specifications with curriculum (aligned with unitree go2 velocity_env_cfg)."""
    base_velocity = custom_mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=custom_mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)
        ),
        limit_ranges=custom_mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class CommandsCfg_PLAY:
    """Command specifications for play/evaluation mode - forward motion only."""
    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=False,  # Disable heading command for forward motion
        heading_control_stiffness=0.0,
        rel_standing_envs=0.0,  # No standing
        rel_heading_envs=0.0,
        debug_vis=True,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0),  # Forward velocity only
            lin_vel_y=(0.0, 0.0),  # No lateral velocity
            ang_vel_z=(0.0, 0.0),  # No angular velocity
            heading=(0.0, 0.0),  # No heading change
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCostMapCfg(ObsGroup):
        """Cost map observations for CNN (2D)."""
        
        semantic_cost_map = ObsTerm(
            func=custom_mdp.teacher_semantic_cost_map,
            params={
                "lidar_cfg": SceneEntityCfg("lidar"),
                "command_name": "base_velocity",
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True  # Single term, no concat needed
    
    @configclass
    class PolicyStateCfg(ObsGroup):
        """State observations for MLP (1D)."""

        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # Joint state
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # Velocity commands
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # Last actions
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True  # Concatenate all 1D vectors

    @configclass
    class CriticCostMapCfg(ObsGroup):
        """Cost map observations for critic CNN (2D)."""
        
        semantic_cost_map = ObsTerm(
            func=custom_mdp.teacher_semantic_cost_map,
            params={
                "lidar_cfg": SceneEntityCfg("lidar"),
                "command_name": "base_velocity",
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticStateCfg(ObsGroup):
        """State observations for critic MLP (1D)."""

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

    policy_cost_map: PolicyCostMapCfg = PolicyCostMapCfg()
    policy_state: PolicyStateCfg = PolicyStateCfg()
    critic_cost_map: CriticCostMapCfg = CriticCostMapCfg()
    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class EventCfg:
    """Configuration for randomization events, aligned with teacher_without_semantic (Go2 reset) + YCB resets."""

    # Startup (aligned with teacher_without_semantic)
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

    # Reset Go2 (aligned with teacher_without_semantic)
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

    # Reset YCB objects positions (keep for teacher_semantic scene)
    # Note: Furniture is global/static and doesn't need reset
    
    reset_cracker_box_0 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cracker_box_0"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_cracker_box_1 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cracker_box_1"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_cracker_box_2 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cracker_box_2"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_sugar_box_0 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sugar_box_0"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_sugar_box_1 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sugar_box_1"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_sugar_box_2 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sugar_box_2"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_tomato_soup_can_0 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("tomato_soup_can_0"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_tomato_soup_can_1 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("tomato_soup_can_1"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_tomato_soup_can_2 = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("tomato_soup_can_2"),
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms aligned with teacher_without_semantic (Isaac Lab velocity)."""

    # -- task
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
    # -- penalties
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
    # -- optional
    flat_orientation_l2 = RewTerm(func=isaac_mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=isaac_mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms aligned with teacher_without_semantic (no bad_orientation)."""
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms: velocity command range expansion only (aligned with unitree)."""

    lin_vel_cmd_levels = CurrTerm(
        func=custom_mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "track_lin_vel_xy_exp"},
    )


@configclass
class CurriculumCfg_Empty:
    """Empty curriculum for play/eval (no curriculum)."""
    pass


##
# Environment Configuration
##

@configclass
class TeacherSemanticEnvCfg(ManagerBasedRLEnvCfg):
    """Teacher training environment with ground truth semantic labels."""

    # Scene settings
    scene: TeacherSceneCfg = TeacherSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # Update scene
        # 使用本地 Go2 资产（运行 assets/download_teacher_objects.py 下载）
        go2_usd = str(_TEACHER_OBJECTS_DIR / "Robots" / "Unitree" / "Go2" / "go2.usd")
        self.scene.robot = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UNITREE_GO2_CFG.spawn.replace(usd_path=go2_usd),
        )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        print(f"[TeacherSemanticEnvCfg] Configured with {self.scene.num_envs} environments")
        print(f"[TeacherSemanticEnvCfg] Total YCB objects: {9 * self.scene.num_envs}")


@configclass
class TeacherSemanticEnvCfg_PLAY(TeacherSemanticEnvCfg):
    """Play/evaluation configuration - forward motion only."""
    
    def __post_init__(self):
        # Call parent initialization first
        super().__post_init__()
        
        # Override commands with play configuration (forward motion only)
        self.commands = CommandsCfg_PLAY()

        # Disable curriculum in play mode
        self.curriculum = CurriculumCfg_Empty()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable observation corruption/noise
        self.observations.policy_cost_map.enable_corruption = False
        self.observations.policy_state.enable_corruption = False
        self.observations.critic_cost_map.enable_corruption = False
        self.observations.critic_state.enable_corruption = False
        
        # Remove random pushing
        self.events.push_robot = None
        
        # Reduce episode length for faster evaluation
        self.episode_length_s = 30.0
        
        print(f"[TeacherSemanticEnvCfg_PLAY] Play mode configured:")
        print(f"  - Environments: {self.scene.num_envs}")
        print(f"  - Velocity command: X=[0.5, 1.0] m/s (forward only)")
        print(f"  - Episode length: {self.episode_length_s}s")
        print(f"  - Observation noise: Disabled")
