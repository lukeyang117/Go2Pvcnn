"""Go2 environment configuration for LiDAR sensor testing with 9 YCB objects.

This configuration is based on go2_pvcnn_env_cfg.py but focuses on LiDAR sensor testing
with 9 semantic objects (3x CrackerBox + 3x SugarBox + 3x TomatoSoupCan) per environment.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR,ISAAC_NUCLEUS_DIR

# Import Isaac Lab MDP functions
from isaaclab.envs import mdp as isaac_mdp

# Import custom PVCNN MDP components
import go2_pvcnn.mdp as custom_mdp

# Import usd root
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import Go2 asset
from go2_pvcnn.assets import UNITREE_GO2_CFG
from go2_pvcnn.sensor.lidar import SemanticLidarCfg, LivoxPatternCfg, LidarCfg


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    #这是网格中**单个子地形（Sub-terrain）**的物理尺寸（长 x 宽），单位是米
    size=(8.0, 8.0),
    #在整个生成的地形（所有行和列的集合）的最外围，添加的保护边界宽度
    border_width=20.0,
    #在 Isaac Lab 中，行（Rows）代表难度等级（Difficulty Level）。
    num_rows=10,
    #列代表同一难度下的不同变体或不同类型的地形。
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        #Other terrain types commented out for simpler flat terrain
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
    ##
    #每个sub-terrain的原点世界坐标生成公式（（row+0.5）*size-num_rows*size[0]*0.5，(col+0.5)*size-num_cols*size[1]*0.5）
    #每个subtarrain的platform_width是指地形中平坦区域的尺寸（长和宽），单位是米
    ##
)
# Shared robot body filter list for contact sensors
_ROBOT_BODY_FILTER = [
    "{ENV_REGEX_NS}/Robot/base",           # 0. 基座
    
    # Legs (腿部，按hip-thigh-calf-foot顺序)
    "{ENV_REGEX_NS}/Robot/FL_hip",         # 1. 左前髋
    "{ENV_REGEX_NS}/Robot/FL_thigh",       # 6. 左前大腿
    "{ENV_REGEX_NS}/Robot/FL_calf",        # 11. 左前小腿
    "{ENV_REGEX_NS}/Robot/FL_foot",        # 15. 左前脚
    
    "{ENV_REGEX_NS}/Robot/FR_hip",         # 2. 右前髋
    "{ENV_REGEX_NS}/Robot/FR_thigh",       # 7. 右前大腿
    "{ENV_REGEX_NS}/Robot/FR_calf",        # 12. 右前小腿
    "{ENV_REGEX_NS}/Robot/FR_foot",        # 16. 右前脚
    
    "{ENV_REGEX_NS}/Robot/RL_hip",         # 4. 左后髋
    "{ENV_REGEX_NS}/Robot/RL_thigh",       # 9. 左后大腿
    "{ENV_REGEX_NS}/Robot/RL_calf",        # 13. 左后小腿
    "{ENV_REGEX_NS}/Robot/RL_foot",        # 17. 左后脚
    
    "{ENV_REGEX_NS}/Robot/RR_hip",         # 5. 右后髋
    "{ENV_REGEX_NS}/Robot/RR_thigh",       # 10. 右后大腿
    "{ENV_REGEX_NS}/Robot/RR_calf",        # 14. 右后小腿
    "{ENV_REGEX_NS}/Robot/RR_foot",        # 18. 右后脚
    
    # Head (头部传感器支架)
    "{ENV_REGEX_NS}/Robot/Head_upper",     # 3. 头部上部
    "{ENV_REGEX_NS}/Robot/Head_lower",     # 8. 头部下部
]


@configclass
class Go2LidarSceneCfg(InteractiveSceneCfg):
    """Configuration for Go2 robot scene with LiDAR and 9 YCB objects."""

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

    # Robot
    robot: ArticulationCfg = MISSING

    

    # LiDAR Sensor configuration
    lidar: SemanticLidarCfg = SemanticLidarCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=SemanticLidarCfg.OffsetCfg(
            pos=[0.3, 0.0, 0.2],  # Forward and up from base
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        ray_alignment="yaw",  # Only track yaw rotation
        mesh_prim_paths=[
            "/World/ground",  # Static ground
            "/world/SM_sofa_1",
            "/world/SM_armchair_1",
            "/world/SM_table_1",
            "/world/SM_sofa_2",
            "/world/SM_armchair_2",
            "/world/SM_table_2",
            "/world/SM_sofa_3",
            "/world/SM_armchair_3",
            "/world/SM_table_3",
            #dynamic YCB objects
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
        drift_range=(-0.0, 0.0),  # No sensor drift for testing
        max_distance=15.0,  # Reduced from 100.0 to allow more no-hit rays (upward rays won't reach far terrain)
        min_range=0.1,
        return_pointcloud=True,
        pointcloud_in_world_frame=False,
        enable_sensor_noise=False,
        debug_vis=True,  # 启用可视化
        # Height map configuration
        return_height_map=True,  # Enable height map generation
        height_map_size=(3.2, 3.2),  # 3.2m x 3.2m grid around robot
        height_map_resolution=0.1,  # 0.1m grid resolution (32x32 grid)
    )

    # Contact sensor for foot contact detection
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=2,
        track_air_time=True,
    )

    # ========================================
    # Dynamic YCB Objects - 9 objects per environment
    # 3x CrackerBox, 3x SugarBox, 3x TomatoSoupCan
    # Using category name as prim path instead of Object_X
    # ========================================
    
    # CrackerBox 0
    cracker_box_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[20.0, -1.0, 0.0],  # 2m in front, 1m left
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    
    # CrackerBox 1
    cracker_box_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[20.5, -1.0, 0.0],  # 2.5m in front, 1m left
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    
    # CrackerBox 2
    cracker_box_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cracker_box_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[30.0, -1.0, 0.0],  # 3m in front, 1m left
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    
    # SugarBox 0
    sugar_box_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[20.0, 0.0, 0.0],  # 2m in front, centered
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    
    # SugarBox 1
    sugar_box_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[20.5, 0.0, 0.0],  # 2.5m in front, centered
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    
    # SugarBox 2
    sugar_box_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[30.0, 0.0, 0.0],  # 3m in front, centered
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    
    # TomatoSoupCan 0
    tomato_soup_can_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[20.0, 10.0, 0.0],  # 2m in front, 1m right
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    
    # TomatoSoupCan 1
    tomato_soup_can_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[20.5, 1.0, 0.0],  # 2.5m in front, 1m right
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    
    # TomatoSoupCan 2
    tomato_soup_can_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[30.0, 1.0, 0.0],  # 3m in front, 1m right
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )

    # Add global furniture placed near flat sub-terrain origins
    # Compute sub-terrain origin formula used in the terrain generator:
    # x = (row + 0.5) * size_x - num_rows * size_x * 0.5
    # y = (col + 0.5) * size_y - num_cols * size_y * 0.5
    # We use row=0 and cols=1..9 (skip the first flat), then offset x by +2.0
    # Based on COBBLESTONE_ROAD_CFG: size=(8.0,8.0), num_rows=10, num_cols=20
    # Explicit furniture assets placed at the precomputed positions
    # (inserted directly into the config; no intermediate lists)
    furniture_1 = AssetBaseCfg(
        prim_path="/world/SM_sofa_1",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Sofa.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-26.0, -76.0, 0.0)),
    )

    furniture_2 = AssetBaseCfg(
        prim_path="/world/SM_armchair_1",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Armchair.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-18.0, -76.0, 0.0)),
    )

    furniture_3 = AssetBaseCfg(
        prim_path="/world/SM_table_1",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_TableA.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-10.0, -76.0, 0.0)),
    )

    furniture_4 = AssetBaseCfg(
        prim_path="/world/SM_sofa_2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Sofa.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.0, -76.0, 0.0)),
    )

    furniture_5 = AssetBaseCfg(
        prim_path="/world/SM_armchair_2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Armchair.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, -76.0, 0.0)),
    )

    furniture_6 = AssetBaseCfg(
        prim_path="/world/SM_table_2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_TableA.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(14.0, -76.0, 0.0)),
    )

    furniture_7 = AssetBaseCfg(
        prim_path="/world/SM_sofa_3",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Sofa.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(22.0, -76.0, 0.0)),
    )

    furniture_8 = AssetBaseCfg(
        prim_path="/world/SM_armchair_3",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Armchair.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(30.0, -76.0, 0.0)),
    )

    furniture_9 = AssetBaseCfg(
        prim_path="/world/SM_table_3",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_TableA.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(38.0, -76.0, 0.0)),
    )

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # Velocity command with heading control
    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=True,
        heading_control_stiffness=1.0,
        rel_standing_envs=0.02,
        rel_heading_envs=0.8,
        debug_vis=True,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-1, 1),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class Go2LidarEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Go2 robot with LiDAR sensor."""

    # ========================================
    # Scene configuration
    # ========================================
    scene: Go2LidarSceneCfg = Go2LidarSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=True)

    # ========================================
    # Actions
    # ========================================
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

    actions: ActionsCfg = ActionsCfg()

    # ========================================
    # Commands
    # ========================================
    commands: CommandsCfg = CommandsCfg()

    # ========================================
    # Observations
    # ========================================
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

    actions: ActionsCfg = ActionsCfg()

    # ========================================
    # Observations
    # ========================================
    @configclass
    class ObservationsCfg:
        """Observation specifications for the MDP."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy group."""

            # Base velocity
            base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
            base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            
            # Projected gravity
            projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

            # Joint positions and velocities
            joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

            # Velocity commands
            velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
            
            # Last actions
            actions = ObsTerm(func=isaac_mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class CriticCfg(ObsGroup):
            """Observations for critic group (privileged information)."""

            # Base velocity
            base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
            base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            
            # Projected gravity
            projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

            # Joint positions and velocities
            joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

            # Velocity commands
            velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
            
            # Last actions
            actions = ObsTerm(func=isaac_mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        # Define observation groups
        policy: PolicyCfg = PolicyCfg()
        critic: CriticCfg = CriticCfg()

    observations: ObservationsCfg = ObservationsCfg()

    # ========================================
    # Rewards
    # ========================================
    @configclass
    class RewardsCfg:
        """Reward specifications for the MDP."""

        # Reward for tracking velocity command
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

        # Reward for smooth actions
        lin_vel_z_l2 = RewTerm(func=isaac_mdp.lin_vel_z_l2, weight=-0.02)
        ang_vel_xy_l2 = RewTerm(func=isaac_mdp.ang_vel_xy_l2, weight=-0.01)
        action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.001)
        action_l2 = RewTerm(func=isaac_mdp.action_l2, weight=-0.001)

    rewards: RewardsCfg = RewardsCfg()

    # ========================================
    # Terminations
    # ========================================
    @configclass
    class TerminationsCfg:
        """Termination specifications for the MDP."""

        time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

    terminations: TerminationsCfg = TerminationsCfg()

    # ========================================
    # Curriculum
    # ========================================
    curriculum: dict = {}

    # ========================================
    # Post initialization
    # ========================================
    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # Update scene with robot config
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Override some settings
        self.viewer.eye = (7.5, 7.5, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
