"""Configuration for Go2 quadruped with PVCNN-based perception."""

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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
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
# Terrain Configuration
##

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


##
# Pre-defined configs
##
from go2_pvcnn.assets import UNITREE_GO2_CFG  # isort: skip

##
# Scene definition
##

# 共享的机器人body filter列表（所有19个body）
# Shared robot body filter list for all object contact sensors (all 19 bodies)
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
class Go2SceneCfg(InteractiveSceneCfg):
    """Configuration for Go2 robot scene with LiDAR and dynamic objects."""

    # ========================================
    # Scene Replication Settings
    # ========================================
    
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

    # Height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # ========================================
    # Contact Sensors - 测试 PhysX 自动环境分组
    # ========================================
    # 假设验证：
    # 虽然 filter pattern 会匹配所有环境的机器人（{ENV_REGEX_NS}/Robot/*），
    # 但 PhysX 应该会根据 sensor prim 所在的环境自动将数据分组到
    # 正确的环境索引，而不是真的返回所有304个body的数据。
    # 
    # 如果这个假设正确，force_matrix_w 的形状应该是：
    #   [num_envs, 1, 19, 3]  而不是  [num_envs, 1, 304, 3]
    
    # 机器人的接触传感器（检测所有碰撞）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    
    # 为每个动态物体创建接触传感器（使用全局定义的 _ROBOT_BODY_FILTER）
    # Object_0: CrackerBox
    object_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=_ROBOT_BODY_FILTER,
    )
    
    # Object_1: SugarBox
    object_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=_ROBOT_BODY_FILTER,
    )
    
    # Object_2: TomatoSoupCan
    object_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_2",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=_ROBOT_BODY_FILTER,
    )

    # NOTE: LiDAR sensor configuration has been moved to go2_lidar_env_cfg.py
    # This old configuration is commented out to avoid import errors
    # lidar_sensor = RayCasterCfg(...)

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
    # ========================================
    # Dynamic Objects - 每个环境3个物体
    # ========================================
    # 注意：RigidObjectCollectionCfg会为每个环境复制物体
    # 3个物体 × 512个环境 = 1536个物体实例（总共）
    # Each environment will have 3 objects (CrackerBox, SugarBox, TomatoSoupCan)
    dynamic_objects = create_dynamic_objects_collection_cfg(num_objects=3)


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
        heading_command=True,  # 启用航向控制模式
        heading_control_stiffness=1.0,  # 航向误差到角速度的转换系数
        rel_standing_envs=0.02,  # 减少静止环境比例，鼓励运动
        rel_heading_envs=0.8,  # 80%的环境使用航向控制，20%使用直接角速度
        debug_vis=True,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), 
            lin_vel_y=(-0.1, 0.1), 
            ang_vel_z=(-1, 1),
            heading=(-math.pi, math.pi),  # 目标航向角范围：-π到π
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
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # NOTE: PVCNN observation commented out due to lidar_sensor migration
        # Cost map from PVCNN semantic segmentation + Height Scanner
        # Returns cost_map tensor: (batch, 256) = single channel × 16 × 16 grid
        # Uses height_scanner for ground truth height information
        # pvcnn_with_costmap = ObsTerm(
        #     func=custom_mdp.pvcnn_features_with_cost_map, 
        #     params={
        #         "sensor_cfg": SceneEntityCfg("lidar_sensor"),
        #         "height_scanner_cfg": SceneEntityCfg("height_scanner"),
        #     }
        # )

        # Base velocity in base frame
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

        # NOTE: PVCNN observation commented out due to lidar_sensor migration
        # Cost map from PVCNN semantic segmentation + Height Scanner (critic uses same observation)
        # pvcnn_with_costmap = ObsTerm(
        #     func=custom_mdp.pvcnn_features_with_cost_map, 
        #     params={
        #         "sensor_cfg": SceneEntityCfg("lidar_sensor"),
        #         "height_scanner_cfg": SceneEntityCfg("height_scanner"),
        #     }
        # )

        # Base velocity in base frame
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
@configclass
class EventCfg:
    """Configuration for randomization events."""

    # ========================================
    # Robot Reset Events
    # ========================================
    # Reset all environments when episode ends
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

    # Reset robot joints
    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # Reset dynamic objects positions based on terrain origins
    reset_objects_position = EventTerm(
        func=custom_mdp.reset_dynamic_objects_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dynamic_objects"),
            "position_range": (-5.0, 5.0, -5.0, 5.0),  # ±5m range relative to terrain origin
            "height_offset": 2.0,  # 2.0m above terrain
        },
    )

    # Push robot
    push_robot = EventTerm(
        func=isaac_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task rewards
    track_lin_vel_xy_exp = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp,
        weight=10.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=custom_mdp.flat_orientation_l2, weight=-0.5)
    joint_torques_l2 = RewTerm(func=custom_mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acc_l2 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7)
    joint_vel_l2 = RewTerm(func=custom_mdp.joint_vel_l2, weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01)
    joint_power = RewTerm(func=custom_mdp.joint_power, weight=-2.0e-5)
    joint_pos_limits = RewTerm(func=custom_mdp.joint_pos_limits, weight=-1.0)
    
    # ========================================
    # 碰撞惩罚系统 - 使用 ContactSensor（已解决环境隔离问题）
    # ========================================
    # 用户通过明确列出所有 robot body 路径解决了 PhysX filter 的环境隔离问题
    # 每个物体有独立的 ContactSensor，可精确获取碰撞信息
    
    # 非足端部位与地面碰撞惩罚
    body_terrain_collision = RewTerm(
        func=custom_mdp.non_foot_ground_contact_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 1.0,  # 接触力阈值
        },
    )
    
    # 机器人与动态物体碰撞惩罚（包括足端，不同物体不同惩罚强度）
    # CrackerBox 碰撞 - 最重惩罚
    collision_crackerbox = RewTerm(
        func=custom_mdp.object_contact_penalty,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("object_0_contact"),
            "threshold": 0.1,
            "exclude_foot": False,  # 不排除足端，所有部位碰撞都惩罚
        },
    )
    
    # SugarBox 碰撞 - 中等惩罚
    collision_sugarbox = RewTerm(
        func=custom_mdp.object_contact_penalty,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("object_1_contact"),
            "threshold": 0.1,
            "exclude_foot": False,
        },
    )
    
    # TomatoSoupCan 碰撞 - 最轻惩罚
    collision_tomatosoupcan = RewTerm(
        func=custom_mdp.object_contact_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("object_2_contact"),
            "threshold": 0.1,
            "exclude_foot": False,
        },
    )
    
    # 足端落在高代价区域的惩罚（基于上一时刻的代价地图）
    foot_step_on_cost = RewTerm(
        func=custom_mdp.foot_cost_map_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "force_threshold": 10.0,  # 足端接触力阈值 (N)
            "grid_resolution": 0.1,  # 与height_scanner一致
            "x_range": (-0.75, 0.75),  # 与cost_map_generator一致
            "y_range": (-0.75, 0.75),
        },
    )

    
    # -- optional: feet air time and sliding
    # feet_air_time_positive_reward = RewTerm(
    #     func=custom_mdp.feet_air_time_positive_reward,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # feet_slide = RewTerm(
    #     func=custom_mdp.feet_slide,
    #     weight=-0.1,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot")},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

    # bad_orientation = DoneTerm(
    #     func=custom_mdp.bad_orientation,
    #     params={"limit_angle": 1.0},
    # )

    # base_height = DoneTerm(
    #     func=custom_mdp.base_height,
    #     params={"minimum_height": 0.1},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


##
# Environment configuration
##


@configclass
class Go2PvcnnEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 quadruped environment with PVCNN."""

    # Scene settings
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)
    
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
        self.scene.robot = self._get_robot_cfg()
        
        print(f"[Go2PvcnnEnvCfg] 配置完成：每个环境3个动态物体，共{self.scene.num_envs}个环境")
        print(f"[Go2PvcnnEnvCfg] Total objects in all environments: {3 * self.scene.num_envs}")
        
    def _get_robot_cfg(self) -> ArticulationCfg:
        """Get the robot articulation configuration."""
        # Use the pre-configured Unitree Go2 from our assets
        return UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Go2PvcnnEnvCfg_PLAY(Go2PvcnnEnvCfg):
    """Configuration for playing/evaluation."""
    
    def __post_init__(self):
        # Post-initialization
        super().__post_init__()
        
        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # Remove random pushing for play
        self.events.push_robot = None
        
        print(f"[Go2PvcnnEnvCfg_PLAY] Configured for play mode with {3 * 50} dynamic objects")