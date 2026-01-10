"""
ABS (Adaptive Behavior System) Test Environment Configuration

ABS模型特点：
- 带视觉感知（LiDAR点云）
- 本体感觉观测
- 动态障碍物场景
- 平地+楼梯混合地形
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.envs import mdp as isaac_mdp
import go2_pvcnn.mdp as go2_mdp

import go2_pvcnn.mdp as mdp
from go2_pvcnn.assets import UNITREE_GO2_CFG
from go2_pvcnn.mdp import create_dynamic_objects_collection_cfg


##
# Terrain Configuration
##

ABS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=3,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


# Shared robot body filter for collision detection
_ROBOT_BODY_FILTER = [
    "{ENV_REGEX_NS}/Robot/base",
    "{ENV_REGEX_NS}/Robot/FL_hip", "{ENV_REGEX_NS}/Robot/FL_thigh", 
    "{ENV_REGEX_NS}/Robot/FL_calf", "{ENV_REGEX_NS}/Robot/FL_foot",
    "{ENV_REGEX_NS}/Robot/FR_hip", "{ENV_REGEX_NS}/Robot/FR_thigh",
    "{ENV_REGEX_NS}/Robot/FR_calf", "{ENV_REGEX_NS}/Robot/FR_foot",
    "{ENV_REGEX_NS}/Robot/RL_hip", "{ENV_REGEX_NS}/Robot/RL_thigh",
    "{ENV_REGEX_NS}/Robot/RL_calf", "{ENV_REGEX_NS}/Robot/RL_foot",
    "{ENV_REGEX_NS}/Robot/RR_hip", "{ENV_REGEX_NS}/Robot/RR_thigh",
    "{ENV_REGEX_NS}/Robot/RR_calf", "{ENV_REGEX_NS}/Robot/RR_foot",
    "{ENV_REGEX_NS}/Robot/Head_upper", "{ENV_REGEX_NS}/Robot/Head_lower",
]


##
# Scene definition
##

@configclass  
class AbsSceneCfg(InteractiveSceneCfg):
    """ABS场景配置：机器人 + 混合地形 + 动态物体 + LiDAR"""
    
    # 混合地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ABS_TERRAIN_CFG,
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
    
    # 机器人
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 动态物体（3个YCB物体）
    dynamic_objects = create_dynamic_objects_collection_cfg()
    
    # Ray2D传感器 - ABS的视觉感知（2D射线扫描）
    # 11条射线，-45°到45°范围扫描障碍物
    # 对应ABS训练代码：theta从-π/4到π/4，步长π/20，共11条射线
    ray2d_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.05, 0.0, 0.0)),  # x_0=-0.05m, y_0=0.0m（机器人前方）
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,  # 单层2D扫描
            vertical_fov_range=(0.0, 0.0),  # 水平方向
            horizontal_fov_range=(-45.0, 45.0),  # -π/4到π/4（度数）
            horizontal_res=8.5,  # ceil(90/8.5)=11条射线（torch.linspace包含两端）
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=6.0,  # 对应训练代码的max_dist=6.0
    )
    
    # 碰撞传感器（监听所有body，但观测时只使用脚部）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    
    # 物体碰撞传感器
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


##
# MDP settings
##

@configclass  
class ActionsCfg:
    """动作配置"""
    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass  
class ObservationsCfg:
    """观测配置"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测：本体感觉 + Ray2D射线扫描
        
        观测维度分解（与ABS训练配置对齐）：
        - contact: 4维 (4个脚的接触状态)
        - base_ang_vel: 3维
        - projected_gravity: 3维
        - commands: 3维 (vx, vy, omega_z)
        - timer: 1维
        - joint_pos: 12维
        - joint_vel: 12维
        - actions: 12维
        - ray2d: 11维 (11条射线距离)
        总计: 4+3+3+3+1+12+12+12+11 = 61维
        """
        # 接触传感器（4个脚）- 二值化接触状态 (+1接触, -1未接触)
        contact = ObsTerm(
            func=go2_mdp.binary_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0}
        )
        
        # 本体感觉
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=go2_mdp.goal_based_velocity_commands,
            params={"goal_distance": 5.0, "max_speed": 1.0}
        )
        
        # 计时器（episode进度，0到1）
        timer = ObsTerm(
            func=lambda env: (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)
        )
        
        # 关节状态
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        actions = ObsTerm(func=isaac_mdp.last_action)
        
        # Ray2D射线扫描（视觉感知，11维）
        ray2d = ObsTerm(
            func=go2_mdp.ray2d_obstacle_distance,
            params={
                "sensor_cfg": SceneEntityCfg("ray2d_sensor"),
                "log2_scale": True,  # 使用log2缩放，与训练一致
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """奖励配置（测试用，权重较小）"""
    # 惩罚大动作
    action_rate_l2 = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=-0.01
    )
    # 惩罚关节加速度
    joint_acc_l2 = RewTerm(
        func=isaac_mdp.joint_acc_l2,
        weight=-2.5e-7
    )
    # 存活奖励
    is_alive = RewTerm(
        func=isaac_mdp.is_alive,
        weight=0.5
    )


@configclass
class TerminationsCfg:
    """终止条件"""
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class EventCfg:
    """事件配置"""
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
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
    reset_scene_objects = EventTerm(
        func=mdp.reset_dynamic_objects_position,
        mode="reset",
    )
    reset_goal_positions = EventTerm(
        func=mdp.reset_goal_positions,
        mode="reset",
        params={"goal_distance": 5.0},
    )


@configclass
class CurriculumCfg:
    """课程学习配置（空）"""
    pass


##
# Environment configuration
##

@configclass
class Go2AbsTestEnvCfg(ManagerBasedRLEnvCfg):
    """ABS测试环境配置"""
    
    # 场景配置
    scene: AbsSceneCfg = AbsSceneCfg(num_envs=4, env_spacing=4.0)
    
    # MDP配置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    # 环境设置
    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class Go2AbsTestEnvCfg_PLAY(Go2AbsTestEnvCfg):
    """ABS测试环境配置（Play模式）"""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.episode_length_s = 30.0
