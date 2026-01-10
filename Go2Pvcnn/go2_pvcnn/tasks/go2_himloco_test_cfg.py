"""
HIMLoco Test Environment Configuration

完全独立的配置，用于测试HIMLoco策略：
- 45维本体感觉观测（无PVCNN/LiDAR）
- 动态障碍物场景
- 平地+楼梯混合地形
- 直接继承ManagerBasedRLEnvCfg，参考训练代码结构
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

HIMLOCO_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),  # 每个子地形8x8米
    border_width=20.0,  # 边界宽度20米
    num_rows=3,  # 3个难度等级（简化测试）
    num_cols=5,  # 每个难度5种变体
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # 50% 平地
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        
        # 25% 金字塔楼梯（向上）
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.15),  # 台阶高度5-15cm
            step_width=0.3,  # 台阶宽度30cm
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        
        # 25% 倒金字塔楼梯（向下）
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.15),  # 台阶高度5-15cm
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


##
# Scene definition
##

@configclass  
class HimlocoSceneCfg(InteractiveSceneCfg):
    """HIMLoco场景配置：机器人 + 混合地形（平地+楼梯） + 动态物体"""
    
    # 混合地形（50%平地 + 50%楼梯）
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # 使用地形生成器
        terrain_generator=HIMLOCO_TERRAIN_CFG,  # 混合地形配置
        max_init_terrain_level=1,  # 初始难度等级
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
    robot: ArticulationCfg = MISSING
    
    # 动态物体
    dynamic_objects = create_dynamic_objects_collection_cfg(num_objects=3)
    
    # 接触传感器
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
    )
    object_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        history_length=3,
        track_air_time=False,
    )
    object_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_2",
        history_length=3,
        track_air_time=False,
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """动作配置"""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True
    )


@configclass  
class ObservationsCfg:
    """观测配置"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测：45维本体感觉"""
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=go2_mdp.goal_based_velocity_commands,
            params={"goal_distance": 5.0, "max_speed": 1.0}
        )
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        actions = ObsTerm(func=isaac_mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """命令配置 - 现在使用goal_based_velocity_commands，不再需要命令生成器"""
    pass


@configclass
class RewardsCfg:
    """奖励配置 - 简化用于测试"""
    # 惩罚大动作
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01
    )
    # 惩罚关节加速度
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7
    )
    # 存活奖励
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.5
    )


@configclass
class TerminationsCfg:
    """终止条件配置"""
    time_out = DoneTerm(
        func=isaac_mdp.time_out,
        time_out=True
    )
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0}
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
class Go2HimlocoTestEnvCfg(ManagerBasedRLEnvCfg):
    """HIMLoco测试环境配置"""
    
    # 场景设置
    scene = HimlocoSceneCfg(num_envs=4, env_spacing=2.5)
    
    # MDP设置
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()
    
    def __post_init__(self):
        """后初始化"""
        # 基础设置
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # 仿真设置
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # 设置机器人
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        print(f"[HIMLoco] 配置完成：{self.scene.num_envs}个环境，每个3个动态物体")


@configclass
class Go2HimlocoTestEnvCfg_PLAY(Go2HimlocoTestEnvCfg):
    """测试模式配置"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 更小的环境数
        self.scene.num_envs = 2
        
        # 禁用随机化
        self.observations.policy.enable_corruption = False
        
        # 移除随机推动
        self.events.reset_base = None
        
        print(f"[HIMLoco Play] 测试模式：{self.scene.num_envs}个环境")
