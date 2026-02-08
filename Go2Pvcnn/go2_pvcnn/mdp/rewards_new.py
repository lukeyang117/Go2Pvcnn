"""Reward functions for Go2 Teacher training with semantic cost map."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

# 复用原有的基础reward函数
from .rewards import (
    track_lin_vel_xy_exp,
    track_ang_vel_z_exp,
    flat_orientation_l2,
    base_height_l2,
    joint_torques_l2,
    joint_acc_l2,
    action_rate_l2,
    joint_vel_l2,
    joint_power,
    joint_pos_limits,
    feet_air_time_positive_reward,
    feet_slide,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def semantic_cost_map_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_ground"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """
    根据足端落在语义代价地图上的位置给予惩罚（Teacher模式）
    
    逻辑：
    1. 检测四足是否着地
    2. 获取着地足端在机器人坐标系下的位置
    3. 映射到代价地图网格索引
    4. 从env中读取cost_map查询代价值
    5. 累加所有着地足端的代价作为惩罚
    
    Args:
        env: 环境实例
        sensor_cfg: 足端接触传感器配置
        asset_cfg: 机器人配置
        force_threshold: 判断足端着地的力阈值 (N)
    
    Returns:
        torch.Tensor: 惩罚值 [num_envs]，着地足端代价的平均值
    """
    # 检查是否有cost_map（需要在observation中生成并存储到env）
    if not hasattr(env, 'cost_map') or env.cost_map is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 获取contact sensor和robot
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 获取足端接触力 [num_envs, num_feet, 3]
    net_forces_w = contact_sensor.data.net_forces_w  # World frame
    
    if net_forces_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算接触力大小
    force_magnitude = torch.norm(net_forces_w, dim=-1)  # [num_envs, num_feet]
    
    # 判断是否着地
    is_contact = force_magnitude > force_threshold  # [num_envs, num_feet]
    
    # 获取cost_map的参数
    cost_map = env.cost_map  # [num_envs, H, W]
    H, W = cost_map.shape[1], cost_map.shape[2]
    
    # 获取足端在机器人坐标系下的位置
    # 通过body_ids获取足端位置
    foot_body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else list(range(net_forces_w.shape[1]))
    foot_pos_w = robot.data.body_pos_w[:, foot_body_ids, :]  # [num_envs, num_feet, 3]
    
    # 转换到机器人坐标系
    robot_pos_w = robot.data.root_pos_w  # [num_envs, 3]
    robot_quat_w = robot.data.root_quat_w  # [num_envs, 4]
    
    # 相对位置（世界坐标）
    foot_pos_rel_w = foot_pos_w - robot_pos_w.unsqueeze(1)  # [num_envs, num_feet, 3]
    
    # 转换到机器人坐标系（base frame）
    foot_pos_b = math_utils.quat_rotate_inverse(
        robot_quat_w.unsqueeze(1), foot_pos_rel_w
    )  # [num_envs, num_feet, 3]
    
    # 提取XY坐标
    foot_x = foot_pos_b[:, :, 0]  # [num_envs, num_feet]
    foot_y = foot_pos_b[:, :, 1]
    
    # 从env获取cost_map的物理范围（需要与cost_map_generator一致）
    # 默认使用与height_scanner相同的范围
    x_range = (-0.75, 0.75) if not hasattr(env, 'cost_map_x_range') else env.cost_map_x_range
    y_range = (-0.75, 0.75) if not hasattr(env, 'cost_map_y_range') else env.cost_map_y_range
    grid_resolution = 0.1 if not hasattr(env, 'cost_map_resolution') else env.cost_map_resolution
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # 映射到网格索引
    x_indices = ((foot_x - x_min) / grid_resolution).long()  # [num_envs, num_feet]
    y_indices = ((foot_y - y_min) / grid_resolution).long()
    
    # 限制在有效范围内
    x_indices = torch.clamp(x_indices, 0, W - 1)
    y_indices = torch.clamp(y_indices, 0, H - 1)
    
    # 查询cost_map的代价值
    batch_indices = torch.arange(env.num_envs, device=env.device).unsqueeze(1).expand_as(x_indices)
    foot_costs = cost_map[batch_indices, y_indices, x_indices]  # [num_envs, num_feet]
    
    # 只惩罚着地的足端
    valid_costs = foot_costs * is_contact.float()  # [num_envs, num_feet]
    
    # 计算平均代价（所有着地足端）
    num_contact_feet = is_contact.sum(dim=1).float() + 1e-6  # 避免除零
    avg_cost = valid_costs.sum(dim=1) / num_contact_feet  # [num_envs]
    
    return avg_cost


def obstacle_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_small_objects"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    检测与动态障碍物的碰撞惩罚
    
    使用专门的contact sensor（filter小物体）检测碰撞
    
    Args:
        env: 环境实例
        sensor_cfg: 小物体接触传感器配置
        threshold: 接触力阈值 (N)
    
    Returns:
        torch.Tensor: 惩罚值 [num_envs]，有碰撞为1.0
    """
    # 获取小物体接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取接触力
    net_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    
    if net_forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算力的大小
    force_magnitude = torch.norm(net_forces, dim=-1)  # [num_envs, num_bodies]
    
    # 检测是否有任何body超过阈值
    has_collision = torch.any(force_magnitude > threshold, dim=-1)  # [num_envs]
    
    return has_collision.float()


def furniture_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_furniture"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    检测与家具/valuable物体的碰撞惩罚
    
    使用专门的contact sensor（filter家具）检测碰撞
    
    Args:
        env: 环境实例
        sensor_cfg: 家具接触传感器配置
        threshold: 接触力阈值 (N)
    
    Returns:
        torch.Tensor: 惩罚值 [num_envs]，有碰撞为1.0
    """
    # 获取家具接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取接触力
    net_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    
    if net_forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算力的大小
    force_magnitude = torch.norm(net_forces, dim=-1)  # [num_envs, num_bodies]
    
    # 检测是否有任何body超过阈值
    has_collision = torch.any(force_magnitude > threshold, dim=-1)  # [num_envs]
    
    return has_collision.float()


def non_foot_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    检测机器人非足端部位与地面的碰撞惩罚
    
    使用机器人的ContactSensor，通过接触力方向判断是否为地面碰撞
    
    Args:
        env: 环境实例
        sensor_cfg: 机器人的接触传感器配置
        threshold: 接触力阈值 (N)
    
    Returns:
        torch.Tensor: 惩罚值 [num_envs]，有碰撞为1.0
    """
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取所有body的接触力
    net_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    
    if net_forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算力的大小和z方向分量
    force_norm = torch.norm(net_forces, dim=-1)  # [num_envs, num_bodies]
    force_z = net_forces[..., 2]
    
    # 地面碰撞特征：力主要向上（z>0）
    is_ground_contact = (force_z > 0.5 * force_norm) & (force_norm > threshold)
    
    # 获取足端索引并排除
    robot: Articulation = env.scene["robot"]
    body_names = robot.data.body_names
    
    # 查找足端索引
    foot_indices = []
    for i, name in enumerate(body_names):
        if "foot" in name.lower():
            foot_indices.append(i)
    
    # 排除足端
    if len(foot_indices) > 0:
        mask = torch.ones(is_ground_contact.shape[-1], device=env.device, dtype=torch.bool)
        for idx in foot_indices:
            if idx < mask.shape[0]:
                mask[idx] = False
        
        non_foot_ground_contact = is_ground_contact[:, mask]
    else:
        non_foot_ground_contact = is_ground_contact
    
    # 任意非足端body触地即返回惩罚
    has_collision = torch.any(non_foot_ground_contact, dim=-1)  # [num_envs]
    
    return has_collision.float()


def command_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    奖励机器人朝向command方向移动
    
    计算实际速度方向与目标速度方向的对齐度
    
    Args:
        env: 环境实例
        command_name: Command管理器中的指令名称
        asset_cfg: 机器人配置
    
    Returns:
        torch.Tensor: 奖励值 [num_envs]，对齐度越高奖励越大
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 获取command和实际速度
    command_vel = env.command_manager.get_command(command_name)[:, :2]  # [num_envs, 2] (vx, vy)
    actual_vel = robot.data.root_lin_vel_b[:, :2]  # [num_envs, 2]
    
    # 计算方向对齐度（归一化后的点积）
    command_norm = torch.norm(command_vel, dim=1, keepdim=True) + 1e-6
    actual_norm = torch.norm(actual_vel, dim=1, keepdim=True) + 1e-6
    
    command_dir = command_vel / command_norm
    actual_dir = actual_vel / actual_norm
    
    # 点积（余弦相似度）
    alignment = torch.sum(command_dir * actual_dir, dim=1)  # [num_envs]
    
    # 转换到[0, 1]范围（-1到1 -> 0到1）
    alignment = (alignment + 1.0) / 2.0
    
    # 只在有速度指令时奖励
    has_command = torch.norm(command_vel, dim=1) > 0.1
    alignment = alignment * has_command.float()
    
    return alignment
