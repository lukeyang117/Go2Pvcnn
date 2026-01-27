"""Reward functions for Go2 PVCNN locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking linear velocity command (xy axes) using exponential function."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking angular velocity command (z axis) using exponential function."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute projected gravity (gravity vector in world frame)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=asset.data.root_quat_w.device).repeat(env.num_envs, 1)
    projected_gravity = math_utils.quat_rotate_inverse(asset.data.root_quat_w, gravity_vec)
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize base height deviation from target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute height deviation
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (change in actions between timesteps)."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power (torque * velocity)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that are too close to limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def feet_air_time_positive_reward(
    env: ManagerBasedRLEnv, 
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward feet air time when in motion."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is below threshold
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0] > 1.0
    # reward feet that are in the air when in motion
    first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > threshold
    reward = torch.sum(first_contact.float() * ~in_contact.float(), dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding when in contact."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    # check contact
    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    # compute velocity
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    # penalize sliding
    reward = torch.sum(torch.norm(body_vel, dim=-1) * in_contact.float(), dim=1)
    return reward


def object_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_0_contact"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """检测动态物体与机器人的碰撞惩罚（可选排除足端）。
    
    使用物体的 ContactSensor（filter 机器人所有body），获取碰撞信息。
    
    Args:
        env: 环境实例
        sensor_cfg: 物体的接触传感器配置
        threshold: 接触力阈值 (N)
        exclude_foot: 是否排除足端的碰撞（默认True，只惩罚非足端碰撞）
    
    Returns:
        torch.Tensor: 惩罚值 [num_envs]，有碰撞为1.0，无碰撞为0.0
    """
    # 获取物体的接触传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取 force_matrix_w
    # 原始形状: [1, 1, num_envs * num_robot_bodies, 3]
    # 数据按环境顺序排列: (env_0的19个body, env_1的19个body, ..., env_255的19个body)
    force_matrix = contact_sensor.data.force_matrix_w
    
    if force_matrix is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 获取环境数量和机器人body数量
    num_envs = env.num_envs
    total_bodies = force_matrix.shape[2]  # 应该是 num_envs * num_robot_bodies
    num_robot_bodies = total_bodies // num_envs
    
    # Reshape 成正确的形状: [num_envs, num_robot_bodies, 3]
    force_reshaped = force_matrix.view(num_envs, num_robot_bodies, 3)
    
    # 计算每个body的接触力范数
    contact_force_norm = torch.norm(force_reshaped, dim=-1)  # [num_envs, num_robot_bodies]
    
   
    
    # 检测是否有任何body的接触力超过阈值
    has_collision = torch.any(contact_force_norm > threshold, dim=-1)  # [num_envs]
    
    return has_collision.float()


def non_foot_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """检测机器人非足端部位与地面的碰撞惩罚。
    
    使用机器人的 ContactSensor，通过接触力方向判断是否为地面碰撞
    （力主要向上），并排除足端。
    
    Args:
        env: 环境实例
        sensor_cfg: 机器人的接触传感器配置（监测所有body）
        threshold: 接触力阈值 (N)
    
    Returns:
        torch.Tensor: 惩罚值 [num_envs]，有碰撞为1.0，无碰撞为0.0
    """
    # 获取接触传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取所有body的接触力：[num_envs, num_bodies, 3]
    net_forces = contact_sensor.data.net_forces_w
    
    if net_forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算力的大小和z方向分量
    force_norm = torch.norm(net_forces, dim=-1)  # [num_envs, num_bodies]
    force_z = net_forces[..., 2]  # [num_envs, num_bodies]
    
    # 地面碰撞特征：力主要向上（z>0），且力足够大
    is_ground_contact = (force_z > 0.5 * force_norm) & (force_norm > threshold)
    
    
    # 获取足端body的索引（通过 body_names 匹配）
    # Go2机器人的足端：FL_foot, FR_foot, RL_foot, RR_foot
    robot = env.scene.articulations["robot"]
    body_names = robot.data.body_names
    
    # 查找足端索引
    foot_indices = []
    for i, name in enumerate(body_names):
        if "foot" in name.lower():
            foot_indices.append(i)
    
    # 如果找到了足端索引，创建mask排除它们
    if len(foot_indices) > 0:
        mask = torch.ones(is_ground_contact.shape[-1], device=env.device, dtype=torch.bool)
        for idx in foot_indices:
            if idx < mask.shape[0]:
                mask[idx] = False
        
        # 只检查非足端的地面碰撞
        non_foot_ground_contact = is_ground_contact[:, mask]  # [num_envs, num_non_foot_bodies]
    
    # 任意非足端body触地即返回惩罚
    has_collision = torch.any(non_foot_ground_contact, dim=-1)  # [num_envs]
    
    return has_collision.float()


def foot_cost_map_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 10.0,
    grid_resolution: float = 0.1,
    x_range: tuple = (-0.75, 0.75),
    y_range: tuple = (-0.75, 0.75),
) -> torch.Tensor:
    """根据足端落在代价地图上的位置给予惩罚。
    
    逻辑：
    1. 检测四足是否着地（接触力超过阈值）
    2. 对于着地的足端，获取其在机器人坐标系下的位置
    3. 将位置映射到代价地图的网格索引
    4. 从上一时刻保存的代价地图中查询该位置的代价值
    5. 根据代价值给予惩罚（代价越高惩罚越大）
    
    Args:
        env: 环境实例
        sensor_cfg: 足端的接触传感器配置
        asset_cfg: 机器人配置
        force_threshold: 判断足端着地的力阈值 (N)
        grid_resolution: 代价地图分辨率 (m/grid)
        x_range: 代价地图X轴范围 (m)
        y_range: 代价地图Y轴范围 (m)
    
    Returns:
        torch.Tensor: 惩罚值 [num_envs]，根据足端所在位置的代价值加权求和
    """
    # 获取接触传感器和机器人
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 获取上一时刻保存的代价地图 (在observations.py中保存)
    if not hasattr(env.unwrapped, 'last_cost_map_2d') or env.unwrapped.last_cost_map_2d is None:
        # 第一次调用时还没有代价地图，返回0惩罚
        return torch.zeros(env.num_envs, device=env.device)
    
    cost_map_2d = env.unwrapped.last_cost_map_2d  # [num_envs, H, W]
    H, W = cost_map_2d.shape[1], cost_map_2d.shape[2]
    
    # 获取足端body的索引和位置
    body_names = robot.data.body_names
    foot_indices = []
    for i, name in enumerate(body_names):
        if "foot" in name.lower():
            foot_indices.append(i)
    
    if len(foot_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 获取足端接触力 [num_envs, num_bodies, 3] -> [num_envs, num_feet, 3]
    net_forces = contact_sensor.data.net_forces_w[:, foot_indices, :]
    force_norm = torch.norm(net_forces, dim=-1)  # [num_envs, num_feet]
    
    # 判断足端是否着地
    is_contact = force_norm > force_threshold  # [num_envs, num_feet]
    
    # 获取足端在世界坐标系下的位置
    foot_pos_w = robot.data.body_pos_w[:, foot_indices, :]  # [num_envs, num_feet, 3]
    
    # 获取机器人base的位置和朝向
    base_pos_w = robot.data.root_pos_w  # [num_envs, 3]
    base_quat_w = robot.data.root_quat_w  # [num_envs, 4]
    
    # 将足端位置转换到机器人坐标系（yaw-aligned frame）
    # 只考虑yaw旋转，忽略roll和pitch
    # math_utils already imported at top of file
    
    # 计算从世界坐标到机器人坐标的变换
    foot_pos_rel = foot_pos_w - base_pos_w.unsqueeze(1)  # [num_envs, num_feet, 3]
    
    # 使用四元数逆变换将位置转到机器人坐标系
    foot_pos_b = math_utils.quat_rotate_inverse(
        base_quat_w.unsqueeze(1).expand(-1, len(foot_indices), -1).reshape(-1, 4),
        foot_pos_rel.reshape(-1, 3)
    ).reshape(env.num_envs, len(foot_indices), 3)
    
    # 提取XY坐标用于映射到网格
    foot_xy_b = foot_pos_b[:, :, :2]  # [num_envs, num_feet, 2]
    
    # 将XY坐标映射到网格索引
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_indices = ((foot_xy_b[:, :, 0] - x_min) / grid_resolution).long()  # [num_envs, num_feet]
    y_indices = ((foot_xy_b[:, :, 1] - y_min) / grid_resolution).long()  # [num_envs, num_feet]
    
    # 限制在有效范围内
    x_indices = torch.clamp(x_indices, 0, W - 1)
    y_indices = torch.clamp(y_indices, 0, H - 1)
    
    # 查询代价地图上的代价值
    # cost_map_2d: [num_envs, H, W]
    # 使用gather操作提取对应位置的代价
    batch_size = env.num_envs
    num_feet = len(foot_indices)
    
    penalty = torch.zeros(batch_size, device=env.device)
    
    for env_idx in range(batch_size):
        for foot_idx in range(num_feet):
            if is_contact[env_idx, foot_idx]:
                # 足端着地，查询代价
                grid_x = x_indices[env_idx, foot_idx]
                grid_y = y_indices[env_idx, foot_idx]
                cost_value = cost_map_2d[env_idx, grid_y, grid_x]
                
                # 累加惩罚（代价值已经是0-1范围）
                penalty[env_idx] += cost_value
    
    # 返回平均惩罚（除以足端数量避免过大）
    penalty = penalty / num_feet
    
    return penalty

