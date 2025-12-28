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
    exclude_foot: bool = True,
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
    
    # 获取 force_matrix_w: [num_envs, 1, num_robot_bodies, 3]
    force_matrix = contact_sensor.data.force_matrix_w
    
    if force_matrix is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算每个body的接触力
    contact_force_norm = torch.norm(force_matrix, dim=-1)  # [num_envs, 1, num_robot_bodies]
    
    if exclude_foot:
        # 足端索引（在 filter list 中的位置）
        # 根据实际的body顺序：
        # FL_foot: 索引4, FR_foot: 索引8, RL_foot: 索引12, RR_foot: 索引16
        foot_indices = [4, 8, 12, 16]
        
        # 创建mask排除足端
        mask = torch.ones(contact_force_norm.shape[-1], device=env.device, dtype=torch.bool)
        for idx in foot_indices:
            if idx < contact_force_norm.shape[-1]:
                mask[idx] = False
        
        # 只检查非足端的碰撞
        contact_force_norm = contact_force_norm[..., mask]  # [num_envs, 1, num_non_foot_bodies]
    
    # 检测是否有任何body的接触力超过阈值
    has_collision = torch.any(contact_force_norm > threshold, dim=(-2, -1))  # [num_envs]
    
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
