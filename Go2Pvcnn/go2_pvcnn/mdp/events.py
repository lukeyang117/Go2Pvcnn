"""Event functions for Go2 PVCNN environment.

This module contains event functions for randomizing dynamic objects and robot states.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObjectCollection, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_dynamic_objects_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("dynamic_objects"),
    position_range: tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0),
    height_offset: float = 0.5,
    terrain_levels: tuple[int, int] = (1, 5),  # Only spawn objects in terrain levels 1-5
) -> None:
    """Reset dynamic objects to random positions based on terrain origins.
    
    This function randomizes the position and orientation of dynamic objects:
    * It samples positions uniformly within the given XY range relative to environment origins
    * It sets Z position to terrain height + height_offset
    * It samples random yaw orientations
    * It sets velocities to zero
    
    Similar to reset_root_state_uniform, this uses env.scene.env_origins for proper 
    terrain-aware positioning instead of robot-relative positioning.
    
    Args:
        env: Environment instance
        env_ids: Environment IDs to reset
        asset_cfg: Dynamic objects asset configuration
        position_range: XY range relative to terrain origin (x_min, x_max, y_min, y_max)
        height_offset: Height offset above terrain
    """
    # Extract the used quantities (to enable type-hinting)
    objects: RigidObjectCollection = env.scene[asset_cfg.name] 
    
    # Get number of objects per environment
    num_objects = objects.num_objects 
    
    # Get number of environments to reset
    num_envs = len(env_ids)
    
    # Get environment origins - Shape: (num_envs, 3)
    env_origins = env.scene.env_origins[env_ids]
    
    
    # Sample random XY positions within range
    x_min, x_max, y_min, y_max = position_range
    
    # Shape: (num_envs, num_objects, 2)
    random_xy = torch.rand(num_envs, num_objects, 2, device=objects.device)
    random_xy[:, :, 0] = random_xy[:, :, 0] * (x_max - x_min) + x_min
    random_xy[:, :, 1] = random_xy[:, :, 1] * (y_max - y_min) + y_min
    
    # Calculate absolute positions: env_origin + random_offset
    # Shape: (num_envs, num_objects, 3)
    positions = torch.zeros(num_envs, num_objects, 3, device=objects.device)
    positions[:, :, :2] = env_origins[:, None, :2] + random_xy
    positions[:, :, 2] = env_origins[:, None, 2] + height_offset
    
    
    # Sample random yaw orientations
    # Shape: (num_envs, num_objects)
    random_yaw = torch.rand(num_envs, num_objects, device=objects.device) * 2 * torch.pi
    
    # Convert to quaternions [qw, qx, qy, qz]
    # For yaw-only rotation: qw = cos(θ/2), qx = 0, qy = 0, qz = sin(θ/2)
    # Shape: (num_envs, num_objects, 4)
    orientations = torch.zeros(num_envs, num_objects, 4, device=objects.device)
    orientations[:, :, 0] = torch.cos(random_yaw / 2)  # qw
    orientations[:, :, 3] = torch.sin(random_yaw / 2)  # qz
    
    # Combine into pose [x, y, z, qw, qx, qy, qz]
    # Shape: (num_envs, num_objects, 7)
    object_pose = torch.cat([positions, orientations], dim=-1)
    
    # Set zero velocities [vx, vy, vz, wx, wy, wz]
    # Shape: (num_envs, num_objects, 6)
    velocities = torch.zeros(num_envs, num_objects, 6, device=objects.device)
    
    
    # Write to physics simulation
    objects.write_object_link_pose_to_sim(object_pose, env_ids=env_ids)
    objects.write_object_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_goal_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    goal_distance: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset goal positions for environments.
    
    This function resets the goal positions when episodes are reset (due to collision,
    timeout, or other termination conditions). It ensures each new episode starts with
    a goal at goal_distance meters ahead in the X direction.
    
    Args:
        env: Environment instance
        env_ids: Environment IDs to reset
        goal_distance: Distance to goal in X direction (meters). Default: 5.0m
        asset_cfg: Robot asset configuration
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w[env_ids, :2]  # (len(env_ids), 2) - [x, y]
    
    # Initialize goal_positions if not exists
    if not hasattr(env.unwrapped, 'goal_positions'):
        env.unwrapped.goal_positions = torch.zeros(env.num_envs, 2, device=env.device)
        env.unwrapped.goal_distance = goal_distance
    
    # Reset goals: current_position + [goal_distance, 0]
    env.unwrapped.goal_positions[env_ids, 0] = robot_pos[:, 0] + goal_distance
    env.unwrapped.goal_positions[env_ids, 1] = robot_pos[:, 1]
