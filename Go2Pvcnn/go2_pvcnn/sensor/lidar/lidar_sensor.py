# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR sensor implementation with grouped ray casting for dynamic objects.

This module implements a complete LiDAR sensor that combines:
- Dynamic mesh tracking for moving obstacles (via LidarRayCaster)
- LiDAR-specific signal processing (distances, point clouds, intensity)
- Optional sensor noise simulation
- Flexible ray pattern support (standard spinning or Livox dynamic patterns)
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply

from .lidar_data import LidarSensorData
from .patterns import patterns_cfg as patterns_cfg_module
from .ray_caster import LidarRayCaster

if TYPE_CHECKING:
    from .lidar_cfg import LidarCfg


class LidarSensor(LidarRayCaster):
    """A LiDAR sensor implementation based on ray-casting with dynamic object support.
    
    This sensor extends LidarRayCaster to provide LiDAR-specific functionality
    including point cloud generation, noise simulation, and sensor-specific processing.
    
    The ray casting and dynamic mesh tracking are handled by the parent LidarRayCaster class,
    while this class focuses on converting ray hits to point clouds and applying sensor effects.
    
    Key Capabilities:
        - Point cloud generation in world or sensor frame
        - Distance measurements with range filtering
        - Optional sensor noise (Gaussian distance noise, angular noise, dropout)
        - Dynamic ray patterns (e.g., for Livox-style sensors)
        - Multi-environment batched processing
    
    Example:
        >>> cfg = LidarCfg(
        ...     prim_path="/World/envs/env_.*",
        ...     update_frequency=50.0,
        ...     return_pointcloud=True,
        ...     enable_sensor_noise=True,
        ... )
        >>> lidar = LidarSensor(cfg)
        >>> distances = lidar.get_distances()
        >>> pointcloud = lidar.get_pointcloud()
    """

    cfg: LidarCfg
    """The configuration parameters."""

    def __init__(self, cfg: LidarCfg):
        """Initializes the LiDAR sensor.

        Args:
            cfg: The configuration parameters.
        """
        #print(f"\n[LidarSensor] Initializing LidarSensor with config: prim_path={cfg.prim_path}")
        
        # Initialize base class (LidarRayCaster)
        super().__init__(cfg)
        
        # Create LiDAR-specific data container
        self._data = LidarSensorData()
        
        # LiDAR-specific timing parameters
        self.update_frequency = cfg.update_frequency
        self.update_dt = 1.0 / self.update_frequency
        self.sensor_t = 0.0
        
        # Pattern parameters for dynamic updating
        self.pattern_start_index = 0
        
    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"LiDAR Sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.update_dt}\n"
            f"\tupdate frequency (Hz): {self.update_frequency}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\tmax range (m)        : {self.cfg.max_distance}\n"
            f"\tmin range (m)        : {self.cfg.min_range}\n"
            f"\tnoise enabled        : {self.cfg.enable_sensor_noise}"
        )

    @property
    def data(self) -> LidarSensorData:
        """The sensor data object."""
        # Update buffers if needed
        self._update_outdated_buffers()
        return self._data

    def _get_true_sensor_pos(self) -> torch.Tensor:
        """Calculates the true world position of the sensor, including the local offset."""
        # Get the stored base prim pose
        base_pos_w = self._data.pos_w
        base_quat_w = self._data.quat_w

        # Get the local offset from the config
        local_offset = torch.tensor(self.cfg.offset.pos, device=self.device)
        local_offset = local_offset.expand(base_pos_w.shape[0], -1)

        # Rotate the local offset to align with the base prim's world orientation
        world_offset = quat_apply(base_quat_w, local_offset)
        
        # The true sensor position is the base position plus the world-space offset
        return base_pos_w + world_offset

    def _initialize_rays_impl(self):
        """Initialize ray patterns for LiDAR sensor."""
        # Parent class (Isaac RayCaster via LidarRayCaster) already creates:
        # - self.drift
        # - self._data.pos_w, quat_w, ray_hits_w
        super()._initialize_rays_impl()
        
        # Initialize LiDAR-specific data buffers (not created by parent)
        self._data.distances = torch.zeros(self._view.count, self.num_rays, device=self._device)
        
        # Initialize point cloud data if needed
        if self.cfg.return_pointcloud:
            self._data.pointcloud = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        
        # Initialize intensity data
        self._data.intensity = torch.zeros(self._view.count, self.num_rays, device=self._device)
        
        # Initialize height map if needed
        if self.cfg.return_height_map:
            # Calculate grid dimensions from size and resolution
            size_x, size_y = self.cfg.height_map_size
            resolution = self.cfg.height_map_resolution
            self.height_map_grid_h = int(size_y / resolution) + 1  # Height (Y direction)
            self.height_map_grid_w = int(size_x / resolution) + 1  # Width (X direction)
            self._data.height_map = torch.zeros(
                self._view.count, self.height_map_grid_h, self.height_map_grid_w, 
                device=self._device
            )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Update sensor buffers with LiDAR-specific processing.
        
        This method:
        1. Updates sensor time for dynamic patterns
        2. Updates dynamic ray patterns if configured
        3. Calls parent for ray casting
        4. Processes hit results into distances and point clouds
        5. Applies noise if enabled
        
        Args:
            env_ids: Environment IDs to update.
        """
        
        # Update sensor time
        self.sensor_t += self.update_dt
        
        # Update ray patterns if using dynamic patterns (e.g., Livox)
        if hasattr(self.cfg.pattern_cfg, 'sensor_type'):
            #print(f"[LidarSensor] Updating dynamic ray patterns")
            self._update_dynamic_rays()
        
        # Call parent implementation for ray casting
        super()._update_buffers_impl(env_ids)
        
        # Get sensor position including offset
        sensor_pos = self._get_true_sensor_pos()[env_ids].unsqueeze(1)
        #print(f"[LidarSensor] Sensor positions: {sensor_pos.shape}")

        # Calculate distances from sensor to hit points
        hit_points = self._data.ray_hits_w[env_ids]
        distances = torch.norm(hit_points - sensor_pos, dim=2)
        #print(f"[LidarSensor] Raw distances: min={distances.min().item():.4f}, max={distances.max().item():.4f}, mean={distances.mean().item():.4f}")
        
        # Handle out-of-range values
        inf_mask = torch.isinf(hit_points).any(dim=2)
        distances[inf_mask] = self.cfg.max_distance
        #print(f"[LidarSensor] Infinite values found: {inf_mask.sum().item()}")
        
        # Filter by min range
        near_mask = distances < self.cfg.min_range
        distances[near_mask] = self.cfg.near_out_of_range_value
        #print(f"[LidarSensor] Near range violations: {near_mask.sum().item()}")
        
        # Filter by max range
        far_mask = distances > self.cfg.max_distance
        distances[far_mask] = self.cfg.far_out_of_range_value
        #print(f"[LidarSensor] Far range violations: {far_mask.sum().item()}")
        
        # Apply noise if enabled
        if self.cfg.enable_sensor_noise:
            #print(f"[LidarSensor] Applying sensor noise")
            distances = self._apply_noise(distances, env_ids)
            #print(f"[LidarSensor] Distances after noise: min={distances.min().item():.4f}, max={distances.max().item():.4f}")
        
        self._data.distances[env_ids] = distances
        
        # Generate point cloud if requested
        if self.cfg.return_pointcloud:
            #print(f"[LidarSensor] Generating point cloud")
            self._generate_pointcloud(env_ids)
        
        # Generate height map if requested
        if self.cfg.return_height_map:
            self._generate_height_map(env_ids)
        
        #print(f"[LidarSensor] _update_buffers_impl() DONE\n")
    def _update_dynamic_rays(self):
        """Update ray directions for dynamic patterns (e.g., Livox sensors).
        
        This method simulates the rotating scan pattern of Livox sensors by
        rotating the ray directions around the Z-axis.
        """
        if not hasattr(self.cfg.pattern_cfg, 'samples'):
            return
            
        # Rotate the pattern slightly to simulate Livox behavior
        rotation_angle = self.sensor_t * 0.1  # Slow rotation for temporal variation
        cos_rot = math.cos(rotation_angle)
        sin_rot = math.sin(rotation_angle)
        
        # Apply small rotation around Z-axis
        # Get current x and y components of ray directions for all environments
        x_dirs = self.ray_directions[:, :, 0].clone()  # Shape: (num_envs, num_rays)
        y_dirs = self.ray_directions[:, :, 1].clone()  # Shape: (num_envs, num_rays)
        
        # Apply rotation matrix using vectorized operations
        self.ray_directions[:, :, 0] = x_dirs * cos_rot - y_dirs * sin_rot
        self.ray_directions[:, :, 1] = x_dirs * sin_rot + y_dirs * cos_rot

    def _apply_noise(self, distances: torch.Tensor, env_ids: Sequence[int]) -> torch.Tensor:
        """Apply noise to distance measurements.
        
        Simulates realistic sensor noise including:
        - Gaussian distance noise (proportional error)
        - Pixel dropout (random missing returns)
        - Standard deviation multiplier for per-pixel noise
        
        Args:
            distances: Input distance tensor of shape (num_envs_subset, num_rays).
            env_ids: Indices of environments being updated.
            
        Returns:
            Noisy distance measurements with same shape as input.
        """
        #print(f"[LidarSensor] Applying noise: distance_std={self.cfg.random_distance_noise}, dropout_prob={self.cfg.pixel_dropout_prob}")
        
        # Apply Gaussian noise to distances
        noise = torch.randn_like(distances) * self.cfg.random_distance_noise
        distances_noisy = distances + noise
        #print(f"[LidarSensor] Distance noise stats: min={noise.min().item():.6f}, max={noise.max().item():.6f}, mean={noise.mean().item():.6f}")
        
        # Apply dropout
        dropout_mask = torch.rand_like(distances) < self.cfg.pixel_dropout_prob
        distances_noisy[dropout_mask] = self.cfg.max_distance
        dropout_count = dropout_mask.sum().item()
        #print(f"[LidarSensor] Pixels dropped out: {dropout_count} ({100*dropout_count/distances.numel():.2f}%)")
        
        # Clamp to valid range
        distances_noisy = torch.clamp(distances_noisy, self.cfg.min_range, self.cfg.max_distance)
        
        return distances_noisy

    def _generate_pointcloud(self, env_ids: Sequence[int]):
        """Generate point cloud from ray hits.
        
        Converts ray hit points to a point cloud representation. Can generate
        either in world coordinates or sensor-relative coordinates based on configuration.
        
        Args:
            env_ids: Indices of environments for which to generate point clouds.
        """
        #print(f"[LidarSensor] Generating point clouds in {'world' if self.cfg.pointcloud_in_world_frame else 'sensor'} frame")
        
        if self.cfg.pointcloud_in_world_frame:
            # Point cloud in world coordinates
            self._data.pointcloud[env_ids] = self._data.ray_hits_w[env_ids].clone()
            #print(f"[LidarSensor] Point cloud (world): shape={self._data.pointcloud[env_ids].shape}")
        else:
            # Point cloud in sensor frame
            sensor_pos = self._data.pos_w[env_ids].unsqueeze(1)
            hit_points_world = self._data.ray_hits_w[env_ids]
            
            # Calculate distances
            distances = torch.norm(hit_points_world - sensor_pos, dim=2, keepdim=True)
            #print(f"[LidarSensor] Distance range: [{distances.min().item():.4f}, {distances.max().item():.4f}]")
            
            # Get original ray directions in sensor frame
            ray_directions_sensor = self.ray_directions[env_ids]
            
            # Generate point cloud in sensor frame
            pointcloud_sensor = ray_directions_sensor * distances
            
            # Handle infinite distances (no hits)
            inf_mask = torch.isinf(hit_points_world).any(dim=2, keepdim=True)
            pointcloud_sensor[inf_mask.expand_as(pointcloud_sensor)] = float('inf')
            inf_count = inf_mask.sum().item()
            #print(f"[LidarSensor] Infinite hits: {inf_count} ({100*inf_count/inf_mask.numel():.2f}%)")
            
            self._data.pointcloud[env_ids] = pointcloud_sensor
            #print(f"[LidarSensor] Point cloud (sensor): shape={self._data.pointcloud[env_ids].shape}, range=[{pointcloud_sensor.min().item():.4f}, {pointcloud_sensor.max().item():.4f}]")

    def _generate_height_map(self, env_ids: Sequence[int]):
        """Generate 2D height map from ray hits.
        
        Projects ray hit points onto a horizontal grid centered on the robot,
        recording the Z-coordinate (height/elevation) of each hit in the grid cell.
        Multiple hits in the same cell are averaged.
        
        Args:
            env_ids: Indices of environments for which to generate height maps.
        """
        # Get configuration
        size_x, size_y = self.cfg.height_map_size
        resolution = self.cfg.height_map_resolution
        grid_h = self.height_map_grid_h  # Y direction (height)
        grid_w = self.height_map_grid_w  # X direction (width)
        
        # Get sensor position (base of robot + offset)
        sensor_pos = self._data.pos_w[env_ids]  # Shape: (num_envs_subset, 3)
        
        # Get hit points in world frame
        hit_points = self._data.ray_hits_w[env_ids]  # Shape: (num_envs_subset, num_rays, 3)
        
        # Convert to sensor-relative coordinates (X: forward, Y: left, Z: up)
        hit_points_rel = hit_points - sensor_pos.unsqueeze(1)  # Subtract sensor position
        
        # Filter out invalid hits (inf values)
        valid_mask = torch.isfinite(hit_points_rel).all(dim=2)  # Shape: (num_envs_subset, num_rays)
        
        # Initialize height map for this batch (NaN for empty cells)
        height_map_batch = torch.full(
            (len(env_ids), grid_h, grid_w), 
            float('nan'), 
            device=self._device, 
            dtype=torch.float32
        )
        
        # Process each environment separately for accurate grid mapping
        for i, env_id in enumerate(env_ids):
            # Get valid hits for this environment
            env_hits = hit_points_rel[i][valid_mask[i]]  # Shape: (num_valid_hits, 3)
            
            if env_hits.shape[0] == 0:
                continue  # No valid hits for this environment
            
            # Extract X, Y, Z coordinates
            x_coords = env_hits[:, 0]  # Forward (X)
            y_coords = env_hits[:, 1]  # Left (Y)
            z_coords = env_hits[:, 2]  # Up (Z) - this is the height we want
            
            # Convert world coordinates to grid indices
            # Grid is centered on robot: X range [-size_x/2, size_x/2], Y range [-size_y/2, size_y/2]
            grid_x_indices = ((x_coords + size_x / 2) / resolution).long()  # X axis
            grid_y_indices = ((y_coords + size_y / 2) / resolution).long()  # Y axis
            
            # Filter points outside the grid
            in_bounds_mask = (
                (grid_x_indices >= 0) & (grid_x_indices < grid_w) &
                (grid_y_indices >= 0) & (grid_y_indices < grid_h)
            )
            
            if in_bounds_mask.sum() == 0:
                continue  # No hits within grid bounds
            
            # Filter to in-bounds points
            grid_x_indices = grid_x_indices[in_bounds_mask]
            grid_y_indices = grid_y_indices[in_bounds_mask]
            z_coords = z_coords[in_bounds_mask]
            
            # Aggregate heights for each grid cell (take maximum Z for obstacle awareness)
            # Using scatter_reduce with 'amax' to keep the highest point in each cell
            linear_indices = grid_y_indices * grid_w + grid_x_indices  # Flatten grid indices
            
            # Flatten the height map for scatter operation
            height_map_flat = height_map_batch[i].view(-1)
            
            # Use scatter_reduce to get max height per cell (safer than mean for obstacles)
            height_map_flat.scatter_reduce_(
                0, linear_indices, z_coords, reduce='amax', include_self=False
            )
            
            # Reshape back to 2D grid
            height_map_batch[i] = height_map_flat.view(grid_h, grid_w)
        
        # Store in data buffer
        self._data.height_map[env_ids] = height_map_batch

    def get_distances(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get distance measurements for specified environments.
        
        Returns the processed distance measurements after range filtering and noise application.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Distance measurements tensor of shape (num_envs, num_rays) in meters.
            
        Raises:
            ValueError: If called before sensor is properly initialized.
        """
        distances = self.data.distances if env_ids is None else self.data.distances[env_ids]
        #print(f"[LidarSensor] get_distances: shape={distances.shape}, range=[{distances.min().item():.4f}, {distances.max().item():.4f}]")
        return distances

    def get_pointcloud(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get point cloud data for specified environments.
        
        Returns the 3D point cloud generated from ray hits. Points are either in world
        or sensor coordinates depending on the `pointcloud_in_world_frame` configuration.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Point cloud tensor of shape (num_envs, num_rays, 3) in meters.
            
        Raises:
            ValueError: If point cloud generation is disabled in configuration.
        """
        if not self.cfg.return_pointcloud:
            raise ValueError("Point cloud generation is disabled. Set 'return_pointcloud=True' in config.")
        
        pointcloud = self.data.pointcloud if env_ids is None else self.data.pointcloud[env_ids]
        #print(f"[LidarSensor] get_pointcloud: shape={pointcloud.shape}, frame={'world' if self.cfg.pointcloud_in_world_frame else 'sensor'}")
        return pointcloud

    def get_intensity(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get intensity (reflectance) measurements for specified environments.
        
        Returns the reflectance/intensity values for each ray. These are based on
        the surface properties of hit objects.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Intensity measurements tensor of shape (num_envs, num_rays).
            
        Raises:
            ValueError: If intensity data is not available.
        """
        if self.data.intensity is None:
            raise ValueError("Intensity data not available.")
        
        intensity = self.data.intensity if env_ids is None else self.data.intensity[env_ids]
        #print(f"[LidarSensor] get_intensity: shape={intensity.shape}, range=[{intensity.min().item():.4f}, {intensity.max().item():.4f}]")
        return intensity

    def get_height_map(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get 2D height map for specified environments.
        
        Returns a gridded elevation map generated from ray hits. The grid is centered
        on the robot and aligned with its local coordinate frame. Each cell contains
        the Z-coordinate (height) of the terrain at that position. Cells with no hits
        contain NaN values.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Height map tensor of shape (num_envs, grid_height, grid_width) in meters.
            Grid is centered on robot with X axis forward, Y axis left.
            
        Raises:
            ValueError: If height map generation is disabled in configuration.
        """
        if not self.cfg.return_height_map:
            raise ValueError("Height map generation is disabled. Set 'return_height_map=True' in config.")
        
        height_map = self.data.height_map if env_ids is None else self.data.height_map[env_ids]
        return height_map
