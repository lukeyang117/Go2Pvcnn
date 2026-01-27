# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the LiDAR sensor.

This module defines the LidarCfg configuration class which extends RayCasterCfg
to add LiDAR-specific sensor parameters including:
- Point cloud generation
- Sensor noise simulation
- Range filtering
- Ray pattern alignment modes
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from isaaclab.utils import configclass

from .ray_caster_cfg import RayCasterCfg

if TYPE_CHECKING:
    from .lidar_sensor import LidarSensor


@configclass
class LidarCfg(RayCasterCfg):
    """Configuration for the LiDAR sensor.
    
    This configuration extends RayCasterCfg to add LiDAR-specific parameters
    for point cloud generation, noise simulation, and range filtering.
    
    Key Features:
        - Dynamic mesh tracking for moving obstacles
        - Configurable ray pattern alignment (base frame, yaw-only, or world)
        - Optional point cloud generation
        - Optional sensor noise simulation
        - Range filtering with configurable out-of-range values
    
    Example Configuration:
        >>> cfg = LidarCfg(
        ...     prim_path="/World/envs/env_.*",
        ...     update_frequency=50.0,
        ...     max_distance=30.0,
        ...     min_range=0.2,
        ...     ray_alignment="yaw",
        ...     return_pointcloud=True,
        ...     enable_sensor_noise=True,
        ...     random_distance_noise=0.03,
        ... )
        >>> lidar = LidarSensor(cfg)
    """

    class_type: type = None
    """Runtime sensor class. Will be set to LidarSensor at runtime."""
    
    def __post_init__(self):
        """Set the class type to LidarSensor after initialization."""
        super().__post_init__()
        from .lidar_sensor import LidarSensor
        self.class_type = LidarSensor
    
    # Ray pattern alignment
    ray_alignment: Literal["base", "yaw", "world"] = "yaw"
    """Ray alignment mode:
    
    - "base": Full base frame rotation applied to rays (standard multi-DoF rotation)
    - "yaw": Only yaw rotation applied, pitch and roll ignored (typical for 2.5D LiDAR)
    - "world": No rotation applied, rays fixed in world frame
    
    Defaults to "yaw" for typical spinning LiDAR behavior.
    """

    # LiDAR-specific timing parameters
    update_frequency: float = 50.0
    """LiDAR update frequency in Hz.
    
    This controls how often new ray casts are performed and point clouds are generated.
    Typical values: 10-100 Hz depending on sensor model.
    Defaults to 50.0 Hz.
    """

    # Range settings
    min_range: float = 0.2
    """Minimum sensing range in meters.
    
    Measurements closer than this distance are considered invalid (near_out_of_range_value).
    Typical range: 0.1-1.0 meters.
    Defaults to 0.2m.
    """

    # Output settings
    return_pointcloud: bool = True
    """Whether to generate and return point cloud data.
    
    If True, point clouds are available via get_pointcloud().
    If False, only distance measurements are available via get_distances().
    Defaults to True.
    """

    pointcloud_in_world_frame: bool = False
    """Coordinate frame for point cloud generation.
    
    - False: Point cloud in sensor-relative coordinates (default)
    - True: Point cloud in world coordinates
    
    Sensor-relative is more typical for robot control tasks.
    Defaults to False (sensor frame).
    """

    # Noise settings
    enable_sensor_noise: bool = False
    """Whether to enable sensor noise simulation.
    
    When enabled, applies distance noise and dropout to simulate realistic sensor behavior.
    Defaults to False.
    """

    random_distance_noise: float = 0.03
    """Standard deviation of Gaussian distance noise in meters.
    
    Applied as: distance_noisy = distance + N(0, sigma)
    Typical values: 0.01-0.1 meters.
    Defaults to 0.03m.
    """

    random_angle_noise: float = 0.15 * 3.14159 / 180
    """Standard deviation of angular noise in radians.
    
    Typical values: 0.1-1.0 degrees.
    Defaults to 0.15 degrees (≈ 0.0026 radians).
    """

    pixel_dropout_prob: float = 0.01
    """Probability of pixel dropout (no return) per ray.
    
    Simulates missing returns due to specular reflection or absorption.
    Typical values: 0.0-0.1 (0-10%).
    Defaults to 0.01 (1%).
    """

    pixel_std_dev_multiplier: float = 0.01
    """Multiplier for per-pixel noise standard deviation.
    
    Used for applying pixel-wise noise variations.
    Defaults to 0.01.
    """

    # Data normalization and out-of-range handling
    normalize_range: bool = False
    """Whether to normalize range values to [0, 1].
    
    If True, distances are normalized by max_distance.
    Defaults to False.
    """

    far_out_of_range_value: float = -1.0
    """Value to assign to measurements beyond max_distance.
    
    Use -1.0 for invalid marker, inf for fallback distance, or other custom values.
    Defaults to -1.0.
    """

    near_out_of_range_value: float = -1.0
    """Value to assign to measurements closer than min_range.
    
    Use -1.0 for invalid marker, 0.0 for minimum distance, or other custom values.
    Defaults to -1.0.
    """

    # Height map generation parameters
    return_height_map: bool = False
    """Whether to generate a 2D height map from ray hits. Defaults to False.
    
    When enabled, creates a gridded elevation map around the robot by projecting
    ray hit points onto a horizontal grid. Useful for terrain-aware navigation.
    """

    height_map_size: tuple[float, float] = (3.2, 3.2)
    """Size of the height map grid in meters (length_x, length_y). Defaults to (3.2, 3.2).
    
    This defines the physical extent of the height map in the robot's local coordinate frame.
    For example, (3.2, 3.2) creates a 3.2m x 3.2m square centered on the robot.
    The grid extends size_x/2 forward/backward and size_y/2 left/right from the sensor origin.
    """

    height_map_resolution: float = 0.1
    """Resolution of the height map grid in meters. Defaults to 0.1 meters.
    
    This determines the spacing between grid cells. A smaller value creates a finer grid
    but uses more memory. The number of grid cells is: (size_x/resolution) x (size_y/resolution).
    For example, 0.1m resolution with 3.2m size creates a 32x32 grid (1024 cells).
    """