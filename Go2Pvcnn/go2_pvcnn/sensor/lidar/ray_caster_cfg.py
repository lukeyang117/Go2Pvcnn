"""Configuration for the ray-cast sensor with dynamic mesh support.

This module defines the RayCasterCfg class which extends Isaac Lab's RayCasterCfg
to add support for:
- Dynamic mesh tracking with moving rigid bodies
- Multi-environment collision group management
- Grouped ray casting for efficient multi-environment simulation
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal, TYPE_CHECKING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster import RayCasterCfg as IsaacRayCasterCfg

if TYPE_CHECKING:
    from .ray_caster import LidarRayCaster


@configclass
class RayCasterCfg(IsaacRayCasterCfg):
    """Configuration for the ray-cast sensor with dynamic mesh support.
    
    This extends the base Isaac Lab RayCasterCfg to support dynamic meshes
    with updating transforms for efficient multi-environment ray casting.
    
    Attributes:
        class_type: Runtime sensor class (set to LidarRayCaster in __post_init__).
        attach_yaw_only: (Deprecated) Use ray_alignment instead.
        min_distance: Minimum sensing distance in meters.
        aux_mesh_and_link_names: Mapping of mesh file names to link names for complex models.
        drift_range: Range of random drift in meters for ray start positions.
        ray_cast_drift_range: Per-axis drift ranges in local frame.
    
    Example Configuration:
        >>> cfg = RayCasterCfg(
        ...     prim_path="/World/envs/env_.*",
        ...     mesh_prim_paths=["{ENV_REGEX_NS}/.*"],
        ...     aux_mesh_and_link_names={"mesh_name": "link_name"},
        ...     min_distance=0.2,
        ...     max_distance=30.0,
        ... )
    """

    class_type: type = None
    """Runtime sensor class. Will be set to LidarRayCaster in __post_init__."""
    
    attach_yaw_only: bool = False
    """(Deprecated) Whether rays only track yaw orientation. Use ray_alignment instead."""
    
    min_distance: float = 0.0
    """Minimum sensing distance in meters. Hits closer than this are ignored."""

    aux_mesh_and_link_names: dict[str, str] = None
    """Mapping of mesh file names to link names for auxiliary mesh search.
    
    Used when complex robot models have multiple visual meshes under an Xform prim.
    Each entry maps a mesh file name (key) to its corresponding link name (value).
    If None (default), an empty dictionary is used.
    
    Example:
        >>> aux_mesh_and_link_names = {
        ...     "torso_visual": "torso_link",
        ...     "arm_visual": "arm_link",
        ... }
    """
    
    drift_range: tuple[float, float] = (0.0, 0.0)
    """Range of random drift in meters for ray starting positions in world frame.
    
    Useful for simulating random calibration errors or vibration.
    Example: (0.0, 0.01) adds random drift uniformly in [-0.01, 0.01] meters.
    """
    
    ray_cast_drift_range: dict[str, tuple[float, float]] = None
    """Per-axis drift ranges in local frame for ray projection points.
    
    If None (default), initialized to no drift: {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    """

    def __post_init__(self):
        """Set the class type to LidarRayCaster after initialization."""
        super().__post_init__()
        from .ray_caster import LidarRayCaster
        self.class_type = LidarRayCaster
        if self.aux_mesh_and_link_names is None:
            self.aux_mesh_and_link_names = {}
        if self.ray_cast_drift_range is None:
            self.ray_cast_drift_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
