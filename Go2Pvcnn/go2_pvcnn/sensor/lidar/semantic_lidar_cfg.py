# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Semantic LiDAR sensor.

This module defines SemanticLidarCfg which extends LidarCfg to add semantic
classification capabilities based on mesh prim paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .lidar_cfg import LidarCfg

if TYPE_CHECKING:
    from .semantic_lidar_sensor import SemanticLidarSensor


@configclass
class SemanticLidarCfg(LidarCfg):
    """Configuration for the Semantic LiDAR sensor.
    
    This extends LidarCfg to add semantic classification based on mesh names.
    Each mesh is automatically classified into one of three categories:
    - Terrain (ground, walls, static environment)
    - Dynamic obstacles (movable objects)
    - Valuable items (furniture, important objects)
    
    The classification is based on keyword matching in mesh_prim_paths using
    the semantic_class_mapping dictionary.
    
    Example Configuration:
        >>> cfg = SemanticLidarCfg(
        ...     prim_path="/World/envs/env_.*",
        ...     mesh_prim_paths=[
        ...         "/World/ground",
        ...         "/world/SM_sofa_1",
        ...         "{ENV_REGEX_NS}/cracker_box_0/_03_cracker_box",
        ...     ],
        ...     semantic_class_mapping={
        ...         "terrain": ["ground", "wall", "floor"],
        ...         "dynamic_obstacle": ["cracker_box", "sugar_box", "tomato_soup_can"],
        ...         "valuable": ["sofa", "armchair", "table"],
        ...     },
        ...     update_frequency=50.0,
        ...     max_distance=40.0,
        ... )
        >>> sensor = SemanticLidarSensor(cfg)
    """

    class_type: type = None
    """Runtime sensor class. Will be set to SemanticLidarSensor at runtime."""
    
    def __post_init__(self):
        """Set the class type to SemanticLidarSensor after initialization."""
        # Call parent's __post_init__ first (but skip setting class_type there)
        # We override class_type below
        from .ray_caster_cfg import RayCasterCfg
        RayCasterCfg.__post_init__(self)
        
        from .semantic_lidar_sensor import SemanticLidarSensor
        self.class_type = SemanticLidarSensor
        
        # Set default semantic mapping if not provided
        if self.semantic_class_mapping is None:
            self.semantic_class_mapping = {
                "terrain": ["ground", "wall", "floor", "plane"],
                "dynamic_obstacle": ["cracker_box", "sugar_box", "tomato_soup_can", 
                                    "_03_cracker_box", "_04_sugar_box", "_05_tomato_soup_can"],
                "valuable": ["sofa", "armchair", "table", "chair", "desk",
                            "SM_Sofa", "SM_Armchair", "SM_Table"],
            }
    
    # Semantic classification settings
    semantic_class_mapping: dict[str, list[str]] = None
    """Mapping from semantic class names to keyword lists for classification.
    
    This dictionary defines how meshes are classified into semantic categories.
    Each key is a semantic class name, and the value is a list of keywords to
    search for in mesh prim paths (case-insensitive).
    
    Default semantic classes:
        - "terrain": Static environment elements (ground, walls, floors)
        - "dynamic_obstacle": Movable objects that should be avoided
        - "valuable": Important items like furniture that require special care
    
    The classification uses the first matching category. If no keywords match,
    the mesh is classified as semantic_id = 0 (unknown/no hit).
    
    Semantic ID mapping:
        - 0: No hit / Unknown
        - 1: Terrain
        - 2: Dynamic obstacle
        - 3: Valuable item
    
    Example:
        >>> semantic_class_mapping = {
        ...     "terrain": ["ground", "wall"],
        ...     "dynamic_obstacle": ["box", "can"],
        ...     "valuable": ["sofa", "table"],
        ... }
    """
    
    return_semantic_labels: bool = True
    """Whether to compute and return semantic labels for ray hits.
    
    If True, semantic_labels are available via get_semantic_labels().
    Defaults to True.
    """
    
    return_mesh_ids: bool = False
    """Whether to return mesh prototype IDs for each ray hit.
    
    Useful for debugging and detailed analysis of which specific mesh was hit.
    Defaults to False.
    """
