# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data class for the Semantic LiDAR sensor."""

from __future__ import annotations

import torch
from dataclasses import dataclass

from .lidar_data import LidarSensorData


@dataclass
class SemanticLidarData(LidarSensorData):
    """Data container for the Semantic LiDAR sensor.
    
    This extends LidarSensorData to include semantic classification information
    for each ray hit based on the mesh type.
    """

    semantic_labels: torch.Tensor = None
    """Semantic class labels for each ray. Shape is (num_instances, num_rays).
    
    Semantic categories:
        - 0: No hit / Unknown
        - 1: Terrain (ground, walls, static environment)
        - 2: Dynamic obstacles (movable objects like boxes, cans)
        - 3: Valuable items (furniture like sofas, chairs, tables)
    
    Note:
        This is populated based on the mesh_prim_paths and semantic_class_mapping
        configuration in SemanticLidarCfg.
    """

    mesh_ids: torch.Tensor = None
    """Mesh prototype IDs for each ray hit. Shape is (num_instances, num_rays).
    
    This tracks which specific mesh was hit by each ray, useful for debugging
    and detailed semantic analysis. Value is -1 for no hit.
    """

    semantic_class_counts: torch.Tensor = None
    """Count of rays for each semantic class. Shape is (num_instances, 4).
    
    Each row represents counts for [class_0, class_1, class_2, class_3]:
        - class_0: No hit / Unknown
        - class_1: Terrain
        - class_2: Dynamic obstacles
        - class_3: Valuable items
    
    This is automatically computed when semantic_labels is updated.
    """

    ray_mesh_ids: torch.Tensor = None
    """Mesh indices for each ray hit from raycast kernel. Shape is (num_instances, num_rays).
    
    This is the mesh index (into mesh_prototype_ids) returned by the raycast kernel.
    Value is -1 for no hit. Used internally for semantic classification.
    """
