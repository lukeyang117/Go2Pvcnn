# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic LiDAR sensor implementation with object classification.

This module implements a Semantic LiDAR sensor that extends the standard LiDAR
to provide semantic classification of ray hits based on the type of mesh/object hit.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .lidar_sensor import LidarSensor
from .semantic_lidar_data import SemanticLidarData

if TYPE_CHECKING:
    from .semantic_lidar_cfg import SemanticLidarCfg


class SemanticLidarSensor(LidarSensor):
    """A Semantic LiDAR sensor that classifies ray hits by object type.
    
    This sensor extends LidarSensor to add semantic classification capabilities.
    Each ray hit is classified into one of three categories based on the mesh
    that was hit:
        - Terrain (ID=1): Ground, walls, static environment
        - Dynamic obstacles (ID=2): Movable objects to avoid
        - Valuable items (ID=3): Important objects requiring special handling
    
    The classification is based on keyword matching against mesh prim paths using
    the semantic_class_mapping configuration.
    
    Key Capabilities:
        - All standard LiDAR features (distances, point clouds, noise)
        - Semantic classification of each ray hit
        - Optional mesh ID tracking for debugging
        - Configurable keyword-based classification rules
    
    Example:
        >>> cfg = SemanticLidarCfg(
        ...     prim_path="/World/envs/env_.*",
        ...     mesh_prim_paths=["/World/ground", "/world/SM_sofa_1", ...],
        ...     return_semantic_labels=True,
        ... )
        >>> sensor = SemanticLidarSensor(cfg)
        >>> semantic_labels = sensor.get_semantic_labels()  # Shape: (num_envs, num_rays)
    """

    cfg: SemanticLidarCfg
    """The configuration parameters."""

    def __init__(self, cfg: SemanticLidarCfg):
        """Initializes the Semantic LiDAR sensor.

        Args:
            cfg: The configuration parameters.
        """
        #print(f"\n[SemanticLidarSensor] Initializing with semantic classification enabled")
        
        # Initialize parent class (LidarSensor -> LidarRayCaster)
        super().__init__(cfg)
        
        # Replace data container with semantic version
        self._data = SemanticLidarData()
        
        # Semantic classification mapping (populated during mesh initialization)
        self.mesh_id_to_semantic_class: dict[int, int] = {}
        """Mapping from warp mesh ID to semantic class ID (0=unknown, 1=terrain, 2=obstacle, 3=valuable)."""
        
        self.mesh_prim_path_to_semantic_class: dict[str, int] = {}
        """Mapping from mesh prim path to semantic class ID for debugging."""
        
    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        base_str = super().__str__()
        semantic_info = (
            f"\n\tSemantic classification: {self.cfg.return_semantic_labels}\n"
            f"\tnumber of semantic classes: {len(self.cfg.semantic_class_mapping)}\n"
            f"\tnumber of classified meshes: {len(self.mesh_id_to_semantic_class)}"
        )
        return base_str + semantic_info

    @property
    def data(self) -> SemanticLidarData:
        """The sensor data object with semantic information."""
        # Update buffers if needed
        self._update_outdated_buffers()
        return self._data

    def _classify_mesh_path(self, mesh_prim_path: str) -> int:
        """Classify a mesh based on its prim path using keyword matching.
        
        Args:
            mesh_prim_path: The full prim path of the mesh.
            
        Returns:
            Semantic class ID:
                - 0: Unknown/unclassified
                - 1: Terrain
                - 2: Dynamic obstacle
                - 3: Valuable item
        """
        # Normalize path to lowercase for case-insensitive matching
        path_lower = mesh_prim_path.lower()
        
        # Define semantic class ID mapping
        semantic_class_ids = {
            "terrain": 1,
            "dynamic_obstacle": 2,
            "valuable": 3,
        }
        
        # Check each semantic class in order
        for class_name, keywords in self.cfg.semantic_class_mapping.items():
            for keyword in keywords:
                if keyword.lower() in path_lower:
                    class_id = semantic_class_ids.get(class_name, 0)
                    return class_id
        
        # No match found
        #print(f"[SemanticLidarSensor] Could not classify '{mesh_prim_path}', defaulting to class 0 (unknown)")
        return 0

    def _initialize_warp_meshes(self):
        """Initialize warp meshes and build semantic classification mapping.
        
        This extends the parent method to also build the mesh_id -> semantic_class
        mapping based on mesh prim paths.
        """
        # Call parent to initialize meshes
        super()._initialize_warp_meshes()
        # Build mapping from mesh_id to semantic class
        # We need to iterate through the meshes dictionary and match with mesh_prototype_id
        for mesh_prim_path_pattern, wp_meshes_list in self.meshes.items():
            # Classify this mesh pattern
            semantic_class = self._classify_mesh_path(mesh_prim_path_pattern)
            # Assign this class to all mesh instances from this pattern
            for wp_mesh in wp_meshes_list:
                mesh_id = wp_mesh.id
                self.mesh_id_to_semantic_class[mesh_id] = semantic_class
                self.mesh_prim_path_to_semantic_class[mesh_prim_path_pattern] = semantic_class
        
        # Build vectorized lookup tensor for fast semantic classification (only once during initialization)
        max_mesh_id = len(self.mesh_prototype_ids)
        self.mesh_id_to_class_tensor = torch.zeros(max_mesh_id, dtype=torch.int32, device=self._device)
        
        for mesh_idx in range(max_mesh_id):
            mesh_id = int(self.mesh_prototype_ids[mesh_idx])
            semantic_class = self.mesh_id_to_semantic_class.get(mesh_id, 0)
            self.mesh_id_to_class_tensor[mesh_idx] = semantic_class
                
        
        

    def _initialize_rays_impl(self):
        """Initialize ray patterns and semantic label buffers."""
        # Call parent to initialize rays and lidar data
        super()._initialize_rays_impl()
        
        # Initialize semantic label buffers
        if self.cfg.return_semantic_labels:
            self._data.semantic_labels = torch.zeros(
                self._view.count, self.num_rays, dtype=torch.int32, device=self._device
            )
            # Initialize semantic class counts buffer (4 classes: 0, 1, 2, 3)
            self._data.semantic_class_counts = torch.zeros(
                self._view.count, 4, dtype=torch.int32, device=self._device
            )
            #print(f"[SemanticLidarSensor] Initialized semantic_labels buffer: shape={self._data.semantic_labels.shape}")
            #print(f"[SemanticLidarSensor] Initialized semantic_class_counts buffer: shape={self._data.semantic_class_counts.shape}")
        
        # Initialize mesh ID tracking (always needed for semantic classification)
        self._data.ray_mesh_ids = torch.full(
            (self._view.count, self.num_rays), -1, dtype=torch.int32, device=self._device
        )
        
        # Legacy mesh_ids buffer if requested
        if self.cfg.return_mesh_ids:
            self._data.mesh_ids = torch.full(
                (self._view.count, self.num_rays), -1, dtype=torch.int64, device=self._device
            )
            #print(f"[SemanticLidarSensor] Initialized mesh_ids buffer: shape={self._data.mesh_ids.shape}")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Update sensor buffers including semantic classification.
        
        This extends the parent method to also compute semantic labels
        based on which mesh was hit by each ray.
        
        Args:
            env_ids: Environment IDs to update.
        """
        # Call parent to update lidar measurements
        super()._update_buffers_impl(env_ids)
        
        # Compute semantic labels if enabled
        if self.cfg.return_semantic_labels or self.cfg.return_mesh_ids:
            self._compute_semantic_labels(env_ids)

    def _compute_semantic_labels(self, env_ids: Sequence[int]):
        """Compute semantic labels for ray hits using returned mesh IDs.
        
        This method uses the mesh IDs returned from ray casting to determine
        which mesh was hit by each ray, then looks up the semantic class.
        
        Args:
            env_ids: Environment IDs to update.
        """
        # Get ray hits for inf detection
        hit_points = self._data.ray_hits_w[env_ids]  # Shape: (num_envs_subset, num_rays, 3)
        
        # Check for no collision using torch.isinf (now that we initialize to inf)
        nohit_mask = torch.isinf(hit_points).any(dim=-1)  # Shape: (num_envs_subset, num_rays)
        
        # Get mesh IDs from raycast results
        ray_mesh_ids = self._data.ray_mesh_ids[env_ids]  # Shape: (num_envs_subset, num_rays)
        
        # Initialize semantic labels to 0 (no hit)
        semantic_labels = torch.zeros(
            (len(env_ids), self.num_rays), dtype=torch.int32, device=self._device
        )
        
        # Vectorized lookup: use pre-built tensor from initialization
        hit_mask = ray_mesh_ids >= 0
        semantic_labels[hit_mask] = self.mesh_id_to_class_tensor[ray_mesh_ids[hit_mask]]
        
        # No-hit rays stay at class 0
        semantic_labels[nohit_mask] = 0
        
        # Store semantic labels
        self._data.semantic_labels[env_ids] = semantic_labels
        
        # Compute semantic class counts for each environment
        semantic_counts = torch.zeros(
            (len(env_ids), 4), dtype=torch.int32, device=self._device
        )
        for class_id in range(4):
            semantic_counts[:, class_id] = (semantic_labels == class_id).sum(dim=1)
        
        self._data.semantic_class_counts[env_ids] = semantic_counts

    def get_semantic_labels(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get semantic class labels for each ray hit.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Semantic labels tensor of shape (num_envs, num_rays) with values:
                - 0: No hit / Unknown
                - 1: Terrain
                - 2: Dynamic obstacle
                - 3: Valuable item
            
        Raises:
            ValueError: If semantic label generation is disabled in configuration.
        """
        if not self.cfg.return_semantic_labels:
            raise ValueError("Semantic label generation is disabled. Set 'return_semantic_labels=True' in config.")
        
        labels = self.data.semantic_labels if env_ids is None else self.data.semantic_labels[env_ids]
        #print(f"[SemanticLidarSensor] get_semantic_labels: shape={labels.shape}, unique_classes={torch.unique(labels).cpu().tolist()}")
        return labels

    def get_mesh_ids(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get mesh prototype IDs for each ray hit.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Mesh IDs tensor of shape (num_envs, num_rays). Value is -1 for no hit.
            
        Raises:
            ValueError: If mesh ID tracking is disabled in configuration.
        """
        if not self.cfg.return_mesh_ids:
            raise ValueError("Mesh ID tracking is disabled. Set 'return_mesh_ids=True' in config.")
        
        mesh_ids = self.data.mesh_ids if env_ids is None else self.data.mesh_ids[env_ids]
        #print(f"[SemanticLidarSensor] get_mesh_ids: shape={mesh_ids.shape}")
        return mesh_ids

    def get_semantic_class_counts(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get count of rays for each semantic class.
        
        Args:
            env_ids: Environment indices. If None, returns data for all environments.
            
        Returns:
            Semantic class counts tensor of shape (num_envs, 4) with counts for:
                - Column 0: No hit / Unknown
                - Column 1: Terrain
                - Column 2: Dynamic obstacles
                - Column 3: Valuable items
            
        Raises:
            ValueError: If semantic label generation is disabled in configuration.
        """
        if not self.cfg.return_semantic_labels:
            raise ValueError("Semantic label generation is disabled. Set 'return_semantic_labels=True' in config.")
        
        counts = self.data.semantic_class_counts if env_ids is None else self.data.semantic_class_counts[env_ids]
        return counts
