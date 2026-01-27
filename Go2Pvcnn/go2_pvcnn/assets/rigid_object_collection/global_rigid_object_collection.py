# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""全局刚体对象集合 - 支持多环境训练中的全局共享物体集合。

Global rigid object collection that supports globally-shared object collections in multi-environment training.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .rigid_object_collection import RigidObjectCollection

if TYPE_CHECKING:
    from .rigid_object_collection_cfg import RigidObjectCollectionCfg


class GlobalRigidObjectCollection(RigidObjectCollection):
    """全局刚体对象集合
    
    Global rigid object collection for multi-environment training.
    
    这个类扩展了 RigidObjectCollection，用于处理全局共享的物体集合（例如所有环境共享的障碍物、墙壁等）。
    与标准的 RigidObjectCollection 不同，全局集合在所有环境中只有一个实例，而不是每个环境一个副本。
    
    This class extends RigidObjectCollection to handle globally-shared object collections
    (e.g., obstacles, walls shared across all environments). Unlike standard RigidObjectCollection,
    global collections have only one instance across all environments instead of one copy per environment.
    
    标准 RigidObjectCollection 的问题：
    Standard RigidObjectCollection issue:
        - num_instances = 1 (全局物体只有1个实例)
        - reset(env_ids=[0-255]) 被调用时会导致索引越界
        - self._external_force_b[env_ids, :] 其中 env_ids 包含 0-255 但 buffer 第一维只有 1
    
    GlobalRigidObjectCollection 的解决方案：
    GlobalRigidObjectCollection solution:
        - 检测 num_instances == 1（全局集合标志）
        - 在 reset() 中忽略 env_ids，只重置索引 0 的状态
        - 避免索引越界错误
    
    使用方法 Usage:
        在场景配置中使用 class_type 参数：
        Use class_type parameter in scene configuration:
        
        ```python
        dynamic_objects = RigidObjectCollectionCfg(
            rigid_objects={...},
            class_type=GlobalRigidObjectCollection,  # ← 使用全局集合类
        )
        ```
    """

    cfg: RigidObjectCollectionCfg
    """配置对象 The configuration object."""

    def __init__(self, cfg: RigidObjectCollectionCfg):
        """初始化全局刚体对象集合
        
        Initialize the global rigid object collection.

        Args:
            cfg: 刚体对象集合配置 The rigid object collection configuration.
        """
        # 调用父类初始化
        # Call parent initialization
        super().__init__(cfg)
        
        # 检测是否为全局集合（所有环境共享一个实例）
        # Detect if this is a global collection (one instance shared across all environments)
        # 注意：此时 num_instances 可能还未设置，需要在 _initialize_impl 后检查
        # Note: num_instances may not be set yet, need to check in _initialize_impl
        self._is_global_collection = False

    def _initialize_impl(self):
        """初始化实现 - 检测全局集合并添加调试信息
        
        Initialize implementation - detect global collection and add debug info.
        """
        # 先调用父类初始化
        # Call parent initialization first
        super()._initialize_impl()
        
        # 检测是否为全局集合（num_instances == 1 表示全局共享）
        # Detect if this is a global collection (num_instances == 1 means globally shared)
        self._is_global_collection = self.num_instances == 1
        
        if self._is_global_collection:
            print(f"\n[GlobalRigidObjectCollection] 检测到全局物体集合")
            print(f"  - num_instances = {self.num_instances} (全局共享)")
            print(f"  - num_objects = {self.num_objects}")
            print(f"  - 将在 reset() 时忽略多环境索引")
            print(f"  - 数据缓冲区形状: _external_force_b = {self._external_force_b.shape}")

    def reset(self, env_ids: torch.Tensor | None = None, object_ids: torch.Tensor | None = None):
        """重置物体集合状态
        
        Reset the rigid object collection state.
        
        对于全局集合（num_instances=1），忽略 env_ids 参数，只重置单一全局状态。
        For global collections (num_instances=1), ignore env_ids and only reset the single global state.
        
        Args:
            env_ids: 要重置的环境 ID Environment IDs to reset (ignored for global collections).
            object_ids: 要重置的物体 ID Object IDs to reset.
        """
        if self._is_global_collection:
            # 全局集合：忽略 env_ids，只重置索引 0 的状态
            # Global collection: ignore env_ids, only reset state at index 0
            
            # 解析物体ID
            # Resolve object IDs
            if object_ids is None:
                object_ids = self._ALL_OBJ_INDICES
            
            # 重置外部力和力矩（只重置第一个环境索引，因为全局集合只有一个实例）
            # Reset external wrench (only reset first env index since global collection has one instance)
            env_idx = 0
            self._external_force_b[env_idx, object_ids] = 0.0
            self._external_torque_b[env_idx, object_ids] = 0.0
            self._external_wrench_positions_b[env_idx, object_ids] = 0.0
            
            # 添加调试信息（可选，生产环境可以注释掉）
            # Add debug info (optional, can be commented out in production)
            # print(f"[GlobalRigidObjectCollection.reset] 重置全局集合 (忽略 env_ids，只重置索引 {env_idx})")
        else:
            # 普通集合：使用标准重置逻辑
            # Normal collection: use standard reset logic
            super().reset(env_ids, object_ids)

    def write_object_link_pose_to_sim(
        self,
        object_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """写入物体位姿到仿真
        
        Write object link pose to simulation.
        
        对于全局集合，忽略 env_ids 参数，因为所有环境共享同一个物体实例。
        For global collections, ignore env_ids since all environments share the same object instance.
        
        Args:
            object_pose: 物体位姿 Object pose tensor. Shape: (num_envs, num_objects, 7) or (num_objects, 7)
            env_ids: 环境 ID Environment IDs (ignored for global collections)
            object_ids: 物体 ID Object IDs
        """
        if self._is_global_collection:
            # 全局集合：忽略 env_ids，只使用第一组位姿数据
            # Global collection: ignore env_ids, only use first pose data
            
            # 解析物体 ID
            if object_ids is None:
                object_ids = self._ALL_OBJ_INDICES
            
            # 如果 object_pose 有 3 个维度 (num_envs, num_objects, 7)，只取第一个环境
            # If object_pose has 3 dimensions (num_envs, num_objects, 7), take first env
            if object_pose.ndim == 3:
                # 取第一个环境的位姿（所有环境共享相同的全局物体）
                object_pose_global = object_pose[0]  # Shape: (num_objects, 7)
            else:
                # 已经是 (num_objects, 7) 或 (1, num_objects, 7)
                object_pose_global = object_pose.squeeze(0) if object_pose.ndim == 3 else object_pose
            
            # 添加调试信息
            print(f"[GlobalRigidObjectCollection.write_object_link_pose_to_sim]")
            print(f"  - Input object_pose shape: {object_pose.shape}")
            print(f"  - Using global pose shape: {object_pose_global.shape}")
            print(f"  - num_instances: {self.num_instances}")
            print(f"  - num_objects: {self.num_objects}")
            print(f"  - object_ids: {object_ids}")
            
            # 确保形状正确
            if object_pose_global.shape[0] != len(object_ids):
                raise ValueError(
                    f"Global pose shape mismatch: expected ({len(object_ids)}, 7), "
                    f"got {object_pose_global.shape}"
                )
            
            # 更新内部缓冲区（只更新索引 0 的环境）
            env_idx = 0
            self._data.object_link_pose_w[env_idx, object_ids] = object_pose_global.clone()
            
            # 更新其他缓冲区（如果使用）
            if self._data._object_link_state_w.data is not None:
                self._data.object_link_state_w[env_idx, object_ids, :7] = object_pose_global.clone()
            if self._data._object_state_w.data is not None:
                self._data.object_state_w[env_idx, object_ids, :7] = object_pose_global.clone()
            if self._data._object_com_state_w.data is not None:
                from isaaclab.utils import math as math_utils
                # get CoM pose in link frame
                com_pos_b = self.data.object_com_pos_b[env_idx, object_ids]
                com_quat_b = self.data.object_com_quat_b[env_idx, object_ids]
                com_pos, com_quat = math_utils.combine_frame_transforms(
                    object_pose_global[..., :3],
                    object_pose_global[..., 3:7],
                    com_pos_b,
                    com_quat_b,
                )
                self._data.object_com_state_w[env_idx, object_ids, :3] = com_pos
                self._data.object_com_state_w[env_idx, object_ids, 3:7] = com_quat

            # 转换四元数格式：wxyz -> xyzw
            from isaaclab.utils import math as math_utils
            poses_xyzw = self._data.object_link_pose_w.clone()
            poses_xyzw[..., 3:] = math_utils.convert_quat(poses_xyzw[..., 3:], to="xyzw")

            # 写入仿真（只使用索引 0）
            view_ids = self._env_obj_ids_to_view_ids(torch.tensor([0], device=self.device), object_ids)
            # reshape_data_to_view 期望 (num_instances, num_objects, 7)，所以保持 3D 形状
            self.root_physx_view.set_transforms(self.reshape_data_to_view(poses_xyzw), indices=view_ids)
            
        else:
            # 普通集合：使用标准逻辑
            super().write_object_link_pose_to_sim(object_pose, env_ids, object_ids)

    def write_object_link_velocity_to_sim(
        self,
        velocities: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):
        """写入物体速度到仿真
        
        Write object link velocity to simulation.
        
        对于全局集合，忽略 env_ids 参数。
        For global collections, ignore env_ids parameter.
        
        Args:
            velocities: 速度张量 Velocity tensor. Shape: (num_envs, num_objects, 6) or (num_objects, 6)
            env_ids: 环境 ID Environment IDs (ignored for global collections)
            object_ids: 物体 ID Object IDs
        """
        if self._is_global_collection:
            # 全局集合：忽略 env_ids，只使用第一组速度数据
            
            # 解析物体 ID
            if object_ids is None:
                object_ids = self._ALL_OBJ_INDICES
            
            # 如果 velocities 有 3 个维度，只取第一个环境
            if velocities.ndim == 3:
                velocities_global = velocities[0]  # Shape: (num_objects, 6)
            else:
                velocities_global = velocities.squeeze(0) if velocities.ndim == 3 else velocities
            
            print(f"[GlobalRigidObjectCollection.write_object_link_velocity_to_sim]")
            print(f"  - Input velocities shape: {velocities.shape}")
            print(f"  - Using global velocities shape: {velocities_global.shape}")
            
            # 更新内部缓冲区
            env_idx = 0
            self._data.object_link_lin_vel_w[env_idx, object_ids] = velocities_global[..., :3].clone()
            self._data.object_link_ang_vel_w[env_idx, object_ids] = velocities_global[..., 3:].clone()
            
            # 更新其他缓冲区
            if self._data._object_link_state_w.data is not None:
                self._data.object_link_state_w[env_idx, object_ids, 7:] = velocities_global.clone()
            if self._data._object_state_w.data is not None:
                self._data.object_state_w[env_idx, object_ids, 7:] = velocities_global.clone()
            if self._data._object_com_state_w.data is not None:
                self._data.object_com_state_w[env_idx, object_ids, 7:] = velocities_global.clone()

            # 写入仿真
            view_ids = self._env_obj_ids_to_view_ids(torch.tensor([0], device=self.device), object_ids)
            # 需要添加 env 维度：(num_objects, 6) -> (1, num_objects, 6)
            velocities_3d = velocities_global.unsqueeze(0)  # Shape: (1, num_objects, 6)
            self.root_physx_view.set_velocities(self.reshape_data_to_view(velocities_3d), indices=view_ids)
            
        else:
            # 普通集合：使用标准逻辑
            super().write_object_link_velocity_to_sim(velocities, env_ids, object_ids)
