# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""全局刚体对象 - 支持多环境共享的单实例对象

Global rigid object that supports single-instance objects shared across multiple environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from collections.abc import Sequence

from .rigid_object import RigidObject

if TYPE_CHECKING:
    from .rigid_object_cfg import RigidObjectCfg


class GlobalRigidObject(RigidObject):
    """支持多环境训练的全局刚体对象
    
    A rigid object that exists as a single instance shared across all environments.
    
    标准 RigidObject 的限制：
    Standard RigidObject limitation:
        - 全局对象路径（无 ENV_REGEX_NS）→ _num_envs = 1
        - reset(env_ids=[0,1,...,255]) → 索引越界 Index out of bounds
    
    GlobalRigidObject 的解决方案：
    GlobalRigidObject solution:
        - 保持 _num_envs = 1（与其他组件兼容）
        - reset() 方法忽略 env_ids（全局对象对所有环境是相同的）
        - 数据缓冲区保持 (1, ...) 形状（节省内存）
    
    核心特性 Key features:
    - 与多环境训练兼容 (Compatible with multi-environment training)
    - 零额外内存开销 (Zero extra memory overhead)
    - 安全的 reset 操作 (Safe reset operations)
    - 与标准 RigidObject API 兼容 (Compatible with standard RigidObject API)
    
    使用场景 Use cases:
    - 静态障碍物（墙壁、柱子等）(Static obstacles like walls, pillars)
    - 共享物体（所有环境看到同一个物体）(Shared objects visible to all environments)
    - YCB 物体集合（用于感知训练）(YCB object collections for perception training)
    
    实现细节 Implementation notes:
    - 仅覆盖 reset() 方法 (Only overrides reset() method)
    - 所有其他功能继承自 RigidObject (All other functionality inherited from RigidObject)
    - 不改变数据结构或张量形状 (Does not change data structures or tensor shapes)
    
    使用示例 Example usage:
        >>> # 在场景配置中 In scene config
        >>> from isaaclab.assets import GlobalRigidObject, RigidObjectCfg
        >>> 
        >>> global_wall = RigidObjectCfg(
        ...     prim_path="/World/GlobalObjects/Wall",  # 无 ENV_REGEX_NS
        ...     spawn=...,
        ... )
        >>> 
        >>> # 在代码中使用 GlobalRigidObject 而不是 RigidObject
        >>> # Use GlobalRigidObject instead of RigidObject in code
        >>> # 或者在 InteractiveScene 中自动检测并使用
        >>> # Or automatically detect and use in InteractiveScene
    """

    def __init__(self, cfg: RigidObjectCfg):
        """初始化全局刚体对象
        
        Initialize the global rigid object.

        Args:
            cfg: 刚体对象配置 The rigid object configuration.
        """
        # 先检测是否为真正的全局对象（在调用父类初始化之前）
        # Detect if this is truly a global object (before calling parent init)
        self._is_global_object = "{ENV_REGEX_NS}" not in cfg.prim_path and "env_" not in cfg.prim_path.lower()
        
        # 调用父类初始化
        # Call parent initialization
        super().__init__(cfg)
        
        # 打印调试信息（不访问 _num_envs，因为它在 RigidObject 中不存在）
        # Print debug info (don't access _num_envs as it doesn't exist in RigidObject)
        if self._is_global_object:
            print(f"[GlobalRigidObject] 检测到全局对象: {cfg.prim_path}")
            print(f"[GlobalRigidObject] 将在 reset 时忽略多环境索引")

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置外部力和力矩（针对全局对象优化）
        
        Reset external wrench (optimized for global objects).
        
        对于全局对象，env_ids 会被忽略，因为全局对象在所有环境中是相同的。
        这避免了索引越界错误，同时保持与多环境训练的兼容性。
        
        For global objects, env_ids is ignored since the global object is the same
        across all environments. This avoids index out of bounds errors while
        maintaining compatibility with multi-environment training.

        Args:
            env_ids: 环境索引（全局对象会忽略此参数）
                     Environment indices (ignored for global objects).
        """
        if self._is_global_object:
            # 全局对象：总是重置索引 0（唯一实例）
            # Global object: always reset index 0 (the only instance)
            # 使用切片 [0:1] 而不是 [0]，以保持张量维度
            reset_indices = slice(0, 1)
        else:
            # 标准对象：使用传入的 env_ids
            # Standard object: use provided env_ids
            if env_ids is None:
                reset_indices = slice(None)
            else:
                reset_indices = env_ids
        
        # 重置外部力和力矩
        # Reset external wrench
        self._external_force_b[reset_indices] = 0.0
        self._external_torque_b[reset_indices] = 0.0
        self._external_wrench_positions_b[reset_indices] = 0.0

    def __str__(self) -> str:
        """返回实例的信息字符串 Returns: A string containing information about the instance."""
        msg = super().__str__()
        if self._is_global_object:
            msg += f"\n\t类型 type: GlobalRigidObject (全局对象 global object)"
        return msg
