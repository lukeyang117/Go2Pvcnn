# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""全局刚体对象集合配置。

Configuration for global rigid object collections.
"""

from isaaclab.utils import configclass

from .global_rigid_object_collection import GlobalRigidObjectCollection
from .rigid_object_collection_cfg import RigidObjectCollectionCfg


@configclass
class GlobalRigidObjectCollectionCfg(RigidObjectCollectionCfg):
    """全局刚体对象集合的配置
    
    Configuration for global rigid object collections.
    
    此配置类自动设置 class_type 为 GlobalRigidObjectCollection。
    用于定义在所有环境中共享的全局物体集合（例如障碍物、墙壁等）。
    
    This configuration class automatically sets class_type to GlobalRigidObjectCollection.
    Used to define global object collections shared across all environments (e.g., obstacles, walls).
    
    使用示例 Usage example:
        ```python
        from isaaclab.assets import GlobalRigidObjectCollectionCfg, RigidObjectCfg
        
        dynamic_objects = GlobalRigidObjectCollectionCfg(
            rigid_objects={
                "object_0": RigidObjectCfg(
                    prim_path="/World/GlobalObjects/Object_0",
                    spawn=...,
                ),
                "object_1": RigidObjectCfg(
                    prim_path="/World/GlobalObjects/Object_1",
                    spawn=...,
                ),
                # ... 更多全局物体
            },
        )
        ```
    """

    class_type: type = GlobalRigidObjectCollection
