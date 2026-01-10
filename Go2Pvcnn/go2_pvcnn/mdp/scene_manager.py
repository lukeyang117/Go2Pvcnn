"""Dynamic USD objects scene manager for Go2 PVCNN environment.

This module manages the spawning, positioning, and movement of dynamic USD objects
that can be detected by the LiDAR sensor.
"""

from __future__ import annotations

import json
import numpy as np
import torch
from typing import TYPE_CHECKING, List, Dict, Tuple
from pathlib import Path

from isaaclab.assets import RigidObject, RigidObjectCfg, RigidObjectCollectionCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Helper Functions for Scene Configuration
##




def create_dynamic_objects_collection_cfg(num_objects: int = 3) -> RigidObjectCollectionCfg:
    """Create RigidObjectCollectionCfg with 3 YCB objects per environment.
    
    创建动态USD物体配置，每个环境3个物体（CrackerBox, SugarBox, TomatoSoupCan）。
    
    Args:
        num_objects: Number of objects per environment (fixed at 3 for YCB objects)
        
    Returns:
        RigidObjectCollectionCfg configured with 3 YCB objects
    """
    # 固定使用3个YCB物体，每个物体对应一个USD文件
    # Fixed 3 YCB objects, each with its own USD file
    ycb_objects = [
        {
            "name": "cracker_box",
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            "mesh_name": "_03_cracker_box",  # USD内部的mesh名称
        },
        {
            "name": "sugar_box",
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            "mesh_name": "_04_sugar_box",
        },
        {
            "name": "tomato_soup_can",
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            "mesh_name": "_05_tomato_soup_can",
        },
    ]
    
    # Create rigid objects dictionary
    rigid_objects = {}
    
    # Create num_objects objects by cycling through the 3 YCB objects
    for i in range(num_objects):
        obj_info = ycb_objects[i % len(ycb_objects)]  # Cycle through YCB objects
        
        # 使用 {ENV_REGEX_NS} 让物体在每个环境中自动复制
        # Use {ENV_REGEX_NS} to automatically replicate objects across environments
        rigid_objects[f"object_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Object_{i}",  # 每个环境: Object_0, Object_1, Object_2, ...
            spawn=sim_utils.UsdFileCfg(
                usd_path=obj_info["usd_path"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,  # 启用接触传感器 - Enable contact sensors
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.5),  # 将通过事件随机化位置 - Will be randomized by events
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)






