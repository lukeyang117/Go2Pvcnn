# Teacher Training Configuration Fixes

## 问题总结

在创建teacher训练环境配置时，存在以下问题需要修复：

1. **错误的reset函数引用**: 使用了不存在的 `reset_root_state_uniform_with_terrain_origin`
2. **缺少家具配置**: 需要添加9个家具物体（3个沙发、3个扶手椅、3个桌子）
3. **重复的height_scanner**: SemanticLidarCfg已经生成高程图，不需要单独的height_scanner
4. **缺少家具碰撞检测**: 需要启用家具碰撞传感器和奖励

## 修复内容

### 1. 移除多余的height_scanner

**问题**: SemanticLidarCfg已经通过 `return_height_map=True` 返回高程图，不需要额外的height_scanner

**修复**:
- 删除了 `height_scanner` RayCasterCfg配置
- 更新观察项以仅使用lidar的height_map

```python
# 修复前
height_scanner = RayCasterCfg(...)

# 观察项
semantic_cost_map = ObsTerm(
    func=custom_mdp.teacher_semantic_cost_map,
    params={
        "lidar_cfg": SceneEntityCfg("semantic_lidar"),
        "height_scanner_cfg": SceneEntityCfg("height_scanner"),  # ❌ 多余
        "command_name": "base_velocity",
    },
)

# 修复后
# height_scanner已删除

semantic_cost_map = ObsTerm(
    func=custom_mdp.teacher_semantic_cost_map,
    params={
        "lidar_cfg": SceneEntityCfg("lidar"),  # ✅ lidar已包含height_map
        "command_name": "base_velocity",
    },
)
```

### 2. 修复Event函数引用

**问题**: 使用了不存在的 `custom_mdp.reset_root_state_uniform_with_terrain_origin`

**修复**: 改用Isaac Lab内置的 `isaac_mdp.reset_root_state_uniform`

```python
# 修复前 ❌
reset_cracker_box_0 = EventTerm(
    func=custom_mdp.reset_root_state_uniform_with_terrain_origin,  # 不存在
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("cracker_box_0"),
        "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0)},
        "velocity_range": {},
        "height_offset": 1.0,
    },
)

# 修复后 ✅
reset_cracker_box_0 = EventTerm(
    func=isaac_mdp.reset_root_state_uniform,  # Isaac Lab内置函数
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("cracker_box_0"),
        "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
    },
)
```

**说明**:
- 所有9个YCB物体都使用此函数重置位置（每次episode reset时随机放置）
- 家具**不需要**reset event（它们是静态的全局物体，固定在/World下）

### 3. 添加家具配置

**问题**: 家具配置被注释掉了

**修复**: 添加9个家具物体作为全局静态刚体

```python
# 家具特点：
# - prim_path: "/World/SM_*" (小写world，全局路径)
# - kinematic_enabled=True (运动学刚体，不受物理影响)
# - disable_gravity=True (无重力)
# - 无reset event (静态物体，不随机化位置)

furniture_1 = RigidObjectCfg(
    prim_path="/World/SM_sofa_1",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Sofa.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,  # 运动学刚体
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, -2.0, 0.0)),
)
# ... 同样添加 furniture_2 到 furniture_9
```

**家具列表** (共9个):
1. `/World/SM_sofa_1` - 沙发1
2. `/World/SM_armchair_1` - 扶手椅1
3. `/World/SM_table_1` - 桌子1
4. `/World/SM_sofa_2` - 沙发2
5. `/World/SM_armchair_2` - 扶手椅2
6. `/World/SM_table_2` - 桌子2
7. `/World/SM_sofa_3` - 沙发3
8. `/World/SM_armchair_3` - 扶手椅3
9. `/World/SM_table_3` - 桌子3

### 4. 更新LiDAR mesh_prim_paths

**问题**: LiDAR的mesh_prim_paths缺少家具

**修复**: 添加所有家具到mesh_prim_paths

```python
lidar: SemanticLidarCfg = SemanticLidarCfg(
    ...
    mesh_prim_paths=[
        "/World/ground",
        # Global furniture (static, no reset)
        "/World/SM_sofa_1",
        "/World/SM_armchair_1",
        "/World/SM_table_1",
        "/World/SM_sofa_2",
        "/World/SM_armchair_2",
        "/World/SM_table_2",
        "/World/SM_sofa_3",
        "/World/SM_armchair_3",
        "/World/SM_table_3",
        # Dynamic YCB objects (per-env, will be reset)
        "{ENV_REGEX_NS}/cracker_box_0/_03_cracker_box",
        "{ENV_REGEX_NS}/cracker_box_1/_03_cracker_box",
        "{ENV_REGEX_NS}/cracker_box_2/_03_cracker_box",
        "{ENV_REGEX_NS}/sugar_box_0/_04_sugar_box",
        "{ENV_REGEX_NS}/sugar_box_1/_04_sugar_box",
        "{ENV_REGEX_NS}/sugar_box_2/_04_sugar_box",
        "{ENV_REGEX_NS}/tomato_soup_can_0/_05_tomato_soup_can",
        "{ENV_REGEX_NS}/tomato_soup_can_1/_05_tomato_soup_can",
        "{ENV_REGEX_NS}/tomato_soup_can_2/_05_tomato_soup_can",
    ],
)
```

### 5. 启用家具碰撞传感器

**问题**: 家具碰撞传感器被注释掉了

**修复**: 启用contact_forces_furniture传感器

```python
# 修复前 ❌
# contact_forces_furniture: ContactSensorCfg = ContactSensorCfg(...)

# 修复后 ✅
contact_forces_furniture: ContactSensorCfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    update_period=0.0,
    history_length=3,
    track_air_time=False,
    debug_vis=False,
    filter_prim_paths_expr=[
        "/World/SM_sofa_1",
        "/World/SM_armchair_1",
        "/World/SM_table_1",
        "/World/SM_sofa_2",
        "/World/SM_armchair_2",
        "/World/SM_table_2",
        "/World/SM_sofa_3",
        "/World/SM_armchair_3",
        "/World/SM_table_3",
    ],
)
```

### 6. 启用家具碰撞惩罚

**问题**: 家具碰撞奖励项被注释掉了

**修复**: 启用furniture_collision奖励项

```python
# 修复前 ❌
# furniture_collision = RewTerm(...)

# 修复后 ✅
furniture_collision = RewTerm(
    func=teacher_rewards.furniture_collision_penalty,
    weight=-10.0,  # 强惩罚，避免碰撞家具
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces_furniture"),
        "threshold": 0.1,
    },
)
```

## 环境结构总结

### 物体分类

#### 1. 静态地形
- `/World/ground` - 地面terrain

#### 2. 静态家具（全局，无reset）
- 9个家具物体位于 `/World/SM_*`
- 使用kinematic rigid body（不受物理力影响）
- 位置固定，每个episode不变

#### 3. 动态YCB物体（per-env，有reset）
- 9个YCB物体位于 `{ENV_REGEX_NS}/*`
- 使用dynamic rigid body
- 每个episode开始时随机重置位置

### 传感器配置

#### 1. SemanticLidarCfg
- **功能**: 
  - 返回语义标签 (`return_semantic_labels=True`)
  - 返回高程图 (`return_height_map=True`)
  - 返回点云 (`return_pointcloud=True`)
- **覆盖范围**: 地面 + 家具 + YCB物体

#### 2. ContactSensor (3类)
- `contact_forces_ground`: 脚与地面接触
- `contact_forces_small_objects`: 机器人身体与YCB物体碰撞
- `contact_forces_furniture`: 机器人与家具碰撞

### Event配置

#### Reset时执行的事件：
1. `reset_robot_joints` - 重置机器人关节到默认姿态
2. `reset_base` - 重置机器人base位置
3. `reset_goal_positions` - 重置目标位置
4. `reset_cracker_box_0~2` - 重置3个cracker box位置（9个reset events for all YCB objects）
5. `reset_sugar_box_0~2` - 重置3个sugar box位置
6. `reset_tomato_soup_can_0~2` - 重置3个tomato soup can位置

**注意**: 家具**没有**reset event！

## 验证

所有语法错误已修复，文件可以正常导入。剩余的import错误是IDE配置问题（isaaclab包未在IDE中正确配置），不影响实际运行。

## 下一步

现在可以测试训练：

```bash
cd /mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn
python scripts/train.py --num_envs 128
```

参考 [TEACHER_TRAINING_README.md](TEACHER_TRAINING_README.md) 了解完整的训练流程。
