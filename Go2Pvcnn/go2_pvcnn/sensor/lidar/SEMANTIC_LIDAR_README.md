# Semantic LiDAR Sensor

## 概述

`SemanticLidarSensor` 是对标准 `LidarSensor` 的扩展，增加了**语义分类**功能。它能够根据射线击中的物体类型（mesh）自动分类为三大类：

1. **地形 (Terrain, ID=1)**: 静态环境元素（地面、墙壁、地板等）
2. **动态障碍物 (Dynamic Obstacle, ID=2)**: 可移动的物体（盒子、罐子等需要避开的物体）
3. **贵重物品 (Valuable Item, ID=3)**: 重要物体（家具如沙发、椅子、桌子等需要特殊处理的物品）

## 继承结构

```
RayCaster (Isaac Lab)
    ↓
LidarRayCaster (自定义，支持动态 mesh)
    ↓
LidarSensor (标准 LiDAR，距离+点云)
    ↓
SemanticLidarSensor (语义 LiDAR，分类+标签)
```

## 核心组件

### 1. SemanticLidarData
扩展 `LidarSensorData`，添加语义信息字段：
- `semantic_labels`: torch.Tensor, shape (num_envs, num_rays)
  - 每条射线的语义类别 ID
- `mesh_ids`: torch.Tensor, shape (num_envs, num_rays) [可选]
  - 每条射线击中的 mesh 原型 ID（用于调试）

### 2. SemanticLidarCfg
扩展 `LidarCfg`，添加语义分类配置：
- `semantic_class_mapping`: dict[str, list[str]]
  - 语义类别名称到关键词列表的映射
  - 通过在 mesh_prim_paths 中搜索关键词来分类
- `return_semantic_labels`: bool (默认 True)
  - 是否计算并返回语义标签
- `return_mesh_ids`: bool (默认 False)
  - 是否返回具体的 mesh ID

### 3. SemanticLidarSensor
扩展 `LidarSensor`，实现语义分类逻辑：
- `mesh_id_to_semantic_class`: dict[int, int]
  - Warp mesh ID → 语义类别 ID 的映射
- `_classify_mesh_path(mesh_prim_path)`: 根据路径关键词分类 mesh
- `_initialize_warp_meshes()`: 初始化时建立 mesh 到语义类别的映射
- `_compute_semantic_labels(env_ids)`: 根据射线击中的 mesh 计算语义标签
- `get_semantic_labels()`: 获取语义标签数据

## 使用示例

### 基本配置

```python
from go2_pvcnn.sensor.lidar import SemanticLidarCfg, LivoxPatternCfg

cfg = SemanticLidarCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    mesh_prim_paths=[
        "/World/ground",              # 将被分类为 terrain
        "/world/SM_sofa_1",           # 将被分类为 valuable
        "{ENV_REGEX_NS}/cracker_box_0/_03_cracker_box",  # 将被分类为 dynamic_obstacle
        # ... 更多 mesh
    ],
    semantic_class_mapping={
        "terrain": ["ground", "wall", "floor"],
        "dynamic_obstacle": ["cracker_box", "sugar_box", "tomato_soup_can"],
        "valuable": ["sofa", "armchair", "table"],
    },
    return_semantic_labels=True,
    max_distance=40.0,
)
```

### 在环境中使用

```python
# 创建传感器
lidar = SemanticLidarSensor(cfg)

# 获取标准 LiDAR 数据
distances = lidar.get_distances()        # (num_envs, num_rays)
pointcloud = lidar.get_pointcloud()      # (num_envs, num_rays, 3)

# 获取语义分类
semantic_labels = lidar.get_semantic_labels()  # (num_envs, num_rays)

# 语义 ID 含义:
# 0 = 未击中/未知
# 1 = 地形 (terrain)
# 2 = 动态障碍物 (dynamic_obstacle)
# 3 = 贵重物品 (valuable)

# 按语义类别过滤点云
terrain_mask = (semantic_labels == 1)
obstacle_mask = (semantic_labels == 2)
valuable_mask = (semantic_labels == 3)

terrain_points = pointcloud[terrain_mask]
obstacle_points = pointcloud[obstacle_mask]
valuable_points = pointcloud[valuable_mask]
```

### 在 RL 环境配置中使用

替换标准的 `LidarCfg` 为 `SemanticLidarCfg`：

```python
# 在 go2_lidar_env_cfg.py 中
from go2_pvcnn.sensor.lidar import SemanticLidarCfg, LivoxPatternCfg

@configclass
class Go2LidarSceneCfg(InteractiveSceneCfg):
    # 使用 SemanticLidarCfg 替代 LidarCfg
    lidar: SemanticLidarCfg = SemanticLidarCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=[...],
        semantic_class_mapping={...},
        return_semantic_labels=True,
        # ... 其他配置
    )
```

### 在 MDP 观测函数中使用

```python
def lidar_semantic_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回 LiDAR 距离和语义标签的组合观测."""
    sensor: SemanticLidarSensor = env.scene.sensors[sensor_cfg.name]
    
    distances = sensor.get_distances()           # (num_envs, num_rays)
    semantic_labels = sensor.get_semantic_labels()  # (num_envs, num_rays)
    
    # 组合为单一观测 (num_envs, num_rays * 2)
    obs = torch.stack([distances, semantic_labels.float()], dim=-1)
    return obs.reshape(env.num_envs, -1)
```

## 分类逻辑

### 关键词匹配
语义分类基于 **不区分大小写的关键词匹配**：

1. 遍历 `semantic_class_mapping` 中的每个类别
2. 对于每个类别的关键词列表，检查是否出现在 mesh prim 路径中
3. 返回第一个匹配的类别 ID
4. 如果没有匹配，返回 0 (未知)

示例：
```python
mesh_prim_path = "/world/SM_Sofa_1"
keywords = ["sofa", "SM_Sofa"]

# "sofa" 出现在路径中（不区分大小写）
# → 分类为 "valuable" (ID=3)
```

### 默认分类规则
如果不提供 `semantic_class_mapping`，使用以下默认规则：

```python
{
    "terrain": ["ground", "wall", "floor", "plane"],
    "dynamic_obstacle": [
        "cracker_box", "sugar_box", "tomato_soup_can",
        "_03_cracker_box", "_04_sugar_box", "_05_tomato_soup_can"
    ],
    "valuable": [
        "sofa", "armchair", "table", "chair", "desk",
        "SM_Sofa", "SM_Armchair", "SM_Table"
    ],
}
```

## 实现细节

### 初始化流程

1. **`__init__`**: 创建 `SemanticLidarData` 容器
2. **`_initialize_warp_meshes`**: 
   - 调用父类初始化 warp meshes
   - 遍历所有 mesh，根据 prim 路径分类
   - 建立 `mesh_id → semantic_class` 映射
3. **`_initialize_rays_impl`**:
   - 调用父类初始化射线
   - 分配 `semantic_labels` 和 `mesh_ids` 缓冲区

### 更新流程

1. **`_update_buffers_impl`**:
   - 调用父类执行 ray casting
   - 调用 `_compute_semantic_labels` 计算语义标签

2. **`_compute_semantic_labels`**:
   - 获取 `ray_hits_w` 结果
   - 检测无穷点（未击中）
   - **[待实现]** 从 raycast 结果中提取 mesh ID
   - 根据 mesh ID 查找语义类别
   - 填充 `semantic_labels` 缓冲区

### 当前限制与改进方向

**当前实现的限制**:
当前的 `_compute_semantic_labels` 是一个**简化版本**，因为 Warp raycast 内核没有返回被击中的 mesh ID。

**改进方案**:
需要修改 `go2_pvcnn/utlis/warp/kernels.py` 中的 `raycast_mesh_kernel_grouped_transformed` 内核，添加 mesh ID 输出：

```python
# 在 warp kernel 中添加
ray_mesh_id: wp.array(dtype=wp.int32),  # 新增输出

# 在命中时记录
if hit_success and t < ray_distance_buf and t > min_dist:
    ray_hits[tid] = start + direction * t
    ray_mesh_id[tid] = mesh_idx  # 记录击中的 mesh 索引
```

然后在 `raycast_mesh_grouped` 函数中返回 mesh ID：
```python
return ray_hits_output, ray_distance, ray_normal, ray_face_id, ray_mesh_id
```

完成后，`_compute_semantic_labels` 就能直接使用 mesh ID 查询语义类别。

## 完整示例

参见 `semantic_lidar_example.py` 获取完整的配置和使用示例。

## API 参考

### SemanticLidarSensor

#### 方法
- `get_distances(env_ids=None)` → torch.Tensor
  - 继承自父类，返回距离测量
- `get_pointcloud(env_ids=None)` → torch.Tensor
  - 继承自父类，返回点云数据
- `get_semantic_labels(env_ids=None)` → torch.Tensor
  - 返回语义类别标签
- `get_mesh_ids(env_ids=None)` → torch.Tensor
  - 返回 mesh ID（需要配置 `return_mesh_ids=True`）

#### 属性
- `data`: SemanticLidarData
  - 传感器数据容器
- `mesh_id_to_semantic_class`: dict[int, int]
  - Mesh ID 到语义类别的映射

### SemanticLidarCfg

#### 新增配置项
- `semantic_class_mapping`: dict[str, list[str]]
  - 语义类别到关键词的映射
- `return_semantic_labels`: bool
  - 是否返回语义标签
- `return_mesh_ids`: bool
  - 是否返回 mesh ID

#### 继承的配置项
所有 `LidarCfg` 的配置项均可使用（距离、点云、噪声等）。
