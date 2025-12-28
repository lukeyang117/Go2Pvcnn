# Isaac Sim 4.5 Nucleus 资产列表

## 📊 总览

已扫描 Isaac Sim 4.5 官方 Nucleus 服务器上的 USD 资产，整理出以下家具和环境相关的资产。

**Nucleus URL**: `http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac`

---

## 🔧 YCB Objects (标准家居物体)

YCB (Yale-CMU-Berkeley) 数据集是机器人抓取研究的标准物体库。

### Axis_Aligned (21 个物体)
仅几何模型，无物理属性：

- `002_master_chef_can.usd` - 罐头
- `003_cracker_box.usd` - 饼干盒
- `004_sugar_box.usd` - 糖盒
- `005_tomato_soup_can.usd` - 番茄汤罐头
- `006_mustard_bottle.usd` - 芥末瓶
- `007_tuna_fish_can.usd` - 金枪鱼罐头
- `008_pudding_box.usd` - 布丁盒
- `009_gelatin_box.usd` - 果冻盒
- `010_potted_meat_can.usd` - 罐装肉
- `011_banana.usd` - 香蕉
- `019_pitcher_base.usd` - 水壶底座
- `021_bleach_cleanser.usd` - 清洁剂
- `024_bowl.usd` - 碗
- `025_mug.usd` - 马克杯
- `035_power_drill.usd` - 电钻
- `036_wood_block.usd` - 木块
- `037_scissors.usd` - 剪刀
- `040_large_marker.usd` - 大号记号笔
- `051_large_clamp.usd` - 大号夹子
- `052_extra_large_clamp.usd` - 超大号夹子
- `061_foam_brick.usd` - 泡沫砖

### Axis_Aligned_Physics (4 个物体)
**推荐用于物理仿真**，包含碰撞检测和物理属性：

- `003_cracker_box.usd` - 饼干盒（带物理）
- `004_sugar_box.usd` - 糖盒（带物理）
- `005_tomato_soup_can.usd` - 番茄汤罐头（带物理）
- `006_mustard_bottle.usd` - 芥末瓶（带物理）

**使用示例**：
```python
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 带物理属性的 YCB 物体
cracker_box = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
```

---

## 🏢 环境资产 (Environments)

### 1. Simple Warehouse (简易仓库) ⭐推荐
**主文件**：
- `warehouse.usd` - 标准仓库
- `full_warehouse.usd` - 完整仓库
- `warehouse_multiple_shelves.usd` - 多货架仓库
- `warehouse_with_forklifts.usd` - 带叉车仓库

**Props**: 1847 个物体（桶、瓶子、箱子、货架等）

**使用示例**：
```python
warehouse = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
```

### 2. Office (办公室)
**主文件**: `office.usd`

**Props**: 543 个物体（桌子、椅子、文件柜、电脑等）

**使用示例**：
```python
office = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"
```

### 3. Simple Room (简单房间)
**主文件**: `simple_room.usd`

**Props**: 包含墙壁、地板等基础结构

### 4. Hospital (医院)
**主文件**: `hospital.usd`

**Props**: 包含医疗设备、病床等

### 5. Modular Warehouse (模块化仓库)
可自定义组合的仓库组件

---

## 📦 道具资产 (Props)

### Blocks (方块)
8 个基础方块，用于测试和简单场景：

- `block.usd` - 标准方块
- `block_instanceable.usd` - 可实例化方块
- `blue_block.usd`, `green_block.usd`, `red_block.usd`, `yellow_block.usd` - 彩色方块
- `nvidia_cube.usd` - NVIDIA 立方体
- `basic_block.usd` - 基础方块

**使用示例**：
```python
block = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block_instanceable.usd"
```

---

## 🎯 推荐的使用场景

### 场景 1: 仓库机器人导航
```python
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 仓库环境
warehouse = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"

# 可交互的物体
cracker_box = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
```

### 场景 2: 办公室机器人
```python
# 办公室环境
office = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"

# 办公室道具在 Props 子目录中
# office_props = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_*.usd"
```

### 场景 3: 物体抓取测试
```python
# 使用带物理属性的 YCB 物体
sugar_box = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"
mustard = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"
```

---

## 📁 文件位置

所有生成的资产列表保存在：

1. **快速概览**: `assets/furniture_assets_quick.json`
   - 包含主要类别和子目录结构
   
2. **详细列表**: `assets/detailed_furniture_list.json`
   - 包含 2400+ 个具体 USD 文件路径
   - YCB 物体完整列表
   - 仓库和办公室所有 Props

---

## ⚡ 快速开始

在你的 Go2 训练环境中使用这些资产：

```python
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 1. 加载仓库环境
env_cfg.scene.warehouse = AssetBaseCfg(
    prim_path="/World/Warehouse",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
    )
)

# 2. 添加可碰撞物体
env_cfg.scene.obstacle = RigidObjectCfg(
    prim_path="/World/Obstacles/Box",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
    )
)

# 3. 在奖励函数中检测碰撞
# 参考 mdp/rewards.py 中的碰撞检测示例
```

---

## 🔍 详细信息

- **YCB 物体总数**: 25 个（21 个几何 + 4 个带物理）
- **仓库 Props**: 1847 个
- **办公室 Props**: 543 个
- **环境场景**: 5 个完整场景

所有资产都可以通过 `ISAAC_NUCLEUS_DIR` 直接引用，无需下载到本地。Isaac Sim 会自动从 Nucleus 服务器加载。

---

## 📝 注意事项

1. **物理仿真**: 如果需要碰撞检测，请使用 `Axis_Aligned_Physics` 目录下的 YCB 物体
2. **性能优化**: 使用 `*_instanceable.usd` 文件可以提高多实例场景的性能
3. **网络连接**: 首次加载资产需要从 AWS S3 下载，之后会缓存到本地
4. **路径引用**: 始终使用 `ISAAC_NUCLEUS_DIR` 变量而不是硬编码 URL

---

生成时间: 2025-11-19
Isaac Sim 版本: 4.5
