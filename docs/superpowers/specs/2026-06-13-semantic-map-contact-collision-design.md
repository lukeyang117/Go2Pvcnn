# 语义高程图接触推断碰撞设计

## 背景

当前 train 配置里，语义碰撞奖励和 flat-small curriculum 使用全局语义 contact sensor：

```text
semantic_contact_small / semantic_contact_large
```

它会把机器人多个 body 和所有语义障碍物建立 filtered contact view。flat-small 修复全列语义物体后，小障碍物数量达到：

```text
10 rows * 20 cols, 每行 small count 总和 400
=> total_small = 8000
```

1024 个环境下，small contact force matrix 接近：

```text
[1024 envs, 13 bodies, 8000 objects, 3]
```

这会让 `Starting the simulation...` 阶段卡在 PhysX/contact-view 初始化。这个瓶颈不是 PPO，也不是 checkpoint，而是全局语义 contact filter 的组合规模。

## 目标

把所有 train 配置中的语义碰撞来源，从“机器人 body 对所有语义物体的全局 filtered contact”改为：

```text
普通机器人 contact force
+ 0.01m 分辨率 semantic/elevation map 近邻查询
=> 推断 small/large semantic collision
```

核心目标：

- 训练时不再创建 `semantic_contact_small` / `semantic_contact_large` 全局 filtered contact sensor。
- 训练 cfg 在 scene creation 前就必须把 `scene.semantic_contact_small` 和 `scene.semantic_contact_large` 设为 `None`，不能只是 reward 不读取它们；否则 PhysX 仍会初始化全局 filtered contact view，启动仍然会很慢。
- PLAY / VIEWER cfg 默认也不再加载 `semantic_contact_small` / `semantic_contact_large`。可视化只保留 semantic objects、semantic/elevation scanner 和普通 `contact_forces`，不背全局 filtered contact sensor。
- 保留完整 semantic objects 和 0.01m 语义高程图，用于观测、reward、curriculum、可视化。
- 语义碰撞奖励和课程升级/降级都基于“真实 body contact 事件 + 语义地图判断”。
- 如果以后需要真实 semantic contact 对照，只放到独立 diagnostic/eval 开关或脚本里，不能作为 train/play/viewer cfg 默认配置。

## 非目标

- 不降低 semantic obstacle 数量作为主要修复手段。
- 不降低 `semantic_height_scanner` 的 0.01m 分辨率。
- 不改变 policy observation/action shape。
- 不改变旧 checkpoint 读取方式。
- 不把每个 obstacle 的具体 USD path 作为训练必需信息。

## 当前可复用基础

已有场景传感器：

```python
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    history_length=3,
    track_air_time=True,
)

semantic_height_scanner = SemanticGridRayCasterCfg(
    pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[1.5, 1.5]),
    mesh_prim_paths=["/World/ground", SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT],
    mesh_semantic_ids={
        "/World/ground": 0,
        SEMANTIC_COURSE_SMALL_ROOT: 1,
        SEMANTIC_COURSE_LARGE_ROOT: 2,
    },
)
```

已有 GPU map 查询基础：

- `extension.mdp.semantic_body_part_clearance`
- `_current_scanner_terrain(scanner, ...)`
- `_current_body_part_sample_points(...)`
- `height_at(...)`
- `semantic_at(...)`

所以新方案不需要从零写语义地图查询，只需要把“contact force 事件”和“语义/高度图局部查询”组合起来。

## 核心判断逻辑

每个 step，对每个环境执行：

```text
1. 从 contact_forces 读取 robot body 的 net_forces_w
2. 找出 force_norm > threshold 的 body
3. 对这些 body 的几何采样点，在 0.01m semantic/elevation map 中查询邻域
4. 如果邻域 semantic id 命中 small 或 large，判定为对应语义碰撞
5. 将 per-step hit 写入 episode sticky flag
```

### Body 分组

使用和现有语义几何 clearance 一致的分组：

```text
foot:  FL/FR/RL/RR foot
calf:  FL/FR/RL/RR calf
thigh: FL/FR/RL/RR thigh
base:  base
```

foot 接触地面是正常事件，所以 foot 的 semantic collision 必须同时满足：

```text
foot force > threshold
且 foot 附近 semantic == small/large
```

calf / thigh / base 正常不应该接触，判断可以更严格：

```text
body force > threshold
且 body 几何邻域 semantic == small/large
```

## 几何查询方式

使用已有的 Go2 近似几何：

```text
foot:  sphere
calf:  capsule samples
thigh: capsule samples
base:  footprint grid / box bottom samples
```

每个采样点不是只查一个 grid cell，而是查半径邻域。因为 semantic map 是 0.01m 分辨率，建议默认：

```text
foot_query_radius_m  = 0.035
calf_query_radius_m  = 0.045
thigh_query_radius_m = 0.045
base_query_radius_m  = 0.030
```

一开始可以复用现有 clearance 的 radius/margin 参数，避免 reward 和 collision bookkeeping 对同一几何体产生两套不一致的空间解释。

## 输出数据

新 helper 输出 per-env tensor：

```text
small_hit: [num_envs] bool
large_hit: [num_envs] bool
small_penalty: [num_envs] float
large_penalty: [num_envs] float
```

内部可选保留 body 级调试信息：

```text
foot_small_hit
calf_small_hit
thigh_small_hit
base_small_hit
```

但默认 TensorBoard 不记录这些细项，避免指标过多。训练期课程指标继续只保留 terrain level。

## Reward 设计

把当前：

```text
semantic_global_contact_collision_reward
```

并入现有：

```text
_semantic_body_part_clearance_reward_term()
```

也就是训练默认不再新增一条独立 `semantic_contact_collision` reward。现有 `semantic_body_part_clearance_reward` 扩展为同一条 body-part semantic safety reward：

```text
clearance penalty:
  body geometry near/under small semantic cells

map-contact collision penalty:
  ordinary body contact force
  + same body geometry semantic-neighborhood hit
```

合并后 reward 语义：

```text
reward = clearance_reward - map_contact_collision_penalty
```

这里的重点是代码层共享同一套 body sample、scanner terrain、semantic/elevation map query、body weights 和半径参数。不要维护两套几何解释，否则 clearance 认为“危险”的区域和 contact collision 认为“碰撞”的区域会不一致。

map-contact penalty 来源不再是 `[N,B,O,3]` 的全局 force matrix，而是：

```text
ordinary body force magnitude
* semantic-neighborhood hit mask
* body weight
* small/large class weight
```

建议保留当前权重语义：

```text
foot weight  = 1.0
calf weight  = 2.0
thigh weight = 2.0
base weight  = 5.0
small_weight = 2.5
large_weight = 2.0
```

flat-small 主要使用 small hit；baseline semantic cfg 同时支持 small/large。

为了避免一个函数变得不可测，内部拆成纯 helper：

```text
query_body_part_semantic_geometry(...)
infer_semantic_contact_from_body_forces(...)
semantic_body_part_clearance_reward(...)
```

外部 cfg 只挂一条 reward term：

```text
semantic_body_part_clearance
```

`rewards.semantic_contact_collision` 在 train/play/viewer cfg 中都设为 `None`。

## Curriculum 设计

当前 flat-small episode-level curriculum 中：

```text
update_episode_small_collision_from_forces(
    semantic_contact_small.data.force_matrix_w
)
```

改为：

```text
update_episode_small_collision_from_map_contacts(
    contact_forces.data.net_forces_w,
    semantic_height_scanner semantic/elevation map,
    robot body poses
)
```

episode sticky flag 语义保持不变：

```text
episode_had_small_collision[env] |= small_hit[env]
```

episode 结束时：

```text
move_up:
  走得够远
  且没有 small semantic collision
  且没有 base_contact / bad_orientation

move_down:
  走得不够远
  或发生 small semantic collision
  或 base_contact / bad_orientation
```

这保持之前“一个环境一个 episode 结束时判断”的课程语义。

## Train 配置变更范围

所有 train cfg 默认做以下调整：

```text
scene.semantic_contact_small = None
scene.semantic_contact_large = None
rewards.semantic_contact_collision = None
rewards.semantic_body_part_clearance = combined clearance + map-contact collision reward
curriculum collision source = map_contact
```

包括：

```text
TeacherElevationTrajectoryMpcSemanticEnvCfg
TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg
```

这里的重点是 **scene 不加载全局 semantic contact sensor 配置**。如果只是把 `rewards.semantic_contact_collision = None`，但 `scene.semantic_contact_small/large` 还存在，IsaacLab 仍会在仿真启动时创建 contact view，性能问题不会解除。

## Play / Viewer / Eval 配置范围

PLAY 和 VIEWER 默认也做以下调整：

```text
scene.semantic_contact_small = None
scene.semantic_contact_large = None
rewards.semantic_contact_collision = None
```

原因和 train 一样：只要 scene 里挂了全局 semantic contact sensor，启动和可视化都会承担 filtered contact view 初始化成本。PLAY/VIEWER 的任务是看策略、看语义物体、看运动行为，不应该默认加载重 contact matrix。

真实 semantic contact 对照如果仍然需要，应放到单独 diagnostic/eval 路径中，并要求显式小规模启用，例如：

```text
--enable-true-semantic-contact
--num_envs 16/32
```

这个开关不属于本次默认 cfg 改动范围，第一步先保证 train/play/viewer 默认不加载全局 semantic contact。

## 性能预期

旧路径主复杂度：

```text
O(num_envs * num_bodies * num_semantic_objects)
```

新路径主复杂度：

```text
O(num_envs * num_body_samples * local_offsets)
```

以 1024 env 估算，body samples 和 local offsets 是固定小常数，不再随 8000 个语义物体线性增长。0.01m map 查询仍然是 GPU tensor 操作，和现有 clearance reward 属于同一量级。

## 风险和处理

### 风险 1：没有真实 contact point，只能用 body pose 近似

处理：

- 使用 body 几何邻域查询，不用单点查询。
- 对 calf/thigh/base 保持比 foot 更高权重。
- 用 controlled crossing eval 对比真实 semantic contact。

### 风险 2：foot 正常接地导致误判

处理：

- foot 必须同时满足 contact force 和 nearby semantic obstacle。
- foot 查询半径不要过大。
- foot 权重低于 calf/thigh/base。

### 风险 3：semantic map 视野只有 1.5m x 1.5m

处理：

- 该判断只服务“当前 body 附近是否接触 semantic obstacle”，视野跟随 base，足够覆盖当前机器人近场。
- 如果 base 高速移动导致边缘漏查，可以后续把 scanner size 从 1.5m 增到 2.0m，但不是第一步。

### 风险 4：训练指标和真实 PhysX semantic contact 不一致

处理：

- 保留小规模 eval 的真实 contact sensor。
- 实施后先跑 16/32 env 对照：真实 semantic contact vs map-contact inferred hit。
- 如果 inferred 漏检，优先调 query radius / force threshold，而不是恢复全局 contact。

## 验证计划

### 静态和单元测试

- 新增纯 tensor 测试：
  - force below threshold 不命中。
  - force above threshold 但 semantic 为 ground 不命中。
  - force above threshold 且 nearby semantic 为 small 命中。
  - foot/calf/thigh/base 权重聚合正确。
- cfg 测试：
  - train cfg 不再挂 `semantic_contact_small/large`。
  - train cfg 的 scene/sensor manager 中不能出现 `semantic_contact_small` 或 `semantic_contact_large`；这是启动性能验收条件，不只是 reward 配置条件。
  - train cfg 的 `semantic_contact_collision` 为 `None`。
  - train cfg 的 `semantic_body_part_clearance` 使用 combined clearance + map-contact collision reward 参数。
  - PLAY / VIEWER cfg 中也不能出现 `semantic_contact_small` 或 `semantic_contact_large`。
  - PLAY cfg 仍不启用 train curriculum。

### 实机 smoke

使用 `env_isaacsim`：

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:0 \
  --num_envs 8 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --planner-backend mpc
```

确认：

```text
Scene 不再创建 semantic_contact_small/large
Reward Manager 没有 semantic_contact_collision
Reward Manager 有 combined semantic_body_part_clearance reward
Curriculum Manager 仍只有 terrain_levels
policy_state/action shape 不变
```

### 1024 启动验证

再次运行用户命令，重点看是否能越过：

```text
Starting the simulation. This may take a few seconds. Please wait...
```

如果能进入训练循环，说明全局 semantic contact filter 瓶颈解除。

### 行为验收

使用 controlled crossing eval：

```text
1000 steps
多个 env
固定 path obstacle opportunity
统计：
  root_crossed
  foot_over_count
  inferred_small_collision
  true_small_contact_eval
  overpass_success
```

最终仍以真实 semantic contact eval 作为验收，而不是只相信 inferred metric。

## 实施顺序

1. 新增 map-contact 碰撞 helper，复用现有 semantic/elevation map 查询。
2. 扩展 `semantic_body_part_clearance_reward`，把 map-contact collision penalty 并入同一条 body-part semantic safety reward。
3. 修改 curriculum collision update，从 `force_matrix_w` 改为 map-contact inferred small hit。
4. train cfg 禁用 `semantic_contact_small/large`。
5. PLAY / VIEWER cfg 禁用 `semantic_contact_small/large`。
6. 确认 train/play/viewer scene creation 不再加载全局 semantic contact sensor 配置。
7. 跑单元测试、cfg 测试、8-env smoke、1024 启动验证。

## 决策

采用方向四作为训练主线：

```text
普通机器人接触力 + 0.01m 语义高程图 = 训练期语义碰撞推断
```

全局 semantic filtered contact 不再作为 train 默认路径，只作为小规模评估和对照工具。

补充决策：

```text
PLAY / VIEWER 默认也不保留全局 semantic filtered contact。
map-contact collision penalty 和 _semantic_body_part_clearance_reward_term() 合并为同一条 body-part semantic safety reward。
```
