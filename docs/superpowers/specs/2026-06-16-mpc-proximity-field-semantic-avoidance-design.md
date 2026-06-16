# MPC Proximity Field Semantic Avoidance Design

日期：2026-06-16

主线：`Go2Pvcnn/extension/batch_mpc_planner`

## 1. 目标

当前目标是让训练时可以使用：

```text
RL env 数量 == MPC env 数量
```

并且在至少 `1024` 个强化学习环境、`1024` 个 MPC 规划环境下，使用 `TeacherElevationTrajectoryMpcSemanticEnvCfg` 启动真实 IsaacLab 训练检查时不因为 MPC semantic avoidance 的显存临时张量爆掉。

本设计只替换现有 `parametric_semantic_avoidance` loss 的内部计算方式：

```text
旧：foot/root/touchdown 与 22500 个 grid cell 做 dense pairwise distance
新：semantic/height map 预计算 soft proximity field，再用 grid_sample 查询 foot/root/touchdown risk
```

## 2. 硬约束

- 不新增 MPC loss 项。
- 不新增 loss 名称。
- 不删除 `parametric_semantic_avoidance`。
- 不改变 `_parametric_sampled_frame_losses()` 的返回 key 集合。
- 不新增 decode-time touchdown projection。
- 不新增优化后 touchdown/root/foot snapping。
- 不新增 hard foot separation。
- 不把 `decode_parametric_trajectory()` 改成语义修正器。
- 只能在已有 loss / 已有配置项范围内调权重或参数；如果需要新增可调参数，必须先单独确认。
- 必须保持 `docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html` 的验收语义不退步。

## 3. 当前问题

当前 `_parametric_semantic_avoidance_loss()` 会构造类似下面的临时张量：

```text
root_delta: [B, H, 22500, 2]
foot_delta: [B, H, 4, 22500, 2]
touchdown_delta: [B, 4, 22500, 2]
```

其中：

```text
B = MPC batch size
H = horizon_steps，当前为 25
22500 = semantic_height_scanner 的 150 x 150 grid cell
```

当 `B=128` 时，这些张量已经可能达到数 GB 级别；当 `B=1024` 或 `4096` 时，不可能作为训练热路径保留。

## 4. 设计方案

保留 `parametric_semantic_avoidance` 这个 loss 的职责：

- root path 避开 high-small / large obstacle。
- foot path 避免贴近 high-risk semantic obstacle。
- touchdown 避免落在 high-risk semantic obstacle 附近。

替换计算方式：

```text
semantic_map / height_map
-> risky_mask
-> soft proximity field
-> grid_sample(root_xy / foot_xy / touchdown_xy)
-> [B] loss
```

### 4.1 risky mask

输入：

```text
terrain.height_map:   [B, 150, 150]
terrain.semantic_map: [B, 150, 150]
root_pos:             [B, H, 3]
command:              [B, 3]
```

保留现有 high-small / large 判定语义：

```text
small_mask = semantic_map == 1
large_mask = semantic_map == 2
root_ground0 = height_at(terrain, root_pos[:, :1, :2])
high_small = small_mask & ((height_map - root_ground0[:, None, None]) > high_small_relative_height_m)
risky_mask = large_mask | high_small
```

### 4.2 corridor gating

为了避免全场障碍都对当前命令产生同等影响，保留现有 command-frame corridor 语义：

```text
heading, left = command_frame_axes(command, root_yaw0)
delta0 = grid_xy - root_pos[:, :1, :2]
along0 = dot(delta0, heading)
lateral0 = dot(delta0, left)
linear_candidate = -0.10 <= along0 <= 1.50 and abs(lateral0) <= 0.45
candidate = risky_mask & linear_candidate & active_command
```

这里 `candidate` 只用于构建 field，不再用于构造 `[B,H,4,22500]` 的 dense distance。

### 4.3 soft proximity field

从 `candidate` 生成 soft risk field：

```text
candidate:  [B, 150, 150]
risk_field: [B, 1, 150, 150]
```

第一版使用 GPU tensor 上的多尺度 pooling，不做 CPU connected component，不做精确 EDT：

```text
base = candidate.float().unsqueeze(1)
near = max_pool2d(base, kernel_size=5, stride=1, padding=2)
mid  = max_pool2d(base, kernel_size=15, stride=1, padding=7)
far  = max_pool2d(base, kernel_size=31, stride=1, padding=15)
risk_field = clamp(1.0 * near + 0.4 * mid + 0.1 * far, max=1.0)
```

这不是新 loss，只是现有 semantic avoidance 内部的风险查询表示。`risk_field` 不需要梯度；它可以在 `no_grad` 语义下由当前 terrain map 构造。梯度只需要从 `grid_sample()` 的 query 坐标回传到 MPC variables。

### 4.4 world xy 查询

新增内部 helper，把 world-frame xy 转成 `grid_sample` 所需的归一化坐标：

```text
points_xy: [B, P, 2]
grid:      [B, P, 1, 2]
field:     [B, 1, 150, 150]
sampled:   [B, P]
```

查询对象：

```text
root_pos[..., :2]      -> root_risk      [B, H]
foot_pos[..., :2]      -> foot_risk      [B, H, 4]
touchdown_w[..., :2]   -> touchdown_risk [B, 4]
```

### 4.5 loss 公式

保留旧 loss 的三部分语义和大致权重比例：

```text
semantic_avoidance =
  30.0 * root_risk_mean
+ 20.0 * foot_risk_mean
+ 25.0 * touchdown_risk_mean
```

其中：

- `root_risk_mean` 对 `[B,H]` 求每 env 平均。
- `foot_risk_mean` 对 `[B,H,4]` 求每 env 平均，可乘现有 `swing_prob` 或 `contact_prob` 权重，但不新增外部配置项。
- `touchdown_risk_mean` 对 `[B,4]` 求每 env 平均。
- 如果当前 env 没有 candidate obstacle，该 env loss 为 `0`。

## 5. 显存预期

旧热路径最大张量：

```text
foot_delta: [B,25,4,22500,2]
```

新热路径：

```text
risk_field: [B,1,150,150]
foot_risk:  [B,25,4]
root_risk:  [B,25]
td_risk:    [B,4]
```

对 `B=1024`：

```text
risk_field float32 ~= 1024 * 150 * 150 * 4 = 87.9 MiB
foot_risk float32  ~= 1024 * 25 * 4 * 4 = 0.39 MiB
```

这会移除当前最危险的 `22500` pairwise 维度。

## 6. 行为兼容要求

实现后必须满足：

- `parametric_semantic_avoidance` 仍对 high-small / large obstacle 产生非零 loss。
- 没有语义障碍或没有 active command 时，该 loss 仍为零或接近零。
- foot/root/touchdown 越靠近 candidate obstacle，risk 越大。
- foot/root/touchdown 从高风险区域向低风险区域移动时，loss 对 query xy 有可用梯度。
- `cost_breakdown` 中的 key 不变化。
- `MpcPlannerCfg.losses` 不新增 loss term。

## 7. 低小障碍验收

必须参考并保持下面文档的验收标准：

```text
docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html
```

最低验收要求：

```text
plane_env_count > 0
crossing_leg_count > 0
fk_semantic_collision_count == 0
fk_semantic_collision_rate == 0
fk_semantic_min_clearance_over_semantic_m >= 0
planned_vs_fk_foot_error_crossing_leg_max_m <= 测试阈值
```

如果已有测试命令中的阈值为 `0.05m` 或 `0.08m`，沿用已有测试阈值，不因为本次显存优化放宽。

必须覆盖：

- forward / backward
- left / right
- turn_left / turn_right
- diag_fl / diag_fr
- mixed_turn_l / mixed_turn_r
- 单次 plan
- 多次 replan
- horizon 为 `25`

## 8. 1024 RL Env + 1024 MPC Env 验收

新增真实 IsaacLab 检查，必须使用：

```text
TeacherElevationTrajectoryMpcSemanticEnvCfg
```

测试目的：

- `scene.num_envs = 1024`
- `mpc_planner_cfg.runtime.parallel_plan_batch_size = 1024`
- planner-owned reference cache 开启
- semantic scanner 开启
- 真实 env 启动、step、replan 不因为 MPC semantic avoidance 显存爆掉

验收命令必须使用：

```text
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python
```

日志必须记录：

- `CUDA_VISIBLE_DEVICES`
- `num_envs`
- `mpc_num_envs`
- `horizon_steps`
- `replan_interval_steps`
- `parallel_plan_batch_size`
- GPU 显存峰值或 `torch.cuda.max_memory_allocated/reserved`
- 是否完成至少一次 MPC replan
- 是否出现 CUDA OOM
- 是否出现 NaN/Inf result

通过条件：

```text
env build exits 0
至少完成一次 planner replan
CUDA OOM 未出现
NaN/Inf result 未出现
parallel_plan_batch_size == 1024
num_envs == 1024
```

## 9. 测试矩阵

必须新增或更新的测试类别：

1. 纯 tensor 单元测试：
   - proximity field shape 为 `[B,1,H,W]`。
   - no-obstacle 输出零 risk。
   - obstacle 附近 risk 大于远处 risk。
   - `grid_sample` 查询对 `points_xy` 有非零梯度。
   - 不构造 `[B,H,4,H*W]` 或 `[B,H,4,H*W,2]` 级别张量。

2. planner focused 测试：
   - `parametric_semantic_avoidance` key 保留。
   - cost breakdown key 集合不变。
   - low-small / high-large 相关 focused tests 不退步。

3. 低小障碍验收：
   - 使用现有 2026-05-28 验收相关测试环境和指标。
   - 指标不能因为本次显存优化变差到失败。

4. 真实 IsaacLab 1024/1024：
   - 使用 `TeacherElevationTrajectoryMpcSemanticEnvCfg`。
   - 跑到至少一次 MPC replan。
   - 记录 GPU memory。

## 10. 不做事项

- 不做精确 Euclidean Distance Transform。
- 不做 CPU per-env connected component。
- 不引入新的 planner backend。
- 不改变 `semantic_height_scanner` 分辨率作为第一手段。
- 不把 RL env 和 MPC env 再拆成抽样关系来规避问题。

## 11. 自查

- 本设计没有新增 loss 项。
- 本设计没有新增 loss 名称。
- 本设计只替换 `parametric_semantic_avoidance` 内部的风险计算表示。
- 本设计保留 25 帧 MPC horizon。
- 本设计不修改 low-small 验收阈值。
- 本设计增加 `TeacherElevationTrajectoryMpcSemanticEnvCfg` 的 1024 env / 1024 MPC 真实显存验收。
