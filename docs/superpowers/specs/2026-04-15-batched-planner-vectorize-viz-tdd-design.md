# Batched Planner：向量化提速 + 可视化对齐 + TDD 测试体系

## 元信息

- **日期**：2026-04-15
- **范围**：`Go2Pvcnn/extension/batched_planner/`、`Go2Pvcnn/extension/viz/go2_foostep_planner.py`、`Go2Pvcnn/tests/`
- **金标准**：`raw/kinematic_footsteps/scripts/go2fp/` 的输出
- **目标设备**：CUDA GPU only
- **目标环境数**：N=2048
- **允许手段**：PyTorch 向量化 + Triton custom kernel

## 1. 背景与问题

### 1.1 性能瓶颈

训练输出（N=1, `--verbose-planner`）：

```
PlannerTiming[plan=696.95ms/1 swing_targets=637.29ms/1 terrain=28.04ms/1
              footholds=9.05ms/1 ik=8.30ms/1 terrain_est=5.46ms/1
              fk=5.35ms/1 base_solve=4.41ms/1 +7 more]
```

`swing_targets` 占 plan 耗时 91%。根因：`swing.py` 中 `batched_compute_swing_targets` 的双层 Python `for batch_idx × for leg` 循环，内部 `_leg_swing_progress_and_stance_anchor` 还有 `for i in range(T)` + `.item()` 的逐帧 GPU→CPU 同步。时间复杂度 **O(N·T)** 全部是 Python 串行。

N=2048 时线性外推：swing_targets ≈ 1300 秒，完全不可用。

### 1.2 可视化未对齐

当前 `go2_foostep_planner.py` 已有 `--planner-playback-mode direct` 支持 kinematic pose 写入，但主循环仍每帧调 `env.step(zero_actions)`，且控制流是「每帧步进 → 读 cache」而非 raw viewer 的「一次性规划 → 逐帧回放 → 播完再规划」。需要改造为 raw 风格的 plan-once/replay-then-replan 控制流，并彻底去掉物理步进。

### 1.3 测试缺乏 raw 交叉验证

现有 14 个 `test_batched_*.py` 仅自对比，缺少与 raw/kinematic_footsteps 的输出一致性验证。

## 2. 训练侧 selective replan 约束

### 2.1 现有 manager 的 selective replan 机制

`BatchedTrajectoryManager.refresh_from_env()` 不会每步对所有 N=2048 个 env 重新规划。它通过 `_compute_replan_mask()` 计算 per-env 布尔 mask，只有满足以下条件之一的 env 才触发 replan：

**Per-env 条件**（产生稀疏 mask）：
- **command 变化**：任一分量 `|cmd_new[i] - cmd_old[i]| > 1e-6`（per-env per-component，不是向量范数）
- **env reset**：`episode_length_buf < last_episode_length_buf` 或 `pending_reset_mask`
- **间隔到期**：`episode_length_buf - last_replan_episode_length_buf >= reference_replan_interval_steps`

**全局退化为全量 replan 的分支**（返回 `torch.ones(num_envs)`）：
- `cache is None`（首次调用）
- `last_episode_length_buf` / `last_commands` / `last_replan_episode_length_buf` 为 None
- `episode_length_buf` 或 `commands` 与上次 shape 不一致
- `cache.horizon_length() != reference_trajectory_horizon`（horizon 配置变更）

对需要 replan 的 env 子集，manager 用 `_subset_state`、`_subset_terrain`、`index_select` 提取子集数据，只对子集调 `batched_generate_trajectory(sub_terrain, sub_states, sub_commands, ...)`，然后通过 `masked_write_reference_cache_rows` 把结果写回全量 cache。不需要 replan 的 env 继续推进 `phase_counter`。

### 2.2 对向量化改造的影响

这意味着：
1. **planner 的实际输入 N 是动态的**：可能是 10、100、500、2048，取决于当步哪些 env 触发了 replan
2. **向量化代码必须支持任意 N**，不能假设 N 是固定的或是 2 的幂
3. **性能目标应按「最大子集 N」评估**：最坏情况是全部 2048 env 同时 replan（首步或全局 reset 后），但稳态下每步的 replan 子集通常远小于 N
4. **Triton kernel 的 grid 维度必须适配动态 N**：不能 hardcode block 数

### 2.3 对测试的影响

L2 测试需覆盖：
- N=1（单 env replan）
- N=小子集（如 N=32 从 2048 中选出）
- N=全量 2048（首步 / reset 后全 replan）
- **全局退化分支**：首帧无 cache、horizon 变更、shape 不一致等触发全量 replan
- **混合 replan**：manager 级测试验证 subset → planner → masked_write → cache 的完整路径

L3 性能基准需增加：
- **稳态 replan 场景**：模拟 N_total=2048 但每步只有 ~5-10% env 触发 replan（N_replan ≈ 100-200）
- **burst replan 场景**：全量 N=2048 同时 replan

## 3. 提速改造设计

### 3.1 swing.py — 核心瓶颈（91% 耗时）

#### 3.1.1 `_leg_swing_progress_and_stance_anchor` 向量化

当前两个 `for i in range(n)` + `.item()` 循环的语义：给定 stance 布尔序列，找连续 swing 区间起止点，算归一化 progress。

向量化算法：

1. **边界检测**：`torch.diff(stance_bool, dim=time_dim)` 检测 stance→swing（lift-off）和 swing→stance（touch-down）事件
2. **区间 ID 分配**：`torch.cumsum(lift_off_events, dim=time_dim)` × `is_swing` 给每段连续 swing 唯一 ID，shape `(N, T, 4)`
3. **区间内序号**：`frame_idx - first_frame_of_run`。`first_frame_of_run` 通过 `torch.scatter_reduce(frame_idx, run_id, reduce='amin')` 或等价的 `segment_reduce` 获取，再用 `run_id` 作 index gather 回每帧。PyTorch ≥2.0 的 `scatter_reduce` 支持 `'amin'`/`'amax'`；若版本不支持，用 `torch.zeros().scatter_(dim, run_id, frame_idx, reduce='amin')` 替代
4. **区间长度**：`torch.zeros().scatter_add_(dim, run_id, ones)` 统计每段 swing 帧数 → `run_id` gather 广播回每帧
5. **归一化**：`progress = idx_in_run.float() / (run_length - 1).clamp(min=1).float()`

全部在 `(N, T, 4)` 上并行，消除所有 Python 循环和 `.item()` 同步。

#### 3.1.2 `batched_compute_swing_targets` 全张量化

消除 `for batch_idx in range(batch_size): for leg in range(4):` 外层循环：

- `_compute_swing_apex`：已 element-wise，直接支持 `(N, 4)` 输入
- `_swing_phase_targets`（Hermite 插值）：当前使用 `if torch.any(mask_first/second)` 分支和 masked write（`swing.py:65-79`），需重构为 branchless `torch.where` 在完整 `(N, T, 4)` 上操作，输出 `(N, T, 4, 3)`。这不是简单的 shape 提升，需要消除条件分支。
- `torch.where(stance, anchor, arc)` 合并

#### 3.1.3 Triton 增强（第二阶段）

N=2048 时如 bench 显示 kernel launch overhead 显著，写 fused Triton kernel：一个 block 处理一个 `(env, leg)` 的全 T 帧，融合 swing progress + Hermite + merge。Triton grid 维度从输入 N 动态计算（`grid = (N * 4,)`），不 hardcode。

### 3.2 terrain.py — `max_height_along_segment`

当前：`for idx in range(self.batch_size)` + `.item()` 逐 env 取端点，逐个 `torch.linspace` + `_sample_map`。在 `trajectory.py` 被调用 4 次（per leg）。

改造：

1. 端点保持 tensor `(N, 2)`，不调 `.item()`
2. Batch 插值：`t = linspace(0, 1, n_samples).view(1, -1, 1)` → `points = lerp(p0, p1, t)` 得 `(N, n_samples, 2)`
3. 一次 `F.grid_sample` 采样所有 env × sample。需确保归一化、边界模式和 dtype 与现有 `_sample_map` 一致（当前 `_sample_map` 使用 `align_corners=True, mode='bilinear', padding_mode='border'`），否则 L1/L2 会因插值差异失败
4. `torch.amax(sampled, dim=1)` 得 `(N,)` 结果
5. `trajectory.py` 的 4 次 per-leg 调用合并为一次：stack `(N×4, 2)` → 采样 → reshape `(N, 4)`

采样点数固定上界（按 resolution 对应的最大合理步数，如 32），短段多余的采样结果 mask 掉。

### 3.3 foothold.py

#### `_precompute_spiral_offsets`

Python 嵌套循环 → `torch.meshgrid` 规则网格 + 按距中心距离排序。搜索结果最终取全局 argmin score，螺旋顺序无影响。仅初始化时调一次，不是热路径，但改了更干净。

#### `batched_evaluate_touchdowns`

`reasons` 列表的 `.item()` 调用改为 lazy（仅 verbose 时取）或改为整型 reason code tensor。

### 3.4 base_solver.py — `batched_solve_base_height` EMA

`for t in range(num_frames)` 串行但 batch 维并行。T 通常 <30 帧，N=2048 时每帧内已让 GPU 满载。

**策略：** bench 后决定。如果 `base_solve` 阶段 <5ms（N=2048），保持现状；否则用 parallel scan 重写。

### 3.5 terrain_estimator.py — `batched_estimate_terrain` EMA

同 3.4 逻辑。bench 后决定。

### 3.6 ik.py

已向量化，保持现状。可选优化：float64 → float32 降精度提速（需验证 IK 解稳定性）。

### 3.7 预期性能目标

| N | 当前估计 | 目标 |
|---|---------|------|
| 1 | ~700ms | <10ms |
| 64 | ~45s | <10ms |
| 256 | ~180s | <15ms |
| 1024 | ~720s | <30ms |
| 2048 | ~1300s | <50ms |

## 4. 可视化对齐设计

### 4.1 核心改造：纯 kinematic 回放

当前 `go2_foostep_planner.py` 依赖 Isaac Lab 物理步进。改为 raw `viewer.py` 的逻辑：

1. 接收 teleop 命令
2. 调 planner `batched_generate_trajectory` 生成完整轨迹
3. 立即显示 touchdown 标记和足端轨迹弧线
4. 逐帧 kinematic 回放：`write_root_pose_to_sim` + `write_joint_state_to_sim`
5. 播完后用末帧状态作为下次规划的初始状态
6. 循环

无物理约束、无 `env.step`。机器人忠实按 planner 输出表演。

### 4.2 具体改动

| 项目 | 当前 | 改后 |
|------|------|------|
| 播放驱动 | `env.step(zero_actions)` 物理步进 | 逐帧 kinematic write |
| 规划触发 | 每帧走物理再读 cache | 一次性生成 → 逐帧回放 → 播完重规划 |
| touchdown 显示时机 | 和轨迹同时更新 | 规划完成后立即显示 |
| 触地标记颜色 | 已按 `LEG_COLORS` 四色区分 | 保持（红/绿/蓝/黄） |
| 状态链 | env 物理状态 → planner | planner 末帧 → 下次 planner 输入 |

### 4.3 不改的部分

- 地形点云可视化（当前 subsampled ray_hits 球）
- 命令箭头（单绿箭头）
- 相机跟随逻辑
- teleop 键位
- 根轨迹橙色球链（保留，加 `--show-root-traj` 开关）

### 4.4 数据验证覆盖层（`--debug-overlay`）

可选 debug 模式：

- stance/swing 状态颜色标记
- IK 关节角度超限位黄色警告
- infeasible touchdown X 标记

## 5. 测试体系设计

### 5.1 架构

```
Go2Pvcnn/tests/
├── conftest.py                            # 共享 fixtures
├── fixtures/
│   ├── terrain_adapter.py                 # 桥接 raw terrain 和 batched terrain
│   └── golden/                            # 串行版 .pt golden reference 文件
├── test_cross_validation_raw.py           # L1: raw ↔ batched 交叉验证
├── test_swing_vectorized.py               # L2: swing 向量化回归
├── test_terrain_vectorized.py             # L2: terrain 向量化回归
├── test_foothold_vectorized.py            # L2: foothold 向量化回归
├── test_base_solver.py                    # L2: base solver 回归
├── test_trajectory_integration.py         # L2: 端到端轨迹回归
├── test_ik.py                             # L2: IK/FK
├── test_gait.py                           # L2: gait
├── test_viz_playback.py                   # L4: 可视化回放逻辑
├── test_batched_manager.py                # L2: manager 集成（扩展现有文件）
└── benchmarks/
    └── bench_planner_scaling.py           # L3: 性能基准
```

### 5.2 L1：raw ↔ batched 交叉验证（核心）

金标准是 `raw/kinematic_footsteps/scripts/go2fp/` 的输出。

#### 前置条件：参数对齐

两边 config 默认值不同（raw `duty_factor=0.55, max_reach=0.22, replan_stop=0.03`；batched `duty_factor=0.6, max_reach=0.15, replan_stop=0.05`）。L1 测试必须使用统一的 **golden alignment dict** 显式传参：

```python
GOLDEN_ALIGNMENT = {
    "gait_name": "trot",
    "step_freq": 2.0,
    "duty_factor": 0.55,
    "step_height": 0.08,
    "hip_height": 0.30,
    "body_clearance_margin": 0.012,
    "foothold_search_radius": 0.15,
    "foothold_search_step": 0.03,
    "max_foothold_step_down": 0.10,
    "max_touchdown_xy_reach": 0.22,
    "replan_stop_speed": 0.03,
}

@pytest.fixture
def aligned_configs():
    """raw TrajectoryConfig 和 BatchedTrajectoryConfig 全部从 GOLDEN_ALIGNMENT 构造，
    确保共享参数完全一致。"""
    raw_cfg = TrajectoryConfig(**{k: v for k, v in GOLDEN_ALIGNMENT.items()
                                  if k in TrajectoryConfig.__dataclass_fields__})
    batched_cfg = BatchedTrajectoryConfig(**{k: v for k, v in GOLDEN_ALIGNMENT.items()
                                             if k in BatchedTrajectoryConfig.__dataclass_fields__})
    return raw_cfg, batched_cfg
```

**有意排除的字段**：
- raw 的 `max_base_roll` / `max_base_pitch` 在 `BatchedTrajectoryConfig` 中不存在，暂不对齐（batched 侧无对应限制逻辑）
- batched 的 `max_roughness`（默认 `0.5`）在 raw 中不存在。注意 `batched_compute_footholds` 内部螺旋搜索当前硬编码 `max_roughness=1.0`，未读 config 字段。L1 不依赖此参数，但如后续接 cfg 需同步更新 golden dict

#### 前置条件：terrain 桥接

`fixtures/terrain_adapter.py` 从同一 heightmap 数据构造 raw `GlobalElevationTerrain` 和 batched `PlannerTerrain`，先验证两边 `height_at` 一致（网格内点 atol=1e-6，边界点 atol=1e-4，见 §7.3）。Terrain 桥接一致性本身作为 L1 的前置断言，失败则跳过依赖 terrain 的交叉验证。

#### 逐模块交叉验证

| 测试类 | 对比什么 | 精度 |
|--------|---------|------|
| `TestGaitCrossValidation` | contact_seq, touchdown_times, stance_time | contact_seq: exact match after `.to(float32)` (batched returns float32 flags); touchdown_times/stance_time: atol=1e-12 |
| `TestFootholdCrossValidation` | touchdown XYZ | atol=1e-8 |
| `TestSwingCrossValidation` | foot_targets `(T, 4, 3)` | atol=1e-8 |
| `TestBaseSolverCrossValidation` | root_pos_w, root_quat_w | atol=1e-8 |
| `TestIKCrossValidation` | joint_angles, body_pos_w | atol=1e-8 |
| `TestTrajectoryEndToEnd` | 所有输出字段 | atol=1e-8, rtol=1e-6 |

端到端覆盖场景：

- flat terrain + trot gait + forward (vx=0.3)
- flat terrain + trot gait + lateral (vy=0.2)
- flat terrain + trot gait + turn (yaw_rate=0.5)
- stairs terrain + trot gait + forward (vx=0.5)
- standstill (cmd=0)

### 5.3 L2：向量化回归

**Golden reference 策略**：向量化前，先用当前串行实现对固定输入生成参考输出并序列化为 `.pt` 文件（存储在 `Go2Pvcnn/tests/fixtures/golden/`）。向量化后的实现必须 allclose 匹配。如果串行代码路径被删除，golden `.pt` 文件作为唯一 reference。

在 L1 基础上，额外覆盖 raw 不方便测试的 edge case：

- N>1 的 broadcast 行为
- batch 内混合 standstill 和 motion 的 env
- 全 stance / 全 swing / 单帧 swing 的 contact_seq
- Hermite 插值连续性（swing 起止点 z 无跳变）

### 5.4 L3：性能基准

新建 `Go2Pvcnn/tests/benchmarks/bench_planner_scaling.py`（复用 `Go2Pvcnn/scripts/bench_batched_planner.py` 的 `_SyntheticEnv`/`_BenchRow` 框架），扫描 N=[1, 64, 256, 1024, 2048]，输出 per-stage JSONL + 性能门限断言。`bench_batched_planner.py` 保持不变作为独立脚本入口。

**必须包含 §2.3 定义的两种 workload**：
- **burst replan**：全量 N 同时 replan（现有 bench 逻辑，`replan_interval=1`）
- **稳态 replan**：N_total=2048 但每步仅 ~5-10% env 触发 replan（通过设置 `replan_interval > 1` + 交错 `episode_length_buf` 模拟）

**CUDA 依赖**：L3 标记 `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`，无 GPU 时自动跳过。

### 5.5 L4：可视化回放逻辑

不依赖 Isaac Lab 渲染：

- `test_kinematic_playback_state_chain`：轨迹末帧 → 下次规划初始状态
- `test_touchdown_visible_before_playback`：touchdown 标记在回放帧 0 已设置
- `test_teleop_command_integration`：命令更新后下次规划使用新命令

### 5.6 TDD 执行流程

1. 写 L1 交叉验证测试（调 raw + 当前串行 batched，确认基线一致或记录差异）
2. 写 L2 回归测试（串行版 golden reference）
3. 向量化改造代码，L1 + L2 红→绿
4. 跑 L3 bench 确认性能
5. 改造可视化，L4 绿
6. 全量 `pytest Go2Pvcnn/tests/ -x`

## 6. 实现顺序

1. **Phase 0 — 基线建立**：写 L1 交叉验证，确认 raw 与当前串行 batched 的基线差异
2. **Phase 1 — swing 向量化**：最高优先级，消除 91% 瓶颈
3. **Phase 2 — terrain + foothold 向量化**
4. **Phase 3 — base_solver / terrain_estimator 评估**：bench 后决定
5. **Phase 4 — Triton kernel**（如需）：bench N=2048 后对瓶颈热点补 Triton
6. **Phase 5 — 可视化改造**：纯 kinematic 回放
7. **Phase 6 — 全量验证**：L1-L4 全绿 + bench 达标

## 7. 风险与回退

### 7.1 数值精度

- `batched_gait_schedule` 返回 float32 contact flags，raw 返回 float64。L1 对比时统一为 float32 再 exact match。
- terrain `grid_sample` 在 float32 下可能引入 ~1e-7 级偏差。L1 terrain 对比使用 `atol=1e-6`。
- 向量化后的 swing progress 使用 cumsum/scatter 而非逐帧 Python 赋值，浮点累积顺序不同。如果 atol=1e-8 不够，放宽到 1e-6 并记录。

### 7.2 import 路径

L1 测试需要同时 import `raw/kinematic_footsteps/scripts/go2fp/` 和 `Go2Pvcnn/extension/batched_planner/`。`conftest.py` 中通过 `sys.path.insert` 添加两个根路径。需确保 raw 侧的 numpy-based 模块不与 batched 侧的 torch-based 模块命名冲突。

### 7.3 terrain 桥接

raw 使用 `GlobalElevationTerrain`（基于 scipy/numpy 的双线性插值），batched 使用 `PlannerTerrain`（基于 `F.grid_sample`）。即使从同一 heightmap 构造，边界处理和插值精度可能略有差异。terrain_adapter 的首要测试是验证两边 `height_at` 在网格内点精度 <1e-6，边界点精度 <1e-4。

### 7.4 无 CUDA 环境

L1/L2 测试在 CPU 上运行（验证正确性不需要 GPU）。L3 性能测试标记 `skipif` no CUDA。Triton kernel 编译需要 CUDA toolkit，通过 `try: import triton` guard。

### 7.5 Phase 4/5 并行性

可视化改造（Phase 5）与 Triton kernel（Phase 4）无依赖关系，可并行进行。spec 中的顺序是推荐优先级，非硬性依赖。

## 8. 相关代码入口

- `Go2Pvcnn/extension/batched_planner/swing.py` — 核心瓶颈
- `Go2Pvcnn/extension/batched_planner/terrain.py` — `max_height_along_segment`
- `Go2Pvcnn/extension/batched_planner/foothold.py` — 螺旋偏移 + evaluate
- `Go2Pvcnn/extension/batched_planner/base_solver.py` — EMA 高度平滑
- `Go2Pvcnn/extension/batched_planner/terrain_estimator.py` — EMA 地形估计
- `Go2Pvcnn/extension/batched_planner/trajectory.py` — 组装调用链
- `Go2Pvcnn/extension/batched_planner/manager.py` — selective replan 调度（§2）
- `Go2Pvcnn/extension/viz/go2_foostep_planner.py` — 可视化入口
- `raw/kinematic_footsteps/scripts/go2fp/` — 金标准参考
- `notes/human/human-13-batched-planner-swing-stance-ik-complexity.md` — 复杂度分析
