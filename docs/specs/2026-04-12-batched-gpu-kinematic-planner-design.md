# Batched GPU Kinematic Planner Design

将 `raw/kinematic_footsteps` 的 `generate_trajectory` 管线用 PyTorch 做 batched GPU 重写，使其天然支持 `(N_envs, ...)` 并行，消除训练时 CPU 进程池的阻塞瓶颈。

## 背景与问题

当前 `Go2Pvcnn/extension` 通过 `raw_go2fp_bridge.py` + `ProcessPoolExecutor` 在 Isaac Lab 训练中调用 `raw/kinematic_footsteps` 的纯 Python/NumPy 规划器。性能瓶颈：

- **startup 阻塞**：为 4096 个环境串行/并行跑 `generate_trajectory`，训练前等待很久
- **interval replan 阻塞**：每隔 `reference_replan_interval_s` 秒触发全量 replan，训练完全停住等 CPU 规划完成
- **IPC 开销**：spawn 子进程的 pickle/numpy 数据搬运
- **CPU↔GPU 搬运**：规划完成后全量 cache 搬到 GPU

## 设计决策总结

| 决策项 | 选择 |
|--------|------|
| 架构 | PyTorch batched GPU 重写 |
| 算法逻辑 | **完全保留** raw 的每一步算法（螺旋搜索顺序、候选迭代、scoring 公式、EMA 等） |
| 设备 | GPU (CUDA)，与 Isaac Lab 仿真同卡 |
| 候选搜索 | 固定 K 个候选（由 config 决定），N×K batch |
| 地形查询 | 直接用 Isaac Lab ray hits tensor，`F.grid_sample` 双线性插值 |
| 代码组织 | 重写 `Go2Pvcnn/extension/`，保留 rewards/observations/metrics |
| 配置位置 | `go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py` |
| replan 策略 | 固定间隔全量 replan |
| 验证 | 数值对齐测试，agent 在终端执行 |

## 1. 目录结构

### 重写范围

`Go2Pvcnn/extension/` 全部重写。当前的三层（`planner/core` 同构复刻、`planner/runtime` 进程池桥、`planner/adapters` Isaac 适配）移除，替换为 batched PyTorch GPU 管线。

### 保留

- `extension/mdp/rewards_reference.py` — 保留 reward 函数签名和 cache 消费逻辑，适配新 cache 格式
- `extension/mdp/observations.py` — 保留 `downsampled_height_scan`
- `extension/mdp/metrics.py` — 保留指标记录

### 新结构

```
Go2Pvcnn/extension/
├── __init__.py
├── convention.py              # MuJoCo↔Isaac Lab 对齐（quat wxyz↔xyzw、坐标系等）
├── batched_planner/
│   ├── __init__.py
│   ├── config.py              # BatchedTrajectoryConfig（与 raw TrajectoryConfig 参数对齐）
│   ├── types.py               # BatchedTrajectoryResult、BatchedRobotState（全 tensor）
│   ├── terrain.py             # BatchedTerrain：F.grid_sample 封装
│   ├── gait.py                # batched_gait_schedule, batched_next_touchdown_times
│   ├── foothold.py            # batched_spiral_search, batched_evaluate_touchdowns, batched_evaluate_candidates
│   ├── swing.py               # batched_compute_swing_targets
│   ├── base_solver.py         # batched_solve_base_trajectory
│   ├── terrain_estimator.py   # batched_estimate_terrain
│   ├── ik.py                  # batched_inverse_kinematics, batched_forward_kinematics
│   └── trajectory.py          # batched_generate_trajectory：主入口
├── mdp/
│   ├── __init__.py
│   ├── rewards_reference.py   # 保留（适配新 cache）
│   ├── observations.py        # 保留
│   └── metrics.py             # 保留
└── viz/
    └── compare_trajectories.py  # 数值对比工具（--no-gui 模式供 agent 用）

go2_pvcnn/tasks/
├── teacher_elevation_trajectory_env_cfg.py  # 迁移并重写
└── register_envs.py                          # 更新 import 路径
```

### 模块职责

| 模块 | 输入 | 输出 | raw 对应 |
|------|------|------|----------|
| `convention.py` | Isaac Lab tensor | 转换后 tensor | 无（新增） |
| `terrain.py` | `(N,1,H,W)` 高程图 + 元数据 | `height_at_batch`, `roughness_at_batch`, `max_height_along_segment_batch` | `raw/go2fp/terrain.py` |
| `gait.py` | config 参数 | `(N,T,4)` contact_seq | `raw/go2fp/gait.py` |
| `foothold.py` | state + terrain + candidates | `(N*K,4,3)` footholds + scores | `raw/go2fp/foothold.py` |
| `swing.py` | contact_seq + footholds | `(N,T,4,3)` foot targets | `raw/go2fp/swing.py` |
| `terrain_estimator.py` | foot_pos + base_pos | `(N,T)` roll/pitch/height | `raw/go2fp/terrain_estimator.py` |
| `base_solver.py` | planar + terrain estimates | `(N,T,3)` pos + `(N,T,4)` quat | `raw/go2fp/base_solver.py` |
| `ik.py` | root + foot targets | `(N,T,12)` joints | `raw/go2fp/ik.py` |
| `trajectory.py` | terrain + state + command | `BatchedTrajectoryResult` | `raw/go2fp/trajectory.py` |

## 2. 核心算法（忠实移植）

核心原则：**逻辑不动，只加 batch 维**。raw 的每一步——螺旋搜索顺序、候选迭代顺序、scoring 公式、EMA 系数、Raibert 公式、body clearance 采样——全部保持一致。

### 2.1 `convention.py`

raw 使用 `wxyz` 四元数，Isaac Lab 使用 `xyzw`（部分 API 混用）。

```python
def quat_wxyz_to_xyzw(q: Tensor) -> Tensor: ...   # (..., 4)
def quat_xyzw_to_wxyz(q: Tensor) -> Tensor: ...   # (..., 4)
def isaac_state_to_planner_state(...) -> BatchedRobotState: ...
def planner_result_to_reference_cache(...) -> ReferenceTrajectoryCache: ...
```

### 2.2 `terrain.py`

Isaac Lab `RayCaster` 输出 `(N, num_rays, 3)` 的 ray_hits_w。直接在 GPU 上操作：

```python
class BatchedTerrain:
    """(N, 1, H, W) 高程图 + 元数据，所有查询返回 batched tensor。"""

    def __init__(self, heightmaps, origins_xy, resolution, yaw):
        # heightmaps: (N, 1, H, W)
        # origins_xy: (N, 2) — 高程图中心世界坐标
        # yaw: (N,) — 机器人朝向

    @classmethod
    def from_ray_hits(cls, ray_hits_w, root_pos, root_quat, grid_size, resolution):
        """从 Isaac Lab ray caster 输出构建 batched 高程图。
        (N, num_rays, 3) → reshape (N, H, W, 3) → 取 z → (N, 1, H, W)。"""

    def height_at(self, xy: Tensor) -> Tensor:
        """(N, M, 2) 世界坐标 → (N, M) 高度值，F.grid_sample 双线性插值。"""

    def roughness_at(self, xy: Tensor) -> Tensor:
        """(N, M, 2) → (N, M) 粗糙度，Sobel 梯度幅值。"""

    def max_height_along_segment(self, p0, p1, n_samples=8) -> Tensor:
        """(N, 4, 2), (N, 4, 2) → (N, 4) 线段最大高度。
        p0→p1 间均匀采 n_samples 点，取 max(height_at(...))。"""
```

`roughness_at` 用 `F.conv2d` Sobel kernel 在 `(N,1,H,W)` 上算梯度幅值，与 raw 用高程差近似斜率语义一致。

### 2.3 `gait.py`

raw 已在 time 维向量化。加 batch 维：

```python
def batched_gait_schedule(n_frames, dt, step_freq, duty_factor, phase_offsets) -> Tensor:
    """(N,) params → (N, T, 4) contact sequence，1=stance 0=swing。
    phase = (t * step_freq[:, None, None] + offsets) % 1.0
    contact = (phase < duty_factor[:, None, None]).float()"""

def batched_next_touchdown_times(step_freq, phase_offsets) -> Tensor:
    """(N,) → (N, 4) 每条腿距下次触地的时间。"""
```

### 2.4 `foothold.py` — 保留螺旋搜索

```python
SPIRAL_OFFSETS: Tensor  # (S, 2) 预计算常量，与 raw 螺旋顺序完全一致

def _precompute_spiral_offsets(search_radius, search_step) -> Tensor:
    """按 raw spiral_search_safe_foothold 相同的螺旋顺序生成偏移。一次性常量。"""

def batched_spiral_search(nominal_xy, terrain, previous_footholds,
                          search_step, max_roughness, max_step_down) -> Tensor:
    """(N, 4, 2) nominal → (N, 4, 3) 选中落脚点。
    1. 展开: candidates = nominal[:,:,None,:] + SPIRAL_OFFSETS * step → (N, 4, S, 2)
    2. 批量 height_at / roughness_at → (N, 4, S)
    3. 约束 mask: roughness <= max, height >= prev_z - max_step_down
    4. score = d_nominal + 0.5 * d_previous（与 raw 一致）
    5. tie-breaking: score += 1e-10 * spiral_index（保留螺旋顺序偏好）
    6. invalid → inf, argmin → 选中
    7. 全部 invalid → fallback to nominal 或 previous"""

def batched_evaluate_touchdowns(touchdown_pos, liftoff_pos, touchdown_mask,
                                terrain, previous_footholds, max_reach):
    """完全复刻 raw evaluate_touchdown_set 的 feasibility 和 scoring。"""

def generate_replan_candidates(command, cfg) -> Tensor:
    """与 raw iter_replan_commands 生成完全相同顺序的 K 个候选。
    (N, 3) → (N, K, 3)"""

def batched_candidate_total_score(original_cmd, candidate_cmd,
                                  touchdown_scores, candidate_indices) -> Tensor:
    """复刻 raw _candidate_total_score。"""

def batched_evaluate_candidates(terrain, states, commands, cfg):
    """(N,) → best_command (N, 3), best_footholds (N, 4, 3)。
    1. expand (N,) → (N*K,)
    2. batched_compute_footholds → (N*K, 4, 3)
    3. batched_evaluate_touchdowns → scores (N*K,)
    4. reshape (N, K) → per-env argmin"""
```

### 2.5 `swing.py`

```python
def batched_compute_swing_targets(contact_seq, lift_off_pos, touchdown_pos,
                                  step_height, terrain_max_heights, clearance=0.02):
    """(N, T, 4) + footholds → (N, T, 4, 3) foot targets。
    swing phase progress 用 cumsum + mask（替代 raw 的 run-length Python 循环）。
    Hermite z 插值、xy 线性插值逻辑与 raw 一致。"""
```

### 2.6 `terrain_estimator.py`

```python
def batched_estimate_terrain(foot_positions, base_positions, base_yaw, alpha=0.05):
    """(N, T, 4, 3) → roll (N,T), pitch (N,T), height (N,T)。
    raw values 计算全 batch 向量化。
    EMA 滤波：time 维串行循环（T ≤ 50），每步 N 个环境并行。
    数值与 raw 完全一致。"""
```

### 2.7 `base_solver.py`

```python
def batched_solve_base_trajectory(initial_pos, initial_yaw, vx, vy, yaw_rate,
                                  n_frames, dt, terrain, foot_targets, contact_seq,
                                  terrain_roll, terrain_pitch, terrain_height):
    """→ root_pos (N,T,3), root_quat (N,T,4)。
    1. integrate_base_planar: cumsum（全 batch 向量化）
    2. solve_base_height: support_z 加权平均 + EMA（time 串行, env 并行）
    3. euler_to_quat batch
    4. body_clearance: 8 采样点展开为 (N, T, 8, 2) → terrain.height_at → max → z 调整"""
```

### 2.8 `ik.py`

raw IK/FK 已在 time 维向量化。改为 `(N, T, 4, ...)` tensor 操作，用 `HIP_OFFSETS` 常量 tensor 广播替代逐腿循环。

### 2.9 `trajectory.py` — 主入口

```python
def batched_generate_trajectory(terrain, states, commands, n_frames, dt, cfg):
    """完全复刻 raw generate_trajectory 的 10 步流程：
    1. standstill mask: |cmd| < eps → (N,) bool
    2. gait_schedule → (N, T, 4)
    3. hip positions（HIP_OFFSETS 常量 tensor）
    4. 候选搜索:
       a. generate_replan_candidates → (N, K, 3)
       b. expand → (N*K, ...)
       c. batched_compute_footholds → (N*K, 4, 3)
       d. batched_evaluate_touchdowns → (N*K,)
       e. batched_candidate_total_score → (N, K)
       f. per-env argmin → best_command, best_footholds
       g. 全部 infeasible → merge standstill mask
       h. best_cmd is standstill → merge standstill mask
    5. max_height_along_segment → (N, 4)
    6. compute_swing_targets → (N, T, 4, 3)
    7. integrate_base_planar → estimate_terrain
    8. solve_base_trajectory
    9. batch IK/FK
    10. torch.where(standstill_mask, standstill_result, motion_result)"""
```

## 3. Isaac Lab 集成

### 3.1 固定间隔全量 Replan

```python
class BatchedTrajectoryManager:
    """固定间隔全量 replan，间隔期间消费缓存的参考轨迹。"""

    def __init__(self, cfg, device):
        self._cfg = cfg
        self._cache: BatchedTrajectoryResult | None = None
        self._phase_counter: Tensor     # (N,) 当前 phase index
        self._step_counter: int = 0     # 全局步数计数器

    def step(self, terrain, states, commands):
        """每个 env step 调用。到 replan 间隔 → 全量重规划。"""
        if self._step_counter % self._cfg.replan_interval_steps == 0:
            self._cache = batched_generate_trajectory(
                terrain, states, commands,
                self._cfg.horizon, self._cfg.dt, self._cfg,
            )
            self._phase_counter.zero_()
        self._phase_counter += 1
        self._step_counter += 1

    def current_reference(self) -> dict[str, Tensor]:
        """返回当前 phase 对应的参考帧。phase 超出 horizon 时 clamp。"""
        idx = self._phase_counter.clamp(max=self._cache.horizon - 1)
        return self._cache.gather_at_phase(idx)
```

env reset 时不触发额外 replan，只重置 `phase_counter`。下次固定间隔到来时统一 replan。

### 3.2 Env Config

迁移到 `go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`：

```python
@configclass
class TeacherElevationTrajectoryEnvCfg(TeacherElevationEnvCfg):
    scene: TeacherElevationTrajectorySceneCfg = ...
    observations: TeacherElevationTrajectoryObservationsCfg = ...
    rewards: TeacherElevationTrajectoryRewardsCfg = ...

    use_batched_reference_trajectory: bool = True
    reference_trajectory_horizon: int = 50
    reference_replan_interval_steps: int = 250

    # 候选搜索（与 raw TrajectoryConfig 对齐）
    replan_velocity_scales: list[float] = [1.0, 0.8, 0.6]
    replan_yaw_biases: list[float] = [0.0, 0.15, -0.15]
    replan_vy_biases: list[float] = [0.0, 0.05, -0.05]

    # 步态
    gait_name: str = "trot"
    step_freq: float = 2.0
    duty_factor: float = 0.6
    step_height: float = 0.08

    # 落脚点搜索
    foothold_search_radius: float = 0.15
    foothold_search_step: float = 0.03
    max_step_down: float = float("inf")

    def __post_init__(self):
        super().__post_init__()
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # 不再注册 EventTerm — BatchedTrajectoryManager 由 env 管理
```

### 3.3 Reward 消费

`rewards_reference.py` 中的 reward 函数签名不变，仍通过 `env.unwrapped._trajectory_reference_cache` 获取参考数据。`BatchedTrajectoryManager.current_reference()` 写入该位置。

### 3.4 register_envs.py

import 路径从 `extension.tasks.` 改为 `go2_pvcnn.tasks.`，gym.register 不变。

## 4. 测试策略

所有测试在 `Go2Pvcnn/tests/` 下，`unittest` 框架，agent 通过 `conda run -n mujoco_env python -m pytest` 执行并看终端输出判断。

### 4.1 测试层级

| 顺序 | 测试文件 | 验证内容 |
|------|----------|----------|
| 1 | `test_batched_convention.py` | quat 转换 round-trip |
| 2 | `test_batched_terrain.py` | height_at / roughness_at / max_height_along_segment 对比 raw |
| 3 | `test_batched_gait.py` | gait_schedule 对比 raw |
| 4 | `test_batched_foothold.py` | spiral_search + evaluate_touchdowns 对比 raw |
| 5 | `test_batched_swing.py` | swing_targets 对比 raw |
| 6 | `test_batched_terrain_estimator.py` | EMA roll/pitch/height 对比 raw |
| 7 | `test_batched_base_solver.py` | root_pos / root_quat 对比 raw |
| 8 | `test_batched_ik.py` | IK/FK joint angles 对比 raw |
| 9 | `test_batched_trajectory.py` | 端到端 N=1 对比 raw |
| 10 | `test_batched_trajectory_batch.py` | N=32 batch 一致性 |

### 4.2 对齐标准

- 位置 / 角度 / 四元数 / touchdown：`atol=1e-5, rtol=1e-5`
- contact_state：精确相等
- 每个测试固定 seed，可复现

### 4.3 测试结构

```python
class TestBatchedXxx(unittest.TestCase):
    def setUp(self):
        # 固定 seed 构造 raw 和 batched 两套输入

    def test_single_env_matches_raw(self):
        # N=1 batched 输出 vs raw 单环境，逐字段 assert_allclose

    def test_batch_consistency(self):
        # N=8 每个 env 独立输入，结果与逐个 N=1 跑一致
```

### 4.4 端到端数值报告

`extension/viz/compare_trajectories.py --no-gui --seed 42` 输出：

```
=== Trajectory Alignment Report ===
root_pos     max_err: X.Xe-XX  PASS/FAIL (< 1e-5)
root_quat    max_err: X.Xe-XX  PASS/FAIL
joint_angles max_err: X.Xe-XX  PASS/FAIL
foot_pos     max_err: X.Xe-XX  PASS/FAIL
contact      exact_match: True/False  PASS/FAIL
touchdown    max_err: X.Xe-XX  PASS/FAIL
=== ALL FIELDS ALIGNED / MISMATCHES FOUND ===
```

## 5. 与现有文档的关系

本设计完成后需同步更新：

- `notes/human/human-09-extension-planner-mapping.md` — 映射关系变为 raw ↔ batched_planner
- `notes/human/human-10-extension-planner-runtime.md` — runtime 改为 GPU 直联
- `notes/human/human-11-extension-trajectory-reward.md` — 移除 raw 参考重规划与并行章节
- 对应 `notes/ai/` 文档同步更新
