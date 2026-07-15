# Joint MPC RTI GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `Go2Pvcnn/extension/joint_mpc_rti/` 新增独立的纯 GPU batched rolling kinematic MPC，以 50 Hz 从真实状态重规划未来 16 步，只发布第一未来 reference，并在连续 Loss 下实现 command 跟踪、小物体跨越、大障碍绕行和粗糙地形运动学安全。

**Architecture:** 新 backend 不依赖 `batch_mpc_planner` 内部算法，只复用项目级 convention、ReferenceTrajectoryCache 和 trajectory-manager 接口。核心是 fixed-shape multiple-shooting SQP-RTI：解析运动学/FK/field Jacobian、Generalized Gauss-Newton、Primal-Dual iLQR、时间维 associative scan 和并行 line search；terrain/SDF 在 RayCaster 同批 env ids 更新时构建，planner 热路径只做世界坐标查询。

**Tech Stack:** Python 3.10、PyTorch、pytest、IsaacLab、CUDA；测试解释器固定为 `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`。

---

## 全局约束

- 设计来源：`docs/superpowers/specs/2026-07-15-joint-mpc-rti-gpu-design.html`。
- 新 package 固定为 `Go2Pvcnn/extension/joint_mpc_rti/`，backend 名固定为 `joint_mpc_rti`。
- 当前 `mpc` 行为、配置和测试必须保持不变。
- 外部 command 固定为 body frame `[vx_body, vy_body, yaw_rate]`，不得在 viewer 和 planner 各旋转一次。
- v1 固定 trot contact/swing schedule，不优化 phase offset、swing duration 或 gait topology。
- 不增加 `crossable_small`、`if small then cross`、`if large then turn left/right`、固定避障侧、指定跨越腿、snapping、hard projection 或 optimize-then-repair。
- 唯一高程图为 `height_w = ray_hits_w[...,2]`；不增加 ground/surface 双高度场。
- 足、膝、小腿和机身采样点由 root/joint 解析 FK 产生；不增加独立 foot decision state，也不增加 FK-consistency behavior loss。
- planner 热路径使用 `float32`、固定 shape、无 Python per-env 循环、无 Python per-horizon 主循环、无 `.item()` 或逐 stage CUDA synchronize。
- 所有 production code 必须遵循 RED → GREEN → REFACTOR；每个任务先看到针对缺失行为的正确失败。
- CPU/纯 tensor 测试使用：

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest <本任务列出的测试路径> -q
```

- IsaacLab 进程必须串行启动；上一进程退出后才能运行下一条 Isaac 测试。

---

## 文件结构

### 新 package

```text
Go2Pvcnn/extension/joint_mpc_rti/
├── __init__.py
├── config.py
├── types.py
├── planner.py
├── model/
│   ├── __init__.py
│   ├── dynamics.py
│   ├── gait_schedule.py
│   ├── go2_kinematics.py
│   └── rollout.py
├── terrain/
│   ├── __init__.py
│   ├── field_cache.py
│   ├── field_builder.py
│   ├── distance_field.py
│   └── query.py
├── losses/
│   ├── __init__.py
│   ├── objective.py
│   ├── barriers.py
│   ├── command.py
│   ├── posture.py
│   ├── contact.py
│   ├── clearance.py
│   ├── semantic.py
│   └── smoothness.py
├── solver/
│   ├── __init__.py
│   ├── sqp_rti.py
│   ├── linearization.py
│   ├── gauss_newton.py
│   ├── primal_dual_ilqr.py
│   ├── associative_scan.py
│   └── line_search.py
├── runtime/
│   ├── __init__.py
│   ├── manager.py
│   ├── warm_start.py
│   └── reference_buffer.py
├── integration/
│   ├── __init__.py
│   ├── command.py
│   ├── field_sync.py
│   ├── isaaclab_adapter.py
│   ├── reference_adapter.py
│   └── viewer_adapter.py
└── diagnostics/
    ├── __init__.py
    ├── metrics.py
    ├── profiler.py
    └── validation.py
```

### 新测试

```text
Go2Pvcnn/tests/joint_mpc_rti/
├── __init__.py
├── helpers.py
├── test_contracts.py
├── test_command_dynamics.py
├── test_kinematics_gait.py
├── test_terrain_fields.py
├── test_losses.py
├── test_solver.py
├── test_rolling_runtime.py
├── test_backend_wiring.py
├── test_behavior.py
├── test_isaaclab_runtime.py
└── test_performance.py
```

### 共享测试 helper 契约

Task 1 同时创建 `Go2Pvcnn/tests/joint_mpc_rti/helpers.py`。后续测试统一从这里导入，不使用未定义的 `_state()`、`_field()` 或 `_command()`：

```python
from __future__ import annotations

import torch


def make_state(batch: int, *, device: str = "cpu", dtype: torch.dtype = torch.float32):
    from extension.joint_mpc_rti.types import JointMpcRtiState

    root_pos = torch.zeros(batch, 3, device=device, dtype=dtype)
    root_pos[:, 2] = 0.32
    joint = torch.tensor([0.0, 0.8, -1.5] * 4, device=device, dtype=dtype).expand(batch, -1).clone()
    return JointMpcRtiState(
        root_pos_w=root_pos,
        root_rpy_w=torch.zeros(batch, 3, device=device, dtype=dtype),
        joint_pos=joint,
        root_lin_vel_b=torch.zeros(batch, 3, device=device, dtype=dtype),
        root_ang_vel_b=torch.zeros(batch, 3, device=device, dtype=dtype),
        joint_vel=torch.zeros(batch, 12, device=device, dtype=dtype),
    )


def make_command(batch: int, *, vx: float = 0.2, vy: float = 0.0, yaw: float = 0.0, device: str = "cpu"):
    return torch.tensor([vx, vy, yaw], dtype=torch.float32, device=device).expand(batch, -1).clone()


def make_flat_field(batch: int, *, device: str = "cpu"):
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch

    return build_field_batch(
        height_w=torch.zeros(batch, 151, 151, device=device),
        semantic_id=torch.zeros(batch, 151, 151, dtype=torch.long, device=device),
        origin_w=torch.zeros(batch, 3, device=device),
        yaw_w=torch.zeros(batch, device=device),
        timestamp=torch.zeros(batch, device=device),
        version=torch.ones(batch, dtype=torch.long, device=device),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )
```

### 修改的现有边界

- `Go2Pvcnn/extension/trajectory_manager_factory.py`
  - 接受 `joint_mpc_rti` 并创建 rolling manager。
- `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - 增加 `joint_mpc_rti_cfg`，不改变默认 `planner_backend="mpc"`。
- `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - 增加 CLI backend 和 rolling viewer dispatch。
- `Go2Pvcnn/extension/mdp/rewards_reference.py`
  - 兼容 pending-next-step reference 时序，不改变旧 manager phase 语义。
- `Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py`
  - 在同一批 env ids 完成 ray/semantic 写入后通知 joint field cache 更新；只有启用新 backend 时生效。

---

### Task 1: Public Contracts、Config 与 Backend Factory

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/__init__.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/types.py`
- Modify: `Go2Pvcnn/extension/trajectory_manager_factory.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/__init__.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/helpers.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_contracts.py`

**Public interfaces:**

```python
@dataclass
class JointMpcRtiState:
    root_pos_w: Tensor       # [B,3]
    root_rpy_w: Tensor       # [B,3]
    joint_pos: Tensor        # [B,12]
    root_lin_vel_b: Tensor   # [B,3]
    root_ang_vel_b: Tensor   # [B,3]
    joint_vel: Tensor        # [B,12]

@dataclass
class JointMpcRtiTrajectory:
    state: Tensor            # [B,H+1,18]
    control: Tensor          # [B,H,18]
    foot_pos_w: Tensor       # [B,H+1,4,3]
    contact_state: Tensor    # [B,H+1,4]
    valid: Tensor            # [B]
    fallback: Tensor         # [B]
    status: Tensor           # [B]
    loss_breakdown: dict[str, Tensor]
```

- [ ] **Step 1: 写 backend/config/type RED 测试**

在 `test_contracts.py` 中写：

```python
from types import SimpleNamespace

import pytest
import torch


def test_joint_mpc_rti_config_defaults_match_fixed_shape_contract() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    assert cfg.runtime.horizon_steps == 16
    assert cfg.runtime.dt == pytest.approx(0.02)
    assert cfg.runtime.sqp_iterations_per_step == 1
    assert tuple(cfg.solver.line_search_alphas) == (1.0, 0.5, 0.25)


def test_factory_accepts_joint_mpc_rti_without_changing_mpc_default() -> None:
    from extension.trajectory_manager_factory import planner_backend_from_cfg

    assert planner_backend_from_cfg(SimpleNamespace()) == "mpc"
    assert planner_backend_from_cfg(SimpleNamespace(planner_backend="joint_mpc_rti")) == "joint_mpc_rti"


def test_state_contract_rejects_wrong_joint_shape() -> None:
    from extension.joint_mpc_rti.types import JointMpcRtiState

    with pytest.raises(ValueError, match="joint_pos"):
        JointMpcRtiState(
            root_pos_w=torch.zeros(2, 3),
            root_rpy_w=torch.zeros(2, 3),
            joint_pos=torch.zeros(2, 11),
            root_lin_vel_b=torch.zeros(2, 3),
            root_ang_vel_b=torch.zeros(2, 3),
            joint_vel=torch.zeros(2, 12),
        )
```

- [ ] **Step 2: 运行 RED**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_contracts.py -q
```

Expected: FAIL，原因是 `extension.joint_mpc_rti` 尚不存在，factory 不接受新 backend。

- [ ] **Step 3: 实现最小 config/types/public exports**

`config.py` 使用嵌套 dataclass，至少定义：

```python
@dataclass
class JointMpcRtiRuntimeCfg:
    horizon_steps: int = 16
    dt: float = 0.02
    sqp_iterations_per_step: int = 1
    max_field_age_steps: int = 2
    dtype: torch.dtype = torch.float32


@dataclass
class JointMpcRtiSolverCfg:
    regularization: float = 1.0e-4
    barrier_relaxation: float = 1.0e-3
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25)


@dataclass
class JointMpcRtiCfg:
    runtime: JointMpcRtiRuntimeCfg = field(default_factory=JointMpcRtiRuntimeCfg)
    solver: JointMpcRtiSolverCfg = field(default_factory=JointMpcRtiSolverCfg)
```

`types.py` 的每个 dataclass 在 `__post_init__` 中检查 batch、末维和 dtype/device 一致性。

- [ ] **Step 4: 扩展 factory**

```python
VALID_PLANNER_BACKENDS = ("mpc", "joint_mpc_rti")

def create_trajectory_manager(cfg, *, device):
    backend = planner_backend_from_cfg(cfg)
    if backend == "joint_mpc_rti":
        from extension.joint_mpc_rti.runtime.manager import JointMpcRtiManager
        return JointMpcRtiManager(cfg, device=device)
    from extension.batch_mpc_planner.manager import MpcTrajectoryManager
    return MpcTrajectoryManager(cfg, device=device)
```

为避免 manager 尚未创建导致 Task 1 import 失败，先在 `runtime/manager.py` 建立最小类，只实现 constructor、`planner_backend` 和 `horizon_steps()`；完整行为在 Task 8 TDD 实现。

- [ ] **Step 5: 运行 GREEN**

运行 Step 2 命令，Expected: `3 passed`。

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/extension/trajectory_manager_factory.py Go2Pvcnn/tests/joint_mpc_rti
git commit -m "feat: add joint MPC RTI public contracts"
```

---

### Task 2: Body-Frame Command、Kinematic Dynamics 与 Fixed Trot

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/integration/command.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/model/dynamics.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_command_dynamics.py`

- [ ] **Step 1: 写 command rotation RED 测试**

```python
@pytest.mark.parametrize(
    ("yaw", "expected"),
    [(0.0, (1.0, 0.0)), (torch.pi / 2, (0.0, 1.0)), (torch.pi, (-1.0, 0.0))],
)
def test_body_forward_command_rotates_once_into_world(yaw, expected) -> None:
    from extension.joint_mpc_rti.integration.command import body_linear_velocity_to_world

    command = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    actual = body_linear_velocity_to_world(command[:, :2], torch.tensor([yaw]))
    torch.testing.assert_close(actual, torch.tensor([expected]), atol=1.0e-5, rtol=0.0)
```

- [ ] **Step 2: 写 dynamics/gait RED 测试**

```python
def test_kinematic_step_integrates_body_velocity_and_joint_velocity() -> None:
    from extension.joint_mpc_rti.model.dynamics import kinematic_step

    x = torch.zeros(1, 18)
    x[:, 5] = torch.pi / 2
    u = torch.zeros(1, 18)
    u[:, 0] = 1.0
    u[:, 6:] = 0.5
    out = kinematic_step(x, u, dt=0.02)
    torch.testing.assert_close(out[:, :2], torch.tensor([[0.0, 0.02]]), atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(out[:, 6:], torch.full((1, 12), 0.01))


def test_fixed_trot_uses_diagonal_pairs_and_has_no_optimized_phase() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule

    contact = fixed_trot_schedule(batch=2, horizon_steps=16, device="cpu")
    assert contact.shape == (2, 17, 4)
    assert torch.equal(contact[:, :, 0], contact[:, :, 3])
    assert torch.equal(contact[:, :, 1], contact[:, :, 2])
    assert torch.equal(contact[:, :, 0], torch.logical_not(contact[:, :, 1]))
```

- [ ] **Step 3: 运行 RED**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_command_dynamics.py -q
```

Expected: FAIL，缺少 command/dynamics/gait 模块。

- [ ] **Step 4: 实现最小解析函数**

`body_linear_velocity_to_world(v_xy_b, yaw)` 使用标准二维旋转；`kinematic_step()` 按设计中的 root position、RPY rate map 和 joint Euler step；`fixed_trot_schedule()` 使用固定 tensor pattern，不读取 terrain 或 semantic。

- [ ] **Step 5: 运行 GREEN 和现有 command-frame 回归**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_command_dynamics.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -k 'command_frame or planner_backend' -q
```

Expected: 新测试通过，旧 `mpc` command/backend 测试不退化。

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti
git commit -m "feat: add joint MPC kinematics and trot schedule"
```

---

### Task 3: Go2 Analytic FK、Link Samples 与 Jacobians

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py`
- Read-only parity source: `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py`

- [ ] **Step 1: 写 FK shape、leg order 和 Jacobian RED 测试**

```python
def test_go2_fk_returns_planner_leg_order_and_link_samples() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    root_pos = torch.zeros(2, 3)
    root_rpy = torch.zeros(2, 3)
    joint = torch.tensor([[0.0, 0.8, -1.5] * 4] * 2)
    geometry = go2_fk(root_pos, root_rpy, joint)
    assert geometry.foot_pos_w.shape == (2, 4, 3)
    assert geometry.knee_pos_w.shape == (2, 4, 3)
    assert geometry.shank_samples_w.shape[:3] == (2, 4, 3)
    assert geometry.body_samples_w.shape[0] == 2


def test_go2_analytic_foot_jacobian_matches_central_difference() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import foot_jacobian_joint, go2_fk

    q = torch.tensor([[0.05, 0.7, -1.4] * 4], dtype=torch.float64)
    root_pos = torch.zeros(1, 3, dtype=torch.float64)
    root_rpy = torch.zeros(1, 3, dtype=torch.float64)
    jac = foot_jacobian_joint(root_pos, root_rpy, q)
    eps = 1.0e-6
    q_plus = q.clone(); q_plus[0, 0] += eps
    q_minus = q.clone(); q_minus[0, 0] -= eps
    fd = (go2_fk(root_pos, root_rpy, q_plus).foot_pos_w - go2_fk(root_pos, root_rpy, q_minus).foot_pos_w) / (2 * eps)
    torch.testing.assert_close(jac[0, :, :, 0], fd[0], atol=2.0e-5, rtol=2.0e-4)
```

- [ ] **Step 2: 运行 RED**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py -q
```

Expected: FAIL，缺少新 FK API。

- [ ] **Step 3: 实现解析 FK/Jacobian**

实现 Go2 四腿 hip offsets、abad/hip/knee 三关节链、`LEG_ORDER=(FL,FR,RL,RR)`；同时返回 knee、三点 shank samples、body bottom/edge/corner samples。禁止 autograd Jacobian 出现在 planner 热路径。

- [ ] **Step 4: 运行 GREEN 与现有 FK parity 子集**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py -k 'fk or kinematic' -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model Go2Pvcnn/tests/joint_mpc_rti
git commit -m "feat: add analytic Go2 geometry for joint MPC"
```

---

### Task 4: 世界坐标高程图、Small/Large SDF 与 Field Version

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/terrain/distance_field.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/terrain/query.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/terrain/field_cache.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/terrain/field_builder.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py`

- [ ] **Step 1: 写 world-query/SDF RED 测试**

```python
def test_world_query_uses_bound_field_pose_and_returns_invalid_outside() -> None:
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    height = torch.zeros(1, 151, 151)
    semantic = torch.zeros(1, 151, 151, dtype=torch.long)
    semantic[:, 75, 85] = 1
    field = build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=torch.tensor([[2.0, 3.0, 0.0]]),
        yaw_w=torch.tensor([torch.pi / 2]),
        timestamp=torch.tensor([5.0]),
        version=torch.tensor([7]),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )
    inside = query_world(field, torch.tensor([[[1.90, 3.00, 0.0]]]))
    outside = query_world(field, torch.tensor([[[5.0, 5.0, 0.0]]]))
    assert inside.valid.item()
    assert inside.small_distance_m.item() <= 0.02
    assert not outside.valid.item()


def test_field_cache_updates_only_selected_env_rows_atomically() -> None:
    from extension.joint_mpc_rti.terrain.field_cache import JointMpcTerrainFieldCache

    cache = JointMpcTerrainFieldCache(num_envs=4, grid_size=151, device="cpu")
    before = cache.version.clone()
    update = dict(
        height_w=torch.zeros(2, 151, 151),
        semantic_id=torch.zeros(2, 151, 151, dtype=torch.long),
        origin_w=torch.zeros(2, 3),
        yaw_w=torch.zeros(2),
        timestamp=torch.ones(2),
    )
    cache.update_rows(env_ids=torch.tensor([1, 3]), **update)
    assert torch.equal(cache.version[[0, 2]], before[[0, 2]])
    assert torch.all(cache.version[[1, 3]] > before[[1, 3]])
```

- [ ] **Step 2: 运行 RED**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py -q
```

- [ ] **Step 3: 实现固定 151×151 单高度场**

`build_field_batch()` 输入 `height_w`，不创建第二高度场。small/large mask 由 semantic ids 构造；SDF 使用全 tensor 固定迭代的 jump-flood 或等价 GPU distance transform，输出以米为单位的 signed/unsigned distance 和 XY gradient。

- [ ] **Step 4: 实现 field cache 原子行更新**

先在临时 row tensors 中完成 height/SDF/gradient/pose/version，再用 `index_copy_` 一次写入同一批 env ids，最后更新 ready/version。旧 row 的 origin/yaw/version 必须与旧 field 保持绑定。

- [ ] **Step 5: 运行 GREEN**

运行 Step 2，Expected: 全部通过；额外验证旋转 scanner 后世界查询不变、越界 invalid、1024 行无串扰。

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/terrain Go2Pvcnn/tests/joint_mpc_rti
git commit -m "feat: add world terrain and semantic distance fields"
```

---

### Task 5: Rollout、Warm Start 与 Reference-Frame Geometry

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/model/rollout.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/runtime/warm_start.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_solver.py`

- [ ] **Step 1: 写 rollout/warm-start RED 测试**

```python
def test_shift_warm_start_injects_measured_x0_and_shifts_controls() -> None:
    from extension.joint_mpc_rti.runtime.warm_start import shift_warm_start

    x = torch.arange(1 * 17 * 18, dtype=torch.float32).reshape(1, 17, 18)
    u = torch.arange(1 * 16 * 18, dtype=torch.float32).reshape(1, 16, 18)
    measured = torch.full((1, 18), -2.0)
    shifted = shift_warm_start(x, u, measured)
    torch.testing.assert_close(shifted.state[:, 0], measured)
    torch.testing.assert_close(shifted.state[:, 1:-1], x[:, 2:])
    torch.testing.assert_close(shifted.control[:, :-1], u[:, 1:])


def test_rollout_geometry_is_derived_from_root_and_joint_only() -> None:
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from .helpers import make_state

    result = rollout_controls(make_state(batch=2), torch.zeros(2, 16, 18), dt=0.02)
    assert result.state.shape == (2, 17, 18)
    assert result.foot_pos_w.shape == (2, 17, 4, 3)
    assert not hasattr(result, "independent_foot_state")
```

- [ ] **Step 2: 运行 RED**

- [ ] **Step 3: 实现固定 shape rollout 和 shift**

rollout 使用编译友好的张量 scan/固定展开；warm start 的 terminal state 用最后控制外推一步，dual/slack 同步左移并为尾部复制稳定值。

- [ ] **Step 4: 运行 GREEN 并检查无 Python 时间循环源码契约**

测试通过后，加入 AST/source 测试确保 production rollout 没有 `for env`；时间维最终在 Task 7 替换为 associative scan，Task 5 可使用内部固定 tensor recurrence helper，但不得暴露可变 horizon。

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model Go2Pvcnn/extension/joint_mpc_rti/runtime Go2Pvcnn/tests/joint_mpc_rti
git commit -m "feat: add rolling rollout and warm start"
```

---

### Task 6: Relaxed Barriers 与细分 Loss

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/barriers.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/command.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/posture.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/contact.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/clearance.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/semantic.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/smoothness.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/objective.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_losses.py`

- [ ] **Step 1: 写 barrier 连续性和 Loss 行为 RED 测试**

```python
def test_relaxed_barrier_is_finite_and_increases_toward_violation() -> None:
    from extension.joint_mpc_rti.losses.barriers import relaxed_barrier

    margin = torch.tensor([0.2, 0.02, 0.0, -0.02], requires_grad=True)
    value = relaxed_barrier(margin, relaxation=0.01)
    assert torch.isfinite(value).all()
    assert torch.all(value[1:] > value[:-1])
    value.sum().backward()
    assert torch.isfinite(margin.grad).all()


def test_small_object_loss_prefers_foot_over_or_bypass_without_root_gate() -> None:
    from extension.joint_mpc_rti.losses.semantic import small_object_losses

    common = dict(
        small_top_height=torch.tensor([[0.08]]),
        small_distance_touchdown=torch.tensor([[0.20]]),
        link_pos_w=torch.tensor([[[[0.0, 0.0, 0.20]]]]),
        link_small_distance=torch.tensor([[[0.0]]]),
        swing_mask=torch.tensor([[[True]]]),
        stance_mask=torch.tensor([[[False]]]),
        extra_margin=0.03,
    )
    over = dict(common, foot_pos_w=torch.tensor([[[[0.0, 0.0, 0.20]]]]), foot_small_distance=torch.tensor([[[0.0]]]))
    low = dict(common, foot_pos_w=torch.tensor([[[[0.0, 0.0, 0.03]]]]), foot_small_distance=torch.tensor([[[0.0]]]))
    bypass = dict(common, foot_pos_w=torch.tensor([[[[0.2, 0.0, 0.03]]]]), foot_small_distance=torch.tensor([[[0.20]]]))
    over_loss = small_object_losses(**over)
    low_loss = small_object_losses(**low)
    bypass_loss = small_object_losses(**bypass)
    assert over_loss["small_object_foot_over"] < low_loss["small_object_foot_over"]
    assert bypass_loss["small_object_foot_over"] < low_loss["small_object_foot_over"]
    assert "small_object_root_avoidance" not in over_loss


def test_touchdown_on_small_is_penalized_even_when_height_matches_surface() -> None:
    from extension.joint_mpc_rti.losses.contact import touchdown_losses
    from extension.joint_mpc_rti.losses.semantic import small_object_losses

    foot = torch.tensor([[[[0.0, 0.0, 0.08]]]])
    contact = touchdown_losses(
        touchdown_pos_w=foot[:, 0],
        queried_height_w=torch.tensor([[[0.08]]]),
        queried_valid=torch.tensor([[[True]]]),
    )
    semantic = small_object_losses(
        foot_pos_w=foot,
        foot_small_distance=torch.tensor([[[-0.01]]]),
        small_top_height=torch.tensor([[[0.08]]]),
        small_distance_touchdown=torch.tensor([[[-0.01]]]),
        link_pos_w=foot,
        link_small_distance=torch.tensor([[[-0.01]]]),
        swing_mask=torch.tensor([[[False]]]),
        stance_mask=torch.tensor([[[True]]]),
        extra_margin=0.03,
    )
    losses = {**contact, **semantic}
    assert losses["touchdown_ground_height"] < 1.0e-6
    assert losses["small_object_touchdown_avoidance"] > 0.0
```

- [ ] **Step 2: 运行 RED**

- [ ] **Step 3: 实现完整 Loss breakdown**

必须实现设计中的精确 key：

```text
command_linear_velocity, command_yaw_rate, command_progress, command_direction
root_support_height, root_roll_pitch, root_vertical_velocity, root_roll_pitch_rate
joint_nominal_posture, joint_position_limit_barrier, joint_velocity_limit_barrier
stance_xy_lock, stance_ground_contact, stance_slip_velocity
swing_nominal_shape, terrain_swing_clearance, swing_velocity_smoothness, touchdown_velocity
touchdown_ground_height, touchdown_valid_map, touchdown_reach_margin, touchdown_foot_separation
foot_ground_penetration, knee_ground_clearance, shank_ground_clearance, body_ground_clearance
small_object_foot_over, small_object_touchdown_avoidance, small_object_link_clearance
large_root_footprint_barrier, large_body_collision, large_foot_collision
large_knee_shank_collision, large_terminal_risk
control_rate, first_control_continuity, joint_acceleration, root_acceleration
terminal_command_velocity, terminal_obstacle_safety, terminal_posture, terminal_contact_viability
```

所有项返回 `[B]` normalized loss 和 residual/Jacobian 所需局部张量；禁止 `.item()`、环境分支和布尔模式选择。

- [ ] **Step 4: 添加源码反 gate 测试**

读取 `losses/` 和 `planner.py`，断言不存在 `crossable_small`、`small_mode`、`avoid_side`、`terrain_row`、`terrain_col`、`snapping`、`hard_projection`。

- [ ] **Step 5: 运行 GREEN**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_losses.py -q
```

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/losses Go2Pvcnn/tests/joint_mpc_rti
git commit -m "feat: add continuous joint MPC loss model"
```

---

### Task 7: GGN、Primal-Dual iLQR、Associative Scan 与 Line Search

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/linearization.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/gauss_newton.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/associative_scan.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/primal_dual_ilqr.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/sqp_rti.py`
- Extend: `Go2Pvcnn/tests/joint_mpc_rti/test_solver.py`

- [ ] **Step 1: 写线性二次已知解 RED 测试**

```python
def test_primal_dual_ilqr_matches_small_dense_lqr_solution() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import solve_lq_subproblem

    problem = make_scalar_integrator_lqr(batch=3, horizon=16, q=1.0, r=0.1, q_terminal=4.0)
    result = solve_lq_subproblem(problem)
    expected = dense_kkt_reference(problem)
    torch.testing.assert_close(result.delta_u, expected.delta_u, atol=2.0e-4, rtol=2.0e-4)


def test_associative_scan_matches_sequential_affine_composition() -> None:
    from extension.joint_mpc_rti.solver.associative_scan import affine_scan

    generator = torch.Generator().manual_seed(7)
    A = 0.1 * torch.randn(4, 16, 2, 2, generator=generator)
    A = A + torch.eye(2).reshape(1, 1, 2, 2)
    b = torch.randn(4, 16, 2, generator=generator)
    parallel = affine_scan(A, b)
    state = torch.zeros(4, 2)
    sequence = []
    for index in range(16):
        state = torch.einsum("bij,bj->bi", A[:, index], state) + b[:, index]
        sequence.append(state)
    sequential = torch.stack(sequence, dim=1)
    torch.testing.assert_close(parallel, sequential, atol=1.0e-5, rtol=1.0e-5)


def test_one_rti_iteration_reduces_merit_from_shifted_warm_start() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step
    from .helpers import make_command, make_flat_field, make_state

    result = step(make_state(8), make_command(8), make_flat_field(8), None, JointMpcRtiCfg())
    assert torch.all(result.merit_after <= result.merit_before + 1.0e-6)
    assert torch.isfinite(result.delta_control).all()
```

- [ ] **Step 2: 运行 RED**

- [ ] **Step 3: 实现解析线性化与 GGN blocks**

`linearization.py` 组合 dynamics/FK/field-query 解析 Jacobian；`gauss_newton.py` 构造每节点 `Qxx/Qxu/Quu/qx/qu` 和 terminal blocks。barrier curvature 只加入凸的对角/低秩正半定部分，`Quu` 加可调 regularization。

测试文件同时实现 `make_scalar_integrator_lqr(...)` 和 `dense_kkt_reference(...)`：前者显式构造标量积分器的固定 shape LQ blocks，后者用 `torch.linalg.solve` 解测试规模的 dense KKT；它们只存在于测试中，不进入 production hot path。

- [ ] **Step 4: 实现 doubling associative scan**

对 `H=16` 使用固定 4 层 doubling：offset `1,2,4,8`。每层用 batched matrix composition 更新消息，不能写 Python horizon loop；固定 4 层可显式调用同一 helper。

- [ ] **Step 5: 实现 primal-dual LQ solve**

等式 dynamics defect 通过 primal/dual 消元进入 LQ 消息；输出 feedforward `k`、feedback `K`、dual update 和预测下降量。先让已知 LQR 与 dense KKT reference 对齐，再接实际 18/18 blocks。

- [ ] **Step 6: 实现并行 line search**

把 `(1.0,0.5,0.25)` 扩展为额外 alpha 维，单次 batched rollout 计算三个 candidate merit；每环境选择有限且改善的最大 alpha，无改善则 alpha=0。

- [ ] **Step 7: 运行 GREEN**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_solver.py -q
```

- [ ] **Step 8: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver Go2Pvcnn/tests/joint_mpc_rti/test_solver.py
git commit -m "feat: add batched SQP RTI solver"
```

---

### Task 8: Planner API、Rolling Manager 与 Pending Reference

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
- Complete: `Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/runtime/reference_buffer.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/integration/reference_adapter.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/integration/isaaclab_adapter.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py`

- [ ] **Step 1: 写 first-future-reference 时序 RED 测试**

```python
def test_manager_publishes_x1_and_rewards_next_real_state_against_pending_reference() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.runtime.manager import JointMpcRtiManager
    from .helpers import make_command, make_flat_field, make_state

    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=2, device="cpu")
    measured_t = make_state(batch=2)
    step = manager.plan_from_tensors(measured_t, make_command(batch=2), make_flat_field(batch=2))
    torch.testing.assert_close(step.full_trajectory.state[:, 0], measured_t.as_vector())
    torch.testing.assert_close(step.pending_reference.joint_angles, step.full_trajectory.state[:, 1, 6:])
    assert step.pending_reference.target_step == 1


def test_reset_clears_only_selected_pending_rows() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.runtime.manager import JointMpcRtiManager
    from .helpers import make_command, make_flat_field, make_state

    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=4, device="cpu")
    manager.plan_from_tensors(make_state(4), make_command(4), make_flat_field(4))
    manager.reset_envs(torch.tensor([False, True, False, True]))
    assert torch.equal(manager.pending_valid, torch.tensor([True, False, True, False]))
```

- [ ] **Step 2: 运行 RED**

- [ ] **Step 3: 实现 planner.step()**

```python
def step(
    measured_state: JointMpcRtiState,
    command_body: Tensor,
    terrain_field: JointMpcTerrainField,
    solver_state: JointMpcRtiSolverState | None,
    cfg: JointMpcRtiCfg,
) -> JointMpcRtiStepResult:
    validate_inputs(measured_state, command_body, terrain_field, cfg)
    contact = fixed_trot_schedule(measured_state.batch_size, cfg.runtime.horizon_steps, measured_state.device)
    warm_start = initialize_or_shift_solver_state(measured_state, solver_state, contact, cfg)
    return sqp_rti_step(measured_state, command_body, terrain_field, contact, warm_start, cfg)
```

顺序固定为 validate → gait → shift/init → one RTI → safety/finite status → full trajectory + x1。

- [ ] **Step 4: 实现 rolling manager**

manager 每次 `refresh_from_env()` 都规划一次，不使用旧 manager 的 horizon phase counter；reference cache 保留 `[B,H+1,...]` 完整 horizon 兼容 ABI，`current_reference()` 对新 backend 直接返回索引 1 的 pending frame，`current_frame_ids()` 固定返回全 1 仅用于兼容消费者。奖励时序测试必须证明从未读取索引 0 的当前状态。

- [ ] **Step 5: 运行 GREEN**

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py
git commit -m "feat: add rolling joint MPC runtime"
```

---

### Task 9: Task Config、Reward 与 Factory Wiring

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Modify: `Go2Pvcnn/extension/mdp/rewards_reference.py`
- Modify: `Go2Pvcnn/extension/trajectory_manager_factory.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py`

- [ ] **Step 1: 写配置和 reward 时序 RED 测试**

```python
def test_task_can_select_joint_mpc_rti_without_changing_default_mpc() -> None:
    source = Path("Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py").read_text()
    assert 'planner_backend: str = "mpc"' in source
    assert "joint_mpc_rti_cfg" in source


def test_reference_reward_uses_pending_next_step_for_rolling_manager() -> None:
    manager = SimpleNamespace(
        planner_backend="joint_mpc_rti",
        pending_reference=lambda: {"joint_angles": torch.ones(2, 12), "valid": torch.ones(2, dtype=torch.bool)},
    )
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.ones(2, 12)))
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(_trajectory_manager=manager, scene={"robot": robot}),
        scene={"robot": robot},
    )
    reward = reference_joint_pos_reward(env, std=0.2)
    torch.testing.assert_close(reward, torch.ones(2))
```

- [ ] **Step 2: 运行 RED**

- [ ] **Step 3: 增加 task cfg 字段和 backend-specific cfg 选择**

默认仍是 `mpc`；只有显式设置 `planner_backend="joint_mpc_rti"` 才实例化新 manager。训练、PLAY、VIEWER 子类可分别 override，但不得隐式改变现有注册任务。

- [ ] **Step 4: reward 兼容 rolling reference**

`ensure_reference_cache()` 根据 manager capability 获取 pending reference；旧 manager 保持当前 phase gather。reset/invalid reference 返回 neutral/disabled 结果，不读取旧 episode pending frame。

- [ ] **Step 5: 运行 GREEN 和旧 factory/reward 回归**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -k 'factory or reference or reward' -q
```

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/extension/mdp/rewards_reference.py Go2Pvcnn/extension/trajectory_manager_factory.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py
git commit -m "feat: wire joint MPC RTI into reference rewards"
```

---

### Task 10: SemanticRayCaster 同步 Field 更新

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/integration/field_sync.py`
- Extend: `Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py`

- [ ] **Step 1: 写相同 env ids/version RED 测试**

测试模拟 RayCaster 分块更新 `[0,2]`，断言 field updater 收到同一 ids、同一 pose/timestamp，且 `[1,3]` version 不变。

- [ ] **Step 2: 运行 RED**

- [ ] **Step 3: 增加可选 observer/hook**

RayCaster 完成 `ray_hits_w` 和 `semantic_map` row 写入后调用已注册的 field-sync observer。未启用 `joint_mpc_rti` 时 observer 为 `None`，旧 scanner 路径没有额外 SDF 开销。

- [ ] **Step 4: 实现独立 CUDA stream/ready event**

field builder 在独立 stream 构建同一 ids；完成后记录 ready event 和 version。planner 读取最近 ready version，不能同步读取半完成数据。

- [ ] **Step 5: 运行 GREEN**

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster \
  Go2Pvcnn/extension/joint_mpc_rti/integration Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py
git commit -m "feat: synchronize joint MPC fields with raycaster"
```

---

### Task 11: Viewer Rolling Backend

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Complete: `Go2Pvcnn/extension/joint_mpc_rti/integration/viewer_adapter.py`
- Extend: `Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py`

- [ ] **Step 1: 写 CLI 和 rolling playback RED 测试**

```python
def test_viewer_cli_accepts_joint_mpc_rti() -> None:
    import importlib.util

    path = Path("Go2Pvcnn/extension/viz/go2_foostep_planner.py")
    spec = importlib.util.spec_from_file_location("go2_footstep_viewer_for_joint_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._parse_args(["--planner-backend", "joint_mpc_rti"])
    assert args.planner_backend == "joint_mpc_rti"


def test_joint_viewer_applies_only_first_future_frame() -> None:
    from extension.joint_mpc_rti.integration.viewer_adapter import JointMpcRtiViewerAdapter

    trajectory = SimpleNamespace(state=torch.zeros(1, 17, 18))
    adapter = JointMpcRtiViewerAdapter.for_test(trajectory)
    frame = adapter.next_playback_frame()
    assert frame.frame_index == 1
    assert adapter.next_playback_frame().frame_index == 1
```

- [ ] **Step 2: 运行 RED**

- [ ] **Step 3: 增加 backend dispatch**

`mpc` 保持 segment playback；`joint_mpc_rti` 每 viewer step 读取真实 state/current field、调用一次 manager、显示完整 horizon、只写入 `x1`。teleop/scripted command 直接以 body frame 传入，不调用 `_viewer_mpc_world_command_from_root_frame()`。

- [ ] **Step 4: 运行 GREEN 和 viewer 静态回归**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/extension/joint_mpc_rti/integration/viewer_adapter.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py
git commit -m "feat: add rolling joint MPC viewer backend"
```

---

### Task 12: 纯 Tensor 行为验收

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py`
- Modify only as failures prove necessary: `Go2Pvcnn/extension/joint_mpc_rti/config.py`, `losses/*`, `solver/*`

- [ ] **Step 1: 写平地 command 测试**

覆盖 forward/backward/lateral/yaw/diagonal，断言方向符号、body-frame 对齐、有限值、stance clearance ≤ `0.005m`、penetration count `0`。

- [ ] **Step 2: 写 small true-overpass 测试**

构造平地小矩形/圆形高程与 small semantic，多个距离/横向偏移。成功必须同时断言同一腿 lift → footprint 上方 → land、touchdown 不在 small、foot/knee/shank/body collision `0`、root 沿 command 通过且没有整体 Z 抬升伪造 clearance。

- [ ] **Step 3: 写 large avoidance 测试**

构造左/右/正前 large 障碍，断言 body/leg SDF margin、无穿越、绕行方向随风险场变化、障碍后恢复 command 方向。

- [ ] **Step 4: 写粗糙地形/台阶测试**

覆盖上/下台阶、局部高差、不同 offset，断言 touchdown surface error ≤ `0.005m`、全身 penetration `0`、joint limits、root-Z continuity 和多 gait 周期滚动稳定。

- [ ] **Step 5: 逐个运行 RED → 调整连续权重/尺度 → GREEN**

禁止为场景添加模式 gate。只允许调整 residual、barrier margin、normalization、regularization、line-search acceptance 和解析 Jacobian 错误。

- [ ] **Step 6: 运行完整纯 tensor suite**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti -q
```

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti
git commit -m "test: verify joint MPC rolling behaviors"
```

---

### Task 13: 真实 IsaacLab Smoke 与 Reference 时序

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_isaaclab_probe.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_isaaclab_runtime.py`

- [ ] **Step 1: 写 headless probe**

启动真实 semantic trajectory env，显式设置 `planner_backend="joint_mpc_rti"`，读取真实 root/joint/foot/RayCaster，运行若干 rolling steps，输出严格 JSON：planner status、reference target step、real-vs-reference joint/root/foot error、field version/age、collision/penetration、timing。

- [ ] **Step 2: 运行 2-env RED/Smoke**

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_isaaclab_probe.py \
  --num-envs 2 --steps 10 --headless
```

Expected before full wiring: FAIL at backend attach or field sync; after implementation: exit 0、finite reference、target step 正确。

- [ ] **Step 3: 运行 16-env 行为 smoke**

分别运行 flat、small、large、rough terrain 条件；验证 reference timing、真实状态注入、field version、无环境串扰和无非有限结果。

- [ ] **Step 4: 运行 pytest wrapper**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_isaaclab_runtime.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/joint_mpc_rti
git commit -m "test: add IsaacLab joint MPC runtime coverage"
```

---

### Task 14: 1024 环境 GPU 性能与固定执行图

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/diagnostics/profiler.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/diagnostics/metrics.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/diagnostics/validation.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_performance.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_perf_probe.py`

- [ ] **Step 1: 写性能 probe 和非同步计时测试**

probe 预分配 `B=1024,H=16` 输入，完成 compile/warm-up 后，用 CUDA events 包围 1000 次 planner hot path；循环内不 synchronize，循环结束后统一同步并报告 total/mean/P50/P95/P99/max/peak memory/nonfinite count。

- [ ] **Step 2: 建立初始基线**

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_perf_probe.py \
  --num-envs 1024 --horizon 16 --steps 1000 --warmup 100
```

- [ ] **Step 3: 固定 workspace 和编译图**

移除热路径动态分配、字典重建、CPU 分支和同步；稳定后使用 `torch.compile` 与 CUDA Graph 或等价固定执行图。诊断关闭时 loss breakdown 使用预分配 tensor views，不触发 host readback。

- [ ] **Step 4: 若仍超时，按顺序优化**

1. 融合 world query + gradient gather；
2. 融合 FK link samples + Jacobian；
3. 合并 residual/GGN block kernel；
4. 检查 associative scan kernel 数；
5. 必要时为 scan/query 编写 Triton/CUDA 专用 kernel。

不得通过删除全身安全采样、减少 1024 batch 或伪造异步 enqueue 时间通过。

- [ ] **Step 5: 达到硬目标**

Required:

```text
1024 envs
H=16
1000 planner calls
total <= 3.0 s
mean <= 3.0 ms
nonfinite = 0
no sustained memory growth
```

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/diagnostics Go2Pvcnn/tests/joint_mpc_rti
git commit -m "perf: optimize joint MPC RTI for 1024 environments"
```

---

### Task 15: Full Regression、Notes 与完成验证

**Files:**
- Update: `notes/todo.md`
- Create/Update: relevant `notes/todo/T*.md` branch page for joint MPC RTI
- Update: `notes/log/index.md`
- Create: per-verification logs under `notes/log/`
- Update: `notes/human/human-09-extension-planner-mapping.md`
- Update: `notes/human/human-10-extension-planner-runtime.md`
- Update: `notes/human/human-11-extension-trajectory-reward.md`
- Update paired AI notes when public runtime contracts change.

- [ ] **Step 1: 运行全部 joint suite**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti -q
```

- [ ] **Step 2: 运行现有 planner/reward/factory 回归**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py \
  Go2Pvcnn/tests/test_mpc_rl_participation.py \
  Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

- [ ] **Step 3: 运行真实 IsaacLab acceptance**

按 Task 13 串行运行 2/16/1024-env probes，保存严格 JSON 与日志；明确区分 planner hot path、field build、RayCaster、Isaac step 和渲染时间。

- [ ] **Step 4: 静态检查**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  $(find Go2Pvcnn/extension/joint_mpc_rti -name '*.py' -type f | sort)
git diff --check
git status --short
```

- [ ] **Step 5: 更新 notes**

记录每个 distinct verification 的命令、输入条件、指标、结果、baseline/candidate ref、未验证项；todo dashboard 只增加简短 active front 和链接，不堆入完整历史。

- [ ] **Step 6: 最终 commit**

```bash
git add notes Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti \
  Go2Pvcnn/extension/trajectory_manager_factory.py \
  Go2Pvcnn/extension/mdp/rewards_reference.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py
git commit -m "feat: complete joint MPC RTI GPU planner"
```

---

## 完成定义

只有以下条件同时满足才可声明完成：

- 新 backend 显式选择可用，旧 `mpc` 默认和行为不变。
- command body/world 方向、固定 trot、解析 FK/Jacobian 和 single-height world query 测试通过。
- Loss breakdown 与设计一致，没有语义硬 gate、独立 foot state 或 FK consistency loss。
- 每个 rolling step 的 `x0` 是真实状态，发布的是 `x1`，PPO 在下一步比较正确 pending reference。
- 平地 command、小物体 true-overpass、大障碍绕行、粗糙地形多 gait 周期全部达到设计验收。
- 1024 环境无串扰、无非有限、无显存持续增长。
- 1000 次 planner hot-path 总计不超过 3 秒。
- 真实 IsaacLab 测试使用 `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim` 串行完成。
- todo/log/human/AI notes 与最终代码和验证证据一致。
