# T302l MPC RL Participation And Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留已完成的 MPC-RL participation / world-foot reference 工作，并把旧的 per-body global filtered ContactSensor 方案替换为两个数量对齐的 IsaacLab 兼容语义真实碰撞 sensor。

**Architecture:** 已完成部分保持不动：selector 选择参与 MPC 的 env，manager 只规划 selected env，reference cache 使用 world foot tracking。新工作只替换 semantic contact 链路：在 `go2_pvcnn/sensor/semantic_contacter` 新增继承 IsaacLab `ContactSensor` 的 `SemanticGlobalContactSensor`，scene 只注册 `semantic_contact_small` 和 `semantic_contact_large` 两个 sensor，reward 通过标准 IsaacLab `RewTerm` / `SceneEntityCfg` 消费两个 sensor。

**Tech Stack:** IsaacLab `ContactSensor` / `ContactSensorCfg` / `SceneEntityCfg`、PhysX tensor `RigidContactView`、PyTorch、RSL-RL、`env_isaacsim` (`/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`)。

---

## Source Spec

- [../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html](../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html)
- Low-small 不可回归约束：[../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html)

## Current State

### Completed And Kept

- [x] MPC reference participation selector 已完成。
  - Commit: `c0286f7 feat: add mpc reference participation selector`
- [x] MPC manager selected-env planning 已完成。
  - Commit: `2b1acc6 feat: select mpc reference envs in manager`
- [x] MPC horizon / cache / replan interval 已对齐到 25 step。
  - Covered by manager/env cfg work and final verification.
- [x] `ReferenceTrajectoryCache.foot_pos_w` 已完成。
  - Commit: `a9d43b7 feat: add world foot positions to reference cache`
- [x] `reference_foot_pos_reward()` 已改为 world-frame foot tracking。
  - Commit: `0736552 fix: track reference feet in world frame`
- [x] 1024 env / 64 MPC env 性能 probe 已完成，历史结果 `epoch_seconds=5.2561346013098955`。
  - Commit: `e80049a test: add mpc rl performance acceptance probe`
- [x] MPC-only human command note 已对齐。
  - Commit: `7338cfc docs: update mpc planner human commands`
- [x] 新 semantic contact 设计 HTML 已更新。
  - Commit: `e4d9a25 docs: update mpc rl semantic contact design`

### Superseded And To Replace

- [ ] 旧的 26 个 per-body semantic contact sensor 配置需要替换。
  - 旧提交：`509192b feat: add semantic contact sensors to mpc rl cfg`
  - 问题：`filter_prim_paths_expr=["/World/semantic_course/small/.*"]` 会在 IsaacLab 内部变成一级 glob `/World/semantic_course/small/*`，只匹配 `row_*`，并触发 PhysX `expected 1024, found 7/5`。
- [ ] 旧 reward `semantic_filtered_contact_collision_reward(...)` 需要替换或兼容迁移。
  - 旧提交：`dde022c feat: add semantic filtered contact reward`
  - 问题：reward 依赖 26 个旧 sensor name 列表，不符合新设计的 2 个全局语义 sensor 接口。
- [ ] 旧 IsaacLab smoke `test_mpc_semantic_contact_isaaclab.py` 需要改成 2-sensor 数量对齐测试。
  - 旧提交：`21e1479 test: validate semantic contact sensors in isaaclab`
  - 问题：原 smoke 只检查旧 per-body sensor shape，没有检查全部 `slot_*` semantic object 覆盖，也没有阻止 `expected 1024, found 7/5` 日志。

## File Structure For Remaining Work

- Create `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/__init__.py`
  - Export `SemanticGlobalContactSensor`.
- Create `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py`
  - Subclass IsaacLab `ContactSensor`.
  - Resolve robot body leaf paths for all envs and selected body names.
  - Resolve semantic leaf obstacle paths under `/World/semantic_course/{small,large}/row_*/col_*/slot_*`.
  - Create a PhysX `RigidContactView` with exact sensor path list and repeated per-sensor filter path lists.
  - Publish `data.force_matrix_w` as `[num_envs, num_bodies, num_semantic_objects, 3]`.
- Modify `Go2Pvcnn/extension/mdp/semantic_contact_rewards.py`
  - Replace or extend the old per-body-sensor reward with `semantic_global_contact_collision_reward(...)`.
  - Keep numeric helper tests for threshold, monotonicity, clipping, NaN/Inf.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - Remove 26 old `semantic_contact_{body}_{small,large}` scene entries.
  - Add `semantic_contact_small` and `semantic_contact_large`.
  - Reward params use `SceneEntityCfg("semantic_contact_small")` and `SceneEntityCfg("semantic_contact_large")`.
- Modify `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`
  - Assert the cfg now exposes only the 2 global semantic contact sensors.
  - Assert reward params point to the 2 sensor names.
- Modify `Go2Pvcnn/tests/test_semantic_contact_rewards.py`
  - Add tests for the new reward interface and tensor shapes.
- Modify `Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py`
  - Run real IsaacLab smoke with `teacher_elevation_trajectory_mpc_semantic_env_cfg.py`.
  - Add final 1024 env quantity-alignment test.
- Update notes/log files after each IsaacLab run:
  - `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-smoke.md`
  - `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-1024.md`

## Global Constraints

- Do not modify `decode_parametric_trajectory()` loss semantics.
- Do not add MPC optimizer losses.
- Do not add touchdown hard projection, touchdown snapping, or hard foot separation.
- Keep 2026-05-28 low-small behavior unchanged.
- Do not modify IsaacLab source.
- All IsaacLab validation uses:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python ...
```

- Final quantity-alignment validation must use `num_envs=1024` and the real cfg file:

```text
Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

---

### Task 10: Add `SemanticGlobalContactSensor` Skeleton And Unit Tests

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/__init__.py`
- Create: `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py`
- Test: `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`

- [ ] **Step 1: Write cfg import test**

Add a test in `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`:

```python
def test_semantic_global_contact_sensor_importable():
    from go2_pvcnn.sensor.semantic_contacter import SemanticGlobalContactSensor
    from isaaclab.sensors import ContactSensor

    assert issubclass(SemanticGlobalContactSensor, ContactSensor)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py::test_semantic_global_contact_sensor_importable -q
```

Expected: FAIL because `go2_pvcnn.sensor.semantic_contacter` does not exist.

- [ ] **Step 3: Add minimal package and class**

Create `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py`:

```python
from __future__ import annotations

from isaaclab.sensors import ContactSensor


class SemanticGlobalContactSensor(ContactSensor):
    """ContactSensor variant for global static semantic-course objects."""
```

Create `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/__init__.py`:

```python
from .semantic_global_contact_sensor import SemanticGlobalContactSensor

__all__ = ["SemanticGlobalContactSensor"]
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py::test_semantic_global_contact_sensor_importable -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py
git commit -m "feat: add semantic global contact sensor shell"
```

---

### Task 11: Implement Semantic Leaf Path Resolution

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py`
- Test: `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`

- [ ] **Step 1: Write pure path-resolution tests**

Add tests that use fake path lists so they do not need IsaacLab:

```python
def test_semantic_global_contact_leaf_filter_keeps_only_slots():
    from go2_pvcnn.sensor.semantic_contacter.semantic_global_contact_sensor import (
        filter_semantic_leaf_obstacle_paths,
    )

    paths = [
        "/World/semantic_course/small/row_00",
        "/World/semantic_course/small/row_00/col_00",
        "/World/semantic_course/small/row_00/col_00/slot_00",
        "/World/semantic_course/small/row_00/col_01/slot_01",
        "/World/semantic_course/large/row_00/col_00/slot_00",
    ]

    assert filter_semantic_leaf_obstacle_paths(paths, "/World/semantic_course/small") == [
        "/World/semantic_course/small/row_00/col_00/slot_00",
        "/World/semantic_course/small/row_00/col_01/slot_01",
    ]
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py::test_semantic_global_contact_leaf_filter_keeps_only_slots -q
```

Expected: FAIL because `filter_semantic_leaf_obstacle_paths` is not implemented.

- [ ] **Step 3: Implement leaf filter**

Add to `semantic_global_contact_sensor.py`:

```python
def filter_semantic_leaf_obstacle_paths(paths: list[str], semantic_root: str) -> list[str]:
    prefix = semantic_root.rstrip("/") + "/"
    out: list[str] = []
    for path in sorted(str(p) for p in paths):
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix):]
        parts = rel.split("/")
        if len(parts) == 3 and parts[0].startswith("row_") and parts[1].startswith("col_") and parts[2].startswith("slot_"):
            out.append(path)
    return out
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py::test_semantic_global_contact_leaf_filter_keeps_only_slots -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py
git commit -m "test: define semantic obstacle leaf filtering"
```

---

### Task 12: Replace Env CFG With Two Global Semantic Contact Sensors

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Modify: `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`

- [ ] **Step 1: Write cfg structure test**

Replace the old per-body sensor expectations with:

```python
def test_mpc_semantic_cfg_uses_two_global_semantic_contact_sensors():
    from isaaclab.managers import SceneEntityCfg
    from go2_pvcnn.sensor.semantic_contacter import SemanticGlobalContactSensor
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        SEMANTIC_CONTACT_BODY_NAMES,
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
    )

    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
    assert cfg.scene.semantic_contact_small.class_type is SemanticGlobalContactSensor
    assert cfg.scene.semantic_contact_large.class_type is SemanticGlobalContactSensor
    assert cfg.scene.semantic_contact_small.filter_prim_paths_expr == ["/World/semantic_course/small/.*"]
    assert cfg.scene.semantic_contact_large.filter_prim_paths_expr == ["/World/semantic_course/large/.*"]

    for body in SEMANTIC_CONTACT_BODY_NAMES:
        assert not hasattr(cfg.scene, f"semantic_contact_{body}_small")
        assert not hasattr(cfg.scene, f"semantic_contact_{body}_large")

    params = cfg.rewards.semantic_contact_collision.params
    assert isinstance(params["small_sensor_cfg"], SceneEntityCfg)
    assert isinstance(params["large_sensor_cfg"], SceneEntityCfg)
    assert params["small_sensor_cfg"].name == "semantic_contact_small"
    assert params["large_sensor_cfg"].name == "semantic_contact_large"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py::test_mpc_semantic_cfg_uses_two_global_semantic_contact_sensors -q
```

Expected: FAIL because cfg still has old per-body sensor attributes and old reward params.

- [ ] **Step 3: Modify scene cfg**

In `teacher_elevation_trajectory_mpc_semantic_env_cfg.py`:

- Import `SemanticGlobalContactSensor`.
- Replace `_semantic_contact_sensor(body_name, semantic_root)` with `_semantic_global_contact_sensor(semantic_root)`.
- Remove all `semantic_contact_{body}_{small,large}` assignments.
- Add:

```python
semantic_contact_small = _semantic_global_contact_sensor(SEMANTIC_COURSE_SMALL_ROOT)
semantic_contact_large = _semantic_global_contact_sensor(SEMANTIC_COURSE_LARGE_ROOT)
```

The helper should return:

```python
ContactSensorCfg(
    class_type=SemanticGlobalContactSensor,
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    update_period=0.0,
    history_length=0,
    track_air_time=False,
    debug_vis=False,
    filter_prim_paths_expr=[f"{semantic_root}/.*"],
)
```

- [ ] **Step 4: Modify reward cfg**

Update reward params:

```python
semantic_contact_collision = RewTerm(
    func=semantic_global_contact_collision_reward,
    weight=1.0,
    params={
        "small_sensor_cfg": SceneEntityCfg("semantic_contact_small"),
        "large_sensor_cfg": SceneEntityCfg("semantic_contact_large"),
        "body_names": SEMANTIC_CONTACT_BODY_NAMES,
        "body_weights": SEMANTIC_CONTACT_BODY_WEIGHTS,
        "force_threshold": 1.0,
        "force_scale": 50.0,
        "force_clip": 1.0,
        "small_weight": 1.0,
        "large_weight": 2.0,
    },
)
```

- [ ] **Step 5: Run cfg tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py
git commit -m "feat: switch mpc semantic cfg to global contact sensors"
```

---

### Task 13: Implement New Reward Interface

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/semantic_contact_rewards.py`
- Modify: `Go2Pvcnn/tests/test_semantic_contact_rewards.py`

- [ ] **Step 1: Add tensor numeric helper test**

Add:

```python
def test_global_semantic_contact_penalty_shapes_and_body_weights():
    from extension.mdp.semantic_contact_rewards import global_semantic_contact_penalty_from_matrices

    small = torch.zeros((2, 3, 4, 3), dtype=torch.float32)
    large = torch.zeros((2, 3, 2, 3), dtype=torch.float32)
    small[0, 1, 2, 0] = 6.0
    large[1, 2, 1, 1] = 11.0

    penalty = global_semantic_contact_penalty_from_matrices(
        small,
        large,
        body_weights=(1.0, 2.0, 5.0),
        force_threshold=1.0,
        force_scale=10.0,
        force_clip=10.0,
        small_weight=1.0,
        large_weight=2.0,
    )

    torch.testing.assert_close(penalty, torch.tensor([1.0, 10.0]))
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_contact_rewards.py::test_global_semantic_contact_penalty_shapes_and_body_weights -q
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement helper**

Add:

```python
def global_semantic_contact_penalty_from_matrices(
    small_force_matrix_w: Tensor,
    large_force_matrix_w: Tensor,
    *,
    body_weights: tuple[float, ...],
    force_threshold: float,
    force_scale: float,
    force_clip: float,
    small_weight: float,
    large_weight: float,
) -> Tensor:
    small = torch.as_tensor(small_force_matrix_w, dtype=torch.float32)
    large = torch.as_tensor(large_force_matrix_w, dtype=torch.float32, device=small.device)
    if small.ndim != 4 or large.ndim != 4 or int(small.shape[-1]) != 3 or int(large.shape[-1]) != 3:
        raise ValueError("force matrices must have shape [N,B,O,3]")
    if int(small.shape[0]) != int(large.shape[0]) or int(small.shape[1]) != int(large.shape[1]):
        raise ValueError("small and large matrices must share [N,B]")
    weights = torch.as_tensor(body_weights, dtype=torch.float32, device=small.device)
    if int(weights.numel()) != int(small.shape[1]):
        raise ValueError("body_weights length must match body dimension")
    small_excess = torch.relu(torch.linalg.vector_norm(small, dim=-1) - float(force_threshold)).sum(dim=-1)
    large_excess = torch.relu(torch.linalg.vector_norm(large, dim=-1) - float(force_threshold)).sum(dim=-1)
    total = (weights[None, :] * (float(small_weight) * small_excess + float(large_weight) * large_excess)).sum(dim=-1)
    return (total / max(float(force_scale), 1.0e-6)).clamp(0.0, float(force_clip))
```

- [ ] **Step 4: Add IsaacLab reward function test with fake env**

Add:

```python
def test_semantic_global_contact_collision_reward_reads_scene_entity_cfgs():
    from isaaclab.managers import SceneEntityCfg
    from extension.mdp.semantic_contact_rewards import semantic_global_contact_collision_reward

    class _Sensor:
        def __init__(self, matrix):
            self.data = type("Data", (), {"force_matrix_w": matrix})()
            self.body_names = ["a", "b"]

    class _Scene(dict):
        pass

    env = type("Env", (), {})()
    env.device = "cpu"
    env.num_envs = 1
    env.scene = _Scene()
    env.scene["semantic_contact_small"] = _Sensor(torch.zeros((1, 2, 1, 3)))
    env.scene["semantic_contact_large"] = _Sensor(torch.ones((1, 2, 1, 3)) * 3.0)

    reward = semantic_global_contact_collision_reward(
        env,
        small_sensor_cfg=SceneEntityCfg("semantic_contact_small"),
        large_sensor_cfg=SceneEntityCfg("semantic_contact_large"),
        body_names=("a", "b"),
        body_weights=(1.0, 1.0),
        force_threshold=1.0,
        force_scale=10.0,
        force_clip=10.0,
        small_weight=1.0,
        large_weight=2.0,
    )
    assert reward.shape == (1,)
    assert reward.item() < 0.0
```

- [ ] **Step 5: Implement reward function**

Add:

```python
def semantic_global_contact_collision_reward(
    env,
    small_sensor_cfg,
    large_sensor_cfg,
    body_names: tuple[str, ...],
    body_weights: tuple[float, ...],
    force_threshold: float = 1.0,
    force_scale: float = 50.0,
    force_clip: float = 1.0,
    small_weight: float = 1.0,
    large_weight: float = 2.0,
) -> Tensor:
    device = torch.device(getattr(env, "device", "cpu"))
    small_sensor = _scene_sensor(env, small_sensor_cfg.name)
    large_sensor = _scene_sensor(env, large_sensor_cfg.name)
    if tuple(getattr(small_sensor, "body_names", ())) != tuple(body_names):
        raise ValueError("small semantic contact sensor body_names do not match reward body_names")
    if tuple(getattr(large_sensor, "body_names", ())) != tuple(body_names):
        raise ValueError("large semantic contact sensor body_names do not match reward body_names")
    penalty = global_semantic_contact_penalty_from_matrices(
        torch.as_tensor(small_sensor.data.force_matrix_w, dtype=torch.float32, device=device),
        torch.as_tensor(large_sensor.data.force_matrix_w, dtype=torch.float32, device=device),
        body_weights=body_weights,
        force_threshold=force_threshold,
        force_scale=force_scale,
        force_clip=force_clip,
        small_weight=small_weight,
        large_weight=large_weight,
    )
    return -penalty.to(device=device)
```

- [ ] **Step 6: Run reward tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_contact_rewards.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/extension/mdp/semantic_contact_rewards.py Go2Pvcnn/tests/test_semantic_contact_rewards.py
git commit -m "feat: add global semantic contact reward"
```

---

### Task 14: Implement PhysX Contact View Initialization

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py`
- Test: `Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py`

- [ ] **Step 1: Add small IsaacLab smoke test**

Update `Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py` to start the real MPC semantic cfg with a small env count first, for example `num_envs=8`, and assert:

```python
assert root.scene.sensors["semantic_contact_small"].data.force_matrix_w.ndim == 4
assert root.scene.sensors["semantic_contact_large"].data.force_matrix_w.ndim == 4
assert root.scene.sensors["semantic_contact_small"].data.force_matrix_w.shape[0] == 8
assert root.scene.sensors["semantic_contact_small"].data.force_matrix_w.shape[1] == len(SEMANTIC_CONTACT_BODY_NAMES)
assert root.scene.sensors["semantic_contact_large"].data.force_matrix_w.shape[1] == len(SEMANTIC_CONTACT_BODY_NAMES)
```

- [ ] **Step 2: Run smoke to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py::test_mpc_semantic_global_contact_sensors_real_isaaclab_small -q
```

Expected: FAIL until `_initialize_impl()` creates a valid contact view.

- [ ] **Step 3: Implement `_initialize_impl()`**

Implementation must copy only the necessary shape/buffer setup from IsaacLab `ContactSensor._initialize_impl()` and change the contact view creation:

```python
sensor_paths = self._resolve_ordered_robot_body_paths()
filter_paths = self._resolve_semantic_filter_leaf_paths()
self._contact_physx_view = self._physics_sim_view.create_rigid_contact_view(
    sensor_paths,
    filter_patterns=[filter_paths] * len(sensor_paths),
)
self._num_bodies = len(self._configured_body_names)
self._data.force_matrix_w = torch.zeros(
    self._num_envs,
    self._num_bodies,
    len(filter_paths),
    3,
    device=self._device,
)
```

The implementation must also set:

```python
self._body_names = list(self._configured_body_names)
self._semantic_filter_paths = list(filter_paths)
```

and expose `body_names` as the configured body order.

- [ ] **Step 4: Implement `_update_buffers_impl()` reshape**

Use:

```python
force_matrix = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
force_matrix = force_matrix.view(self._num_envs, self._num_bodies, self.contact_physx_view.filter_count, 3)
self._data.force_matrix_w[env_ids] = force_matrix[env_ids]
```

If `track_pose` and `track_air_time` are not needed for these sensors, keep them disabled and raise a clear `RuntimeError` if enabled.

- [ ] **Step 5: Run small smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py::test_mpc_semantic_global_contact_sensors_real_isaaclab_small -q
```

Expected: PASS and no PhysX `expected 1024, found 7/5` errors.

- [ ] **Step 6: Write log and commit**

Create `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-smoke.md` with:

```text
Purpose: Validate SemanticGlobalContactSensor with real IsaacLab small env count.
Command: CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest ...
Result: PASS/FAIL
Observed shapes: small=[N,B,O,3], large=[N,B,O,3]
PhysX filter errors: none or copied error lines
```

Commit:

```bash
git add Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-smoke.md
git commit -m "feat: implement semantic global contact sensor"
```

---

### Task 15: 1024 Env Quantity Alignment Acceptance

**Files:**
- Modify: `Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py`
- Create: `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-1024.md`

- [ ] **Step 1: Add 1024 alignment test**

The test must use the real cfg from `teacher_elevation_trajectory_mpc_semantic_env_cfg.py` and `num_envs=1024`.

Assertions:

```python
expected_small = count_stage_paths("/World/semantic_course/small/row_*/col_*/slot_*")
expected_large = count_stage_paths("/World/semantic_course/large/row_*/col_*/slot_*")
small = root.scene.sensors["semantic_contact_small"]
large = root.scene.sensors["semantic_contact_large"]

assert small.data.force_matrix_w.shape == (1024, len(SEMANTIC_CONTACT_BODY_NAMES), expected_small, 3)
assert large.data.force_matrix_w.shape == (1024, len(SEMANTIC_CONTACT_BODY_NAMES), expected_large, 3)
assert small.body_names == list(SEMANTIC_CONTACT_BODY_NAMES)
assert large.body_names == list(SEMANTIC_CONTACT_BODY_NAMES)
assert small.contact_physx_view.sensor_count == 1024 * len(SEMANTIC_CONTACT_BODY_NAMES)
assert large.contact_physx_view.sensor_count == 1024 * len(SEMANTIC_CONTACT_BODY_NAMES)
assert small.contact_physx_view.filter_count == expected_small
assert large.contact_physx_view.filter_count == expected_large
```

- [ ] **Step 2: Run 1024 test**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py::test_mpc_semantic_global_contact_sensors_quantity_alignment_1024 -q
```

Expected: PASS. The run must not emit:

```text
Filter pattern '/World/semantic_course/small/*' did not match the correct number of entries
Filter pattern '/World/semantic_course/large/*' did not match the correct number of entries
```

- [ ] **Step 3: Write acceptance log**

Create `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-1024.md` with:

```text
Purpose: Validate real 1024-env quantity alignment for semantic global contact sensors.
Command: CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest ...
num_envs: 1024
body_count: 13
expected_small_slot_count: <value>
expected_large_slot_count: <value>
small_shape: [1024, 13, expected_small_slot_count, 3]
large_shape: [1024, 13, expected_large_slot_count, 3]
PhysX filter errors: none
Result: PASS/FAIL
```

- [ ] **Step 4: Commit**

```bash
git add Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-1024.md
git commit -m "test: validate semantic contact quantity alignment"
```

---

### Task 16: Final Regression And Notes Alignment

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302l-mpc-rl-participation-and-reward-plan.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-final.md`

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_semantic_contact_rewards.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py -q
```

Expected: PASS.

- [ ] **Step 2: Run MPC backend regression subset**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: PASS.

- [ ] **Step 3: Run 1024 performance probe again**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
```

Expected:

```text
num_envs = 1024
selected_mpc_envs = 64
epoch_seconds <= 10
```

- [ ] **Step 4: Run real train one iteration**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py --experiment teacher_elevation_trajectory_mpc_semantic --planner-backend mpc --num_envs 1024 --max_iterations 1 --headless --device cuda:0
```

Expected: exit code 0 and no semantic contact filter-count errors.

- [ ] **Step 5: Write final verification log**

Create `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-final.md` with commands, pass/fail results, performance metrics, and remaining risk.

- [ ] **Step 6: Update todo dashboard and log index**

Update:

```text
notes/todo.md
notes/log/index.md
```

The dashboard should show T302l remaining semantic global contact tasks as closed when all tests pass.

- [ ] **Step 7: Commit**

```bash
git add notes/todo.md notes/log/index.md notes/todo/T302l-mpc-rl-participation-and-reward-plan.md notes/log/YYYY-MM-DD-HHMM-t302l-semantic-global-contact-final.md
git commit -m "docs: record semantic global contact verification"
```

---

## Self-Review

- Spec coverage:
  - Two global semantic contact sensors: Task 10, Task 12, Task 14.
  - Sensor path and semantic object quantity alignment: Task 11, Task 14, Task 15.
  - IsaacLab reward interface: Task 12, Task 13.
  - 1024 env real cfg validation: Task 15.
  - Performance and train validation: Task 16.
- Placeholder scan:
  - This plan intentionally uses concrete file paths, commands, expected results, and test names. No open-ended placeholder steps remain.
- Type/interface consistency:
  - Sensor names are consistently `semantic_contact_small` and `semantic_contact_large`.
  - Reward function is consistently `semantic_global_contact_collision_reward`.
  - Force matrix shape is consistently `[num_envs, num_bodies, num_semantic_objects, 3]`.
