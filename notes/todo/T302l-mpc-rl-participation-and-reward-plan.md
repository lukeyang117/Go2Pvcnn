# MPC RL Participation And Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入 MPC 到 RL 训练时，只让配置允许的 env 参与 MPC reference，并把 foot tracking 和语义碰撞 reward 改成 world-frame / IsaacLab 真实 contact 的实现。

**Architecture:** 训练 runtime 增加一个独立 participation selector，MPC manager 只规划 selected env 并用 mask 控制 reference reward。Reference cache 增加 `foot_pos_w` 支持 world-frame foot tracking；真实语义碰撞 reward 独立放在 `extension/mdp/semantic_contact_rewards.py`，使用 per-body filtered `ContactSensor.data.force_matrix_w`，不再依赖 semantic height scanner。

**Tech Stack:** IsaacLab ManagerBasedRLEnv、IsaacLab ContactSensor / ContactSensorCfg、PyTorch、RSL-RL、`env_isaacsim` (`/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`)。

---

## Source Spec

- [../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html](../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html)
- Low-small 不可回归约束：[../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html)

## File Structure

- Create `Go2Pvcnn/extension/batch_mpc_planner/participation.py`
  - `MpcReferenceParticipationCfg`
  - `MpcTerrainDifficultyPair`
  - `select_mpc_reference_envs(...)`
  - terrain name / col / row filtering and round-robin cursor logic.
- Modify `Go2Pvcnn/extension/batch_mpc_planner/config.py`
  - Add participation config to `MpcPlannerCfg`.
  - Add task-cfg override fields.
- Modify `Go2Pvcnn/extension/batch_mpc_planner/manager.py`
  - Keep `_selection_cursor`.
  - Read terrain metadata from IsaacLab terrain.
  - Select envs by participation selector before `plan_segment`.
  - Use 25-step aligned replan behavior.
- Modify `Go2Pvcnn/extension/reference/cache.py`
  - Add `foot_pos_w` to `ReferenceTrajectoryCache`.
  - Update `to`, `is_ready`, validation helpers, expand/masked-write/standstill utilities.
- Modify `Go2Pvcnn/extension/batch_mpc_planner/adapter.py`
  - Populate `foot_pos_w` from `MpcPlannerResult.foot_pos`.
  - Preserve `foot_pos_w` in clone/scatter/blend/standstill caches.
- Modify `Go2Pvcnn/extension/batched_together_planner/adapter.py`
  - Populate `foot_pos_w` where a result already exposes world foot positions.
  - Keep `foot_pos_root` for compatibility.
- Modify `Go2Pvcnn/extension/mdp/rewards_reference.py`
  - `reference_foot_pos_reward()` compares IsaacLab `body_pos_w` with `cache.foot_pos_w`.
  - MPC backend uses `manager.current_frame_ids()` instead of `episode_length % horizon`.
- Create `Go2Pvcnn/extension/mdp/semantic_contact_rewards.py`
  - `semantic_filtered_contact_collision_reward(...)`
  - numeric helper for `force_matrix_w` aggregation and clipped force penalty.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - Add per-body filtered contact sensors for global semantic course small/large roots.
  - Replace old highmap-based `swing_leg_collision_reward` with real contact reward.
  - Set `reference_trajectory_horizon = 25`, `reference_replan_interval_steps = 25`.
  - Set training diagnostics sync off.
- Tests:
  - Modify/add focused tests under `Go2Pvcnn/tests/test_mpc_rl_participation.py`.
  - Modify existing cache/adapter tests in `Go2Pvcnn/tests/test_batch_mpc_backend.py`.
  - Add contact reward numeric tests under `Go2Pvcnn/tests/test_semantic_contact_rewards.py`.
  - Add env cfg sensor smoke tests under `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`.

## Global Constraints

- Do not modify `decode_parametric_trajectory()` loss semantics.
- Do not add MPC optimizer losses.
- Do not add touchdown hard projection, touchdown snapping, or hard foot separation.
- Run IsaacLab smoke/performance with:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python ...
```

- Performance target: real RL training entry, `num_envs=1024`, `selected_mpc_envs=64`, single RL epoch `<= 10s`.

---

### Task 1: Reference Cache Adds `foot_pos_w`

**Files:**
- Modify: `Go2Pvcnn/extension/reference/cache.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/adapter.py`
- Modify: `Go2Pvcnn/extension/batched_together_planner/adapter.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Write failing cache ABI tests**

Add tests near the existing cache adapter tests in `Go2Pvcnn/tests/test_batch_mpc_backend.py`:

```python
def test_mpc_reference_cache_exports_world_feet():
    result = _make_simple_mpc_result(batch=2, horizon=5)
    cache = mpc_result_to_reference_cache(result)
    assert cache.foot_pos_w is not None
    assert cache.foot_pos_w.shape == (2, 5, 4, 3)
    torch.testing.assert_close(cache.foot_pos_w, result.foot_pos)


def test_reference_cache_clone_scatter_blend_preserve_world_feet():
    result = _make_simple_mpc_result(batch=3, horizon=6)
    cache = mpc_result_to_reference_cache(result)
    cloned = clone_reference_cache(cache)
    assert cloned.foot_pos_w is not None
    torch.testing.assert_close(cloned.foot_pos_w, cache.foot_pos_w)

    new = mpc_result_to_reference_cache(_make_simple_mpc_result(batch=1, horizon=6, offset=10.0))
    scatter_cache_rows(cloned, new, torch.tensor([1], device=cache.root_pos_w.device))
    torch.testing.assert_close(cloned.foot_pos_w[1], new.foot_pos_w[0])
```

If `_make_simple_mpc_result` does not exist, add a local helper in the test file:

```python
def _make_simple_mpc_result(batch: int, horizon: int, offset: float = 0.0):
    root = torch.zeros((batch, horizon, 3), dtype=torch.float32)
    root[..., 0] = torch.arange(horizon, dtype=torch.float32).view(1, horizon) + offset
    foot = root[:, :, None, :].expand(batch, horizon, 4, 3).clone()
    foot[..., 0] += torch.tensor([0.2, 0.2, -0.2, -0.2]).view(1, 1, 4)
    foot[..., 1] += torch.tensor([0.1, -0.1, 0.1, -0.1]).view(1, 1, 4)
    return SimpleNamespace(
        root_pos=root,
        root_rpy=torch.zeros((batch, horizon, 3), dtype=torch.float32),
        foot_pos=foot,
        joint_angles=torch.zeros((batch, horizon, 12), dtype=torch.float32),
        contact_state=torch.ones((batch, horizon, 4), dtype=torch.bool),
        planned_touchdown_w=foot,
    )
```

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_reference_cache_exports_world_feet Go2Pvcnn/tests/test_batch_mpc_backend.py::test_reference_cache_clone_scatter_blend_preserve_world_feet -q
```

Expected: FAIL because `ReferenceTrajectoryCache` has no `foot_pos_w`.

- [x] **Step 3: Implement `foot_pos_w` in cache ABI**

Update `ReferenceTrajectoryCache`:

```python
@dataclass
class ReferenceTrajectoryCache:
    root_pos_w: torch.Tensor | None = None
    root_quat_w: torch.Tensor | None = None
    joint_angles: torch.Tensor | None = None
    foot_pos_w: torch.Tensor | None = None
    foot_pos_root: torch.Tensor | None = None
    contact_state: torch.Tensor | None = None
    planned_touchdown_w: torch.Tensor | None = None
    phase_index: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
```

Update every constructor call in `cache.py` to pass `foot_pos_w`. For functions that only have `foot_pos_root`, reconstruct a compatibility value:

```python
foot_pos_w = None
if cache.foot_pos_w is not None:
    foot_pos_w = exp2(cache.foot_pos_w)
```

Update `to()`:

```python
foot_pos_w=_move(self.foot_pos_w),
```

Update `is_ready()` and validation shape checks so `foot_pos_w` is optional for legacy caches but required by MPC semantic reward tests:

```python
if self.foot_pos_w is not None:
    check_float("foot_pos_w", self.foot_pos_w, (4, 3))
```

- [x] **Step 4: Implement adapter propagation**

In `Go2Pvcnn/extension/batch_mpc_planner/adapter.py`, update `mpc_result_to_reference_cache()`:

```python
foot_pos = _as_device_tensor(result.foot_pos, like=root_pos_w)
foot_pos_root = foot_pos - root_pos_w.unsqueeze(2)
return ReferenceTrajectoryCache(
    root_pos_w=root_pos_w,
    root_quat_w=_as_device_tensor(root_quat_w, like=root_pos_w),
    joint_angles=_as_device_tensor(result.joint_angles, like=root_pos_w),
    foot_pos_w=foot_pos,
    foot_pos_root=foot_pos_root,
    contact_state=_as_device_tensor(result.contact_state, like=root_pos_w, dtype=torch.bool),
    planned_touchdown_w=_as_device_tensor(result.planned_touchdown_w, like=root_pos_w),
    phase_index=phase_index,
    valid_mask=valid_mask,
)
```

Update `standstill_cache_from_state()`:

```python
foot_pos_w=foot_pos_w.unsqueeze(1).expand(num_envs, int(horizon), 4, 3).contiguous(),
```

Update `clone_reference_cache`, `scatter_cache_rows`, and `blend_reference_caches` to include `foot_pos_w` using the same shape branch as `foot_pos_root`.

- [x] **Step 5: Run tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_reference_cache_exports_world_feet Go2Pvcnn/tests/test_batch_mpc_backend.py::test_reference_cache_clone_scatter_blend_preserve_world_feet -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/reference/cache.py Go2Pvcnn/extension/batch_mpc_planner/adapter.py Go2Pvcnn/extension/batched_together_planner/adapter.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add world foot positions to reference cache"
```

---

### Task 2: World-Frame `reference_foot_pos_reward` And MPC Phase Tracking

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/rewards_reference.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/manager.py`
- Test: `Go2Pvcnn/tests/test_mpc_rl_participation.py`

- [x] **Step 1: Write failing reward tests**

Create `Go2Pvcnn/tests/test_mpc_rl_participation.py` with:

```python
from types import SimpleNamespace
import torch

from extension.mdp.rewards_reference import reference_foot_pos_reward


class _FakeManager:
    planner_backend = "mpc"

    def __init__(self, cache, mask, frame_ids):
        self._cache = cache
        self._mask = mask
        self._frame_ids = frame_ids
        self.refresh_count = 0

    def refresh_from_env(self, env):
        self.refresh_count += 1
        return self._cache

    def reference_reward_mask(self):
        return self._mask

    def current_frame_ids(self):
        return self._frame_ids


def test_reference_foot_pos_reward_uses_world_feet_and_manager_phase():
    current = torch.zeros((2, 4, 3), dtype=torch.float32)
    ref = current.clone()
    ref[0] += 0.0
    ref[1] += 1.0
    cache = SimpleNamespace(
        foot_pos_w=torch.stack((ref, ref + 10.0), dim=1),
        root_pos_w=torch.zeros((2, 2, 3)),
        is_ready=lambda: True,
        horizon_length=lambda: 2,
    )
    manager = _FakeManager(cache, torch.tensor([1.0, 0.0]), torch.tensor([0, 0]))
    robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=current, root_pos_w=torch.zeros(2, 3), root_quat_w=torch.zeros(2, 4)))
    scene = {"robot": robot}
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(_trajectory_manager=manager),
        scene=scene,
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.tensor([1, 1]),
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3])
    reward = reference_foot_pos_reward(env, sigma=0.5, asset_cfg=asset_cfg)
    torch.testing.assert_close(reward[0], torch.tensor(1.0))
    torch.testing.assert_close(reward[1], torch.tensor(0.0))
```

- [x] **Step 2: Run test and verify failure**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py::test_reference_foot_pos_reward_uses_world_feet_and_manager_phase -q
```

Expected: FAIL because current reward reads `foot_pos_root` and uses `episode_length % horizon` for MPC.

- [x] **Step 3: Change frame selection**

In `_select_reference_frame(env)`, use manager phase for any manager that exposes `current_frame_ids()`:

```python
manager = _trajectory_manager(env)
if manager is not None and hasattr(manager, "current_frame_ids"):
    frame_ids = manager.current_frame_ids()
else:
    frame_ids = _reference_indices(env, horizon)
```

- [x] **Step 4: Change reward to world frame**

Replace `_current_foot_positions_root()` use in `reference_foot_pos_reward()` with:

```python
asset = env.scene[asset_cfg.name]
current_foot = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
ref_foot = _reference_field(env, cache, "foot_pos_w", frame_ids).to(dtype=current_foot.dtype)
err = foot_position_error(current_foot, ref_foot)
reward = exponential_tracking_reward(err, sigma=sigma)
```

Keep the existing mask:

```python
manager = _trajectory_manager(env)
if manager is not None and hasattr(manager, "reference_reward_mask"):
    mask = manager.reference_reward_mask().to(device=reward.device, dtype=reward.dtype)
    reward = reward * mask
```

- [x] **Step 5: Run tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py::test_reference_foot_pos_reward_uses_world_feet_and_manager_phase -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: focused test PASS; backend tests PASS or only unrelated existing skips.

- [x] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/mdp/rewards_reference.py Go2Pvcnn/tests/test_mpc_rl_participation.py
git commit -m "fix: track reference feet in world frame"
```

---

### Task 3: Participation Selector With AND Exclusion And Round-Robin

**Files:**
- Create: `Go2Pvcnn/extension/batch_mpc_planner/participation.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Test: `Go2Pvcnn/tests/test_mpc_rl_participation.py`

- [x] **Step 1: Write failing selector tests**

Append:

```python
from extension.batch_mpc_planner.participation import (
    MpcReferenceParticipationCfg,
    MpcTerrainDifficultyPair,
    select_mpc_reference_envs,
)


def test_participation_exclude_pair_is_terrain_and_row_logic():
    terrain_types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    terrain_levels = torch.tensor([0, 3, 3, 7, 7], dtype=torch.long)
    cfg = MpcReferenceParticipationCfg(
        enabled=True,
        exclude_pairs=(MpcTerrainDifficultyPair(terrain_cols=(1,), terrain_rows=(7,)),),
        selection_mode="round_robin",
    )
    selected, next_cursor, eligible = select_mpc_reference_envs(
        num_envs=5,
        device=torch.device("cpu"),
        terrain_types=terrain_types,
        terrain_levels=terrain_levels,
        terrain_names=["flat", "stairs", "rough"],
        cfg=cfg,
        sample_count=5,
        cursor=0,
        return_eligible=True,
    )
    assert eligible.tolist() == [True, True, True, False, True]
    assert selected.tolist() == [True, True, True, False, True]
    assert next_cursor == 0


def test_participation_round_robin_wraps_inside_eligible_ids():
    terrain_types = torch.zeros(6, dtype=torch.long)
    terrain_levels = torch.zeros(6, dtype=torch.long)
    cfg = MpcReferenceParticipationCfg(enabled=True, selection_mode="round_robin")
    selected, next_cursor, _ = select_mpc_reference_envs(
        num_envs=6,
        device=torch.device("cpu"),
        terrain_types=terrain_types,
        terrain_levels=terrain_levels,
        terrain_names=["flat"],
        cfg=cfg,
        sample_count=4,
        cursor=4,
        return_eligible=True,
    )
    assert selected.tolist() == [True, True, False, False, True, True]
    assert next_cursor == 2
```

- [x] **Step 2: Run tests and verify failure**

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py::test_participation_exclude_pair_is_terrain_and_row_logic Go2Pvcnn/tests/test_mpc_rl_participation.py::test_participation_round_robin_wraps_inside_eligible_ids -q
```

Expected: FAIL because `participation.py` does not exist.

- [x] **Step 3: Implement selector dataclasses**

Create `participation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class MpcTerrainDifficultyPair:
    terrain_cols: tuple[int, ...] | None = None
    terrain_names: tuple[str, ...] | None = None
    terrain_rows: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class MpcReferenceParticipationCfg:
    enabled: bool = True
    include_terrain_cols: tuple[int, ...] | None = None
    include_terrain_names: tuple[str, ...] | None = None
    include_terrain_rows: tuple[int, ...] | None = None
    exclude_pairs: tuple[MpcTerrainDifficultyPair, ...] = field(default_factory=tuple)
    selection_mode: str = "round_robin"
```

- [x] **Step 4: Implement filtering helpers and selector**

Add:

```python
def _isin(values: Tensor, allowed: tuple[int, ...] | None) -> Tensor:
    if allowed is None:
        return torch.ones_like(values, dtype=torch.bool)
    if len(allowed) == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    out = torch.zeros_like(values, dtype=torch.bool)
    for item in allowed:
        out = torch.logical_or(out, values == int(item))
    return out


def _name_mask(terrain_types: Tensor, terrain_names: list[str] | None, names: tuple[str, ...] | None) -> Tensor:
    if names is None:
        return torch.ones_like(terrain_types, dtype=torch.bool)
    if terrain_names is None:
        return torch.zeros_like(terrain_types, dtype=torch.bool)
    wanted_cols = tuple(i for i, name in enumerate(terrain_names) if str(name) in set(names))
    return _isin(terrain_types, wanted_cols)


def _eligible_mask(num_envs: int, device: torch.device, terrain_types, terrain_levels, terrain_names, cfg):
    base = torch.ones(num_envs, dtype=torch.bool, device=device)
    if terrain_types is not None:
        types = torch.as_tensor(terrain_types, dtype=torch.long, device=device).reshape(-1)
        base &= _isin(types, cfg.include_terrain_cols)
        base &= _name_mask(types, terrain_names, cfg.include_terrain_names)
    if terrain_levels is not None and cfg.include_terrain_rows is not None:
        rows = torch.as_tensor(terrain_levels, dtype=torch.long, device=device).reshape(-1)
        base &= _isin(rows, cfg.include_terrain_rows)
    for pair in cfg.exclude_pairs:
        pair_terrain = torch.ones(num_envs, dtype=torch.bool, device=device)
        if terrain_types is not None:
            types = torch.as_tensor(terrain_types, dtype=torch.long, device=device).reshape(-1)
            pair_terrain = _isin(types, pair.terrain_cols) | _name_mask(types, terrain_names, pair.terrain_names)
        rows = torch.as_tensor(terrain_levels, dtype=torch.long, device=device).reshape(-1)
        pair_row = _isin(rows, pair.terrain_rows)
        base &= torch.logical_not(pair_terrain & pair_row)
    return base
```

Implement:

```python
def select_mpc_reference_envs(...):
    eligible = _eligible_mask(...)
    ids = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
    selected = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not bool(cfg.enabled) or int(ids.numel()) == 0 or int(sample_count) <= 0:
        return (selected, int(cursor), eligible) if return_eligible else (selected, int(cursor))
    count = min(int(sample_count), int(ids.numel()))
    start = int(cursor) % int(ids.numel())
    order = torch.cat((ids[start:], ids[:start]), dim=0)
    chosen = order[:count]
    selected[chosen] = True
    next_cursor = (start + count) % int(ids.numel())
    return (selected, next_cursor, eligible) if return_eligible else (selected, next_cursor)
```

- [x] **Step 5: Add config fields**

In `config.py`, import the dataclasses and add to `MpcPlannerCfg`:

```python
reference_participation: MpcReferenceParticipationCfg = field(default_factory=MpcReferenceParticipationCfg)
```

Add simple task-cfg overrides:

```python
rp = out.reference_participation
_set_if_has(task_cfg, "mpc_reference_participation_enabled", bool, rp, "enabled")
_tuple_ints_if_has(task_cfg, "mpc_reference_include_terrain_cols", rp, "include_terrain_cols")
_tuple_ints_if_has(task_cfg, "mpc_reference_include_terrain_rows", rp, "include_terrain_rows")
```

For names and exclude pairs, first implementation can rely on direct `mpc_planner_cfg.reference_participation` assignment in env cfg to avoid parsing nested tuples from task cfg.

- [x] **Step 6: Run tests**

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py::test_participation_exclude_pair_is_terrain_and_row_logic Go2Pvcnn/tests/test_mpc_rl_participation.py::test_participation_round_robin_wraps_inside_eligible_ids -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/participation.py Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/tests/test_mpc_rl_participation.py
git commit -m "feat: add mpc reference participation selector"
```

---

### Task 4: Integrate Selector Into MPC Manager And Align Horizon/Replan

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/manager.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_mpc_rl_participation.py`

- [x] **Step 1: Write failing manager selection test**

Add a focused unit test with a fake manager subclass or monkeypatch `plan_segment`:

```python
def test_mpc_manager_selects_only_participating_envs(monkeypatch):
    planned_batches = []

    def fake_plan_segment(terrain, states, command, cfg):
        planned_batches.append(int(states.root_pos.shape[0]))
        return _make_simple_mpc_result(batch=int(states.root_pos.shape[0]), horizon=25)

    monkeypatch.setattr("extension.batch_mpc_planner.manager.plan_segment", fake_plan_segment)
    env = _make_fake_mpc_env(num_envs=8, terrain_types=[0, 0, 1, 1, 1, 2, 2, 2], terrain_levels=[0, 1, 7, 7, 3, 7, 2, 1])
    env.cfg.reference_trajectory_horizon = 25
    env.cfg.reference_replan_interval_steps = 25
    env.cfg.mpc_parallel_plan_batch_size = 2
    env.cfg.mpc_planner_cfg.reference_participation.exclude_pairs = (
        MpcTerrainDifficultyPair(terrain_cols=(1,), terrain_rows=(7,)),
    )
    manager = MpcTrajectoryManager(env.cfg, device=torch.device("cpu"))
    cache = manager.refresh_from_env(env)
    assert planned_batches == [2]
    assert cache.root_pos_w.shape[1] == 25
    assert int(manager.reference_reward_mask().sum().item()) == 2
```

If fake env helpers do not exist, implement the minimal fake env in the test file. It must expose `unwrapped`, `scene`, `scene.sensors`, `scene.terrain.terrain_types`, `scene.terrain.terrain_levels`, `scene.terrain.cfg.terrain_generator.sub_terrains`, `command_manager.get_command`, `episode_length_buf`, and robot data buffers.

- [x] **Step 2: Run test and verify failure**

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py::test_mpc_manager_selects_only_participating_envs -q
```

Expected: FAIL because manager still samples globally.

- [x] **Step 3: Add manager cursor and terrain metadata extraction**

In `MpcTrajectoryManager.__init__`:

```python
self._selection_cursor = 0
```

Add helper:

```python
def _terrain_selection_metadata_from_env(self, env):
    root = self._env_root(env)
    terrain = getattr(root.scene, "terrain", None)
    if terrain is None:
        return None, None, None
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_cfg = getattr(terrain, "cfg", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    sub_terrains = getattr(terrain_generator, "sub_terrains", None)
    names = list(sub_terrains.keys()) if isinstance(sub_terrains, dict) else None
    return terrain_types, terrain_levels, names
```

- [x] **Step 4: Replace global sampling with selector**

In `refresh_from_env()`:

```python
if global_due:
    terrain_types, terrain_levels, terrain_names = self._terrain_selection_metadata_from_env(env)
    selected, self._selection_cursor = select_mpc_reference_envs(
        num_envs=num_envs,
        device=self._device,
        terrain_types=terrain_types,
        terrain_levels=terrain_levels,
        terrain_names=terrain_names,
        cfg=cfg.reference_participation,
        sample_count=int(cfg.runtime.parallel_plan_batch_size),
        cursor=self._selection_cursor,
    )
else:
    selected = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
```

- [x] **Step 5: Align env cfg horizon/replan and diagnostics**

In `TeacherElevationTrajectoryMpcSemanticEnvCfg`:

```python
reference_trajectory_horizon: int = 25
reference_replan_interval_steps: int = 25
mpc_parallel_plan_batch_size: int = 64
mpc_diagnostics_emit_runtime_counters: bool = False
mpc_diagnostics_profile_cuda_sync: bool = False
```

In PLAY cfg keep debug counters if useful, but set horizon/replan to 25.

- [x] **Step 6: Run focused tests**

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py::test_mpc_manager_selects_only_participating_envs -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py Go2Pvcnn/tests/test_mpc_rl_participation.py
git commit -m "feat: select mpc reference envs in manager"
```

---

### Task 5: Real Semantic Contact Reward Numeric Core

**Files:**
- Create: `Go2Pvcnn/extension/mdp/semantic_contact_rewards.py`
- Test: `Go2Pvcnn/tests/test_semantic_contact_rewards.py`

- [x] **Step 1: Write numeric tests**

Create `Go2Pvcnn/tests/test_semantic_contact_rewards.py`:

```python
import torch

from extension.mdp.semantic_contact_rewards import filtered_contact_penalty_from_force_matrix


def test_filtered_contact_penalty_threshold_monotonic_clip_and_finite():
    force = torch.zeros((1, 1, 4, 3), dtype=torch.float32)
    force[..., 0] = torch.tensor([0.5, 2.0, 6.0, 100.0]).view(1, 1, 4)
    penalty = filtered_contact_penalty_from_force_matrix(
        force,
        force_threshold=1.0,
        force_scale=5.0,
        force_clip=1.0,
    )
    assert torch.isfinite(penalty).all()
    assert penalty.shape == (1,)
    assert penalty.item() == 1.0


def test_filtered_contact_penalty_zero_below_threshold():
    force = torch.zeros((2, 1, 3, 3), dtype=torch.float32)
    force[..., 0] = 0.5
    penalty = filtered_contact_penalty_from_force_matrix(
        force,
        force_threshold=1.0,
        force_scale=5.0,
        force_clip=1.0,
    )
    torch.testing.assert_close(penalty, torch.zeros(2))
```

- [x] **Step 2: Run tests and verify failure**

```bash
pytest Go2Pvcnn/tests/test_semantic_contact_rewards.py -q
```

Expected: FAIL because module does not exist.

- [x] **Step 3: Implement numeric helper**

Create:

```python
from __future__ import annotations

import torch
from torch import Tensor


def filtered_contact_penalty_from_force_matrix(
    force_matrix_w: Tensor,
    *,
    force_threshold: float,
    force_scale: float,
    force_clip: float,
) -> Tensor:
    force = torch.as_tensor(force_matrix_w, dtype=torch.float32)
    if force.ndim != 4 or int(force.shape[-1]) != 3:
        raise ValueError(f"force_matrix_w must have shape [N,B,F,3], got {tuple(force.shape)}")
    per_filter = torch.linalg.vector_norm(force, dim=-1)
    total_force = per_filter.sum(dim=(1, 2))
    scaled = torch.relu(total_force - float(force_threshold)) / max(float(force_scale), 1.0e-6)
    return scaled.clamp(0.0, float(force_clip))
```

- [x] **Step 4: Implement reward function**

Add:

```python
def _scene_sensor(env, name: str):
    sensors = getattr(env.scene, "sensors", None)
    if sensors is not None:
        try:
            return sensors[name]
        except Exception:
            return getattr(sensors, name)
    return env.scene[name]


def semantic_filtered_contact_collision_reward(
    env,
    small_sensor_names: tuple[str, ...],
    large_sensor_names: tuple[str, ...],
    body_weights: tuple[float, ...],
    force_threshold: float = 1.0,
    force_scale: float = 50.0,
    force_clip: float = 1.0,
    small_weight: float = 1.0,
    large_weight: float = 2.0,
) -> torch.Tensor:
    device = torch.device(getattr(env, "device", "cpu"))
    out = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    weights = torch.as_tensor(body_weights, dtype=torch.float32, device=device)
    if len(small_sensor_names) != int(weights.numel()) or len(large_sensor_names) != int(weights.numel()):
        raise ValueError("sensor name counts must match body_weights")
    for idx, name in enumerate(small_sensor_names):
        sensor = _scene_sensor(env, name)
        matrix = torch.as_tensor(sensor.data.force_matrix_w, dtype=torch.float32, device=device)
        out = out + weights[idx] * float(small_weight) * filtered_contact_penalty_from_force_matrix(
            matrix,
            force_threshold=force_threshold,
            force_scale=force_scale,
            force_clip=force_clip,
        ).to(device=device)
    for idx, name in enumerate(large_sensor_names):
        sensor = _scene_sensor(env, name)
        matrix = torch.as_tensor(sensor.data.force_matrix_w, dtype=torch.float32, device=device)
        out = out + weights[idx] * float(large_weight) * filtered_contact_penalty_from_force_matrix(
            matrix,
            force_threshold=force_threshold,
            force_scale=force_scale,
            force_clip=force_clip,
        ).to(device=device)
    return -out
```

- [x] **Step 5: Run numeric tests**

```bash
pytest Go2Pvcnn/tests/test_semantic_contact_rewards.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/mdp/semantic_contact_rewards.py Go2Pvcnn/tests/test_semantic_contact_rewards.py
git commit -m "feat: add semantic filtered contact reward"
```

---

### Task 6: Env CFG Adds Per-Body Filtered Contact Sensors

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`

- [x] **Step 1: Write cfg tests**

Create `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`:

```python
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    SEMANTIC_CONTACT_BODY_NAMES,
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
)
from extension.semantic_course import SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT


def test_mpc_semantic_cfg_has_one_body_filtered_contact_sensors():
    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
    for body in SEMANTIC_CONTACT_BODY_NAMES:
        small = getattr(cfg.scene, f"semantic_contact_{body}_small")
        large = getattr(cfg.scene, f"semantic_contact_{body}_large")
        assert small.prim_path == f"{{ENV_REGEX_NS}}/Robot/{body}"
        assert large.prim_path == f"{{ENV_REGEX_NS}}/Robot/{body}"
        assert small.filter_prim_paths_expr == [f"{SEMANTIC_COURSE_SMALL_ROOT}/.*"]
        assert large.filter_prim_paths_expr == [f"{SEMANTIC_COURSE_LARGE_ROOT}/.*"]
        assert small.update_period == 0.0
        assert large.update_period == 0.0
```

- [x] **Step 2: Run test and verify failure**

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py::test_mpc_semantic_cfg_has_one_body_filtered_contact_sensors -q
```

Expected: FAIL because sensors do not exist.

- [x] **Step 3: Add sensor constants and imports**

In env cfg:

```python
from isaaclab.sensors import ContactSensorCfg, patterns
from extension.mdp.semantic_contact_rewards import semantic_filtered_contact_collision_reward

SEMANTIC_CONTACT_BODY_NAMES = (
    "FL_foot", "FR_foot", "RL_foot", "RR_foot",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "base",
)
SEMANTIC_CONTACT_BODY_WEIGHTS = (
    1.0, 1.0, 1.0, 1.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    5.0,
)
```

- [x] **Step 4: Add one-body filtered sensors**

Inside `TeacherElevationTrajectoryMpcSemanticSceneCfg`, add one attribute per body/class. Use explicit attributes rather than a dict because IsaacLab configclass discovers class attributes.

Example pattern:

```python
semantic_contact_FL_foot_small = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
    update_period=0.0,
    history_length=0,
    track_air_time=False,
    debug_vis=False,
    filter_prim_paths_expr=[f"{SEMANTIC_COURSE_SMALL_ROOT}/.*"],
)
semantic_contact_FL_foot_large = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
    update_period=0.0,
    history_length=0,
    track_air_time=False,
    debug_vis=False,
    filter_prim_paths_expr=[f"{SEMANTIC_COURSE_LARGE_ROOT}/.*"],
)
```

Repeat for every body in `SEMANTIC_CONTACT_BODY_NAMES`.

- [x] **Step 5: Replace reward term**

In `TeacherElevationTrajectoryMpcSemanticRewardsCfg`, replace `swing_leg_collision` with:

```python
semantic_contact_collision = RewTerm(
    func=semantic_filtered_contact_collision_reward,
    weight=1.0,
    params={
        "small_sensor_names": tuple(f"semantic_contact_{body}_small" for body in SEMANTIC_CONTACT_BODY_NAMES),
        "large_sensor_names": tuple(f"semantic_contact_{body}_large" for body in SEMANTIC_CONTACT_BODY_NAMES),
        "body_weights": SEMANTIC_CONTACT_BODY_WEIGHTS,
        "force_threshold": 1.0,
        "force_scale": 50.0,
        "force_clip": 1.0,
        "small_weight": 1.0,
        "large_weight": 2.0,
    },
)
swing_leg_collision = None
```

- [x] **Step 6: Run cfg tests**

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py
git commit -m "feat: add semantic contact sensors to mpc rl cfg"
```

---

### Task 7: IsaacLab Contact Sensor Smoke In `env_isaacsim`

**Files:**
- Create: `Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py`
- Log: `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-contact-smoke.md`

- [x] **Step 1: Write real IsaacLab smoke test**

Create a test that starts the MPC semantic env with a small number of envs and validates sensor shapes:

```python
def test_mpc_semantic_contact_sensors_real_isaaclab():
    import gymnasium as gym
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        SEMANTIC_CONTACT_BODY_NAMES,
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
    )

    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
    cfg.scene.num_envs = 4
    env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
    try:
        env.reset()
        root = env.unwrapped
        for body in SEMANTIC_CONTACT_BODY_NAMES:
            for suffix in ("small", "large"):
                sensor = root.scene.sensors[f"semantic_contact_{body}_{suffix}"]
                matrix = sensor.data.force_matrix_w
                assert matrix.shape[0] == 4
                assert matrix.shape[1] == 1
                assert matrix.shape[-1] == 3
    finally:
        env.close()
```

- [x] **Step 2: Run smoke in requested env**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py::test_mpc_semantic_contact_sensors_real_isaaclab -q
```

Expected: PASS. If it fails because `filter_count == 0`, inspect whether `"/World/semantic_course/small/.*"` needs an explicit prim list from `semantic_course.py`.

- [x] **Step 3: Write log**

Create `notes/log/YYYY-MM-DD-HHMM-t302l-semantic-contact-smoke.md` with:

```markdown
# T302l Semantic Contact Sensor Smoke

- Purpose: Validate per-body filtered contact sensors in real IsaacLab.
- Env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
- Command: `CUDA_VISIBLE_DEVICES=0 ...`
- num_envs: 4
- Result: PASS
- Key Metrics:
  - sensor_count: 26
  - force_matrix_shape: [4, 1, filter_count, 3]
- Follow-up: none if PASS.
```

- [x] **Step 4: Commit**

```bash
git add Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py notes/log/YYYY-MM-DD-HHMM-t302l-semantic-contact-smoke.md
git commit -m "test: validate semantic contact sensors in isaaclab"
```

---

### Task 8: RL Performance Acceptance 1024 Env / 64 MPC Env / <=10s Epoch

**Files:**
- Create: `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py`
- Log: `notes/log/YYYY-MM-DD-HHMM-t302l-rl-1024-64-performance.md`

- [ ] **Step 1: Add probe script**

Create a script that launches the real training env, attaches the manager, runs one short runner epoch/iteration or a fixed rollout matching the training loop, and prints JSON:

```python
def main():
    import json
    import time
    import gymnasium as gym
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import TeacherElevationTrajectoryMpcSemanticEnvCfg
    from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
    cfg.scene.num_envs = 1024
    cfg.mpc_parallel_plan_batch_size = 64
    cfg.reference_trajectory_horizon = 25
    cfg.reference_replan_interval_steps = 25
    cfg.mpc_diagnostics_emit_runtime_counters = False
    cfg.mpc_diagnostics_profile_cuda_sync = False
    env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
    try:
        root = env.unwrapped
        attach_trajectory_manager_if_enabled(root, cfg, experiment_name="teacher_elevation_trajectory_mpc_semantic", device=root.device)
        obs, _ = env.reset()
        actions = root.action_manager.action_term_dim
        start = time.perf_counter()
        for _ in range(25):
            zero = root.action_manager.action_term_dim
            action = root.action_space.sample()
            env.step(action)
        elapsed = time.perf_counter() - start
        print(json.dumps({"num_envs": 1024, "selected_mpc_envs": 64, "epoch_seconds": elapsed}))
    finally:
        env.close()
```

If RSL-RL runner one-iteration timing is available without long startup overhead, prefer the real runner. If not, this probe is the minimum runtime proxy and the final task must still run a real `train.py --max_iterations 1` command.

- [ ] **Step 2: Run performance probe**

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
```

Expected JSON includes:

```json
{"num_envs":1024,"selected_mpc_envs":64,"epoch_seconds":9.99}
```

Acceptance: `epoch_seconds <= 10.0`.

- [ ] **Step 3: Run real train entry one iteration**

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py --experiment teacher_elevation_trajectory_mpc_semantic --num_envs 1024 --max_iterations 1 --headless --device cuda:0
```

Expected: exits 0 and logs one iteration timing. If CLI does not accept `--device cuda:0`, use AppLauncher's supported device flag from `--help` and record the exact command.

- [ ] **Step 4: Write performance log**

Create `notes/log/YYYY-MM-DD-HHMM-t302l-rl-1024-64-performance.md`:

```markdown
# T302l RL 1024 Env / 64 MPC Env Performance

- Purpose: Validate training speed target.
- Env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- num_envs: 1024
- selected_mpc_envs: 64
- epoch_seconds: `<measured>`
- Acceptance: `epoch_seconds <= 10.0`
- Result: PASS or FAIL
- Follow-up: if FAIL, do not change loss semantics; profile selected-env planning, contact sensor count, and reward time first.
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py notes/log/YYYY-MM-DD-HHMM-t302l-rl-1024-64-performance.md
git commit -m "test: add mpc rl performance acceptance probe"
```

---

### Task 9: Low-Small Regression And Final Notes Alignment

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302l-mpc-rl-participation-and-reward-plan.md`
- Modify: `notes/log/index.md`
- Log: `notes/log/YYYY-MM-DD-HHMM-t302l-final-verification.md`

- [ ] **Step 1: Run local focused tests**

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_semantic_contact_rewards.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py -q
```

Expected: PASS.

- [ ] **Step 2: Run backend regression subset**

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: PASS.

- [ ] **Step 3: Run low-small regression in env_isaacsim**

Run the existing low-small full matrix / FK semantic collision probe used by the 2026-05-28 design. Use the same command shape recorded in [../log/2026-05-28-2259-t302k-low-small-full-matrix-and-fk-inner-loop.md](../log/2026-05-28-2259-t302k-low-small-full-matrix-and-fk-inner-loop.md), with:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python ...
```

Expected:

```text
crossing-covered rows: no FK semantic collision regression
max crossing FK error <= previous accepted 0.08m gate
```

- [ ] **Step 4: Write final verification log**

Create `notes/log/YYYY-MM-DD-HHMM-t302l-final-verification.md`:

```markdown
# T302l MPC RL Participation Final Verification

- Purpose: Final verification for MPC RL participation and reward redesign.
- Design: `docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html`
- Env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
- Commands:
  - focused pytest: PASS
  - backend pytest: PASS
  - IsaacLab contact smoke: PASS
  - RL 1024/64 performance: PASS or FAIL
  - low-small regression: PASS
- Key Metrics:
  - selected_mpc_envs: 64
  - epoch_seconds: `<measured>`
  - small_contact_count / large_contact_count separation: verified
  - FK semantic collisions on crossing-covered rows: 0
- Follow-up:
  - none if all PASS
```

- [ ] **Step 5: Update todo dashboard and log index**

Update `notes/todo.md`:

```markdown
- Current focus: **T302l MPC RL participation and reward integration**.
- Active branch page: [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md).
```

Add recent log row to `notes/log/index.md`.

- [ ] **Step 6: Commit**

```bash
git add notes/todo.md notes/log/index.md notes/todo/T302l-mpc-rl-participation-and-reward-plan.md notes/log/YYYY-MM-DD-HHMM-t302l-final-verification.md
git commit -m "docs: record mpc rl final verification"
```

---

## Self-Review

- Spec coverage:
  - 25 horizon/replan alignment: Task 4.
  - env participation selector with AND exclusion: Task 3 and Task 4.
  - round-robin selection: Task 3.
  - selected env reward mask: Task 4 and Task 2.
  - world-frame foot tracking: Task 1 and Task 2.
  - real IsaacLab semantic contact reward: Task 5 and Task 6.
  - per-body filtered ContactSensor for global semantic objects: Task 6 and Task 7.
  - small/large collision separation: Task 5, Task 6, Task 9.
  - numeric force implementation: Task 5.
  - 1024 env / 64 MPC env / <=10s epoch: Task 8.
  - env_isaacsim verification: Task 7, Task 8, Task 9.
  - low-small no-regression: Task 9.
- Placeholder scan: no placeholder markers, no unspecified tests, no missing command shapes.
- Type consistency:
  - `foot_pos_w` is introduced in Task 1 before reward consumption in Task 2.
  - `MpcReferenceParticipationCfg` is introduced in Task 3 before manager consumption in Task 4.
  - `semantic_filtered_contact_collision_reward` is introduced in Task 5 before env cfg consumption in Task 6.
