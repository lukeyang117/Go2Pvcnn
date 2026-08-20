# Parallelism Mixed Terrain Root Height Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 让 flat 和 flat_dense_small_obstacles 使用平地基准面加 0.30m 的固定 root target，并统一 large train、play、distillation、viewer 的 terrain 比例与语义障碍物数量。

**Architecture:** 保留复杂地形的 terrain-following root 逻辑，只在 flat terrain mask 为 false 的分支中使用固定 root target。reference manager 和 viewer 统一把两个 flat 名称识别为 flat；small obstacle 高程继续传给 touchdown、swing、IK 和 geometry collision。

**Tech Stack:** Python 3.10, PyTorch, IsaacLab config classes, pytest.

## Global Constraints

- flat 比例为 0.1，flat_dense_small_obstacles 比例为 0.1。
- random_rough、hf_pyramid_slope、hf_pyramid_slope_inv、boxes 比例为 0.1。
- pyramid_stairs、pyramid_stairs_inv 比例为 0.2。
- flat 的 semantic obstacle 数量为 small=0, large=2。
- flat_dense_small_obstacles 的 semantic obstacle 数量为 small=40, large=0。
- 其他 terrain 的 semantic obstacle 数量为 small=5, large=2。
- 仅排除 flat_dense_small_obstacles 的 terrain curriculum。
- flat root target 为 flat_base_z_m + flat_root_clearance_m = 0.0 + 0.30 = 0.30m。
- reference 第 0 帧必须保留 IsaacLab 当前真实 root 状态。
- 不修改 touchdown、swing clearance、IK、geometry collision、RL observation、reward、termination。
- 不覆盖工作区中未相关的训练脚本改动。

---

### Task 1: Lock behavior with tests

**Files:**
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_terrain_following_root.py
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py

**Interfaces:**
- rollout_root must return the configured flat target height for flat rows.
- ParallelismReferenceManager._terrain_following_mask must return False for both flat terrain names.
- The large config must declare the exact proportions and obstacle overrides.

- [x] Step 1: Change the flat root test to expect a fixed target.

~~~python
def test_flat_mask_uses_fixed_root_height_independent_of_obstacle_height():
    height = torch.full((1, 151, 151), 0.20)
    terrain = _terrain_from_height(height)
    cfg = ParallelismCfg(flat_base_z_m=0.0, flat_root_clearance_m=0.30)
    root = rollout_root(
        _state(root_z=0.42),
        torch.zeros(1, 3),
        terrain,
        cfg,
        terrain_following_mask=torch.tensor([False]),
    )
    assert torch.allclose(root.root_pos_w[:, 0, 2], torch.tensor([0.42]))
    assert torch.allclose(root.root_pos_w[:, -1, 2], torch.tensor([0.30]))
~~~

- [x] Step 2: Add a configurable target test using flat_base_z_m=0.07 and flat_root_clearance_m=0.30, expecting frame 0 to remain 0.42m and the final frame to be 0.37m.
- [x] Step 3: Add the dense-flat mask case with names flat_dense_small_obstacles, flat, random_rough and expected mask [False, False, True].
- [x] Step 4: Add source assertions for all eight proportions, the three obstacle count rules, and exclusion of only flat_dense_small_obstacles.
- [x] Step 5: Run the focused pytest command and verify RED because the current implementation still queries terrain height and recognizes only flat.

Run:

~~~bash
pytest -q Go2Pvcnn/tests/tracking/test_parallelism_terrain_following_root.py \
  Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py \
  Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py
~~~

---

### Task 2: Implement fixed flat root target and terrain configuration

**Files:**
- Modify: Go2Pvcnn/extension/parallelism/config.py
- Modify: Go2Pvcnn/extension/parallelism/root.py
- Modify: Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py

**Interfaces:**
- Add ParallelismCfg.flat_base_z_m: float = 0.0.
- Add ParallelismCfg.flat_root_clearance_m: float = 0.30.
- Expose ParallelismCfg.flat_root_z_target_m as flat_base_z_m + flat_root_clearance_m.
- _flat_root_z must not call query_height_semantic_valid.

- [x] Step 1: Add the configuration fields and computed target.

~~~python
flat_base_z_m: float = 0.0
flat_root_clearance_m: float = 0.30

@property
def flat_root_z_target_m(self) -> float:
    return float(self.flat_base_z_m) + float(self.flat_root_clearance_m)
~~~

- [x] Step 2: Replace _flat_root_z with a smooth trajectory from the measured frame 0 to the fixed target.

~~~python
def _flat_root_z(state, root0, rpy0, terrain, cfg):
    del state, rpy0, terrain
    target = root0.new_tensor(float(cfg.flat_root_z_target_m))
    frame = torch.arange(int(cfg.horizon), dtype=root0.dtype, device=root0.device)
    u = (frame / float(max(int(cfg.root_leveling_frames), 1))).clamp(0.0, 1.0)
    smoothstep = u * u * (3.0 - 2.0 * u)
    return root0[:, None, 2] + smoothstep[None, :] * (target - root0[:, None, 2])
~~~

- [x] Step 3: Set terrain proportions to 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2 in the terrain generator.
- [x] Step 4: Set explicit semantic overrides:

~~~python
plane_counts=(SemanticObstacleCount(small=0, large=2),)
non_plane_counts=(SemanticObstacleCount(small=5, large=2),)
terrain_obstacle_count_overrides={
    "flat": SemanticObstacleCount(small=0, large=2),
    "flat_dense_small_obstacles": SemanticObstacleCount(small=40, large=0),
}
~~~

- [x] Step 5: Run Task 1 tests and verify the root and configuration behavior is GREEN.

---

### Task 3: Align manager, fallback, and viewer classification

**Files:**
- Modify: Go2Pvcnn/tracking/managers/parallelism_reference_manager.py
- Modify: Go2Pvcnn/extension/viz/go2_foostep_planner.py
- Modify: Go2Pvcnn/tests/parallelism/test_viewer_adapter.py
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py

**Interfaces:**
- Manager flat names are exactly {"flat", "flat_dense_small_obstacles"}.
- Viewer selection returns False for both flat names.
- Standstill fallback uses the fixed flat target for flat-class terrain and support-height logic for non-flat terrain.

- [x] Step 1: Update manager terrain classification to use flat_names = {"flat", "flat_dense_small_obstacles"}.
- [x] Step 2: Make _standard_stand_state select cfg.flat_root_z_target_m for flat-class environments and preserve current support-height calculation for non-flat environments.
- [x] Step 3: Update viewer selection so follow = terrain_name.lower() not in {"flat", "flat_dense_small_obstacles"}.
- [x] Step 4: Add viewer and fallback regression tests.
- [x] Step 5: Run:

~~~bash
pytest -q Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py \
  Go2Pvcnn/tests/parallelism/test_viewer_adapter.py
~~~

---

### Task 4: Full verification and focused commit

**Files:** Only files listed in Tasks 1-3.

- [x] Step 1: Run:

~~~bash
pytest -q \
  Go2Pvcnn/tests/tracking/test_parallelism_terrain_following_root.py \
  Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py \
  Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py \
  Go2Pvcnn/tests/parallelism/test_viewer_adapter.py
~~~

- [x] Step 2: Run syntax compilation:

~~~bash
python -m compileall -q \
  Go2Pvcnn/extension/parallelism \
  Go2Pvcnn/tracking \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py
~~~

- [x] Step 3: Run git diff --check and confirm unrelated training scripts remain unstaged.
- [x] Step 4: Commit only the implementation:

~~~bash
git add Go2Pvcnn/extension/parallelism/config.py \
  Go2Pvcnn/extension/parallelism/root.py \
  Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py \
  Go2Pvcnn/tracking/managers/parallelism_reference_manager.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/tests/tracking/test_parallelism_terrain_following_root.py \
  Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py \
  Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py \
  Go2Pvcnn/tests/parallelism/test_viewer_adapter.py
git commit -m "fix: keep flat obstacle root reference height fixed"
~~~
