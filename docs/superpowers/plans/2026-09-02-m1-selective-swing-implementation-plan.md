# M1 Selective Swing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 M1 在保持 Go2 默认规划行为不变的前提下，按每条腿最终选中的候选轨迹决定 rolling 或 swing，并使用 M1 轮子几何生成足够的越障 clearance。

**Architecture:** 在现有通用 planner 中增加候选级 `needs_swing[B,4,50]` 元数据。候选仍先经过现有 IK、落点和碰撞验证，评分选出 `selected_index[B,4]` 后才读取对应标记；M1 对 selected=false 的腿使用固定腿角随 Root 滚动，selected=true 的腿使用 terrain-aware swing。Go2 不设置该元数据，继续走原有全腿摆动路径。

**Tech Stack:** Python 3.10, PyTorch, pytest, Isaac Lab / Isaac Sim viewer。

## Global Constraints

- 每条腿固定生成 `50` 个候选点。
- semantic 轨迹检查使用当前轮心到候选点的整条 XY 路径，而不是只检查端点。
- 只有最终 selected candidate 的 `needs_swing` 可以触发该腿 swing。
- 小障碍可以触发 swing；大障碍继续使用现有绕行逻辑。
- M1 规划只输出 12 个 ABAD/HIP/KNEE 关节；4 个轮轴关节保持当前值，不输入、不输出、不评分。
- M1 swing 高度使用局部高程、轮子垂向包络和额外 clearance；轮半径为 `0.095958 m`，碰撞余量为 `0.003 m`。
- 不生成 BASE 碰撞体，不新增 link 间 self-collision。
- 未传 `--robot` 或传入 `--robot go2` 时，Go2 FK/IK、配置和现有执行路径保持不变。
- Viewer 只做临时参数 override，不把 slider 状态写回默认配置。

## File Map

- Modify: `Go2Pvcnn/extension/parallelism/types.py` - 为 planner diagnostics 增加候选级 swing 元数据。
- Modify: `Go2Pvcnn/extension/parallelism/robot_backend.py` - 声明每个 backend 的小障碍 swing semantic 与 terrain-following swing 能力。
- Modify: `Go2Pvcnn/extension/parallelism/planner.py` - 计算每候选轨迹的 swing 标记，并在 selected 后逐腿组装 rolling/swing 目标。
- Modify: `Go2Pvcnn/extension/parallelism/swing.py` - 支持 M1 的 wheel-envelope clearance 输入，保持原 Go2 调用兼容。
- Modify: `Go2Pvcnn/extension/parallelism/m1_kinematics.py` - 固化 M1 swing clearance、轮子水平投影包络和 M1 默认配置。
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py` - 在已有 Parallelism viewer diagnostics 中显示 selected swing 状态。
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py` - 通用候选级标记和逐腿执行测试。
- Test: `Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py` - M1 小障碍、平地和轮轴保持测试。
- Test: `Go2Pvcnn/tests/parallelism/test_m1_robot_backend.py` - M1 clearance 与 wheel envelope 测试。
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py` - 新 diagnostics 字段的构造和显示兼容性。

---

### Task 1: 增加候选级 swing contract

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/types.py`
- Modify: `Go2Pvcnn/extension/parallelism/robot_backend.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`

**Interfaces:**
- `RobotBackend.swing_semantic_ids: tuple[int, ...]`：当前 M1 为 `(1,)`，Go2 为空；large obstacle semantic `2` 不触发普通 swing。
- `ParallelismDiagnostics.candidate_needs_swing: Tensor`：shape `[B,4,C]`，与 `candidate_valid` 对齐。
- `ParallelismDiagnostics.selected_needs_swing: Tensor`：shape `[B,4]`，由 `candidate_needs_swing` 按 `selected_index` gather 得到。

- [x] **Step 1: Write the failing test**

```python
def test_candidate_swing_metadata_is_per_leg_and_per_candidate():
    result = plan_trajectory(
        _state(), torch.tensor([[0.2, 0.0, 0.0]]), _terrain(), ParallelismCfg()
    )
    assert result.diagnostics.candidate_needs_swing.shape == (1, 4, 50)
    assert result.diagnostics.selected_needs_swing.shape == (1, 4)
    assert torch.equal(
        result.diagnostics.selected_needs_swing,
        result.diagnostics.candidate_needs_swing.gather(
            -1, result.diagnostics.selected_index.unsqueeze(-1)
        ).squeeze(-1),
    )
```

- [x] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_planner.py::test_candidate_swing_metadata_is_per_leg_and_per_candidate`

Expected: FAIL because the diagnostics contract has no candidate swing fields.

- [x] **Step 3: Write the minimal implementation**

Add the two tensors to `ParallelismDiagnostics`, add `swing_semantic_ids` to `RobotBackend`, set Go2 to `()` and M1 to `(1,)`, and populate zero-shaped values in `_hold_trajectory`. In the normal planner path gather `selected_needs_swing` from the selected candidate index.

- [x] **Step 4: Run the test to verify it passes**

Run the same pytest command. Expected: PASS for the new metadata assertion; the four known zero-command baseline failures remain isolated until the compatibility check in Task 3.

- [x] **Step 5: Run the existing constructor compatibility tests**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

Expected: PASS. Existing test fixtures must remain constructible; new fields should have defaults only if that is required for old fixture compatibility.

### Task 2: Compute semantic swing requirement over each candidate path

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`

**Interfaces:**
- Add `_candidate_needs_swing(state, candidates, terrain, cfg, backend) -> Tensor` with shape `[B,4,C]`.
- The path is sampled at `cfg.half_cycle` points from current wheel center XY to each candidate XY.
- M1 sets true when any path sample has semantic ID `1`; terrain-following M1 batches set true for every candidate. Go2 returns all false and does not change its execution path.

The tests use these concrete local helpers in `test_planner.py`:

```python
def _m1_state():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    return ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )

def _m1_backend():
    from extension.parallelism.robot_backend import get_robot_backend
    return get_robot_backend("m1")

def _semantic_terrain_id(semantic_id):
    return _semantic_terrain(batch=1, semantic_id=semantic_id)
```

- [x] **Step 1: Write the failing tests**

```python
def test_m1_only_marks_candidates_whose_xy_path_hits_small_obstacle():
    terrain = _semantic_terrain_id(1)
    result = plan_trajectory(
        _m1_state(), torch.tensor([[0.5, 0.0, 0.0]]), terrain,
        M1_CFG, robot_backend=_m1_backend(),
    )
    marked = result.diagnostics.candidate_needs_swing
    assert marked.shape == (1, 4, 50)
    assert int(marked.sum()) > 0
    assert int(marked.sum()) < 200

def test_large_obstacle_does_not_set_normal_swing_flag():
    terrain = _semantic_terrain_id(2)
    result = plan_trajectory(
        _m1_state(), torch.tensor([[0.5, 0.0, 0.0]]), terrain,
        M1_CFG, robot_backend=_m1_backend(),
    )
    assert not bool(result.diagnostics.candidate_needs_swing.any())
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_planner.py -k 'candidate.*swing or large.*obstacle'`

Expected: FAIL because no path-level semantic metadata is produced.

- [x] **Step 3: Implement the path query**

Use vectorized interpolation and `query_height_semantic_valid`:

```python
tau = torch.linspace(
    0.0, 1.0, int(cfg.half_cycle), dtype=current_foot.dtype, device=current_foot.device
)
path_xy = current_foot[:, :, None, None, :2] * (1.0 - tau) \
    + candidates.candidate_w[:, :, :, None, :2] * tau
path_query = query_height_semantic_valid(
    terrain, path_xy.reshape(batch, leg_count * candidate_count * int(cfg.half_cycle), 2)
)
path_semantic = path_query.semantic.reshape(batch, leg_count, candidate_count, -1)
semantic_hit = (path_semantic.unsqueeze(-1) == ids).any(dim=(-1, -2))
```

Use only `backend.swing_semantic_ids` for the semantic branch. Do not use `cfg.obstacle_semantic_ids` here because it includes large obstacles.

- [x] **Step 4: Add terrain-following behavior**

For M1 only, broadcast `terrain_following_mask[B]` to `[B,4,C]` and OR it with the small-obstacle path result. Keep the mask false for flat and flatDense; a small obstacle can still set only the affected candidate true.

- [x] **Step 5: Run the focused tests**

Run the two test names from Step 1. Expected: PASS and no Go2 test behavior changes.

### Task 3: Execute only selected M1 legs as swing

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Modify: `Go2Pvcnn/extension/parallelism/swing.py`
- Test: `Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`

**Interfaces:**
- Extend `_assemble_foot_targets(state, root_pos, root_rpy, selected_foothold_w, terrain, cfg, robot_backend=None, leg_swing=None)` where `leg_swing` has shape `[B,4]`; `None` preserves existing Go2 behavior.
- For an M1 leg with `leg_swing=False`, target the wheel center obtained by FK using the current 12 DOF joint vector under each planned Root pose.
- For `leg_swing=True`, retain the existing diagonal half-cycle swing and selected touchdown behavior.

- [x] **Step 1: Write the failing selective-execution tests**

```python
def test_m1_selected_non_swing_leg_rolls_without_z_lift():
    state = _m1_state()
    result = plan_trajectory(
        state, torch.tensor([[0.5, 0.0, 0.0]]), _flat_terrain(), M1_CFG,
        robot_backend=_m1_backend(),
    )
    selected = result.diagnostics.selected_needs_swing[0]
    assert bool((~selected).any())
    for leg in torch.where(~selected)[0].tolist():
        torch.testing.assert_close(
            result.joint_pos[0, :, leg * 3:(leg + 1) * 3],
            state.joint_pos[0, leg * 3:(leg + 1) * 3].expand(24, 3),
            atol=2.0e-3, rtol=0.0,
        )

def test_unselected_candidate_cannot_trigger_another_leg_swing():
    from extension.parallelism.planner import _assemble_foot_targets
    backend = _m1_backend()
    state = _m1_state()
    root_pos = state.root_pos_w[:, None].expand(-1, 24, -1)
    root_rpy = state.root_rpy_w[:, None].expand(-1, 24, -1)
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    selected_foothold = current_foot.clone()
    selected_foothold[..., 0] += 0.10
    targets = _assemble_foot_targets(
        state, root_pos, root_rpy, selected_foothold, _flat_terrain(),
        M1_CFG, robot_backend=backend,
        leg_swing=torch.tensor([[False, True, False, True]]),
    )
    rolling = backend.fk(root_pos.reshape(24, 3), root_rpy.reshape(24, 3), state.joint_pos.expand(24, 12)).foot_pos_w
    assert torch.allclose(targets[:, :, 0], rolling[:, :, 0], atol=2.0e-3)
    assert torch.allclose(targets[:, :, 2], rolling[:, :, 2], atol=2.0e-3)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py -k 'selected.*swing or non_swing'`

Expected: FAIL because `_assemble_foot_targets` currently swings both diagonal pairs unconditionally.

- [x] **Step 3: Build the rolling target**

Compute `rolling_foot_pos = backend.fk(root_pos.reshape(batch * horizon, 3), root_rpy.reshape(batch * horizon, 3), current_joint[:, None].expand(-1, horizon, -1).reshape(batch * horizon, 12)).foot_pos_w`, reshape to `[B,H,4,3]`, and initialize all target feet from this tensor for M1. This keeps a non-swing leg attached to the Root-relative rolling trajectory.

- [x] **Step 4: Overlay swing trajectories only for selected swing legs**

Construct the two existing diagonal swing curves, then use `torch.where(leg_swing[..., None, None], swing_target, rolling_target)` for each leg. For a swing leg, the stance half remains the selected world touchdown; for a rolling leg, every frame remains the rolling FK target.

- [x] **Step 5: Pass selected flags after candidate selection**

In `plan_trajectory`, gather `selected_needs_swing`, pass it to `_assemble_foot_targets` only for the M1 backend, and keep `None` for Go2. Store both candidate and selected values in diagnostics.

- [x] **Step 6: Run focused and regression tests**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py Go2Pvcnn/tests/parallelism/test_planner.py -k 'm1 or assemble or swing or flat'`

Expected: the new selective tests pass; any four baseline zero-command failures must be reported separately unless their expectations are updated in this task with evidence that the current hold contract is intentional.

### Task 4: Apply M1 wheel-envelope clearance and keep Go2 swing unchanged

**Files:**

- Modify: `Go2Pvcnn/extension/parallelism/m1_kinematics.py`
- Modify: `Go2Pvcnn/extension/parallelism/swing.py`
- Test: `Go2Pvcnn/tests/parallelism/test_m1_robot_backend.py`
- Test: `Go2Pvcnn/tests/parallelism/test_swing.py`

**Interfaces:**
- Define `M1_SWING_CLEARANCE_M = 0.105` as the initial effective M1 clearance, derived from wheel radius, collision margin and a small extra safety margin.
- Keep `terrain_aware_swing_curve` signature backward compatible for Go2; M1 uses its configured effective clearance.

- [x] **Step 1: Write the failing geometry tests**

```python
def test_m1_clearance_contains_wheel_radius_and_margin():
    from extension.parallelism.m1_kinematics import M1_CFG, M1_WHEEL_RADIUS_M
    assert M1_CFG.swing_clearance_m >= M1_WHEEL_RADIUS_M + M1_CFG.collision_margin_m

def test_m1_small_obstacle_swing_apex_clears_wheel_envelope():
    terrain = _terrain_with_mid_obstacle()
    start = torch.tensor([[[0.0, 0.0, 0.0]]])
    touchdown = torch.tensor([[[1.0, 0.0, 0.0]]])
    swing = terrain_aware_swing_curve(
        start, touchdown, terrain, frames=11,
        clearance_m=M1_CFG.swing_clearance_m, min_apex_m=0.0,
    )
    assert float(swing[..., 2].max()) >= 0.08 + M1_WHEEL_RADIUS_M + M1_CFG.collision_margin_m
```

- [x] **Step 2: Run the tests to verify the current bug**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_m1_robot_backend.py -k 'clearance or apex'`

Expected: FAIL with the current M1 `swing_clearance_m=0.0`.

- [x] **Step 3: Set the M1 effective clearance**

Set the M1 config to the named constant, retain `foot_contact_offset_m=M1_WHEEL_RADIUS_M`, and leave Go2 `ParallelismCfg` values unchanged.

- [x] **Step 4: Verify wheel horizontal projection is documented and applied**

Add a named M1 projection radius of `M1_WHEEL_RADIUS_M + 0.04650 / 2.0` and use it when the local terrain maximum is queried for swing. Do not change candidate disk radius unless a focused test proves the candidate search is the failure source.

- [x] **Step 5: Run M1 and Go2 swing tests**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_m1_robot_backend.py Go2Pvcnn/tests/parallelism/test_swing.py`

Expected: M1 clearance tests and existing Go2 curve tests pass.

### Task 5: Viewer diagnostics and M1 smoke validation

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`
- Test: `Go2Pvcnn/tests/test_viewer_reset.py`

**Interfaces:**
- Extend the Parallelism diagnostic formatter to tolerate and display `selected_needs_swing` when present.
- Existing viewers and fixtures without the new field must continue to render diagnostics without an exception.

- [x] **Step 1: Write the failing viewer test**

```python
def test_parallelism_diagnostics_reports_selected_swing_state():
    text = _format_parallelism_reject_diagnostics(result_with_selected_swing())
    assert "selected_swing=" in text
```

- [x] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -k selected_swing`

Expected: FAIL because the formatter does not report the new field.

- [x] **Step 3: Add optional diagnostics formatting**

Read the field with `getattr`, format four boolean values for the first batch, and leave the existing six reject-bit fields unchanged. Do not make viewer startup depend on the field.

- [x] **Step 4: Run all Python regression tests in scope**

Run: `PYTHONPATH="$PWD/Go2Pvcnn" /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism Go2Pvcnn/tests/test_viewer_reset.py`

Expected: all newly added tests pass. Existing baseline failures must be resolved or explicitly reported with their pre-existing cause before claiming completion.

- [x] **Step 5: Run Isaac Sim M1 flat smoke**

Run with the project Isaac Sim environment and the user-provided display/CUDA setup:

```bash
export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn/Go2Pvcnn
/opt/VirtualGL/bin/vglrun -d egl \
  /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --device cuda:0 --num_envs 1 --terrain task --robot m1 \
  --planner-backend parallelism --n-frames 30 --plan-dt 0.02 \
  --terrain-row 0 --terrain-col 0 --warmup-steps 0
```

Expected: M1 USD and lighted terrain load; flat route keeps Root height fixed, selected non-swing legs roll with the Root, and no wheel joint is rewritten. A second smoke on a small-obstacle column must show a swing only for selected legs whose selected path hits semantic small obstacle.

## Plan Self-Review

- Coverage: candidate generation, semantic path metadata, selected execution, wheel clearance, Go2 isolation, viewer diagnostics, unit tests and Isaac Sim smoke are all assigned above.
- Placeholder scan: no unresolved implementation placeholder is allowed; every task names its file, interface, test command and expected result.
- Type consistency: `candidate_needs_swing` is `[B,4,C]`, `selected_needs_swing` is `[B,4]`, and `leg_swing` consumes `[B,4]`.
- Scope: no RL policy, wheel velocity planning, BASE collision or gait redesign is included.

## Execution Record

- 聚焦 M1/planner/swing/viewer 回归：`51 passed`。
- 完整范围回归：`125 passed, 6 failed`。
- 两个 `test_offline_obstacle_diagnostics.py` 失败是已有 stale expected values；当前 HEAD 代码本身也产生 `[0,31,46,0]` 和 `[0,40,44,0]`，未因本次 M1 selective swing 修改而改变。
- `test_viewer_reset.py` 的 4 个失败分别依赖不存在的 `/mnt/mydisk/lhy/IsaacLab` 路径，查找另一版本 `scripts/play.py` 的循环源码，以及与当前 `_play_loop_should_continue` 语义不一致的 `max_steps=0` 断言；均不属于本次 M1 规划改动。
- Isaac Sim M1 flat smoke 已成功加载 M1 USD、地形和光照；平地 `selected_swing=[0,0,0,0]`，前进命令可推动 Root。
