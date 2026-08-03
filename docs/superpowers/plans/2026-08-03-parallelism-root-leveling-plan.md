# Parallelism Root 姿态快速水平化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 Parallelism 在倾斜 root 状态开始规划时保留第 0 帧真实姿态，并在前 12 帧内将 roll/pitch 平滑恢复为水平。

**Architecture:** 在 `extension/parallelism/root.py` 的 root rollout 中生成 `[B, horizon, 3]` 的 RPY 轨迹。roll/pitch 使用 GPU 上的 smoothstep 插值，yaw 保持既有 rollout；该 RPY 张量继续作为 candidate、IK、FK 和碰撞检查的统一输入。通过 Parallelism 单元测试锁定第 0 帧对齐、第 12 帧水平和无效规划 fallback 行为。

**Tech Stack:** Python 3.10、PyTorch、pytest、现有 Parallelism planner。

## Global Constraints

- `root_leveling_frames = 12` 是默认值，并且必须通过 `ParallelismCfg` 可调。
- 第 0 帧必须等于当前实测 root RPY。
- 第 12 帧及之后 roll/pitch 必须为 0。
- 规划必须保持 batch/GPU 张量并行，不新增逐帧 Python 控制循环。
- 不修改 root XY、yaw、touchdown candidate、semantic filter、score、碰撞体、RL reward、termination 或 curriculum。
- 无效规划且启用 standstill fallback 时，输出必须继续保持当前实测 root、joint、foot 状态。

---

### Task 1: Add Regression Tests For Tilted Root Leveling

**Files:**
- Modify: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`
- Test: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`

**Interfaces:**
- Consumes: `ParallelismState`, `ParallelismCfg`, `rollout_root`, `plan_trajectory`.
- Produces: tests that define the expected root RPY trajectory and preserve existing XY behavior.

- [x] **Step 1: Add the failing test for the 12-frame leveling contract**

Add a test with `root_rpy_w=(0.2, 0.15, 0.4)` and assert:

```python
result.root_rpy_w[0, 0, :2] == (0.2, 0.15)
result.root_rpy_w[0, 12, :2] == (0.0, 0.0)
result.root_rpy_w[0, 23, :2] == (0.0, 0.0)
result.root_rpy_w[0, 0, 2] == 0.4
```

Also assert the roll/pitch magnitudes are non-increasing and that `root_pos_w[..., :2]` still follows the existing commanded XY rollout.

- [x] **Step 2: Run the focused test and verify it fails**

Run:

```bash
PYTHONPATH=Go2Pvcnn python -m pytest -q Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py -k levels
```

Expected: FAIL because current rollout copies the initial roll and pitch through all 24 frames.

---

### Task 2: Implement Batched Root Leveling

**Files:**
- Modify: `Go2Pvcnn/Go2Pvcnn/extension/parallelism/config.py:21-48`
- Modify: `Go2Pvcnn/Go2Pvcnn/extension/parallelism/root.py:36-80`

**Interfaces:**
- Consumes: `ParallelismCfg.root_leveling_frames`, current measured `state.root_rpy_w`, existing yaw delta.
- Produces: `RootRollout.root_rpy_w` with shape `[B, horizon, 3]`, preserving the existing public interface.

- [x] **Step 1: Add the configurable leveling duration**

Add this field to `ParallelismCfg` next to the horizon timing fields:

```python
root_leveling_frames: int = 12
```

- [x] **Step 2: Generate smoothstep weights without a time-step loop**

Inside `rollout_root`, generate one horizon-index tensor and compute:

```python
level_frames = max(int(cfg.root_leveling_frames), 1)
frame = torch.arange(int(cfg.horizon), dtype=root0.dtype, device=root0.device)
u = torch.clamp(frame / float(level_frames), 0.0, 1.0)
s = u * u * (3.0 - 2.0 * u)
level_scale = (1.0 - s)[None, :, None]
```

Then replace the current constant roll/pitch assignments with:

```python
root_rpy[..., 0] = rpy0[:, None, 0] * level_scale[..., 0]
root_rpy[..., 1] = rpy0[:, None, 1] * level_scale[..., 0]
root_rpy[..., 2] = yaw
```

Validate `root_leveling_frames` through the `max(..., 1)` clamp so invalid zero or negative configuration values cannot divide by zero.

- [x] **Step 3: Run the focused leveling and existing root tests**

Run:

```bash
PYTHONPATH=Go2Pvcnn python -m pytest -q \
  Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py \
  Go2Pvcnn/tests/parallelism/test_planner.py
```

Expected: all tests PASS, including tilted root leveling and existing displacement/fallback contracts.

---

### Task 3: Verify End-to-End Tilted Planning And Commit

**Files:**
- Modify: none beyond Task 1 and Task 2.
- Test: existing `Go2Pvcnn/tests/parallelism/` suite.

**Interfaces:**
- Consumes: the updated `rollout_root` output through `plan_trajectory`.
- Produces: evidence that tilted planning uses the same leveled root trajectory for candidate and final planning, without changing fallback semantics.

- [x] **Step 1: Run the full Parallelism test suite**

Run:

```bash
PYTHONPATH=Go2Pvcnn python -m pytest -q Go2Pvcnn/tests/parallelism
```

Expected: all Parallelism tests PASS.

- [x] **Step 2: Run a batched tilted-state probe**

Use a batch containing horizontal, roll-only, pitch-only and combined tilted states. Verify:

```text
root_rpy[:, 0, :2] == measured_rpy[:, :2]
root_rpy[:, 12:, :2] == 0
root_rpy.shape == [B, 24, 3]
```

Also verify planner output remains finite and that invalid-plan standstill fallback still returns the measured state when the terrain valid mask is false.

- [x] **Step 3: Run repository hygiene checks**

Run:

```bash
python -m compileall -q Go2Pvcnn/Go2Pvcnn/extension/parallelism
git diff --check
git status --short
```

Expected: no compile errors, no whitespace errors, and only the intended implementation/test files are modified.

- [x] **Step 4: Commit the implementation**

```bash
git add Go2Pvcnn/extension/parallelism/config.py \
        Go2Pvcnn/extension/parallelism/root.py \
        tests/parallelism/test_root_candidates_kinematics.py \
        docs/superpowers/plans/2026-08-03-parallelism-root-leveling-plan.md
git commit -m "fix: level parallelism root attitude within horizon"
```
