# Parallelism Reset Standard Stance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 viewer 每次 reset 后，先将 Go2 的 root 和四足调整到地形上的标准落地姿态，再开始 Parallelism 第一次规划。

**Architecture:** 保留 Parallelism planner 的候选、IK、碰撞和 standstill 逻辑不变。在 viewer reset 流程中，完成地形 tile reset 后使用已有 scanner 高程查询，将当前四足整体平移到共同地面高度，并刷新仿真数据；之后规划器读取到的第 0 帧就是四足落地的真实状态。

**Tech Stack:** Python, PyTorch, Isaac Lab, pytest.

## Global Constraints

- 不修改 `Go2Pvcnn/extension/parallelism/planner.py` 的规划判定逻辑。
- 不修改键盘 `R` 与面板 `Reset to terrain platform` 的职责。
- 不改变 standstill 的判断规则。
- 不暂存或覆盖当前工作区中与本任务无关的用户修改。

---

### Task 1: Add reset grounding behavior

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/test_viewer_reset.py`

**Interfaces:**
- Reuse `_viewer_ground_robot_from_scanner(...)` as the reset grounding primitive.
- `_reset_viewer_env(...)` continues to return the selected terrain origin.
- Grounding occurs after terrain selection and warmup, before the first planner state read.

- [ ] **Step 1: Add a focused reset test**

Verify that a reset path invokes the scanner-based grounding helper after terrain selection and warmup, without changing reset snapshot semantics.

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
pytest -q Go2Pvcnn/tests/test_viewer_reset.py -k "ground or reset"
```

Expected: the new grounding assertion fails before implementation.

- [ ] **Step 3: Implement minimal grounding in `_reset_viewer_env`**

After the second terrain selection and warmup, call:

```python
if scanner is not None and foot_ids is not None:
    _viewer_ground_robot_from_scanner(
        base_env,
        scanner,
        foot_ids,
        foot_contact_offset=0.0,
    )
```

Keep the existing reset snapshot restoration after grounding so an explicit `R` reset still restores its saved joint configuration while the root remains terrain-grounded.

- [ ] **Step 4: Run focused reset tests**

Run:

```bash
pytest -q Go2Pvcnn/tests/test_viewer_reset.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_viewer_reset.py docs/superpowers/plans/2026-08-10-parallelism-reset-standard-stance.md
git commit -m "fix: ground viewer robot before parallelism planning"
```

### Task 2: Verify the original stair reproduction

**Files:**
- No source changes expected.
- Test output: `/tmp/parallelism_stairs_reset_grounded.log`

**Interfaces:**
- Use the existing viewer entry point with `row=0`, `col=12`.
- Verify planner diagnostics after reset, not only the rendered pose.

- [ ] **Step 1: Run a deterministic zero-command reproduction**

Run the existing viewer command with a scripted zero command for one planning cycle.

- [ ] **Step 2: Check the reset state**

Expected:

```text
root_z ~= terrain_height + configured_root_clearance
per_leg_valid has nonzero values for all four legs
standstill=False
```

- [ ] **Step 3: Check that no planner geometry code changed**

Run:

```bash
git diff HEAD~1 -- Go2Pvcnn/extension/parallelism
```

Expected: no diff.

