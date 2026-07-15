# Flat-Small Path Swing-Over Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `semantic_foot_over_clearance` reward prefer MPC-aligned swing-over behavior on path-small opportunities instead of rewarding contact avoidance alone.

**Architecture:** Keep the same reward term and policy interfaces. Tighten the tensor reward helper so strict swing-over is the primary positive signal, dense/near shaping is capped lower, and root-crossed-without-overpass receives a small negative value inside the same reward.

**Tech Stack:** PyTorch tensor ops, existing `MpcPlannerTerrain` query helpers, pytest, `env_isaacsim`.

## Global Constraints

- Do not add a new reward term.
- Do not add or change MPC planner losses.
- Do not change policy observation or action shape.
- Do not remove `bad_orientation`.
- Do not solve this by only increasing `semantic_foot_over_clearance.weight`.

---

### Task 1: RED Tests For Path Swing-Over Semantics

**Files:**
- Modify: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

**Interfaces:**
- Consumes: `semantic_foot_over_clearance_bonus_from_tensors(...) -> torch.Tensor`
- Produces: failing tests for `missed_over_penalty`, `root_crossed_margin_m`, and stricter auxiliary caps.

- [x] **Step 1: Add tests**

Add tests that prove:

- a crossed path-small opportunity with no swing overpass returns a negative reward;
- a swing foot overpass avoids that penalty and returns positive reward;
- stance feet do not cancel the missed-over penalty;
- cfg exposes `dense_approach_bonus_fraction=0.05`, `strict_near_bonus_fraction=0.25`, `missed_over_penalty=0.15`, and `root_crossed_margin_m=0.02`.

- [x] **Step 2: Run RED**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_foot_over_bonus_penalizes_crossed_path_small_without_swing_overpass \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_foot_over_bonus_swing_overpass_removes_missed_penalty \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_foot_over_bonus_stance_overpass_does_not_satisfy_swing_overpass \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected: fails because the reward helper does not accept missed-over params and cfg still has old cap values.

### Task 2: GREEN Reward Helper

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`

**Interfaces:**
- Consumes: path samples and swing mask already computed in `semantic_foot_over_clearance_bonus_from_tensors`.
- Produces: same function return type, now allowing negative values down to `-missed_over_penalty`.

- [x] **Step 1: Add params**

Add:

```python
missed_over_penalty=0.0,
root_crossed_margin_m=0.02,
```

- [x] **Step 2: Compute strict sample overpass**

Use existing foot-to-path-sample along/lateral errors and candidate clearance. Gate by `path_small`, strict footprint limits, and swing mask.

- [x] **Step 3: Compute missed-over penalty**

An env is penalized when a path-small sample is crossed and no strict sample overpass exists.

- [x] **Step 4: Run focused tests**

Same command as Task 1 should pass.

### Task 3: GREEN Config

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`

**Interfaces:**
- Consumes: new reward helper params.
- Produces: flat-small cfg with stricter auxiliary caps and missed-over params.

- [x] **Step 1: Update params**

Set:

```python
"dense_approach_bonus_fraction": 0.05,
"strict_near_bonus_fraction": 0.25,
"missed_over_penalty": 0.15,
"root_crossed_margin_m": 0.02,
```

- [x] **Step 2: Run cfg static test**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected: pass.

### Task 4: Verification And Notes

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302s-env-level-collision-curriculum-plan.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-07-01-flat-small-path-swing-over-reward.md`

- [x] **Step 1: Run focused verification**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

- [x] **Step 2: Run pycompile**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

- [x] **Step 3: Run diff check**

```bash
git diff --check -- \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  notes/log/index.md \
  notes/todo.md \
  notes/todo/T302s-env-level-collision-curriculum-plan.md
```

- [x] **Step 4: Update notes**

Record the test result and next training/eval command. Do not claim behavior improvement until a real warm-start run and controlled crossing eval prove it.
