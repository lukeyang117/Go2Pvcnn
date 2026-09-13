# M1 Trot Stance Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make M1 use a trot-phase planner where every stance foot remains fixed in world XYZ, and only the finally selected candidates marked `need_swing` leave the ground.

**Architecture:** Keep the existing candidate selection and analytic IK interfaces. For M1, initialize every foot target from the current world foot pose, use a ground-level XY trot path for candidates without `need_swing`, overlay terrain-aware clearance only for selected swing legs, and pass all target feet through analytic IK so stance joints may move while the foot remains fixed. Go2 continues through the existing `leg_swing=None` path.

**Tech Stack:** Python 3.10, PyTorch, pytest, Isaac Lab / Isaac Sim.

## Global Constraints

- M1 stance feet preserve world `XYZ`, not only `Z`.
- Only a selected candidate with `need_swing=True` may leave the ground.
- A M1 candidate without `need_swing` moves in XY at fixed contact Z during its nominal trot swing phase, then remains fixed in XYZ during stance.
- M1 ABAD/HIP/KNEE joints may change through analytic IK; wheel joints remain untouched by Parallelism.
- Joint-limit failures remain invalid; no joint clipping is used as a workaround.
- If the lowered M1 stance still produces out-of-limit or unreachable targets, adjust the M1 default stance parameters rather than changing Go2 behavior.
- Go2 planner behavior and interfaces remain unchanged.

## File Map

- Modify: `Go2Pvcnn/extension/parallelism/planner.py` - build fixed stance targets and gate swing/collision paths by the selected swing flag.
- Modify: `Go2Pvcnn/extension/parallelism/m1_kinematics.py` - adjust M1 stance only if the regression exposes a reachable workspace issue.
- Test: `Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py` - assert XYZ-fixed stance and selective swing behavior.
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py` - preserve the Go2 path and generic planner behavior.

### Task 1: Lock the M1 stance target in world XYZ

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/planner.py:376-439`
- Test: `Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py`

**Interfaces:**
- `_assemble_foot_targets(..., leg_swing: Tensor | None)` keeps `None` as the existing Go2 behavior.
- For M1, `leg_swing` has shape `[B,4]`; false legs use a contact-Z XY path in their nominal trot swing phase and a fixed world XYZ target in stance.

- [x] Write a failing test that passes `leg_swing=False` for all four legs; assert each nominal swing half moves only in XY and each stance half holds the final foot XYZ.
- [x] Run the focused test and confirm it fails because the current M1 branch keeps non-swing legs fixed for the whole horizon.
- [x] Initialize M1 targets from `foot0`, overlay a fixed-Z XY ground path for false legs, and hold `selected_foothold_w` after each trot swing half.
- [x] Run the focused test and confirm it passes.

### Task 2: Gate trot swing and collision checking by need_swing

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/planner.py:127-251, 403-439`
- Test: `Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py`

**Interfaces:**
- A true M1 leg uses its diagonal phase: FL/RR in the first half, FR/RL in the second half.
- A false M1 leg remains grounded, follows the contact-Z XY path in its nominal swing half, and is fixed in XYZ in its stance half.

- [x] Add a test asserting that a false leg has contact-Z XY motion in its nominal swing phase, constant XYZ in stance, and contact state true across both phases.
- [x] Run it and confirm failure under the current fixed-target implementation.
- [x] Make `_swing_collision_mask` use the same ground path for false candidates and terrain-aware paths for true candidates.
- [x] Preserve existing Go2 behavior when `candidate_needs_swing` is `None`.
- [x] Run M1 planner tests and confirm selected swing legs leave the ground only in their assigned phase.

### Task 3: Preserve analytic IK and validate limits/workspace

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/m1_kinematics.py` only if required
- Test: `Go2Pvcnn/tests/parallelism/test_m1_robot_backend.py`
- Test: `Go2Pvcnn/tests/parallelism/test_m1_planner_backend.py`

- [x] Add a regression that evaluates the full fixed-stance target trajectory through M1 analytic IK and requires all frames reachable and within limits.
- [x] Run the regression before any stance parameter adjustment.
- [x] If it fails, adjust M1 default hip/knee stance parameters and recompute `M1_ROOT_Z_M` from FK geometry; never clamp out-of-limit IK output.
- [x] Run all M1 backend and planner tests.

### Task 4: Verify Go2 compatibility and offline behavior

**Files:**
- No production files beyond the tasks above.
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_offline_obstacle_diagnostics.py`

- [x] Run the focused Go2 planner tests and confirm the legacy path remains green.
- [x] Run the full parallelism suite and report any pre-existing exact-count baseline failures separately from this M1 change.
- [x] Run `git diff --check` and inspect the final diff for unrelated edits.
