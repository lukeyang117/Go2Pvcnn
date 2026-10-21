# M1 RL NaN Root Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Stop M1 AME leg-state explosions and NaN reward propagation at their source, correct wheel-angle reward selection, and preserve Go2 and terrain curriculum behavior.

**Architecture:** Keep fixes M1-specific. Bound normalized mixed actions and target slew in the M1 action term; separate M1 leg and wheel implicit actuators with finite limits; terminate invalid M1 states and make joint-state reward outputs finite; implement selector-aware M1 reward terms. Go2 code and terrain curriculum thresholds remain unchanged.

**Tech Stack:** Python 3.10, PyTorch, Isaac Lab/Isaac Sim 4.5, pytest, TensorBoard.

## Global Constraints

- All source changes are confined to the `m1_rl` worktree.
- Keep action/observation dimensions at 16/1594/1597.
- Keep terrain layout, terrain curriculum thresholds, and Go2 configs unchanged.
- Normalized M1 actions are bounded to `[-1,1]`; wheel command scale remains `8 rad/s`.
- Leg actuator safety velocity is `20 rad/s`; wheel actuator safety velocity is `8 rad/s`; leg/wheel effort ceilings are `150/50 N·m` per the USD drive limits.
- Never silently write NaN/Inf into reward sums; invalid-state termination and finite M1 reward outputs are both required.

---

### Task 1: Add failing M1 root-cause regression tests

**Files:**
- Create: `Go2Pvcnn/tests/m1_ame/test_m1_ame_nan_root_fix.py`
- Modify: `Go2Pvcnn/tests/m1_ame/test_m1_mixed_actions.py`

- [ ] Test normalized action clipping and bounded leg target delta using unique per-joint values.
- [ ] Test M1 wheel indices remain velocity targets while 12 leg indices remain position targets.
- [ ] Test a wheel-angle change does not contribute to M1 joint-position penalty while a leg-angle change does.
- [ ] Test non-finite joint values and over-limit joint velocities trigger the M1 invalid-state detector.
- [ ] Test invalid joint tensors yield finite reward outputs.
- [ ] Run the tests and confirm failures correspond to the missing behavior.

### Task 2: Bound M1 actions and leg-target rate

**Files:**
- Modify: `Go2Pvcnn/m1_ame/m1_mixed_action_math.py`
- Modify: `Go2Pvcnn/m1_ame/m1_mixed_actions.py`
- Modify: `Go2Pvcnn/m1_ame/m1_ame_env_cfg.py`

- [ ] Add configurable normalized action clipping (default M1 value `1.0`).
- [ ] Add a `0.15 rad/control-step` leg target-delta limit and absolute M1 position limits.
- [ ] Seed/reset the prior target from the articulation default leg pose.
- [ ] Run action math and M1 action tests.

### Task 3: Separate M1 leg and wheel actuators

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/assets/m1.py`

- [ ] Replace the single all-joint actuator group with exact named leg and wheel groups.
- [ ] Set leg `effort_limit_sim=150`, `velocity_limit_sim=20`, `stiffness=25`, `damping=0.5`.
- [ ] Set wheel `effort_limit_sim=50`, `velocity_limit_sim=8`, `stiffness=0`, `damping=0.5`.
- [ ] Test group coverage, disjointness, and the configured safety limits.

### Task 4: Repair M1 reward selectors and invalid-state handling

**Files:**
- Modify: `Go2Pvcnn/m1_ame/m1_ame_rewards.py`
- Modify: `Go2Pvcnn/m1_ame/m1_ame_runner.py`

- [ ] Add selector-aware M1 joint-position and joint-velocity reward functions.
- [ ] Add finite-safe wrappers for M1 state-based penalties before squaring or multiplying.
- [ ] Add invalid-state termination for non-finite joint/root/body state and excessive leg/wheel speeds.
- [ ] Log per-episode invalid-state termination and maximum leg/wheel velocity.
- [ ] Verify the Go2 reward module and curriculum file are unchanged.

### Task 5: Verify reproducible checkpoint and training smoke tests

**Files:**
- Modify tests only if the new requirements are not covered.

- [ ] Replay `model_12300.pt`, seed 42, 1024 envs for at least 120 steps with production action limits.
- [ ] Assert no non-finite joint state/reward and velocity bounds are respected.
- [ ] Run 1-env and 1024-env Isaac Lab one-iteration smoke tests.
- [ ] Run M1 AME, curriculum, M1 backend and Go2 parity tests; run `git diff --check`.
- [ ] Commit and push `m1_rl`; verify remote SHA matches and `m1` worktree is unchanged.
