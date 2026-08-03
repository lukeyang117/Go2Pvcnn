# Parallelism Reference Joint-Gated Velocity Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align Parallelism RL observations, rewards, curriculum, and play with the current-phase reference root twist transformed through the live policy root frame, while gating velocity curriculum upgrades on strict episode joint tracking.

**Architecture:** Keep the Parallelism planner unchanged. Add tracking-side helpers that expose one canonical reference root twist in the live policy root frame, use it in observations/rewards/curriculum, and maintain GPU episode tracking statistics for joint mean/max and root velocity errors.

**Tech Stack:** Python 3.10, PyTorch, Isaac Lab manager terms, pytest lightweight fake env tests.

## Global Constraints

- Do not modify Parallelism planning logic.
- Actor and critic must not receive raw `base_velocity` as the velocity target.
- Reference root velocity conversion must use the current policy robot root pose every step.
- Rewards, curriculum, velocity termination, and play must share the same reference twist interface.
- Joint tracking reward follows InstinctLab-style sum-square Gaussian tracking.
- Velocity curriculum only upgrades when episode joint mean, episode joint max, and reference root velocity errors pass.
- 1024-env runtime paths must stay Torch/GPU batched.

---

### Task 1: Canonical Reference Twist Interface

**Files:**
- Modify: `Go2Pvcnn/tracking/managers/parallelism_reference_manager.py`
- Modify: `Go2Pvcnn/tracking/mdp/observations.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`

**Interfaces:**
- Produces: `ParallelismReferenceManager.current_root_lin_vel_b_policy -> torch.Tensor [N,3]`
- Produces: `ParallelismReferenceManager.current_root_ang_vel_b_policy -> torch.Tensor [N,3]`
- Produces: `parallelism_ref_root_lin_vel_b_t(env)` and `parallelism_ref_root_ang_vel_b_t(env)` using the policy-frame twist.

- [ ] **Step 1: Write failing tests**
  - Add a fake policy root yaw/quat test proving the same reference world velocity changes when the live policy root yaw changes.
  - Add an observation test proving root velocity observations return policy-frame reference twist.

- [ ] **Step 2: Verify tests fail**
  - Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py::test_reference_root_velocity_uses_live_policy_root_frame Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py::test_reference_root_velocity_observation_uses_policy_frame -q`
  - Expected: FAIL because the new policy-frame properties do not exist or still use reference yaw only.

- [ ] **Step 3: Implement**
  - Add quaternion-to-rotation-matrix and batched rotation helpers.
  - Compute reference finite-difference twist from the current phase, treat it as reference-root-frame, convert to world with reference pose, then convert to live policy root frame using `env.scene["robot"].data.root_quat_w`.
  - Update observation terms to return the new policy-frame properties.

- [ ] **Step 4: Verify**
  - Run the two tests above and the full tracking lightweight tests.

### Task 2: InstinctLab-Style Joint Rewards and Reference Velocity Rewards

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/rewards.py`
- Modify: `Go2Pvcnn/tracking/mdp/__init__.py`
- Modify: `Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`

**Interfaces:**
- Produces: `reference_root_lin_vel_reward(env, std: float)`
- Produces: `reference_root_ang_vel_reward(env, std: float)`
- Changes: `reference_joint_pos_reward(..., tracking_tolerance=0.0)` uses sum-square Gaussian.
- Changes: `reference_joint_vel_reward(..., tracking_tolerance=0.0)` uses sum-square Gaussian.

- [ ] **Step 1: Write failing tests**
  - Add tests for sum-square Gaussian joint reward.
  - Add tests for tolerance clipping.
  - Add tests for root velocity rewards using reference twist, not command.

- [ ] **Step 2: Verify tests fail**
  - Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q`
  - Expected: FAIL before reward implementation.

- [ ] **Step 3: Implement**
  - Replace tracking task velocity rewards with reference velocity rewards.
  - Keep command only for planner and command sampling.
  - Remove `velocity_commands` observation terms from inherited actor/critic state in the tracking config.

- [ ] **Step 4: Verify**
  - Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py -q`

### Task 3: Episode Joint and Root Tracking Statistics for Curriculum

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/rewards.py`
- Modify: `Go2Pvcnn/tracking/mdp/curriculums.py`
- Modify: `Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`

**Interfaces:**
- Produces: `update_parallelism_tracking_error_stats(env) -> dict[str, torch.Tensor]`
- Produces: `parallelism_tracking_errors(env)` reading episode means/max for curriculum.
- Changes: `parallelism_velocity_curriculum(..., joint_mean_threshold, joint_max_threshold)` gates upgrades by episode joint mean/max and root velocity errors.

- [ ] **Step 1: Write failing tests**
  - Add test showing a low current joint error but high episode max blocks curriculum upgrade.
  - Add test showing root reference velocity error blocks curriculum upgrade.

- [ ] **Step 2: Verify tests fail**
  - Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q`

- [ ] **Step 3: Implement**
  - Maintain `_parallelism_tracking_error_*` tensors on env.
  - Update stats in reward/error calls with Torch operations.
  - Reset selected env stats inside curriculum after consuming them.
  - Preserve level clamp and range interpolation.

- [ ] **Step 4: Verify**
  - Run tracking tests.

### Task 4: Static Verification and Commit

**Files:**
- Modify: files changed by Tasks 1-3.

- [ ] **Step 1: Run checks**
  - Run: `pytest Go2Pvcnn/tests/tracking -q`
  - Run: `python -m py_compile Go2Pvcnn/tracking/mdp/rewards.py Go2Pvcnn/tracking/mdp/curriculums.py Go2Pvcnn/tracking/mdp/observations.py Go2Pvcnn/tracking/managers/parallelism_reference_manager.py Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`
  - Run: `git diff --check`

- [ ] **Step 2: Commit**
  - Run: `git add ... && git commit -m "feat: align parallelism rl reference tracking curriculum"`

## Self-Review

- Spec coverage: reference twist conversion, policy/play alignment, InstinctLab-style joint rewards, joint mean/max velocity curriculum gate, and Torch batched constraints are covered.
- Placeholder scan: no placeholders remain.
- Type consistency: manager properties and MDP reward/curriculum names are defined before use.
