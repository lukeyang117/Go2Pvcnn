# Parallelism Root Relative Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Parallelism RL 的 root reference 改为当前 policy root 坐标系下的下一帧相对位置和相对姿态，并同步 root tracking metric。

**Architecture:** `ParallelismReferenceManager` 继续维护规划轨迹和 `t -> t+1` 时间对齐，新增相对 root pose 的缓存访问器。`tracking.mdp.observations` 只读取 manager 的相对 pose；`tracking.mdp.rewards` 继续保留速度 reward/curriculum 所需的速度误差，同时新增位置和 axis-angle 姿态误差统计。配置和环境日志注册使用新的 observation 与 metric 名称。

**Tech Stack:** Python 3.10, PyTorch, IsaacLab manager-based RL, pytest。

## Global Constraints

- Parallelism planner、root 轨迹生成、swing 轨迹和重规划逻辑不修改。
- policy 在真实状态 `t` 执行动作，reference 输入使用 `t+1`；末帧使用 `min(t+1, horizon-1)`。
- root 相对位置使用当前 policy root frame；相对姿态使用 `q_policy^-1 * q_reference` 的 3 维 axis-angle。
- actor/critic root reference 维度保持 6 维。
- reward、termination、curriculum 的具体计算公式不修改；内部速度误差保留给现有 curriculum。
- 所有修改提交到当前 `Parallelism-flat-rl` 分支。

---

### Task 1: Extend manager with relative root pose accessors

**Files:**
- Modify: `Go2Pvcnn/tracking/managers/parallelism_reference_manager.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`

**Interfaces:**
- Produces `current_root_pos_b_policy` with shape `(num_envs, 3)`.
- Produces `current_root_rot_b_policy` with shape `(num_envs, 3)`.
- Both accessors compare the live policy root at `t` with the manager reference frame at `min(t+1, horizon-1)`.

- [ ] **Step 1: Add failing manager tests.**

  Add tests that create a one-environment fake plan with a reference root one meter forward and a 90-degree yaw target. Verify the identity policy root returns `[1, 0, 0]` and an identity relative rotation, while a 90-degree yaw policy root rotates the position into policy coordinates and returns a near-zero/expected relative axis-angle.

- [ ] **Step 2: Run the focused tests and confirm failure.**

  Run:

  ```bash
  pytest Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py -q
  ```

  Expected: the new accessor tests fail because the manager does not expose relative root pose properties.

- [ ] **Step 3: Implement batched pose conversion.**

  Add a small wxyz quaternion helper in the manager for reference RPY conversion and use torch-only batched math:

  ```python
  target_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
  ref_pos_w = self._take(self.root_pos_w, target_phase)
  policy_pos_w = torch.as_tensor(self._robot().data.root_pos_w, dtype=ref_pos_w.dtype, device=ref_pos_w.device)
  policy_quat_w = torch.as_tensor(self._robot().data.root_quat_w, dtype=ref_pos_w.dtype, device=ref_pos_w.device)
  ref_pos_b = quat_rotate_inverse(policy_quat_w, ref_pos_w - policy_pos_w)
  ref_quat_w = rpy_to_quat(self._take(self.root_rpy_w, target_phase))
  relative_quat = quat_mul(quat_inverse(policy_quat_w), ref_quat_w)
  ref_rot_b = axis_angle_from_quat(relative_quat)
  ```

  Keep the conversion on the manager device and clamp the quaternion scalar before `acos` to avoid NaNs.

- [ ] **Step 4: Run focused tests and verify pass.**

  Run the same pytest command. Expected: all manager tests pass, including the existing `t=22` replan boundary test.

- [ ] **Step 5: Commit the manager change.**

  ```bash
  git add Go2Pvcnn/tracking/managers/parallelism_reference_manager.py Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py
  git commit -m "feat: expose relative parallelism root pose"
  ```

### Task 2: Replace root reference observations

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/observations.py`
- Modify: `Go2Pvcnn/tracking/mdp/__init__.py`
- Modify: `Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`

**Interfaces:**
- Adds `parallelism_ref_root_pos_b_t(env) -> Tensor[B, 3]`.
- Adds `parallelism_ref_root_rot_b_t(env) -> Tensor[B, 3]`.
- Removes the old root velocity observation terms from policy and critic configuration.

- [ ] **Step 1: Update observation tests first.**

  Replace the old velocity-observation assertions with checks that the observation functions return the manager's relative position and relative rotation tensors and preserve shape `(2, 3)`.

- [ ] **Step 2: Run the focused MDP tests and confirm the expected import/config failures.**

  Run:

  ```bash
  pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q
  ```

- [ ] **Step 3: Implement the two observation terms and exports.**

  The functions should be thin manager accessors:

  ```python
  def parallelism_ref_root_pos_b_t(env):
      return get_parallelism_reference_manager(env).current_root_pos_b_policy

  def parallelism_ref_root_rot_b_t(env):
      return get_parallelism_reference_manager(env).current_root_rot_b_policy
  ```

  Export both names from `tracking.mdp`.

- [ ] **Step 4: Update actor and critic config terms.**

  Replace both old config terms with:

  ```python
  parallelism_ref_root_pos = ObsTerm(func=tracking_mdp.parallelism_ref_root_pos_b_t)
  parallelism_ref_root_rot = ObsTerm(func=tracking_mdp.parallelism_ref_root_rot_b_t)
  ```

  The two terms remain 3 dimensions each, so the network input dimension remains unchanged.

- [ ] **Step 5: Run MDP and static config tests.**

  ```bash
  pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py -q
  ```

- [ ] **Step 6: Commit observation/config changes.**

  ```bash
  git add Go2Pvcnn/tracking/mdp/observations.py Go2Pvcnn/tracking/mdp/__init__.py Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py
  git commit -m "feat: use relative root pose observations"
  ```

### Task 3: Change root tracking metrics without changing velocity reward/curriculum

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/rewards.py`
- Modify: `Go2Pvcnn/tracking/env.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`

**Interfaces:**
- Adds per-step `root_pos_error` and `root_rot_error`.
- Adds episode `episode_root_pos_error` and `episode_root_rot_error`.
- Keeps `lin_vel_error` and `ang_vel_error` internally for existing velocity curriculum and reward behavior.
- Logs `Episode_Tracking/episode_reference_root_pos_error` and `Episode_Tracking/episode_reference_root_rot_error`.

- [ ] **Step 1: Add failing metric tests.**

  Extend the fake manager with `current_root_pos_b_policy` and `current_root_rot_b_policy`, then assert zero pose error when the live root matches the reference and nonzero errors for a known offset/axis-angle.

- [ ] **Step 2: Run the focused test and confirm failure.**

  ```bash
  pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q
  ```

- [ ] **Step 3: Accumulate pose metrics.**

  In `_current_parallelism_tracking_errors`, read the manager's relative root pose and compute:

  ```python
  root_pos_error = torch.linalg.vector_norm(ref_pos_b, dim=-1)
  root_rot_error = torch.linalg.vector_norm(ref_rot_b, dim=-1)
  ```

  Add sum buffers, reset them with the other episode buffers, and return averaged episode values. Do not delete the existing velocity buffers because curriculum still consumes them.

- [ ] **Step 4: Update environment metric names.**

  Change the metric mapping in `tracking/env.py` to the new position/rotation names. Update static tests to require the new names and reject the old public root velocity metric names.

- [ ] **Step 5: Run tracking tests.**

  ```bash
  pytest Go2Pvcnn/tests/tracking -q
  ```

- [ ] **Step 6: Commit metric changes.**

  ```bash
  git add Go2Pvcnn/tracking/mdp/rewards.py Go2Pvcnn/tracking/env.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py
  git commit -m "feat: report relative root pose tracking metrics"
  ```

### Task 4: Full verification and documentation consistency

**Files:**
- Verify: `docs/superpowers/specs/2026-08-04-parallelism-root-relative-observation-design.html`
- Verify: all modified tracking files

- [ ] **Step 1: Run syntax and whitespace checks.**

  ```bash
  python -m compileall -q Go2Pvcnn/tracking
  git diff --check HEAD~3..HEAD
  ```

- [ ] **Step 2: Run all focused tests.**

  ```bash
  pytest Go2Pvcnn/tests/tracking -q
  ```

  Expected: all tracking tests pass.

- [ ] **Step 3: Verify the final branch state.**

  ```bash
  git status --short --branch
  git log -4 --oneline
  ```

  Expected: no uncommitted code changes and the three implementation commits are present after the existing documentation commit.
