# Parallelism Per-Leg Tracking Rewards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add active-swing worst-foot and worst-joint tracking rewards, tune retained locomotion rewards for the 0.24 s trot cycle, and expose per-leg tracking metrics without changing planner, observation, termination, or curriculum behavior.

**Architecture:** `tracking.mdp.rewards` remains the single source for current-frame tracking errors and episode accumulators. New reward terms consume the same next-frame joint/foot reference used by existing tracking rewards, use Torch masks/reductions across environments, legs, and joints, and return one scalar per environment. `ParallelismTrackingEnv` expands fixed-shape per-leg episode tensors into TensorBoard names only at reset, while environment configuration owns all final weights and thresholds.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab manager-based RL, RSL-RL, pytest.

## Global Constraints

- Keep every existing reward term; no reward may be deleted or set to `None` as part of this change.
- Add `reference_active_swing_foot_max` with weight `1.5` and std `0.12 m`.
- Add `reference_joint_max` with weight `0.75` and std `0.8 rad`, taking the maximum over all 12 joints.
- Change `joint_pos` weight from `-0.7` to `-0.2`.
- Keep `feet_air_time` weight `0.1` and change its threshold from `0.5 s` to `0.20 s`.
- Change `air_time_variance` weight from `-1.0` to `-0.1`.
- Change `action_rate` weight from `-0.1` to `-0.03`.
- Keep all other reward weights and parameters unchanged.
- Keep Parallelism planner, reference timing, standstill, observation, action, termination, and curriculum formulas/thresholds unchanged.
- Use Torch tensor masks and reductions only in per-step code; do not add Python loops over environments.
- Keep all runtime tensors on the input tensor's dtype and device.
- Commit every completed implementation task on `Parallelism-flat-rl`.

---

### Task 1: Add worst-foot and worst-joint reward terms

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/rewards.py`
- Modify: `Go2Pvcnn/tracking/mdp/__init__.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`

**Interfaces:**
- Produces `reference_active_swing_foot_max_reward(env, asset_cfg, std=0.12) -> Tensor[B]`.
- Produces `reference_joint_max_reward(env, asset_cfg, std=0.8) -> Tensor[B]`.
- Active swing is `~manager.current_contact_state`; environments with no active swing receive zero from the new swing reward.
- Joint max covers all 12 joints in IsaacLab articulation order.

- [ ] **Step 1: Extend the fake environment and write failing reward tests.**

  Add explicit `joint_names`, a two-leg swing mask, and tests equivalent to:

  ```python
  def test_active_swing_worst_foot_reward_uses_only_worst_active_leg():
      env = _fake_env()
      env.parallelism_reference_manager.current_contact_state[:] = torch.tensor([False, True, True, False])
      env.scene["robot"].data.body_pos_w[:, 0, 0] += 0.05
      env.scene["robot"].data.body_pos_w[:, 3, 0] += 0.10
      reward = reference_active_swing_foot_max_reward(env, std=0.12)
      expected = torch.exp(torch.tensor(-(0.10 / 0.12) ** 2))
      assert torch.allclose(reward, torch.full((2,), expected))

  def test_active_swing_worst_foot_reward_ignores_stance_error():
      env = _fake_env()
      env.parallelism_reference_manager.current_contact_state[:] = torch.tensor([False, True, True, False])
      env.scene["robot"].data.body_pos_w[:, 1, 0] += 1.0
      assert torch.allclose(reference_active_swing_foot_max_reward(env), torch.ones(2))

  def test_joint_max_reward_uses_worst_of_all_twelve_joints():
      env = _fake_env()
      env.scene["robot"].data.joint_pos[:, 7] = 0.8
      expected = torch.exp(torch.tensor(-1.0))
      assert torch.allclose(reference_joint_max_reward(env, std=0.8), torch.full((2,), expected))
  ```

- [ ] **Step 2: Run focused tests and verify RED.**

  ```bash
  env PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py
  ```

  Expected: import failures for the two new reward functions.

- [ ] **Step 3: Implement the minimal Torch reward functions.**

  Reuse the existing root-frame foot conversion and Gaussian convention:

  ```python
  def reference_active_swing_foot_max_reward(env, asset_cfg=SceneEntityCfg("robot", body_names=".*_foot"), std=0.12):
      asset = env.scene[asset_cfg.name]
      manager = get_parallelism_reference_manager(env)
      ref_w = manager.step_foot_pos_w
      actual_w = _actual_foot_pos_w(asset, asset_cfg, ref_w)
      root_pos = torch.as_tensor(asset.data.root_pos_w, dtype=ref_w.dtype, device=ref_w.device)
      root_quat = torch.as_tensor(asset.data.root_quat_w, dtype=ref_w.dtype, device=ref_w.device)
      foot_error = torch.linalg.vector_norm(
          _points_to_root_frame(actual_w, root_pos, root_quat)
          - _points_to_root_frame(ref_w, root_pos, root_quat),
          dim=-1,
      )
      swing = ~torch.as_tensor(manager.current_contact_state, dtype=torch.bool, device=ref_w.device)
      has_swing = swing.any(dim=-1)
      worst = torch.where(swing, foot_error, torch.full_like(foot_error, -torch.inf)).amax(dim=-1)
      reward = torch.exp(-torch.square(worst / float(std)))
      return torch.where(has_swing, reward, torch.zeros_like(reward))

  def reference_joint_max_reward(env, asset_cfg=SceneEntityCfg("robot"), std=0.8):
      asset = env.scene[asset_cfg.name]
      ref = get_parallelism_reference_manager(env).step_joint_pos
      actual = torch.as_tensor(asset.data.joint_pos, dtype=ref.dtype, device=ref.device)
      worst = torch.abs(actual - ref).amax(dim=-1)
      return torch.exp(-torch.square(worst / float(std)))
  ```

  Export both functions from `tracking.mdp.__init__`.

- [ ] **Step 4: Run focused tests and verify GREEN.**

  Run the command from Step 2. Expected: all tracking MDP tests pass.

- [ ] **Step 5: Commit the reward functions.**

  ```bash
  git add Go2Pvcnn/tracking/mdp/rewards.py Go2Pvcnn/tracking/mdp/__init__.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py
  git commit -m "feat: add worst-leg parallelism tracking rewards"
  ```

### Task 2: Add accurate per-leg episode tracking statistics

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/rewards.py`
- Modify: `Go2Pvcnn/tracking/env.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`

**Interfaces:**
- `_current_parallelism_tracking_errors` additionally returns `foot_error_per_leg[B,4]`, `foot_z_error_per_leg[B,4]`, `swing_mask[B,4]`, and `joint_max_error_per_leg[B,4]`.
- `parallelism_tracking_episode_errors` returns global active-swing mean/max and Z mean/max plus per-leg swing mean/max/Z mean and joint max tensors.
- Per-leg tensor order is always `FL, FR, RL, RR`.

- [ ] **Step 1: Write failing current-frame and episode-stat tests.**

  Add a fake articulation `find_bodies(".*_foot")` implementation with a deliberately shuffled body order and verify the metric path restores `FL, FR, RL, RR`. Add a two-step episode test that checks:

  ```python
  assert stats["episode_active_swing_foot_mean_error"].shape == (2,)
  assert stats["episode_swing_foot_mean_error_per_leg"].shape == (2, 4)
  assert stats["episode_swing_foot_max_error_per_leg"].shape == (2, 4)
  assert stats["episode_swing_foot_z_mean_error_per_leg"].shape == (2, 4)
  assert stats["episode_joint_max_error_per_leg"].shape == (2, 4)
  ```

  Verify stance-only errors do not enter swing sum/count, and `reset_parallelism_tracking_error_stats(env, tensor([1]))` clears only environment 1.

- [ ] **Step 2: Run focused tests and verify RED.**

  Use the Task 1 focused pytest command. Expected: missing per-leg keys and accumulator attributes.

- [ ] **Step 3: Resolve and cache canonical foot/joint indices.**

  Add helpers that resolve articulation names once, map normalized names to canonical orders, and cache device tensors on the environment:

  ```python
  _LEG_NAMES = ("FL", "FR", "RL", "RR")
  _FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

  foot_ids, foot_names = asset.find_bodies(".*_foot")
  ordered_ids = [foot_ids[foot_names.index(name)] for name in _FOOT_NAMES]

  joint_leg_ids = torch.tensor([
      [joint_names.index(f"{leg}_hip_joint"), joint_names.index(f"{leg}_thigh_joint"), joint_names.index(f"{leg}_calf_joint")]
      for leg in _LEG_NAMES
  ], device=device)
  ```

  The list construction occurs only during first-time cache setup; all per-step indexing and reductions remain Torch operations. For lightweight tests with exactly four body positions, allow canonical `[0,1,2,3]` fallback. Do not use the old real-robot `body_pos_w[:, -4:]` assumption for metric computation.

- [ ] **Step 4: Extend current-frame errors and episode accumulators.**

  Compute foot and Z errors in the current policy root frame, active masks from the same reference frame, and joint per-leg max with a batched gather. Allocate fixed-shape buffers:

  ```python
  env._parallelism_tracking_active_swing_foot_sum       # [B]
  env._parallelism_tracking_active_swing_foot_count     # [B], long
  env._parallelism_tracking_active_swing_foot_max       # [B]
  env._parallelism_tracking_active_swing_foot_z_sum     # [B]
  env._parallelism_tracking_active_swing_foot_z_max     # [B]
  env._parallelism_tracking_swing_foot_sum_per_leg      # [B,4]
  env._parallelism_tracking_swing_foot_count_per_leg    # [B,4], long
  env._parallelism_tracking_swing_foot_max_per_leg      # [B,4]
  env._parallelism_tracking_swing_foot_z_sum_per_leg    # [B,4]
  env._parallelism_tracking_joint_max_per_leg           # [B,4]
  ```

  Update each accumulator only when the existing per-environment `update_mask` is true. Means divide by dtype-converted `count.clamp_min(1)`. Add every buffer to selective reset.

- [ ] **Step 5: Expose TensorBoard names from `ParallelismTrackingEnv._reset_idx`.**

  Keep existing scalar mappings and add global mappings:

  ```python
  ("episode_active_swing_foot_mean_error", "Episode_Tracking/episode_active_swing_foot_mean_error")
  ("episode_active_swing_foot_max_error", "Episode_Tracking/episode_active_swing_foot_max_error")
  ("episode_active_swing_foot_z_mean_error", "Episode_Tracking/episode_active_swing_foot_z_mean_error")
  ("episode_active_swing_foot_z_max_error", "Episode_Tracking/episode_active_swing_foot_z_max_error")
  ```

  At reset only, enumerate `("FL", "FR", "RL", "RR")` and expand the four `[B,4]` tensors into:

  ```text
  Episode_Tracking/episode_swing_foot_{LEG}_mean_error
  Episode_Tracking/episode_swing_foot_{LEG}_max_error
  Episode_Tracking/episode_swing_foot_{LEG}_z_mean_error
  Episode_Tracking/episode_joint_{LEG}_max_error
  ```

- [ ] **Step 6: Run tracking tests and verify GREEN.**

  ```bash
  env PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/tracking
  ```

- [ ] **Step 7: Commit per-leg statistics.**

  ```bash
  git add Go2Pvcnn/tracking/mdp/rewards.py Go2Pvcnn/tracking/env.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py
  git commit -m "feat: log per-leg parallelism tracking errors"
  ```

### Task 3: Register rewards and tune retained reward parameters

**Files:**
- Modify: `Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py`

**Interfaces:**
- Registers `reference_active_swing_foot_max` and `reference_joint_max` reward terms.
- Overrides inherited reward configurations without removing any existing terms.

- [ ] **Step 1: Add failing static configuration assertions.**

  Require both new term names and exact final values:

  ```python
  assert "reference_active_swing_foot_max = RewTerm" in source
  assert "reference_joint_max = RewTerm" in source
  assert 'weight=1.5' in active_swing_source
  assert '"std": 0.12' in active_swing_source
  assert 'weight=0.75' in joint_max_source
  assert '"std": 0.8' in joint_max_source
  assert "self.rewards.joint_pos.weight = -0.2" in source
  assert "self.rewards.feet_air_time.params[\"threshold\"] = 0.20" in source
  assert "self.rewards.air_time_variance.weight = -0.1" in source
  assert "self.rewards.action_rate.weight = -0.03" in source
  ```

  Keep the existing assertion that `reference_foot_pos` is never set to `None`.

- [ ] **Step 2: Run static test and verify RED.**

  ```bash
  env PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py
  ```

- [ ] **Step 3: Register new terms and override retained reward settings.**

  Add to `ParallelismTrackingRewardsCfg`:

  ```python
  reference_active_swing_foot_max = RewTerm(
      func=tracking_mdp.reference_active_swing_foot_max_reward,
      weight=1.5,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), "std": 0.12},
  )
  reference_joint_max = RewTerm(
      func=tracking_mdp.reference_joint_max_reward,
      weight=0.75,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "std": 0.8},
  )
  ```

  In `ParallelismTrackingFlatEnvCfg.__post_init__`, after `super().__post_init__()`, set only:

  ```python
  self.rewards.joint_pos.weight = -0.2
  self.rewards.feet_air_time.params["threshold"] = 0.20
  self.rewards.air_time_variance.weight = -0.1
  self.rewards.action_rate.weight = -0.03
  ```

  Do not change any other reward, termination, or curriculum field.

- [ ] **Step 4: Run static and MDP tests and verify GREEN.**

  ```bash
  env PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py
  ```

- [ ] **Step 5: Commit configuration changes.**

  ```bash
  git add Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py
  git commit -m "feat: tune parallelism imitation rewards"
  ```

### Task 4: Full verification and real 1024-environment smoke test

**Files:**
- Verify: `Go2Pvcnn/tracking`
- Verify: `Go2Pvcnn/tests/tracking`
- Verify: `docs/superpowers/specs/2026-08-04-parallelism-per-leg-tracking-reward-design.html`

**Interfaces:**
- Confirms all added rewards and metrics are finite under real IsaacLab tensors.
- Confirms training reaches iteration 4 with 1024 environments.

- [ ] **Step 1: Run syntax, whitespace, and complete tracking tests.**

  ```bash
  /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m compileall -q Go2Pvcnn/tracking Go2Pvcnn/scripts
  git diff --check
  env PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/tracking
  ```

  Expected: compile and whitespace checks produce no output; all tracking tests pass.

- [ ] **Step 2: Start a real headless 1024-environment training smoke test.**

  ```bash
  env OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/lib/python3.10/site-packages/torch/lib:/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/lib/python3.10/site-packages/nvidia/cudnn/lib:/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/lib/python3.10/site-packages/nvidia/cuda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
    Go2Pvcnn/scripts/train.py --experiment parallelism_tracking_flat --num_envs 1024 \
    --device cuda:0 --headless --max_iterations 4
  ```

  Expected: iterations `0/4` through `3/4` complete; both new `Episode_Reward` entries and finite new `Episode_Tracking` entries appear after resets.

- [ ] **Step 3: Inspect smoke metrics and performance.**

  Confirm logs contain finite values for:

  ```text
  Episode_Reward/reference_active_swing_foot_max
  Episode_Reward/reference_joint_max
  Episode_Tracking/episode_active_swing_foot_mean_error
  Episode_Tracking/episode_swing_foot_FL_mean_error
  Episode_Tracking/episode_swing_foot_RR_max_error
  Episode_Tracking/episode_joint_RL_max_error
  ```

  Record collection time and steps/s. Compare only for obvious regressions; a four-iteration smoke run is not a convergence test.

- [ ] **Step 4: Commit any verification-only test adjustments, then confirm a clean worktree.**

  ```bash
  git status --short
  ```

  Expected: no uncommitted source or test changes remain. Generated logs remain untracked/ignored.
