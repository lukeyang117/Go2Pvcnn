# Parallelism RL Reference Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align Parallelism RL observations, rewards, terminations, metrics, and replanning around the next joint-position frame and the current-to-next velocity interval while preserving the complete policy action interface and the user's tuned reward weights.

**Architecture:** `ParallelismReferenceManager` will maintain the current plan phase and a per-step snapshot of the reference target selected before physics. `ParallelismTrackingEnv.step()` will prepare that snapshot before calling IsaacLab's physics step, so post-step rewards, terminations, and metrics use the same target even when the manager replans at the 22-to-23 boundary. The training logger will create `<date>/<git-short-hash>/` directories without `tag.txt`.

**Tech Stack:** Python, PyTorch, IsaacLab ManagerBasedRLEnv, Gymnasium, pytest, Git.

**Status:** Implemented and verified on August 4, 2026. The user's tuned tracking
reward weights are preserved.

## Global Constraints

- Keep Parallelism trajectories at exactly 24 frames with valid indices `0..23`.
- Keep the policy action as the complete 12-dimensional action interpreted by IsaacLab as `default_joint_pos + 0.25 * action`.
- Do not change reward formulas, metric formulas, mean/max aggregation, curriculum logic structure, or planner logic.
- Preserve the user's reward weights: `reference_joint_pos.weight=2.0` and `reference_foot_pos.weight=1.75`.
- Use the approved first-stage reference termination thresholds: root-z `0.50`, projected-gravity `1.50`, foot-z `0.30`, joint max `2.00`.
- Use curriculum thresholds `ang_vel_threshold=1.0` and `joint_max_threshold=1.0`.
- At the end of the old plan, finish reward/termination/metric evaluation against the old final frame before installing the new plan.
- New-plan frame `0` is the measured IsaacLab state; the next policy observation uses new-plan frame `1`.
- Do not create `tag.txt`; use the current Git `HEAD` short hash as the training directory name.

---

### Task 1: Add failing tests for phase timing and boundary snapshots

**Files:**
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`

**Interfaces:**
- The manager will expose `prepare_step_reference()`, `step_joint_pos`, `step_joint_vel`, `step_foot_pos_w`, `step_root_pos_w`, `step_root_rpy_w`, `step_root_lin_vel_b_policy`, and `step_root_ang_vel_b_policy`.
- The manager will expose `next_joint_pos` for the policy position observation.

- [ ] **Step 1: Add a failing manager test for the regular next target.**

```python
def test_prepare_step_reference_targets_next_position_and_current_to_next_velocity(monkeypatch):
    env = _fake_env(num_envs=1)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.joint_pos[env_ids] = torch.arange(manager.horizon, dtype=torch.float32).view(
            1, -1, 1
        ).repeat(int(env_ids.numel()), 1, 12)
        manager.root_pos_w[env_ids] = manager.joint_pos[env_ids, :, :3]

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    manager.phase[:] = 4
    manager.prepare_step_reference()

    assert torch.allclose(manager.step_joint_pos, torch.full((1, 12), 5.0))
    assert torch.allclose(manager.step_joint_vel, torch.full((1, 12), 1.0 / manager.dt))
```

- [ ] **Step 2: Run the focused test and verify it fails because the snapshot API is absent.**

Run:

```bash
pytest -q tests/tracking/test_parallelism_reference_manager.py -k "prepare_step_reference"
```

Expected: FAIL with an attribute or method error for the new manager API. If `pytest` is unavailable in the active shell, run the same command through the repository's IsaacLab Python environment.

- [ ] **Step 3: Add a failing boundary test for replan after phase 22.**

```python
def test_phase_22_step_replans_before_the_next_observation(monkeypatch):
    env = _fake_env(num_envs=1)
    env.episode_length_buf = torch.tensor([22])
    manager = ParallelismReferenceManager(env, autostart=False)
    planned_cycles = []

    def fake_plan(env_ids, cycle):
        planned_cycles.append(int(cycle[0]))
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.joint_pos[env_ids] = torch.arange(manager.horizon, dtype=torch.float32).view(
            1, -1, 1
        ).repeat(int(env_ids.numel()), 1, 12)

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    planned_cycles.clear()
    manager.prepare_step_reference()
    assert manager.phase.item() == 22
    assert torch.allclose(manager.step_joint_pos, torch.full((1, 12), 23.0))

    env.episode_length_buf[:] = 23
    manager.refresh()
    assert manager.phase.item() == 0
    assert planned_cycles == [1]
```

- [ ] **Step 4: Add a failing MDP test that observation position uses `next_joint_pos`, while velocity remains current-to-next.**

```python
def test_reference_observation_uses_next_position_and_current_to_next_velocity():
    env = _fake_env()
    manager = env.parallelism_reference_manager
    manager.next_joint_pos = torch.ones(2, 12)
    manager.current_joint_vel = torch.full((2, 12), 2.0)
    assert torch.allclose(parallelism_ref_joint_pos_rel_t(env), torch.ones(2, 12))
    assert torch.allclose(parallelism_ref_joint_vel_t(env), torch.full((2, 12), 2.0))
```

- [ ] **Step 5: Commit the tests only.**

```bash
git add Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py \
  Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py \
  Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py
git commit -m "test: define parallelism next-frame tracking timing"
```

### Task 2: Implement reference snapshots and 22-frame cycle boundary

**Files:**
- Modify: `Go2Pvcnn/tracking/managers/parallelism_reference_manager.py`
- Modify: `Go2Pvcnn/tracking/env.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`

**Interfaces:**
- `ParallelismReferenceManager.prepare_step_reference() -> None` refreshes the plan at the current phase and snapshots the target frame for the upcoming physics step.
- `next_joint_pos -> Tensor[B, 12]` returns `joint_pos[min(phase + 1, horizon - 1)]`.
- `step_*` properties return the snapshot created before the current environment step.

- [ ] **Step 1: Implement a 23-transition planning stride.**

Use:

```python
planning_stride = max(self.horizon - 1, 1)
cycle = torch.div(episode_length, planning_stride, rounding_mode="floor")
phase = torch.remainder(episode_length, planning_stride)
```

This produces phases `0..22` for action-start states. Frame `23` is the target of phase `22`, not a separate action-start phase.

- [ ] **Step 2: Add per-environment snapshot buffers.**

Allocate buffers for:

```python
step_joint_pos
step_joint_vel
step_foot_pos_w
step_root_pos_w
step_root_rpy_w
step_root_lin_vel_b_policy
step_root_ang_vel_b_policy
```

All buffers must be batched on the manager device and initialized with the same shapes as the corresponding trajectory tensors.

- [ ] **Step 3: Implement `prepare_step_reference()`.**

The method must:

1. Call `refresh()` using the current `episode_length_buf`.
2. Select `start_phase = phase`.
3. Select `target_phase = clamp(phase + 1, max=horizon - 1)`.
4. Snapshot target position/root/foot/contact frame.
5. Compute joint and root velocity reference from `start_phase` to `target_phase`.
6. Keep the final boundary velocity zero because both indices clamp to frame `23`.

- [ ] **Step 4: Add `next_joint_pos` and preserve current velocity semantics.**

`next_joint_pos` is the target position used by policy observation. Existing `current_joint_vel` and root velocity properties remain the current-to-next interval used by observations; reward/metric will consume the step snapshot instead of recalculating after phase advancement.

- [ ] **Step 5: Call `prepare_step_reference()` before IsaacLab physics.**

Override `ParallelismTrackingEnv.step()`:

```python
def step(self, action):
    get_parallelism_reference_manager(self).prepare_step_reference()
    return super().step(action)
```

Do not call the old manual `manager.step()` counter from this path.

- [ ] **Step 6: Run the focused manager and environment tests.**

Run:

```bash
pytest -q tests/tracking/test_parallelism_reference_manager.py tests/tracking/test_parallelism_tracking_registration_static.py
```

Expected: all focused tests pass, including phase `22 -> 23`, new cycle phase `0`, and no access to frame `24`.

- [ ] **Step 7: Commit the manager boundary implementation.**

```bash
git add Go2Pvcnn/tracking/managers/parallelism_reference_manager.py \
  Go2Pvcnn/tracking/env.py
git commit -m "fix: align parallelism replanning with frame 22 boundary"
```

### Task 3: Switch observation and post-step tracking consumers to the snapshot

**Files:**
- Modify: `Go2Pvcnn/tracking/mdp/observations.py`
- Modify: `Go2Pvcnn/tracking/mdp/rewards.py`
- Modify: `Go2Pvcnn/tracking/mdp/terminations.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`

**Interfaces:**
- Policy joint-position observation consumes `manager.next_joint_pos`.
- Policy velocity observations continue consuming current-to-next velocity.
- Reward and termination reference values consume `manager.step_*`.
- Metric implementation remains structurally unchanged; only its reference source changes from `current_*` to `step_*`.

- [ ] **Step 1: Write failing MDP tests for the snapshot source.**

Add a fake manager with distinct values:

```python
manager.current_joint_pos = torch.zeros(2, 12)
manager.next_joint_pos = torch.ones(2, 12)
manager.step_joint_pos = torch.full((2, 12), 2.0)
manager.current_joint_vel = torch.full((2, 12), 3.0)
manager.step_joint_vel = torch.full((2, 12), 4.0)
```

Assert that:

```python
parallelism_ref_joint_pos_rel_t(env) == 1.0
parallelism_ref_joint_vel_t(env) == 3.0
reference_joint_pos_reward(env) uses 2.0
reference_joint_vel_reward(env) uses 4.0
```

- [ ] **Step 2: Run the tests and verify they fail before changing consumers.**

Run:

```bash
pytest -q tests/tracking/test_parallelism_tracking_mdp.py -k "snapshot or reference_observation"
```

Expected: FAIL because the existing consumers still use `current_*`.

- [ ] **Step 3: Change the policy joint-position observation to `next_joint_pos`.**

Keep the public observation function name and shape unchanged so the actor/critic configuration remains compatible:

```python
ref = manager.next_joint_pos
```

Keep joint velocity and root velocity observation functions on current-to-next velocity.

- [ ] **Step 4: Change reward and metric reference reads to `step_*`.**

Only replace reference source expressions:

```python
manager.current_joint_pos -> manager.step_joint_pos
manager.current_joint_vel -> manager.step_joint_vel
manager.current_foot_pos_w -> manager.step_foot_pos_w
manager.current_root_lin_vel_b_policy -> manager.step_root_lin_vel_b_policy
manager.current_root_ang_vel_b_policy -> manager.step_root_ang_vel_b_policy
```

Do not change the Gaussian reward, tolerance, coordinate conversion, cache, mean, max, or episode accumulation code.

- [ ] **Step 5: Change termination reference reads to `step_*`.**

Use the cached target frame for root, joint, foot, and projected-gravity reference values. Keep the boolean comparisons and thresholds unchanged in this task.

- [ ] **Step 6: Run the MDP tests.**

Run:

```bash
pytest -q tests/tracking/test_parallelism_tracking_mdp.py
```

Expected: all tests pass with the same reward and metric formulas, but reference data comes from the action-step snapshot.

- [ ] **Step 7: Commit the consumer alignment.**

```bash
git add Go2Pvcnn/tracking/mdp/observations.py \
  Go2Pvcnn/tracking/mdp/rewards.py \
  Go2Pvcnn/tracking/mdp/terminations.py
git commit -m "fix: align parallelism tracking consumers to action target"
```

### Task 4: Apply training thresholds and hash-based log directories

**Files:**
- Modify: `Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py`
- Modify: `Go2Pvcnn/tests/test_train_script_static.py`

**Interfaces:**
- Training output is `logs/rsl_rl/<experiment>/<date-time>/<git-short-hash>/`.
- No `tag.txt` is created.
- Existing old run directories remain loadable through explicit `--load_run`.

- [ ] **Step 1: Add failing static tests for thresholds and output layout.**

Assert the config source contains:

```python
"threshold": 0.50
"threshold": 1.50
"threshold": 0.30
"threshold": 2.00
"ang_vel_threshold": 1.0
"joint_max_threshold": 1.0
```

Assert the train source constructs a short Git hash directory below the timestamp directory and does not write `tag.txt`.

- [ ] **Step 2: Run the static tests and verify the threshold/layout assertions fail.**

Run:

```bash
pytest -q tests/tracking/test_parallelism_tracking_env_cfg_static.py tests/test_train_script_static.py
```

Expected: FAIL only on the new threshold and hash-directory assertions.

- [ ] **Step 3: Update termination thresholds while preserving user reward weights.**

Set:

```python
root_z threshold = 0.50
projected_gravity threshold = 1.50
foot_z threshold = 0.30
joint_pos threshold = 2.00
```

Preserve:

```python
reference_joint_pos.weight = 2.0
reference_foot_pos.weight = 1.75
```

- [ ] **Step 4: Update the two curriculum thresholds.**

Set:

```python
ang_vel_threshold = 1.0
joint_max_threshold = 1.0
```

Leave the curriculum function and all other thresholds unchanged.

- [ ] **Step 5: Add a dependency-free Git short-hash helper to `train.py`.**

Implement:

```python
def get_git_short_hash(repo_root: str | os.PathLike[str]) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
```

If Git lookup fails, raise a clear runtime error instead of silently creating an unversioned run directory.

- [ ] **Step 6: Create the date/hash training directory.**

Replace the current timestamp-only construction with:

```python
date_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
git_hash = get_git_short_hash(GO2PVCNN_ROOT)
log_dir = os.path.join(log_root_path, date_dir, git_hash)
os.makedirs(log_dir, exist_ok=True)
```

For distributed training, synchronize the complete nested `log_dir` through the existing temporary path. Do not create `tag.txt`.

- [ ] **Step 7: Preserve resume compatibility.**

When `--load_run` names a new nested run, resolve:

```text
logs/rsl_rl/<experiment>/<date>/<hash>/
```

When it names an old timestamp-only run, preserve the existing direct path behavior. Automatic latest-run selection must sort both layouts by modification/name without selecting the hash directory as the run itself.

- [ ] **Step 8: Run static tests.**

Run:

```bash
pytest -q tests/tracking/test_parallelism_tracking_env_cfg_static.py tests/test_train_script_static.py
```

Expected: PASS.

- [ ] **Step 9: Commit thresholds and logging changes.**

```bash
git add Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py \
  Go2Pvcnn/scripts/train.py \
  Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py \
  Go2Pvcnn/tests/test_train_script_static.py
git commit -m "feat: version parallelism training runs by git hash"
```

### Task 5: Full verification

**Files:**
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py`
- Test: `Go2Pvcnn/tests/test_train_script_static.py`

- [ ] **Step 1: Run all focused tracking and train tests.**

```bash
pytest -q \
  tests/tracking/test_parallelism_reference_manager.py \
  tests/tracking/test_parallelism_tracking_mdp.py \
  tests/tracking/test_parallelism_tracking_env_cfg_static.py \
  tests/test_train_script_static.py
```

- [ ] **Step 2: Run Python compilation checks for modified modules.**

```bash
python -m py_compile \
  tracking/managers/parallelism_reference_manager.py \
  tracking/env.py \
  tracking/mdp/observations.py \
  tracking/mdp/rewards.py \
  tracking/mdp/terminations.py \
  tracking/parallelism_tracking_env_cfg.py \
  scripts/train.py
```

- [ ] **Step 3: Inspect the final diff and preserve only intended user changes.**

```bash
git diff HEAD~4..HEAD --stat
git status --short
git diff --check HEAD~4..HEAD
```

Confirm the final commits contain the user's reward weights and the updated foot reward behavior, and no `tag.txt` creation.

- [ ] **Step 4: Commit any final test-only adjustments.**

```bash
git add tests
git commit -m "test: verify parallelism rl timing and run layout"
```
