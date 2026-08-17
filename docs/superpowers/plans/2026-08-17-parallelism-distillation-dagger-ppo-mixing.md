# Parallelism DAgger PPO Mixing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the approved teacher/student DAgger schedule so fresh distillation training uses 100% teacher-controlled environments through 30%, linearly transitions to student-controlled environments through 80%, and then uses 100% student control while retaining PPO and imitation training.

**Architecture:** `HybridDistillationPPO.act()` computes deterministic teacher actions and sampled student actions every step, then selects whole environments with one batch-level teacher ratio. Rollout storage records the selected action and a `ppo_active_mask`; imitation loss uses every sample, critic value loss uses every sample, and PPO actor surrogate uses only samples where student actually controlled the environment. The original PPO experiments and the teacher checkpoint remain isolated from this schedule.

**Tech Stack:** Python, PyTorch, RSL-RL rollout storage, IsaacLab environment wrapper, pytest, Bash.

## Global Constraints

- Default validation is fresh training: do not pass `--resume`, `--load_run`, or `--load_checkpoint`.
- Fresh training loads only the fixed `--teacher_checkpoint`; student actor, student critic, and optimizer start from new initialization.
- Teacher action is deterministic and teacher parameters remain frozen.
- Student training keeps its learnable action standard deviation.
- Student actor does not receive Parallelism reference observations or `base_lin_vel`.
- Critic continues to receive the privileged teacher observation.
- The environment action is selected per environment; teacher and student actions are not numerically averaged.
- `teacher_ratio + student_ratio = 1.0` for every rollout.
- Teacher ratio schedule is 1.0 through 30%, linearly decays to 0.0 from 30% to 80%, and stays 0.0 afterwards.
- Controller assignment is fixed for the lifetime of an episode; a new ratio is applied only after that environment resets.
- IsaacLab `last_action` remains the source of critic action input, so each episode's critic action comes from its fixed controller.
- Imitation loss is computed for all rollout samples.
- Critic value loss is computed for all rollout samples.
- PPO actor surrogate is computed only for student-controlled rollout samples.
- Existing non-distillation PPO configurations must keep their current behavior.

---

### Task 1: Extend rollout storage with action-source masks

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`

**Interfaces:**
- `RolloutStorage.Transition.ppo_active` is an optional tensor shaped `(num_envs, 1)` or `(num_envs,)`.
- `RolloutStorage` stores `ppo_active_masks` with shape `(num_transitions_per_env, num_envs, 1)`.
- `mini_batch_generator(..., include_privileged_actions=True, include_ppo_mask=True)` returns the existing fields followed by the PPO mask.
- Existing PPO and old distillation generator call signatures remain valid when `include_ppo_mask=False`.

- [ ] **Step 1: Write a failing storage test**

Add a test that creates a small `RolloutStorage`, sets `Transition.ppo_active` to a mixed boolean/float mask, adds the transition, and asserts the mini-batch contains the same mask.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py -k ppo_active
```

Expected: FAIL because the transition and generator do not expose the mask yet.

- [ ] **Step 3: Implement the storage change**

Initialize the mask buffer to ones so old PPO behavior remains unchanged:

```python
self.ppo_active_masks = torch.ones(
    num_transitions_per_env, num_envs, 1, device=device
)
```

In `Transition.clear()`, reset `ppo_active` to `None`. In `add_transitions()`, copy `transition.ppo_active` when present and otherwise keep ones. Extend only the optional hybrid generator return path.

- [ ] **Step 4: Run the focused test**

Run the same pytest command and expect PASS. Also run the existing storage/tracking tests to confirm old generator tuple shapes remain unchanged.

- [ ] **Step 5: Commit the storage change**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py
git commit -m "feat: store distillation PPO action-source masks"
```

### Task 2: Implement the 30%/80% environment DAgger schedule

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/hybrid_distillation_ppo.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`

**Interfaces:**
- Constructor parameters:
  - `teacher_ratio_warmup_pct=0.30`
  - `teacher_ratio_decay_end_pct=0.80`
  - `teacher_ratio_min=0.0`
- `_compute_teacher_ratio()` returns the scalar scheduled teacher fraction.
- `act(obs, teacher_obs)` returns the selected environment action and stores:
  - `transition.actions`
  - `transition.privileged_actions`
  - `transition.ppo_active`
  - `transition.action_source_teacher`
- `update()` returns `teacher_ratio`, `student_ratio`, `teacher_action_share`, and `ppo_active_ratio`.

- [ ] **Step 1: Write schedule and action-source tests**

Add tests for:

```python
assert algorithm._compute_teacher_ratio() == 1.0  # progress < 0.30
assert algorithm._compute_teacher_ratio() == 0.5  # progress == 0.55
assert algorithm._compute_teacher_ratio() == 0.0  # progress >= 0.80
```

Use a deterministic `torch.rand` patch or a fixed generator to verify the selected action is teacher for the teacher mask and student for the complement, and that `ppo_active` is zero for teacher rows and one for student rows.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py -k "teacher_ratio or action_source"
```

Expected: FAIL because the current algorithm always returns student action and has no mask.

- [ ] **Step 3: Implement the schedule**

Use the current iteration and command-line-derived total iteration count:

```python
progress = min(max(
    float(self.current_iteration) / float(self.total_iterations), 0.0
), 1.0)
if progress < warmup:
    return 1.0
if progress < decay_end:
    return 1.0 - (progress - warmup) / max(decay_end - warmup, 1e-6)
return teacher_ratio_min
```

Use one scalar ratio for the whole batch. Build a per-environment boolean mask, select teacher or student action with `torch.where`, and set:

```python
ppo_active = (~teacher_mask).to(student_action.dtype)
```

    The default implementation should allocate approximately `round(num_envs * teacher_ratio)` newly reset environment rows, while randomizing row assignment so environment IDs do not permanently identify teacher or student behavior. Once assigned, an environment keeps the same controller until its `done` flag is observed. The actual batch share is logged from the persistent mask.

    The algorithm keeps two buffers:

    ```python
    self._teacher_control_mask: Tensor[num_envs, bool]
    self._needs_control_assignment: Tensor[num_envs, bool]
    ```

    `process_env_step()` marks done environment IDs for reassignment; the next `act()` assigns only those IDs using the current schedule.

- [ ] **Step 4: Run the focused tests**

    Run the focused tests for ratio, action-source persistence, and the complete tracking test set. Expect all schedule and action-source assertions to pass without changing legacy PPO tests.

- [ ] **Step 5: Commit the algorithm change**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/algorithms/hybrid_distillation_ppo.py \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py
git commit -m "feat: add percentage-based teacher student rollout mixing"
```

### Task 3: Mask PPO actor loss while retaining imitation and critic losses

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/hybrid_distillation_ppo.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`

**Interfaces:**
- `HybridDistillationPPO.update()` requests `include_ppo_mask=True`.
- PPO surrogate is reduced over active student samples only; if a mini-batch has no active samples, surrogate and entropy actor terms are zero without NaN.
- Value loss is reduced over all samples.
- Imitation MSE and action L1 are reduced over all samples.
- Runner logs the new algorithm information without changing old PPO logging.

- [ ] **Step 1: Write a failing loss-mask test**

Construct a minimal hybrid algorithm with a batch containing one teacher-controlled row and one student-controlled row. Verify that changing the teacher row advantage does not change the masked PPO surrogate, while changing either row's teacher action changes imitation loss.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py -k loss_mask
```

Expected: FAIL because current PPO loss averages all rows and the generator has no mask.

- [ ] **Step 3: Implement masked reductions**

For the PPO actor terms:

```python
active = ppo_active_mask.reshape(-1) > 0.5
if torch.any(active):
    surrogate_loss = torch.max(
        surrogate[active], surrogate_clipped[active]
    ).mean()
    entropy_loss = entropy_batch.reshape(-1)[active].mean()
else:
    surrogate_loss = actions_log_prob_batch.new_zeros(())
    entropy_loss = actions_log_prob_batch.new_zeros(())
```

Keep value loss and imitation loss over the full batch. Use `ppo_active` only for actor surrogate and entropy. Return the measured active ratio.

- [ ] **Step 4: Update runner logging**

In the hybrid branch of `OnPolicyRunner.learn()`, preserve the current training metrics and add:

```text
Distillation/teacher_ratio
Distillation/student_ratio
Distillation/teacher_action_share
Distillation/ppo_active_ratio
Distillation/action_mse
Distillation/action_l1
```

No `--resume` behavior is changed; this task only ensures fresh training reports the new schedule.

- [ ] **Step 5: Run focused and complete tests**

Run the loss-mask test and then:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/tracking
```

Expected: all existing tracking tests pass.

- [ ] **Step 6: Commit the loss-path change**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/algorithms/hybrid_distillation_ppo.py \
  Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py
git commit -m "feat: mask PPO loss to student-controlled rollouts"
```

### Task 4: Update configuration and fresh-training launcher

**Files:**
- Modify: `Go2Pvcnn/agent/train_cfg.py`
- Modify: `Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh`
- Modify: `docs/superpowers/specs/2026-08-17-parallelism-distillation-ppo-design-zh.html`

**Interfaces:**
- Configuration passes:

```python
"teacher_ratio_warmup_pct": 0.30,
"teacher_ratio_decay_end_pct": 0.80,
"teacher_ratio_min": 0.0,
```

- Fresh launcher passes only:

```text
--experiment parallelism_tracking_cross_large_complex_distillation
--num_envs 1024
--headless
--max_iterations 2000
--teacher_checkpoint <fixed teacher model>
```

- [ ] **Step 1: Add a config assertion test**

Add a static test that reads `get_train_cfg("parallelism_tracking_cross_large_complex_distillation")` without launching Isaac Sim and asserts the algorithm class and three schedule values.

- [ ] **Step 2: Run the test to verify current values fail**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py -k schedule
```

Expected: FAIL until configuration is changed from the previous hybrid defaults.

- [ ] **Step 3: Update the config and launcher**

Set the schedule values to 0.30, 0.80, and 0.0. Ensure the fresh launcher does not contain `--resume`, `--load_run`, or `--load_checkpoint`, and continues to load the fixed teacher with `--teacher_checkpoint`.

- [ ] **Step 4: Run shell and config checks**

Run:

```bash
bash -n Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py -k schedule
```

Expected: shell syntax passes and the schedule assertion passes.

- [ ] **Step 5: Commit the configuration change**

```bash
git add Go2Pvcnn/agent/train_cfg.py \
  Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh \
  docs/superpowers/specs/2026-08-17-parallelism-distillation-ppo-design-zh.html \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py
git commit -m "feat: configure fresh percentage-based distillation training"
```

### Task 5: Verify end-to-end construction without resume

**Files:**
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py`

- [ ] **Step 1: Run compile and diff checks**

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/rsl_rl/rsl_rl/algorithms/hybrid_distillation_ppo.py \
  Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py \
  Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py \
  Go2Pvcnn/agent/train_cfg.py
git diff --check
```

- [ ] **Step 2: Run the complete tracking suite**

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/tracking
```

Expected: all tracking tests pass. Isaac Sim app construction tests may remain unavailable in a plain Python process if `omni.kit` is not importable.

- [ ] **Step 3: Verify launcher semantics**

```bash
rg -n -- "--resume|--load_run|--load_checkpoint|--teacher_checkpoint|max_iterations|num_envs" \
  Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh
```

Expected: only `--teacher_checkpoint` is present among checkpoint/resume flags, with `--num_envs 1024` and `--max_iterations 2000`.

- [ ] **Step 4: Commit the verified implementation**

```bash
git status --short
git log --oneline -5
git commit -am "feat: restore DAgger teacher student PPO rollout schedule"
```
