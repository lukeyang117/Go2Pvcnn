# Parallelism Distillation DAgger Teacher Ratio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative DAgger-style teacher/student action schedule to the distillation experiment without changing non-distillation PPO training.

**Architecture:** Keep the existing `parallelism_tracking_cross_large_complex_distillation` experiment and add a distillation-only teacher ratio schedule. The environment action becomes a per-env choice between deterministic teacher action and deterministic student action; the supervised target always remains the teacher action. The schedule uses current runner iteration divided by runner total iteration, so regular training uses `it / --max_iterations` and resume continues from the checkpoint iteration.

**Tech Stack:** Python 3.10, PyTorch, project-local RSL-RL, IsaacLab manager-based env, pytest.

## Global Constraints

- Branch: `parallelism-large-obstacles-distillation-rl`.
- Only `parallelism_tracking_cross_large_complex_distillation` uses this schedule.
- Original PPO experiments must not read or depend on `teacher_ratio`.
- Do not use action std for distillation rollout; student rollout uses `act_inference()`.
- Do not add PPO gradients; student updates remain supervised by frozen teacher action.
- Default schedule: `warmup_pct=0.10`, `decay_end_pct=0.80`, `teacher_ratio_min=0.0`.
- Environment action is selected by per-env mask, not action interpolation.

---

### Task 1: Add Static Schedule Tests

**Files:**
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py`

**Interfaces:**
- Consumes: `Distillation._compute_teacher_ratio()`
- Produces: tests proving schedule defaults and config isolation.

- [ ] **Step 1: Add schedule math test**

Add a test that constructs `Distillation` with a small `StudentTeacherCNN`, calls `set_iteration()`, and asserts:

```python
assert ratio_at(0, 100) == 1.0
assert ratio_at(10, 100) == 1.0
assert ratio_at(45, 100) == 0.5
assert ratio_at(80, 100) == 0.0
assert ratio_at(100, 100) == 0.0
```

- [ ] **Step 2: Add train config schedule test**

Assert `get_train_cfg("parallelism_tracking_cross_large_complex_distillation")["algorithm"]` contains:

```python
teacher_ratio_warmup_pct == 0.10
teacher_ratio_decay_end_pct == 0.80
teacher_ratio_min == 0.0
```

- [ ] **Step 3: Run tests and verify expected failure**

Run:

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/Go2Pvcnn
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  tests/tracking/test_parallelism_distillation_static.py \
  tests/tracking/test_parallelism_distillation_env_cfg.py
```

Expected: fail because schedule fields and methods do not exist yet.

---

### Task 2: Implement Distillation Action Mixing

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/distillation.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`

**Interfaces:**
- Produces: `Distillation.set_iteration(iteration: int, total_iterations: int) -> None`
- Produces: `Distillation._compute_teacher_ratio() -> float`
- Produces: `Distillation.act(obs: torch.Tensor, teacher_obs: torch.Tensor) -> torch.Tensor`
- Produces TensorBoard keys through `loss_dict`: `teacher_ratio`, `teacher_action_share`.

- [ ] **Step 1: Extend `Distillation.__init__`**

Add keyword parameters:

```python
teacher_ratio_warmup_pct=0.10
teacher_ratio_decay_end_pct=0.80
teacher_ratio_min=0.0
```

Store clamped float values and initialize:

```python
self.current_iteration = 0
self.total_iterations = 1
self.last_teacher_ratio = 1.0
self.last_teacher_action_share = 1.0
```

- [ ] **Step 2: Add schedule helper**

Implement:

```python
def set_iteration(self, iteration: int, total_iterations: int) -> None:
    self.current_iteration = max(int(iteration), 0)
    self.total_iterations = max(int(total_iterations), 1)

def _compute_teacher_ratio(self) -> float:
    progress = min(max(float(self.current_iteration) / float(self.total_iterations), 0.0), 1.0)
    if progress < self.teacher_ratio_warmup_pct:
        return 1.0
    if progress < self.teacher_ratio_decay_end_pct:
        span = max(self.teacher_ratio_decay_end_pct - self.teacher_ratio_warmup_pct, 1.0e-6)
        ratio = 1.0 - (progress - self.teacher_ratio_warmup_pct) / span
        return max(float(self.teacher_ratio_min), min(1.0, ratio))
    return float(self.teacher_ratio_min)
```

- [ ] **Step 3: Replace stochastic student rollout**

In `Distillation.act()`:

```python
student_action = self.policy.act_inference(obs).detach()
teacher_action = self.policy.evaluate(teacher_obs).detach()
teacher_ratio = self._compute_teacher_ratio()
mask = torch.rand(obs.shape[0], 1, device=obs.device) < teacher_ratio
env_action = torch.where(mask, teacher_action, student_action)
self.transition.actions = env_action
self.transition.privileged_actions = teacher_action
self.last_teacher_ratio = teacher_ratio
self.last_teacher_action_share = float(mask.float().mean().item())
return env_action
```

- [ ] **Step 4: Let runner set schedule iteration**

At the start of each training iteration in `OnPolicyRunner.learn()`, before rollout:

```python
if self.training_type == "distillation" and hasattr(self.alg, "set_iteration"):
    self.alg.set_iteration(it, tot_iter)
```

- [ ] **Step 5: Add schedule metrics to `update()` output**

Return:

```python
"teacher_ratio": self.last_teacher_ratio,
"teacher_action_share": self.last_teacher_action_share,
```

The existing runner already logs all `loss_dict` keys under `Distillation/`.

---

### Task 3: Add Distillation-Only Config Fields

**Files:**
- Modify: `Go2Pvcnn/agent/train_cfg.py`

**Interfaces:**
- Consumes: `Distillation.__init__(teacher_ratio_warmup_pct, teacher_ratio_decay_end_pct, teacher_ratio_min)`
- Produces: distillation-only algorithm config schedule.

- [ ] **Step 1: Add config keys**

Inside `_parallelism_distillation_train_cfg()["algorithm"]`, add:

```python
"teacher_ratio_warmup_pct": 0.10,
"teacher_ratio_decay_end_pct": 0.80,
"teacher_ratio_min": 0.0,
```

- [ ] **Step 2: Keep PPO config unchanged**

Confirm `_teacher_elevation_trajectory_mpc_semantic_train_cfg()` does not contain any `teacher_ratio` key.

---

### Task 4: Verification and Commit

**Files:**
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_train_script_static.py`

**Interfaces:**
- Produces passing static tests and compile checks.

- [ ] **Step 1: Run focused tests**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/Go2Pvcnn
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  tests/tracking/test_parallelism_distillation_static.py \
  tests/tracking/test_parallelism_distillation_env_cfg.py \
  tests/test_train_script_static.py
```

- [ ] **Step 2: Run compile checks**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/rsl_rl/rsl_rl/algorithms/distillation.py \
  Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py \
  Go2Pvcnn/agent/train_cfg.py
```

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn docs/superpowers/plans/2026-08-16-parallelism-distillation-dagger-teacher-ratio.md
git commit -m "feat: add dagger teacher ratio distillation"
```
