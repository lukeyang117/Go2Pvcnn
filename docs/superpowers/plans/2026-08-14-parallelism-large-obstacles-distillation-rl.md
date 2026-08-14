# Parallelism Large Obstacles Distillation RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a teacher-student distillation experiment where the teacher sees Parallelism reference trajectory and the student does not.

**Architecture:** Create a new environment config inheriting `ParallelismTrackingCrossLargeComplexEnvCfg`, adding student/teacher observation groups. Extend the project-local `rsl_rl` with `Distillation` and `StudentTeacherCNN` so existing `ActorCriticCNN` teacher checkpoints can supervise a CNN student without changing the original teacher RL experiment.

**Tech Stack:** Python 3.10, IsaacLab manager-based envs, project-local `Go2Pvcnn/rsl_rl`, PyTorch, TensorBoard, pytest.

## Global Constraints

- Branch: `parallelism-large-obstacles-distillation-rl`.
- New experiment: `parallelism_tracking_cross_large_complex_distillation`.
- Teacher observation includes Parallelism reference trajectory.
- Teacher observation matches the existing PPO teacher checkpoint input and does not add `velocity_commands`; student observation excludes all `parallelism_ref_*` terms but keeps velocity command, proprioception, previous action, elevation/semantic map.
- Environment executes student action during distillation.
- Teacher action is used only as the supervised target.
- Do not change behavior of `parallelism_tracking_cross_large_complex`.
- Keep `parallelism_consecutive_standstill`, `parallelism_geometry_collision`, terrain curriculum, velocity curriculum.
- Use explicit `--teacher_checkpoint` for loading a frozen teacher.
- Save and play distillation checkpoints through the student policy.

---

### Task 1: Add Distillation Algorithm Support

**Files:**
- Create: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/distillation.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/__init__.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`

**Interfaces:**
- Produces: `rsl_rl.algorithms.Distillation`
- Consumes: `RolloutStorage(training_type="distillation", ..., privileged_actions)`

- [x] **Step 1: Write static import test**

```python
def test_project_rsl_rl_exports_distillation():
    from rsl_rl.algorithms import Distillation

    assert Distillation.__name__ == "Distillation"
```

- [x] **Step 2: Implement `Distillation`**

Copy the local site-packages interface shape: `act(obs, teacher_obs)`, `process_env_step`, `update`, and return `{"behavior": mean_behavior_loss, "action_mse": ..., "action_l1": ..., "action_error_max": ...}`.

- [x] **Step 3: Export from `algorithms/__init__.py`**

```python
from .distillation import Distillation
```

- [x] **Step 4: Run test**

Run: `PYTHONPATH=Go2Pvcnn pytest -q Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`

---

### Task 2: Add StudentTeacherCNN

**Files:**
- Create: `Go2Pvcnn/rsl_rl/rsl_rl/modules/student_teacher_cnn.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/modules/__init__.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`

**Interfaces:**
- Produces: `StudentTeacherCNN(num_student_obs, num_teacher_obs, num_actions, **cfg)`
- Produces: `load_state_dict()` that accepts PPO `actor.*` checkpoints and distillation checkpoints.
- Produces: `act()`, `act_inference()`, `evaluate()`, `reset()`, `get_hidden_states()`, `detach_hidden_states()`.

- [x] **Step 1: Write module construction test**

```python
def test_student_teacher_cnn_builds_with_different_obs_dims():
    from rsl_rl.modules import StudentTeacherCNN

    model = StudentTeacherCNN(
        num_student_obs=600,
        num_teacher_obs=650,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={"output_channels": [8, 16], "kernel_size": [3, 3], "max_pool": [True, True], "activation": "elu"},
        student_hidden_dims=[32],
        teacher_hidden_dims=[32],
    )
    assert model.act_inference(torch.zeros(2, 600)).shape == (2, 12)
    assert model.evaluate(torch.zeros(2, 650)).shape == (2, 12)
```

- [x] **Step 2: Implement CNN student/teacher actor builders**

Use the same map split rule as `ActorCriticCNN`: the last `2 * 16 * 16` flattened values are map channels.

- [x] **Step 3: Implement PPO actor checkpoint loading**

For keys starting with `actor_cnns.`, `cnn_encoder.`, or `actor.`, strip the actor prefix only where needed and load into teacher submodules.

- [x] **Step 4: Export from `modules/__init__.py`**

```python
from .student_teacher_cnn import StudentTeacherCNN
```

---

### Task 3: Update Runner for Distillation

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py`

**Interfaces:**
- Consumes: `algorithm.class_name == "Distillation"`
- Consumes: `extras["observations"]["teacher"]`
- Produces: `runner.load_teacher(path)`
- Produces: distillation logging fields.

- [x] **Step 1: Add test that runner source recognizes Distillation**

```python
source = Path("Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text()
assert '"Distillation"' in source
assert "load_teacher" in source
assert 'extras["observations"].get("teacher"' in source
```

- [x] **Step 2: Modify runner initialization**

Detect training type:

```python
self.training_type = "distillation" if self.alg_cfg["class_name"] == "Distillation" else "rl"
```

Use `teacher` observations as privileged observations in distillation.

- [x] **Step 3: Modify learn loop**

For distillation, call `actions = self.alg.act(obs, teacher_obs)` and skip `compute_returns`.

- [x] **Step 4: Modify logging/save/load/inference**

Log `Loss/behavior`, `Distillation/action_mse`, `Distillation/action_l1`, `Distillation/action_error_max`. Save `model_state_dict` normally. `get_inference_policy()` returns student inference for distillation.

---

### Task 4: Add Distillation Environment Config

**Files:**
- Create: `Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py`
- Modify: `Go2Pvcnn/tracking/register_envs.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/scripts/play.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py`

**Interfaces:**
- Produces: `ParallelismTrackingCrossLargeComplexDistillationEnvCfg`
- Produces: `ParallelismTrackingCrossLargeComplexDistillationEnvCfg_PLAY`
- Produces gym ids for train/play.

- [x] **Step 1: Write config tests**

Assert:

```python
cfg.experiment_name == "parallelism_tracking_cross_large_complex_distillation"
cfg.terminations.parallelism_consecutive_standstill is not None
cfg.rewards.parallelism_geometry_collision is not None
cfg.curriculum.terrain_levels is not None
cfg.observations.student_state.parallelism_ref_joint_pos is None
cfg.observations.teacher_state.parallelism_ref_joint_pos is not None
```

- [x] **Step 2: Implement observation classes**

Teacher groups inherit current policy groups and keep `velocity_commands = None` for checkpoint compatibility. Student groups inherit current policy groups, set all `parallelism_ref_*` terms to `None`, and restore `velocity_commands`.

- [x] **Step 3: Register experiment**

Add to train/play experiment choices and gym registration.

---

### Task 5: Add Train Config and Wrapper Flattening

**Files:**
- Modify: `Go2Pvcnn/agent/train_cfg.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/scripts/play.py`
- Test: `Go2Pvcnn/tests/test_train_script_static.py`

**Interfaces:**
- Produces: `get_train_cfg("parallelism_tracking_cross_large_complex_distillation")`
- Produces CLI arg: `--teacher_checkpoint`

- [x] **Step 1: Add train cfg test**

Assert algorithm class is `Distillation`, policy class is `StudentTeacherCNN`, obs groups contain `student` and `teacher`.

- [x] **Step 2: Update wrappers**

Use `train_cfg["obs_groups"]`:

```python
student_obs = flatten(obs_dict, obs_groups["student"])
teacher_obs = flatten(obs_dict, obs_groups["teacher"])
return student_obs, {"observations": {"teacher": teacher_obs}}
```

- [x] **Step 3: Load teacher checkpoint**

If experiment is distillation and not resuming a distillation checkpoint, require `--teacher_checkpoint` and call `runner.load_teacher(path)`.

---

### Task 6: Verification

**Files:**
- Test: `Go2Pvcnn/tests/tracking/parallelism_cross_large_complex_distillation_training_smoke_probe.py`

**Interfaces:**
- Produces: 1024-env smoke probe that confirms distillation loss and iteration increase.

- [x] **Step 1: Run focused pytest**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/Go2Pvcnn
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q \
  tests/tracking/test_parallelism_distillation_static.py \
  tests/tracking/test_parallelism_distillation_env_cfg.py \
  tests/test_train_script_static.py
```

- [x] **Step 2: Run import compile**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py \
  Go2Pvcnn/rsl_rl/rsl_rl/modules/student_teacher_cnn.py \
  Go2Pvcnn/rsl_rl/rsl_rl/algorithms/distillation.py \
  Go2Pvcnn/scripts/train.py \
  Go2Pvcnn/scripts/play.py
```

- [x] **Step 3: Run real smoke**

Use 1024 envs and 4 iterations with an existing teacher checkpoint. Expected: process exits 0 and logs `Loss/behavior`.

- [x] **Step 4: Commit**

```bash
git add Go2Pvcnn docs/superpowers/plans/2026-08-14-parallelism-large-obstacles-distillation-rl.md
git commit -m "feat: add parallelism distillation rl"
```
