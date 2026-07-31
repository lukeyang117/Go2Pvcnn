# Parallelism Flat RL Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a flat-ground Go2 RL tracking task that feeds the current-frame parallelism planner joint/root reference to the lower-level policy.

**Architecture:** Keep `extension/parallelism` as the planner and add training-layer code under `Go2Pvcnn/Go2Pvcnn/tracking/`. A small reference manager owns 24-frame caches, replans on env reset and after 24 consumed frames, and exposes current-frame reference tensors to MDP observations, rewards, terminations, and curriculum.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab manager-based RL env cfg, Gymnasium task registration, local RSL-RL training scripts.

## Global Constraints

- New RL tracking code lives under `Go2Pvcnn/Go2Pvcnn/tracking/`.
- `raw/whole_body_tracking` and `raw/InstinctLab` are reference-only; do not import them and do not commit them.
- Actor and critic both receive `elevation_semantic_map`.
- Policy receives only the current parallelism reference frame, not future frames.
- `base_velocity` command resampling period is `24 * 0.02 = 0.48s`.
- Parallelism replans only on environment reset and after a 24-frame cache is consumed.
- `parallelism_ref_joint_pos_too_far` is enabled by default.
- Real training smoke test must start with 1024 envs and show epoch/iteration increasing.

---

## File Structure

- Create `Go2Pvcnn/tracking/__init__.py`: package marker and task import side effect.
- Create `Go2Pvcnn/tracking/managers/parallelism_reference_manager.py`: 24-frame reference cache, reset/cycle replanning, current-frame getters.
- Create `Go2Pvcnn/tracking/mdp/observations.py`: current-frame reference observation terms.
- Create `Go2Pvcnn/tracking/mdp/rewards.py`: joint position/velocity tracking rewards and curriculum metrics helper.
- Create `Go2Pvcnn/tracking/mdp/terminations.py`: root/foot/joint reference-too-far termination terms.
- Create `Go2Pvcnn/tracking/mdp/curriculums.py`: velocity range curriculum over `base_velocity`.
- Create `Go2Pvcnn/tracking/mdp/__init__.py`: export MDP terms.
- Create `Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`: flat env cfg reusing Go2 assets, semantic scanner, actions, observations, rewards, terminations, curriculum.
- Create `Go2Pvcnn/tracking/register_envs.py`: register `Isaac-Go2-Parallelism-Tracking-Flat-v0`.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`: import tracking registration.
- Modify `Go2Pvcnn/scripts/train.py`: add a task option for parallelism tracking.
- Add tests under `Go2Pvcnn/tests/tracking/`.

---

### Task 1: Static Package And Registration Scaffolding

**Files:**
- Create: `Go2Pvcnn/tracking/__init__.py`
- Create: `Go2Pvcnn/tracking/register_envs.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`

**Interfaces:**
- Produces Gym task id: `Isaac-Go2-Parallelism-Tracking-Flat-v0`
- Produces cfg entry point: `Go2Pvcnn.tracking.parallelism_tracking_env_cfg:ParallelismTrackingFlatEnvCfg`

- [ ] **Step 1: Write failing static registration test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_parallelism_tracking_task_id_is_registered() -> None:
    source = (ROOT / "Go2Pvcnn/tracking/register_envs.py").read_text()
    assert "Isaac-Go2-Parallelism-Tracking-Flat-v0" in source
    assert "ParallelismTrackingFlatEnvCfg" in source


def test_main_task_registration_imports_tracking_registration() -> None:
    source = (ROOT / "Go2Pvcnn/go2_pvcnn/tasks/register_envs.py").read_text()
    assert "Go2Pvcnn.tracking.register_envs" in source
```

- [ ] **Step 2: Run the failing test**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py -q`

Expected: fails because tracking registration files do not exist.

- [ ] **Step 3: Create registration files**

`Go2Pvcnn/tracking/__init__.py` imports `Go2Pvcnn.tracking.register_envs`.

`Go2Pvcnn/tracking/register_envs.py` calls `gym.register(...)` with the task id and cfg entry point.

`Go2Pvcnn/go2_pvcnn/tasks/register_envs.py` imports `Go2Pvcnn.tracking.register_envs` after existing registrations.

- [ ] **Step 4: Run the test**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tracking/__init__.py Go2Pvcnn/tracking/register_envs.py Go2Pvcnn/go2_pvcnn/tasks/register_envs.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py
git commit -m "feat: register parallelism tracking task"
```

---

### Task 2: Parallelism Reference Manager

**Files:**
- Create: `Go2Pvcnn/tracking/managers/__init__.py`
- Create: `Go2Pvcnn/tracking/managers/parallelism_reference_manager.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`

**Interfaces:**
- Produces class `ParallelismReferenceManager(env, cfg: ParallelismCfg | None = None, command_name: str = "base_velocity")`
- Produces method `reset(env_ids: Sequence[int] | torch.Tensor | None = None) -> None`
- Produces method `step() -> None`
- Produces properties `current_joint_pos`, `current_joint_vel`, `current_root_pos_w`, `current_root_rpy_w`, `current_foot_pos_w`, `current_root_lin_vel_b`, `current_root_ang_vel_b`, `phase`

- [ ] **Step 1: Write manager tests with a fake planner callback**

Create tests that instantiate the manager with fake tensors and assert:

```python
manager.reset(torch.tensor([0, 1]))
assert torch.all(manager.phase == 0)
first_plan_count = manager.plan_count.clone()
for _ in range(23):
    manager.step()
assert torch.equal(manager.plan_count, first_plan_count)
manager.step()
assert torch.all(manager.phase == 0)
assert torch.all(manager.plan_count == first_plan_count + 1)
```

Also assert `current_joint_vel` is finite-difference of cached joint positions divided by `dt`.

- [ ] **Step 2: Run failing manager tests**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py -q`

Expected: fails because manager does not exist.

- [ ] **Step 3: Implement manager**

Implement cache tensors shaped:

```python
root_pos_w: [N, 24, 3]
root_rpy_w: [N, 24, 3]
joint_pos: [N, 24, 12]
foot_pos_w: [N, 24, 4, 3]
contact_state: [N, 24, 4]
valid: [N, 24]
phase: [N]
```

Use `extension.parallelism.planner.plan_parallelism` if available; if the function name differs, import the actual planner entry point from `extension.parallelism.planner`.

- [ ] **Step 4: Run manager tests**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tracking/managers Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py
git commit -m "feat: add parallelism reference manager"
```

---

### Task 3: MDP Observations, Rewards, And Terminations

**Files:**
- Create: `Go2Pvcnn/tracking/mdp/__init__.py`
- Create: `Go2Pvcnn/tracking/mdp/observations.py`
- Create: `Go2Pvcnn/tracking/mdp/rewards.py`
- Create: `Go2Pvcnn/tracking/mdp/terminations.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py`

**Interfaces:**
- Observations return tensors from `env.parallelism_reference_manager`.
- Rewards expose `reference_joint_pos_reward(env, asset_cfg, std)` and `reference_joint_vel_reward(env, asset_cfg, std)`.
- Terminations expose `parallelism_ref_joint_pos_too_far`, `parallelism_ref_root_z_too_far`, `parallelism_ref_projected_gravity_too_far`, `parallelism_ref_foot_z_too_far`.

- [ ] **Step 1: Write fake-env MDP tests**

Use a `SimpleNamespace` fake env with:

```python
env.parallelism_reference_manager.current_joint_pos = torch.zeros(2, 12)
env.parallelism_reference_manager.current_joint_vel = torch.zeros(2, 12)
env.scene["robot"].data.joint_pos = torch.zeros(2, 12)
env.scene["robot"].data.joint_vel = torch.zeros(2, 12)
```

Assert observation shapes, reward equals high value when errors are zero, and joint termination triggers when one joint exceeds `0.8`.

- [ ] **Step 2: Run failing tests**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q`

Expected: fails because MDP modules do not exist.

- [ ] **Step 3: Implement MDP modules**

Use only torch operations for thresholds:

```python
error = torch.max(torch.abs(asset.data.joint_pos - ref), dim=-1).values
return error > threshold
```

For reward:

```python
err = torch.mean(torch.square(asset.data.joint_pos - ref), dim=-1)
return torch.exp(-err / (std * std))
```

- [ ] **Step 4: Run tests**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tracking/mdp Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py
git commit -m "feat: add parallelism tracking mdp terms"
```

---

### Task 4: Flat Tracking Env Cfg And Curriculum

**Files:**
- Create: `Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py`
- Create: `Go2Pvcnn/tracking/mdp/curriculums.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py`

**Interfaces:**
- Produces `ParallelismTrackingFlatEnvCfg`.
- Produces `ParallelismTrackingFlatEnvCfg_PLAY`.
- Produces curriculum function `parallelism_velocity_curriculum(env, env_ids, command_name="base_velocity")`.

- [ ] **Step 1: Write static env cfg tests**

Assert source contains:

```python
"resampling_time_range=(0.48, 0.48)"
"downsampled_elevation_semantic_scan"
"parallelism_ref_joint_pos_too_far"
"ParallelismTrackingFlatEnvCfg"
"Isaac-Go2-Parallelism-Tracking-Flat-v0"
```

- [ ] **Step 2: Run failing static tests**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py -q`

Expected: fails because env cfg does not exist.

- [ ] **Step 3: Implement env cfg**

Reuse existing Go2 scene patterns:

- `UNITREE_GO2_CFG`
- flat terrain generator or plane terrain
- `SemanticGridRayCasterCfg`
- `JointPositionActionCfg`
- policy and critic observation groups with state and map groups
- terminations from `Go2Pvcnn.tracking.mdp.terminations`
- rewards from `Go2Pvcnn.tracking.mdp.rewards`

- [ ] **Step 4: Update train script**

Add mapping for `--task parallelism_tracking_flat` or existing task selector style to instantiate `ParallelismTrackingFlatEnvCfg` and task id `Isaac-Go2-Parallelism-Tracking-Flat-v0`.

- [ ] **Step 5: Run static tests**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py -q`

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/tracking/parallelism_tracking_env_cfg.py Go2Pvcnn/tracking/mdp/curriculums.py Go2Pvcnn/scripts/train.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_env_cfg_static.py
git commit -m "feat: add parallelism tracking flat env cfg"
```

---

### Task 5: Integration Smoke Tests

**Files:**
- Create: `Go2Pvcnn/tests/tracking/parallelism_training_smoke_probe.py`
- Test by command.

**Interfaces:**
- Probe starts IsaacLab training with 1024 envs and a tiny iteration count.
- Probe prints lines containing iteration/epoch progress.

- [ ] **Step 1: Add smoke probe script**

The script should call the real training entry point with:

```bash
--task parallelism_tracking_flat
--num_envs 1024
--headless
--max_iterations 2
```

Use environment exports matching current IsaacSim setup:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=0
```

- [ ] **Step 2: Run unit/static tests**

Run:

```bash
pytest Go2Pvcnn/tests/tracking -q
```

Expected: pass.

- [ ] **Step 3: Run real 1024-env training smoke test**

Run:

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn
export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=0
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py --task parallelism_tracking_flat --num_envs 1024 --headless --max_iterations 2
```

Expected: logs show training started and iteration/epoch counter increases beyond zero.

- [ ] **Step 4: Commit**

```bash
git add Go2Pvcnn/tests/tracking/parallelism_training_smoke_probe.py
git commit -m "test: add parallelism training smoke probe"
```

---

## Self-Review

- Spec coverage: plan covers tracking package location, one-frame reference, actor/critic map input, 0.48s command period, reset/cycle replanning, joint-pos termination, velocity curriculum, registration, and 1024-env smoke test.
- Placeholder scan: no task uses TODO/TBD placeholders; implementation details are concrete enough to execute against the current codebase.
- Type consistency: all task interfaces consistently use `env.parallelism_reference_manager` and current-frame tensor property names.
