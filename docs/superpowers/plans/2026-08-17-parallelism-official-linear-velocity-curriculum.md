# Parallelism Official Linear Velocity Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Parallelism reference-error speed curriculum with Go2Pvcnn's reward-based linear-velocity curriculum, keep yaw commands fixed at `[-1.0, 1.0]`, and guarantee that the current three-dimensional command is passed to Parallelism on every replan.

**Architecture:** The distillation environment will own a curriculum configuration that disables `parallelism_velocity` and registers `go2_mdp.lin_vel_cmd_levels`, which expands only `lin_vel_x` and `lin_vel_y` from `track_lin_vel_xy` reward. `ParallelismReferenceManager` will expose a strict `[batch, 3]` command contract by reading `command_manager.get_command("base_velocity")` at planning time and forwarding `[vx, vy, yaw]` to `plan_trajectory`.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab config classes, pytest, headless Isaac Sim.

## Global Constraints

- `ang_vel_z` must be sampled directly from `[-1.0, 1.0]` and must not be curriculum-controlled.
- Only `lin_vel_x` and `lin_vel_y` may be expanded by curriculum.
- Keep the latest reward weights: `track_lin_vel_xy=2.0`, `track_ang_vel_z=1.5`, both with `std=0.5`.
- Student observation remains reference-free and must not regain `base_lin_vel`.
- Teacher/student assignment, PPO, imitation loss, termination, terrain curriculum, and existing unrelated dirty files must remain unchanged.
- Parallelism planner input must be a contiguous tensor with shape `[batch, 3]` ordered as `[vx, vy, yaw]`.
- A real 1024-environment Isaac Sim run must complete 4 training iterations.

---

### Task 1: Update the design record

**Files:**
- Modify: `docs/superpowers/specs/2026-08-17-parallelism-official-linear-velocity-curriculum-design.html`

**Interfaces:**
- Documents: command manager -> reference manager -> planner data flow.
- Documents: the 1024-environment, 4-iteration acceptance test.

- [ ] **Step 1: Add the command-to-planner contract**

Document that every replan reads:

```python
command = env.command_manager.get_command("base_velocity")[:, :3].contiguous()
plan_trajectory(state, command, terrain, cfg, ...)
```

- [ ] **Step 2: Verify the design document**

Run:

```bash
rg -n "command_manager|get_command|plan_trajectory|1024|4 iteration|ang_vel_z" \
  docs/superpowers/specs/2026-08-17-parallelism-official-linear-velocity-curriculum-design.html
```

- [ ] **Step 3: Commit only the design update**

```bash
git add docs/superpowers/specs/2026-08-17-parallelism-official-linear-velocity-curriculum-design.html
git commit -m "docs: specify parallelism command curriculum interface"
```

### Task 2: Add failing tests for the new curriculum configuration

**Files:**
- Create: `tests/tracking/test_parallelism_official_velocity_curriculum.py`

**Interfaces:**
- Consumes: `tracking.parallelism_cross_large_complex_distillation_env_cfg.ParallelismTrackingCrossLargeComplexDistillationEnvCfg`.
- Produces: regression checks for curriculum registration, command ranges, and reward values.

- [ ] **Step 1: Write the failing tests**

```python
from tracking.parallelism_cross_large_complex_distillation_env_cfg import (
    ParallelismTrackingCrossLargeComplexDistillationEnvCfg,
)


def test_distillation_uses_official_linear_velocity_curriculum_only():
    cfg = ParallelismTrackingCrossLargeComplexDistillationEnvCfg()

    assert cfg.curriculum.parallelism_velocity is None
    assert cfg.curriculum.lin_vel_cmd_levels is not None


def test_distillation_uses_full_yaw_range_without_yaw_curriculum():
    cfg = ParallelismTrackingCrossLargeComplexDistillationEnvCfg()

    assert tuple(cfg.commands.base_velocity.ranges.ang_vel_z) == (-1.0, 1.0)
    assert tuple(cfg.commands.base_velocity.limit_ranges.ang_vel_z) == (-1.0, 1.0)


def test_distillation_keeps_latest_velocity_rewards():
    cfg = ParallelismTrackingCrossLargeComplexDistillationEnvCfg()

    assert cfg.rewards.track_lin_vel_xy.weight == 2.0
    assert cfg.rewards.track_lin_vel_xy.params["std"] == 0.5
    assert cfg.rewards.track_ang_vel_z.weight == 1.5
    assert cfg.rewards.track_ang_vel_z.params["std"] == 0.5
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run:

```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl \
  /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/tracking/test_parallelism_official_velocity_curriculum.py
```

Expected: FAIL because the distillation curriculum does not yet register `lin_vel_cmd_levels` and the yaw range is not forced to `[-1.0, 1.0]`.

### Task 3: Add failing tests for the command-to-planner interface

**Files:**
- Modify: `tests/tracking/test_parallelism_reference_manager.py`

**Interfaces:**
- Consumes: `ParallelismReferenceManager._command`.
- Produces: a strict `[batch, 3]` command tensor for planner calls.

- [ ] **Step 1: Add a test for all three command components**

```python
def test_command_to_planner_contract_reads_latest_vx_vy_yaw():
    env = _fake_env(num_envs=2)
    command = torch.tensor([[0.35, -0.2, 0.8], [-0.4, 0.15, -0.6]])
    env.command_manager = SimpleNamespace(get_command=lambda _name: command)
    manager = ParallelismReferenceManager(env, autostart=False)

    result = manager._command(torch.tensor([1, 0]))

    assert result.shape == (2, 3)
    assert torch.allclose(result, command[[1, 0]])
    assert result.is_contiguous()
```

- [ ] **Step 2: Add a test that extra command channels are not forwarded**

```python
def test_command_to_planner_contract_truncates_extra_channels():
    env = _fake_env(num_envs=1)
    command = torch.tensor([[0.2, 0.1, -0.7, 99.0]])
    env.command_manager = SimpleNamespace(get_command=lambda _name: command)
    manager = ParallelismReferenceManager(env, autostart=False)

    result = manager._command(torch.tensor([0]))

    assert result.shape == (1, 3)
    assert torch.allclose(result, command[:, :3])
```

- [ ] **Step 3: Run the focused tests and verify the expected failure**

Run:

```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl \
  /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/tracking/test_parallelism_reference_manager.py \
    -k "command_to_planner_contract"
```

Expected: FAIL because `_command` currently does not explicitly enforce the three-channel contiguous contract.

### Task 4: Implement the curriculum and command contract

**Files:**
- Modify: `Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py`
- Modify: `Go2Pvcnn/tracking/managers/parallelism_reference_manager.py`

**Interfaces:**
- Produces: `cfg.curriculum.lin_vel_cmd_levels`.
- Produces: `_command(env_ids) -> Tensor[batch, 3]` ordered `[vx, vy, yaw]`.

- [ ] **Step 1: Add a distillation curriculum config**

Add a config class extending the existing mixed-terrain curriculum:

```python
@configclass
class ParallelismTrackingCrossLargeComplexDistillationCurriculumCfg(
    ParallelismTrackingCrossLargeComplexCurriculumCfg
):
    parallelism_velocity = None
    lin_vel_cmd_levels = CurrTerm(go2_mdp.lin_vel_cmd_levels)
```

Use this class as the `curriculum` type of the distillation environment.

- [ ] **Step 2: Force the distillation command yaw range**

In `__post_init__`, keep the existing x/y ranges and set:

```python
self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)
```

Do not add an angular curriculum term.

- [ ] **Step 3: Enforce the planner command contract**

Update `ParallelismReferenceManager._command`:

```python
command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
if command.ndim != 2 or int(command.shape[-1]) < 3:
    raise ValueError("Parallelism command must have shape [batch, 3 or more]")
return command[:, :3].index_select(0, env_ids).contiguous()
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run:

```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl \
  /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/tracking/test_parallelism_official_velocity_curriculum.py \
  tests/tracking/test_parallelism_reference_manager.py \
  -k "official_velocity or command_to_planner_contract or panel_speed_replan"
```

Expected: PASS.

### Task 5: Run the complete tracking test suite

**Files:**
- No additional files.

- [ ] **Step 1: Run all tracking tests**

```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl \
  /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/tracking
```

Expected: all tracking tests pass.

- [ ] **Step 2: Inspect the diff**

```bash
git diff --check
git diff --stat
git status --short
```

Confirm that the pre-existing dirty script and reward configuration changes are preserved.

### Task 6: Run the 1024-environment real Isaac Sim smoke test

**Files:**
- Use: `scripts/train.py`
- Use: `scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh`

- [ ] **Step 1: Launch a bounded 1024-environment headless run**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn

export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=0
export ISAAC_ENV=/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim
export LD_LIBRARY_PATH="$ISAAC_ENV/lib/python3.10/site-packages/torch/lib:$ISAAC_ENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$ISAAC_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$ISAAC_ENV/lib/python3.10/site-packages/nvidia/cuda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl \
  "$ISAAC_ENV/bin/python" Go2Pvcnn/scripts/train.py \
  --experiment parallelism_tracking_cross_large_complex_distillation \
  --num_envs 1024 \
  --max_iterations 4 \
  --headless \
  --device cuda:0 \
  --teacher_checkpoint /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/parallelism_tracking_cross_large_complex/model_12199.pt
```

- [ ] **Step 2: Verify runtime behavior**

The run must:

- initialize 1024 environments;
- report student, teacher, and critic observation dimensions without mismatch;
- complete collection for all 4 iterations;
- complete PPO and imitation updates;
- show no planner command shape error;
- write TensorBoard scalars for `track_lin_vel_xy`, `track_ang_vel_z`, and the curriculum term.

- [ ] **Step 3: Commit the implementation**

```bash
git add Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py \
  Go2Pvcnn/tracking/managers/parallelism_reference_manager.py \
  tests/tracking/test_parallelism_official_velocity_curriculum.py \
  tests/tracking/test_parallelism_reference_manager.py
git commit -m "feat: align parallelism command curriculum interface"
```

