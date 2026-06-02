# T302m Teacher Elevation MPC Semantic Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前 working tree 收敛到 `teacher_elevation_trajectory_mpc_semantic + mpc backend` 单一主线，同时保留 train/play/viewer、low-small MPC、RL participation、semantic contactor、semantic raycaster 的当前效果。

**Architecture:** 先冻结 golden tests 和入口契约，再压平/收窄 runtime 入口，最后清理旧 backend、旧 cfg、旧 debug/probe 代码。生产代码只保留当前 semantic MPC 热路径；debug/probe helper 迁到 `Go2Pvcnn/tests` 或删除。

**Tech Stack:** IsaacLab, RSL-RL, PyTorch, `env_isaacsim` (`/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`), `extension/batch_mpc_planner`, semantic raycaster/contact sensor。

---

## Source Spec

- [../../docs/superpowers/specs/2026-05-31-teacher-elevation-mpc-semantic-cleanup-design.html](../../docs/superpowers/specs/2026-05-31-teacher-elevation-mpc-semantic-cleanup-design.html)
- Low-small 不可回归约束：[../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html)
- MPC-RL 不可回归约束：[../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html](../../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html)

## Global Constraints

- 不提交 git；只修改当前 working tree。
- 不修改 IsaacLab 源码。
- 不新增 MPC loss，除非用户重新批准。
- 不恢复 touchdown hard projection、touchdown snapping、hard foot spacing。
- 不删除当前 semantic raycaster / semantic contactor 的验收能力。
- debug/probe-only 代码不能留在 `Go2Pvcnn/extension/batch_mpc_planner`。
- 需要启动 IsaacLab 的验证使用：

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python ...
```

## Golden Runtime Command

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:0 \
  --num_envs 1024 \
  --max_iterations 10000 \
  --experiment teacher_elevation_trajectory_mpc_semantic \
  --planner-backend mpc
```

## File Responsibility Map

- `Go2Pvcnn/scripts/train.py`: 只保留 semantic MPC training entry。
- `Go2Pvcnn/scripts/play.py`: 只保留 semantic MPC play entry。
- `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`: 只注册 semantic MPC train/play Gym id。
- `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`: 当前唯一 task cfg，最终要自包含，不再继承旧 teacher cfg。
- `Go2Pvcnn/agent/train_cfg.py`: 只返回当前 semantic MPC train cfg。
- `Go2Pvcnn/extension/trajectory_manager_factory.py`: 只创建 `MpcTrajectoryManager`。
- `Go2Pvcnn/extension/batch_mpc_planner/`: 只保留生产 MPC planner 热路径。
- `Go2Pvcnn/tests/`: 保留当前 golden tests/probes；debug variant 和 probe helper 只能放这里。

---

### Task 1: Freeze Cleanup Guard Tests

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- Modify: `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`

- [x] **Step 1: Add/adjust cleanup guard tests**

Add tests that assert:

```python
def test_cleanup_entrypoints_only_expose_mpc_semantic_experiment():
    train_source = (REPO_ROOT / "Go2Pvcnn/scripts/train.py").read_text()
    play_source = (REPO_ROOT / "Go2Pvcnn/scripts/play.py").read_text()
    register_source = (REPO_ROOT / "Go2Pvcnn/go2_pvcnn/tasks/register_envs.py").read_text()
    assert "teacher_elevation_trajectory_mpc_semantic" in train_source
    assert "teacher_elevation_trajectory_mpc_semantic" in play_source
    assert "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0" in register_source
    assert "teacher_without_semantic" not in train_source
    assert "teacher_semantic" not in train_source
    assert "teacher_elevation_semantic_map" not in train_source
    assert "Isaac-Teacher-Without-Semantic-Go2-v0" not in register_source
```

Add tests that assert:

```python
def test_cleanup_mpc_factory_has_no_legacy_or_together_backend():
    source = (REPO_ROOT / "Go2Pvcnn/extension/trajectory_manager_factory.py").read_text()
    assert "MpcTrajectoryManager" in source
    assert "batched_together_planner" not in source
    assert "batched_planner" not in source
    assert '"legacy"' not in source
    assert '"together"' not in source
```

Add tests that assert:

```python
def test_cleanup_batch_mpc_planner_has_no_debug_variants_module():
    root = REPO_ROOT / "Go2Pvcnn/extension/batch_mpc_planner"
    assert not (root / "debug_variants.py").exists()
    combined = "\n".join(path.read_text() for path in root.rglob("*.py"))
    assert "debug_loss_variant" not in combined
    assert "apply_mpc_debug_variant_cfg" not in combined
```

- [x] **Step 2: Run guard tests and record current failures**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_entrypoints_only_expose_mpc_semantic_experiment \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_mpc_factory_has_no_legacy_or_together_backend \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_batch_mpc_planner_has_no_debug_variants_module -q
```

Expected before implementation: FAIL, because old experiment/backend/debug routes still exist.

---

### Task 2: Narrow Train/Play/Register Entrypoints

**Files:**
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/scripts/play.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/__init__.py`

- [x] **Step 1: Simplify `train.py` experiment choices**

Keep only:

```python
parser.add_argument(
    "--experiment",
    type=str,
    default="teacher_elevation_trajectory_mpc_semantic",
    choices=["teacher_elevation_trajectory_mpc_semantic"],
    help="Experiment: teacher_elevation_trajectory_mpc_semantic (MPC + semantic grid trajectory reward).",
)
parser.add_argument(
    "--planner-backend",
    type=str,
    default="mpc",
    choices=["mpc"],
    help="Trajectory planner backend. Cleanup build supports only mpc.",
)
```

- [x] **Step 2: Simplify `train.py` imports and env map**

Inside `main()`, import only:

```python
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
)
```

Set:

```python
EXPERIMENT_ENV_MAP = {
    "teacher_elevation_trajectory_mpc_semantic": (
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
        "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
    ),
}
```

Keep planner verbosity/backend assignment but it now only applies to the single experiment.

- [x] **Step 3: Simplify `play.py` experiment choices/imports/map**

Keep only `teacher_elevation_trajectory_mpc_semantic` in parser choices, imports, and `experiment_play_map`.

- [x] **Step 4: Simplify `register_envs.py`**

Remove old cfg imports and old `gym.register(...)` blocks. Keep only:

```python
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
    TeacherElevationTrajectoryMpcSemanticEnvCfg,
    TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
)
```

and the two semantic MPC Gym ids.

- [x] **Step 5: Simplify `tasks/__init__.py`**

Remove exports of old cfg classes. Keep registration import.

- [x] **Step 6: Run entrypoint guard tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_entrypoints_only_expose_mpc_semantic_experiment -q
```

Expected: PASS.

---

### Task 3: Make Trajectory Manager Factory MPC-Only

**Files:**
- Modify: `Go2Pvcnn/extension/trajectory_manager_factory.py`

- [x] **Step 1: Remove backend branching**

Set:

```python
VALID_PLANNER_BACKENDS = ("mpc",)
TRAJECTORY_MANAGER_EXPERIMENTS = ("teacher_elevation_trajectory_mpc_semantic",)
```

Make `create_trajectory_manager()` always import and return `MpcTrajectoryManager`.

- [x] **Step 2: Keep command/reset hooks unchanged**

Do not change `_wrap_command_hook`, `_wrap_env_reset`, or selected-env manager behavior.

- [x] **Step 3: Run factory guard test**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_mpc_factory_has_no_legacy_or_together_backend -q
```

Expected: PASS.

---

### Task 4: Remove Batch MPC Debug Variants From Production Package

**Files:**
- Move/delete: `Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify or delete tests/probes importing production debug variants.

- [x] **Step 1: Remove production imports**

Remove from `planner.py`:

```python
from .debug_variants import apply_mpc_debug_variant_cfg
```

Remove the block:

```python
if cfg.debug_loss_variant not in (None, "", "baseline") and not bool(getattr(cfg, "debug_loss_variant_cfg_applied", False)):
    cfg = apply_mpc_debug_variant_cfg(cfg, cfg.debug_loss_variant, command=command)
```

- [x] **Step 2: Remove debug fields from `MpcPlannerCfg`**

Remove fields named:

```python
debug_loss_variant
debug_loss_variant_cfg_applied
```

- [x] **Step 3: Remove viewer debug variant CLI dependency**

In `go2_foostep_planner.py`, stop importing `apply_mpc_debug_variant_cfg` from production. If the current viewer still exposes `--mpc-debug-variant`, remove the argument or make it reject non-baseline values with a clear error.

- [x] **Step 4: Delete production debug variants file**

Delete:

```text
Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py
```

If a probe still needs the code, move the required helper into:

```text
Go2Pvcnn/tests/fixtures/mpc_debug_variants.py
```

- [x] **Step 5: Run debug guard**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_batch_mpc_planner_has_no_debug_variants_module -q
```

Expected: PASS.

---

### Task 5: Flatten Semantic MPC Env Cfg Inheritance

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Reference-read only during implementation: old teacher cfg files.

- [x] **Step 1: Copy required base scene/env cfg into current module**

Inline the required pieces from old parent cfgs:

- robot asset cfg
- terrain importer cfg
- contact force sensor
- sky light
- commands
- actions
- events
- terminations
- curriculum
- locomotion base rewards
- sim/env base parameters

- [x] **Step 2: Keep current semantic MPC overrides**

Preserve:

```python
planner_owned_reference_cache = True
use_batched_reference_trajectory = True
planner_backend = "mpc"
reference_height_scanner_name = "semantic_height_scanner"
reference_trajectory_horizon = 25
reference_replan_interval_steps = 25
mpc_parallel_plan_batch_size = 64
```

- [x] **Step 3: Remove imports from old teacher cfg modules**

The current cfg must no longer import:

```python
teacher_elevation_trajectory_env_cfg
teacher_elevation_env_cfg
teacher_semantic_env_cfg
teacher_without_semantic_env_cfg
```

- [x] **Step 4: Run cfg import smoke**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import TeacherElevationTrajectoryMpcSemanticEnvCfg
cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
assert cfg.planner_backend == "mpc"
assert cfg.reference_trajectory_horizon == 25
assert cfg.reference_replan_interval_steps == 25
print("cfg import ok")
PY
```

Expected: prints `cfg import ok`.

---

### Task 6: Delete Old Backend And Old Experiment Files

**Files:**
- Delete: `Go2Pvcnn/extension/batched_planner/`
- Delete: `Go2Pvcnn/extension/batched_together_planner/`
- Delete old task cfg files not imported by current semantic MPC cfg.

- [x] **Step 1: Verify no production imports remain**

Run:

```bash
rg -n "batched_planner|batched_together_planner|teacher_without_semantic|teacher_semantic|teacher_elevation_semantic_map|teacher_elevation_trajectory_env_cfg" Go2Pvcnn --glob '*.py'
```

Expected: matches only in tests/docs slated for cleanup, not production runtime.

- [x] **Step 2: Delete old backend directories**

Delete old backend directories after imports are gone.

- [x] **Step 3: Delete old experiment cfg files**

Delete old cfg files after semantic MPC cfg is self-contained.

---

### Task 7: Clean Batch MPC Duplicate Logic

Status: deferred intentionally. This cleanup pass did not refactor planner internals because current route cleanup and tests pass, and a duplicate-query refactor could change planner behavior outside the cleanup scope.

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py`
- Modify tests as needed.

- [ ] **Step 1: Identify repeated terrain queries**

Search for repeated `height_at(...)` / `semantic_at(...)` on the same sampled tensors inside one optimization pass.

- [ ] **Step 2: Introduce loss context only if it removes real duplication**

If needed, add a small internal context object that contains:

```python
touchdown_samples
swing_arc_samples
fk_joint_angles
fk_foot_pos
fk_leg_points
low_small_circles
height_samples
semantic_samples
```

- [ ] **Step 3: Keep behavior unchanged**

The refactor must not change loss equations or default weights.

- [x] **Step 4: Run MPC focused tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: PASS after old-test cleanup. Refactor steps above remain deferred; focused regression still passed after cleanup.

---

### Task 8: Clean Tests To Current Mainline

**Files:**
- Modify/delete old tests under `Go2Pvcnn/tests`

- [x] **Step 1: Delete tests for old experiment/backend routes**

Remove tests whose only purpose is:

- old `teacher_*` experiment registrations
- `legacy` backend
- `together` backend
- old dense MPC route
- old debug variants in production planner

- [x] **Step 2: Keep golden tests**

Keep or update tests/probes for:

- current train smoke
- current play/viewer smoke
- low-small acceptance
- RL participation/world-foot reward
- semantic contact quantity/drop
- semantic raycaster smoke

- [x] **Step 3: Run focused suite**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py \
  Go2Pvcnn/tests/test_mpc_rl_participation.py \
  Go2Pvcnn/tests/test_semantic_contact_rewards.py \
  Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py -q
```

Expected: PASS.

---

### Task 9: IsaacLab Acceptance

Status: complete on card1. Card3 was blocked by an existing 1024-env long training process, but card1 acceptance passed: semantic contact drop probe exit 0, 1024-env train smoke exit 0, and 1024/64/25-step performance `epoch_seconds=5.8828s`.

**Files:**
- No code edits unless acceptance exposes a real bug.
- Update notes/log after each run.

- [x] **Step 1: Semantic contact small drop probe**

Card1 result: PASS with `CUDA_VISIBLE_DEVICES=1`.

Run:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_contact_robot_drop_probe.py::test_semantic_contact_robot_drop_probe_real_isaaclab_small -q
```

Expected: PASS.

- [x] **Step 2: 1024 env train command smoke**

Card1 result: PASS with `--max_iterations 1`. Full `10000` iteration long run was not executed.

Run the golden runtime command. If full `10000` iterations is too long for smoke, use the same command shape with `--max_iterations 1` first, then report that full long run was not executed.

- [x] **Step 3: Performance acceptance**

Card1 result: PASS, `epoch_seconds=5.882832678034902` for 1024 env / 64 selected MPC env / 25 steps.

Run existing 1024/64/25-step performance probe.

Expected: epoch stays under 10 seconds.

- [x] **Step 4: Update notes**

Update:

- `notes/todo.md`
- this plan file
- `notes/log/index.md`
- one new log file under `notes/log/`

Record commands, GPU, pass/fail, key metrics, and remaining risk.

---

### Task 10: Unify MPC Tuning Entry Under `mpc_planner_cfg`

Status: complete locally on 2026-06-01.

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/extension/mdp/rewards_reference.py`
- Modify: focused probes/tests under `Go2Pvcnn/tests/`

- [x] **Step 1: Make `MpcPlannerCfg` the RL task tuning entry**

`TeacherElevationTrajectoryMpcSemanticEnvCfg` now tunes MPC through:

```python
mpc_planner_cfg: MpcPlannerCfg = field(default_factory=MpcPlannerCfg)
```

Runtime horizon/replan/dt/parallel batch size, diagnostics, low-small FK collision weight, and MPC participation exclude pairs are set on `self.mpc_planner_cfg`.

- [x] **Step 2: Stop production task cfg from exposing duplicated MPC aliases**

Removed the active task cfg fields:

- `reference_trajectory_horizon`
- `reference_replan_interval_steps`
- `mpc_parallel_plan_batch_size`
- `mpc_diagnostics_emit_runtime_counters`
- `mpc_diagnostics_profile_cuda_sync`
- `plan_dt`

Viewer CLI still exposes `--n-frames` / `--plan-dt`, but writes them into `env_cfg.mpc_planner_cfg.runtime`.

- [x] **Step 3: Make planner bridge prefer official config object**

`planner_cfg_from_task_cfg()` now returns a deep copy of `task_cfg.mpc_planner_cfg` when present. The old top-level override path remains only as compatibility for legacy tests/fake configs without `mpc_planner_cfg`.

- [x] **Step 4: Verify focused regressions**

Local result:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q
# 15 passed

pytest Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
# 133 passed, 1 warning
```

Related log: [../log/2026-06-01-1743-t302m-mpc-planner-cfg-unification.md](../log/2026-06-01-1743-t302m-mpc-planner-cfg-unification.md).

---

### Task 11: Remove MPC Participation Include Whitelist Fields

Status: complete locally on 2026-06-01.

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/participation.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/tests/test_mpc_rl_participation.py`

- [x] **Step 1: Remove include fields from `MpcReferenceParticipationCfg`**

Removed:

- `include_terrain_cols`
- `include_terrain_names`
- `include_terrain_rows`

Participation now defaults all envs to eligible and only removes envs through `exclude_pairs`.

- [x] **Step 2: Remove include alias bridge**

Removed legacy `mpc_reference_include_*` mapping from `planner_cfg_from_task_cfg()`.

- [x] **Step 3: Verify focused regressions**

Local result:

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py -q
# 4 passed

pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
# 129 passed, 1 warning
```

Source scan:

```bash
rg -n "include_terrain_cols|include_terrain_names|include_terrain_rows|mpc_reference_include" Go2Pvcnn notes -g '*.py' -g '*.md'
# no matches
```

Related log: [../log/2026-06-01-2228-t302m-remove-participation-include-fields.md](../log/2026-06-01-2228-t302m-remove-participation-include-fields.md).
