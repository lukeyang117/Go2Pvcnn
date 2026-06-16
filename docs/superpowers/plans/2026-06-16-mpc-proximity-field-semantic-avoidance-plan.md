# MPC Proximity Field Semantic Avoidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the internal dense-grid implementation of existing `parametric_semantic_avoidance` with a proximity-field + `grid_sample` implementation so `TeacherElevationTrajectoryMpcSemanticEnvCfg` can run `1024` RL envs with `1024` MPC envs without CUDA OOM.

**Architecture:** Keep the existing MPC loss set and `parametric_semantic_avoidance` key unchanged. Add small internal helpers in the batch MPC planner to build a soft GPU proximity field from the current terrain semantic/height map and sample root/foot/touchdown risk through differentiable `grid_sample`. Preserve the 2026-05-28 low-small acceptance behavior and validate with both focused tensor tests and real IsaacLab 1024/1024 startup/replan checks.

**Tech Stack:** PyTorch, IsaacLab, `env_isaacsim`, existing `Go2Pvcnn/extension/batch_mpc_planner` parametric MPC backend.

---

## Source Spec

- `docs/superpowers/specs/2026-06-16-mpc-proximity-field-semantic-avoidance-design.md`
- `docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html`

## File Structure

- Modify `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - Replace `_parametric_semantic_avoidance_loss()` internals.
  - Add private helpers for risky mask, proximity field, world-xy sampling, and shape-safe flatten/unflatten.
  - Keep `_parametric_sampled_frame_losses()` return keys unchanged.
- Modify `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - Add static/contract tests that loss keys and config loss terms do not change.
- Modify `Go2Pvcnn/tests/test_batch_mpc_parametric.py`
  - Add pure tensor tests for proximity field shape, risk ordering, zero-obstacle behavior, gradient through query coordinates, and no dense `[B,H,4,H*W]` allocation pattern.
- Modify `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py`
  - Add CLI options or a focused mode for `num_envs=1024`, `mpc_num_envs=1024`, `TeacherElevationTrajectoryMpcSemanticEnvCfg`, at least one replan, and CUDA memory reporting.
- Modify `Go2Pvcnn/tests/test_mpc_runtime_headless.py` or add a new test file if the existing runtime fixture is too heavy
  - Add a real IsaacLab smoke/probe wrapper for the 1024/1024 acceptance command.
- Update notes after implementation and verification
  - `notes/todo.md`
  - relevant branch page, likely `notes/todo/T302u-semantic-map-contact-collision-plan.md` or a new child page if the work expands
  - `notes/log/index.md`
  - one per-verification log under `notes/log/`

## Constraints

- Do not add a new MPC loss term.
- Do not add a new loss name.
- Do not remove `parametric_semantic_avoidance`.
- Do not change `decode_parametric_trajectory()` semantics.
- Do not add decode-time projection, touchdown snapping, root/foot snapping, or hard foot separation.
- Do not relax low-small acceptance thresholds.
- Do not solve the 1024/1024 target by lowering `semantic_height_scanner` resolution or by making `mpc_num_envs < num_envs`.
- Weight tuning is allowed only through existing loss/config fields.

---

### Task 1: Contract Tests For No New Loss Surface

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- Read: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Read: `Go2Pvcnn/extension/batch_mpc_planner/config.py`

- [x] **Step 1: Add a static test that `parametric_semantic_avoidance` remains an existing key**

Add a test that parses `planner.py` and asserts `_parametric_sampled_frame_losses()` still returns `"parametric_semantic_avoidance"` and does not add a new `"proximity"` or `"distance_field"` cost key.

```python
def test_mpc_semantic_avoidance_keeps_existing_loss_key_only():
    source = Path("Go2Pvcnn/extension/batch_mpc_planner/planner.py").read_text()
    assert '"parametric_semantic_avoidance"' in source
    forbidden_keys = (
        '"parametric_proximity"',
        '"parametric_distance_field"',
        '"semantic_proximity"',
        '"semantic_distance_field"',
    )
    for key in forbidden_keys:
        assert key not in source
```

- [x] **Step 2: Add a config test that no new `MpcLossesCfg` field is introduced**

Capture the current expected loss field names from `MpcPlannerCfg().losses` and assert the implementation does not add a field for proximity/distance field.

```python
def test_mpc_cfg_does_not_add_proximity_loss_term():
    from extension.batch_mpc_planner.config import MpcPlannerCfg

    fields = set(vars(MpcPlannerCfg().losses).keys())
    assert "semantic_proximity" not in fields
    assert "distance_field" not in fields
    assert "proximity_field" not in fields
    assert "semantic_contact_avoid" in fields
```

- [x] **Step 3: Run the new tests to confirm RED/GREEN behavior**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_avoidance_keeps_existing_loss_key_only \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_cfg_does_not_add_proximity_loss_term -q
```

Expected before implementation: first test may pass if only checking existing key, but it protects against later accidental new keys. Second test should pass and remain passing.

---

### Task 2: Pure Tensor Tests For Proximity Field Helpers

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`

- [x] **Step 1: Add failing tests for helper imports**

Add tests importing the intended private helpers:

```python
from extension.batch_mpc_planner.planner import (
    _build_semantic_proximity_field,
    _sample_proximity_field_at_world_xy,
)
```

Expected before implementation: import fails.

- [x] **Step 2: Add shape and zero-obstacle tests**

Use a small `8x8` terrain so the test is fast.

```python
def test_semantic_proximity_field_shape_and_zero_obstacle():
    height = torch.zeros((2, 8, 8), dtype=torch.float32)
    semantic = torch.zeros((2, 8, 8), dtype=torch.long)
    root_xy = torch.zeros((2, 2), dtype=torch.float32)
    command = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    root_yaw = torch.zeros(2)

    field = _build_semantic_proximity_field(
        height_map=height,
        semantic_map=semantic,
        root_xy=root_xy,
        root_ground_z=torch.zeros(2),
        command=command,
        root_yaw=root_yaw,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )

    assert field.shape == (2, 1, 8, 8)
    assert torch.count_nonzero(field).item() == 0
```

- [x] **Step 3: Add risk ordering test**

Place one obstacle in front of env0 and assert near query risk is greater than far query risk.

```python
def test_semantic_proximity_field_near_obstacle_has_higher_risk():
    height = torch.zeros((1, 8, 8), dtype=torch.float32)
    semantic = torch.zeros((1, 8, 8), dtype=torch.long)
    semantic[0, 4, 5] = 2
    command = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    root_yaw = torch.zeros(1)

    field = _build_semantic_proximity_field(
        height_map=height,
        semantic_map=semantic,
        root_xy=torch.zeros((1, 2)),
        root_ground_z=torch.zeros(1),
        command=command,
        root_yaw=root_yaw,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )
    near = _sample_proximity_field_at_world_xy(
        field,
        torch.tensor([[[0.10, 0.00]]], dtype=torch.float32),
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )
    far = _sample_proximity_field_at_world_xy(
        field,
        torch.tensor([[[-0.35, -0.35]]], dtype=torch.float32),
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )
    assert near.item() > far.item()
```

- [x] **Step 4: Add gradient-through-query test**

```python
def test_proximity_sampling_has_gradient_to_query_xy():
    field = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    field[:, :, :, 4:] = 1.0
    points = torch.tensor([[[0.05, 0.0]]], dtype=torch.float32, requires_grad=True)

    risk = _sample_proximity_field_at_world_xy(
        field,
        points,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )
    risk.sum().backward()

    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.count_nonzero(points.grad).item() > 0
```

- [x] **Step 5: Run tests to confirm RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py::test_semantic_proximity_field_shape_and_zero_obstacle \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py::test_semantic_proximity_field_near_obstacle_has_higher_risk \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py::test_proximity_sampling_has_gradient_to_query_xy -q
```

Expected: FAIL because helpers do not exist yet.

---

### Task 3: Implement Proximity Helpers

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`

- [x] **Step 1: Add `_grid_world_xy_from_height_shape()` or reuse existing `_terrain_grid_world_xy()`**

Use the existing terrain-grid coordinate convention where possible. If helper tests need standalone shape input, add a private helper that mirrors `_terrain_grid_world_xy()` with explicit height/width.

- [x] **Step 2: Implement `_build_semantic_proximity_field()`**

Implementation requirements:

```python
def _build_semantic_proximity_field(
    *,
    height_map: Tensor,
    semantic_map: Tensor,
    root_xy: Tensor,
    root_ground_z: Tensor,
    command: Tensor,
    root_yaw: Tensor,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
) -> Tensor:
    # Implement in Task 3 Step 2 using the behavior requirements below.
    raise NotImplementedError
```

Behavior:

- Accept `[B,H,W]` height and semantic maps.
- Treat `semantic == 2` as risky large obstacle.
- Treat `semantic == 1` as risky only when height relative to root ground exceeds the existing high-small threshold used by `_parametric_semantic_avoidance_loss()`.
- Apply current command-frame corridor gating.
- Use `torch.nn.functional.max_pool2d` for multi-scale soft field.
- Return `[B,1,H,W]`.
- Do not call `.cpu()`.
- Do not use Python loops over envs.
- Do not detach query coordinates.

- [x] **Step 3: Implement `_sample_proximity_field_at_world_xy()`**

Implementation requirements:

```python
def _sample_proximity_field_at_world_xy(
    field: Tensor,
    points_xy: Tensor,
    *,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
) -> Tensor:
    # Implement in Task 3 Step 3 using the behavior requirements below.
    raise NotImplementedError
```

Behavior:

- Accept `points_xy` shape `[B,...,2]`.
- Flatten point dimensions to `[B,P,2]`.
- Convert world xy to `grid_sample` normalized coordinates.
- Call `F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)`.
- Return original point shape without the final xy dimension.

- [x] **Step 4: Run helper tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py::test_semantic_proximity_field_shape_and_zero_obstacle \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py::test_semantic_proximity_field_near_obstacle_has_higher_risk \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py::test_proximity_sampling_has_gradient_to_query_xy -q
```

Expected: PASS.

---

### Task 4: Replace Dense Semantic Avoidance Internals

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`

- [x] **Step 1: Add a regression test that `_parametric_semantic_avoidance_loss()` returns `[B]` and backpropagates**

Construct a small terrain, simple root/foot/touchdown tensors with `requires_grad=True`, and assert:

```python
loss.shape == (batch,)
loss.sum().backward()
root_pos.grad is not None
foot_pos.grad is not None
```

- [x] **Step 2: Replace only `_parametric_semantic_avoidance_loss()` internals**

Keep the function signature unchanged:

```python
def _parametric_semantic_avoidance_loss(
    terrain: MpcPlannerTerrain,
    root_pos: Tensor,
    foot_pos: Tensor,
    touchdown_w: Tensor,
    command: Tensor,
    *,
    root_yaw: Tensor,
) -> Tensor:
```

Implementation outline:

```text
if terrain.semantic_map is None:
    return zero

normalize height/semantic to [B,H,W]
build risk_field with _build_semantic_proximity_field()
root_risk = sample(root_pos[..., :2])
foot_risk = sample(foot_pos[..., :2])
td_risk = sample(touchdown_w[..., :2])
return 30 * root_mean + 20 * foot_mean + 25 * td_mean, masked to zero where no candidate exists
```

- [x] **Step 3: Avoid dense allocation patterns**

Remove the dense operations:

```python
root_delta = root_pos[..., None, :2] - grid_xy[:, None, :, :]
foot_delta = foot_pos[..., None, :2] - grid_xy[:, None, None, :, :]
touchdown_delta = touchdown_w[..., None, :2] - grid_xy[:, None, :, :]
```

Then add a static test or grep assertion in `test_batch_mpc_parametric.py` that these exact patterns are absent.

- [x] **Step 4: Run focused parametric tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: PASS.

---

### Task 5: Preserve Planner Loss Contracts

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- Run existing: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Add a cost-breakdown key regression**

Use an existing small `plan_segment()` fixture or helper. Assert:

```python
assert "parametric_semantic_avoidance" in result.cost_breakdown
assert "semantic_proximity" not in result.cost_breakdown
assert "semantic_distance_field" not in result.cost_breakdown
```

- [x] **Step 2: Run backend suite**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: PASS.

---

### Task 6: Low-Small Regression Acceptance

**Files:**
- Run existing: `Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py`
- Run existing or selected: `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
- Reference: `docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html`

- [x] **Step 1: Run low-small body/leg collision tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py -q
```

Expected: PASS. Required metrics must remain within existing thresholds:

```text
fk_semantic_collision_count == 0
fk_semantic_collision_rate == 0
planned_vs_fk_foot_error_crossing_leg_max_m <= existing test threshold
```

- [x] **Step 2: Run focused semantic obstacle tests**

Run a focused set that covers low-small/high-large semantic behavior:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q
```

If this file is too broad for one pass, run the existing low-small focused tests selected by `-k "low_small or semantic_avoidance or high_large"`, and record exactly which tests were selected.

- [x] **Step 3: Do not relax thresholds**

If low-small tests fail, investigate root cause. Do not change thresholds to make the tests pass. Only existing loss weights/parameters may be tuned, and any tuning must be recorded.

---

### Task 7: Add 1024 RL Env + 1024 MPC Env Real Probe

**Files:**
- Modify: `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py`
- Possibly modify/add: `Go2Pvcnn/tests/test_mpc_runtime_headless.py` or `Go2Pvcnn/tests/test_mpc_rl_1024_mpc_probe.py`

- [x] **Step 1: Extend probe CLI**

Add arguments:

```text
--num-envs
--mpc-num-envs
--steps
--require-replan
--print-cuda-memory
```

Default for the new acceptance mode:

```text
num_envs=1024
mpc_num_envs=1024
steps >= 30
require_replan=True
cfg class = TeacherElevationTrajectoryMpcSemanticEnvCfg
```

- [x] **Step 2: Record CUDA memory**

Inside the probe, after env creation and after stepping:

```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
    print(f"cuda_max_memory_allocated={torch.cuda.max_memory_allocated()}")
    print(f"cuda_max_memory_reserved={torch.cuda.max_memory_reserved()}")
```

Also print:

```text
num_envs
mpc_num_envs
horizon_steps
replan_interval_steps
parallel_plan_batch_size
sampled_plan_count or runtime_counters
```

- [x] **Step 3: Add a pytest wrapper or documented command**

If a pytest wrapper is practical, add:

```python
def test_teacher_mpc_semantic_1024_env_1024_mpc_probe():
    script = Path("Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py").read_text()
    assert "--num-envs" in script
    assert "--mpc-num-envs" in script
    assert "--require-replan" in script
    assert "--print-cuda-memory" in script
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg" in script
```

Otherwise add a static test that validates the probe exposes the required CLI flags, and run the real probe manually as verification.

- [x] **Step 4: Run real IsaacLab 1024/1024 probe**

Use GPU0 by default; if GPU0 is occupied, switch to an idle GPU and record the replacement in the verification log.

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py \
  --num-envs 1024 \
  --mpc-num-envs 1024 \
  --steps 30 \
  --require-replan \
  --print-cuda-memory
```

Expected:

```text
exit code 0
num_envs=1024
parallel_plan_batch_size=1024
sampled_plan_count > 0 or equivalent replan evidence
no CUDA OOM
no NaN/Inf planner result
cuda memory metrics printed
```

---

### Task 8: Full Focused Verification

**Files:**
- Run: focused tests and pycompile

- [x] **Step 1: Run focused local tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py -q
```

Expected: PASS.

- [x] **Step 2: Run semantic focused regression**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q
```

If runtime is too long, use the low-small/high-large focused selector and record it:

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q -k "low_small or high_large or semantic"
```

- [x] **Step 3: Run pycompile**

Run:

```bash
python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
```

Expected: exit `0`.

- [x] **Step 4: Run diff check**

Run:

```bash
git diff --check
```

Expected: exit `0`.

---

### Task 9: Notes And Logs

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302u-semantic-map-contact-collision-plan.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-06-16-mpc-proximity-field-semantic-avoidance.md`

- [x] **Step 1: Update branch memory**

Record:

```text
problem: dense semantic avoidance constructs B x H x 4 x 22500 tensors
solution: existing parametric_semantic_avoidance now uses proximity field + grid_sample
constraints: no new loss, no loss name change, no low-small threshold relaxation
```

- [x] **Step 2: Create verification log**

Log must include:

```text
purpose
stage
related todo
commands
input conditions
low-small metrics
1024/1024 GPU memory metrics
result
follow-up
git refs
```

- [x] **Step 3: Update log index**

Add a row summarizing:

```text
MPC proximity field semantic avoidance
pass/fail
focused tests
1024 env / 1024 MPC result
CUDA memory
```

---

### Task 10: Final Review Checklist

**Files:**
- Inspect all modified code and tests

- [x] **Step 1: Check no forbidden implementation slipped in**

Run:

```bash
rg -n "projection|snap|hard separation|distance_field|semantic_proximity|parametric_proximity" \
  Go2Pvcnn/extension/batch_mpc_planner Go2Pvcnn/tests
```

Expected:

- No new decode-time projection/snapping implementation.
- No new cost key or config loss term.
- Mentions in tests/specs are acceptable only if they are guard assertions.

- [x] **Step 2: Check dense allocation pattern is gone**

Run:

```bash
rg -n "foot_pos\\[\\.\\.\\., None, :2\\].*grid_xy|grid_xy\\[:, None, None|22500" \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests
```

Expected:

- Old dense semantic avoidance foot/root/touchdown pairwise pattern is absent.
- `22500` appears only in comments/tests/spec explanations, not as a hot-path allocation assumption.

- [x] **Step 3: Summarize remaining risk**

Before closing, report:

```text
which module changed
which contracts stayed unchanged
which low-small acceptance tests passed
whether 1024/1024 real probe passed
GPU memory evidence
unverified parts
```
