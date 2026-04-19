# Viewer Runtime Diagnostics Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build diagnostics-oriented tests that localize whether the current viewer failures originate in command injection, planner stages, final trajectory output, or playback-to-robot synchronization.

**Architecture:** Extend the existing batched planner test suite rather than creating a parallel harness. Add one shared diagnostics helper module for numeric metrics and lightweight runtime fixtures, then add two test files: one for viewer/runtime-path diagnostics and one for planner stage diagnostics, plus an optional batched benchmark smoke. Keep the main path batched and device-local so the tests remain representative of the Isaac Lab runtime under `env_isaaclab`.

**Tech Stack:** Python, pytest, torch, Isaac Lab runtime fixtures, existing `extension.batched_planner.*` instrumentation, existing `Go2Pvcnn/tests` helpers.

---

## File Structure

- Create: `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
  - Shared helper module for command cases, batched metrics, stage-summary extraction, fake/runtime fixture adapters, and playback sync helpers.
- Create: `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
  - Runtime-facing diagnostics covering `WASD/QE/standstill`, playback/readback consistency, leg ordering, and batched smoke assertions.
- Create: `Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py`
  - Planner-internal diagnostics covering `input`, `standstill`, `footholds`, `touchdown_eval`, `swing_targets`, `base_approx`, `terrain_est`, `base_solve`, `ik`, `fk`, `mix`, and final result deltas.
- Optional create/modify: `Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py`
  - Batched timing smoke if the runtime diagnostics need a dedicated perf probe.
- Modify only if tests reveal a visibility gap: `Go2Pvcnn/extension/batched_planner/trajectory.py`
  - Add opt-in diagnostics wrapper output only if current instrumentation cannot expose required stage snapshots without changing planner semantics.

Helper boundary rules:

- `make_real_runtime_fixture(...)`
  - boots `TeacherElevationTrajectoryEnvCfg_PLAY` in headless mode under `/home/lhy/anaconda3/envs/env_isaaclab`
  - owns numeric command injection, planner invocation, playback apply, sync/update, and authoritative robot readback
- `make_unit_test_runtime_adapter(...)`
  - provides fast fake env/robot objects for logic-level tests only
  - must never satisfy tests whose name or assertions require the real runtime boundary
- runtime tests must call the real fixture explicitly, so they cannot accidentally pass with pure stubs

## Task 1: Build Shared Diagnostics Fixtures

**Files:**
- Create: `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
- Test: `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py`

- [ ] **Step 1: Write the failing fixture-consumer test**

```python
def test_runtime_fixture_exposes_headless_command_injection_contract():
    from Go2Pvcnn.tests.fixtures.viewer_runtime_diagnostics import build_command_cases

    cases = build_command_cases(device=torch.device("cpu"), num_envs=1)

    assert "forward" in cases
    assert cases["forward"].shape == (1, 3)
    assert torch.linalg.norm(cases["forward"], dim=-1).item() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_runtime_fixture_exposes_headless_command_injection_contract -v`
Expected: FAIL because `viewer_runtime_diagnostics` helper module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_command_cases(*, device: torch.device, num_envs: int) -> dict[str, torch.Tensor]:
    base = torch.zeros((num_envs, 3), dtype=torch.float64, device=device)
    return {
        "standstill": base.clone(),
        "forward": base.clone().index_fill_(1, torch.tensor([0], device=device), 0.3),
        "yaw_left": base.clone().index_fill_(1, torch.tensor([2], device=device), 0.6),
    }
```

Then expand it to include the full command set and helper dataclasses for:
- command metadata
- batched plan motion summaries
- touchdown displacement summaries
- playback readback error summaries
- stage snapshot containers
- real-runtime fixture setup and teardown helpers
- explicit playback sync helper that performs:
  1. robot state write
  2. scene flush if required
  3. sim/scene update step
  4. authoritative robot data refresh
  5. post-sync readback tensor collection

- [ ] **Step 4: Run targeted tests to verify they pass**

Run: `pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_runtime_fixture_exposes_headless_command_injection_contract -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py
git commit -m "test: add viewer runtime diagnostics fixtures"
```

## Task 2: Add Viewer Runtime Diagnostics Tests

**Files:**
- Create: `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
- Modify: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py` only if a tiny shared fake robot/env helper should be reused instead of duplicated
- Test: `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`

- [ ] **Step 1: Write the failing runtime diagnostics tests**

```python
def test_viewer_forward_command_changes_plan_motion_metrics():
    fixture = make_runtime_diagnostics_fixture(device=torch.device("cpu"), num_envs=1)
    result = fixture.replan(command_name="forward")
    metrics = summarize_result_motion(result)
    assert (metrics.plan_dx**2 + metrics.plan_dy**2) ** 0.5 > 1e-4


def test_viewer_yaw_command_changes_yaw_and_touchdown_metrics():
    fixture = make_runtime_diagnostics_fixture(device=torch.device("cpu"), num_envs=1)
    result = fixture.replan(command_name="yaw_left")
    metrics = summarize_result_motion(result)
    assert abs(metrics.plan_dyaw) > 1e-4
    assert metrics.max_touchdown_shift > 1e-4
```

Also write failing tests for:
- `test_viewer_lateral_command_changes_plan_motion_metrics`
- `test_viewer_playback_matches_reference_frame_numeric`
- `test_viewer_standstill_has_no_single_leg_outlier`
- `test_viewer_leg_order_matches_planner_contract`
- `test_viewer_batched_runtime_smoke_preserves_parallel_path`

- [ ] **Step 2: Run tests to verify they fail for the right reason**

Run: `pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -v`
Expected: FAIL because runtime diagnostics helpers and/or playback readback helpers are incomplete or missing.

- [ ] **Step 3: Write minimal implementation**

Implement the runtime fixture in `viewer_runtime_diagnostics.py`:

```python
@dataclass
class RuntimeDiagnosticsFixture:
    env: object
    robot: object
    device: torch.device
    command_cases: dict[str, torch.Tensor]

    def replan(self, *, command_name: str):
        command = self.command_cases[command_name]
        terrain = PlannerTerrain.from_ray_hits(self.env.scene.sensors["height_scanner"].data.ray_hits_w)
        state = planner_state_from_env(self.env)
        return batched_generate_trajectory(terrain, state, command, requested_n_frames=20, dt=0.02)

    def apply_and_readback(self, result, *, frame_idx: int):
        _apply_direct_playback_to_robot(self.robot, result, frame_idx=frame_idx)
        self.sync_scene()
        return self.read_robot_state()
```

Implementation requirements:
- `make_real_runtime_fixture(...)` must construct `TeacherElevationTrajectoryEnvCfg_PLAY` headlessly with no livestream
- the real fixture must expose `inject_command()`, `replan()`, `apply_and_readback()`, and `close()`
- command injection is numeric, headless, and batched
- playback assertions happen only after the explicit sync/readback contract
- metrics stay batched/device-local until the final scalar summary
- batched smoke asserts CUDA/device locality when CUDA is available
- failure messages include command, plan deltas, touchdown shifts, playback errors, and base orientation

- [ ] **Step 4: Run targeted tests to verify they pass**

Run: `pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -v`
Expected: PASS for the new runtime diagnostics tests.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "test: add viewer runtime diagnostics coverage"
```

## Task 3: Add Planner Stage Diagnostics Tests

**Files:**
- Create: `Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py`
- Create or modify: `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
- Modify only if strictly needed: `Go2Pvcnn/extension/batched_planner/trajectory.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py`

- [ ] **Step 1: Write the failing stage diagnostics tests**

```python
def test_planner_stage_outputs_respond_to_forward_command():
    diag = run_stage_diagnostics(command_name="forward", num_envs=1)
    assert diag.stage_values["input"]["cmd_norm"].item() > 0.0
    assert not diag.stage_values["standstill"]["mask"].item()
    assert diag.result_metrics.planar_delta > 1e-4


def test_planner_standstill_stage_outputs_remain_symmetric():
    diag = run_stage_diagnostics(command_name="standstill", num_envs=1)
    assert diag.stage_values["standstill"]["mask"].item()
    assert diag.result_metrics.max_single_leg_deviation <= 1e-2
```

Also write failing tests for:
- `test_planner_stage_outputs_respond_to_yaw_command`
- `test_planner_output_vs_playback_divergence_report`
- one batched `num_envs=32` smoke assertion that stage snapshots remain batched tensors

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py -v`
Expected: FAIL because stage snapshot reporting is not wired into the diagnostics helper yet.

- [ ] **Step 3: Write minimal implementation**

Preferred implementation path:
- reuse `PlannerInstrumentation` stage names already emitted by `batched_generate_trajectory`
- add a diagnostics wrapper in `viewer_runtime_diagnostics.py` that:
  - runs planner once per command case
  - collects stage timing summary
  - captures stage snapshots for `input`, `standstill`, `gait`, `footholds`, `touchdown_eval`, `swing_targets`, `base_approx`, `terrain_est`, `base_solve`, `ik`, `fk`, `mix`, and final `result`
  - records a concrete `result` snapshot object with final motion deltas, touchdown deltas, base orientation summary, and playback divergence summary
  - records standstill/mix masks and final motion metrics

Fallback implementation path if tests prove necessary:

```python
def batched_generate_trajectory_with_diagnostics(..., return_stage_snapshots: bool = False):
    result = batched_generate_trajectory(...)
    if not return_stage_snapshots:
        return result, None
    return result, snapshots
```

Constraints for the fallback:
- opt-in only
- no semantic change to the production planner path
- keep snapshots as batched tensors, not Python-expanded per-env lists
- the snapshot payload must include at minimum:
  - `input_cmd`, `input_cmd_norm`
  - `standstill_mask`
  - `gait.contact_seq`, `gait.touchdown_times`, `gait.stance_time`
  - `footholds.touchdowns`
  - `touchdown_eval.feasible`
  - `swing_targets.foot_targets`
  - `base_approx.pos_xy_approx`, `base_approx.yaw_approx`
  - `terrain_est.roll`, `terrain_est.pitch`, `terrain_est.height`
  - `base_solve.root_pos`, `base_solve.root_quat`
  - `ik.joint_angles`
  - `fk.body_pos_w`
  - `mix.mask`
  - `result` summary tensors

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/extension/batched_planner/trajectory.py
git commit -m "test: add planner stage diagnostics"
```

## Task 4: Add Optional Batched Timing Smoke

**Files:**
- Create: `Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py`
- Test: `Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py`

- [ ] **Step 1: Write the failing benchmark smoke**

```python
def test_viewer_runtime_batched_smoke(bench_device):
    report = run_runtime_batched_smoke(device=bench_device, num_envs=32)
    assert report.plan_ms > 0.0
    assert report.total_ms > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py -v`
Expected: FAIL because the benchmark module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def run_runtime_batched_smoke(*, device: torch.device, num_envs: int):
    fixture = make_runtime_diagnostics_fixture(device=device, num_envs=num_envs)
    start = time.perf_counter()
    result = fixture.replan(command_name="forward")
    mid = time.perf_counter()
    fixture.apply_and_readback(result, frame_idx=0)
    end = time.perf_counter()
    return SimpleNamespace(plan_ms=(mid - start) * 1e3, total_ms=(end - start) * 1e3)
```

Keep this as a smoke/print-style benchmark, not a brittle hard gate unless conservative CUDA thresholds are easy to justify from observed numbers.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py
git commit -m "test: add viewer runtime diagnostics benchmark"
```

## Task 5: Final Verification

**Files:**
- Verify: `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
- Verify: `Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py`
- Verify: `Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py`

- [ ] **Step 1: Run focused diagnostics suite**

Run: `pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py -v`
Expected: PASS

- [ ] **Step 2: Run benchmark smoke if added**

Run: `pytest Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py -v`
Expected: PASS

- [ ] **Step 3: Run one existing adjacent regression suite**

Run: `pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py Go2Pvcnn/tests/test_batched_planner_instrumentation.py -v`
Expected: PASS

- [ ] **Step 4: Review git diff**

Run: `git diff --stat HEAD~1..HEAD`
Expected: only diagnostics tests/helpers and any narrowly scoped opt-in planner diagnostics hooks

- [ ] **Step 5: Commit final verification or fixups**

```bash
git add Go2Pvcnn/tests Go2Pvcnn/extension/batched_planner/trajectory.py
git commit -m "test: finalize viewer diagnostics suite"
```
