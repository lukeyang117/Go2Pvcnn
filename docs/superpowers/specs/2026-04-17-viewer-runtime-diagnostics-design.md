# Viewer Runtime Diagnostics Design

## Metadata

- **Date**: 2026-04-17
- **Topic**: Isaac Lab viewer runtime diagnostics for batched planner
- **Status**: Draft for review
- **Primary environment**: `/home/lhy/anaconda3/envs/env_isaaclab`

## 1. Problem Statement

Current viewer behavior indicates a mismatch between teleop intent, planner output, and robot playback:

- `WASD`: `touchdowns` and robot appear not to move at all.
- `QE`: `touchdowns` update and the planned trajectory looks reasonable, but the Go2 body does not move like a walking robot.
- In some cases the robot base appears flipped or strongly misoriented.
- In standstill on flat terrain, one leg appears visibly abnormal.

The immediate goal is **not** to fix the implementation first. The goal is to build a set of diagnostics-oriented tests that can determine where the fault originates:

1. command input did not meaningfully reach the planner
2. planner internal stage output is already wrong
3. final planner result is numerically healthy, but viewer playback is wrong
4. leg ordering or robot state mapping is inconsistent
5. standstill degeneration introduces a single-leg outlier

These tests must stay close to the real Isaac Lab runtime path while avoiding GUI dependence. They should prefer batched/GPU-friendly execution and use numeric output and assertions rather than visual inspection.

## 2. Scope and Non-Goals

### In Scope

- Add new diagnostics-focused tests under `Go2Pvcnn/tests/`
- Exercise the real Isaac Lab environment/runtime path in `env_isaaclab`
- Run headless, without livestream and without interactive visualization
- Measure planner-stage outputs, final trajectory outputs, and playback-to-robot consistency
- Produce failure messages and structured numeric summaries that directly support debugging
- Include lightweight timing metrics so diagnostics do not accidentally force a slow CPU/serial path

### Out of Scope

- Fixing viewer or planner code in this phase
- Building a full end-to-end visual replay harness
- Creating CPU-only fallback diagnostics as the primary path
- Reworking architecture outside what is required to diagnose these viewer failures

## 3. Constraints

1. Tests should run in:
   - `/home/lhy/anaconda3/envs/env_isaaclab`
2. Tests should default to:
   - `headless`
   - no `livestream`
   - no GUI dependency
3. Diagnostics should prefer:
   - real batched tensors
   - CUDA/GPU-friendly execution when available
   - minimal per-env Python loops in hot paths
4. The first objective is localization of faults, not broad feature coverage.

## 4. Design Overview

We will add a diagnostics suite with two complementary layers:

1. **Viewer runtime diagnostics**
   - Verifies what happens across the real runtime path:
     `command -> replan -> trajectory result -> playback apply -> robot state readback`
   - This is the closest numeric analogue to the current viewer failure.

2. **Planner stage diagnostics**
   - Verifies which internal planner stage first diverges from expectation:
     `gait -> foothold -> touchdown_eval -> swing_targets -> base_approx -> terrain_est -> base_solve -> ik/fk`
   - This isolates planner-originated errors from playback-originated errors.

Together these let us answer:

- Is `WASD` failing before or after planning?
- Is `QE` healthy at the planner level but broken at playback?
- Does standstill already contain the one-leg anomaly before robot application?
- Is the base flip introduced in planner output or in viewer playback?

## 5. Test Artifacts

### 5.1 New test files

- `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
- `Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py`

### 5.2 Optional benchmark file

- `Go2Pvcnn/tests/benchmarks/bench_viewer_runtime_diagnostics.py`

The benchmark file is optional and should remain lightweight. It exists to ensure the diagnostics themselves do not silently force an undesirable slow path.

## 6. Runtime Configuration

The diagnostics suite should construct the real play env using the same task family as the viewer path:

- `TeacherElevationTrajectoryEnvCfg_PLAY`
- real `height_scanner`
- real `PlannerTerrain.from_ray_hits(...)`
- real `batched_generate_trajectory(...)`

The suite should configure the environment in a headless, no-livestream mode. The design assumes no visual output is needed. Debugging depends entirely on numeric inspection and assertions.

Recommended default cases:

- `num_envs = 1` for viewer-failure reproduction
- `num_envs = 32` for batched/parallellism smoke checks

The `num_envs = 1` case is primary because the reported bug is viewer-specific and easiest to localize in a single-env scenario.

## 7. Core Diagnostic Model

Each test should treat the runtime as four separable boundaries:

1. **Teleop/command boundary**
   - What command vector is actually being sent?
2. **Planner stage boundary**
   - How do intermediate planner values respond?
3. **Trajectory result boundary**
   - Does the final `BatchedTrajectoryResult` reflect the command?
4. **Playback boundary**
   - After applying a result frame to the robot, does the robot state match the selected reference frame?

This separation is critical because the existing symptoms already suggest planner markers and robot playback may be diverging.

## 8. Metrics and Probes

### 8.1 Command-to-plan metrics

For each command case, collect:

- `plan_dx`
- `plan_dy`
- `plan_dz`
- `plan_dyaw`
- `plan_standstill`

Interpretation:

- If `WASD` commands keep these near zero, the planner path is not responding to translational commands.
- If `QE` updates `dyaw` but `WASD` leaves `dx/dy` near zero, input mapping or translational planner behavior is suspect.

### 8.2 Touchdown metrics

- per-leg touchdown displacement relative to previous plan
- per-leg touchdown displacement relative to current foot positions
- front/back and left/right symmetry checks

Interpretation:

- If touchdowns change but robot does not, planner likely works and playback is the likely boundary of failure.
- If one leg is a strong outlier in standstill, either foot ordering or IK-related interpretation is suspect.

### 8.3 Base orientation metrics

- `roll`
- `pitch`
- `yaw`
- quaternion consistency across consecutive frames
- optional up-vector sanity check

Interpretation:

- Detects base flip, sign inversion, or impossible body orientation changes.

### 8.4 Playback consistency metrics

After applying a trajectory frame to the robot:

- `root_pose_error`
- `joint_pos_error`
- `foot_pos_error`
- optional `body_pos_error` if available

Interpretation:

- If planner output is healthy but playback errors are large, the failure is in writeback/sync/playback contract rather than planning.

### 8.5 Standstill anomaly metrics

- root variance across the horizon
- joint variance across the horizon
- maximum single-leg deviation
- left/right and front/back leg symmetry error

Interpretation:

- Standstill should be time-constant and leg-consistent.
- A single large per-leg outlier is a high-value signal for leg-order mismatch or playback corruption.

### 8.6 Lightweight performance metrics

Collect wall-clock timing for:

- replan call
- playback-apply step
- metric collection
- total diagnostic iteration

These timings do not gate correctness unless explicitly configured. Their primary purpose is to prevent diagnostics from accidentally collapsing into a CPU-heavy or serial-only workflow.

## 9. Command Cases

The suite should at minimum exercise:

- `standstill = [0.0, 0.0, 0.0]`
- `forward = [vx>0, 0.0, 0.0]`
- `backward = [vx<0, 0.0, 0.0]`
- `lateral_left = [0.0, vy>0, 0.0]`
- `lateral_right = [0.0, vy<0, 0.0]`
- `yaw_left = [0.0, 0.0, yaw>0]`
- `yaw_right = [0.0, 0.0, yaw<0]`

The exact default magnitudes should match the viewer’s intended teleop scale closely enough to reproduce reported behavior.

## 10. Proposed Test Cases

### 10.1 Viewer runtime diagnostics

#### `test_viewer_forward_command_changes_plan_motion_metrics`

Purpose:
- Determine whether forward `WASD`-equivalent input actually changes plan-level translational output.

Expected signals:
- `plan_dx` should be materially non-zero
- touchdown displacement should not remain identically zero

#### `test_viewer_lateral_command_changes_plan_motion_metrics`

Purpose:
- Same as above for `A/D`-style lateral motion.

Expected signals:
- `plan_dy` should be materially non-zero

#### `test_viewer_yaw_command_changes_yaw_and_touchdown_metrics`

Purpose:
- Confirm `QE` command path produces healthy planner response.

Expected signals:
- `plan_dyaw` non-zero
- touchdown change non-zero

#### `test_viewer_playback_matches_reference_frame_numeric`

Purpose:
- Determine whether playback into the robot matches the selected reference frame.

Expected signals:
- low `root_pose_error`
- low `joint_pos_error`

If this fails while planner metrics are healthy, the bug is downstream of planning.

#### `test_viewer_standstill_has_no_single_leg_outlier`

Purpose:
- Reproduce the “one leg is weird at standstill” complaint numerically.

Expected signals:
- low horizon variance
- low max single-leg deviation
- no strong asymmetry outlier

#### `test_viewer_leg_order_matches_planner_contract`

Purpose:
- Ensure the robot foot body ordering matches planner `LEG_ORDER = (FL, FR, RL, RR)`.

Expected signals:
- body names either match directly or are explicitly reordered

This should fail loudly if the runtime order is ambiguous or mismatched.

### 10.2 Planner stage diagnostics

#### `test_planner_stage_outputs_respond_to_forward_command`

Collect and inspect:
- `contact_seq`
- `touchdown_times`
- `stance_time`
- `touchdowns`
- `foot_targets`
- `base_approx`
- `base_solve`

Purpose:
- Find whether translational commands die at a specific stage.

#### `test_planner_stage_outputs_respond_to_yaw_command`

Purpose:
- Compare with the known “QE looks better” path.

This provides a reference-good case against the failing translational case.

#### `test_planner_standstill_stage_outputs_remain_symmetric`

Purpose:
- Determine whether standstill itself already contains a single-leg anomaly before playback.

#### `test_planner_output_vs_playback_divergence_report`

Purpose:
- Produce a compact numeric report showing:
  - planner is good / playback is bad
  - planner is already bad
  - both are bad

This test is primarily diagnostic and should emit high-signal failure details.

## 11. Instrumentation Strategy

The planner stage diagnostics require stage-level observability without changing production behavior in a risky way.

Recommended approach:

- Prefer existing stage boundaries already named in `batched_generate_trajectory(...)`
- If possible, reuse or extend planner instrumentation hooks already present in the runtime
- Avoid adding CPU-only inspection code or serial per-env deep copies
- Any new diagnostics helper should preserve batched tensors and device locality as much as possible

If intermediate planner tensors are not currently exposed, the design allows adding test-focused helpers or opt-in diagnostics wrappers, provided they do not alter planner semantics.

## 12. Assertions and Failure Reporting

A failing test must print enough structured detail to support debugging without visual replay.

Minimum failure payload should include:

- command vector
- `plan_dx, plan_dy, plan_dyaw`
- touchdown displacement per leg
- root playback error
- joint playback error
- roll/pitch/yaw
- max single-leg deviation

Example failure interpretation:

- `plan_dx ~ 0`, `touchdowns ~ 0`, `WASD cmd != 0`
  - translational command path did not reach meaningful planning
- `plan_dx > 0`, `touchdowns change`, `playback_root_error large`
  - planner healthy, playback unhealthy
- `standstill single_leg_deviation large`
  - leg ordering, IK interpretation, or standstill output corruption

## 13. Performance and Parallelism Requirements

The diagnostics suite must not default to a CPU-only model.

Requirements:

- keep planner tensors batched
- preserve device-local operations where practical
- avoid rewriting stage probes as per-env Python loops in hot sections
- include at least one batched smoke test to guard against accidental serialization

The optional benchmark file should include a small batched case and print timing summaries, but should stay conservative enough to run during targeted diagnostics.

## 14. Risks

1. **Isaac Lab runtime dependencies may make tests heavier than standard unit tests**
   - acceptable because these are diagnostics tests, not tiny pure unit tests

2. **Intermediate stage visibility may require additional hooks**
   - acceptable if implemented as opt-in diagnostics helpers with no production behavior change

3. **Playback mismatch may depend on scene sync semantics**
   - this is a feature, not a bug, for diagnostics: the suite should expose exactly that ambiguity

## 15. Success Criteria

This design is successful if, after implementation, we can answer all of the following without opening a viewer window:

1. Does translational teleop (`WASD`) produce meaningful planner motion?
2. Does yaw teleop (`QE`) produce healthy planner motion?
3. Is the robot playback numerically consistent with the selected trajectory frame?
4. Does standstill already contain a one-leg anomaly at planner output time?
5. Is foot ordering aligned with planner leg ordering?
6. Are diagnostics still batched/GPU-friendly enough to be practical?

## 16. Implementation Notes for the Next Phase

The next phase should produce:

- one real-runtime diagnostics test file
- one planner-stage diagnostics test file
- one optional benchmark file
- helper utilities only where necessary to expose numeric stage data cleanly

Implementation should remain reviewable in small steps:

1. establish reusable runtime fixture in `env_isaaclab`
2. add viewer runtime numeric probes
3. add planner stage probes
4. add batched/performance smoke coverage

