# Parallelism Terrain Root Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU-parallel multi-point root terrain sampling and viewer controls, while keeping keyboard `R` reset behavior separate from the new terrain-platform reset button.

**Architecture:** Extend `ParallelismCfg` with symmetric sampling range/count and RPY deadband settings. Replace non-flat single-point roll/pitch estimation with least-squares slope fitting over uniformly sampled height points, then apply existing clamp/smoothing/rate limiting. Extend `ViewerTestTerminalState` and the Omni UI panel with root controls and a one-shot platform-reset flag; preserve `TeleopCommand.reset_requested` and its keyboard `R` path unchanged.

**Tech Stack:** Python, PyTorch, IsaacLab `omni.ui`, pytest.

## Global Constraints

- Only root trajectory generation and viewer debug controls may change.
- Do not change touchdown sampling, swing trajectory, IK, FK, collision geometry, semantic filtering, scoring, or RL interfaces.
- Keyboard `R` remains the existing normal reset path for flat and small-obstacle workflows.
- Panel platform reset is a separate one-shot action for the currently selected terrain tile.
- Root sampling remains Torch-parallel and must not create a root-trajectory candidate dimension.

### Task 1: Update the design contract

**Files:**
- Modify: `docs/superpowers/specs/2026-08-09-parallelism-terrain-root-sampling-design.html`

- [ ] Add the explicit distinction between keyboard `R` and panel `Reset to terrain platform`.
- [ ] State that panel reset must not reuse or modify `TeleopCommand.reset_requested`.
- [ ] Verify the HTML contains no placeholders.

### Task 2: Add configuration fields and failing root math tests

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/extension/parallelism/root.py`
- Test: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`

**Interfaces:**
- Add fields:
  - `terrain_following_pitch_sample_range_m: float = 0.35`
  - `terrain_following_pitch_sample_count: int = 7`
  - `terrain_following_roll_sample_range_m: float = 0.35`
  - `terrain_following_roll_sample_count: int = 5`
  - `terrain_following_rpy_deadband_rad: float = 0.02`
- Add pure helpers:
  - `_uniform_symmetric_samples(range_m, count, *, dtype, device) -> Tensor`
  - `_fit_height_slope(samples, offsets) -> Tensor`
- The fitted pitch/roll must be finite, bounded by existing limits, and use the slope sign convention already used by `_terrain_following_rpy`.
- When fitted absolute angle is below deadband, target angle must be zero.
- For flat profiles, output roll/pitch is zero.
- For a linear slope `h(s)=0.2s`, fitted magnitude is `atan(0.2)` within tolerance.

- [ ] Write failing tests for symmetric sample generation, line fitting, flat zero, slope sign, and deadband.
- [ ] Run:
  `env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`
- [ ] Implement config fields and helpers with Torch operations only.
- [ ] Run the focused tests until green.

### Task 3: Replace non-flat single-point RPY sampling

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/root.py`
- Test: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`

- [ ] Replace the four single offset points in `_terrain_following_rpy()` with:
  - `pitch_s = linspace(-pitch_range, +pitch_range, pitch_count)`
  - `roll_s = linspace(-roll_range, +roll_range, roll_count)`
  - height queries for each line
  - least-squares slope fitting
- [ ] Preserve current sign conventions:
  - `pitch_raw = -atan(pitch_slope)`
  - `roll_raw = atan(roll_slope)`
- [ ] Apply clamp, deadband, smoothing, and rate limiting after fitting.
- [ ] Keep frame-0 actual `rpy0` behavior through the existing smoothing/rate-limit path.
- [ ] Run focused root tests.

### Task 4: Add viewer root controls and separate platform reset

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Extend `ViewerTestTerminalState` with the new root config values and:
  - `platform_reset_requested: bool = False`
- Extend `_parallelism_cfg_from_viewer_args()` to pass the root values into `ParallelismCfg`.
- Add `_request_viewer_platform_reset(state)` or equivalent one-shot consume helper.
- Add a platform reset helper that uses the selected terrain origin and existing ground/scanner synchronization, but does not set `TeleopCommand.reset_requested`.

- [ ] Write failing tests that panel state values map to `ParallelismCfg`.
- [ ] Write a failing test that platform reset consumes only `platform_reset_requested`, while `reset_requested` remains independent.
- [ ] Add sliders for root clearance, z smoothing/rate, pitch/roll range/count, RPY smoothing/rate/limits/deadband.
- [ ] Add a `Reset to terrain platform` button that sets the one-shot platform flag.
- [ ] In the main loop, process platform reset in its own branch before planning; keep the existing `if active_cmd.reset_requested:` branch unchanged.
- [ ] Run viewer adapter tests.

### Task 5: Verify and commit

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/extension/parallelism/root.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`
- Modify: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`
- Modify: `docs/superpowers/specs/2026-08-09-parallelism-terrain-root-sampling-design.html`

- [ ] Run focused tests:
  `env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`
- [ ] Run related tracking tests:
  `env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/tracking/test_parallelism_terrain_following_root.py Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py`
- [ ] Run:
  `env_isaacsim/bin/python -m compileall -q Go2Pvcnn/extension/parallelism Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- [ ] Run `git diff --check`.
- [ ] Commit with:
  `git commit -m "feat: add terrain root sampling controls"`
