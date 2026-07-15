# T302v MPC QP Safety-Constrained Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new explicit `mpc_qp` planner backend that is isolated from the current `mpc` backend and starts with single-shot configurable QP iterations.

**Architecture:** `mpc_qp` lives in a new `Go2Pvcnn/extension/batch_mpc_qp_planner/` package and shares only stable boundary contracts: `ReferenceTrajectoryCache`, planner state/terrain types, command-frame convention, and viewer playback ABI. The first implementation creates the backend selection, config surface, cache-compatible planner output, diagnostics, and smoke-test path; the safety-constrained QP internals then grow behind that isolated backend without touching `extension/batch_mpc_planner`.

**Tech Stack:** Python, PyTorch, IsaacLab via `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`, CUDA GPU card 1 via `CUDA_VISIBLE_DEVICES=1`, pytest.

## Global Constraints

- Current `planner_backend="mpc"` behavior remains unchanged.
- New backend is selected only by explicit `planner_backend="mpc_qp"` or `--planner-backend mpc_qp`.
- `mpc_qp` default `qp_iterations` is `1`; diagnostic sweeps may set `1`, `2`, or `3`.
- Velocity/progress is a low-priority soft objective; semantic and height safety dominate.
- Do not allocate dense `[B, K, 22500, ...]` semantic pairwise tensors.
- Real IsaacLab testing must use `CUDA_VISIBLE_DEVICES=1` and `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.
- Passing metrics follow [../../docs/superpowers/specs/2026-07-03-mpc-qp-safety-constrained-backend-design.md](../../docs/superpowers/specs/2026-07-03-mpc-qp-safety-constrained-backend-design.md) and [../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html).

## Current State Addendum

- 2026-07-11 viewer timing debug exists behind `--timing-debug`; `--timing-sync-cuda` synchronizes CUDA around viewer timing points for more accurate GPU attribution.
- GPU1 headless scripted run with `qp_iterations=3` measured first nonzero command `loop_until_playback≈546.46ms`, `plan≈484.73ms`, `qp_total≈478.16ms`, `qp_nominal≈208.39ms`, and `qp_solve≈257.26ms`.
- The same run showed startup/reload is dominated by IsaacSim, not QP: scene creation `31.23s`, simulation start `12.45s`.
- Headless/WebRTC run printed `stdin is not a TTY; teleop keys are disabled`; interactive key latency must distinguish command-input focus from planner compute time.
- Follow-up child: zero-command hold/replan rows after scripted command release average `loop≈22.24ms`, `plan≈0.26ms`, `playback≈12.69ms`; this is not first-command QP latency but may affect smoothness if reproduced interactively.
- 2026-07-12 interactive TTY repro: key input reaches the loop, but a short `w` creates only one nonzero command row. With default `key_hold_timeout=0.18s` and nonzero `qp_iterations=3` planning at `≈0.26-0.29s`, the key expires before the next loop, then `mpc_qp` zero-command final-frame hold truncates playback. See [../log/2026-07-12-mpc-qp-interactive-key-hold-repro.md](../log/2026-07-12-mpc-qp-interactive-key-hold-repro.md).
- 2026-07-12 fix: viewer zero-after-nonzero now drains active `mpc_qp` motion before final-frame hold. GPU1 scripted pulse shows frames `1-20` after command returns to zero keep `need_replan=false`, `force_zero_hold=false`, `plan_ms=None`. See [../log/2026-07-12-mpc-qp-key-pulse-drain-fix.md](../log/2026-07-12-mpc-qp-key-pulse-drain-fix.md).
- 2026-07-12 follow-up latency investigation: once command reaches terminal stdin, GPU1 motion starts after QP in `<1s` (`350-565ms` measured, visible root motion next playback frame). Remaining likely issue is the input boundary because viewer keyboard control currently only reads `sys.stdin`, not WebRTC/browser keyboard events, plus GPU contention on busy cards. See [../log/2026-07-12-mpc-qp-post-drain-latency-investigation.md](../log/2026-07-12-mpc-qp-post-drain-latency-investigation.md).
- 2026-07-12 real-chain A/B: same headless/livestream command with no TTY reports `stdin is not a TTY=true` and produces `0` nonzero command rows; same command with TTY stdin injection produces one nonzero row and first-motion root delta `0.08149m`. This confirms current keyboard control is terminal stdin, not WebRTC/browser-window input. See [../log/2026-07-12-mpc-qp-real-chain-input-ab.md](../log/2026-07-12-mpc-qp-real-chain-input-ab.md).
- 2026-07-12 zero-command static hold: user selected policy B. Idle `mpc_qp` zero command now builds a static hold result from current robot state before terrain/QP planning; nonzero command still runs QP and release-to-zero drains the active trajectory. Card2 real idle run: zero QP rows `0`, terrain build rows `0`, max zero `plan_ms=0.791ms`, root/foot idle delta `0`. See [../log/2026-07-12-mpc-qp-zero-command-static-hold.md](../log/2026-07-12-mpc-qp-zero-command-static-hold.md).

---

### Task 1: Backend Selection And Config Surface

**Files:**
- Modify: `Go2Pvcnn/extension/trajectory_manager_factory.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Consumes: `planner_backend_from_cfg(cfg) -> str`
- Produces: `planner_backend="mpc_qp"` selection, `mpc_qp_planner_cfg.runtime.qp_iterations`, viewer `--planner-backend mpc_qp`.

- [x] **Step 1: Write failing backend selection tests**

Add tests proving `mpc_qp` is accepted, creates a distinct manager backend, exposes `qp_iterations=1`, and viewer argparse accepts `mpc_qp`.

- [x] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: fails because `mpc_qp` package and backend selection do not exist.

- [x] **Step 3: Implement minimal backend selection**

Add `mpc_qp` to the factory allowlist without changing the default `mpc` path. Add `mpc_qp_planner_cfg` with default `qp_iterations=1`. Add viewer argparse choice.

- [x] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: selection/config tests pass.

### Task 2: Cache-Compatible `mpc_qp` Planner Output

**Files:**
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/__init__.py`
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/manager.py`
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/adapter.py`
- Test: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Consumes: `MpcRobotState`, `MpcPlannerTerrain`, `MpcPlannerResult`, `build_mpc_terrain_from_scanner()`, `mpc_result_to_reference_cache()`.
- Produces: `MpcQpTrajectoryManager.refresh_from_env(env) -> ReferenceTrajectoryCache`, `plan_segment_qp(terrain, state, command, cfg) -> MpcPlannerResult`.

- [x] **Step 1: Write failing cache ABI tests**

Tests assert `plan_segment_qp()` returns finite root, feet, joints, contact, touchdown, `qp_iterations_configured == 1`, and a cache with full `(B, H, ...)` shape.

- [x] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: fails because planner output functions are missing.

- [x] **Step 3: Implement minimal cache-compatible planner**

Use current command-frame convention and a safe nominal trajectory as the first output path. Keep `qp_iterations` diagnostics explicit and isolated.

- [x] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: cache ABI tests pass.

### Task 3: First Safety Constraint Diagnostics

**Files:**
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/distance_field.py`
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/constraints.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Produces: fixed-shape semantic/height diagnostic tensors in `result.loss_breakdown`.

- [x] **Step 1: Write failing safety diagnostic tests**

Tests assert touchdown semantic violation count is zero on a simple obstacle map, height clearance violation is non-negative, and no dense semantic pairwise tensors are built.

- [x] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: fails because diagnostics do not exist.

- [x] **Step 3: Implement fixed-shape diagnostics**

Implement lightweight semantic and height checks over touchdown, foot, root, and sparse swing samples. Report violations but do not touch current `mpc`.

- [x] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: diagnostic tests pass.

### Task 4: Static Regression And Viewer Selection

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- Modify: `Go2Pvcnn/tests/test_mpc_rl_participation.py`
- Test: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Produces: explicit test coverage that current `mpc` still works and `mpc_qp` is opt-in.

- [x] **Step 1: Add regression tests**

Tests assert default configs still report `planner_backend="mpc"` and `mpc_qp` does not change current `MpcTrajectoryManager`.

- [x] **Step 2: Run focused static tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py -q
```

Expected: pass.

### Task 5: Real IsaacLab Smoke On GPU Card 1

**Files:**
- Modify: `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py` only if needed for `mpc_qp` selection.
- Create: `notes/log/2026-07-03-mpc-qp-safety-backend-smoke.md`

**Interfaces:**
- Produces: real GPU evidence for `mpc_qp` startup/replan path.

- [x] **Step 1: Run real smoke with env_isaacsim on card 1**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_smoke_16.json --planner-backend mpc_qp --qp-iterations 1
```

Expected: exit `0`, at least one replan, finite cache, no OOM.

- [x] **Step 2: Record log**

Record command, GPU card, metrics, result, and follow-up in `notes/log/2026-07-03-mpc-qp-safety-backend-smoke.md`.

## Status

- Created from spec commit `8168b15`.
- Current implementation status: first scaffold implemented and verified; full constrained QP solver and 1024/1024 `qp_iterations` sweep remain open.

### Task 6: Batched Safety QP Core

**Files:**
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/distance_field.py`
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Consumes: nominal `MpcPlannerResult`, high-resolution `MpcPlannerTerrain`, `MpcRobotState`, command-frame axes.
- Produces: fixed-shape touchdown/root QP corrections, step-cap diagnostics, semantic/height slack diagnostics.

- [x] **Step 1: Write failing QP behavior tests**

Add tests proving the QP update moves touchdown points off semantic objects without falling back to the current foot when another nearby safe point exists, executes configured iteration counts, and reduces root/foot stride under high height variation.

- [x] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: fails because there is no real QP update yet.

- [x] **Step 3: Implement fixed-shape batched QP step**

Implement a compact projected constrained step over per-leg touchdown XY and root terminal XY. Use GPU tensor operations, signed semantic distance samples, height/risk step caps, bounded line-search/clamp, and large safety slack penalties. Keep current `mpc` untouched.

- [x] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: QP behavior and existing scaffold tests pass.

### Task 7: Safety Metrics And Compatibility Regression

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/constraints.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Produces: design-aligned counters for max semantic/height/step-cap violation, touchdown-on-small, crossing mask, and fallback count.

- [x] **Step 1: Add metric tests**
- [x] **Step 2: Run focused QP and current MPC regression**
- [x] **Step 3: Update diagnostics and notes**

### Task 8: IsaacLab GPU1 Sweep

**Files:**
- Create/update: `notes/log/2026-07-03-mpc-qp-safety-qp-core-and-sweep.md`

**Interfaces:**
- Produces: real `env_isaacsim` evidence for `qp_iterations in {1,2,3}` and 1024/1024 if stable.

- [x] **Step 1: Run small real smoke on GPU1**
- [x] **Step 2: Run `qp_iterations` sweep on GPU1**
- [x] **Step 3: Attempt 1024 RL / 1024 MPC real probe**
- [x] **Step 4: Record pass/fail metrics and remaining gaps**

## Current State After QP Core Pass

- Implemented fixed-shape projected QP safety step in `batch_mpc_qp_planner/qp.py`.
- `qp_iterations` now executes the configured number in `plan_segment_qp`; default remains `1`.
- Added semantic touchdown projection to nearby safe cells before anchor fallback.
- Added terrain height-variation step cap that reduces root/foot progress on rough/high-change paths.
- Added low-small diagnostics: `qp_crossing_leg_count`, `qp_touchdown_on_small_count`, `qp_fk_semantic_collision_count`, `qp_fk_semantic_collision_rate`, and `qp_fk_semantic_min_clearance_over_semantic_m`.
- Added perf-probe historical counters: `max_qp_iterations_executed_seen` and `qp_replan_event_count`.

## Remaining Gaps

- Current QP core covers touchdown semantic keepout, foot-path low-small clearance metrics, terrain-risk step cap, FK knee/shank/root-underbody diagnostics, and FK shank clearance lift. Full constrained QP can still be strengthened with a denser selected-frame body model, but the named design counters now have focused coverage.
- Real 1024/1024 smoke passes memory envelope and now reports split QP timing fields. The profile still shows nominal zero-Adam parametric export as the dominant cost, while `qp_solve_ms` is separately visible for backend tuning.
- Viewer `go2_foostep_planner.py --planner-backend mpc_qp` static selection exists, but visual acceptance has not been rerun after the QP core pass.

### Task 9: FK Body/Leg Safety Metrics And Shank Clearance

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/constraints.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- Create: `notes/log/2026-07-03-mpc-qp-fk-body-leg-safety.md`

**Interfaces:**
- Produces: FK knee/shank/root-underbody semantic and height diagnostics for design acceptance, with QP-side shank clearance lift and IK refresh after QP foot corrections.

- [x] **Step 1: Write failing FK/body metric test**

Added a test requiring `qp_fk_body_leg_collision_count`, `qp_root_underbody_collision_count`, per-part semantic counters, and `qp_fk_body_leg_height_violation_max` to be collision-free on a low-small crossing case.

- [x] **Step 2: Run RED**

Focused test failed first on missing fields, then exposed a real `qp_fk_shank_semantic_collision_count=2` / `qp_fk_body_leg_height_violation_max≈0.0818m` after fields were added.

- [x] **Step 3: Implement FK/body diagnostics and repair**

Added sparse FK diagnostics via existing `fk_leg_points_from_joint_angles()`, root-underbody fixed stencil checks, QP-side FK shank clearance lift, and IK recomputation after QP foot corrections.

- [x] **Step 4: Verify focused/regression/real GPU1**

Focused `11 passed`, regression `168 passed, 1 warning`, pycompile/diff check exit `0`, 16/16 GPU1 smoke exit `0`, and 1024/1024 GPU1 smoke exit `0`.

### Task 10: QP Stage Timing Counters

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/manager.py`
- Modify: `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- Create: `notes/log/2026-07-03-mpc-qp-stage-timing-counters.md`

**Interfaces:**
- Produces: `qp_nominal_ms`, `qp_solve_ms`, `qp_repair_ms`, `qp_diagnostics_ms`, and `qp_total_ms` in QP result diagnostics, manager runtime counters, and perf-probe historical maxima.

- [x] **Step 1: Write failing timing tests**

Added tests requiring per-batch timing tensors in `plan_segment_qp()` and perf probe historical fields `max_qp_solve_ms_seen` / `max_qp_total_ms_seen`.

- [x] **Step 2: Run RED**

Focused QP suite failed with missing `max_qp_solve_ms_seen` source string and missing `qp_nominal_ms` in `result.loss_breakdown`.

- [x] **Step 3: Implement timing diagnostics**

Added lightweight `time.perf_counter()` timing around nominal planning, QP iterations, semantic repair, safety diagnostics, and total QP backend time. Manager now extracts finite max/mean QP timing fields from `loss_breakdown`; perf probe carries historical max timing across reset/step/final summaries.

- [x] **Step 4: Verify focused/regression/real GPU1**

Focused `12 passed`; QP + participation + current MPC backend regression `169 passed, 1 warning`; pycompile/diff check exit `0`; 16/16 GPU1 smoke exit `0`; 1024/1024 GPU1 smoke exit `0` with `max_qp_solve_ms_seen≈12.44ms`, `max_qp_total_ms_seen≈1668.19ms`, and CUDA allocated/reserved `7.57GB/9.28GB`.

### Task 11: Viewer Controlled Low-Small Crossing Acceptance

**Files:**
- Create: `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Create: `notes/log/2026-07-03-mpc-qp-viewer-controlled-crossing.md`

**Interfaces:**
- Produces: real IsaacLab viewer/playback evidence that opt-in `mpc_qp` crosses a low-small semantic object without semantic collision, touchdown-on-small, or small penetration.

- [x] **Step 1: Add viewer crossing probe**

Added `mpc_qp_viewer_crossing_probe.py`, which launches the real viewer runtime fixture with `planner_backend="mpc_qp"`, inserts a controlled low-small obstacle in a foot lane, plays back the reference, and emits one JSON row per cycle plus a hard acceptance summary.

- [x] **Step 2: Run first controlled lane and expose shank collision**

Initial lane placement proved foot-over success but exposed a final-frame shank semantic collision. The QP repair path now applies semantic-frame shank clearance lift and reruns the lift after final semantic touchdown repair.

- [x] **Step 3: Verify both forward foot lanes on GPU1**

Commands:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m -0.12

CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m 0.12
```

Both exit `0` with `viewer_crossing_acceptance_passed=true`, `fk_foot_over_low_small_success_count=1`, `max_fk_semantic_collision_count=0`, `max_fk_foot_small_penetration_rate=0.0`, and `max_fk_touchdown_on_small_rate=0.0`. The two lanes cover crossing masks `[1,0,0,0]` and `[0,1,0,0]`.

## Current State After Viewer Acceptance

- `mpc_qp` remains opt-in and current `mpc` remains the default.
- `runtime.qp_iterations=1` remains the default and is configurable for sweeps.
- Focused/current-backend regression after the viewer repair reports `170 passed, 1 warning`; pycompile and `git diff --check` exit `0`.
- Real GPU1 viewer controlled crossing now passes for both tested forward foot lanes on low-small semantic obstacles.
- Remaining scope is broader quality tuning and stress coverage, not the initial isolated backend acceptance.

### Task 12: Broader Direction Matrix And Lateral Swing-Over Repair

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- Create: `notes/log/2026-07-03-mpc-qp-direction-matrix-followup.md`

**Interfaces:**
- Produces: fixed-shape low-small swing-over repair for non-forward commands and broader GPU1 matrix evidence.

- [x] **Step 1: Add RED for lateral command low-small crossing**

Added `test_mpc_qp_lateral_command_low_small_crossing_keeps_fk_leg_over_obstacle`; RED failed on missing `qp_low_small_swing_over_repair_count`.

- [x] **Step 2: Add low-small swing-over repair**

Added `runtime.low_small_swing_repair_*` config fields and a fixed-radius small-obstacle swing repair in `qp.py`: swing frames near `semantic==1` candidates are blended toward the obstacle center and lifted before IK recomputation.

- [x] **Step 3: Add knee semantic to FK clearance lift**

Extended the existing shank clearance lift to also consider semantic knee hits. Focused QP suite passes after this change.

- [x] **Step 4: Verify focused/regression and GPU1 probes**

Focused QP `14 passed`; QP + participation + current MPC backend regression `171 passed, 1 warning`; pycompile/diff check exit `0`.

GPU1 evidence:

- `left:0.0,0.35,0.0`, `lateral=-0.12`: previously failed foot-over; now `viewer_crossing_acceptance_passed=true`, FK semantic collision `0`, touchdown-on-small `0`, small penetration `0`.
- 8-command matrix with `lateral=-0.12`: improved from `6/8` to `7/8` foot-over successes with collision `0`; remaining fail is `diag_fl` lacking touchdown-after in that lane.
- 8-command matrix with `lateral=+0.12`: includes successful `diag_fl`, but still exposes unresolved `mixed_turn_l` knee semantic collision (`fk_semantic_collision_count=7`).

## Current Remaining Gap After Direction Matrix

Superseded by Task 13 below. At this point the full original T302v objective was not yet complete: the broader matrix still had a hard unresolved lane case, `mixed_turn_l` with `lateral=+0.12`, due to knee semantic collision after playback.

### Task 13: Body-Leg Root-Lift Safety Matrix Follow-Up

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- Create: `notes/log/2026-07-03-mpc-qp-body-leg-root-lift-direction-matrix.md`

**Interfaces:**
- Produces: body-leg XY candidate repair diagnostics, semantic-volume body clearance, low-small crossing root-lift diagnostics, and GPU1 evidence that the previous hard knee collision is cleared.

- [x] **Step 1: Add RED for mixed-turn knee semantic collision**

Added `test_mpc_qp_mixed_turn_repairs_knee_semantic_collision_with_xy_avoidance`; RED reproduced a knee semantic collision and then missing body-leg root-lift diagnostics.

- [x] **Step 2: Add body-leg repair layers**

Added fixed-shape foot-target XY candidate repair scored by recomputed IK/FK knee/shank collision, semantic-volume clearance for semantic cells, and `runtime.low_small_crossing_root_lift_m` so low-small crossing horizons proactively lift root clearance. Removed duplicate final crossing lift after GPU1 evidence showed double-lifting hurt IK/readback quality.

- [x] **Step 3: Verify hard case, matrix, and 1024 smoke**

Focused QP `15 passed`; QP + participation + current MPC backend regression `172 passed, 1 warning`; pycompile and diff check exit `0`.

GPU1 evidence:

- Hard case `mixed_turn_l:0.45,0.15,0.6`, `lateral=+0.12`: `viewer_crossing_acceptance_passed=true`, FK semantic collision `0`, touchdown-on-small `0`, penetration `0`.
- 8-command matrix with `lateral=+0.12`: hard safety passes with max FK semantic collision `0`, touchdown-on-small `0`, penetration `0`; strict summary remains false because only `5/8` rows have foot-over success and several rows have no crossing opportunity in that lane.
- 8-command matrix with `lateral=-0.12`: hard safety passes with max FK semantic collision `0`, touchdown-on-small `0`, penetration `0`; strict summary remains false because `diag_fl` still lacks the strict lift-then-land/touchdown-after success.
- 1024/1024 GPU1 smoke: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=1024`, CUDA allocated/reserved `7.51GB/9.27GB`, `max_qp_solve_ms_seen≈25.48`, `max_qp_repair_ms_seen≈10.38`.

## Current Remaining Gap After Body-Leg Root-Lift Matrix

The hard semantic safety objective is now covered for the tested matrix: FK foot/knee/shank collision, touchdown-on-small, and foot-small penetration are all zero across both tested lanes. The remaining T302v gap is strict trajectory-quality acceptance, not a semantic collision blocker:

- left-lane `diag_fl` still misses strict foot-over/lift-then-land/touchdown-after success despite zero collision and zero penetration
- right-lane summary counts rows with no crossing opportunity as foot-over failures

Next work should decide the strict matrix acceptance contract, then tune `diag_fl` swing/touchdown behavior if every row must satisfy strict foot-over success.

### Task 14: Contact-Over Repair Diagnostic Follow-Up

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- Create: `notes/log/2026-07-03-mpc-qp-contact-over-repair-regression.md`

**Interfaces:**
- Produces: `qp_low_small_contact_over_repair_count` in `loss_breakdown` for contact-foot samples that cross low-small semantic cells.

- [x] **Step 1: Reproduce RED**

Focused QP regression failed with `KeyError: 'qp_low_small_contact_over_repair_count'` in `test_mpc_qp_lifts_contact_leg_when_low_small_lies_on_crossing_path`.

- [x] **Step 2: Implement fixed-shape contact-over repair**

Added a GPU tensor repair path that lifts contact foot samples over `semantic == 1` cells to `terrain_z + low_small_swing_clearance_m`, recomputes IK, and reports `qp_low_small_contact_over_repair_count`. `plan_segment_qp()` now defaults the metric to zero when no contact-over repair fires.

- [x] **Step 3: Verify focused/regression/real GPU1**

Focused failed case `1 passed`; focused QP suite `17 passed`; QP + participation + current MPC backend regression `174 passed, 1 warning`; pycompile/diff check exit `0`; real GPU1 16/16 and 1024/1024 smokes exit `0`. The 1024/1024 smoke completed 30 steps with CUDA allocated/reserved `7.76GB/9.49GB`, `max_qp_solve_ms_seen≈149.44`, and `max_qp_repair_ms_seen≈139.94`.

### Task 15: Strict Contact-Free Crossing Final Gate

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- Create: `notes/log/2026-07-03-mpc-qp-strict-contact-crossing-final.md`

**Interfaces:**
- Produces: stricter viewer acceptance that requires no `fk_stance_on_small_rate`, plus QP/viewer contact cleanup for FK-consistent low-small contact.

- [x] **Step 1: Reproduce the remaining real viewer gap**

GPU1 `diag_fl` with `lateral=-0.12` reproduced the strict failure: `crossing_leg_count=1`, FK semantic collision `0`, penetration `0`, touchdown-on-small `0`, but `fk_foot_over_low_small_success=0` and `fk_stance_on_small_rate=0.055`. Focused tensor tests were already green, so the root cause was the real viewer/FK/contact metric path rather than basic `plan_segment_qp` shape behavior.

- [x] **Step 2: Extend contact-over repair to footprint keepout**

Changed low-small contact-over detection to query a small footprint around planned/FK feet instead of only the foot center. Increased `runtime.low_small_contact_reland_forward_m` to `0.16` so repaired contact lands beyond the real small-obstacle footprint plus sampling margin.

- [x] **Step 3: Add FK-consistent contact cleanup**

Added final FK low-small contact suppression in the QP planner and an additional viewer-result cleanup aligned with the viewer acceptance FK/yaw convention. Tightened `mpc_qp_viewer_crossing_probe.py` so `viewer_crossing_acceptance_passed` requires `max_fk_stance_on_small_rate <= 1e-6`.

- [x] **Step 4: Verify final strict gate and throughput**

Focused QP suite `20 passed`; QP + participation + current MPC backend regression `177 passed, 1 warning`; pycompile and diff check exit `0`.

GPU1 evidence:

- `diag_fl`, `lateral=-0.12`: strict failure turned into pass with `fk_foot_over_low_small_success=1`, `fk_foot_over_low_small_lift_then_land=1`, `fk_foot_over_low_small_touchdown_after=1`, and `fk_stance_on_small_rate=0`.
- 8-command matrix with `lateral=-0.12`: exit `0`, `crossing_opportunity_count=8`, `fk_foot_over_low_small_required_success_count=8`, max FK semantic collision `0`, stance-on-small `0`, touchdown-on-small `0`, penetration `0`.
- 8-command matrix with `lateral=+0.12`: exit `0`, `crossing_opportunity_count=5`, `fk_foot_over_low_small_required_success_count=5`, max FK semantic collision `0`, stance-on-small `0`, touchdown-on-small `0`, penetration `0`. Rows without crossing opportunity are ignored for strict foot-over count but still gated by hard safety.
- 1024/1024 GPU1 smoke: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=1024`, `qp_replan_event_count=2`, CUDA allocated/reserved `7.48GB/9.26GB`, `epoch_seconds=7.35`, `max_qp_solve_ms_seen≈32.93`, `max_qp_repair_ms_seen≈13.40`.

## Current State After Strict Contact-Free Crossing

T302v is now verify-state rather than active-blocked: `mpc_qp` remains opt-in, current `mpc` remains unchanged, default `qp_iterations=1` remains configurable, strict reachable low-small crossing passes on the tested two-lane 8-command GPU1 viewer matrix, and 1024/1024 GPU1 smoke remains within prior memory range. Future work should treat T302v as a regression guard unless new terrain cases expose playback smoothness/readback issues.

Documentation sync: [human-12](../human/human-12-batched-planner-train-viewer-commands.md) now records the current `mpc_qp` opt-in commands, `--qp-iterations`, strict crossing probes, 1024/1024 smoke, and final acceptance metrics. See [2026-07-03-human12-mpc-qp-command-update.md](../log/2026-07-03-human12-mpc-qp-command-update.md).

Viewer CLI follow-up: [go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py) now accepts and applies `--qp-iterations` for `mpc_qp`, fixing the documented command's argparse failure before IsaacLab startup. This is a CLI/config propagation fix only; QP trajectory behavior is unchanged. See [2026-07-06-mpc-qp-viewer-qp-iterations-cli-fix.md](../log/2026-07-06-mpc-qp-viewer-qp-iterations-cli-fix.md).

## New Direction: Full SQP/QP Over Differentiable Fields

User decision on 2026-07-06: the observed foot trajectory discontinuity around small obstacles, hard stairs, and box terrain should not be solved by post-hoc repair, nearby touchdown lookup, fixed-offset search, ring search, endpoint candidates, or any other candidate-selection method. The next `mpc_qp` design pass must make a fixed-shape SQP/RTI-style QP the main path: root, foot controls, `touchdown_xy`, and slack variables are optimized directly; `touchdown_z` is bound through a differentiable height field.

Design intent:

- Keep `planner_backend="mpc_qp"` isolated from the current `mpc` backend.
- Treat existing safety repair, loss-driven update, and candidate/line-search code as temporary scaffold or diagnostic fallback until replaced.
- Prefer fixed-shape QP variables, constraints, and slack penalties over frame-wise hard replacement for touchdown keepout, swing clearance, stance/contact consistency, FK body-leg clearance, and terrain height variation.
- Touchdown variables must be solved by the QP. Do not find touchdown by nearby-cell lookup, fixed offset foothold selection, obstacle ring search, or endpoint candidate selection.
- Convert discrete height/semantic maps into differentiable vector fields before the QP loop: height `h, grad_h`, semantic risk `s, grad_s`, roughness/edge/support `r, grad_r`, and clearance fields or equivalent fixed sampled residuals.
- Encode alternating diagonal gait through fixed stance/swing masks and QP constraints so all four legs cannot drift together and one leg cannot chase velocity while degrading support.
- Add acceptance metrics for smoothness/readback quality in addition to the existing hard safety metrics: foot frame jump, joint frame jump, FK foot vs target foot error, stance continuity, swing arc continuity, and high-elevation stair/box visual stability.
- Do not claim this direction implemented until code removes/demotes repair/candidate search and real IsaacLab viewer/probe evidence is rerun.
- Required viewer acceptance terrain: `terrain-row=8`, `terrain-col=12` using `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --terrain-row 8 --terrain-col 12`. Do not hard-code `CUDA_VISIBLE_DEVICES`; the user selects the visible GPU externally.

Implementation update on 2026-07-06:

- Added `batch_mpc_qp_planner/fields.py`, `gait.py`, `variables.py`, and `qp_assembly.py` as the first fixed-shape SQP/QP scaffold.
- `fields.py` exposes differentiable height, semantic-risk, and roughness values/gradients.
- `gait.py` exposes fixed alternating diagonal swing/stance masks.
- `variables.py` makes `touchdown_xy` an explicit decision-vector slice and includes semantic/clearance/reachability/stability slacks.
- `qp_assembly.py` returns fixed-shape `H`, `g`, `A`, `b`, `E`, `e`, `lower`, and `upper` buffers for a given batch/horizon.
- `solver.py` no longer uses endpoint scale candidates or fixed-offset touchdown candidate selection in the continuous main path. Touchdown updates now use bounded field-gradient residuals, and FK/readback quality is improved by fixed residual refinement plus configurable `qp_iterations`.
- Focused static/local evidence: `test_mpc_qp_backend.py` `49 passed`; current MPC/participation regression `157 passed, 1 warning`; pycompile and `git diff --check` pass. Real viewer startup on `terrain-row=8,col=12` attaches `mpc_qp`, applies the tile override, enters playback loop without planner exception, and exits cleanly on Ctrl-C; numeric visual-quality scoring for this exact tile remains a future automated probe.

Fixed gait update on 2026-07-06:

- `plan_segment_qp()` now creates fixed alternating diagonal gait masks in continuous mode.
- `decode_controls_to_result()` accepts explicit `contact_state` and uses the fixed diagonal stance mask for output.
- Stance feet are anchored during their stance half of the horizon; swing feet keep the sampled Bezier trajectory.
- Added `qp_fixed_gait_active` diagnostic.
- Focused static/local evidence: `test_mpc_qp_backend.py` `51 passed`; current MPC/participation regression `157 passed, 1 warning`; pycompile pass. Real viewer `terrain-row=8,col=12` should be rerun after this gait change for visual confirmation.

### Task 18: Replace Candidate/Repair Scaffold With Full SQP/QP Main Path

Status: active/P0. This supersedes the candidate endpoint line-search direction discussed during hard-terrain debugging. The current local solver may contain transitional candidate/search code; the implementation plan must remove or bypass it from the `mpc_qp` main path.

Required architecture:

- New or revised files under `Go2Pvcnn/extension/batch_mpc_qp_planner/`:
  - `fields.py`: differentiable height, semantic-risk, roughness/edge/support, and clearance field values/gradients.
  - `gait.py`: fixed alternating diagonal gait masks for stance/swing phases.
  - `variables.py`: fixed QP variable layout for root, foot controls, touchdown variables, and slacks.
  - `qp_assembly.py`: fixed-shape assembly of `H`, `g`, `A`, `b`, `E`, `e`, and box bounds from linearized residuals.
  - `solver.py`: fixed-shape batched SQP/RTI QP solve; no candidate touchdown or endpoint selection.
- `touchdown_xy` is part of the QP decision vector; `touchdown_z = h(touchdown_xy)` through linearized field binding.
- Map use inside the QP loop is field query + gradient query only; do not scan cells or allocate obstacle-list/pairwise tensors.
- Default `qp_iterations=1` remains configurable for sweeps.
- Velocity/progress tracking remains low priority relative to semantic safety, roughness/high-variation footholds, clearance, reachability, stability, and smoothness.

Required tests:

- Static tests proving no candidate/search/repair main-path calls are used for touchdown or endpoint selection.
- Unit tests for field value/gradient shapes and finite values.
- Unit tests proving `touchdown_xy` appears in the QP variable layout and `touchdown_z` is height-field bound.
- Unit tests proving fixed diagonal gait masks: `FL/RR` swing against `FR/RL` stance, then the opposite phase.
- Unit tests proving QP matrix shapes are fixed for a given batch/horizon and do not depend on obstacle count.
- Viewer/probe acceptance must include `terrain-row=8`, `terrain-col=12`.

Next implementation node:

- [x] **Task 16: Continuous Bezier trajectory design/plan and first scaffold**

  Replace the repair-dominant path with a continuous trajectory-centered solve. The QP should optimize four-leg/root trajectory parameters under semantic keepout, height clearance, leg reach/FK clearance approximations, terrain-risk/foothold costs, and temporal smoothness costs. Any remaining repair must be explicit legacy fallback and separately diagnosed.

  Current local scaffold:

  - Added continuous Bezier design: [../../docs/superpowers/specs/2026-07-06-mpc-qp-continuous-bezier-trajectory-design.html](../../docs/superpowers/specs/2026-07-06-mpc-qp-continuous-bezier-trajectory-design.html).
  - Added implementation plan: [../../docs/superpowers/plans/2026-07-06-mpc-qp-continuous-bezier-trajectory-plan.md](../../docs/superpowers/plans/2026-07-06-mpc-qp-continuous-bezier-trajectory-plan.md).
  - Added `batch_mpc_qp_planner/bezier.py`, `continuous.py`, and `losses.py`.
  - `plan_segment_qp()` now defaults to `continuous_trajectory_enabled=True`: nominal `mpc` output is a warm start, output foot frames are sampled from cubic Bezier controls, touchdown z is terrain-bound, and the repair main path is inactive by default.
  - Legacy projected repair behavior remains available only when `cfg.runtime.continuous_trajectory_enabled=False`, and legacy repair-focused tests were made explicit about that mode.
  - `mpc`/`mpc_qp` isolation is now a documented design and test contract: current `mpc` does not receive `qp_continuous_*` diagnostics.
  - Focused verification: `Go2Pvcnn/tests/test_mpc_qp_backend.py` reports `26 passed`; current `mpc`/participation regression reports `157 passed, 1 warning`; pycompile and `git diff --check` exit `0`.

- [ ] **Task 17: Full Continuous-QP Loss/Solver Pass**

  Implement the real optimizer over Bezier/root variables. Use fixed-shape preallocated buffers, terrain/semantic query-only access inside iterations, four-leg/root joint loss, and loss-first tuning. Do not reintroduce hard repair for test failures; tune loss weights, fixed sampling density, warm start, or `qp_iterations`.

  Current progress:

  - Added `batch_mpc_qp_planner/solver.py`.
  - `qp_iterations` now calls a fixed-shape continuous update over Bezier controls instead of only re-decoding unchanged controls.
  - First implemented loss-driven update targets touchdown footprint height variation: fixed +/-xy probes, semantic penalty, bounded step, P3 terrain z rebinding, P1/P2 smooth endpoint carry.
  - Added continuous low-small swing clearance update over Bezier samples: low-small semantic cells under swing samples create a clearance deficit, and the solver lifts `P1/P2.z` by the Bezier-basis-scaled amount while keeping `P3.z` terrain-bound.
  - Added diagnostics `qp_continuous_low_small_clearance_deficit_max`, `qp_continuous_solver_swing_clearance_lift_count`, and `qp_continuous_solver_swing_clearance_deficit_before_max`.
  - Added continuous FK/readback diagnostics and update: target-foot vs clamped-IK FK error is reported, `P1/P2` receive a bounded readback correction, and low-small swing clearance is re-applied afterward so terrain/semantic clearance remains the final priority.
  - Added diagnostics `qp_continuous_fk_readback_error_max`, `qp_continuous_fk_readback_error_mean`, `qp_continuous_joint_frame_jump_max`, `qp_continuous_solver_fk_readback_update_count`, and `qp_continuous_solver_fk_readback_error_before_max`.
  - Added continuous root/base progress update: controls now carry root trajectory, decode solves IK against optimized root, and terrain height variation can cap root XY progress so hard height edges do not keep full nominal speed progress.
  - Added diagnostics `qp_continuous_root_terrain_risk_reduces_progress`, `qp_continuous_root_height_variation_max`, and `qp_continuous_root_progress_scale_min`.
  - Fixed continuous diagnostics so foot frame jump measures time-frame motion rather than inter-leg spacing, and joint frame jump reports maximum per-joint delta rather than a 12-joint vector norm.
  - Added continuous body-leg clearance update: fixed-shape IK/FK knee/shank semantic clearance deficits lift the continuous root z trajectory inside `continuous_qp_update()` instead of calling legacy repair.
  - Added root/foot start easing via `continuous_start_tangent_scale`, keeping velocity tracking soft while reducing aggressive initial relative motion.
  - Viewer acceptance now gates continuous playback readback, foot jump, per-joint jump, low-small clearance, semantic collision, stance-on-small, touchdown-on-small, and penetration.
  - Added RED/GREEN test proving more QP iterations reduce bad touchdown footprint height variation without activating the repair main path.
  - Added RED/GREEN test proving continuous iterations lift swing samples over low-small semantic cells without activating the repair main path.
  - Added RED/GREEN test proving continuous output reports FK/readback error without activating the repair main path.
  - Added RED/GREEN test proving root progress reduces on high height variation without activating the repair main path.
  - Focused QP verification now reports `34 passed`.
  - Current MPC/participation regression remains `157 passed, 1 warning`.
  - Real GPU1 16-env smoke after solver update exits `0` with `completed_steps=30`, `qp_replan_event_count=2`, CUDA allocated/reserved about `0.106GB/0.130GB`, `max_qp_solve_ms_seen≈26.94`, and `max_qp_repair_ms_seen≈0.0006`.
  - Real GPU1 viewer crossing acceptance now passes for `forward:0.45,0.0,0.0`, `qp_iterations=1`, `lateral_offset=-0.12`: `viewer_crossing_acceptance_passed=true`, required low-small crossing `1/1`, FK semantic collision `0`, stance/touchdown/penetration `0`, readback `≈0.00277m`, foot jump `≈0.04688m`, joint jump `≈1.22270rad`.
  - Added hard-terrain probe support for full `10x20` semantic terrain grids: `RealViewerRuntimeFixture(compact_semantic_grid=False)` keeps the complete terrain, and `move_env0_to_terrain_tile(..., ground_robot=True)` now selects a tile, moves env0/scanner to the tile origin, grounds root Z from scanner terrain, and refreshes the scanner again.
  - Added `Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py`, which emits JSON rows for selected or auto-scanned hard tiles, checks continuous readback/foot jump/joint jump/FK semantic/touchdown metrics, and uses local root-path height variation rather than whole-scan height range to decide whether progress reduction is required.
  - Real GPU1 hard-terrain probe now passes for tile `row=9,col=19`, commands `forward:0.35,0.0,0.0` and `diag_left:0.30,0.12,0.0`, `qp_iterations=1`: terrain scan height range `≈0.439539m`, `viewer_hard_terrain_acceptance_passed=true`, FK semantic collision `0`, touchdown-on-small `0`, max readback `≈1.53e-05m`, max foot jump `≈0.0251m`, max joint jump `≈0.756rad`.
  - Diagnosed required hard tile `row=8,col=12`: original continuous output had smooth foot/joint jumps but max FK/playback readback `≈0.44m` because touchdown/foot samples were terrain-bound below the root while the leg target exceeded Go2 reach and clamped at the calf lower joint limit.
  - Added hard-terrain readback detail diagnostics to `mpc_qp_hard_terrain_probe.py`: worst frame/leg, root/target/FK foot z, body/hip-relative foot coordinates, reach norm, and joint-limit saturation counts.
  - Added continuous reachability update in `solver.py`: fixed-shape sample residual over gait-sampled foot controls detects swing targets beyond the leg workspace and shortens touchdown/control XY toward the terrain-bound start anchor without candidate search or repair.
  - Added gait-segment FK readback update so residuals from the first and second diagonal swing phases update the matching Bezier controls; z remains governed by terrain binding and swing-clearance loss, not arbitrary readback repair.
  - Stabilized low-small swing clearance by excluding contact-transition boundary frames from clearance diagnostics and clamping per-iteration lift; this avoids P1/P2 z blow-up while keeping mid-swing obstacle clearance.
  - Tuned body-leg clearance loss margins (`body_leg_root_lift_margin_m=0.06`, `body_leg_root_lift_max_m=0.20`) so FK knee/shank semantic collision is cleared inside the continuous QP path.
  - Focused QP verification now reports `54 passed`; pycompile passes for QP planner and probes.
  - Required hard terrain `row=8,col=12`: `qp_iterations=1` now improves max readback from `≈0.44m` to `≈0.0583m` with FK semantic collision `0`, touchdown-on-small `0`, foot jump `≈0.04277m`, joint jump `≈0.16809rad`; strict `0.05m` readback gate still fails by about `8mm`.
  - Required hard terrain `row=8,col=12`: `qp_iterations=2` passes numeric acceptance with max readback `≈0.04941m`, FK semantic collision `0`, touchdown-on-small `0`, foot jump `≈0.04277m`, joint jump `≈0.16241rad`.
  - Added terrain-clearance and swing-height diagnostics after visual report: planned/FK foot clearance min, planned/FK terrain penetration count, swing height over terrain, and low-small swing height over terrain.
  - Added fixed-shape terrain clearance update over sampled Bezier points and separate `P1.z`/`P2.z` low-small swing-height updates. Default low-small clearance is now `0.06m` so flat small obstacles do not force high arcs by default.
  - Required hard terrain `row=8,col=12`, `qp_iterations=2`, after terrain-clearance tuning: planned terrain penetration `0`, FK semantic collision `0`, touchdown-on-small `0`, max readback `≈0.02118m`, foot jump `≈0.04663m`, joint jump `≈0.20897rad`; remaining FK terrain clearance min is `≈-0.00219m` for one readback point and is tracked under a `5mm` FK terrain tolerance.
  - Flat-small crossing probe with `forward:0.45,0.0,0.0`, `qp_iterations=2` did not produce a crossing opportunity and reported poor readback (`≈0.24835m`), so it cannot be used as evidence that small-obstacle crossing is fixed.
  - Added low-small root progress residual using fixed command-frame samples over the semantic/height field. This fixes the root-level failure where a low-small obstacle ahead did not create a crossing window; the synthetic RED/GREEN test now requires root to pass the obstacle window without activating repair.
  - Added a reachability gate for low-small foot-over residuals. If pulling a swing foot into the obstacle lane would exceed the leg workspace, the residual is rejected instead of forcing an unsafe trajectory.
  - Focused QP verification now reports `58 passed`; pycompile passes.
  - Required hard terrain `row=8,col=12`, `qp_iterations=2`, after low-small progress changes: still accepted with max readback `≈0.02118m`, planned terrain penetration `0`, FK semantic collision `0`, touchdown-on-small `0`, and the same single `≈2.19mm` FK terrain contact under tolerance.
  - Real default flat-small crossing probe now reaches root progress `≈0.36m` but still fails strict acceptance: crossing opportunity remains `0`, readback is `≈0.313m`, and foot-over needs a reachability-aware redesign rather than stronger lateral forcing.
  - Foot-lane probing showed that forcing lateral foot-over can produce FK semantic collision/terrain penetration. The current direction is therefore to keep root progress but redesign foot-over as a reachable trajectory loss, not a hard lateral move.
  - Added a continuous crossing-arc residual over Bezier `P1/P2` only: near-lane crossing legs can receive fixed-shape lateral and vertical arc updates, while `P3` touchdown remains terrain-bound and no endpoint candidates/search are introduced.
  - Added synthetic RED/GREEN coverage requiring the continuous QP path to create a low-small crossing leg from trajectory loss. Focused QP verification now reports `59 passed`; pycompile passes.
  - Required hard terrain `row=8,col=12`, `qp_iterations=2`, after crossing-arc residuals: still accepted with max readback `≈0.02118m`, planned terrain penetration `0`, FK semantic collision `0`, touchdown-on-small `0`, foot jump `≈0.04663m`, and joint jump `≈0.20897rad`.
  - Real default flat-small crossing probe still fails strict acceptance after crossing-arc residuals: `crossing_opportunity_count=0`, `max_qp_continuous_fk_readback_error_m≈0.31284`, and no semantic/touchdown/penetration violations. The next blocker is optimized-vs-IK/FK trajectory consistency, not endpoint selection.
  - Added high-arc/readback regression coverage after visual report that planned touchdown is on ground but actual FK foot does not coincide. The failing synthetic case had safe touchdown and no semantic collision but `qp_continuous_fk_readback_error_max≈0.212m`.
  - Fixed low-small root/body support height so nearby low-small semantic cells do not lift root support height to the obstacle top. This addresses the old design requirement that flat low-small obstacles should not raise root z.
  - Fixed terrain-clearance projection to use the same fixed diagonal gait segment basis as the decoder, so early/late swing terrain residuals are projected onto the correct Bezier segment.
  - Added a low-small-gated joint-limit readback residual and crossing arc target lane config. These are not sufficient for default flat-small strict crossing because the planned path still may not enter the semantic over-cell window at `qp_iterations=2`.
  - Focused high-arc RED/GREEN now passes, full QP suite reports `60 passed`, and pycompile passes.
  - Required hard terrain `row=8,col=12`, `qp_iterations=2`, still passes after root-support/readback changes: max playback readback `≈0.03522m`, planned penetration `0`, FK semantic collision `0`, touchdown-on-small `0`, foot jump `≈0.04478m`, joint jump `≈0.17770rad`, one FK terrain contact `≈-0.00219m` remains within the probe tolerance.
  - Default flat-small strict crossing remains open: `qp_iterations=2` still has `crossing_opportunity_count=0` and readback `≈0.23475m`; `qp_iterations=4` creates `crossing_opportunity_count=1` but worsens readback to `≈0.50064m`, introduces planned/FK penetration, and joint jump `≈2.09rad`. Therefore more QP iterations are not the fix.

  Still open:

  - Replace isolated per-loss updates with shared coupled residuals over root, Bezier controls, touchdown variables, and fixed gait samples.
  - FK/readback, reachability, joint-limit, FK terrain clearance, and body/knee/shank clearance must depend on shared root+foot variables. Touchdown semantic/roughness and smoothness can stay local.
  - Redesign optimized-vs-IK/FK trajectory consistency so root z can move down when FK feet float above terrain-bound planned feet; do not solve this by increasing swing height, endpoint candidates, or hard repair.
  - Strengthen semantic/object keepout over full Bezier samples and touchdown footprint through differentiable fields/losses, not search.
  - Required metrics now include semantic pass rate, semantic/object collision rate, FK/planned foot overlap/readback, terrain penetration, low-small crossing, foot/joint jumps, and row `8`, col `12` hard-terrain acceptance.
  - Tune loss weights, fixed sampling density, warm start, or `qp_iterations` if probes fail. Do not add hard constraints, nearby lookup, candidate endpoints, or repair layers to hide failures.

  Latest evidence: [2026-07-06 FK readback and root support](../log/2026-07-06-mpc-qp-fk-readback-root-support.md), [2026-07-06 crossing arc residual](../log/2026-07-06-mpc-qp-crossing-arc-residual.md), [2026-07-06 low-small progress and reach gate](../log/2026-07-06-mpc-qp-low-small-progress-and-reach-gate.md), [2026-07-06 terrain clearance and swing height](../log/2026-07-06-mpc-qp-terrain-clearance-and-swing-height.md), [2026-07-06 row8 col12 reachability](../log/2026-07-06-mpc-qp-row8-col12-reachability.md), [2026-07-06 fixed gait main path](../log/2026-07-06-mpc-qp-fixed-gait-main-path.md), [2026-07-06 fixed-shape SQP scaffold](../log/2026-07-06-mpc-qp-fixed-shape-sqp-scaffold.md), [2026-07-06 hard terrain probe](../log/2026-07-06-mpc-qp-hard-terrain-probe.md), and [2026-07-06 continuous viewer acceptance](../log/2026-07-06-mpc-qp-continuous-viewer-acceptance.md).

### Task 19: Shared Coupled Loss For Root/Foot Readback

Status: active/P0. Current user priority is flat-ground low-small crossing quality; row `8`, col `12` hard terrain is no longer the tuning driver for this node.

Design contract:

- Shared variable block: `root_delta_xy[t]`, `root_delta_z[t]`, `P1/P2 xyz`, `P3 xy`, fixed slacks.
- Coupled residuals: FK/readback, reachability, joint limits, FK terrain clearance, and body/knee/shank clearance see both root and foot variables.
- Local residuals: touchdown semantic/roughness mainly see `P3.xy`; smoothness mainly sees the sampled root or foot trajectory.
- Touchdown z remains bound to `height_at(terrain, touchdown_xy)`.
- Current `mpc` stays untouched; only opt-in `mpc_qp` changes.
- No candidate endpoint search, no nearby touchdown lookup, no repair fallback for the continuous main path.

Implementation plan:

- [x] Add RED test where terrain-bound planned feet and high root z create FK/readback mismatch; QP must lower root z through shared readback and reduce FK/planned error without activating repair.
- [x] Add shared readback-root update in `batch_mpc_qp_planner/solver.py`, reusing fixed-shape samples and terrain height queries.
- [x] Expose diagnostics: `qp_continuous_solver_fk_root_z_update_count`, `qp_continuous_solver_fk_root_z_delta_max`, and before/after readback max.
- [x] Run focused QP tests and pycompile.
- [ ] Flat-small P0 follow-up: replace the remaining root-lift-vs-readback tug-of-war with a truly coupled body/knee/shank + FK/readback update. Current flat-small `qp_iterations=3` evidence has crossing opportunity and FK foot-over success, but still fails because `playback_readback_error_max_m≈0.1205`, `fk_semantic_collision_count=3` on knee/shank, and `qp_fk_body_leg_collision_count≈31`.
- [ ] Keep flat-small gates explicit before returning to hard terrain: `crossing_opportunity_count > 0`, `fk_foot_over_low_small_success=1`, FK/planned readback `<=0.05m`, FK foot/knee/shank semantic collision `0`, touchdown/stance on small `0`, planned/FK terrain penetration `0`, low-small swing height `<=0.18m`, and joint jump `<=1.25rad`.

### 2026-07-06 Flat-Small Focus Update

Latest user direction: ignore row `8`, col `12` for now and focus on flat-ground low-small obstacle crossing quality.

Implemented/verified this pass:

- Body/knee/shank clearance update now also moves swing Bezier `P1/P2` laterally through the existing body-leg clearance loss, instead of only lifting root z. `P3/touchdown` remains terrain-bound and is not candidate-searched.
- Body/underbody clearance is included in the same continuous body-leg clearance pass, and FK readback root-z lowering is capped by underbody clearance instead of blindly lowering root through semantic objects.
- Body-leg semantic diagnostics now use a `1e-5m` numerical penetration tolerance, so a point exactly at the configured clearance margin is not counted as collision.
- Viewer flat-small probe now prints solver-stage diagnostics for reachability, FK readback, body-leg clearance, and joint-limit readback.
- Focused high-arc synthetic regression passes: foot-over success `1`, FK/planned readback `<=0.05m`, FK body-leg collision `0`, low-small swing-height gate `<=0.18m`.
- Full QP unit suite remains `61 passed`; pycompile passes.

Flat-small real probe is still **not accepted**:

- Command: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 3`
- Best current safe-ish probe after body-leg coupling: FK foot/knee/shank semantic collision `0`, touchdown/stance on small `0`, planned/FK terrain penetration `0`, foot-over success `1`, but `crossing_opportunity_count=0`, `qp_fk_body_leg_collision_count=5` from body/underbody diagnostics, and FK/planned readback remains `≈0.182m`.
- Additional diagnostics show `roll_pitch_abs_max≈0.167rad`, `raw_ik_joint_limit_violation_max≈0.838rad`, reachability/readback updates are active, but optimized foot controls and FK feet still diverge around mid-horizon.

Rejected directions from this pass:

- Do not clear root roll/pitch to zero as a post-hoc flat-small fix. A synthetic reproduction showed that direct root attitude leveling can clear body diagnostics but breaks foot-over, increases FK terrain penetration, and leaves readback high.
- Do not globally enable joint-limit readback outside semantic cells. A real probe showed it triggers but collapses foot-over and introduces planned/FK penetration.

Next step:

- Implement a true coupled root attitude/root position/Bezier foot-control residual for flat-small IK/FK consistency. Root `rpy` cannot be modified independently from foot controls; it must be optimized together with foot controls and body-leg clearance, then sampled once.
- Keep tuning inside existing design losses: body/knee/shank/root safety, reachability + IK/FK readback, swing clearance/crossing, smoothness, and low-priority tracking. Do not add hard repair, candidates, or touchdown lookup.

Latest evidence: [2026-07-06 flat-small body-leg/readback focus](../log/2026-07-06-mpc-qp-flat-small-body-leg-readback.md).

### 2026-07-07 Flat-Small Coupled Progress / P0 Boundary

Status: active/P0 follow-up.

What changed:

- `mpc_qp` continuous main path still uses one shared coupled loss over root, rpy, and Bezier foot controls. No candidate endpoint search, touchdown lookup, or repair fallback was added.
- Low-small target detection is anchored to the nominal/root-start XY so crossing residuals do not deactivate merely because an intermediate QP iteration moves root/controls.
- Crossing residuals now include existing-design midpoint along/lane terms and persistent sampled-foot/FK-foot crossing height/lane terms.
- Fixed the continuous boundary condition: Bezier `P0` preserves the current/nominal foot 3D position; only touchdown `P3.z` is terrain-bound.
- `P0` is explicitly fixed during the coupled update because it is a boundary condition, not an optimization variable.

Verification:

- Focused pytest: `test_mpc_qp_continuous_low_small_qp_creates_crossing_leg_from_trajectory_loss` passes; `test_mpc_qp_continuous_low_small_high_arc_remains_fk_reachable` still fails at `qp_iterations=2`.
- Real GPU2 flat-small probe:
  - Pre-P0 fix, `qp_iterations=3`: foot-over success `1`, semantic collision `0`, penetration `0`, readback `0`, but foot jump `≈0.3500m` and joint jump `≈1.311rad`.
  - Diagnosis: worst foot jump was frame `0 -> 1`, leg `2`, because `P0.z` had been terrain-bound below the real current foot.
  - After P0 fix, `qp_iterations=4`: foot-over success `1`, semantic collision `0`, penetration `0`, but foot jump `≈0.28145m`, joint jump `≈3.135rad`, and FK low-small swing height `≈0.1867m`.
  - After P0 fix, `qp_iterations=5`: foot jump improves to `≈0.2446m`, but semantic collision `5`, joint jump `≈3.125rad`, and FK low-small swing height `≈0.1849m`.
  - Joint-jump diagnosis points to an IK branch/limit discontinuity: leg `3` thigh jumps from near `0` to `≈3.13rad` while calf is clamped at the lower limit.

Rejected/ineffective directions:

- Stronger global reachability/joint-limit weights reduced some joint jump but removed foot-over.
- Smaller root/foot update steps reduced foot/joint jumps but removed foot-over.
- P1/P2 relative-control trust-region and control-polygon step penalties did not improve acceptance and were removed.

Next step:

- Add/tune an IK branch-continuity or differentiable leg/phase weighting residual inside the coupled objective so the selected crossing motion stays away from thigh/calf limit flips.
- Keep using loss / `qp_iterations` / trust-region tuning only; do not add hard constraints, candidates, search, or repair.

Latest evidence: [2026-07-07 flat-small coupled progress](../log/2026-07-07-mpc-qp-flat-small-coupled-progress.md).

### 2026-07-07 Idle Fast Jitter Fix

Status: idle symptom fixed; non-idle flat-small/edge suite still open.

What changed:

- `mpc_qp` gait masks are now command-aware. Zero-command rows use all-stance masks instead of the fixed alternating diagonal swing schedule.
- `plan_segment_qp()` passes the command into gait selection and reports `qp_idle_all_stance_active`.
- Idle rows are anchored to the incoming root pose and joint angles across the whole horizon, preventing IK branch changes and replan-boundary joint snaps.
- Viewer CLI runtime overrides now sync `horizon_steps`, `replan_interval_steps`, `dt`, and `qp_iterations` into `mpc_qp_planner_cfg` for `--planner-backend mpc_qp`.
- Continuous decode binds `P0.z` to terrain for stance anchors and preserves stance feet while allowing swing frames to use IK/FK readback.

Verification:

- Focused pytest for config/gait/idle/stance/readback: `7 passed, 57 deselected`.
- Real GPU2 `env_isaacsim` idle probe: all-stance in two consecutive replans; foot, joint, and root frame jumps all `0`; replan-boundary foot and joint deltas both `0`.
- Full QP suite: `53 passed, 11 failed`.

Remaining open:

- The full suite failures are non-idle low-small/edge/reachability items: semantic touchdown improvement already starts at `0`, root progress can shrink over low-small cases, height-edge FK penetration remains, and high-arc/reachability readback still needs coupled loss tuning.
- Continue with existing design losses and `qp_iterations`/weight tuning only. Do not add hard repair, candidate endpoints, or touchdown lookup.

Latest evidence: [2026-07-07 idle jitter fix](../log/2026-07-07-mpc-qp-idle-jitter-fix.md).

### 2026-07-07 Remaining Unit Failures Fixed

Status: local focused QP unit suite is green; real viewer/probe acceptance still needs rerun.

What changed:

- The non-idle failures left after the idle jitter pass were fixed inside the isolated `mpc_qp` package.
- Continuous decode now keeps QP output/FK readback consistency, including unclamped IK decode and low-small swing-sample clearance before IK where configured.
- Coupled root/foot updates now preserve root start XY, keep root z optimizable, preserve original nominal progress instead of compounding per-iteration drift, and update crossing arc `P1/P2/P3` directly as trajectory variables.
- Low-small, terrain height, body-leg, reachability, and readback parameters were tuned inside the existing QP/loss design. No `Go2Pvcnn/extension/batch_mpc_planner/` changes were made for this pass.
- Height diagnostics now ignore sub-micrometer numerical deficits so exact-contact terrain samples do not trip false failures.

Verification:

- Focused QP suite: `pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q` -> `64 passed`.
- Pycompile: `python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py` -> exit `0`.

Remaining open:

- This pass only proves local QP unit coverage. Flat small-obstacle real viewer quality and hard terrain row `8`, col `12` still need real `env_isaacsim` reruns before claiming visual/runtime acceptance.
- If real probes still look bad, continue within the approved design knobs: existing losses, weights, fixed sampling density, warm start, and `qp_iterations`. Do not add candidate endpoints, touchdown lookup, or hard repair layers.

Latest evidence: [2026-07-07 remaining QP unit failures fixed](../log/2026-07-07-mpc-qp-remaining-unit-failures-fixed.md).

### 2026-07-07 Idle Jitter Root/Foot Displacement Repro

Status: reproduced in real env-step path; not reproduced in zero-command `mpc_qp` direct playback/reference output.

What changed:

- Added diagnostic-only probe `Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py`.
- The probe records root and foot displacement in two modes:
  - `playback`: zero-command `mpc_qp` reference is directly played back through the viewer write path.
  - `env-step`: IsaacLab is stepped with zero actions while recording actual root/foot motion.
- No planner behavior or QP loss was changed.

Verification:

- Direct playback, default task terrain: planned and actual root/foot step maxima are `0`; foot readback is `≈1.05e-6m`.
- Direct playback, row `8`, col `12`: planned and actual root/foot step maxima are `0`; foot readback is `≈2.14e-6m`.
- Zero-action env-step, default task terrain: root total drift `≈0.13160m`, foot total drift `≈0.06240m`.
- Zero-action env-step, row `8`, col `12`: root total drift/fall `≈24.49m`, foot total drift/fall `≈24.86m`.

Conclusion:

- The current evidence does not support "zero-command `mpc_qp` trajectory itself jitters"; the reference is static.
- The visible idle motion is more likely in the runtime/env-step path: physics zero-action stability, policy/action application, reset/grounding, or the live viewer main-loop mode the user is observing.

Next step:

- Instrument the actual viewer loop the user runs to log `command`, `need_replan`, `playback_frame`, `playback_path`, and actual root/foot deltas per frame.
- Only return to QP loss tuning if that instrumentation shows nonzero planned reference motion.

Latest evidence: [2026-07-07 idle jitter root/foot displacement repro](../log/2026-07-07-mpc-qp-idle-jitter-repro-root-foot-displacement.md).

### 2026-07-07 Idle Jitter MPC Baseline A/B

Status: active diagnostic conclusion. The user specifically asked to compare against `mpc` and use a quantitative metric before blaming viewer/env behavior.

Metric contract:

- Main metric: `planned_root_trajectory_replan_delta_m`, `planned_foot_trajectory_replan_delta_m`, and `planned_joint_trajectory_replan_delta_rad`, comparing the full planned horizon across replans.
- Secondary metric: moving-cycle `max_actual_root_step_m`, `max_actual_foot_step_m`, and `max_actual_joint_step_rad`.
- Boundary continuity is still tracked, but it is not sufficient for the conclusion because current `mpc_qp` already has near-zero first-frame planned boundary delta.

Observed A/B on stop-after-motion playback (`--pre-command 0.45,0,0 --pre-cycles 2 --cycles 4`):

- `mpc` baseline: moving root/foot/joint step `≈0.01136m / 0.04012m / 0.33939rad`; full-horizon root/foot/joint replan delta `≈0.17988m / 0.21539m / 1.07066rad`.
- current `mpc_qp`: moving root/foot/joint step `≈0.10657m / 0.09633m / 1.26483rad`; full-horizon root/foot/joint replan delta `≈0.71195m / 0.48323m / 3.45675rad`; first-frame planned foot boundary remains only `≈7.57e-7m`.

Conclusion:

- The user-visible complaint is valid under the `mpc` A/B: current `mpc_qp` changes too much over the whole horizon and has larger moving-frame root/foot/joint jumps.
- This is not mainly a first-frame replan-boundary problem. The fix should target full-horizon replan consistency and moving-segment smoothness inside isolated `mpc_qp`, without candidate endpoints, touchdown lookup, hard repair, or default `mpc` edits.

Latest evidence: [2026-07-07 idle jitter MPC baseline A/B](../log/2026-07-07-mpc-qp-idle-jitter-mpc-baseline-ab.md).

### 2026-07-07 Flat Nonzero Speed Continuity A/B

Status: active diagnostic conclusion. The user reports that, with the same nonzero velocity command on flat terrain, `mpc_qp` speed tracking and foot/root continuity look much more distorted than `mpc`.

Metric contract:

- Root speed distortion: mean forward-speed error, root speed mean/std.
- Within-horizon continuity: max root step, root acceleration, max/mean foot step, foot acceleration, joint step/acceleration.
- Replan consistency: first-frame boundary delta plus full-horizon shape delta between adjacent rolling replans.

Observed pure planner A/B on flat terrain with `vx=0.45`, `vy=0`, `yaw=0`, horizon `25`, dt `0.02`, six rolling replans:

- `mpc`: root forward error mean `≈0.09225m/s`, root speed std `≈0.10788m/s`, root step max mean `≈0.01450m`, foot step max mean `≈0.06128m`, full-horizon root delta mean `≈0.00658m`.
- current `mpc_qp`: root forward error mean `≈0.95976m/s`, root speed std `≈1.07956m/s`, root step max mean `≈0.08462m`, foot step max mean `≈0.16386m`, full-horizon root delta mean `≈0.07443m`.
- Ratios: root forward error `≈10.4x`, root speed std `≈10.0x`, root acceleration `≈126.6x`, root step `≈5.8x`, foot step `≈2.7x`, full-horizon root delta `≈11.3x`, full-horizon foot delta `≈2.8x`.

Conclusion:

- The complaint is reproduced at planner-output level; it is not only a viewer/runtime artifact.
- The strongest signal is coupled root trajectory distortion, with secondary foot and joint discontinuity.
- Next work should inspect existing design-approved `mpc_qp` continuous update/loss terms and replan consistency. Keep `mpc` untouched and do not add candidate endpoint/search/repair behavior.

Latest evidence: [2026-07-07 flat speed continuity A/B](../log/2026-07-07-mpc-qp-flat-speed-continuity-ab.md).

### 2026-07-07 Zero-Command Hold Last Plan Frame

Status: implemented/verified for manager-owned `mpc_qp` reference cache.

User-corrected behavior:

- When velocity command is zero, do not replan from the current state.
- Use the previous planned reference's final frame as the new static reference horizon.

Implementation:

- Added a default no-op planning-selection hook to `MpcTrajectoryManager`; this keeps default `mpc` behavior unchanged.
- Overrode it in `MpcQpTrajectoryManager` so selected rows with `command_norm <= runtime.idle_command_threshold` and valid old cache are removed from the QP planning batch.
- Added `MpcQpTrajectoryManager._apply_hold_cache_rows_from_previous_final()` to copy the previous cache final frame across `root_pos_w`, `root_quat_w`, `joint_angles`, `foot_pos_w`, `foot_pos_root`, `contact_state`, and `planned_touchdown_w`.
- Hold rows reset their phase counter to `0`.

Verification:

- Focused manager test `1 passed`.
- Full QP unit suite `67 passed`.
- Base MPC suite `153 passed, 1 warning`.
- Pycompile and diff check pass.
- Direct playback stop-after-motion probe reports idle cycles with root/foot/joint step `0`; that probe does not exercise manager cache-hold directly, so the focused manager test is the cache-hold acceptance.

Latest evidence: [2026-07-07 zero-command hold last plan frame](../log/2026-07-07-mpc-qp-zero-command-hold-last-plan-frame.md).

### 2026-07-07 Viewer Zero-Command Playback Hold

Status: implemented/verified for the interactive viewer direct playback path.

Root cause:

- The manager-level zero-command cache hold did not cover the viewer window because `go2_foostep_planner.py` directly plans and plays `ViewerTrajectoryResult`.
- In a moving-to-zero command transition, the command printed as zero while an old moving result still reached `_viewer_direct_playback_step()`.

Implementation:

- Added an `mpc_qp`-only zero-command hold predicate in the viewer.
- The predicate treats root, quat, joint, or foot motion as a moving result.
- Added a playback guard immediately before `_viewer_direct_playback_step()` so stale moving results are converted to previous-final-frame hold before writing to the robot.
- Kept default `mpc` behavior unchanged.

Verification:

- Focused viewer tests: `3 passed`.
- Full QP unit suite: `69 passed`.
- Pycompile pass.
- Real IsaacSim scripted moving-to-zero viewer run on visible card 3 with `qp_iterations=3`:
  - before guard: zero-command frames still had `foot_delta_max_m≈0.18604` and `root_delta_m≈0.08941`.
  - after guard: zero-command sampled frames have `foot_delta_max_m=0` and `root_delta_m=0`.

Latest evidence: [2026-07-07 viewer zero-command playback hold](../log/2026-07-07-mpc-qp-viewer-zero-command-playback-hold.md).
