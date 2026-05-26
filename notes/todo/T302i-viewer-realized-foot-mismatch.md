# T302i Viewer Realized Foot Mismatch

## Current State

- T302i is a child of [T302h](T302h-semantic-obstacle-jitter-reproduction.md), related to [T300f](T300-unified-dense-mpc-backend.md) swing trajectory quality and [T301](T301-viewer-r-key-grounded-reset.md) viewer playback.
- User report: in the current viewer, low-small obstacle crossing shows planned `touchdowns` / colored swing markers that do not align with the actual Go2 foot ends, and the visual swing path is discontinuous.
- This branch treats the problem as separate from the already-passing T302h low-small planner task gate:
  - T302h rolling25 planned trajectory passed `semantic_task=0/2`, `foot_over=2/2`, and contact/penetration `0`.
  - The same log still recorded residual visual risks: moderate `min_z_quadratic_r2=0.325-0.397` and `playback_foot_error_max` around `0.29m`.
- Scope for this branch:
  - reproduce under real IsaacLab using `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`;
  - keep command output under `tmp/t302i-viewer-realized-foot-mismatch/`;
  - do not directly reuse current test conclusions as authority, because the user says current tests and todo have diverged;
  - do not change production planner/viewer code until the mismatch is localized.
- 2026-05-25 reproduction evidence:
  - Real IsaacLab low-small rolling25 probe reproduced a mismatch with no semantic contact/penetration: both commands have `semantic_task_violation=1` from continuity/overpass gates, not contact gates.
  - Planned-vs-realized foot mismatch is large at the selected playback frame: `playback_foot_error_max=0.286667-0.288352m`, while root error stays near zero.
  - Rolling segment terminal foot mismatch is also large: up to `0.375311m`.
  - Swing/continuity metrics match the screenshot complaint: `replan_boundary_foot_step_to_median=14.279-19.159`, `foot_accel_max_to_mean=22.194-29.451`, `min_z_quadratic_r2=0.324858-0.396874`.
  - Viewer entrypoint behavior is not clean for rolling25 MPC: `--planner-backend mpc --n-frames 25` still hits a together fixed-horizon guard; `--n-frames 50` attaches the MPC manager and reaches playback setup but was interrupted by timeout before natural completion.
- 2026-05-25 clamp trace evidence:
  - Focused low-small forward trace with `--trace-foot-mismatch` localizes the mismatch to IK clamp feasibility, not Isaac readback or joint writeback.
  - Worst row: `segment=0`, `frame=24`, `worst_leg=FL`, `planned_foot=[15.415548, 1.815011, 0.0]`, `planned_touchdown=[15.415548, 1.815011, 0.0]`, `actual_foot=[15.393563, 1.648044, 0.231984]`.
  - `actual_vs_internal_fk_error_norm=4.34e-7m` and `joint_error_max_abs=0.0`, so the simulated robot matches the planned joint sequence / FK.
  - Raw IK for the planned target needs `FL_calf=-0.0`, while the Go2 calf upper limit is `-0.8378`; clamped IK sets `FL_calf=-0.8378`, producing `internal_fk_error_norm=0.286668m`.
  - Interpretation: `rolling_segment_terminal_foot_error_max` is caused by exporting/planning unreachable `foot_pos` / touchdown targets after clamped IK, not by viewer marker extraction or Isaac FK mismatch.
- 2026-05-25 new reachable-crossing probe:
  - Added `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py` and local metric tests.
  - Local checks: `pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py` -> `5 passed`; `py_compile` passed.
  - Real IsaacLab forward baseline starts through `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python` and emits the requested metrics under `tmp/t302i-viewer-realized-foot-mismatch/`.
  - All-direction baseline current failures: max planned-vs-FK `0.452m`, max touchdown IK/FK `1.063m`, swing step ratio up to `20.058`, mixed-yaw direction cosine `-0.595`, pure-yaw small penetration `0.0075`. Stance/touchdown small contact remains `0`.
- 2026-05-25 probe-only loss direction:
  - `reachable_loss_v1/v2` improve some IK/FK feasibility but make swing continuity worse.
  - `reachable_struct_v1` improves FK swing continuity (`20.058 -> 6.263`) but worsens planned-vs-FK (`0.323 -> 0.372`).
  - `reachable_struct_v2` adds touchdown-phase IK/FK residual and partially improves the tradeoff: touchdown IK/FK `0.682 -> 0.594`, swing step `20.058 -> 15.809`, swing accel `13.487 -> 13.441`, no small contact, foot-over still `1`.
  - `reachable_struct_v2` is still rejected because whole-trajectory planned-vs-FK worsens (`0.323 -> 0.350`); next direction should target full-horizon reachability / raw joint-limit barrier rather than only touchdown residual.
  - `reachable_struct_v3` adds full-horizon worst FK residual and raw joint-limit excess. It improves touchdown IK/FK `0.682 -> 0.590` and swing step `20.058 -> 11.313`, but still worsens whole planned-vs-FK `0.323 -> 0.347` and lateral drift `0.079 -> 0.429`.
  - Current conclusion: penalty stacking is not enough; next production-facing direction should be reachability-aware foot target generation/parameterization or export contract, while keeping viewer visualization unchanged.
- 2026-05-25 small obstacle size sweep:
  - Default small obstacle is `diameter=0.12m`, `height=0.16m`; probe now supports `--semantic-small-diameter-m` in addition to height.
  - Smaller diameter improves forward IK/FK mismatch: baseline planned-vs-FK is `0.323` at `0.12m`, `0.319` at `0.10m`, and `0.291` at `0.08m`; touchdown IK/FK similarly drops from `0.682` to `0.670/0.641`.
  - Probe-only `reachable_loss_small_v2` at `0.10m` gives the best mismatch so far (`planned-vs-FK 0.319 -> 0.236`, touchdown `0.670 -> 0.577`, no contact, foot-over `1`) but still worsens swing accel (`12.029 -> 20.566`).
  - Current interpretation: small footprint matters and `0.10m` is a reasonable next probe candidate, but loss tuning still needs continuity-safe approach/cross behavior before all-direction acceptance.
- 2026-05-25 FK-realized distance-window crossing probe:
  - Added probe-only `reachable_fk_cross_v1/v2/v3` loss variants to test the user's proposed behavior: approach to a suitable command-frame distance first, then cross with FK-realized feet, while preventing low-base/spider shortcuts.
  - `reachable_fk_cross_v1` validated the approach/cross idea but found the low-base shortcut: forward swing step `20.058 -> 8.784`, touchdown `0.682 -> 0.631`, raw IK violation `2.471 -> 0.872`, but root height fell `0.143 -> 0.077` and lateral drift rose `0.079 -> 0.442`.
  - `reachable_fk_cross_v2` fixed low-base posture (`root_height=0.140`) but still drifted laterally (`0.436`) and did not improve planned-vs-FK enough.
  - `reachable_fk_cross_v3` is the best diagnostic so far for forward: planned-vs-FK `0.323 -> 0.320`, touchdown `0.682 -> 0.600`, raw IK violation `2.471 -> 0.838`, swing step `20.058 -> 8.931`, root height `0.143 -> 0.209`, no contact.
  - All-direction v3 remains rejected: lateral mismatch improves, but diagonal introduces small contact/penetration, mixed-yaw direction tracking remains poor (`cos -0.595 -> 0.108`), and pure-yaw continuity regresses (`step 9.863 -> 18.807`).
  - Current interpretation: distance-window + FK-realized crossing is a useful loss direction, but the next loss-only slice must split pure-yaw out of crossing pressure and fix mixed-yaw command-frame direction/lateral path handling.
- 2026-05-25 command-split v4-v9 probe:
  - v4 split extra loss but not the original cfg enough; diagonal/mixed contacts remained and pure-yaw continuity still regressed.
  - v5 introduced command-specific cfg, but still inherited too much `reachable_loss_small_v1`; mixed/diagonal contact worsened and pure-yaw swing step/accel became `19.957/19.582`.
  - v6 made pure yaw baseline-like with crossing/foot-over disabled: pure-yaw penetration cleared and accel stayed similar, but step still worsened (`7.814 -> 10.040`); mixed-yaw direction remained weak (`cos=0.086`).
  - v7 added explicit mixed-yaw command-direction cosine/progress loss: direction recovered (`cos -0.595 -> 0.9998`, drift `0.400 -> 0.029`) with contact `0`, but root height collapsed to `0.099`.
  - v8 added mixed-yaw root-height/posture guard: direction stayed clean and root height improved to `0.123`, but planned-vs-FK/touchdown regressed to `0.431/0.786`.
  - v9 added mixed-yaw reachability barrier: root height `0.147`, planned-vs-FK `0.350`, touchdown `0.564`, swing step `4.355`, contact `0`, but lateral drift regressed to `0.473` and direction cosine dropped to `0.897`.
  - Current interpretation: mixed-yaw now has three useful but competing signals: v7 direction, v8 posture, v9 reachability. Next variant should combine these more softly; if it still cannot satisfy all gates, loss-only is likely hitting a structural limit.
- 2026-05-25 v10 soft-combo probe:
  - v10 combined v7 direction, v8 posture, and v9 reachability with softer weights.
  - Mixed-yaw direction/drift were acceptable relative to baseline (`cos -0.595 -> 0.995`, drift `0.400 -> 0.120`) and root height stayed above v7 (`0.129`), but reachability collapsed: raw IK violation `2.040 -> 2.980`, planned-vs-FK `0.388 -> 0.430`, touchdown `0.662 -> 0.732`.
  - Current interpretation: continuing to stack scalar losses is not producing a simultaneous solution for mixed-yaw direction, posture, and IK/FK reachability. Next credible direction is a planner target/output contract change: reachable target generation, FK-realized optimization/selection, or FK-realized export after clamped IK with diagnostics retained.
- 2026-05-25 V9 viewer runtime port:
  - Added a real MPC debug variant path so `reachable_fk_cross_v9` can be selected in the viewer with `--mpc-debug-variant reachable_fk_cross_v9`; viewer marker behavior is unchanged.
  - Fixed the viewer startup mismatch where `--planner-backend mpc --n-frames 25` still constructed the together fixed-horizon cfg and raised the `n-frames=50` guard.
  - Viewer smoke reaches `[Viewer] Attached mpc trajectory manager`, `Planner horizon: 25 frames @ dt=0.020s`, and playback setup.
  - Latest mixed-yaw runtime recheck baseline -> V9: direction `-0.595 -> 0.999`, drift `0.400 -> 0.069`, planned-vs-FK `0.388 -> 0.353`, raw IK `2.040 -> 1.683`, swing step `12.330 -> 6.370`, no small contact, but touchdown IK/FK worsens `0.662 -> 0.745` and root height drops to `0.117`.
  - Current interpretation: V9 is now available for visual inspection, but remains diagnostic only and should not be promoted as a fix.
- 2026-05-26 command-frame endpoint / foot-height reproduction:
  - Added explicit probe diagnostics for the user's visual hypothesis: planned/FK swing along-command forward/backward steps, touchdown-behind-planned/FK/swing along distance, planned-vs-FK along error, and foot-height relative to root.
  - Real IsaacLab mixed-yaw baseline reproduces "swing goes forward, touchdown is behind": `planned_swing_along_forward_step_max_m=0.347791`, while `touchdown_behind_planned_foot_along_max_m=0.569856` and `touchdown_behind_swing_foot_along_max_m=0.569856`.
  - Baseline also shows command-frame planned-vs-FK mismatch `0.356298`, terminal planned-vs-FK `0.388287`, touchdown IK/FK `0.661772`, and direction cosine `-0.595472`.
  - V9 reduces but does not remove the endpoint conflict: touchdown remains behind planned/swing by `0.316026/0.314493`; touchdown IK/FK worsens to `0.745050`.
  - V9 reproduces the user's supplemental foot-height issue: `planned_swing_foot_above_root_z_max_m=0.120522` and `fk_swing_foot_above_root_z_max_m=0.069278`; baseline did not show above-root feet in this row.
  - Current interpretation: this is a real planner-output geometry conflict, not a viewer marker issue. Next root-cause trace should inspect which touchdown/swing/terrain terms can pull touchdown behind a forward-moving swing endpoint, and separately why V9 allows above-root foot arcs under low root height.
- 2026-05-26 touchdown chain trace:
  - Added optional `--trace-touchdown-chain` probe output that compares `nominal_raw`, `initial_grounded_decode`, `optimized_export`, and `fk_from_clamped_ik` touchdowns in command-frame coordinates.
  - Mixed-yaw baseline trace does not support the exact "nominal far forward then pulled back" sequence for this row: nominal touchdown along starts at `[-2.114, -2.776, -0.910, -1.519]`, grounding keeps along unchanged, and optimization moves it forward to `[-1.974, -1.331, -0.664, -1.159]`.
  - The same row still has the visible conflict: swing forward step `0.347791`, touchdown behind swing `0.569856`, terminal planned-vs-FK `0.388287`, touchdown IK/FK `0.661772`.
  - Current interpretation: the main mismatch is not nominal being pulled backward in this case; it is that swing path extrema/endpoints are not coupled tightly enough to the sampled/exported touchdown endpoint, then IK clamp/FK adds another offset.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302i.1 | verify | P0 | Reproduce low-small rolling25 viewer-style planned-foot vs realized-foot mismatch with real IsaacLab metrics | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `tmp/t302i-viewer-realized-foot-mismatch/` |
| T302i.2 | todo | P0 | If the probe reproduces high playback error, split planned output error into marker/touchdown extraction, direct playback writeback, and IK/FK realized foot readback | `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py` |
| T302i.3 | todo | P1 | Quantify swing visual continuity beyond existing R2/jump metrics with per-leg frame-local planned-vs-realized traces | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `tmp/t302i-viewer-realized-foot-mismatch/` |
| T302i.4 | todo | P0 | Decide whether viewer MPC should accept rolling25 `--n-frames 25` without constructing the together fixed-horizon cfg first | `Go2Pvcnn/extension/viz/go2_foostep_planner.py` |
| T302i.5 | todo | P0 | Decide planner output contract for unreachable `foot_pos` / touchdown targets after clamped IK: export FK-realized feet, enforce hard feasibility before export, or constrain low-small target generation upstream | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py`, `Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py` |
| T302i.6 | todo | P0 | Create a new low-small reachable-crossing IsaacLab probe that reuses existing helper code but makes the new acceptance metrics explicit | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`, `tmp/t302i-viewer-realized-foot-mismatch/` |
| T302i.7 | active | P0 | Implement probe-only loss variants for reachable low-small crossing and compare against the new all-direction baseline before production changes | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`, `Go2Pvcnn/extension/batch_mpc_planner/losses/`, `Go2Pvcnn/extension/batch_mpc_planner/config.py` |
| T302i.8 | blocked | P0 | Full-horizon reachability barrier / raw joint-limit excess loss direction; `struct_v3` still worsened exported planned-vs-FK and lateral drift | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py` |
| T302i.9 | todo | P0 | Decide next non-penalty architecture: reachable foot target generation, FK-derived foot export, or joint/reachable-coordinate optimization | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/extension/batch_mpc_planner/variables.py`, `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py` |
| T302i.10 | active | P0 | Treat small obstacle diameter as a tested variable; combine `0.10m` candidate size with continuity-safe loss before all-direction testing | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`, `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py` |
| T302i.11 | active | P0 | Continue loss-only distance-window direction with pure-yaw bypass, mixed-yaw command-frame direction handling, and FK no-contact gate before considering production changes | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py` |
| T302i.12 | active | P0 | Trace command-frame endpoint inconsistency: swing feet follow command direction, but planned touchdown can be placed behind the swing/planned endpoint; include foot-above-root diagnostics | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`, `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/extension/batch_mpc_planner/losses/` |

## Closed Children Archive

- None yet.

## Related Logs

- [../log/2026-05-25-1600-t302i-viewer-realized-foot-mismatch-reproduction.md](../log/2026-05-25-1600-t302i-viewer-realized-foot-mismatch-reproduction.md)
- [../log/2026-05-25-1723-t302i-ik-clamp-foot-mismatch-trace.md](../log/2026-05-25-1723-t302i-ik-clamp-foot-mismatch-trace.md)
- [../log/2026-05-25-1904-t302i-reachable-crossing-probe-baseline.md](../log/2026-05-25-1904-t302i-reachable-crossing-probe-baseline.md)
- [../log/2026-05-25-1951-t302i-reachable-loss-variant-probe.md](../log/2026-05-25-1951-t302i-reachable-loss-variant-probe.md)
- [../log/2026-05-25-2035-t302i-reachable-struct-v2-probe.md](../log/2026-05-25-2035-t302i-reachable-struct-v2-probe.md)
- [../log/2026-05-25-2110-t302i-reachable-struct-v3-barrier-probe.md](../log/2026-05-25-2110-t302i-reachable-struct-v3-barrier-probe.md)
- [../log/2026-05-25-2150-t302i-small-obstacle-size-and-loss-sweep.md](../log/2026-05-25-2150-t302i-small-obstacle-size-and-loss-sweep.md)
- [../log/2026-05-25-2129-t302i-fk-cross-distance-window-probe.md](../log/2026-05-25-2129-t302i-fk-cross-distance-window-probe.md)
- [../log/2026-05-25-2233-t302i-command-split-v4-v9-probe.md](../log/2026-05-25-2233-t302i-command-split-v4-v9-probe.md)
- [../log/2026-05-25-2244-t302i-v10-soft-combo-probe.md](../log/2026-05-25-2244-t302i-v10-soft-combo-probe.md)
- [../log/2026-05-25-2326-t302i-v9-viewer-runtime-port.md](../log/2026-05-25-2326-t302i-v9-viewer-runtime-port.md)
- [../log/2026-05-26-1002-t302i-command-frame-endpoint-height-reproduction.md](../log/2026-05-26-1002-t302i-command-frame-endpoint-height-reproduction.md)
- [../log/2026-05-26-1018-t302i-touchdown-chain-trace.md](../log/2026-05-26-1018-t302i-touchdown-chain-trace.md)
- [../log/2026-05-25-1222-t302h-rolling25-low-small-foot-over-production.md](../log/2026-05-25-1222-t302h-rolling25-low-small-foot-over-production.md)
- [../log/2026-05-22-1358-mpc-swing-trajectory-quality-reproduction.md](../log/2026-05-22-1358-mpc-swing-trajectory-quality-reproduction.md)
- Spec: [../../docs/superpowers/specs/2026-05-25-t302i-low-small-reachable-crossing-loss-design.md](../../docs/superpowers/specs/2026-05-25-t302i-low-small-reachable-crossing-loss-design.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `working tree, T302h rolling25 production pass`
- Current Work Ref: `working tree @ c54dc5c`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)

## Next Step

Use the following execution plan for subsequent testing and implementation. Treat it as the T302i working contract unless the user updates it.

Immediate next slice after `reachable_fk_cross_v3`:

1. Keep the distance-window idea, but gate it by translation command only:
   - pure yaw should get no foot-over/crossing pressure;
   - pure yaw keeps no-contact, stability, reachability, and continuity losses only.
2. For mixed translation+yaw, compute crossing direction from translational command in the correct world/body frame and add a full-path lateral corridor loss:
   - do not let the root or FK feet win by side-bypassing;
   - do not require speed magnitude tracking while inside the approach/cross window.
3. Add FK-realized no-small-contact/penetration pressure into the same gated coordinate space:
   - diagonal v3 contact proves foot-over credit alone is insufficient.
4. Run forward first, then all-direction only if forward improves planned-vs-FK and swing without drift/posture regression.

### Plan A: New Probe And Metric Contract

1. Create a new probe file, not a direct expansion of the old entrypoint:
   - Target file: `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`.
   - It may import/reuse helper functions from `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`.
   - Keep the new file focused on reachable low-small crossing acceptance, not the broader T302h semantic sweep.
2. Use the required runtime environment:
   - Final acceptance command must use `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.
   - Use IsaacLab startup, not only local helper/unit tests.
   - Write terminal/stdout/stderr outputs under `tmp/t302i-viewer-realized-foot-mismatch/`.
3. Test command coverage:
   - low-small forward: e.g. `vx>0, vy=0, yaw=0`;
   - low-small lateral: e.g. `vx=0, vy!=0, yaw=0`;
   - low-small diagonal: e.g. `vx!=0, vy!=0, yaw=0`;
   - low-small mixed yaw: e.g. `vx/vy` plus nonzero `yaw`;
   - pure or near-pure yaw: `translation_norm` near zero, `yaw!=0`; this should require no-contact/stability, not crossing.
4. Use a command-aligned frame for every translation crossing metric:
   - `forward_axis = normalize([vx, vy])` when `sqrt(vx^2 + vy^2)` is above threshold.
   - `lateral_axis` is perpendicular to `forward_axis`.
   - Compute obstacle/root/foot `along` and `lateral` by projection into this frame.
   - Do not use world `x` as the crossing direction.

### Plan B: Required Metrics

1. Swing-foot continuity:
   - `fk_swing_foot_step_max_to_median`: max frame-to-frame FK-realized swing-foot step divided by median step.
   - `fk_swing_foot_accel_max_to_mean`: max FK-realized swing-foot acceleration divided by mean acceleration.
   - `replan_boundary_fk_foot_step_to_median`: FK-realized foot jump at rolling segment boundary divided by median in-segment step.
   - These metrics address the user-visible discontinuous swing path.
2. Touchdown IK/FK consistency:
   - `touchdown_ik_fk_error_max`: max distance between planned touchdown and `FK(clamped IK(root, touchdown_or_foot_target))`.
   - `terminal_planned_vs_fk_foot_error_max`: max distance between exported planned foot and FK-realized foot at rolling segment terminal frames.
   - `raw_ik_joint_limit_violation_max`: max raw IK violation beyond joint limits.
   - `calf_upper_saturation_max` or equivalent per-joint diagnostic for calf near/over `-0.8378`.
   - Acceptance intent: the planner must not place touchdown or planned foot markers at Cartesian points that the actual clamped-IK robot cannot reach.
3. No contact with small obstacles:
   - `fk_stance_on_small_rate`: FK-realized stance feet on small semantic cells.
   - `fk_touchdown_on_small_rate`: touchdown targets on small semantic cells.
   - `fk_foot_small_penetration_rate`: FK-realized feet penetrating or occupying the small obstacle footprint/top.
   - Required target for low-small approach/crossing: all `0`.
4. Direction tracking without hard speed magnitude:
   - `command_direction_cosine`: cosine similarity between net root displacement and planar command direction.
   - `along_progress_m`: root progress along command direction.
   - `lateral_drift_m` or equivalent command-frame lateral deviation.
   - `speed_magnitude_tracking_error`: report for diagnostics, but do not fail approach/crossing solely because speed magnitude is lower while crossing.
   - Acceptance intent: the robot may slow down or approach over multiple replans, but it must keep the requested direction.
5. FK-realized foot-over obstacle arc:
   - `fk_foot_over_low_small_success`: at least one FK-realized swing foot crosses from obstacle-front to obstacle-back in command-frame `along`.
   - `fk_foot_over_low_small_min_lateral`: same foot remains within the obstacle lane while crossing; side bypass does not count.
   - `fk_foot_over_low_small_clearance_max`: same foot rises above obstacle top plus clearance near the obstacle lane.
   - `fk_foot_over_low_small_lift_then_land`: same foot rises above clearance, then later descends.
   - `fk_foot_over_low_small_touchdown_after`: touchdown lands beyond the obstacle in `along` and not on small semantic cells.
   - Acceptance intent: success requires a foot to fly over the small obstacle from above and land after it; decoded but unreachable foot arcs do not count.
6. Anti-low-base / anti-spider stability:
   - `root_height_min`.
   - `base_bottom_clearance_min`.
   - `roll_pitch_abs_max`.
   - `foot_lateral_spread_max`.
   - `foot_to_root_lateral_offset_max`.
   - `hip_abduction_limit_margin_min`.
   - Acceptance intent: do not pass by dropping the base, excessive body tilt, or spreading legs around the obstacle.
7. Yaw-specific behavior:
   - Translation + yaw cases use the command-aligned translation metrics plus no-contact/stability metrics.
   - Pure yaw cases should not be required to satisfy `fk_foot_over_low_small_success`.
   - Pure yaw must still satisfy small-obstacle no-contact, IK/FK consistency, swing continuity, and stability metrics.

### Plan C: Loss Behavior To Test

1. Preserve viewer behavior:
   - Do not fix the problem by changing the viewer to display FK-realized feet.
   - The viewer should continue showing planner outputs; planner outputs should become feasible.
2. Implement or test loss-only behavior inside `extension/batch_mpc_planner`:
   - No discrete phase/state machine.
   - No selector/postprocess repair as the primary fix.
   - Prefer continuous loss gates.
3. Approach-safe behavior:
   - If the robot is still before the crossing window, the segment may only approach the obstacle.
   - It must make safe positive progress along the command direction.
   - It must not touch small obstacles, create unreachable touchdowns, lower the base, or spread legs excessively.
4. Cross-when-ready behavior:
   - Crossing pressure becomes strong only when the robot is in a reasonable command-frame along-distance window near the small obstacle.
   - Crossing credit must use FK-realized feet after clamped IK.
   - Velocity magnitude tracking can be weakened in this window; direction tracking and lateral drift bounds stay active.
5. Rolling replan expectation:
   - A first 25-frame segment may walk closer without crossing.
   - A later rolling replan should cross once in the crossing window.
   - The acceptance probe should distinguish safe approach-only segments from failed crossing-when-ready segments.

### Plan D: Acceptance And Regression Gates

1. Primary T302i acceptance:
   - `terminal_planned_vs_fk_foot_error_max` and `touchdown_ik_fk_error_max` must drop materially from the reproduced `0.286-0.375m` mismatch range.
   - FK-realized swing continuity metrics must not show large boundary/acceleration spikes.
   - Small-obstacle contact metrics must be `0`.
   - Direction metrics must show positive progress and bounded lateral drift for translation commands.
   - FK-realized foot-over must pass for translation commands once the robot reaches the crossing window.
   - Pure yaw must remain contact-free and stable without requiring crossing.
2. Regression coverage:
   - Keep high-small and large-obstacle behavior in mind; after low-small probe direction works, run a focused non-regression pass so the loss does not turn high-small/large into unsafe crossing.
   - Keep existing T302h/T302i reproduction logs as baseline evidence.
3. Logging:
   - Every IsaacLab run gets a dedicated log under `notes/log/`.
   - Update this T302i branch page after each meaningful test pass.
   - Keep output files under `tmp/t302i-viewer-realized-foot-mismatch/`.

## Node Details

### T302i.1 Low-Small Viewer-Style Reproduction

- why-created: user supplied screenshots where colored planned trajectory/touchdown markers visually diverge from actual foot ends during low-small crossing, and noted swing discontinuity. Existing T302h acceptance proves task-level planned crossing but explicitly leaves playback foot error and parabola quality open.
- command:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 24 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00' --variants baseline > tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_repro.jsonl 2>&1
```

- output:
  - `tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_repro.jsonl`
- result:
  - `forward_v050`: `playback_foot_error_max=0.286667`, `playback_foot_error_mean=0.125156`, `rolling_segment_terminal_foot_error_max=0.286667`, `semantic_task_violation=1`, `semantic_task_contact_violation=0`, `semantic_task_continuity_violation=1`, `foot_over_low_small_success=1`, `small_overpass_success=0`, `touchdown_on_semantic_rate=0`, `foot_semantic_penetration_rate=0`, `replan_boundary_foot_step_to_median=14.279280`, `foot_accel_max_to_mean=22.193964`, `min_z_quadratic_r2=0.396874`.
  - `forward_yaw_v050_vy025_yaw100`: `playback_foot_error_max=0.288352`, `playback_foot_error_mean=0.137288`, `rolling_segment_terminal_foot_error_max=0.375311`, `semantic_task_violation=1`, `semantic_task_contact_violation=0`, `semantic_task_continuity_violation=1`, `foot_over_low_small_success=1`, `small_overpass_success=0`, `touchdown_on_semantic_rate=0`, `foot_semantic_penetration_rate=0`, `replan_boundary_foot_step_to_median=19.159033`, `foot_accel_max_to_mean=29.451223`, `min_z_quadratic_r2=0.324858`.
- interpretation rule:
  - `semantic_task=0` with high `playback_foot_error_max` means the planner's abstract/reference path passes but viewer/Isaac realized foot display diverges.
  - high jump/boundary/R2 failure with low playback error means the issue is mainly decoded trajectory continuity.
  - touchdown semantic/contact failures mean T302h task acceptance regressed and should be handled under T302h as well.

### T302i.4 Viewer Entrypoint Horizon Mismatch

- why-created: the user specifically asked to use IsaacLab startup for the current viewer failure. The probe reproduces the numeric mismatch, but the viewer entrypoint itself shows a separate MPC startup mismatch.
- commands:

```bash
timeout -s INT -k 20s 90s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc --n-frames 25 --plan-dt 0.02 --warmup-steps 6 --scripted-command "0.50 0.00 0.00" --scripted-command-cycles 1 > tmp/t302i-viewer-realized-foot-mismatch/viewer_mpc_low_small_smoke.log 2>&1
timeout -s INT -k 20s 120s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc --n-frames 50 --plan-dt 0.02 --warmup-steps 6 --scripted-command "0.50 0.00 0.00" --scripted-command-cycles 1 > tmp/t302i-viewer-realized-foot-mismatch/viewer_mpc_low_small_smoke_n50.log 2>&1
```

- results:
  - `n-frames=25` exits with `ValueError: together backend requires --n-frames=50 for the fixed T116 horizon, got 25`, despite `--planner-backend mpc`.
  - `n-frames=50` reaches `[Viewer] Attached mpc trajectory manager`, prints `Planner horizon: 50 frames @ dt=0.020s`, and uses `[Viewer][Playback] path=render+scene_sync`, but the command was interrupted by the 120s timeout before natural completion.

### T302i.5 IK Clamp Export Contract

- why-created: focused trace shows the worst low-small planned touchdown requires a calf angle outside Go2 limits; `plan_segment()` exports the original decoded `foot_pos` / touchdown while the robot executes clamped IK joints, so viewer markers and actual foot ends diverge.
- command:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 24 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline --trace-foot-mismatch > tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_trace_forward_clamp.jsonl 2>&1
```

- output:
  - `tmp/t302i-viewer-realized-foot-mismatch/rolling25_low_small_trace_forward_clamp.jsonl`
- result:
  - worst terminal row: `segment=0`, `frame=24`, `worst_leg=0/FL`.
  - `foot_error_norm=0.286667m`.
  - `actual_vs_internal_fk_error_norm=4.34e-7m`.
  - `ik_clamp_worst_joint_name=FL_calf`, `ik_raw_worst_joint_value=-0.0`, `ik_clamped_worst_joint_value=-0.837800`, upper limit `-0.8378`.
  - `planned_touchdown_xyz` equals `planned_foot_xyz`, so the touchdown marker is showing the exported unreachable target.
- interpretation:
  - the current visual mismatch is an output-contract / feasibility issue: the planner returns unreachable Cartesian feet while returning clamped feasible joints.
