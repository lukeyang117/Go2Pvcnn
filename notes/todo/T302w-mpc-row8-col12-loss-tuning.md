# T302w MPC Row8 Col12 Loss Tuning

## Current State

- Opened by user request on 2026-07-06 22:26 CST.
- This branch is for the existing `planner_backend="mpc"` path only.
- It is parallel to, and isolated from, [T302v mpc_qp](T302v-mpc-qp-safety-constrained-backend-plan.md).
- Current result on 2026-07-13: farther-walk repro confirmed that initial flat start hid the rough-tile failure. The latest fix clears body/knee/shank heightfield penetration on row `8`, col `11` and row `8`, col `12` using only existing MPC FK body-leg collision internals/weights. FK planned-vs-realized readback remains micron-scale and low-small hard metrics still pass. Foot initial-frame clearance and long-sequence touchdown/current marker distance remain separate residuals.
- Reproduction command:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc --n-frames 25 --plan-dt 0.02 --terrain-row 8 --terrain-col 12
```

- User-observed issue: on terrain row `8`, col `12`, planned feet and realized/FK feet appeared not to stay aligned because velocity/progress tracking seemed too strong relative to foot/FK consistency.
- Required strategy: reproduce different speed magnitudes/directions first, then tune existing losses/weights/parameters only. Do not add or delete optimizer losses.
- Required acceptance reference: [../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html).

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302w.1 | verify | P0 | Reproduce row `8`, col `12` `mpc` nonzero speed/direction FK/IK and touchdown marker metrics. | `Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py`, `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |
| T302w.2 | verify | P0 | Tune only existing `mpc` loss weights to reduce touchdown-contact alignment while preserving FK/IK. | `Go2Pvcnn/extension/batch_mpc_planner/config.py`, `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py` |
| T302w.3 | verify | P0 | Run the low-small redesign acceptance metrics and confirm no regression against the 2026-05-28 spec. | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`, spec-linked metrics |
| T302w.4 | open | P1 | Keep stop-after-motion zero-command mismatch as a separate issue if the user reopens it. | `Go2Pvcnn/extension/batch_mpc_planner/planner.py` |
| T302w.5 | verify | P0 | Reproduce and mitigate row `8`, col `11` / `12` farther-walk body/knee/shank heightfield penetration while preserving low-small metrics. | `Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py`, `notes/log/2026-07-13-mpc-row8-col11-col12-heightfield-penetration-repro.md`, `notes/log/2026-07-13-mpc-row8-col11-col12-body-shank-clearance-fix.md` |
| T302w.6 | verify | P0 | Quantify `optimize_steps=0..25` on deterministic S4 low-small cylinder/cone crossing and identify the smallest conservative candidate. | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`, `notes/log/2026-07-15-mpc-low-small-optimize-steps-sweep.md` |
| T302w.7 | verify | P0 | Reproduce row8/col12 planned root/foot, analytic FK, Isaac readback, touchdown/current-foot, and terrain-clearance metrics at `optimize_steps=16`. | `Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py`, `notes/log/2026-07-15-mpc-row8-col12-opt16-position-repro.md` |
| T302w.8 | open | P0 | Compare row8/col12 step `16` with `24/25` under the same current runtime and separate optimizer convergence from analytic FK convention mismatch. | `Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py` |
| T302w.9 | open | P1 | Repair stale row8/col12 probe fixture calls (`compact_semantic_grid`, `move_env0_to_terrain_tile`) before its documented command is directly runnable again. | `Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py`, `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py` |

## Scope Guard

- This branch must not modify `Go2Pvcnn/extension/batch_mpc_qp_planner/`.
- This branch must not tune `mpc_qp` weights, QP iterations, differentiable fields, or viewer acceptance logic.
- This branch must not add any new optimizer loss.
- Allowed changes are limited to existing `mpc` loss weights/parameters, existing runtime optimization settings, and focused diagnostics/tests needed to reproduce and verify.
- Preserve the low-small redesign constraints:
  - no decode-time hard projection;
  - no touchdown snapping;
  - no hard foot separation;
  - no optimize-then-snap repair.

## Acceptance Metrics

From the 2026-05-28 low-small loss redesign spec:

- `plane_env_count > 0`
- `crossing_leg_count > 0`
- `fk_semantic_collision_count == 0`
- `fk_semantic_collision_rate == 0`
- `fk_semantic_min_clearance_over_semantic_m >= 0`
- `planned_vs_fk_foot_error_crossing_leg_max_m <= 0.05` preferred, `0.08` only as a documented fallback threshold.

Additional row `8`, col `12` focus:

- Record planned-vs-FK foot error for the reproduced viewer terrain.
- Record whether velocity/progress losses dominate the final loss breakdown.
- Confirm `planner_backend` is `mpc`, not `mpc_qp`.

## Related Logs

- [../log/2026-07-06-mpc-row8-col12-rpy-diagnostic-fix.md](../log/2026-07-06-mpc-row8-col12-rpy-diagnostic-fix.md)
- [../log/2026-07-06-mpc-row8-col12-stop-after-motion-repro.md](../log/2026-07-06-mpc-row8-col12-stop-after-motion-repro.md)
- [../log/2026-07-06-mpc-row8-col12-nonzero-weight-tuning.md](../log/2026-07-06-mpc-row8-col12-nonzero-weight-tuning.md)
- [../log/2026-07-13-mpc-row8-col11-col12-heightfield-penetration-repro.md](../log/2026-07-13-mpc-row8-col11-col12-heightfield-penetration-repro.md)
- [../log/2026-07-13-mpc-row8-col11-col12-body-shank-clearance-fix.md](../log/2026-07-13-mpc-row8-col11-col12-body-shank-clearance-fix.md)
- [../log/2026-07-15-mpc-low-small-optimize-steps-sweep.md](../log/2026-07-15-mpc-low-small-optimize-steps-sweep.md)
- [../log/2026-07-15-mpc-row8-col12-opt16-position-repro.md](../log/2026-07-15-mpc-row8-col12-opt16-position-repro.md)

## Git Refs

- Last Feature Commit: `8168b15`
- Last Verified Commit: `8168b15` plus dirty working tree on 2026-07-06 23:00 CST
- Current Work Ref: `8168b15` plus dirty working tree on 2026-07-06 23:00 CST
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py](../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py)
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html)

Latest verification ref on 2026-07-13: `8168b15` plus dirty working tree; see [../log/2026-07-13-mpc-row8-col11-col12-body-shank-clearance-fix.md](../log/2026-07-13-mpc-row8-col11-col12-body-shank-clearance-fix.md).

## Next Step

- If user accepts the current weight-only improvement, keep this branch in verify state. If the remaining visual issue is specifically the current-foot-to-future-touchdown marker distance, clarify marker semantics because that is not a pure FK/IK mismatch.

## Node Details

### T302w.7 Row8/Col12 Optimize-16 Position Reproduction

- why-created: user observed foot endpoint/root/planned-versus-actual position problems on row `8`, col `12` after reducing optimizer steps to `16`.
- result: reproduced without code changes. Single-pass planned-foot versus analytic-FK error reaches `0.193788m` forward and `0.226555m` diagonal; forward `x6` reaches `0.243642m`.
- actual playback boundary: planned root/foot versus Isaac readback remains micron-close (`1.092e-6m` root, `4.266e-6m` foot), so the viewer write/readback path is not drifting.
- rough-terrain failure: forward `x6` reaches root Z step `0.205379m`, raw IK/calf saturation `0.837800rad`, and FK foot/knee/shank/body min clearances `-0.172954/-0.122214/-0.125601/-0.048149m`.
- interpretation: planner foot targets versus analytic FK/joint realization are inconsistent at 16 steps; touchdown-current distance is a separate future-target metric.
- evidence: [2026-07-15 row8/col12 optimize-16 position repro](../log/2026-07-15-mpc-row8-col12-opt16-position-repro.md).

### T302w.8 Optimize-16 Versus 24/25 Current-Runtime A/B

- why-created: the 16-step low-small result does not generalize to rough row8/col12, and current opt16 metrics conflict sharply with older opt24 evidence.
- next: run identical current full-grid forward/diagonal/x6 cases at step `16` and `24/25`, then compare analytic FK error, Isaac readback, root Z steps, joint saturation, and heightfield penetration.

### T302w.9 Row8/Col12 Probe API Drift

- why-created: the existing untracked probe no longer launches directly because the fixture removed `compact_semantic_grid` and `move_env0_to_terrain_tile`.
- current workaround: process-only monkeypatch used for this diagnostic; no file changes.
- next: update the probe to current full-grid fixture APIs only after user authorizes code changes.

### T302w.6 Low-Small Optimizer-Step Ablation

- why-created: user asked whether the viewer MPC gradient-descent count can be reduced from 25 without losing cylinder/cone low-small crossing behavior.
- result: deterministic S4 diagonal sweep completed for every `optimize_steps=0..25` on one cylinder and one cone anchor; all counts crossed with zero FK semantic collision.
- quality boundary: step `6` first meets crossing planned-vs-FK error `<=0.05m`; step `16` first also meets conservative raw IK joint-limit violation `<=0.01rad`.
- recommendation: step `16` is the smallest practical candidate, not a production-default change. It was about `20-34%` faster than step `25` in this run, with about `5.2%` less progress and lower clearance margin.
- limitation: the compact fixture exposed only one anchor per requested shape; independent placements/seeds and multi-replan playback remain unverified.
- evidence: [2026-07-15 optimize-step sweep](../log/2026-07-15-mpc-low-small-optimize-steps-sweep.md).

### T302w.1 Reproduce Row8 Col12 MPC Mismatch

- why-created: user reported planned foot and actual/FK foot mismatch on `terrain-row 8`, `terrain-col 12` using `planner_backend=mpc`.
- original hypothesis: existing `mpc` loss weighting gives velocity/progress tracking too much authority, letting optimized target feet drift away from reachable FK feet.
- evidence:
  - initial diagnostic baseline reported `planned_fk_after_frame0_error_max_m ~= 0.188628m`;
  - temporary probe-only `ik_fk x4 + kinematics x4` reduced the apparent error only to `~0.1497m`, and lowering progress/swing did not fix it;
  - root cause was `_root_rpy_from_viewer_result()` using yaw-only fallback from `root_quat_w`, so FK was evaluated against a flat body attitude on sloped terrain;
  - after full RPY extraction, row `8`, col `12` `mpc` baseline reports `planned_fk_after_frame0_error_max_m = 2.7079e-6`, `terminal_planned_vs_fk_foot_error_max = 2.7079e-6`, and `playback_readback_error_max_m = 4.266e-6`.
- dynamic move-then-stop evidence:
  - sequence `move_v050 x2; stop x4` without reset reproduces the user-visible issue;
  - row `8`, col `12` stop phase has `planned_vs_fk_foot_error_all_max_m ~= 0.17768`, `playback_readback_error_max_m ~= 0.17768`, and `touchdown_to_current_actual_foot_error_max_m ~= 0.17768`;
  - row `0`, col `0` control has stop phase `~0.02500m` for the same metrics;
  - row `8`, col `12` stop phase also has `raw_ik_joint_limit_violation_max ~= 0.83780` and `calf_upper_saturation_max ~= 0.83780`, while the control row is `0`.
- latest nonzero-speed evidence:
  - baseline command matrix over forward/back/lateral/diagonal/mixed directions has `max_playback_readback_error_m = 4.5942e-6`, `max_terminal_planned_vs_fk_foot_error_m = 2.7079e-6`, and raw IK violation around `1e-5rad`;
  - large visible metrics are touchdown marker distances: `max_touchdown_to_contact_frame_foot_error_m = 0.53763`, `max_touchdown_to_current_actual_foot_error_m = 0.61080`;
  - therefore the latest nonzero-speed reproduction does not show FK/IK inconsistency; it shows future touchdown marker alignment differences.

### T302w.2 Tune Existing Loss Weights

- why-created: user explicitly requested loss tuning and disallowed new loss terms.
- result: tuned existing `mpc` weights only. Connected two already-existing weight configs that were previously ineffective/not exposed (`root_foot_center.weight`, `touchdown_endpoint.weight`) and set row/viewer cfg weights to:
  - `ik_fk_residual.weight = 16.0`;
  - `kinematics.weight = 3.0`;
  - `root_foot_center.weight = 4.0`;
  - `touchdown_endpoint.weight = 16.0`;
  - `progress.weight = 0.35`;
  - `swing_direction.weight = 0.25`.
- final nonzero matrix:
  - `max_playback_readback_error_m = 5.4001e-6`;
  - `max_terminal_planned_vs_fk_foot_error_m = 4.2716e-6`;
  - `max_raw_ik_joint_limit_violation = 2.3365e-5`;
  - `max_touchdown_to_contact_frame_foot_error_m = 0.47605`;
  - `max_touchdown_to_current_actual_foot_error_m = 0.60490`.
- guard: no new loss was added, and no `mpc_qp` path was touched for this branch.

### T302w.3 Verify Low-Small Redesign Metrics

- why-created: user required the 2026-05-28 spec metrics after tuning and said they must not be broken.
- evidence:
  - after weight update, `mpc_low_small_reachable_crossing_probe.py --variants baseline --cycles 1 --requested-n-frames 25 --warmup-steps 6` exited `0`;
  - diagonal case had `crossing_leg_count = 1`;
  - diagonal `fk_semantic_collision_count = 0`;
  - diagonal `fk_semantic_min_clearance_over_semantic_m = 0.08004`;
  - diagonal `planned_vs_fk_foot_error_crossing_leg_max_m = 9.7759e-7`;
  - summary `max_terminal_planned_vs_fk_foot_error = 9.7759e-7`.
- caveat: the broader probe output still reports nonzero contact/touchdown-on-small rates (`max_fk_touchdown_on_small_rate = 0.25`, `max_fk_stance_on_small_rate = 0.01`, `max_fk_foot_small_penetration_rate = 0.02`). These are tracked separately from the hard 2026-05-28 FK semantic collision / planned-vs-FK acceptance keys.

### T302w.5 Reproduce Farther-Walk Penetration Probe

- why-created: the user clarified that the initial flat start is not representative; the real rough-tile failure appears only after walking farther.
- evidence:
  - sequence probe `move_v050:0.50,0.00,0.00x6` on row `8`, col `11` stays much milder through the sequence;
  - row `8`, col `12` becomes strongly bad after several cycles: FK foot, knee, shank, and body footprint samples all penetrate the height field, and `root_step_z_max_m` grows to `0.16187m`;
  - late-cycle row `8`, col `12` also shows `planned_vs_fk_foot_error_all_max_m = 0.24908m`, `touchdown_to_contact_frame_foot_error_max_m = 0.40622m`, and `raw_ik_joint_limit_violation_max = 0.83780`.
- mitigation:
  - root cause was partly underbody/root-z safety mismatch: the loss did not rotate underbody samples with root yaw, and decode root `z` only used root-center terrain support rather than yaw-rotated body footprint support;
  - final config kept `fk_body_leg_collision.weight = 640.0`, raised `shank_margin_m = 0.12`, `knee_margin_m = 0.04`, `underbody_margin_m = 0.05`, and set `shank_sample_count = 5`;
  - final row8/col11 summary: body/knee/shank penetration counts `0`, `min_fk_shank_ground_clearance_m = 0.044592`, `max_terminal_planned_vs_fk_foot_error_m = 3.815e-6`;
  - final row8/col12 summary: body/knee/shank penetration counts `0`, `min_fk_shank_ground_clearance_m = 0.044592`, `max_terminal_planned_vs_fk_foot_error_m = 2.708e-6`;
  - residual: both rough-tile probes retain small initial-frame foot clearance residual around `-0.00463m`, and long-sequence touchdown/current marker distances are not solved by this FK body-leg collision change.
