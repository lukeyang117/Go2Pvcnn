# 2026-07-15 MPC Row8/Col12 Optimize-16 Position Reproduction

## Purpose

Reproduce the reported foot endpoint, root, planned-position, and actual-position problem on terrain row `8`, col `12` with the existing MPC optimizer reduced to `16` steps. No repository code or planner configuration was changed.

## Stage

- `extension/batch_mpc_planner` viewer-style rough-terrain diagnostics.
- Existing `Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py` metrics.

## Related Todo

- [T302w.7 / T302w.8](../todo/T302w-mpc-row8-col12-loss-tuning.md#t302w7-row8col12-optimize-16-position-reproduction)

## Command / Procedure

Core probe settings:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py \
  --device cuda:0 \
  --terrain-row 8 \
  --terrain-col 12 \
  --requested-n-frames 25 \
  --playback-frames 25 \
  --warmup-steps 4 \
  --optimize-steps 16
```

Runs:

- single pass: forward `0.50,0,0`, then diagonal `0.35,0.20,0`;
- continuous no-reset sequence: forward `0.50,0,0 x6`;
- explicit 25-frame planned root/foot versus Isaac readback pass.

The existing untracked probe had drifted behind the current fixture API: `compact_semantic_grid` and `move_env0_to_terrain_tile` no longer exist. To respect the no-code-change request, the process used in-memory compatibility only: ignored the stale constructor keyword, disabled the current compact `4x1` semantic grid, and mapped the old tile move to current terrain selection/root placement/scanner grounding calls. Files on disk were not edited.

Raw output:

- `tmp/mpc_row8_col12_opt16/single_cycle.jsonl`
- `tmp/mpc_row8_col12_opt16/forward_x6.jsonl`
- `tmp/mpc_row8_col12_opt16/root_foot_readback.jsonl`

## Input Conditions

- Backend: `mpc`, not `mpc_qp`.
- `optimize_steps=16` confirmed in probe header.
- Full task terrain: row `8`, col `12`.
- Horizon/playback: `25/25`, `dt=0.02`.
- Existing weights only: `ik_fk=8`, `kinematics=1`, `fk_body_leg_collision=120`, `progress=1`, `swing_direction=1`.

## Key Metrics

Single pass:

| Case | Planned foot vs analytic FK | Planned foot vs Isaac readback | Root max step Z | Touchdown vs current actual foot | Raw IK violation | Heightfield result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| forward | `0.193788m` | `4.266e-6m` | `0.006803m` | `0.326320m` | `0.837800rad` | FK foot min `-0.045789m`, `17` penetrations |
| diagonal after forward | `0.226555m` | `3.692e-6m` | `0.151759m` | `0.369960m` | `0.837800rad` | FK foot/shank/body min `-0.080202/-0.045851/-0.047683m`; counts `35/28/32` |

Continuous forward `x6` summary:

- max planned-foot versus analytic-FK error: `0.243642m`;
- max planned-foot versus Isaac-foot readback error: `4.266e-6m`;
- max touchdown versus current actual foot: `0.446452m`;
- max root step: XY `0.017032m`, Z `0.205379m`;
- raw IK violation / calf upper saturation: `0.837800rad`;
- min FK foot/knee/shank/body clearances: `-0.172954/-0.122214/-0.125601/-0.048149m`;
- max FK foot/knee/shank/body penetration counts: `40/13/46/42`.

Explicit playback readback for the forward plan:

- planned root position versus Isaac root position: max `1.092e-6m`;
- planned foot position versus Isaac body foot position: max `4.266e-6m`.

## Result

Reproduced at `optimize_steps=16`.

The evidence separates three different quantities:

1. Planner root/foot arrays versus Isaac playback readback are micron-close. The viewer write/readback path does not show root or foot position drift.
2. Planner foot arrays versus the analytic FK reconstructed from the result joint angles diverge by `0.194-0.244m`. This is the main planned-versus-realizable trajectory inconsistency at 16 steps.
3. Touchdown markers are future targets and remain `0.326-0.446m` from the current actual feet; this should not be labeled pure playback error.

The 16-step rough-terrain plan also reaches the calf limit, produces large root-Z discontinuity in later/diagonal cases, and produces analytic FK heightfield penetration. Therefore the earlier low-small recommendation for 16 steps does not generalize to row8/col12 rough terrain.

## Follow-Up

- [T302w.8] Compare step `16` directly with step `24/25` under the same current full-grid runtime and determine whether the mismatch is optimizer convergence, the analytic FK convention, or both.
- [T302w.9] Repair the stale row8/col12 probe API before treating its command as directly runnable; this test intentionally used process-only compatibility and did not change code.

## Git Refs

- Baseline Ref: `1c951ec` plus pre-existing dirty working tree.
- Candidate Ref: same; no code change.
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py](../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../todo/T302w-mpc-row8-col12-loss-tuning.md](../todo/T302w-mpc-row8-col12-loss-tuning.md)

