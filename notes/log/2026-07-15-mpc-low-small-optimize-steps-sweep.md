# 2026-07-15 MPC Low-Small Optimize-Steps Sweep

## Purpose

Quantify whether the current `planner_backend=mpc` can reduce gradient-descent iterations from the 25-step viewer baseline while preserving low-small obstacle crossing for cylinder and cone shapes.

## Stage

- `extension/batch_mpc_planner` low-small crossing acceptance.
- Probe-only instrumentation in `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`.
- No production planner loss, weight, decode, or runtime default changed.

## Related Todo

- [T302w.6](../todo/T302w-mpc-row8-col12-loss-tuning.md#t302w6-low-small-optimizer-step-ablation)

## Command / Procedure

The probe uses the same semantic MPC task, `cuda:0` mapped from physical GPU 1, one environment, 25 frames, `dt=0.02`, one deterministic S4 low-small anchor per requested native shape, a fixed diagonal command, and an identical relative start pose. Only `runtime.optimize_steps` changes.

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --commands 'diagonal_v050:0.35 0.35 0.00' \
  --variants baseline \
  --cycles 1 \
  --requested-n-frames 25 \
  --warmup-steps 6 \
  --longitudinal-offset-m -0.35 \
  --shape-kinds cylinder,cone \
  --optimize-steps 0-25
```

Raw mixed Isaac/JSONL output:

- `tmp/mpc_optimize_steps_sweep/cylinder_cone_diagonal_steps_0_25.jsonl`
- Forward control: `tmp/mpc_optimize_steps_sweep/cylinder_cone_steps_0_25.jsonl`

## Metrics And Thresholds

- Crossing behavior: `crossing_leg_count > 0`, `fk_foot_over_low_small_success == 1`.
- Collision safety: `fk_semantic_collision_count == 0`, `fk_semantic_min_clearance_over_semantic_m >= 0`.
- Existing preferred consistency gate: `planned_vs_fk_foot_error_crossing_leg_max_m <= 0.05`.
- Additional conservative reachability gate for this sweep: `raw_ik_joint_limit_violation_max <= 0.01 rad`.
- Runtime: CUDA-synchronized `plan_time_ms`.

## Key Results

All 26 iteration counts crossed with one crossing leg and zero FK semantic collision for both shapes in the covered deterministic anchor. This alone is insufficient to call all counts equivalent because low-step target reachability is much worse.

| Shape / steps | Cross | FK collision | Min semantic clearance | Crossing planned-vs-FK error | Raw IK limit violation | Progress | Plan time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cylinder / 0 | 1 | 0 | `0.08061m` | `0.06720m` | `0.26051rad` | `0.25000m` | `140.98ms` |
| cylinder / 6 | 1 | 0 | `0.08929m` | `0.04932m` | `0.15866rad` | `0.22702m` | `492.46ms` |
| cylinder / 16 | 1 | 0 | `0.03730m` | `0.04071m` | `0.00233rad` | `0.20643m` | `1183.92ms` |
| cylinder / 25 | 1 | 0 | `0.05068m` | `0.03551m` | `0.00079rad` | `0.21773m` | `1483.61ms` |
| cone / 0 | 1 | 0 | `0.12911m` | `0.06720m` | `0.26052rad` | `0.25000m` | `81.23ms` |
| cone / 6 | 1 | 0 | `0.15880m` | `0.04932m` | `0.15866rad` | `0.22702m` | `472.69ms` |
| cone / 16 | 1 | 0 | `0.08548m` | `0.04071m` | `0.00233rad` | `0.20643m` | `1177.68ms` |
| cone / 25 | 1 | 0 | `0.10140m` | `0.03551m` | `0.00079rad` | `0.21773m` | `1789.10ms` |

- Step 6 is the first count where both shapes meet the existing preferred crossing planned-vs-FK error threshold.
- Step 16 is the first count where both shapes also meet the `0.01rad` raw IK reachability gate.
- Relative to step 25 in this run, step 16 reduced synchronized plan time by about `20.2%` for cylinder and `34.2%` for cone. It retained positive semantic clearance but reduced progress by about `5.2%` and reduced clearance margin.
- The forward control did not produce a crossing leg at any count because 25 frames moved the root only about `0.21-0.25m` from a `-0.35m` start. It is not used for the crossing conclusion.

## Result

Partial pass with a practical recommendation:

- `0` iterations can cross this covered diagonal case, but it is not equivalent to 25 iterations in target reachability.
- `6` iterations is the minimum for the existing preferred crossing consistency gate, but raw IK violation remains too large.
- `16` iterations is the minimum conservative candidate that preserves the covered cylinder/cone crossing and safety gates while materially reducing planning time.
- `25` iterations retains better progress, clearance margin, and IK consistency. Do not change the production default from this single-anchor probe alone.

## Verification

- Real Isaac sweep: `52/52` expected result rows, exit `0`.
- Focused unaffected unit subset: `19 passed, 13 deselected`.
- Full probe unit file: `21 passed, 11 failed`; all 11 failures are pre-existing stale tests for removed `reachable_fk_cross_v1..v9` debug variants.
- `py_compile`: exit `0`.
- `git diff --check` for the probe: exit `0`.

## Follow-Up

- Before changing `MpcRuntimeCfg.optimize_steps`, run step 16 versus step 25 across more independent low-small placements/seeds and a multi-replan playback trajectory. The compact S4 fixture exposed only one cylinder and one cone anchor, so this log does not claim statistical equivalence.
- Keep the stateful `cycles>1` result separate: it changes the physical initial state between cycles and is not an independent repeat of this ablation.

## Git Refs

- Baseline Ref: `1c951ec` plus pre-existing dirty working tree.
- Candidate Ref: `1c951ec` plus probe-only instrumentation and notes.
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../todo/T302w-mpc-row8-col12-loss-tuning.md](../todo/T302w-mpc-row8-col12-loss-tuning.md)

