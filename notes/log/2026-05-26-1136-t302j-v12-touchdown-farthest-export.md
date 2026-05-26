# T302j V12 Touchdown Farthest Export

## Purpose

Verify the user's latest priority: for low-small mixed-yaw crossing, the exported `touchdowns` must not sit behind any swing foot point along the translational command direction.

## Stage

- `extension/batch_mpc_planner`
- Debug/runtime variant: `reachable_fk_cross_v12`

## Related Todo

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)
- Parent: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --commands 'mixed_yaw_v050:0.50 0.25 1.00' \
  --variants reachable_fk_cross_v12 \
  --cycles 1 \
  --requested-n-frames 300 \
  --warmup-steps 6 \
  > tmp/t302i-viewer-realized-foot-mismatch/t302j_v12c_farthest_export_mixed_yaw_only.jsonl 2>&1
```

## Input Conditions

- Real IsaacLab startup through `env_isaacsim`.
- One mixed translation+yaw command: `vx=0.50`, `vy=0.25`, `yaw=1.00`.
- Rolling 25-frame replans inside a 300-frame requested rollout.

## Key Metrics

Baseline/V9/V11 references from earlier T302j logs:

| Variant | touchdown behind swing | touchdown IK/FK | command direction cosine | small contact/penetration |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.569856` | `0.661772` | `-0.595472` | `0` |
| V9 | `0.314493` | `0.745050` | `0.998519` | `0` |
| V11 | `0.124781` | `0.786326` | `0.995655` | `0` |
| V12 loss-only | `0.180266` | `0.702144` | `0.994882` | `0` |
| V12 hinge | `0.150514` | `0.563487` | `0.881158` | `0` |
| V12 farthest export | `0.000000` | `0.563487` | `0.881158` | `0` |

Detailed V12 farthest-export metrics:

- `touchdown_behind_swing_foot_along_max_m=0.0`
- `touchdown_behind_planned_foot_along_max_m=0.12862664461135864`
- `touchdown_behind_fk_foot_along_max_m=0.21159471571445465`
- `touchdown_ik_fk_error_max=0.5634872317314148`
- `terminal_planned_vs_fk_foot_error_max=0.3423762321472168`
- `command_direction_cosine=0.881157636642456`
- `lateral_drift_m=0.7863688468933105`
- `fk_stance_on_small_rate=0.0`
- `fk_touchdown_on_small_rate=0.0`
- `fk_foot_small_penetration_rate=0.0`
- `planned_swing_foot_above_root_z_max_m=0.08920276165008545`
- `fk_swing_foot_above_root_z_max_m=0.1371837705373764`
- `fk_swing_foot_step_max_to_median=7.267355042725247`
- `fk_swing_foot_accel_max_to_mean=10.235263007254359`
- `root_height_min=0.10140658915042877`

## Result

Pass for the latest P0 endpoint/export contract: `touchdown_behind_swing_foot_along_max_m` is exactly `0.0` for the mixed-yaw reproduction row.

The successful change is not just increasing scalar loss. V12 now exports touchdown markers from the command-direction farthest swing point per leg, grounded to terrain height, so the visible touchdown cannot be behind the visible planned swing arc.

## Conclusion

- The user's current priority is fixed for V12: planner/exported touchdown is the farthest point along translation command direction for the swing.
- Secondary issues remain:
  - FK-realized foot can still be ahead of exported touchdown by `0.2116m`.
  - Direction/lateral behavior regressed versus V11 (`cos=0.881`, `lateral_drift=0.786m`).
  - Planned/FK foot above root remains high (`0.089/0.137m`).
  - Swing acceleration ratio is still high (`10.235`).

## Follow-Up

Next slice should keep V12 farthest-export contract and then address FK-realized endpoint consistency, direction/lateral drift, and foot-above-root guard without moving viewer markers to FK readback.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree @ c54dc5c plus V12 farthest-export changes`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py](../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
