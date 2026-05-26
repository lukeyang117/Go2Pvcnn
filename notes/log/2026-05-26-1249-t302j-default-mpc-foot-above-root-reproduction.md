# T302j Default MPC Foot Above Root Reproduction

## Purpose

Reproduce the user's visual complaint that while crossing small obstacles, the swing foot trajectory can rise above the root.

## Stage

- `extension/batch_mpc_planner`
- Default MPC runtime path, not V12 debug-gated.

## Related Todo

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)
- Parent: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command

First attempt with `CUDA_VISIBLE_DEVICES=1 --device cuda:1` failed because Isaac/Omniverse saw only one visible GPU and `cuda:1` became an invalid ordinal.

Successful command:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:1 \
  --commands 'forward_v050:0.50 0.00 0.00,mixed_yaw_v050:0.50 0.25 1.00' \
  --variants baseline \
  --cycles 1 \
  --requested-n-frames 300 \
  --warmup-steps 6 \
  > tmp/t302i-viewer-realized-foot-mismatch/t302j_default_mpc_above_root_repro.jsonl 2>&1
```

## Input Conditions

- Default small obstacle: `height=0.16m`, `diameter=0.12m`.
- Default MPC behavior after farthest-touchdown export moved into the normal MPC path.
- Commands:
  - `forward_v050 = (0.50, 0.00, 0.00)`
  - `mixed_yaw_v050 = (0.50, 0.25, 1.00)`

## Key Metrics

| Command | planned swing foot above root | FK swing foot above root | root height min | base bottom clearance min | touchdown behind swing | small contact/penetration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forward_v050 | `+0.025513m` | `+0.073637m` | `0.142901m` | `0.042901m` | `0.0m` | `0` |
| mixed_yaw_v050 | `-0.005118m` | `-0.008270m` | `0.289896m` | `0.189896m` | `0.0m` | `0` |

Additional forward metrics:

- `touchdown_behind_fk_foot_along_max_m=0.104883`
- `touchdown_ik_fk_error_max=0.682283`
- `terminal_planned_vs_fk_foot_error_max=0.323130`
- `command_direction_cosine=0.999533`
- `lateral_drift_m=0.078791`
- `fk_swing_foot_step_max_to_median=20.058457`
- `fk_swing_foot_accel_max_to_mean=13.486868`

Additional mixed-yaw metrics:

- `touchdown_behind_fk_foot_along_max_m=0.098926`
- `touchdown_ik_fk_error_max=0.733100`
- `terminal_planned_vs_fk_foot_error_max=0.388287`
- `command_direction_cosine=-0.595472`
- `lateral_drift_m=0.399854`
- `fk_swing_foot_step_max_to_median=12.329788`
- `fk_swing_foot_accel_max_to_mean=11.311711`

## Result

Pass as reproduction. The forward low-small crossing case reproduces the foot-above-root failure:

- Planned swing foot exceeds root height by `2.55cm`.
- FK-realized swing foot exceeds root height by `7.36cm`.

The mixed-yaw row does not reproduce foot-above-root in this run, but it still fails direction tracking. This separates the visual high-foot issue from the mixed-yaw direction issue.

## Conclusion

The current default MPC farthest-touchdown export fixed the exported touchdown being behind the planned swing path, but it did not solve foot-height shape. The forward low-small crossing still uses an excessive swing arc relative to root height, and FK/clamped-IK realization amplifies it.

## Follow-Up

Next step should add a default MPC foot-above-root guard or reshape the swing arc so low-small clearance is achieved over the obstacle without allowing foot height above root. The guard must preserve obstacle clearance and avoid the low-base/spider workaround.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree @ c54dc5c plus default MPC farthest-touchdown export`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py](../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py)
