# T302i Touchdown Chain Trace

## Purpose

Test the hypothesis that nominal first places touchdown far away due to velocity, then small-obstacle / IK constraints pull touchdown back while swing cannot adjust, causing command-frame discontinuity.

## Stage

`extension/batch_mpc_planner` reachable low-small crossing probe.

## Related Todo

- [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --commands 'mixed_yaw_v050:0.50 0.25 1.00' \
  --variants baseline \
  --cycles 1 \
  --requested-n-frames 300 \
  --warmup-steps 6 \
  --trace-touchdown-chain \
  > tmp/t302i-viewer-realized-foot-mismatch/touchdown_chain_mixed_yaw_baseline.jsonl 2>&1
```

## Key Metrics

Command-frame touchdown `along_m` by layer, four legs:

- `nominal_raw`: `[-2.114, -2.776, -0.910, -1.519]`
- `initial_grounded_decode`: `[-2.114, -2.776, -0.910, -1.519]`
- `optimized_export`: `[-1.974, -1.331, -0.664, -1.159]`
- `fk_from_clamped_ik`: `[-0.468, -0.899, -0.621, -0.588]`

Layer deltas:

- `nominal -> initial_grounded`: along delta all `0.0`, only z grounding changes around `0.0016m`.
- `initial_grounded -> optimized_export`: along moves forward by `[0.140, 1.446, 0.246, 0.360]m`.
- `optimized_export -> FK`: FK moves further forward by `[1.506, 0.432, 0.043, 0.571]m`, with z deltas around `0.194-0.235m`.

Same run still shows:

- `planned_swing_along_forward_step_max_m=0.347791`
- `touchdown_behind_swing_foot_along_max_m=0.569856`
- `terminal_planned_vs_fk_foot_error_max=0.388287`
- `touchdown_ik_fk_error_max=0.661772`

## Result

The hypothesis is partially corrected by evidence:

- Nominal did not first place touchdown far in front and then get pulled back in this mixed-yaw baseline row. The nominal touchdown already starts behind the obstacle in command-frame along.
- Grounding does not change touchdown along; it only adjusts z slightly.
- Optimization moves touchdown forward, not backward, relative to nominal.
- The visible rear-touchdown conflict appears because the swing trajectory can move farther forward than the sampled/exported touchdown endpoint.
- IK clamp/FK then moves the realized foot farther from the exported touchdown, adding the planned-vs-realized mismatch.

## Follow-Up

Next fix direction should couple swing path endpoint and sampled touchdown in command frame, and enforce reachability/FK consistency before export. Do not treat nominal distance alone as the root cause.

## Git Refs

- Baseline Ref: `c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
