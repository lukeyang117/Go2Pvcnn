# T302j V11 Endpoint Consistency Probe

## Purpose

Implement and test a probe-only `reachable_fk_cross_v11` debug variant for command-frame touchdown endpoint consistency, sampled-touchdown FK reachability, and foot-above-root guard.

## Stage

`extension/batch_mpc_planner` debug variant and T302i reachable low-small probe.

## Related Todo

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)
- Parent: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Commands

Local:

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
```

Real IsaacLab:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --commands 'mixed_yaw_v050:0.50 0.25 1.00' \
  --variants reachable_fk_cross_v11 \
  --cycles 1 \
  --requested-n-frames 300 \
  --warmup-steps 6 \
  > tmp/t302i-viewer-realized-foot-mismatch/t302j_v11_mixed_yaw_only.jsonl 2>&1
```

## Result

Partial pass as diagnosis. V11 successfully targets endpoint consistency and foot-above-root, but does not yet solve touchdown IK/FK consistency.

## Key Metrics

Mixed-yaw baseline / V9 / V11:

- `touchdown_behind_swing_foot_along_max_m`: `0.569856 -> 0.314493 -> 0.124781`
- `touchdown_behind_planned_foot_along_max_m`: `0.569856 -> 0.316026 -> 0.211320`
- `planned_swing_foot_above_root_z_max_m`: `-0.005118 -> 0.120522 -> 0.013994`
- `fk_swing_foot_above_root_z_max_m`: `-0.008270 -> 0.069278 -> 0.007172`
- `command_direction_cosine`: `-0.595472 -> 0.998519 -> 0.995655`
- `lateral_drift_m`: `0.399854 -> 0.069376 -> 0.099620`
- `raw_ik_joint_limit_violation_max`: `2.039664 -> 1.683324 -> 1.095759`
- `terminal_planned_vs_fk_foot_error_max`: `0.388287 -> 0.353239 -> 0.371493`
- `touchdown_ik_fk_error_max`: `0.661772 -> 0.745050 -> 0.786326`
- `fk_swing_foot_step_max_to_median`: `12.329788 -> 6.369946 -> 10.653463`
- small contact/penetration metrics remain `0`.

Local verification:

- `py_compile`: pass.
- `pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py`: `31 passed`.

## Conclusion

V11 confirms the endpoint/height hypothesis is actionable:

- endpoint backtracking can be reduced from `0.57m` to `0.125m`;
- V9's above-root foot arc can be suppressed;
- direction tracking can remain good.

However, V11 does not solve the underlying touchdown IK/FK mismatch. The sampled-touchdown reachability term is present but insufficient or poorly aligned with the exported touchdown metric. Next variant should focus on sampled touchdown reachability before adding more endpoint pressure.

## Follow-Up

- T302j.2 remains open: sampled-touchdown FK reachability needs stronger or better-conditioned implementation.
- Consider V12:
  - reduce endpoint/height weights slightly to recover continuity;
  - increase direct sampled-touchdown reachability;
  - compute sampled touchdown reachability using the same exported `planned_touchdown_w`/phase path used by metrics.

## Git Refs

- Baseline Ref: `c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py](../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
