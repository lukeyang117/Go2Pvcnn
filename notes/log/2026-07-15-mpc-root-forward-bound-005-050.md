# 2026-07-15 MPC Root Forward Bound 0.05-0.50m

## Purpose

Restrict the existing smooth parametric root forward correction so the nominal `0.25m` endpoint can move within approximately `0.05-0.50m`, matching the current `25 * 0.02s = 0.5s` planning horizon at up to `1.0m/s`.

## Stage

- `extension/batch_mpc_planner` parametric trajectory decoding.
- CPU-only parametric contract test.

## Related Todo

- [T302w.11](../todo/T302w-mpc-row8-col12-loss-tuning.md#t302w11-root-forward-smooth-bound-005-050m)

## Command / Procedure

Changed only the positive bound passed to `_smooth_asymmetric_zero_offset` for `root_goal_delta_raw[:, 0]` from `0.75` to `0.25`. Updated the existing saturation test accordingly.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py
```

## Input Conditions

- Nominal forward root endpoint: `0.25m`.
- Smooth correction bounds: `[-0.20, +0.25]m`.
- Loss weights and progress loss unchanged.
- No IsaacLab or viewer startup.

## Key Metrics

- Large negative raw value: terminal root X `0.05m`.
- Zero raw value: terminal root X `0.25m`.
- Large positive raw value: terminal root X `0.50m`.
- Zero-input gradient remains finite and positive.
- Focused tests: `22 passed in 1.74s`.
- `git diff --check`: pass.

## Result

Pass. Root forward correction remains smooth and zero-preserving while its terminal forward range is reduced from approximately `0.05-1.00m` to `0.05-0.50m` around the unchanged `0.25m` nominal endpoint.

## Conclusion

This changes the optimizer's available root endpoint range only. It does not change the one-sided progress target, which remains capped at `0.35m`, and it does not prove row8/col12 behavior without a runtime probe.

## Follow-Up

- Keep the progress-loss cap as a separate issue unless the user requests it.
- Run row8/col12 or low-small runtime behavior only if requested.

## Git Refs

- Baseline Ref: `56b9ae6` plus pre-existing dirty working tree.
- Candidate Ref: `56b9ae6` plus this local change and pre-existing dirty working tree.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)

