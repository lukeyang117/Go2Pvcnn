# T302i Reachable Struct V3 Barrier Probe

## Purpose

Test whether a probe-only full-horizon reachability barrier can improve both exported planned-vs-FK consistency and FK-realized swing continuity.

## Stage

- `extension/batch_mpc_planner`
- `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
- related todo: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command / Procedure

Local RED/GREEN:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
python -m py_compile Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
```

IsaacLab forward comparison:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_loss_v1,reachable_struct_v1,reachable_struct_v2,reachable_struct_v3 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_reachable_struct_v3.jsonl 2>&1
```

## Input Conditions

- Python: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Device: `cuda:0`
- Case: low-small forward
- Total rollout: `300`
- Rolling replan horizon: `25`
- Warmup: `6`

## Key Metrics

Local checks:

- RED failed on missing `reachable_struct_v3`.
- GREEN: `pytest` -> `10 passed`.
- `py_compile`: pass.

Forward comparison:

| Variant | Planned-vs-FK | Touchdown IK/FK | Raw IK Violation | Swing Step Ratio | Swing Accel Ratio | Direction Cos | Lateral Drift | Small Penetration | Foot-Over |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | `0.323130` | `0.682283` | `2.470606` | `20.058457` | `13.486868` | `0.999533` | `0.078791` | `0` | `1` |
| `reachable_loss_v1` | `0.275787` | `0.635461` | `0.851645` | `21.040013` | `16.267543` | `0.998257` | `0.141103` | `0` | `1` |
| `reachable_struct_v1` | `0.371620` | `0.675879` | `0.837800` | `6.262593` | `7.331959` | `0.992749` | `0.333843` | `0` | `1` |
| `reachable_struct_v2` | `0.350369` | `0.594303` | `0.837800` | `15.809417` | `13.440590` | `0.997748` | `0.158822` | `0` | `1` |
| `reachable_struct_v3` | `0.347461` | `0.589840` | `0.837800` | `11.313112` | `11.765254` | `0.983657` | `0.428828` | `0` | `1` |

## Result

Partial pass as a diagnostic, rejected as a fix direction.

`reachable_struct_v3` improves touchdown IK/FK and swing continuity relative to `struct_v2`, but still worsens whole-trajectory planned-vs-FK against baseline and introduces large lateral drift.

## Conclusion

Adding heavier loss terms does not solve the core contract. The optimizer can reduce sampled touchdown residual and smooth FK motion while still exporting decoded Cartesian feet that are not the same as clamped-IK FK feet.

The next design should move from pure penalty stacking to a reachability-aware trajectory representation or generation rule:

- generate candidate foot targets inside the IK/FK reachable set before optimization;
- or optimize joint / reachable-foot coordinates and derive Cartesian feet from FK;
- or enforce a differentiable projection onto the reachable set before the exported `foot_pos` and touchdown are sampled.

This still preserves the user's constraint that the viewer should remain unchanged; the planner output must become feasible.

## Follow-Up

- Do not run all-direction for `struct_v3`.
- Discuss the production-facing direction before implementation:
  - penalty-only may not be enough;
  - likely fix belongs around `plan_segment()` decode/export contract or reachable foot target parameterization.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_reachable_struct_v3.jsonl`
