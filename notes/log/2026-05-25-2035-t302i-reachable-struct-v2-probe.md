# T302i Reachable Struct V2 Probe

## Purpose

Test a probe-only `reachable_struct_v2` loss that jointly penalizes decoded-foot IK/FK residual, touchdown-phase IK/FK residual, and FK-realized foot smoothness before any production planner change.

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
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_loss_v1,reachable_loss_v2,reachable_struct_v1,reachable_struct_v2 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_reachable_struct_v2.jsonl 2>&1
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

- RED failed on missing `reachable_struct_v2` extra loss.
- GREEN: `pytest` -> `9 passed`.
- `py_compile`: pass.

Forward comparison:

| Variant | Planned-vs-FK | Touchdown IK/FK | Raw IK Violation | Swing Step Ratio | Swing Accel Ratio | Direction Cos | Lateral Drift | Small Penetration | Foot-Over |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | `0.323130` | `0.682283` | `2.470606` | `20.058457` | `13.486868` | `0.999533` | `0.078791` | `0` | `1` |
| `reachable_loss_v1` | `0.275787` | `0.635461` | `0.851645` | `21.040013` | `16.267543` | `0.998257` | `0.141103` | `0` | `1` |
| `reachable_loss_v2` | `0.287762` | `0.599881` | `0.903613` | `25.737349` | `26.145389` | `0.985843` | `0.326323` | `0` | `1` |
| `reachable_struct_v1` | `0.371620` | `0.675879` | `0.837800` | `6.262593` | `7.331959` | `0.992749` | `0.333843` | `0` | `1` |
| `reachable_struct_v2` | `0.350369` | `0.594303` | `0.837800` | `15.809417` | `13.440590` | `0.997748` | `0.158822` | `0` | `1` |

## Result

Partial pass as a diagnostic, not an accepted fix.

`reachable_struct_v2` improves touchdown IK/FK error and swing step ratio compared with baseline, but worsens whole-trajectory planned-vs-FK error. It is not strong enough for all-direction testing or production.

## Conclusion

The next loss direction should not simply increase touchdown residual weight. The failure now looks like a whole decoded-foot reachability issue: touchdown-phase residual improves, but exported `foot_pos` still enters unreachable regions after clamped IK.

The next probe-only direction should add a full-horizon reachability barrier / saturation-distance term for decoded feet, while keeping FK continuity weights lower than `struct_v1`.

## Follow-Up

- Create `reachable_struct_v3` as a probe-only full-horizon reachability barrier:
  - stronger penalty on max decoded-foot FK residual, not only mean/contact residual;
  - explicit raw joint-limit excess or calf saturation proxy;
  - low-weight FK step/accel smoothness;
  - no selector/postprocess/viewer change.
- Re-run forward comparison only. Do not run all-direction until both planned-vs-FK and swing continuity improve together.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_reachable_struct_v2.jsonl`
