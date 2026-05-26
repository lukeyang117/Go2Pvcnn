# T302i Reachable Loss Variant Probe

## Purpose

Test probe-only loss variants against the new T302i reachable-crossing metrics before changing production planner behavior.

## Stage

- `extension/batch_mpc_planner`
- `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
- related todo: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command / Procedure

Local checks:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
python -m py_compile Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
```

Forward v1/v2 comparison:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_loss_v1,reachable_loss_v2 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_reachable_loss_v2.jsonl 2>&1
```

Forward structured comparison:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_loss_v1,reachable_loss_v2,reachable_struct_v1 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_reachable_struct_v1.jsonl 2>&1
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

- `pytest`: `8 passed`.
- `py_compile`: pass.

Forward comparison:

| Variant | Planned-vs-FK | Touchdown IK/FK | Raw IK Violation | Swing Step Ratio | Swing Accel Ratio | Direction Cos | Small Penetration | Foot-Over | Root Z Min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | `0.323130` | `0.682283` | `2.470606` | `20.058457` | `13.486868` | `0.999533` | `0` | `1` | `0.142901` |
| `reachable_loss_v1` | `0.275787` | `0.635461` | `0.851645` | `21.040013` | `16.267543` | `0.998257` | `0` | `1` | `0.141283` |
| `reachable_loss_v2` | `0.287762` | `0.599881` | `0.903613` | `25.737349` | `26.145389` | `0.985843` | `0` | `1` | `0.266507` |
| `reachable_struct_v1` | `0.371620` | `0.675879` | `0.837800` | `6.262593` | `7.331959` | `0.992749` | `0` | `1` | `0.217714` |

## Result

Partial pass as direction finding, not an accepted fix.

The probe-only variants execute under IsaacLab and expose the tradeoff:

- `reachable_loss_v1` improves raw IK violation and planned-vs-FK error, but swing continuity gets worse.
- `reachable_loss_v2` slightly improves touchdown IK/FK and root height, but worsens swing continuity and lateral drift.
- `reachable_struct_v1` directly improves FK swing continuity and root height, but worsens planned-vs-FK and touchdown IK/FK.

## Conclusion

Simple config scaling is insufficient. A structured loss that only smooths FK-realized feet conflicts with Cartesian planned-foot/touchdown feasibility. The next step should explicitly target touchdown feasibility and decoded-foot/FK consistency together, not just FK trajectory smoothness.

Concrete next direction:

- Add a `reachable_struct_v2` probe-only loss that includes:
  - FK residual on decoded feet;
  - FK residual on touchdown/export target semantics or equivalent touchdown event proxy;
  - FK foot step/accel smoothing with lower weights than `struct_v1`;
  - command-frame direction/approach gating to avoid forcing over-aggressive crossing.
- Do not run full all-direction sweep for `struct_v1`; forward result is not good enough.

## Follow-Up

- Implement `reachable_struct_v2` in the probe only.
- Re-run forward comparison before any all-direction test.
- If `struct_v2` does not reduce both touchdown IK/FK and swing continuity together, stop and redesign the loss around touchdown export generation rather than decoded foot only.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_reachable_loss_v2.jsonl`
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_reachable_struct_v1.jsonl`
