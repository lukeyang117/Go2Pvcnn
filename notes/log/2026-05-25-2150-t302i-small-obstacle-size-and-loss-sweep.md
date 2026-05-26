# T302i Small Obstacle Size And Loss Sweep

## Purpose

Check whether the low-small obstacle geometry is too large for the current MPC crossing/reachability behavior, and test probe-only loss variants after adding small obstacle diameter control.

## Stage

- `extension/semantic_course`
- `extension/batch_mpc_planner` probe-only loss variants
- `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
- related todo: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Command / Procedure

Local checks:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
python -m py_compile Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py
```

IsaacLab size sweep:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_loss_small_v1 --cycles 1 --requested-n-frames 300 --warmup-steps 6 --semantic-small-diameter-m 0.12 --semantic-small-height-m 0.16 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_diam012_small_v1.jsonl 2>&1
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_loss_small_v1 --cycles 1 --requested-n-frames 300 --warmup-steps 6 --semantic-small-diameter-m 0.10 --semantic-small-height-m 0.16 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_diam010_small_v1.jsonl 2>&1
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_loss_small_v1 --cycles 1 --requested-n-frames 300 --warmup-steps 6 --semantic-small-diameter-m 0.08 --semantic-small-height-m 0.16 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_diam008_small_v1.jsonl 2>&1
```

Focused `0.10m` loss comparison:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,reachable_loss_small_v1,reachable_loss_small_v2 --cycles 1 --requested-n-frames 300 --warmup-steps 6 --semantic-small-diameter-m 0.10 --semantic-small-height-m 0.16 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_diam010_small_v2.jsonl 2>&1
```

## Input Conditions

- Python: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Device: `cuda:0`
- Case: low-small forward
- Height: `0.16m`
- Diameters: `0.12m`, `0.10m`, `0.08m`
- Variants:
  - `baseline`
  - `reachable_loss_small_v1`: smaller low-small loss window and semantic soft margins.
  - `reachable_loss_small_v2`: `small_v1` plus extra low-small continuity/regularization.

## Key Metrics

Local checks:

- `pytest`: `13 passed`.
- `py_compile`: pass.

Size sweep:

| Diameter | Variant | Planned-vs-FK | Touchdown IK/FK | Raw IK Violation | Swing Step | Swing Accel | Direction Cos | Lateral Drift | Penetration | Foot-Over |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.12` | `baseline` | `0.323130` | `0.682283` | `2.470606` | `20.058457` | `13.486868` | `0.999533` | `0.078791` | `0` | `1` |
| `0.12` | `small_v1` | `0.257755` | `0.602595` | `0.837800` | `17.680351` | `22.480219` | `0.998424` | `0.129678` | `0` | `1` |
| `0.10` | `baseline` | `0.318882` | `0.669759` | `1.041937` | `17.737562` | `12.028861` | `0.999929` | `0.029791` | `0` | `1` |
| `0.10` | `small_v1` | `0.259253` | `0.590736` | `0.837800` | `16.807978` | `21.142055` | `0.998372` | `0.130444` | `0` | `1` |
| `0.08` | `baseline` | `0.291299` | `0.641019` | `2.253829` | `18.646264` | `12.234158` | `0.996988` | `0.195845` | `0` | `1` |
| `0.08` | `small_v1` | `0.249960` | `0.576746` | `0.837800` | `16.861091` | `20.981064` | `0.998211` | `0.136119` | `0` | `1` |

Focused `0.10m` loss comparison:

| Variant | Planned-vs-FK | Touchdown IK/FK | Raw IK Violation | Swing Step | Swing Accel | Direction Cos | Lateral Drift | Penetration | Foot-Over |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | `0.318882` | `0.669759` | `1.041937` | `17.737562` | `12.028861` | `0.999929` | `0.029791` | `0` | `1` |
| `small_v1` | `0.259253` | `0.590736` | `0.837800` | `16.807978` | `21.142055` | `0.998372` | `0.130444` | `0` | `1` |
| `small_v2` | `0.235963` | `0.576998` | `0.837800` | `17.664060` | `20.566445` | `0.998856` | `0.104328` | `0` | `1` |

## Result

Partial pass.

The probe now supports `--semantic-small-diameter-m`, and local tests cover the diameter/height override path. Smaller small-obstacle diameter improves IK/FK mismatch in the real IsaacLab forward probe, but does not solve swing acceleration.

`reachable_loss_small_v2` gives the best mismatch at `0.10m` (`planned-vs-FK 0.319 -> 0.236`, touchdown `0.670 -> 0.577`) while preserving no-contact and foot-over. It still leaves swing acceleration high (`12.029 -> 20.566`), so it is not accepted as a complete fix.

## Conclusion

Small obstacle footprint matters. The default `0.12m` is not impossible, but `0.10m` or `0.08m` makes the reachable-crossing problem easier. However, simply shrinking the loss window and adding feasibility weights still creates a continuity tradeoff.

The best next direction is:

- Keep `0.10m` as the current probe candidate size unless the user prefers `0.08m`.
- Combine the smaller obstacle size with the earlier approach-then-cross logic and a stronger FK-realized continuity term.
- Do not change production default yet; verify all-direction behavior first.

## Follow-Up

- Run all-direction for the best size/loss candidate only after adding a continuity-safe version.
- Update T302i plan to include obstacle size as a tested variable, not a silent fixture change.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_diam012_small_v1.jsonl`
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_diam010_small_v1.jsonl`
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_diam008_small_v1.jsonl`
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_diam010_small_v2.jsonl`
