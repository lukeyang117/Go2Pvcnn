# 2026-05-28 21:06 T302k Plane Low-Small FK Collision Probe

## Purpose

Add the test-only plane low-small FK semantic collision probe required by Task 9. This does not add an optimizer loss or a hard trajectory repair.

## Stage

`extension/batch_mpc_planner` IsaacLab diagnostic probe and backend tests.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'plane_low_small_metrics'
```

Initial failure:

```text
ImportError: cannot import name 'compute_plane_low_small_fk_metrics'
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'plane_low_small_metrics'
python -m py_compile Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_batch_mpc_backend.py
```

IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --requested-n-frames 300 \
  --cycles 1 \
  --commands 'forward:0.50 0.00 0.00,left:0.00 0.50 0.00,turn_left:0.00 0.00 1.00' \
  > tmp/t302k-low-small-redesign/plane_fk_collision_smoke.jsonl 2>&1
```

## Input Conditions

- Baseline ref: `eed5d18`.
- Environment: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.
- GPU selection: `CUDA_VISIBLE_DEVICES=0`.
- Commands: `forward`, `left`, `turn_left`.
- Cycles: `1`.
- Requested frames: `300`.
- Probe output: `tmp/t302k-low-small-redesign/plane_fk_collision_smoke.jsonl`.

## Key Metrics

- Focused helper test: `1 passed, 118 deselected`.
- Pycompile: pass.
- IsaacLab smoke exit code: `0`.
- JSONL cycle rows: `3`.
- Commands recorded: `forward`, `left`, `turn_left`.
- `cuda_visible_devices`: `0`.
- `plane_env_count`: `[1, 1, 1]`.
- `terrain_is_plane`: `[1, 1, 1]`.
- Required metadata keys present: yes.
- Required FK semantic metric keys present: yes.
- `crossing_leg_count`: `[0, 0, 0]`.
- `fk_semantic_collision_count`: `[0, 0, 0]`.
- `fk_semantic_collision_rate`: `[0.0, 0.0, 0.0]`.
- `planned_vs_fk_foot_error_crossing_leg_max_m`: `[0.0, 0.0, 0.0]`.

## Result

Pass for metric computation, JSONL metadata, and IsaacLab smoke execution.

## Conclusion

The probe now reports plane-only low-small diagnostic coverage and FK semantic collision metrics from actual planned target feet and IK/FK realized foot/knee/shank points. The smoke proves the logging path works on plane terrain, but it did not trigger a crossing leg, so it is not full behavioral acceptance.

## Follow-Up

Continue with Task 10 full local verification and the full IsaacLab command matrix. Rows with `crossing_leg_count == 0` must be marked not-covered rather than pass.

## Git Refs

- Baseline Ref: `eed5d18`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
