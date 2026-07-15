# MPC QP Hard Terrain Probe

## Purpose

Verify the continuous `mpc_qp` path on a real high-height-variation semantic terrain tile, and make the hard-terrain probe use a correct root/scanner positioning contract.

## Stage

MPC-QP backend / continuous trajectory hard-terrain viewer probe.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py --device cuda:0 --tiles 9:19 --commands 'forward:0.35,0.0,0.0;diag_left:0.30,0.12,0.0' --cycles 1 --requested-n-frames 25 --playback-frames 25 --qp-iterations 1
```

## Result

Pass after fixing the probe positioning contract.

Key evidence:

- selected tile: `row=9`, `col=19`
- scanner/tile XY error: `0.0m`
- terrain height range in scanner window: `0.439539m`
- summary: `viewer_hard_terrain_acceptance_passed=true`
- FK semantic collision: `0`
- touchdown on small: `0`
- max playback readback: `1.52587890625e-05m`
- max continuous foot frame jump: `0.025128424m`
- max continuous joint frame jump: `0.756024712rad`
- max local root-path height variation: `2.861e-06m`
- root terrain-risk progress reduction: `0.0` because the selected high-range tile did not put the planned local root path across the high edge.

## Root-Cause Note

The first hard-terrain probe attempt produced false failure metrics: readback and FK error were about `2.45m`. Root cause was the probe moving env0 XY to the selected tile while leaving root Z clamped near world `0.85`; tile `row=9,col=19` terrain was around `-2.2..-1.76m`, so the robot was floating far above the terrain. The fixture now exposes `move_env0_to_terrain_tile(..., ground_robot=True)`, which selects the tile, moves env0 to the tile origin, refreshes the scanner, grounds the robot from scanner terrain, and refreshes again.

## Verification

- `pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q` -> `40 passed`
- `python -m py_compile Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py` -> exit `0`
- real GPU1 probe above -> exit `0`

## Follow-Up

The hard-terrain probe now has a trustworthy positioning contract. Broader coverage should use `--auto-scan-top-k` or explicit stair/box tiles to find cases where the local planned path itself crosses a high edge; those cases should be tuned via continuous loss weights or `qp_iterations`, not by adding hard repair.

## Git Refs

- Baseline Ref: local working tree before this probe
- Candidate Ref: local working tree after adding hard-terrain probe and fixture grounding helper
- Key Files:
  - `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
  - `Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
