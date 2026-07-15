# 2026-07-06 MPC QP Root Z Small Obstacle Repro

## Purpose

Reproduce the viewer report that around crossing a small semantic obstacle, `mpc_qp` root height can stay high enough that played feet do not touch the terrain.

## Stage

MPC-QP backend / continuous trajectory / viewer playback readback.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Commands

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 2
```

Then a temporary inline diagnostic using `RealViewerRuntimeFixture` with the same forward command, `qp_iterations=2`, and 50 frames, reading planned root/foot, terrain height, playback actual feet, and playback actual root.

## Metrics

Existing crossing probe:

- `viewer_crossing_acceptance_passed=false`
- `crossing_opportunity_count=0`
- `playback_readback_error_max_m≈0.23475`
- `planned_vs_fk_foot_error_all_max_m≈0.28224`
- planned feet remain terrain-bound: planned clearance min `0`, planned penetration `0`

Temporary root/foot time-series diagnostic:

- planned root height offset starts at `0.42m` and remains `0.42m` through frame 30.
- frame 0 actual foot clearance min `≈0.04964m`, planned clearance min `0`.
- frame 10 actual foot clearance min `≈0.04323m`, planned clearance min `0`.
- frame 20 actual foot clearance min `≈0.03549m`, planned clearance min `0`.
- frame 25 actual foot clearance min `≈0.00921m`, planned clearance min `0`.
- frame 40 root is over semantic small obstacle: root ground `≈0.14250m`, root z `≈0.27750m`, root offset `≈0.13500m`, actual foot clearance min `≈0`.
- frame 49 root is still only slightly beyond obstacle center (`root_along≈0.020m`), root offset `≈0.13701m`, actual foot clearance min `≈0`.

Follow-up 100-frame single-cycle probe:

- `viewer_crossing_acceptance_passed=false`
- `along_progress_m≈0.36m`
- `playback_readback_error_max_m≈0.23321`
- `planned_vs_fk_foot_error_all_max_m≈0.28037`
- `crossing_opportunity_count=0`

Follow-up 3-cycle rolling probe:

- cycle 0: `playback_readback_error_max_m≈0.23475`, root `along_end≈0.010m`, root offset start `0.42m`, mid `0.42m`, end `≈0.1318m`.
- cycle 1: `playback_readback_error_max_m≈0.18604`, root `along_start≈0.010m`, `along_end≈0.260m`, root offset start `0.26m`, after-obstacle mean `≈0.28335m`, max `≈0.40409m`.
- cycle 2: `playback_readback_error_max_m≈1.9e-6`, root `along_start≈0.260m`, `along_end≈0.510m`, root offset stays `0.26m`.

## Conclusion

The failure is reproduced, but the evidence is more specific than "touchdown z is wrong": planned feet are terrain-bound, while playback/FK feet float above the terrain during the approach/early crossing as root height offset stays at `0.42m`. Once the sampled root reaches the small obstacle top, root z drops relative to terrain and actual feet contact again.

The rolling probe shows the high root behavior is not a permanent tail: by cycle 2, after the root is well beyond the obstacle, offset stabilizes at `0.26m` and playback readback is nearly zero. The problematic window is the approach/crossing transition and the immediately following cycle, where root offset and IK/FK readback remain too high for grounded-looking feet.

## Follow-Up

- Add a stable root-height/foot-grounding diagnostic to the crossing probe.
- Fix should be in `mpc_qp` root-height/readback/grounding residuals, not touchdown snapping or hard repair.
- Add a test target for the transition window: cycle 0/1 readback and foot clearance should be bounded while preserving terrain-bound planned feet.

## Git Refs

- Baseline Ref: dirty workspace after prior MPC-QP continuous work.
- Candidate Ref: no code changes for this reproduction.
- Key Files:
  - `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`
