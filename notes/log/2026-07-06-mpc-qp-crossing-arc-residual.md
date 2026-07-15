# 2026-07-06 MPC QP Crossing Arc Residual

## Purpose

Translate the old MPC low-small loss idea into the continuous `mpc_qp` path without endpoint candidates or touchdown repair: when a low-small obstacle sits in the command corridor, optimize the swing Bezier arc itself (`P1/P2`) so the trajectory can form an obstacle-crossing arc while `P3` touchdown stays terrain-bound.

## Stage

MPC-QP backend / continuous Bezier trajectory / low-small crossing loss.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 2
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py --device cuda:0 --tiles 8:12 --commands 'forward:0.35,0.0,0.0;diag_left:0.30,0.12,0.0' --cycles 1 --requested-n-frames 25 --playback-frames 25 --qp-iterations 2 --warmup-steps 4
```

## Changes

- Added a focused RED/GREEN unit test requiring the continuous QP path to create a low-small crossing leg from trajectory loss, not from repair.
- Added fixed-shape low-small crossing-arc residuals in `solver.py`:
  - near-lane crossing legs adjust only `P1/P2.xy`;
  - crossing arc clearance adjusts only `P1/P2.z`;
  - touchdown `P3.xy` is not selected by candidates, and `P3.z` remains height-field bound.
- Added runtime tuning fields for the crossing-arc lane margin, lateral step, vertical margin, and vertical step.

## Metrics

Static:

- Focused RED failed on `fk_foot_over_low_small_success == 0`.
- Focused GREEN passed.
- Full focused QP suite: `59 passed`.
- Pycompile: pass.

Required hard terrain `row=8`, `col=12`, `qp_iterations=2`:

- `viewer_hard_terrain_acceptance_passed=true`.
- `max_playback_readback_error_m≈0.02118`.
- `max_qp_continuous_planned_foot_terrain_penetration_count=0`.
- `max_fk_semantic_collision_count=0`.
- `max_qp_touchdown_on_small_count=0`.
- `max_qp_continuous_foot_frame_jump_m≈0.04663`.
- `max_qp_continuous_joint_frame_jump_rad≈0.20897`.

Default flat-small viewer probe:

- `viewer_crossing_acceptance_passed=false`.
- `crossing_opportunity_count=0`.
- `max_playback_readback_error_m≈0.31284`.
- `max_qp_continuous_fk_readback_error_m≈0.31284`.
- FK semantic collision, stance-on-small, touchdown-on-small, and planned penetration remain `0`.

## Conclusion

The new crossing-arc residual is safe on the required hard terrain and passes synthetic trajectory-loss coverage, but it does not yet satisfy the real default flat-small strict crossing probe. The remaining blocker is old-design loss item 4: optimized target foot trajectory and IK/FK readback still diverge badly in that viewer setup. The next change should strengthen or redesign the FK/readback consistency residual and crossing-leg diagnostics, not add hard constraints, endpoint candidates, or touchdown repair.

## Git Refs

- Baseline Ref: dirty workspace after low-small progress/reach-gate work.
- Candidate Ref: current dirty workspace after crossing-arc residual.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
