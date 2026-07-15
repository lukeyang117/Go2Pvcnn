# 2026-07-06 MPC QP Flat-Small Body-Leg Readback

## Purpose

Focus `mpc_qp` on flat-ground low-small obstacle crossing quality after visual feedback that the trajectory still looks bad. Row `8`, col `12` hard terrain is intentionally not the tuning driver in this pass.

## Stage

MPC-QP backend / continuous trajectory solver / flat-small crossing.

## Related Todo

- [T302v Task 19](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md#task-19-shared-coupled-loss-for-rootfoot-readback)

## Candidate Ref

- Workspace changes on 2026-07-06.
- Key files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/constraints.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
  - `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`

## Changes

- Added continuous body-leg lateral coupling inside the existing body/knee/shank clearance pass:
  - collision residuals can move swing `P1/P2` laterally
  - `P3/touchdown` remains terrain-bound and is not candidate-searched
  - root-z lift still handles body/underbody clearance
- Added fixed underbody points to the continuous body-leg pass and capped root-z FK readback lowering by underbody clearance.
- Tuned `body_leg_xy_repair_step_m` from `0.03` to `0.04`.
- Adjusted body-leg semantic collision diagnostics to use a `1e-5m` numerical penetration tolerance.
- Added viewer probe diagnostic keys for reachability, FK readback, body-leg clearance, and joint-limit readback.

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_low_small_high_arc_remains_fk_reachable -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 3
```

## Result

- Focused high-arc synthetic regression: passed.
  - foot-over success `1`
  - FK/planned readback `<=0.05m`
  - FK body-leg collision `0`
  - low-small swing-height gate `<=0.18m`
- Full QP suite: `61 passed`.
- Pycompile: passed.
- Real flat-small probe still fails strict acceptance:
  - `viewer_crossing_acceptance_passed=false`
  - `crossing_opportunity_count=0`
  - FK foot-over success can be `1` in the safe body-leg coupling run
  - FK foot/knee/shank semantic collision `0`
  - touchdown/stance on small `0`
  - planned/FK terrain penetration `0` in the safe body-leg coupling run
  - FK/planned readback remains high at about `0.182m`
  - body/underbody diagnostic collision remains nonzero (`qp_fk_body_leg_collision_count≈5`)

## Investigation Notes

- Synthetic reproduction with root roll `≈0.167rad` reproduces the readback/body issue: the same flat-small scene that passes with level root gets FK/readback `≈0.126m` and body/terrain issues when roll is added.
- Directly zeroing root roll/pitch is not acceptable: it breaks FK/foot-over and can introduce terrain penetration because foot controls were optimized against the original root attitude.
- Globally enabling joint-limit readback is not acceptable: it triggers on raw IK limit violation but collapses foot-over and introduces penetration.

## Conclusion

This pass improves the body-leg part of the flat-small problem but does not complete acceptance. The remaining blocker is true coupled optimization of root attitude/root position and Bezier foot controls. Root `rpy` cannot be repaired independently; it must participate in the same IK/FK consistency, reachability, and body-leg clearance residuals as the foot controls.

