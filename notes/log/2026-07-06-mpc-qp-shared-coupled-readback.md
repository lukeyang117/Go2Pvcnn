# 2026-07-06 MPC QP Shared Coupled Readback

## Purpose

Record the first shared coupled readback update for `mpc_qp`: FK/readback can now update root z together with Bezier foot controls in low-risk flat scenes, instead of only moving `P1/P2/P3`.

## Stage

MPC-QP backend / continuous trajectory solver / root-foot coupled loss.

## Related Todo

- [T302v Task 19](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md#task-19-shared-coupled-loss-for-rootfoot-readback)

## Candidate Ref

- Workspace changes on 2026-07-06 21:34 CST.
- Key files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
  - `docs/superpowers/specs/2026-07-06-mpc-qp-continuous-bezier-trajectory-design.html`
  - `notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md`

## Changes

- Design now documents shared coupled variables, coupled-vs-local loss groups, and acceptance metrics beyond semantic pass rate.
- Todo now makes shared root/foot readback Task 19 and drops stale open items that implied foot-only fixes or repair/candidate directions.
- Added config fields:
  - `continuous_fk_root_z_gain`
  - `continuous_fk_root_z_max_step_m`
  - `continuous_fk_root_z_error_threshold_m`
- Added root-z readback update in `solver.py`:
  - computes FK/planned foot z residual from fixed samples
  - lowers root z only when feet are clearly floating above terrain-bound planned feet
  - gates the lowering off when semantic cells or high terrain variation are present
  - keeps touchdown z terrain-bound
  - exposes `qp_continuous_solver_fk_root_z_update_count`, `qp_continuous_solver_fk_root_z_delta_max`, and `qp_continuous_solver_fk_root_readback_error_before_max`
- Added RED/GREEN unit coverage for root-z readback lowering without activating repair.

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_fk_readback_lowers_root_when_feet_float_without_repair -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py
git diff --check -- docs/superpowers/specs/2026-07-06-mpc-qp-continuous-bezier-trajectory-design.html notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md Go2Pvcnn/extension/batch_mpc_qp_planner/config.py Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py Go2Pvcnn/tests/test_mpc_qp_backend.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 2
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py --device cuda:0 --tiles 8:12 --commands 'forward:0.35,0.0,0.0;diag_left:0.30,0.12,0.0' --cycles 1 --requested-n-frames 25 --playback-frames 25 --qp-iterations 2 --warmup-steps 4
```

## Result

- Focused RED first failed on missing `qp_continuous_solver_fk_root_z_update_count`; GREEN passed.
- Full QP suite: `61 passed`.
- Pycompile: passed.
- `git diff --check`: passed.
- Flat-small crossing probe: process exit `0`, but strict viewer crossing still fails.
  - `viewer_crossing_acceptance_passed=false`
  - `crossing_opportunity_count=0`
  - `max_fk_semantic_collision_count=0`
  - `max_fk_touchdown_on_small_rate=0`
  - `max_qp_continuous_planned_foot_terrain_penetration_count=0`
  - `max_qp_continuous_fk_readback_error_m≈0.23475`
  - `max_qp_continuous_foot_frame_jump_m≈0.12811`
  - `max_qp_continuous_joint_frame_jump_rad≈0.66301`
- Required hard terrain row `8`, col `12`: passed.
  - `viewer_hard_terrain_acceptance_passed=true`
  - `max_fk_semantic_collision_count=0`
  - `max_qp_touchdown_on_small_count=0`
  - `max_playback_readback_error_m≈0.03522`
  - `max_qp_continuous_planned_foot_terrain_penetration_count=0`
  - `min_qp_continuous_fk_foot_terrain_clearance_m≈-0.00219`
  - `max_qp_continuous_foot_frame_jump_m≈0.04478`
  - `max_qp_continuous_joint_frame_jump_rad≈0.17770`

## Conclusion

The shared root-z readback scaffold is implemented and covered without activating the repair main path. It does not regress required row `8`, col `12` hard-terrain acceptance. It does not yet fix the default flat-small strict crossing/readback problem; that remains open for the next coupled loss pass.

## Follow-Up

- Extend coupled residuals beyond low-risk root-z lowering so flat-small crossing can improve FK/planned overlap without forcing unsafe swing height or endpoint candidates.
- Add full Bezier semantic/object keepout and FK-aware clearance terms as loss updates, not hard repair.
