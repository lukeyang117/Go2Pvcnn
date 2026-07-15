# 2026-07-06 MPC QP FK Readback And Root Support

## Purpose

Investigate the viewer report that `mpc_qp` planned touchdowns are on terrain but actual FK feet do not coincide, and that planned/played swing trajectories are too high or visually detached.

## Stage

MPC-QP backend / continuous Bezier trajectory / IK-FK readback consistency.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q -k high_arc
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 2
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 4
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py --device cuda:0 --tiles 8:12 --commands 'forward:0.35,0.0,0.0;diag_left:0.30,0.12,0.0' --cycles 1 --requested-n-frames 25 --playback-frames 25 --qp-iterations 2 --warmup-steps 4
```

## Changes

- Added a regression test for a low-small high-arc case where touchdown is safe but FK/readback mismatch was `~0.212m`.
- Added fixed-shape low-small body support height handling: when root/body support samples are over nearby low-small semantic cells, root height uses the local ground baseline instead of the small-obstacle top.
- Updated terrain clearance projection to use the same fixed diagonal gait segment basis as `sample_controls_with_optional_gait()`, so early/late swing clearance residuals project onto the correct Bezier segment.
- Added an experimental joint-limit readback residual gated to semantic low-small target samples. It is present in code but did not affect the default flat-small failure because the planned target never entered the semantic window.
- Added `continuous_low_small_crossing_arc_target_lane_m` config and kept default at `0.05m`; reducing to `0.03m` made the high-arc reachability regression fail, so it was reverted.

## Metrics

Static:

- New RED failed at `qp_continuous_fk_readback_error_max≈0.21235m`.
- GREEN focused high-arc test passes after root support + gait-basis changes.
- Focused local regression: `4 passed`.
- Full QP suite: `60 passed`.
- Pycompile: pass.

Required hard terrain `row=8`, `col=12`, `qp_iterations=2`:

- `viewer_hard_terrain_acceptance_passed=true`.
- max playback readback `≈0.03522m`.
- planned terrain penetration `0`.
- FK semantic collision `0`.
- touchdown-on-small `0`.
- foot jump `≈0.04478m`.
- joint jump `≈0.17770rad`.
- one FK terrain contact remains `≈-0.00219m`, within the existing hard-probe tolerance.

Default flat-small strict crossing, `qp_iterations=2`:

- `viewer_crossing_acceptance_passed=false`.
- `crossing_opportunity_count=0`.
- max playback/readback `≈0.23475m`.
- FK semantic collision, stance-on-small, touchdown-on-small, planned penetration, and FK penetration remain `0`.

Default flat-small strict crossing, `qp_iterations=4`:

- `crossing_opportunity_count=1`, but quality worsens.
- max playback/readback `≈0.50064m`.
- planned penetration count `6`, FK penetration count `3`, joint jump `≈2.09rad`.
- Conclusion: more QP iterations alone is not a valid fix.

## Conclusion

Touchdown z is terrain-bound and is not the remaining cause. The actual mismatch comes from trajectory/readback consistency: root support height, stance anchors, crossing-arc lateral forcing, and IK joint limits are not yet coupled strongly enough. The low-small root support and segmented clearance changes fix a real high-arc/readback regression and keep the required hard tile accepted, but the default flat-small strict crossing still needs a dedicated reachable over-cell trajectory loss. Increasing QP iterations can create an opportunity but destabilizes clearance/readback, so the next step should tune or add a coupled loss rather than adding hard constraints or relying on more iterations.

## Git Refs

- Baseline Ref: dirty workspace after crossing-arc residual.
- Candidate Ref: current dirty workspace after root-support/readback investigation.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
