# 2026-07-07 MPC QP Flat Speed Continuity A/B

## Purpose

Reproduce the user's report that, on flat terrain with the same nonzero velocity command, `mpc_qp` has larger velocity distortion and foot/root/joint continuity distortion than the existing `mpc` backend.

## Stage

MPC-QP planner output diagnostics. This pass does not modify planner code.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Command / Procedure

First attempted a real `RealViewerRuntimeFixture` A/B on visible card 3:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python <inline fixture probe>
```

The process printed Isaac/Kit startup logs and GPU bad-state warnings, then exited before the fixture reached `after_fixture`. No Python exception was emitted. Because this did not produce planner metrics, the reproducible evidence below uses a pure planner A/B.

Pure planner A/B:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python <inline pure planner probe>
```

The probe used:

- terrain: synthetic flat `MpcPlannerTerrain`, 151 x 151 zero height map, zero semantic map
- initial state: root z `0.32m`, nominal four foot offsets
- command: `vx=0.45, vy=0.0, yaw=0.0`
- horizon: `25`
- dt: `0.02`
- cycles: `6` rolling replans, where each next state is the previous result final frame
- `mpc`: `MpcPlannerCfg`, `optimize_steps=24`
- `mpc_qp`: `MpcQpPlannerCfg`, `optimize_steps=24`, `qp_iterations=3`

## Metrics

Main metrics:

- root velocity distortion: mean forward-speed error, speed mean/std
- within-horizon continuity: max root step, root acceleration, foot step/acceleration, joint step/acceleration
- replan continuity: first-frame boundary delta and full-horizon shape delta between adjacent replans

Selected summary:

| Metric | `mpc` | `mpc_qp` | Ratio |
| --- | ---: | ---: | ---: |
| root forward abs error mean | `0.09225 m/s` | `0.95976 m/s` | `10.40x` |
| root speed mean | `0.50090 m/s` | `1.71639 m/s` | `3.43x` |
| root speed std | `0.10788 m/s` | `1.07956 m/s` | `10.01x` |
| root step max mean | `0.01450 m` | `0.08462 m` | `5.83x` |
| root acceleration max mean | `2.814 m/s^2` | `356.317 m/s^2` | `126.61x` |
| foot step max mean | `0.06128 m` | `0.16386 m` | `2.67x` |
| foot acceleration max mean | `184.19 m/s^2` | `783.67 m/s^2` | `4.25x` |
| joint step max mean | `0.73381 rad` | `1.37529 rad` | `1.87x` |
| full-horizon root delta max mean | `0.00658 m` | `0.07443 m` | `11.31x` |
| full-horizon foot delta max mean | `0.07785 m` | `0.21482 m` | `2.76x` |
| full-horizon joint delta max mean | `0.51380 rad` | `1.13596 rad` | `2.21x` |

Boundary result:

- `mpc` boundary root/foot/joint deltas were `0`.
- `mpc_qp` boundary foot delta was near zero after cycle 1, but boundary root and joint deltas remained nonzero in rolling replans: root max `0.02683m`, joint max `0.65855rad`.

## Result

Reproduced. On flat terrain with identical nonzero velocity, current `mpc_qp` produces much larger speed distortion and continuity distortion than `mpc`.

The strongest quantitative signal is not only foot displacement. It is the coupled root trajectory:

- mean root forward-speed error is about `10.4x` larger
- root speed variance is about `10.0x` larger
- root acceleration is about `126.6x` larger
- full-horizon root replan shape delta is about `11.3x` larger

Foot and joint metrics are also worse:

- foot max step mean is about `2.7x` larger
- foot acceleration max mean is about `4.3x` larger
- joint step max mean is about `1.9x` larger

## Conclusion

The user's visual complaint is valid at the planner-output level. This is not just a viewer/runtime artifact.

The likely next investigation should focus on isolated `mpc_qp` continuous coupled loss/update terms that can create root lateral/XY oscillation or high-frequency root trajectory changes on flat terrain, especially the continuous FK readback/root XY/root Z updates and full-horizon replan consistency. Do not change `mpc`, and do not add candidate endpoint/search/repair behavior.

## Follow-Up

Add an open T302v child for nonzero flat speed continuity:

- reproduce in a reusable probe/test file rather than inline script
- inspect which `continuous_qp_update` diagnostics fire on flat terrain
- tune existing design-approved coupled losses or `qp_iterations`; do not introduce design-outside losses before user approval

## Git Refs

- Baseline Ref: `8168b15`
- Candidate Ref: working tree, no code changes in this pass
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
