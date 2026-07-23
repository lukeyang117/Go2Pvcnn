# Joint MPC RTI Final Flat Gate Closure

## Purpose

Close final-plan Task 16 with real same-refresh KKT diagnostics, the complete 19-cell flat command matrix, the focused owner suite, and the full `joint_mpc_rti` package.

## Stage And Todo

- Stage: final perceptive-kinematic Task 16 flat gate
- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan: [final implementation plan](../../docs/superpowers/plans/2026-07-23-joint-mpc-rti-perceptive-kinematic-implementation-plan.md)

## Git Refs

- Baseline Ref: `aaca7cc`
- Candidate Ref: `work/joint-mpc-kinematic` dirty Task 16 candidate
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/{config.py,planner.py,model/nominal.py,model/perceptive_plan.py,solver/trajectory_scan.py}` and `Go2Pvcnn/tests/joint_mpc_rti/`

## Corrections Since First Diagnosis

- The H30 auxiliary preview holds the terminal root pose; selector, IK, preview nominal, and current-map safety now use the same clamped preview root.
- Warm root uses the approved near/far profile: first 12 edges at `0.85` command scale, final 18 edges held.
- Startup root hold uses the monotonic lifecycle phase and no longer repeats every half gait.
- Touchdown candidates require support-joint safety over the complete following stance interval.
- Nominal and line search use the same joint margin, and LQ context preserves the measured current stance anchor z.
- Flat acceptance now records the real `planner.step()` KKT tensors for every refresh instead of converting missing optional metrics to zero.
- Dual KKT uses the design's scaled stationarity residual:

  \[
  r_d=\frac{\|H_{aug}\Delta Z+g_{aug}\|_\infty}
  {1+\|H_{aug}\Delta Z\|_\infty+\|g_{aug}\|_\infty}.
  \]

## Verification

Focused owner union:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_lq_problem.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_tensor_constants.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_flat_acceptance.py -q
```

Result: `115 passed in 129.87s`.

Full package:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti -q
```

Result: `284 passed in 160.31s`.

Formal CUDA flat v13 used the monitored runner with the child command after `--`. Result: `19/19` cells passed, `gate.passed=true`, elapsed `54.746s`, peak process-tree RSS about `1607.1MiB`, and peak observed GPU memory `1842MiB`.

Worst applicable metrics across all 19 cells:

| Metric | Worst | Gate |
| --- | ---: | ---: |
| KKT primal residual | `5.133e-5` | `<=1e-4` |
| KKT dual residual | `9.014e-5` | `<=1e-4` |
| root velocity error | `0.17719m/s` | `<=0.2m/s` |
| root yaw-rate error | `0.01308rad/s` | `<=0.2rad/s` |
| joint safe margin | `0.19032rad` | `>=0.1rad` |
| joint step | `0.33438rad` | `<=0.35rad` |
| stance XY slip | `2.668e-5m` | `<=5e-4m` |
| stance anchor residual | `5.141e-5m` | `<=5e-4m` |
| publish ratio | `1.0` | `1.0` |
| alpha-zero ratio | `0.0` | `<=0.05` |

## Conclusion

Task 16 flat is closed on the final H30 pure-kinematic candidate. Small-obstacle, large-obstacle/viewer, and `1024 x 1000 <5s` performance remain unverified and must proceed in Tasks 17-19 without weakening the flat gates.
