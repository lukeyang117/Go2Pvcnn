# Joint MPC RTI Forward Cold-Nominal Filter Diagnostic

## Purpose

Identify the exact position/velocity filter causing the first forward invalid frame after infeasible-fallback status propagation.

## Stage

Task 13 root-cause diagnosis before the next nominal-scale experiment.

## Procedure

Wrapped the existing line search read-only and ran the ranked B=3 trace for 16 steps under the monitored CUDA runner. For forward row 1, printed selected feasibility, all five position/velocity masks, nominal minimum joint margin and node/joint index, and nominal maximum joint step and edge/joint index.

## Result

- Monitor completed in `9.53s`; task GPU memory about `2.09GiB`; no trigger.
- Forward validity is false for planner frames 1-12 and recovers at frame 13.
- At frame 10, all five candidates fail both position and velocity filters.
- The alpha-zero nominal minimum margin is `-0.837rad` at node 1, joint flat index 5 (leg-1 calf), and its maximum adjacent joint step is `1.422rad` on the same joint.
- The same infeasible cold trajectory shifts toward x0 through frames 10-12. A fresh cold build at frame 13 produces a feasible alpha-1 candidate and restores validity.

## Conclusion

Forward invalidity begins with the cold nominal, not only the phase-matched warm terminal. Smooth tuning cannot repair a first-step cold point outside the physical/velocity filters. Re-test the approved `command_scale=0.8` with corrected solver status and universal validity metric.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `model/nominal.py`, `solver/trajectory_qp.py`, `solver/line_search.py`, `planner.py`
