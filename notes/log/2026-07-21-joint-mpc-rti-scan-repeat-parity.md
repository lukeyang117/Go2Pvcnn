# Joint MPC RTI Exact-Repeat Scan Parity

## Purpose

Isolate associative-scan batch behavior from rolling active-set sensitivity at the backward frame-10 QP.

## Stage

Task 13 solver diagnosis before parameter tuning.

## Procedure

Captured the exact B=1 frame-10 `TrajectoryQp`, repeated every QP tensor identically to B=3, and solved the repeated batch with the existing H30/32 scan under the monitored CUDA runner.

## Result

- Monitor completed in `6.57s`; task GPU memory about `1.02GiB`; no trigger.
- Maximum difference among the three repeated rows: `0.0`.
- Maximum difference from the original B=1 solve: `2.68e-7`.
- Original and repeated maximum direction magnitude: `0.78125465` for every row.

## Conclusion

The associative scan preserves the exact-repeat batch contract for this failing QP. The rolling B1/B3 divergence is caused by the trajectory reaching a joint-bound active-set boundary and then amplifying small prior state differences, not by direct environment mixing in scan. Proceed with the approved Task 13 order: first reduce joint trust, then consider posture weight.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `solver/trajectory_qp.py`, `solver/trajectory_scan.py`
