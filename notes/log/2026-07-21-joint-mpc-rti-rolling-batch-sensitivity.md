# Joint MPC RTI Rolling Batch Sensitivity Diagnostic

## Purpose

Locate the first backward-command failure transition and verify that ranked-batch behavior is representative of the same command run alone.

## Stage

Task 13 systematic root-cause diagnosis. No production or tuning parameter change.

## Procedure

- Ran the 24-step backward command `(-1,-0.5,-1)` alone at B=1 under the monitored CUDA runner.
- Ran the same command as row 2 of the ranked B=3 batch under the same monitor.
- Printed per-frame gait phase, selected alpha, trajectory validity, minimum joint margin, maximum joint step, root XY step, maximum foot XY step, and minimum flat clearance.

## Resource Result

- B=1 completed in `9.64s`, task GPU memory about `1.02GiB`.
- B=3 completed in `12.15s`, task GPU memory about `2.09GiB`.
- Both monitors completed without a resource trigger.

## Findings

- Frames 1-9 are nearly identical, with only small floating-point differences.
- The first discrete divergence is phase/frame 10: B=1 selects alpha `0.5`; B=3 selects alpha `0.25`.
- B=1 then selects alpha zero at frame 12 and reports invalid through frame 23 while its nominal leaves joint bounds.
- B=3 remains valid through frame 22, but alpha zero persists from frame 15; joint margin reaches `0` at frame 21, joint step reaches `0.394rad` at frame 22, and clearance reaches `-64.6mm`.
- The different alpha at one rolling step changes the shifted warm trajectory and creates a large later behavioral branch. Ranked B=3 behavior therefore cannot yet be assumed representative of formal B275 behavior.

## Conclusion

The immediate root-cause question is whether the frame-10 candidate losses are genuinely nearly tied or whether scan/line-search results violate batch-size parity. Inspect candidate losses, filter masks, and directions at the first divergence before tuning weights.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `solver/trajectory_scan.py`, `solver/line_search.py`, `solver/sqp_rti.py`, `runtime/warm_start.py`, `run_joint_acceptance.py`
