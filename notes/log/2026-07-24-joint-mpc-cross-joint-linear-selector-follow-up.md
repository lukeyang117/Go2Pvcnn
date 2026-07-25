# Joint MPC Cross Joint-Linear Selector Follow-up

## Purpose

Verify the complete small-cross common gate and investigate the preview liftoff swept mismatch.

## Result

- The direct numeric regression passes: a strict-cross-success trace with one stance foot `20 mm` above current ground and another `2 mm` below fails both `stance_ground_gap` and `stance_ground_penetration`, and the complete small report fails.
- Fixed joint-linear selector components for foot, knee, calf, and thigh were added. Selector/nominal focused verification is `84 passed`; B512 and B1024 selector shape gates pass after capsule-slice streaming.
- Controlled `vx=0.2` cuboid crossing remains red. Actual part collision counts are zero, but publication first becomes invalid at refresh 59. The complete common metrics remain active.
- A preview warm-joint/tail blend experiment caused early publication collapse (`valid_ratio=0.0879`) and was reverted. The stable baseline is restored; the remaining issue is nominal/preview trajectory ownership, not acceptance-threshold tuning.

## Conclusion

Small crossing correctly evaluates `M_common union M_small`, including current-map stance grounding, gap, penetration, and forbidden semantic checks. Task 17 is not complete because strict crossing remains red.

## Git Refs

- Baseline Ref: `5250bf1`
- Candidate Ref: uncommitted `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py`, `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`, `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py`
