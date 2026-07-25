# Joint MPC Cross Common-Metric Regression

- Purpose: verify that small/cross acceptance consumes the complete flat/common metric set, including stance grounding and forbidden semantic checks.
- Stage: final perceptive-kinematic plan Task 17.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Changes

- Marked `stance_on_forbidden_semantic` and `touchdown_on_forbidden_semantic` as `P+A+M` metrics because their evidence includes current-map queries.
- Added a regression asserting that a moving small command includes every applicable `COMMON_METRICS` ID in the small gate. The three zero-drift horizon metrics remain explicitly N/A for nonzero motion commands by contract.

## Verification

```text
metric/common focused tests: 22 passed in 1.70s
perceptive_plan.py + test_nominal.py: completed without failures
git diff --check: passed
```

## Conclusion

Cross now has the same applicable flat/common gate as flat, including stance ground gap/penetration, stance slip/anchor/stationary, current-map semantic stance/touchdown, swing, joint, root, lifecycle, map freshness, KKT, publish, and validity metrics. The controlled small crossing remains open for the previously recorded preview liftoff selector/publication swept-geometry mismatch.

## Git Refs

- Baseline Ref: `work/joint-mpc-kinematic`
- Candidate Ref: uncommitted working tree
- Key Files: `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py`
