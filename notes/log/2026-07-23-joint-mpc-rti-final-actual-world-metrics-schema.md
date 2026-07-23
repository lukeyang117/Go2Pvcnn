# Final Actual-World Metrics Schema

## Purpose

Complete final-plan Task 14 by replacing legacy metric names/applicability with the frozen current-map P/A/M schema and a strict foot-event crossing detector.

## Stage And Refs

- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Baseline Ref: `6c4df17`
- Candidate Ref: current Task 14 checkpoint
- Branch: `work/joint-mpc-kinematic`

## Implementation

- Defined 72 final metric IDs, including split zero drift durations, alpha/no-feasible/publish/stop lifecycle, map freshness/transform, target-change reasons, KKT, nominal safety, and report-only acceleration/jerk diagnostics.
- Added explicit `P`, `A`, and `M` source metadata and preserved it through JSON/shard round trips.
- Small reports inherit every common metric. Zero translation removes only direction/cross terms; stance, swing, lead, joint, collision, numerical, and lifecycle evidence remains applicable.
- Planned and actual collision masks are combined by logical OR per body part, so an actual collision fails even when planned geometry is safe.
- Replaced the old root-band crossing proxy with an optional strict foot-event path using lift/touchdown edges, continuous segment-to-footprint intersection, sole vertical margin, actual-root direction, after-obstacle landing, normal-ground landing mask, and whole-body collision window.
- A formal opportunity with insufficient actual root progress remains a failed crossing rather than disappearing from the denominator.

## Verification

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_run_joint_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_process_watchdog.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py -q
```

Result: `57 passed in 11.97s`.

## Conclusion

The metric/report and strict-cross contracts pass focused tests. This is not flat/small behavior acceptance: formal traces and canonical actual viewer traces remain Tasks 16-17.

## Key Files

- `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`
- `Go2Pvcnn/tests/joint_mpc_rti/acceptance_thresholds.py`
- `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`
- `Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py`
- `Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py`
