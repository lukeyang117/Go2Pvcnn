# Joint MPC RTI Universal Trajectory Validity Metric

## Purpose

Prevent invalid planner frames from being silently excluded from every other behavior metric and thereby masking a failed flat/small cell.

## Stage

Task 12/13 shared acceptance contract correction discovered after infeasible-fallback status propagation.

## Root Cause

Each `MetricResult` recorded a `valid_count`, but no applicable metric required that count to equal the full trace length. Joint, line-search, stance, and tracking metrics masked invalid nodes. The post-status-fix default trace had only `13/25` valid forward nodes and `15/25` valid backward nodes without an explicit validity failure.

## TDD Evidence

- RED: `1 failed, 5 passed`; `trajectory_valid_ratio` was absent.
- Added `trajectory_valid_ratio = valid.float().mean()` to the shared flat metric set, inherited by small.
- Added the frozen threshold `>=1.0`.
- GREEN: joint metrics plus flat aggregation `10 passed in 3.75s`.

## Result

Every flat and small cell now fails explicitly if any node is invalid. No optimizer, loss, line-search candidate/filter, threshold for an existing metric, or detector was changed.

## Follow-up

Rerun the default ranked CUDA report so the new validity metric is present, then continue only with candidates that keep `trajectory_valid_ratio=1.0`.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `joint_metrics.py`, `acceptance_thresholds.py`, `test_joint_metrics.py`
