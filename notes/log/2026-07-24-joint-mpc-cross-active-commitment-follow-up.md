# Joint MPC Cross Active Commitment Follow-up

## Purpose

Continue Task 17 after the selector/nominal published-sweep mismatch. The
controlled small cuboid trace must cross the footprint without collision or
publication stop while retaining all flat/common metrics.

## Changes

- Added a nominal regression for an active crossing candidate rejected by the
  selector sweep but safe under exact nominal checks.
- Kept active primary crossing candidates eligible through structural map,
  plane, and region checks; exact nominal safety remains authoritative.
- Unified the primary crossing offset calculation with the delayed swing
  profile and added fixed-shape preview/nominal retry paths.
- Added primary crossing mutual exclusion so two legs do not enter the same
  small footprint in one refresh.

## Verification

- `test_nominal.py`: `43 passed` before the final mutual-exclusion edits; the
  focused commitment/retry regression remains green.
- Perceptive crossing offset tests: `2 passed`.
- Controlled 144-refresh CUDA trace: still failed `strict_cross_success`.
- Latest direct trace: strict crossing `0`; the run later degraded to about
  `56.6%` valid and `43.4%` stop after the speculative retry path. This is not
  acceptance evidence.

## Diagnosis

The stable portion of the trace reaches a single crossing leg, but a later
published joint-linear swept edge still fails the foot safety gate or the
crossing commitment expires before the actual touchdown-after-obstacle event.
The current worktree therefore does not close Task 17.

## Follow-up

Revert or simplify the speculative multi-attempt nominal changes before the
next controlled run, then trace the first single-leg crossing event through
actual foot footprint, touchdown-after, and direction gates.

## Git Traceability

- Baseline Ref: `work/joint-mpc-kinematic`
- Candidate Ref: current worktree
- Current Work Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`,
  `Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py`,
  `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`
