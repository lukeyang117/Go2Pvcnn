# Joint MPC RTI Kinematic Task 10 Line Search

## Purpose

Replace the historical control/merit search with the approved five-candidate direct-state loss-only rule.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 10 of 16

## TDD Evidence

RED: the old module lacked `FILTER_NAMES`, fixed state candidates, joint-limit filters, and the new result contract.

GREEN:

- Focused line-search suite: `7 passed`.
- Contract + line search + Task 9 CPU regression: `21 passed, 1 CUDA deselected`.
- `py_compile` and `git diff --check`: exit `0` before final note updates.

## Contract

- Candidate alphas are exactly `(1.0, 0.5, 0.25, 0.125, 0.0)`.
- Candidate tensor is exactly `[B,5,31,18]` and is evaluated in one objective call.
- Filters are exactly `finite`, `joint_position`, and `joint_velocity`.
- Selection uses only candidate objective loss; equal loss within `1e-7` chooses the larger alpha.
- `alpha=0` is the nominal fallback.
- No collision hard gate, terrain hard gate, constraint ranking, recovery candidate, improvement-over-base rule, or output projection remains in this module.
- Two historical improving-over-base tests were removed; old SQP/planner migration remains Task 11 work.

## Git Refs

- Baseline Ref: `0e04839`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py`

## Follow-Up

Route exactly one direct-Z linearization, active scan solve, and five-candidate line search through the new RTI planner, then delete old production control/recovery/projection logic.
