# Joint MPC RTI Kinematic Task 08 Active Bounds

## Purpose

Implement the frozen active constraint set for merged state boxes and joint-velocity edges with exactly two unrolled refinements.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 8 of 16

## TDD Evidence

RED: collection failed because `ActiveConstraints`, selection, dense active KKT, and refinement APIs did not exist.

During GREEN, dense diagnostics caught and fixed a real box-row indexing defect: cross-product advanced indexing produced `rank=2` instead of one independent row per active variable. The corrected row/column paired indexing restores exact z0 equalities.

Results:

- Full direct-QP/active suite: `6 passed in 5.44s`.
- Plan compile-budget selection: `1 passed, 22 deselected in 2.42s`.

## Contract

- Active box masks are `[B,31,18]`; active joint-difference masks are `[B,30,12]`.
- Local structure is capped at `18+12=30` rows per interval.
- First refinement fixes merged box boundaries; second keeps those boundaries and adds velocity boundaries.
- Lower and upper sides remain mutually exclusive.
- Redundant velocity rows whose two joint endpoints are already box-fixed are omitted from dense KKT assembly.
- Dense active KKT satisfies state and velocity bounds within `2e-5` and is repeatable for the same active set.
- `joint_kkt_compile_budget` rejects more than 32 rows and enforces padded rows `<=32`, combined RHS `<=51`, and `BLOCK_R<=64` before solve.
- No behavior, terrain, stance, startup, or recovery constraint was added.

## Git Refs

- Baseline Ref: `b57f366`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py`, `Go2Pvcnn/extension/joint_mpc_rti/solver/primal_dual_ilqr.py`

## Follow-Up

Implement the fixed H30/32 associative trajectory solve and dense active-KKT parity.
