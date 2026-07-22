# Joint MPC RTI Warm x1 Manifold Implementation

## Purpose

Implement and evaluate Task 14E方案 A: one explicit pre-LQ warm published-x1 continuing-stance manifold initialization.

## Stage And Contract

- Todo: [T302v Joint MPC RTI GPU](../todo/T302v-joint-mpc-rti-gpu.md)
- Backend: pure-kinematic H30 `joint_mpc_rti`
- One QP, one SQP/RTI, five alphas, four filters and seven losses unchanged
- Cold exactly once, then warm-only
- No recovery, candidate repair, semantic branch, projection or second solve

For `continuing = stance(x0) AND stance(x1)`, shifted x1 FK supplies z and the persistent anchor supplies XY. Existing batched analytic IK replaces only continuing-leg x1 joints. The operation preserves x0, x1 root, swing/onset joints and x2..x30.

## TDD And Verification

- Initial 6mm RED: failed with the expected `0.005999997m` XY residual.
- Main GREEN: exact-FK XY and preserved z pass.
- Invalid paths: unreachable, joint-position and x0-to-x1 velocity failures remain warm and mark nominal invalid.
- Downstream: support target near zero and alpha-zero continuing exact-FK feasibility pass.
- Dynamic batches: `B=1/40/512/1024` finite fixed-shape pass.
- Nominal/loss/QP/scan/line-search/RTI/backend: `124 passed in 23.86s`.
- Contract/analytic-IK/gait/kinematics/terrain: `47 passed in 5.69s`.
- Final nominal: `28 passed in 4.00s`.
- `py_compile` and `git diff --check`: pass.

## Representative Viewer

- Device: `cuda:1`
- Scenario: actual-state S4 sphere `small_forward`
- Report: `/tmp/joint_mpc_viewer_warm_x1_manifold.json`
- Executed cycles: `48`, actual samples: `49`

Closed metrics:

- invalid cycles: `0` (Task 14D had phases `13..23` invalid)
- trajectory validity: `1.0`
- root velocity error: `0.107394m/s`
- root direction error: `0.079696rad`
- joint step: `0.346827rad`
- stance XY slip max/mean: `0.338244/0.013802mm`
- swing clearance: `+9.276um`
- strict crossing: `1.0`
- collision and maximum penetration: `0`
- cold/warm lifecycle: `1 cold`, `47 warm`, unexpected restart/invariant fault `0`

Open failures:

- stance ground gap: `0.084633m > 0.012m`
- airborne touchdown: `0.020408 > 0`
- stance-root carry ratio: `29.7235 > 0.1`

At the worst continuing-stance event, nominal x1 anchor XY error is exactly `0`; full/selected errors are `0.362/0.091mm`, both within the publication tolerance. The XY manifold therefore works. The failed gate is vertical: the approved target preserves shifted foot z, so an old-horizon airborne z can become a new published stance z without being grounded.

## Decision

Retain the scoped Task 14E implementation as solver-valid progress, but do not claim behavior closure. Stop before Task 14F, ranked/formal and Stage B. The next architecture review must decide ownership of continuing-stance z and touchdown-onset z in pre-LQ nominal construction. Do not silently extend this operation to XYZ, add post-solve repair, add another solve, relax thresholds or tune loss scalars.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `work/joint-mpc-kinematic` plus uncommitted Task 14E work
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py`
