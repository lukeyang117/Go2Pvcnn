# Joint MPC RTI Published Root XY Priority

## Purpose

Implement and evaluate Task 14D: preserve nominal command progress by fixing only the published `x1` QP root-XY correction while retaining the six mixed published-kinematics rows.

## Stage And Contract

- Planner: `Go2Pvcnn/extension/joint_mpc_rti`
- Scenario: actual-state S4 sphere `small_forward` on `cuda:1`
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Frozen architecture: H30, one QP, one SQP/RTI, five alphas, four filters, seven losses, cold once then warm only

The existing box bounds now fix `delta p_B,1,xy=0`. No KKT row, loss, filter, projection, recovery, repair, or second solve was added.

## Solver Root Cause And Fix

The first affine seed used the unconstrained minimum-norm solution `J^T(JJ^T)^-1 b`. It could therefore assign motion to fixed `x1` root XY. `feasible_step` correctly ignored already-active box coordinates, but that meant an infeasible fixed coordinate could survive both refinements when other bounds blocked the path to the active KKT solution.

A synthetic three-blocker regression reproduced the defect with final `x1 XY=[0.021,0.021]m`. The seed is now constructed in the free subspace of the initial fixed boxes: fixed values are written first, then the six-row correction is solved with the fixed columns removed. The regression is green for dense and production scan.

A real-FK feasibility probe with `6mm` stance targets found an exact fixed-root solution using root z/RPY and joints; maximum joint correction was about `0.013rad`, with zero box and velocity violation. Task 14D is therefore mathematically feasible at the linear QP layer.

## Verification

- New RED: one expected failure, fixed x1 XY residual `0.021m`
- New GREEN: `1 passed`
- QP/scan/RTI/backend including CUDA compile: `58 passed`
- Full focused QP/scan/loss/line-search/RTI/backend union: `93 passed`
- Contract/gait/terrain supplement: `37 passed`
- `py_compile`: pass
- `git diff --check`: pass

## Representative Viewer

Report: `/tmp/joint_mpc_viewer_published_root_xy_seed_fixed.json`

The solver invariant is closed:

- full-QP published root-XY deviation: `0m`, violations `0`
- selected published root-XY deviation: `0m`, violations `0`
- root velocity error: `0.12535m/s`, pass
- root direction error: `0.09765rad`, pass
- stance slip/anchor: `0.01893/0.01893mm`, pass
- swing clearance: `+3.748um`, pass
- strict crossing, collision, penetration, cold-once/warm-only, and `20ms` foot lead: pass

The representative gate remains red:

- trajectory validity: `0.77551 < 1.0`
- joint step metric: `0.37823rad > 0.35rad`
- stance ground gap: `0.08101m > 0.012m`
- airborne touchdown: `0.04082 > 0`

Phases `13..23` produce eleven invalid cycles. All five candidates pass finite, joint-position, and joint-velocity filters but fail `published_kinematics`, including alpha zero. The stance target grows to approximately `-6.09mm`; the QP direction keeps x1 root XY exactly zero, so the remaining root/RPY/joint first-order correction cannot satisfy the exact-FK `0.5mm` continuing-stance filter in one RTI.

## Decision

Retain the free-subspace seed fix because it enforces the approved box/KKT solver contract. Reject Task 14D as a complete behavior solution: it makes root tracking green but exposes an architectural one-linearization feasibility conflict and retains the joint discontinuity. Do not run ranked/formal or Stage B. Further behavior work requires a user-approved architecture amendment; do not add projection, another SQP/QP, recovery, candidate repair, or relaxed thresholds implicitly.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `solver/trajectory_qp.py`, `test_trajectory_qp.py`, Task 14D design and plan
