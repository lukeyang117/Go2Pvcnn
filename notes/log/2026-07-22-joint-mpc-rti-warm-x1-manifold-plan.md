# Joint MPC RTI Warm x1 Support Manifold Plan

## Purpose

Record the Task 14D follow-up diagnosis, the user-approved Task 14E architecture, and the exact implementation gate before production code changes.

## Stage

- Todo: [T302v Joint MPC RTI GPU](../todo/T302v-joint-mpc-rti-gpu.md)
- Stage: Task 14E, warm nominal construction before LQ linearization
- Status: approved and planned; RED/GREEN implementation pending

## Input Conditions

- Pure kinematic `Z[B,31,18]`
- Fixed x1 root XY from Task 14D
- One QP and exactly one SQP/RTI update per cycle
- Five alphas, four filters and seven loss families unchanged
- Cold exactly once, then initialized rows remain warm-only
- Representative blocker: S4 sphere `small_forward`, phases `13..23`

## Diagnostic

Warm shifting maps old x2 to new published x1. Only old x1 was constrained by the mixed published-kinematics KKT, so old x2 can be outside the persistent continuing-stance manifold. Existing warm construction shifts/rebases the trajectory and rebuilds references but does not reconstruct x1 stance consistency.

A synthetic fixed-root probe displaced both continuing-stance feet by 6mm in XY. Existing `go2_analytic_ik` reduced both errors to `1.49e-8m`; maximum joint correction was `0.01797rad`, and root state was unchanged. This establishes local kinematic reachability for the observed failure.

## Approved Change

After warm shift/rebase and x0 overwrite, compute shifted x1 FK. For only

```text
continuing_i = stance_i(x0) AND stance_i(x1)
```

build

```text
target_foot_i = [persistent_anchor_i.x, persistent_anchor_i.y, shifted_foot_i.z]
```

and replace only that leg's x1 joints with the existing analytic IK result. Preserve x0, x1 root position/RPY, swing legs, touchdown-onset legs, and x2..x30 exactly.

Reachability, finite IK, physical joint position bounds, and the existing `30rad/s * 0.02s` x0-to-x1 velocity bound enter warm nominal validity. Failure never clears `initialized`, never routes to cold, and never invokes repair, recovery, semantic search, projection, or a second solve.

## Verification Gate

1. RED/GREEN 6mm reproduction and exact FK XY/z checks.
2. Invariance and invalid-path tests.
3. Mixed support target, alpha-zero exact-FK, and `B=1/40/512/1024` contracts.
4. Focused nominal/QP/line-search/RTI/backend regression.
5. The same S4 sphere `small_forward` viewer on `cuda:1`.

Task 14E is evaluated on validity, phases `13..23`, stance gap and airborne touchdown. The phase `23->0` joint-step issue is not part of this change and requires a separate Task 14F decision if it remains red.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `work/joint-mpc-kinematic` with uncommitted Task 14 work
- Key Files: `docs/superpowers/specs/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-design.html`, `docs/superpowers/plans/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-implementation-plan.md`, `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`
