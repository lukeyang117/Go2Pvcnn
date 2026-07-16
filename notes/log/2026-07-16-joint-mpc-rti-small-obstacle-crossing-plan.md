# Joint MPC RTI Small-Obstacle Crossing Implementation Plan

- Purpose: record the implementation breakdown approved after the small-obstacle crossing design review.
- Stage: T302v.4 implementation planning.
- Related todo: [T302v.4](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `17e81d3`.
- Candidate Ref: plan-only working tree.
- Plan: [../../docs/superpowers/plans/2026-07-16-joint-mpc-rti-small-obstacle-crossing-implementation-plan.md](../../docs/superpowers/plans/2026-07-16-joint-mpc-rti-small-obstacle-crossing-implementation-plan.md).
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/terrain/`, `model/go2_kinematics.py`, `losses/semantic.py`, `planner.py`, `Go2Pvcnn/tests/joint_mpc_rti/`.

## Result

The plan contains six ordered tasks: RED gait/signed-field tests, signed CPU/CUDA fields, thigh geometry/Jacobians, GGN-visible small-link residuals, native-shape strict-cross/zero-collision acceptance, and full regression/performance evidence.

## Verification

- Placeholder scan found no `TBD`, `TODO`, `待定`, `占位`, or placeholder text.
- The plan contains 25 checkbox steps.
- `git diff --check` passed before the plan commit.

## Follow-Up

Execute inline on `joint_mpc` as explicitly requested by the user, beginning with failing tests before production changes.
