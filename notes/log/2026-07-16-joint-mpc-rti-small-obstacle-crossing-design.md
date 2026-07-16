# Joint MPC RTI Small-Obstacle Crossing Design

- Purpose: record the follow-up design for fixing the measured small-obstacle crossing failures while preserving the existing joint MPC RTI contracts.
- Stage: T302v.4 design / `joint_mpc_rti`.
- Related todo: [T302v.4](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `b99cda0`.
- Candidate Ref: design-only working tree before implementation.
- Spec: [../../docs/superpowers/specs/2026-07-16-joint-mpc-rti-small-obstacle-crossing-design.html](../../docs/superpowers/specs/2026-07-16-joint-mpc-rti-small-obstacle-crossing-design.html).
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`, `Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py`, `Go2Pvcnn/extension/joint_mpc_rti/losses/rollout_objective.py`, `Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py`.

## Context

The design extends the existing GPU RTI spec and the joint-order/stance-grounding spec:

- [GPU RTI design](../../docs/superpowers/specs/2026-07-15-joint-mpc-rti-gpu-design.html)
- [Joint order and stance grounding design](../../docs/superpowers/specs/2026-07-16-joint-mpc-rti-joint-order-stance-grounding-design.md)

It responds to the controlled small-obstacle quantification: strict cross success was `1/223 (0.45%)`, with calf collision invalidating `97.8%` of opportunities and foot collision invalidating `78.0%`.

## Design Result

- Keep `H=16`, `dt=0.02s`, measured `x0`, rolling MPC, and publish-only-`x1`.
- Change fixed trot timing to `half_cycle_steps=8`, so `H16` covers one full `stance -> swing -> stance` cycle.
- Keep the no-hard-gate behavior contract: no `crossable_small`, specified crossing leg, shape branch, fixed bypass side, snapping, projection, or repair.
- Upgrade both small and large semantic distance channels to signed boundary distance so inside-obstacle samples keep an exit gradient. The written contract now requires complementary occupied/free exact EDT passes, a half-cell boundary correction, finite empty/full-channel behavior, and query-time analytic interpolation gradients; multiplying the old zero-inside unsigned EDT by a sign is explicitly rejected.
- Align optimization geometry and acceptance geometry: foot sphere `0.022m`, calf capsule `0.040m`, thigh capsule `0.040m`.
- Add foot/calf/thigh small clearance residuals to both GGN/LQ and full merit, instead of leaving collision only as final scoring/diagnostic.
- Add strict cross success and stance-ground metrics to acceptance.

## Verification

Design-document checks only:

- HTML parser check passed for the new spec, including the signed-distance clarification requested during user review.
- Placeholder scan found no `TBD`, `TODO`, `待定`, `占位`, or `placeholder`.

No production planner or test code was changed in this pass; the user is still reviewing the design.

## Follow-Up

Write the implementation plan, then implement RED/GREEN tests for signed distance, GGN/LQ visibility, strict cross, stance-ground metrics, variable `num_envs`, and the old joint/viewer/old-MPC regressions.
