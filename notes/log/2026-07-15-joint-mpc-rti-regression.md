# Joint MPC RTI Regression Verification

- Purpose: verify joint behavior and preserve old MPC behavior.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `cb2fff4`; Candidate Ref: `joint_mpc` working tree.
- Commands: full `Go2Pvcnn/tests/joint_mpc_rti`; old MPC backend/parametric/participation/env-cfg/eval subset; `py_compile`; `git diff --check`.
- Result: joint `71 passed`; old MPC `193 passed`; compile/diff checks exit `0`.
- Coverage: forward/backward/lateral/yaw/diagonal, left/right large avoidance, left/right small-object foot-over, up/down step, fixed gait rolling.
- Conclusion: `(1.0,0.25)` passes behavior; single alpha was rejected by the down-step test.
