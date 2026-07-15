# Joint MPC RTI IsaacLab Smoke

- Purpose: verify real state/scanner/reference integration.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `cb2fff4`; Candidate Ref: `joint_mpc` working tree.
- Command: `CUDA_VISIBLE_DEVICES=0 .../env_isaacsim/bin/python Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_isaaclab_probe.py --num-envs 1 --steps 1 --headless`.
- Result: `completed_steps=1`, `field_ready_count=1`, field version `0`, backend `joint_mpc_rti`, finite reference, `target_step=1`, `x0_error_max=0.0`.
- First timing: `13717.41 ms`, including JIT and graph capture; not steady-state.
- Follow-up: the second-step exit was traced to field construction inside the RayCaster callback plus returning a graph capture before first replay. Both are fixed in the next verification.
- Conclusion: this log is the original one-step baseline; see [the multi-step fix](2026-07-16-joint-mpc-rti-multistep-isaac-fix.md).
