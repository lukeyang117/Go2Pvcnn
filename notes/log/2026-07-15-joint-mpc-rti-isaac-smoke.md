# Joint MPC RTI IsaacLab Smoke

- Purpose: verify real state/scanner/reference integration.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `cb2fff4`; Candidate Ref: `joint_mpc` working tree.
- Command: `CUDA_VISIBLE_DEVICES=0 .../env_isaacsim/bin/python Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_isaaclab_probe.py --num-envs 1 --steps 1 --headless`.
- Result: `completed_steps=1`, `field_ready_count=1`, field version `0`, backend `joint_mpc_rti`, finite reference, `target_step=1`, `x0_error_max=0.0`.
- First timing: `13717.41 ms`, including JIT and graph capture; not steady-state.
- Two-step follow-up: process ended before final JSON, reproducing the prior second-step hard-exit class.
- Conclusion: one-step boundary contracts pass; multi-step Isaac stability remains open.
