# Joint MPC RTI Performance Acceptance

- Purpose: verify the fixed-shape planner hot path for 1024 environments.
- Stage: `joint_mpc_rti` Task 14.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `cb2fff4`; Candidate Ref: `joint_mpc` working tree.
- Command: `CUDA_VISIBLE_DEVICES=0 .../env_isaacsim/bin/python Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_perf_probe.py --num-envs 1024 --horizon 16 --steps 1000 --warmup 100 --line-search-alphas 1.0 0.25`.
- Conditions: shared 151×151 field storage, float32, compiled kernels, diagonal state Riccati, production `JointMpcCudaGraphRunner`, named diagnostics disabled.
- Metrics: total `2885.6289 ms`; mean `2.8856 ms`; P50 `2.7341 ms`; P95 `4.5702 ms`; P99 `5.0362 ms`; max `5.5460 ms`; peak `282.58 MiB`; nonfinite `0`.
- Result: pass for required total and mean.
- Follow-up: real IsaacLab multi-step stability remains open.
