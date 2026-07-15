# 2026-07-11 MPC QP Viewer Startup GPU Unavailable

## Purpose

Check whether the user-reported `mpc_qp` viewer flash exit can be reproduced from this workspace with the same command shape.

## Stage

MPC-QP viewer / IsaacSim startup.

## Related Todo

- [T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Command

```bash
timeout 80s env CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --qp-iterations 3
```

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Time: `2026-07-11 22:38:33 CST`
- No code edits in this pass.

## Key Output

- Exit code: `139`
- `nvidia-smi` failed before the run: `couldn't communicate with the NVIDIA driver`
- IsaacSim startup reported:
  - `No device could be created`
  - `CUDA error: no CUDA-capable device is detected`
  - `Failed to create primary CUDA context`
  - `Fatal Python error: Segmentation fault`

## Result

The local run did not reach `mpc_qp` planning. It crashed during IsaacSim/GPU initialization.

## Conclusion

This pass is not evidence that one `mpc_qp` prediction is too slow. It only proves the current agent execution context cannot see a CUDA/graphics device for IsaacSim, so real planner timing must be captured either from the user's working viewer process or from a GPU-visible shell.

## Follow-Up

Add or enable viewer-side printing of existing `qp_nominal_ms`, `qp_solve_ms`, `qp_diagnostics_ms`, and `qp_total_ms` from `result.loss_breakdown` to measure the real first replan cost.

## Git Refs

- Baseline Ref: current dirty workspace
- Candidate Ref: unchanged
- Key Files:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
