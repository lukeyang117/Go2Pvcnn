# Joint MPC RTI Final Pipeline CUDA Checkpoint

## Purpose

Continue the approved 2026-07-23 pure-kinematic final plan through the production QP and warm-only planner routing, and remove the CUDA Graph blocker caused by the eager dense reference solver.

## Stage

- Final plan Task 10 correctness/capture checkpoint
- Final plan Task 12 production pipeline and runtime checkpoint
- Related todo: [T302v Joint MPC RTI GPU](../todo/T302v-joint-mpc-rti-gpu.md)

## Baseline And Candidate

- Baseline Ref: `7f15334`
- Candidate Ref: uncommitted worktree checkpoint before the Task 10/12 commit
- Branch: `work/joint-mpc-kinematic`

## Root Cause

`solve_trajectory_qp_scan(LqProblem)` still called `solve_dense_qp`, which builds variable-size equality/inequality matrices per environment with Python batch loops and boolean indexing. CUDA Graph capture failed at dynamic `torch.eye`/mask materialization. Replacing isolated tensor constructors could not make that algorithm capture-safe.

## Changes

- Route the final planner through `perceptive_sqp_rti_update` and hard-safe publish/stop semantics.
- Preserve the last accepted finite trajectory on no-feasible-candidate cycles and remain warm-only after initialization.
- Replace the production dense call with a fixed-shape batched block-pentadiagonal solve and two active-mask refinements.
- Keep the dense QP only as an eager test reference and compare B=`1/8/40` directions directly.
- Cache all capture-time constants used by the new LQ/line-search path.
- Copy `preview_tail_state` into and through CUDA Graph fixed-address solver state.

## Verification

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py -q
```

Result: `11 passed`.

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py -q
```

Result: `11 passed in 44.91s`, including real CUDA Graph capture/replay.

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_final_contract.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_field.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_swept_safety.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_lq_problem.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py -q
```

Result: `128 passed in 105.81s`.

## Conclusion

The final planner path is functionally routed and CUDA Graph capture-safe at B=1. The dense reference is no longer reachable from the production `LqProblem` solver path.

## Open Follow-Up

- The current solve is fixed-shape and batched but temporally sequential. It does not yet satisfy the final plan's true H30 padded-to-H32 five-level associative recovery requirement.
- `dense_parity_error` remains a compatibility field and is not computed in production; Task 13 diagnostics must remove or explicitly mark this test-only concept.
- Flat, small, large, real-viewer, B=1024, and `1024 x 1000 < 5s` acceptance have not been run on this candidate.

## Key Files

- `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py`
- `Go2Pvcnn/extension/joint_mpc_rti/solver/lq_problem.py`
- `Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py`
- `Go2Pvcnn/extension/joint_mpc_rti/solver/sqp_rti.py`
- `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
- `Go2Pvcnn/extension/joint_mpc_rti/runtime/cuda_graph.py`
