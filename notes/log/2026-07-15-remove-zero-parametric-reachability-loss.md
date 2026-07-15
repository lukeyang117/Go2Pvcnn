# 2026-07-15 Remove Zero Parametric Reachability Loss

## Purpose

Remove the dead `parametric_reachability` loss whose implementation compared `target_foot_pos` with the same tensor passed as `foot_pos`, making it identically zero.

## Stage

- `extension/batch_mpc_planner` sampled parametric loss contract.
- CPU-only verification; no IsaacLab startup or runtime smoke requested.

## Related Todo

- [T302w.10](../todo/T302w-mpc-row8-col12-loss-tuning.md#t302w10-remove-zero-parametric-reachability-loss)

## Change

- Deleted `target_fk_error = norm(target_foot_pos - foot_pos)` from `_parametric_sampled_frame_losses()`.
- Deleted the `parametric_reachability` entry from the returned loss dictionary and therefore from `cost_breakdown` / `loss_breakdown`.
- Removed the key from the test-side required parametric loss set.
- Added explicit CPU assertions that neither breakdown contains `parametric_reachability`.
- No dedicated configuration weight existed for this dead loss, so no `MpcLossesCfg` field was removed.
- Preserved `ik_fk_residual.weight` and `parametric_trajectory_fk_consistency`; those belong to the separate, working target-foot versus FK-foot loss.

## Verification

RED:

- focused contract failed because `parametric_reachability` was still present in `cost_breakdown`.

GREEN:

- focused absence contract: `1 passed`.
- CPU MPC backend + parametric suite: `170 passed in 6.03s`.
- `py_compile`: exit `0`.
- `git diff --check`: exit `0`.
- production search for `target_fk_error|parametric_reachability`: no matches; only the two negative test assertions remain.

## Result

Pass. Current MPC no longer computes, exports, or requires the dead loss. No IsaacLab process was started.

## Git Refs

- Baseline Ref: `1c951ec` plus pre-existing dirty working tree.
- Candidate Ref: same working tree with this focused deletion.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)

