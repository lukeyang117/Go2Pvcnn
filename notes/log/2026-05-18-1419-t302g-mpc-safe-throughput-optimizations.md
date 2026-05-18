# T302g MPC Safe Throughput Optimizations

## Purpose

Optimize MPC execution without reducing `optimize_steps`, changing heavy-loss scheduling, or weakening T302 collision-safety losses.

## Stage

`extension/batch_mpc_planner` planner/loss runtime.

## Related Todo

[T302g](../todo/T302g-mpc-semantic-rl-training-config.md), especially `T302g.5a`.

## Commands

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_plan_segment_skips_optimizer_for_zero_command \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_plan_segment_optimizes_only_nonzero_command_rows \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_ik_fk_residual_reuses_precomputed_ik_without_changing_value \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_result_and_package_do_not_depend_on_old_mode_fields \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_profile_prints_plan_optimizer_and_loss_stages -q

CUDA_VISIBLE_DEVICES='' /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Standalone CUDA timing probes used `CUDA_VISIBLE_DEVICES=1` with 64 envs, horizon `25`, and `optimize_steps=24`.

## Input Conditions

- Heavy-loss scheduling was not changed.
- `optimize_steps` remained `24`.
- T302 collision/semantic loss weights were not changed.
- `T302G_MPC_DISABLE_QUERY_CACHE=1` can disable the new query cache for diagnostics.

## Key Metrics

- Focused MPC tests: `5 passed`
- CPU backend suite: `85 passed, 1 skipped`
- py_compile for modified MPC files: exit `0`
- Fixed decoded loss equivalence with query cache enabled vs disabled: `per max diff 0.0`
- Standalone 64-env all-nonzero profile remained about `2.1-2.2s` after warmup.
- Standalone 64-env half-zero command probe: wall time about `1.70s`, because only nonzero rows enter optimizer.

## Result

Partial pass. Safe zero-command prefiltering is implemented and tested. Terrain query cache is implemented and fixed-decoded equivalent, but its all-nonzero speedup is limited in the flat synthetic profile. IK/FK merge was investigated but not enabled in the production registry because it changed optimization trajectories/costs in the 64-env CUDA probe.

## Conclusion

The highest-confidence throughput improvement in this slice is zero/near-zero command row skipping before optimizer execution. Terrain query cache does not change the fixed decoded objective and remains available, but it should be evaluated on real semantic height fields before relying on it for the 4096 target. IK/FK merging needs a more careful design if pursued; naive reuse of clamped IK/FK intermediate tensors changed the optimized trajectory enough to be unsafe without strict T302 reruns.

## Follow-Up

- Rerun 4096 headless profiling when GPU memory is available.
- Run T302 strict JSONL non-regression after any further optimizer/loss internal changes.
- If IK/FK reuse is revisited, add gradient-level equivalence tests or accept it only after strict metric evidence.

## Git Refs

- Baseline Ref: working tree on top of `946811f`
- Candidate Ref: working tree, 2026-05-18 14:19 CST
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
