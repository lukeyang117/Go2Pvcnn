# MPC Touchdown Keepout Runtime Cache

## Purpose

Investigate and fix the 1024 RL / 1024 MPC training collection slowdown where later collection windows stretched from about `6s` to `17s`, `41s`, and `46s` while learning stayed near `0.53s`.

## Stage

Batch MPC planner / parametric sampled losses / `TeacherElevationTrajectoryMpcSemanticEnvCfg` 1024-env runtime.

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Commands

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh && conda activate env_isaacsim
T302G_MPC_PROFILE_LIMIT=12 T302G_MPC_PROFILE_LOSS_TERMS=20 CUDA_VISIBLE_DEVICES=1 python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 60 --require-replan --print-cuda-memory --summary-path /tmp/mpc_probe_term_profile_60.json --optimize-steps 24
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
T302G_MPC_PROFILE_LIMIT=8 CUDA_VISIBLE_DEVICES=1 python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 60 --require-replan --print-cuda-memory --summary-path /tmp/mpc_probe_final_1024_60.json --optimize-steps 24
```

## Input Conditions

- `num_envs=1024`
- `mpc_num_envs=1024`
- `horizon_steps=25`
- `optimize_steps=24`
- `replan_interval_steps=25`
- Environment cfg path: `TeacherElevationTrajectoryMpcSemanticEnvCfg`

## Key Metrics

- Before this fix, a 60-step cached-semantic probe still hit a later `mixed_zero_split` long tail:
  - inner nonzero subset `batch=911`
  - `plan.parametric_ms=35861.512`
  - `loss.total_ms=34898.914`
  - `term.touchdown_clearance_ms=33820.227`
  - outer `plan.mixed_zero_split_ms=35865.191`
  - `epoch_seconds=45.894`
- After caching low-small component circles once per replan:
  - third replan inner subset `batch=922`
  - `plan.parametric_ms=2607.813`
  - `loss.total_ms=374.120`
  - `term.touchdown_clearance_ms=21.731`
  - outer `plan.mixed_zero_split_ms=2611.850`
  - final `epoch_seconds=15.258`
  - CUDA max allocated `7547703296`, reserved `9290383360`
- Focused regression: `169 passed in 5.27s`.
- Pycompile exit `0`.

## Result

Pass. The long tail was not caused by `mixed_zero_split` itself. `mixed_zero_split` was timing the outer call while the inner nonzero subset planner spent tens of seconds in `term.touchdown_clearance`.

Root cause: `parametric_touchdown_keepout_loss()` rebuilt `low_small_component_circles()` inside each sampled-frame loss call. That helper performs fixed-grid component labeling over the semantic map, and the semantic map is static during a replan. With `optimize_steps=24`, the same component extraction was repeated around every optimizer iteration when touchdown semantic triggers were present.

Fix: `parametric_touchdown_keepout_loss()` now accepts optional precomputed `LowSmallCircles`; `_optimize_parametric_variables()` builds that context once per replan and passes it through `_parametric_sampled_frame_losses()`.

## Conclusion

The 1024/1024 memory gate remains stable, and the MPC collection long tail is removed for the reproduced 60-step probe. No loss key was added and no loss weight was changed.

## Follow-Up

`plan.parametric_ms` is still around `2.6-2.9s` per 1024-env replan with 24 optimizer steps, so further speed work can target one-time per-replan geometry/context construction or reduce optimize-step cost. The urgent 35s touchdown keepout long tail is closed.

## Git Refs

- Baseline Ref: working tree before this fix on branch `costmap-teacher-ablation`
- Candidate Ref: current working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
