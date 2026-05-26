# T302k Dense Path Retirement

## Purpose

Remove the obsolete dense residual MPC path so the current `extension/batch_mpc_planner` implementation has one active trajectory contract: parametric touchdown/root/curve optimization.

## Stage

`extension/batch_mpc_planner` parametric MPC cleanup.

## Related Todo

[../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Command

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py \
  Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py \
  Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q
```

Additional checks:

```bash
rg -n "extension\\.batch_mpc_planner\\.(nominal|variables|optimizer|losses\\.registry)|build_nominal_trajectory|decode_trajectory|init_optimization_variables|compute_total_loss|use_parametric_trajectory|mpc_use_parametric_trajectory|foot_pos_residual|root_pos_residual|DecodedMpcTrajectory|MpcOptimizationVariables" Go2Pvcnn/extension/batch_mpc_planner Go2Pvcnn/tests
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/*.py Go2Pvcnn/extension/batch_mpc_planner/losses/*.py Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py Go2Pvcnn/tests/mpc_root_cause_probe.py Go2Pvcnn/tests/test_mpc_runtime_headless.py
```

IsaacLab smoke after cleanup:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 --variants parametric_v1 --requested-n-frames 300 --warmup-steps 6 \
  --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,pure_yaw:0.00 0.00 1.00' \
  > tmp/t302k-parametric-mpc/low_small_parametric_v1_t302k_cleanup_gpu3.jsonl 2>&1
```

## Input Conditions

- Working tree on top of T302k parametric MPC changes.
- User explicitly requested old MPC code cleanup and said current MPC code does not need to be preserved.

## Key Metrics

- Focused suite: `209 passed, 1 warning`.
- Old dense source modules removed:
  - `Go2Pvcnn/extension/batch_mpc_planner/nominal.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/optimizer.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/variables.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
- Source scan for old dense symbols returned no matches.
- Pycompile pass for planner package and edited probes.
- IsaacLab cleanup smoke:
  - `cycle_count=3`
  - `fk_foot_over_low_small_success_count=2`
  - `max_replan_initial_foot_error=0.0`
  - `max_replan_initial_touchdown_to_current_foot_error=0.4458`
  - `max_terminal_planned_vs_fk_foot_error=1.97e-6`
  - `max_touchdown_ik_fk_error=0.4755`

## Result

Pass locally; IsaacLab smoke executes successfully with the existing current-touchdown mismatch still visible.

`plan_segment()` now has no dense residual fallback branch, and `MpcRuntimeCfg.use_parametric_trajectory` / task override `mpc_use_parametric_trajectory` are removed. Tests and probes no longer import or monkeypatch the deleted dense modules.

## Conclusion

T302k.8 is closed for source cleanup. The active MPC contract is parametric-only. Remaining behavior work is still T302k.12 current-stance/current-foot touchdown consistency and high/large semantic acceptance.

## Follow-Up

Continue T302k.12. The cleanup smoke still shows planned touchdown/current-foot mismatch around `0.446m`, and pure-yaw has small stance/foot penetration in this run.

## Git Refs

- Baseline Ref: `working tree @ 1b799cd` plus T302k local changes before dense cleanup.
- Candidate Ref: `working tree 2026-05-26 17:57 +0800`.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/__init__.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/__init__.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
