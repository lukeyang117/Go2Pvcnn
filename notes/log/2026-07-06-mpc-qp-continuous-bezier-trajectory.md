# MPC QP Continuous Bezier Trajectory Scaffold

## Purpose

Start replacing the repair-dominant `mpc_qp` main path with a continuous trajectory path that samples foot frames from Bezier controls.

## Stage

MPC-QP backend / continuous trajectory redesign.

## Related Todo

[T302v MPC QP safety-constrained backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

Implemented the first scaffold from:

- [continuous Bezier design](../../docs/superpowers/specs/2026-07-06-mpc-qp-continuous-bezier-trajectory-design.html)
- [continuous Bezier plan](../../docs/superpowers/plans/2026-07-06-mpc-qp-continuous-bezier-trajectory-plan.md)

## Changes

- Added `Go2Pvcnn/extension/batch_mpc_qp_planner/bezier.py`.
- Added `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`.
- Added `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`.
- Added `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`.
- Added continuous path config fields on `MpcQpRuntimeCfg`.
- Updated `plan_segment_qp()` so default `continuous_trajectory_enabled=True` uses nominal output as warm start, decodes terrain-bound Bezier controls, samples output foot frames, and does not call the repair main path.
- Added a first fixed-shape continuous solver update: each `qp_iteration` evaluates fixed touchdown footprint height-variation candidates, takes a bounded control-point step toward lower foothold variation, rebinds touchdown z to terrain, and carries P1/P2 smoothly with endpoint motion.
- Added continuous low-small swing clearance handling: the solver samples each Bezier curve at fixed horizon phases, detects low-small semantic cells under swing samples, and lifts `P1/P2.z` by the Bezier-basis-scaled clearance deficit. `P3.z` remains terrain-bound and the default continuous path still does not call repair.
- Added diagnostics `qp_continuous_low_small_clearance_deficit_max`, `qp_continuous_solver_swing_clearance_lift_count`, and `qp_continuous_solver_swing_clearance_deficit_before_max`.
- Added continuous FK/readback handling: diagnostics now report target-foot vs clamped-IK FK foot error, joint frame jump, and the solver performs a bounded `P1/P2` readback update before re-applying swing clearance as the final safety priority.
- Added diagnostics `qp_continuous_fk_readback_error_max`, `qp_continuous_fk_readback_error_mean`, `qp_continuous_joint_frame_jump_max`, `qp_continuous_solver_fk_readback_update_count`, and `qp_continuous_solver_fk_readback_error_before_max`.
- Added continuous root/base progress handling: controls now carry root trajectory, decode solves IK against the optimized root path, and the solver applies a terrain-height-variation progress cap so high-edge paths do not keep full nominal speed progress.
- Added diagnostics `qp_continuous_root_terrain_risk_reduces_progress`, `qp_continuous_root_height_variation_max`, and `qp_continuous_root_progress_scale_min`.
- Kept legacy projected-repair behavior available behind `continuous_trajectory_enabled=False`.
- Updated old repair-focused tests to opt into the legacy mode explicitly.
- Added isolation test proving current `mpc` does not receive `qp_continuous_*` diagnostics.

## Key Metrics

- Focused QP suite: `pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q` -> `31 passed`.
- Current MPC/participation regression: `pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py -q` -> `157 passed, 1 warning`.
- Pycompile: exit `0`.
- `git diff --check`: exit `0`.
- Real GPU1 smoke:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_continuous_16.json --planner-backend mpc_qp --qp-iterations 1
```

  Initial scaffold result: exit `0`, `completed_steps=30`, `planner_backend="mpc_qp"`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, `qp_replan_event_count=2`, CUDA allocated/reserved `0.11GB/0.13GB`, `max_qp_solve_ms_seen≈181.03`, `max_qp_repair_ms_seen≈0.0006`.

  Continuous solver follow-up command:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_continuous_solver_16.json --planner-backend mpc_qp --qp-iterations 1
```

  Result: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, `qp_replan_event_count=2`, CUDA allocated/reserved `0.108GB/0.130GB`, `max_qp_solve_ms_seen≈14.89`, `max_qp_total_ms_seen≈172.43`, `max_qp_repair_ms_seen≈0.0005`.

  Semantic keepout / low-small clearance follow-up command:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_continuous_semantic_16.json --planner-backend mpc_qp --qp-iterations 1
```

  Result: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, `qp_replan_event_count=2`, CUDA allocated/reserved `0.105GB/0.130GB`, `max_qp_solve_ms_seen≈15.72`, `max_qp_total_ms_seen≈171.37`, `max_qp_repair_ms_seen≈0.0005`.

  FK/readback follow-up command:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_continuous_readback_16.json --planner-backend mpc_qp --qp-iterations 1
```

  Result: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, `qp_replan_event_count=2`, CUDA allocated/reserved `0.114GB/0.130GB`, `max_qp_solve_ms_seen≈21.33`, `max_qp_total_ms_seen≈184.47`, `max_qp_repair_ms_seen≈0.0005`.

  Root/base terrain-risk follow-up command:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_continuous_root_16.json --planner-backend mpc_qp --qp-iterations 1
```

  Result: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, `qp_replan_event_count=2`, CUDA allocated/reserved `0.106GB/0.130GB`, `max_qp_solve_ms_seen≈26.94`, `max_qp_total_ms_seen≈213.55`, `max_qp_repair_ms_seen≈0.0006`.

## Result

Pass for static/local scaffold plus 16-env real IsaacLab smoke. This is not yet the full continuous-QP solver, but `qp_iterations` now performs fixed-shape loss-driven Bezier/root control updates for touchdown foothold height variation, touchdown semantic keepout, low-small swing clearance, FK/readback consistency, and root terrain-risk progress reduction. The current implementation establishes trajectory sampling, touchdown z terrain binding, diagnostics, default repair-main-path demotion, and `mpc`/`mpc_qp` isolation.

## Follow-up

Implement the remaining continuous-QP loss/solver pass with fixed-shape buffers and no hard repair fallback as the main solution. Next highest-value items are stronger full-curve semantic/object keepout, richer root/base orientation-height optimization, and viewer acceptance probes for stairs/box/small-obstacle visual continuity.

## Git Refs

- Current Work Ref: local uncommitted workspace on 2026-07-06
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/bezier.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- `Go2Pvcnn/tests/test_mpc_qp_backend.py`
