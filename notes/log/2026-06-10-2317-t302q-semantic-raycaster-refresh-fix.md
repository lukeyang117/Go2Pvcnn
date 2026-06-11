# T302q Semantic Raycaster Late Refresh Fix

## Purpose

Find the exact bottleneck inside `SemanticGridRayCaster._update_buffers_impl()` for the flat-small training slowdown and verify the fix against the baseline semantic MPC training route.

## Stage

RL collection / semantic height scanner.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Added env-gated internal timing in:

- [../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py](../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py)

Timing env vars:

```bash
SEMANTIC_RAYCASTER_TIMING=8
SEMANTIC_RAYCASTER_TIMING_CUDA_SYNC=1
T302G_STEP_TIMING=1
T302G_STEP_TIMING_CUDA_SYNC=1
T302G_STEP_TIMING_STEPS=3
```

Compared:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless --device cuda:0 --num_envs 1024 --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --planner-backend mpc --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07 \
  --load_checkpoint model_19999.pt
```

and:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless --device cuda:0 --num_envs 1024 --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic --planner-backend mpc
```

Logs:

```text
/tmp/t302q_timing/flat_small_semantic_raycaster_timing.log
/tmp/t302q_timing/baseline_semantic_raycaster_timing.log
/tmp/t302q_timing/flat_small_semantic_raycaster_timing_after_fix.log
/tmp/t302q_timing/flat_small_normal_after_refresh_fix_iter.log
```

## Input Conditions

- Flat-small reward mount `semantic_body_part_clearance` is not active in the current cfg.
- Flat-small uses only small semantic obstacles; large semantic root can intentionally remain empty.
- Both timing runs use the same 1024 env count and scanner grid `151x151`.

## Key Metrics

Before fix, flat-small:

- `faces=438956`, `rays_per_env=22801`.
- `refresh=1684-1877ms` per scanner chunk.
- `raycast=1.6-2.1ms`, `semantic_elev=1.6-1.9ms`, `map_write=0.16-0.20ms`.
- `observation.compute=3639-3733ms`.

Baseline before fix:

- `faces=2039524`, `rays_per_env=22801`.
- `refresh=0.26-0.33ms` steady state.
- `raycast=3.8-4.8ms`.
- `observation.compute=26-27ms` steady state.

After fix, flat-small:

- First late refresh still costs one startup rebuild: `refresh=2126.04ms`.
- Subsequent scanner updates: `refresh=0.01-0.02ms`.
- Steady scanner chunk: `total=9.5-9.9ms`.
- `observation.compute=21.53-21.90ms` steady state.
- Normal resumed training produced collection times in the target range after startup:
  - `collection: 8.181s`
  - `collection: 6.338s`
  - `collection: 3.899s`
  - `collection: 5.796s`
  - `collection: 3.768s`
  - later observed `collection: 6.857s`

## Result

Pass. The slowdown was not `raycast_mesh`, not face-id-to-semantic lookup, and not the removed body-part reward. The true hot path was repeated late semantic mesh refresh work before each raycast.

## Conclusion

`_refresh_late_semantic_mesh_if_needed()` previously set `_late_semantic_mesh_refresh_done` only when all configured nonzero semantic ids were present. Flat-small intentionally has no large obstacles, so semantic id `2` may be absent forever. That made every scanner update traverse semantic USD roots and rebuild/check the combined mesh path again.

The fix marks the late refresh as done after a successful late rebuild, even if some optional semantic roots are empty. This preserves the current combined mesh snapshot and prevents repeated USD traversal/rebuild checks in steady-state training.

## Follow-Up

- Keep the env-gated scanner timing helper for future diagnostics.
- If later configs dynamically add semantic geometry after sensor startup, they need an explicit refresh mechanism instead of relying on this per-update late-refresh polling.
- Further optimization such as lower scanner resolution is optional now; the main performance target is already restored.

## Git Refs

- Baseline Ref: working tree before `SemanticGridRayCaster` refresh fix
- Candidate Ref: working tree after `SemanticGridRayCaster._refresh_late_semantic_mesh_if_needed()` fix
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py](../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
