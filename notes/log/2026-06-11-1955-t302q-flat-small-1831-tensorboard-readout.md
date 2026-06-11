# T302q Flat-Small TensorBoard Readout 2026-06-11 18:31

## Purpose

Read TensorBoard scalars for `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-11_18-31-19` after signal-first clearance radius/margin tuning, and decide whether continuing this same training run is useful.

## Stage

Training metrics / TensorBoard scalar interpretation.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)
- [../todo/T302r-go2-geometry-clearance-reward-plan.md](../todo/T302r-go2-geometry-clearance-reward-plan.md)

## Command / Procedure

Used `env_isaacsim` TensorBoard event reader:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-11_18-31-19", size_guidance={"scalars": 0})
ea.Reload()
...
PY
```

Also checked `env_cfg.yaml` for active clearance params and disabled speed curriculum.

## Input Conditions

- Run path: `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-11_18-31-19`.
- Event steps: `19999..20823`, `825` scalar points.
- Checkpoints: `model_20000.pt` through `model_20700.pt`.
- Signal-first clearance params in `env_cfg.yaml`:
  - query radius `0.5` for foot/calf/thigh/base
  - margins: foot/base `0.2`, calf/thigh `0.4`
  - `lin_vel_cmd_levels: null`

## Key Metrics

- `semantic_body_part_clearance` is no longer identically zero:
  - nonzero `484/825`
  - min `-3.886e-06`
  - full mean `-3.121e-07`
  - last-20 mean `-2.355e-07`
- `semantic_contact_collision` is sparse and disappears late:
  - nonzero `31/825`
  - min `-0.002048`
  - last-100 mean `-1.041e-06`
  - last-20 mean `0.0`
- Curriculum remains closed:
  - `semantic_success_rate=0` for all points
  - `consecutive_success_count=0` for all points
  - `semantic_gate_pass=0` for all points
  - `flat_move_up_count=0`, `non_flat_move_up_count=0`
  - `mean_terrain_level: 0.494 -> 0.0`, last-100 mean `0.0`
- Episode completion bookkeeping:
  - `completed_flat_episodes` last-20 mean `1.553`
  - `successful_full_no_collision_episodes` last-20 mean `1.551`
  - both remain below `min_completed_episodes=8`, so success rate is suppressed to `0`
- Locomotion is already saturated/stable:
  - `Train/mean_reward` last `36.761`, last-100 mean `36.622`
  - `Train/mean_episode_length` last `1000`, last-20 mean `999.262`
  - `track_lin_vel_xy` last-100 mean `1.465`
  - `base_contact=0`, `bad_orientation` last-20 mean `0.00125`
- Performance remains usable:
  - collection time last-100 mean `5.541s`, last-20 mean `5.686s`
  - total fps last-20 mean `6840`

## Result

Diagnostic pass. Signal-first clearance tuning successfully made `semantic_body_part_clearance` nonzero, but the signal is extremely small and the curriculum gate still never opens.

## Conclusion

Do not continue this exact run for a long time. It has mostly converged on stable locomotion at the lowest flat-small curriculum level; the semantic dense reward is technically alive but too small to drive meaningful behavior, and the curriculum sample gate prevents terrain/obstacle progression.

## Follow-Up

- Redesign/rename curriculum metrics and gate sample aggregation before another long run.
- Increase or rescale the clearance reward if it is meant to influence policy, or add part-level diagnostics to check whether the tiny nonzero signal is enough after reward normalization.
- Run behavior eval/play on `model_20700.pt` only as a sanity check, not as evidence that avoidance is solved.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree on 2026-06-11
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
