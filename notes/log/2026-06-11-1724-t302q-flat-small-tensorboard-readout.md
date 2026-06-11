# T302q Flat-Small TensorBoard Readout 2026-06-11 17:05

## Purpose

Read TensorBoard scalars for `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-11_17-05-24` and summarize semantic/curriculum/performance behavior after geometry clearance and velocity-curriculum changes.

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
ea = EventAccumulator("logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-11_17-05-24", size_guidance={"scalars": 0})
ea.Reload()
...
PY
```

Also checked `env_cfg.yaml` for active curriculum/reward config.

## Input Conditions

- Run path: `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-11_17-05-24`.
- Event steps: `19999..20131`, `133` scalar points.
- Checkpoints: `model_20000.pt`, `model_20100.pt`.

## Key Metrics

- `env_cfg.yaml`: `lin_vel_cmd_levels: null`.
- `Episode_Reward/semantic_body_part_clearance`: all `0.0`, `133/133` zero.
- `Episode_Reward/semantic_contact_collision`: sparse nonzero `14/133`; min `-0.0015918`; last-20 mean `0.0`.
- `Curriculum/terrain_levels/plane_env_count`: always `1024`.
- `Curriculum/terrain_levels/mean_terrain_level`: `0.4942 -> 0.00488`; last-20 mean `0.00610`.
- `Curriculum/terrain_levels/plane_collision_rate`: max `0.0006348`; last-20 mean `0.0`.
- `completed_flat_episodes`: last `1.4`; last-20 mean `1.56`.
- `successful_full_no_collision_episodes`: last `1.4`; last-20 mean `1.55875`.
- `semantic_success_rate`: all `0.0`.
- `consecutive_success_count`: all `0.0`.
- `semantic_gate_pass`: all `0.0`.
- `flat_move_up_count`: all `0.0`.
- `non_flat_move_up_count`: all `0.0`.
- `Train/mean_reward`: `-1.51 -> 32.71`; last-20 mean `31.56`.
- `Train/mean_episode_length`: reaches `1000`; last-20 mean `999.97`.
- `Perf/collection time`: first `8.856s`, last `4.109s`, last-20 mean `5.611s`.
- `Perf/total_fps`: last-20 mean `6883`.

## Result

Diagnostic pass. Velocity curriculum is disabled and flat mask is fixed, but geometry clearance is still inactive in this training run.

## Conclusion

The run looks stable from reward/episode-length/performance, but the intended pre-contact semantic clearance signal is still not contributing. The semantic curriculum gate also remains closed; however, completed/successful episode counters are near equal while `semantic_success_rate` is zero, so the gate metric is likely being suppressed by the `min_completed_episodes` threshold rather than reflecting total failure.

## Follow-Up

- Probe why geometry clearance still samples no penalized small cells under real 1024-env training conditions.
- Revisit semantic gate thresholds or aggregation if the flat-small run intentionally has very few completed episodes per curriculum call.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree on 2026-06-11
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
