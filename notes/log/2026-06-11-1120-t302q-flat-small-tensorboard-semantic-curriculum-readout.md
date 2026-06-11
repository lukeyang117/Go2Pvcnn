# T302q Flat-Small TensorBoard Semantic Curriculum Readout

## Purpose

Interpret semantic reward and curriculum TensorBoard scalars for the flat-small continuation run requested by the user.

## Stage

Training metrics / TensorBoard scalar interpretation.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Inspected:

- `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-10_23-24-57`

Read the event file with TensorBoard `event_accumulator` from:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python
```

## Input Conditions

- Event scalar range: steps `19999 -> 24998`.
- Scalar count for the inspected run: `5000` points per main metric.
- Saved checkpoints run from `model_20000.pt` through `model_24998.pt`.

## Key Metrics

- `Episode_Reward/semantic_contact_collision`: last `0.0`, min `-0.0310559`, last-20 mean `-0.000595816`, nonzero `3221/5000`.
- `Episode_Reward/semantic_body_part_clearance`: always `0.0`, nonzero `0/5000`.
- `Curriculum/terrain_levels/mean_terrain_level`: first `1.43247`, last `1.54719`, last-20 mean `1.56239`.
- `Curriculum/terrain_levels/plane_collision_rate`: last `0.0264423`, last-20 mean `0.0186298`, threshold in cfg `0.03`.
- `Curriculum/terrain_levels/plane_env_count`: always `52`.
- `Curriculum/terrain_levels/completed_flat_episodes`: last-20 mean `0.08`.
- `Curriculum/terrain_levels/successful_full_no_collision_episodes`: last-20 mean `0.04125`.
- `Curriculum/terrain_levels/semantic_success_rate`: always `0.0`.
- `Curriculum/terrain_levels/consecutive_success_count`: always `0.0`.
- `Curriculum/terrain_levels/semantic_gate_pass`: always `0.0`.
- `Curriculum/terrain_levels/flat_move_up_count`: always `0.0`.
- `Curriculum/terrain_levels/non_flat_move_up_count`: last `0.55`, last-20 mean `0.56`.
- `Curriculum/lin_vel_cmd_levels`: reaches and stays at `1.0`.
- `Episode_Termination/time_out`: last-20 mean `1.6025`.
- `Episode_Termination/base_contact`: last-20 mean `0.0`.
- `Episode_Termination/bad_orientation`: last-20 mean `0.0225`.
- `Train/mean_reward`: first `-1.52223`, last `29.0467`, last-20 mean `28.5271`.

## Result

Diagnostic pass. The run is healthy from a general reward/command-curriculum perspective, but the new semantic shaping reward is inactive and the episode-level semantic curriculum gate never passes.

## Conclusion

- Real small semantic contacts are sparse and low magnitude near the end, so `semantic_contact_collision` is not dominating training.
- The newly added `semantic_body_part_clearance` reward does not contribute any optimization signal in this run.
- `plane_collision_rate` is below the configured `0.03` threshold on the last-20 mean, but this is not enough for the current flat-small gate.
- The current gate depends on completed flat episodes satisfying timeout and no small contact, base contact, or bad orientation, then reaching `semantic_success_rate >= 0.95` for `5` consecutive checks. This run never reaches that logged success-rate path.
- `plane_env_count=52` and nonzero `non_flat_move_up_count` in the flat-small run remain suspicious bookkeeping signals and should not be over-interpreted as clean flat-only terrain accounting.

## Follow-Up

- Inspect why `semantic_body_part_clearance` is always zero if it is expected to shape leg clearance.
- Inspect flat-only terrain type bookkeeping if `non_flat_move_up_count` should be impossible for this config.

## Git Refs

- Current Work Ref: working tree on 2026-06-11
- Key Files:
  - [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-10_23-24-57](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-10_23-24-57)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
