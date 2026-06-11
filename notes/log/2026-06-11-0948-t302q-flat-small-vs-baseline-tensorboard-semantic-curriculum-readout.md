# T302q Flat-Small vs Baseline TensorBoard Semantic/Curriculum Readout

## Purpose

Read the TensorBoard scalars for the resumed flat-small avoidance run and compare its semantic and curriculum-related metrics against the baseline `teacher_elevation_trajectory_mpc_semantic` run.

## Stage

Training metrics / TensorBoard scalar interpretation.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Inspected runs:

- `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-10_23-24-57`
- `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07`

Read event files with:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python
```

using TensorBoard `event_accumulator`.

## Input Conditions

- Flat-small run is a resume continuation from baseline checkpoint step `19999`.
- Flat-small TensorBoard covers steps `19999 -> 24998` (`5000` scalar points).
- Baseline TensorBoard covers steps `0 -> 19999` (`20000` scalar points).

## Key Metrics

Flat-small:

- `Episode_Reward/semantic_contact_collision`: last `0.0`, last-20 mean `-0.000596`.
- `Episode_Reward/semantic_body_part_clearance`: always exactly `0.0` for all `5000` points.
- `Curriculum/terrain_levels/mean_terrain_level`: `1.432 -> 1.547`, last-20 mean `1.562`.
- `Curriculum/terrain_levels/plane_collision_rate`: last `0.0264`, last-20 mean `0.0186`.
- `Curriculum/terrain_levels/completed_flat_episodes`: last-20 mean `0.08`.
- `Curriculum/terrain_levels/successful_full_no_collision_episodes`: last-20 mean `0.04125`.
- `Curriculum/terrain_levels/semantic_success_rate`: always `0.0`.
- `Curriculum/terrain_levels/consecutive_success_count`: always `0.0`.
- `Curriculum/terrain_levels/semantic_gate_pass`: always `0.0`.
- `Curriculum/terrain_levels/plane_env_count`: always `52`.
- `Curriculum/terrain_levels/non_flat_move_up_count`: last-20 mean `0.56`.
- `Episode_Termination/time_out`: last-20 mean `1.6025`.
- `Episode_Termination/bad_orientation`: last-20 mean `0.0225`.

Baseline:

- `Episode_Reward/semantic_contact_collision`: last `-0.000076`, last-20 mean `-0.000852`.
- No `semantic_body_part_clearance` scalar in baseline.
- `Curriculum/terrain_levels/mean_terrain_level`: last `0.970`, last-20 mean `0.959`.
- `Curriculum/terrain_levels/plane_collision_rate`: last `0.0250`, last-20 mean `0.0267`.
- `Curriculum/terrain_levels/consecutive_success_count`: last `3.65`, last-20 mean `6.93`.
- `Curriculum/terrain_levels/semantic_gate_pass`: last `0.35`, last-20 mean `0.3525`.
- `Curriculum/terrain_levels/plane_env_count`: always `52`.
- `Episode_Termination/time_out`: last-20 mean `1.4875`.
- `Episode_Termination/bad_orientation`: last-20 mean `0.12625`.

## Result

Diagnostic pass. The semantic collision reward remains sparse in both runs. The new body-part clearance reward exists in the flat-small run but never fires. The new flat-small episode-level semantic gate is much stricter than the baseline gate and never passes in this run.

## Conclusion

- `semantic_contact_collision` is near zero in both runs, so real semantic contacts are rare by the end of training and this term is not dominating optimization.
- `semantic_body_part_clearance` being identically zero strongly suggests the reward is currently inactive in practice: either the robot never enters the penalty region, or the geometric/query condition is too sparse to contribute training signal.
- Flat-small’s new gate logic depends on completed flat episodes that both time out and avoid small contact, base contact, and bad orientation, then also requires `semantic_success_rate >= 0.95` for `5` consecutive checks. This run records some successful no-collision flat episodes, but not enough proportionally or consecutively, so `semantic_gate_pass` stays `0`.
- `plane_collision_rate` in flat-small is already below the configured `0.03` threshold on the last-20 mean, but the episode-level success gate is the real blocker.
- `plane_env_count=52` and nonzero `non_flat_move_up_count` even in a flat-only run are suspicious. These metrics likely reflect terrain-type bookkeeping inherited from the generic terrain curriculum path, so they should not be over-interpreted as a clean “all envs are flat” accounting signal.

## Follow-Up

- If `semantic_body_part_clearance` is intended to shape behavior, verify why it never becomes negative in this run.
- If the flat-small curriculum should truly be flat-only, inspect why `plane_env_count` remains `52` and why `non_flat_move_up_count` is nonzero.

## Git Refs

- Current Work Ref: working tree on 2026-06-11
- Key Files:
  - [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-10_23-24-57](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-10_23-24-57)
  - [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
