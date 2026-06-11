# T302q Flat-Small Curriculum And Clearance Root Cause Probe

## Purpose

Investigate why the flat-small run reports `plane_env_count=52`, nonzero `non_flat_move_up_count`, `semantic_gate_pass=0`, and always-zero `semantic_body_part_clearance`.

## Stage

RL curriculum metrics / flat-small semantic clearance reward diagnostics.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Static cfg / mask probe:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
# launched IsaacLab AppLauncher headless, instantiated
# TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
# simulated terrain_types = arange(1024) % num_cols,
# and called plane_env_mask_from_terrain().
PY
```

Real env metadata / reward probe:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
# launched 64-env Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-v0,
# attached trajectory manager, reset, stepped zero actions,
# and recorded terrain_types, flat gate info, semantic map ids, and clearance reward.
PY
```

Reward-internal probe:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
# launched 64-env flat-small env, queried current body-part sample points
# through _current_scanner_terrain(), semantic_at(), and height_at().
PY
```

## Input Conditions

- Static probe uses the flat-small cfg with `num_rows=10`, `num_cols=20`, and `sub_terrains={"flat": ...}`.
- Real env probes use `64` envs, `parallel_plan_batch_size=16`, headless IsaacLab, zero actions after reset.
- TensorBoard context run: `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-10_23-24-57`.

## Key Metrics

Static cfg / mask probe:

- `terrain_names=('flat',)`.
- simulated `terrain_types` unique values `0..19`.
- current `plane_env_mask_from_terrain()` count: `52/1024`.
- simulated column counts: first four columns `52` each, remaining columns `51` each.
- expected flat count if all flat columns are treated as flat: `1024`.

Real 64-env probe:

- actual `terrain_names=['flat']`.
- actual `terrain_type_unique=0..19`.
- current `plane_mask_count_current_logic=4/64`.
- flat gate info reports `plane_env_count=4`.
- scanner semantic map shape `[64,151,151]`.
- scanner semantic map contains small id pixels: step-0 counts `{0:1459028, 1:236}`.
- `semantic_body_part_clearance` reward: min `-0.0`, max `-0.0`, mean `0.0`, nonzero `0/64` across inspected steps.

Reward-internal probe:

- body-part sample points per env: `20`.
- sampled semantic ids at body points: `{0:1280}` for all inspected steps.
- sampled body points on small semantic id: `0`.
- sampled points with positive height deficit: `0`.
- sampled points on small id with positive deficit: `0`.
- scanner map did contain small pixels in at least one env, e.g. `138` small pixels in env `3` at reset.

## Result

Diagnostic pass. Two separate issues are evidenced:

1. Flat curriculum bookkeeping misclassifies most flat columns as non-flat because `plane_env_mask_from_terrain()` compares `terrain_types` against the index of `"flat"` in `terrain_names`; with only `("flat",)`, that means only `terrain_types == 0` is treated as flat.
2. The new `semantic_body_part_clearance` reward is inactive in the observed real env states because the current foot/calf/thigh sample points do not query any small semantic cells and have no positive clearance deficit.

## Conclusion

- `plane_env_count=52` is explained by `1024 / 20` column assignment: the current mask only recognizes column `0` as flat even though the flat-small cfg has all columns generated from the single flat sub-terrain.
- Nonzero `non_flat_move_up_count` is therefore mostly a bookkeeping artifact: columns `1..19` are actual flat terrain but enter the non-flat branch.
- The always-zero `semantic_body_part_clearance` TensorBoard metric is consistent with real env probes: scanner maps contain small obstacle ids, but the reward samples only exact/current body part points, and those points did not overlap small semantic ids or clearance deficits in the probes.
- The training signal for small-obstacle avoidance is therefore weak: real contact penalty is sparse, near-field clearance shaping is not firing, and curriculum gate statistics are distorted by flat mask bookkeeping.

## Follow-Up

- Fix or redesign flat column recognition before trusting `plane_env_count`, `semantic_success_rate`, `semantic_gate_pass`, or `non_flat_move_up_count` for flat-small.
- Revisit `semantic_body_part_clearance` support region if it is expected to shape near misses rather than only exact body-point-over-obstacle cases.
- Re-run a short flat-small train/TensorBoard probe after those changes before interpreting curriculum progress.

## Git Refs

- Current Work Ref: working tree on 2026-06-11
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
