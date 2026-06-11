# T302q Flat-Small Plane Mask Fix

## Purpose

Fix the flat-small curriculum bookkeeping bug where a single flat sub-terrain with multiple generated terrain columns only counted column `0` as flat.

## Stage

RL curriculum metrics / flat-small semantic curriculum gate.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

RED:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py::test_plane_env_mask_treats_single_flat_subterrain_columns_as_flat -q
```

GREEN / focused regression:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py::test_plane_env_mask_treats_single_flat_subterrain_columns_as_flat Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py::test_plane_env_mask_from_terrain -q
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
python -m py_compile Go2Pvcnn/go2_pvcnn/mdp/curriculums.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py
```

Real IsaacLab probe:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
# 64-env flat-small env, attached trajectory manager, reset, inspected terrain_types and flat gate info.
PY
```

## Input Conditions

- Flat-small cfg has `sub_terrains={"flat": ...}` and `num_cols=20`.
- IsaacLab still reports `terrain_types` as column ids `0..19`.
- The intended flat-small curriculum contract is that all columns generated from the single flat sub-terrain count as flat.

## Key Metrics

- RED failed as expected: only index `0` was `True`; index `1` was `False`.
- Target GREEN: `2 passed`.
- Full semantic curriculum term file: `9 passed`.
- Focused curriculum/static suite: `21 passed`.
- `py_compile`: exit `0`.
- Real 64-env IsaacLab post-fix probe:
  - `terrain_names=["flat"]`
  - `terrain_type_unique=0..19`
  - `plane_mask_count=64`
  - `flat_gate_info.plane_env_count=64`

## Result

Pass. `plane_env_mask_from_terrain()` now treats all columns as plane/flat when the terrain generator exposes exactly one terrain name and that name is listed in `plane_terrain_names`.

## Conclusion

This fixes the `plane_env_count=52/1024` flat-small bookkeeping root cause. For flat-small, future `plane_collision_rate`, `completed_flat_episodes`, `semantic_success_rate`, `semantic_gate_pass`, `flat_move_up_count`, and `non_flat_move_up_count` should now be based on all flat columns rather than only column `0`.

## Follow-Up

- Run a short flat-small training/TensorBoard probe before comparing new curriculum metrics against the previous `2026-06-10_23-24-57` run.
- `semantic_body_part_clearance` remains a separate issue: this change only fixes flat mask/curriculum bookkeeping.

## Git Refs

- Baseline Ref: working tree before plane mask fix
- Candidate Ref: working tree on 2026-06-11
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py](../../Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py)
