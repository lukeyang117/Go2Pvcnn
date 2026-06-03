# T302n Row-Gated Semantic Curriculum

## Purpose

Verify the new row-based semantic obstacle curriculum design:

- semantic objects are generated once from terrain row and terrain type.
- `plane_counts[row]` and `non_plane_counts[row]` are row difficulty tables.
- the old semantic-level runtime rebuild route is removed from production code.
- flat env row upgrades are gated by flat semantic collision rate; non-flat envs keep the original terrain curriculum behavior.

## Stage

Teacher semantic RL terrain curriculum / static semantic course.

## Related Todo

- [../todo/T302n-semantic-obstacle-curriculum-plan.md](../todo/T302n-semantic-obstacle-curriculum-plan.md)

## Commands

Focused local suite:

```bash
pytest \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py \
  Go2Pvcnn/tests/test_semantic_course_curriculum_layout.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py \
  Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner \
  -q
```

Py compile:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/semantic_curriculum.py \
  Go2Pvcnn/extension/semantic_course.py \
  Go2Pvcnn/go2_pvcnn/mdp/curriculums.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/tests/semantic_obstacle_curriculum_isaaclab_probe.py
```

Real IsaacLab row probe:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/semantic_obstacle_curriculum_isaaclab_probe.py \
  --num-envs 8 \
  --row 9 \
  --force-low-collision-steps 1 \
  --output-json /tmp/t302n_probe_row.json \
  --trace-json /tmp/t302n_probe_trace.json
```

## Key Metrics

- Focused local suite: `24 passed`.
- Py compile: exit `0`.
- IsaacLab probe exit: `0`.
- Probe trace reached:

```text
before_gym_make -> after_gym_make -> after_env_reset -> before_curriculum_compute -> after_curriculum_compute -> after_small_paths -> after_large_paths -> after_output_json
```

- Row 9 flat expected/actual:
  - small: `8 / 8`
  - large: `2 / 2`
- Row 9 non-flat expected/actual:
  - small: `4 / 4`
  - large: `1 / 1`
- Force matrix shapes:
  - small: `[8, 13, 416, 3]`
  - large: `[8, 13, 82, 3]`
- Force matrices finite:
  - small: `true`
  - large: `true`
- Runtime semantic level state:
  - `has_runtime_semantic_level=false`
  - curriculum trace `has_runtime_level=false`

## Result

Pass.

The production path no longer contains the old `semantic_obstacle_levels` runtime rebuild term or `semantic_obstacle_curriculum_level` wiring. The active terrain curriculum route is now `terrain_levels_vel_semantic_plane_gate`, which applies the semantic gate only to flat env move-up.

## Follow-Up

The real probe used 8 envs for fast acceptance. A later performance acceptance can rerun the same probe with 1024 envs if training throughput changes are suspected.

## Git Refs

- Baseline Ref: `f23858e`
- Candidate Ref: working tree

## Key Files

- [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
- [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
- [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
- [../../Go2Pvcnn/tests/semantic_obstacle_curriculum_isaaclab_probe.py](../../Go2Pvcnn/tests/semantic_obstacle_curriculum_isaaclab_probe.py)
