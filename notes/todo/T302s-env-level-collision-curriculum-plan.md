# T302s Env-Level Collision Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 flat-small 课程改成每个 env 在 episode 结束时独立判断升降级，并把 curriculum TensorBoard 输出精简到 terrain difficulty。

**Architecture:** 保留现有 `terrain_levels_vel_semantic_plane_gate` 入口，避免改 cfg 的 curriculum term 名称；内部去掉 flat 全局 semantic gate，把 sticky small collision / base contact / bad orientation 直接映射到当前 reset env 的 `move_up` / `move_down`。`SemanticObstacleCurriculumState` 只保留 episode sticky 状态和最近一次调试值，不再驱动全局 gate；TensorBoard curriculum return 只暴露 `mean_terrain_level`。

**Tech Stack:** IsaacLab ManagerBased RL env、PyTorch tensor masks、`semantic_contact_small.data.force_matrix_w`、pytest、`env_isaacsim` real smoke。

---

## Source Design

- [../../docs/superpowers/specs/2026-06-11-flat-small-env-level-collision-curriculum-design.html](../../docs/superpowers/specs/2026-06-11-flat-small-env-level-collision-curriculum-design.html)
- Triggering TensorBoard readout: [../log/2026-06-11-1955-t302q-flat-small-1831-tensorboard-readout.md](../log/2026-06-11-1955-t302q-flat-small-1831-tensorboard-readout.md)

## Conflicting Old Todo Cleanup

The following old assumptions are closed for T302q/T302r and must not be reimplemented:

- Global `semantic_gate_pass` controls flat move-up after consecutive success windows.
- `min_completed_episodes` blocks all flat env upgrades when too few flat episodes reset in one curriculum call.
- `completed_flat_episodes`, `successful_full_no_collision_episodes`, `semantic_success_rate`, `consecutive_success_count`, `semantic_gate_pass`, `flat_move_up_count`, `non_flat_move_up_count`, and `plane_collision_rate` are public TensorBoard curriculum metrics.
- `small_collision` forces immediate downgrade in the first implementation.

## File Structure

- Modify `Go2Pvcnn/extension/semantic_curriculum.py`
  - Add env-level episode result helper.
  - Keep old count/layout helpers.
  - Keep compatibility fields only as internal debug state if needed.
- Modify `Go2Pvcnn/go2_pvcnn/mdp/curriculums.py`
  - Replace global flat gate with env-level `flat_move_up_i = terrain_move_up_i AND episode_success_i`.
  - Add downgrade for base contact / bad orientation.
  - Return only `mean_terrain_level`.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - Add `clearance_scale=1000.0` to flat-small reward params.
  - Remove no-longer-used gate threshold overrides from flat-small config if they are only for global gate.
- Modify tests:
  - `Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py`
  - `Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`

## Task 1: RED Tests For Env-Level Curriculum

- [x] Replace old global-gate tests with tests proving:
  - a single successful flat env upgrades immediately even if only one flat env resets
  - a flat env with sticky small collision does not upgrade
  - small collision does not force downgrade
  - base contact / bad orientation force downgrade
  - non-flat terrain still follows IsaacLab distance curriculum
  - curriculum return keys are exactly `{"mean_terrain_level"}`

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
```

Observed before implementation: `8 failed, 167 passed, 1 warning`; failures were old extra return keys and missing `clearance_scale`.

## Task 2: GREEN Env-Level Episode Logic

- [x] Implement helper logic that computes per-env episode success from reset ids:

```text
episode_success_i =
  time_out_i
  AND NOT episode_had_small_collision_i
  AND NOT base_contact_i
  AND NOT bad_orientation_i
```

- [x] In `terrain_levels_vel_semantic_plane_gate`, use:

```text
flat_move_up_i = terrain_move_up_i AND episode_success_i
flat_move_down_i = terrain_move_down_i OR base_contact_i OR bad_orientation_i
```

- [x] Clear sticky flags only for env ids passed to the curriculum call.
- [x] Return only `mean_terrain_level`.

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
```

Observed after implementation: focused selected tests pass as part of the `184 passed, 1 warning` verification.

## Task 3: Reward Scale Wiring

- [x] Add `clearance_scale` to `semantic_body_part_clearance_reward`.
- [x] Apply scale to the raw negative reward before return while preserving clipping behavior.
- [x] Wire flat-small config with `clearance_scale=1000.0`.
- [x] Add static tests for the config param and pure reward scale behavior.

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: reward scale test and cfg static tests pass.

## Task 4: Focused Verification

- [x] Run focused curriculum/reward/backend tests:

```bash
pytest \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  -q
```

- [x] Run pycompile for touched production files:

```bash
python -m py_compile \
  Go2Pvcnn/extension/semantic_curriculum.py \
  Go2Pvcnn/go2_pvcnn/mdp/curriculums.py \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

## Task 5: Real IsaacLab Smoke

- [x] Run a small `env_isaacsim` smoke after local tests:

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

Observed: process exits `0`; Curriculum Manager has only `terrain_levels`; Reward Manager includes `semantic_body_part_clearance`.

## Task 6: Notes And Logs

- [x] Create a verification log under `notes/log/`.
- [x] Update `notes/log/index.md`.
- [x] Update `notes/todo.md`.
- [x] Update this branch page with actual command results and next step.

## Related Logs

- [../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md](../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md)
- [../log/2026-06-11-2211-t302s-env-level-collision-curriculum-implementation.md](../log/2026-06-11-2211-t302s-env-level-collision-curriculum-implementation.md)
- [../log/2026-06-11-1955-t302q-flat-small-1831-tensorboard-readout.md](../log/2026-06-11-1955-t302q-flat-small-1831-tensorboard-readout.md)

## Git Refs

- Last Feature Commit: `da46138`
- Last Verified Commit: `da46138`
- Current Work Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)

## Next Step

- Run a short resumed training/TensorBoard check to confirm `mean_terrain_level` can move and scaled clearance reward is visible without destabilizing locomotion.
