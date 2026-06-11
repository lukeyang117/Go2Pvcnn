# T302s Env-Level Collision Curriculum Implementation

- Purpose: implement the approved flat-small env-level episode-end collision curriculum and remove conflicting global semantic gate behavior.
- Stage: RL curriculum / flat-small semantic avoidance reward wiring.
- Related todo: [T302s](../todo/T302s-env-level-collision-curriculum-plan.md)
- Procedure:
  - Replaced old global-gate tests with env-level episode-end curriculum tests.
  - Watched RED failures before implementation: `8 failed, 167 passed, 1 warning`.
  - Implemented env-level flat move-up/down logic, TensorBoard return cleanup, old gate API cleanup, and clearance reward scaling.
  - Ran focused local tests, pycompile, grep cleanup, diff check, and an `env_isaacsim` smoke.
- Input conditions:
  - Design source: [../../docs/superpowers/specs/2026-06-11-flat-small-env-level-collision-curriculum-design.html](../../docs/superpowers/specs/2026-06-11-flat-small-env-level-collision-curriculum-design.html)
  - Flat-small run `2026-06-11_18-31-19` showed stable locomotion but global gate remained closed.
- Key metrics:
  - RED: `pytest ... test_semantic_obstacle_curriculum* test_semantic_body_part_clearance_reward.py test_batch_mpc_backend.py -q` -> `8 failed, 167 passed, 1 warning`.
  - GREEN focused: `184 passed, 1 warning`.
  - `py_compile`: exit `0`.
  - `git diff --check`: exit `0`.
  - Active code/test grep for old gate API: no production matches for `plane_collision_rate_threshold`, `min_completed_episodes`, `consecutive_success_required`, `record_completed_flat_episodes`, or noisy curriculum return metric names.
  - Real smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

  - Real smoke result: exit `0`; Curriculum Manager reports only `terrain_levels`; Reward Manager includes `semantic_body_part_clearance`.
- Result:
  - Flat move-up is now per-env and episode-end:

```text
flat_move_up_i = terrain_move_up_i AND time_out_i
                 AND NOT episode_had_small_collision_i
                 AND NOT base_contact_i
                 AND NOT bad_orientation_i
```

  - Flat move-down is:

```text
flat_move_down_i = terrain_move_down_i OR base_contact_i OR bad_orientation_i
```

  - Small collision blocks upgrade but does not force downgrade.
  - Non-flat envs keep the original IsaacLab distance curriculum.
  - Curriculum return is only `mean_terrain_level`.
  - `semantic_body_part_clearance_reward()` accepts `clearance_scale`; flat-small config uses `1000.0`.
- Conclusion:
  - Local and small real smoke verification support the new curriculum contract.
- Follow-up:
  - Start a short resumed training run and inspect TensorBoard: expect curriculum scalars to only expose terrain difficulty and clearance reward to be visible at a larger magnitude.
- Baseline Ref: `da46138`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py](../../Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
