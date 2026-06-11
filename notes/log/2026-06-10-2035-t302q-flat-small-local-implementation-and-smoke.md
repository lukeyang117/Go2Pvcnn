# T302q Flat Small Local Implementation And Smoke

## Purpose

Record the local implementation, focused verification, real IsaacLab train smoke, and checkpoint resume compatibility for the flat small-obstacle avoidance RL continuation.

## Stage

RL config / near-field semantic avoidance reward / episode-level semantic curriculum.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Procedure

- Implemented the new flat-small continuation experiment:
  - `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance`
  - `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`
  - `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY`
- Added `semantic_body_part_clearance_reward()` with current IsaacLab body poses for foot/calf/thigh and cached scanner map root anchors.
- Modified the existing semantic curriculum to use episode-level true small-contact success from `semantic_contact_small.data.force_matrix_w`.
- Ran focused local regression and production `py_compile`.
- Ran two real IsaacLab 16-env / 1-iteration train smokes on GPU1:
  - fresh new experiment run
  - resume from existing teacher checkpoint `model_14000.pt`

## Input Conditions

- Baseline ref: `da46138`.
- Existing warm-start checkpoint:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07/model_14000.pt
```

- GPU status before smoke: GPU1/GPU2/GPU3 were idle or near-idle; smoke used `CUDA_VISIBLE_DEVICES=1`.

## Commands

Focused local regression:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_entrypoints_are_registered \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play -q
```

Production compile check:

```bash
python -m py_compile \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/extension/mdp/__init__.py \
  Go2Pvcnn/extension/semantic_curriculum.py \
  Go2Pvcnn/extension/trajectory_manager_factory.py \
  Go2Pvcnn/go2_pvcnn/mdp/curriculums.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/register_envs.py \
  Go2Pvcnn/scripts/train.py \
  Go2Pvcnn/scripts/play.py \
  Go2Pvcnn/agent/train_cfg.py
```

Fresh IsaacLab train smoke:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --num_envs 16 \
  --max_iterations 1 \
  --headless
```

Checkpoint resume smoke:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --num_envs 16 \
  --max_iterations 1 \
  --headless \
  --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07 \
  --load_checkpoint model_14000.pt
```

## Key Metrics

- Focused local regression: `31 passed in 1.65s`.
- Production compile check: exit `0`.
- Fresh IsaacLab train smoke: exit `0`.
- Resume IsaacLab train smoke: exit `0`.
- New env smoke observation/action contract:
  - policy elevation semantic map: `(16, 2, 16, 16)`
  - policy state: `(16, 45)`
  - critic elevation semantic map: `(16, 2, 16, 16)`
  - critic state: `(16, 48)`
  - action: `(16, 12)`
- Reward manager includes `semantic_body_part_clearance` as an active term.
- Planner attach confirmed after the fix:

```text
[Planner] Attached mpc trajectory manager for teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance
```

## Result

Pass after one runtime wiring fix. The first real smoke created the env and showed the new reward term, but failed on the first training step because `reference_foot_pos_reward()` required `env.unwrapped._trajectory_manager`. Root cause: the new experiment inherited planner-owned reference rewards, but `extension/trajectory_manager_factory.py` allowlisted only `teacher_elevation_trajectory_mpc_semantic`.

The fix added `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance` to `TRAJECTORY_MANAGER_EXPERIMENTS` and added a static regression assertion that the new experiment appears in the factory source. After that, fresh train smoke and resume smoke both exited `0`.

## Conclusion

The flat-small avoidance config is locally implemented, focused-test verified, real-env creation verified, reward execution smoke-verified, and compatible with the existing `model_14000.pt` checkpoint shape. The remaining runtime follow-up is the longer `mpc_policy_eval.py --mode small_collision` smoke against a useful flat-small run/checkpoint.

## Follow-Up

- Run T302q Task 9 small-collision eval smoke.
- Optional diagnostic: add a targeted runtime probe if scanner cache clone counts need explicit evidence beyond code-level cache signature tests.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: local working tree on top of `da46138`
- Key Files:
  - `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
  - `Go2Pvcnn/extension/trajectory_manager_factory.py`
  - `Go2Pvcnn/extension/semantic_curriculum.py`
  - `Go2Pvcnn/go2_pvcnn/mdp/curriculums.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
  - `Go2Pvcnn/scripts/train.py`
  - `Go2Pvcnn/scripts/play.py`
  - `Go2Pvcnn/agent/train_cfg.py`
  - `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`
  - `Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py`
  - `Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`
