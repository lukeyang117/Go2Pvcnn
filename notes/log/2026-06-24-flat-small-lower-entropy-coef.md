# Flat-Small Lower Entropy Coef

## Purpose

Reduce PPO exploration pressure only for `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance` after two keep-std continuation runs showed policy std rising during training and collapsing around obstacle-crossing attempts.

## Stage

Training config / RSL-RL PPO config.

## Related Todo

[T302s env-level collision curriculum](../todo/T302s-env-level-collision-curriculum-plan.md)

## Change

- Base `teacher_elevation_trajectory_mpc_semantic` keeps `algorithm.entropy_coef = 0.01`.
- Flat-small avoidance now overrides `algorithm.entropy_coef = 0.002`.

Key file:

- [../../Go2Pvcnn/agent/train_cfg.py](../../Go2Pvcnn/agent/train_cfg.py)

## Verification

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_train_cfg_uses_lower_entropy_without_affecting_base \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_entrypoints_are_registered \
  -q

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/agent/train_cfg.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py

git diff --check -- \
  Go2Pvcnn/agent/train_cfg.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py
```

## Result

- Focused pytest: `2 passed`
- `py_compile`: exit `0`
- `git diff --check`: exit `0`

## Conclusion

This keeps `--keep_std` useful at resume time while reducing PPO's incentive to grow action entropy during the continuation run. The next training check should watch `Policy/mean_noise_std`, `bad_orientation`, `base_contact`, `mean_episode_length`, and controlled-crossing success before training far past `300-500` iterations.

## Git Refs

- Baseline Ref: `feea80f`
- Candidate Ref: working tree
