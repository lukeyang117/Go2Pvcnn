# 2026-06-17 11:29 Flat-small Remove Speed Curriculum Again

## Purpose

按用户要求，删除 `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg` 的速度课程，不影响 base semantic MPC cfg、地形课程、`GoalAnchoredVelocityCommand` 命令类型和 MPC 参考奖励。

## Stage

RL config / curriculum wiring / flat-small avoidance train cfg

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Input Conditions

- 当前代码曾在 T302u.7 为 flat-small `GoalAnchoredVelocityCommand` 重新启用 `lin_vel_cmd_levels`。
- 用户明确要求删除 `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg` 的速度课程。
- 已有训练进程在 GPU0/1/3 附近运行；真实 smoke 使用 `CUDA_VISIBLE_DEVICES=2` 和 `--device cuda:0`，避免干扰用户长训。

## Changes

- `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg.__post_init__()` 设置 `self.curriculum.lin_vel_cmd_levels = None`。
- 保留 base `TeacherElevationTrajectoryMpcSemanticEnvCfg.curriculum.lin_vel_cmd_levels`。
- 保留 flat-small `GoalAnchoredVelocityCommand`、命令范围、`terrain_levels` 地形课程、MPC reference rewards。
- 测试桩补齐 `reference_contact_reward`，匹配当前本地 reference-contact reward 代码。

## Key Metrics

- RED: `test_flat_small_avoidance_cfg_static_contract` 补齐 fake module 后按预期失败在 `cfg.curriculum.lin_vel_cmd_levels is None`。
- GREEN: focused `1 passed`; related command/curriculum `6 passed`; flat-small cfg subset `3 passed, 146 deselected`; pycompile exit `0`; `git diff --check` exit `0`。
- Real smoke: 4-env flat-small MPC train smoke exit `0`；Command Manager has `GoalAnchoredVelocityCommand`；Curriculum Manager exactly one active term, `terrain_levels`。

## Result

Flat-small training cfg no longer mounts the velocity curriculum. The base semantic MPC cfg still keeps `lin_vel_cmd_levels`, so the change is scoped to `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`.

## Conclusion

This restores the user's desired flat-small contract: fixed command-defined velocity behavior plus terrain curriculum only, while preserving the goal-anchored command path and MPC reference reward wiring.

## Git Refs

- Baseline Ref: `1c951ec`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
