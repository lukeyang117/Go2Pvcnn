# 2026-06-15 16:49 Flat-small Goal-anchored Speed Curriculum

## Purpose

让 flat-small avoidance 继续使用旧的 `lin_vel_cmd_levels` 速度课程，同时对齐当前 `GoalAnchoredVelocityCommand` 的方向逻辑。

## Stage

RL config / command generation / velocity curriculum

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Input Conditions

- 用户明确要求：`lin_vel_cmd_levels` 用现有这一条即可，只需要对齐当前速度方向问题。
- 当前 flat-small cfg 之前关闭了 `self.curriculum.lin_vel_cmd_levels`。
- `GoalAnchoredVelocityCommand` 已经按目标点相对 root/body 坐标系每步更新 `vx/vy` 符号和 yaw，但没有 `ranges/limit_ranges`，旧课程函数无法更新它的速度大小。

## Changes

- `GoalAnchoredVelocityCommandCfg` 增加 `ranges` 和 `limit_ranges` 兼容字段。
- `GoalAnchoredVelocityCommand._resample_command()` 优先从 `ranges.lin_vel_x/y` 推导 `vx_abs/vy_abs` 的采样范围。
- `lin_vel_cmd_levels()` 保持旧课程逻辑，只放宽为同时支持 `UniformLevelVelocityCommandCfg` 和 `GoalAnchoredVelocityCommandCfg`。
- `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg` 重新启用 `lin_vel_cmd_levels`，初始速度范围回到旧模型友好的 `lin_vel_x/y=(-0.1,0.1)`，极限范围为 `lin_vel_x=(-1.0,1.0)`、`lin_vel_y=(-0.5,0.5)`。

## Key Metrics

- Focused RED:
  - `test_goal_anchored_command_uses_curriculum_ranges_for_abs_speed` failed before implementation.
  - `test_flat_small_avoidance_cfg_static_contract` failed before implementation.
- GREEN:
  - `pytest -q Go2Pvcnn/tests/test_goal_anchored_velocity_command.py`: `5 passed`
  - focused cfg tests: `2 passed`
  - curriculum focused test: `1 passed`
  - combined related regression: `18 passed`
  - `python -m py_compile ...`: exit `0`
  - `git diff --check`: exit `0`

## Result

Flat-small training now uses the old `lin_vel_cmd_levels` curriculum again. The curriculum updates signed body-frame ranges, while `GoalAnchoredVelocityCommand` converts those signed ranges into fixed speed magnitudes and keeps direction aligned to the reset-time goal point in the current root/body frame.

## Conclusion

This preserves the old curriculum semantics for the policy reward and TensorBoard scalar, while fixing the current goal-anchored command path so speed magnitude can grow gradually instead of starting directly at high speed.

## Git Refs

- Baseline Ref: `71ae5c7`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
