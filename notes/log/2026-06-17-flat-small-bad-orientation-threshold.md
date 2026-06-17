# 2026-06-17 Flat-small Bad Orientation Threshold

## Purpose

用户反馈 flat-small 学跨小障碍物时容易触发 `bad_orientation`，要求在 `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg` 中放宽阈值。

## Stage

RL config / termination wiring / flat-small avoidance train cfg

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Changes

- `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg.__post_init__()` 中设置 `self.terminations.bad_orientation.params["limit_angle"] = 1.0`。
- 改动仅针对 flat-small cfg；base semantic MPC cfg 默认仍是 `0.8`。

## Verification

- `python -m py_compile Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`: exit `0`

## Git Refs

- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
