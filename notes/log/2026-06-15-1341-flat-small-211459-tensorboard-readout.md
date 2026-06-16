# 2026-06-15 13:41 Flat-small 21:14 TensorBoard Readout

## Purpose

判断 `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-14_21-14-59` 是否值得继续训练。

## Stage

training metrics / flat-small avoidance / goal-anchored command / MPC runtime configuration

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Procedure

- 读取 run 的 `env_cfg.yaml` 和 TensorBoard scalar。
- 检查当前训练进程。

## Input Conditions

- Run path: `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-14_21-14-59`
- Saved command cfg:
  - `GoalAnchoredVelocityCommand`
  - `vx_abs_range=(0.6, 1.0)`
  - `vy_abs_range=(0.0, 0.4)`
  - `yaw_stiffness=0.5`
  - `yaw_range=(-0.8, 0.8)`
- Saved MPC runtime:
  - `parallel_plan_batch_size=1`
- Live process:
  - `python Go2Pvcnn/scripts/train.py ... --num_envs 4096 ... --mpc_num_envs 1`

## Key Metrics

- Scalar steps: `19999 -> 24367`
- `Train/mean_episode_length`: last500 `638.141`, last100 `627.166`, max `772.0`, last `582.44`
- `Train/mean_reward`: last500 `3.58743`, last100 `0.088211`, max `8.57772`, last `2.89012`
- `Curriculum/terrain_levels/mean_terrain_level`: last500 `1.56693`, last100 `1.80711`, max `1.87877`, last `1.71948`
- `Episode_Reward/track_lin_vel_xy`: last500 `0.80803`, last100 `0.788091`
- `Episode_Reward/track_ang_vel_z`: last500 `0.395199`, last100 `0.382496`
- `Metrics/base_velocity/error_vel_xy`: last100 `0.0244287`
- `Metrics/base_velocity/error_vel_yaw`: last100 `0.0254417`
- `Episode_Reward/semantic_body_part_clearance`: last500 `-0.0676428`, last100 `-0.123783`, min `-4.18368`
- `Episode_Reward/semantic_foot_over_clearance`: nonzero `84/4369`, last100 `0.0`, max `0.0111687`
- `Episode_Reward/reference_foot_pos`: last500 `3.09119e-05`, last100 `2.93314e-05`
- `Episode_Termination/bad_orientation`: first100 `368.657`, last100 `3.56`
- `Episode_Termination/base_contact`: last100 `0.00208333`
- `Perf/collection time`: last100 `9.26708s`

## Result

这条 run 明显好于 `2026-06-14_00-22-58` 的崩溃 run：姿态崩溃基本恢复，episode 长度和 tracking 都上来了，地形课程也从 0 打开到约 1.8。

但它不适合长时间继续训练：课程仍停在低层，`semantic_foot_over_clearance` 最近窗口为 0，`semantic_body_part_clearance` last100 比 last500 更负，说明小障碍跨越信号没有稳定形成。同时 `--mpc_num_envs 1` 让 4096 个环境里只有极少数环境持续拿到 MPC 参考，轨迹模仿线索过弱。

## Conclusion

不建议继续长训这条 run。若还在跑，可最多观察到 `model_25000` 附近；如果 `mean_terrain_level` 仍不能超过 `3`、`mean_episode_length` 不能接近 `800+`、`semantic_foot_over_clearance` 仍为 0，应停止。

更推荐停止后用更合理的运行条件重启：

- `--mpc_num_envs` 至少 `64`，更理想是 `128` 或按显存压测上调。
- 速度比当前更保守，尤其降低横向速度和 yaw 压力。
- 先对 `model_24300.pt` 做受控跨越 eval，确认是否已有真实跨越行为。

## Follow-Up

- 如果继续沿 goal-anchored command 训练，下一条 run 应避免 `--mpc_num_envs 1`。
- 若受控 eval 仍无 foot-over，应继续改 reward/curriculum，而不是单纯延长训练。

## Git Refs

- Baseline Ref: `71ae5c7`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py](../../Go2Pvcnn/go2_pvcnn/mdp/commands/velocity_command.py)
