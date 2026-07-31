# Parallelism 平地 RL Tracking 设计

日期：2026-07-31

## 目标

在 `Parallelism-flat-rl` 分支上新增一个平地 RL tracking 任务，让下层强化学习 policy 跟踪 parallelism planner 生成的 Go2 关节轨迹。第一版只做平地，保留高程图和语义图观测接口，为后续接入语义地形和强化学习环境扩展做准备。

参考代码包括：

- `raw/whole_body_tracking`
- `raw/InstinctLab`

这些 raw 目录只作为设计参考。实现时不直接 import raw 包，不复制 raw 的工程结构，不把 raw 目录纳入提交。

## 设计原则

1. `extension/parallelism` 只负责规划。
2. RL 训练任务代码集中放在 `Go2Pvcnn/Go2Pvcnn/tracking/`。
3. 参考 InstinctLab 的 reference-as-command 思路，但 parallelism reference 第一版只输入当前 1 帧。
4. actor 和 critic 都接收高程图/语义图 CNN map。
5. 速度 command 同时输入 parallelism planner 和 policy，且 24 帧规划周期内保持不变。

## 总体数据流

```text
base_velocity command 课程采样 vx/vy/vyaw
        |
        v
parallelism planner 每 0.48s 规划 24 帧
        |
        v
parallelism reference cache 保存 24 帧轨迹
        |
        v
policy 每个 control step 读取当前 phase 的 1 帧 reference
        |
        v
policy 输出 12 维 joint position action
```

parallelism planner 每次生成 24 帧，`plan_dt = 0.02s`：

```text
planner_horizon_frames = 24
planner_cycle_s = 24 * 0.02 = 0.48s
```

因此 `base_velocity` command 的重采样周期也必须是 0.48s：

```text
resampling_time_range = (0.48, 0.48)
```

一个 planner cycle 内：

- `vx, vy, vyaw` 保持不变。
- parallelism 使用这组速度生成 24 帧 reference。
- policy 每个 step 只读取当前 phase 的 1 帧 reference。
- phase 到第 24 帧后，重新采样速度并重新规划。

## Reference 定义

参考 InstinctLab 的拆分方式，reference 不作为散乱的普通 observation 拼接，而由 tracking 任务提供统一的 parallelism reference source。第一版实际输入 policy 的 reference 为：

```text
parallelism_ref_joint_pos_rel_t
parallelism_ref_joint_vel_t
parallelism_ref_root_lin_vel_b_t
parallelism_ref_root_ang_vel_b_t
```

定义如下：

| 名称 | shape | 坐标系/含义 |
| --- | --- | --- |
| `parallelism_ref_joint_pos_rel_t` | `(num_envs, 12)` | 当前 phase 的参考关节角减去 `default_joint_pos`。这是关节角，不是 root 坐标系下的位置。 |
| `parallelism_ref_joint_vel_t` | `(num_envs, 12)` | 当前 phase 的参考关节速度。 |
| `parallelism_ref_root_lin_vel_b_t` | `(num_envs, 3)` | 当前 phase 的参考 root 线速度，表达在 root/body frame。 |
| `parallelism_ref_root_ang_vel_b_t` | `(num_envs, 3)` | 当前 phase 的参考 root 角速度，表达在 root/body frame。 |

parallelism 内部仍保留完整 24 帧轨迹，用于 phase 推进、reward、termination 和后续扩展。policy 第一版不输入未来帧。

## 速度 Command 与课程

速度 command 名称沿用 `base_velocity`。它同时服务两件事：

1. 输入 parallelism planner，生成 24 帧 reference。
2. 输入 policy observation，让 policy 知道当前目标速度。

速度表达在 root/body frame：

```text
velocity_commands = [vx, vy, vyaw]
```

课程只改变采样范围，不改变 0.48s 重采样周期。建议 level 0 到 level max 线性插值：

```text
level 0:
  vx   [-0.1, 0.1]
  vy   [-0.05, 0.05]
  vyaw [-0.2, 0.2]

level max:
  vx   [-1.0, 1.0]
  vy   [-0.5, 0.5]
  vyaw [-1.0, 1.0]
```

每个 episode 结束时更新课程。第一版采用简单、可解释的成功/失败规则：

成功条件：

```text
episode 以 time_out 结束
没有 base_contact
没有 bad_orientation
没有 reference-too-far termination
mean_lin_vel_error < 0.25 m/s
mean_ang_vel_error < 0.35 rad/s
mean_joint_error < 0.35 rad
```

其中：

```text
lin_vel_error = norm(real_base_lin_vel_b[:2] - cmd[:2])
ang_vel_error = abs(real_base_ang_vel_b_z - cmd_yaw)
joint_error = mean(abs(real_joint_pos - ref_joint_pos))
```

课程更新：

```text
success: level += 1
failure: level -= 1
level clamp 到 [0, max_level]
```

实现时需要记录这些指标，方便后续把简单规则替换成滑动成功率或分 env 统计。

## Actor Observation

actor 使用低维状态加 CNN map。低维状态：

```text
base_lin_vel
base_ang_vel
projected_gravity
joint_pos_rel
joint_vel_rel
last_action
velocity_commands
parallelism_ref_joint_pos_rel_t
parallelism_ref_joint_vel_t
parallelism_ref_root_lin_vel_b_t
parallelism_ref_root_ang_vel_b_t
```

地图观测：

```text
elevation_semantic_map
```

shape：

```text
(num_envs, 2, 16, 16)
```

通道：

```text
channel 0: elevation height map
channel 1: semantic id map
```

当前平地训练时语义图基本为 0，高程图为平面，但 actor 接口保留。

## Critic Observation

critic 第一版和 actor 对齐，也接收：

```text
base_lin_vel
base_ang_vel
projected_gravity
joint_pos_rel
joint_vel_rel
last_action
velocity_commands
parallelism_ref_joint_pos_rel_t
parallelism_ref_joint_vel_t
parallelism_ref_root_lin_vel_b_t
parallelism_ref_root_ang_vel_b_t
elevation_semantic_map
```

第一版不做 privileged critic，避免训练输入和部署输入差异过大。

## Action

动作沿用 Go2 的 joint position action：

```text
12 维 joint position action
use_default_offset = True
```

action scale 第一版可沿用现有 Go2 训练配置，或使用 `0.25` 作为保守初值。

## Reward

第一版 reward 重点是关节轨迹 tracking 和速度 tracking：

```text
reference_joint_pos_reward
reference_joint_vel_reward
track_lin_vel_xy_exp
track_ang_vel_z_exp
action_rate_l2
joint_pos_limits
joint_vel / torque / action_smoothness 稳定项
```

当前只做平地，所以 semantic collision reward 不作为主训练项。地图输入先保留接口。

## Termination

参考 InstinctLab 的 reference-too-far 思路，并适配 Go2 parallelism reference。第一版启用：

```text
time_out
base_contact
bad_orientation
parallelism_ref_root_z_too_far
parallelism_ref_projected_gravity_too_far
parallelism_ref_foot_z_too_far
parallelism_ref_joint_pos_too_far
terrain_out_of_bounds
```

建议阈值：

```text
parallelism_ref_root_z_too_far:
  abs(real_root_z - ref_root_z) > 0.25m

parallelism_ref_projected_gravity_too_far:
  projected gravity diff > 0.8

parallelism_ref_foot_z_too_far:
  any abs(real_foot_z - ref_foot_z) > 0.25m

parallelism_ref_joint_pos_too_far:
  max(abs(real_joint_pos - ref_joint_pos)) > 0.8rad
```

`parallelism_ref_joint_pos_too_far` 默认开启。阈值初始使用 0.8rad，避免训练初期 episode 过碎；稳定后可收紧到 0.5rad。

重要规则：

```text
parallelism planner invalid 不直接 termination
```

如果某一轮 planner 所有 candidate invalid，使用已有 standstill/fallback 逻辑。planner 可行性问题不能直接当作 policy 失败。

## 平地范围

第一版只做 flat：

```text
terrain = plane 或 flat generator
semantic obstacle curriculum 关闭
SemanticGridRayCaster 保留
elevation_semantic_map 保留
```

这样 policy 输入结构与后续语义地形保持兼容，但训练目标先聚焦在平地关节轨迹跟踪。

## 文件结构

新增代码集中放在：

```text
Go2Pvcnn/Go2Pvcnn/tracking/
```

建议结构：

```text
Go2Pvcnn/Go2Pvcnn/tracking/
  __init__.py

  parallelism_tracking_env_cfg.py
    平地 RL env cfg
    包含 scene / commands / actions / observations / rewards / terminations / curriculum

  mdp/
    __init__.py

    observations.py
      parallelism_ref_joint_pos_rel_t
      parallelism_ref_joint_vel_t
      parallelism_ref_root_lin_vel_b_t
      parallelism_ref_root_ang_vel_b_t
      地图观测调用现有 extension.mdp.observations.downsampled_elevation_semantic_scan

    rewards.py
      reference_joint_pos_reward
      reference_joint_vel_reward
      需要时包装现有 rewards_reference

    terminations.py
      parallelism_ref_root_z_too_far
      parallelism_ref_projected_gravity_too_far
      parallelism_ref_foot_z_too_far
      parallelism_ref_joint_pos_too_far

    curriculums.py
      parallelism_velocity_curriculum
      根据 episode success/failure 更新 base_velocity range

    commands.py
      如有需要，提供 parallelism reference command/source

  managers/
    __init__.py
    parallelism_reference_manager.py
      管理 24 帧 trajectory cache
      控制 0.48s 重规划
      提供当前 phase 的 1 帧 reference
```

训练入口后续注册新 task，例如：

```text
Go2-Parallelism-Tracking-Flat-v0
```

## 与参考项目的关系

`raw/whole_body_tracking` 的参考点：

- policy command 只使用当前参考帧。
- target root/body velocity 主要用于 reward，但本设计将 planner 目标 root velocity 作为 policy 输入，服务 Go2 速度命令跟踪。
- 关节误差在 raw 中是 metric/reward，不是默认 termination。

`raw/InstinctLab` 的参考点：

- reference-as-command/source 的组织方式。
- base/link/reference-too-far termination。
- joint_pos_far_from_ref 的 termination 思路。

本设计只参考这些思想，不直接依赖 raw 代码。

## 验证要求

实现后需要验证：

1. task 能在平地启动训练。
2. `base_velocity` 每 0.48s 更新一次，24 帧内不变化。
3. parallelism planner 每 24 帧重新规划一次。
4. policy 每步只读取当前 phase 的 1 帧 reference。
5. actor 和 critic 都包含 `elevation_semantic_map`。
6. `parallelism_ref_joint_pos_too_far` 默认开启，并能在指标中看到触发次数。
7. 课程 level 能随 episode 成功/失败变化。
8. raw 目录没有被 import，也没有被提交。
