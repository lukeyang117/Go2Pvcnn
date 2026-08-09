# Parallelism 非平地 Root 轨迹设计

## 目标

在现有 Parallelism foot planner 基础上，新增一套面向非平地的 root 轨迹生成逻辑，使机器人可以在 `random_rough`、`hf_pyramid_slope`、`hf_pyramid_slope_inv`、`boxes`、`pyramid_stairs`、`pyramid_stairs_inv` 上根据高度图调整 root 高度和姿态。

本阶段只修改 root 轨迹生成，不修改 foot touchdown 候选、IK、FK、collision filter、semantic filter、score 和 RL tracking 接口主逻辑。

## 地形分流原则

使用 IsaacLab terrain type 编号分流。terrain type 编号从 `scene.terrain.terrain_types` 读取，再通过 `terrain_generator.sub_terrains` 的 key 顺序映射为地形名。

分流规则固定为：

```text
terrain_name == "flat"  -> 使用当前 cross-obstacles root 轨迹
terrain_name != "flat"  -> 使用新的 terrain-following root 轨迹
```

如果运行环境没有提供 terrain type 或 terrain name，则默认当作 `flat`，保持当前行为。

## 一个训练配置

新增一个单独的 Parallelism RL 配置，基于 `teacher_elevation_trajectory_mpc_semantic_env_cfg.py` 中的地形集合：

```text
flat
random_rough
hf_pyramid_slope
hf_pyramid_slope_inv
boxes
pyramid_stairs
pyramid_stairs_inv
```

这个配置不是拆成 flat 和 ladder 两个任务，而是在同一个训练环境里同时包含这些 terrain type。

只有 `flat` 子地形会叠加之前的 40 个小障碍逻辑。其他 terrain type 不放这 40 个小障碍：

```text
flat:
    semantic small obstacles = 40
    使用 cross-obstacles root

non-flat:
    semantic small obstacles = 0
    使用 terrain-following root
```

`boxes` 视为可行走地形，语义为 0，不作为需要跨越的语义障碍物。

## 当前 Flat Root 保留逻辑

`flat` 使用现有 root 轨迹：

```text
前 12 帧 stance = FR, RL
root_z[0:12] = mean(height(FR_foot_xy), height(RL_foot_xy)) + root_clearance_m

后 12 帧 stance = FL, RR
root_z[12:24] = mean(height(FL_foot_xy), height(RR_foot_xy)) + root_clearance_m
```

roll/pitch 仍然在 `root_leveling_frames` 内回正，yaw 根据 `vyaw` 积分。

这个逻辑适合平地和小障碍物跨越，因为小障碍物不应该直接把 root 顶高。

## Non-flat Terrain-following Root

非 `flat` 地形使用新的 root 轨迹。流程是：

```text
1. 对 command 做 soft clamp，得到 planner 用的有效速度
2. 根据有效速度预测 24 帧 root_xy 和 yaw
3. 在每帧 root_xy 下方查询 height map
4. root_z_raw = height(root_xy) + terrain_following_root_clearance_m
5. frame 0 强制等于当前真实 root_z
6. frame 1..23 对 root_z_raw 做平滑和限速
7. 通过 root 附近前后/左右高程差估计 roll/pitch
8. frame 0 强制等于当前真实 root_rpy
9. frame 1..23 对 roll/pitch 做平滑和限幅，yaw 继续由速度积分
```

root 高度只使用 root 自身预测点的高度，不使用 stance 足端高度，也不使用 body footprint quantile。

## Root 高度模型

对每个环境、每个规划帧：

```text
terrain_z[t] = height_map(root_xy[t])
root_z_target[t] = terrain_z[t] + terrain_following_root_clearance_m
```

第 0 帧：

```text
root_z[0] = current_root_z
```

第 1 到第 23 帧：

```text
root_z[t] = smooth_and_rate_limit(root_z_target[t])
```

平滑和限速的目的是避免楼梯边缘、boxes 边缘、rough 高程噪声导致 root_z 一帧跳变太大。

## Root 姿态模型

root yaw 仍由 `vyaw` 积分得到。roll 和 pitch 根据局部地形坡度估计。

使用预测 yaw 构造前向和左向：

```text
forward = [cos(yaw), sin(yaw)]
left = [-sin(yaw), cos(yaw)]
```

查询四个高度：

```text
h_front = height(root_xy + forward * pitch_sample_m)
h_back  = height(root_xy - forward * pitch_sample_m)
h_left  = height(root_xy + left * roll_sample_m)
h_right = height(root_xy - left * roll_sample_m)
```

估计姿态：

```text
pitch_raw = -atan2(h_front - h_back, 2 * pitch_sample_m)
roll_raw  =  atan2(h_left - h_right, 2 * roll_sample_m)
```

第 0 帧：

```text
root_roll[0] = current_roll
root_pitch[0] = current_pitch
```

第 1 到第 23 帧：

```text
root_roll/root_pitch = smooth_and_rate_limit(clamp(raw_roll_pitch))
```

roll/pitch 是弱跟随，不要求完全贴合台阶边缘。z 是主变量，roll/pitch 是辅助变量。

## 非平地速度 Soft Clamp

非 `flat` root 规划对速度做 soft clamp：

```text
vx_soft_limit = 0.5
vy_soft_limit = 0.25
vyaw_soft_limit = 0.5
```

如果速度没有超过 soft limit，保持不变。超过后只缩小超出部分：

```text
v_eff = sign(v) * (soft_limit + (abs(v) - soft_limit) * excess_scale)
```

默认：

```text
vx_excess_scale = 0.5
vy_excess_scale = 0.5
vyaw_excess_scale = 0.5
```

示例：

```text
vx = 1.0
vx_eff = 0.5 + (1.0 - 0.5) * 0.5 = 0.75
```

这样不会把高速命令硬截断到 0.5，而是让非平地速度变钝，降低 root 路径跨过台阶或 boxes 边缘时的规划失败率。

## 可调参数

新增参数建议放在 Parallelism 配置中：

```text
terrain_following_root_clearance_m = 0.30
terrain_following_root_z_smoothing = 0.35
terrain_following_root_z_rate_limit_m = 0.035
terrain_following_root_height_deadband_m = 0.005

terrain_following_pitch_sample_m = 0.20
terrain_following_roll_sample_m = 0.16
terrain_following_rpy_smoothing = 0.25
terrain_following_roll_limit_rad = 0.25
terrain_following_pitch_limit_rad = 0.35
terrain_following_rpy_rate_limit_rad = 0.03

terrain_following_vx_soft_limit = 0.5
terrain_following_vy_soft_limit = 0.25
terrain_following_vyaw_soft_limit = 0.5
terrain_following_vx_excess_scale = 0.5
terrain_following_vy_excess_scale = 0.5
terrain_following_vyaw_excess_scale = 0.5
```

常用调参方向：

```text
楼梯 root 抬不起来:
    增大 terrain_following_root_clearance_m
    增大 terrain_following_root_z_rate_limit_m

非平地姿态太激进:
    减小 terrain_following_rpy_smoothing
    减小 terrain_following_roll_limit_rad / terrain_following_pitch_limit_rad

高速容易规划失败:
    减小 vx/vy/vyaw_excess_scale

rough 上下抖:
    增大 terrain_following_root_height_deadband_m
    减小 terrain_following_root_z_smoothing
```

## 数据流

训练或 play 中，每次重规划：

```text
ParallelismReferenceManager._plan
    -> 读取 live state
    -> 从 semantic_height_scanner 读取高程图/语义图
    -> 读取 terrain_types 并映射 terrain name
    -> 按 terrain_name 选择 root 轨迹模式
    -> plan_trajectory
        -> flat: rollout_root 使用旧逻辑
        -> non-flat: rollout_root 使用 terrain-following 逻辑
        -> build_candidates
        -> IK/FK/filter/score
    -> 写入 24 帧 reference
    -> 第 0 帧覆盖为当前真实状态
```

## 文件边界

建议修改：

```text
Go2Pvcnn/extension/parallelism/config.py
Go2Pvcnn/extension/parallelism/root.py
Go2Pvcnn/extension/parallelism/planner.py
Go2Pvcnn/tracking/managers/parallelism_reference_manager.py
Go2Pvcnn/tracking/register_envs.py
Go2Pvcnn/scripts/train.py
Go2Pvcnn/scripts/play.py
```

建议新增：

```text
Go2Pvcnn/tracking/parallelism_ladder_env_cfg.py
Go2Pvcnn/tests/tracking/test_parallelism_terrain_following_root.py
Go2Pvcnn/tests/tracking/test_parallelism_ladder_env_cfg_static.py
```

配置文件名用 `parallelism_ladder_env_cfg.py`，但内容覆盖所有非 flat 地形，不只楼梯。这样符合当前分支名 `parallelism-ladder-rl`，同时不把功能限制死。

## 测试要求

静态测试：

```text
1. 新配置包含 teacher terrain 的全部 terrain names
2. flat terrain type 的 small obstacle count 为 40
3. non-flat terrain type 的 semantic obstacle count 为 0
4. register_envs 注册新 task id
5. train.py / play.py 支持新 experiment name
```

root 单元测试：

```text
1. flat mask 下 root_z 仍等于旧 stance-foot 逻辑
2. non-flat mask 下 root_z 跟随 root_xy 的 height map
3. frame 0 root_pos/root_rpy 等于当前真实状态
4. 非平地速度 soft clamp 生效
5. roll/pitch 根据前后/左右高度差变化，并受限幅约束
```

IsaacSim smoke test：

```text
1. 1024 env headless 启动
2. 至少跑 4 iteration，有 Learning iteration 输出
3. plan_valid_count 不长期为 0
4. Episode_Termination/parallelism_consecutive_standstill 不持续爆发
5. play 能选择新 experiment 并看到非 flat root reference 随地形高度变化
```

## 非目标

本阶段不做：

```text
1. 不改 foot touchdown 候选圆半径和采样方式
2. 不改 swing 曲线 z 逻辑
3. 不改 collision 几何体和 collision filter
4. 不改 score 结构
5. 不做四足组合搜索
6. 不做 root-foot 联立优化
```

## 成功标准

```text
1. flat + small obstacles 行为和当前一致
2. non-flat root_z 能随 height map 上下变化
3. slope 上 root pitch 有合理跟随
4. stairs/boxes 上 root 不再长期保持平地高度
5. RL 配置能在一个任务里覆盖 flat 小障碍和 teacher 非平地
6. 所有新增逻辑保持 torch batch 并行
```
