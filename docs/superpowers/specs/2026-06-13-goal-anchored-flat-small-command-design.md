# Flat-Small Goal-Anchored Body Velocity Command Design

## 背景

当前 flat-small avoidance 训练希望从旧模型继续加载。旧模型的 `base_velocity` 语义来自 IsaacLab `UniformVelocityCommand`：命令是机器人 root/body 坐标系下的 `[vx_body, vy_body, yaw_rate]`，并且现有 reward、observation、MPC 都直接消费这个合同。

因此新方案不能把 policy 输入改成世界系速度。目标是只改变 flat-small 训练的 command 生成方式，让机器人在世界里朝一个远方向持续走远，同时仍然给旧策略熟悉的 body-frame velocity command。

## 目标

- 只影响 `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`。
- 新增一个 flat-small 专用 command term，但 command 名字仍为 `base_velocity`。
- policy 观测、tracking reward、MPC manager 继续读取同一个 body-frame command tensor。
- reset 时为每个 env 采样一个世界目标方向和固定 x/y 速度幅值。
- 每个 step 根据目标方向在当前 root 坐标系下的象限更新 `vx_body/vy_body` 正负，并根据 heading error 更新 yaw。
- 不把任务变成导航任务；目标点只是世界方向锚点，用来减少局部打转并增加走远距离。

## 非目标

- 不改 observation shape。
- 不改 action shape。
- 不改 PPO checkpoint 兼容性。
- 不改 `track_lin_vel_xy` / `track_ang_vel_z` reward 语义。
- 不改 MPC manager 的 command 读取接口。
- 不影响 baseline `teacher_elevation_trajectory_mpc_semantic` 的普通 velocity command 和速度 curriculum。

## 推荐方案

新增 `GoalAnchoredVelocityCommand` 和对应 config，例如：

```text
GoalAnchoredVelocityCommandCfg
```

它继承 IsaacLab command term 的行为边界，输出属性仍是：

```text
command: Tensor[num_envs, 3]
```

语义仍是：

```text
[vx_body, vy_body, yaw_rate]
```

flat-small cfg 中把：

```text
commands.base_velocity = UniformLevelVelocityCommandCfg(...)
```

替换为：

```text
commands.base_velocity = GoalAnchoredVelocityCommandCfg(...)
```

这样所有下游消费者都不用改，因为它们按名字 `base_velocity` 读取命令。

## 数据流

### Reset / Resample

对每个 reset env：

```text
root_xy = robot.data.root_pos_w[:, :2]
root_yaw = robot.data.heading_w

theta_world ~ Uniform(-pi, pi)
goal_xy = root_xy + goal_distance * [cos(theta_world), sin(theta_world)]

vx_abs ~ Uniform(vx_abs_range[0], vx_abs_range[1])
vy_abs ~ Uniform(vy_abs_range[0], vy_abs_range[1])
```

推荐初始参数：

```text
goal_distance = 10.0
goal_reached_threshold = 1.0
vx_abs_range = (0.6, 1.0)
vy_abs_range = (0.6, 1.0)
yaw_stiffness = 0.5
yaw_range = (-0.8, 0.8)
rel_standing_envs = 0.0
resampling_time_range = (100.0, 100.0)
```

`vx_abs` 和 `vy_abs` 在一个 episode 内保持不变。

### Per-Step Update

每个 step 根据当前 root pose 更新命令：

```text
dir_world = normalize(goal_xy - root_xy)
dir_body = rotate_world_to_body(dir_world, root_yaw)

vx_body = sign(dir_body.x) * vx_abs
vy_body = sign(dir_body.y) * vy_abs

heading_error = wrap_to_pi(atan2(dir_world.y, dir_world.x) - root_yaw)
yaw_rate = clamp(yaw_stiffness * heading_error, yaw_range[0], yaw_range[1])
```

如果 `goal_xy` 距离当前 root 小于 `goal_reached_threshold`，沿当前目标方向再续一个 `goal_distance`，保持持续走远：

```text
goal_xy = root_xy + goal_distance * dir_world
```

### Reward 对齐

奖励不改：

```text
track_lin_vel_xy:
  compare command[:, :2] with root_lin_vel_b[:, :2]

track_ang_vel_z:
  compare command[:, 2] with root_ang_vel_b[:, 2]
```

这保证旧 checkpoint 仍然看到熟悉的 body-frame velocity tracking 任务。

## 为什么不直接改成世界系速度

旧策略没有学过世界系速度命令。若直接把 command 解释成 world-frame，policy 输入维度不变但语义变了，旧模型会把世界速度误当 body 速度，继续训练初期会非常混乱。

本方案只在 command generator 内部使用世界目标方向；对 policy、reward、MPC 暴露的仍是 body-frame command。

## 为什么用硬象限而不是连续比例

用户希望 x/y 速度大小在 reset 后固定，不希望运动速度随目标角度比例变小。

硬象限规则满足：

```text
abs(vx_body) = vx_abs
abs(vy_body) = vy_abs
```

每个 step 只根据目标方向落在当前 root 坐标系的哪个象限来改变正负号。这样速度幅值稳定，走远能力更强。

连续比例方案更平滑，但会在目标接近前方时让 `vy_body` 变小，不符合本轮训练意图。

## 兼容性

保持不变：

- `command_manager.get_command("base_velocity")`
- policy observation 中的 `velocity_commands`
- `track_lin_vel_xy_exp`
- `track_ang_vel_z_exp`
- MPC manager `_commands_from_env()`
- checkpoint observation/action shape

只变更：

- flat-small cfg 的 `base_velocity` command term class。
- flat-small saved cfg 中的 command 参数。

## 测试计划

### 静态测试

- flat-small cfg 使用 `GoalAnchoredVelocityCommandCfg`。
- baseline semantic cfg 仍使用 `UniformLevelVelocityCommandCfg`。
- command 名字仍是 `base_velocity`。
- flat-small reward、obs、MPC command name 不变。

### 单元测试

构造 fake robot/root pose，验证：

- reset 采样 `goal_xy`、`vx_abs`、`vy_abs`。
- `vx_abs/vy_abs` 在非 reset step 不变。
- root yaw 变化后，目标方向投影到 body 坐标系，`vx/vy` 正负随象限变化。
- `yaw_rate` 按 heading error 和 yaw range clamp。
- 到达目标后 goal 会向同方向续 10m。

### 真实 smoke

使用 `env_isaacsim` 跑小环境：

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

检查：

- Command Manager 仍只有 `base_velocity`。
- `base_velocity` shape 为 `[num_envs, 3]`。
- `abs(vx_body)` 和 `abs(vy_body)` 在 `[0.6, 1.0]`。
- `yaw_rate` 在 `[-0.8, 0.8]`。
- Reward Manager 和 Curriculum Manager 正常启动。

## 风险

- 硬象限规则会让 `vx/vy` 在坐标轴附近跳变。当前设计接受这个风险，因为用户明确更关心速度大小稳定。
- `vy_body` 可能比旧 flat-small 训练更大，短期可能降低稳定性；但这是有意为之，用于增强横向/斜向移动和走远能力。
- yaw 上限过大可能导致转向占主导；先用 `0.8 rad/s`，必要时再从 TensorBoard 和 rollout 里调。

## 成功标准

- 加载旧 checkpoint 后训练不因 command 语义改变而崩坏。
- `mean_episode_length` 维持稳定。
- `mean_terrain_level` 不因走不远而塌到 0。
- controlled crossing / first-layer eval 中 path obstacle opportunity 增加。
- 后续 TensorBoard 中 semantic contact 不再只靠随机碰撞，训练有更稳定的走远与跨越机会。
