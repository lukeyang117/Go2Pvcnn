# Go2 SemLoco 语义感知足端规划对比实验设计

## 目标与范围

本实验复现 Watch Your Step / SemLoco 的语义感知 Raibert 足端规划思想，作为 Go2 的独立 PPO 对比基线。策略仍由现有 `ActorCriticCNN`（CNN + MLP）输出动作，规划器不直接修改动作；规划器生成的足端落点仅通过新增奖励影响 PPO。

本次采用单阶段真实刚体障碍训练，不采用论文中的 Virtual Obstacle -> Rigid Obstacle 两阶段训练。不加入 AME，不复用或修改 AMP、Distillation、Teacher、AME 的训练路径，也不改变现有 `cross_large_complex_ppo` 配置和网络。

实验名称为 `cross_large_complex_semloco`，日志写入 `logs/rsl_rl/cross_large_complex_semloco/`。

## 现有配置与隔离边界

- 复用当前 Go2 的 observation、语义地图/高程图输入、CNN+MLP 网络和 PPO 超参数。
- 新增独立 `semloco/` 包、环境配置、训练入口和 play 入口。
- 现有 PPO、AMP、Distillation、Teacher、AME 文件不做行为修改。
- planner 与 reward 只通过环境实例上的明确缓存接口通信；不使用全局变量。
- 语义类别沿用工程定义：`0=terrain`、`1=small obstacle`、`2=large obstacle`。

## 模块设计

### `semloco/semantic_raibert_planner.py`

提供无副作用、可批量执行的 `SemanticRaibertPlanner`：

```python
plan(
    root_pos_w: Tensor,          # [N, 3]
    root_quat_w: Tensor,         # [N, 4]
    root_lin_vel_b: Tensor,      # [N, 3]
    root_ang_vel_b: Tensor,      # [N, 3]
    foot_pos_w: Tensor,          # [N, 4, 3]
    velocity_command: Tensor,    # [N, 3], vx/vy/vyaw
    gait_phase: Tensor,          # [N, 4], [0, 1]
    semantic_height_scanner: Tensor,
) -> PlannerOutput
```

`PlannerOutput` 包含：

```python
target_foothold_w: Tensor        # [N, 4, 3]
nominal_foothold_w: Tensor       # [N, 4, 3]
valid: Tensor                    # [N, 4] bool
collision_free: Tensor           # [N, 4] bool
```

Raibert 目标以名义落点为基准：

```text
ΔP_linear = [T_stance/2 * vx, T_stance/2 * vy, 0]
ΔP_yaw_i  = [0, T_stance/2 * vyaw * x_nominal_i, 0]
P_raibert_i = P_nominal_i + ΔP_linear + ΔP_yaw_i
```

上述增量在机体坐标中计算，再用 `root_quat_w` 转换到世界系；`x_nominal_i` 是第 `i` 条腿的名义前后位置。四足 trot 相位为 FL/RR 同相、FR/RL 反相。每条腿进入 swing 时才重新选择目标，目标在 touchdown 前保持不变。

每个目标周围搜索 `5 x 5` 个候选点，偏移为 `[-0.08, -0.04, 0, 0.04, 0.08] m`。候选点使用语义高度图插值，并以 `foot_radius=0.04 m` 做 AABB 膨胀碰撞检测。代价为：

```text
J = 1.0 * max(dx, 0) + 0.5 * max(-dx, 0)
    + 8.0 * abs(dy) + 100.0 * collision
```

优先级为碰撞约束、横向偏移、前向偏移。无可行候选时，返回名义点、`valid=False`，不触发 episode termination。

### `semloco/semloco_rewards.py`

提供两个纯函数，输入环境缓存和当前状态，输出 `[N]` reward：

1. `semantic_foothold_tracking_reward`
   - 仅对 swing 中且 `valid=True` 的腿计算；
   - `exp(-||foot-target||^2 / sigma_foot)`，四腿求和后按有效腿数归一化；
   - `sigma_foot=0.10`；无有效腿时严格返回零。
2. `semloco_clearance_penalty`
   - 对所有 swing 腿计算，不依赖 planner 是否有效；
   - 参考高度 `z_ref = swing_height * sqrt(sin(pi * phase)) + 0.02`；
   - 使用单边 ReLU：`-sum(max(0, z_ref-z_foot)^2)`；
   - `swing_height=0.08 m`。

奖励权重作为独立环境配置字段，可在实验中记录和调整；默认 foothold tracking 与 clearance 使用正向 tracking/负向 penalty 的语义，不覆盖基础运动奖励。

### `semloco/semloco_env_cfg.py`

定义 `SemlocoCrossLargeComplexEnvCfg`，从当前 cross-large-complex 配置复制必要的 robot、scene、observation、reward 和 termination 设置，并只追加：

- `semantic_raibert_planner` 配置与缓存初始化；
- `semloco_foothold_tracking` reward；
- `semloco_clearance` penalty；
- planner 更新回调，挂在物理 step 后、reward 计算前；
- 不改变原有 action/observation shape。

规划缓存按环境维度保存：`_semloco_target_foothold_w [N,4,3]`、`_semloco_target_valid [N,4]`、`_semloco_swing_mask [N,4]`。环境 reset 时清空缓存；新 swing 开始且上一目标已 touchdown 后才写入对应腿。

### 训练与运行入口

- `semloco/semloco_train_cfg.py`：独立实验名和 runner 配置，其他 PPO 参数与基础 PPO 对齐。
- `semloco/semloco_runner.py`：注册环境和配置，保持与现有 train 入口相同的 CLI/checkpoint 约定。
- `scripts/train_cross_large_complex_semloco.py` 与 headless shell：支持从零训练和 `--resume`，不修改现有 shell。
- `scripts/play_cross_large_complex_semloco.py`：复用现有 play 的 checkpoint 加载和可视化参数。

## 每步数据流与时间对齐

在 `env.step(action_t)` 中：

1. policy 使用时刻 `t` observation 产生 `action_t`；
2. 仿真推进到 `t+1`，更新 root、足端、接触和 scanner；
3. 根据 `t+1` gait phase 检测每条腿的 swing 起点；
4. 仅对新进入 swing 的腿运行 planner 并锁定目标；
5. 计算 foothold tracking 与 clearance；
6. 与基础 reward 相加，写入 PPO transition `t` 对应的 reward；
7. 将 `t+1` observation 和缓存状态交给下一次 policy 调用。

因此动作与奖励仍遵循现有 PPO 的一步延迟约定，规划目标不会在一个 swing 内抖动。规划结果跨越 PPO rollout 边界持续保存在环境缓存中，不区分“上一次/下一次规划”；reset 或 touchdown 后才替换。

## 失败、无效与边界行为

- planner 无效：对应腿 foothold tracking 为 0；clearance penalty、基础奖励和 episode 继续运行。
- scanner 数据缺失或维度不符：planner 返回全腿 invalid，并记录一次受控 warning，不抛出训练中断异常。
- reset：所有目标置零、valid 置 false，首个 swing 重新规划。
- 数值保护：候选代价、距离和 reward 使用有限值检查；NaN/Inf 候选视为不可行。
- 不允许 planner 直接改 action、关节目标或 termination。

## 测试与验收

### 单元测试

- Raibert 前向/横向/yaw 修正方向和四腿相位映射；
- 5x5 候选搜索在无障碍时选择代价最低点；
- 语义障碍膨胀后候选被正确拒绝；
- 无可行点返回 invalid 且不改变输入；
- swing 起点锁定目标，swing 中途不重规划，touchdown 后允许更新；
- foothold reward 的 valid mask、归一化和无有效腿返回零；
- clearance penalty 的 ReLU 单边性质及 planner invalid 时仍计算；
- planner 输出批量 shape、device 和 finite 约束。

### 真实 IsaacLab smoke

使用独立 SemLoco 入口启动 `num_envs=1024`、headless、`max_iterations=4`：

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn
./scripts/train_cross_large_complex_semloco_headless.sh \
  --num-envs 1024 --max-iterations 4 --headless
```

验收条件：进程正常退出；无 NaN/Inf；observation/action/reward shape 不变；日志出现 SemLoco 两项 reward；现有 PPO/AMP/Distillation/AME 静态测试继续通过。

## 不在本次范围内

- Virtual Obstacle 预训练阶段；
- AME、AMP 或 Distillation 融合；
- 新的神经网络结构或 observation 通道；
- planner 直接控制动作；
- 重新设计 terrain 生成器或碰撞语义标签。
