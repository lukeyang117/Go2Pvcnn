# M1 RL 状态爆炸与奖励隔离设计

## 已复现根因

在 `model_12300.pt`、seed 42、1024 环境下复现到 env 978、terrain type 19（`pyramid_stairs_inv`）、level 0。回合从 `episode_length=0` 开始、command 为零速，因此 NaN 与随机 episode 长度无关。第 6 控制步 RBL HIP/KNEE 速度达到约 `108/-119 rad/s`、驱动力矩达到 `±150 N·m`；第 7 步 RBL ABAD/HIP/KNEE 达到约 `3710/1209/1431 rad/s`；第 8 步关节状态升到 `1e16–1e21` 并产生 `-inf` 加速度，第 9 步出现 NaN。

相同 checkpoint 和 seed 的 1024 环境消融测试中，仅将 normalized policy action 限制到 `[-1, 1]`，运行 120 步没有非有限状态，RBL 短暂峰值后稳定在约 `2.1 rad/s`。M1 USD 为腿关节提供 `150 N·m` 力矩上限，但 `maxJointVelocity` 实际接近 float32 上限；当前资产给所有关节使用同一 `stiffness=25, damping=0.5` actuator，未设置 PhysX velocity limit。

## 目标与范围

- 限制策略输出到 mixed-action 所定义的 normalized 区间，防止极端均值动作持续将腿关节推向限位。
- 给 M1 腿和轮子配置不同的 actuator 物理上限；不改 Go2 资产或共享 actor-critic 的默认行为。
- 增加 M1 非有限/超速状态 termination，并确保本步奖励保持有限，坏状态不会进入 episode reward 和 PPO rollout。
- 修复 M1 `joint-position` 奖励忽略 `joint_ids`、错误惩罚轮子角度的问题。
- 课程升级/降级阈值、地图、terrain schedule、16 维 observation/action contract 保持不变。

## 设计

### Action 边界

M1 mixed action 在输入动作一进入 action term 时逐维裁剪到 `[-1,1]`。腿部仍使用 position control 和现有 `leg_scale=0.15`，wheel 仍使用 velocity control 和 `wheel_scale=8`。腿部 position target 再做逐控制步变化率限制，初始/复位 target 设为当前默认腿姿，避免零 target 突跳。leg joint limit 的绝对裁剪继续保留。

M1 policy 的 std 使用 per-action 上界：腿部允许现有 M1 探索范围；轮子 std 上界固定为 `0.10`，不允许所有 16 维共同涨至 `0.40`。Go2 scalar std 配置不变。

### M1 专属 actuators

- Leg joints：implicit position drive，effort limit 沿用 USD 的 `150 N·m`；配置有限 simulation velocity limit `20 rad/s` 作为仿真保护上限，不声称是硬件规格。
- Wheel joints：纯 velocity drive（`stiffness=0`），effort limit 沿用 USD 的 `50 N·m`，velocity limit 与 M1 命令上限一致为 `8 rad/s`。
- 只在 `go2_pvcnn/assets/m1.py` 改 M1 actuator groups；不动 Go2 资产。

### 非有限状态保护

- M1 终止条件检查 joint position/velocity/acceleration/applied torque、root/body velocity 是否有限，以及 leg/wheel joint velocity 是否超过各自保护上限。
- M1 state-based reward wrappers 在计算平方、乘法、限位距离前将输入限定在其有效物理范围，始终产生有限数；invalid-state termination 负责立即 reset，配套终止统计显式记录原因。不是只把最终 episode NaN 静默吞掉。
- TensorBoard 增加 invalid-state termination 及最大腿/轮关节速度指标。

### 奖励关节选择器

- 新增 M1 专用 `joint_position_penalty`，只读取 `M1_LEG_JOINT_CFG.joint_ids`。
- 修复 M1 的 `joint_vel` term，使其确实只惩罚腿部；joint acceleration/torque/energy 保持当前 16 维配置。
- Go2 reward helpers 保持不变。

## 验证要求

1. 单元测试覆盖动作裁剪、腿目标变化率、wheel joint 不进入 joint-position reward、非有限与超速 detector、每个 M1 reward 在坏状态下仍有限。
2. 同 checkpoint/seed/1024-env replay 运行至少 120 control steps；要求没有非有限状态，所有腿速度不超过 20 rad/s、轮速不超过 8 rad/s。
3. Isaac Lab 1-env 和 1024-env 各运行 1 PPO iteration。
4. M1/课程/Go2 回归测试通过；不改课程阈值及 Go2 默认配置。
