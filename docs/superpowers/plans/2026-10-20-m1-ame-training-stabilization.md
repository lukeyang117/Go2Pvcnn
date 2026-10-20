# M1 AME Training Stabilization Plan

## Goal

修复 M1 纯 PPO AME 训练中的探索尺度、KL 自适应学习率、腿部物理控制、步态奖励和课程 reset 回退问题，同时保持 Go2 AME 行为、地图、课程升级阈值和 M1 主分支不变。

## Architecture

- 训练入口继续使用 `m1_ame`，动作保持 16 维：12 个腿部位置动作和 4 个轮子速度动作。
- policy 继续接收完整 16 维关节状态；AME 算法、网络结构和 PPO 超参数沿用 Go2，M1 只增加必要的维度适配。
- Parallelism 不接入本次纯 PPO 修复。
- M1 专属差异通过 `m1_ame` 配置、动作适配层和 M1 reward/curriculum 文件隔离；Go2 默认路径不改变。

## Tech Stack

- Python 3.10
- PyTorch
- Isaac Lab / Isaac Sim
- `rsl_rl` PPO/AME runner
- pytest

## Global Constraints

- 不修改课程升级阈值，不改变地图和地形课程顺序。
- 恢复 Go2 的初始 reset 保护，避免 reset 帧错误触发课程降级。
- 保留轮子 `wheel_scale=8.0` 的物理接口，但限制其策略探索标准差，避免一次采样产生过大的轮速命令。
- M1 的腿部目标必须按实际关节限位裁剪，并降低腿部位置动作幅度以避免力矩饱和；轮子继续由 policy 学习。
- 只对 M1 启用按动作维度归一化 KL；Go2 保留原有 KL 计算和自适应学习率逻辑。
- 所有行为修复先有失败测试，再实现；提交前验证 Go2 回归、M1 单元测试和 IsaacLab 配置初始化。

## Tasks

### 1. Add regression tests first

- 新增 M1 稳定性测试，覆盖 16 维动作切分、腿部限位裁剪、轮子/腿部 std 分组上限、KL 维度归一化、步态 reward 非空和课程 reset guard。
- 运行目标测试确认在当前代码上失败，并记录失败原因。

### 2. Restore M1 gait shaping

- 使用 M1 四个真实足端 body selector 恢复 `feet_air_time`、`air_time_variance`、`feet_slide` 和 `fixed_gait_contact_match`。
- 沿用 Go2 AME 的 reward 权重、周期和接触阈值，不改变课程阈值。

### 3. Stabilize M1 action exploration and physical targets

- 让 AME actor-critic 支持标量或逐动作 `init_noise_std`，并保证 std 始终为正且在 M1 专属上下界内。
- M1 对腿部使用较小的 raw std 和位置动作尺度；轮子使用独立、更小的 raw std，保持轮速比例和 16 维 action contract。
- 在 M1 mixed action adapter 中对腿部 position target 按 M1 的 ABAD/hip/knee 限位裁剪，避免高噪声动作直接把关节推入极限。
- 增加可观测的 action std min/max 记录，便于 TensorBoard 诊断。

### 4. Fix M1-only KL adaptation

- 在 PPO 增加关闭状态下与旧逻辑完全一致的 KL 归一化选项。
- M1 AME 开启按 action dimension 归一化，避免从 12 维扩展到 16 维后 KL 被系统性放大并把学习率压到 `1e-5`。
- 保持 Go2 的默认配置不变，并保留 adaptive KL。

### 5. Restore curriculum reset protection

- 将 Go2 的 `_is_initial_reset` 判断和 terrain curriculum reset guard 恢复到 M1 对应文件。
- 不修改 `terrain_move_up` / `terrain_move_down` 的判定阈值或地形课程升级规则。

### 6. Verify and deliver

- 运行新增测试、M1 相关测试和 Go2 回归测试。
- 运行 M1 1 环境短启动/训练 smoke test；在资源允许时运行 1024 环境配置初始化或短收集验证。
- 检查 `m1` 主工作树保持不变，提交并推送 `m1_rl`，记录 commit 和验证结果。

## Commands

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest \
  /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl/tests \
  -q
```

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic_m1/Go2Pvcnn_m1_rl
git diff --check
git status --short --branch
```

## Commit Strategy

- `test: cover M1 AME stabilization regressions`
- `fix: stabilize M1 AME exploration and curriculum`
- 如测试和工作树检查通过，将修复提交推送到 `origin/m1_rl`。
