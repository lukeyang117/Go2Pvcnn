# Go2 平地摔倒根因分析

**日期:** 2026-03-13  
**现象:** 机器人在平地上摔倒，用户 play 时观察到  
**方法:** 系统化调试 - Phase 1 根因调查

---

## 1. 关键证据

### 1.1 用户反馈
- 机器人**在平地**摔倒（非地形难度导致）
- 通过 play 模式观察

### 1.2 与 unitree_rl_lab 对比

| 维度 | unitree_rl_lab (Go2 velocity) | teacher_semantic |
|------|-------------------------------|------------------|
| **地形** | 仅 flat（其它 sub-terrain 全部注释） | 全地形 + 楼梯/坡/盒子 |
| **Curriculum** | terrain_levels + lin_vel_cmd_levels | 空 |
| **2D 观测** | height_scanner（高度扫描） | cost_map + height_map（语义代价） |
| **训练速度** | limit_ranges（需查具体值） | lin_vel_x=(-0.2, 0.2) |
| **Play 速度** | 继承 limit_ranges | lin_vel_x=(**0.5, 1.0**) |

### 1.3 数据流追溯

**训练时 action 流程:**
```
PPO.act(obs) → distribution.sample()  # 带噪声采样！
  → wrapper: clip(-100, 100)
  → env.action_manager
  → JointPositionAction: target = default + 0.25 * action
```

**init_noise_std=1.0, noise_std_type="log"**  
→ 初始 std ≈ 1.0，每个关节 action 叠加 N(0,1) 噪声  
→ 12 个关节同时受大幅噪声 → 动作极易失稳

---

## 2. 根因假设（按优先级）

### 假设 A：Play 与训练的命令分布严重不符（高置信度）

**证据:**
- `CommandsCfg`（训练）：`lin_vel_x=(-0.2, 0.2)`, `lin_vel_y=(-0.2, 0.2)`, 25% standing
- `CommandsCfg_PLAY`（play）：`lin_vel_x=(0.5, 1.0)`, `lin_vel_y=0`, `ang_vel_z=0`

**含义:**
- 训练时策略从未见过 0.5–1.0 m/s 的纯前向指令
- 测试时指令分布完全偏移（OOD）
- 策略对“大步幅前向行走”未建立有效映射 → 易失控、易倒地

**验证:** 用训练分布（如 lin_vel_x=(-0.2, 0.2)）在 play 里测试，观察是否仍摔倒。

---

### 假设 B：训练早期 action 噪声过大（高置信度）

**证据:**
- `ActorCriticCNN.act()` 返回 `self.distribution.sample()`，而非 `distribution.mean`
- `init_noise_std=1.0`，log_std 初始化为 `log(1.0) ≈ 0`
- 采样结果 = mean + N(0, std)，std 初始 ~1.0
- action scale=0.25 → 关节 delta = 0.25 × (mean + noise)
- 噪声 ±1 时，单关节 delta ≈ ±0.25 rad，多关节叠加会非常剧烈

**含义:**
- 训练早期，即使策略 mean 合理，噪声也会产生极大扰动
- 在平地上，这类动作足以打破平衡 → base 触地 → base_contact
- 早期 episode 短、base_contact 多，可能与噪声关系更大，而非地形

**验证:** 将 `init_noise_std` 降到 0.3 重训，观察早期 base_contact 是否显著下降。

---

### 假设 C：观测与任务不匹配（中置信度）

**证据:**
- unitree 用 `height_scanner`（2D 高度）
- 我们用 `teacher_semantic_cost_map`（语义代价 + command 对齐）
- cost map 含 obstacle、distance、gradient、height、command_bonus 等
- 在平地、障碍少时，cost map 以低代价为主，结构简单
- 策略可能过度依赖复杂 cost map，而对“平地直走”的 proprio 信息利用不足

**含义:**
- 在平地这种简单场景，高度图更贴近 locomotion；cost map 可能带来干扰
- 若 cost map 与 proprio 冲突，策略可能学到错误映射
- 但 unitree 的 height_scanner 在 CriticCfg 中被注释，说明他们也可能主要靠 proprio

**验证:** 临时移除 cost map，仅用 state，在平地上测试基础 locomotion 是否明显改善。

---

### 假设 D：缺少 curriculum 导致难度不匹配（中置信度）

**证据:**
- unitree 有 `terrain_levels_vel`、`lin_vel_cmd_levels`
- teacher_semantic 的 `CurriculumCfg` 为空
- 我们一开始就暴露在全地形 + 高指令范围下

**含义:**
- 若有 curriculum，失败 env 可回到更简单地形/更慢指令
- 无 curriculum 时，一旦开始跌倒，会持续在高难度上失败，难以“退回简单模式”学习
- 这主要解释“难以收敛”，对“平地也摔”是次要因素

---

### 假设 E：YCB 物体与 cost map 干扰（低置信度）

**证据:**
- 每 env 9 个 YCB 物体，reset 时 x,y ∈ (-3, 3)
- cost map 会标记这些障碍
- 为避开障碍，策略可能产生急转、急停等不稳定动作

**含义:**
- 在平地上，若物体靠近，策略为避障可能牺牲稳定性
- 但 base_contact 更多是整体倾倒，不一定是足端碰物体，该假设需进一步对照实验

---

## 3. 训练过程现象解释（修正版）

| 阶段 | time_out | base_contact | 可能原因 |
|------|-----------|--------------|----------|
| 初期 | 高 | 低 | 随机策略偏保守；或部分 env 在较简单地形；噪声导致动作幅度有限 |
| 中期 | 降 | 升 | 策略为获得 track_lin_vel 开始主动运动；噪声仍大；动作更激进 → 更易倒地 |
| 后期 | ~0 | 高 | 策略陷入“尝试走→倒地”的循环；无 curriculum 难以恢复；base_contact  predominates |

**核心：**  
- 地形难度不是主因（用户确认在平地也摔）
- 更可能是：  
  1）训练时的 action 噪声；  
  2）Play 时的命令 OOD；  
  3）观测/任务设计（cost map vs height）的次优匹配。

---

## 4. 建议验证步骤（不直接改策略，先确认根因）

1. **验证假设 A（命令 OOD）**
   - 修改 `CommandsCfg_PLAY`，令 `lin_vel_x=(0.1, 0.2)` 与训练分布一致
   - 在 play 中观察是否仍频繁倒地

2. **验证假设 B（action 噪声）**
   - 将 `init_noise_std` 从 1.0 降至 0.3
   - 重新训练，对比早期 `Episode_Termination/base_contact` 与 mean episode length

3. **验证假设 C（观测影响）**
   - 暂时用零/常数 cost map 替代真实 cost map
   - 在平地上测试 locomotion 是否稳定

---

## 5. 结论

- 平地摔倒的主要可疑原因：
  1. **Play 命令 OOD**：0.5–1.0 m/s 前向 vs 训练 -0.2–0.2
  2. **训练早期 action 噪声过大**：init_noise_std=1.0 导致动作扰动过大
- 次要因素：观测设计（cost map vs height）、缺少 curriculum、YCB 物体分布
- 建议：先完成上述验证，再根据结果决定是否改网络、curriculum 或观测结构。
