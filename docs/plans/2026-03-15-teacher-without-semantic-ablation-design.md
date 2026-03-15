# Teacher Without Semantic  ablation 设计

**日期:** 2026-03-15  
**目的:** 验证高程图/CNN 输入是否导致机器狗训练后无法行走

---

## 1. 设计概览

最小化修改现有代码，增加 state-only 对照实验：

1. 将 `train_cfg` 从 `train.py` 拆分到 `Go2Pvcnn/agent`
2. 新建 `teacher_without_semantic_env_cfg.py`（仅 state，无 CNN）
3. 通过 `--experiment` 切换 teacher_semantic / teacher_without_semantic

---

## 2. 文件变更

### 2.1 新增

| 路径 | 说明 |
|------|------|
| `Go2Pvcnn/agent/__init__.py` | 导出 `get_train_cfg` |
| `Go2Pvcnn/agent/train_cfg.py` | 训练配置（按 experiment 返回 dict） |
| `Go2Pvcnn/go2_pvcnn/tasks/teacher_without_semantic_env_cfg.py` | 仅 state 的环境配置 |

### 2.2 修改

| 路径 | 改动 |
|------|------|
| `train.py` | 增加 `--experiment`；`get_train_cfg()`；按 experiment 选 env |
| `register_envs.py` | 注册 `Isaac-Teacher-Without-Semantic-Go2-v0` |

---

## 3. teacher_without_semantic 环境

- **继承:** `TeacherSemanticEnvCfg`
- **覆盖:** `observations` → `ObservationsCfg_StateOnly`
- **观测:** 仅 policy/critic state（base_ang_vel, projected_gravity, joint_pos/vel, velocity_commands, actions）
- **移除:** policy_cost_map, critic_cost_map（无 CNN 输入）

---

## 4. 策略与 obs_groups

| experiment | policy 类 | obs_groups |
|------------|-----------|------------|
| teacher_semantic | ActorCriticCNN | policy: [policy_cost_map, policy_state], critic: [critic_cost_map, critic_state] |
| teacher_without_semantic | ActorCritic | policy: [policy], critic: [critic] |

---

## 5. 使用方式

```bash
# 原始 teacher_semantic（默认）
python train.py --num_envs 256 --headless

# 仅 state 的 ablation
python train.py --experiment teacher_without_semantic --num_envs 256 --headless
```

---

## 6. 验证假设

若 `teacher_without_semantic` 能学会行走，而 `teacher_semantic` 不能，则说明：

- 高程图 / cost map / CNN 输入可能导致问题
- 或观测/任务设计存在不匹配
