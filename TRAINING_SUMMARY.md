# Go2 PVCNN-PPO 训练系统总结

## 📋 系统概述

成功实现了基于PVCNN语义分割的Go2四足机器人导航训练系统，使用PPO算法训练带有视觉感知的策略网络。

---

## 🔧 核心修改清单

### 1. PVCNN模型适配（3通道输入 + 4类别输出）

**文件**: `Go2Pvcnn/go2_pvcnn/pvcnn_wrapper.py`

#### 输入通道适配（9→3）
- **原始**: S3DIS训练时使用9通道（XYZ + RGB + Normals）
- **修改后**: Go2只用3通道（XYZ坐标）
- **实现**:
  ```python
  # 第一层卷积权重裁剪
  if 'point_features.0' in key and 'weight' in key:
      if original_weight.shape[1] == 9:
          new_weight = original_weight[:, :3, ...]  # 只保留XYZ通道
  ```

#### 输出类别适配（13→4）
- **原始**: S3DIS有13个室内场景类别
- **修改后**: Go2环境4个类别
  - **Class 0**: Terrain（地形，可通行）
  - **Class 1**: CrackerBox（Object_0，障碍物）
  - **Class 2**: SugarBox（Object_1，障碍物）
  - **Class 3**: TomatoSoupCan（Object_2，障碍物）
- **实现**:
  ```python
  # 分类器输出层权重裁剪
  if 'classifier' in key and original_param.shape[0] == 13:
      new_param = original_param[:4, ...]  # 只保留前4个类别
  ```

---

### 2. 点云预处理优化

**文件**: `Go2Pvcnn/go2_pvcnn/mdp/observations.py`

#### 无效点过滤
```python
# 过滤Inf/NaN/零点
valid_mask = ~(torch.isinf(point_cloud).any(dim=-1) | 
               torch.isnan(point_cloud).any(dim=-1) |
               (point_cloud.abs().sum(dim=-1) < 1e-6))
valid_points = point_cloud[valid_mask]
```

#### 智能采样到2046点
- **点太多**: 使用FPS（最远点采样）
  ```python
  from pytorch3d.ops import sample_farthest_points
  sampled_points, _ = sample_farthest_points(valid_points, K=2046)
  ```
- **点太少**: 复制点到目标数量
  ```python
  num_repeats = (2046 + num_valid - 1) // num_valid
  point_cloud = valid_points.repeat(num_repeats, 1)[:2046]
  ```

---

### 3. Cost Map生成

**文件**: `Go2Pvcnn/go2_pvcnn/mdp/cost_map.py`

#### 3通道代价地图（64×64网格）
```python
class CostMapGenerator:
    def generate_cost_map(point_xyz, semantic_logits, semantic_confidence):
        # Channel 0: 距离代价（到最近障碍物）
        distance_cost = compute_distance_cost(obstacle_map)
        
        # Channel 1: 高度梯度代价（地形陡峭度）
        gradient_cost = compute_gradient_cost(height_map)
        
        # Channel 2: 语义置信度代价（1 - confidence）
        confidence_cost = 1.0 - confidence_map
        
        # 堆叠成(batch, 3, 64, 64)
        cost_map = torch.stack([distance_cost, gradient_cost, confidence_cost], dim=1)
        
        # 展平为(batch, 12288)以便拼接到观测向量
        return cost_map.view(batch, -1)
```

---

### 4. ActorCriticCNN网络

**文件**: `Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic_cnn.py`

#### 网络架构
```python
class ActorCriticCNN(nn.Module):
    def __init__(self, num_obs, num_privileged_obs, num_actions, ...):
        # CNN编码器（处理cost_map）
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),    # (3,64,64) -> (32,30,30)
            nn.Conv2d(32, 64, kernel_size=3, stride=2),   # (32,30,30) -> (64,14,14)
            nn.Conv2d(64, 128, kernel_size=3, stride=2),  # (64,14,14) -> (128,6,6)
            Flatten(),
            nn.Linear(128*6*6, 256)                       # (4608) -> (256)
        )
        
        # Actor: CNN特征(256) + Proprio(48) -> Actions
        self.actor = nn.Sequential(
            nn.Linear(256 + 48, 256),
            nn.Linear(256, num_actions)
        )
        
        # Critic: CNN特征(256) + Proprio(48) -> Value
        self.critic = nn.Sequential(
            nn.Linear(256 + 48, 256),
            nn.Linear(256, 1)
        )
```

#### 前向传播流程
```python
def _extract_features(self, observations):
    # 1. 分离观测
    cost_map_flat = observations[:, -12288:]  # 最后12288维
    proprio = observations[:, :-12288]         # 前面的proprio观测
    
    # 2. Reshape cost_map
    cost_map_2d = cost_map_flat.view(-1, 3, 64, 64)
    
    # 3. CNN编码
    cnn_features = self.cnn_encoder(cost_map_2d)  # (batch, 256)
    
    # 4. 拼接特征
    combined = torch.cat([cnn_features, proprio], dim=1)
    return combined
```

---

### 5. PPO算法集成PVCNN

**文件**: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py`

#### 参数扩展
```python
class PPO:
    def __init__(self, ..., pvcnn_model=None, lambda_seg=0.1):
        self.pvcnn_model = pvcnn_model      # PVCNN模型（用于多任务学习）
        self.lambda_seg = lambda_seg         # 语义分割损失权重
```

#### 优化器包含PVCNN参数
```python
# 在train_go2_pvcnn.py中
runner.alg.pvcnn_model = pvcnn_wrapper.model
params = list(runner.alg.actor_critic.parameters()) + \
         list(pvcnn_wrapper.model.parameters())
runner.alg.optimizer = Adam(params, lr=learning_rate)
```

---

### 6. 环境观测配置

**文件**: `Go2Pvcnn/go2_pvcnn/tasks/go2_pvcnn_env_cfg.py`

#### Policy观测组（Actor输入）
```python
class PolicyCfg(ObsGroup):
    # PVCNN特征 (2046*4=8184维，展平)
    pvcnn_features = ObsTerm(func=custom_mdp.pvcnn_features)
    
    # 地形高度扫描 (11*17=187维)
    height_scan = ObsTerm(func=isaac_mdp.height_scan)
    
    # 本体感知 (48维)
    base_lin_vel = ObsTerm(...)      # 3维
    base_ang_vel = ObsTerm(...)      # 3维
    projected_gravity = ObsTerm(...) # 3维
    joint_pos = ObsTerm(...)         # 12维
    joint_vel = ObsTerm(...)         # 12维
    velocity_commands = ObsTerm(...) # 3维
    actions = ObsTerm(...)           # 12维
    
    # Cost Map (12288维=3*64*64，展平)
    cost_map = ObsTerm(func=custom_mdp.cost_map_from_lidar)
```

#### Critic观测组（特权信息）
```python
class CriticCfg(ObsGroup):
    # 与Policy相同的所有观测
    # 可以添加额外的特权信息（如真实物体位置等）
```

---

## 🔄 完整训练流程

### 阶段1: 初始化（Initialization）

```
1. 加载环境配置
   └─> Go2PvcnnEnvCfg
       ├─> Scene配置（机器人、地形、动态物体）
       ├─> Observations配置（policy组 + critic组）
       ├─> Rewards配置
       └─> Terminations配置

2. 创建PVCNN Wrapper
   └─> create_pvcnn_wrapper()
       ├─> 加载预训练checkpoint
       ├─> 适配输入通道（9→3）
       ├─> 适配输出类别（13→4）
       └─> 冻结为eval模式

3. 创建Isaac Lab环境
   └─> gym.make("Go2PvcnnEnv")
       ├─> 初始化场景（地形、机器人、物体）
       ├─> 初始化传感器（LiDAR、高度扫描）
       └─> 初始化观测/奖励/终止管理器

4. 包装环境
   └─> RslRlPvcnnEnvWrapper(env, pvcnn_wrapper)
       ├─> 注入pvcnn_wrapper到env.unwrapped
       ├─> 第一次reset（计算初始观测）
       └─> 验证观测维度
```

### 阶段2: 创建训练代理（Create Agent）

```
5. 配置PPO参数
   agent_cfg = {
       "policy": {
           "class_name": "ActorCriticCNN",
           "cnn_channels": [32, 64, 128],
           "cnn_feature_dim": 256,
           ...
       },
       "algorithm": {
           "learning_rate": 1e-3,
           "lambda_seg": 0.1,
           ...
       }
   }

6. 创建RSL-RL Runner
   └─> OnPolicyRunner(env, agent_cfg)
       ├─> 创建ActorCriticCNN网络
       ├─> 创建PPO算法
       └─> 创建RolloutStorage

7. 注入PVCNN到PPO
   └─> runner.alg.pvcnn_model = pvcnn_wrapper.model
       ├─> 重建optimizer（包含PVCNN参数）
       └─> 设置lambda_seg权重
```

### 阶段3: 训练循环（Training Loop）

```
对于每个iteration:
    
    ┌─────────────────────────────────────────┐
    │ 8. 数据收集 (Rollout)                  │
    └─────────────────────────────────────────┘
    对于每个step (num_steps_per_env=24):
        
        a) 获取观测 (Observation)
           └─> env.step() 调用observation_manager
               ├─> pvcnn_features():
               │   ├─> 从LiDAR获取点云
               │   ├─> 过滤无效点
               │   ├─> 采样到2046点
               │   ├─> 通过PVCNN前向传播
               │   └─> 返回展平的语义特征 (8184维)
               │
               ├─> cost_map_from_lidar():
               │   ├─> 从LiDAR获取点云
               │   ├─> 通过PVCNN获取语义logits
               │   ├─> 生成3通道cost_map (64×64)
               │   └─> 展平为向量 (12288维)
               │
               └─> 其他观测（height_scan, proprio等）
        
        b) 选择动作 (Action Selection)
           └─> actor_critic.act(obs)
               ├─> _extract_features():
               │   ├─> 分离cost_map和proprio
               │   ├─> Reshape cost_map为(3,64,64)
               │   ├─> CNN编码 -> (256维)
               │   └─> 拼接CNN特征+proprio
               │
               ├─> actor(features) -> mean
               ├─> 从N(mean, std)采样动作
               └─> 返回动作 + log_prob
        
        c) 执行动作 (Execute Action)
           └─> env.step(action)
               ├─> 物理仿真步进
               ├─> 计算奖励
               ├─> 检查终止条件
               └─> 返回next_obs, reward, done
        
        d) 存储经验 (Store Experience)
           └─> storage.add_transitions(obs, action, reward, done, value, ...)
    
    ┌─────────────────────────────────────────┐
    │ 9. 策略更新 (Policy Update)            │
    └─────────────────────────────────────────┘
    对于每个epoch (num_learning_epochs=5):
        
        a) 计算优势函数 (Compute Advantages)
           └─> storage.compute_returns(last_value, gamma, lam)
               └─> GAE: A_t = δ_t + γλδ_{t+1} + ...
        
        b) Mini-batch更新 (num_mini_batches=4)
           对于每个mini-batch:
               
               i) 采样batch数据
                  └─> obs, actions, values, returns, advantages, log_probs
               
               ii) 前向传播
                   └─> actor_critic.evaluate(obs, actions)
                       ├─> actor输出新的mean
                       ├─> 计算新的log_prob
                       └─> critic输出新的value
               
               iii) 计算损失
                    ├─> Surrogate Loss (PPO clip):
                    │   ratio = exp(new_log_prob - old_log_prob)
                    │   L_CLIP = min(ratio*A, clip(ratio)*A)
                    │
                    ├─> Value Loss:
                    │   L_V = (value - return)^2
                    │
                    ├─> Entropy Loss:
                    │   L_ENT = -mean(entropy)
                    │
                    └─> 总损失:
                        L = -L_CLIP + value_coef*L_V - entropy_coef*L_ENT
               
               iv) 反向传播
                   └─> optimizer.step()
                       ├─> 更新ActorCriticCNN参数
                       └─> 更新PVCNN参数（如果未冻结）
    
    ┌─────────────────────────────────────────┐
    │ 10. 日志记录 (Logging)                 │
    └─────────────────────────────────────────┘
    └─> TensorBoard记录
        ├─> Loss/policy_loss
        ├─> Loss/value_loss
        ├─> Policy/mean_reward
        ├─> Policy/episode_length
        └─> ...
```

### 阶段4: 保存与评估

```
11. 定期保存模型
    └─> runner.save(log_dir)
        ├─> 保存ActorCriticCNN权重
        ├─> 保存optimizer状态
        └─> 保存iteration信息

12. 训练完成
    └─> env.close()
        └─> 关闭Isaac Sim
```

---

## 📊 数据流详解

### 观测流（Observation Flow）

```
LiDAR传感器
    ↓ (num_rays=2046, XYZ坐标)
点云预处理
    ├─> 过滤无效点（Inf/NaN/零点）
    ├─> FPS采样或复制到2046点
    └─> 形状: (batch, 2046, 3)
    ↓
PVCNN前向传播
    ├─> 输入: (batch, 3, 2046)  # XYZ only
    ├─> 4层PVConv编码
    ├─> 全局特征聚合
    └─> 输出: {
          'logits': (batch, 4, 2046),      # 4类语义
          'confidence': (batch, 2046),      # 置信度
          'global_features': (batch, 128)   # 全局特征
        }
    ↓
分支1: PVCNN Features            分支2: Cost Map
    ↓                                ↓
展平logits                        投影到2D网格
(batch, 8184)                    ├─> Distance cost
                                 ├─> Gradient cost
                                 └─> Confidence cost
                                      ↓
                                 展平(batch, 12288)
    ↓                                ↓
    └────────────┬──────────────────┘
                 ↓
         拼接所有观测
    ┌─────────────────────────────┐
    │ pvcnn_features:    8184维   │
    │ height_scan:       187维    │
    │ proprio:           48维     │
    │ cost_map:          12288维  │
    │ ────────────────────────    │
    │ 总计:              20707维  │
    └─────────────────────────────┘
                 ↓
         ActorCriticCNN
    ┌─────────────────────────────┐
    │ 分离cost_map (12288)        │
    │ 分离proprio (8419)          │
    │         ↓                   │
    │ CNN编码cost_map -> 256维   │
    │ 拼接CNN特征+proprio         │
    │         ↓                   │
    │ Actor  -> 动作分布(12维)   │
    │ Critic -> 状态价值(1维)    │
    └─────────────────────────────┘
```

---

## 🎯 关键设计决策

### 1. **为什么展平cost_map？**
- Isaac Lab的ObservationManager要求所有观测都是1D向量
- 在ActorCriticCNN内部reshape回2D进行CNN处理
- 既满足框架要求，又利用了空间结构信息

### 2. **为什么同时使用pvcnn_features和cost_map？**
- `pvcnn_features`: 逐点语义标签（高维但完整）
- `cost_map`: 2D空间投影（低维但结构化）
- 提供互补信息，提升策略鲁棒性

### 3. **为什么冻结PVCNN？**
- PVCNN已在S3DIS上预训练
- RL数据量有限，避免过拟合
- 专注于学习运动策略，而非视觉特征

### 4. **为什么使用FPS采样？**
- 保持点云均匀分布
- 优于随机采样（保留重要几何特征）
- pytorch3d高效实现

---

## 🚀 运行命令

### 训练
```bash
cd /mnt/mydisk/lhy/testPvcnnWithIsaacsim
bash Go2Pvcnn/scripts/train_go2_pvcnn.sh \
    Go2Pvcnn/scripts/train_go2_pvcnn.py \
    --num_envs 2048 \
    --max_iterations 5000 \
    --headless
```

### 可视化训练
```bash
# TensorBoard
tensorboard --logdir=logs/rsl_rl/go2_pvcnn
```

### 恢复训练
```bash
bash Go2Pvcnn/scripts/train_go2_pvcnn.sh \
    Go2Pvcnn/scripts/train_go2_pvcnn.py \
    --resume \
    --load_run 2025-12-15_09-03-08 \
    --num_envs 2048
```

---

## 📁 关键文件路径

```
Go2Pvcnn/
├── scripts/
│   ├── train_go2_pvcnn.py          # 主训练脚本
│   └── train_go2_pvcnn.sh          # 启动脚本
├── go2_pvcnn/
│   ├── pvcnn_wrapper.py            # PVCNN模型包装器
│   ├── wrapper/
│   │   └── pvcnn_env_wrapper.py    # RSL-RL环境包装器
│   ├── mdp/
│   │   ├── observations.py         # 观测函数
│   │   └── cost_map.py             # Cost map生成器
│   └── tasks/
│       └── go2_pvcnn_env_cfg.py    # 环境配置
└── rsl_rl/
    └── rsl_rl/
        ├── algorithms/ppo.py       # PPO算法
        └── modules/
            └── actor_critic_cnn.py # CNN策略网络
```

---

## ✅ 验证清单

- [x] PVCNN成功加载并适配（9→3通道，13→4类别）
- [x] 点云预处理正常（过滤+采样）
- [x] Cost map生成正常（3×64×64）
- [x] 观测维度匹配（policy和critic一致）
- [x] ActorCriticCNN正常前向传播
- [x] PPO训练循环运行
- [x] TensorBoard日志记录正常

---

## 📈 预期训练指标

- **Episode Reward**: 逐渐增加（从负值到正值）
- **Episode Length**: 稳定在最大长度
- **Policy Loss**: 收敛到小值
- **Value Loss**: 收敛
- **Learning Rate**: 根据KL散度自适应调整

---

## 🔍 Debug技巧

### 查看观测维度
```python
# 在train_go2_pvcnn.py中添加
print(f"Policy obs shape: {env.observation_manager.group_obs_dim['policy']}")
print(f"Critic obs shape: {env.observation_manager.group_obs_dim['critic']}")
```

### 检查PVCNN输出
```python
# 在observations.py中添加
if call_count % 100 == 1:
    print(f"PVCNN output shapes: {pvcnn_output.keys()}")
    for k, v in pvcnn_output.items():
        print(f"  {k}: {v.shape}")
```

### 监控训练稳定性
```bash
# 查看loss是否有NaN/Inf
grep -i "nan\|inf" logs/rsl_rl/go2_pvcnn/*/summaries.txt
```

---

**训练系统已完全就绪！🎉**
