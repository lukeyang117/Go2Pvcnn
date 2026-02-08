# Teacher Semantic Training System

这是一个基于真实语义标签的教师训练系统，不使用PVCNN推理，直接从LiDAR语义标签生成代价地图。

## 文件结构

### 1. 核心文件

#### `/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/go2_pvcnn/mdp/cost_map_new.py`
- **功能**: 基于真实语义标签的代价地图生成器
- **类**: `TeacherCostMapGenerator`
- **特点**:
  - 直接使用LiDAR语义标签（无PVCNN推理）
  - 结合高程图计算多项代价
  - 支持command方向的代价减免（鼓励沿目标方向移动）
  - 代价类型：
    - Obstacle cost: 障碍物位置代价
    - Distance cost: 离机器人距离代价
    - Gradient cost: 地形陡峭度代价
    - Height cost: 绝对高度代价
    - Command alignment bonus: 沿command方向的奖励

#### `/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/go2_pvcnn/mdp/rewards_new.py`
- **功能**: Teacher模式专用的reward函数
- **包含的reward**:
  - `semantic_cost_map_penalty`: 基于代价地图的足端惩罚
  - `obstacle_collision_penalty`: 动态障碍物碰撞惩罚
  - `furniture_collision_penalty`: 家具碰撞惩罚（已注释）
  - `non_foot_ground_contact_penalty`: 非足端接地惩罚
  - `command_alignment_reward`: command方向对齐奖励
  - 复用原有的基础reward: 速度跟踪、关节限制等

#### `/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/go2_pvcnn/tasks/teacher_semantic_env_cfg.py`
- **功能**: Teacher训练环境配置
- **特点**:
  - 使用`SemanticLidarCfg`获取真实语义标签
  - 配置了ground contact、small objects、furniture三类contact sensor
  - 9个YCB动态物体（3×CrackerBox, 3×SugarBox, 3×TomatoSoupCan）
  - 家具部分已注释（可按需启用）
  - 继承自`ManagerBasedRLEnvCfg`

#### `/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/scripts/train.py`
- **功能**: 训练启动脚本
- **使用的包**: `rsl-rl-2-01` (替代rsl_rl)
- **wrapper**: 使用`go2_pvcnn.wrapper`目录下的wrapper
- **特点**:
  - 支持单GPU和多GPU训练
  - 简化的环境wrapper（无PVCNN推理）
  - 完整的训练循环和checkpoint管理

### 2. 语义类别映射

LiDAR语义分割使用以下类别：
- **Class 0**: No hit / Unknown
- **Class 1**: Terrain (地形，可通行)
- **Class 2**: Dynamic obstacle (动态障碍物 - CrackerBox, SugarBox, TomatoSoupCan)
- **Class 3**: Valuable (重要物体 - 家具，如sofa, armchair, table)

### 3. Contact Sensor配置

三个独立的contact sensor:
1. **contact_forces_ground**: 监测足端与地面接触
2. **contact_forces_small_objects**: 监测与YCB小物体的碰撞（base only, with filter）
3. **contact_forces_furniture**: 监测与家具的碰撞（已注释，可启用）

## 使用方法

### 1. 环境依赖

确保已安装：
```bash
# 已安装的rsl-rl-2-01包
pip show rsl-rl-2-01

# Isaac Lab环境
# Go2 资产包
```

### 2. 单GPU训练

```bash
cd /mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn
python scripts/train.py --num_envs 256 --headless --max_iterations 5000
```

### 3. 多GPU训练

```bash
# 使用GPU 0,1训练
export GPU_IDS="0,1"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train.py --num_envs 512 --headless --distributed --max_iterations 5000
```

### 4. 恢复训练

```bash
python scripts/train.py --num_envs 256 --headless --resume --load_run 2026-02-01_12-00-00
```

### 5. 参数说明

- `--num_envs`: 环境数量（会被GPU数量平分）
- `--max_iterations`: 最大训练迭代次数
- `--seed`: 随机种子
- `--resume`: 是否恢复训练
- `--load_run`: 要加载的run目录名
- `--load_checkpoint`: 要加载的checkpoint文件名
- `--headless`: 无头模式（不显示GUI）
- `--distributed`: 启用多GPU训练

## 训练配置

### PPO超参数

在`train.py`中配置：
```python
ppo_cfg = PPOCfg(
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1e-3,
    clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.01,
    max_grad_norm=1.0,
)
```

### Reward权重

在`teacher_semantic_env_cfg.py`中配置：
```python
track_lin_vel_xy_exp = RewTerm(weight=10.0, ...)
track_ang_vel_z_exp = RewTerm(weight=0.5, ...)
command_alignment = RewTerm(weight=2.0, ...)
obstacle_collision = RewTerm(weight=-5.0, ...)
# ... 更多reward配置
```

## 输出文件

训练输出保存在：
```
logs/rsl_rl/teacher_semantic/YYYY-MM-DD_HH-MM-SS/
├── env_cfg.yaml          # 环境配置
├── runner_cfg.yaml       # Runner配置
├── ppo_cfg.yaml          # PPO配置
├── model_*.pt           # 模型checkpoints
└── summaries/           # TensorBoard日志
```

## TODO和扩展

### 已完成
- ✅ 代价地图生成（cost_map_new.py）
- ✅ Teacher reward函数（rewards_new.py）
- ✅ 环境配置（teacher_semantic_env_cfg.py）
- ✅ 训练脚本（train.py）

### 待实现
- ⚠️ 语义代价地图observation term（需要在observations.py中添加）
- ⚠️ 启用家具物体（取消注释相关代码）
- ⚠️ 实现height map到代价地图的完整pipeline
- ⚠️ 添加observation term调用cost_map_new.py

### 下一步工作

1. **添加observation term**:
   ```python
   # 在go2_pvcnn/mdp/observations.py中添加
   def teacher_semantic_cost_map(env, sensor_cfg, height_scanner_cfg):
       # 从LiDAR获取semantic_labels和point_xyz
       # 从height_scanner获取height_map
       # 调用TeacherCostMapGenerator生成cost_map
       # 返回cost_map flatten
   ```

2. **在环境配置中启用observation**:
   ```python
   # 在teacher_semantic_env_cfg.py的ObservationsCfg中
   semantic_cost_map = ObsTerm(
       func=custom_mdp.teacher_semantic_cost_map,
       params={
           "sensor_cfg": SceneEntityCfg("lidar"),
           "height_scanner_cfg": SceneEntityCfg("height_scanner"),
       }
   )
   ```

3. **测试训练pipeline**:
   ```bash
   # 小规模测试
   python scripts/train.py --num_envs 4 --max_iterations 10
   ```

## 注意事项

1. **环境注册**: 需要在`__init__.py`中注册环境：
   ```python
   gym.register(
       id="Isaac-Teacher-Semantic-Go2-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       kwargs={"env_cfg_entry_point": TeacherSemanticEnvCfg},
   )
   ```

2. **Import路径**: 确保所有import路径正确，特别是：
   - `from rsl_rl_2_01 import ...` (使用2.01版本)
   - `from go2_pvcnn.mdp import rewards_new as teacher_rewards`
   - `from go2_pvcnn.wrapper import ...`

3. **语义映射**: 确保LiDAR的semantic_class_mapping与cost_map_new.py中的obstacle_class_ids一致

4. **GPU内存**: 大量环境+语义LiDAR可能占用较多GPU内存，建议：
   - 单GPU: 256-512 envs
   - 双GPU: 512-1024 envs

## 与PVCNN版本的区别

| 特性 | PVCNN版本 | Teacher版本 |
|------|----------|------------|
| 语义获取 | PVCNN推理 | LiDAR真实语义 |
| 代价地图 | PVCNN特征+cost_map.py | 真实语义+cost_map_new.py |
| Wrapper | RslRlPvcnnEnvWrapper | SimpleRslRlEnvWrapper |
| rsl_rl包 | rsl_rl (3.x) | rsl-rl-2-01 (2.0.1) |
| Command bonus | 无 | 有（减少代价） |
| 训练目标 | 学习感知+控制 | 纯控制（感知完美） |
