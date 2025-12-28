# Go2 PVCNN项目架构说明

## 项目概述

这是一个使用PVCNN（Point-Voxel CNN）处理LiDAR点云数据来训练Unitree Go2四足机器人行走的强化学习项目。

## 技术栈

- **仿真环境**: Isaac Lab 2.21 + Isaac Sim 4.5
- **感知模型**: PVCNN (预训练在S3DIS室内场景数据集)
- **强化学习**: RSL-RL PPO算法
- **机器人**: Unitree Go2四足机器人

## 核心架构

### 1. 数据流

```
LiDAR传感器 (Livox Mid-360)
    ↓ 采集点云 (num_envs, 2046, 3)
PVCNN特征提取器
    ↓ 提取语义特征 (num_envs, 128)
观测组合 [PVCNN特征 + 机器人状态]
    ↓ (num_envs, obs_dim)
PPO Policy Network
    ↓ (num_envs, action_dim)
机器人执行动作
    ↓
环境反馈奖励
```

### 2. 模块结构

#### `pvcnn_wrapper.py` - PVCNN推理模块
**功能**:
- 加载预训练PVCNN模型
- 处理点云输入（采样、填充、格式转换）
- 提取128维全局特征
- 支持批量推理

**关键方法**:
- `extract_features(point_cloud)`: 主推理接口
- `_resample_points()`: 点云重采样到固定数量
- `get_feature_dim()`: 返回特征维度

#### `pvcnn_observer.py` - 观测项定义
**功能**:
- 定义所有MDP观测项
- 整合PVCNN特征到观测空间
- 提供机器人状态观测

**关键函数**:
- `get_pvcnn_features()`: 从LiDAR获取点云并通过PVCNN提取特征
- `get_base_velocity()`: 基座速度
- `get_projected_gravity()`: 投影重力
- `get_joint_positions/velocities()`: 关节状态

#### `go2_pvcnn_env_cfg.py` - 环境配置
**功能**:
- 定义场景（地形、机器人、传感器）
- 配置观测空间（policy和critic）
- 设置奖励函数
- 定义终止条件

**关键配置类**:
- `Go2SceneCfg`: 场景配置（LiDAR、地形、机器人）
- `ObservationsCfg`: 观测配置
- `RewardsCfg`: 奖励配置
- `Go2PvcnnEnvCfg`: 主环境配置

#### `pvcnn_env_wrapper.py` - RSL-RL包装器
**功能**:
- 将Isaac Lab环境包装为RSL-RL兼容格式
- 注入PVCNN wrapper到观测管理器
- 处理观测、动作、奖励的格式转换

**关键方法**:
- `_inject_pvcnn_wrapper()`: 将PVCNN注入观测项
- `step()`: 环境步进
- `reset()`: 环境重置
- `get_observations()`: 获取观测

#### `train_go2_pvcnn.py` - 训练脚本
**功能**:
- 初始化环境和PVCNN
- 配置PPO算法
- 启动训练循环
- 保存检查点和日志

**流程**:
1. 解析命令行参数
2. 创建PVCNN wrapper
3. 创建Isaac Lab环境
4. 包装为RSL-RL环境
5. 创建PPO runner
6. 开始训练

### 3. 观测空间设计

#### Policy观测 (173维)
```python
[
    PVCNN特征(128),           # 点云语义特征
    基座线速度(3),            # vx, vy, vz
    基座角速度(3),            # wx, wy, wz
    投影重力(3),              # gx, gy, gz (base frame)
    关节位置(12),             # 12个关节角度
    关节速度(12),             # 12个关节速度
    上一次动作(12)            # 上一步的动作
]
```

#### Critic观测 (Policy + 160维)
```python
[
    Policy观测(173),
    高度扫描(160)             # 特权信息：地形高度
]
```

### 4. 奖励函数

#### 任务奖励
- `track_lin_vel_xy_exp`: 跟踪XY平面线速度 (权重: 1.0)
- `track_ang_vel_z_exp`: 跟踪Z轴角速度 (权重: 0.5)
- `feet_air_time`: 足端腾空时间 (权重: 0.125)

#### 惩罚项
- `lin_vel_z_l2`: Z轴速度惩罚 (权重: -2.0)
- `ang_vel_xy_l2`: XY轴角速度惩罚 (权重: -0.05)
- `dof_torques_l2`: 关节力矩惩罚 (权重: -1e-5)
- `dof_acc_l2`: 关节加速度惩罚 (权重: -2.5e-7)
- `action_rate_l2`: 动作变化率惩罚 (权重: -0.01)
- `undesired_contacts`: 不期望接触惩罚 (权重: -1.0)
- `dof_pos_limits`: 关节位置限制惩罚 (权重: -1.0)

### 5. 训练超参数

#### PPO配置
```python
learning_rate = 1e-3
num_steps_per_env = 24
num_learning_epochs = 5
num_mini_batches = 4
clip_param = 0.2
gamma = 0.99
lam = 0.95
```

#### 网络架构
```python
Policy Network:
  - Hidden layers: [512, 256, 128]
  - Activation: ELU
  - Output: 12 (关节动作)

Critic Network:
  - Hidden layers: [512, 256, 128]
  - Activation: ELU
  - Output: 1 (价值估计)
```

### 6. LiDAR配置

```python
传感器类型: Livox Mid-360
扫描模式: 真实模式（从.npy文件加载）
点云数量: 2046个点
更新频率: 10 Hz
测量范围: 0.2m - 20.0m
视场角: 360° (水平) × 59° (垂直)
传感器噪声: 启用，σ = 0.01m
坐标系: 传感器本地坐标系
```

### 7. 性能优化

#### 计算优化
- PVCNN在GPU上批量推理
- LiDAR更新频率降低到10Hz
- 点云数量限制为2046（平衡精度和速度）
- 禁用不必要的可视化

#### 内存优化
- 使用环境变量控制并行环境数
- 支持梯度检查点（如需要）
- 及时清理中间特征

### 8. 环境变量

训练前需要设置的环境变量（已在启动脚本中自动设置）：

```bash
# Isaac Sim 4.5 必需
unset __GLX_VENDOR_LIBRARY_NAME
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH

# Python路径
export PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:$PYTHONPATH
```

## 使用流程

### 1. 测试设置
```bash
conda activate env_isaacsim
cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn
python scripts/test_setup.py
```

### 2. 开始训练
```bash
# 小规模测试
./scripts/train_go2_pvcnn.sh --num_envs 256

# 完整训练
./scripts/train_go2_pvcnn.sh --num_envs 4096 --headless
```

### 3. 监控训练
```bash
tensorboard --logdir logs/rsl_rl/go2_pvcnn/
```

### 4. 恢复训练
```bash
./scripts/train_go2_pvcnn.sh --resume --load_run 2024-01-01_12-00-00
```

## 扩展点

### 添加新观测
在 `pvcnn_observer.py` 中添加新函数，然后在 `go2_pvcnn_env_cfg.py` 的 `ObservationsCfg` 中引用。

### 修改奖励函数
在 `go2_pvcnn_env_cfg.py` 的 `RewardsCfg` 中修改权重或添加新奖励项。

### 更换PVCNN模型
修改 `pvcnn_wrapper.py` 中的模型加载代码，支持不同的checkpoint。

### 调整LiDAR配置
在 `go2_pvcnn_env_cfg.py` 的 `Go2SceneCfg.lidar_sensor` 中修改参数。

## 注意事项

1. **PVCNN只推理**: 模型参数冻结，不参与训练
2. **点云数量**: 2046个点是性能和精度的平衡点
3. **特征通道**: 当前使用占位符填充RGB和法向量，可根据需要替换为真实特征
4. **Go2模型**: 需要手动指定Go2的USD/URDF文件路径
5. **内存需求**: 4096环境约需32GB GPU内存

## 参考

- Isaac Lab文档: https://isaac-sim.github.io/IsaacLab/
- PVCNN论文: https://arxiv.org/abs/1907.03739
- RSL-RL: https://github.com/leggedrobotics/rsl_rl
