# Go2 PVCNN Locomotion

强化学习项目：使用PVCNN处理LiDAR点云数据来训练Unitree Go2机器狗行走。

## 项目概述

本项目结合了以下技术：
- **Isaac Lab 2.21** + **Isaac Sim 4.5**: 物理仿真环境
- **PVCNN**: 点云特征提取网络（预训练在S3DIS数据集上）
- **RSL-RL PPO**: 强化学习算法
- **Unitree Go2**: 四足机器人

### 核心思想

使用LiDAR传感器获取环境点云数据，通过预训练的PVCNN模型提取语义特征，将这些特征作为观测输入到PPO策略网络，训练机器狗在复杂地形上行走。

## 目录结构

```
Go2Pvcnn/
├── go2_pvcnn/
│   ├── __init__.py
│   ├── pvcnn_wrapper.py          # PVCNN模型加载和推理
│   ├── pvcnn_observer.py         # 自定义观测项
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── go2_pvcnn_env_cfg.py  # 环境配置
│   │   └── register_envs.py      # Gym环境注册
│   └── wrapper/
│       ├── __init__.py
│       └── pvcnn_env_wrapper.py  # RSL-RL环境包装器
├── scripts/
│   ├── train_go2_pvcnn.py        # 训练脚本
│   └── train_go2_pvcnn.sh        # 启动脚本（设置环境变量）
├── setup.py                       # 安装配置
└── README.md                      # 本文件
```

## 安装

### 前置条件

1. **Conda环境**: `env_isaacsim`
2. **Isaac Lab 2.21** 已安装
3. **Isaac Sim 4.5** 已安装
4. **Go2Pvcnn** 已通过pip安装

```bash
# 激活环境
conda activate env_isaacsim

# 验证Go2Pvcnn已安装
python -c "import go2_pvcnn; print('Go2Pvcnn installed successfully')"
```

### PVCNN权重

预训练的PVCNN权重应位于：
```
/home/lhy/testPvcnnWithIsaacsim/pvcnn/runs/s3dis.pvcnn.area5.c1/best.pth.tar
```

## 使用方法

### 训练

使用提供的启动脚本（推荐）：

```bash
# 基础训练（4096个环境）
./scripts/train_go2_pvcnn.sh

# 自定义环境数量
./scripts/train_go2_pvcnn.sh --num_envs 2048

# 无头模式（更快）
./scripts/train_go2_pvcnn.sh --headless --num_envs 4096

# 记录训练视频
./scripts/train_go2_pvcnn.sh --video --num_envs 1024

# 恢复训练
./scripts/train_go2_pvcnn.sh --resume --load_run 2024-01-01_12-00-00
```

或直接运行Python脚本：

```bash
# 先设置环境变量
unset __GLX_VENDOR_LIBRARY_NAME
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH

# 运行训练
python scripts/train_go2_pvcnn.py --num_envs 4096
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_envs` | 4096 | 并行环境数量 |
| `--max_iterations` | 5000 | 最大训练迭代次数 |
| `--seed` | 42 | 随机种子 |
| `--num_points` | 2046 | PVCNN输入点云数量 |
| `--checkpoint_path` | `best.pth.tar` | PVCNN权重路径 |
| `--headless` | False | 无头模式运行 |
| `--video` | False | 录制训练视频 |
| `--resume` | False | 恢复训练 |

## 技术细节

### PVCNN集成

1. **点云采集**: LiDAR传感器（Livox Mid-360）采集2046个点
2. **特征提取**: PVCNN处理点云，输出128维全局特征
3. **观测空间**: PVCNN特征 + 机器人状态（速度、关节位置等）

### 观测空间

**Policy观测**（用于决策）:
- PVCNN特征: 128维
- 基座速度: 6维（线速度3 + 角速度3）
- 投影重力: 3维
- 关节位置: 12维
- 关节速度: 12维
- 上一次动作: 12维

**Critic观测**（特权信息）:
- Policy观测
- 高度扫描: 160维

### 奖励函数

- **任务奖励**:
  - 跟踪目标线速度
  - 跟踪目标角速度
  
- **惩罚项**:
  - Z轴线速度
  - XY轴角速度
  - 关节力矩
  - 关节加速度
  - 动作变化率
  - 不期望的接触（大腿触地）
  - 关节位置限制

- **额外奖励**:
  - 足端腾空时间

## 环境变量说明

Isaac Sim 4.5需要特定的环境变量才能正常运行：

```bash
# 重置GLX供应商库（必需）
unset __GLX_VENDOR_LIBRARY_NAME
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# 添加NVIDIA库路径（必需）
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH
```

这些变量已在`train_go2_pvcnn.sh`中自动设置。

## 配置修改

### 修改Go2机器人模型

编辑 `go2_pvcnn/tasks/go2_pvcnn_env_cfg.py`:

```python
def _get_robot_cfg(self) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="<YOUR_GO2_USD_PATH>",  # 修改这里
            # ... 其他配置
        ),
    )
```

### 修改LiDAR配置

编辑 `go2_pvcnn/tasks/go2_pvcnn_env_cfg.py` 中的 `lidar_sensor`:

```python
lidar_sensor = LidarSensorCfg(
    pattern_cfg=LivoxPatternCfg(
        sensor_type="mid360",    # 传感器类型
        samples=2046,            # 点数
        use_simple_grid=False,   # 使用真实扫描模式
    ),
    max_distance=20.0,           # 最大距离
    update_frequency=10.0,       # 更新频率
    # ... 其他配置
)
```

### 修改PPO超参数

编辑 `scripts/train_go2_pvcnn.py` 中的 `agent_cfg`:

```python
agent_cfg = OnPolicyRunnerCfg(
    learning_rate=1e-3,         # 学习率
    num_steps_per_env=24,       # 每环境步数
    num_learning_epochs=5,      # 学习轮数
    num_mini_batches=4,         # 小批量数
    # ... 其他配置
)
```

## 监控训练

训练日志保存在 `logs/rsl_rl/go2_pvcnn/` 目录下。

使用TensorBoard查看训练曲线：

```bash
tensorboard --logdir logs/rsl_rl/go2_pvcnn/
```

## 故障排除

### 问题1: PVCNN模型加载失败

**错误**: `FileNotFoundError: Checkpoint not found`

**解决**: 检查PVCNN权重路径是否正确：
```bash
ls /home/lhy/testPvcnnWithIsaacsim/pvcnn/runs/s3dis.pvcnn.area5.c1/best.pth.tar
```

### 问题2: Isaac Sim渲染错误

**错误**: GLX相关错误

**解决**: 确保运行了启动脚本或手动设置了环境变量：
```bash
unset __GLX_VENDOR_LIBRARY_NAME
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH
```

### 问题3: 内存不足

**错误**: CUDA out of memory

**解决**: 减少并行环境数量：
```bash
./scripts/train_go2_pvcnn.sh --num_envs 1024
```

### 问题4: LiDAR传感器未找到

**错误**: `LidarSensor not available`

**解决**: 确保已安装LiDAR集成：
- 检查 `omni.isaac.lab.sensors` 中是否有 `LidarSensor`
- 参考 `simple_lidar_integration.py` 中的集成方法

## 参考代码

本项目参考了以下代码：
- VR-Robo项目的PPO训练框架
- PVCNN的S3DIS评估代码
- Isaac Lab的LiDAR传感器集成

## 注意事项

1. **点云数量**: PVCNN训练时使用的是4096个点，但这里使用2046个点以平衡性能和计算效率
2. **特征通道**: PVCNN期望9个通道（XYZ + RGB + 法向量），当前使用占位符填充额外通道
3. **推理频率**: LiDAR更新频率设置为10Hz以减少计算负担
4. **只推理**: PVCNN模型仅用于推理，不进行训练

## 性能优化

对于大规模训练（>2048环境）：
1. 使用无头模式: `--headless`
2. 降低LiDAR更新频率: 修改 `update_frequency`
3. 减少点云数量: 修改 `samples` 参数
4. 禁用调试可视化: 设置 `debug_vis=False`

## 许可

本项目基于以下开源项目：
- Isaac Lab (BSD-3-Clause)
- PVCNN (MIT License)
- RSL-RL (BSD-3-Clause)

## 联系

如有问题，请查看代码注释或联系项目维护者。
