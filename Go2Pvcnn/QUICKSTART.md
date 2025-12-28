# Go2 PVCNN项目 - 快速开始指南

## 项目已创建完成！

我已经为你创建了一个完整的使用PVCNN的Go2机器狗强化学习项目。

## 📁 创建的文件

```
Go2Pvcnn/
├── go2_pvcnn/                      # 主模块
│   ├── __init__.py
│   ├── pvcnn_wrapper.py            # ✅ PVCNN模型推理封装
│   ├── pvcnn_observer.py           # ✅ 自定义观测项（LiDAR→PVCNN）
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── go2_pvcnn_env_cfg.py    # ✅ 环境配置（场景、观测、奖励）
│   │   └── register_envs.py        # ✅ Gym环境注册
│   └── wrapper/
│       ├── __init__.py
│       └── pvcnn_env_wrapper.py    # ✅ RSL-RL环境包装器
├── scripts/
│   ├── train_go2_pvcnn.py          # ✅ 主训练脚本
│   ├── train_go2_pvcnn.sh          # ✅ 启动脚本（设置环境变量）
│   └── test_setup.py               # ✅ 测试脚本
├── setup.py                         # ✅ 安装配置
└── README.md                        # ✅ 详细文档
```

## 🚀 使用步骤

### 1. 激活环境
```bash
conda activate env_isaacsim
```

### 2. 测试设置（推荐）
```bash
cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn
python scripts/test_setup.py
```

这会测试：
- PVCNN模型是否能正确加载
- PVCNN推理是否正常工作
- 所有导入是否成功

### 3. 开始训练
```bash
# 使用启动脚本（推荐）
/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/scripts/train_go2_pvcnn.sh --num_envs 2048

# 或无头模式（更快）
/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/scripts/train_go2_pvcnn.sh --headless --num_envs 4096

# 运行测试
/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/scripts/train_go2_pvcnn.sh test
```

## ⚙️ 重要配置

### 需要修改的地方

**1. Go2机器人模型路径**

编辑 `go2_pvcnn/tasks/go2_pvcnn_env_cfg.py`，第372行左右：

```python
usd_path="<PATH_TO_GO2_USD>",  # ← 改成你的Go2 USD/URDF路径
```

你需要将这里改成实际的Go2机器人模型文件路径。

**2. 其他可选配置**

- **LiDAR点云数量**: 默认2046个点，可在启动时通过 `--num_points` 修改
- **环境数量**: 默认4096，可通过 `--num_envs` 修改
- **训练迭代次数**: 默认5000，可通过 `--max_iterations` 修改

## 🔑 关键特性

### 1. PVCNN集成
- ✅ 自动加载预训练权重（S3DIS数据集）
- ✅ 提取128维全局特征
- ✅ 支持批量推理（高效）
- ✅ 自动处理点云采样/填充

### 2. 观测空间设计
```
Policy观测 (用于决策):
├── PVCNN特征: 128维          ← 点云语义特征
├── 基座速度: 6维              ← [lin_vel, ang_vel]
├── 投影重力: 3维
├── 关节位置: 12维
├── 关节速度: 12维
└── 上一次动作: 12维

Critic观测 (特权信息):
├── Policy观测
└── 高度扫描: 160维
```

### 3. LiDAR配置
- 传感器: Livox Mid-360（真实扫描模式）
- 点数: 2046个
- 更新频率: 10Hz（平衡性能）
- 范围: 0.2m - 20.0m

### 4. 环境变量自动设置
启动脚本会自动设置Isaac Sim所需的环境变量：
```bash
unset __GLX_VENDOR_LIBRARY_NAME
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH
```

## 📊 监控训练

训练开始后，可以用TensorBoard查看：

```bash
tensorboard --logdir logs/rsl_rl/go2_pvcnn/
```

## 🛠️ 故障排除

### 问题1: Go2模型未找到
**解决**: 修改 `go2_pvcnn/tasks/go2_pvcnn_env_cfg.py` 中的USD路径

### 问题2: PVCNN权重未找到
**检查**: 
```bash
ls /home/lhy/testPvcnnWithIsaacsim/pvcnn/runs/s3dis.pvcnn.area5.c1/best.pth.tar
```

### 问题3: 内存不足
**解决**: 减少环境数量
```bash
/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/scripts/train_go2_pvcnn.sh --num_envs 1024
```

### 问题4: LiDAR传感器错误
**检查**: 确保已安装LiDAR集成（见README.md中的LiDAR部分）

## 📚 代码说明

### PVCNN推理流程
```python
# 1. LiDAR采集点云 (num_envs, 2046, 3)
point_cloud = lidar_sensor.data.point_cloud

# 2. 添加额外特征通道 → (num_envs, 2046, 9)
#    [XYZ坐标(3) + RGB(3) + 法向量(3)]

# 3. PVCNN提取特征 → (num_envs, 128)
features = pvcnn_wrapper.extract_features(point_cloud_with_features)

# 4. 特征作为观测输入PPO
```

### 奖励函数
- **跟踪速度**: 跟随命令速度（前进、转向）
- **惩罚项**: 不自然运动、关节限制、碰撞等
- **额外奖励**: 足端腾空时间（鼓励行走）

## 🎯 下一步

1. **测试设置**: 运行 `test_setup.py` 确保一切正常
2. **小规模测试**: 先用少量环境测试（如256个）
3. **调整参数**: 根据结果调整学习率、奖励权重等
4. **扩大规模**: 逐步增加环境数量到4096

## 📖 完整文档

详细文档请查看 `Go2Pvcnn/README.md`

## ✅ 项目特点

- ✅ 完全模块化设计，易于修改
- ✅ 自动环境变量设置
- ✅ 支持视频录制
- ✅ 支持恢复训练
- ✅ 详细的日志和监控
- ✅ 测试脚本验证设置

祝训练顺利！🚀
