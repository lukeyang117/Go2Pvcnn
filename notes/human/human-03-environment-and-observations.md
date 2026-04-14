# Human Environment And Observations

## 导航

- 文档类型：`human` 阶段文档
- 对应 AI 文档：[../ai/ai-03-environment-and-observations.md](../ai/ai-03-environment-and-observations.md)
- 上一篇：[human-02-training-and-entrypoints.md](human-02-training-and-entrypoints.md)
- 下一篇：[human-04-lidar-and-pvcnn.md](human-04-lidar-and-pvcnn.md)
- 总索引：[../index.md](../index.md)

## 作用

说明任务配置、场景、奖励、观测和 curriculum 是怎么拼起来的。

## Mermaid 代码结构图

```mermaid
graph LR
    register["任务注册\n../../Go2Pvcnn/go2_pvcnn/tasks/register_envs.py"]
    semantic["teacher_semantic\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_semantic_env_cfg.py"]
    nosemantic["teacher_without_semantic\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_without_semantic_env_cfg.py"]
    elevation["teacher_elevation\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_env_cfg.py"]
    traj["teacher_elevation_trajectory\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py"]
    obs["观测函数\n../../Go2Pvcnn/go2_pvcnn/mdp/observations.py"]
    rewards["奖励/终止/事件\n../../Go2Pvcnn/go2_pvcnn/mdp/"]
    sensors["ContactSensor / RayCaster\nisaaclab.sensors"]
    curriculum["课程与命令\n../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py"]

    register -->|"选择不同 env cfg"| semantic
    register -->|"共享基类链"| nosemantic
    nosemantic -->|"继承并加 height_scanner"| elevation
    elevation -->|"进一步加 trajectory reward"| traj
    semantic -->|"ObsTerm / RewTerm 绑定函数"| obs
    nosemantic -->|"ObsTerm / RewTerm 绑定函数"| obs
    elevation -->|"height_scan 观测"| obs
    traj -->|"downsampled height + reference reward"| obs
    semantic -->|"配置 scene 里的 lidar / contact"| sensors
    elevation -->|"配置 scene 里的 height_scanner"| sensors
    nosemantic -->|"reset / push / command 范围"| curriculum
    obs -->|"policy / critic tensor"| rewards
```

## 重点目录

- [tasks](../../Go2Pvcnn/go2_pvcnn/tasks)
- [curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)

## 上游输入

- 入口脚本选择的 env cfg
- 机器人模型、地形、传感器配置

## 下游消费者

- LiDAR / ray caster
- PVCNN 特征抽取
- PPO runner 的观测输入

## 待补充

- policy / critic 观测各自包含什么
- curriculum 在哪些 env cfg 中启用
- PLAY 和训练版配置的真实差异
