# Human PPO And Runner

## 导航

- 文档类型：`human` 阶段文档
- 对应 AI 文档：[../ai/ai-05-ppo-and-runner.md](../ai/ai-05-ppo-and-runner.md)
- 上一篇：[human-04-lidar-and-pvcnn.md](human-04-lidar-and-pvcnn.md)
- 下一篇：[human-06-assets-paths-and-experiments.md](human-06-assets-paths-and-experiments.md)
- 总索引：[../index.md](../index.md)

## 作用

说明 `rsl_rl` 里的 runner、PPO 和 rollout / update / checkpoint 主循环怎样和当前项目耦合。

## Mermaid 训练主循环图

```mermaid
graph LR
    train["训练入口\n../../Go2Pvcnn/scripts/train.py"]
    wrapper["SimpleRslRlEnvWrapper\n../../Go2Pvcnn/scripts/train.py"]
    runner["OnPolicyRunner\n../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py"]
    storage["RolloutStorage\n../../Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py"]
    algo["PPO\n../../Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py"]
    env["ManagerBasedRLEnv\nIsaac Lab"]
    obs["观测组\npolicy / critic"]
    ckpt["checkpoint / train_cfg / env_cfg\n../../logs/rsl_rl/"]

    train -->|"创建 env 后包成 VecEnv"| wrapper
    wrapper -->|"step/reset/get_observations"| env
    env -->|"返回 obs / reward / done"| obs
    obs -->|"rollout 数据写入"| storage
    storage -->|"mini-batch 采样"| algo
    algo -->|"参数更新"| runner
    runner -->|"learn() 驱动 rollout + update"| wrapper
    runner -->|"dump yaml / save model"| ckpt
```

## 重点文件

- [on_policy_runner.py](../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py)
- [ppo.py](../../Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py)
- [rollout_storage.py](../../Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py)

## 上游输入

- env wrapper 提供的观测、奖励、done
- PVCNN 特征或监督数据

## 下游消费者

- checkpoint 保存与恢复
- 日志记录
- 下轮训练迭代

## 待补充

- 同步 PVCNN 训练何时开启
- PPO 更新和 PVCNN 更新的耦合点
- 多 GPU 或分布式相关分支
