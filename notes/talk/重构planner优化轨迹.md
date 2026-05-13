# 重构 planner 优化轨迹：聊天记录

记录时间：2026-05-11

说明：这是把当前会话中可见的聊天内容导入到本文件，方便下一个 agent 接手。用户发言按当前上下文尽量逐字保留；之前 assistant 的逐字回复没有完整出现在当前上下文时，不伪造原文，只记录可恢复的执行摘要和结论。

## 用户原始发言

```text
[$brainstorming](/mnt/mydisk/lhy/testPvcnnWithIsaacsim/.agents/skills/brainstorming/SKILL.md) 你现在回顾/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/batched_together_planner，我现在想重新实现一个planner，然后接入/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/viz/go2_foostep_planner.py，当前的问题就是我觉得这个不符合我的预期，我希望是一个mcp，来算梯度下降，从而优化轨迹，你觉得这样要怎么做，考虑接入强化学习，所以训练要快，无cpu的内容

不用

mpc，我说错了

1，但是stance和swing的时序也要放入优化

contact_state: bool [B,T,4]是什么

我希望这个也要加入梯度下降，不能固定

2

我认为这个可以做个config，直接能接入RL的配置，这样方便我调参，就像高程图，一个实现类，一个配置类，然后强化学习配置里面导入这个配置类就可以，然后现在就先长一点把，因为用长一点效果更好，同时这个参数可以加入配置类，后期来调

1

1，同时我的这个mpc不能只走步态，还要考虑能跨过小障碍物，要求是要和together planner一样的，只不过从轨迹搜索变成了梯度下降，来调整轨迹

2，旧概念不要保留了，很罗嗦，MPC 优化变量大概是：root spline residual [B,T,3] root rpy residual [B,T,3] foot xyz residual [B,T,4,3] contact_logits [B,T,4]  ，轨迹和touchdowns都在foot xyz residual [B,T,4,3] ，同时每条损失也要像rl配置reward一样，可以选调的权重

全都需要

可以

那这个硬诊断层的作用是什么？，同时这个loss除了权重，暴露出一些可以调的参数，到配置类里

这个硬约束可以添加一个enable的参数，然后如果要测试，或者验证，可以启动这个，来帮助测试，但是也可以当测试好后，disable，这样训练快一点，上述配置我认可

我的训练打算是4096个环境，你觉得这个怎么做能保证速度和精度

你分别解释这些参数是做什么的

还要考虑环境的并行性，还有如果给的command变了，replan_interval_steps到了，或者触发了reset，都要重新规划，这样的话，你觉得每个step大改要多久

比如我的rl配置，速度每隔一段时间就会变，你觉得要怎么做适配呢

我没太搞懂每 env 随机相位更新是什么意思

可以，每个环境的reset和时间步到了，还有速度变了的时刻可以异步，不是所有环境都要重新规划，而是每个step，需要更新才更新，不更新就用上次规划出来的轨迹的下一个时刻，你觉得这样的话，4096个环境需要每个step多久

可以

1

这个任务不能挂到T100，新建一个子节点，新建md文件

使用subagent，对design审查一下，对我的需求完成度，强化学习并行，还有测试不复用性，新写测试，对isaaclab的适配性，测试也要在isaaclab进行，不用cpu，只用gpu，还有用硬检测层来做测试（worker），还有配置类和实现类，文件结构设计，都审查一下

用这些结论修改design

/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/batch_mpc_planner,在这个路径，你在这里添加代码 ，加到设计中，并且设计一些文件系统

可以用subagent来修改代码，测试，主agent负责审查，而且你需要把设计先写成todo，给subagent来执行

继续

要使用isaaclab环境测试，环境：env_isaacsim          *  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim，需要启动isaaclab，来做headless的测试，用测试层

继续吧

使用cuda2，3来测试，这个测试一定要跑，

他现在的输出是什么，你和我说说

/mnt/mydisk/lhy/testPvcnnWithIsaacsim/notes/human/human-16-isaaclab-applauncher-webrtc-migration.md，你把这个viewer和启动训练的命令行更新一下mcp planner，我自己来测试

(env_isaacsim) lhy@enine:~/testPvcnnWithIsaacsim$ python \
Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:2 \
  --num_envs 4096 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory \
  --planner-backend mpc
usage: train.py [-h] [--num_envs NUM_ENVS] [--seed SEED] [--max_iterations MAX_ITERATIONS] [--video] [--video_length VIDEO_LENGTH]
                [--video_interval VIDEO_INTERVAL] [--resume] [--load_run LOAD_RUN] [--load_checkpoint LOAD_CHECKPOINT] [--distributed]，为什么， 
                [--experiment {teacher_semantic,teacher_without_semantic,teacher_elevation,teacher_elevation_semantic_map,teacher_elevation_trajectory}]

启动了命令行，然后报错Traceback (most recent call last):
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/scripts/train.py", line 560, in <module>
    raise SystemExit(main())
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/scripts/train.py", line 542, in main
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/third_party/rsl_rl/rsl_rl_2_01/runners/on_policy_runner.py", line 119, in learn
    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/scripts/train.py", line 431, in step
    obs_dict, rewards, dones, truncated, extras = self.env.step(actions)
  File "/mnt/mydisk/lhy/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py", line 207, in step
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
  File "/mnt/mydisk/lhy/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py", line 148, in compute
    value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/mdp/rewards_reference.py", line 208, in reference_root_pose_reward
    cache, frame_ids = _select_reference_frame(env)
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/mdp/rewards_reference.py", line 173, in _select_reference_frame
    cache = ensure_reference_cache(env)
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/mdp/rewards_reference.py", line 134, in ensure_reference_cache
    cache = manager.refresh_from_env(env)
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/batch_mpc_planner/manager.py", line 261, in refresh_from_env
    sub_terrain = subset_mpc_terrain(terrain, selected_ids)
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/batch_mpc_planner/terrain.py", line 40, in subset_mpc_terrain
    height = terrain.height_map.index_select(0, ids)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 89.07 GiB. GPU 2 has a total capacity of 23.55 GiB of which 10.75 GiB is free. Including non-PyTorch memory, this process has 12.74 GiB memory in use. Of the allocated memory 3.98 GiB is allocated by PyTorch, and 16.74 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/stable/notes/cuda.html#environment-variables)

python Go2Pvcnn/scripts/train.py   --headless   --device cuda:2   --num_envs 4096   --max_iterations 1   --experiment teacher_elevation_trajectory   --planner-backend mpc 你帮我启动这条命令，然后直到它能跑完，现在会报错，环境采用env_isaacsim          *  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim

python Go2Pvcnn/scripts/train.py   --headless   --device cuda:2   --num_envs 4096   --max_iterations 1   --experiment teacher_elevation_trajectory   --planner-backend mpc，这个命令行会不会完整启动mpc planner？还是相比together，会省去一些东西

(env_isaacsim) lhy@enine:~/testPvcnnWithIsaacsim$ python \
Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:2 \
  --num_envs 1 \
  --terrain task \
  --planner-backend mpc
Traceback (most recent call last):
  File "/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn/extension/viz/go2_foostep_planner.py", line 23, in <module>
    from extension.batched_together_planner.types import HARD_REASON_NAMES, TogetherPlannerStatus
ModuleNotFoundError: No module named 'extension.batched_together_planner'

# Selected text:

## Selection 1
掉

## My request for Codex:
如图是我启动可视化，足端乱飞，你觉得是什么问题，做些先分析，做测试，同时isaaclab的headless启动你需要使用参考当前viewer的代码来启动，来做测试

上述足端乱飞已经解决，但是给了速度，root会移动，但是四条腿不会跟着移动，你需要修改先参考viewer的使用isaaclab的路径，做测试来排查，排查出来后，再修改代码，cuda2和3只需要用一个测试就可以，不用测试两次

现在发现了新的问题，足端稍微摆动，但是达不到步态的效果，你先做测试复刻这个失败（无法走路的姿态），使用isaacsim，参考viwer的代码逻辑，然后使用测试层来排查问题，你可以先测试，然后不改代码，把测试结果和我聊一下，再决定怎么改

这个情况是不是说明loss没有发挥作用，需要调loss权重，还是代码出现了功能性错误？你查看代码分析一下

可以

刚开始会摆动，之后就不摆动了，同时按QE的时候，应该是yaw的转动，但是绕x轴也转动了，转得特别大，你再更长时效性得测试复刻这个问题

可以

没有任何变化，你去做测试，还是有倾斜，足端只会抽出，没有步态，先复刻

我按了QE之后变成图片这样，但是base确实是保持水平得，但是足端特别奇怪，同时足端的问题，我也能理解，它虽然再forward情况下，前进了，但是没有征程走路的步态，你先做测试复刻

可以

你现在觉得是哪里出问题了，出五个可能的方向

你每个方向都测试一下，排查一下哪个解决了问题

可以

现在往前走勉强看起来是走路，但是横移很奇怪，还是一样，腿不follow base，做测试复刻，QE的情况也测试一下

你提一下最有可能的问题，提5个

你测试一下，哪个优化最好

我希望横移，前进，后退，旋转，都要有，你觉得要怎么做，从上述测试来说

可以

随着启动越来越久（replan），姿态越来越奇怪，做测试复现这个

主要是足端越跑越偏，base没有问题

你分析5个问题方向吧，是什么造成的

足端锚点是什么意思，如果要修改，怎么改

你对上述5个方向都测试一下，比对一下效果

这个abs优化是什么

漂移指的是什么意思

foot_err均值也是漂移吗

有什么根本解决这个问题的方法吗，你上述测试，明显还是存在问题

你能把这个聊天记录导入/mnt/mydisk/lhy/testPvcnnWithIsaacsim/notes/talk/重构planner优化轨迹.md，方便下一个agent读取吗

不是要笔记，而是完整的聊天内容
```

## 当前上下文中可恢复的 assistant / 执行摘要

下面不是逐字聊天原文，而是当前上下文里可恢复的执行结果，供下一个 agent 对齐状态。

### 1. 初始设计方向

用户一开始说的是 “mcp”，随后更正为 “mpc”。核心目标是把旧的 `batched_together_planner` 思路重构为新的梯度下降式 MPC planner，并接入 viewer 与 RL 训练链路。

已接受的设计方向：

- 新后端名为 `mpc`，通过 `--planner-backend mpc` 选择。
- 代码路径为 [Go2Pvcnn/extension/batch_mpc_planner](../../Go2Pvcnn/extension/batch_mpc_planner)。
- 接入 viewer：[Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)。
- 接入训练：[Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)。
- 强化学习配置可直接导入 planner 配置类，像高程图配置一样可调。
- 全部训练热路径应在 GPU 上，不走 CPU hot path。
- 支持 4096 env。
- 每个 env 按 reset、command 改变、`replan_interval_steps` 到期异步重规划；不需要所有 env 同步重规划。
- 不保留旧 together planner 的罗嗦旧概念。

MPC 优化变量按用户确认：

- `root_pos_residual [B,T,3]`
- `root_rpy_residual [B,T,3]`
- `foot_pos_residual [B,T,4,3]`
- `contact_logits [B,T,4]`

解释过的关键点：

- `contact_state: bool [B,T,4]` 表示每个 env、每个 horizon step、每条腿是否接触地面。
- 用户要求 stance/swing 时序不能固定，也要进入梯度下降；因此使用 `contact_logits [B,T,4]`。
- trajectory 和 touchdowns 都应该放在 `foot_pos_residual [B,T,4,3]` 中表达。
- 每条 loss 应该像 RL reward 一样有可调权重，同时 loss 内部参数也暴露到配置类。
- 硬诊断层用于测试/验证，可以 `enable`；训练时可 disable 加速。

### 2. 子代理与设计审查

用户要求使用 subagent 审查设计，审查范围包括：

- 需求完成度。
- 强化学习并行性。
- 测试不复用性，新写测试。
- IsaacLab 适配性。
- 测试必须在 IsaacLab 里进行。
- 不用 CPU，只用 GPU。
- 使用硬检测层做测试。
- 配置类与实现类。
- 文件结构设计。

之后按这些结论修改了 design，并把实现路径固定到 [Go2Pvcnn/extension/batch_mpc_planner](../../Go2Pvcnn/extension/batch_mpc_planner)。

### 3. 已实现或曾经确认的文件方向

当前上下文显示，之前已经添加/修改过这些方向：

- [Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
- [Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
- [Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
- [Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
- [Go2Pvcnn/extension/batch_mpc_planner/optimizer.py](../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py)
- [Go2Pvcnn/extension/batch_mpc_planner/variables.py](../../Go2Pvcnn/extension/batch_mpc_planner/variables.py)
- [Go2Pvcnn/extension/batch_mpc_planner/types.py](../../Go2Pvcnn/extension/batch_mpc_planner/types.py)
- [Go2Pvcnn/extension/batch_mpc_planner/adapter.py](../../Go2Pvcnn/extension/batch_mpc_planner/adapter.py)
- [Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
- [Go2Pvcnn/extension/batch_mpc_planner/diagnostics.py](../../Go2Pvcnn/extension/batch_mpc_planner/diagnostics.py)
- `Go2Pvcnn/extension/batch_mpc_planner/losses/`
- [Go2Pvcnn/extension/trajectory_manager_factory.py](../../Go2Pvcnn/extension/trajectory_manager_factory.py)
- [Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py)
- [Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
- [Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)

### 4. 用户给出的运行环境和测试要求

用户明确要求：

- IsaacLab 环境：`/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
- 需要启动 IsaacLab 做 headless 测试。
- GPU 测试用 `cuda:2` 或 `cuda:3`；后续用户说只需要用一个测，不用两个都测。
- viewer 测试要参考当前 viewer 代码的 IsaacLab 启动方式。
- 测试层要能复现用户看到的问题，再决定是否改代码。

用户给过的训练命令：

```bash
python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:2 \
  --num_envs 4096 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory \
  --planner-backend mpc
```

用户给过的 viewer 命令：

```bash
python Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:2 \
  --num_envs 1 \
  --terrain task \
  --planner-backend mpc
```

### 5. 曾经出现并修复过的问题

当前上下文可恢复的问题链：

- `train.py --planner-backend mpc` 初始 CLI 不认识该参数，后来已接入。
- 4096 训练时 `subset_mpc_terrain` OOM，错误表现为试图分配 `89.07 GiB`；原因与 terrain ray shape / `index_select` 维度误解有关，后来修复 terrain ingestion。
- 直接运行 viewer 时导入失败：`ModuleNotFoundError: No module named 'extension.batched_together_planner'`，后来修复 direct script bootstrap/import path。
- viewer camera `.numpy()` grad crash 和二次 replan autograd graph reuse 问题，后来通过 detach / inference-mode 处理。
- “足端乱飞”一度被定位为 joint order mismatch，后来修复 `_mpc_state_from_env` 中 robot-order joints 到 planner-order 的转换，以及 playback 时 planner-order 到 robot-order 的转换。
- “root 会移动但四条腿不跟着移动”后来通过输出 IK-solved `joint_angles[B,T,12]` 修复。
- “足端稍微摆动但达不到步态”后来加过 gait coupling losses、command-relative nominal foot seed、contact schedule loss，但效果只是部分改善。
- `Q/E` yaw 视觉上像 roll，一部分来自 xyzw/wxyz 显示混淆；viewer 后来主打印 `wxyz`，旧 `xyzw` 只做 debug。
- command matrix 中 backward/yaw 的问题一度与 diagnostics 中动态 quadrant 重排足端有关；后来 viewer readback 固定为 planner leg order `FL/FR/RL/RR`，MPC manager 做 foot-id canonicalization。之后 command matrix 通过过。

### 6. 曾经确认过的验证结果

当前上下文里记录过的验证证据：

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
# 12 passed
```

```bash
MPC_TEST_DEVICE=cuda:2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -q
# 8 passed, 1 skipped
```

4096 训练命令曾经跑通过 1 iteration，并产出 `model_0.pt`：

```bash
python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:2 \
  --num_envs 4096 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory \
  --planner-backend mpc
```

注意：当前上下文也记录了“true 4096 runtime counters still unstable due high-scale Isaac/PhysX/CUDA issues”，所以下一个 agent 不要只看 unit test，要按用户要求做 IsaacLab headless runtime 复现。

## 最近一轮问题：长时间 replan 后足端越跑越偏

用户描述：

```text
随着启动越来越久（replan），姿态越来越奇怪，做测试复现这个

主要是足端越跑越偏，base没有问题
```

已做过的可恢复分析：

- 先做过长时 base roll/pitch 复现，60 cycles forward/lateral/yaw 和 72 cycles command switching；base roll/pitch 保持 `0.0000`。
- 用户澄清主问题不是 base，而是足端越跑越偏。
- 后续做了 120-cycle foot drift focused test。

### 关键定义

`漂移`：

- 计算 `rel = foot_pos_world - base_pos_world`。
- 再看 `rel_rad = ||rel||`。
- `drift = rel_rad_last - rel_rad_first`。
- 含义是足端布局相对 base 慢慢跑远/跑偏，不是整台机器人在世界中移动。

`foot_err`：

- 计算 `||actual_foot_pos - planned_foot_pos||`。
- 它不是漂移本身。
- 如果 drift 大但 foot_err 小，说明 planner 自己在输出漂移的足端布局，仿真基本照着执行了。
- 如果 foot_err 大但 drift 小，说明 playback/IK/关节限制可能没有跟上 planner。

`abs漂移均值`：

- 临时评估指标，等于多个 command case 的 `abs(rel_rad_delta)` 平均值。
- 它不是 planner loss，只是这轮 A/B 对比时用于排序的指标。

### 120-cycle foot drift 测试结果

```text
forward:
  rel_rad: 0.3978 -> 0.4696
  drift: +0.0718
  foot_err: 0.0000 -> 0.0925
  mean foot_err: 0.0607

lateral_left:
  rel_rad: 0.4021 -> 0.3940
  drift: -0.0081
  foot_err stable: ~0.0465

backward:
  rel_rad: 0.3975 -> 0.4608
  drift: +0.0633
  final foot_err: 0.0462
```

结论：长时 replan 足端漂移存在，主要在 forward/backward；base 稳定。

## 曾经提出的 5 个可能方向

1. Replan first-frame foot anchor constraint 太弱。
2. Soft `contact_logits/prob` 没有和真实 stance/swing/contact 同步。
3. Foot trajectory losses 只看局部/平均，缺少长期 world/body anchor memory。
4. IK saturation / joint limit clipping 造成 plan-to-playback mismatch，并通过反馈累积。
5. Command-driven nominal 太激进，尤其 forward/backward stride。

## 5 个方向的 A/B 测试结果

当时第一版批量脚本失败，后来用独立 per-variant IsaacLab 启动，`cuda:2`，每个 60 cycles。

| Variant | abs_drift_mean | err_mean | 结论 |
| --- | ---: | ---: | --- |
| baseline | 0.0484 | 0.0457 | 基线 |
| dir1_anchor_proxy | 0.0484 | 0.0455 | 无明显改善 |
| dir2_contact_sync | 0.0485 | 0.0456 | 无明显改善 |
| dir3_longterm_follow | 0.0430 | 0.0582 | 漂移略降但 foot error 变差，不建议单独用 |
| dir4_ik_guard | 0.0484 | 0.0455 | 无明显改善 |
| dir5_nominal_less_aggressive | 0.0285 | 0.0405 | 最好，但 backward 仍未根治 |

更细的指标：

```text
baseline:
  forward drift +0.0718, foot_err_mean 0.0577, foot_err_last 0.0598
  backward drift +0.0627, foot_err_mean 0.0355, foot_err_last 0.0390
  lateral_left drift -0.0106, foot_err_mean 0.0438, foot_err_last 0.0450

dir1_anchor_proxy:
  forward +0.0720
  backward +0.0627
  lateral -0.0106

dir2_contact_sync:
  forward +0.0727
  backward +0.0622
  lateral -0.0107

dir3_longterm_follow:
  forward +0.0468
  backward +0.0538
  lateral +0.0283

dir4_ik_guard:
  forward +0.0722
  backward +0.0622
  lateral -0.0107

dir5_nominal_less_aggressive:
  forward +0.0201, foot_err_mean 0.0509, foot_err_last 0.0510
  backward +0.0612, foot_err_mean 0.0275, foot_err_last 0.0326
  lateral_left -0.0043, foot_err_mean 0.0432, foot_err_last 0.0441
```

解释：

- forward 漂移主要像是 nominal stride 太激进。
- lateral_left 已经相对稳定。
- backward 仍明显未解决，说明不是简单调权重/调 stride 能完全解决。
- 用户指出“上述测试，明显还是存在问题”，因此后续应做结构性修复，而不是只调参数。

## 对“足端锚点”的解释

足端锚点指 stance foot 在世界坐标系中的“应该钉住的位置”。

四足步态里，脚处于 stance 时不应该跟着 base 一起滑；base 往前走时，站立脚应该近似留在世界位置，直到该脚进入 swing 后才抬起并落到新 foothold。

当前问题像是每次 replan 都重新生成一段 nominal foot trajectory，缺少“上一轮这个脚已经站在哪里”的长期记忆，于是足端布局会一点点被新规划推偏。

## 根本解决方向

当前上下文里最后的结论是：这不是单纯 loss 权重问题，也不是只靠硬诊断层能解决；更像是 MPC 缺少 persistent gait/foothold memory。

建议的结构性修改：

1. 在 manager 中维护 per-env 持久状态：

```text
stance_anchor_w[B,4,3]
last_touchdown_w[B,4,3]
phase[B]
contact/stance state from previous horizon，可选
```

2. stance 脚只在 touchdown 事件更新 world anchor，不要每次 replan 都重置。

3. stance 阶段的脚应围绕 world anchor 保持，swing 阶段才生成目标 touchdown。

4. `build_nominal_trajectory()` 不应该每次 replan 都从 `t=0` 重新开始 gait phase；phase 应该由 manager 持有并随 step 推进。

5. reset / command change 时允许 resync phase 或重置对应 env 的 gait memory，但不是所有 env 同步重置。

6. MPC 仍然优化 residuals：

```text
root_pos_residual
root_rpy_residual
foot_pos_residual
contact_logits
```

但 residual 应围绕持久 gait/foothold state 构造，而不是围绕每次新生成的无记忆 nominal。

7. 诊断层增加：

```text
stance foot slip threshold
foot-base radius threshold
touchdown update too frequent
joint-limit near boundary
```

## 短期缓解参数

当时提出过短期缓解，但不认为这是根治：

```text
nominal_stride_scale: 0.9 -> 0.7
nominal_max_stride_m: 0.18 -> 0.13
nominal_backward_stride_scale: 0.70 -> 0.55
```

这些参数能降低 forward 漂移，但 backward 仍存在明显问题。

## 下一个 agent 建议接手方式

1. 先不要继续纯调 loss 权重；优先写一个 IsaacLab headless runtime test，复现 120-cycle foot drift，并固定阈值。

2. 测试要使用用户指定环境：

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python
```

3. 测试设备用 `cuda:2` 或 `cuda:3` 中一个即可。

4. 实现 manager-owned persistent gait/foothold memory，然后把它传入 nominal/planner。

5. 修改后用同一个 drift 测试比较：

```text
forward
backward
lateral_left
yaw_left / yaw_right
command switching
```

6. 验证时不要只看 base roll/pitch；用户已经明确 base 没问题，核心是 foot relative-to-base drift 和步态质量。

7. 注意 leg order 固定为 planner order：

```text
FL / FR / RL / RR
```

8. 不要引入 CPU hot path；4096 env 训练路径要保持 GPU batch。

## 相关入口链接

- Planner 新实现：[Go2Pvcnn/extension/batch_mpc_planner](../../Go2Pvcnn/extension/batch_mpc_planner)
- Viewer：[Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
- Train：[Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
- Trajectory manager factory：[Go2Pvcnn/extension/trajectory_manager_factory.py](../../Go2Pvcnn/extension/trajectory_manager_factory.py)
- RL trajectory cfg：[Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py)
- IsaacLab/WebRTC 命令说明：[notes/human/human-16-isaaclab-applauncher-webrtc-migration.md](../human/human-16-isaaclab-applauncher-webrtc-migration.md)
- Planner 阅读入口：[notes/human/human-08-extension-planner-reading-guide.md](../human/human-08-extension-planner-reading-guide.md)
- Planner 映射：[notes/human/human-09-extension-planner-mapping.md](../human/human-09-extension-planner-mapping.md)

##  最新聊天内容
现在的问题本质是：每次 replan 都允许 foot trajectory 重新围绕当前状态和 nominal 生成，所以小误差会一轮轮写进下一轮初值，形成随机游走/偏移。要根治，需要让“脚该站在哪里”有一个跨 replan 的稳定参考，而不是每段 horizon 自己决定。

我建议的根本方案：

引入 per-env foot anchor memory
每个 env 保存 stance_anchor_w[B,4,3]、last_touchdown_w[B,4,3]、phase[B]。stance 脚锚定到上一次 touchdown，不随 replan 重算。

只在 touchdown 事件更新 foothold
脚不是每次 replan 都重新找位置，而是 swing 脚到 touchdown 时才更新目标落脚点。stance 期间保持世界坐标锚点。

MPC 只优化 residual，不改 gait 状态语义
MPC 可以调 root、swing 高度、foot residual、contact logits，但 stance 脚有强约束：不能长期滑走。这样既保留梯度下降，又避免 foot layout 随 replan 漂。

把 stance/swing phase 做成 manager-owned persistent state
不能让每次 build_nominal_trajectory() 从 t=0 重启 gait phase。phase 必须由 manager 跨 step 推进；reset/command-change 才重新同步。

增加投影/诊断层作为 safety net
训练时可以关闭硬诊断，但开发/测试时打开：

stance foot slip 超阈值报警
foot-base 半径超阈值报警
touchdown 更新过频报警
joint-limit 贴边报警
短期修法是调 nominal_stride_scale/max_stride，能缓解 forward，但你说得对：它不是根治。根治必须让足端落点有跨 replan 的“记忆”和“事件更新规则”。现在每次 replan 都像一只金鱼重新决定脚在哪，跑久了当然会迷路。