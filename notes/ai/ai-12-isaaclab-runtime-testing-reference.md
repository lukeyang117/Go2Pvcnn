# AI Isaac Lab Runtime Testing Reference

## Navigation

- doc role: AI runtime testing/debugging reference for Isaac Lab viewer and planner work
- paired human doc: none
- previous: [ai-11-extension-trajectory-reward.md](ai-11-extension-trajectory-reward.md)
- next: [ai-13-batched-planner-swing-stance-ik-complexity.md](ai-13-batched-planner-swing-stance-ik-complexity.md)
- master index: [../index.md](../index.md)

## Purpose

总结 batched planner / viewer 在真实 Isaac Lab 运行时下做测试时遇到的典型问题、可复现命令、诊断套路和修复经验。

目标不是解释训练主线，而是给后续所有“需要真的把 Isaac Lab 拉起来”的测试工作一个稳定参考。

## Recommended Runtime

- preferred conda env: `env_isaaclab`
- preferred interpreter: `env_isaaclab` 自带 `python`
- do not assume当前 shell 的 Python 已经能导入 Isaac Lab

## Fast Rule

如果问题涉及下面任一层，就不要只靠纯单测判断：

1. `AppLauncher`
2. Isaac articulation / rigid object 写回
3. `height_scanner`
4. `robot.data.*` 实际状态
5. viewer playback / render sync
6. quaternion convention on Isaac boundary

这些问题必须至少补一次真实 runtime 复现。

## Preferred Commands

### 1. 验证环境是否可用

```bash
conda run -n env_isaaclab python -u - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
PY
```

### 2. 运行真实 runtime 诊断测试

```bash
conda run -n env_isaaclab pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py \
Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py -q
```

注意：

- 这类测试在非 Isaac Lab 环境里可能直接 `skip`
- 只看到 `skip` 不能说明 viewer/runtime 没问题

如果 `conda run` 下 here-doc 的 stdout 不稳定，再切成：

```bash
conda activate env_isaaclab
python -u ...
```

### 3. 直接启动 viewer 做 deterministic 复现

```bash
conda activate env_isaaclab
python -u \
Go2Pvcnn/extension/viz/go2_foostep_planner.py \
--headless \
--num_envs 1 \
--terrain flat \
--n-frames 50 \
--plan-dt 0.02 \
--scripted-command "0 0 -0.3" \
--scripted-command-cycles 8
```

`--scripted-command` / `--scripted-command-cycles` 适合做 fixed-command 多轮重规划复现，避免键盘输入噪声。

## Problems Encountered

### 1. `real runtime` 在普通 pytest 环境里经常 `skip`

现象：

- `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
- `Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py`

在普通 shell 环境里只看到 `skip`

原因：

- 这些测试依赖真实 Isaac Lab `AppLauncher`
- 还依赖 CUDA / rendering / scene startup

解决：

- 用 `env_isaaclab` 跑
- 不要把 `skip` 误判成“问题不存在”

### 2. `conda run` 跑 here-doc 时输出不稳定

现象：

- `conda run -n env_isaaclab python - <<'PY' ... PY` 有时能跑但 stdout 不完整

解决：

- 优先直接调用环境里的 Python：

```bash
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -u ...
```

- 对长启动过程，加 `-u`

### 3. viewer 的 root quaternion 写回约定错了

现象：

- planner 结果看起来正常
- actual base state / visual result 不对
- 进一步查 `write_root_pose_to_sim()` 后发现 root pose 写回边界有问题

最终结论：

- Isaac Lab `write_root_pose_to_sim()` 需要 `(w, x, y, z)`
- 之前 viewer 回放把 planner quaternion 从 `wxyz` 转成了 `xyzw` 再写回

代码依据：Isaac Lab 源码里的 `articulation.py::write_root_pose_to_sim(...)`

修复位置：

- [Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)

测试位置：

- [Go2Pvcnn/tests/test_batched_planner_runtime_path.py](../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py)

### 4. `robot.data.root_quat_w` 的真实约定必须现场确认，不能凭记忆

现象：

- 只看变量名 `root_quat_w` 很容易猜它是 `xyzw`
- 真实 runtime 读取后发现当前资产/runtime 边界里更接近 `wxyz`

解决：

- 在 viewer 里同时打印：
  - `actual_quat_raw`
  - `actual_rpy_if_wxyz`
  - `actual_rpy_if_xyzw`
  - `plan_rpy`

当 `actual_rpy_if_wxyz` 和 `plan_rpy` 对齐时，说明 raw quaternion 就该按 `wxyz` 解读。

### 5. 足端对比一开始出现 `0.5m+` 大误差，但不是 planner 真错

现象：

- `joint_err` 非常小
- `foot_err` 却极大

原因：

- Isaac 资产里的 `body_pos_w[:, foot_ids]` 顺序不能直接等同 planner 的 `FL/FR/RL/RR`
- 直接按数组下标比较会得到假的大误差

解决：

- 按物理象限重排 actual foot：
  - front-left
  - front-right
  - rear-left
  - rear-right

修复位置：

- [Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)

### 6. `foot_ids` 可能是 `list`，也可能是 tensor

现象：

- 诊断代码里直接 `foot_ids.tolist()` 可能报：
  `AttributeError: 'list' object has no attribute 'tolist'`

解决：

- 增加统一 helper，把 `list / tuple / tensor` 都转成 `list[int]`

修复位置：

- [Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)

## Runtime Diagnostics That Helped

这次最有用的不是“再猜一个修复”，而是把 viewer 变成会自己报状态的诊断器。

建议至少打印这两组：

### Actual base

- `actual_pos`
- `actual_quat_raw`
- `actual_rpy_if_wxyz`
- `actual_rpy_if_xyzw`
- `plan_pos`
- `plan_rpy`

用法：

- 快速判定“看起来歪”到底是 base 真歪了，还是边界写回/显示解释错了

### Actual kinematics

- `joint_err_max`
- `joint_err_mean`
- `foot_err_max`
- `foot_err_mean`

用法：

- base 正常时，继续判断是 joint 写回问题，还是 foot/body mapping 问题

## Interpretation Pattern

### Case A

- `actual_rpy_if_wxyz ~= plan_rpy`
- `joint_err` 很小
- `foot_err` 很大

优先怀疑：

- foot body 顺序
- foot point定义不一致
- 视觉 mesh / body anchor 偏置

### Case B

- `actual_rpy_if_wxyz` 和 `plan_rpy` 明显不一致

优先怀疑：

- quaternion order
- `write_root_pose_to_sim()` 调用约定
- root state sync / render sync 顺序

### Case C

- planner diagnostics 本身就出现大的 roll/pitch 漂移

优先怀疑：

- terrain estimator
- support contact 使用方式
- viewer multi-cycle state chaining

## Multi-Cycle Replan Rule

对 turn / viewer 漂移类问题，不要只看单轮 `cycle=0`。

至少看：

1. `cycle=0`
2. `cycle=1`
3. `cycle=2`
4. `cycle=4+`

原因：

- 单轮经常只暴露轻微偏差
- 真正的 viewer bug 常常要多轮 state chaining 后才放大

## Current Recommended Workflow

以后只要是 viewer/runtime 类问题，建议固定顺序：

1. 跑纯单测，确认 helper 和边界函数没明显坏
2. 跑 `env_isaaclab` 下的 real runtime diagnostics
3. 跑 scripted viewer command，多轮重规划
4. 先看 `ActualBase`
5. 再看 `ActualKinematics`
6. 最后才做肉眼判断截图/视频

不要一开始只靠视频印象决定根因。

## Relevant Files

- [Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
- [Go2Pvcnn/tests/test_viz_playback.py](../../Go2Pvcnn/tests/test_viz_playback.py)
- [Go2Pvcnn/tests/test_batched_planner_runtime_path.py](../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py)
- [Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
- [Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py](../../Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py)
- [Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
- [Go2Pvcnn/extension/batched_planner/trajectory.py](../../Go2Pvcnn/extension/batched_planner/trajectory.py)
- [Go2Pvcnn/extension/batched_planner/terrain_estimator.py](../../Go2Pvcnn/extension/batched_planner/terrain_estimator.py)

## One-Line Memory Aid

Isaac Lab viewer 调试不要只看画面，先确认三件事：

1. root quaternion 写回顺序对不对
2. actual base state 和 planner state 是否一致
3. foot body 顺序是不是和 planner leg order 一致
