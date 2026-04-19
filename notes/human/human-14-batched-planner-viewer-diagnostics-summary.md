# Human Batched Planner Viewer Diagnostics Summary

## 导航

- 文档类型：`human` 诊断总结
- 对应 AI 文档：暂无
- 上一篇：[human-13-batched-planner-swing-stance-ik-complexity.md](human-13-batched-planner-swing-stance-ik-complexity.md)
- 相关运行时主文档：[human-10-extension-planner-runtime.md](human-10-extension-planner-runtime.md)
- 总索引：[../index.md](../index.md)

## 一句话结论

基于当前已经跑通的真实 headless Isaac Lab diagnostics，可以先给出一个很强的判断：

- **`batched_generate_trajectory(...)` 主体不是当前 “WASD 不动 / QE marker 正常但狗姿态怪 / standstill 单腿怪” 的首要嫌疑。**
- 更可能的问题层在：
  1. **viewer 真正运行时的输入链**
     `stdin / terminal teleop / 焦点`
  2. **viewer 主循环里的 playback 同步契约**
     `write -> render` 和 `write -> scene.write_data_to_sim -> render -> scene.update`
     之间的差异
  3. **你当前工作区里 viewer 相关代码的本地修改**
     因为我们验证通过的是测试 fixture 所采用的“对齐 replay contract”的路径，而不是直接证明你现在手上的 `go2_foostep_planner.py` 主循环一定无误

## 本次证据分层

### 1. Isaac Lab 启动本身不是根问题

当前 diagnostics fixture 已经可以沿着 `go2_foostep_planner.py` 的真实启动思路创建：

- `TeacherElevationTrajectoryEnvCfg_PLAY`
- headless `AppLauncher`
- `gym.make("Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0")`
- `_attach_reference_manager_if_enabled(...)`

也就是说：

- **“Isaac Lab 根本起不来”不是现在的主结论。**
- 之前我们看到的启动异常，更像是：
  - GPU 资源选择不稳
  - PhysX / CUDA OOM
  - 启动后半初始化状态没有被清理好

这部分现在已经通过测试侧的 device candidate / resource skip / app cleanup 收紧了。

### 2. planner 对 `forward / lateral / yaw / standstill` 的响应是正常的

从实际输出看：

- `forward`
  - `input.command_vx_mean = +0.3000`
  - `base_approx.path_dx_mean = +0.1140`
  - `base_solve.path_dx_mean = +0.1140`
  - `result.path_dx_mean = +0.1140`
- `yaw_left`
  - `input.command_yaw_mean = +0.3000`
  - `base_approx.yaw_delta_mean = +0.1140`
  - `base_solve.yaw_delta_mean = +0.1140`
  - `result.yaw_delta_mean = +0.1140`
- `standstill`
  - `standstill_ratio = 1.0`
  - `path_dx_mean = 0`
  - `yaw_delta_mean = 0`

这说明：

- 只要 command 真正送进 planner，`WASD` 对应的平移命令和 `QE` 对应的 yaw 命令都会被 planner 正常消费。
- 所以 **“WASD 没反应”更像是 command 没进 planner，或者 viewer 真运行时没有触发 replan / 采用了别的输入状态源**，而不是 `trajectory.py` 不支持平移。

### 3. standstill 单腿异常并没有在 planner 输出里复现

当前 `standstill` 的 diagnostics 结果是：

- touchdown delta 基本为 `0`
- left/right touchdown mean 都接近 `0`
- `result.root_pos_w` 沿 horizon 是时间常量

这意味着：

- **“平地静止时有一条腿很怪”并不是 planner 在 standstill 结果里天然就算坏了。**
- 更像是：
  - playback 写回不同步
  - viewer 读写时序不一致
  - 或者某条腿在 viewer 链路里被错误解释

但当前测试也验证了 foot order：

- `foot_names == LEG_ORDER (FL, FR, RL, RR)`

所以在当前这套 runtime fixture 里，**腿顺序本身也没有复现出错。**

### 4. “QE 看起来正常但机器人不正常”更像 playback / viewer loop 问题

这次最关键的证据是：

```text
[playback-diag] case=forward frame_idx=7
root_pos_max_abs=0.000000
root_pos_mean_abs=0.000000
joint_pos_max_abs=0.000000
joint_pos_mean_abs=0.000000
```

它说明：

- 在 diagnostics fixture 采用的写回契约下，
  `planner result` 和 `robot readback` 可以做到几乎 `0` 误差对齐

而 diagnostics fixture 的 playback 契约是：

1. `_apply_direct_playback_to_robot(...)`
2. `scene.write_data_to_sim()`
3. `sim.render()`
4. `scene.update(physics_dt)`
5. 再读 `robot.data.*`

但是当前 viewer 主循环里，主路径仍然是：

- `_apply_direct_playback_to_robot(...)`
- `base_env.sim.render()`

也就是少了：

- `scene.write_data_to_sim()`
- `scene.update(...)`

所以当前最强嫌疑依然是：

- **planner 结果本身是正常的**
- **但 `go2_foostep_planner.py` 真正 viewer 主循环里的 playback 同步链路比 diagnostics fixture 更弱**
- 这非常符合你最早描述的现象：
  - touchdown / trajectory 看起来正常
  - 但 Go2 本体姿态不正常
  - 甚至像整体翻转或某条腿怪

## 对原始现象的重新归因

### 现象 A：`WASD` 时 touchdowns 和机器人都不动

当前更可能的归因顺序：

1. **viewer 输入链问题**
   - `TerminalTeleop` 走的是 stdin，不是 viewer 窗口事件
   - 如果焦点不在终端，或者按键没有真正变成 `teleop_cmd.values`，planner 根本不会收到平移命令
2. **viewer replan 触发条件没有被平移命令更新到**
   - `need_replan` 依赖 `last_cmd` 和 `teleop_cmd.values`
3. **不是 planner 不支持平移**
   - 因为 diagnostics 里 `forward / lateral` 已经明确跑通

### 现象 B：`QE` 时 touchdowns 正常，但狗不像走路，base 还会反

当前更可能的归因顺序：

1. **planner 本体正常**
   - `yaw_left` 的 `yaw_delta_mean`、touchdown delta 都是正常的
2. **robot playback / scene sync 链路有问题**
   - diagnostics 证明只要补齐 replay-style sync，数值可以对齐
   - viewer 主循环当前同步链不完整
3. **因此“marker 正常、robot 怪”最像 playback 问题**

### 现象 C：standstill 时一条腿很怪

当前更可能的归因顺序：

1. **不是 standstill planner 输出本身坏了**
2. **更像 viewer 写回或显示层问题**
3. 腿顺序在当前 runtime fixture 下没有复现异常，所以它不是当前第一嫌疑

## 这份结论的边界

我需要把边界说清楚：

- 这些测试证明的是：
  - 真实 headless Isaac Lab runtime 下
  - 如果 command 数值直接注入
  - 如果 playback 采用 replay-style sync contract
  - 那么 planner 输出和 robot readback 是健康的

- 它们**不能直接证明**：
  - 你当前工作区里的 `go2_foostep_planner.py` 主循环一定就没问题
  - 真实交互时 `stdin -> teleop_cmd.values` 一定正常
  - 所有 GUI / livestream / focus / render-only 路径都健康

所以更准确地说：

- **当前 tests 已经把“问题主要在 planner 核心算法里”的可能性显著压低了**
- **把嫌疑集中到了 viewer 输入链和 viewer playback 同步链**

## 当前最值得继续查的两件事

### 1. 直接对照 viewer 主循环和 diagnostics fixture 的 playback 差异

要对照的不是 planner，而是：

- diagnostics fixture：
  - `write_data_to_sim()`
  - `render()`
  - `scene.update(...)`
- viewer 主循环：
  - 目前只有 `render()`

这两条链的差异，最值得优先确认。

### 2. 在真实 viewer 里记录 `teleop_cmd.values`

因为 diagnostics 是直接数值注入 command，它证明的是“planner 收到命令以后没问题”。

所以如果你要继续定位 `WASD`，最应该先确认：

- 你按键时 `teleop_cmd.values` 到底有没有变
- `need_replan` 是否真的因为命令变化被触发

如果这里没有变，就不用先怀疑 planner。

## 相关测试入口

- viewer runtime diagnostics：
  - `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
- planner stage diagnostics：
  - `Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py`

推荐两个最有信息量的命令：

```bash
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py::test_planner_stage_diagnostics_emit_summary -s -q
```

```bash
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py::test_planner_output_vs_playback_emit_report -s -q
```

前者看：

- command 有没有进 planner
- stage 从哪一层开始变化

后者看：

- planner result 和 playback readback 是否数值对齐
