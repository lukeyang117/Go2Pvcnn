# T301 Viewer R-Key Grounded Reset

## Current State

- 新增独立 viewer 交互任务：`R` 键重置不再把 Go2 传回初始世界位置。
- 当前实现方向已改为：
  - 保持当前 root 世界 `xy` 位置不变
  - 保持当前 root 朝向不变
  - 仅把关节状态恢复到初始站立姿态
  - 再根据 semantic scanner / 高程图把足端重新贴回地面
- 本轮已完成 viewer helper 代码修改与轻量单测，尚未补一个真实 headless runtime 的针对性 `R` 行为断言。
- 新增 `T301b` step-mode 子节点：viewer 默认连续播放，运行时按 `M` 切换单帧模式，再按 `M` 回连续播放；单帧暂停时 IsaacLab/Kit 窗口继续 render/pump，机器狗状态和轨迹 marker 只在空格节拍更新；`W/A/S/D/Q/E/R` 仍监听，其中运动命令只锁存为下一段轨迹输入，当前轨迹走完前不切换。轻量 pytest、`env_isaacsim` pytest/compile、真实 headless viewer smoke 已通过。

## Open Children

- [T301a](#t301a-viewer-r-reset语义改造与helper验证): helper 语义与局部验证已完成，待补 runtime 级针对性断言。
- [T301b](#t301b-step-mode空格单帧播放与轨迹边界命令切换): viewer step-mode 改为运行时 `M` 切换，按空格推进一帧，命令变化延迟到当前轨迹结束后生效；实现与 headless IsaacLab smoke 已完成，人工 livestream 手感仍可选验收。

## Closed Children Archive

- 无。

## Related Logs

- [../log/2026-05-15-2045-t301-viewer-r-key-grounded-reset.md](../log/2026-05-15-2045-t301-viewer-r-key-grounded-reset.md)
- [../log/2026-05-22-0004-t301b-step-mode-viewer-play.md](../log/2026-05-22-0004-t301b-step-mode-viewer-play.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `c92627d` plus working tree changes verified through [../log/2026-05-22-0004-t301b-step-mode-viewer-play.md](../log/2026-05-22-0004-t301b-step-mode-viewer-play.md)
- Current Work Ref: `working tree on top of c92627d (T301b step-mode viewer/play)`
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)

## Next Step

- 在真实 IsaacLab headless runtime 里补一个更贴近交互语义的 targeted reset 断言：
  - reset 前先人工改 root `xy/yaw`
  - 执行 viewer reset helper
  - 验证 root `xy/yaw` 保持不变、joint 恢复初始站姿、足端高度接近地面
- 可选人工验收 `T301b`：
  - 远程服务器用 `--headless --livestream 2` 启动 viewer
  - 浏览器连接 WebRTC 后在终端按空格逐帧推进
  - 按 `M` 进入单帧模式，再按 `M` 回连续播放
  - 在当前轨迹未播完时按 `W/A/S/D/Q/E`，确认只影响下一段轨迹

## Node Details

### T301a viewer R reset语义改造与helper验证

- status: `verify`
- why-created:
  - 用户要求 `R` 键恢复 Go2 状态，但不能回到初始世界位置和初始朝向。
  - 正确语义应当是“留在当前地方，只恢复站姿，并按地形落脚”。
- implementation summary:
  - `ViewerResetSnapshot` 仅保留初始关节状态，不再缓存初始 root pose/velocity
  - reset 时先保留当前 root pose
  - `env.reset()` + warmup 后显式回写当前 root pose 与初始 joint state
  - 使用 scanner 构造的本地 terrain 对四足当前位置采样高程，整体修正 root z 使足端贴地
  - 额外清零 `base_velocity` command，避免 reset 后旧命令残留
- evidence:
  - 本地 `python -m pytest Go2Pvcnn/tests/test_viewer_reset.py -q` 通过
  - `env_isaacsim` 下同一测试通过
  - `py_compile` 与 `git diff --check` 通过
- remaining risk:
  - 还没有真实终端按键 `R` 的 end-to-end 行为日志
  - 当前贴地策略使用四足平均高程修正 root z；若后续要适配更激烈地形，可能需要升级成 support-plane 级 reset

### T301b step-mode空格单帧播放与轨迹边界命令切换

- status: `verify`
- why-created:
  - 用户最初希望显式命令行参数启动 step-mode，后续改为 viewer 默认连续播放，运行时按 `M` 切换单帧/连续模式。
  - 终端监听空格：单帧模式下每按一次空格只播放/推进一帧；不按就只保持窗口 render/pump。
  - `W/A/S/D/Q/E/R` 键仍要监听；运动命令按下后只作为下一段轨迹输入，除非当前轨迹播放完，否则不切换轨迹。
- implementation plan:
  1. 在 viewer 侧新增可运行时切换的 `ViewerStepGate`；`TerminalTeleop` 监听 `M` 产生 `mode_toggle_requested`，空格产生一次性 `step_requested`。
  2. 修改 `_viewer_loop_need_replan`，支持 `defer_command_replan_until_trajectory_end=True`；仅当 step gate 运行时启用时，命令变化不触发当前轨迹中途重规划，只有 `result is None`、轨迹耗尽、或 `R` reset 触发重规划。
  3. viewer 主循环在 `_viewer_direct_playback_step` 前检查 step gate；未按空格时只 poll 键盘并继续 render/update IsaacLab 窗口，不播放下一帧。
  4. viewer 轨迹 marker 和机器狗状态使用同一个空格节拍更新；未按空格时不刷新到下一段/下一帧轨迹点。
  5. `play.py` 保留显式 `--step-mode` 和终端空格 gate；本次用户要求只改 viewer 可视化运行时切换。
  6. 先用轻量 pytest 覆盖 “`M` 可运行时切换 step gate”“命令变化只在 step gate 启用时延迟”“空格一次放行一帧”“暂停时窗口继续 render/update”“visualizer 只在放行帧更新”；再用 `env_isaacsim` 跑同一测试、`py_compile`，最后启动一次 headless IsaacLab viewer smoke 验证默认连续、`M` 进入单帧、空格推进、`M` 回连续。
- acceptance:
  - viewer 不再需要 `--step-mode`；默认连续播放。
  - viewer 按 `M` 后，空格是唯一机器狗/轨迹 marker 帧推进触发；`W/A/S/D/Q/E` 可在暂停期间改变下一段命令；再按 `M` 回连续播放。
  - 不按空格时 IsaacLab/Kit 窗口仍继续 render/pump，可控制窗口/相机/WebRTC。
  - 当前轨迹 `playback_frame < result.num_frames` 时，运动命令变化不触发重规划；轨迹结束后使用最新锁存命令规划下一段。
  - `R` 仍即时 reset。
- evidence:
  - local `python -m pytest Go2Pvcnn/tests/test_viewer_reset.py -q`: `12 passed`
  - `env_isaacsim` `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_viewer_reset.py -q`: `12 passed`
  - `env_isaacsim` `py_compile`: exit `0`
  - `git diff --check`: exit `0`
  - real IsaacLab headless viewer smoke without `--step-mode`: scene/environment created, default continuous playback reached `[Viewer][Playback] path=render+scene_sync`, PTY-fed `m` printed `[Viewer][StepMode] enabled`, PTY-fed space advanced in single-frame mode, PTY-fed `m` printed `[Viewer][StepMode] disabled`, Ctrl-C exited without traceback
- remaining risk:
  - 真实人工终端按键节奏难以完全自动化；自动验证以 helper/parser/headless startup 为主，最终手感需要人工连 livestream 或本地 GUI 试用。
