# T301 Viewer R-Key Grounded Reset

## Current State

- 新增独立 viewer 交互任务：`R` 键重置不再把 Go2 传回初始世界位置。
- 当前实现方向已改为：
  - 保持当前 root 世界 `xy` 位置不变
  - 保持当前 root 朝向不变
  - 仅把关节状态恢复到初始站立姿态
  - 再根据 semantic scanner / 高程图把足端重新贴回地面
- 本轮已完成 viewer helper 代码修改与轻量单测，尚未补一个真实 headless runtime 的针对性 `R` 行为断言。

## Open Children

- [T301a](#t301a-viewer-r-reset语义改造与helper验证): helper 语义与局部验证已完成，待补 runtime 级针对性断言。

## Closed Children Archive

- 无。

## Related Logs

- [../log/2026-05-15-2045-t301-viewer-r-key-grounded-reset.md](../log/2026-05-15-2045-t301-viewer-r-key-grounded-reset.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `24b59cb` plus working tree changes verified through [../log/2026-05-15-2045-t301-viewer-r-key-grounded-reset.md](../log/2026-05-15-2045-t301-viewer-r-key-grounded-reset.md)
- Current Work Ref: `working tree on top of 24b59cb (T301 viewer R-key grounded reset)`
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)

## Next Step

- 在真实 IsaacLab headless runtime 里补一个更贴近交互语义的 targeted reset 断言：
  - reset 前先人工改 root `xy/yaw`
  - 执行 viewer reset helper
  - 验证 root `xy/yaw` 保持不变、joint 恢复初始站姿、足端高度接近地面

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
