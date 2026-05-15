# T301 Viewer R-Key Grounded Reset

## Purpose

- 为 `Go2Pvcnn/extension/viz/go2_foostep_planner.py` 增加符合用户要求的 `R` 键 reset 行为：
  - 保持当前世界位置
  - 保持当前朝向
  - 恢复初始关节站姿
  - 根据 scanner / 高程图把足端放回地面

## Stage

- viewer interaction / IsaacLab planner visualization

## Related Todo

- [../todo/T301-viewer-r-key-grounded-reset.md](../todo/T301-viewer-r-key-grounded-reset.md)

## Command / Procedure

1. 修改 viewer reset helper 逻辑：
   - `ViewerResetSnapshot` 仅保存初始 `joint_pos/joint_vel`
   - reset 前保存当前 `root_pos/root_quat`
   - reset 后回写当前 root pose + 初始 joint state
   - 清零 `base_velocity` command buffer
   - 用 scanner 构造 terrain 并按当前四足位置采样地面高程，修正 root z
2. 新增轻量测试文件：
   - `Go2Pvcnn/tests/test_viewer_reset.py`
3. 运行验证：
   - `python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_viewer_reset.py`
   - `python -m pytest Go2Pvcnn/tests/test_viewer_reset.py -q`
   - `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_viewer_reset.py -q`
   - `git diff --check -- Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_viewer_reset.py`

## Input Conditions

- 工作树已包含用户正在推进的 T300/T300e MPC 修改，未回退任何无关改动
- viewer 目标路径：`Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- 用户后续澄清了 reset 语义：
  - root 世界位置不要回去
  - yaw/朝向保持不变
  - 只恢复 joint 初始状态
  - 足端要按高程图落地

## Key Metrics

- `py_compile`: exit `0`
- local pytest: `3 passed`
- `env_isaacsim` pytest: `3 passed`
- `git diff --check`: exit `0`

## Result

- pass with scoped verification

## Conclusion

- `R` 键 reset 的内部 helper 语义已改成用户要求的版本：
  - reset 不再回到启动点
  - 保留当前 root `xy/yaw`
  - 关节恢复初始站姿
  - root z 根据四足当前位置对应的地形高程整体修正，使足端重新贴地
- 本轮验证覆盖 helper 与轻量测试层；尚未补真实终端按键 `R` 的 headless runtime 行为日志。

## Follow-up

- 增加一个真实 IsaacLab runtime targeted test：
  - 人工改 root pose
  - 调用 reset helper
  - 验证 root `xy/yaw` 保持、joint 恢复、足端贴地

## Git Refs

- Baseline Ref: `24b59cb`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
