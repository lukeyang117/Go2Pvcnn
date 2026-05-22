# T301b Step-Mode Viewer/Play

## Purpose

- 为 viewer/play 增加显式 `--step-mode`：
  - 默认暂停
  - 终端空格按一次推进一帧 / 一个 env step
  - 暂停时 IsaacLab/Kit 窗口仍持续 render/pump，可继续控制窗口/相机/WebRTC
  - viewer 中 `W/A/S/D/Q/E/R` 仍监听
  - viewer 中运动命令变化只影响下一段轨迹，当前轨迹播完前不切换轨迹
  - viewer 轨迹 marker 和机器狗状态使用同一个空格节拍更新

## Stage

- viewer interaction / IsaacLab planner visualization
- play policy rollout / IsaacLab rendering loop

## Related Todo

- [../todo/T301-viewer-r-key-grounded-reset.md](../todo/T301-viewer-r-key-grounded-reset.md#t301b-step-mode空格单帧播放与轨迹边界命令切换)

## Command / Procedure

1. 增加 viewer helper 与 CLI：
   - `ViewerStepGate`
   - `--step-mode`
   - `TerminalTeleop.step_requested`
   - step-mode 下锁存 `W/A/S/D/Q/E` 命令
   - `_viewer_loop_need_replan(..., defer_command_replan_until_trajectory_end=True)`
2. 增加 play helper 与 CLI：
   - `_TerminalStepGate`
   - `--step-mode`
   - policy/env step 前等待空格
3. 补充 paused render pump 和 visualizer gate：
   - `_viewer_pump_paused_window`
   - `_viewer_update_visualizer_when_permitted`
   - `_pump_play_paused_window`
   - Ctrl-C terminal cleanup 防重入
4. 增加轻量测试：
   - step-mode 命令变化不打断当前 viewer 轨迹
   - 空格 gate 一次只放行一帧
   - play gate 默认关闭时不阻塞
   - viewer step-mode pause 时仍 render/update window
   - visualizer update 只在 frame permitted 时执行
5. 运行验证：
   - `python -m pytest Go2Pvcnn/tests/test_viewer_reset.py -q`
   - `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_viewer_reset.py -q`
   - `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/scripts/play.py Go2Pvcnn/tests/test_viewer_reset.py`
   - `git diff --check -- Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/scripts/play.py Go2Pvcnn/tests/test_viewer_reset.py notes/todo/T301-viewer-r-key-grounded-reset.md`
   - Headless IsaacLab viewer smoke:

```bash
timeout -s INT -k 20s 140s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --device cuda:0 \
  --num_envs 1 \
  --terrain task \
  --planner-backend together \
  --n-frames 50 \
  --plan-dt 0.02 \
  --warmup-steps 0 \
  --scripted-command "0.20 0.00 0.00" \
  --scripted-command-cycles 1 \
  --step-mode
```

## Input Conditions

- Baseline git ref: `c92627d`
- Server/headless IsaacLab through `env_isaacsim`
- Smoke used PTY input to feed spaces, then Ctrl-C for clean shutdown
- First attempted smoke with `--n-frames 5` reached AppLauncher but failed expected together horizon guard; rerun used legal `--n-frames 50`

## Key Metrics

- local pytest: `8 passed`
- `env_isaacsim` pytest: `8 passed`
- `env_isaacsim` `py_compile`: exit `0`
- `git diff --check`: exit `0`
- real headless viewer smoke:
  - AppLauncher loaded headless kit
  - IsaacLab environment created
  - viewer printed `Step mode: enabled; press Space to advance one frame.`
  - no-space interval stayed alive without playback output, exercising paused render/window pump
  - PTY-fed spaces reached `[Viewer][Playback] path=render+scene_sync`
  - Ctrl-C exited cleanly

## Result

- pass with scoped verification

## Conclusion

- viewer `--step-mode` now gates playback so Space advances one frame.
- paused viewer/play loops keep rendering/pumping the IsaacLab window instead of blocking the whole app loop.
- viewer trajectory visualization now updates only on the same permitted step as machine-dog playback/replan.
- viewer step-mode keeps listening to `W/A/S/D/Q/E/R`; movement commands are latched and do not trigger mid-trajectory replans.
- play `--step-mode` now gates policy/env steps on Space.
- Default non-step-mode behavior remains unchanged by helper tests and code path shape.

## Follow-up

- Optional manual livestream acceptance:
  - start viewer with `--headless --livestream 2 --step-mode`
  - connect WebRTC
  - verify terminal Space advances visual playback one frame at a time
  - press movement keys mid-trajectory and confirm the next trajectory, not the current one, changes command

## Git Refs

- Baseline Ref: `c92627d`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
  - [../todo/T301-viewer-r-key-grounded-reset.md](../todo/T301-viewer-r-key-grounded-reset.md)
