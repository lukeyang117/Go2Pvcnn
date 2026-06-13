# 2026-06-13 18:10 Human 12 Play Keyboard Command Update

## Purpose

更新 [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md)，把 flat-small PLAY 的指定地形、键盘速度控制、`pynput` 安装状态和 headless X/DISPLAY 限制写进命令指南。

## Stage

Documentation / play command guide.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

Manual documentation edit with evidence from:

- [2026-06-13-1735-play-keyboard-control-terrain-selection.md](2026-06-13-1735-play-keyboard-control-terrain-selection.md)
- [2026-06-13-1756-play-pynput-install-headless-smoke.md](2026-06-13-1756-play-pynput-install-headless-smoke.md)

Verification:

```bash
rg -n "keyboard-control|terrain-row|pynput|curriculum.terrain_levels|terminal raw-input" notes/human/human-12-batched-planner-train-viewer-commands.md
```

## Input Conditions

- `pynput 1.8.2` is installed in `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.
- Current headless shell has no valid X `DISPLAY`, so `pynput.keyboard` cannot initialize there.
- Real `play.py --keyboard-control` smoke exits `0` because the script warns and disables keyboard capture.

## Key Evidence

- Human 12 now documents:
  - flat-small PLAY closes training `curriculum.terrain_levels`.
  - `--terrain-row` / `--terrain-col` env0 initial terrain selection.
  - `--keyboard-control` and `W/S/A/D/Q/E`, `+/-`, `Space/X`, `Esc`.
  - `pynput` installed but requiring X/DISPLAY.
  - pure headless SSH/WebRTC browser keypresses do not automatically enter the `pynput` backend.

## Result

Documentation updated.

## Conclusion

The command guide now matches the current tested play behavior and avoids implying that installing `pynput` alone is enough for headless keyboard capture.

## Follow-Up

If pure SSH interactive control is required, add and document a terminal raw-input backend.

## Git Refs

- Baseline Ref: `23182ce`
- Candidate Ref: working tree
- Key Files:
  - [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md)
