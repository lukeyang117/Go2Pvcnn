# Human-16 MPC Command Update

- timestamp: 2026-05-11 13:43 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: pass

## Purpose

Update `notes/human/human-16-isaaclab-applauncher-webrtc-migration.md` so viewer and training launch commands use `mpc` planner backend for user-side manual validation.

## Stage

Documentation-only update (no runtime/code-path behavior changes).

## Changes

- Updated command section title to `推荐启动命令（MPC Planner）`.
- Changed remote viewer command:
  - `--planner-backend together` -> `--planner-backend mpc`
- Added a training launch command for manual `mpc` verification:
  - `Go2Pvcnn/scripts/train.py`
  - `--experiment teacher_elevation_trajectory`
  - `--planner-backend mpc`
  - `--num_envs 4096 --max_iterations 1`

## Verification

- `git diff --check -- notes/human/human-16-isaaclab-applauncher-webrtc-migration.md` -> pass.
- `rg "planner-backend" -n notes/human/human-16-isaaclab-applauncher-webrtc-migration.md` -> both updated commands use `mpc`.

## Notes

- This pass only updates documentation commands as requested; it does not run training/viewer itself.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: working tree with `human-16` command updates
- Key Files:
  - [../../notes/human/human-16-isaaclab-applauncher-webrtc-migration.md](../../notes/human/human-16-isaaclab-applauncher-webrtc-migration.md)
