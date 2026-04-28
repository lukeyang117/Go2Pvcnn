# Human 12 Command Guide Update

## Purpose

Align the train, play, and viewer command guide with the current `teacher_elevation_trajectory` code after the native `batched_together_planner` migration.

## Stage

T100 command documentation.

## Related Todo

- [T100 batched together planner GPU migration](../todo/T100-batched-together-planner-gpu-migration.md)

## Command / Procedure

- Updated [human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md).
- Switched examples from environment-wrapper/legacy assumptions to `/home/lhy/anaconda3/envs/env_isaaclab/bin/python`.
- Made `--planner-backend together` explicit for train/play/viewer, while documenting `--planner-backend legacy` rollback commands.
- Aligned viewer commands with the current `task` terrain, `35` frame horizon, and `0.02` planner dt contract.
- Replaced stale command examples with current train smoke, long train, distributed train, resume, viewer smoke, viewer interactive/WebRTC, and play commands.

## Input Conditions

- Working tree on top of `7cf6c11`.
- Current code default backend is `planner_backend = "together"`.
- User requested command documentation only.

## Key Evidence

- Train guide now uses:
  - `Go2Pvcnn/scripts/train.py`
  - `--device cuda:0`
  - `--experiment teacher_elevation_trajectory`
  - `--planner-backend together`
- Viewer guide now uses:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `--terrain task`
  - `--n-frames 35`
  - `--plan-dt 0.02`
  - `--planner-backend together`
- Play guide now documents required `--run_dir` and `--checkpoint`, with together and legacy backend options.

## Result

Pass for documentation update. No training or viewer runtime command was executed for this documentation-only change.

## Follow-Up

- Keep the command guide synced if parser defaults, backend choices, or verified checkpoints change.

## Git Refs

- Last Feature Commit: `pending`
- Current Work Ref: `working tree on top of 7cf6c11`
