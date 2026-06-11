# Human 12 Flat Small Command Update

## Purpose

Record the command-guide update for the new flat small-obstacle avoidance experiment and warm-start workflow.

## Stage

Documentation / train-play-eval command guide.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Procedure

- Updated [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md).
- Added the new experiment name and Gym ids.
- Added flat-small cfg ownership and key runtime contracts.
- Added copyable train smoke, warm-start, short-tuning, play, and eval command examples.
- Documented that `train.py --resume --load_run` can use the old teacher checkpoint by passing the old run as an absolute path.
- Documented the current `mpc_policy_eval.py` caveat: checkpoint lookup still defaults to the old `teacher_elevation_trajectory_mpc_semantic` experiment path.

## Input Conditions

- T302q local implementation and smoke already passed:
  - focused `31 passed`
  - production `py_compile` exit `0`
  - fresh 16-env / 1-iteration train smoke exit `0`
  - resume from `2026-06-04_18-16-07/model_14000.pt` exit `0`

## Result

Documentation update complete. No runtime command was executed for this documentation-only step.

## Follow-Up

- Run T302q Task 9 small-collision eval smoke after `mpc_policy_eval.py` supports the flat-small experiment checkpoint path or after choosing a compatible checkpoint placement.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: local working tree on top of `da46138`
- Key Files:
  - `notes/human/human-12-batched-planner-train-viewer-commands.md`
  - `notes/log/2026-06-10-2043-human-12-flat-small-command-update.md`
