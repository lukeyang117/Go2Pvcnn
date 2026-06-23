# Human 12 Keep Std Command Update

## Purpose

Update the human command guide with the current flat-small continuation command for `2026-06-17_12-01-10/model_14700.pt`, including `--keep_std` and `--mpc_num_envs 1024`.

## Stage

Documentation / training command guide.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

```bash
rg -n "keep_std|model_14700|mpc_num_envs 1024|默认 resume|planner-backend.*默认" notes/human/human-12-batched-planner-train-viewer-commands.md
git diff --check -- notes/human/human-12-batched-planner-train-viewer-commands.md
```

## Input Conditions

- `train.py` now exposes `--keep_std`.
- `OnPolicyRunner.load()` defaults to dropping checkpoint `std`; `--keep_std` preserves it.
- The user wants to continue `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10/model_14700.pt` and keep old std.

## Key Metrics

- `rg` confirmed the guide contains `--keep_std`, `model_14700.pt`, `--mpc_num_envs 1024`, and current `--planner-backend mpc` wording.
- `git diff --check`: exit `0`.

## Result

- [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md) now includes the recommended 1024-env flat-small continuation command with `--keep_std`.
- The key-parameter section explains `--keep_std` and `--mpc_num_envs`.
- The `--planner-backend` default note was aligned with current `train.py`: only `mpc` is supported and it is the default.

## Conclusion

Docs now match the train/resume behavior added in [2026-06-23-train-keep-std-resume-option.md](2026-06-23-train-keep-std-resume-option.md).

## Follow-Up

No runtime command was run because this is a docs-only update.

## Git Refs

- Baseline Ref: `704db79`
- Candidate Ref: working tree
- Key Files:
  - [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md)
