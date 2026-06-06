# T302p MPC Command Frame Alignment Plan

## Purpose

Record the implementation todo/plan for fixing the MPC command-frame mismatch reproduced in T302o flat-forward evaluation.

## Stage

MPC semantic policy evaluation / batch MPC command-frame contract planning.

## Related Todo

- [../todo/T302p-mpc-command-frame-alignment-plan.md](../todo/T302p-mpc-command-frame-alignment-plan.md)
- Parent evidence: [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Procedure

- Read repository constraints:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
- Read active dashboard and planner context:
  - [../todo.md](../todo.md)
  - [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)
  - [../human/human-08-extension-planner-reading-guide.md](../human/human-08-extension-planner-reading-guide.md)
  - [../human/human-09-extension-planner-mapping.md](../human/human-09-extension-planner-mapping.md)
- Used `writing-plans` instructions and user-approved design direction.
- Created a new T302p branch page under `notes/todo/`.
- Updated the dashboard to make T302p the active front.
- Added this planning log to the log index.

## Input Conditions

- User asked to start writing todo and use todo as the plan.
- User requirement: align training command, viewer keyboard command, and MPC command direction to root-yaw/body frame externally.
- User requirement: only align coordinate frames; do not add new loss, modify loss weights, or change the previous low-small design intent.
- Existing reproduction:
  - flat terrain
  - no semantic obstacle signal
  - fixed command `[1.0, 0.0, 0.0]`
  - robot yaw about `16.05deg`
  - default MPC side drift about `-0.0937m`
  - yaw-rotated command side drift about `-0.0044m`

## Key Plan Contents

- T302p Task 1: RED static/unit guards for root-yaw axes, viewer no-pre-rotation, and eval command-source diagnostics.
- T302p Task 2: make `command_frame_axes()` interpret command XY as root-yaw/body-frame and output world axes.
- T302p Task 3: audit and convert MPC world-geometry heading uses in `semantic_policy.py`, `planner.py`, and `losses/terrain_clearance.py`.
- T302p Task 4: remove viewer pre-rotation before MPC planning while preserving body-frame arrow display.
- T302p Task 5: add flat all-direction diagnostics in `mpc_policy_eval.py`.
- T302p Task 6: run local focused regression and no-loss-change diff check.
- T302p Task 7: run real IsaacLab flat all-direction smoke on GPU 0 in `env_isaacsim`.
- T302p Task 8: rerun semantic compatibility regression for low-small crossing and obstacle avoidance.
- T302p Task 9: keep notes/logs aligned.

## Result

Plan recorded. No runtime code implementation was performed in this step.

## Conclusion

The next implementation step is T302p Task 1: write failing tests that encode the body/root-yaw public command contract and prevent viewer/eval boundary drift.

## Follow-Up

- Execute T302p Task 1 inline.
- Preserve T302o eval and livestream behavior while changing the MPC command-frame boundary.
- Record each verification pass in a separate log file.

## Git Refs

- Baseline Ref: local dirty worktree with T302o diagnostics and design HTML present.
- Candidate Ref: local notes-only planning update.
- Key Files:
  - `notes/todo/T302p-mpc-command-frame-alignment-plan.md`
  - `notes/todo.md`
  - `notes/log/index.md`
  - `notes/log/2026-06-06-1834-t302p-mpc-command-frame-alignment-plan.md`
