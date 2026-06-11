# T302q Flat Small Avoidance Reward Plan

## Purpose

Record the implementation todo/plan for the flat small-obstacle avoidance RL continuation reward and episode-level curriculum gate.

## Stage

RL config / near-field semantic avoidance reward planning.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)
- Upstream curriculum: [../todo/T302n-semantic-obstacle-curriculum-plan.md](../todo/T302n-semantic-obstacle-curriculum-plan.md)
- Upstream contact route: [../todo/T302l-mpc-rl-participation-and-reward-plan.md](../todo/T302l-mpc-rl-participation-and-reward-plan.md)
- Eval reference: [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Procedure

- Read repository constraints:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
- Read active dashboard and related todo/log context:
  - [../todo.md](../todo.md)
  - [../todo/T302n-semantic-obstacle-curriculum-plan.md](../todo/T302n-semantic-obstacle-curriculum-plan.md)
  - [../todo/T302l-mpc-rl-participation-and-reward-plan.md](../todo/T302l-mpc-rl-participation-and-reward-plan.md)
  - [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)
  - [index.md](index.md)
- Used `writing-plans` instructions, with user override to save the plan under `notes/todo/`.
- Created the T302q branch plan under `notes/todo/`.
- Updated the dashboard to make T302q the active front.
- Added this planning log to the log index.

## Input Conditions

- User requested continuing after interruption and asked to update todo first, then create a todo Markdown file as the plan.
- User-approved design:
  - New config for flat small-obstacle avoidance.
  - Warm-start from `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07`.
  - New reward should use MPC-style heightfield clearance, but not modify MPC or duplicate MPC FK.
  - Scanner map update is not every step; reward must cache semantic/elevation maps with `map_root_pos_w` and `map_root_yaw_w`.
  - Do not add root displacement/yaw staleness gate.
  - Modify current row-based semantic curriculum rather than creating a new one.
  - Curriculum metric is episode-level true small collision success, not per-step collision statistics.

## Key Plan Contents

- T302q Task 1: static cfg and registration contracts for `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance`.
- T302q Task 2: pure tensor clearance helper using cached semantic/elevation maps.
- T302q Task 3: IsaacLab current body-part sampling for foot, calf/shank, and thigh.
- T302q Task 4: scanner map anchor cache with root pose/yaw captured on scanner update.
- T302q Task 5: IsaacLab `RewTerm` wrapper and new cfg-only reward wiring.
- T302q Task 6: current curriculum changed to episode-level `semantic_contact_small.data.force_matrix_w` success gate.
- T302q Task 7: focused local regression and pycompile.
- T302q Task 8: real IsaacLab smoke and checkpoint compatibility.
- T302q Task 9: small-collision eval smoke and final notes alignment.

## Result

Plan recorded. No runtime code implementation was performed in this step.

## Conclusion

The next implementation step is T302q Task 1: write failing static cfg/registration tests and add the minimal flat-small avoidance cfg without changing the existing main MPC semantic cfg.

## Follow-Up

- Execute T302q Task 1.
- Preserve existing `teacher_elevation_trajectory_mpc_semantic` PLAY/VIEWER and MPC reference behavior.
- Record each verification pass in a separate log file.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: local notes-only planning update on top of `da46138`
- Key Files:
  - `docs/superpowers/specs/2026-06-10-flat-small-obstacle-avoidance-reward-design.html`
  - `notes/todo/T302q-flat-small-avoidance-reward-plan.md`
  - `notes/todo.md`
  - `notes/log/index.md`
  - `notes/log/2026-06-10-1945-t302q-flat-small-avoidance-plan.md`
