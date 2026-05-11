# T116 K5 Mode-First Small-Obstacle Crossing Design

## Summary

- stage: design
- todo: [T100/T116](../todo/T100-batched-together-planner-gpu-migration.md#t116-k5-mode-first-small-obstacle-crossing-rewrite)
- spec: [../../docs/superpowers/specs/2026-05-09-k5-mode-first-cross-small-design.md](../../docs/superpowers/specs/2026-05-09-k5-mode-first-cross-small-design.md)
- result: design recorded; no implementation or runtime test executed in this log

## User Requirements Captured

- Exit `/inspire`; use brainstorming/design for the next architecture pass.
- Use fixed `K=5` candidates for every mode.
- Keep planner hot paths GPU-only, fixed-shape, and tensorized: no NumPy, no CPU sync, no dynamic env/candidate subbatch, and no hot-path Python loops over env/candidate/leg.
- No-semantic-obstacle planning should not bypass; candidates should keep command direction and vary speed from current command down to zero.
- `small` obstacles should normally be crossed by gait/touchdown design rather than bypassed.
- `small` obstacles may bypass only when too high or not safely crossable, same broad handling as `large`.
- `large` obstacles should reject center-forward approach and use command-relative bypass with reduced speed.
- Merge previous front/rear crossing states into one `CROSS_SMALL`; do not keep hidden crossing progress.
- All modes should use a longer horizon so the planner can evaluate complete four-leg crossing.
- Runtime acceptance must use Isaac Lab headless under `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.
- Runtime tests must be bounded with timeouts and cleaned up so they do not occupy GPU indefinitely.
- Do not create a new implementation planner file; modify existing code in place and remove obsolete old logic where it conflicts.

## Design Decisions

- Mode is global per replan:
  - `CRUISE`
  - `APPROACH_SMALL`
  - `CROSS_SMALL`
  - `BYPASS_OBSTACLE`
- Candidate contract is:

```text
c_k = (beta_k, route_k, J_k)
```

- Removed from candidate contract:
  - `mode_k`
  - `valid_k`
  - `HOLD`
  - `REFUSE_OR_HOLD`
- Routes are command-relative:
  - `CENTER`: along command direction
  - `LEFT`: command-left normal
  - `RIGHT`: command-right normal
- Candidate tables are fixed shape, but geometry is dynamic:
  - tables define `beta`, `route`, and schedule shape
  - touchdown, apex, base path, and foot path depend on current root/feet, semantic geometry, height map, command direction, terrain support, and reachability
- Shared horizon:

```text
T = 1.0s
dt = 0.02s
steps = 50
event_cap = 2
```

## Cleanup Contract

- Implementation must modify existing together planner files:
  - `Go2Pvcnn/extension/batched_together_planner/config.py`
  - `Go2Pvcnn/extension/batched_together_planner/schedule.py`
  - `Go2Pvcnn/extension/batched_together_planner/terrain.py`
  - `Go2Pvcnn/extension/batched_together_planner/parameterization.py`
  - `Go2Pvcnn/extension/batched_together_planner/costs.py`
  - `Go2Pvcnn/extension/batched_together_planner/planner.py`
  - `Go2Pvcnn/extension/batched_together_planner/types.py`
- Remove or replace obsolete behavior instead of keeping dormant alternatives:
  - old `K=3` route-only assumptions
  - old `front_cross` / `rear_follow` / `clear` semantics that conflict with merged `CROSS_SMALL`
  - old 35-step horizon assumptions where they apply to together planner trajectory contracts
  - stale diagnostics/tests that only describe obsolete state labels
- Tests should prefer existing together planner semantic/core/guardrail/runtime files. Add a new test file only if existing ownership would become misleading.

## Required Test Matrix

- Deterministic planner fixtures:
  - `F1_cruise_no_semantic_k5_speed_ladder`
  - `F2_cruise_uneven_terrain_selects_slower_center_speed`
  - `F3_lateral_command_direction_guard`
  - `F4_forward_command_no_backward_progress`
  - `F5_approach_small_does_not_cross_or_touch`
  - `F6_cross_small_four_leg_success`
  - `F7_cross_small_dynamic_geometry_changes_with_obstacle_position`
  - `F8_cross_small_rejects_touchdown_on_small`
  - `F9_cross_small_rejects_foot_path_collision`
  - `F10_cross_small_rejects_base_body_leg_collision`
  - `F11_too_high_small_uses_bypass`
  - `F12_large_blocks_center_uses_bypass`
  - `F13_k5_shape_consistency_all_modes`
  - `F14_horizon_50_contract`
- Headless Isaac Lab runtime cases:
  - `R1_cruise_no_semantic_no_bypass`
  - `R2_small_cross_runtime_four_leg_success`
  - `R3_small_cross_runtime_no_touchdown_on_small`
  - `R4_small_cross_runtime_no_foot_path_collision`
  - `R5_small_cross_runtime_no_base_body_leg_penetration`
  - `R6_large_runtime_bypass_direction_guard`
  - `R7_lateral_runtime_no_opposite_direction_rejection`
- Final acceptance must rerun all relevant deterministic, guardrail, and runtime tests on the final code state. Earlier partial passes are not authoritative after later code changes.

## Verification

- Design and notes were updated only.
- No code implementation was performed in this log.
- No deterministic or Isaac Lab tests were run in this log.

## Next

- Review the design with the user.
- If accepted, convert T116 into implementation todo leaves.
- Implementation should use subagent-driven development with the main agent owning review, cleanup, and final-code-state test authority.
