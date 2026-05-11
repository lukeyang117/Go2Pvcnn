# T114 State-Machine Touchdown Front-End Redesign

## Meta

- Time: `2026-05-08 12:51 +0800`
- Stage: `together planner design refinement`
- Result: `pass`
- Todo: [T100/T114](../todo/T100-batched-together-planner-gpu-migration.md#t114-state-machine-touchdown-front-end-redesign)

## Purpose

- Record the user-approved design direction for replacing endpoint-centric semantic touchdown selection with a unified pure-GPU state-machine front-end.
- Capture the new requirements for `small` approach/cross/bypass behavior, candidate-as-action-segment semantics, touchdown boundary margin, four-leg consistency, anchor-to-touchdown path clearance, and final rerun-authority rules.

## Design Decisions

- Primary spec written at [../../docs/superpowers/specs/2026-05-08-semantic-touchdown-state-machine-design.md](../../docs/superpowers/specs/2026-05-08-semantic-touchdown-state-machine-design.md).
- The redesign keeps one unified front-end framework active in all scenes, including scenes without `small` obstacles.
- Required state set:
  - `cruise`
  - `approach`
  - `ready_to_cross`
  - `front_cross`
  - `rear_follow`
  - `bypass`
  - `clear`
- `small` is explicitly tri-modal:
  - `approach`
  - `cross`
  - `bypass`
- `small` is not forced to cross; if local terrain, posture, or path-clearance quality is poor, bypass remains valid.
- Touchdown quality now explicitly includes:
  - legal terrain support
  - distance margin from `small` boundaries
  - four-leg consistency
  - whole-body posture quality
  - anchor-to-touchdown foot clearance
  - anchor-to-touchdown leg clearance
- Candidates are action segments rather than only touchdown endpoints.
- The design carries forward the rerun-authority rule:
  - earlier focused passes become `superseded / non-authoritative` once later overlapping edits occur
  - final acceptance must rerun the affected test union on one final code state

## Review Result

- First focused review found four blocker gaps:
  - `front_cross` / `rear_follow` / `clear` were listed but not strongly covered by explicit fixture/acceptance language
  - no dedicated near-boundary off-surface negative fixture was defined
  - pure-GPU / no-`for` / no-`numpy` constraints were not yet promoted into explicit guardrail acceptance
  - action-segment diagnostics were not yet required strongly enough at acceptance level
- The spec was revised to add:
  - explicit semantics for `front_cross`, `rear_follow`, and `clear`
  - explicit fixtures:
    - `F8_front_cross_state`
    - `F9_rear_follow_state`
    - `F10_clear_state_after_cross`
    - `F15_near_boundary_penalize_or_invalidate`
  - explicit guardrail acceptance requiring no-`for` / no-`numpy` / no-CPU checks in the final affected test union
  - required metric `candidate_action_segment_diagnostics_present`
- Blocker-only re-review then recommended passing the design.

## Verification

- Design-only step; no implementation code changed in this log.
- Design structure includes:
  - problem framing
  - goals / non-goals
  - captured user requirements
  - alternative approaches with trade-offs
  - recommended state-machine approach
  - explicit state semantics
  - candidate-as-action-segment structure
  - todo-first contract
  - test authority / rerun rules
  - deterministic fixtures
  - explicit metrics
  - acceptance indicators
- Design review counts:
  - focused review passes: `1`
  - blocker-only re-review passes: `1`
- Deterministic fixtures in spec: `15`
  - `F1`, `F1b`, `F2` through `F15`
- Explicit metrics in spec: `21`
  - including `state_mode`, `touchdown_small_margin`, `front_pair_consistency`, `rear_pair_follow_consistency`, `anchor_to_touchdown_foot_clearance`, `anchor_to_touchdown_leg_clearance`, and `candidate_action_segment_diagnostics_present`

## Conclusion

- The state-machine touchdown front-end redesign is now written, review-hardened, and synchronized into repository memory as ready for user review.
- No todo tree has been written for `T114`; the next valid step is user review of the written spec, followed by explicit `写 todo` if the user wants execution breakdown.

## Follow-up

- Ask the user to review [../../docs/superpowers/specs/2026-05-08-semantic-touchdown-state-machine-design.md](../../docs/superpowers/specs/2026-05-08-semantic-touchdown-state-machine-design.md).
- If the user accepts the spec, convert it directly into todo leaves under `T100/T114`.

## Git Refs

- Baseline Ref: `current working tree after T113/T208 focused verification chain`
- Candidate Ref: `working tree with T114 state-machine touchdown design spec and notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-08-semantic-touchdown-state-machine-design.md](../../docs/superpowers/specs/2026-05-08-semantic-touchdown-state-machine-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
