# Semantic Touchdown/Bypass/Collision Redesign

## Meta

- Time: `2026-05-07 19:52 +0800`
- Stage: `together planner design refinement`
- Result: `pass`
- Todo: [T100/T113](../todo/T100-batched-together-planner-gpu-migration.md#t113-semantic-touchdown-bypass--collision-redesign)

## Purpose

- Record the user-approved design direction for tightening semantic touchdown legality, always-on fixed `K=3` candidate generation, foothold-level large-obstacle bypass, height-aware swing clearance, and body/thigh/calf collision coverage in the `together` planner.
- Capture the agreed testing and metric contract before any todo breakdown or implementation.

## Design Decisions

- Primary spec written at [../../docs/superpowers/specs/2026-05-07-semantic-touchdown-bypass-collision-design.md](../../docs/superpowers/specs/2026-05-07-semantic-touchdown-bypass-collision-design.md).
- The redesign keeps the user-approved boundary:
  - touchdown/support reads `height + semantic`
  - trajectory/collision remains merged-height-driven in this phase
- `support_at()` remains a legal-support query only; it does not own high-level bypass preferences.
- `small` and `large` are never legal touchdown/support surfaces.
- `small` crossing is a preference rather than a hard behind-obstacle rule; legal terrain before the obstacle remains admissible.
- `large` avoidance must begin at foothold/touchdown policy rather than only through later route penalties.
- Fixed `K=3` candidates are always active, including obstacle-free scenes; center is not inherently privileged over terrain quality and safety.
- Collision coverage must include:
  - one body hull
  - four thighs
  - four calves
- Planner-hot-path implementation must remain pure GPU and fixed shape.

## Review Result

- First focused design review checked requirement coverage and acceptance-indicator coverage.
- Review found two blocking gaps:
  - the spec needed an explicit acceptance condition and deterministic fixture for “soft collision penalty while candidate remains feasible”
  - the spec needed an explicit boundary statement for which obstacle geometries are covered by the merged height surface
- The spec was revised to add:
  - `F9_mild_clearance_penalty_but_feasible`
  - a matching acceptance indicator that proves mild close approach increases soft penalty without triggering infeasible
  - an explicit merged-height-surface coverage boundary for raised obstacles representable as height-bearing occupied surface
- A blocker-only re-review then recommended passing the design with no remaining blockers.

## Verification

- Design-only step; no implementation code changed in this log.
- Design structure includes:
  - problem statement
  - goals / non-goals
  - captured requirements
  - approaches with trade-offs
  - recommended design
  - module responsibility split
  - testing layers
  - deterministic fixtures
  - explicit metrics
  - acceptance indicators
- Design review counts:
  - focused review passes: `1`
  - blocker-only re-review passes: `1`
- Deterministic fixtures in spec: `9`
  - `F1` through `F9`
- Explicit metrics in spec: `11`
  - `candidate_count`
  - `touchdown_semantic_valid_ratio`
  - `small_surface_touchdown_count`
  - `large_surface_touchdown_count`
  - `small_cross_preference_outcome`
  - `large_forward_refusal_ratio`
  - `body_min_clearance`
  - `leg_min_clearance`
  - `collision_penalty_breakdown`
  - `support_xy_z_consistency`
  - `forward_progress_metric`

## Conclusion

- The semantic touchdown/bypass/collision redesign is now written, review-hardened, and synchronized into repository memory as ready for user review.
- No todo tree has been written yet; the next valid step is user review of the written spec, followed by explicit `写 todo` if the user wants execution breakdown.

## Follow-up

- Ask the user to review [../../docs/superpowers/specs/2026-05-07-semantic-touchdown-bypass-collision-design.md](../../docs/superpowers/specs/2026-05-07-semantic-touchdown-bypass-collision-design.md).
- If the user accepts the spec, convert it directly into todo leaves under `T100/T113`.

## Git Refs

- Baseline Ref: `8e8acc0`
- Candidate Ref: `working tree on top of 8e8acc0 (2026-05-07 19:52 +0800); semantic touchdown/bypass/collision design spec + notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-07-semantic-touchdown-bypass-collision-design.md](../../docs/superpowers/specs/2026-05-07-semantic-touchdown-bypass-collision-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
