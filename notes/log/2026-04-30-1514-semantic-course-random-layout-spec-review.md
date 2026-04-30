# Semantic Course Random Layout Spec Review

## Meta

- Time: `2026-04-30 15:14 +0800`
- Stage: `semantic static course design review`
- Result: `issues incorporated`
- Todo: [T200/T207](../todo/T200-semantic-static-course-viewer.md#t207-deterministic-full-sub-terrain-semantic-layout--footprint-grounding)

## Purpose

- Record the first spec-review pass for the semantic course random layout and grounding design.
- Capture the refinements applied before user spec review.

## Review Findings Incorporated

- Seed/config contract was underspecified.
  - Added exact defaults, config objects, and compatibility-preserving function signatures.
- Tile-size resolution was too loose.
  - Added `resolve_tile_size(...)` priority order and axis inference rules.
- Targeted runtime semantic-hit tests were not actionable.
  - Added anchor selection / targeted env scan contract and acceptance criteria for S4 small and large hits.
- Grounding defaults were too tunable.
  - Fixed default policy to `height_quantile=1.0` and `embed_depth_m=0.015`.
- Fallback layout behavior was not testable.
  - Added `layout_fallback_used`, canonical no-fallback expectations, and fallback constraints.

## Verification

- Design/spec review step only; no implementation code changed.
- Updated spec: [../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md](../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md)

## Git Refs

- Baseline Ref: `177c4a7`
- Candidate Ref: `working tree on top of 177c4a7 (2026-04-30 15:14 +0800); spec review fixes uncommitted`
- Key Files:
  - [../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md](../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
