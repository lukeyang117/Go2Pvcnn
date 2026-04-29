# Semantic Static Course Parallel Review Convergence

## Meta

- Time: `2026-04-29 23:18 +0800`
- Stage: `semantic static course parallel review convergence`
- Result: `blocking review findings absorbed into spec`
- Todo: [T200](../todo/T200-semantic-static-course-viewer.md)

## Purpose

- Record the main-agent decisions after parallel technical/completeness/detail reviews.
- Preserve the blocker resolutions before worker dispatch.

## Main-Agent Decisions

- The semantic viewer scene must set `replicate_physics = False`.
- `extension/semantic_course.py` must always create stable empty container roots:
  - `/World/semantic_course/small`
  - `/World/semantic_course/large`
- The approved semantic raster target is explicit:
  - `151 x 151`
  - `ray_hits_w` length `151 * 151`
- Semantic hit diagnostics must ignore invalid sampled rays and include:
  - terrain/small/large counts
  - `valid_sample_count`
  - one elevation-lift metric
- The authoritative `ray_hits_w -> planner terrain` conversion for this rollout is the stable world-window rule already used by the direct viewer path.
- Success criteria:
  - semantic correctness is required on default `together`
  - `legacy` only needs a semantic smoke if it remains visible in the CLI
- Stage exposure:
  - interactive semantic viewer defaults env `0` to representative `S4`
  - tests force env `0` onto representative rows for `S1..S4`

## Verification

- Parallel subagent reviews completed on sensor, course/lifecycle, and viewer/metrics slices.
- Blocking findings were folded back into the spec and execution notes.

## Git Refs

- Baseline Ref: `a4dc6c2`
- Candidate Ref: `working tree on top of a4dc6c2 (2026-04-29 23:18 +0800); review-convergence note update`
- Key Files:
  - [../../docs/superpowers/specs/2026-04-29-semantic-static-course-viewer-design.md](../../docs/superpowers/specs/2026-04-29-semantic-static-course-viewer-design.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
