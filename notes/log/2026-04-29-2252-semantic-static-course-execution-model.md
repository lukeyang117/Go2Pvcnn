# Semantic Static Course Execution Model

## Meta

- Time: `2026-04-29 22:52 +0800`
- Stage: `semantic static course execution model`
- Result: `parallel review / worker ownership model recorded`
- Todo: [T200](../todo/T200-semantic-static-course-viewer.md)

## Purpose

- Record the user-directed execution model after the design/spec phase.
- Preserve the division of responsibilities between the main agent and parallel subagents before implementation starts.

## Execution Model

- Main agent responsibilities:
  - own design and sequencing
  - own technical decisions and integration order
  - review subagent outputs and decide what is accepted
  - own final verification and acceptance
- Parallel review subagents:
  - `R1`: `semantic_raycaster` technical/risk review
  - `R2`: semantic-course placement / `prestartup` / terrain attachment review
  - `R3`: viewer metrics, semantic hit diagnostics, and test-completeness review
- Planned worker split after reviews:
  - `W1`: sensor + tests
  - `W2`: semantic course + viewer config + tests
  - `W3`: viewer integration + tests

## Verification

- Design notes updated; no implementation code changed yet.
- This log exists to anchor the requested subagent workflow before worker dispatch.

## Git Refs

- Baseline Ref: `a4dc6c2`
- Candidate Ref: `working tree on top of a4dc6c2 (2026-04-29 22:52 +0800); notes-only execution-model update`
- Key Files:
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
