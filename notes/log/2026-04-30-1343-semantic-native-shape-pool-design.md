# Semantic Native Shape Pool Design

## Meta

- Time: `2026-04-30 13:43 +0800`
- Stage: `semantic native shape pool design`
- Result: `design recorded`
- Todo: [T200](../todo/T200-semantic-static-course-viewer.md)

## Purpose

- Record the user-approved design increment for expanding semantic-course geometry from cuboids-only to the full Isaac Sim native shape pool.

## Design Decisions

- This increment uses only native Isaac Sim shapes:
  - `sphere`
  - `cuboid`
  - `cylinder`
  - `capsule`
  - `cone`
- `small` and `large` remain the only semantic classes.
- Both classes share the same shape pool.
- Shape selection is deterministic per `(stage, row, col, slot, semantic_class)`.
- Scale remains class-driven:
  - `small`: diameter `0.12`, height `0.22`
  - `large`: diameter `0.45`, height `0.55`
- Grounding becomes shape-aware via per-shape bottom-to-center offsets.

## Verification

- Design-only step; no implementation code changed yet.
- Notes, branch memory, and log index are updated.

## Git Refs

- Baseline Ref: `deea8ec`
- Candidate Ref: `working tree on top of deea8ec (2026-04-30 13:43 +0800); design notes update`
- Key Files:
  - [../../docs/superpowers/specs/2026-04-30-semantic-native-shape-pool-design.md](../../docs/superpowers/specs/2026-04-30-semantic-native-shape-pool-design.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
