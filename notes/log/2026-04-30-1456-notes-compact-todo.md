# Notes Compact Todo

## Meta

- Time: `2026-04-30 14:56 +0800`
- Stage: `notes workflow compact-todo`
- Result: `pass`
- Todo: [T000](../todo/T000-notes-workflow.md)

## Purpose

- Compress the root dashboard and log index after the semantic static-course and native shape-pool work expanded the note set.
- Keep the next-agent handoff fast while preserving detailed evidence in branch pages and archive logs.

## Changes

- Reduced `notes/todo.md` to a tighter dashboard:
  - shorter `Start Here`
  - focused `Active Fronts`
  - only active open leaves
  - removed expanded T200 design-history rows from the dashboard
- Compacted `notes/todo/T200-semantic-static-course-viewer.md`:
  - moved completed children into `Closed Children Archive`
  - kept only active/open follow-up and current-state summaries
  - updated git refs to the latest landed feature commits
- Compacted `notes/log/index.md`:
  - kept the recent operational evidence
  - moved older T200 design-chain rows into:
    - [archive/2026-04-semantic-static-course-design.md](archive/2026-04-semantic-static-course-design.md)

## Validation

- `wc -l notes/todo.md notes/log/index.md notes/todo/T200-semantic-static-course-viewer.md`
  - `notes/todo.md`: `92`
  - `notes/log/index.md`: `72`
  - `notes/todo/T200-semantic-static-course-viewer.md`: `129`
- Required dashboard/index sections still present:
  - `Start Here`
  - `Root Map`
  - `Open Leaves`
  - `Topic Log Index`
  - `Archived Logs`

## Git Refs

- Baseline Ref: `d13a21a`
- Candidate Ref: `working tree on top of d13a21a (2026-04-30 14:56 +0800); notes-only compact-todo update`
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
  - [archive/2026-04-semantic-static-course-design.md](archive/2026-04-semantic-static-course-design.md)
