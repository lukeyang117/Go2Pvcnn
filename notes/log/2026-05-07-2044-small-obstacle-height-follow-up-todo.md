# Small Obstacle Height Follow-Up Todo

## Meta

- Time: `2026-05-07 20:44 +0800`
- Stage: `semantic course requirement follow-up`
- Result: `pass`
- Todo: [T200/T208](../todo/T200-semantic-static-course-viewer.md#t208-small-obstacle-height-reduction-follow-up)

## Purpose

- Record the new user-requested follow-up that current `small obstacle` assets are still too tall and should be reduced.
- Keep this geometry adjustment separate from the ongoing `T113` planner-semantics leaves.

## Follow-Up Decision

- Created new leaf `T208` under `T200 semantic static course viewer`.
- Scope is semantic-course geometry and related deterministic/runtime contract updates, not planner logic.
- The user preference captured here is:
  - length / width may remain larger
  - height should be lower than the current `small obstacle` contract

## Verification

- Todo-only step; no implementation code changed in this log.
- Updated required repository memory surfaces:
  - [../todo.md](../todo.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)

## Conclusion

- The obstacle-geometry adjustment is now tracked explicitly and will not be lost while `T113` planner semantics continue.

## Git Refs

- Baseline Ref: `8e8acc0`
- Candidate Ref: `working tree on top of 8e8acc0 (2026-05-07 20:44 +0800); T208 todo memory update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
