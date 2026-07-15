# Compact Todo Around MPC QP Focus

## Purpose

Compact the repository todo/log memory system so current agent startup focuses on `mpc_qp/T302v`, while older non-`mpc_qp` planner/RL branches become archived context.

## Stage

Notes workflow / todo-log compaction.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

- Rewrote [../todo.md](../todo.md) as a short dashboard centered on T302v.
- Created [../todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md](../todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md) to summarize outdated non-`mpc_qp` roots.
- Rewrote [index.md](index.md) to keep current `mpc_qp` logs in Recent Logs and move older topics to a compact Topic Log Index.
- Did not delete or move per-test log files.

## Validation

```bash
wc -l notes/todo.md notes/log/index.md notes/todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md
rg -n "## Start Here|## Root Map|## Open Leaves|## Topic Log Index|T302v|mpc_qp" notes/todo.md notes/log/index.md notes/todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md
git diff --check -- notes/todo.md notes/log/index.md notes/todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md
```

## Result

Pass.

- [../todo.md](../todo.md): `81` lines, short dashboard centered on `T302v/mpc_qp`.
- [index.md](index.md): `46` lines, current `mpc_qp` Recent Logs plus compact Topic Log Index.
- [../todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md](../todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md): `45` lines, summarizes outdated non-`mpc_qp` roots.
- Required headings and `T302v` / `mpc_qp` anchors were confirmed with `rg`.
- `git diff --check` exit `0`.

## Follow-Up

Keep root dashboard short. Reopen archived non-`mpc_qp` branches only on explicit user request.

## Git Refs

- Baseline Ref: current worktree before compaction
- Candidate Ref: current worktree after compaction
- Key Files:
  - [../todo.md](../todo.md)
  - [index.md](index.md)
  - [../todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md](../todo/archive/2026-07-06-outdated-non-mpc-qp-roots.md)
