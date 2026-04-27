# Test Log Index

This page indexes verification evidence. Keep it short enough to scan.

## Recent Logs

| Time | Topic | Stage | Result | Key Metrics | Todo | File |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-04-27 19:14 | batched together continued train/cadence/regression testing | batched_together_planner train/runtime regression | pass with follow-up caveats | train 32/128 env 1-iter pass; legacy 16 env pass; real cadence full-N pass; regression fixed; main pytest 59+36 pass | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1914-batched-together-continued-testing.md](2026-04-27-1914-batched-together-continued-testing.md) |
| 2026-04-27 18:36 | batched together env_isaaclab final verification | batched_together_planner train/runtime/viewer | pass with scoped caveats | full suite 50 passed; CUDA smoke; py_compile pass; together viewer smoke; train/play smoke reviewed | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1836-batched-together-env-isaaclab-final-verification.md](2026-04-27-1836-batched-together-env-isaaclab-final-verification.md) |
| 2026-04-27 18:28 | viewer together backend smoke | extension/viz viewer runtime | pass | together backend plan/playback; legacy rollback plan/playback; runtime path 9 passed; guardrails 5 passed; raw-default cfg drift fixed | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1828-viewer-together-backend-smoke.md](2026-04-27-1828-viewer-together-backend-smoke.md) |
| 2026-04-27 17:11 | batched together cadence decision | batched_together_planner runtime design | design decision recorded | full-N update only on command change, reset, or 0.7s interval; host trigger required | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1711-batched-together-cadence-decision.md](2026-04-27-1711-batched-together-cadence-decision.md) |
| 2026-04-27 16:30 | batched together design review revisions | batched_together_planner design review | issues incorporated | GPU cache ABI; manager-owned phase; safe fallback; raw semantic contract; AST guardrail | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1630-batched-together-design-review-revisions.md](2026-04-27-1630-batched-together-design-review-revisions.md) |
| 2026-04-27 16:22 | batched together planner GPU migration design | batched_together_planner design | design recorded | native Isaac GPU backend; fixed-shape full-batch manager; A+B parity; static guardrail | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1622-batched-together-planner-gpu-migration-design.md](2026-04-27-1622-batched-together-planner-gpu-migration-design.md) |
| 2026-04-27 13:49 | notes workflow bootstrap | notes workflow | pass | dashboard memory system created and linked into existing notes | [T000](../todo.md#root-map) | [2026-04-27-1349-notes-workflow-bootstrap.md](2026-04-27-1349-notes-workflow-bootstrap.md) |

## Topic Log Index

- T100 batched together planner GPU migration:
  - [2026-04-27-1914-batched-together-continued-testing.md](2026-04-27-1914-batched-together-continued-testing.md)
  - [2026-04-27-1836-batched-together-env-isaaclab-final-verification.md](2026-04-27-1836-batched-together-env-isaaclab-final-verification.md)
  - [2026-04-27-1828-viewer-together-backend-smoke.md](2026-04-27-1828-viewer-together-backend-smoke.md)
  - [2026-04-27-1711-batched-together-cadence-decision.md](2026-04-27-1711-batched-together-cadence-decision.md)
  - [2026-04-27-1630-batched-together-design-review-revisions.md](2026-04-27-1630-batched-together-design-review-revisions.md)
  - [2026-04-27-1622-batched-together-planner-gpu-migration-design.md](2026-04-27-1622-batched-together-planner-gpu-migration-design.md)
- T000 notes workflow:
  - [2026-04-27-1349-notes-workflow-bootstrap.md](2026-04-27-1349-notes-workflow-bootstrap.md)

## Archived Logs

- none yet

## How To Add A New Entry

1. Create one log file under `notes/log/`.
2. Add a row to `Recent Logs`.
3. Add or update the topic group in `Topic Log Index`.
4. Update `notes/todo.md` and relevant branch pages.
5. If `Recent Logs` grows too long, move older rows to `notes/log/archive/`.

## Copyable Row Template

```text
| YYYY-MM-DD HH:MM | topic | stage | pass/fail/partial | metric summary | [T001](../todo.md#open-leaves) | [YYYY-MM-DD-HHMM-topic.md](YYYY-MM-DD-HHMM-topic.md) |
```
