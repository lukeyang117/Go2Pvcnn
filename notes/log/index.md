# Test Log Index

This page indexes verification evidence. Keep it short enough to scan.

## Recent Logs

| Time | Topic | Stage | Result | Key Metrics | Todo | File |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-04-29 23:59 | semantic static-course env_isaaclab compact runtime smoke | semantic static course real runtime | pass with scoped caveat | `semantic_height_scanner_contract` pass; `together_semantic_smoke` pass; compact `4x1` terrain grid; full-grid interactive startup still unverified | [T200](../todo/T200-semantic-static-course-viewer.md) | [2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md](2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md) |
| 2026-04-29 23:48 | semantic static-course implementation + local verification | semantic static course implementation | local tests pass; real runtime smoke incomplete | sensor `4 passed`; course/config `12 passed`; viewer `33+29 passed`; runtime contract reached live semantic-course spawning but timed out at `120s` | [T200](../todo/T200-semantic-static-course-viewer.md) | [2026-04-29-2348-semantic-static-course-implementation-and-local-verification.md](2026-04-29-2348-semantic-static-course-implementation-and-local-verification.md) |
| 2026-04-29 23:18 | semantic static-course parallel review convergence | semantic static course review integration | blockers absorbed into spec | `replicate_physics=False`; stable semantic root containers; `151x151`; valid-hit diagnostics; default `together` success target | [T200](../todo/T200-semantic-static-course-viewer.md) | [2026-04-29-2318-semantic-static-course-parallel-review-convergence.md](2026-04-29-2318-semantic-static-course-parallel-review-convergence.md) |
| 2026-04-29 22:52 | semantic static-course execution model | semantic static course workflow | parallel review / worker split recorded | main agent owns decisions/integration; parallel reviews `R1/R2/R3`; worker split `W1/W2/W3` | [T200](../todo/T200-semantic-static-course-viewer.md) | [2026-04-29-2252-semantic-static-course-execution-model.md](2026-04-29-2252-semantic-static-course-execution-model.md) |
| 2026-04-29 22:34 | semantic static-course viewer spec review | semantic static course design review | approved with refinements | subagent `Approved`; explicit row-band split; fixed `S3/S4` default counts/anchors; semantic hit counts marked required diagnostics | [T200](../todo/T200-semantic-static-course-viewer.md) | [2026-04-29-2234-semantic-static-course-viewer-spec-review.md](2026-04-29-2234-semantic-static-course-viewer-spec-review.md) |
| 2026-04-29 22:09 | semantic static-course viewer design | semantic static course / semantic raycaster / viewer design | design recorded | `semantic_height_scanner`; `1.5x1.5m @ 0.01m`; `0/1/2 semantic map`; `prestartup` required by source-order inspection | [T200](../todo/T200-semantic-static-course-viewer.md) | [2026-04-29-2209-semantic-static-course-viewer-design.md](2026-04-29-2209-semantic-static-course-viewer-design.md) |
| 2026-04-28 12:54 | viewer zero-command handoff idempotence | extension/viz together hold handoff | pass with scoped caveat | red test caught repeated second-segment `delta_z=-0.10`; targeted viewer tests 2 passed; wider subset 56 passed; diagnostic became `0.4->0.3` once then `0.3->0.3` | [T100/T110](../todo/T100-batched-together-planner-gpu-migration.md#t110-zero-command-rehome-upright-recovery) | [2026-04-28-1254-viewer-zero-command-handoff-idempotence.md](2026-04-28-1254-viewer-zero-command-handoff-idempotence.md) |
| 2026-04-28 11:32 | together zero-command rehome recovery | batched_together_planner core/viewer stop smoke | pass with scoped caveat | red test caught frozen root z; T110/core/parity/guardrail 26 passed; runtime+viz 29 passed; combined subset 55 passed; headless viewer zero command near z/rpy recovery | [T100/T110](../todo/T100-batched-together-planner-gpu-migration.md#t110-zero-command-rehome-upright-recovery) | [2026-04-28-1132-together-zero-command-rehome.md](2026-04-28-1132-together-zero-command-rehome.md) |
| 2026-04-28 10:07 | viewer together root-z ratchet fix | extension/viz together playback | pass with scoped caveat | red test reproduced `0.3964m` z climb; `test_viz_playback` 20 passed; together parity/core/runtime/guardrail 35 passed; headless viewer reached real playback then timeout | [T100/T109](../todo/T100-batched-together-planner-gpu-migration.md#t109-viewer-together-root-z-ratchet) | [2026-04-28-1007-viewer-together-root-z-ratchet.md](2026-04-28-1007-viewer-together-root-z-ratchet.md) |
| 2026-04-28 09:52 | human-12 train/viewer/play command guide update | T100 command documentation | pass | commands aligned to `env_isaaclab`, default `together`, legacy rollback, viewer `task/35/0.02` contract | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-28-0952-human-12-command-guide-update.md](2026-04-28-0952-human-12-command-guide-update.md) |
| 2026-04-27 19:14 | batched together continued train/cadence/regression testing | batched_together_planner train/runtime regression | pass with follow-up caveats | train 32/128 env 1-iter pass; legacy 16 env pass; real cadence full-N pass; regression fixed; main pytest 59+36 pass | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1914-batched-together-continued-testing.md](2026-04-27-1914-batched-together-continued-testing.md) |
| 2026-04-27 18:36 | batched together env_isaaclab final verification | batched_together_planner train/runtime/viewer | pass with scoped caveats | full suite 50 passed; CUDA smoke; py_compile pass; together viewer smoke; train/play smoke reviewed | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1836-batched-together-env-isaaclab-final-verification.md](2026-04-27-1836-batched-together-env-isaaclab-final-verification.md) |
| 2026-04-27 18:28 | viewer together backend smoke | extension/viz viewer runtime | pass | together backend plan/playback; legacy rollback plan/playback; runtime path 9 passed; guardrails 5 passed; raw-default cfg drift fixed | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1828-viewer-together-backend-smoke.md](2026-04-27-1828-viewer-together-backend-smoke.md) |
| 2026-04-27 17:11 | batched together cadence decision | batched_together_planner runtime design | design decision recorded | full-N update only on command change, reset, or 0.7s interval; host trigger required | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1711-batched-together-cadence-decision.md](2026-04-27-1711-batched-together-cadence-decision.md) |
| 2026-04-27 16:30 | batched together design review revisions | batched_together_planner design review | issues incorporated | GPU cache ABI; manager-owned phase; safe fallback; raw semantic contract; AST guardrail | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1630-batched-together-design-review-revisions.md](2026-04-27-1630-batched-together-design-review-revisions.md) |
| 2026-04-27 16:22 | batched together planner GPU migration design | batched_together_planner design | design recorded | native Isaac GPU backend; fixed-shape full-batch manager; A+B parity; static guardrail | [T100](../todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1622-batched-together-planner-gpu-migration-design.md](2026-04-27-1622-batched-together-planner-gpu-migration-design.md) |
| 2026-04-27 13:49 | notes workflow bootstrap | notes workflow | pass | dashboard memory system created and linked into existing notes | [T000](../todo.md#root-map) | [2026-04-27-1349-notes-workflow-bootstrap.md](2026-04-27-1349-notes-workflow-bootstrap.md) |

## Topic Log Index

- T200 semantic static-course viewer:
  - [2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md](2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md)
  - [2026-04-29-2348-semantic-static-course-implementation-and-local-verification.md](2026-04-29-2348-semantic-static-course-implementation-and-local-verification.md)
  - [2026-04-29-2318-semantic-static-course-parallel-review-convergence.md](2026-04-29-2318-semantic-static-course-parallel-review-convergence.md)
  - [2026-04-29-2252-semantic-static-course-execution-model.md](2026-04-29-2252-semantic-static-course-execution-model.md)
  - [2026-04-29-2234-semantic-static-course-viewer-spec-review.md](2026-04-29-2234-semantic-static-course-viewer-spec-review.md)
  - [2026-04-29-2209-semantic-static-course-viewer-design.md](2026-04-29-2209-semantic-static-course-viewer-design.md)
- T100 batched together planner GPU migration:
  - [2026-04-28-1254-viewer-zero-command-handoff-idempotence.md](2026-04-28-1254-viewer-zero-command-handoff-idempotence.md)
  - [2026-04-28-1132-together-zero-command-rehome.md](2026-04-28-1132-together-zero-command-rehome.md)
  - [2026-04-28-1007-viewer-together-root-z-ratchet.md](2026-04-28-1007-viewer-together-root-z-ratchet.md)
  - [2026-04-28-0952-human-12-command-guide-update.md](2026-04-28-0952-human-12-command-guide-update.md)
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
