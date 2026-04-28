# Investigation Dashboard

This page is the fast-start dashboard for agent work. It is not a full database. Detailed memory lives in [todo/](todo/); evidence lives in [log/](log/).

## Start Here

- Current focus: manually confirm the viewer after the together root-z and zero-command recovery fixes, then continue raw semantic parity and long-run scaling.
- Read next:
  - [current train/viewer/play command guide](human/human-12-batched-planner-train-viewer-commands.md)
  - [viewer zero-command handoff idempotence log](log/2026-04-28-1254-viewer-zero-command-handoff-idempotence.md)
  - [together zero-command rehome log](log/2026-04-28-1132-together-zero-command-rehome.md)
  - [viewer together root-z ratchet fix log](log/2026-04-28-1007-viewer-together-root-z-ratchet.md)
  - [T100 zero-command rehome child](todo/T100-batched-together-planner-gpu-migration.md#t110-zero-command-rehome-upright-recovery)
  - [command guide update log](log/2026-04-28-0952-human-12-command-guide-update.md)
  - [continued train cadence regression log](log/2026-04-27-1914-batched-together-continued-testing.md)
  - [viewer together backend smoke](log/2026-04-27-1828-viewer-together-backend-smoke.md)
  - [T100 batched together planner GPU migration](todo/T100-batched-together-planner-gpu-migration.md)
  - [final env_isaaclab verification](log/2026-04-27-1836-batched-together-env-isaaclab-final-verification.md)
  - [cadence decision log](log/2026-04-27-1711-batched-together-cadence-decision.md)
  - [design review revisions](log/2026-04-27-1630-batched-together-design-review-revisions.md)
  - [design log](log/2026-04-27-1622-batched-together-planner-gpu-migration-design.md)
  - [planner reading guide](human/human-08-extension-planner-reading-guide.md)
  - [planner mapping](human/human-09-extension-planner-mapping.md)
  - [raw batch planner notes](../raw/kinematic_footsteps/notes/index.md)
- Avoid redoing:
  - Do not migrate raw viewer/adapter CPU compatibility code into the training path.
  - Do not preserve legacy dynamic sub-batch replanning in the new `together` backend.
  - Keep viewer CPU logging/camera/visualization exceptions separate from the training-path guardrail.
- Current git base: `7cf6c11`

## Status Legend

- `todo`: not started
- `doing`: under investigation
- `blocked`: waiting on a condition
- `verify`: changed or hypothesized, awaiting verification
- `done`: completed and closed
- `drop`: abandoned direction

## Active Fronts

| Leaf | Why Active Or Next | Suggested Action |
| --- | --- | --- |
| T110 | Zero-command together rehome is implemented and smoke-verified; manual visual confirmation remains useful. | Rerun interactive together viewer and confirm stop command visually recovers upright rather than crouching. |
| T109 | Root-z ratchet is fixed in viewer handoff and covered by a regression test; interactive visual confirmation is still useful. | Rerun interactive together viewer under manual teleop and watch for any remaining visible lift-off. |
| T103 | Complex raw terrain/support/CEM parity still needs broader scenarios beyond flat P0 parity. | Use [T100](todo/T100-batched-together-planner-gpu-migration.md#t103-raw-planner-core-semantic-migration) and extend parity cases. |
| T107 | CUDA full-batch smoke, real cadence/full-N env test, and 1-iteration train at 32/128 envs pass; long-run throughput and multi-device remain open. | Run multi-iteration profiling and larger env counts when performance numbers are needed. |

## Root Map

| Root | Status | Stage | Branch | Current | Refs |
| --- | --- | --- | --- | --- | --- |
| T000 | done | notes workflow | [T000](todo/T000-notes-workflow.md) | memory system bootstrapped and linked into existing notes | feature `7cf6c11`; verified `7cf6c11` |
| T100 | verify | batched together planner -> IsaacLab training/runtime/viewer | [T100](todo/T100-batched-together-planner-gpu-migration.md) | implementation landed and smoke-verified in `env_isaaclab`: together backend, manager/factory/reward wiring, viewer together core path, flat raw parity, static guardrail, 1-iteration train 32/128 envs | feature `pending`; verified `59 passed`, `36 passed`, CUDA smoke, cadence/full-N real env, train/play/viewer smoke |

## Open Leaves

| Leaf | Parent | Status | Priority | Why Active | Next Read |
| --- | --- | --- | --- | --- | --- |
| T110 | T100 | verify | P0 | Core fix and headless zero-command smoke passed; interactive visual confirmation remains. | [T100 branch](todo/T100-batched-together-planner-gpu-migration.md#t110-zero-command-rehome-upright-recovery) |
| T109 | T100 | verify | P0 | Regression fixed and headless viewer reached real playback; visual manual confirmation remains. | [T100 branch](todo/T100-batched-together-planner-gpu-migration.md#t109-viewer-together-root-z-ratchet) |
| T103 | T100 | verify | P1 | Flat raw tensor parity now passes; complex terrain/support/CEM scenarios still need expansion. | [T100 branch](todo/T100-batched-together-planner-gpu-migration.md#t103-raw-planner-core-semantic-migration) |
| T107 | T100 | verify | P1 | CUDA benchmark, real cadence/full-N, and 1-iteration train at 32/128 envs pass; long multi-iteration training, env counts beyond 128, and multi-device are still not covered. | [T100 branch](todo/T100-batched-together-planner-gpu-migration.md#t107-performance-and-scaling-benchmarks) |

## Branch Pages

- [todo/README.md](todo/README.md)
- [T000-notes-workflow.md](todo/T000-notes-workflow.md)
- [T100-batched-together-planner-gpu-migration.md](todo/T100-batched-together-planner-gpu-migration.md)

## Recent Logs

| Time | Topic | Result | Todo | File |
| --- | --- | --- | --- | --- |
| 2026-04-28 12:54 | viewer zero-command handoff idempotence | pass with scoped caveat | [T100/T110](todo/T100-batched-together-planner-gpu-migration.md#t110-zero-command-rehome-upright-recovery) | [2026-04-28-1254-viewer-zero-command-handoff-idempotence.md](log/2026-04-28-1254-viewer-zero-command-handoff-idempotence.md) |
| 2026-04-28 11:32 | together zero-command rehome recovery | pass with scoped caveat | [T100/T110](todo/T100-batched-together-planner-gpu-migration.md#t110-zero-command-rehome-upright-recovery) | [2026-04-28-1132-together-zero-command-rehome.md](log/2026-04-28-1132-together-zero-command-rehome.md) |
| 2026-04-28 10:07 | viewer together root-z ratchet fix | pass with scoped caveat | [T100/T109](todo/T100-batched-together-planner-gpu-migration.md#t109-viewer-together-root-z-ratchet) | [2026-04-28-1007-viewer-together-root-z-ratchet.md](log/2026-04-28-1007-viewer-together-root-z-ratchet.md) |
| 2026-04-28 09:52 | human-12 train/viewer/play command guide update | pass | [T100](todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-28-0952-human-12-command-guide-update.md](log/2026-04-28-0952-human-12-command-guide-update.md) |
| 2026-04-27 19:14 | batched together continued train/cadence/regression testing | pass with follow-up caveats | [T100](todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1914-batched-together-continued-testing.md](log/2026-04-27-1914-batched-together-continued-testing.md) |
| 2026-04-27 18:36 | batched together env_isaaclab final verification | pass with scoped caveats | [T100](todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1836-batched-together-env-isaaclab-final-verification.md](log/2026-04-27-1836-batched-together-env-isaaclab-final-verification.md) |
| 2026-04-27 18:28 | viewer together backend smoke | pass | [T100](todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1828-viewer-together-backend-smoke.md](log/2026-04-27-1828-viewer-together-backend-smoke.md) |
| 2026-04-27 17:11 | batched together cadence decision | design decision recorded | [T100](todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1711-batched-together-cadence-decision.md](log/2026-04-27-1711-batched-together-cadence-decision.md) |
| 2026-04-27 16:30 | batched together design review revisions | issues incorporated | [T100](todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1630-batched-together-design-review-revisions.md](log/2026-04-27-1630-batched-together-design-review-revisions.md) |
| 2026-04-27 16:22 | batched together planner GPU migration design | design recorded | [T100](todo/T100-batched-together-planner-gpu-migration.md) | [2026-04-27-1622-batched-together-planner-gpu-migration-design.md](log/2026-04-27-1622-batched-together-planner-gpu-migration-design.md) |
| 2026-04-27 13:49 | notes workflow bootstrap | pass | [T000](todo/T000-notes-workflow.md) | [2026-04-27-1349-notes-workflow-bootstrap.md](log/2026-04-27-1349-notes-workflow-bootstrap.md) |

## Maintenance

- Keep this page short: dashboard only.
- Put detailed background and conclusions in branch pages.
- Put metrics and command output in per-test logs.
- Use `$compact-todo` when this page, branch pages, or log index grow too large.
