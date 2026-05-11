# T100 Pre-T116 Historical Context

## Current State

- This page preserves the pre-`T116` together-planner history that is still useful as background but is no longer an active implementation front.
- `T116` is the only active architecture subtree under `T100`; do not reopen these nodes as active work unless a future task explicitly says to.
- The goal of this page is traceability, not task execution. Use it to recover prior constraints, metrics, and evidence when `T116` needs historical context.

## Open Children

- none

## Closed Children Archive

- `T101-T108`: together backend architecture, API, manager/cache, backend switch, parity, benchmark, and guardrail foundation.
- `T109-T111`: viewer/runtime peripheral fixes and WebRTC/server-side runtime hygiene.
- `T112`: semantic-aware together planner/viewer baseline before the later small-obstacle redesign rounds.
- `T113`: semantic touchdown/bypass/collision redesign baseline.
- `T114`: state-machine touchdown front-end redesign baseline.
- `T115`: grounded rear-follow runtime-validation baseline.

## Related Logs

- [2026-05-08-2239-t115f-final-authority.md](../log/2026-05-08-2239-t115f-final-authority.md)
- [2026-05-08-2229-t115e-runtime-acceptance-cases.md](../log/2026-05-08-2229-t115e-runtime-acceptance-cases.md)
- [2026-05-08-1705-t114g-traceability-authority.md](../log/2026-05-08-1705-t114g-traceability-authority.md)
- [2026-05-07-2136-t113e-diagnostics-traceability.md](../log/2026-05-07-2136-t113e-diagnostics-traceability.md)
- [2026-05-06-2340-semantic-aware-together-viewer-implementation.md](../log/2026-05-06-2340-semantic-aware-together-viewer-implementation.md)
- [2026-05-06-2248-human-16-isaaclab-applauncher-webrtc-migration-guide.md](../log/2026-05-06-2248-human-16-isaaclab-applauncher-webrtc-migration-guide.md)
- [2026-05-06-2106-viewer-persistent-loop-fix.md](../log/2026-05-06-2106-viewer-persistent-loop-fix.md)
- [2026-05-06-2054-isaaclab-livestream-dedup-fix.md](../log/2026-05-06-2054-isaaclab-livestream-dedup-fix.md)
- [2026-05-06-2011-viewer-webrtc-public-ip-fix.md](../log/2026-05-06-2011-viewer-webrtc-public-ip-fix.md)
- [2026-04-28-1254-viewer-zero-command-handoff-idempotence.md](../log/2026-04-28-1254-viewer-zero-command-handoff-idempotence.md)
- [2026-04-28-1132-together-zero-command-rehome.md](../log/2026-04-28-1132-together-zero-command-rehome.md)
- [2026-04-28-1007-viewer-together-root-z-ratchet.md](../log/2026-04-28-1007-viewer-together-root-z-ratchet.md)
- [2026-04-27-1914-batched-together-continued-testing.md](../log/2026-04-27-1914-batched-together-continued-testing.md)
- [2026-04-27-1836-batched-together-env-isaaclab-final-verification.md](../log/2026-04-27-1836-batched-together-env-isaaclab-final-verification.md)
- [2026-04-27-1828-viewer-together-backend-smoke.md](../log/2026-04-27-1828-viewer-together-backend-smoke.md)
- [2026-04-27-1711-batched-together-cadence-decision.md](../log/2026-04-27-1711-batched-together-cadence-decision.md)
- [2026-04-27-1630-batched-together-design-review-revisions.md](../log/2026-04-27-1630-batched-together-design-review-revisions.md)
- [2026-04-27-1622-batched-together-planner-gpu-migration-design.md](../log/2026-04-27-1622-batched-together-planner-gpu-migration-design.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `working tree verification on top of 7cf6c11`
- Current Work Ref: `historical summary preserved while T116 remains active in the main T100 page`
- Key Files:
  - [T100-batched-together-planner-gpu-migration.md](T100-batched-together-planner-gpu-migration.md)
  - [../log/index.md](../log/index.md)

## Next Step

- Keep this page stable as background memory.
- If future work needs one of these older branches again, reopen it from here with an explicit new todo node instead of expanding the active `T100` page by default.

## Node Details

### Foundation Summary (`T101-T108`)

- `T101`: established the native IsaacLab GPU together backend, full-batch planner-call rule, and fixed-shape runtime direction.
- `T102`: defined module/API boundaries, manager protocol, GPU cache ABI, and result-schema compatibility.
- `T103`: documented raw planner parity boundaries and explicitly excluded CPU/viewer-only behavior from the active hot path.
- `T104`: defined full-batch manager/cache blending semantics and host-side trigger rules.
- `T105`: wired the `together` / `legacy` backend switch across env cfg, train/play, and viewer entry points.
- `T106`: captured parity and behavior test expectations for schedule, kinematics, trajectory, costs, support query, manager behavior, and mixed-batch execution.
- `T107`: preserved scaling and cadence benchmark evidence; this remains context only unless a future task reactivates performance work.
- `T108`: added static training-path guardrails against CPU sync, forbidden tensor branching, and hot-path Python loops.

### Peripheral Runtime Summary (`T109-T111`)

- `T109-T110`: viewer root-z and zero-command rehome/runtime issues were fixed and should be treated as background runtime hygiene, not active architecture guidance.
- `T111`: remote WebRTC/server-side viewer fixes landed; browser visual confirmation was intentionally kept outside the current `T116` completion scope.
- Carry-forward lesson:
  - headless Isaac Lab checks must be timeout-wrapped
  - cleanup must preserve exit code
  - viewer-only CPU exceptions stay outside the training path

### Semantic And State-Machine Baselines (`T112-T115`)

- `T112`: introduced semantic-aware together planner/viewer behavior and row/col targeting; this is the earliest semantic baseline for later small-obstacle work.
- `T113`: established the semantic touchdown/bypass/collision redesign baseline with fixed `K=3`, legal-support-only touchdown, and collision-aware clearance metrics.
- `T114`: added the old state-machine front-end baseline with `approach/front_cross/rear_follow/clear` semantics, pair/posture/path diagnostics, and final-code-state authority rules.
- `T115`: added grounded rear-follow and headless Isaac runtime acceptance, plus the three-surface crossing checks and authoritative rerun discipline.

### Carry-Forward Constraints Into `T116`

- Preserve fixed-shape full-batch planner execution and GPU residency.
- Preserve the guardrail rule against CPU sync, dynamic sub-batch planning, and hot-path Python loops.
- Preserve runtime hygiene for headless Isaac Lab acceptance.
- Treat `T113-T115` fixtures and logs as historical evidence only; their old `K=3`, `front_cross/rear_follow/clear`, and `35`-step assumptions are superseded by `T116`.
