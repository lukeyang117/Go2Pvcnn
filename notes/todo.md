# Investigation Dashboard

This page is the fast-start dashboard for agent work. It is not a full database. Detailed memory lives in [todo/](todo/); evidence lives in [log/](log/).

## Start Here

- Current focus: T116 `K=5` mode-first small-obstacle crossing has now closed the `T116i` follow-up for nonzero speed candidate tables and hard-reason diagnostics.
  - T116 supersedes the old active `K=3`, 35-step, `front_cross/rear_follow` small-crossing implementation details; old logs remain evidence, not the next target.
  - Final authority was recorded in `T116g`/`T116h`; `T116i` extends that same mainline by removing nonzero `beta=0` candidates and adding fixed-shape hard-reason/rank diagnostics, with final review evidence recorded.
  - Separate tooling front: T002 `compact-todo` has one live branch-compact pass plus a new tree/index hardening pass; grouped stale-test cleanup behavior is still the remaining pressure gap.
  - New design front: T300 has completed subagent-driven implementation integration for `extension/batch_mpc_planner`; current state is runtime-acceptance verification.
  - T300d now has `env_isaacsim` headless MPC selector evidence, runtime-counter instrumentation, viewer direct-script MPC stabilization, long replan foot-drift reproduction evidence, a five-direction variant sweep, an iterative direction search selecting `dir10`, a second expansion round, and a mixed-command / command-switch sweep showing the `dir10` drift benefit plus yaw-segment foot-step side effects; `dir14` helps mixed boundary drift. Production `batch_mpc_planner` now implements the accepted `dir15 + dir19` synthesis with terrain-grounded contact foot z. User visual inspection then exposed a remaining yaw-only foot alternation problem; standalone IsaacLab probes separate planned grounding from actual playback and show the failure is yaw-dominant while pure forward/back/lateral are much cleaner. True 4096 counter extraction remains blocked by large-scale Isaac runtime instability.
  - T300e continuous swing-window redesign is now implemented locally and has `env_isaacsim` runtime probe evidence: no `MpcFootholdMemory`, no output-side foot grounding, no `contact_logits`; scanner height/semantic losses, body-frame nominal/tracking, swing-center urgency, clamped-output IK/FK feasibility, root-foot center, root-height, and yaw-frame support-plane losses are active. Backend verification passes, NaN/contact-collapse is fixed, and the latest targeted runtime pass cleaned the prior `backward_fast` plus mixed-yaw stance-airborne residuals with a clean command-matrix pytest artifact.
- Read next:
  - [T300 unified dense MPC branch](todo/T300-unified-dense-mpc-backend.md)
  - [T300 spec](../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md)
  - [T300 spec hardening log](log/2026-05-11-1110-t300-spec-hardening.md)
  - [T300 subagent review log](log/2026-05-11-1104-t300-subagent-design-review.md)
  - [T300 design log](log/2026-05-11-1050-t300-unified-dense-mpc-backend-design.md)
  - [T300d subagent integration log](log/2026-05-11-1157-t300d-subagent-implementation-review.md)
  - [T300d env_isaacsim runtime log](log/2026-05-11-1243-t300d-env-isaacsim-mpc-headless-runtime.md)
  - [T300d 4096 runtime counters attempt log](log/2026-05-11-1318-t300d-4096-runtime-counters-attempt.md)
  - [T300d 4096 train max-iter1 success log](log/2026-05-11-1428-mpc-4096-train-maxiter1-success.md)
  - [T300d viewer entrypoint + autograd replan fix log](log/2026-05-11-1505-t300d-mpc-viewer-entrypoint-and-autograd-replan-fix.md)
  - [T300d terrain ray-shape OOM fix log](log/2026-05-11-1411-mpc-terrain-ray-shape-oom-fix.md)
  - [T300d leg-order command-matrix recovery log](log/2026-05-11-2005-mpc-leg-order-command-matrix-recovery.md)
  - [T300d long replan foot drift reproduction log](log/2026-05-12-2147-mpc-long-replan-foot-drift-reproduction.md)
  - [T300d long replan variant sweep log](log/2026-05-12-2243-mpc-long-replan-variant-sweep.md)
  - [T300d iterative long replan direction search log](log/2026-05-12-2346-mpc-long-replan-iterative-direction-search.md)
  - [T300d second direction expansion log](log/2026-05-13-0030-mpc-long-replan-second-direction-expansion.md)
  - [T300d mixed/sequence long-horizon sweep log](log/2026-05-13-0932-mpc-mixed-sequence-long-horizon-sweep.md)
  - [T300d dir15/dir19 production grounding log](log/2026-05-13-1253-mpc-dir15-dir19-production-grounding.md)
  - [T300d yaw foot alternation reproduction log](log/2026-05-13-1352-mpc-yaw-foot-alternation-reproduction.md)
  - [T300d yawfix direction sweep log](log/2026-05-13-1429-mpc-yawfix-direction-sweep.md)
  - [T300d yawfix4-plus long sweep log](log/2026-05-13-1611-mpc-yawfix4plus-long-sweep.md)
  - [T300d yaw anchor memory production log](log/2026-05-13-1633-mpc-yaw-anchor-memory-production.md)
  - [T300d yaw gait failure probe log](log/2026-05-13-1757-mpc-yaw-gait-failure-probe.md)
  - [T300d all-speed gait probe log](log/2026-05-13-1810-mpc-all-speed-gait-probe.md)
  - [T300d command switch gait probe log](log/2026-05-13-1815-mpc-command-switch-gait-probe.md)
  - [T300d forward/backward alternation probe log](log/2026-05-13-1835-mpc-forward-backward-alternation-probe.md)
  - [T300d forward/backward pair alternation log](log/2026-05-13-1843-mpc-forward-backward-pair-alternation.md)
  - [T300d root-cause minimal verification log](log/2026-05-13-1910-mpc-root-cause-minimal-verification.md)
  - [T300d IK/FK residual headless comparison log](log/2026-05-13-2023-mpc-ikfk-residual-headless-comparison.md)
  - [T300e continuous swing-window implementation log](log/2026-05-15-1755-mpc-continuous-swing-window-implementation.md)
  - [T300e continuous window runtime fix log](log/2026-05-15-1903-mpc-continuous-window-runtime-fix.md)
  - [T300e IK/FK and grounding runtime tuning log](log/2026-05-15-1937-mpc-ikfk-grounding-runtime-tuning.md)
  - [T300e contact support and touchdown anchor acceptance log](log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md)
  - [human-16 mpc command update log](log/2026-05-11-1343-human16-mpc-command-update.md)
  - [T116i nonzero speed/hard-reason design](../docs/superpowers/specs/2026-05-10-nonzero-speed-hard-reason-design.md)
  - [T116i main review/final verification log](log/2026-05-10-2223-t116i-main-review-final-verification.md)
  - [T116i review-fix log](log/2026-05-10-2214-t116i-review-fix-viewer-output-small-runtime.md)
  - [T116i implementation log](log/2026-05-10-2156-t116i-nonzero-speed-hard-reason-implementation.md)
  - [T116i todo log](log/2026-05-10-2138-t116i-nonzero-speed-hard-reason-todo.md)
  - [T117 test/todo cleanup branch](todo/T117-together-planner-test-and-todo-cleanup.md)
  - [T116h final authority log](log/2026-05-10-2017-t116h-final-review-authority.md)
  - [T117 approved test deletion log](log/2026-05-10-2102-t117-approved-test-deletion.md)
  - [T100 batched together planner GPU migration](todo/T100-batched-together-planner-gpu-migration.md)
  - [T100 pre-T116 historical context](todo/T100-pre-t116-history.md)
  - Planner preread when editing implementation: [extension planner reading guide](human/human-08-extension-planner-reading-guide.md), [extension planner mapping](human/human-09-extension-planner-mapping.md), [train/viewer/play command guide](human/human-12-batched-planner-train-viewer-commands.md)
- Avoid redoing:
  - Do not migrate raw viewer/adapter CPU compatibility code into the training path.
  - Do not preserve legacy dynamic sub-batch replanning in the new `together` backend.
  - Keep viewer CPU logging/camera/visualization exceptions separate from the training-path guardrail.
  - Do not add a parallel new planner implementation for T116; modify existing together planner files in place and delete/rewrite obsolete old logic.
  - Do not continue T113/T114/T115 as active architecture. They are historical baselines and evidence only for T116.
- Current git base: `130c635`

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
| T002b | One live branch-compact session passed and tree/Obsidian index preservation is now explicit; the remaining pressure gap is grouped stale-test review and decision batching across `Go2Pvcnn/tests/`. | Use compact sessions to keep grouped note/test cleanup behavior sharp as the tree shrinks around final T116 authority. |
| T117 | T116 is closed through `T116h`; remaining work is lightweight note/index cleanup around final authority and reduced test surface. | Keep compressing non-T116 surfaces while preserving final authority and the new `T117` cleanup branch. |
| T300e | Continuous swing-window MPC redesign is implemented in active code; backend verification and `env_isaacsim` tuning show NaN/contact-collapse fixed, `backward_fast` targeted residual cleaned, mixed-yaw targeted probes clean, and command-matrix pytest artifact clean. | Broaden acceptance with longer unmonkeypatched yaw/viewer and command-switch probes; keep 4096 runtime counters as the remaining scale-stability front. |
| T301a | Viewer `R` reset semantics now preserve current root `xy/yaw`, restore standing joints, and ground feet from scanner terrain; remaining work is one targeted real-runtime reset assertion. | Add a focused IsaacLab headless reset check without disturbing the active T300e branch. |

## Root Map

| Root | Status | Stage | Branch | Current | Refs |
| --- | --- | --- | --- | --- | --- |
| T000 | done | notes workflow | [T000](todo/T000-notes-workflow.md) | memory system bootstrapped and linked into existing notes | feature `7cf6c11`; verified `7cf6c11` |
| T001 | done | local skill design / requirement-analysis workflow | [T001](todo/T001-inspire-skill-design.md) | `/inspire` skill package is implemented, review-hardened, and no longer active for T116 unless explicitly invoked | feature `pending`; verified `skill file tree + stage-gate grep checks` |
| T002 | verify | local skill design / interactive memory and test grooming workflow | [T002](todo/T002-compact-todo-interactive-memory-and-test-grooming.md) | `compact-todo` skill body now preserves child-tree/Obsidian index paths during compaction; grouped stale-test review remains | feature `pending`; verified `skill readback + live branch-compact pressure pass + tree/index hardening readback` |
| T100 | doing | batched together planner -> IsaacLab training/runtime/viewer | [T100](todo/T100-batched-together-planner-gpu-migration.md) | T116i is closed with main review/final verification evidence; `T117` remains notes/test cleanup | feature `pending`; verified through `T116i` final review checks |
| T300 | verify | unified dense MPC backend design and implementation | [T300](todo/T300-unified-dense-mpc-backend.md) | `batch_mpc_planner` now implements T300e continuous swing-window MPC; backend/compile verification passes and targeted `env_isaacsim` acceptance cleaned the previous `backward_fast` and command-matrix blockers | feature `pending`; verified `43 backend tests + py_compile + root-cause JSONL probe + command-matrix selector` |
| T200 | done | semantic static course -> semantic raycaster -> viewer integration | [T200](todo/T200-semantic-static-course-viewer.md) | supporting semantic terrain work is background for T116, including small obstacle height reduction; no active T200 work is in the current front | feature `130c635`; verified `130c635` with runtime-output caveat |
| T301 | verify | viewer interaction / grounded reset behavior | [T301](todo/T301-viewer-r-key-grounded-reset.md) | `R` reset helper now preserves current root `xy/yaw`, restores initial standing joints, clears command buffer, and grounds feet from scanner terrain; targeted real-runtime reset evidence remains | feature `pending`; verified `local+env_isaacsim test_viewer_reset + py_compile + diff check` |

## Open Leaves

| Leaf | Parent | Status | Priority | Why Active | Next Read |
| --- | --- | --- | --- | --- | --- |
| T002b | T002 | verify | P1 | One live branch-compact pass is complete; remaining verification is grouped stale-test cleanup behavior and decision batching for `Go2Pvcnn/tests/`. | [T002 branch](todo/T002-compact-todo-interactive-memory-and-test-grooming.md#t002b-live-usage-pressure-verification) |
| T117 | T100/T116 | verify | P1 | Non-keep `Go2Pvcnn/tests/` surfaces were deleted with user approval; remaining work is note compression and new index organization for the reduced test tree. | [T117 branch](todo/T117-together-planner-test-and-todo-cleanup.md) |
| T300e | T300 | verify | P0 | Continuous swing-window redesign is implemented and latest `env_isaacsim` targeted acceptance cleaned mixed-yaw, `backward_fast`, and command-matrix evidence; remaining work is broader long-horizon viewer/yaw/4096 confidence. | [T300e branch](todo/T300e-mpc-continuous-swing-window-plan.md) + [acceptance log](log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md) |
| T301a | T301 | verify | P1 | Viewer `R` reset helper now matches the corrected user semantics; remaining gap is one real-runtime targeted reset assertion. | [T301 branch](todo/T301-viewer-r-key-grounded-reset.md#t301a-viewer-r-reset语义改造与helper验证) + [log](log/2026-05-15-2045-t301-viewer-r-key-grounded-reset.md) |

## Branch Pages

- [todo/README.md](todo/README.md)
- [T000-notes-workflow.md](todo/T000-notes-workflow.md)
- [T001-inspire-skill-design.md](todo/T001-inspire-skill-design.md)
- [T002-compact-todo-interactive-memory-and-test-grooming.md](todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
- [T100-batched-together-planner-gpu-migration.md](todo/T100-batched-together-planner-gpu-migration.md)
- [T117-together-planner-test-and-todo-cleanup.md](todo/T117-together-planner-test-and-todo-cleanup.md)
- [T100-pre-t116-history.md](todo/T100-pre-t116-history.md)
- [T300-unified-dense-mpc-backend.md](todo/T300-unified-dense-mpc-backend.md)
- [T301-viewer-r-key-grounded-reset.md](todo/T301-viewer-r-key-grounded-reset.md)
- [T200-semantic-static-course-viewer.md](todo/T200-semantic-static-course-viewer.md)

## Recent Logs

| Time | Topic | Result | Todo | File |
| --- | --- | --- | --- | --- |
| 2026-05-15 20:45 | T301 viewer R-key grounded reset | helper-level pass with local and `env_isaacsim` verification | [T301](todo/T301-viewer-r-key-grounded-reset.md) | [2026-05-15-2045-t301-viewer-r-key-grounded-reset.md](log/2026-05-15-2045-t301-viewer-r-key-grounded-reset.md) |
| 2026-05-15 20:01 | T300e MPC contact support and touchdown anchor acceptance | targeted runtime pass; prior blockers clean | [T300/T300e](todo/T300e-mpc-continuous-swing-window-plan.md) | [2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md](log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md) |
| 2026-05-15 19:37 | T300e MPC IK/FK and grounding runtime tuning | partial runtime improvement; backward-fast remains | [T300/T300e](todo/T300e-mpc-continuous-swing-window-plan.md) | [2026-05-15-1937-mpc-ikfk-grounding-runtime-tuning.md](log/2026-05-15-1937-mpc-ikfk-grounding-runtime-tuning.md) |
| 2026-05-15 19:03 | T300e MPC continuous window runtime fix | partial runtime pass; residual risks remain | [T300/T300e](todo/T300e-mpc-continuous-swing-window-plan.md) | [2026-05-15-1903-mpc-continuous-window-runtime-fix.md](log/2026-05-15-1903-mpc-continuous-window-runtime-fix.md) |
| 2026-05-15 17:55 | T300e MPC continuous swing-window implementation | local pass; IsaacLab runtime blocked by missing package | [T300/T300e](todo/T300e-mpc-continuous-swing-window-plan.md) | [2026-05-15-1755-mpc-continuous-swing-window-implementation.md](log/2026-05-15-1755-mpc-continuous-swing-window-implementation.md) |
| 2026-05-13 18:10 | T300d MPC all-speed gait probe | pass as reproduction across directions | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1810-mpc-all-speed-gait-probe.md](log/2026-05-13-1810-mpc-all-speed-gait-probe.md) |
| 2026-05-13 18:15 | T300d MPC command switch gait probe | pass as switch reproduction | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1815-mpc-command-switch-gait-probe.md](log/2026-05-13-1815-mpc-command-switch-gait-probe.md) |
| 2026-05-13 17:57 | T300d MPC yaw gait failure probe | pass as reproduction | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1757-mpc-yaw-gait-failure-probe.md](log/2026-05-13-1757-mpc-yaw-gait-failure-probe.md) |
| 2026-05-13 13:52 | T300d MPC yaw foot alternation reproduction | pass as reproduction | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1352-mpc-yaw-foot-alternation-reproduction.md](log/2026-05-13-1352-mpc-yaw-foot-alternation-reproduction.md) |
| 2026-05-13 14:29 | T300d MPC yawfix direction sweep | pass with candidate caveats | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1429-mpc-yawfix-direction-sweep.md](log/2026-05-13-1429-mpc-yawfix-direction-sweep.md) |
| 2026-05-13 16:11 | T300d MPC yawfix4-plus long sweep | pass with no production-ready local mask | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1611-mpc-yawfix4plus-long-sweep.md](log/2026-05-13-1611-mpc-yawfix4plus-long-sweep.md) |
| 2026-05-13 16:33 | T300d MPC yaw anchor memory production implementation | pass with short smoke, long acceptance pending | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1633-mpc-yaw-anchor-memory-production.md](log/2026-05-13-1633-mpc-yaw-anchor-memory-production.md) |
| 2026-05-13 12:53 | T300d MPC dir15/dir19 production grounding | pass with scoped verification | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1253-mpc-dir15-dir19-production-grounding.md](log/2026-05-13-1253-mpc-dir15-dir19-production-grounding.md) |
| 2026-05-13 10:25 | T300d MPC touchdown-grounding probe | pass with new airborne-touchdown signal | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-1025-mpc-touchdown-grounding-probe.md](log/2026-05-13-1025-mpc-touchdown-grounding-probe.md) |
| 2026-05-13 09:32 | T300d MPC mixed-command and sequence long-horizon sweep | pass with new mixed/yaw transition side-effect signal | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-0932-mpc-mixed-sequence-long-horizon-sweep.md](log/2026-05-13-0932-mpc-mixed-sequence-long-horizon-sweep.md) |
| 2026-05-13 00:30 | T300d MPC long replan second direction expansion | pass with no new winner beyond `dir10` | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-13-0030-mpc-long-replan-second-direction-expansion.md](log/2026-05-13-0030-mpc-long-replan-second-direction-expansion.md) |
| 2026-05-12 23:46 | T300d MPC iterative long replan direction search | pass with selected direction | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-12-2346-mpc-long-replan-iterative-direction-search.md](log/2026-05-12-2346-mpc-long-replan-iterative-direction-search.md) |
| 2026-05-12 22:43 | T300d MPC long replan variant sweep | pass with no direct fix winner | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-12-2243-mpc-long-replan-variant-sweep.md](log/2026-05-12-2243-mpc-long-replan-variant-sweep.md) |
| 2026-05-12 21:47 | T300d MPC long replan foot drift reproduction | pass as reproduction | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-12-2147-mpc-long-replan-foot-drift-reproduction.md](log/2026-05-12-2147-mpc-long-replan-foot-drift-reproduction.md) |
| 2026-05-11 20:05 | T300d MPC leg-order command-matrix recovery | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-2005-mpc-leg-order-command-matrix-recovery.md](log/2026-05-11-2005-mpc-leg-order-command-matrix-recovery.md) |
| 2026-05-11 16:55 | T300d MPC long replan foot motion and yaw display fix | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1655-mpc-long-replan-foot-motion-and-yaw-display.md](log/2026-05-11-1655-mpc-long-replan-foot-motion-and-yaw-display.md) |
| 2026-05-11 16:35 | T300d MPC gait-coupling loss minimal fix | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1635-mpc-gait-coupling-loss-minimal-fix.md](log/2026-05-11-1635-mpc-gait-coupling-loss-minimal-fix.md) |
| 2026-05-11 15:43 | T300d MPC viewer forward static-joint IK fix | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1543-mpc-viewer-forward-static-joint-ik-fix.md](log/2026-05-11-1543-mpc-viewer-forward-static-joint-ik-fix.md) |
| 2026-05-11 15:28 | T300d MPC viewer flying-feet order regression | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1528-mpc-viewer-flying-feet-order-regression.md](log/2026-05-11-1528-mpc-viewer-flying-feet-order-regression.md) |
| 2026-05-11 15:05 | T300d MPC viewer entrypoint + replan autograd fix | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1505-t300d-mpc-viewer-entrypoint-and-autograd-replan-fix.md](log/2026-05-11-1505-t300d-mpc-viewer-entrypoint-and-autograd-replan-fix.md) |
| 2026-05-11 14:28 | T300d MPC 4096 train command max-iter1 success | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1428-mpc-4096-train-maxiter1-success.md](log/2026-05-11-1428-mpc-4096-train-maxiter1-success.md) |
| 2026-05-11 14:11 | T300d MPC terrain ray-shape OOM fix | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1411-mpc-terrain-ray-shape-oom-fix.md](log/2026-05-11-1411-mpc-terrain-ray-shape-oom-fix.md) |
| 2026-05-11 13:43 | human-16 mpc command update | pass | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1343-human16-mpc-command-update.md](log/2026-05-11-1343-human16-mpc-command-update.md) |
| 2026-05-11 13:18 | T300d 4096 runtime counters attempt | partial pass with 4096 runtime blocker | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1318-t300d-4096-runtime-counters-attempt.md](log/2026-05-11-1318-t300d-4096-runtime-counters-attempt.md) |
| 2026-05-11 12:43 | T300d env_isaacsim MPC headless runtime verification | pass with runtime-output caveat | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1243-t300d-env-isaacsim-mpc-headless-runtime.md](log/2026-05-11-1243-t300d-env-isaacsim-mpc-headless-runtime.md) |
| 2026-05-11 11:57 | T300d subagent implementation review/integration | pass (focused verification) | [T300/T300d](todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner) | [2026-05-11-1157-t300d-subagent-implementation-review.md](log/2026-05-11-1157-t300d-subagent-implementation-review.md) |
| 2026-05-11 11:10 | T300 spec hardening | pass (design hardened) | [T300/T300b](todo/T300-unified-dense-mpc-backend.md#t300b-subagent-review-convergence-and-spec-hardening-before-implementation-plan) | [2026-05-11-1110-t300-spec-hardening.md](log/2026-05-11-1110-t300-spec-hardening.md) |
| 2026-05-11 11:04 | T300 subagent design review convergence | partial pass with P0 hardening blockers | [T300/T300b](todo/T300-unified-dense-mpc-backend.md#t300b-subagent-review-convergence-and-spec-hardening-before-implementation-plan) | [2026-05-11-1104-t300-subagent-design-review.md](log/2026-05-11-1104-t300-subagent-design-review.md) |
| 2026-05-11 10:50 | T300 unified dense MPC backend design | design recorded | [T300/T300a](todo/T300-unified-dense-mpc-backend.md#t300a-written-spec-review-gate-and-implementation-plan-handoff) | [2026-05-11-1050-t300-unified-dense-mpc-backend-design.md](log/2026-05-11-1050-t300-unified-dense-mpc-backend-design.md) |
| 2026-05-10 22:23 | T116i main review and final verification | pass with runtime-output caveat | [T100/T116i](todo/T100-batched-together-planner-gpu-migration.md#t116i-nonzero-speed-candidates-and-hard-reason-diagnostics) | [2026-05-10-2223-t116i-main-review-final-verification.md](log/2026-05-10-2223-t116i-main-review-final-verification.md) |
| 2026-05-10 22:14 | T116i review fix viewer output and small runtime | pass with pytest-output caveat | [T100/T116i](todo/T100-batched-together-planner-gpu-migration.md#t116i-nonzero-speed-candidates-and-hard-reason-diagnostics) | [2026-05-10-2214-t116i-review-fix-viewer-output-small-runtime.md](log/2026-05-10-2214-t116i-review-fix-viewer-output-small-runtime.md) |
| 2026-05-10 21:56 | T116i nonzero speed and hard-reason implementation | pass with scoped runtime coverage | [T100/T116i](todo/T100-batched-together-planner-gpu-migration.md#t116i-nonzero-speed-candidates-and-hard-reason-diagnostics) | [2026-05-10-2156-t116i-nonzero-speed-hard-reason-implementation.md](log/2026-05-10-2156-t116i-nonzero-speed-hard-reason-implementation.md) |
| 2026-05-10 21:39 | compact-todo tree/Obsidian index hardening | pass | [T002/T002c](todo/T002-compact-todo-interactive-memory-and-test-grooming.md#t002c-tree-preserving-child-compaction-and-obsidian-index-hardening) | [2026-05-10-2139-compact-todo-tree-obsidian-index-hardening.md](log/2026-05-10-2139-compact-todo-tree-obsidian-index-hardening.md) |
| 2026-05-10 21:38 | T116i nonzero speed and hard-reason todo | todo recorded | [T100/T116i](todo/T100-batched-together-planner-gpu-migration.md#t116i-nonzero-speed-candidates-and-hard-reason-diagnostics) | [2026-05-10-2138-t116i-nonzero-speed-hard-reason-todo.md](log/2026-05-10-2138-t116i-nonzero-speed-hard-reason-todo.md) |
| 2026-05-10 18:43 | compact-todo non-T116 subtree compression | pass | [T002/T002b](todo/T002-compact-todo-interactive-memory-and-test-grooming.md#t002b-live-usage-pressure-verification) | [2026-05-10-1843-compact-todo-non-t116-subtree-compression.md](log/2026-05-10-1843-compact-todo-non-t116-subtree-compression.md) |
| 2026-05-10 21:02 | T117 approved test deletion | pass | [T117/T117c](todo/T117-together-planner-test-and-todo-cleanup.md#t117c-remove-non-mainline-parity-and-historical-traceability-surfaces) | [2026-05-10-2102-t117-approved-test-deletion.md](log/2026-05-10-2102-t117-approved-test-deletion.md) |
| 2026-05-10 20:43 | compact-todo together-planner test cleanup scan | scan recorded | [T100/T117](todo/T100-batched-together-planner-gpu-migration.md#t117-together-planner-test-and-todo-cleanup-after-t116h) | [2026-05-10-2043-compact-todo-together-planner-test-cleanup-scan.md](log/2026-05-10-2043-compact-todo-together-planner-test-cleanup-scan.md) |
| 2026-05-10 20:17 | T116h final review and authority | pass | [T100/T116h](todo/T100-batched-together-planner-gpu-migration.md#t116h-final-integration-authoritative-rerun-review-and-noteslog-closure) | [2026-05-10-2017-t116h-final-review-authority.md](log/2026-05-10-2017-t116h-final-review-authority.md) |
| 2026-05-10 19:53 | T116g `env_isaacsim` runtime diagnostics | pass | [T100/T116g](todo/T100-batched-together-planner-gpu-migration.md#t116g-env_isaacsim-headless-runtime-diagnostics-and-acceptance-tests) | [2026-05-10-1953-t116g-env-isaacsim-runtime-diagnostics.md](log/2026-05-10-1953-t116g-env-isaacsim-runtime-diagnostics.md) |

## Maintenance

- Keep this page short: dashboard only.
- Put detailed background and conclusions in branch pages.
- Put metrics and command output in per-test logs.
- Use `$compact-todo` when this page, branch pages, or log index grow too large.
