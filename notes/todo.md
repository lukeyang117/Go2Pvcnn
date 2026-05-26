# Investigation Dashboard

This page is the fast-start dashboard for agent work. It is not a full database. Detailed memory lives in [todo/](todo/); evidence lives in [log/](log/).

## Start Here

- Current focus: **T302k parametric MPC trajectory contract**.
  - The user approved a structural redesign because dense per-frame `foot_pos_residual` tuning is now the blocker.
  - Active design: [MPC parametric trajectory contract](../docs/superpowers/specs/2026-05-26-mpc-parametric-trajectory-contract-design.md).
  - Active execution page and implementation todo: [T302k](todo/T302k-parametric-mpc-trajectory-contract.md).
  - Core contract:
    - optimize touchdown `xy` only;
    - derive touchdown `z` from `height_at(terrain, touchdown_xy)`;
    - optimize root and foot cubic-curve parameters;
    - sample 25 frames for losses;
    - run clamped IK and export FK-realized feet.
- Do not continue old V9/V10/V11/V12 scalar-loss tuning unless it is needed as a regression comparison.
- T302h/T302i/T302j are now context/evidence branches for T302k, not the main execution route.

## Status Legend

- `active`: current execution front.
- `verify`: implemented or evidenced, needs broader confidence or cleanup.
- `context`: useful background, not the next execution target.
- `done`: historical or closed.

## Active Fronts

| Front | State | Why It Matters Now | Next Step |
| --- | --- | --- | --- |
| T302k | active | Approved replacement for dense-foot residual MPC; carries the implementation todo for parametric root/foot curves and FK-realized output. | Start [T302k.1](todo/T302k-parametric-mpc-trajectory-contract.md#t302k1-parametric-geometry-helpers). |
| T302j | context | Latest runtime evidence shows structured low-small touchdown/swing coupling still fails FK foot-over/height; this motivates T302k. | Read only for evidence before touching current `planner.py`. |
| T302i | context | Root cause: planned Cartesian foot/touchdown targets can be unreachable after clamped IK; viewer/Isaac readback matches FK. | Preserve metrics and probes for T302k acceptance. |
| T302h | context | Production low-small/high-small/large behavior mostly passes task gates, but visual/R2 and planned-vs-realized mismatch remain. | Use as non-regression baseline. |
| T302g | context | Semantic RL rollout front remains open for 4096 metric capture, but it is not the current trajectory-contract task. | Defer until T302k acceptance or explicit user request. |

## Root Map

| Root | Status | Stage | Branch | Current | Refs |
| --- | --- | --- | --- | --- | --- |
| T000 | done | notes workflow | [T000](todo/T000-notes-workflow.md) | memory system bootstrapped and linked into existing notes | feature `7cf6c11`; verified `7cf6c11` |
| T002 | verify | compact-todo workflow | [T002](todo/T002-compact-todo-interactive-memory-and-test-grooming.md) | skill preserves child-tree/Obsidian index paths; grouped stale-test cleanup remains | feature `pending`; verified live compact passes |
| T100 | context | batched together planner | [T100](todo/T100-batched-together-planner-gpu-migration.md) | T116/T116i old together-planner small crossing is closed; not current MPC route | final evidence in T116 logs |
| T300 | context | unified dense MPC backend | [T300](todo/T300-unified-dense-mpc-backend.md) | original dense MPC backend and swing-window history; superseded for current trajectory representation by T302k | see T300/T300e logs |
| T302 | context | MPC collision/semantic behavior | [T302](todo/T302-mpc-body-leg-height-field-collision-safety.md) | collision/semantic baseline remains important for non-regression | strict JSONL `17/17` pass |
| T302h | context | semantic obstacle jitter/crossing | [T302h](todo/T302h-semantic-obstacle-jitter-reproduction.md) | task-level semantic behavior mostly passes; exposes remaining visual/reachability mismatch | rolling25 low-small production log |
| T302i | context | viewer realized-foot mismatch | [T302i](todo/T302i-viewer-realized-foot-mismatch.md) | clamp trace proves output/feasibility contract issue | clamp trace + reachable baseline logs |
| T302j | context | touchdown endpoint consistency | [T302j](todo/T302j-touchdown-endpoint-consistency.md) | endpoint/export repairs are partial and motivate parametric trajectory redesign | structured touchdown runtime log |
| T302k | active | parametric MPC trajectory contract | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | implementation todo written; ready for execution | design commit `d922eef` |
| T302g | context | MPC semantic RL config | [T302g](todo/T302g-mpc-semantic-rl-training-config.md) | 4096 timing capture remains open but unrelated to immediate parametric contract | working tree evidence |
| T301 | context | viewer reset/step-mode | [T301](todo/T301-viewer-r-key-grounded-reset.md) | viewer controls are background unless testing manual playback | step-mode/reset logs |
| T200 | done | semantic static course | [T200](todo/T200-semantic-static-course-viewer.md) | semantic terrain support is background | feature `130c635` |

## Open Leaves

| Leaf | Parent | Status | Priority | Why Active | Next Read |
| --- | --- | --- | --- | --- | --- |
| T302k.1 | T302k | todo | P0 | Establish command-frame axes, bounded Bezier parameters, and curve sampling helpers. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k1-parametric-geometry-helpers) |
| T302k.2 | T302k | todo | P0 | Replace dense foot frame variables with compact parametric variables. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k2-parametric-variables) |
| T302k.3 | T302k | todo | P0 | Decode root/foot curves and grounded touchdowns into 25 sampled frames. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k3-curve-decode) |
| T302k.4 | T302k | todo | P0 | Export FK-realized feet behind a config/debug switch. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k4-fk-realized-output) |
| T302k.5 | T302k | todo | P0 | Port losses to sampled curves: reachability, collision, semantic, gait, command, regularization. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k5-parametric-losses) |
| T302k.6 | T302k | todo | P0 | Add low-small parametric probe variant and local/unit acceptance. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k6-low-small-probe) |
| T302k.7 | T302k | todo | P0 | Run real IsaacLab rolling25 low-small/high-small/large acceptance and log it. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k7-isaaclab-acceptance) |
| T302k.8 | T302k | todo | P1 | Retire obsolete dense-foot residual repair path after parametric acceptance. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k8-dense-path-retirement) |
| T302g.5 | T302g | partial | P2 | 4096 metric capture remains open but is not part of T302k. | [T302g](todo/T302g-mpc-semantic-rl-training-config.md) |

## Branch Pages

- [todo/README.md](todo/README.md)
- [T000-notes-workflow.md](todo/T000-notes-workflow.md)
- [T001-inspire-skill-design.md](todo/T001-inspire-skill-design.md)
- [T002-compact-todo-interactive-memory-and-test-grooming.md](todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
- [T100-batched-together-planner-gpu-migration.md](todo/T100-batched-together-planner-gpu-migration.md)
- [T117-together-planner-test-and-todo-cleanup.md](todo/T117-together-planner-test-and-todo-cleanup.md)
- [T100-pre-t116-history.md](todo/T100-pre-t116-history.md)
- [T300-unified-dense-mpc-backend.md](todo/T300-unified-dense-mpc-backend.md)
- [T300e-mpc-continuous-swing-window-plan.md](todo/T300e-mpc-continuous-swing-window-plan.md)
- [T302-mpc-body-leg-height-field-collision-safety.md](todo/T302-mpc-body-leg-height-field-collision-safety.md)
- [T302g-mpc-semantic-rl-training-config.md](todo/T302g-mpc-semantic-rl-training-config.md)
- [T302h-semantic-obstacle-jitter-reproduction.md](todo/T302h-semantic-obstacle-jitter-reproduction.md)
- [T302i-viewer-realized-foot-mismatch.md](todo/T302i-viewer-realized-foot-mismatch.md)
- [T302j-touchdown-endpoint-consistency.md](todo/T302j-touchdown-endpoint-consistency.md)
- [T302k-parametric-mpc-trajectory-contract.md](todo/T302k-parametric-mpc-trajectory-contract.md)
- [T301-viewer-r-key-grounded-reset.md](todo/T301-viewer-r-key-grounded-reset.md)
- [T200-semantic-static-course-viewer.md](todo/T200-semantic-static-course-viewer.md)

## Recent Logs

| Time | Topic | Result | Todo | File |
| --- | --- | --- | --- | --- |
| 2026-05-26 14:27 | T302k parametric MPC todo and dashboard cleanup | plan/todo recorded; old T302h/i/j fronts demoted to context; no code changed | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1427-t302k-parametric-mpc-todo-dashboard-cleanup.md](log/2026-05-26-1427-t302k-parametric-mpc-todo-dashboard-cleanup.md) |
| 2026-05-26 13:36 | T302j structured low-small touchdown runtime | partial improvement, not accepted; marker/swing conflict reduced but FK foot-over/height remain open | [T302j](todo/T302j-touchdown-endpoint-consistency.md) | [2026-05-26-1336-t302j-structured-low-small-touchdown-runtime.md](log/2026-05-26-1336-t302j-structured-low-small-touchdown-runtime.md) |
| 2026-05-26 12:59 | T302j low-small crossing acceptance contract | pass; current default MPC fails only no-above-root in forward case | [T302j](todo/T302j-touchdown-endpoint-consistency.md) | [2026-05-26-1259-t302j-low-small-crossing-acceptance-test-contract.md](log/2026-05-26-1259-t302j-low-small-crossing-acceptance-test-contract.md) |
| 2026-05-25 17:23 | T302i IK clamp foot mismatch trace | pass; unreachable Cartesian target after calf clamp causes `0.286667m` foot mismatch | [T302i](todo/T302i-viewer-realized-foot-mismatch.md) | [2026-05-25-1723-t302i-ik-clamp-foot-mismatch-trace.md](log/2026-05-25-1723-t302i-ik-clamp-foot-mismatch-trace.md) |
| 2026-05-25 12:22 | T302h rolling25 low-small foot-over production | pass for task gate; R2/playback mismatch remain | [T302h](todo/T302h-semantic-obstacle-jitter-reproduction.md) | [2026-05-25-1222-t302h-rolling25-low-small-foot-over-production.md](log/2026-05-25-1222-t302h-rolling25-low-small-foot-over-production.md) |
| 2026-05-22 13:58 | MPC swing trajectory quality reproduction | pass as reproduction; optimized `foot_pos_residual` first breaks swing shape | [T300](todo/T300-unified-dense-mpc-backend.md) | [2026-05-22-1358-mpc-swing-trajectory-quality-reproduction.md](log/2026-05-22-1358-mpc-swing-trajectory-quality-reproduction.md) |

## Maintenance

- Keep this page short: dashboard only.
- Put detailed background and conclusions in branch pages.
- Put metrics and command output in per-test logs.
- Use `$compact-todo` when this page, branch pages, or log index grow too large.
