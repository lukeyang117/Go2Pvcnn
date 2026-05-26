# Investigation Dashboard

This page is the fast-start dashboard for agent work. It is not a full database. Detailed memory lives in [todo/](todo/); evidence lives in [log/](log/).

## Start Here

- Current focus: **T302k parametric MPC trajectory contract**.
  - The user approved a structural redesign because dense per-frame `foot_pos_residual` tuning is now the blocker.
  - Active design: [MPC parametric trajectory contract](../docs/superpowers/specs/2026-05-26-mpc-parametric-trajectory-contract-design.md).
  - Active execution page and implementation todo: [T302k](todo/T302k-parametric-mpc-trajectory-contract.md).
  - Current implementation status:
    - T302k.1-T302k.6 are implemented locally in the working tree;
    - default `plan_segment` now uses parametric decode + sampled losses + IK + FK-realized foot export;
    - T302k.9 local semantic/endpoint losses are partially implemented;
    - T302k.10 fixed the parametric foot curve timing so diagonal trot pairs alternate instead of all four feet moving together;
    - T302k.11 makes every parametric replan export frame0 foot positions from the current IsaacLab state, while frame1+ remains FK-realized; IsaacLab confirms replan initial foot error `0.0`, but initial touchdown-to-current-foot error remains up to `0.447m`;
    - viewer MPC now rotates root/body-frame `vx/vy` commands into world-frame commands at the viewer boundary before calling `plan_segment`;
    - T302k.14 makes parametric root roll/pitch follow the contact-weighted foot support plane after frame0, so downhill/downstairs terrain no longer leaves the planned root orientation horizontal;
    - T302k.15 body-relative foot drift is locally fixed for the major long-step/mid-replan accumulation: full-cycle terminal feet now anchor to a stable body-yaw footprint under the terminal root, with four-leg touchdown deltas de-meaned; `env_isaacsim` reduces lateral total drift to `~0.088m`, yaw to `~0.132-0.140m`, and z drift to `~0.009m`;
    - IsaacLab GPU3 smoke shows low-small foot-over succeeds with the trot-phase decode, but high-small/large semantic acceptance still fails;
    - T302k.8 cleanup removed the old dense residual planner modules and config switch; current `plan_segment` is parametric-only.
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
| T302k | active | Approved replacement for dense-foot residual MPC; sampled parametric losses now run and low-small works, trot pair timing is fixed, but high/large rolling acceptance and touchdown endpoint quality are not accepted yet. | Continue [T302k](todo/T302k-parametric-mpc-trajectory-contract.md). |
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
| T302k | active | parametric MPC trajectory contract | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | default planner path is parametric FK-realized with sampled losses; low-small smoke passes, high/large still fails after T302k.9 local slice | helper commit `1b799cd`; verified working tree |
| T302g | context | MPC semantic RL config | [T302g](todo/T302g-mpc-semantic-rl-training-config.md) | 4096 timing capture remains open but unrelated to immediate parametric contract | working tree evidence |
| T301 | context | viewer reset/step-mode | [T301](todo/T301-viewer-r-key-grounded-reset.md) | viewer controls are background unless testing manual playback | step-mode/reset logs |
| T200 | done | semantic static course | [T200](todo/T200-semantic-static-course-viewer.md) | semantic terrain support is background | feature `130c635` |

## Open Leaves

| Leaf | Parent | Status | Priority | Why Active | Next Read |
| --- | --- | --- | --- | --- | --- |
| T302k.7 | T302k | partial | P0 | Real IsaacLab smoke: low-small translation passes; high-small/large and endpoint quality still fail. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) |
| T302k.9 | T302k | partial | P0 | Local semantic avoidance/root shaping and endpoint losses exist, but high/large IsaacLab acceptance still fails. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) |
| T302k.10 | T302k | verify | P0 | Parametric foot curves now use per-leg local swing phase so diagonal trot pairs alternate instead of all four feet swinging together. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) |
| T302k.11 | T302k | verify | P0 | Replan output now starts from current IsaacLab foot positions at frame0 instead of FK-rewriting the initial feet. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) |
| T302k.12 | T302k | todo | P0 | Replan touchdowns still do not match current stance/current foot; IsaacLab max initial touchdown-to-current-foot error `0.447m`. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) |
| T302k.13 | T302k | verify | P1 | Viewer teleop/scripted `vx/vy` is root/body-frame; MPC receives world-frame XY after root-yaw rotation. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) |
| T302k.14 | T302k | verify | P1 | Parametric root roll/pitch now follows the foot support plane on sloped/downstairs terrain after frame0. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) |
| T302k.15 | T302k | verify | P0 | Long-step mid-replan major body-relative foot drift is reduced by stable terminal body-footprint anchoring; visual acceptance and residual yaw body-x drift remain to check. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) |
| T302k.8 | T302k | done | P1 | Obsolete dense residual planner modules and tests were removed; source scan confirms no old dense imports remain. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#t302k8-dense-path-retirement) |
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
| 2026-05-26 20:40 | T302k long-step root-relative foot drift reproduction | reproduced; body-yaw relative foot coordinates drift across 8 mid-replan cycles while frame0/readback alignment remains near zero | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-2040-t302k-long-step-root-relative-foot-drift-repro.md](log/2026-05-26-2040-t302k-long-step-root-relative-foot-drift-repro.md) |
| 2026-05-26 21:33 | T302k body-relative foot anchor fix | pass for major accumulation; `env_isaacsim` 8-cycle probe keeps horizon `25` and reduces lateral/yaw/z body-relative drift while preserving frame0/readback alignment | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-2133-t302k-body-relative-foot-anchor-fix.md](log/2026-05-26-2133-t302k-body-relative-foot-anchor-fix.md) |
| 2026-05-26 20:21 | T302k support-plane root roll/pitch | pass locally and in `env_isaacsim`; parametric decode now estimates roll/pitch from the contact-weighted foot support plane while preserving frame0 current orientation | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-2021-t302k-support-plane-root-roll-pitch.md](log/2026-05-26-2021-t302k-support-plane-root-roll-pitch.md) |
| 2026-05-26 19:49 | viewer MPC body-frame command | pass locally; root/body-frame `vx/vy` rotates by current root yaw before MPC planning; focused viewer tests pass | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1949-viewer-mpc-body-frame-command.md](log/2026-05-26-1949-viewer-mpc-body-frame-command.md) |
| 2026-05-26 17:57 | T302k dense path retirement | pass locally; old dense residual modules/config switch removed, focused suite passed, IsaacLab cleanup smoke runs with existing touchdown mismatch | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1757-t302k-dense-path-retirement.md](log/2026-05-26-1757-t302k-dense-path-retirement.md) |
| 2026-05-26 17:17 | T302k IsaacLab current-foot/touchdown replan check | partial; frame0 foot matches IsaacLab current foot, but planned touchdown still differs from current foot | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1717-t302k-isaaclab-current-foot-touchdown-check.md](log/2026-05-26-1717-t302k-isaaclab-current-foot-touchdown-check.md) |
| 2026-05-26 17:13 | T302k parametric current-foot replan anchor | pass locally; frame0 exported feet now match current IsaacLab state, frame1+ remains FK-realized | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1713-t302k-parametric-current-foot-replan-anchor.md](log/2026-05-26-1713-t302k-parametric-current-foot-replan-anchor.md) |
| 2026-05-26 16:49 | T302k parametric trot-phase foot curves | pass for gait timing and low-small smoke; high/large still open | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1649-t302k-parametric-trot-phase-foot-curves.md](log/2026-05-26-1649-t302k-parametric-trot-phase-foot-curves.md) |
| 2026-05-26 15:54 | T302k parametric semantic and endpoint losses | partial; local tests pass and low-small remains good, but high/large semantic acceptance still fails | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1554-t302k-parametric-semantic-endpoint-losses.md](log/2026-05-26-1554-t302k-parametric-semantic-endpoint-losses.md) |
| 2026-05-26 15:24 | T302k sampled losses and IsaacLab smoke | partial; local tests pass and low-small translation crosses, but high/large semantic acceptance fails | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1524-t302k-parametric-sampled-loss-and-isaaclab-smoke.md](log/2026-05-26-1524-t302k-parametric-sampled-loss-and-isaaclab-smoke.md) |
| 2026-05-26 14:50 | T302k parametric default FK output | pass; default planner path uses parametric curves and exports FK-realized feet; full backend+parametric tests `122 passed` | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1450-t302k-parametric-default-fk-output.md](log/2026-05-26-1450-t302k-parametric-default-fk-output.md) |
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
