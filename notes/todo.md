# Investigation Dashboard

This page is the fast-start dashboard for agent work. Detailed memory lives in [todo/](todo/); evidence lives in [log/](log/).

## Start Here

- Current focus: **T302k low-small MPC parameter tuning / residual reachability checks**.
- Active branch page: [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md).
- Active implementation plan: [T302k low-small loss redesign plan](todo/T302k-low-small-loss-redesign-plan.md).
- Active code surface:
  - [Go2Pvcnn/extension/batch_mpc_planner/participation.py](../Go2Pvcnn/extension/batch_mpc_planner/participation.py)
  - [Go2Pvcnn/extension/batch_mpc_planner/manager.py](../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [Go2Pvcnn/extension/reference/cache.py](../Go2Pvcnn/extension/reference/cache.py)
  - [Go2Pvcnn/extension/mdp/rewards_reference.py](../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [Go2Pvcnn/extension/mdp/semantic_contact_rewards.py](../Go2Pvcnn/extension/mdp/semantic_contact_rewards.py)
  - [Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/](../Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/)
  - [Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [Go2Pvcnn/scripts/play.py](../Go2Pvcnn/scripts/play.py)
  - [Go2Pvcnn/extension/viz/go2_foostep_planner.py](../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py](../Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py)
  - [Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [Go2Pvcnn/extension/batch_mpc_planner/planner.py](../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [Go2Pvcnn/tests/test_batch_mpc_parametric.py](../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
  - [Go2Pvcnn/tests/test_batch_mpc_backend.py](../Go2Pvcnn/tests/test_batch_mpc_backend.py)
- Current contract:
  - T302m cleanup is implemented locally: production train/play/register/factory/viewer are narrowed to `teacher_elevation_trajectory_mpc_semantic + mpc`; old `batched_planner`, `batched_together_planner`, old teacher cfgs, old script entrypoints, and production debug variants are deleted from the working tree.
  - T302m MPC tuning entry is unified locally: `TeacherElevationTrajectoryMpcSemanticEnvCfg` now tunes planner runtime/diagnostics/participation through `mpc_planner_cfg: MpcPlannerCfg`; production task cfg no longer exposes duplicated top-level MPC aliases.
  - T302m MPC participation config is now blacklist-only: `include_terrain_cols`, `include_terrain_names`, and `include_terrain_rows` were removed; use `exclude_pairs` to remove envs from MPC reference participation.
  - T302m local/static verification passed: cleanup guards `3 passed`, viewer tests `16 passed`, current focused suite `43 passed`, backend suite `128 passed`, production pycompile pass, and production old-route scan has no matches.
  - T302m real IsaacLab acceptance passed on card1 after fixing train/play local `rsl_rl` imports, the RSL-RL wrapper observation contract, and the active PPO config: contact drop probe pass, 1024-env 1-iteration train smoke pass, 1024/64/25-step performance `epoch_seconds=5.8828s`.
  - T302l design approved in [../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html](../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html).
  - T302l implementation plan lives in [todo/T302l-mpc-rl-participation-and-reward-plan.md](todo/T302l-mpc-rl-participation-and-reward-plan.md).
  - T302l PLAY/VIEWER split is verified locally and in `env_isaacsim`: `TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY` is no-MPC for `scripts/play.py`; `TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER` preserves MPC viewer behavior; `model_14000.pt` headless play ran 5 steps with no planner attach.
  - MPC RL runtime must align `reference_trajectory_horizon = reference_replan_interval_steps = 25`.
  - Only selected envs participate in MPC reference reward; selection filters by terrain/difficulty and excludes only AND-matching terrain+difficulty pairs.
  - `reference_foot_pos_reward()` must compare IsaacLab and MPC feet in world frame.
  - RL semantic collision reward must use IsaacLab real contact, not semantic height-map collision approximation.
  - Current T302l semantic contact route uses 2 custom global semantic sensors, `semantic_contact_small` and `semantic_contact_large`, each covering all selected robot bodies and all `row_*/col_*/slot_*` semantic objects.
  - Final semantic contact acceptance on card1 passed with `num_envs=1024`, `force_matrix_w` shapes `[1024, 13, 640, 3]` and `[1024, 13, 100, 3]`, and `epoch_seconds=5.6489s` for 1024 env / 64 MPC env / 25 steps.
  - Robot-drop semantic contact probe on card1 passed: small and large obstacle contacts are detectable with no NaN/Inf and no empty-env cross-talk; small-obstacle contact is much sparser than large-obstacle contact in the controlled drop setup.
  - Design approved in [../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html).
  - Implementation plan lives in [todo/T302k-low-small-loss-redesign-plan.md](todo/T302k-low-small-loss-redesign-plan.md).
  - Task 1 restored the nominal extraction contract locally: `semantic_policy.py` builds `ParametricTrajectoryNominal`, `planner.py` builds nominal before optimization, and decode consumes `nominal + variables`.
  - Task 2 added optional `is_plane_terrain` metadata through scanner terrain construction, subset, planner normalization, and MPC manager IsaacLab terrain type inference.
  - Task 3 added GPU low-small component circle approximation in `semantic_geometry.py`.
  - Task 4 replaced sampled `parametric_low_small_crossing` with `parametric_touchdown_keepout`.
  - Task 5 added sampled `parametric_swing_foot_clearance`.
  - Task 6 added FK realized `parametric_fk_body_leg_collision` and it now participates in the sampled Adam loss path.
  - Task 7 added `parametric_trajectory_fk_consistency` and it now participates in the sampled Adam loss path.
  - Task 8 added sampled `parametric_plane_root_z_target` gated by `is_plane_terrain`.
  - Task 9 added plane-only low-small FK semantic collision probe metrics and JSONL GPU/run metadata. The diagnostic now runs after optimization, uses rolling segment terrain snapshots, and counts FK semantic collision only on crossing-triggered legs.
  - Full matrix on GPU0 passed hard acceptance for covered crossing rows: `20` cycle rows, `12` covered rows, `0` FK semantic collisions, max crossing FK error `0.0634m`; four rows exceed preferred `0.05m` but stay within accepted `0.08m`.
  - New low-small direction: no hard projection, no touchdown snapping, no hard foot separation; debug by tuning confirmed loss weights/parameters only.
- Old dense residual MPC (`nominal.py`, `optimizer.py`, `variables.py`, `losses/registry.py`) is retired. Do not reopen V9/V10/V11/V12 scalar-loss branches unless explicitly requested.

## Status Legend

- `active`: current execution front.
- `verify`: implemented/evidenced, keep as regression guard.
- `context`: useful background, not current work.
- `done`: closed history.
- `closed`: unfinished historical route closed by the current T302k direction.

## Active Fronts

| Front | State | Why It Matters Now | Next Step |
| --- | --- | --- | --- |
| T302m | verify | Current working tree has been cleaned to the single semantic MPC route; MPC tuning is unified under `mpc_planner_cfg`; participation filtering is blacklist-only; local/static tests pass and card1 IsaacLab acceptance passes. | Keep as regression guard; run train smoke only if changing task cfg/runtime wiring again. |
| T302l | verify | MPC participation/contact route and PLAY/VIEWER split are verified; PLAY no longer attaches MPC, viewer uses VIEWER cfg, and low-small hard metrics remain clean. | Keep as regression guard; rerun PLAY smoke only if changing play wrapper, policy observation shape, or task cfg. |
| T302k | active | Current parametric MPC path; low-small loss redesign implementation is verified on covered rows, with only parameter tuning left unless user approves a new loss. | Inspect loss breakdown and tune confirmed parameters only if continuing soft FK-error reduction. |

## Root Map

| Root | Status | Stage | Branch | Current | Refs |
| --- | --- | --- | --- | --- | --- |
| T302m | verify | teacher elevation MPC semantic cleanup | [T302m](todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md) | Single-route cleanup implemented locally; card1 IsaacLab acceptance and 1024/64/25 performance pass | design [2026-05-31](../docs/superpowers/specs/2026-05-31-teacher-elevation-mpc-semantic-cleanup-design.html) |
| T302l | verify | MPC RL participation and reward integration | [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md) | Selector/world-foot/global contact work and PLAY/VIEWER split verified; prior card1 1024 quantity/perf acceptance retained | design [2026-05-30](../docs/superpowers/specs/2026-05-30-mpc-rl-participation-and-runtime-design.html) |
| T302k | active | parametric MPC trajectory contract | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | Low-small loss redesign and plane-only FK semantic collision testing | design commit `97c5b60` |
| T302h | closed | semantic obstacle jitter/crossing evidence | [T302h](todo/T302h-semantic-obstacle-jitter-reproduction.md) | Closed as implementation route; retained as reproduction/evidence for T302k | rolling25 low-small production evidence |
| T302i | closed | viewer realized-foot mismatch evidence | [T302i](todo/T302i-viewer-realized-foot-mismatch.md) | Closed as loss-sweep route; IK/FK mismatch evidence retained for T302k reachability | clamp trace and reachable probes |
| T302j | closed | touchdown endpoint consistency evidence | [T302j](todo/T302j-touchdown-endpoint-consistency.md) | Closed as dense/default-MPC endpoint route; endpoint lessons folded into T302k | structured touchdown logs |
| T302g | context | MPC semantic RL config | [T302g](todo/T302g-mpc-semantic-rl-training-config.md) | Deferred until parametric planner behavior stabilizes | global-sync sampled MPC evidence |
| T302 | context | MPC collision/semantic baseline | [T302](todo/T302-mpc-body-leg-height-field-collision-safety.md) | Baseline metric history only | strict JSONL history |
| T300 | context | old dense MPC backend | [T300](todo/T300-unified-dense-mpc-backend.md) | Superseded by T302k | dense path retired |
| T100 | context | batched together planner | [T100](todo/T100-batched-together-planner-gpu-migration.md) | Historical non-MPC planner path | keep for rollback/context |
| T301 | context | viewer reset/step mode | [T301](todo/T301-viewer-r-key-grounded-reset.md) | Viewer controls background | use only for viewer regressions |
| T200 | done | semantic static course | [T200](todo/T200-semantic-static-course-viewer.md) | Course/runtime support complete enough for current planner work | feature `130c635` |
| T002 | done | compact-todo workflow | [T002](todo/T002-compact-todo-interactive-memory-and-test-grooming.md) | Skill implemented; this session used it for cleanup | compact session logs |
| T000 | done | notes workflow | [T000](todo/T000-notes-workflow.md) | memory system bootstrapped | feature `7cf6c11` |

## Open Leaves

| Leaf | Parent | Status | Priority | Why Active | Next Read |
| --- | --- | --- | --- | --- | --- |
| T302m.1 | T302m | verify | P0 | Route cleanup, local regression, card1 contact/drop, 1024-env train smoke, and 1024/64/25-step perf pass. | [T302m cleanup plan](todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md) |
| T302l.1 | T302l | verify | P0 | Two global semantic contact sensors are implemented; exact-path body resolution fixes 1024-env `gym.make` stall; card1 quantity and 25-step performance pass. | [T302l implementation plan](todo/T302l-mpc-rl-participation-and-reward-plan.md) |
| T302l.2 | T302l | verify | P0 | PLAY/VIEWER split verified: PLAY no planner attach with `model_14000.pt`, VIEWER cfg static contract covered, low-small regression FK semantic collisions `0`. | [Task 20](todo/T302l-mpc-rl-participation-and-reward-plan.md#task-20-play--viewer-cfg-split) |
| T302k.12 | T302k | active | P0 | Replan touchdown/current-foot and touchdown IK/FK mismatch remain the main trajectory/reachability issue. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#open-children) |
| T302k.18 | T302k | verify | P0 | Low-small loss redesign is implemented and hard acceptance passes on covered full-matrix rows; remaining work is parameter tuning only unless user approves new loss. | [T302k low-small loss redesign plan](todo/T302k-low-small-loss-redesign-plan.md) |
| T302k.17 | T302k | verify | P0 | Nominal extraction Task 1 is implemented, committed, and covered by local regression tests. | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md#open-children) |

## Branch Pages

- [todo/README.md](todo/README.md)
- [T302m-teacher-elevation-mpc-semantic-cleanup-plan.md](todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md)
- [T302l-mpc-rl-participation-and-reward-plan.md](todo/T302l-mpc-rl-participation-and-reward-plan.md)
- [T302k-parametric-mpc-trajectory-contract.md](todo/T302k-parametric-mpc-trajectory-contract.md)
- [T302k-low-small-loss-redesign-plan.md](todo/T302k-low-small-loss-redesign-plan.md)
- [T302h-semantic-obstacle-jitter-reproduction.md](todo/T302h-semantic-obstacle-jitter-reproduction.md)
- [T302i-viewer-realized-foot-mismatch.md](todo/T302i-viewer-realized-foot-mismatch.md)
- [T302j-touchdown-endpoint-consistency.md](todo/T302j-touchdown-endpoint-consistency.md)
- [T302g-mpc-semantic-rl-training-config.md](todo/T302g-mpc-semantic-rl-training-config.md)
- [T302-mpc-body-leg-height-field-collision-safety.md](todo/T302-mpc-body-leg-height-field-collision-safety.md)
- [T300-unified-dense-mpc-backend.md](todo/T300-unified-dense-mpc-backend.md)
- [T100-batched-together-planner-gpu-migration.md](todo/T100-batched-together-planner-gpu-migration.md)
- [T301-viewer-r-key-grounded-reset.md](todo/T301-viewer-r-key-grounded-reset.md)
- [T200-semantic-static-course-viewer.md](todo/T200-semantic-static-course-viewer.md)

## Recent Logs

| Time | Topic | Result | Todo | File |
| --- | --- | --- | --- | --- |
| 2026-06-02 00:06 | T302l PLAY / VIEWER cfg split | pass; local focused tests pass, headless PLAY with `model_14000.pt` completes 5 steps with no planner attach, low-small covered rows `2` with FK semantic collisions `0` and max crossing FK error `0.0416m` | [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md) | [2026-06-02-0006-t302l-play-viewer-cfg-split.md](log/2026-06-02-0006-t302l-play-viewer-cfg-split.md) |
| 2026-05-31 23:06 | T302m card1 IsaacLab acceptance | pass; contact drop probe, 1024-env 1-iteration train smoke, and 1024/64/25-step perf pass; `epoch_seconds=5.8828s` | [T302m](todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md) | [2026-05-31-2306-t302m-card1-isaaclab-acceptance.md](log/2026-05-31-2306-t302m-card1-isaaclab-acceptance.md) |
| 2026-05-31 22:49 | T302m teacher elevation MPC semantic cleanup | local pass; cleanup guards `3 passed`, viewer `16 passed`, focused `43 passed`, backend `128 passed`; IsaacLab card3 smoke blocked by existing 20.6GB 1024-env train process causing OOM | [T302m](todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md) | [2026-05-31-2249-t302m-teacher-mpc-semantic-cleanup.md](log/2026-05-31-2249-t302m-teacher-mpc-semantic-cleanup.md) |
| 2026-05-31 10:49 | T302l semantic contact robot drop probe | pass; real robot drop detects small and large semantic contacts; empty envs stay zero; no NaN/Inf; small active frames `5` vs large active frames `150` | [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md) | [2026-05-31-1049-t302l-semantic-contact-robot-drop-probe.md](log/2026-05-31-1049-t302l-semantic-contact-robot-drop-probe.md) |
| 2026-05-30 23:13 | T302l semantic global contact card1 performance | pass after exact-path body resolution fix; card1 quantity alignment PASS, shapes `[1024,13,640,3]` and `[1024,13,100,3]`, 1024/64 probe `5.6489s` | [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md) | [2026-05-30-2313-t302l-semantic-global-contact-card1-perf.md](log/2026-05-30-2313-t302l-semantic-global-contact-card1-perf.md) |
| 2026-05-30 21:23 | T302l MPC RL final verification | pass; focused `7 passed`, backend `140 passed`, contact smoke PASS, 1024/64 probe `5.256s`, train entry PASS, low-small covered rows `0` FK semantic collisions and max crossing FK error `0.0634m`; PhysX global-filter warning recorded | [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md) | [2026-05-30-2123-t302l-final-verification.md](log/2026-05-30-2123-t302l-final-verification.md) |
| 2026-05-30 21:14 | T302l 1024/64 performance | pass; probe `5.256s <= 10s`, train entry exits `0` with `--planner-backend mpc` | [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md) | [2026-05-30-2114-t302l-rl-1024-64-performance.md](log/2026-05-30-2114-t302l-rl-1024-64-performance.md) |
| 2026-05-30 21:03 | T302l semantic contact smoke | pass in `env_isaacsim`; 26 per-body filtered contact sensors expose force matrices `[4,1,filter_count,3]` | [T302l](todo/T302l-mpc-rl-participation-and-reward-plan.md) | [2026-05-30-2103-t302l-semantic-contact-smoke.md](log/2026-05-30-2103-t302l-semantic-contact-smoke.md) |
| 2026-05-28 22:59 | T302k low-small full matrix and FK inner-loop losses | pass for hard acceptance on crossing-covered rows; max FK semantic collision `0`, max crossing FK error `0.0634m`; four rows remain soft tuning risk over `0.05m` | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2259-t302k-low-small-full-matrix-and-fk-inner-loop.md](log/2026-05-28-2259-t302k-low-small-full-matrix-and-fk-inner-loop.md) |
| 2026-05-28 21:06 | T302k plane low-small FK semantic collision probe | pass for metric/logging smoke; plane rows and required FK semantic keys present; crossing legs not covered in smoke | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2106-t302k-plane-low-small-fk-collision-probe.md](log/2026-05-28-2106-t302k-plane-low-small-fk-collision-probe.md) |
| 2026-05-28 21:25 | T302k plane root-z target | pass locally; plane-only root-z target sampled key added | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2125-t302k-plane-root-z-target.md](log/2026-05-28-2125-t302k-plane-root-z-target.md) |
| 2026-05-28 21:17 | T302k FK trajectory consistency | pass locally; final optimized-target vs FK-realized consistency key added | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2117-t302k-fk-trajectory-consistency.md](log/2026-05-28-2117-t302k-fk-trajectory-consistency.md) |
| 2026-05-28 21:10 | T302k FK body leg collision | pass locally; final loss key added for realized FK body/leg terrain collision, with post-optimization limitation recorded | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2110-t302k-fk-body-leg-collision.md](log/2026-05-28-2110-t302k-fk-body-leg-collision.md) |
| 2026-05-28 20:57 | T302k swing target clearance | pass locally; sampled loss key `parametric_swing_foot_clearance` added | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2057-t302k-swing-target-clearance.md](log/2026-05-28-2057-t302k-swing-target-clearance.md) |
| 2026-05-28 20:48 | T302k touchdown circle keepout | pass locally; sampled loss key is now `parametric_touchdown_keepout` | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2048-t302k-touchdown-circle-keepout.md](log/2026-05-28-2048-t302k-touchdown-circle-keepout.md) |
| 2026-05-28 20:34 | T302k low-small GPU circles | pass locally; fixed-shape component circles stay on input device | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2034-t302k-low-small-gpu-circles.md](log/2026-05-28-2034-t302k-low-small-gpu-circles.md) |
| 2026-05-28 20:25 | T302k plane terrain metadata | pass locally; `is_plane_terrain` flows through MPC terrain and manager infers `flat/plane` from IsaacLab terrain names | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2025-t302k-plane-terrain-metadata.md](log/2026-05-28-2025-t302k-plane-terrain-metadata.md) |
| 2026-05-28 20:14 | T302k nominal extraction contract | pass locally; decode consumes `nominal + variables`; pure-yaw high/large semantic candidate restored | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-28-2014-t302k-nominal-extraction-contract.md](log/2026-05-28-2014-t302k-nominal-extraction-contract.md) |
| 2026-05-28 | T302k low-small loss redesign design/plan | design committed and implementation plan created under todo; no code implementation yet | [T302k plan](todo/T302k-low-small-loss-redesign-plan.md) | [HTML design](../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html) |
| 2026-05-26 21:33 | T302k body-relative foot anchor fix | pass for major accumulated foot drift; residual yaw body-x drift remains background | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-2133-t302k-body-relative-foot-anchor-fix.md](log/2026-05-26-2133-t302k-body-relative-foot-anchor-fix.md) |
| 2026-05-26 20:21 | T302k support-plane root roll/pitch | pass locally and in `env_isaacsim`; root roll/pitch follows support plane after frame0 | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-2021-t302k-support-plane-root-roll-pitch.md](log/2026-05-26-2021-t302k-support-plane-root-roll-pitch.md) |
| 2026-05-26 17:57 | T302k dense path retirement | pass locally; old dense residual modules/config switch removed | [T302k](todo/T302k-parametric-mpc-trajectory-contract.md) | [2026-05-26-1757-t302k-dense-path-retirement.md](log/2026-05-26-1757-t302k-dense-path-retirement.md) |

## Maintenance

- Keep this page as a dashboard, not a changelog.
- Put detailed background in branch pages and evidence in logs.
- Old unfinished T302h/T302i/T302j leaves are closed as routes and preserved as context, not deleted.
