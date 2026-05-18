# T302g MPC Semantic RL Training Config

## Current State

- T302g is a child of [T302](T302-mpc-body-leg-height-field-collision-safety.md).
- Goal: promote the T302 MPC collision-safe backend into an independent semantic RL train/play task without changing the existing `teacher_elevation_trajectory` / together defaults.
- Written design: [../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md](../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md)
- Historical initial plan: [../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md](../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md)
- Subagent coverage review: [../log/2026-05-18-1053-t302g-subagent-requirement-coverage-review.md](../log/2026-05-18-1053-t302g-subagent-requirement-coverage-review.md)
- Implementation slice log: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md)
- Review result: partial pass; requirements are mostly captured, and the spec is being hardened here. Future execution uses this branch page's child todo tree, not a new plan file.
- Active design pivot: replace the old per-env dirty-subset/backlog mainline with fixed-interval `global_sync` MPC refresh. At each global replan tick, randomly sample `mpc_parallel_plan_batch_size` envs for MPC planning; only sampled-and-successfully-planned envs receive `reference_foot_pos` imitation reward. Reset or command-change envs immediately set `reference_reward_mask[env] = false` until a later global replan samples and replans them again.
- Implementation state: T302g.1-T302g.3 are implemented and backend-tested; T302g.4 dirty-subset behavior is historical and no longer the target; T302g.5a-T302g.5c are implemented and backend-tested for global-sync sampled MPC behavior; T302g.5 still needs a real 4096 `collect_time_s`; T302g.6 still needs the full strict JSONL rerun.
- Latest timing diagnosis: [../log/2026-05-18-1258-t302g-4096-timing-diagnostics.md](../log/2026-05-18-1258-t302g-4096-timing-diagnostics.md) shows the command runs and saves a checkpoint, but collect time is still `75.88-86.56s`. The bottleneck is `reference_foot_pos` triggering MPC planning, not `swing_leg_collision` or semantic observation.
- MPC-internal profile instrumentation is now added and covered by a focused test. Standalone 64-env CUDA profiling shows stable `plan.total_ms ~= 2103ms`, with `optimizer.loss ~= 945ms` and `optimizer.backward ~= 1039ms`; largest forward terms are `touchdown_surface`, `ik_fk_residual`, `semantic_obstacle`, and `leg_kinematics`. Evidence: [../log/2026-05-18-1338-t302g-mpc-internal-profile.md](../log/2026-05-18-1338-t302g-mpc-internal-profile.md).
- Static semantic course generation now uses a terrain-importer hook so semantic geometry exists before scanner initialization while `replicate_physics=True` remains enabled. 64-env smoke confirms scanner `unique_semantic_ids=[0,1,2]` and observation semantic map includes `[0,1,2]`; the run then hits a separate MPC OOM inside `high_obstacle_avoidance_loss`, so 4096 timing remains blocked by MPC memory/throughput. Evidence: [../log/2026-05-18-1438-t302g-replicate-physics-static-semantic-course.md](../log/2026-05-18-1438-t302g-replicate-physics-static-semantic-course.md).
- Safe throughput slice: heavy-loss scheduling and `optimize_steps` remain unchanged; zero/near-zero command rows are skipped before optimizer execution, terrain query cache is fixed-decoded equivalent, and IK/FK merge was not enabled because it changed optimized trajectories in CUDA probes. Evidence: [../log/2026-05-18-1419-t302g-mpc-safe-throughput-optimizations.md](../log/2026-05-18-1419-t302g-mpc-safe-throughput-optimizations.md).
- Resource-limited 10-env T302 optimize-step sweep completed low-small + high-small subset for `5/15/20/25`: `20` was the only value with zero failures in the completed subset; `5` and `15` degraded, and `25` produced one stance-semantic anomaly. Evidence: [../log/2026-05-18-1521-t302-optimize-steps-small-sweep.md](../log/2026-05-18-1521-t302-optimize-steps-small-sweep.md).
- Global-sync sampled MPC implementation now passes backend coverage: fixed global replan tick, random `mpc_parallel_plan_batch_size` planning subset, sampled-only imitation reward, continued tracking between global ticks, and reset/command reward-mask invalidation. The active runtime-counter vocabulary is now also cleaned to global-sync terms: `global_due`, `global_due_count`, and `sampled_plan_count`. Evidence: [../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md](../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md), [../log/2026-05-18-2200-t302g-global-sync-counter-cleanup.md](../log/2026-05-18-2200-t302g-global-sync-counter-cleanup.md).
- P0 acceptance:
  - independent new config file, train/play experiment, and Gym ids
  - high-resolution `semantic_height_scanner` for MPC and collision reward
  - CNN input `2 x 16 x 16` height + priority semantic map
  - MPC foot-only imitation reward
  - swing/leg collision reward reads current IsaacLab body buffers, not planner FK
  - global-sync MPC replan reads current IsaacLab state for the sampled envs, not prior MPC cache
  - `mpc_parallel_plan_batch_size` controls how many envs are randomly sampled for MPC planning and imitation reward each global replan tick
  - reset/command-changed envs immediately disable `reference_reward_mask`, even if they previously received a valid MPC plan
  - 4096 real IsaacLab collect-data timing under `10s`
  - T302 strict metrics do not regress

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302g.1 | done | P0 | Create/register independent MPC semantic train/play config with explicit MPC reference-manager flags and no default/together regression | `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`, `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`, `Go2Pvcnn/scripts/train.py`, `Go2Pvcnn/scripts/play.py` |
| T302g.2 | done | P0 | Add high-resolution semantic scanner observation helpers for `2 x 16 x 16` elevation + priority semantic CNN maps | `Go2Pvcnn/extension/mdp/observations.py`, new config file, `Go2Pvcnn/tests/test_batch_mpc_backend.py` |
| T302g.3 | done | P0 | Add `swing_leg_collision_reward` using IsaacLab body/contact buffers, scanner pose/yaw map queries, and swing>stance weighting | `Go2Pvcnn/extension/mdp/rewards_reference.py`, `Go2Pvcnn/extension/mdp/__init__.py`, `Go2Pvcnn/tests/test_batch_mpc_backend.py` |
| T302g.4 | drop | P0 | Remove dirty-subset/backlog scheduling from the T302g active path; it conflicts with the new global-sync sampled-planning design | `Go2Pvcnn/extension/batch_mpc_planner/manager.py`, `Go2Pvcnn/tests/test_batch_mpc_backend.py` |
| T302g.5 | partial | P0 | Add real 4096 RSL-RL collect-data timing gate for the new MPC semantic task, under `10s` without lowering MPC quality | `Go2Pvcnn/tests/test_mpc_runtime_headless.py`, optional fixture changes if needed |
| T302g.5a | done | P0 | Implement global-sync sampled MPC planning: fixed replan tick, random `mpc_parallel_plan_batch_size` envs, sampled-only imitation reward | `Go2Pvcnn/extension/batch_mpc_planner/manager.py`, `Go2Pvcnn/extension/batch_mpc_planner/config.py`, `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py` |
| T302g.5b | done | P0 | Add hard reward-mask invalidation for sampled envs that later reset or receive command changes | `Go2Pvcnn/extension/batch_mpc_planner/manager.py`, `Go2Pvcnn/extension/mdp/rewards_reference.py`, `Go2Pvcnn/tests/test_batch_mpc_backend.py` |
| T302g.5c | done | P0 | Remove or rewrite tests/assertions that require dirty-subset counters/backlog behavior in the new T302g global-sync mode | `Go2Pvcnn/tests/test_batch_mpc_backend.py`, `Go2Pvcnn/tests/test_mpc_runtime_headless.py` |
| T302g.6 | partial | P0 | Run T302 strict non-regression and record implementation/timing evidence | `Go2Pvcnn/tests/`, `notes/log/`, T302 strict JSONL probe path |
| T302g.7 | done | P1 | Align notes/logs after each verified implementation slice | `notes/todo.md`, this branch page, `notes/log/index.md`, per-verification logs |

## Closed Children Archive

- T302g.1: implemented new config/register/train/play/factory integration. Evidence: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md).
- T302g.2: implemented `downsampled_elevation_semantic_scan` and priority semantic pooling. Evidence: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md).
- T302g.3: implemented and wired `swing_leg_collision_reward`. Evidence: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md).
- T302g.4: dirty-subset/current-state manager behavior was verified in the earlier implementation pass, but is now dropped from the T302g active direction because it conflicts with the global-sync sampled-planning design. Evidence remains historical only: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md).
- T302g.5a: implemented `mpc_replan_mode="global_sync"` and `mpc_parallel_plan_batch_size` sampled planning. Evidence: [../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md](../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md).
- T302g.5b: implemented reset/command reward-mask invalidation for already-planned envs. Evidence: [../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md](../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md).
- T302g.5c: backend/runtime tests now assert T302g global-sync sampled counters instead of dirty-subset budget behavior. Evidence: [../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md](../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md).

## Related Logs

- [../log/2026-05-18-1036-t302g-mpc-semantic-rl-training-design-and-plan.md](../log/2026-05-18-1036-t302g-mpc-semantic-rl-training-design-and-plan.md)
- [../log/2026-05-18-1053-t302g-subagent-requirement-coverage-review.md](../log/2026-05-18-1053-t302g-subagent-requirement-coverage-review.md)
- [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md)
- [../log/2026-05-18-1258-t302g-4096-timing-diagnostics.md](../log/2026-05-18-1258-t302g-4096-timing-diagnostics.md)
- [../log/2026-05-18-1338-t302g-mpc-internal-profile.md](../log/2026-05-18-1338-t302g-mpc-internal-profile.md)
- [../log/2026-05-18-1438-t302g-replicate-physics-static-semantic-course.md](../log/2026-05-18-1438-t302g-replicate-physics-static-semantic-course.md)
- [../log/2026-05-18-1419-t302g-mpc-safe-throughput-optimizations.md](../log/2026-05-18-1419-t302g-mpc-safe-throughput-optimizations.md)
- [../log/2026-05-18-1521-t302-optimize-steps-small-sweep.md](../log/2026-05-18-1521-t302-optimize-steps-small-sweep.md)
- [../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md](../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md)
- T302 strict baseline: [../log/2026-05-17-0804-t302-strict-collision-metric-tuning.md](../log/2026-05-17-0804-t302-strict-collision-metric-tuning.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `working tree, 2026-05-18 17:29 CST backend verification`
- Current Work Ref: `working tree on top of 446a875 (2026-05-18 17:29 CST)`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md](../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md)
  - [../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md](../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md)
  - [../../docs/specs/2026-04-12-batched-gpu-kinematic-planner-design.md](../../docs/specs/2026-04-12-batched-gpu-kinematic-planner-design.md)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/extension/mdp/observations.py](../../Go2Pvcnn/extension/mdp/observations.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)

## Next Step

- Run real IsaacLab timing for the global-sync sampled T302g path when GPU resources allow, then run full T302 strict JSONL non-regression before closing the branch.

## Node Details

### T302g.1 Independent MPC semantic train/play config and registry

- why-created: the new RL task must be isolated from `teacher_elevation_trajectory` and together defaults.
- implementation contract:
  - create `teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - define train and play config classes
  - set `planner_owned_reference_cache = True`, `use_batched_reference_trajectory = True`, `planner_backend = "mpc"`, and `reference_height_scanner_name = "semantic_height_scanner"` explicitly in both classes
  - register both Gym ids with `gym.spec(...)` coverage
  - add train/play experiment mapping
  - add regression assertion that existing `teacher_elevation_trajectory` mapping/default planner path is unchanged
- acceptance:
  - new Gym ids resolve
  - new experiment choices resolve
  - old trajectory experiment remains unchanged
- evidence: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md).

### T302g.2 Semantic scanner CNN observation

- why-created: the policy needs the same spatial map size as current trajectory config while preserving semantic priority.
- implementation contract:
  - use one high-resolution `semantic_height_scanner` at `0.01m`
  - expose `(num_envs, 2, 16, 16)` observation
  - channel 0 area-pools elevation
  - channel 1 uses priority pooling: large obstacle > small obstacle > terrain
  - semantic ids are never averaged
- acceptance:
  - shape test passes
  - priority pooling test proves a large cell wins over small/terrain
  - missing scanner/map errors include the scanner name
- evidence: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md).

### T302g.3 Swing/leg collision reward from current IsaacLab state

- why-created: the RL task needs a policy-facing reward for current simulated leg collision, separate from MPC planner losses.
- implementation contract:
  - read `robot.data.body_pos_w` for `.*_thigh`, `.*_calf`, and `.*_foot`
  - read current foot contacts from `contact_forces` with `body_names=".*_foot"`
  - classify stance by current contact force above `swing_collision_contact_force_threshold`
  - classify swing as not-stance
  - map each leg body to its foot contact state by body name prefix and apply `swing_collision_swing_weight > swing_collision_stance_weight`
  - sample `semantic_height_scanner.data.elevation_map` and `semantic_height_scanner.data.semantic_map`
  - transform world body points into scanner-local grid coordinates using scanner pose/yaw and pattern size/resolution
  - do not call planner FK/IK helpers and do not read MPC reference cache as current state
- acceptance:
  - swing penalty is greater than stance for the same collision sample
  - large semantic penalty is greater than small
  - yawed/translated scanner query test catches fixed-world-range sampling
  - source guard proves no FK/IK helper/cache dependency in the reward
- evidence: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md). Remaining gap: add a dedicated yawed/translated scanner reward test if this becomes a review blocker; implementation uses shared yaw-aware `height_at`/`semantic_at`.

### T302g.4 Dropped dirty-subset MPC replanning path

- status: drop from active T302g mainline.
- why-dropped: dirty-subset/backlog scheduling replans reset/command/stale envs asynchronously and keeps a backlog, which conflicts with the new design where MPC refresh happens only at fixed global replan ticks.
- historical contract:
  - selected dirty rows were capped by `mpc_max_dirty_envs_per_step`
  - dirty-subset counters/backlog were used to diagnose partial replans
  - reset/command changes marked envs dirty instead of only disabling imitation reward
- replacement:
  - T302g.5a global-sync sampled planning
  - T302g.5b hard reward-mask invalidation on reset/command change
  - T302g.5c test cleanup for old dirty-subset assumptions
- evidence: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md) remains historical only.

### T302g.5 4096 RSL-RL collect-data timing gate

- why-created: user requires real IsaacLab 4096 training collection time under `10s`.
- implementation contract:
  - instantiate `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0` or direct `TeacherElevationTrajectoryMpcSemanticEnvCfg`
  - assert the new cfg/task is active before timing
  - prefer the RSL-RL rollout/collect-data path; raw `env.step(...)` alone is not final acceptance
  - if fixture reuse is needed, extend it with explicit `task_id`/`env_cfg_cls` and new-task assertions
  - optimize batching/tensor/scanner/reward paths if too slow
  - do not reduce MPC optimizer quality, T302 collision losses, or policy-facing foot trajectory horizon to pass timing
- acceptance:
  - measured collect-data pass under `10s`
  - global-sync counters show `mpc_parallel_plan_batch_size` sampled envs at each fixed replan tick
  - reset/command-changed envs have `reference_reward_mask=False` until a later global replan samples and replans them
  - timing log records device, env count, steps/rollout length, and counters
- evidence: partial in [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md). The new test gate exists and exits `0` under `MPC_TEST_DEVICE=cuda:1`, but the current AppLauncher/pytest path loses the numeric `collect_time_s` metric before shutdown.

### T302g.5a Global-sync sampled MPC planning

- status: done for backend behavior; real IsaacLab timing remains under T302g.5.
- why-created: the new active design requires fixed-interval global MPC refresh with random planning subsets instead of dirty-subset/backlog scheduling.
- evidence: [../log/2026-05-18-1258-t302g-4096-timing-diagnostics.md](../log/2026-05-18-1258-t302g-4096-timing-diagnostics.md).
- current measurements:
  - `reference_foot_pos` reward term costs about `2.65-2.90s` per steady step.
  - MPC counters show `sampled_plan_count=64`, `planner_ms` about `2.65-2.86s`, `cache_ms < 1ms`.
  - `swing_leg_collision` costs about `3.5-5.7ms`.
  - observation costs about `9-10ms` after warmup.
  - standalone 64-env CUDA `plan_segment` costs about `2.10s` after warmup with profiling enabled; optimizer forward loss and backward dominate roughly equally.
  - largest measured forward loss terms are `touchdown_surface`, `ik_fk_residual`, `semantic_obstacle`, `leg_kinematics`, then `swing_center_urgency`, `obstacle_risk`, `swing_direction`, and `semantic_contact_avoid`.
  - zero-command row splitting reduces optimizer batch size for mixed zero/nonzero commands; a standalone half-zero 64-env probe ran in about `1.70s`.
  - fixed decoded objective with terrain query cache enabled vs disabled has max per-env diff `0.0`.
  - optimize-step small sweep suggests `20` steps is the best candidate among `5/15/20/25` on the completed low-small + high-small subset.
- implementation contract:
  - add `mpc_parallel_plan_batch_size` to task/runtime config, default `4096`
  - at each fixed global replan tick, sample `min(num_envs, mpc_parallel_plan_batch_size)` env ids randomly
  - read current IsaacLab state/terrain/command only for sampled envs
  - call `plan_segment(...)` on the sampled env batch
  - scatter successful results into the full reference cache
  - set `reference_reward_mask=True` only for sampled envs with valid MPC results
  - set unsampled envs to `reference_reward_mask=False`
  - do not run per-env dirty/stale/backlog replans in this mode
  - keep old non-T302g behavior isolated if other configs still need it
- constraints:
  - do not reduce T302 strict MPC quality parameters just to pass timing.
  - keep replans reading current IsaacLab state for sampled envs.
  - `mpc_parallel_plan_batch_size` controls both planning participation and imitation-reward participation.
  - do not change heavy-loss scheduling for this user-requested optimization slice.
- acceptance:
  - default `mpc_parallel_plan_batch_size=4096` samples all 4096 envs.
  - setting it smaller samples exactly that many envs when `num_envs` is large enough.
  - unsampled envs do not call MPC and receive zero `reference_foot_pos` imitation reward.
  - sampled envs keep following their existing MPC plan between global replan ticks unless reset/command invalidates the mask.
  - full reference cache shape remains `num_envs x horizon`.
  - runtime counters/logs expose sampled count and planner time.
- evidence: [../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md](../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md).

### T302g.5b Reset/command invalidates MPC imitation reward

- status: done for backend behavior.
- why-created: an env can receive a valid MPC plan and then become invalid for that plan through reset or velocity-command change before the next global replan tick.
- implementation contract:
  - reset event for env `i` immediately sets `reference_reward_mask[i] = false`
  - command/velocity change for env `i` immediately sets `reference_reward_mask[i] = false`
  - this rule applies even if env `i` was sampled and planned successfully earlier in the same generation
  - reset/command change must not trigger immediate per-env replan in global-sync mode
  - next fixed global replan tick may sample env `i`; only then can valid planning re-enable the reward
- acceptance:
  - test proves a sampled/planned env loses reward mask after reset
  - test proves a sampled/planned env loses reward mask after command change
  - test proves unsampled envs keep reward mask false
  - `swing_leg_collision_reward` is not gated by this MPC imitation mask
- evidence: [../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md](../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md).

### T302g.5c Dirty-subset test/code cleanup

- status: done for active T302g tests/config; legacy dirty-subset code remains available for non-T302g mode.
- why-created: tests or implementation branches that require dirty-subset counters/backlog behavior will fight the new T302g global-sync sampled-planning design.
- implementation contract:
  - remove T302g active-path assertions that require dirty-subset counters/backlog semantics, stale priority, or `mpc_max_dirty_envs_per_step`
  - rewrite timing-gate expectations around sampled-count/global-generation counters
  - keep historical dirty-subset tests only if they are explicitly scoped to a legacy/non-T302g mode
  - remove config references from the T302g semantic task when they only serve dirty-subset scheduling
- acceptance:
  - no T302g test expects reset/command change to enqueue an immediate MPC replan
  - no T302g task config depends on dirty-subset budget as the training throughput control
  - tests document `mpc_parallel_plan_batch_size` as the single planning/reward participation knob for global-sync mode
- evidence: [../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md](../log/2026-05-18-1729-t302g-global-sync-sampled-mpc.md).

### T302g.6 T302 strict non-regression

- why-created: RL integration must not weaken T302's MPC collision-safety metrics.
- implementation contract:
  - run backend tests
  - run the strict single-process JSONL probe path from [../log/2026-05-17-0804-t302-strict-collision-metric-tuning.md](../log/2026-05-17-0804-t302-strict-collision-metric-tuning.md)
  - keep root-bottom/swing-foot/knee/shank collision ratios at `0.0`
  - keep `17/17` strict rows passing
  - keep stance semantic count `0`
- acceptance:
  - no T302 metric regression
  - no MPC quality reduction made for speed
- evidence: partial backend regression only: `Go2Pvcnn/tests/test_batch_mpc_backend.py -q` returned `79 passed`. Full strict JSONL rerun remains open.

### T302g.7 Notes/log alignment

- why-created: this branch is now the execution todo source of truth.
- implementation contract:
  - update this branch page after each child node changes state
  - add one log per distinct verification pass
  - update `notes/log/index.md` and `notes/todo.md` when focus/result changes
  - keep links repository-relative
- acceptance:
  - code behavior, verification evidence, and notes agree
- evidence: [../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md](../log/2026-05-18-1142-t302g-mpc-semantic-rl-implementation.md).

### T302g.0 Original independent MPC semantic RL rollout config

- status: superseded by finer child nodes `T302g.1` through `T302g.7`.
- why-created: the user wants MPC integrated into RL training as a new semantic task without disturbing together or T302.
- hypothesis: a high-resolution semantic scanner can serve MPC and collision reward, while a downsampled `2 x 16 x 16` map feeds the CNN; global-sync sampled MPC planning can keep 4096 collect-data under `10s` by using `mpc_parallel_plan_batch_size` as the single planning/reward participation knob.
- evidence: design and plan written; subagent review found five plan-hardening items; implementation and timing evidence pending.
