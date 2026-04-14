# Batched Planner Single-Shot / Decoupled Runtime Design

Date: 2026-04-14

## Goal

Refactor `Go2Pvcnn/extension/batched_planner` so it matches the updated
`raw/kinematic_footsteps` runtime semantics and removes the current
whole-batch lifecycle coupling that makes training throughput collapse under
asynchronous resets.

The new design must:

1. align trajectory generation behavior to the updated raw planner
2. decouple replanning at the per-env level instead of per-batch
3. keep `teacher_elevation_trajectory` on the planner-only runtime path
4. add performance observability for planner internals and train-time rollout
5. allow the viewer to play planner trajectories directly without physical
   constraint enforcement

## Requirements

1. `batched_planner` must follow raw single-shot command semantics.
2. Multi-candidate replanning must be removed rather than hidden behind a flag.
3. Any single env reset must not force unrelated envs to replan.
4. Any single env command change must not force unrelated envs to replan.
5. An env whose plan falls back to standstill must keep consuming a standstill
   cache until its own next replan event.
6. Planner performance instrumentation must cover both internal planner stages
   and train-time rollout outcomes.
7. Benchmark coverage must include env counts up to `2048`.
8. A train-side verbose flag may print planner diagnostics when explicitly
   requested for debugging.
9. The viewer must support direct planner playback mode without physical
   constraint enforcement.

## Non-Goals

- Do not preserve the old whole-batch replan lifecycle as a compatibility mode.
- Do not preserve multi-candidate command recovery as a compatibility mode.
- Do not change the viewer into a different runtime path from training.
- Do not require raw and Isaac Lab heightmap generation to be re-designed in
  this iteration.
- Do not require the viewer playback path to obey robot physical constraints in
  this iteration.

## Confirmed Assumptions

### Raw local heightmap size

For this design, raw and batched planner are treated as aligned on local
heightmap size.

The execution path in `raw/kinematic_footsteps` passes:

- `local_extent_xy=(1.5, 1.5)`

and the terrain window constructor uses:

- `start_x = center_x - extent_x * 0.5`
- `nx = round(extent_x / local_resolution)`

which means the raw runtime currently uses an approximately
`1.5 m x 1.5 m` local window rather than a `3.0 m x 3.0 m` window.

### Standstill cache semantics

If a specific env replans into standstill, that env keeps consuming a cached
standstill trajectory until one of its own future replan triggers fires:

- env reset
- env command change
- env interval-based replan

The env must not continue using a stale motion trajectory after a failed
single-shot plan.

## Current Problems

### 1. Whole-batch replan coupling

`BatchedTrajectoryManager.refresh_from_env()` currently treats replanning as a
batch-global event:

- if any env resets, the whole batch replans
- if any env command changes, the whole batch replans
- if any env hits interval cadence, the whole batch replans

This is incorrect for asynchronous RL rollouts and is a likely root cause of
poor collection throughput at moderate env counts such as `100`.

### 2. Planner semantics still differ from raw

The current batched planner still enumerates multiple recovery commands through:

- velocity scales
- yaw biases
- lateral biases

This no longer matches the updated raw planner, which now behaves as:

- try the given command once
- if infeasible, return standstill

### 3. Missing planner-level observability

Current train metrics only show outer rollout results such as:

- `Steps per second`
- `Collection time`

They do not reveal whether the time is dominated by:

- terrain construction
- gait / stance schedule
- foothold search
- swing generation
- terrain estimator
- base solver
- IK
- manager lifecycle overhead

## Design Summary

### Source of truth

`Go2Pvcnn/extension/batched_planner` remains the only runtime planner for:

- training
- viewer
- planner-owned reference cache generation

`raw/kinematic_footsteps` remains a behavioral baseline and parity oracle.

### Core semantic change

The batched planner changes from:

- multi-candidate recovery planner

to:

- single-shot planner with explicit standstill fallback

### Core lifecycle change

The manager changes from:

- whole-batch lifecycle coupling

to:

- per-env lifecycle tracking with masked cache updates

### Core observability change

The system gains:

- planner micro-benchmarks
- train macro-benchmarks
- optional verbose planner diagnostics during train execution

### Core viewer playback change

The viewer gains an explicit planner playback mode:

- the robot pose shown in Isaac Lab may be driven directly from planner output
- this mode is intended for trajectory inspection rather than physics-faithful
  execution
- physical constraints are intentionally not enforced in this playback mode

## Architecture

### 1. Single-shot planner core

Primary file:

- `Go2Pvcnn/extension/batched_planner/trajectory.py`

Responsibilities after refactor:

- accept the commanded velocity once
- run gait schedule, foothold planning, touchdown evaluation, swing generation,
  terrain estimation, base solving, and IK once for that command
- return standstill immediately when:
  - the command is effectively zero / below stop threshold
  - single-shot touchdown planning is infeasible
  - no valid motion result exists for that command

Planned removals:

- `_iter_replan_commands(...)`
- multi-candidate velocity scaling loop
- yaw-bias candidate loop
- vy-bias candidate loop
- candidate-scoring-based command replacement

Resulting runtime policy:

1. evaluate the current command once
2. if feasible, return motion trajectory
3. otherwise, return standstill trajectory

This is the required raw parity target.

Planner output semantics are intentionally binary:

- `motion trajectory`
- `standstill trajectory`

No intermediate degraded-motion recovery mode remains in the planner contract.

### 2. Per-env manager lifecycle

Primary file:

- `Go2Pvcnn/extension/batched_planner/manager.py`

Responsibilities after refactor:

- track replanning need independently for each env
- build planner inputs only for envs that need replanning
- update cached reference tensors only for those envs
- advance phase counters independently for envs that do not need replanning

Required state becomes explicitly per-env:

- `phase_index`
- `last_commands`
- `last_replan_episode_length`
- `pending_reset_mask`
- any derived replan reason mask

Old behavior to delete:

- `torch.any(...)` style whole-batch trigger logic that forces unrelated envs to
  rebuild trajectories

New lifecycle policy:

- cache missing: replan only affected envs
- reset: replan only reset envs
- command change: replan only changed envs
- interval expiry: replan only expired envs
- otherwise: advance phase only

Cache update policy:

- planner results may still be generated in batches, but only over the selected
  env subset
- the global cache object is updated by masked writeback into the selected env
  rows
- the canonical cache exposed to rewards and viewer remains full-shaped for all
  envs; partial replan complexity stays inside the manager rather than changing
  downstream cache-consumer contracts

This intentionally requires a large rewrite of manager logic and does not keep
the old lifecycle shape.

### 3. Standstill cache handling

Files affected:

- `Go2Pvcnn/extension/batched_planner/manager.py`
- `Go2Pvcnn/extension/batched_planner/trajectory.py`

Policy:

- when a replanning env falls back to standstill, its cache row becomes a
  standstill trajectory cache
- that row stays valid until that same env hits its next replan event
- unrelated envs continue using their own motion or standstill caches

This avoids both:

- repeated per-step standstill recomputation
- accidental reuse of an older motion plan after failure

### 4. Planner instrumentation

Planner-stage timing must be fine-grained enough to separate these stages:

- terrain input materialization
- gait / stance schedule
- foothold planning
- touchdown feasibility evaluation
- swing target generation
- terrain estimator
- base solver
- IK
- total trajectory generation
- manager total refresh cost

Instrumentation should live close to planner-owned code so it can be used by:

- standalone planner benchmarks
- optional train-time verbose output

Recommended implementation shape:

- a lightweight timing helper or profiler accumulator in planner-owned code
- no verbose printing by default
- explicit opt-in output only

### 5. Train-side debug output

Primary file:

- `Go2Pvcnn/scripts/train.py`

Add an explicit CLI flag, e.g.:

- `--verbose-planner`

When enabled, train may print compact per-step or periodic planner diagnostics
such as:

- how many envs replanned this step
- replan reasons by category
- how many envs are currently on standstill cache
- manager refresh time
- planner total time
- optional stage breakdown if available

Default behavior must remain quiet.

Integration guidance:

- planner-owned timing data should feed the existing training logging surface
  through a compact manager-facing summary rather than ad hoc prints scattered
  across planner stages
- `--verbose-planner` may enable periodic human-readable diagnostics, while the
  benchmark path remains responsible for fuller timing dumps

## Data Flow

### Training path

1. Isaac Lab env produces robot state, scanner hits, command manager output, and
   episode lengths.
2. Manager computes per-env replan masks.
3. For envs in the replan mask:
   - build `PlannerTerrain`
   - build batched robot state subset
   - call single-shot `batched_generate_trajectory()`
4. Manager writes back only affected env rows into the canonical reference
   cache.
5. Reward code gathers from the shared cache as before.

### Viewer path

1. Viewer teleop writes commands through the shared command manager proxy.
2. Viewer env step triggers reward-side cache ensure, which calls the same
   manager.
3. Viewer visualizes the resulting shared cache.
4. In planner playback mode, viewer may apply planner output directly to the
   displayed robot state without requiring physical constraint consistency.

No viewer-only planner semantics are introduced.

### Viewer playback mode

The viewer should support a direct planner playback path whose purpose is
trajectory inspection rather than controller validation.

Policy:

- planner output may directly drive the displayed robot pose / joints
- playback does not need to satisfy physical feasibility or dynamic
  consistency
- no physical-constraint enforcement is required in this mode
- this mode exists specifically so the user can inspect whether the planner is
  generating the intended motion, touchdown sequence, and body trajectory

This is distinct from:

- zero-action simulation with reference overlays
- policy-tracking playback
- physically constrained controller validation

Those remain separate concerns.

## Testing Strategy

### Behavioral tests

Add or update tests to cover:

1. single-shot success path matches raw for representative cases
2. single-shot failure returns standstill immediately
3. mixed batch behavior:
   - some envs succeed
   - some envs standstill
4. per-env manager lifecycle:
   - one env reset does not replan unrelated envs
   - one env command change does not replan unrelated envs
   - interval expiry replans only expired envs
5. standstill cache persistence:
   - failed env remains on standstill cache until its own next replan trigger
6. viewer playback path:
   - planner output can be visualized directly without requiring physical
     constraint enforcement

### Planner micro-benchmarks

Add benchmark coverage for planner components at:

- `1`
- `16`
- `64`
- `100`
- `256`
- `512`
- `1024`
- `2048`

Measure at least:

- terrain construction
- gait / stance schedule
- foothold
- swing
- terrain estimator
- base solver
- IK
- trajectory total
- manager refresh total

Output should include:

- absolute elapsed time
- per-env normalized time
- batch size
- standstill env count
- replanned env count

### Train macro-benchmarks

Run `teacher_elevation_trajectory` at:

- `num_envs = 1, 16, 64, 100, 256, 512, 1024, 2048`
- `max_iterations = 1`

Track:

- `Steps per second`
- `Collection time`
- optional verbose planner counters when enabled

## Acceptance Criteria

The design is considered successfully implemented when all of the following are
true.

### Behavior

1. `batched_planner` no longer enumerates multiple recovery commands.
2. A failed single-shot motion plan returns standstill immediately.
3. Manager replanning is per-env rather than whole-batch.
4. A standstill env keeps using standstill cache until its own next replan.
5. Raw parity tests for the updated single-shot semantics pass.
6. Viewer playback mode can display planner-generated motion even when the
   simulated robot is not physically tracking that motion.

### Performance visibility

7. Planner micro-benchmarks exist and report per-stage timings up to `2048`.
8. Train macro-benchmarks exist and report `Steps per second` and
   `Collection time` across env-count sweeps.
9. Train can optionally emit planner debug output through a verbose flag.

### Performance direction

10. The known `100 env` asynchronous reset case no longer triggers obvious
   whole-batch replanning behavior.
11. Train-side collection metrics improve or, at minimum, become explainable
    through the added timing breakdowns.

## Risks

### Cache writeback complexity

Per-env masked cache updates are more complex than whole-batch replacement.
Shape consistency and canonical cache layout must be tested carefully.

### Benchmark interference

Verbose planner output and timing instrumentation must not distort the default
training path when disabled.

### Partial-batch planner calls

Once replanning is masked, planner code must correctly handle arbitrary subset
batch sizes without hidden assumptions tied to the full env count.

## Open Implementation Guidance

- Prefer deleting obsolete logic rather than hiding it behind optional flags.
- Keep planner instrumentation lightweight and planner-owned.
- Avoid spreading lifecycle logic into rewards or viewer code.
- Keep the user-facing runtime contract simple:
  - same planner for train and viewer
  - same single-shot semantics as raw
  - explicit verbose mode for debugging only
