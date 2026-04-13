# Batched Planner / Train / Viewer Alignment Design

Date: 2026-04-13

## Goal

Unify the `teacher_elevation_trajectory` training path, the Isaac Lab viewer at
`Go2Pvcnn/extension/viz/go2_foostep_planner.py`, and the runtime behavior of
`Go2Pvcnn/extension/batched_planner` so they all use the same batched planner as
the single runtime source of truth.

The batched planner should be aligned to `raw/kinematic_footsteps` in behavior,
while the raw code remains a comparison baseline rather than an active runtime
path.

## Requirements

1. `teacher_elevation_trajectory` must use `extension/batched_planner` only.
2. `raw` must not remain a training runtime path.
3. Placeholder drift must not remain a normal trajectory-training path.
4. The Isaac Lab viewer must call the same planner path used by training.
5. `batched_planner` behavior must be aligned to `raw` for trajectory semantics.
6. Heightmap window size and scanner footprint do not need exact raw parity.
7. Notes must document launch commands for `train`, `viewer`, and `play`, plus
   parameter explanations.

## Non-Goals

- Do not refactor `play.py` behavior in this iteration.
- Do not require the Isaac Lab height scanner footprint to match the raw local
  elevation window exactly.
- Do not keep viewer-only planner adapters as the long-term solution.
- Do not use `raw` as the default runtime implementation after alignment.

## Current Problems

### Runtime split-brain

`teacher_elevation_trajectory` is configured as a trajectory-guided experiment,
but the current reward-side reference cache path still has a placeholder
generator fallback. This means the experiment name and the actual runtime source
of the trajectory can diverge.

### Viewer-specific patches

The current viewer was forced to work around planner input issues in its own
code path:

- dtype mismatches between Isaac Lab tensors and planner tensors
- marker clearing behavior that Isaac Lab does not accept
- terrain query shape mismatches for single-map multi-query usage
- device mismatches once the planner reaches deeper motion branches

These are signs that planner runtime contracts are not yet stable enough to be
used as a shared source of truth.

### Planner contract ambiguity

`batched_planner` currently mixes several implicit assumptions:

- CPU/GPU tensor placement
- float32 vs float64 intermediate values
- single-env vs multi-env shapes
- single-terrain vs batched-terrain query semantics

Those assumptions are survivable in isolated unit tests but break once the
planner is driven by Isaac Lab state and scanner outputs.

## Design Summary

### Source of truth

`Go2Pvcnn/extension/batched_planner` becomes the only runtime trajectory
generator for `teacher_elevation_trajectory`.

`raw/kinematic_footsteps` remains:

- a comparison oracle
- a regression-test oracle
- a behavior-alignment baseline

but not a normal runtime source for training or viewer execution.

### Shared runtime chain

Both training and viewer will use the same logical pipeline:

1. Read Isaac Lab robot state.
2. Read Isaac Lab `height_scanner` ray hits.
3. Convert those into the formal planner input boundary.
4. Call `batched_planner`.
5. Consume the result:
   - training writes reference cache for rewards
   - viewer visualizes the exact same planner output

No viewer-only planner semantics are allowed after alignment.

### Raw alignment target

The implementation goal is not line-by-line code similarity. The target is
behavioral alignment for:

- standstill behavior
- stop-speed fallback
- gait/contact schedule
- root trajectory
- foot swing targets
- planned touchdown positions
- replan candidate enumeration and selection
- single-env outputs under representative commands and terrains

## Architecture

### 1. Planner core

Files:

- `Go2Pvcnn/extension/batched_planner/trajectory.py`
- `Go2Pvcnn/extension/batched_planner/foothold.py`
- `Go2Pvcnn/extension/batched_planner/base_solver.py`
- `Go2Pvcnn/extension/batched_planner/terrain.py`
- `Go2Pvcnn/extension/batched_planner/terrain_estimator.py`
- supporting config/types files

Responsibilities:

- define stable planner input/output contracts
- accept Isaac-derived terrain query objects without viewer-specific hacks
- keep device/dtype behavior deterministic
- preserve parity with `raw` trajectory semantics

Planned changes:

- formalize dtype/device rules
- formalize accepted terrain query shapes
- remove assumptions that only work for tests or standstill branches
- align motion-branch logic to raw behavior where tests show drift
- remove or replace the current `batch_size > 1` recursive fallback so the
  training path remains viable at large env counts

Performance constraint:

- the shared train-time planner path must be designed for `num_envs=4096`
  teacher runs and must not rely on a Python-level per-env recursion loop as
  the normal execution path

### 2. Planner boundary adapter

A small formal boundary layer should exist near the planner, not inside the
viewer.

Responsibilities:

- convert Isaac Lab state into planner state
- convert `height_scanner.data.ray_hits_w` into the planner terrain-query
  object expected by the core planner
- guarantee shape, device, and dtype consistency

Canonical boundary contract:

- input robot state arrives batched as `(N, ...)`
- input scanner ray hits arrive batched per env from Isaac Lab as `(N, R, 3)` or
  `(N, H, W, 3)` depending on scanner layout
- invalid hits (`nan`/`inf`) must be filtered in a deterministic way
- terrain query world ranges must be derived by one canonical rule shared by
  train and viewer
- single-terrain and multi-terrain query semantics must be explicit rather than
  inferred by caller-specific adapters
- the planner boundary must define exactly what shapes are accepted by:
  - `height_at`
  - `roughness_at`
  - `max_height_along_segment`

Formal input object decision:

- the only external terrain input accepted by planner entry points must be a
  planner-owned `PlannerTerrain` object
- internal tensor storage may remain implementation-defined, but train and
  viewer must not pass raw heightmap tensors directly into planner entry points
- `PlannerTerrain` is the formal ABI for terrain input to
  `batched_generate_trajectory(...)`

Ownership decision:

- this adapter must live in planner-owned code and be imported by both train and
  viewer
- viewer-local terrain adapters are not allowed in the final design
- training-local terrain adapters are not allowed in the final design

This can live either in:

- `extension/batched_planner/terrain.py` and related planner helpers, or
- a small planner-owned adapter module beside the planner package

The important design rule is ownership: training and viewer both call the same
boundary implementation, and that implementation defines the only accepted
terrain-query contract.

### 3. Training integration

Files likely affected:

- `Go2Pvcnn/scripts/train.py`
- `Go2Pvcnn/extension/mdp/rewards_reference.py`
- possibly a small new runtime helper under `extension/reference/` or
  `extension/batched_planner/`

Responsibilities:

- remove raw runtime selection as a normal trajectory-training path
- remove placeholder reference generation as a normal trajectory-training path
- ensure `teacher_elevation_trajectory` always populates reference cache from
  batched planner output

Training-side ownership:

- a planner-owned runtime manager must own trajectory-cache build/update for
  training
- reward code must consume an already-managed cache rather than lazily inventing
  a placeholder when the cache is missing
- the design must define where this manager is called in the env lifecycle
  relative to reset, step, and replanning cadence

Design decision:

- `scripts/train.py` must delete the `--use-raw-reference-trajectory` CLI
  argument entirely
- raw reference generation must not remain as a train-time CLI path for
  `teacher_elevation_trajectory`
- if the batched planner cannot be built, training should fail clearly rather
  than silently substituting placeholder drift

Reference-cache lifecycle requirements:

- cache build must happen on first valid trajectory step of each env
- cache rebuild must happen on env reset
- cache rebuild must happen when the planner replan interval is reached
- cache invalidation rules must explicitly handle terrain changes and command
  changes
- the spec implementation must state whether command changes trigger immediate
  replanning or only interval-based replanning, and that same rule must be used
  by training and viewer

Resolved replan policy:

- command changes trigger immediate replanning
- env reset triggers immediate replanning
- interval-based replanning remains in place as a fallback cadence even without
  command changes
- training and viewer must share this exact policy

### 4. Viewer integration

File:

- `Go2Pvcnn/extension/viz/go2_foostep_planner.py`

Responsibilities:

- launch Isaac Lab
- collect keyboard input
- call the same planner runtime path used by training
- visualize the exact resulting planner output

Viewer must not own:

- planner-specific shape repair
- planner-specific terrain semantics
- planner-specific fallback logic

Those belong to the shared planner runtime boundary.

### 5. Documentation

Notes deliverable will include:

- `train` launch command examples
- `viewer` launch command examples
- `play` launch command examples
- key parameters and what they change
- trajectory-runtime constraints
- common failure cases and quick diagnosis steps

`play.py` code will not be changed in this iteration, but its command-line usage
will still be documented.

## Data Flow

### Training

1. Isaac Lab env produces robot state and `height_scanner` data.
2. A planner-owned runtime manager receives state, scanner data, and current
   commands.
3. Shared planner boundary converts these into planner inputs.
4. `batched_planner` generates the trajectory batch on the configured replanning
   cadence.
5. The result is converted into the reference cache format and stored on the env
   runtime state.
6. Reference-tracking rewards consume that managed cache.

Training cadence requirement:

- the design must explicitly preserve or replace
  `reference_replan_interval_steps`
- the same cadence must govern cache refresh in train and expected phase advance
  semantics in viewer comparisons
- the chosen policy is: immediate replan on reset, immediate replan on command
  change, otherwise interval-based replan on the configured cadence

### Viewer

1. Isaac Lab env produces the same robot state and `height_scanner` data.
2. Keyboard teleop produces the current command.
3. Shared planner boundary converts state/scanner data into planner inputs.
4. `batched_planner` generates the trajectory batch.
5. Viewer renders root trajectory, foot trajectories, touchdowns, command arrow,
   and sampled heightmap points.

### Comparison baseline

For aligned test cases, the same logical state, command, and terrain queries are
fed to `raw` and `batched_planner`, then compared numerically.

Parity rule:

- candidate enumeration order must be deterministic
- candidate tie-break behavior must be specified explicitly
- stop-speed boundary behavior must be checked explicitly
- parity is not considered sufficient unless near-tie candidate cases are
  included in fixtures

## Error Handling

### Hard failures

Training should fail loudly when:

- the planner runtime cannot be constructed
- terrain query data is invalid
- reference cache generation fails
- viewer/training inputs violate planner contract
- planner output/device normalization is violated

This is preferable to falling back to placeholder drift in a trajectory
experiment.

### Soft handling

Viewer-only presentation concerns may remain local to the viewer, for example:

- marker visibility toggles
- camera placement
- teleop timeout behavior

Those should not affect planner semantics.

## Testing Strategy

### Raw alignment tests

Extend or add tests that compare `batched_planner` against `raw` for:

- zero command
- below-stop-speed command
- forward motion
- turning motion
- lateral motion
- representative stair-like local terrain queries
- representative near-tie replanning cases
- reset-and-replan lifecycle cases

Compare at minimum:

- `root_pos_w`
- `root_quat_w`
- `foot_pos_w`
- `contact_state`
- `planned_touchdown_w`
- effective replanning command choice when multiple candidates are close

Raw reproduction checks:

- keep a dedicated check for “does batched reproduce raw behavior” separate from
  generic planner contract tests
- include deterministic fixture inputs for raw-comparison runs
- include terrain-boundary outputs in the oracle for at least selected fixtures
  so parity is not judged only by final trajectory tensors

### Contract tests

Add planner boundary tests for:

- dtype stability
- device stability
- single-env and multi-env parity
- single-terrain multi-query semantics
- body-clearance terrain sampling behavior
- output cache dtype/device normalization
- invalid-ray handling
- canonical world-range derivation from scanner hits

Formal contract rules to validate:

- planner internal math may use a chosen canonical dtype, but output dtype/device
  normalization must be explicit
- reference cache dtype/device must be explicitly defined
- viewer visualization casts must happen only after planner outputs leave the
  shared runtime path
- the shared runtime path must avoid accidental `.item()`, CPU materialization,
  or Python loops that introduce hidden GPU/CPU synchronization

ABI decisions to validate:

- planner entry points accept `PlannerTerrain`, planner state tensors, and
  command tensors as the only formal runtime inputs
- planner outputs must preserve a single explicit runtime device rule and a
  single explicit normalized dtype rule before cache conversion
- reference cache layout and placement must be explicitly defined rather than
  inferred from caller behavior

### Performance checks

Add explicit training-efficiency checks for:

- planner invocation cost at realistic env counts
- whether `batch_size > 1` still falls back to per-env recursion
- whether terrain-query conversion introduces Python loops on the hot path
- whether cache rebuild cadence is amortized rather than recomputed every step

Minimum efficiency acceptance:

- no normal `teacher_elevation_trajectory` training run should rely on a
  Python-level per-env planner recursion path at `num_envs=4096`
- the implementation must include at least one measured or asserted check that
  the shared runtime path scales beyond `num_envs=1`

### Runtime smoke tests

After implementation:

- run minimal `train.py --headless --num_envs 1 --max_iterations 1 --experiment teacher_elevation_trajectory`
- run a larger train-side smoke/perf check at a higher env count chosen to
  expose recursion or synchronization problems
- run viewer with livestream-compatible startup
- confirm viewer motion branch no longer crashes under teleop commands
- confirm viewer and train use the same planner-owned boundary module rather
  than separate adapters

## Implementation Approach Options

### Option A: Planner-first alignment

Fix planner contracts and raw parity first, then reconnect training and viewer to
that shared path.

Pros:

- cleanest long-term architecture
- consistent runtime behavior
- lowest chance of train/viewer drift

Cons:

- requires touching several planner files before user-visible payoff

Recommendation: yes

### Option B: Boundary-adapter-first

Keep planner internals mostly unchanged, but formalize a single adapter used by
train and viewer.

Pros:

- narrower code churn

Cons:

- may hide real planner inconsistencies
- can still diverge from raw semantics internally

Recommendation: only if planner-first proves too risky mid-implementation

### Option C: Continue patching viewer and train separately

Pros:

- fastest local unblocking

Cons:

- directly violates the shared-source-of-truth goal
- likely to regress again

Recommendation: no

## Recommended Plan

1. Normalize planner input contracts in `extension/batched_planner`.
2. Define a planner-owned terrain-query boundary with explicit shape, dtype,
   device, and invalid-hit rules.
3. Replace lazy reward-side placeholder ownership with a planner-owned
   training-time cache manager tied to reset and replanning cadence.
4. Align planner motion behavior to raw where tests show differences,
   including near-tie replanning cases.
5. Remove per-env recursive normal-path behavior that would block large-env
   training.
6. Wire training reference-cache generation to batched planner only.
7. Simplify viewer so it uses the same runtime path with no private planner
   semantics.
8. Add or update raw-alignment, contract, lifecycle, and performance tests.
9. Write notes for `train`, `viewer`, and `play` commands and parameter meaning.

## Acceptance Criteria

This work is complete when all of the following are true:

1. `teacher_elevation_trajectory` no longer relies on raw runtime or placeholder
   runtime for normal training.
2. Viewer and training call the same batched planner path.
3. Motion commands in viewer produce planned motion and nontrivial touchdown
   updates instead of only static overlaps.
4. The current viewer crash on planner motion branches is eliminated.
5. Batched planner outputs are numerically aligned with raw for representative
   test cases.
6. Raw-reproduction checks cover near-tie replanning and lifecycle-sensitive
   cases, not only generic motion cases.
7. Training cache ownership, invalidation, and replanning cadence are explicit
   and no longer depend on reward-side placeholder generation.
8. Minimal headless train smoke test reaches the trajectory path without falling
   back to placeholder logic.
9. The shared train-time path is not relying on Python-level per-env recursion
   as the normal execution model for large env counts.
10. Notes document launch commands and parameter meanings for `train`, `viewer`,
    and `play`.

## Open Decisions Settled in This Spec

- Runtime truth source: `batched_planner`
- Raw usage: comparison baseline only
- Placeholder usage: not a normal trajectory-training path
- Play code changes this iteration: no
- Heightmap footprint parity with raw: not required
- Trajectory behavior parity with raw: required
- Replan policy: immediate on reset, immediate on command change, interval-based
  otherwise
- Terrain input ABI: planner-owned `PlannerTerrain` object only
