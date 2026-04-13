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

### 2. Planner boundary adapter

A small formal boundary layer should exist near the planner, not inside the
viewer.

Responsibilities:

- convert Isaac Lab state into planner state
- convert `height_scanner.data.ray_hits_w` into the planner terrain-query
  object expected by the core planner
- guarantee shape, device, and dtype consistency

This can live either in:

- `extension/batched_planner/terrain.py` and related planner helpers, or
- a small planner-owned adapter module beside the planner package

The important design rule is ownership: training and viewer both call the same
boundary implementation.

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

Design decision:

- `--use-raw-reference-trajectory` should be removed or rejected for the
  `teacher_elevation_trajectory` runtime path
- if the batched planner cannot be built, training should fail clearly rather
  than silently substituting placeholder drift

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
2. Shared planner boundary converts these into planner inputs.
3. `batched_planner` generates the trajectory batch.
4. The result is converted into the reference cache format.
5. Reference-tracking rewards consume that cache.

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

## Error Handling

### Hard failures

Training should fail loudly when:

- the planner runtime cannot be constructed
- terrain query data is invalid
- reference cache generation fails
- viewer/training inputs violate planner contract

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

Compare at minimum:

- `root_pos_w`
- `root_quat_w`
- `foot_pos_w`
- `contact_state`
- `planned_touchdown_w`

### Contract tests

Add planner boundary tests for:

- dtype stability
- device stability
- single-env and multi-env parity
- single-terrain multi-query semantics
- body-clearance terrain sampling behavior

### Runtime smoke tests

After implementation:

- run minimal `train.py --headless --num_envs 1 --max_iterations 1 --experiment teacher_elevation_trajectory`
- run viewer with livestream-compatible startup
- confirm viewer motion branch no longer crashes under teleop commands

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
2. Align planner motion behavior to raw where tests show differences.
3. Wire training reference-cache generation to batched planner only.
4. Simplify viewer so it uses the same runtime path with no private planner
   semantics.
5. Add or update raw-alignment and contract tests.
6. Write notes for `train`, `viewer`, and `play` commands and parameter meaning.

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
6. Minimal headless train smoke test reaches the trajectory path without falling
   back to placeholder logic.
7. Notes document launch commands and parameter meanings for `train`, `viewer`,
   and `play`.

## Open Decisions Settled in This Spec

- Runtime truth source: `batched_planner`
- Raw usage: comparison baseline only
- Placeholder usage: not a normal trajectory-training path
- Play code changes this iteration: no
- Heightmap footprint parity with raw: not required
- Trajectory behavior parity with raw: required
