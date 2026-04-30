# Semantic Native Shape Pool Design

## Metadata

- **Date**: 2026-04-30
- **Topic**: expand the semantic static course from cuboids-only to the full Isaac Sim native shape pool
- **Status**: Draft for review
- **Primary environment**: `/home/lhy/anaconda3/envs/env_isaaclab`

## 1. Problem Statement

The semantic static-course viewer path now works end to end, but the obstacle geometry is visually monotonous because the course still spawns only cuboids.

The user wants the semantic course to include the native geometric shape kinds already available in Isaac Sim / Isaac Lab, rather than custom mesh assets. The explicit request is to broaden obstacle variety with circular and conic shapes while keeping the semantic setup coherent and easy to test.

The approved scope for this increment is:

- use only Isaac Sim native shapes
- no custom mesh shapes in this phase
- keep semantic classes unchanged:
  - `small`
  - `large`
- add more shape variety inside those two classes
- use a deterministic pseudo-random shape choice so each `(stage, row, col, slot, semantic_class)` always resolves to the same shape
- extend scanner-side support as needed so every approved native shape in the pool is actually visible to `semantic_height_scanner`

## 2. Scope

### In Scope

- Extend `Go2Pvcnn/extension/semantic_course.py` so obstacle slots can spawn multiple native shape kinds
- Extend `Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py` as needed so the approved pool is fully ingestible by the semantic scanner
- Use the shared native shape pool:
  - `sphere`
  - `cuboid`
  - `cylinder`
  - `capsule`
  - `cone`
- Keep `small` and `large` as the only semantic classes
- Keep the current stage system `S1..S4`
- Keep deterministic tile-owned placement and deterministic stage layouts
- Add/extend tests for shape selection, parameter mapping, grounded offsets, and stable reproducibility

### Out of Scope

- Adding custom mesh-based shapes such as triangular prism / triangular pyramid
- Changing semantic ids
- Changing scanner naming or viewer marker semantics
- Changing training config in this phase

## 3. Native Shape Inventory

The currently confirmed native shape configs available through Isaac Lab are:

- `SphereCfg`
- `CuboidCfg`
- `CylinderCfg`
- `CapsuleCfg`
- `ConeCfg`

These are the full approved shape pool for this increment.

This increment also explicitly includes any scanner-side shape ingestion work needed so that all five shapes are ray-castable and semantically labeled at runtime.

## 4. Design Overview

The current semantic course stores obstacle slots as if every obstacle were a cuboid. This increment generalizes obstacle generation by introducing a shape-spec layer while preserving the existing semantic-course and viewer architecture.

The data flow becomes:

1. `stage -> local anchors`
2. `anchor + semantic class + deterministic selector -> shape kind`
3. `shape kind + class scale profile -> shape parameters`
4. `shape parameters -> grounded placement offset`
5. `grounded obstacle -> native Isaac shape spawn`

The semantic ids and viewer diagnostic surface remain unchanged. The public `semantic_height_scanner` contract and name stay the same, while `semantic_raycaster` internals are allowed to expand supported USD/native geometry types as needed for the approved pool.

## 5. Hard Decisions And Constraints

1. Only native shape configs are allowed in this increment.
2. `small` and `large` share the same shape pool.
3. The semantic class remains the semantic meaning source; shape kind is purely geometric variety.
4. Shape selection must be deterministic for the same `(stage, row, col, slot, semantic_class)`.
5. Rebooting the viewer must not change the shape assigned to an existing slot.
6. Shape choice must not introduce a third semantic layer or custom viewer logic.
7. Grounding must stay geometry-correct for every native shape kind.
8. The semantic scanner must be able to ingest every approved shape kind in the pool for this increment to count as complete.

## 6. Data Model Changes

### 6.1 Current Limitation

`semantic_course.py` currently stores obstacle slots with cuboid-only assumptions:

- `size`
- `world_center`
- cuboid-specific spawn helper

This is not expressive enough for the full native shape pool.

### 6.2 New Shape Spec Layer

Add a shape-spec concept inside `semantic_course.py`.

Recommended fields:

- `shape_kind`
  - one of: `sphere`, `cuboid`, `cylinder`, `capsule`, `cone`
- `shape_params`
  - native parameters specific to the chosen shape
- `target_diameter`
- `target_height`
- `ground_offset`

The exact type names can vary, but the abstraction must let the semantic course reason about shape independently from semantic class.

### 6.3 Updated Course Structures

`CourseAnchor` should gain enough information to represent the selected shape kind and the shape scale profile that will later become spawn parameters.

`GroundedCourseObstacle` should represent a fully resolved obstacle:

- semantic class
- shape kind
- resolved shape parameters
- grounded world pose
- prim path

The cuboid-only `size` field should no longer be the sole source of truth.

## 7. Shared Shape Pool

### 7.1 Shape Pool Contents

The approved shared pool is:

```python
("sphere", "cuboid", "cylinder", "capsule", "cone")
```

Both `small` and `large` draw from this same pool.

The exact stable key formula is implementation-defined, as long as it deterministically depends on `(stage, row, col, slot, semantic_class)` and is covered by tests.

### 7.2 Deterministic Shape Selection

Shape selection must be deterministic rather than runtime-random.

Approved rule:

- derive a stable integer key from:
  - `stage`
  - `row`
  - `col`
  - `slot`
  - `semantic_class`
- map that key to the shared shape pool via modulo

This guarantees:

- the same tile/slot always gets the same shape
- shape variety is visually distributed across the course
- there is no dependence on process-local random seeds

## 8. Small / Large Scale Profiles

The semantic class still decides size scale.

Approved target scales:

- `small`
  - `target_diameter = 0.12`
  - `target_height = 0.22`
- `large`
  - `target_diameter = 0.45`
  - `target_height = 0.55`

These values remain the primary scale inputs for all native shapes.

## 9. Shape Parameter Mapping

The first-implementation parameter mapping is:

### 9.1 Cuboid

- `size = (target_diameter, target_diameter, target_height)`

### 9.2 Sphere

- `radius = target_diameter / 2`

### 9.3 Cylinder

- `radius = target_diameter / 2`
- `height = target_height`
- `axis = "Z"`

### 9.4 Cone

- `radius = target_diameter / 2`
- `height = target_height`
- `axis = "Z"`

### 9.5 Capsule

- `radius = target_diameter / 2`
- `height = max(target_height - target_diameter, epsilon)`
- `axis = "Z"`

This keeps total scale visually aligned with the semantic class while staying within native-shape conventions.

## 10. Grounding Rules

Grounding must be geometry-aware.

### 10.1 Approved Bottom-To-Center Offsets

- `cuboid`: `target_height / 2`
- `cylinder`: `target_height / 2`
- `cone`: `target_height / 2`
- `sphere`: `radius`
- `capsule`: `radius + height / 2`

### 10.2 Grounding Formula

For every obstacle:

`world_center_z = terrain_z + bottom_to_center_offset`

The existing grounding helper should be generalized to use shape-kind-specific offsets rather than cuboid half-height only.

## 11. Spawn Layer Changes

The cuboid-only spawn helper should be generalized.

Recommended evolution:

- replace `_spawn_grounded_cuboid()` with `_spawn_grounded_shape()`

Dispatch based on `shape_kind`:

- `sphere` -> `sim_utils.SphereCfg`
- `cuboid` -> `sim_utils.CuboidCfg`
- `cylinder` -> `sim_utils.CylinderCfg`
- `capsule` -> `sim_utils.CapsuleCfg`
- `cone` -> `sim_utils.ConeCfg`

All of them should keep the same semantic-course ownership rules:

- static
- kinematic
- gravity disabled
- collision enabled

## 11.1 Scanner Compatibility Changes

Because the approved pool includes `capsule` and `cone`, this increment explicitly includes scanner-side compatibility work inside `semantic_raycaster` if those geometry types are not already fully tessellated and consumed today.

Required outcome:

- every shape in the approved pool is both spawned and scanned
- no approved native shape is “visual only”

## 12. What Does Not Change

This increment should not change:

- stage definitions `S1..S4`
- anchor counts per stage
- semantic ids:
  - `0 = terrain`
  - `1 = small`
  - `2 = large`
- the public `semantic_height_scanner` name/contract
- viewer diagnostic keys
- compact runtime smoke strategy

It also should not introduce custom mesh assets solely to emulate shape kinds already covered by the approved native pool.

## 13. Test Strategy

### 13.1 Shape Selector Tests

Add tests that verify:

- identical `(stage,row,col,slot,semantic_class)` always produces identical shape kind
- different slots distribute across the shared pool predictably
- `small` and `large` both draw from the same pool

### 13.2 Parameter Mapping Tests

Add tests that verify:

- cuboid mapping
- sphere radius mapping
- cylinder radius/height mapping
- capsule radius/height mapping
- cone radius/height mapping

### 13.3 Grounding Tests

Add tests that verify:

- each shape kind uses the correct bottom-to-center offset
- the resulting `world_center_z` matches `terrain_z + expected_offset`

### 13.4 Spawn Tests

Add focused tests that verify:

- `_spawn_grounded_shape()` dispatches to the correct Isaac shape config
- resulting prim paths remain under the existing semantic-course roots

### 13.5 Compatibility Tests

Keep or extend tests that verify:

- semantic root paths remain stable
- semantic raycaster still consumes the spawned native shapes
- no semantic class or shape kind forces zero-marker viewer crashes
- compact `env_isaaclab` semantic smoke is rerun with a layout guaranteed to include at least one `capsule` and one `cone`

## 14. Risks And Mitigations

### 14.1 Risk: Shape-Specific Grounding Is Wrong

Mitigation:

- explicit bottom-to-center offset tests for every shape kind

### 14.2 Risk: Native Shape Diversity Breaks Scanner Assumptions

Mitigation:

- explicitly include scanner-side support work for any approved native shape not fully handled today
- verify scanner compatibility through unit tests

### 14.3 Risk: Deterministic Randomization Is Not Stable

Mitigation:

- use a pure, testable deterministic selector function
- do not depend on process RNG state

## 15. Implementation Readiness

This increment is ready for implementation because the following are now explicit:

- approved native shape pool
- scanner-side scope for `capsule` / `cone` compatibility
- semantic-class policy
- deterministic selection rule
- parameter mapping per shape kind
- grounding offsets per shape kind
- spawn dispatch strategy
- tests required to keep shape diversity safe

No further design clarification is needed before implementation.
