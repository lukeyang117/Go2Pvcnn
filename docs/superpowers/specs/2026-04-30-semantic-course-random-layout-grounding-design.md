# Semantic Course Random Layout And Grounding Design

## Metadata

- **Date**: 2026-04-30
- **Topic**: distribute semantic-course objects across each sub-terrain with deterministic randomness and robust terrain grounding
- **Status**: Draft for review
- **Primary environment**: `/home/lhy/anaconda3/envs/env_isaaclab`

## 1. Problem Statement

The semantic static-course viewer path works, but object placement is still too centralized. Current anchors in `extension/semantic_course.py` are fixed near the tile origin, with local coordinates such as `0.20..0.70m` and `y` mirrored around `0.0`. Since the active terrain tiles are `8m x 8m`, the semantic objects visually cluster near the central spawn/scanner area instead of coexisting with the broader sub-terrain.

The desired behavior is:

- semantic objects should spread across most of each sub-terrain
- each sub-terrain should have a different randomized layout
- layout randomness must be reproducible
- objects should stay upright
- object bottoms should align to irregular terrain height well enough that objects do not float and do not become fully buried
- slight embedding is acceptable and preferred over obvious floating on uneven terrain

The user-selected direction is:

- full sub-terrain spread over scanner-window visibility
- deterministic pseudo-random layout
- robust upright grounding, not terrain-normal tilt

## 2. Scope

### In Scope

- Update `Go2Pvcnn/extension/semantic_course.py` layout generation from fixed center anchors to deterministic per-tile sampling.
- Keep the existing semantic stages and object counts:
  - `S1`: no semantic objects
  - `S2`: `4` small
  - `S3`: `4` small, `1` large
  - `S4`: `6` small, `1` large
- Add a stable semantic-course seed and stable integer mixing so layouts are reproducible.
- Infer or resolve terrain tile size so local sampling covers the full sub-terrain rather than the central `1.5m` scan window.
- Add margin, center-safety, and minimum-distance constraints.
- Upgrade grounding from one point per object to footprint-based multi-point terrain sampling.
- Bias grounding toward slight embedding rather than visible floating.
- Update tests and runtime diagnostics expectations that currently assume immediate central visibility.

### Out Of Scope

- Changing the training environment config in this phase.
- Adding non-native custom object meshes.
- Changing semantic ids or marker colors.
- Making objects dynamic or reset-randomized.
- Tilting objects to match estimated terrain normals.
- Guaranteeing that the default viewer spawn scanner sees all semantic classes immediately.

## 3. Architecture Boundary

The primary implementation target is `Go2Pvcnn/extension/semantic_course.py`.

Responsibilities remain split as follows:

- `extension/semantic_course.py`
  - semantic stage definitions
  - per-tile deterministic layout generation
  - shape selection
  - footprint sample generation
  - terrain grounding
  - prestartup spawning
- `go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py`
  - keeps the viewer-only semantic config and prestartup event wiring
  - may pass a layout seed or grounding parameters through event params if useful
- `extension/viz/go2_foostep_planner.py`
  - should not own layout logic
  - continues to consume `semantic_height_scanner` and visualize class-partitioned hit points
- `go2_pvcnn/sensor/semantic_raycaster`
  - semantic contract should remain unchanged
  - only needs changes if the new distribution exposes a scanner/mesh traversal bug

This keeps the viewer path close to the existing semantic-course architecture and avoids routing placement policy into the visualization loop.

## 4. Public API And Defaults

The design must keep current no-argument callers deterministic while adding explicit configuration points for layout and grounding.

### 4.1 Constants

Add explicit defaults in `extension/semantic_course.py`:

- `DEFAULT_SEMANTIC_COURSE_SEED = 20260430`
- `DEFAULT_SEMANTIC_COURSE_TILE_SIZE = (8.0, 8.0)`
- `DEFAULT_TILE_MARGIN_M = 0.50`
- `DEFAULT_CENTER_SAFETY_HALF_EXTENT_M = 0.85`
- `DEFAULT_MIN_SPACING_CLEARANCE_M = 0.15`
- `DEFAULT_MAX_LAYOUT_ATTEMPTS = 64`
- `DEFAULT_GROUNDING_HEIGHT_QUANTILE = 1.0`
- `DEFAULT_GROUNDING_EMBED_DEPTH_M = 0.015`

`DEFAULT_GROUNDING_HEIGHT_QUANTILE = 1.0` means the default robust terrain height is the maximum finite footprint sample height. Tunability may exist, but the default must be exact and testable.

### 4.2 Layout And Grounding Config Objects

Add lightweight config structures or equivalent keyword groups:

- `SemanticCourseLayoutCfg`
  - `tile_margin_m`
  - `center_safety_half_extent_m`
  - `min_spacing_clearance_m`
  - `max_layout_attempts`
- `SemanticCourseGroundingCfg`
  - `height_quantile`
  - `embed_depth_m`

Both should have defaults matching the constants above.

### 4.3 Function Signatures

The implementation should preserve existing calls like `build_course_anchors(terrain_origins)`.

Recommended public/helper signatures:

```python
def resolve_tile_size(
    terrain_origins,
    *,
    terrain_generator=None,
    fallback_tile_size: tuple[float, float] = DEFAULT_SEMANTIC_COURSE_TILE_SIZE,
) -> tuple[float, float]:
    ...

def build_course_anchors(
    terrain_origins,
    *,
    terrain_generator=None,
    tile_size: tuple[float, float] | None = None,
    semantic_course_seed: int = DEFAULT_SEMANTIC_COURSE_SEED,
    layout_cfg: SemanticCourseLayoutCfg = DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG,
) -> list[CourseAnchor]:
    ...

def spawn_semantic_course_prestartup(
    env,
    _env_ids,
    *,
    default_stage: str = DEFAULT_VIEWER_REPRESENTATIVE_STAGE.value,
    semantic_course_seed: int = DEFAULT_SEMANTIC_COURSE_SEED,
    tile_size: tuple[float, float] | None = None,
    layout_cfg: SemanticCourseLayoutCfg = DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG,
    grounding_cfg: SemanticCourseGroundingCfg = DEFAULT_SEMANTIC_COURSE_GROUNDING_CFG,
) -> None:
    ...
```

Exact type names can vary, but these contracts must remain true:

- no-seed callers stay deterministic through `DEFAULT_SEMANTIC_COURSE_SEED`
- runtime prestartup can pass explicit seed/layout/grounding parameters through event params later
- implementation tests can call the pure helpers without Isaac runtime

### 4.4 Tile-Size Resolution Contract

`resolve_tile_size(...)` should use this exact priority:

1. if `tile_size` is explicitly supplied to `build_course_anchors(...)`, validate and use it
2. if `terrain_generator.size` exists and contains two positive values, use it
3. infer missing axes from `terrain_origins`
4. use `DEFAULT_SEMANTIC_COURSE_TILE_SIZE` for any axis that cannot be inferred

Axis inference rules:

- `tile_x`: median absolute difference between adjacent row origins' `x` values in the same column
- `tile_y`: median absolute difference between adjacent column origins' `y` values in the same row
- ignore zero or non-finite differences
- if only one axis can be inferred, combine it with the fallback for the missing axis

At runtime, `spawn_semantic_course_prestartup(...)` should pass the terrain generator when available, for example from `scene.terrain.cfg.terrain_generator` or the equivalent terrain importer config location. If the runtime terrain object does not expose that config, origin inference and fallback remain the compatibility path.

## 5. Layout Design

### 5.1 Replace Fixed Coordinates With Stage Specs

The current `_STAGE_LAYOUTS` should stop storing exact `(x, y)` anchors. It should become a stage-count specification, or equivalent structure, that describes how many objects each semantic class owns per stage.

The counts stay unchanged:

| Stage | Small Count | Large Count |
| --- | ---: | ---: |
| `S1` | 0 | 0 |
| `S2` | 4 | 0 |
| `S3` | 4 | 1 |
| `S4` | 6 | 1 |

`course_anchor_counts(stage)` remains the public count query.

### 5.2 Tile-Local Coordinate System

Terrain origins are treated as sub-terrain centers, consistent with existing Isaac Lab terrain origin comments and current placement behavior. For a tile of size `(tile_x, tile_y)`, valid random local coordinates are sampled from:

- `x in [-tile_x / 2 + margin, tile_x / 2 - margin]`
- `y in [-tile_y / 2 + margin, tile_y / 2 - margin]`

For the current semantic terrain, `tile_x = tile_y = 8.0`, so objects can occupy most of the `8m x 8m` sub-terrain instead of only the old center band.

Tile size resolution must follow the public helper contract in section 4.4.

### 5.3 Deterministic Pseudo-Randomness

Every generated coordinate must derive from a stable seed and stable key fields:

- `semantic_course_seed`
- `stage`
- `row`
- `col`
- `slot_index`
- `semantic_class`
- candidate attempt index

Do not use Python's built-in `hash()` because it is process-randomized. Use a stable integer mixer or digest-based helper.

The expected behavior:

- same seed + same tile + same slot produces the same coordinate across runs
- different row/col values produce different layouts
- different slots/classes inside the same tile do not collapse onto the same coordinate

### 5.4 Sampling Constraints

The sampler should use deterministic rejection sampling with bounded attempts.

Required constraints:

- **tile margin**: keep object footprint inside the tile and away from sub-terrain borders
- **center safety box**: avoid the robot spawn/scanner center region around local `(0, 0)`
- **minimum object distance**: reduce overlap and visual clustering across small and large objects
- **large-first placement**: when a stage contains large and small objects, place large objects first so small objects distribute around them

Default constants are fixed in section 4.1. The class-aware minimum spacing should be:

`radius_a + radius_b + layout_cfg.min_spacing_clearance_m`

If rejection sampling cannot satisfy every constraint, use a deterministic fallback based on a coarse grid or radial sweep. Startup should not fail merely because a sparse randomized layout exhausted attempts.

Fallback behavior must be testable:

- `CourseAnchor` or companion debug metadata must expose `layout_fallback_used`
- fallback must still satisfy tile margin and center-safety constraints
- fallback may relax only the minimum object-distance constraint
- the canonical current terrain case, `8m x 8m` with default S4 counts and default layout config, must not use fallback

### 5.5 Viewer Visibility Consequence

Because the selected direction prioritizes full sub-terrain spread, the default viewer spawn scanner is no longer guaranteed to hit semantic objects immediately.

This is intentional. The automated runtime tests that require semantic hits should become targeted tests:

- select a known generated obstacle in a representative stage
- move or configure env `0` near that obstacle for the scan
- verify semantic hit counts from that targeted pose

The production viewer remains truthful to the full-tile layout instead of forcing every object into the initial `1.5m` scan window.

## 6. Grounding Design

### 6.1 Current Limitation

Current grounding samples terrain height at only the object center:

`center_z = terrain_z_at_anchor + bottom_to_center_offset(shape)`

This works on flat terrain but is fragile on rough terrain, stairs, boxes, and slopes. An object can float at one side or bury too deeply at another side because the terrain under the footprint is not represented.

### 6.2 Footprint Samples

Grounding should generate multiple terrain sample points per obstacle.

Recommended footprint samples:

- center
- four cardinal points
- four diagonal/corner points

For cuboids, use half extents from the cuboid size. For sphere, cylinder, capsule, and cone with vertical `Z` axis, use the horizontal radius as a circular footprint approximation.

This keeps the implementation simple while covering the meaningful support area for every current native shape kind.

### 6.3 Robust Bottom Height

The bottom height should be computed from all footprint terrain heights rather than a single center height.

The default policy should prefer slight embedding over visible floating:

1. sample terrain heights under the footprint
2. compute a robust high value, such as a high quantile or max-like value
3. subtract a small `embed_depth`
4. add the existing shape-aware `bottom_to_center_offset(shape)`

Formula:

`world_center_z = robust_ground_z - embed_depth + bottom_to_center_offset(shape)`

Default values are exact:

- `grounding_height_quantile = 1.0`
- `embed_depth_m = 0.015`

With the default config, `robust_ground_z` is the maximum finite footprint sample height. Other values may be exposed later as explicit parameters, but tests should assert the default policy exactly.

The design intent is:

- avoid objects floating above local protrusions
- allow small penetration into high spots
- avoid placing the center so low that the object is mostly buried

### 6.4 Upright Objects Only

Objects remain upright. This design does not estimate local normals or rotate objects with slopes.

Reason:

- upright objects are more stable on stairs and boxy terrain
- normal-following can look strange at terrain discontinuities
- the current semantic scanner and planner care primarily about geometry surfaces, not physically realistic prop resting orientation

## 7. Targeted Runtime Scan Contract

Full-tile distribution intentionally breaks the old assumption that env `0` at tile center sees semantic objects. Runtime tests that require semantic hits need an explicit targeting helper.

Add pure/runtime helper contracts equivalent to:

```python
def select_course_anchor(
    anchors: list[CourseAnchor],
    *,
    stage: SemanticCourseStage | str,
    semantic_class: str,
    slot_index: int | None = None,
) -> CourseAnchor:
    ...

def set_scene_env_to_course_anchor(scene, *, env_id: int, anchor: CourseAnchor) -> None:
    ...
```

The runtime helper should:

- set the env's terrain row/col to the anchor's tile when terrain level/type buffers are available
- set `terrain.env_origins[env_id]` to a scan pose whose `x/y` overlaps the selected anchor, not merely the tile center
- keep enough terrain/base height clearance for the robot reset or scanner update to be valid

The exact implementation may live in test fixtures if it is not needed by production viewer code.

Acceptance criteria:

- targeted S4 small scan reports semantic id `1` within a bounded number of zero-action steps, no more than one scanner update period plus four env steps
- targeted S4 large scan reports semantic id `2` under the same bound
- the two targeted scans together prove both small and large semantic geometry remain visible to `semantic_height_scanner`

The existing `set_scene_env_to_representative_stage(...)` remains useful for stage selection, but it is not sufficient as the only semantic-hit runtime mechanism after center-safety exclusion.

## 8. Data Model Changes

`CourseAnchor` should keep enough information to describe a resolved randomized slot:

- `row`
- `col`
- `stage`
- `semantic_class`
- `slot_index`
- `shape_kind`
- `shape_params`
- `target_diameter`
- `target_height`
- `ground_offset`
- `local_xy`
- `world_xy`
- `prim_path`
- `layout_fallback_used`

The existing fields mostly remain valid. The difference is that `local_xy` becomes generated, not looked up from a fixed anchor table.

`GroundedCourseObstacle` should continue to represent the final spawn-ready object, but its `world_center.z` should come from footprint grounding.

The pure helper `ground_course_anchors(...)` should be generalized so unit tests can verify multi-point grounding without requiring Isaac runtime. The runtime helper `_ground_with_runtime_terrain_sampler(...)` should batch all footprint sample points through the existing `/World/ground` raycast path.

## 9. Testing Strategy

### 9.1 Layout Unit Tests

Add or update tests in `Go2Pvcnn/tests/test_semantic_course.py`:

- stage counts remain exact for `S1..S4`
- same seed/terrain origins generate identical anchors
- different rows/cols produce different local layouts
- local points stay inside tile margins
- local points avoid the center safety box
- object spacing constraints hold for non-fallback cases
- canonical `8m x 8m` S4 default layout does not use fallback
- a deliberately tight synthetic tile can use fallback, and fallback still satisfies margin and center safety
- shape selection remains deterministic and still covers the native pool

### 9.2 Grounding Unit Tests

Add tests with fake terrain height functions:

- footprint samples are generated for each shape class
- grounding uses multiple sample points, not just center height
- `center_z` follows `robust_ground_z - embed_depth + shape_offset`
- uneven terrain produces slight embedding where expected and no full burial in the tested cases

### 9.3 Runtime And Viewer Tests

Keep existing semantic viewer/runtime coverage, with one important expectation change:

- default spawn scanner may not see small/large objects after full-tile distribution
- tests that require semantic hits should target a known obstacle location

Runtime tests should still verify:

- `semantic_height_scanner` exists
- `semantic_map` remains valid
- planner terrain reconstruction still works from semantic scanner hits
- targeted S4 small and large scans report semantic ids `1` and `2`
- compact shape-pool acceptance still includes `capsule` and `cone`

### 9.4 Manual Viewer Acceptance

Manual viewer acceptance should confirm:

- objects are distributed across each sub-terrain, not clustered around tile center
- repeated viewer launches with the same seed show the same layout
- different sub-terrains visibly differ
- objects do not visibly float
- objects may embed slightly on rough terrain, but are not swallowed by the terrain
- scanner colors still partition terrain/small/large hits when the robot is near semantic objects

## 10. Risks And Mitigations

### Risk: Default Viewer Looks Empty Near Spawn

Mitigation:

- document this as an intentional trade-off from the user-selected full-tile distribution
- use targeted runtime tests for semantic-hit assertions
- optionally add a later viewer helper command/flag for jumping near a generated obstacle, outside this design

### Risk: Rejection Sampling Fails In Tight Tiles

Mitigation:

- keep bounded attempts
- place large objects first
- use deterministic fallback placement
- expose `layout_fallback_used` for tests and diagnostics
- make spacing constraints conservative and tile-size aware

### Risk: Objects Still Float On Sharp Steps

Mitigation:

- use footprint samples
- prefer high quantile or max-like ground height
- subtract a small embedding depth
- keep objects upright to avoid unstable rotations

### Risk: Tests Become Overfit To Exact Random Coordinates

Mitigation:

- test reproducibility, bounds, safety, and spacing rather than every exact coordinate
- use exact-coordinate assertions only for a tiny stable smoke case if needed

## 11. Implementation Readiness

This design is ready for implementation planning. The selected behavior and constraints are fixed:

- full sub-terrain spread
- deterministic per-tile randomness
- upright objects
- footprint-based terrain grounding
- slight embedding allowed
- no guarantee of default central scanner visibility

The next step after review is a detailed implementation plan focused on `extension/semantic_course.py`, followed by test updates and targeted runtime verification.
