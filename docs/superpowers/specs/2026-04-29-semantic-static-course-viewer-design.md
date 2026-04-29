# Semantic Static Course Viewer Design

## Metadata

- **Date**: 2026-04-29
- **Topic**: terrain-aligned static semantic obstacle course for the trajectory viewer
- **Status**: Draft for review
- **Primary environment**: `/home/lhy/anaconda3/envs/env_isaaclab`

## 1. Problem Statement

The current trajectory viewer only scans `/World/ground` through `height_scanner`, so it cannot test two behaviors the user now wants to verify before touching training:

1. a height scan that truly sees both terrain and semantic obstacles
2. viewer hit visualization that changes color when the scan lands on semantic objects

The semantic course must follow terrain difficulty rather than viewer-only toggles:

- `S1`: no semantic obstacles
- `S2`: four small obstacles
- `S3`: large obstacle plus small obstacles
- `S4`: large obstacle plus more small obstacles

The user also clarified the architectural direction:

- do **not** modify `teacher_elevation_trajectory_env_cfg.py` yet
- create a viewer-first configuration that can later be migrated into the training config
- keep semantic objects static and terrain-attached
- bind semantic stages to terrain difficulty
- use `semantic_raycaster`, and treat its redesign as part of scope

## 2. Scope

### In Scope

- Add a viewer-only env config derived from `TeacherElevationTrajectoryEnvCfg_PLAY`
- Remove inherited `height_scanner` in that viewer config
- Add `semantic_height_scanner` returning:
  - `elevation_map` at `1.5 x 1.5 m`, `0.01 m` resolution
  - `semantic_map` with matching shape and values `0=terrain`, `1=small`, `2=large`
- Add a tile-based semantic course module under `Go2Pvcnn/extension/`
- Spawn static semantic cuboids before sensor initialization
- Redesign `semantic_raycaster` so it can ingest terrain plus semantic obstacle roots robustly
- Update `go2_foostep_planner.py` to consume `semantic_height_scanner`
- Add focused tests for sensor behavior, semantic course generation, and viewer integration

### Out of Scope

- Changing `teacher_elevation_trajectory_env_cfg.py` in this phase
- Wiring semantic maps into training observations or rewards
- Adding viewer CLI flags for semantic curriculum
- Refreshing obstacle positions at reset, replan, or runtime
- General cleanup of all legacy dynamic semantic LiDAR code unless it directly blocks this feature

## 3. Hard Decisions And Constraints

1. The viewer-first config must stay structurally close to the training trajectory config.
2. The old inherited `height_scanner` must be deleted in the viewer config, not left around unused.
3. The active sensor name must be `semantic_height_scanner`.
4. Semantic stage is tied to terrain difficulty, not a viewer argument.
5. Semantic obstacles are static stage geometry, not dynamic per-env assets.
6. Training and viewer both rely on the same future terrain-course logic: more difficult terrain means richer semantic obstacle layouts.
7. The semantic scan must return geometry that includes obstacle surfaces, not only terrain plus post-hoc labels.

### Source-Order Constraint

Source inspection of Isaac Lab establishes a hard ordering:

1. scene is created
2. `prestartup` events may run
3. `sim.reset()` triggers timeline `PLAY`
4. sensors initialize their warp meshes on `PLAY`
5. `startup` events run only after manager loading

Therefore semantic obstacles **cannot** be spawned in `startup` if `semantic_raycaster` must merge them into its static mesh. They must exist by `prestartup`.

## 4. Design Overview

The design uses a viewer-only trajectory config extension plus a shared semantic course module:

1. `teacher_elevation_trajectory_semantic_viewer_env_cfg.py`
   - inherits `TeacherElevationTrajectoryEnvCfg_PLAY`
   - deletes inherited `height_scanner`
   - adds `semantic_height_scanner`
   - repoints inherited observation terms and planner manager scanner references to `semantic_height_scanner`

2. `extension/semantic_course.py`
   - owns semantic stage definitions, tile-to-stage mapping, obstacle sizes, obstacle anchors, terrain height sampling, and stage spawning

3. `go2_pvcnn/sensor/semantic_raycaster/*`
   - upgraded from a narrow proof-of-concept into the authoritative static semantic grid scanner for this workflow

4. `extension/viz/go2_foostep_planner.py`
   - instantiates the viewer-only config
   - reads `semantic_height_scanner`
   - colors terrain, small-obstacle, and large-obstacle hits differently

This keeps training untouched for now while making the viewer path directly reusable when the semantic scanner is later migrated into the main trajectory task config.

## 5. Viewer Config Design

### 5.1 New Config File

Add:

- `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py`

It inherits from `TeacherElevationTrajectoryEnvCfg_PLAY` and changes only the semantic-scanner-related pieces.

### 5.2 Scene Overrides

The derived scene config must do both:

- `height_scanner = None`
- `semantic_height_scanner = SemanticGridRayCasterCfg(...)`

`InteractiveScene` skips `None` assets/sensors, so this is the clean way to prevent the parent scanner from being built.

### 5.3 Semantic Scanner Contract

`semantic_height_scanner` keeps the planner footprint and resolution:

- `prim_path = "{ENV_REGEX_NS}/Robot/base"`
- top-down offset matching the trajectory scanner pattern
- yaw-only alignment
- `GridPatternCfg(resolution=0.01, size=[1.5, 1.5])`

Returned data contract:

- `ray_hits_w`
- `elevation_map`
- `semantic_map`

Where:

- `elevation_map.shape == semantic_map.shape`
- the grid is the full `1.5 x 1.5 m @ 0.01 m` raster
- semantic ids are exactly:
  - `0`: terrain
  - `1`: small obstacle
  - `2`: large obstacle

### 5.4 Observation And Planner Alignment

The viewer config must not leave inherited references pointing at `height_scanner`.

Required viewer-config overrides:

- inherited observation terms that consume the scanner must use `SceneEntityCfg("semantic_height_scanner")`
- `reference_height_scanner_name = "semantic_height_scanner"` so planner manager paths that already support a configurable scanner name stay aligned
- `semantic_height_scanner.update_period = decimation * sim.dt`

No helper abstraction for scanner-name indirection is added in this phase. The user explicitly wants the viewer config to become the future migration template.

### 5.5 Gym Registration

No new Gym env id is required for this phase. The viewer already passes an explicit config object to `gym.make(...)`, so the semantic viewer config can be instantiated directly and supplied there.

## 6. Semantic Course Design

### 6.1 Module Location

Add:

- `Go2Pvcnn/extension/semantic_course.py`

This module is shared logic, not a `tasks/` local helper.

### 6.2 Ownership

The module owns:

- stage enum or equivalent constants: `S1` to `S4`
- row/difficulty to stage mapping
- obstacle dimensions for `small` and `large`
- fixed obstacle anchors per stage
- stage root path conventions under `/World/semantic_course`
- terrain height sampling helper for obstacle placement
- prestartup stage-generation routine

### 6.3 Stage Mapping

The first implementation maps terrain difficulty by terrain row.

Recommended first rule:

- split `num_rows` into four contiguous bands
- lowest band -> `S1`
- next band -> `S2`
- next band -> `S3`
- highest band -> `S4`

This mirrors current terrain curriculum semantics closely enough for the first semantic-course version and stays deterministic across viewer and later training.

### 6.4 Tile-Based, Not Env-Based

Semantic obstacles are generated per terrain tile, not per environment instance.

For each terrain tile `(row, col)`:

- determine stage from the row/difficulty mapping
- choose the fixed anchor layout for that stage
- place cuboids into world space using that tile's terrain origin

This is required so that later training curriculum can move envs between terrain rows without regenerating semantic objects. The course already exists in the world.

### 6.5 Stage Layouts

The approved course progression is:

- `S1`: no semantic objects
- `S2`: `4` small obstacles
- `S3`: large obstacle plus small obstacles
- `S4`: large obstacle plus more small obstacles

Exact dimensions are intentionally left tunable, but the layout structure must be fixed and deterministic.

### 6.6 Root Paths

Organize stage props under fixed global roots:

- `/World/semantic_course/small`
- `/World/semantic_course/large`

Per-tile descendants should be stable and enumerable, for example:

- `/World/semantic_course/small/row_03/col_07/slot_00`
- `/World/semantic_course/large/row_08/col_12/slot_00`

This gives `semantic_raycaster` stable roots to crawl recursively.

## 7. Placement And Initialization

### 7.1 Initialization Mode

Semantic geometry must be created in `prestartup`, not `startup`.

That event runs after scene creation but before `sim.reset()` and before sensor mesh initialization on timeline `PLAY`.

### 7.2 Geometry Type

Use cuboids for both semantic classes.

- `small`: crossing-oriented obstacle class
- `large`: detour-oriented obstacle class

They are semantic classes distinguished by size and semantic id, not by different asset types.

### 7.3 Terrain Attachment

Obstacle `z` placement must be computed from the terrain surface, not guessed from env origin height.

The semantic course module should include a terrain height sampling helper that performs downward terrain queries at each world-space anchor and then sets:

`cuboid_center_z = terrain_surface_z + 0.5 * cuboid_height`

This avoids floating props and buried props across mixed terrain.

### 7.4 Static Lifetime

Generated semantic props are not refreshed at reset, replan, or runtime. Reset only affects the robot and managers, not the semantic course geometry.

## 8. `semantic_raycaster` Redesign

This feature depends on redesigning `go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py`, not merely using it as-is.

### 8.1 Current Weaknesses To Fix

The current implementation is too narrow for the approved course:

- it assumes a small number of exact mesh paths
- it effectively treats each configured path as a single geometry fetch point
- it is fragile for roots that contain many children
- it does not explicitly model empty semantic roots as a valid case

### 8.2 Required New Behavior

For each configured semantic root path:

1. resolve the root prim
2. recursively collect all supported geometry descendants under that root
3. convert each descendant into world-space trimesh triangles
4. assign the configured semantic id to every triangle from that root
5. concatenate all collected geometry into the final merged static mesh

Supported geometry types remain:

- `Mesh`
- `Plane`
- `Cube`
- `Sphere`
- `Cylinder`

### 8.3 Semantic Roots

The intended root mapping is:

- `/World/ground` -> `0`
- `/World/semantic_course/small` -> `1`
- `/World/semantic_course/large` -> `2`

This is a root-level semantic assignment, not a per-leaf config.

### 8.4 Empty-Root Handling

The scanner should tolerate empty semantic roots gracefully.

Examples:

- all tiles happen to be `S1` in a reduced debug layout
- a future config disables one semantic class temporarily

If a semantic root contains no geometry descendants, the scanner should skip it without failing the whole sensor initialization.

### 8.5 Data Guarantees

After redesign, the sensor must continue to provide:

- `ray_hits_w`
- `elevation_map`
- `semantic_map`

with matching grid shape and deterministic semantic id assignment.

## 9. Viewer Integration

`extension/viz/go2_foostep_planner.py` changes are intentionally narrow:

1. instantiate the semantic viewer config instead of the current trajectory play config
2. fetch `base_env.scene.sensors["semantic_height_scanner"]`
3. build local terrain for planner from that scanner's `ray_hits_w`
4. color viewer hit markers using `semantic_map`

### 9.1 Planner Path

Planner terrain construction continues to use the scanner hit geometry.

This is the key point: the planner sees the same obstacle-modified surface that the viewer sees. There is no separate semantic-only visualization layer.

### 9.2 Marker Coloring

The current heightmap marker path should split into three semantic classes:

- terrain hits: current default terrain color
- small-obstacle hits: distinct color
- large-obstacle hits: distinct color

All three colors must differ.

### 9.3 Viewer Diagnostics

On each replan, add lightweight semantic counts from the visible scan:

- terrain hit count
- small hit count
- large hit count

This provides a numeric confirmation that the viewer is actually scanning the semantic course being shown.

## 10. Test Strategy

### 10.1 `semantic_raycaster` Tests

Add tests that verify:

- recursive geometry collection under semantic roots
- correct semantic ids for terrain/small/large
- identical `elevation_map` and `semantic_map` shape
- stable raster size for the approved `1.5 x 1.5 m @ 0.01 m` grid
- obstacle surfaces change elevation where expected
- empty semantic roots do not crash initialization

### 10.2 Semantic Course Tests

Add tests for `extension/semantic_course.py` that verify:

- row-to-stage mapping is deterministic
- `S1` yields no props
- `S2` yields exactly four small props
- `S3` yields large plus small props
- `S4` yields large plus more small props
- generated prim paths land under the correct global roots
- terrain-attached placement uses surface height plus half obstacle height

### 10.3 Initialization-Order Tests

Add at least one test or smoke harness that proves semantic props exist before sensor initialization.

This guards the most failure-prone design assumption discovered during source inspection.

### 10.4 Viewer Integration Tests

Add headless tests for the viewer path that verify:

- the semantic viewer config builds successfully in `env_isaaclab`
- `semantic_height_scanner` exists and returns valid maps
- planner terrain construction from `ray_hits_w` still succeeds
- semantic hit counts differ across `S1/S2/S3/S4`
- semantic marker coloring receives the correct class partitions

### 10.5 Manual Smoke

Run the interactive viewer in `/home/lhy/anaconda3/envs/env_isaaclab` and confirm:

- semantic obstacles are visibly grounded on the terrain
- scan points rise onto obstacle surfaces
- small and large obstacle hits display different colors
- stage changes correlate with terrain difficulty regions

## 11. Risks And Mitigations

### 11.1 Risk: Wrong Initialization Timing

Mitigation:

- use `prestartup`
- add explicit timing verification

### 11.2 Risk: Scanner Still Misses Child Geometry

Mitigation:

- redesign root traversal recursively
- add root-recursion tests

### 11.3 Risk: Height Attachment Is Wrong On Complex Terrain

Mitigation:

- centralize terrain height sampling in `extension/semantic_course.py`
- test grounded placement numerically

### 11.4 Risk: Viewer Config Breaks Observation Manager

Mitigation:

- replace inherited `height_scanner` references in observation terms
- set `reference_height_scanner_name = "semantic_height_scanner"`

## 12. Implementation Readiness

This spec is ready for implementation planning because it resolves the major ambiguity points:

- initialization order
- scanner naming
- inherited scanner deletion
- tile-vs-env ownership
- semantic course stage mapping
- semantic raycaster redesign scope
- viewer integration surface
- required tests

No remaining design dependency blocks writing the implementation plan.
