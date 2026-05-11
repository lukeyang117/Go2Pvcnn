# Semantic-Aware Together Planner And Viewer Spawn Design

## Metadata

- **Date**: 2026-05-06
- **Topic**: make the `together` planner use semantic small/large obstacles in the viewer first, while preserving the future training interface
- **Status**: Draft for review
- **Primary viewer command**: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:2 --num_envs 1 --terrain task --planner-backend together`

## 1. Problem Statement

The semantic static-course viewer can now spawn and scan semantic obstacles:

- terrain id `0`
- small obstacle id `1`
- large obstacle id `2`

The current `together` planner path still treats the scanner as height-only. In `go2_foostep_planner.py`, the viewer reads `scanner.data.semantic_map` only for marker coloring and diagnostics. `TogetherPlannerTerrain.from_ray_hits(...)` keeps only `hits[..., 2]`, so `plan_segment(...)`, `support_at(...)`, and `compute_costs(...)` cannot distinguish small from large obstacles.

The desired behavior is:

- small obstacles are not footstep targets
- small obstacles may be crossed when they are in the velocity direction and are low enough for safe foot and body clearance
- small obstacles that are too high, placed on high terrain, or unsafe for the body should be avoided
- large obstacles must be avoided by feet and by the Go2 body
- large obstacles should trigger an active lateral route choice in the viewer, not merely produce a high cost on the original straight path
- the first implementation is evaluated in the viewer, then migrated into `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py` for reinforcement learning training
- the training path must remain fixed-shape, GPU-resident, and aligned with existing `together` guardrails

The viewer should also expose `row` and `col` parameters so the user can spawn Go2 on a selected sub-terrain tile while inspecting semantic behavior.

## 2. Scope

### In Scope

- Add viewer CLI parameters for sub-terrain spawn targeting:
  - `--terrain-row`, default `0`
  - `--terrain-col`, default `0`
- Place env `0` on `terrain.terrain_origins[row, col]` after viewer reset and after manual `R` reset.
- Extend `TogetherPlannerTerrain` with optional semantic maps.
- Feed `scanner.data.semantic_map` into the `together` terrain object in the viewer.
- Preserve height-only behavior when no semantic map exists.
- Add semantic-aware costs for:
  - foot touchdown / stance collision
  - swing clearance over crossable small obstacles
  - body collision along the root trajectory
  - large-obstacle avoidance
- Add active lateral route candidates for semantic mode:
  - center
  - left offset
  - right offset
- Keep the future training entry aligned with `TogetherTrajectoryManager._terrain_from_env(...)`.
- Add tests and metrics that prove the behavior and guardrails.

### Out Of Scope

- Changing PPO observation tensors in this phase.
- Training a policy in this phase.
- Adding a global path planner or persistent world-scale costmap.
- Moving semantic objects at runtime.
- Making obstacle geometry dynamic.
- Replacing the existing `semantic_raycaster`.
- Modifying the legacy `extension/batched_planner` backend.

## 3. Current Execution Chain

```text
go2_foostep_planner.py main()
Go2Pvcnn/extension/viz/go2_foostep_planner.py
↓
TeacherElevationTrajectorySemanticViewerEnvCfg_PLAY
Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py
↓
SemanticGridRayCaster produces ray_hits_w and semantic_map
Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py
↓
_compute_together_local_terrain(scanner)
Go2Pvcnn/extension/viz/go2_foostep_planner.py
↓
TogetherPlannerTerrain.from_ray_hits(...)
Go2Pvcnn/extension/batched_together_planner/terrain.py
↓
plan_segment(...)
Go2Pvcnn/extension/batched_together_planner/planner.py
↓
expand_segment(...) + compute_costs(...)
Go2Pvcnn/extension/batched_together_planner/parameterization.py
Go2Pvcnn/extension/batched_together_planner/costs.py
↓
viewer kinematic playback
Go2Pvcnn/extension/viz/go2_foostep_planner.py
```

The semantic contract should enter at the terrain object, not in the viewer loop or reward code. The viewer is the first caller, and training can later use the same terrain API through the manager.

## 4. Design Overview

The design has four layers:

1. **Viewer spawn targeting**
   - `--terrain-row` and `--terrain-col` select the sub-terrain tile for env `0`.
   - This improves manual inspection without changing planner or training semantics.

2. **Semantic terrain API**
   - `TogetherPlannerTerrain` stores optional `semantic_maps`.
   - Existing height queries are unchanged.
   - Semantic queries remain batched tensors on the same device.

3. **Semantic route candidates**
   - In semantic mode, the planner evaluates center, left, and right route candidates in one fixed-shape GPU batch.
   - Center remains preferred unless collision/clearance costs make it worse.
   - This makes low small obstacles in the velocity direction crossable, while large obstacles and unsafe small obstacles produce active lateral avoidance.

4. **Collision and clearance cost**
   - Feet cannot land on small or large obstacle cells.
   - Small obstacle crossing is allowed only if foot and body clearance are safe.
   - Large obstacle contact by feet or body is heavily penalized.

5. **Shared terrain extraction contract**
   - Viewer and training must use the same semantic terrain constructor path.
   - The contract is:
     `ray_hits + semantic_map + world_x_range + world_y_range -> TogetherPlannerTerrain`.
   - No caller may reinterpret scanner axes, crop windows differently, or build a separate semantic terrain representation outside that shared path.

## 5. Viewer Row / Col Spawn Targeting

### 5.1 CLI Contract

Add to `go2_foostep_planner.py`:

```text
--terrain-row int, default 0
--terrain-col int, default 0
```

Meaning:

- `row=0, col=0` is the default selected sub-terrain.
- Row and column refer to `base_env.scene.terrain.terrain_origins[row, col]`.
- The selected tile applies to env `0`; the viewer already defaults to single-env diagnostics.

### 5.2 Reset Behavior

After:

- initial `env.reset()`
- manual `R` reset

the viewer should:

1. validate that `scene.terrain.terrain_origins` exists
2. validate `0 <= row < num_rows` and `0 <= col < num_cols`
3. set terrain curriculum buffers for env `0` when exposed:
   - `terrain.terrain_levels[0] = row`
   - `terrain.terrain_types[0] = col`
   - `terrain.env_origins[0] = terrain.terrain_origins[row, col]`
4. write the robot root pose to the selected origin
5. zero root velocity when the robot API exposes `write_root_velocity_to_sim`
6. call `scene.write_data_to_sim()`
7. render/update enough steps for scanner buffers to reflect the new pose

The reset helper should follow the existing viewer playback contract:

```text
write robot state -> scene.write_data_to_sim() -> sim.render() -> scene.update(physics_dt)
```

### 5.3 Error Handling

Invalid `row/col` should raise a clear `ValueError` or `RuntimeError` before planning starts. The viewer should not silently clamp indices.

If the terrain importer lacks generated origins, the viewer should report that `--terrain-row/--terrain-col` requires generated terrain origins.

### 5.4 Non-Goals

The `row/col` feature is viewer-only:

- it does not enter `TogetherPlannerConfig`
- it does not enter training env cfg
- it does not affect batched manager replacement masks
- it does not change semantic course generation

## 6. Semantic Terrain API

### 6.1 Data Model

Extend `TogetherPlannerTerrain`:

```python
@dataclass(frozen=True)
class TogetherPlannerTerrain:
    heightmaps: Tensor              # [B, 1, H, W]
    world_x_range: tuple[float, float]
    world_y_range: tuple[float, float]
    semantic_maps: Tensor | None = None  # [B, 1, H, W], integer ids stored in tensor form
```

`semantic_maps` must:

- share batch, height, width, and device with `heightmaps`
- share the exact same grid indexing, resolution, and world range contract as `heightmaps`
- use `torch.long` or a tensor dtype that can be compared to integer ids
- default to `None`

If `semantic_maps is None`, all semantic costs behave as if every cell is terrain id `0`.

### 6.2 Constructors

Add optional semantic map input:

```python
TogetherPlannerTerrain.from_heightmap(
    heightmap,
    world_x_range=...,
    world_y_range=...,
    semantic_map=None,
)

TogetherPlannerTerrain.from_ray_hits(
    ray_hits,
    world_x_range=...,
    world_y_range=...,
    semantic_map=None,
)
```

Existing callers remain valid.

### 6.3 Query Functions

Add GPU-resident queries:

```python
semantic_at(points_xy) -> Tensor
obstacle_height_at(points_xy, semantic_id: int) -> Tensor
obstacle_mask_at(points_xy, semantic_id: int) -> Tensor
terrain_reference_height_at(points_xy) -> Tensor
obstacle_relative_height_at(points_xy, semantic_id: int) -> Tensor
```

Implementation guidance:

- Use the same `_batched_points(...)` shape contract as `height_at(...)`.
- Use nearest-neighbor grid sampling or equivalent fixed-shape gather for semantic ids.
- Freeze the query ABI to fixed tensors only:
  - input `points_xy`: `[B, Q, 2]` or a documented reshape-equivalent static suffix
  - output ids/heights/masks: `[B, Q]`
  - local reference sampling count must be a fixed config constant, not data-dependent
- Avoid `.cpu()`, `.item()`, `.tolist()`, `nonzero`, `index_select`, `masked_select`, dynamic loops, or dynamic sub-batch calls in the training path.
- Keep all operations vectorized over `[B, ...]`.

Exact semantic-height definitions:

- `semantic_at(points_xy)`:
  nearest-neighbor semantic id on the semantic grid at each queried point
- `obstacle_mask_at(points_xy, semantic_id)`:
  boolean mask where `semantic_at(points_xy) == semantic_id`
- `obstacle_height_at(points_xy, semantic_id)`:
  sampled merged-surface height from `height_at(points_xy)` where the semantic id matches, else `0`
- `terrain_reference_height_at(points_xy)`:
  the minimum finite height among a fixed-shape terrain-only local reference stencil sampled around each query point
- `obstacle_relative_height_at(points_xy, semantic_id)`:
  `obstacle_height_at(points_xy, semantic_id) - terrain_reference_height_at(points_xy)`

The local terrain reference stencil must be fixed-shape and config-driven. The first implementation should use a constant sample count and constant radius. No implementation may choose sample count or window shape from obstacle occupancy at runtime.

### 6.4 Shared Extractor

Add one shared helper in the together planner surface, for example:

```python
build_together_terrain_from_scanner(
    ray_hits,
    *,
    world_x_range,
    world_y_range,
    semantic_map=None,
) -> TogetherPlannerTerrain
```

Requirements:

- viewer and training manager both call the same helper
- scanner-window interpretation must be identical across viewer and training
- no caller-side duplicated semantic terrain construction logic
- helper output must be GPU-resident and shape-stable

### 6.5 Viewer Wiring

In `_compute_together_local_terrain(scanner)`:

- read `scanner.data.semantic_map[env_id]` when present
- pass it through the shared terrain extractor
- keep returning `terrain, ray_hits`

If the scanner lacks `semantic_map`, pass `None` and keep current behavior.

### 6.6 Training Wiring

In `TogetherTrajectoryManager._terrain_from_env(env)`:

- read the configured scanner with `_scanner_name()`
- read `scanner.data.ray_hits_w`
- if `scanner.data.semantic_map` exists, pass it through the same shared terrain extractor used by the viewer
- if not, use height-only mode

This lets later `teacher_elevation_trajectory_env_cfg.py` migration work by changing the scanner config and `reference_height_scanner_name`, without changing planner interfaces again.

Training-path restriction:

- `TogetherTrajectoryManager._terrain_from_env(...)` must not call viewer helpers
- training migration must remain inside:
  `TogetherTrajectoryManager._terrain_from_env(...) -> shared terrain extractor -> plan_segment(...) -> together_result_to_reference_cache(...)`
- training migration must not route through any legacy CPU-canonical cache bridge

## 7. Semantic Route Candidates

### 7.1 Why Route Candidates Are Needed

Pure collision penalties on a single center rollout can say "this path is bad" but cannot produce a visible side-step unless the rollout generator has a side-step option. The viewer should show active avoidance for large obstacles, so semantic mode needs explicit route candidates.

### 7.2 Candidate Set

Use a fixed candidate count in semantic mode:

```text
center: lateral route offset 0.0
left:   lateral route offset +semantic_lateral_offset_m
right:  lateral route offset -semantic_lateral_offset_m
```

Default:

```text
semantic_lateral_offset_m = 0.45
```

The candidate dimension is fixed. The planner can expand `[B, ...]` state and command tensors to `[B, C, ...]`, reshape to `[B*C, ...]`, run the existing rollout/cost path, then choose the best candidate per env through fixed-shape tensor gather.

Hard constraint:

- candidate expansion exists only on a static candidate axis
- no env-wise or candidate-wise dynamic subbatch planner calls are allowed
- no caller may loop over candidates or candidate ids

### 7.3 When To Use Candidates

Recommended first implementation:

- if `terrain.semantic_maps is None`: keep the current one-candidate path
- if `terrain.semantic_maps is not None`: evaluate all `C=3` candidates

This keeps height-only training cost unchanged. Semantic training later accepts the known `3x` planner cost.

Semantic candidate expansion must still live inside one planner invocation. For training refresh, this still counts as one full-batch `plan_segment(...)` attempt per refresh trigger.

### 7.4 Candidate Bias

Cost must prefer the center candidate when it is safe. Add a small route penalty:

```text
J_route = abs(route_offset_m) * semantic_lateral_bias_weight
```

This gives the desired behavior:

- low, crossable small obstacle in the velocity direction: center can win by crossing
- large obstacle in the velocity direction: center receives collision cost, left/right wins
- small obstacle that is too high or unsafe: center receives collision/clearance cost, left/right wins

### 7.5 Route Shape

The route offset should be smooth across the horizon:

```text
lateral_offset(t) = route_offset_m * smoothstep(phase)
```

where `phase = t / horizon_s`.

Apply the offset in the initial body lateral direction transformed by the current yaw. This keeps the route candidate aligned with Go2's command frame.

## 8. Obstacle Semantics And Costs

### 8.1 Config Additions

Add semantic parameters to `TogetherPlannerConfig`:

```text
semantic_small_id = 1
semantic_large_id = 2
semantic_lateral_offset_m = 0.45
semantic_lateral_bias_weight = 0.05
semantic_collision_weight = 20.0
semantic_large_collision_weight = 80.0
small_crossable_height_max = 0.28
small_foot_clearance = 0.06
small_body_clearance = 0.04
large_body_clearance = 0.08
max_root_lift_for_small = 0.10
body_footprint_forward_m = 0.28
body_footprint_lateral_m = 0.14
body_footprint_sample_count = 9
```

The exact numeric defaults can be tuned during implementation, but these fields should exist as named config values instead of magic constants in cost code.

### 8.2 Foot Touchdown And Stance

Feet cannot step on semantic obstacles.

For all planned touchdown points and contact foot positions:

- id `1` small: penalize foot placement
- id `2` large: penalize foot placement more strongly

This is independent of whether small can be crossed. Crossing means swing clearance over the obstacle, not standing on it.

### 8.3 Small Obstacle Crossing

Small crossing is allowed when all are true:

- semantic id under the swing/body corridor is small
- `obstacle_relative_height_at(...)` for the relevant swing/body samples is less than or equal to `small_crossable_height_max`
- swing foot height clears obstacle top by `small_foot_clearance`
- body footprint height clears obstacle top by `small_body_clearance`
- required root lift does not exceed `max_root_lift_for_small`

If any condition fails, the small obstacle acts as an avoidance obstacle.

Training-path ABI:

- swing corridor sampling count must be fixed
- body corridor sampling count must be fixed
- no data-dependent neighborhood expansion is allowed
- the implementation should operate on static tensors such as:
  - body samples: `[B, C, T, S_body, 2]`
  - foot samples: `[B, C, T, 4, S_foot, 2]`
  - semantic ids/heights/masks: matching static suffixes

### 8.4 Large Obstacle Avoidance

Large obstacles are not crossable in the first design.

Penalties:

- foot touchdown/stance on large: high
- swing path through large: high
- body footprint touching large: very high
- root center path through large: high

The large-body cost should dominate the center route when a large obstacle is in front of the robot, so a lateral candidate wins.

### 8.5 Body Collision Approximation

Use a fixed body footprint sample grid in root frame, for example:

```text
center
front center
rear center
left center
right center
front-left
front-right
rear-left
rear-right
```

Transform each sample by the root yaw trajectory and add to `root_pos[..., :2]`.

For each body sample:

- query semantic id
- query obstacle/terrain height at the same point
- compare against a conservative body underside height

The first implementation can approximate underside height as:

```text
body_underside_z = root_z - body_underside_offset_m
```

Add `body_underside_offset_m` to config if needed. The value should be conservative and testable. If not explicitly added, document the derivation from `hip_height`.

The body sampling contract must be fixed-shape. No implementation may iterate over body samples in Python.

### 8.6 Cost Breakdown

Extend `cost_breakdown` with semantic terms:

```text
J_semantic_touchdown
J_semantic_swing
J_semantic_body
J_route
```

This matters for debugging viewer behavior. The viewer does not need to print every term by default, but tests should be able to inspect them.

## 9. Result Selection

If semantic mode evaluates `C=3` candidates:

1. produce candidate results for all `[B, C]`
2. compute total cost for each candidate
3. add route bias
4. select `best_idx = argmin(cost, dim=1)`
5. gather full trajectory/result tensors back to `[B, ...]`

The gather must be fixed-shape and GPU-resident. Avoid dynamic sub-batches and CPU indexing. Since `argmin` returns a fixed `[B]` tensor and the candidate dimension is static, this is compatible with the existing full-batch design.

Freeze semantic diagnostics as part of the stable result/debug ABI:

```text
selected_route_offset
semantic_candidate_costs
```

Requirements:

- `cost_breakdown` always includes:
  - `J_semantic_touchdown`
  - `J_semantic_swing`
  - `J_semantic_body`
  - `J_route`
- `selected_route_offset` is always present as a `[B]` tensor
- `semantic_candidate_costs` is always present as a `[B, C]` tensor
- when semantic mode is disabled, these values must still exist with zero/default semantics so tests and adapters do not branch on mode

## 10. Performance Constraints

The semantic path can cost about `3x` more than height-only because it evaluates center/left/right candidates. That is acceptable for viewer-first validation and later semantic training, but height-only training must not regress.

### 10.1 Training-Path Hard Contract

The together training path must obey the same hard contract already enforced by the repository guardrail:

- no NumPy/SciPy/Pandas/Sklearn/OpenCV imports
- no Python `for`, `while`, list/set/dict comprehensions, or generator expressions in training-path together files
- no `.cpu()`, `.item()`, `.tolist()`, `.numpy()`
- no `nonzero`, `index_select`, `index_copy_`, `masked_select`
- no `torch.equal`, `torch.allclose`, `torch.cuda.synchronize`
- no `torch.linalg.svd`, `torch.svd`
- no `torch.split`, `torch.chunk`
- no tensor-derived Python `bool(...)`, `int(...)`, or `float(...)` decisions for control flow or subbatching
- no dynamic env/candidate/row filtering before planner calls
- no repeated planner calls over candidate ids or env ids

Allowed selection/building primitives should remain in the static-axis tensor family already consistent with the backend:

- `reshape`
- `expand`
- `broadcast`
- `torch.where`
- `argmin`
- fixed-axis `gather`
- fixed-axis `grid_sample`

### 10.2 Runtime Shape Contract

All new semantic planner tensors must have fixed static suffixes chosen from config constants:

- candidate count `C=3`
- fixed body sample count
- fixed foot corridor sample count
- fixed local terrain reference sample count

No implementation may let obstacle occupancy or semantic hit count change tensor rank or planner batch shape.

Hard performance constraints:

- no CPU copies in the together training path
- full alignment with the repository's guardrail forbidden-op list in `test_batched_together_guardrails.py`
- no dynamic sub-batch planner calls
- no per-env, per-candidate, or per-sample Python loops or comprehensions in training-path tensor logic
- candidate count is fixed and small
- semantic maps remain on the same device as heightmaps
- route selection uses fixed-shape tensor operations

Performance budgets for implementation verification:

- height-only planner path latency regression: at most `+5%` against the current height-only together baseline on the same device/batch/horizon smoke
- semantic planner path wall time: at most `3.5x` the height-only path on the same device/batch/horizon smoke
- semantic planner peak CUDA memory: at most `4.0x` the height-only path on the same device/batch/horizon smoke
- no host-transfer-related regressions attributable to semantic mode

Expected verification:

- existing guardrail test still passes
- CUDA smoke for `plan_segment(...)` on semantic terrain completes without sync/device errors
- no height-only regression in existing together tests

## 11. Testing And Metrics

### 11.1 Unit Tests

Add focused tests under existing together planner test files or a new file such as:

- `Go2Pvcnn/tests/test_batched_together_semantic_terrain.py`
- `Go2Pvcnn/tests/test_batched_together_semantic_costs.py`

Required tests:

- `TogetherPlannerTerrain.from_ray_hits(..., semantic_map=...)` preserves heightmap behavior and stores semantic map shape.
- `TogetherPlannerTerrain` rejects mismatched semantic-map shape, resolution, or device contract.
- `semantic_at(...)` returns expected ids for known grid points.
- `semantic_at(...)` returns terrain id `0` when no semantic map exists.
- `terrain_reference_height_at(...)` and `obstacle_relative_height_at(...)` match the frozen fixed-stencil contract on synthetic fixtures.
- foot touchdown cost penalizes small and large obstacle cells.
- low small obstacle in the center corridor keeps center route selected when foot/body clearance is enough.
- high small obstacle in the center corridor selects a lateral route.
- large obstacle in the center corridor selects a lateral route.
- body footprint collision with large obstacle increases `J_semantic_body`.
- `cost_breakdown` always includes `J_semantic_touchdown`, `J_semantic_swing`, `J_semantic_body`, and `J_route`.
- `selected_route_offset` and `semantic_candidate_costs` exist with stable shapes in semantic and height-only mode.
- fixed-shape `[B, C] -> [B]` candidate selection regression proves selection stays device-resident and gather-based.
- height-only path produces the same output as before within explicit named tolerances against the current baseline tests.
- manager-path regression proves `_terrain_from_env(...)` forwards `semantic_map` through the shared terrain extractor without breaking cache/runtime invariants.

### 11.2 Viewer Tests

Extend viewer/runtime tests:

- parser exposes `--terrain-row` and `--terrain-col` defaults as `0`
- valid `row/col` moves env `0` to `terrain_origins[row, col]`
- invalid `row/col` raises a clear error
- manual `R` reset preserves selected `row/col` and refreshes scanner buffers before replanning
- targeted semantic viewer smoke on `together` reaches `[Viewer][Plan]`
- route diagnostics or trajectory delta show lateral deviation for large obstacle cases

### 11.3 Acceptance Metrics

The real implementation gate is a named deterministic synthetic-fixture suite, not the manual viewer pass.

Fixture contract for all acceptance cases:

- batch size `B=1`
- candidate count `C=3`
- horizon `35`
- dt `0.02`
- world ranges `(-0.75, 0.75)` for both axes
- synthetic grid shape `151 x 151`
- start root pose `(0.0, 0.0, 0.30)`
- command `(0.30, 0.0, 0.0)`
- exact tolerance constants:
  - clearance tolerance `0.01m`
  - root-lift tolerance `0.01m`
  - route-offset tensor tolerance `1e-6`

Named fixtures:

- `F1_small_low_cross_center`
  - flat base terrain `z=0`
  - one small obstacle centered at `(0.45, 0.0)`
  - obstacle top chosen so `obstacle_relative_height_at(...) = 0.16`
- `F2_small_high_avoid`
  - same geometry placement as `F1`
  - obstacle top chosen so `obstacle_relative_height_at(...) = 0.34`
- `F3_large_center_avoid`
  - flat base terrain `z=0`
  - one large obstacle centered at `(0.45, 0.0)`
  - obstacle top chosen so `obstacle_relative_height_at(...) = 0.45`

- `F1_small_low_cross_center`:
  - `selected_route_offset == 0.0` within route-offset tensor tolerance
  - no touchdown cell has semantic id `1` or `2`
  - minimum swing clearance over the obstacle is at least `small_foot_clearance - 0.01`
  - minimum body clearance over the obstacle is at least `small_body_clearance - 0.01`
  - maximum root lift over start height is at most `max_root_lift_for_small + 0.01`
- `F2_small_high_avoid`:
  - `abs(selected_route_offset) >= semantic_lateral_offset_m - 0.01`
  - body footprint does not collide with the obstacle
- `F3_large_center_avoid`:
  - `abs(selected_route_offset) >= semantic_lateral_offset_m - 0.01`
  - no body footprint sample intersects semantic id `2`
  - no touchdown cell has semantic id `2`
- guardrails:
  - existing together guardrail test passes
  - existing together manager/runtime tests pass
  - one explicit height-only control fixture proves center remains preferred on empty terrain
  - performance budgets in section `10` pass on the semantic smoke

### 11.4 Manual Viewer Acceptance

Run the user's viewer command with explicit tile selection:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:2 \
  --num_envs 1 \
  --terrain task \
  --planner-backend together \
  --terrain-row 8 \
  --terrain-col 0
```

This pass is supplemental only. It does not replace the deterministic-fixture gate above.

Recommended command additions for manual inspection:

```bash
  --scripted-command "0.30 0.0 0.0" \
  --scripted-command-cycles 2
```

Expected visual result:

- low small obstacles in the command direction can be crossed without stepping on them
- high small obstacles are avoided
- large obstacles are avoided laterally
- the body does not visibly intersect large obstacles

Manual viewing may confirm subjective motion quality, but collision sign-off comes from deterministic fixture metrics, not visual judgment.

## 12. Parallel Implementation Boundaries

Implementation should avoid parallel writes to the same hot files unless sequencing is clear.

Recommended ownership split:

1. **Terrain API worker**
   - Owns:
     - `Go2Pvcnn/extension/batched_together_planner/terrain.py`
     - `Go2Pvcnn/extension/batched_together_planner/types.py`
     - shared terrain extractor helper
     - terrain API tests
   - Does not edit planner/cost selection.

2. **Semantic cost and route worker**
   - Owns:
     - `Go2Pvcnn/extension/batched_together_planner/config.py`
     - `Go2Pvcnn/extension/batched_together_planner/parameterization.py`
     - `Go2Pvcnn/extension/batched_together_planner/costs.py`
     - `Go2Pvcnn/extension/batched_together_planner/planner.py`
     - semantic planner tests
   - Waits for the terrain API contract to land.

3. **Viewer spawn/wiring worker**
   - Owns:
     - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
     - viewer parser/reset tests
   - May proceed only after the terrain API worker freezes the shared extractor and query semantics.

4. **Training manager wiring worker**
   - Owns:
     - `Go2Pvcnn/extension/batched_together_planner/manager.py`
     - `Go2Pvcnn/extension/batched_together_planner/adapter.py`
     - runtime path tests
     - parity/guardrail-adjacent tests for the manager path
   - May proceed in parallel with the viewer worker after the shared extractor contract lands.

5. **Review/test-only agent**
   - Read-only.
   - Checks metrics, guardrails, test coverage, and future training boundary.

Agents must not revert unrelated dirty worktree changes. Agents are not alone in the codebase and must adapt to existing edits.

## 13. Migration To Training

After viewer acceptance:

1. Update or derive `teacher_elevation_trajectory_env_cfg.py` to expose a semantic scanner equivalent to the viewer scanner.
2. Set `reference_height_scanner_name = "semantic_height_scanner"` for semantic training runs.
3. Keep `planner_backend = "together"`.
4. Keep the training path on the existing device-preserving manager/cache chain only:
   `TogetherTrajectoryManager._terrain_from_env(...) -> shared terrain extractor -> plan_segment(...) -> together_result_to_reference_cache(...)`
5. Do not reuse viewer-only helpers and do not route through the legacy CPU-canonical cache bridge.
6. Run manager/runtime tests with a real semantic scanner.
7. Run a semantic-training acceptance gate separate from viewer acceptance:
   - semantic manager path tests pass
   - together guardrail test passes
   - semantic planner performance budgets pass
8. Start with smaller env counts to measure the expected `3x` semantic candidate cost.
9. Only then expand to PPO training.

The planner API should already support this path through `TogetherTrajectoryManager._terrain_from_env(...)`.

## 14. Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Center route still wins near large obstacles because lateral candidate is too weak | Make `J_semantic_body` and large collision weights dominate route bias; add deterministic large-avoid test |
| Small obstacles become always avoided, defeating the "can cross" goal | Add low-small-cross acceptance where center route must win |
| Small obstacles become always crossed, even when too high | Add high-small-avoid acceptance where lateral route must win |
| Body collision approximation misses large obstacles | Use fixed footprint samples and test front/side/corner body collisions |
| Semantic route path breaks height-only parity | Keep one-candidate height-only path and add regression test |
| Semantic mode slows training too much | Use fixed `C=3`; benchmark before PPO rollout |
| Parallel agents conflict in planner files | Sequence terrain API before cost/route work; keep viewer worker separate |
| Viewer row/col silently selects the wrong tile | Raise on invalid indices; print selected row/col and world origin |

## 15. Open Questions

- Exact default `body_underside_offset_m` should be chosen during implementation from Go2 geometry or conservative observed root height.
- Exact lateral offset default may need visual tuning after the first viewer smoke.
- Whether `selected_route_offset` should be added to `TogetherPlannerResult` or kept in `cost_breakdown` depends on test ergonomics during implementation.

These questions do not block the first implementation plan because each has a conservative default path.
