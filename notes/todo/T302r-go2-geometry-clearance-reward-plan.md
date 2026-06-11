# T302r Go2 Geometry Clearance Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sparse point-only `semantic_body_part_clearance_reward` with fixed-shape Go2 geometry neighborhood queries so flat-small training gets a dense pre-contact semantic clearance signal.

**Architecture:** Keep the existing reward entry and current scanner terrain ownership, but change the internal representation from center points to Go2 body primitives: foot spheres, calf/thigh capsules, and a base oriented footprint box. Every primitive is converted into fixed-size batched query tensors over the current scanner semantic/elevation maps; no per-env Python loops, USD traversal, PhysX geometry queries, or MPC loss/reference changes are allowed.

**Tech Stack:** PyTorch tensor ops, IsaacLab robot/scanner runtime data, `extension.batch_mpc_planner.terrain.MpcPlannerTerrain`, `height_at()`, `semantic_at()`, focused pytest, `env_isaacsim` probes.

---

## Source Spec

- [../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html](../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html)
- Parent T302q plan: [T302q-flat-small-avoidance-reward-plan.md](T302q-flat-small-avoidance-reward-plan.md)
- Root-cause evidence: [../log/2026-06-11-1420-t302q-flat-small-curriculum-clearance-root-cause-probe.md](../log/2026-06-11-1420-t302q-flat-small-curriculum-clearance-root-cause-probe.md)

## Current State

- Current `semantic_contact_collision` has sparse nonzero true contact, so small objects can collide with the robot.
- Current `semantic_body_part_clearance` can stay exactly `0` because it only queries a small number of foot/calf/thigh centerline points.
- Current reward code:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
- Current implementation now uses geometry section counts with first-version defaults `calf_sections=7` and `thigh_sections=7`; old `shank_*` aliases remain accepted for compatibility.
- Local focused verification and a 16-env real IsaacLab smoke pass. TensorBoard readout for `2026-06-11_17-05-24` showed the reward still remained all zero under training.
- Radius/margin probe on 2026-06-11 found that radius alone is insufficient: radius `0.50m` with original margins hit small semantic cells but produced `positive_deficit=0`; radius `0.50m` plus enlarged margins produced nonzero clearance reward in `1/64` envs. Flat-small cfg is now intentionally set to signal-first params (`0.50m` query radius, `0.20m` foot/base margin, `0.40m` calf/thigh margin) so the next short training/TensorBoard run can confirm the dense reward is alive.
- Flat-mask/curriculum bookkeeping is already fixed in T302q and must not be regressed.

## Confirmed Requirements

- [ ] Keep `semantic_contact_collision` as the authoritative real-contact penalty.
- [ ] Keep the public reward output shape `[num_envs]`.
- [ ] Keep the existing scanner-current-map contract from T302q; do not restore reward-private scanner root anchors.
- [ ] Use foot sphere, calf capsule, thigh capsule, and base oriented footprint approximations derived from Go2 USD dimensions.
- [ ] Use fixed-shape GPU batched queries over scanner `semantic_map` and `elevation_map`.
- [ ] No per-env Python loops in the reward hot path.
- [ ] No USD or PhysX geometry queries in the reward hot path.
- [ ] Do not modify MPC planner loss, reference cache, command shaping, or planner targets.
- [ ] Keep T302q observation/action ABI and checkpoint compatibility unchanged.

## File Structure

- Modify: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
  - Add geometry query helper functions, cached circular offsets, base footprint samples, capsule section samples, geometry aggregation, and the updated reward entry parameters.
- Modify: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`
  - Add pure tensor regression tests for sphere/capsule/base neighborhood coverage, fixed shapes, no NaN/Inf, and old point-sampling miss cases.
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - Replace sparse sample-count params with geometry params and add base weight/radius/footprint config.
- Modify only if required: `Go2Pvcnn/extension/mdp/__init__.py`
  - Keep the reward export stable.
- Update: `notes/todo.md`, this page, `notes/log/index.md`, and one per-verification log under `notes/log/`.

## Target Runtime Parameters

```text
foot_sphere_radius_m: 0.022
foot_query_radius_m: 0.035
foot_margin_m: 0.015
foot_weight: 0.5

calf_capsule_radius_m: 0.040
calf_query_radius_m: 0.045
calf_sections: 7
calf_margin_m: 0.040
calf_weight: 2.0

thigh_capsule_radius_m: 0.040
thigh_query_radius_m: 0.045
thigh_sections: 7
thigh_margin_m: 0.040
thigh_weight: 1.5

base_half_extents_m: [0.20, 0.06, 0.07]
base_footprint_grid: [5, 3]
base_query_radius_m: 0.030
base_margin_m: 0.020
base_weight: 1.0

neighbor_offsets: cached per query radius and map resolution
penalty_clip: 1.0
```

## Reward Rule

For each primitive group, generate `query_xy` and `surface_z` with fixed shape:

```text
query_xy:  [N, Q, 2]
surface_z: [N, Q]
semantic:  [N, Q]
height:    [N, Q]
```

Then compute:

```text
small_mask = semantic in small_semantic_ids
deficit = relu(height + margin - surface_z)
group_penalty = reduce_mean_or_sum(small_mask * deficit^2)
reward = -clip(weighted_group_sum, 0.0, penalty_clip)
```

`surface_z` is the lower surface of the fitted body primitive, not just the primitive center. For foot/capsule this means center/section `z - primitive_radius`; for base this means root/body footprint sample `z - base_half_z`.

## Open Children

| Leaf | Status | Priority | Why Active | Next Read |
| --- | --- | --- | --- | --- |
| T302r.1 | verify | P0 | Pure tensor geometry tests cover foot sphere, calf/thigh capsule, base footprint, fixed shape, and no per-env loop guard. | [implementation log](../log/2026-06-11-1551-t302r-go2-geometry-clearance-implementation.md) |
| T302r.2 | verify | P0 | Cached circular offsets and fixed-shape sphere/capsule/base query builders are implemented. | [reward file](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py) |
| T302r.3 | verify | P0 | Geometry-aware semantic/elevation neighborhood reductions replace the wrapper hot path. | [reward file](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py) |
| T302r.4 | verify | P0 | Flat-small cfg is wired with first-version geometry params while observation/action shape stays unchanged in real smoke. | [env cfg](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py) |
| T302r.5 | active | P0 | Short TensorBoard sanity remains open after signal-first params; check whether `semantic_body_part_clearance` becomes nonzero in training. | [radius/margin log](../log/2026-06-11-1810-t302r-clearance-radius-margin-probe.md) |
| T302r.6 | verify | P0 | Real radius/margin probe now separates scanner small-cell presence, body query small hits, positive deficits, and reward nonzero rate. | [radius/margin log](../log/2026-06-11-1810-t302r-clearance-radius-margin-probe.md) |

## Task 1: Pure Tensor RED Tests For Geometry Coverage

**Files:**
- Modify: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`
- Read: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`

- [ ] **Step 1: Add imports for planned geometry helpers**

Add these imports from `extension.mdp.semantic_body_part_clearance`:

```python
from extension.mdp.semantic_body_part_clearance import (
    _body_geometry_query_points,
    _cached_circle_offsets,
    _current_body_part_sample_points,
    _semantic_clearance_penalty_from_points,
    _semantic_geometry_clearance_penalty,
)
```

- [ ] **Step 2: Add a fixture where the old center point misses a nearby small cell**

Add a test that places a foot center at the map center but puts a small semantic cell one cell away inside `foot_query_radius_m=0.035`. The expected result is a negative reward from geometry neighborhood query.

```python
def test_foot_sphere_neighborhood_detects_adjacent_small_cell() -> None:
    elevation = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    elevation[:, 4, 5] = 0.10
    semantic[:, 4, 5] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.08, 0.08),
        world_y_range=(-0.08, 0.08),
    )
    centers = {"foot": torch.tensor([[[[0.0, 0.0, 0.11]]]], dtype=torch.float32)}

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part=centers,
        root_pos_w=torch.tensor([[0.0, 0.0, 0.20]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        small_semantic_ids=(1,),
        foot_radius_m=0.022,
        foot_query_radius_m=0.035,
        foot_margin_m=0.015,
        foot_weight=0.5,
        penalty_clip=1.0,
    )

    assert reward.shape == (1,)
    assert reward.item() < 0.0
```

- [ ] **Step 3: Add capsule coverage tests for calf and thigh**

Add two tests with `calf`/`thigh` centers formed as `[N,4,7,3]`. Put the small cell near one section but not exactly under the section center. Expected: negative reward and finite output.

```python
def test_calf_capsule_neighborhood_detects_adjacent_small_cell() -> None:
    elevation = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    elevation[:, 4, 5] = 0.16
    semantic[:, 4, 5] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.08, 0.08),
        world_y_range=(-0.08, 0.08),
    )
    calf = torch.zeros((1, 4, 7, 3), dtype=torch.float32)
    calf[..., 2] = 0.18

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part={"calf": calf},
        root_pos_w=torch.tensor([[0.0, 0.0, 0.25]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        small_semantic_ids=(1,),
        calf_radius_m=0.040,
        calf_query_radius_m=0.045,
        calf_margin_m=0.040,
        calf_weight=2.0,
        penalty_clip=1.0,
    )

    assert torch.isfinite(reward).all()
    assert reward.item() < 0.0
```

```python
def test_thigh_capsule_neighborhood_detects_adjacent_small_cell() -> None:
    elevation = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    elevation[:, 5, 4] = 0.20
    semantic[:, 5, 4] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.08, 0.08),
        world_y_range=(-0.08, 0.08),
    )
    thigh = torch.zeros((1, 4, 7, 3), dtype=torch.float32)
    thigh[..., 2] = 0.23

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part={"thigh": thigh},
        root_pos_w=torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        small_semantic_ids=(1,),
        thigh_radius_m=0.040,
        thigh_query_radius_m=0.045,
        thigh_margin_m=0.040,
        thigh_weight=1.5,
        penalty_clip=1.0,
    )

    assert torch.isfinite(reward).all()
    assert reward.item() < 0.0
```

- [ ] **Step 4: Add base footprint coverage test**

Add a test where the base center is not directly over the small cell, but the oriented footprint grid covers it.

```python
def test_base_footprint_grid_detects_small_cell_under_body_extent() -> None:
    elevation = torch.zeros((1, 17, 17), dtype=torch.float32)
    semantic = torch.zeros((1, 17, 17), dtype=torch.long)
    elevation[:, 8, 13] = 0.14
    semantic[:, 8, 13] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.24, 0.24),
        world_y_range=(-0.24, 0.24),
    )

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part={},
        root_pos_w=torch.tensor([[0.0, 0.0, 0.18]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        small_semantic_ids=(1,),
        include_base=True,
        base_half_extents_m=(0.20, 0.06, 0.07),
        base_footprint_grid=(5, 3),
        base_query_radius_m=0.030,
        base_margin_m=0.020,
        base_weight=1.0,
        penalty_clip=1.0,
    )

    assert reward.shape == (1,)
    assert reward.item() < 0.0
```

- [ ] **Step 5: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: new tests fail because `_semantic_geometry_clearance_penalty`, `_body_geometry_query_points`, and `_cached_circle_offsets` do not exist yet.

## Task 2: Cached Circular Offsets And Fixed-Shape Query Builder

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
- Test: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`

- [ ] **Step 1: Add cached circle offsets helper**

Implement a helper with module-level cache keyed by `(device, dtype, radius_m, resolution_m)`. Keep `K` bounded; first version may use the nine grid offsets plus diagonals/axis midpoints inside the radius, capped to stable order.

```python
_CIRCLE_OFFSET_CACHE: dict[tuple[str, torch.dtype, float, float], torch.Tensor] = {}


def _cached_circle_offsets(
    *,
    radius_m: float,
    resolution_m: float,
    device: torch.device,
    dtype: torch.dtype,
    max_offsets: int = 13,
) -> torch.Tensor:
    radius = float(radius_m)
    resolution = max(float(resolution_m), 1.0e-6)
    key = (str(device), dtype, round(radius, 6), round(resolution, 6))
    cached = _CIRCLE_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached
    steps = max(1, int(torch.ceil(torch.tensor(radius / resolution)).item()))
    coords = torch.arange(-steps, steps + 1, dtype=dtype, device=device) * resolution
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    offsets = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    keep = torch.linalg.vector_norm(offsets, dim=-1) <= radius + 1.0e-6
    offsets = offsets[keep]
    dist = torch.linalg.vector_norm(offsets, dim=-1)
    order = torch.argsort(dist)
    offsets = offsets[order]
    if int(offsets.shape[0]) > int(max_offsets):
        offsets = offsets[: int(max_offsets)]
    _CIRCLE_OFFSET_CACHE[key] = offsets.contiguous()
    return _CIRCLE_OFFSET_CACHE[key]
```

- [ ] **Step 2: Add map resolution helper**

```python
def _terrain_resolution_xy(terrain) -> tuple[float, float]:
    height_map = torch.as_tensor(terrain.height_map)
    _, height, width = height_map.shape
    x0, x1 = terrain.world_x_range
    y0, y1 = terrain.world_y_range
    return abs(float(x1) - float(x0)) / max(int(width) - 1, 1), abs(float(y1) - float(y0)) / max(int(height) - 1, 1)
```

- [ ] **Step 3: Add query expansion helper**

```python
def _expand_centers_with_offsets(
    centers: torch.Tensor,
    *,
    surface_z: torch.Tensor,
    offsets_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_centers = centers.reshape(centers.shape[0], -1, 3)
    flat_surface_z = surface_z.reshape(surface_z.shape[0], -1)
    query_xy = flat_centers[..., :2].unsqueeze(2) + offsets_xy.view(1, 1, -1, 2)
    expanded_surface_z = flat_surface_z.unsqueeze(-1).expand(-1, -1, offsets_xy.shape[0])
    return query_xy.reshape(centers.shape[0], -1, 2), expanded_surface_z.reshape(centers.shape[0], -1)
```

- [ ] **Step 4: Add base footprint helper**

Implement root-yaw oriented footprint samples. Use `extract_yaw_batch()` from `extension.convention`.

```python
def _base_footprint_centers(
    *,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    half_extents_m: tuple[float, float, float],
    footprint_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    from extension.convention import extract_yaw_batch

    half_x, half_y, half_z = [float(v) for v in half_extents_m]
    grid_x, grid_y = [int(v) for v in footprint_grid]
    xs = torch.linspace(-half_x, half_x, grid_x, dtype=root_pos_w.dtype, device=root_pos_w.device)
    ys = torch.linspace(-half_y, half_y, grid_y, dtype=root_pos_w.dtype, device=root_pos_w.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    local = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    yaw = extract_yaw_batch(root_quat_w)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    world_x = root_pos_w[:, 0:1] + local[:, 0].view(1, -1) * cos_yaw.view(-1, 1) - local[:, 1].view(1, -1) * sin_yaw.view(-1, 1)
    world_y = root_pos_w[:, 1:2] + local[:, 0].view(1, -1) * sin_yaw.view(-1, 1) + local[:, 1].view(1, -1) * cos_yaw.view(-1, 1)
    centers = torch.stack((world_x, world_y, root_pos_w[:, 2:3].expand_as(world_x)), dim=-1)
    surface_z = centers[..., 2] - half_z
    return centers, surface_z
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: helper import tests progress; penalty tests can still fail until Task 3 aggregation exists.

## Task 3: Geometry-Aware Semantic Clearance Aggregation

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
- Test: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`

- [ ] **Step 1: Implement per-group query penalty helper**

```python
def _geometry_group_penalty(
    *,
    terrain,
    centers: torch.Tensor,
    surface_z: torch.Tensor,
    query_radius_m: float,
    margin_m: float,
    small_semantic_ids,
    cache,
) -> torch.Tensor:
    from extension.batch_mpc_planner.terrain import height_at, semantic_at

    x_res, y_res = _terrain_resolution_xy(terrain)
    resolution = min(x_res, y_res)
    offsets = _cached_circle_offsets(
        radius_m=query_radius_m,
        resolution_m=resolution,
        device=centers.device,
        dtype=centers.dtype,
        max_offsets=13,
    )
    query_xy, query_surface_z = _expand_centers_with_offsets(centers, surface_z=surface_z, offsets_xy=offsets)
    terrain_z = height_at(terrain, query_xy, cache=cache).to(dtype=centers.dtype, device=centers.device)
    semantic_id = semantic_at(terrain, query_xy, cache=cache)
    small_mask = _semantic_id_mask(semantic_id.to(dtype=torch.long), small_semantic_ids).to(dtype=centers.dtype)
    deficit = torch.relu(terrain_z + float(margin_m) - query_surface_z)
    return (small_mask * deficit.square()).mean(dim=1)
```

- [ ] **Step 2: Implement `_semantic_geometry_clearance_penalty`**

Use `TerrainQueryCache()` once per reward call. Missing groups should contribute zero. The function must accept old tests with only one group present.

```python
def _semantic_geometry_clearance_penalty(
    *,
    terrain,
    centers_by_part,
    root_pos_w,
    root_quat_w,
    small_semantic_ids=(1,),
    foot_radius_m=0.022,
    foot_query_radius_m=0.035,
    foot_margin_m=0.015,
    foot_weight=0.5,
    calf_radius_m=0.040,
    calf_query_radius_m=0.045,
    calf_margin_m=0.040,
    calf_weight=2.0,
    thigh_radius_m=0.040,
    thigh_query_radius_m=0.045,
    thigh_margin_m=0.040,
    thigh_weight=1.5,
    include_base=False,
    base_half_extents_m=(0.20, 0.06, 0.07),
    base_footprint_grid=(5, 3),
    base_query_radius_m=0.030,
    base_margin_m=0.020,
    base_weight=1.0,
    penalty_clip=1.0,
):
    from extension.batch_mpc_planner.terrain import TerrainQueryCache

    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    device = height_map.device
    dtype = height_map.dtype
    num_envs = int(height_map.shape[0])
    total = torch.zeros(num_envs, dtype=dtype, device=device)
    cache = TerrainQueryCache()

    if "foot" in centers_by_part:
        centers = _part_points_3d(centers_by_part["foot"], part_name="foot", num_envs=num_envs, dtype=dtype, device=device)
        surface_z = centers[..., 2] - float(foot_radius_m)
        total = total + float(foot_weight) * _geometry_group_penalty(
            terrain=terrain,
            centers=centers,
            surface_z=surface_z,
            query_radius_m=foot_query_radius_m,
            margin_m=foot_margin_m,
            small_semantic_ids=small_semantic_ids,
            cache=cache,
        )

    if "calf" in centers_by_part:
        centers = _part_points_3d(centers_by_part["calf"], part_name="calf", num_envs=num_envs, dtype=dtype, device=device)
        surface_z = centers[..., 2] - float(calf_radius_m)
        total = total + float(calf_weight) * _geometry_group_penalty(
            terrain=terrain,
            centers=centers,
            surface_z=surface_z,
            query_radius_m=calf_query_radius_m,
            margin_m=calf_margin_m,
            small_semantic_ids=small_semantic_ids,
            cache=cache,
        )

    if "thigh" in centers_by_part:
        centers = _part_points_3d(centers_by_part["thigh"], part_name="thigh", num_envs=num_envs, dtype=dtype, device=device)
        surface_z = centers[..., 2] - float(thigh_radius_m)
        total = total + float(thigh_weight) * _geometry_group_penalty(
            terrain=terrain,
            centers=centers,
            surface_z=surface_z,
            query_radius_m=thigh_query_radius_m,
            margin_m=thigh_margin_m,
            small_semantic_ids=small_semantic_ids,
            cache=cache,
        )

    if include_base:
        root_pos = torch.as_tensor(root_pos_w, dtype=dtype, device=device)
        root_quat = torch.as_tensor(root_quat_w, dtype=dtype, device=device)
        centers, surface_z = _base_footprint_centers(
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            half_extents_m=base_half_extents_m,
            footprint_grid=base_footprint_grid,
        )
        total = total + float(base_weight) * _geometry_group_penalty(
            terrain=terrain,
            centers=centers,
            surface_z=surface_z,
            query_radius_m=base_query_radius_m,
            margin_m=base_margin_m,
            small_semantic_ids=small_semantic_ids,
            cache=cache,
        )

    return -torch.clamp(total, min=0.0, max=float(penalty_clip))
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: geometry tests pass; existing point-based tests still pass while the wrapper has not yet been switched.

## Task 4: Wire Reward Wrapper To Geometry Path

**Files:**
- Modify: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Update `_current_body_part_sample_points` contract**

Keep the return keys stable enough for tests, but update defaults to `calf_sections=7` and `thigh_sections=7`. The calf key may replace `shank` internally, while accepting `shank_sample_count` as a compatibility alias for old callers during this plan.

```python
def _current_body_part_sample_points(
    robot,
    *,
    body_ids,
    calf_sections=7,
    thigh_sections=7,
    shank_sample_count=None,
    thigh_sample_count=None,
):
    if shank_sample_count is not None:
        calf_sections = int(shank_sample_count)
    if thigh_sample_count is not None:
        thigh_sections = int(thigh_sample_count)
    ...
    return {
        "foot": foot.unsqueeze(2),
        "calf": _segment_samples(calf, foot, int(calf_sections)),
        "thigh": _segment_samples(thigh, calf, int(thigh_sections)),
    }
```

- [ ] **Step 2: Update `semantic_body_part_clearance_reward` signature**

Add geometry parameters while preserving old aliases only long enough to avoid hidden import failures.

```python
def semantic_body_part_clearance_reward(
    env,
    *,
    asset_cfg,
    scanner_cfg,
    contact_sensor_cfg=None,
    small_semantic_ids=(1,),
    foot_margin_m=0.015,
    calf_margin_m=0.04,
    thigh_margin_m=0.04,
    base_margin_m=0.02,
    foot_weight=0.5,
    calf_weight=2.0,
    thigh_weight=1.5,
    base_weight=1.0,
    foot_sphere_radius_m=0.022,
    foot_query_radius_m=0.035,
    calf_capsule_radius_m=0.040,
    calf_query_radius_m=0.045,
    calf_sections=7,
    thigh_capsule_radius_m=0.040,
    thigh_query_radius_m=0.045,
    thigh_sections=7,
    base_half_extents_m=(0.20, 0.06, 0.07),
    base_footprint_grid=(5, 3),
    base_query_radius_m=0.030,
    include_base=True,
    penalty_clip=1.0,
    shank_margin_m=None,
    shank_weight=None,
    shank_sample_count=None,
    thigh_sample_count=None,
    **unused_compat,
):
```

- [ ] **Step 3: Call `_semantic_geometry_clearance_penalty` from wrapper**

Use `robot.data.root_pos_w` and `robot.data.root_quat_w` for base orientation.

```python
points = _current_body_part_sample_points(
    robot,
    body_ids=body_ids,
    calf_sections=calf_sections,
    thigh_sections=thigh_sections,
    shank_sample_count=shank_sample_count,
    thigh_sample_count=thigh_sample_count,
)
return _semantic_geometry_clearance_penalty(
    terrain=terrain,
    centers_by_part=points,
    root_pos_w=robot.data.root_pos_w,
    root_quat_w=robot.data.root_quat_w,
    small_semantic_ids=small_semantic_ids,
    foot_radius_m=foot_sphere_radius_m,
    foot_query_radius_m=foot_query_radius_m,
    foot_margin_m=foot_margin_m,
    foot_weight=foot_weight,
    calf_radius_m=calf_capsule_radius_m,
    calf_query_radius_m=calf_query_radius_m,
    calf_margin_m=calf_margin_m if shank_margin_m is None else shank_margin_m,
    calf_weight=calf_weight if shank_weight is None else shank_weight,
    thigh_radius_m=thigh_capsule_radius_m,
    thigh_query_radius_m=thigh_query_radius_m,
    thigh_margin_m=thigh_margin_m,
    thigh_weight=thigh_weight,
    include_base=include_base,
    base_half_extents_m=base_half_extents_m,
    base_footprint_grid=base_footprint_grid,
    base_query_radius_m=base_query_radius_m,
    base_margin_m=base_margin_m,
    base_weight=base_weight,
    penalty_clip=penalty_clip,
)
```

- [ ] **Step 4: Update cfg params**

In `_semantic_body_part_clearance_reward_term()`, replace the old sparse params with geometry params:

```python
"calf_margin_m": 0.04,
"thigh_margin_m": 0.04,
"base_margin_m": 0.02,
"foot_weight": 0.5,
"calf_weight": 2.0,
"thigh_weight": 1.5,
"base_weight": 1.0,
"foot_sphere_radius_m": 0.022,
"foot_query_radius_m": 0.035,
"calf_capsule_radius_m": 0.040,
"calf_query_radius_m": 0.045,
"calf_sections": 7,
"thigh_capsule_radius_m": 0.040,
"thigh_query_radius_m": 0.045,
"thigh_sections": 7,
"base_half_extents_m": (0.20, 0.06, 0.07),
"base_footprint_grid": (5, 3),
"base_query_radius_m": 0.030,
"include_base": True,
"penalty_clip": 1.0,
```

- [ ] **Step 5: Update static cfg test expectations**

In `Go2Pvcnn/tests/test_batch_mpc_backend.py`, update the flat-small reward wiring test so it asserts:

```python
assert cfg.rewards.semantic_body_part_clearance.params["calf_sections"] == 7
assert cfg.rewards.semantic_body_part_clearance.params["thigh_sections"] == 7
assert cfg.rewards.semantic_body_part_clearance.params["include_base"] is True
assert cfg.rewards.semantic_body_part_clearance.params["base_footprint_grid"] == (5, 3)
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: tests pass or reveal exact compatibility gaps to fix in the same task.

## Task 5: Performance And Shape Guards

**Files:**
- Modify: `Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py`
- Modify if needed: `Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py`

- [ ] **Step 1: Add a 1024-env fixed-shape smoke test**

Create a synthetic map/body pose test that verifies `[1024]` output and finite values. Keep it CPU-safe; if too slow on CPU, use `N=128` in pytest and reserve `N=1024` for the real probe command.

```python
def test_geometry_clearance_returns_fixed_shape_for_batched_envs() -> None:
    num_envs = 128
    elevation = torch.zeros((num_envs, 17, 17), dtype=torch.float32)
    semantic = torch.zeros((num_envs, 17, 17), dtype=torch.long)
    elevation[:, 8, 8] = 0.10
    semantic[:, 8, 8] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.24, 0.24),
        world_y_range=(-0.24, 0.24),
    )
    foot = torch.zeros((num_envs, 4, 1, 3), dtype=torch.float32)
    foot[..., 2] = 0.11
    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part={"foot": foot},
        root_pos_w=torch.zeros((num_envs, 3), dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_envs, 1),
        small_semantic_ids=(1,),
    )

    assert reward.shape == (num_envs,)
    assert torch.isfinite(reward).all()
```

- [ ] **Step 2: Add a no per-env loop static guard**

Use a source scan in the test file to prevent `for env_id in range(num_envs)` style loops in the reward file.

```python
def test_geometry_reward_hot_path_has_no_per_env_loop() -> None:
    source = (GO2PVCNN_ROOT / "extension/mdp/semantic_body_part_clearance.py").read_text(encoding="utf-8")
    forbidden = ["for env_id in", "for env_idx in", "range(num_envs)", "range(env.num_envs)"]
    for text in forbidden:
        assert text not in source
```

- [ ] **Step 3: Run focused performance-safe tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Expected: all tests pass locally.

## Task 6: Real IsaacLab Probe

**Files:**
- Read/update only if needed: `Go2Pvcnn/scripts/train.py`
- Create log: `notes/log/YYYY-MM-DD-HHMM-t302r-geometry-clearance-real-probe.md`

- [ ] **Step 1: Run a 64-env real probe**

Use the same environment required by the user:

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --task Isaac-Teacher-ManagerBased-RslRl-Go2-FlatSmallAvoidance-v0 \
  --num_envs 64 \
  --max_iterations 1 \
  --headless
```

Expected:

```text
exit code 0
semantic_body_part_clearance appears in reward terms
no NaN/Inf
collection time does not regress catastrophically
```

- [ ] **Step 2: Run a direct reward nonzero probe**

If a one-off probe script or monkeypatch is faster than full training, use it to print:

```text
semantic_small_pixels
geometry_clearance_nonzero_envs
geometry_clearance_min
geometry_clearance_mean
geometry_clearance_max_abs
```

Expected: when scanner has small pixels near robot geometry, `geometry_clearance_nonzero_envs > 0` in at least some frames. If it is still `0`, inspect whether obstacles are outside scanner range or all body surfaces are above `height + margin`.

- [ ] **Step 3: Create verification log**

Create a log under `notes/log/` with:

```text
purpose
stage
related todo T302r
command/procedure
input conditions
key metrics
result
conclusion
follow-up
git refs
```

## Task 7: Short Training / TensorBoard Sanity

**Files:**
- Read: `logs/rsl_rl/...`
- Update: `notes/log/index.md`
- Update: `notes/todo.md`
- Update: this page

- [ ] **Step 1: Run short resume training**

Use the existing warm-start checkpoint path unless a newer approved checkpoint exists:

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --task Isaac-Teacher-ManagerBased-RslRl-Go2-FlatSmallAvoidance-v0 \
  --num_envs 1024 \
  --max_iterations 50 \
  --headless \
  --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07 \
  --checkpoint model_14000.pt
```

- [ ] **Step 2: Inspect scalar behavior**

Check TensorBoard scalar/event files for:

```text
semantic_body_part_clearance no longer all zero
semantic_contact_collision remains sparse
Perf/collection_time remains in the expected post-raycaster-fix range
semantic_success_rate / semantic_gate_pass interpretation remains episode-level
plane_env_count remains 1024 for flat-only runs after T302q flat-mask fix
```

- [ ] **Step 3: Update notes**

Update:

```text
notes/todo.md
notes/todo/T302r-go2-geometry-clearance-reward-plan.md
notes/log/index.md
notes/log/YYYY-MM-DD-HHMM-t302r-short-training-tensorboard.md
```

## Related Logs

- [../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md](../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md)
- [../log/2026-06-11-1551-t302r-go2-geometry-clearance-implementation.md](../log/2026-06-11-1551-t302r-go2-geometry-clearance-implementation.md)
- [../log/2026-06-11-1724-t302q-flat-small-tensorboard-readout.md](../log/2026-06-11-1724-t302q-flat-small-tensorboard-readout.md)
- [../log/2026-06-11-1513-go2-body-geometry-clearance-html-design.md](../log/2026-06-11-1513-go2-body-geometry-clearance-html-design.md)
- [../log/2026-06-11-1428-t302q-flat-small-plane-mask-fix.md](../log/2026-06-11-1428-t302q-flat-small-plane-mask-fix.md)
- [../log/2026-06-11-1420-t302q-flat-small-curriculum-clearance-root-cause-probe.md](../log/2026-06-11-1420-t302q-flat-small-curriculum-clearance-root-cause-probe.md)
- [../log/2026-06-11-1120-t302q-flat-small-tensorboard-semantic-curriculum-readout.md](../log/2026-06-11-1120-t302q-flat-small-tensorboard-semantic-curriculum-readout.md)

## Git Refs

- Last Feature Commit: `da46138`
- Last Verified Commit: `da46138`
- Current Work Ref: `working tree on top of da46138 (2026-06-11 15:19)`
- Key Files:
  - [../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html](../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)

## Next Step

- Follow [T302s](T302s-env-level-collision-curriculum-plan.md) for the immediate clearance scale wiring (`clearance_scale=1000.0`) before another long TensorBoard run; then run a larger 64/1024-env real probe to measure scaled `semantic_body_part_clearance` and collection time.

## Node Details

### T302r.1 Geometry RED tests

- why-created: TensorBoard and real probe showed sparse true contact but zero clearance reward.
- hypothesis: Nearby semantic cells should be detected by primitive radius neighborhoods even when centerline points miss them.
- evidence: [../log/2026-06-11-1420-t302q-flat-small-curriculum-clearance-root-cause-probe.md](../log/2026-06-11-1420-t302q-flat-small-curriculum-clearance-root-cause-probe.md)

### T302r.2 Query builders

- why-created: Scanner maps are discrete 2D fields; body geometry must become fixed-shape `query_xy` tensors.
- hypothesis: Cached circular offsets plus section/footprint centers provide enough coverage without dynamic loops.
- evidence: HTML design estimates about `915-975` query points per env with `K=13`.

### T302r.3 Geometry reward aggregation

- why-created: Old point reward multiplies a semantic hit by height deficit at one point; geometry reward must evaluate primitive lower surfaces over neighborhoods.
- hypothesis: Per-group GPU reductions preserve speed and provide denser pre-contact gradients.
- evidence: existing MPC terrain helper already supports batched `height_at()` and `semantic_at()`.

### T302r.4 Cfg wiring

- why-created: Training must consume geometry parameters instead of the old sparse `shank_sample_count=2` / `thigh_sample_count=2`.
- hypothesis: Reward params can change without policy/critic observation or action shape changes.
- evidence: reward is scalar term only; T302q already preserved checkpoint compatibility.

### T302r.5 Real verification

- why-created: Pure tensor tests prove coverage, but scanner pose, body ids, and IsaacLab maps must be verified in the actual runtime.
- hypothesis: after geometry query path, `semantic_body_part_clearance` should stop being identically zero in flat-small runs.
- evidence: current root-cause probe found scanner small pixels but old body sample semantic ids all ground.
