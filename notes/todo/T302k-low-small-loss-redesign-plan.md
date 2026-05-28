# T302k Low-Small Loss Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan intentionally lives under `notes/todo/` because the user asked to use todo as the implementation plan.

**Goal:** Rebuild the parametric MPC low-small obstacle losses and plane-only verification so touchdowns, target swing feet, IK/FK realized geometry, and root height behavior match the confirmed Chinese design.

**Architecture:** Keep `decode_parametric_trajectory()` as a trajectory decoder with no semantic hard repair. Put nominal command shaping in `semantic_policy.py`, put GPU semantic-circle utilities and sampled collision/consistency losses in small planner-adjacent helpers, and keep IsaacLab verification in probe/test surfaces. Debugging after implementation may tune only confirmed weights/parameters; adding any new loss or hard constraint requires user approval.

**Tech Stack:** Python, PyTorch/CUDA tensors, IsaacLab via `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`, pytest, repository notes/log workflow.

---

## Design And Scope Links

- Design HTML: [../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html](../../docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html)
- Parent branch: [T302k parametric MPC trajectory contract](T302k-parametric-mpc-trajectory-contract.md)
- Dashboard: [../todo.md](../todo.md)

## Hard Scope Guard

- Do not add decode-time hard projection.
- Do not directly snap touchdowns to legal points.
- Do not hard-separate four-foot touchdowns.
- Do not optimize and then overwrite foot/root with a hard repair.
- Do not add any loss outside this plan without asking the user first.
- After this plan starts, debugging may tune only confirmed weights and parameters.
- 第 0 条 is test/diagnostic only. It must not enter the optimizer loss dictionary.

## Current Code Reality Check

After cleaning the worktree, current HEAD has:

- `decode_parametric_trajectory(state, terrain, command, variables, horizon=...)`.
- High/large root bypass logic still inside `parametric.py`.
- No committed `ParametricTrajectoryNominal` contract.
- Old parametric sampled losses still include `parametric_low_small_crossing`, `parametric_semantic_contact`, `parametric_semantic_avoidance`, `parametric_touchdown_endpoint`, and terrain/foot/root regularizers.

Therefore Task 1 is a required precondition: reintroduce the nominal extraction contract before low-small loss work.

## File Map

### Existing files to modify

- `Go2Pvcnn/extension/batch_mpc_planner/types.py`
  - Add optional terrain metadata needed by plane-only root z loss and IsaacLab tests.
- `Go2Pvcnn/extension/batch_mpc_planner/terrain.py`
  - Preserve new terrain metadata through scanner terrain construction and subset operations.
- `Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py`
  - Own nominal command shaping and `ParametricTrajectoryNominal`.
- `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
  - Make decode consume `nominal + variables`; remove semantic command shaping from decode.
- `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py`
  - Expose FK knee/shank samples if current return type is insufficient.
- `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - Wire nominal construction, new losses, FK geometry, and loss dictionary.
- `Go2Pvcnn/extension/batch_mpc_planner/config.py`
  - Add only confirmed parameters and loss weights.
- `Go2Pvcnn/tests/test_batch_mpc_parametric.py`
  - Unit tests for nominal contract and decode no-hard-repair behavior.
- `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - Unit tests for GPU circle keepout, clearance, FK collision, consistency, and plane root z loss.
- `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
  - Extend or reuse for plane-only IsaacLab acceptance metrics.

### New files to create

- `Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py`
  - GPU semantic small-component circle approximation and semantic probe helpers.
- `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
  - New confirmed loss functions with small, testable interfaces.
- `notes/log/YYYY-MM-DD-t302k-low-small-loss-redesign-*.md`
  - One log per meaningful local or IsaacLab verification pass.

## Confirmed Parameters

第 0 条 test-only:

- `low_small_semantic_probe_half_width_m`
- `low_small_semantic_probe_count`

Loss parameters:

- `touchdown_keepout_radius_extra_m`
- `swing_foot_clearance_margin_m`
- `fk_foot_clearance_margin_m`
- `fk_knee_clearance_margin_m`
- `fk_shank_clearance_margin_m`
- `fk_root_clearance_margin_m`
- `fk_underbody_clearance_margin_m`
- `fk_shank_sample_count`
- `fk_underbody_sample_count`
- `root_z_target_height_m`

No extra parameters for trajectory consistency.

---

## Task 1: Restore Nominal Extraction Contract

**Files:**

- Modify: `Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Write failing tests for nominal ownership**

Add tests that assert:

```python
def test_build_parametric_nominal_shapes_high_large_command():
    terrain = _terrain_with_large_obstacle()
    state = _state()
    command = torch.tensor([[0.5, 0.0, 0.0]])
    nominal = build_parametric_nominal(state, terrain, command, horizon=25)
    assert nominal.command.shape == (1, 3)
    assert nominal.shape_diagnostics.command_shaped.item() is True
    assert abs(float(nominal.command[0, 1])) > 0.0
```

```python
def test_decode_parametric_consumes_nominal_without_command_shaping():
    state = _state()
    terrain = _flat_terrain()
    command = torch.tensor([[0.5, 0.0, 0.0]])
    nominal = build_parametric_nominal(state, terrain, command, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    assert decoded.root_pos.shape == (1, 25, 3)
    assert decoded.target_foot_pos.shape == (1, 25, 4, 3)
```

- [x] **Step 2: Run the tests and confirm failure**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q -k 'nominal or consumes_nominal'
```

Expected before implementation: import/signature failures for `build_parametric_nominal` or `decode_parametric_trajectory(..., nominal, ...)`.

- [x] **Step 3: Implement `ParametricTrajectoryNominal` in `semantic_policy.py`**

Add:

```python
@dataclass(frozen=True)
class ParametricTrajectoryNominal:
    command: Tensor
    forward: Tensor
    left: Tensor
    root_goal_delta: Tensor
    root_lateral_bias: Tensor
    terminal_yaw: Tensor
    terminal_rel_xy: Tensor
    shape_diagnostics: NominalCommandShapeDiagnostics
```

Move nominal construction helpers into `semantic_policy.py`:

- `_rotate_xy`
- `_canonical_body_footprint`
- `build_parametric_nominal`

`build_parametric_nominal()` must:

- call `shape_nominal_command_for_semantic_obstacles()`;
- use `command_frame_axes()` from `parametric.py`;
- build `root_goal_delta`, `root_lateral_bias`, `terminal_yaw`, `terminal_rel_xy`;
- return shaped `nominal.command`.

- [x] **Step 4: Change decode signature**

Change:

```python
decode_parametric_trajectory(state, terrain, command, variables, horizon=horizon)
```

to:

```python
decode_parametric_trajectory(state, terrain, nominal, variables, horizon=horizon)
```

Inside decode, remove command shaping/high-large search and consume:

```python
cmd = torch.as_tensor(nominal.command, dtype=dtype, device=device)
forward = torch.as_tensor(nominal.forward, dtype=dtype, device=device)
left = torch.as_tensor(nominal.left, dtype=dtype, device=device)
root_goal_delta = decoded_variable_delta + nominal.root_goal_delta
terminal_yaw = nominal.terminal_yaw
terminal_rel_xy = nominal.terminal_rel_xy
root_lat = raw_root_lat + nominal.root_lateral_bias
```

- [x] **Step 5: Build nominal once before optimizer loop**

In `_parametric_result_from_state()`:

```python
command_tensor = torch.as_tensor(command, dtype=..., device=...)
nominal = build_parametric_nominal(state, terrain, command_tensor, cfg, horizon=horizon)
planning_command = nominal.command
variables = init_parametric_variables(state, planning_command, horizon=horizon)
decoded, loss_breakdown = _optimize_parametric_variables(
    terrain,
    state,
    planning_command,
    loss_command=command,
    variables=variables,
    nominal=nominal,
    horizon=horizon,
    cfg=cfg,
)
```

In `_optimize_parametric_variables()`, pass `nominal` into every decode call.

- [x] **Step 6: Verify nominal contract locally**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or semantic'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/extension/batch_mpc_planner/planner.py
```

Expected: focused tests pass and pycompile passes.

- [x] **Step 7: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "refactor: move parametric nominal construction out of decode"
```

---

## Task 2: Add Terrain Metadata For Plane-Only Root Z

**Files:**

- Modify: `Go2Pvcnn/extension/batch_mpc_planner/types.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/terrain.py`
- Modify: caller that builds `MpcPlannerTerrain` from IsaacLab scanner data
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Write failing metadata tests**

Add:

```python
def test_mpc_terrain_preserves_is_plane_terrain_metadata():
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros(2, 5, 5),
        semantic_map=torch.zeros(2, 5, 5, dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
        is_plane_terrain=torch.tensor([True, False]),
    )
    sub = subset_mpc_terrain(terrain, torch.tensor([1]))
    assert sub.is_plane_terrain.tolist() == [False]
```

- [x] **Step 2: Run the test and confirm failure**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'is_plane_terrain'
```

Expected before implementation: constructor or attribute failure.

- [x] **Step 3: Add metadata field**

In `MpcPlannerTerrain`:

```python
is_plane_terrain: Tensor | None = None
```

In terrain construction/subset helpers:

- Accept `is_plane_terrain`.
- Preserve it in `_normal_terrain()` and `_subset_terrain()`.
- If unavailable, leave it `None` and root z plane loss returns zero.

- [x] **Step 4: Wire IsaacLab plane mask**

Find the MPC manager terrain builder that has access to IsaacLab terrain type/origin metadata. Add an automatic boolean mask:

```python
is_plane_terrain = current_subterrain_name == "flat" or current_subterrain_name == "plane"
```

Do not expose `terrain_col` or manual row/col config. Row remains difficulty and must not gate this loss.

- [x] **Step 5: Verify metadata**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'is_plane_terrain or build_mpc_terrain'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/types.py Go2Pvcnn/extension/batch_mpc_planner/terrain.py
```

- [x] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/types.py Go2Pvcnn/extension/batch_mpc_planner/terrain.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: carry plane terrain metadata into mpc terrain"
```

---

## Task 3: GPU Low-Small Circle Approximation

**Files:**

- Create: `Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Write failing GPU circle tests**

Add tests:

```python
def test_low_small_gpu_circles_split_disconnected_components():
    semantic = torch.zeros(1, 9, 9, dtype=torch.long)
    semantic[0, 2:4, 2:4] = 1
    semantic[0, 6:8, 6:8] = 1
    circles = low_small_component_circles(
        semantic,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
        max_components=4,
    )
    assert circles.center_xy.shape == (1, 4, 2)
    assert int(circles.valid[0].sum()) == 2
```

```python
def test_low_small_gpu_circles_stay_on_input_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    semantic = torch.zeros(1, 9, 9, dtype=torch.long, device=device)
    semantic[0, 3:6, 3:6] = 1
    circles = low_small_component_circles(
        semantic,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
        max_components=4,
    )
    assert circles.center_xy.device == semantic.device
    assert circles.radius.device == semantic.device
```

- [x] **Step 2: Run tests and confirm failure**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'low_small_gpu_circles'
```

Expected before implementation: import failure for `semantic_geometry`.

- [x] **Step 3: Implement GPU circle dataclass**

Create:

```python
@dataclass(frozen=True)
class LowSmallCircles:
    center_xy: Tensor
    radius: Tensor
    valid: Tensor
    truncated: Tensor
```

- [x] **Step 4: Implement fast component approximation**

Implement `low_small_component_circles(...)` with these constraints:

- Input stays on GPU.
- No per-env CPU conversion.
- Use tensor operations over `[B, H, W]`.
- Component splitting may use bounded iterative label propagation or a fixed `max_components` seed/grow approximation.
- If more components exist than `max_components`, set `truncated[b] = True`.

The public signature:

```python
def low_small_component_circles(
    semantic_map: Tensor,
    *,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    max_components: int = 8,
) -> LowSmallCircles:
    ...
```

- [x] **Step 5: Verify GPU circle helper**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'low_small_gpu_circles'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py
```

- [x] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add gpu low-small semantic circle helper"
```

---

## Task 4: Replace Touchdown Semantic Loss With Circle Keepout

**Files:**

- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Create/modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Write failing touchdown keepout tests**

Add:

```python
def test_touchdown_keepout_only_triggers_when_touchdown_on_semantic():
    terrain = _terrain_with_low_small_square()
    touchdown = torch.tensor([[[0.0, 0.0, 0.1], [0.5, 0.5, 0.0], [0.6, 0.5, 0.0], [0.7, 0.5, 0.0]]])
    loss = parametric_touchdown_keepout_loss(
        terrain,
        touchdown,
        radius_extra_m=0.05,
        max_components=8,
    )
    assert loss.item() > 0.0
```

```python
def test_touchdown_keepout_is_zero_for_nonsemantic_touchdowns():
    terrain = _terrain_with_low_small_square()
    touchdown = torch.tensor([[[0.5, 0.5, 0.0], [0.6, 0.5, 0.0], [0.7, 0.5, 0.0], [0.8, 0.5, 0.0]]])
    loss = parametric_touchdown_keepout_loss(
        terrain,
        touchdown,
        radius_extra_m=0.05,
        max_components=8,
    )
    assert loss.item() == pytest.approx(0.0)
```

- [x] **Step 2: Run tests and confirm failure**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'touchdown_keepout'
```

- [x] **Step 3: Add config fields**

Add a parametric low-small group or reuse existing loss cfg without resurrecting old semantics. Required fields:

```python
touchdown_keepout_radius_extra_m: float = 0.05
low_small_circle_max_components: int = 8
```

Keep old task-cfg override names only if needed for backward compatibility, but do not keep old `parametric_touchdown_semantic_ground` behavior.

- [x] **Step 4: Implement loss**

In `parametric_losses.py`:

```python
def parametric_touchdown_keepout_loss(
    terrain: MpcPlannerTerrain,
    touchdown_w: Tensor,
    *,
    radius_extra_m: float,
    max_components: int,
) -> Tensor:
    semantic = semantic_at(terrain, touchdown_w[..., :2])
    trigger = semantic != 0
    circles = low_small_component_circles(...)
    dist = torch.linalg.vector_norm(touchdown_xy[:, :, None, :] - circles.center_xy[:, None, :, :], dim=-1)
    deficit = torch.relu(circles.radius[:, None, :] + float(radius_extra_m) - dist)
    circle_cost = torch.where(circles.valid[:, None, :], deficit.square(), torch.zeros_like(deficit))
    per_leg = circle_cost.amax(dim=-1)
    return (per_leg * trigger.to(per_leg.dtype)).mean(dim=1)
```

- [x] **Step 5: Replace old loss keys**

In `_parametric_sampled_frame_losses()`:

- remove `parametric_low_small_crossing`;
- remove `parametric_touchdown_semantic_ground` if present;
- remove `parametric_touchdown_spacing` if present;
- add `parametric_touchdown_keepout`.

- [x] **Step 6: Verify**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'touchdown_keepout or exposes_sampled_frame_losses'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py
```

- [x] **Step 7: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: replace touchdown semantic loss with circle keepout"
```

---

## Task 5: Add Swing Target Clearance Loss

**Files:**

- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Write failing tests**

```python
def test_swing_target_clearance_penalizes_target_below_height_map():
    terrain = _flat_terrain_with_height(0.10)
    foot = torch.zeros(1, 25, 4, 3)
    foot[..., 2] = 0.105
    swing_prob = torch.ones(1, 25, 4)
    loss = parametric_swing_foot_clearance_loss(
        terrain,
        foot,
        swing_prob,
        margin_m=0.02,
    )
    assert loss.item() > 0.0
```

- [x] **Step 2: Implement loss**

```python
def parametric_swing_foot_clearance_loss(
    terrain: MpcPlannerTerrain,
    target_foot_pos: Tensor,
    swing_prob: Tensor,
    *,
    margin_m: float,
) -> Tensor:
    batch, horizon = target_foot_pos.shape[:2]
    terrain_z = height_at(terrain, target_foot_pos[..., :2].reshape(batch, horizon * 4, 2)).reshape(batch, horizon, 4)
    deficit = torch.relu(terrain_z + float(margin_m) - target_foot_pos[..., 2])
    return (deficit.square() * swing_prob.to(deficit.dtype)).mean(dim=(1, 2))
```

- [x] **Step 3: Wire config and loss key**

Add/confirm parameter:

```python
swing_foot_clearance_margin_m: float = 0.02
```

Add loss key:

```python
"parametric_swing_foot_clearance": swing_clearance
```

- [x] **Step 4: Verify**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'swing_target_clearance or sampled_frame_losses'
```

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add swing target terrain clearance loss"
```

---

## Task 6: Add FK Body/Leg Terrain Collision Loss

**Files:**

- Modify: `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [x] **Step 1: Write failing FK geometry tests**

```python
def test_fk_leg_points_exposes_knee_and_shank_samples():
    root_pos = torch.zeros(1, 25, 3)
    root_rpy = torch.zeros(1, 25, 3)
    joint = torch.zeros(1, 25, 12)
    points = fk_leg_points_from_joint_angles(root_pos, root_rpy, joint, shank_sample_count=3)
    assert points.foot_pos_world.shape == (1, 25, 4, 3)
    assert points.knee_pos_world.shape == (1, 25, 4, 3)
    assert points.shank_pos_world.shape == (1, 25, 4, 3, 3)
```

- [x] **Step 2: Write failing collision loss tests**

```python
def test_fk_body_leg_collision_penalizes_shank_below_terrain():
    terrain = _flat_terrain_with_height(0.10)
    points = _fake_fk_points_with_shank_z(0.05)
    root_pos = torch.zeros(1, 25, 3)
    loss = parametric_fk_body_leg_collision_loss(
        terrain,
        root_pos,
        points,
        margins=FkCollisionMargins(
            foot=0.015,
            knee=0.01,
            shank=0.01,
            root=0.02,
            underbody=0.015,
        ),
        underbody_sample_count=5,
    )
    assert loss.item() > 0.0
```

- [x] **Step 3: Extend FK geometry**

Ensure `fk_leg_points_from_joint_angles()` returns foot, knee, and shank sample points. If current dataclass lacks fields, add them without breaking `fk_feet_from_joint_angles()`.

- [x] **Step 4: Implement underbody samples**

Use root pose to generate fixed body-frame sample offsets under root:

```python
offsets = [
    (0.0, 0.0, -root_body_half_height),
    (front, left, -root_body_half_height),
    (front, -left, -root_body_half_height),
    (-rear, left, -root_body_half_height),
    (-rear, -left, -root_body_half_height),
]
```

Keep sample count configurable by selecting the first `fk_underbody_sample_count` deterministic offsets.

- [x] **Step 5: Add loss**

For every part:

```python
def _terrain_collision_cost(points: Tensor, margin_m: float) -> Tensor:
    terrain_z = height_at(terrain, points[..., :2].reshape(batch, -1, 2)).reshape(points.shape[:-1])
    return torch.relu(terrain_z + float(margin_m) - points[..., 2]).square()
```

Aggregate foot, knee, shank, root, and underbody costs per batch.

- [x] **Step 6: Wire planner**

Inside `_parametric_result_from_state()` after solving joint sequence, compute FK leg points and feed them into loss evaluation. If losses need FK during Adam, compute IK/FK inside `_parametric_sampled_frame_losses()` or add a second loss pass that remains differentiable through joint solution if current solver supports it. If not differentiable, record the limitation and keep this as post-optimization diagnostic only until user approves a differentiable substitute.

- [x] **Step 7: Verify**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'fk_body_leg_collision or shank'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/kinematics.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py
```

- [x] **Step 8: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/kinematics.py Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add fk body leg terrain collision loss"
```

---

## Task 7: Add Optimized-vs-FK Trajectory Consistency

**Files:**

- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Write failing tests**

```python
def test_trajectory_consistency_penalizes_absolute_and_root_relative_error():
    root = torch.zeros(1, 25, 3)
    rpy = torch.zeros(1, 25, 3)
    target = torch.zeros(1, 25, 4, 3)
    fk = target.clone()
    fk[..., 0] += 0.10
    loss = parametric_trajectory_fk_consistency_loss(root, rpy, target, fk)
    assert loss.item() > 0.0
```

- [ ] **Step 2: Implement loss**

```python
def parametric_trajectory_fk_consistency_loss(
    root_pos: Tensor,
    root_rpy: Tensor,
    target_foot_pos: Tensor,
    fk_foot_pos: Tensor,
) -> Tensor:
    abs_cost = (target_foot_pos - fk_foot_pos).square().sum(dim=-1).mean(dim=(1, 2))
    opt_rel = world_to_root_frame(root_pos, root_rpy, target_foot_pos)
    fk_rel = world_to_root_frame(root_pos, root_rpy, fk_foot_pos)
    rel_cost = (opt_rel - fk_rel).square().sum(dim=-1).mean(dim=(1, 2))
    return abs_cost + rel_cost
```

No extra parameters.

- [ ] **Step 3: Wire loss key**

Use key:

```python
"parametric_trajectory_fk_consistency"
```

- [ ] **Step 4: Verify**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'trajectory_consistency'
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add parametric fk trajectory consistency loss"
```

---

## Task 8: Add Plane Root Z Target Loss

**Files:**

- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Write failing tests**

```python
def test_plane_root_z_target_only_applies_to_plane_rows():
    root = torch.zeros(2, 25, 3)
    root[0, :, 2] = 0.40
    root[1, :, 2] = 0.40
    state_root = torch.zeros(2, 3)
    state_root[:, 2] = 0.32
    plane = torch.tensor([True, False])
    loss = parametric_plane_root_z_target_loss(
        root,
        state_root,
        plane,
        target_height_m=None,
    )
    assert loss[0].item() > 0.0
    assert loss[1].item() == pytest.approx(0.0)
```

- [ ] **Step 2: Implement loss**

```python
def parametric_plane_root_z_target_loss(
    root_pos: Tensor,
    root0: Tensor,
    is_plane_terrain: Tensor | None,
    *,
    target_height_m: float | None,
) -> Tensor:
    batch = int(root_pos.shape[0])
    if is_plane_terrain is None:
        return torch.zeros((batch,), dtype=root_pos.dtype, device=root_pos.device)
    target = root0[:, 2] if target_height_m is None else torch.full((batch,), float(target_height_m), dtype=root_pos.dtype, device=root_pos.device)
    err = (root_pos[..., 2] - target[:, None]).square().mean(dim=1)
    return torch.where(is_plane_terrain.to(device=root_pos.device), err, torch.zeros_like(err))
```

- [ ] **Step 3: Add config parameter**

```python
root_z_target_height_m: float | None = None
```

`None` means initialize target from current state root z.

- [ ] **Step 4: Wire loss key**

Use key:

```python
"parametric_plane_root_z_target"
```

- [ ] **Step 5: Verify**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'plane_root_z_target'
```

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add plane root z target loss"
```

---

## Task 9: Build Plane Low-Small FK Semantic Collision Probe

**Files:**

- Modify/Create: `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- Log: `notes/log/YYYY-MM-DD-t302k-plane-low-small-fk-collision-probe.md`

- [ ] **Step 1: Add metric helper tests**

Add pure helper tests for metric computation:

```python
def test_plane_low_small_metrics_count_semantic_collision_and_fk_error():
    metrics = compute_plane_low_small_fk_metrics(
        target_foot_pos=target,
        fk_points=fk_points,
        terrain=terrain,
        plane_mask=torch.tensor([True]),
        probe_half_width_m=0.06,
        probe_count=3,
    )
    assert "fk_semantic_collision_count" in metrics
    assert "planned_vs_fk_foot_error_crossing_leg_max_m" in metrics
```

- [ ] **Step 2: Implement test-only target trigger**

Use `target_foot_pos` 25 frames:

```python
p_probe = target_foot_xy + delta * lateral_axis
crossing_leg_mask = (semantic_at(terrain, p_probe) == 1).any(dim=(time_dim, probe_dim))
```

This remains diagnostic only.

- [ ] **Step 3: Implement FK semantic collision metrics**

Collision condition:

```python
collision = (semantic_at(terrain, point_xy) == 1) & (point_z < height_at(terrain, point_xy))
```

Compute all documented metrics:

- `plane_env_count`
- `crossing_leg_mask`
- `crossing_leg_count`
- `fk_semantic_collision_count`
- `fk_semantic_collision_rate`
- `fk_semantic_collision_by_part`
- `fk_semantic_collision_by_leg`
- `fk_semantic_min_clearance_over_semantic_m`
- `fk_semantic_first_collision_frame`
- `planned_vs_fk_foot_error_*`

- [ ] **Step 4: Add IsaacLab command matrix**

Commands must cover:

```text
forward
backward
left
right
turn_left
turn_right
diag_fl
diag_fr
mixed_turn_l
mixed_turn_r
```

Each command must run single plan and repeated replan with horizon 25.

- [ ] **Step 5: Enforce env and GPU logging**

Probe invocation must use:

```bash
CUDA_VISIBLE_DEVICES=<0|1|2|3> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python ...
```

Each JSONL/log row must record:

- `cuda_visible_devices`
- command name and velocity
- terrain type
- horizon
- replan count/cycle
- whether the row is plane/flat

- [ ] **Step 6: Run a smoke on one GPU**

Example:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --requested-n-frames 300 \
  --cycles 1 \
  --commands 'forward:0.50 0.00 0.00,left:0.00 0.50 0.00,turn_left:0.00 0.00 1.00' \
  > tmp/t302k-low-small-redesign/plane_fk_collision_smoke.jsonl 2>&1
```

Expected:

- process exits 0;
- at least one plane env row;
- rows include all required metric names.

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "test: add plane low-small fk semantic collision probe"
```

---

## Task 10: Full Verification And Notes Alignment

**Files:**

- Modify: `notes/todo.md`
- Modify: `notes/todo/T302k-parametric-mpc-trajectory-contract.md`
- Modify: this plan page
- Create: `notes/log/YYYY-MM-DD-t302k-low-small-loss-redesign-local.md`
- Create: `notes/log/YYYY-MM-DD-t302k-low-small-loss-redesign-isaaclab.md`
- Modify: `notes/log/index.md`

- [ ] **Step 1: Run local focused tests**

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or low_small or touchdown_keepout or fk_body_leg or plane_root_z'
python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py \
  Go2Pvcnn/extension/batch_mpc_planner/parametric.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py \
  Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py \
  Go2Pvcnn/extension/batch_mpc_planner/kinematics.py
```

- [ ] **Step 2: Run IsaacLab plane command matrix**

Use one or more of four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --requested-n-frames 300 \
  --cycles 2 \
  --commands 'forward:0.50 0.00 0.00,backward:-0.50 0.00 0.00,left:0.00 0.50 0.00,right:0.00 -0.50 0.00,turn_left:0.00 0.00 1.00,turn_right:0.00 0.00 -1.00,diag_fl:0.35 0.35 0.00,diag_fr:0.35 -0.35 0.00,mixed_turn_l:0.35 0.25 1.00,mixed_turn_r:0.35 -0.25 -1.00' \
  > tmp/t302k-low-small-redesign/plane_fk_collision_full_gpu0.jsonl 2>&1
```

- [ ] **Step 3: Acceptance checks**

For every command that has `crossing_leg_count > 0`:

```text
fk_semantic_collision_count == 0
fk_semantic_collision_rate == 0
fk_semantic_min_clearance_over_semantic_m >= 0
planned_vs_fk_foot_error_crossing_leg_max_m <= 0.05m preferred, <= 0.08m acceptable only if documented
```

If a command has `crossing_leg_count == 0`, mark it as not-covered instead of pass.

- [ ] **Step 4: Write logs**

Each log must include:

- purpose;
- stage;
- related todo;
- exact command;
- `CUDA_VISIBLE_DEVICES`;
- input commands;
- key metrics;
- pass/fail;
- follow-up;
- git refs.

- [ ] **Step 5: Update todo pages**

Update:

- [../todo.md](../todo.md)
- [T302k-parametric-mpc-trajectory-contract.md](T302k-parametric-mpc-trajectory-contract.md)
- this plan page checkboxes

Open leaves should show:

- T302k.18 plan page as active while implementation is in progress.
- T302k.12 as the parent reachability/collision issue being addressed.
- T302k.17 as prerequisite if nominal extraction is not yet committed.

- [ ] **Step 6: Commit notes**

```bash
git add notes/todo.md notes/todo/T302k-parametric-mpc-trajectory-contract.md notes/todo/T302k-low-small-loss-redesign-plan.md notes/log/index.md notes/log/YYYY-MM-DD-t302k-low-small-loss-redesign-*.md
git commit -m "docs: track t302k low-small loss redesign execution"
```

---

## Plan Self-Review

- [ ] Spec coverage: tasks cover nominal extraction, plane metadata, GPU circles, touchdown keepout, swing target clearance, FK geometry collision, trajectory consistency, plane root z target, IsaacLab plane-only test metrics, and notes/log alignment.
- [ ] Placeholder scan: this plan intentionally contains no `TBD` placeholders. Any implementation uncertainty is expressed as a decision point that must either be solved or escalated to the user.
- [ ] Scope guard: no task permits hard projection, snapping, hard touchdown separation, or adding unapproved losses.
- [ ] Type consistency: planned helper names are stable: `low_small_component_circles`, `parametric_touchdown_keepout_loss`, `parametric_swing_foot_clearance_loss`, `parametric_fk_body_leg_collision_loss`, `parametric_trajectory_fk_consistency_loss`, and `parametric_plane_root_z_target_loss`.
