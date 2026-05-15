# T300e MPC Continuous Swing Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 2026-05-15 MPC continuous swing-window redesign so contact timing is a single continuous swing window per leg, terrain/semantic scanner losses drive foot placement, and output-side grounding plus planner-owned foothold memory are removed.

**Architecture:** Keep the existing `planner_backend="mpc"` module, but replace its contact parameterization, nominal builder, terrain helpers, loss registry, and manager memory path in place. The optimizer decodes `swing_center/swing_width` into continuous `swing_prob/contact_prob`, samples terrain/semantic maps inside loss computation, emits touchdown positions from swing-window end events, and computes joint losses from IK on optimized root+foot targets.

**Tech Stack:** PyTorch tensor ops on GPU, IsaacLab semantic height scanner tensors, `grid_sample` terrain/semantic sampling, existing Go2 IK/FK helpers, pytest backend tests, opt-in IsaacLab runtime probes.

---

## Current State

- Design spec: [../../docs/superpowers/specs/2026-05-15-mpc-continuous-swing-window-redesign.md](../../docs/superpowers/specs/2026-05-15-mpc-continuous-swing-window-redesign.md)
- Parent branch: [T300 unified dense MPC backend](T300-unified-dense-mpc-backend.md)
- Static review conclusion: current code does not implement the redesign yet.
- Work must preserve existing MPC backend integration points and update focused tests before runtime probes.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T300e.1 | pending | P0 | Add GPU terrain/semantic query helpers for loss use | `Go2Pvcnn/extension/batch_mpc_planner/terrain.py` |
| T300e.2 | pending | P0 | Replace contact logits with continuous swing-window variables | `variables.py`, `config.py` |
| T300e.3 | pending | P0 | Rewrite nominal builder around body-frame root integration and world-frame foot swing targets | `nominal.py` |
| T300e.4 | pending | P0 | Implement terrain, semantic, touchdown, and swing-window losses | `losses/terrain_clearance.py`, `losses/gait_coupling.py`, `losses/registry.py` |
| T300e.5 | pending | P0 | Implement IK-derived joint limit, IK/FK, root-center, support-plane, and body-frame tracking losses | `losses/kinematics.py`, `losses/tracking.py`, `losses/gait_coupling.py` |
| T300e.6 | pending | P0 | Remove foothold memory and output-side grounding from planner/manager/viewer direct path | `types.py`, `planner.py`, `manager.py`, `go2_foostep_planner.py` |
| T300e.7 | pending | P0 | Update focused backend tests and add red/green coverage for new contracts | `Go2Pvcnn/tests/test_batch_mpc_backend.py` |
| T300e.8 | pending | P1 | Run py_compile, focused pytest, and IsaacLab probes; record evidence | `notes/log/`, `notes/todo.md`, `T300-unified-dense-mpc-backend.md` |

## Closed Children Archive

- None yet.

## Related Logs

- [../log/2026-05-13-1910-mpc-root-cause-minimal-verification.md](../log/2026-05-13-1910-mpc-root-cause-minimal-verification.md)
- [../log/2026-05-13-2023-mpc-ikfk-residual-headless-comparison.md](../log/2026-05-13-2023-mpc-ikfk-residual-headless-comparison.md)

## Git Refs

- Last Feature Commit: `1740fc1`
- Last Verified Commit: `pending`
- Current Work Ref: `working tree on top of 1740fc1`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/variables.py](../../Go2Pvcnn/extension/batch_mpc_planner/variables.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)

## Next Step

- Start with T300e.1 and T300e.2 in an isolated implementation session. Do not begin runtime probes until focused backend tests pass.

## File Structure

- `Go2Pvcnn/extension/batch_mpc_planner/terrain.py`: owns `MpcPlannerTerrain` GPU query helpers: `height_at`, `semantic_at`, `slope_at`, `support_at`, and terrain subsetting/building.
- `Go2Pvcnn/extension/batch_mpc_planner/config.py`: owns runtime defaults, swing-window bounds, touchdown/support loss weights, and semantic ids.
- `Go2Pvcnn/extension/batch_mpc_planner/variables.py`: owns optimizer variables and decode logic for `swing_center/swing_width -> swing_prob/contact_prob`.
- `Go2Pvcnn/extension/batch_mpc_planner/nominal.py`: owns vectorized root/foot nominal generation from current IsaacLab state and terrain height.
- `Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py`: owns body-frame command tracking.
- `Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py`: owns swing-window, diagonal pair, swing direction, root-foot center, and support-plane losses.
- `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`: owns stance/swing terrain loss, touchdown surface/semantic loss, and semantic obstacle losses.
- `Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py`: owns IK-derived joint limit and IK/FK residual losses.
- `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`: owns active loss aggregation and deletion of old terrain/contact approximation terms.
- `Go2Pvcnn/extension/batch_mpc_planner/planner.py`: owns `plan_segment`, touchdown export, status/diagnostics, and result construction.
- `Go2Pvcnn/extension/batch_mpc_planner/manager.py`: owns IsaacLab state/scanner reads, dirty scheduling, cache update, and removal of foothold memory.
- `Go2Pvcnn/extension/viz/go2_foostep_planner.py`: remove direct viewer MPC foothold memory parity code.
- `Go2Pvcnn/tests/test_batch_mpc_backend.py`: focused unit/backend tests.

---

## Node Details

### T300e.1 Terrain And Semantic Query Helpers

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/terrain.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Write failing terrain helper tests**

Add tests that construct a small batched height map and semantic map, then assert helper outputs:

```python
def test_mpc_terrain_height_semantic_slope_and_support_queries() -> None:
    height = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.2],
                [0.0, 0.2, 0.4],
            ]
        ],
        dtype=torch.float32,
    )
    semantic = torch.tensor(
        [
            [
                [0, 0, 0],
                [0, 1, 2],
                [0, 0, 0],
            ]
        ],
        dtype=torch.long,
    )
    terrain = MpcPlannerTerrain(
        height_map=height,
        semantic_map=semantic,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    query = torch.tensor([[[0.0, 0.0], [0.9, 0.0]]], dtype=torch.float32)

    sampled_h = height_at(terrain, query)
    sampled_sem = semantic_at(terrain, query)
    sampled_slope = slope_at(terrain, query, sample_step=0.25)
    support_xy, support_z, support_slope, invalid = support_at(
        terrain,
        query,
        search_radius=0.5,
        search_step=0.25,
        max_support_slope=1.0,
    )

    assert sampled_h.shape == (1, 2)
    assert sampled_sem.shape == (1, 2)
    assert sampled_slope.shape == (1, 2)
    assert support_xy.shape == (1, 2, 2)
    assert support_z.shape == (1, 2)
    assert support_slope.shape == (1, 2)
    assert invalid.shape == (1, 2)
    assert sampled_sem[0, 0].item() == 1
    assert not bool(invalid[0, 0].item())
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_terrain_height_semantic_slope_and_support_queries -q
```

Expected: FAIL because `height_at`, `semantic_at`, `slope_at`, or `support_at` are not exported yet.

- [ ] **Step 3: Implement GPU query helpers**

Add public helpers in `terrain.py` with these exact contracts:

```text
height_at(terrain, points_xy, mode="bilinear") -> Tensor shaped like points_xy without the final xy dimension
semantic_at(terrain, points_xy) -> LongTensor shaped like points_xy without the final xy dimension
slope_at(terrain, points_xy, sample_step) -> Tensor shaped like points_xy without the final xy dimension
support_at(terrain, points_xy, search_radius, search_step, max_support_slope) -> support_xy, support_z, support_slope, invalid_support
```

Implementation requirements:

- Convert world xy to normalized `grid_sample` coordinates from `world_x_range/world_y_range`.
- Use bilinear `grid_sample` for `height_at`; reshape arbitrary query dimensions into `[B, Q, 2]` internally and restore the original query shape on return.
- Use nearest-neighbor `grid_sample` for `semantic_at`; if `semantic_map is None`, return zeros on the same device with `torch.long` dtype.
- Implement `slope_at` by sampling `x +/- sample_step` and `y +/- sample_step`, then returning `sqrt(dzdx^2 + dzdy^2)`.
- Implement `support_at` with a fixed offset tensor from `torch.arange(-radius, radius + eps, step)` on the active device; score legal semantic-terrain candidates by distance plus slope penalty.
- Return the original query xy/height and `invalid_support=True` when no legal terrain candidate exists, so downstream losses remain finite.

- [ ] **Step 4: Run helper test**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_terrain_height_semantic_slope_and_support_queries -q
```

Expected: PASS.

- [ ] **Step 5: Commit terrain helper slice**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/terrain.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add mpc terrain query helpers"
```

### T300e.2 Continuous Swing Window Variables

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/variables.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Write failing decode test**

Add a test that expects decode to expose swing-window fields and one continuous swing region:

```python
def test_mpc_decode_uses_continuous_swing_window_variables() -> None:
    _, state, command, cfg = _mpc_plan_inputs(batch=2, horizon=25)
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((2, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((2, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)
    variables = init_optimization_variables(nominal, cfg.runtime)
    decoded = decode_trajectory(nominal, variables, cfg.runtime)

    assert decoded.swing_center.shape == (2, 4)
    assert decoded.swing_width.shape == (2, 4)
    assert decoded.swing_prob.shape == (2, 25, 4)
    assert decoded.contact_prob.shape == (2, 25, 4)
    assert torch.all(decoded.swing_width >= cfg.runtime.swing_window_min_width)
    assert torch.all(decoded.swing_width <= cfg.runtime.swing_window_max_width)
    assert torch.allclose(decoded.swing_prob + decoded.contact_prob, torch.ones_like(decoded.swing_prob), atol=1e-5)
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_decode_uses_continuous_swing_window_variables -q
```

Expected: FAIL because decode still uses `contact_logits`.

- [ ] **Step 3: Update config defaults**

In `config.py`, set runtime defaults and add swing-window parameters:

```python
horizon_steps: int = 25
dt: float = 0.02
step_freq: float = 2.0
duty_factor: float = 0.5
nominal_swing_height_m: float = 0.10
nominal_stride_scale: float = 0.5
nominal_yaw_stride_scale: float = 0.5
swing_window_min_width: float = 0.30
swing_window_max_width: float = 0.70
swing_window_center_scale: float = 0.20
swing_window_temperature: float = 40.0
```

Keep task config overrides where existing callers depend on them.

- [ ] **Step 4: Replace variable dataclass fields**

In `variables.py`, change `MpcOptimizationVariables`:

```python
@dataclass
class MpcOptimizationVariables:
    root_pos_residual: Tensor
    root_rpy_residual: Tensor
    foot_pos_residual: Tensor
    swing_center_raw: Tensor
    swing_width_raw: Tensor
```

Change `parameters()` to return these five tensors.

- [ ] **Step 5: Expand decoded trajectory dataclass**

In `variables.py`:

```python
@dataclass(frozen=True)
class DecodedMpcTrajectory:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    swing_center: Tensor
    swing_width: Tensor
    swing_start: Tensor
    swing_end: Tensor
    swing_prob: Tensor
    contact_prob: Tensor
```

- [ ] **Step 6: Implement window decode**

Decode raw fields using the spec mapping:

```python
center_prior = nominal["swing_center"]
width_prior = nominal["swing_width"]
center = torch.remainder(center_prior + float(runtime_cfg.swing_window_center_scale) * torch.tanh(variables.swing_center_raw), 1.0)
width_min = float(runtime_cfg.swing_window_min_width)
width_max = float(runtime_cfg.swing_window_max_width)
width = width_min + (width_max - width_min) * torch.sigmoid(variables.swing_width_raw)
```

Generate `swing_prob` with circular distance over `frame_phase = arange(T)/T`. Keep all tensors on device.

- [ ] **Step 7: Run decode test**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_decode_uses_continuous_swing_window_variables -q
```

Expected: PASS.

- [ ] **Step 8: Commit variables slice**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/variables.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: parameterize mpc contact with swing windows"
```

### T300e.3 Vectorized Nominal Builder

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/nominal.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Write failing nominal root integration test**

```python
def test_mpc_nominal_integrates_body_frame_command_with_yaw() -> None:
    terrain, state, _, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    state = MpcRobotState(
        root_pos=torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32),
        root_rpy=torch.tensor([[0.0, 0.0, 0.5 * torch.pi]], dtype=torch.float32),
        foot_pos=state.foot_pos[:1].to(torch.float32),
        joint_angles=state.joint_angles[:1].to(torch.float32),
    )
    command = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32)

    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)

    assert nominal["root_pos"].shape == (1, 25, 3)
    assert nominal["root_pos"][0, -1, 1] > nominal["root_pos"][0, -1, 0]
    torch.testing.assert_close(nominal["root_pos"][0, :, 2], torch.full((25,), 0.3))
```

- [ ] **Step 2: Write failing touchdown-frame target test**

```python
def test_mpc_nominal_touchdown_target_uses_swing_time_root_frame() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.2, 0.0, 0.5]], dtype=torch.float32)

    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)

    assert "swing_center" in nominal
    assert "swing_width" in nominal
    assert "touchdown_target_w" in nominal
    front_x = nominal["touchdown_target_w"][0, :2, 0]
    rear_x = nominal["touchdown_target_w"][0, 2:, 0]
    assert not torch.allclose(front_x.mean(), rear_x.mean())
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_nominal_integrates_body_frame_command_with_yaw Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_nominal_touchdown_target_uses_swing_time_root_frame -q
```

Expected: FAIL because current nominal uses world-frame xy increments and lacks swing-window target fields.

- [ ] **Step 4: Rewrite root nominal**

In `nominal.py`, implement vectorized root integration:

```python
frame = torch.arange(horizon, dtype=root_pos0.dtype, device=root_pos0.device)
interval = frame[:-1]
yaw_interval = root_rpy0[:, 2:3] + interval.view(1, -1) * dt * cmd[:, 2:3]
v_world = _rotate_body_xy_to_world(cmd[:, None, :2], yaw_interval)
delta_xy = v_world * dt
root_xy_tail = root_pos0[:, None, :2] + torch.cumsum(delta_xy, dim=1)
root_xy = torch.cat((root_pos0[:, None, :2], root_xy_tail), dim=1)
```

- [ ] **Step 5: Add randomized diagonal prior**

Generate GPU phase flip:

```python
if bool(runtime_cfg.randomize_replan_phase):
    flip = torch.randint(0, 2, (batch, 1, 1), device=root_pos0.device, dtype=torch.long).to(root_pos0.dtype) * 0.5
else:
    flip = torch.zeros((batch, 1, 1), device=root_pos0.device, dtype=root_pos0.dtype)
phase = (frame.view(1, horizon, 1) / float(horizon) + offsets + flip) % 1.0
```

Return `swing_center` and `swing_width` priors in `nominal`.

- [ ] **Step 6: Build world-frame foot nominal**

Use current `foot_pos0_w` as stance anchors. Use gather-style `[B,4]` swing/touchdown frame indices to compute `foot_start_body`, `target_body_xy`, and `target_world_xy` from the root pose at swing/touchdown timing. Use `height_at(terrain, target_world_xy)` for target z. Keep semantic map unused in nominal.

- [ ] **Step 7: Run nominal tests**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_nominal_integrates_body_frame_command_with_yaw Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_nominal_touchdown_target_uses_swing_time_root_frame -q
```

Expected: PASS.

- [ ] **Step 8: Commit nominal slice**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/nominal.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: rebuild mpc nominal from body-frame swing windows"
```

### T300e.4 Terrain, Touchdown, And Semantic Losses

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/optimizer.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Write failing touchdown semantic test**

```python
def test_mpc_touchdown_semantic_loss_penalizes_small_and_large_obstacles() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 2] = 1
    semantic[:, 2, 3] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1, 1), world_y_range=(-1, 1))
    touchdown_xy = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [0.0, 0.5]]], dtype=torch.float32)
    touchdown_z = torch.zeros((1, 4), dtype=torch.float32)

    loss = touchdown_semantic_loss(terrain, touchdown_xy, touchdown_z, small_weight=10.0, large_weight=50.0)

    assert loss.shape == (1,)
    assert float(loss[0]) > 0.0
```

- [ ] **Step 2: Write failing stance/swing terrain test**

```python
def test_mpc_stance_and_swing_terrain_losses_use_height_map() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1, 1),
        world_y_range=(-1, 1),
    )
    foot = torch.zeros((1, 3, 4, 3), dtype=torch.float32)
    contact = torch.ones((1, 3, 4), dtype=torch.float32)
    swing = torch.ones_like(contact) - contact

    stance_loss = stance_ground_loss(terrain, foot, contact)
    swing_loss = swing_clearance_terrain_loss(terrain, foot, swing, min_clearance_m=0.05)

    assert stance_loss.shape == (1,)
    assert swing_loss.shape == (1,)
    assert torch.isfinite(stance_loss).all()
    assert torch.isfinite(swing_loss).all()
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -k "touchdown_semantic_loss or stance_and_swing_terrain_losses" -q
```

Expected: FAIL because new loss functions are missing.

- [ ] **Step 4: Implement loss functions**

In `terrain_clearance.py`, replace old approximation losses with the following formulas:

```text
stance_ground_loss:
  terrain_z = height_at(terrain, foot_pos[..., :2])
  return mean_over_time_legs(contact_prob * smooth_l1(foot_pos[..., 2] - terrain_z))

swing_clearance_terrain_loss:
  terrain_z = height_at(terrain, foot_pos[..., :2])
  clearance_error = relu(terrain_z + min_clearance_m - foot_pos[..., 2])
  return mean_over_time_legs(swing_prob * clearance_error^2)

touchdown_surface_loss:
  terrain_z = height_at(terrain, touchdown_w[..., :2])
  slope = slope_at(terrain, touchdown_w[..., :2], sample_step=cfg.touchdown_slope_sample_step)
  support_xy, support_z, support_slope, invalid = support_at(
    terrain,
    touchdown_w[..., :2],
    search_radius=cfg.support_search_radius_m,
    search_step=cfg.support_search_step_m,
    max_support_slope=cfg.max_support_slope,
  )
  return ground + slope + support_distance + support_height + support_slope + invalid_penalty

touchdown_semantic_loss:
  semantic = semantic_at(terrain, touchdown_xy)
  return mean_legs(small_weight * (semantic == 1) + large_weight * (semantic == 2))

semantic_obstacle_loss:
  sample foot contact, swing path, and root footprint stencil through `semantic_at(terrain, query_xy)` calls
  return separate foot/body obstacle penalties for registry breakdown
```

- [ ] **Step 5: Update registry signature**

Change:

```python
compute_total_loss(decoded, nominal, joint_angles, command, cfg)
```

to:

```python
compute_total_loss(decoded, nominal, state, command, terrain, cfg)
```

Update `optimizer.py` to pass `state` and `terrain`.

- [ ] **Step 6: Remove old active losses from registry**

Delete active use of:

```text
_command_adaptive_weights
contact_schedule_tracking_loss
old swing_clearance_loss
old terrain_clearance_loss
old obstacle_margin_loss
touchdown_support = ||decoded.foot_pos - nominal["foot_pos"]||
old swing_stride_loss
```

- [ ] **Step 7: Run terrain/semantic loss tests**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -k "touchdown_semantic_loss or stance_and_swing_terrain_losses" -q
```

Expected: PASS.

- [ ] **Step 8: Commit terrain loss slice**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/extension/batch_mpc_planner/optimizer.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add scanner-driven mpc terrain losses"
```

### T300e.5 Kinematic, Support Geometry, And Body-Frame Tracking Losses

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Write failing body-frame tracking test**

```python
def test_mpc_tracking_loss_uses_body_frame_velocity() -> None:
    root_pos = torch.zeros((1, 2, 3), dtype=torch.float32)
    root_rpy = torch.zeros((1, 2, 3), dtype=torch.float32)
    root_rpy[:, :, 2] = 0.5 * torch.pi
    root_pos[:, 1, 1] = 0.02
    command = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    loss = command_tracking_loss(root_pos, root_rpy, command, dt=0.02)

    assert float(loss[0]) < 1e-4
```

- [ ] **Step 2: Write failing root support geometry test**

```python
def test_mpc_root_support_geometry_losses_are_finite() -> None:
    root = torch.zeros((1, 5, 3), dtype=torch.float32)
    rpy = torch.zeros((1, 5, 3), dtype=torch.float32)
    foot = torch.tensor([[[[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]]]], dtype=torch.float32)
    foot = foot.expand(1, 5, 4, 3).contiguous()
    contact = torch.ones((1, 5, 4), dtype=torch.float32)

    center = root_foot_center_loss(root, foot)
    plane = support_plane_roll_pitch_loss(rpy, foot, contact, swing_weight=0.2)

    assert center.shape == (1,)
    assert plane.shape == (1,)
    assert torch.isfinite(center).all()
    assert torch.isfinite(plane).all()
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -k "body_frame_velocity or root_support_geometry" -q
```

Expected: FAIL because losses are old or missing.

- [ ] **Step 4: Implement body-frame tracking**

In `tracking.py`, compute per-step velocity:

```python
dxy_w = root_pos[:, 1:, :2] - root_pos[:, :-1, :2]
yaw = root_rpy[:, :-1, 2]
dxy_b = rotate_world_to_body_xy(dxy_w, yaw)
vel_b = dxy_b / float(dt)
yaw_rate = (root_rpy[:, 1:, 2] - root_rpy[:, :-1, 2]) / float(dt)
```

Compare `vel_b` to `command[:, :2]` and `yaw_rate` to `command[:, 2]`.

- [ ] **Step 5: Implement support geometry losses**

In `gait_coupling.py`, add these losses with explicit tensor behavior:

```text
root_foot_center_loss:
  foot_center_xy = foot_pos[..., :2].mean(dim=2)
  return mean_time(||root_pos[..., :2] - foot_center_xy||_2)

support_plane_roll_pitch_loss:
  weights = swing_weight + (1 - swing_weight) * contact_prob
  fit z = ax + by + c by weighted least squares per [B,T]
  normal = normalize([-a, -b, 1])
  estimated_roll = atan2(normal_y, normal_z)
  estimated_pitch = atan2(-normal_x, sqrt(normal_y^2 + normal_z^2))
  return mean_time(||root_rpy[..., :2] - estimated_rp||_2)

swing_window_loss:
  combine width bounds, diagonal-pair center agreement, half-cycle group separation, and soft phase-prior terms

swing_direction_loss:
  sample swing start and touchdown foot/root poses by window phase
  compare body-frame start-to-end foot displacement with step_gain/yaw_gain expected displacement
```

Plane fitting may use weighted least squares over four foot points per frame; add a small diagonal regularizer before solving the normal equations.

- [ ] **Step 6: Implement IK-derived joint limit loss**

In `kinematics.py`, expose a loss-side IK path:

```text
solve_ik_for_loss(root_pos, root_rpy, foot_pos):
  flatten [B,T] to [B*T], transform world feet into each leg's hip/body frame, reuse existing Go2 inverse-kinematics math, then reshape to [B,T,12]

joint_limit_loss_from_root_foot(root_pos, root_rpy, foot_pos, joint_limit_margin_rad):
  ik_joint = solve_ik_for_loss(root_pos, root_rpy, foot_pos)
  lower_violation = relu(joint_lower + margin - ik_joint)
  upper_violation = relu(ik_joint - joint_upper + margin)
  return mean_time_joints(lower_violation^2 + upper_violation^2)
```

Use existing IK math and per-joint limits. Do not use repeated `state.joint_angles` as the loss source.

- [ ] **Step 7: Wire registry**

In `registry.py`, add breakdown terms:

```text
swing_window
diagonal_pair
phase_prior
swing_direction
ik_joint_limit
ik_fk_residual
root_foot_center
support_plane_rp
tracking
```

- [ ] **Step 8: Run focused loss tests**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -k "body_frame_velocity or root_support_geometry or decode_uses_continuous" -q
```

Expected: PASS.

- [ ] **Step 9: Commit support/kinematic loss slice**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add mpc swing and support geometry losses"
```

### T300e.6 Planner And Manager Contract Cleanup

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/types.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/manager.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Write failing no-memory/no-grounding test**

```python
def test_mpc_backend_has_no_foothold_memory_or_output_grounding_symbols() -> None:
    root = GO2PVCNN_ROOT / "extension" / "batch_mpc_planner"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    forbidden = [
        "MpcFootholdMemory",
        "_ground_contact_feet_to_terrain",
        "_initialize_foothold_memory",
        "_foothold_memory_for",
        "_update_foothold_memory",
        "_stance_anchor_w",
        "_running_foot_rel_body",
        "_yaw_foot_rel_body",
    ]
    for token in forbidden:
        assert token not in source, token
```

- [ ] **Step 2: Write failing result output test**

```python
def test_mpc_plan_segment_outputs_optimized_feet_without_post_grounding() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.foot_pos.shape == (1, 25, 4, 3)
    assert result.joint_angles.shape == (1, 25, 12)
    assert result.touchdown_seq.shape[0:2] == (1, 4)
    assert result.planned_touchdown_w.shape == (1, 25, 4, 3)
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -k "no_foothold_memory or outputs_optimized_feet" -q
```

Expected: FAIL because memory/grounding symbols still exist.

- [ ] **Step 4: Remove `MpcFootholdMemory` type and imports**

Delete the dataclass from `types.py` and remove all imports in `planner.py`, `manager.py`, `nominal.py`, tests, and viewer code.

- [ ] **Step 5: Remove manager memory fields and methods**

Delete:

```text
_stance_anchor_w
_running_foot_rel_body
_yaw_foot_rel_body
_prev_contact_state
_stable_contact_steps
_prev_yaw_dominance
_yaw_entry_steps
_initialize_foothold_memory
_foothold_memory_for
_update_foothold_memory
```

Ensure replan uses `sub_states.foot_pos` directly.

- [ ] **Step 6: Remove output-side grounding**

Delete `_ground_contact_feet_to_terrain`. In `plan_segment`, call IK and touchdown export on `decoded.foot_pos` directly:

```python
contact_state = decoded.contact_prob > float(cfg.runtime.contact_threshold)
foot_pos = decoded.foot_pos
joint_seq = solve_joint_angles_from_trajectory(decoded.root_pos, decoded.root_rpy, foot_pos)
```

- [ ] **Step 7: Update touchdown export**

Replace transition-frame extraction with swing-window touchdown interpolation:

```python
touchdown_w = sample_touchdown_positions(decoded.foot_pos, decoded.swing_center, decoded.swing_width)
touchdown_seq = touchdown_w.unsqueeze(2).expand(-1, -1, cfg.runtime.touchdown_event_cap, -1).contiguous()
planned_touchdown_w = touchdown_w.unsqueeze(1).expand(-1, horizon, -1, -1).contiguous()
```

- [ ] **Step 8: Run no-memory tests**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -k "no_foothold_memory or outputs_optimized_feet" -q
```

Expected: PASS.

- [ ] **Step 9: Commit planner/manager cleanup slice**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/types.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "refactor: remove mpc foothold memory and post grounding"
```

### T300e.7 Focused Backend Regression Suite

**Files:**
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- Modify: any tests broken by removed old fields

- [ ] **Step 1: Delete obsolete tests**

Remove tests that explicitly verify deleted behavior:

```text
test_yaw_foothold_memory_uses_fixed_body_seed_until_reentry
test_nominal_yaw_body_anchor_uses_fixed_seed_with_displacement_cap
```

- [ ] **Step 2: Add active registry cleanup test**

```python
def test_mpc_loss_registry_no_longer_uses_deleted_terms() -> None:
    source = (GO2PVCNN_ROOT / "extension" / "batch_mpc_planner" / "losses" / "registry.py").read_text(encoding="utf-8")
    forbidden = [
        "_command_adaptive_weights",
        "contact_schedule_tracking_loss",
        "touchdown_support",
        "obstacle_margin_loss",
        "swing_stride_loss",
    ]
    for token in forbidden:
        assert token not in source, token
```

- [ ] **Step 3: Add loss breakdown contract test**

```python
def test_mpc_loss_breakdown_exposes_continuous_window_terms() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.diagnostics.enabled = True
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    expected = {
        "swing_window",
        "diagonal_pair",
        "stance_ground",
        "swing_clearance_terrain",
        "touchdown_surface",
        "touchdown_semantic",
        "swing_direction",
        "ik_joint_limit",
        "ik_fk_residual",
        "root_foot_center",
        "support_plane_rp",
    }
    assert expected.issubset(result.loss_breakdown)
```

- [ ] **Step 4: Run focused backend suite**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit test cleanup slice**

```bash
git add Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "test: update mpc backend contracts for swing windows"
```

### T300e.8 Verification And Notes Evidence

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T300-unified-dense-mpc-backend.md`
- Create: `notes/log/YYYY-MM-DD-HHMM-mpc-continuous-swing-window-implementation.md`

- [ ] **Step 1: Run py_compile**

Run:

```bash
python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/terrain.py \
  Go2Pvcnn/extension/batch_mpc_planner/nominal.py \
  Go2Pvcnn/extension/batch_mpc_planner/variables.py \
  Go2Pvcnn/extension/batch_mpc_planner/optimizer.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/batch_mpc_planner/manager.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py
```

Expected: exit code `0`.

- [ ] **Step 2: Run focused pytest**

Run:

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run root-cause probe small matrix**

Run after activating `env_isaacsim`:

```bash
MPC_TEST_DEVICE=cuda:2 \
MPC_ROOT_CAUSE_CYCLES=4 \
MPC_ROOT_CAUSE_SEQUENCES='linear:forward,backward;yaw:yaw_left,yaw_right;mixed:forward_yaw_left,lateral_left_yaw_right' \
MPC_ROOT_CAUSE_OUTPUT=/tmp/mpc_continuous_window_root_cause.jsonl \
timeout 900s python Go2Pvcnn/tests/mpc_root_cause_probe.py
```

Expected: command exits `0`; output contains runtime summaries without exceptions.

- [ ] **Step 4: Run yaw gait probe**

Run after activating `env_isaacsim`:

```bash
MPC_TEST_DEVICE=cuda:2 \
MPC_YAW_GAIT_CYCLES=8 \
MPC_YAW_GAIT_OUTPUT=/tmp/mpc_continuous_window_yaw_probe.jsonl \
timeout 900s python Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py
```

Expected: command exits `0`; planned touchdown semantic collision count is `0`; yaw actual air and IK/FK error are reported for comparison.

- [ ] **Step 5: Update notes**

Create a log file with:

```text
purpose
stage
related todo: T300e
commands
input conditions
key metrics
result
conclusion
follow-up
git refs
```

Update:

```text
notes/todo.md
notes/todo/T300-unified-dense-mpc-backend.md
notes/log/index.md
```

- [ ] **Step 6: Commit verification notes**

```bash
git add notes/todo.md notes/todo/T300-unified-dense-mpc-backend.md notes/log/index.md notes/log/YYYY-MM-DD-HHMM-mpc-continuous-swing-window-implementation.md
git commit -m "docs: record mpc continuous swing window verification"
```

## Plan Self-Review

- Spec coverage: Tasks T300e.1-T300e.8 cover terrain helpers, swing-window variables, vectorized nominal, terrain/semantic/touchdown losses, IK/support geometry losses, memory/grounding removal, focused tests, and runtime evidence.
- Red-flag scan: no open-ended task steps are intended; each child has files, commands, formulas, and expected outcomes.
- Type consistency: `height_at/semantic_at/slope_at/support_at`, `swing_center/swing_width`, `swing_prob/contact_prob`, `touchdown_w`, and registry breakdown names are used consistently across tasks.
