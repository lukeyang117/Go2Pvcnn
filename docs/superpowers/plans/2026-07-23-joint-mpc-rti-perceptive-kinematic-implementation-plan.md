# Joint MPC RTI Perceptive Kinematic Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current soft-semantic and repair-heavy `joint_mpc_rti` implementation with the approved pure-kinematic H30 planner: current-map touchdown selection, convex landing regions, continuous warm-retargeted swing/stance nominal, one LQ/QP, five-alpha hard-safe line search, complete diagnostics, and flat/small/large behavior plus `1024 x 1000 < 5s` acceptance.

**Architecture:** Optimize only `Z_nodes=[z0,...,z30]`, where `z=[p_B,theta_B,q] in R^18`; export exactly `Z_future=[z1,...,z30]`. Every 20 ms refresh rebuilds a fixed-channel 2.5D perceptive field from the same scanner frame, shifts/rebases the last accepted trajectory, reselects safe touchdown targets and convex regions, retargets the relevant swing/stance segments, linearizes once, solves one fixed H30/32 trajectory QP, and evaluates five candidates with one packed nonlinear cost and swept world-geometry safety pass. Cold construction occurs only once after explicit reset; runtime failures stop publication while preserving the last finite cache.

**Tech Stack:** Python 3.10, PyTorch 2.7, `torch.compile`, CUDA Graph, Triton fixed-size linear algebra, batched Go2 analytic FK/IK, 2.5D `max_pool2d` fields, SQP/RTI with GGN LQ approximation, H30/32 associative scan, Isaac Lab viewer/runtime, pytest.

**Approved Design:** `docs/superpowers/specs/2026-07-22-joint-mpc-rti-perceptive-kinematic-final-design.html`

**Supersedes For Execution:** `docs/superpowers/plans/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-implementation-plan.md`. That file remains historical evidence, but its seven soft-loss objective, Gaussian occupancy field, `phase/11` swing endpoint, and x1-only support repair are not implementation sources.

## Execution Status (2026-07-23)

- Tasks 1-9 and 11 are implemented and committed through `7f15334`.
- Task 10 is implemented: the production path no longer calls the dense reference, B=`1/8/40` direction parity is within `2e-5`, and CUDA Graph capture/replay passes through 36D separator factors padded H30->H32 with five explicit combine levels and two active-mask refinements. Performance is not closed: a B=40 graph smoke averages `122.67 ms/refresh`, so Task 19 must optimize the factor representation/kernels before the formal `1024 x 1000 <5s` gate.
- Task 12 planner/warm-only/CUDA Graph routing is implemented in the current worktree and passes the focused Task 1-12 regression (`128 passed`). Its commit and later Task 15 legacy deletion remain open.
- Tasks 13-16 are implemented and GREEN. Task 16 records real same-refresh KKT metrics and closes all 19 formal flat cells; focused verification is `115 passed`, the full package is `284 passed`, and formal CUDA v13 has `gate.passed=true` with primal/dual KKT maxima `5.133e-5/9.014e-5`. Tasks 17-19 remain open; no small/large/performance acceptance is claimed yet.
- 2026-07-24 Task 17 controlled diagnosis: small crossing now runs every flat/common metric, including current-world-map stance grounding and forbidden semantics. Selector/nominal focused regression is `84 passed`, but refresh 59 remains invalid on preview liftoff edge 25: discrete nodes are collision-free while the published joint-linear swept edge penetrates foot `13.92mm` and calf `9.92mm`. Selector validates a continuous IK curve, so Task 17 cannot close through candidate or weight tuning; the next owner-level step must unify selector and published discrete liftoff/swept geometry without weakening any gate.

---

## Non-Negotiable Contracts

```text
dt = 0.02 s
prediction intervals = future frames = 30
internal state nodes = 31
prediction duration = 0.60 s
gait period = 24 intervals = 0.48 s
swing = 12 intervals; stance = 12 intervals
leg order = [FL, FR, RL, RR]
leg phase offsets = [0, 12, 12, 0]
touchdown edge = phase 11 -> phase 12
swing endpoint tau = r / 12 at phase-12 touchdown node
optimization state = [p_B, roll/pitch/yaw, 12 joint angles]
SQP linearizations per refresh = 1
LQ/QP subproblems per refresh = 1
line-search alpha = [1, 0.5, 0.25, 0.125, 0]
map age = 0 refreshes
cold builds per reset lifecycle = exactly 1
formal performance = 1024 environments x 1000 complete warm refreshes < 5.0 s
```

The implementation must not add contact force, torque, momentum, inertia, rigid-body dynamics, friction cones, contact sensors, post-publication repair, automatic cold fallback, a second QP, or a second SQP iteration. Actual collision evidence is root/joint readback passed through FK and queried against the current world-aligned semantic/elevation map.

## File Ownership Map

```text
Go2Pvcnn/extension/joint_mpc_rti/
|- config.py                         # frozen dimensions and grouped tuning parameters
|- types.py                          # frame metadata, field/plan/trajectory/diagnostic tensor contracts
|- planner.py                        # ten-stage one-refresh orchestration only
|- model/gait_schedule.py            # node/edge 24/12/12 trot and event timing
|- model/go2_kinematics.py           # FK, Jacobians, sphere/capsule/OBB geometry
|- model/analytic_ik.py              # fixed-branch batched IK and reachability
|- model/perceptive_plan.py          # event preview, 25-candidate selector, latch, region
|- model/nominal.py                  # cold once, shift/rebase, retarget, stance/swing IK
|- terrain/perceptive_field.py       # current 2.5D channels and max-pool geometry layers
|- terrain/query.py                  # packed world/grid queries
|- terrain/swept_safety.py           # node and interval hard-safe geometry queries
|- solver/lq_problem.py              # eight residual families and fixed constraint rows
|- solver/trajectory_qp.py           # block QP, active masks, dense reference
|- solver/trajectory_scan.py         # H30 padded to H32 associative solve
|- solver/line_search.py              # five packed candidates and exact filters
|- solver/sqp_rti.py                  # one linearize/solve/search call
|- diagnostics/metrics.py            # fixed-shape per-refresh diagnostics
|- diagnostics/profiler.py           # eight stage timings
|- runtime/manager.py                 # current frame, reset, cache, public reference
`- runtime/cuda_graph.py              # preallocated steady-state replay

Go2Pvcnn/tests/joint_mpc_rti/
|- test_final_contract.py
|- test_fixed_gait_schedule.py
|- test_perceptive_field.py
|- test_swept_safety.py
|- test_perceptive_plan.py
|- test_nominal.py
|- test_lq_problem.py
|- test_trajectory_qp.py
|- test_trajectory_scan.py
|- test_line_search_v2.py
|- test_rti_pipeline.py
|- test_rolling_runtime.py
|- test_diagnostics.py
|- joint_metrics.py
|- acceptance_thresholds.py
|- run_joint_acceptance.py
|- test_flat_acceptance.py
|- test_small_acceptance.py
|- test_large_acceptance.py
`- joint_mpc_rti_full_refresh_probe.py
```

`integration/isaaclab_adapter.py`, `integration/reference_adapter.py`, `integration/viewer_adapter.py`, the factory, and the external reference-cache ABI remain the boundary. Their internals may change only to carry the new 30-future-frame output, same-refresh metadata, and diagnostics.

---

### Task 1: Restore Test Collection And Freeze The Final Public Contract

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_final_contract.py`
- Rewrite conflict sections: `Go2Pvcnn/tests/joint_mpc_rti/test_contracts.py`
- Rewrite conflict sections: `Go2Pvcnn/tests/joint_mpc_rti/acceptance_thresholds.py`
- Rewrite conflict sections: `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`
- Rewrite conflict sections: `Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_perf_probe.py`
- Rewrite conflict sections: `Go2Pvcnn/tests/joint_mpc_rti/test_performance.py`
- Rewrite conflict sections: `Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/types.py`

- [ ] **Step 1: Preserve the current collection failure as RED evidence**

Run:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_contracts.py --collect-only -q
```

Expected: collection fails on committed merge markers. Record the exact files returned by:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' \
  Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti
```

- [ ] **Step 2: Remove only conflict syntax and add final-contract RED tests**

Select the five-alpha, H30, pure-kinematic branch in each conflict and preserve any non-duplicate metric coverage. Add:

```python
def test_final_contract_is_h30_current_map_and_one_rti():
    cfg = JointMpcRtiCfg()
    assert cfg.runtime.horizon_steps == 30
    assert cfg.runtime.future_frames == 30
    assert cfg.runtime.state_nodes == 31
    assert cfg.runtime.dt == 0.02
    assert cfg.runtime.max_field_age_steps == 0
    assert cfg.runtime.sqp_iterations_per_step == 1
    assert cfg.gait.period_steps == 24
    assert cfg.gait.swing_steps == cfg.gait.stance_steps == 12
    assert cfg.solver.line_search_alphas == (1.0, 0.5, 0.25, 0.125, 0.0)


def test_trajectory_contract_separates_internal_nodes_from_future_frames():
    fields = JointMpcRtiTrajectory.__dataclass_fields__
    assert "state_nodes" in fields
    assert "future_state" in fields
```

Run the new file. Expected: FAIL because `future_frames`, `state_nodes`, current-map age zero, and split trajectory fields are missing.

- [ ] **Step 3: Implement frozen config and tensor dataclasses**

Add computed constants and split output without adding dynamics:

```python
@dataclass
class JointMpcRtiRuntimeCfg:
    horizon_steps: int = 30
    future_frames: int = 30
    state_nodes: int = 31
    dt: float = 0.02
    sqp_iterations_per_step: int = 1
    max_field_age_steps: int = 0


@dataclass(frozen=True)
class JointMpcRtiTrajectory:
    state_nodes: Tensor       # [B,31,18], includes measured z0
    future_state: Tensor      # [B,30,18], exactly z1...z30
    derived_velocity: Tensor  # [B,30,18]
    foot_pos_w: Tensor        # [B,31,4,3]
    contact_state: Tensor     # [B,31,4]
    valid: Tensor
    publish: Tensor
    stop: Tensor
    status: Tensor
    line_search_alpha: Tensor
```

Keep temporary read-only `state` compatibility only inside the adapter migration in Task 12; production code must use the explicit names before cleanup.

- [ ] **Step 4: Run GREEN collection and contract tests**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_final_contract.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_contracts.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_performance.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py -q
```

Expected: collection succeeds and tests affected by the final contract pass. The conflict-marker search returns no output.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/config.py \
  Go2Pvcnn/extension/joint_mpc_rti/types.py \
  Go2Pvcnn/tests/joint_mpc_rti
git commit -m "test: restore final joint mpc contract baseline"
```

---

### Task 2: Implement Exact H30 Node/Edge Gait Semantics

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py`

- [ ] **Step 1: Write RED boundary tests**

```python
def test_phase_11_to_12_is_touchdown_edge_and_phase_12_is_stance_node():
    s = fixed_trot_schedule(torch.tensor([0]), horizon_steps=30)
    assert s.swing_edge[0, 11, 0]
    assert s.touchdown_edge[0, 11, 0]
    assert s.stance_node[0, 12, 0]
    assert s.swing_tau_node[0, 11, 0] == pytest.approx(11.0 / 12.0)
    assert s.swing_tau_node[0, 12, 0] == pytest.approx(1.0)


def test_h30_contains_30_edges_and_31_nodes():
    s = fixed_trot_schedule(torch.arange(24), horizon_steps=30)
    assert s.phase_node.shape == (24, 31, 4)
    assert s.phase_edge.shape == (24, 30, 4)
    assert s.swing_edge.shape == (24, 30, 4)
```

Expected RED: current code exposes one node mask and reaches `tau=1` at phase 11 via `/11`.

- [ ] **Step 2: Implement one broadcasted schedule**

```python
phase_node = (phase0[:, None, None] + node[None, :, None] + offsets) % 24
phase_edge = phase_node[:, :-1]
swing_edge = phase_edge < 12
stance_edge = ~swing_edge
stance_node = (phase_node >= 12) & (phase_node < 24)
touchdown_edge = (phase_node[:, :-1] == 11) & (phase_node[:, 1:] == 12)
swing_tau_node = torch.where(
    phase_node <= 12,
    phase_node.to(dtype).div(12.0),
    torch.zeros_like(phase_node, dtype=dtype),
)
```

Offsets must come from `tensor_constants.constant_like`; no tensor allocation is allowed inside CUDA Graph capture.

- [ ] **Step 3: Add all-phase event tests**

For all 24 start phases and four legs, assert 12 swing edges, 12 stance edges, diagonal complementarity, one touchdown per period, next touchdown distance in `[0,24]`, and no phase advancement inside alpha evaluation.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py
git commit -m "fix: define exact h30 trot edge semantics"
```

---

### Task 3: Build The Current-Refresh Perceptive Field

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/terrain/perceptive_field.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/query.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/field_cache.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/integration/field_sync.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/types.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_field.py`

- [ ] **Step 1: Write RED tests for channels, max pooling, and freshness**

```python
def test_field_uses_max_pool_layers_and_preserves_small_vs_large_semantics():
    field = build_perceptive_field(height, semantic, valid, frame, cfg)
    assert field.height_w.shape == (B, 151, 151)
    assert field.inflated_height_w.shape == (B, N_GEOMETRY, 151, 151)
    assert field.landing_safe.shape == (B, 151, 151)
    assert not field.landing_safe[field.small_mask].any()
    assert not field.landing_safe[field.large_mask].any()
    assert field.small_mask.any() and field.large_mask.any()


def test_stale_or_mismatched_frame_is_invalid_not_previous_field_fallback():
    assert not validate_frame_freshness(field_refresh_id=7, state_refresh_id=8).all()
    assert validate_frame_freshness(field_refresh_id=8, state_refresh_id=8).all()
```

Expected RED: current field uses Gaussian occupancy/EDT and permits `max_field_age_steps=2`.

- [ ] **Step 2: Implement fixed-channel 2.5D field construction**

Use `max_pool2d` for each geometry radius/margin and landing footprint:

```python
def inflate(mask: Tensor, radius_m: float, resolution: float) -> Tensor:
    n = math.ceil(radius_m / resolution)
    return F.max_pool2d(mask[:, None].float(), 2 * n + 1, 1, n)[:, 0] > 0


field = JointMpcPerceptiveField(
    height_w=height,
    semantic_id=semantic,
    valid_mask=valid,
    small_mask=small,
    large_mask=large,
    unknown_mask=~valid,
    inflated_height_w=radius_heights,
    landing_safe=valid & ~inflate(small | large | ~valid, landing_radius, resolution),
    slope_xy=slope_xy,
    roughness=roughness,
    origin_w=origin,
    yaw_w=yaw,
    refresh_id=refresh_id,
    timestamp=timestamp,
)
```

No temporal safety result may be cached. Preallocated storage is allowed only when every channel and metadata row is overwritten for the current refresh.

- [ ] **Step 3: Extend packed world queries**

Query height, semantic, validity, all geometry radius heights, landing mask, slope, roughness, and boundary distances in one packed call. Out-of-map and unknown return `valid=False` and are unsafe.

- [ ] **Step 4: Run GREEN and scanner parity**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_field.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/terrain \
  Go2Pvcnn/extension/joint_mpc_rti/integration/field_sync.py \
  Go2Pvcnn/extension/joint_mpc_rti/types.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_field.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py
git commit -m "feat: build current-refresh perceptive field"
```

---

### Task 4: Implement World-Geometry Swept Safety

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/terrain/swept_safety.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_swept_safety.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py`

- [ ] **Step 1: Write RED tests for geometry and tunneling**

```python
def test_interval_collision_detects_safe_endpoints_with_unsafe_middle():
    z0, z1, field = endpoint_safe_but_segment_crosses_cuboid()
    endpoint = evaluate_nodes(torch.stack((z0, z1), dim=1), field, cfg)
    swept = evaluate_swept_intervals(torch.stack((z0, z1), dim=1), field, cfg)
    assert endpoint.safe.all()
    assert not swept.safe.all()


@pytest.mark.parametrize("part", ("foot", "knee", "calf", "thigh", "base"))
def test_each_part_has_independent_clearance_and_reject_bit(part):
    result = evaluate_swept_intervals(part_collision_case(part), field, cfg)
    assert result.collision_by_part[part].any()
```

- [ ] **Step 2: Expose fixed FK geometry primitives**

Return foot sphere/sole corners, knee sphere, calf and thigh capsule endpoints/radii, and base OBB center/rotation/half extents. All tensors remain in world coordinates until `query_world` projection.

- [ ] **Step 3: Implement fixed subdivision and conservative checks**

```python
u = SWEEP_FRACTIONS.view(1, 1, S, 1)
z_swept = z[:, :-1, None] + u * (z[:, 1:, None] - z[:, :-1, None])
geometry = go2_collision_geometry(z_swept)
clearance = packed_geometry_clearance(geometry, current_field)
safe = clearance.valid.all(-1) & (clearance.margin >= 0).all(-1)
```

Capsules and OBB use fixed surface samples plus a conservative interval bound. Complete sole overlap determines forbidden touchdown/stance semantics.

- [ ] **Step 4: Run GREEN and finite-difference checks**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_swept_safety.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py \
  Go2Pvcnn/extension/joint_mpc_rti/terrain/swept_safety.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_swept_safety.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py
git commit -m "feat: add swept whole-robot safety detector"
```

---

### Task 5: Implement Per-Leg Touchdown Event Preview And Selector

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/types.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py`

- [ ] **Step 1: Write RED selector tests**

```python
def test_selector_uses_25_candidates_per_leg_without_four_leg_product():
    plan = select_touchdowns(measured, command, schedule, warm, field, cfg)
    assert plan.candidate_w.shape == (B, 4, 25, 3)
    assert plan.safe_mask.shape == (B, 4, 25)
    assert plan.selected_index.shape == (B, 4)


def test_small_cross_candidate_must_land_after_obstacle_and_be_sweep_safe():
    plan = select_touchdowns(measured, forward, schedule, warm, small_field, cfg)
    selected = gather_selected(plan.candidate_w, plan.selected_index)
    assert ((selected[..., :2] - obstacle_out) @ e_cmd >= cfg.touchdown.d_landing).all()
    assert plan.selected_sweep_safe.all()


def test_selector_rerun_is_warm_and_latched_safe_target_does_not_drift():
    p1 = select_touchdowns(...)
    p2 = select_touchdowns(..., previous_plan=p1)
    assert torch.equal(p2.target_w[p1.latched], p1.target_w[p1.latched])
```

- [ ] **Step 2: Implement fixed event timing and hip-frame candidates**

Compute the next touchdown for every leg and a second preview endpoint only when H30 ends inside the next swing. Generate a `5 x 5` hip-yaw grid with no Python environment or candidate loop.

- [ ] **Step 3: Implement safety masks and score components**

```python
safe = valid_map & landing_safe & reachable & plane_valid & whole_sweep_safe
score = (
    w_cmd * squared(candidate_xy - command_target_xy)
    + w_warm * squared(candidate_xy - previous_target_xy)
    + w_slope * slope.square()
    + w_rough * roughness.square()
    + w_edge / distance_to_forbidden.clamp_min(epsilon)
)
selected_index = masked_argmin(score, safe)
```

For small obstacles, add the hard after-obstacle mask; for large obstacles, candidates and whole-body corridor inside inflated large are invalid.

- [ ] **Step 4: Run GREEN for B=1/40/512/1024 shapes**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py \
  Go2Pvcnn/extension/joint_mpc_rti/config.py \
  Go2Pvcnn/extension/joint_mpc_rti/types.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py
git commit -m "feat: select safe touchdown events every refresh"
```

---

### Task 6: Build Convex Touchdown Regions And Local Planes

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/types.py`
- Extend: `Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py`

- [ ] **Step 1: Write RED region tests**

Assert selected center containment, positive minimum half extents, every cell and complete sole inside landing-safe, local plane residual below threshold, and invalidation rather than expansion across forbidden cells.

```python
assert torch.all((plan.region_A @ plan.target_w[..., :2, None]).squeeze(-1) + plan.region_b >= 0)
assert torch.all(plan.region_area[plan.region_valid] > 0)
assert torch.all(plan.region_plane_residual[plan.region_valid] <= cfg.region.max_plane_residual)
```

- [ ] **Step 2: Implement hip-aligned maximal safe rectangles**

Scan fixed positive/negative hip-frame directions in the landing mask, cap each half extent, subtract `region_margin`, and reject any extent below `min_half_extent`.

- [ ] **Step 3: Fit the fixed-shape weighted plane**

Solve the `3 x 3` normal equations for `h(x,y)=h0+h_x dx+h_y dy`; return plane normal, touchdown z, residual, corners, `A_td`, and `b_td`.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py \
  Go2Pvcnn/extension/joint_mpc_rti/types.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py
git commit -m "feat: construct convex touchdown regions"
```

---

### Task 7: Rebuild Cold-Once Warm-Retarget Nominal

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/runtime/warm_start.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/model/analytic_ik.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`

- [ ] **Step 1: Write RED lifecycle and retarget tests**

```python
def test_only_first_optimize_after_reset_is_cold():
    first = build_nominal(..., initialized=False)
    second = build_nominal(..., previous=accepted(first), initialized=True)
    assert first.used_cold_start.all()
    assert second.used_warm_start.all()
    assert not second.used_cold_start.any()


def test_warm_selector_change_retargets_without_becoming_cold():
    result = build_nominal(..., previous=accepted, perceptive_plan=new_target)
    assert result.used_warm_start.all()
    assert torch.linalg.vector_norm(result.retargeted - result.rebased) > 0


def test_phase12_node_reaches_touchdown_without_phase11_kink():
    result = build_nominal(..., phase0=torch.tensor([0]))
    foot = result.foot_reference_w[0, :, 0]
    torch.testing.assert_close(foot[12], result.touchdown_reference_w[0, 12, 0])
    assert torch.linalg.vector_norm(foot[12] - foot[11]) > 0
```

- [ ] **Step 2: Implement shift/rebase and exact z0 injection**

```python
shift[:, :-1] = accepted[:, 1:]
shift[:, -1] = terminal_fill(accepted, command)
rebase = rigid_xy_yaw_rebase(shift, measured)
rebase[:, 0] = measured.as_vector()
```

Preserve root/joint trend, gait progress, continuing stance anchors, event metadata, and terminal trend. An initialized missing/nonfinite cache sets an invariant fault and stop mask; it never calls cold construction.

- [ ] **Step 3: Implement continuous swing/stance references and IK retarget**

Use quintic XY, one apex computed from current radius-height field, fixed outward offset, two-piece smooth z, full XYZ persistent stance anchors, and selected touchdown plane for future stance. Blend analytic IK only on affected nodes and overwrite `z0` last.

- [ ] **Step 4: Require nominal hard safety before LQ**

Retry the next ranked touchdown candidate without a second selector build. If every ranked candidate produces unsafe/unreachable nominal, return `nominal_safe=False` and stop for that environment.

Run:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_analytic_ik.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py \
  Go2Pvcnn/extension/joint_mpc_rti/runtime/warm_start.py \
  Go2Pvcnn/extension/joint_mpc_rti/model/analytic_ik.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_analytic_ik.py
git commit -m "feat: retarget warm nominal to current touchdown plan"
```

---

### Task 8: Implement The Eight LQ Residual Families

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/lq_problem.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_lq_problem.py`

- [ ] **Step 1: Write RED residual-key and derivative tests**

```python
EXPECTED = {"velocity", "posture", "root", "swing", "touchdown", "smooth", "warm", "slack"}

def test_lq_problem_has_exact_residual_families():
    problem = build_lq_problem(nominal, context, cfg)
    assert set(problem.cost_breakdown) == EXPECTED


def test_lq_gradient_matches_finite_difference():
    assert max_abs(autograd_gradient(problem) - finite_difference_gradient(problem)) < 2e-3
```

- [ ] **Step 2: Implement direct-Z residuals**

Implement velocity tracking, low-speed posture hold, root height/RPY/corridor, swing position/velocity, touchdown center, first/second finite differences, and warm consistency. Slack penalty is represented separately but included under the `slack` diagnostic key.

- [ ] **Step 3: Assemble GGN block bands**

Accumulate only diagonal, first off-diagonal, and second off-diagonal `18 x 18` blocks plus gradient. The implementation may use autograd only in eager reference tests; production uses analytic or batched Jacobian assembly.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_lq_problem.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver/lq_problem.py \
  Go2Pvcnn/extension/joint_mpc_rti/config.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_lq_problem.py
git commit -m "feat: assemble perceptive kinematic lq costs"
```

---

### Task 9: Add Full-Horizon Kinematic Constraints And KKT Diagnostics

**Files:**
- Extend: `Go2Pvcnn/extension/joint_mpc_rti/solver/lq_problem.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py`

- [ ] **Step 1: Write RED constraint tests**

Cover exact `z0`, joint/rate/trust bounds, root z/RPY bounds, every stance node XYZ anchor, touchdown region/plane, linearized part clearance, and typed slack caps.

```python
assert problem.z0_fixed.all()
assert problem.stance_rows.shape[:3] == (B, 31, 4)
assert problem.touchdown_region_rows.shape[-1] == 4
assert problem.kkt_primal_residual(solution).max() <= 1e-4
```

- [ ] **Step 2: Implement scaled fixed-shape constraint rows**

Rows are allocated for all nodes/events and masked inactive rather than dynamically appended. Stance uses complete FK Jacobians. Touchdown region is active at onset and its following stance segment. Clearance rows are local QP guidance; exact publication remains Task 11.

- [ ] **Step 3: Implement active masks, slack diagnostics, and dense reference**

Return primal/dual residual, `slack_max[B,type]`, and `active_constraint_count[B,type]`. Keep the fixed compile budget and use Schur/arrowhead structure rather than materializing a per-environment Python KKT in production.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_lq_problem.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver/lq_problem.py \
  Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_lq_problem.py
git commit -m "feat: constrain full kinematic horizon"
```

---

### Task 10: Adapt The H30/32 Associative Solver

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/solver/associative_scan.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py`

- [ ] **Step 1: Write RED parity tests**

```python
@pytest.mark.parametrize("batch", (1, 8, 40))
def test_scan_matches_dense_constrained_solution(batch):
    dense = solve_dense_qp(problem(batch))
    scan = solve_trajectory_qp_scan(problem(batch))
    torch.testing.assert_close(scan.direction, dense.direction, atol=2e-5, rtol=2e-5)
```

Also assert exactly 30 real interval factors, two neutral padding factors, five combine levels, and no second LQ construction during active refinement.

- [ ] **Step 2: Map `LqProblem` blocks to conditional factors**

Use the existing associative combine/fixed solvers where parity holds. Recover all `Delta z1...Delta z30`; force `Delta z0=0`.

- [ ] **Step 3: Add fixed active-mask refinement without host loops**

All production operations remain batch tensor operations. Dense solve remains test-only.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver/associative_scan.py \
  Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py
git commit -m "feat: solve perceptive lq with h30 scan"
```

---

### Task 11: Replace Line Search With Five Exact Hard-Safe Candidates

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/solver/sqp_rti.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py`

- [ ] **Step 1: Write RED candidate/filter tests**

Assert shape `[B,5,31,18]`, five fixed alpha values, current field only, exact joint/root/stance/region/plane filters, full swept part safety, terminal preview safety, and `alpha=0` receiving the same checks.

```python
assert result.alpha_feasible.shape == (B, 5)
assert result.alpha_reject_bits.shape[:2] == (B, 5)
assert not result.alpha_feasible[tunneling_candidate]
assert result.stop[~result.alpha_feasible.any(dim=1)].all()
```

- [ ] **Step 2: Implement one packed nonlinear evaluation**

Flatten candidate dimension to `B*5`, evaluate nonlinear cost and exact filters once, reshape, and select minimum feasible cost with larger-alpha tie break.

- [ ] **Step 3: Add planned small-cross direction filter**

For intervals whose swept sole intersects the small footprint, require `v_foot_xy dot e_root_plan >= v_forward_min`. Command direction continues to define before/after and opportunity. The final actual-direction metric is Task 14.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_swept_safety.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py \
  Go2Pvcnn/extension/joint_mpc_rti/solver/sqp_rti.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py
git commit -m "feat: enforce five-candidate hard safety"
```

---

### Task 12: Route The Ten-Stage Planner And Warm-Only Runtime

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/runtime/cuda_graph.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/integration/isaaclab_adapter.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/integration/reference_adapter.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/integration/viewer_adapter.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py`

- [ ] **Step 1: Write RED orchestration and lifecycle tests**

Count exactly one selector, one nominal retarget, one linearization, one QP solve, and one five-alpha call per refresh. Assert 31 internal nodes, 30 future frames, x1 publication, frame mismatch stop, one cold then warm-only, last finite cache preservation, and per-row reset isolation.

- [ ] **Step 2: Implement the ten-stage `planner.step`**

```python
field = build_perceptive_field(current_scanner_frame)
fresh = validate_frame_freshness(measured.refresh_id, field.refresh_id)
shifted = shift_rebase(previous.accepted, measured)
plan = build_perceptive_plan(measured, command, schedule, shifted, field, previous.plan)
nominal = retarget_nominal(measured, shifted, plan, schedule, field)
problem = build_lq_problem(nominal, plan, field, cfg)
direction = solve_trajectory_qp_scan(problem)
search = parallel_line_search(nominal, direction, plan, field, cfg)
result = publish_or_stop(search, fresh, previous)
```

- [ ] **Step 3: Implement cache and phase state machine**

Only finite hard-safe selected trajectories replace accepted cache. No feasible candidate publishes stop and keeps the previous cache. Cache corruption in initialized state sets `warm_cache_invariant_fault`; explicit reset alone clears `initialized` and phase.

- [ ] **Step 4: Make CUDA Graph capture allocation-free**

Pre-register constants/workspaces, copy all current map channels and metadata before replay, and add a capture/replay test that rejects `torch.tensor`, `.item()`, host sync, and dynamic shape inside the captured planner.

Run:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/planner.py \
  Go2Pvcnn/extension/joint_mpc_rti/runtime \
  Go2Pvcnn/extension/joint_mpc_rti/integration \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py
git commit -m "refactor: route final perceptive rti pipeline"
```

---

### Task 13: Add Fixed Diagnostics, Profiling, And Viewer Controls

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/diagnostics/metrics.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/diagnostics/profiler.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/types.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_diagnostics.py`
- Modify: `Go2Pvcnn/tests/test_viewer_reset.py`

- [ ] **Step 1: Write RED fixed-shape diagnostics tests**

Assert selector counts/reasons/scores, region quality, warm/rebase/target reasons, nominal safety, KKT residual/slack/active rows, and per-alpha cost/feasible/reject/part-clearance fields. Assert diagnostics do not call planner a second time.

- [ ] **Step 2: Implement tensor-only diagnostics**

Use the shapes in design Section 14.4. Target-change reason is a four-bit tensor allowing simultaneous command/map/unsafe causes; latched is a separate state.

- [ ] **Step 3: Implement eight profiler stages**

Record field, selector, region, nominal/IK, linearization, scan QP, line-search safety, and cache/diagnostics with CUDA events in benchmark mode and no per-environment synchronization.

- [ ] **Step 4: Wire viewer overlays and live parameter groups**

Display current semantic/elevation masks, candidates/reasons, selected target, region/plane, warm/nominal/five candidates, world geometry clearance, gait phase, selected alpha, publish/stop reason, and stage timings. Viewer still invokes exactly one production SQP.

Run:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_diagnostics.py \
  Go2Pvcnn/tests/test_viewer_reset.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/diagnostics \
  Go2Pvcnn/extension/joint_mpc_rti/types.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_diagnostics.py \
  Go2Pvcnn/tests/test_viewer_reset.py
git commit -m "feat: expose final rti diagnostics in viewer"
```

---

### Task 14: Rebuild Actual-World Metrics And Acceptance Schema

**Files:**
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/acceptance_thresholds.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_run_joint_acceptance.py`

- [ ] **Step 1: Write RED schema and applicability tests**

Require all design Section 15 IDs; report-only acceleration/jerk/lead fields remain finite diagnostics. Zero translation makes only direction/cross terms N/A. Flat and small both require every `M_common` metric.

- [ ] **Step 2: Implement planned and actual world-geometry metrics**

For every refresh, compute P from selected trajectory and A from actual root/joint readback, apply FK, and query the matching current map. Any P or A part collision fails its cell. No Isaac contact sensor data is accepted.

- [ ] **Step 3: Implement strict-cross event windows**

Use continuous swept sole overlap, vertical margin, before/after in command direction, touchdown on normal ground, whole-body zero collision, and actual root direction:

```python
e_root = (p_root_td - p_root_lift) / norm.clamp_min(epsilon)
k_obs = swept_sole_intersects_current_small(...)
direction_ok = ((v_foot_xy[k_obs] * e_root).sum(-1) >= v_forward_min).all()
strict = opportunity & before & over_xy & over_z & direction_ok & after & land_ok & body_ok
```

If root speed is below `v_min`, direction angle is N/A but a formal opportunity remains a failed cross rather than disappearing from the denominator.

- [ ] **Step 4: Add split lifecycle and smoothness metrics**

Implement alpha0-selected/no-feasible/publish/stop ratios, three fixed zero-drift durations, target changes by four reason bits, latched drift, acceleration max/mean/p95, jerk max, KKT residuals, nominal clearance, and map/frame parity.

Run:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_run_joint_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_process_watchdog.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py \
  Go2Pvcnn/tests/joint_mpc_rti/acceptance_thresholds.py \
  Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_run_joint_acceptance.py
git commit -m "test: define current-map whole-robot acceptance"
```

---

### Task 15: Delete Superseded Production Paths

**Files:**
- Delete after import migration: `Go2Pvcnn/extension/joint_mpc_rti/terrain/cost_map.py`
- Delete after import migration: `Go2Pvcnn/extension/joint_mpc_rti/runtime/reference_buffer.py`
- Delete after import migration: `Go2Pvcnn/extension/joint_mpc_rti/losses/semantic.py`
- Delete after import migration: `Go2Pvcnn/extension/joint_mpc_rti/losses/terrain.py`
- Delete after import migration: `Go2Pvcnn/extension/joint_mpc_rti/losses/step.py`
- Delete after import migration: `Go2Pvcnn/extension/joint_mpc_rti/losses/swing_speed.py`
- Delete after import migration: `Go2Pvcnn/extension/joint_mpc_rti/losses/contact.py`
- Delete after import migration: `Go2Pvcnn/extension/joint_mpc_rti/losses/rollout_objective.py`
- Delete unreachable duplicate solvers identified by import graph
- Modify: package `__init__.py` files and imports
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_final_import_graph.py`

- [ ] **Step 1: Write RED forbidden-import tests**

```python
FORBIDDEN = (
    "losses.semantic", "losses.terrain", "losses.step", "losses.swing_speed",
    "losses.contact", "losses.rollout_objective", "terrain.cost_map",
    "runtime.reference_buffer",
)

def test_production_import_graph_has_no_superseded_path():
    imports = production_imports("extension.joint_mpc_rti")
    assert not any(name in imports for name in FORBIDDEN)
```

- [ ] **Step 2: Move remaining required behavior to final owners**

No old loss, repair, pending-reference, Gaussian occupancy, EDT safety, fallback-to-cold, output projection, or x1-only collision symbol may remain reachable from planner/manager/viewer.

- [ ] **Step 3: Delete unreachable modules and duplicate solvers**

Keep `fixed_general*`, `fixed_spd*`, `associative_scan.py`, and `trajectory_scan.py` only when imported by the final scan. Delete `primal_dual_ilqr.py`, `associative_tvlqr.py`, `gauss_newton.py`, or `linearization.py` only after `rg` proves no final import or test-only dense parity dependency.

- [ ] **Step 4: Run import graph and focused suite**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_final_import_graph.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py -q
```

- [ ] **Step 5: Commit**

```bash
git add -A Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti/test_final_import_graph.py
git commit -m "refactor: remove superseded joint mpc paths"
```

---

### Task 16: Close The Flat Gate

**Files:**
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_flat_acceptance.py`
- Modify as required by evidence: final implementation/config files only

- [x] **Step 1: Run the 19-command flat RED gate**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py \
  --stage flat --timeout-s 900 --output /tmp/joint_mpc_flat.json
```

Expected initial result: at least one `M_common` metric identifies remaining behavior work; every failure includes cell, numerator/denominator, source, worst-case key, and module diagnostics.

- [x] **Step 2: Add one RED regression per observed root cause**

Place the smallest reproducer in the owning test file: nominal for stance/swing, LQ for residual/constraint direction, line search for exact filter, runtime for lifecycle, or metrics for evidence computation. Do not change thresholds or add scenario repair.

- [x] **Step 3: Tune only approved parameters after structural tests pass**

Use viewer-visible parameter groups. Preserve zero command, low-speed posture, stance XYZ anchors, joint bounds, collision zero, and response gates together.

- [x] **Step 4: Require complete flat GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_flat_acceptance.py -q
```

Then rerun the monitored command and require all 19 cells and every applicable `M_common` metric to pass.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti/test_flat_acceptance.py
git commit -m "test: close final flat kinematic gate"
```

---

### Task 17: Close The Small-Obstacle Gate

**Files:**
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py`
- Modify as required by evidence: selector/region/nominal/LQ/line-search/config only

- [ ] **Step 1: Start with one controlled cuboid RED cell**

Use `vx=0.2 m/s`, one environment, one phase, and one offset. Require behind candidate, valid region, safe nominal, whole-leg sweep, direction margin, touchdown after obstacle, current-map whole-body collision zero, and strict cross. The same trace must also run the complete flat `M_common` metric set: joint/rate limits, foot continuity, command velocity tracking, swing clearance, root direction/yaw tracking, lifecycle/numerical validity, and stance grounding. Every continuing and touchdown-onset stance foot must be on current-map normal ground, not small/large/unknown semantic cells, with the existing gap and penetration thresholds.

- [ ] **Step 2: Expand in fixed order**

Run all 24 entry phases and offsets for cuboid, then sphere/cylinder/capsule/cone at `0.2`, `0.6`, and `1.0 m/s`, then the remaining single-axis commands. Every small cell must also rerun and pass all `M_common` metrics. Crossing may use the bounded `root_lateral_offset_from_nominal_m` allowance of `0.10m` only inside the active crossing event; the original flat value `0.06m` must still be reported, and all other common thresholds remain unchanged. The allowance must not affect stance ground/semantic checks, velocity tracking, foot continuity, root step jump, roll/pitch, or collision gates.

- [ ] **Step 3: Add actual viewer readback evidence**

Run 15 canonical shape-speed traces from before lift through touchdown plus one complete 24-frame gait. Require each trace strict-cross, every applicable `M_common` metric, normal-ground stance/touchdown, and per-part collision rate zero. The root lateral exception is evaluated only over the crossing window and must be absent after the following complete gait period.

- [ ] **Step 4: Require complete small GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py \
  --stage small --timeout-s 1800 --output /tmp/joint_mpc_small.json

PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py -q
```

The controlled crossing test must consume the same `evaluate_trace`/`require_small_gate` path as the formal small report rather than asserting only `strict_cross_success`. Add a regression where a foot crosses the obstacle but a continuing stance foot is on the small semantic footprint; crossing must fail even when the geometric crossing predicates are true.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py
git commit -m "test: close final small obstacle gate"
```

---

### Task 18: Close Large Bypass And RL Batch Interfaces

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_large_acceptance.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/integration/reference_adapter.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`

- [ ] **Step 1: Write RED large-bypass and RL-mask tests**

Require cuboid/cylinder/wall-like obstacles, left/right space, no part entering inflated large, no touchdown/stance on large, forward progress ratio at least `0.70`, and no prolonged stop. Assert optional RL logits only bias already safe candidates and cannot make an unsafe candidate selectable.

- [ ] **Step 2: Implement stable large corridor choice**

Score left/right body corridors by length, clearance, and switch penalty; apply one smooth root XY corridor only for large encounters. Keep flat/small root XY nominal unchanged.

- [ ] **Step 3: Verify fixed RL outputs**

Assert `future_state[B,30,18]`, `candidate_safe_mask[B,4,25]`, target/region, stop/publish, and diagnostics work at B=1 and B=1024 with no environment cross-talk.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_large_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_isaaclab_runtime.py -q
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py \
  Go2Pvcnn/extension/joint_mpc_rti/integration/reference_adapter.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_large_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py
git commit -m "test: close large bypass and rl interface gate"
```

---

### Task 19: Meet `1024 x 1000 < 5s` And Run Final Same-Candidate Verification

**Files:**
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_full_refresh_probe.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/test_performance.py`
- Modify after profiling: final implementation files only
- Update: `notes/todo.md`
- Update: `notes/todo/T302v-joint-mpc-rti-gpu.md`
- Update: `notes/log/index.md`
- Create: one final verification log under `notes/log/`

- [ ] **Step 1: Establish B=1/B=1024 numerical parity**

For identical rows, compare target index, region, nominal, QP direction, alpha feasibility/cost/reject bits, selected future trajectory, stop/publish, and diagnostics. No cache/reset/mask row may affect another environment.

- [ ] **Step 2: Run three formal performance workloads**

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=Go2Pvcnn \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_full_refresh_probe.py \
  --batch-size 1024 --refreshes 1000 --warmup 100 --repeats 5 \
  --workloads flat-only small-only realistic-mixed
```

Each repeat for each workload must be `<5.0s`. Report total/average batch refresh, throughput, peak memory, graph replay count, and all eight stage timings. The derived `<5ms` value is per 1024-environment batch, not per environment.

- [ ] **Step 3: Optimize only execution structure**

Use preallocation, packed queries, `torch.compile`, Triton fixed solves, CUDA Graph replay, and MPX-style batch/time/alpha parallelism. Do not reduce 25 candidates, sweep samples required by parity, safety margins, metric coverage, or map freshness.

- [ ] **Step 4: Run the final full verification on one unchanged candidate**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti -q

git diff --check
rg -n '^(<<<<<<<|=======|>>>>>>>)|TODO|TBD' \
  Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti
```

Then rerun flat, small, large, canonical viewer, B=1/B=1024 parity, and all three performance workloads without changing code/config between gates.

- [ ] **Step 5: Align notes and commit final evidence**

Record exact commands, environment/GPU, candidate commit, P/A/M metrics, canonical traces, performance stage timings, remaining skipped tests, and refs. Then:

```bash
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti \
  notes/todo.md notes/todo/T302v-joint-mpc-rti-gpu.md notes/log/index.md notes/log
git commit -m "test: verify final perceptive kinematic joint mpc"
```

---

## Final Acceptance Checklist

- [ ] Production path is exactly field -> selector -> region -> warm shift/rebase -> retarget -> one linearization -> one LQ/QP -> five-alpha hard safety -> publish/cache.
- [ ] External trajectory is exactly 30 future frames; internal solver trajectory is exactly 31 nodes including measured z0.
- [ ] Gait is fixed 24/12/12 diagonal trot with phase 11->12 touchdown and `tau=r/12` endpoint.
- [ ] Every non-reset optimize after initialization is warm, including map/command changes, alpha0, no feasible candidate, and invalid optimization.
- [ ] Every refresh uses matching state/map/transform metadata with map age zero.
- [ ] Touchdown selector reruns every refresh, uses 25 candidates per leg, builds safe convex regions, and records reasoned target changes.
- [ ] Nominal is continuous, stance XYZ is world-fixed, touchdown is reached at phase 12, and nominal itself is hard-safe before LQ.
- [ ] One direct-Z LQ/QP contains approved costs and full-horizon kinematic constraints; scan/dense parity is `<2e-5`.
- [ ] Five candidates use identical exact nonlinear filters; alpha0 is not an automatic fallback.
- [ ] Foot, knee, calf, thigh, and base planned/actual collision rates are each zero on current maps.
- [ ] Flat passes every `M_common` metric for all 19 commands.
- [ ] Small passes every `M_common` and `M_small` metric; all 15 canonical actual traces strict-cross without collision.
- [ ] Large bypass meets progress and collision gates.
- [ ] Viewer runs the same one-SQP backend and exposes every approved parameter/diagnostic without a second solve.
- [ ] B=1/B=1024 parity passes, and flat-only/small-only/mixed `1024 x 1000` each pass `<5.0s` five times.
- [ ] Old soft collision, repair, pending reference, stale field, fallback-to-cold, and duplicate solver paths are absent from the production import graph.

## Execution Rule

Execute tasks in order. For every behavior correction: add the smallest failing regression, verify the expected RED cause, implement the minimal owning-module change, run focused GREEN, then run the immediately adjacent regression group. Do not tune around a structural failure, weaken a threshold, discard an opportunity denominator, or claim collision success from planner-only geometry when actual readback is red.
