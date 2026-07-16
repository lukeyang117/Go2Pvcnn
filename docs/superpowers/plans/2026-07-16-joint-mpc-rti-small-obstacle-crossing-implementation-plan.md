# Joint MPC RTI Small-Obstacle Crossing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `joint_mpc_rti` produce signed-field-driven, collision-free small-obstacle crossings while preserving rolling `x1`, stance grounding, command tracking, arbitrary batch size, and the five-second performance target.

**T302v.5 Continuation:** Preserve the verified `254/254` crossing and zero-collision result while making rolling x1 recover real stance support when command progress stops inside a small-object influence region. The continuation is part of the same design and retains the no-hard-gate constraint.

**Architecture:** Keep the existing `x0 -> H16 -> x1` RTI pipeline. Replace unsigned semantic EDT with signed boundary distance, extend fixed-shape FK with thigh samples and analytic link Jacobians, and place identical foot/calf/thigh residuals in LQ/GGN and merit. A deterministic five-shape x three-speed probe owns final cross and per-part collision acceptance.

**Tech Stack:** Python 3.10, PyTorch, C++/CUDA extension, pytest, Isaac Lab `env_isaacsim`.

---

## File Structure

- `terrain/distance_field.py`, `cuda_edt.py`, `csrc/work_efficient_edt.*`, `field_builder.py`, `field_cache.py`: signed small/large field construction and publication.
- `model/go2_kinematics.py`, `model/rollout.py`: thigh samples and foot/calf/thigh analytic point Jacobians.
- `losses/semantic.py`, `losses/rollout_objective.py`, `planner.py`, `config.py`: shared collision residual definitions and LQ/GGN visibility.
- `tests/joint_mpc_rti/small_obstacle_crossing_probe.py`: native-shape rolling collision/cross metrics.
- `tests/joint_mpc_rti/small_obstacle_stop_probe.py`: native-shape approach, stop-phase, support-recovery, collision, and drift metrics.
- Existing `tests/joint_mpc_rti/test_*.py`: RED/GREEN and regression acceptance.
- `notes/todo*`, `notes/log*`: required repository memory and evidence.

### Task 1: RED Gait and Signed-Field Contracts

**Files:**
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py`

- [ ] **Step 1: Add the gait RED test**

```python
def test_default_h16_covers_stance_swing_stance() -> None:
    cfg = JointMpcRtiCfg()
    assert cfg.gait.half_cycle_steps == 8
    contact = fixed_trot_schedule(1, 16, torch.device("cpu"), half_cycle_steps=8)[0]
    for leg in range(4):
        transitions = contact[1:, leg].to(torch.int8) - contact[:-1, leg].to(torch.int8)
        assert (transitions == -1).any()
        assert (transitions == 1).any()
```

- [ ] **Step 2: Add signed CPU/CUDA RED tests**

```python
def test_semantic_distance_is_signed_and_half_cell_corrected() -> None:
    field = make_box_field(x_cells=(70, 80), y_cells=(70, 80))
    assert field.small_distance_m[0, 75, 75] < 0
    torch.testing.assert_close(field.small_distance_m[0, 75, 69], torch.tensor(0.005), atol=1e-6, rtol=0)
    torch.testing.assert_close(field.small_distance_m[0, 75, 70], torch.tensor(-0.005), atol=1e-6, rtol=0)

def test_signed_distance_degenerate_channels_are_finite() -> None:
    empty = make_semantic_field(torch.zeros(1, 151, 151, dtype=torch.long))
    full = make_semantic_field(torch.ones(1, 151, 151, dtype=torch.long))
    assert torch.isfinite(empty.small_distance_m).all() and (empty.small_distance_m > 0).all()
    assert torch.isfinite(full.small_distance_m).all() and (full.small_distance_m < 0).all()
```

- [ ] **Step 3: Run and confirm RED**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py \
  -k 'stance_swing_stance or signed or half_cell'
```

Expected: default gait is `4`; obstacle interiors are zero instead of negative.

### Task 2: GREEN Signed CPU/CUDA Fields

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/distance_field.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/field_builder.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/field_cache.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/cuda_edt.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/csrc/work_efficient_edt.cpp`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/csrc/work_efficient_edt_cuda.cu`

- [ ] **Step 1: Change default timing and implement CPU reference**

```python
def signed_boundary_distance(mask: Tensor, *, resolution: float) -> Tensor:
    occupied = torch.as_tensor(mask, dtype=torch.bool)
    outside = jump_flood_distance(occupied, resolution=resolution)
    inside = jump_flood_distance(~occupied, resolution=resolution)
    half = 0.5 * float(resolution)
    diagonal = float(resolution) * math.sqrt(2.0 * float(occupied.shape[-1] - 1) ** 2)
    signed = torch.where(occupied, -(inside - half).clamp_min(half), (outside - half).clamp_min(half))
    has_obstacle = occupied.flatten(1).any(1).view(-1, 1, 1)
    has_free = (~occupied).flatten(1).any(1).view(-1, 1, 1)
    signed = torch.where(has_obstacle, signed, signed.new_full((), diagonal))
    return torch.where(has_free, signed, signed.new_full((), -diagonal))
```

Set `JointMpcRtiGaitCfg.half_cycle_steps = 8`.

- [ ] **Step 2: Implement fixed-workspace CUDA signed output**

Compute four transforms per environment: small occupied/free and large occupied/free. Keep float output `[2,B,151,151]`, expand reusable int16 workspace to `[4,B,151,151]`, then combine:

```cpp
output[index] = occupied
    ? -fmaxf(sqrtf(inside_sq) * resolution - 0.5F * resolution, 0.5F * resolution)
    :  fmaxf(sqrtf(outside_sq) * resolution - 0.5F * resolution, 0.5F * resolution);
```

No occupied seed returns positive grid diagonal; no free seed returns negative grid diagonal. Do not emit `inf` or `NaN`.

- [ ] **Step 3: Publish signed fields through builder/cache**

Use `signed_boundary_distance` on CPU. Preserve the public field names `small_distance_m` and `large_distance_m` for compatibility, but all values and query gradients now follow the signed contract.

- [ ] **Step 4: Run GREEN field tests and commit**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py
git add Go2Pvcnn/extension/joint_mpc_rti/config.py Go2Pvcnn/extension/joint_mpc_rti/terrain Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py
git commit -m "feat: add signed semantic distance fields"
```

Expected: CPU/CUDA parity, boundary signs, rotated query gradient, empty/full channels, cache rows, and gait tests pass.

### Task 3: RED/GREEN Thigh Geometry and Jacobians

**Files:**
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/model/rollout.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`

- [ ] **Step 1: Add RED shape and finite-difference tests**

```python
def test_fk_exposes_fixed_thigh_samples() -> None:
    geometry = go2_fk(torch.zeros(2, 3), torch.zeros(2, 3), nominal_joint_pos(2))
    assert geometry.thigh_samples_w.shape == (2, 4, 3, 3)

@pytest.mark.parametrize("part", ("foot", "calf", "thigh"))
def test_link_sample_jacobian_matches_finite_difference(part: str) -> None:
    analytic = link_sample_jacobians(root_pos, root_rpy, joint_pos)[part]
    finite = finite_difference_link_samples(part, root_pos, root_rpy, joint_pos, epsilon=1e-6)
    torch.testing.assert_close(analytic, finite, atol=3e-5, rtol=3e-4)
```

- [ ] **Step 2: Run and confirm RED**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py -k 'thigh or link_sample_jacobian'
```

Expected: missing thigh samples/Jacobian API.

- [ ] **Step 3: Implement fixed thigh samples and point Jacobians**

Use alpha `(0.25, 0.5, 0.75)` for hip-to-knee thigh and knee-to-foot calf. Thigh Jacobian is `alpha * knee_jacobian`; calf Jacobian is `(1-alpha)*knee_jacobian + alpha*foot_jacobian`. Return `[B,4,3,3,3]` per sampled link and carry thigh through `JointMpcRollout` and candidate selection.

- [ ] **Step 4: Run GREEN and commit**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py Go2Pvcnn/tests/joint_mpc_rti/test_solver.py
git add Go2Pvcnn/extension/joint_mpc_rti/model Go2Pvcnn/extension/joint_mpc_rti/planner.py Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py
git commit -m "feat: add thigh collision geometry"
```

### Task 4: RED/GREEN GGN-Visible Small-Link Residuals

**Files:**
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_losses.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_solver.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/losses/semantic.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/losses/rollout_objective.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`

- [ ] **Step 1: Add RED parity and direction tests**

```python
@pytest.mark.parametrize("part", ("foot", "calf", "thigh"))
def test_small_clearance_is_in_merit_and_lq(part: str) -> None:
    breakdown, problem = penetrating_part_fixture(part)
    assert f"small_object_{part}_clearance" in breakdown
    assert part_gradient(problem, part).abs().max() > 0

@pytest.mark.parametrize("part", ("foot", "calf", "thigh"))
def test_lq_direction_increases_penetrating_part_signed_distance(part: str) -> None:
    before, after = apply_one_lq_direction(part)
    assert after > before
```

- [ ] **Step 2: Run and confirm RED**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti/test_losses.py Go2Pvcnn/tests/joint_mpc_rti/test_solver.py -k 'small_clearance or lq_direction'
```

Expected: named residuals missing; calf/thigh LQ direction zero.

- [ ] **Step 3: Implement shared residual semantics**

Add physical radii `foot=0.022`, `calf=0.040`, `thigh=0.040` and separate weights. Merit returns `small_object_foot_clearance`, `small_object_calf_clearance`, `small_object_thigh_clearance`. Each uses `signed_distance - radius - margin`, continuous height weight, and proximity-weighted normalization.

- [ ] **Step 4: Add thigh packed queries and LQ gradients**

Extend `_LinearizationQueries` with thigh. Chain barrier derivative, query XY gradient, and analytic point Jacobian into root XY and corresponding joint columns. Add positive diagonal GGN curvature from squared Jacobians. Reuse exactly the same radii, margins, weights, and residual names as merit.

- [ ] **Step 5: Run GREEN and commit**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti/test_losses.py Go2Pvcnn/tests/joint_mpc_rti/test_solver.py Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti/test_losses.py Go2Pvcnn/tests/joint_mpc_rti/test_solver.py
git commit -m "feat: optimize small obstacle link clearance"
```

### Task 5: Native-Shape Cross and Zero-Collision Acceptance

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/small_obstacle_crossing_probe.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py`

- [ ] **Step 1: Implement reusable metrics**

```python
@dataclass(frozen=True)
class CrossingMetrics:
    cross_successes: int
    cross_opportunities: int
    collision_frames: dict[str, int]
    valid_frames: dict[str, int]
    max_penetration_m: dict[str, float]
    invalid_count: int
```

Use native sphere/cuboid/cylinder/capsule/cone, speeds `0.1/0.2/0.4m/s`, multiple longitudinal phases, foot sphere `0.022m`, calf/thigh capsules `0.040m`, and existing base samples. Strict cross is `stance -> contiguous swing -> stance`, both stances off semantic, swing XY over object, no side bypass, and no part contact.

- [ ] **Step 2: Add and run the RED acceptance**

```python
def test_native_small_matrix_crosses_without_body_collision() -> None:
    result = run_crossing_matrix()
    assert result.overall_cross_success_rate >= 0.95
    assert min(result.cross_success_rate_by_case.values()) >= 0.90
    for part in ("foot", "calf", "thigh", "base"):
        assert result.collision_frames[part] == 0
        assert max(case.collision_frames[part] for case in result.cases.values()) == 0
    assert result.invalid_count == 0
```

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py -k native_small_matrix
```

Expected: reproduce nonzero foot/calf collisions and low strict-cross success.

- [ ] **Step 3: Tune only approved continuous parameters**

Adjust weights, margins, proximity normalization, and swing target amplitude. After every change rerun native matrix plus flat command/stance tests. Do not add shape branches, hard crossing gates, fixed avoidance side, specified leg, snapping, projection, or repair.

- [ ] **Step 4: Run GREEN behavior acceptance and commit**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py
git add Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti/small_obstacle_crossing_probe.py Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py
git commit -m "test: require collision free small crossings"
```

Expected: overall strict cross `>=95%`, every shape-speed `>=90%`, foot/calf/thigh/base collision frames all zero overall and per case, invalid count zero, stance and command tests green.

### Task 6: Full Regression, Performance, and Repository Memory

**Files:**
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_performance.py`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302v-joint-mpc-rti-gpu.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-07-16-joint-mpc-rti-small-obstacle-crossing-implementation.md`

- [ ] **Step 1: Run joint and legacy regressions**

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py Go2Pvcnn/tests/test_viewer_reset.py
```

Expected: zero failures, including factory `num_envs=1/40/512/1024`, field versions, joint order, rolling `x1`, stance ground, command direction, and old MPC/viewer contracts.

- [ ] **Step 2: Run real viewer acceptance**

Run the existing real Isaac viewer probe for zero, forward/backward, lateral, yaw, speed-varied, and mixed commands. Require joint-order error zero, stance gap `<=0.012m`, penetration `<=0.001m`, valid ratio `1.0`, and numerical zero-command XY drift.

- [ ] **Step 3: Run full signed-field performance acceptance**

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_perf_probe.py --num-envs 1024 --steps 1000 --horizon 16 --include-field-refresh
```

Expected: total `<5.0s`, version `+1000`, no nonfinite values. Timing includes complementary small/large EDT, half-cell combine, MPC, and `x1`; report warmup, GPU, average, P95, max, and contention state.

- [ ] **Step 4: Update notes and final verification**

Record exact commands/metrics, candidate commit, changed contracts, and any remaining real-simulation boundary with repository-relative links.

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; known `${data}/NvStreamer-*` and `raw/mpx/` remain untracked and uncommitted.

- [ ] **Step 5: Commit verification evidence**

```bash
git add notes docs/superpowers/plans/2026-07-16-joint-mpc-rti-small-obstacle-crossing-implementation-plan.md Go2Pvcnn/tests/joint_mpc_rti/test_performance.py
git commit -m "docs: verify joint mpc small crossing"
```

### Task 7: T302v.5 Stop-On-Small Support Recovery

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/small_obstacle_stop_probe.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_losses.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_solver.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/losses/semantic.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/losses/rollout_objective.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`

- [x] **Step 1: Add the reusable stop-phase probe**

Use native sphere/cuboid/cylinder/capsule/cone, forward approach at `0.2m/s`, multiple root stop offsets, and at least `32` zero-command rolling x1 hold cycles. Track scheduled stance frames, grounded support count, floating stance frames, maximum surface gap, consecutive zero-support frames, root XY drift, stance-on-small, per-part collision frames/penetration, and invalid count.

- [x] **Step 2: Add and run RED support acceptance**

```python
def test_native_small_stop_matrix_recovers_grounded_support() -> None:
    result = run_stop_matrix()
    assert result.support_recovery_rate == 1.0
    assert result.max_consecutive_zero_support_frames <= 4
    assert result.max_stop_root_xy_drift_m <= 0.015
    assert result.stance_on_small_frames == 0
    assert all(result.collision_frames[part] == 0 for part in PARTS)
    assert result.invalid_count == 0
```

```bash
CUDA_VISIBLE_DEVICES=<gpu> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py -k 'native_small_stop_matrix'
```

Expected RED on `625768f`: all `65/65` cases reproduce floating and zero grounded support for at least `64` consecutive cycles.

- [x] **Step 3: Add unit RED for continuous phase handoff**

Require mid-swing foot-over weight to peak at the swing center and continuously approach zero at lift-off/touchdown. Require safe-landing weight to rise only near late swing and only as signed distance becomes safely positive. Test that there is no branch on command zero, semantic id, shape name, or selected leg.

- [x] **Step 4: Implement merit residuals**

Add configurable continuous exponents/weights. Apply phase-shaped mid-swing weight to `small_object_foot_over`. Add `small_object_safe_landing` using late-swing phase weight, signed-distance sigmoid, and squared surface contact error. Preserve the same single `height_w`, physical foot radius, signed distance, and temperature contracts.

- [x] **Step 5: Implement matching LQ/GGN terms**

Use the identical phase/signed-distance weights and surface residual in `_add_foot_terrain_linearization`. Chain surface Z error through the analytic foot Jacobian and include positive diagonal GGN curvature. Do not implement zero-command gates, semantic branches, projection, snapping, or specified-leg repair.

Also add continuous `stance_support_viability` to merit and LQ/GGN. Aggregate scheduled stance surface errors with a differentiable harmonic/smooth-min form so the cost is high only when all scheduled support feet are off the surface; preserve individual `stance_ground_contact` and collision losses.

- [x] **Step 6: Tune against both stop and cross matrices**

Tune only phase exponents, continuous landing weight, existing foot-over amplitude, and swing target amplitude. After every change run the stop matrix, strict cross matrix, flat rolling stance test, and loss/LQ parity tests. Do not weaken collision geometry or acceptance thresholds.

- [x] **Step 7: Run GREEN and regression**

```bash
CUDA_VISIBLE_DEVICES=<gpu> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/joint_mpc_rti/test_losses.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_solver.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py
```

Expected: stop support recovery `100%`, each case has at most four consecutive zero-support transition frames; total zero-support frames remain a diagnostic so separate short swing→stance handoffs are not incorrectly merged into one sustained floating event. Collision/stance-on-small/invalid counts remain zero, strict cross remains at least `95%` overall and `90%` per shape-speed, and flat stance/command tracking remain green.

- [x] **Step 8: Run full joint, legacy, real viewer, notes, and commit**

Run Task 6 regressions and real nine-command viewer again. Update T302v.5 notes/log with exact metrics. Keep the idle-GPU five-second performance gate open until it is measured uncontended.

**Final implementation result:** the stop probe uses 65 native shape/offset cases and 32 zero-command hold cycles. Root controls are re-based to the current command reference after RTI trajectory shift while joint controls retain their warm start. Foot-over uses a mid-swing phase envelope; safe landing, touchdown avoidance, and support viability use separate continuous signed-distance margins but remain internally identical between merit and LQ/GGN. Far from small objects, a sharp continuous distance kernel strengthens ordinary stance grounding without changing semantic behavior. Final acceptance is 65/65 stop recovery with maximum one consecutive zero-support frame, zero root drift, zero stance-on-small, and zero foot/calf/thigh/base collision frames; crossing is 100% overall and per case with the same zero-collision/zero-stance-on-small result.
