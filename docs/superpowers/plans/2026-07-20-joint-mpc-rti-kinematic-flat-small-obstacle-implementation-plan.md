# Joint MPC RTI Pure-Kinematic Flat-Small Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current recovery-heavy joint MPC implementation with a fixed-shape pure-kinematic H30 RTI planner that passes the shared flat and small-obstacle behavior gates and then meets the unchanged `1024 x H30 x 1000 <=5.0s` full-refresh performance gate.

**Architecture:** Optimize only `Z=[p_B,theta_B,q]` over 31 nodes, construct the complete nominal in one batched call, linearize exactly seven losses into one block-banded trajectory QP, solve it with a fixed H30/32 associative factor scan, and choose among five parallel loss-only line-search candidates. Reuse the existing `Go2Pvcnn/extension/joint_mpc_rti/` package and public manager/adapters, but delete the old 15+15 adaptive-contact, recovery, startup, post-LQ projection, collision-restoration, and constraint-ranked line-search paths after their valid metric coverage has moved to the new implementation.

**Tech Stack:** Python 3.10, PyTorch 2.7, `torch.compile`, CUDA Graph, Triton fixed-size kernels, Isaac Lab, pytest, batched analytic FK/IK, SQP-RTI/GGN, fixed-shape active-set KKT, associative Gaussian factor composition.

**Approved Design:** `docs/superpowers/specs/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-design.html`

**Supersedes For Execution:** `docs/superpowers/plans/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-implementation-plan.md`. Historical tests and evidence remain useful, but that plan's 15+15 scheduler/recovery architecture is not an implementation source.

---

## Fixed Contracts

These values and structures may not be changed during implementation or tuning:

```text
dt = 0.02
H = 30, nodes = 31
gait period = 24, swing = 12, stance = 12
optimization variable = Z[B,31,18]
line-search alpha = (1.0, 0.5, 0.25, 0.125, 0.0)
loss names = command, step, contact, swing_speed, terrain, posture, smooth
RTI iterations per control step = 1
published reference = x1
scan intervals = 30 padded to 32
dense/scan parity = max abs error < 2e-5
Stage B = realistic synchronous 1024 x H30 x 1000 <= 5.0s
```

Allowed tuning is limited to the seven loss weights and their approved subweights, `h_swing`, clearance margins that do not weaken physical geometry, smoothing temperatures, command/step reference scales, posture references, measurement-mismatch decay, trust-region sizes, regularization, `small_sigma_m`, `large_sigma_m`, `small_gain`, `large_gain`, and `h_wall`. The two active-set refinement passes and convolution/Scharr/query structure are frozen. Do not add loss categories, candidate filters, recovery logic, output projection, scenario-specific objective branches, or discrete semantic hard branches in the optimizer.

## Target File Map

```text
Go2Pvcnn/extension/joint_mpc_rti/
|- config.py                         # only approved fixed contracts and tuning parameters
|- types.py                          # measured input, nominal, optimized trajectory, compact solver state
|- planner.py                        # short one-RTI orchestration only
|- model/gait_schedule.py            # fixed 24/12+12 tensor schedule
|- model/analytic_ik.py              # batched Go2 analytic IK
|- model/go2_kinematics.py           # batched FK and complete Jacobians
|- model/nominal.py                  # cold and shifted warm nominal, one [B,31,18] call
|- terrain/cost_map.py               # grouped conv occupancy/height propagation/Scharr fields
|- terrain/query.py                  # one packed differentiable query
|- losses/command.py                 # command residual
|- losses/step.py                    # touchdown reference residual
|- losses/contact.py                 # stance lock and ground contact residual
|- losses/swing_speed.py             # swing-foot/root speed residual
|- losses/terrain.py                 # unified foot/link/root terrain residual
|- losses/posture.py                 # root/joint posture residual
|- losses/smoothness.py              # first/second state differences
|- losses/objective.py               # exact seven-key nonlinear objective and GGN blocks
|- solver/trajectory_qp.py           # block bands, active constraints, dense reference
|- solver/trajectory_scan.py         # fixed H30/32 associative solve
|- solver/line_search.py              # five candidates, three filters, seven-loss argmin
|- solver/sqp_rti.py                  # one linearize/solve/search update
|- runtime/warm_start.py              # compact shifted trajectory state helpers
|- runtime/manager.py                 # batch lifecycle, x1 buffer, graph dispatch
`- runtime/cuda_graph.py              # fixed-address new solver state replay

Go2Pvcnn/tests/joint_mpc_rti/
|- test_new_contract.py
|- test_nominal.py
|- test_trajectory_losses.py
|- test_trajectory_qp.py
|- test_trajectory_scan.py
|- test_line_search_v2.py
|- test_rti_pipeline.py
|- joint_metrics.py
|- acceptance_thresholds.py
|- scenario_matrix.py
|- run_joint_acceptance.py
|- run_monitored_joint_mpc.py
|- test_process_watchdog.py
|- test_flat_acceptance.py
|- test_small_acceptance.py
`- joint_mpc_rti_full_refresh_probe.py
```

Do not create another backend directory. `integration/isaaclab_adapter.py`, `integration/reference_adapter.py`, `integration/viewer_adapter.py`, and the existing external manager factory remain the ABI boundary.

---

### Task 1: Freeze The New Contract And Reject Old Structure

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_new_contract.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/types.py`
- Test: `Go2Pvcnn/tests/joint_mpc_rti/test_new_contract.py`

- [ ] **Step 1: Write the failing fixed-contract tests**

```python
def test_production_contract_is_h30_24_12_and_one_rti():
    cfg = JointMpcRtiCfg()
    assert cfg.runtime.horizon_steps == 30
    assert cfg.runtime.dt == 0.02
    assert cfg.runtime.sqp_iterations_per_step == 1
    assert cfg.gait.period_steps == 24
    assert cfg.gait.swing_steps == 12
    assert cfg.gait.stance_steps == 12
    assert cfg.solver.line_search_alphas == (1.0, 0.5, 0.25, 0.125, 0.0)


def test_loss_config_has_exactly_seven_top_level_weights():
    cfg = JointMpcRtiCfg()
    assert set(cfg.losses.weights()) == {
        "command", "step", "contact", "swing_speed",
        "terrain", "posture", "smooth",
    }


def test_solver_state_has_no_recovery_or_independent_control():
    fields = set(JointMpcRtiSolverState.__dataclass_fields__)
    assert fields == {"trajectory", "gait_phase", "valid"}
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_new_contract.py -q
```

Expected: FAIL on old `half_cycle_steps=15`, old four alpha values, old loss fields, and recovery/control solver-state fields.

- [ ] **Step 3: Replace config and compact types**

Use these public structures:

```python
@dataclass(frozen=True)
class JointMpcRtiSolverState:
    trajectory: Tensor       # [B,31,18], previous accepted Z*
    gait_phase: Tensor       # [B], int64 in [0,23]
    valid: Tensor            # [B], bool


@dataclass(frozen=True)
class JointMpcRtiTrajectory:
    state: Tensor            # [B,31,18]
    derived_velocity: Tensor # [B,30,18], diagnostics only, not optimized
    foot_pos_w: Tensor       # [B,31,4,3]
    contact_state: Tensor    # [B,31,4]
    valid: Tensor
    fallback: Tensor
    status: Tensor
    line_search_alpha: Tensor
    loss_breakdown: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class JointMpcRtiLossCfg:
    command: float = 1.0
    step: float = 1.0
    contact: float = 1.0
    swing_speed: float = 1.0
    terrain: float = 1.0
    posture: float = 1.0
    smooth: float = 1.0

    def weights(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in (
            "command", "step", "contact", "swing_speed",
            "terrain", "posture", "smooth",
        )}
```

Keep measured velocities in `JointMpcRtiState` because they are sensor inputs and diagnostics; they are not appended to optimization variable `z`.

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/config.py \
  Go2Pvcnn/extension/joint_mpc_rti/types.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_new_contract.py
git commit -m "refactor: freeze pure kinematic joint mpc contract"
```

---

### Task 2: Implement The Fixed Tensor Gait Schedule

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py`
- Delete after GREEN: `Go2Pvcnn/tests/joint_mpc_rti/test_h30_adaptive_contact.py`

- [ ] **Step 1: Write RED tests for all phases and batches**

```python
def test_fixed_trot_schedule_returns_b31x4_without_extension_state():
    phase = torch.tensor([0, 7, 23], dtype=torch.long)
    schedule = fixed_trot_schedule(phase, horizon_steps=30)
    assert schedule.phase.shape == (3, 31, 4)
    assert schedule.swing.shape == (3, 31, 4)
    assert schedule.stance.shape == (3, 31, 4)
    assert torch.equal(schedule.swing, ~schedule.stance)
    assert torch.equal(schedule.swing[:, :, 0], schedule.swing[:, :, 3])
    assert torch.equal(schedule.swing[:, :, 1], schedule.swing[:, :, 2])
    assert torch.equal(schedule.swing[:, :, 0], ~schedule.swing[:, :, 1])
    assert not hasattr(schedule, "recovery")
    assert not hasattr(schedule, "extension_age")
```

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py -q
```

Expected: FAIL because the current scheduler exposes 15+15 adaptive/recovery state.

- [ ] **Step 3: Implement one broadcasted schedule**

```python
@dataclass(frozen=True)
class FixedTrotSchedule:
    phase: Tensor
    swing: Tensor
    stance: Tensor
    swing_tau: Tensor


def fixed_trot_schedule(phase0: Tensor, *, horizon_steps: int = 30) -> FixedTrotSchedule:
    phase0 = torch.as_tensor(phase0, dtype=torch.long)
    k = torch.arange(horizon_steps + 1, device=phase0.device)
    leg_offset = torch.tensor((0, 12, 12, 0), device=phase0.device)
    phase = (phase0[:, None, None] + k[None, :, None] + leg_offset[None, None]) % 24
    swing = phase < 12
    tau = phase.to(torch.float32).div(11.0).clamp(0.0, 1.0)
    return FixedTrotSchedule(phase=phase, swing=swing, stance=~swing, swing_tau=tau)
```

- [ ] **Step 4: Run GREEN and delete adaptive scheduler tests**

Run the Step 2 command, then:

```bash
rg -n "adaptive_contact_schedule|advance_contact_scheduler|recovery_state|swing_extension_age" \
  Go2Pvcnn/extension/joint_mpc_rti/model Go2Pvcnn/tests/joint_mpc_rti
```

Expected: only historical files scheduled for deletion remain. Remove `test_h30_adaptive_contact.py` after its command-matrix checks are moved in Task 12.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_h30_adaptive_contact.py
git commit -m "refactor: replace adaptive contact with fixed trot schedule"
```

---

### Task 3: Add Batched Analytic IK And Preserve FK Geometry

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/model/analytic_ik.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_analytic_ik.py`

- [ ] **Step 1: Write RED FK/IK and no-loop tests**

```python
def test_batched_analytic_ik_matches_fk_for_b31x4_targets():
    root_pos, root_rpy, foot_target = reachable_targets(batch=8, nodes=31)
    q, reachable = go2_analytic_ik(root_pos, root_rpy, foot_target)
    fk = go2_fk(root_pos, root_rpy, q.reshape(8, 31, 12)).foot_pos_w
    assert q.shape == (8, 31, 4, 3)
    assert reachable.all()
    torch.testing.assert_close(fk, foot_target, atol=2e-5, rtol=2e-5)


def test_analytic_ik_source_has_no_python_time_or_leg_loop():
    source = inspect.getsource(go2_analytic_ik)
    tree = ast.parse(textwrap.dedent(source))
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
```

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_analytic_ik.py -q
```

Expected: FAIL because `go2_analytic_ik` does not exist.

- [ ] **Step 3: Implement vectorized three-link IK**

Transform all targets to each hip frame with `R_BW`, solve abduction and planar thigh/calf angles with clamped cosine law, and return a separate `reachable` mask. Do not silently clip the output into joint limits; cold nominal must report unreachable targets so Task 4 can shorten only the approved command step scale.

```python
def go2_analytic_ik(root_pos_w: Tensor, root_rpy_w: Tensor, foot_target_w: Tensor) -> tuple[Tensor, Tensor]:
    rotation_wb = rpy_to_rotation_matrix(root_rpy_w)
    target_b = torch.einsum(
        "...ji,...li->...lj",
        rotation_wb,
        foot_target_w - root_pos_w.unsqueeze(-2),
    )
    hip_local = target_b - HIP_OFFSETS.to(target_b)
    # Broadcasted left/right hip offset, abduction, cosine-law calf and thigh solution.
    q = solve_go2_leg_geometry(hip_local)
    reachable = analytic_reachability_mask(hip_local, q)
    return q, reachable
```

- [ ] **Step 4: Run GREEN plus existing FK/Jacobian tests**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_analytic_ik.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py -q
```

Expected: PASS, with complete foot/knee/calf/thigh/base FK coverage unchanged.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/analytic_ik.py \
  Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_analytic_ik.py
git commit -m "feat: add batched analytic ik for nominal trajectories"
```

---

### Task 4: Build Cold And Rolling Nominal In One Call

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/runtime/warm_start.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`

- [ ] **Step 1: Write RED shape, formula, shift, and source tests**

```python
def test_nominal_builder_returns_complete_b31_state_in_one_call():
    previous = invalid_solver_state_from_measurement(measured, phase)
    result = build_nominal(measured, command, field, phase, previous=previous, cfg=cfg)
    assert result.state.shape == (1024, 31, 18)
    assert result.foot_reference_w.shape == (1024, 31, 4, 3)
    assert result.contact_state.shape == (1024, 31, 4)
    torch.testing.assert_close(result.state[:, 0], measured.as_vector())


def test_warm_nominal_is_shift_rebase_and_measurement_decay():
    previous = accepted_solver_state(accepted, phase)
    result = build_nominal(measured_next, command, field, phase + 1, previous=previous, cfg=cfg)
    assert result.used_warm_start.all()
    torch.testing.assert_close(result.state[:, 0], measured_next.as_vector())
    torch.testing.assert_close(result.state[:, 30, 6:], result.state[:, 6, 6:], atol=1e-6, rtol=0)


def test_nominal_source_has_no_for_or_while():
    tree = ast.parse(textwrap.dedent(inspect.getsource(build_nominal)))
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
```

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py -q
```

Expected: FAIL because the one-call builder does not exist.

- [ ] **Step 3: Implement cold nominal**

```python
@dataclass(frozen=True)
class NominalTrajectory:
    state: Tensor
    foot_reference_w: Tensor
    touchdown_reference_w: Tensor
    contact_state: Tensor
    used_warm_start: Tensor
    valid: Tensor


def build_nominal(measured, command_body, terrain_field, gait_phase, previous, cfg):
    schedule = fixed_trot_schedule(gait_phase, horizon_steps=30)
    cold = build_cold_nominal(measured, command_body, terrain_field, schedule, cfg)
    warm = build_shifted_nominal(measured, command_body, terrain_field, schedule, previous, cfg)
    use_warm = previous.valid[:, None, None]
    state = torch.where(use_warm, warm.state, cold.state)
    foot = torch.where(use_warm.unsqueeze(-1), warm.foot_reference_w, cold.foot_reference_w)
    touchdown = torch.where(
        use_warm.unsqueeze(-1), warm.touchdown_reference_w, cold.touchdown_reference_w
    )
    valid = torch.where(previous.valid, warm.valid, cold.valid)
    return NominalTrajectory(
        state=state,
        foot_reference_w=foot,
        touchdown_reference_w=touchdown,
        contact_state=schedule.stance,
        used_warm_start=previous.valid,
        valid=valid,
    )
```

Cold nominal must reduce the configured command step scale with one batch mask if analytic IK reports unreachable points; it may not search semantic XY or create candidate trajectories.

- [ ] **Step 4: Implement shifted warm nominal**

Use tensor slicing for `previous[:,1:31]`, gait-period terminal joint copy `q30=q6`, SE(2) root rebase from old predicted `x1` to measured root, and one fixed `beta[31]` measurement mismatch vector. Query new terrain references after rebase; do not rebuild warm `q` through IK.

2026-07-21 amendment: when building Contact context, use FK at node zero for a stance segment already active at the horizon start, but use the matching `touchdown_reference_w` at every future stance onset and tensorially hold it until liftoff. Add a RED test where shifted warm FK deliberately differs from the regenerated touchdown target. This is a Step/Contact reference-consistency correction, not warm IK repair or a new constraint/loss.

The Step event must be the `tau=1` swing endpoint, phase `swing_steps-1` (`11`), not the first stance node phase `12`. Contact starts at phase `12` with the same touchdown anchor, giving a continuous target across the phase boundary.

- [ ] **Step 5: Run GREEN at B=1,40,512,1024**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py -q
```

Expected: PASS for all batch sizes, no source loops, finite state, exact measured `z0`, and no semantic XY modification.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py \
  Go2Pvcnn/extension/joint_mpc_rti/runtime/warm_start.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py
git commit -m "feat: build vectorized cold and rolling nominal"
```

---

### Task 5: Build The Unified Elevation-Semantic Cost Map

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/terrain/cost_map.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/field_cache.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/terrain/query.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/types.py`
- Test: `Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py`

- [ ] **Step 1: Write RED phase-aware field tests**

```python
def test_small_is_real_height_for_swing_and_virtual_wall_for_touchdown():
    query = make_small_object_query(height=0.08, small_distance=-0.01)
    swing = effective_surface(query, body_part="foot", stance=False, cfg=cfg)
    stance = effective_surface(query, body_part="foot", stance=True, cfg=cfg)
    assert swing.height_w.item() == pytest.approx(0.08)
    assert stance.height_w.item() >= cfg.terrain.virtual_wall_height


def test_large_is_virtual_wall_for_all_parts_and_phases():
    for part in ("foot", "knee", "calf", "thigh", "base"):
        surface = effective_surface(large_query, body_part=part, stance=False, cfg=cfg)
        assert surface.height_w.item() >= cfg.terrain.virtual_wall_height


def test_convolution_propagates_small_height_and_nonzero_boundary_gradient():
    fields = build_soft_semantic_fields(height, semantic_id, cfg)
    boundary = fields.small_occupancy[:, :, center_y, center_x + kernel_radius - 1]
    gradient = fields.small_gradient_xy[:, :, center_y, center_x + kernel_radius - 1]
    assert torch.all(boundary > 0.0)
    assert torch.all(torch.linalg.vector_norm(gradient, dim=1) > 0.0)
    assert fields.small_height[:, :, center_y, center_x + 1].item() == pytest.approx(object_top)


def test_soft_semantic_query_is_differentiable_with_respect_to_xy():
    point_xy = torch.tensor([[[0.03, 0.0]]], requires_grad=True)
    risk = query_soft_semantic(fields, point_xy).small_occupancy.sum()
    risk.backward()
    assert torch.isfinite(point_xy.grad).all()
    assert point_xy.grad.abs().sum() > 0.0
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py -k "effective_surface or packed" -q
```

Expected: FAIL because no unified `effective_surface` contract exists.

- [ ] **Step 3: Implement fixed grouped convolution occupancy and propagated height**

```python
mask = torch.stack((semantic_id.eq(SMALL_ID), semantic_id.eq(LARGE_ID)), dim=1).to(height_w.dtype)
weighted_height = mask * height_w[:, None]
conv_input = torch.cat((mask, weighted_height), dim=1)
conv = torch.nn.functional.conv2d(
    conv_input,
    fixed_semantic_kernels(cfg, dtype=height_w.dtype, device=height_w.device),
    padding=cfg.terrain.kernel_radius_cells,
    groups=4,
)
mass_small, mass_large, height_num_small, height_num_large = conv.unbind(dim=1)
occupancy_small = 1.0 - torch.exp(-float(cfg.terrain.small_gain) * mass_small)
occupancy_large = 1.0 - torch.exp(-float(cfg.terrain.large_gain) * mass_large)
height_small = height_num_small / mass_small.clamp_min(1e-6)
height_large = height_num_large / mass_large.clamp_min(1e-6)
```

Use one static maximum kernel size; encode `small_sigma_m` and `large_sigma_m` in their fixed per-channel kernels. Kernel size, padding, and channel count cannot change after CUDA Graph capture.

- [ ] **Step 4: Add Scharr gradients and differentiable packed query**

```python
occupancy = torch.stack((occupancy_small, occupancy_large), dim=1)
gradient_xy = torch.nn.functional.conv2d(
    occupancy,
    fixed_scharr_kernels(dtype=height_w.dtype, device=height_w.device),
    padding=1,
    groups=2,
).reshape(B, 2, 2, grid_h, grid_w)
```

Query occupancy, propagated class height, and explicit gradient with one bilinear `grid_sample`/packed gather. Return original height, exact signed distance, soft occupancy, propagated height, world XY gradient, validity, and raw semantic id. Preserve exact EDT and physical foot/link/base radii. The optimization path may consume soft occupancy/height/gradient; acceptance detectors consume raw semantic/elevation/geometry.

- [ ] **Step 5: Run terrain tests**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py -q
```

Expected: PASS; CUDA-only cases may skip when CUDA is unavailable, but CPU contracts must all pass.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/terrain/cost_map.py \
  Go2Pvcnn/extension/joint_mpc_rti/terrain/field_cache.py \
  Go2Pvcnn/extension/joint_mpc_rti/terrain/query.py \
  Go2Pvcnn/extension/joint_mpc_rti/types.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py
git commit -m "feat: unify elevation and semantic terrain fields"
```

---

### Task 6: Replace The Objective With Exactly Seven Losses

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/losses/command.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/step.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/losses/contact.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/swing_speed.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/losses/terrain.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/losses/posture.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/losses/smoothness.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/losses/objective.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_losses.py`

- [ ] **Step 1: Write RED exact-key and behavior tests**

```python
def test_objective_has_exactly_seven_losses():
    breakdown = trajectory_loss_breakdown(candidate, context, cfg)
    assert tuple(breakdown) == (
        "command", "step", "contact", "swing_speed",
        "terrain", "posture", "smooth",
    )


def test_swing_speed_penalizes_foot_not_faster_than_root():
    slow = swing_speed_loss(foot_step=0.01, root_step=0.02, margin=0.002)
    fast = swing_speed_loss(foot_step=0.03, root_step=0.02, margin=0.002)
    assert slow > fast


def test_smooth_loss_contains_first_and_second_state_differences():
    straight = make_state_sequence(second_difference=0.0)
    kinked = make_state_sequence(second_difference=0.1)
    assert smooth_loss(kinked, cfg) > smooth_loss(straight, cfg)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_losses.py -q
```

Expected: FAIL on old dozens of breakdown keys and control-based smoothness.

- [ ] **Step 3: Implement the seven residual families on `state[B,31,18]`**

```python
def trajectory_loss_breakdown(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> dict[str, Tensor]:
    return {
        "command": command_loss(state, context.command_body, cfg),
        "step": step_loss(state, context.touchdown_reference_w, context.schedule, cfg),
        "contact": contact_loss(state, context.stance_anchor_w, context.terrain, context.schedule, cfg),
        "swing_speed": swing_speed_loss(state, context.schedule, cfg),
        "terrain": terrain_loss(state, context.terrain, context.schedule, cfg),
        "posture": posture_loss(state, context.support_height, cfg),
        "smooth": smooth_loss(state, cfg),
    }
```

Each function must return `[B]` nonlinear cost and expose residual/Jacobian helpers used by Task 7. Derive root, joint, and foot velocities only from adjacent state nodes. `terrain_loss` must query all foot/knee/calf/thigh/base samples in one packed call, use convolution occupancy/propagated height/Scharr gradients, and include small touchdown avoidance without a hard gate. The existing `command_early_swing` subweight applies only to the current first edge, with continuous command activity `1-exp(-||v_cmd||^2/s_cmd^2)`; future swing edges retain full command pressure so rolling warm shifts cannot publish a future phase kink. Add tests for zero-command hold and future-transition command pressure. Add a source test that rejects `semantic_id ==`, `semantic_id.eq`, or class-mask branching inside `losses/terrain.py`; raw semantic ids are detector-only.

2026-07-21 zero-command amendment: multiply the existing Command residual by the square root of `1 + (command_hold_multiplier - 1) * exp(-||v_cmd||^2 / command_activity_scale^2)`. This must affect nonlinear scoring and GGN through the same residual, remain continuous, equal the configured multiplier at exactly zero, and converge to one for the formal nonzero command range. Add RED tests for those three properties. Do not add command-zero branching, output projection, a new loss key, or a line-search-only score.

- [ ] **Step 4: Run loss tests and finite-difference Jacobian tests**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_losses.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_losses.py -q
```

Expected: new tests PASS. Rewrite or delete old tests that assert removed loss names; retain geometry and derivative tests under the seven parent keys.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/losses \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_losses.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_losses.py
git commit -m "refactor: define seven-state trajectory losses"
```

---

### Task 7: Linearize The Seven Losses Into A Direct-Z QP

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/solver/linearization.py`
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py`

- [ ] **Step 1: Write RED dense Hessian/gradient tests**

```python
def test_direct_z_qp_matches_autograd_gradient_and_hessian_vector():
    qp = linearize_trajectory(nominal, context, cfg)
    dense_h, dense_g = qp.to_dense()
    auto_g = torch.autograd.grad(total_loss(nominal.state), nominal.state, create_graph=True)[0]
    torch.testing.assert_close(dense_g, auto_g.flatten(1), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(dense_h @ vector, residual_jtj_vector(vector), atol=5e-4, rtol=5e-4)


def test_qp_fixes_delta_z0_and_builds_joint_position_velocity_trust_bounds():
    qp = linearize_trajectory(nominal, context, cfg)
    assert torch.count_nonzero(qp.lower[:, 0]) == 0
    assert torch.count_nonzero(qp.upper[:, 0]) == 0
    assert qp.joint_difference_lower.shape == (B, 30, 12)
    assert qp.joint_difference_upper.shape == (B, 30, 12)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py -q
```

Expected: FAIL because the current LQ is control/dynamics based.

- [ ] **Step 3: Implement block bands and merged bounds**

```python
@dataclass(frozen=True)
class TrajectoryQp:
    diagonal: Tensor          # [B,31,18,18]
    first_offdiag: Tensor     # [B,30,18,18]
    second_offdiag: Tensor    # [B,29,18,18]
    gradient: Tensor          # [B,31,18]
    lower: Tensor             # merged position/trust lower bound on delta Z
    upper: Tensor
    joint_difference_lower: Tensor
    joint_difference_upper: Tensor
```

Assemble GGN blocks from the seven residual Jacobians. Merge position and trust bounds into one box per state component so a node contributes at most 18 active box rows; velocity contributes at most 12 edge rows. Do not add stance, collision, startup, or recovery constraint rows.

- [ ] **Step 4: Implement dense reference helpers**

`TrajectoryQp.to_dense()` and `solve_dense_active_kkt(qp, active)` are test-only/eager references. The active KKT must use at most `18+12=30` rows per interval and call `joint_kkt_compile_budget` before any compiled solve.

- [ ] **Step 5: Run GREEN**

Run the Step 2 command. Expected: PASS for float64 derivative parity and float32 finite blocks.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver/linearization.py \
  Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py
git commit -m "feat: linearize seven losses into direct trajectory qp"
```

---

### Task 8: Implement Fixed-Shape Active Bounds

**Files:**
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/solver/primal_dual_ilqr.py`

- [ ] **Step 1: Write RED active-set and resource tests**

```python
def test_active_set_selects_merged_box_and_joint_velocity_boundaries():
    free = make_direction_with_position_and_velocity_violations()
    active = select_active_constraints(qp, free)
    assert active.box_mask.shape == (B, 31, 18)
    assert active.velocity_mask.shape == (B, 30, 12)
    assert active.max_rows_per_interval <= 30


def test_two_unrolled_refinements_match_dense_active_kkt():
    scan_input = refine_active_set(qp, free_direction, refinements=2)
    dense = solve_dense_active_kkt(qp, scan_input.active)
    assert_constraints_hold(qp, dense, atol=2e-5)
```

- [ ] **Step 2: Run RED**

Run the Task 7 test command. Expected: FAIL because active selection/refinement is absent.

- [ ] **Step 3: Implement two tensorized active refinements**

```python
def select_active_constraints(qp: TrajectoryQp, direction: Tensor) -> ActiveConstraints:
    box_low = direction < qp.lower
    box_high = direction > qp.upper
    dq = direction[:, 1:, 6:] - direction[:, :-1, 6:]
    vel_low = dq < qp.joint_difference_lower
    vel_high = dq > qp.joint_difference_upper
    return ActiveConstraints.from_masks(box_low, box_high, vel_low, vel_high)


def refine_active_set(qp, solve_fn, refinements=2):
    first = solve_fn(qp, select_active_constraints(qp, solve_fn(qp, ActiveConstraints.empty(qp))))
    second_active = select_active_constraints(qp, first)
    second = solve_fn(qp, second_active)
    return ActiveSetSolution(direction=second, active=second_active)
```

The production function must contain two explicit refinement calls rather than a Python loop. `refinements` is validated to equal the frozen configured value used by the compiled graph.

- [ ] **Step 4: Enforce inherited compile budget before solve**

Keep and adapt:

```python
constraint_rows <= 32
next_power_of_two(constraint_rows) <= 32
combined_rhs <= 51
block_r <= 64
```

The new target is at most 30 active local rows. The 32-row number is a ceiling, not permission to add two new behavior constraints.

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_solver.py -k "compile_budget" -q
```

Expected: PASS; unsafe 33-row input is rejected before any CUDA call.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py \
  Go2Pvcnn/extension/joint_mpc_rti/solver/primal_dual_ilqr.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_solver.py
git commit -m "feat: enforce fixed-shape trajectory active bounds"
```

---

### Task 9: Implement The H30/32 Associative Trajectory Solver

**Files:**
- Create: `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/solver/associative_scan.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py`

- [ ] **Step 1: Write RED associativity and dense parity tests**

```python
def test_conditional_factor_combine_is_associative():
    a, b, c = random_spd_conditional_factors(dtype=torch.float64)
    assert_factor_close(combine(combine(a, b), c), combine(a, combine(b, c)), atol=1e-10)


@pytest.mark.parametrize("batch", [1, 7, 40])
def test_h30_scan_matches_dense_active_kkt(batch):
    qp = random_feasible_h30_qp(batch=batch)
    scan = solve_trajectory_qp_scan(qp)
    dense = solve_trajectory_qp_dense_reference(qp)
    assert (scan.direction - dense.direction).abs().max() < 2e-5
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py -q
```

Expected: FAIL because the direct-Z H30/32 solver does not exist.

- [ ] **Step 3: Implement local separator factors**

Use `y_k=[delta_z_(k-1),delta_z_k] in R^36`, convert the block-pentadiagonal QP and active local KKT rows into 30 interval conditional factors, and append two identity/no-cost padding factors. Padding may not alter original variables.

- [ ] **Step 4: Implement the explicit five-level tree**

```python
level1 = combine_pairs(factors)   # 32 -> 16
level2 = combine_pairs(level1)    # 16 -> 8
level3 = combine_pairs(level2)    # 8 -> 4
level4 = combine_pairs(level3)    # 4 -> 2
root = combine_pairs(level4)      # 2 -> 1
direction = recover_all_nodes(root, saved_levels)
```

Do not call the generic PyTorch 2.7 `associative_scan`; its symbolic-vmap batched matmul path is a known failure in this repository.

- [ ] **Step 5: Run CPU parity and monitored B=1 CUDA compile smoke**

CPU:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py -q
```

CUDA, only after CPU passes:

```bash
timeout 120s env PYTHONPATH=Go2Pvcnn MPC_TEST_DEVICE=cuda:0 \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py -k cuda_b1_compile -q
```

Expected: CPU parity PASS. CUDA PASS or a recorded environment skip; timeout/resource growth is a failure, not a skip.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py \
  Go2Pvcnn/extension/joint_mpc_rti/solver/associative_scan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py
git commit -m "feat: solve h30 trajectory qp with associative scan"
```

---

### Task 10: Replace Line Search With The Approved Five-Candidate Rule

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py`
- Delete old constraint-selection tests from: `Go2Pvcnn/tests/joint_mpc_rti/test_solver.py`

- [ ] **Step 1: Write RED rule tests**

```python
def test_line_search_builds_five_state_candidates_and_selects_lowest_loss():
    result = parallel_line_search(nominal, direction, objective, limits, dt=0.02)
    assert result.candidates.shape == (B, 5, 31, 18)
    assert result.alphas.tolist() == [1.0, 0.5, 0.25, 0.125, 0.0]
    assert result.selected_loss.equal(result.candidate_loss.min(dim=1).values)


def test_line_search_filters_only_nonfinite_joint_position_and_velocity():
    assert FILTER_NAMES == ("finite", "joint_position", "joint_velocity")


def test_equal_loss_prefers_larger_alpha():
    nominal = torch.zeros(1, 31, 18)
    direction = torch.zeros_like(nominal)
    lower = torch.tensor((-1.0472, -0.6632, -2.721) * 4)
    upper = torch.tensor((1.0472, 2.966, -0.837) * 4)
    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda z: torch.zeros(z.shape[0], device=z.device),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=torch.full((12,), 30.0),
        dt=0.02,
        tie_tolerance=1e-7,
    )
    assert result.alpha.eq(1.0).all()
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py -q
```

Expected: FAIL because current line search ranks feasibility/constraint restoration and uses old alphas.

- [ ] **Step 3: Implement the exact rule**

```python
ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0)
candidate = nominal[:, None] + alpha[None, :, None, None] * direction[:, None]
finite = torch.isfinite(candidate).all(dim=(2, 3))
position_ok = joint_position_valid(candidate[..., 6:], limits)
velocity_ok = joint_velocity_valid(candidate[..., 6:], limits, dt)
valid = finite & position_ok & velocity_ok
loss = seven_loss_objective(candidate.reshape(B * 5, 31, 18)).reshape(B, 5)
selectable = torch.where(valid, loss, torch.inf)
minimum = selectable.amin(dim=1, keepdim=True)
tie = selectable <= minimum + cfg.solver.line_search_tie_tolerance
selected_index = tie.to(torch.int64).argmax(dim=1)  # descending alpha order
```

`alpha=0` nominal must be feasible by construction and is the only fallback. Do not return constraint violation fields.

- [ ] **Step 4: Delete conflicting old tests and run GREEN**

Delete tests that require safer-base selection, required-control injection, constraint-component restoration, collision hard gates, recovery fallback, or post-candidate FK/KKT projection. Retain generic finite, shape, tie, position, and velocity tests under `test_line_search_v2.py`.

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_line_search_v2.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_solver.py
git commit -m "refactor: use five-candidate loss-only line search"
```

---

### Task 11: Route One RTI Through The New Planner And Delete Old Production Logic

**Files:**
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/solver/sqp_rti.py`
- Rewrite: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/runtime/cuda_graph.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/integration/reference_adapter.py`
- Modify: `Go2Pvcnn/extension/joint_mpc_rti/diagnostics/validation.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py`

- [ ] **Step 1: Write RED end-to-end one-RTI tests**

```python
def test_planner_runs_one_nominal_linearize_scan_search_and_publishes_x1(monkeypatch):
    calls = install_pipeline_spies(monkeypatch)
    result = planner_step(measured, command, field, solver_state=None, cfg=cfg)
    assert calls == ["nominal", "linearize", "scan", "line_search"]
    assert result.full_trajectory.state.shape == (B, 31, 18)
    assert result.pending_reference.target_step == 1
    torch.testing.assert_close(result.pending_reference.root_pos_w, result.full_trajectory.state[:, 1, :3])


def test_planner_source_has_no_old_repair_calls():
    source = inspect.getsource(planner_step)
    forbidden = ("recovery", "startup", "restore_candidate", "minimum_norm", "enforce_first_stance")
    assert not any(name in source for name in forbidden)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py -q
```

Expected: FAIL on the old 5000-line planner path.

- [ ] **Step 3: Implement the short orchestration**

```python
def step(measured_state, command_body, terrain_field, solver_state, cfg):
    batch = measured_state.batch_size
    if solver_state is None:
        previous = JointMpcRtiSolverState(
            trajectory=measured_state.as_vector()[:, None].expand(-1, 31, -1).clone(),
            gait_phase=torch.zeros(batch, dtype=torch.long, device=measured_state.device),
            valid=torch.zeros(batch, dtype=torch.bool, device=measured_state.device),
        )
    else:
        previous = solver_state
    nominal = build_nominal(
        measured_state,
        command_body,
        terrain_field,
        previous.gait_phase,
        previous,
        cfg,
    )
    context = build_loss_context(nominal, measured_state, command_body, terrain_field, cfg)
    update = sqp_rti_update(nominal.state, context, cfg)
    state = update.state
    geometry = go2_fk(state[..., :3], state[..., 3:6], state[..., 6:])
    derived_velocity = (state[:, 1:] - state[:, :-1]) / float(cfg.runtime.dt)
    trajectory = JointMpcRtiTrajectory(
        state=state,
        derived_velocity=derived_velocity,
        foot_pos_w=geometry.foot_pos_w,
        contact_state=nominal.contact_state,
        valid=nominal.valid,
        fallback=update.used_nominal,
        status=update.status,
        line_search_alpha=update.alpha,
        loss_breakdown=update.loss_breakdown,
    )
    pending = JointMpcPendingReference(
        root_pos_w=state[:, 1, :3],
        root_rpy_w=state[:, 1, 3:6],
        joint_angles=state[:, 1, 6:],
        foot_pos_w=geometry.foot_pos_w[:, 1],
        contact_state=nominal.contact_state[:, 1],
        valid=nominal.valid,
        target_step=1,
    )
    next_solver_state = JointMpcRtiSolverState(
        trajectory=state,
        gait_phase=(previous.gait_phase + 1) % 24,
        valid=nominal.valid,
    )
    return JointMpcRtiStepResult(
        full_trajectory=trajectory,
        pending_reference=pending,
        solver_state=next_solver_state,
    )
```

`sqp_rti_update` performs exactly one linearization, one active-set scan QP solve, and one line search.

- [ ] **Step 4: Update runtime and external ABI**

CUDA Graph fixed state becomes only previous trajectory, phase, and valid. Reference adapter consumes `state`, `foot_pos_w`, and `contact_state`; diagnostics consume `derived_velocity` instead of optimized `control`. Batch changes still require manager/cache/graph rebuild.

- [ ] **Step 5: Delete old production helpers and modules**

Remove old planner recovery/startup/projection/KKT/collision-restoration helpers and old control rollout dependencies. Delete `model/dynamics.py`, old control `model/rollout.py`, `losses/semantic.py`, `losses/clearance.py`, and `losses/rollout_objective.py` only after `rg` shows no production or external consumer. Keep low-level fixed solve kernels if `trajectory_scan.py` imports them.

Run:

```bash
rg -n "recovery|startup_root|restore_candidate|minimum_norm_leg|enforce_first_stance|adaptive_contact|constraint_violation" \
  Go2Pvcnn/extension/joint_mpc_rti
```

Expected: no production matches except explicit error text in migration guards, if retained.

- [ ] **Step 6: Run focused integration GREEN**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py -q
```

Expected: PASS for B=1/40, x0 injection, x1 publication, warm shift, reset, and fixed-shape runtime.

- [ ] **Step 7: Commit**

```bash
git add -A Go2Pvcnn/extension/joint_mpc_rti Go2Pvcnn/tests/joint_mpc_rti
git commit -m "refactor: route joint mpc through pure kinematic rti"
```

Checkpoint: do not start behavior matrices until Tasks 1-11 and all focused CPU tests pass.

---

### Task 12: Rebuild Shared Metrics, Applicability, And Monitored Test Runner

**Files:**
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/acceptance_thresholds.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py`
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py`
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_process_watchdog.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_run_joint_acceptance.py`

- [ ] **Step 1: Write RED applicability and watchdog tests**

```python
def test_flat_marks_only_small_specific_metrics_not_applicable():
    report = evaluate_trace(flat_trace, scenario="flat")
    assert report.metric("joint_position_violation").applicable
    assert report.metric("stance_ground_gap").applicable
    assert not report.metric("strict_cross_success").applicable
    assert report.metric("strict_cross_success").na_reason == "no small obstacle in flat scenario"


def test_small_includes_every_flat_metric_plus_small_metrics():
    assert applicable_metrics("flat") < applicable_metrics("small")


def test_watchdog_terminates_only_child_process_group_on_timeout(tmp_path):
    result = run_monitored([sys.executable, "-c", "import time; time.sleep(60)"], timeout_s=0.2)
    assert result.terminated
    assert result.reason == "hard_timeout"
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_run_joint_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_process_watchdog.py -q
```

Expected: FAIL because old metrics contain recovery fields and heavy probes lack one supervisor.

- [ ] **Step 3: Define one trace and per-metric applicability**

The trace must include actual root/joint/foot, command, gait phase, terrain surface, small signed distance, per-part collision, line alpha, nominal root, validity, and timestamps. Remove recovery/extension/liftoff-guard fields. Every metric result stores value, numerator, denominator, valid count, applicability, N/A reason, threshold, pass, and worst-case key.

For exactly zero translation command, `stance_root_carry_ratio_abs` is mathematically N/A because its denominator is near-zero root displacement. Keep absolute stance slip max/mean, stationary ratio, anchor residual, ground gap/penetration, drift, joints, and numerical metrics applicable.

- [ ] **Step 4: Preserve the formal 275 command matrix**

```python
VX = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8, -1.0, 1.0)
VY = (0.0, -0.3, 0.3, -0.5, 0.5)
YAW = (0.0, -0.5, 0.5, -1.0, 1.0)
COMMANDS = tuple(itertools.product(VX, VY, YAW))
assert len(COMMANDS) == 275
```

Small scenes add shape, phase, and offset dimensions. Pure yaw/zero translation keeps drift, yaw, stance, joint, collision, and numerical metrics while translation-direction/crossing metrics receive explicit mathematical N/A.

- [ ] **Step 5: Implement the process-group watchdog**

Use `subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)`, parent-side heartbeat every 5 seconds, and `os.killpg(child.pid, signal.SIGTERM)` followed by task-group-only `SIGKILL` after grace. Poll wall time, process-tree RSS, child `ptxas` RSS, selected-GPU PID memory/utilization, available memory, and swap. Enforce the approved thresholds: B1 120 s, ptxas 8 GiB, tree 16 GiB, available drop 16 GiB, swap delta 256 MiB, and 30 s CPU-only compiler growth without task GPU progress.

- [ ] **Step 6: Delete silent heavy tests after coverage mapping**

Delete or rewrite direct long-running pytest/probe routes in `test_behavior.py`, `small_obstacle_attitude_probe.py`, `small_obstacle_stop_probe.py`, and old `test_coupled_gait.py`. Keep scene construction helpers only if `run_joint_acceptance.py` invokes them under the watchdog and they emit per-cell progress. Do not remove any metric listed in Sections 10-13 of the approved design.

- [ ] **Step 7: Run GREEN**

Run the Step 2 command. Expected: PASS without launching CUDA or Isaac Lab.

- [ ] **Step 8: Commit**

```bash
git add -A Go2Pvcnn/tests/joint_mpc_rti
git commit -m "test: unify flat-small metrics and monitored runner"
```

---

### Task 13: Close The Flat Behavior Gate First

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_flat_acceptance.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`
- Tune only: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Create evidence: `notes/log/2026-07-20-joint-mpc-rti-kinematic-flat-gate.md`

- [ ] **Step 1: Write RED flat acceptance aggregation tests**

```python
def test_flat_gate_rejects_any_failed_applicable_metric():
    report = fake_flat_report(fail=("joint_velocity_violation", "stance_ground_gap"))
    assert not require_flat_gate(report).passed


def test_flat_gate_does_not_require_small_only_metrics():
    report = passing_flat_report_with_small_metrics_na()
    assert require_flat_gate(report).passed
```

- [ ] **Step 2: Run short ranked flat cells under watchdog**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py -- \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  --stage flat --ranked-cells 3 --steps 96 --heartbeat-seconds 5
```

Expected before tuning: finite report with explicit per-metric failures, not a silent process.

- [ ] **Step 3: Tune only the approved parameters in this order**

1. Joint position/velocity/safe-margin failures: reduce trust region, then increase posture weight.
2. Stance slip/gap failures: increase contact weight, then contact ground/slip subweight.
3. Foot-before-root failures: increase swing-speed early-phase subweight and reduce command early-swing phase weight; do not add foot-lead loss.
4. Root tracking failures: increase command weight after stance and joint gates remain green.
5. Oscillation/jump failures: increase first/second smooth subweights.

After each edit rerun the same ranked cells and record the changed parameter plus all metric deltas.

- [ ] **Step 4: Run the complete 275-command flat matrix**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py -- \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  --stage flat --all-commands --steps 144 --heartbeat-seconds 5 \
  --report-json /tmp/joint_mpc_flat_report.json
```

Expected: every applicable metric passes in every command cell; small-only metrics are explicit N/A; no average masks a failed cell.

- [ ] **Step 5: Run real viewer actual-state flat smoke**

Use the monitored runner around `joint_mpc_rti_viewer_reproduction_probe.py --scenario flat`; require actual root/joint/foot finite, x1 publication equality, and the same flat metrics. Planner-internal state is not sufficient evidence.

- [ ] **Step 6: Record evidence and commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/config.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_flat_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  notes/log/2026-07-20-joint-mpc-rti-kinematic-flat-gate.md \
  notes/log/index.md notes/todo.md notes/todo/T302v-joint-mpc-rti-gpu.md
git commit -m "test: close pure kinematic flat behavior gate"
```

Checkpoint: do not run or tune the formal small matrix until flat is fully green.

---

### Task 14: Close The Small-Obstacle Behavior Gate

**Files:**
- Create: `Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`
- Tune only: `Go2Pvcnn/extension/joint_mpc_rti/config.py`
- Create evidence: `notes/log/2026-07-20-joint-mpc-rti-kinematic-small-gate.md`

- [ ] **Step 1: Write RED strict-crossing and universal-metric tests**

```python
def test_small_gate_requires_all_flat_metrics_and_small_metrics():
    report = passing_crossing_report()
    report.cells[0].metrics["stance_ground_gap"].passed = False
    assert not require_small_gate(report).passed


def test_bypass_or_stop_is_not_strict_crossing_success():
    assert not strict_crossing_event(bypass_trace).success
    assert not strict_crossing_event(stopped_trace).success
```

- [ ] **Step 2: Run ranked shape/phase/offset cells**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py -- \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  --stage small --ranked-cells 3 --steps 160 --heartbeat-seconds 5
```

Expected before tuning: finite per-part/per-phase collision and strict-cross opportunity report.

- [ ] **Step 3: Tune only approved terrain and existing loss parameters**

1. Swing foot/link collision: increase terrain weight or clearance subweight, then `h_swing`, then `small_sigma_m`.
2. Touchdown/stance on small: increase terrain touchdown penalty, `small_gain`, or a non-weakened small margin.
3. Root roll/pitch/lateral excess: increase posture weight or reduce root trust region.
4. Crossing stalls while safety is green: increase step weight/reference scale, then swing-speed weight.
5. Joint or stance regression: undo the last small tuning and rebalance existing weights; never add recovery or hard gates.

After every edit rerun ranked small cells and the ranked flat regression cells.

- [ ] **Step 4: Run formal small matrix**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py -- \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  --stage small --all-commands --all-shapes --all-phases --all-offsets \
  --steps 160 --heartbeat-seconds 5 \
  --report-json /tmp/joint_mpc_small_report.json
```

Expected: all applicable flat metrics pass; foot/knee/calf/thigh/base collision rate is zero per valid cell; penetration `<=1mm`; touchdown/stance-on-small and airborne touchdown are zero; root limits pass; strict crossing is `>=95%` overall and `>=90%` per shape-speed.

- [ ] **Step 5: Run real viewer small crossing**

Run actual-state crossing under the watchdog for the representative signed commands. Require the same detector and thresholds as the planner matrix, not a private viewer success flag.

- [ ] **Step 6: Record evidence and commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti/config.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  notes/log/2026-07-20-joint-mpc-rti-kinematic-small-gate.md \
  notes/log/index.md notes/todo.md notes/todo/T302v-joint-mpc-rti-gpu.md
git commit -m "test: close pure kinematic small obstacle gate"
```

---

### Task 15: Freeze Behavior And Run The Joint Flat-Small Gate

**Files:**
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_run_joint_acceptance.py`
- Create evidence: `notes/log/2026-07-20-joint-mpc-rti-kinematic-flat-small-joint-gate.md`

- [ ] **Step 1: Add the combined-stage test**

```python
def test_joint_stage_requires_fresh_flat_and_small_reports_from_same_ref():
    result = require_joint_gate(flat_report, small_report)
    assert result.code_ref == flat_report.code_ref == small_report.code_ref
    assert result.passed == (flat_report.passed and small_report.passed)
```

- [ ] **Step 2: Run all short CPU tests before the expensive joint run**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti -m "not cuda and not isaac and not performance" -q
```

Expected: all pass; no test runs silently for more than 30 seconds.

- [ ] **Step 3: Run fresh flat and small reports without parameter changes**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py -- \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  --stage flat-small --formal --heartbeat-seconds 5 \
  --report-json /tmp/joint_mpc_flat_small_report.json
```

Expected: latest flat and small reports share one HEAD/config/detector and both pass all applicable per-cell metrics.

- [ ] **Step 4: Freeze the behavior configuration hash**

Write the config, geometry, detector, command matrix, map parameters, and code ref hashes into the report. Stage B must assert the same behavior hash before timing.

- [ ] **Step 5: Commit the joint gate evidence**

```bash
git add Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_run_joint_acceptance.py \
  notes/log/2026-07-20-joint-mpc-rti-kinematic-flat-small-joint-gate.md \
  notes/log/index.md notes/todo.md notes/todo/T302v-joint-mpc-rti-gpu.md
git commit -m "test: freeze passing flat-small joint mpc behavior"
```

Checkpoint: Stage B is unauthorized until this task is green on a fresh report.

---

### Task 16: Meet Stage B And Perform The Same-Candidate Final Rerun

**Files:**
- Rewrite: `Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_full_refresh_probe.py`
- Modify: `Go2Pvcnn/tests/joint_mpc_rti/test_performance.py`
- Optimize only as profile proves necessary:
  - `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py`
  - `Go2Pvcnn/extension/joint_mpc_rti/terrain/query.py`
  - `Go2Pvcnn/extension/joint_mpc_rti/terrain/cuda_edt.py`
  - `Go2Pvcnn/extension/joint_mpc_rti/runtime/cuda_graph.py`
  - `Go2Pvcnn/extension/joint_mpc_rti/losses/objective.py`
- Create evidence: `notes/log/2026-07-20-joint-mpc-rti-kinematic-final-behavior-performance.md`

- [ ] **Step 1: Write RED Stage B contract tests**

```python
def test_full_refresh_probe_uses_frozen_behavior_and_realistic_workload():
    args = probe_defaults()
    assert args.num_envs == 1024
    assert args.horizon == 30
    assert args.steps == 1000
    assert args.small_footprint == (11, 11)
    assert args.large_footprint == (41, 41)
    assert args.include_exact_field and args.include_all_five_candidates


def test_stage_b_rejects_behavior_hash_mismatch():
    with pytest.raises(RuntimeError, match="behavior hash"):
        run_probe(args_with_modified_loss_weight())
```

- [ ] **Step 2: Run static/resource audit and B=1 compile smoke**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_performance.py -q
```

Then run B=1 under `run_monitored_joint_mpc.py` with a 120-second hard timeout. Expected: fixed dimensions pass, no resource trigger, finite output.

- [ ] **Step 3: Profile in the approved order**

Collect CUDA-event timings for exact field refresh, grouped semantic convolution plus Scharr, packed query, seven-loss evaluation/GGN, one free plus two active-set scan solves, five-candidate nonlinear score, and graph replay overhead. Optimize only the largest measured component, in this order of allowed transformations:

1. Preallocate and reuse all H30/32 factor, active-mask, candidate, and query workspaces.
2. Fuse the fixed four-channel semantic convolution, propagated-height numerator, Scharr gradients, packed query gathers, and gradient rotation without changing kernels or sampled points.
3. Lower the explicit five-level scan to fixed Triton combines while preserving dense parity.
4. Reuse FK/query intermediates across the five candidate scores where states are identical.
5. Capture the complete fixed-address refresh in CUDA Graph after all parity tests pass.

After each optimization rerun trajectory-scan parity, seven-loss equality, five-candidate selection, and a 1/40/512/1024 finite smoke.

- [ ] **Step 4: Run the formal idle-GPU Stage B gate**

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py -- \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_full_refresh_probe.py \
  --device cuda:0 --num-envs 1024 --horizon 30 --steps 1000 \
  --require-behavior-report /tmp/joint_mpc_flat_small_report.json
```

Expected: total `<=5000ms`, mean `<=5ms/update`, field version `+1000`, `nonfinite_count=0`, exact field and complete MPC included, no watchdog trigger.

- [ ] **Step 5: Freshly rerun behavior and performance on the final HEAD**

Run Task 15 Step 3 again to a new report, then Task 16 Step 4 against that new report. Any code/config change after either run invalidates both reports and requires both reruns.

- [ ] **Step 6: Run final source and repository checks**

```bash
rg -n "adaptive_contact|recovery_state|startup_root|restore_candidate|minimum_norm_leg|constraint_violation" \
  Go2Pvcnn/extension/joint_mpc_rti
git diff --check
```

Expected: no old production behavior path, and no whitespace errors. Confirm old silent probes/tests are deleted or runner-managed.

- [ ] **Step 7: Record evidence and commit**

```bash
git add Go2Pvcnn/extension/joint_mpc_rti \
  Go2Pvcnn/tests/joint_mpc_rti \
  notes/log/2026-07-20-joint-mpc-rti-kinematic-final-behavior-performance.md \
  notes/log/index.md notes/todo.md notes/todo/T302v-joint-mpc-rti-gpu.md
git commit -m "perf: close pure kinematic joint mpc final gates"
```

---

## Final Acceptance Checklist

- [ ] Production optimization variable is only `Z[B,31,18]`; any velocity tensor is derived diagnostics.
- [ ] Nominal cold/warm construction has no Python environment/time/leg loop and passes B=1/40/512/1024.
- [ ] Gait is fixed H30, period 24, swing/stance 12/12.
- [ ] Objective exposes exactly seven loss keys and no scenario-specific branch.
- [ ] Semantic observation uses fixed grouped convolution, propagated class height, Scharr XY gradients, and bilinear trajectory queries; optimizer terrain loss contains no raw-class hard branch.
- [ ] KKT contains only fixed `z0`, merged joint-position/trust bounds, joint-velocity bounds, and trust region.
- [ ] H30/32 associative scan matches dense active KKT within `2e-5`.
- [ ] Line search uses exactly five alphas, three filters, seven-loss argmin, and larger-alpha tie break.
- [ ] Old recovery/startup/projection/restoration/adaptive-contact production code is deleted.
- [ ] Shared applicability-aware JointMetrics checks every applicable metric in flat and small.
- [ ] Flat formal matrix passes before small formal matrix.
- [ ] Small formal matrix passes strict crossing and zero full-body collision without regressing flat metrics.
- [ ] Heavy tests run only under heartbeat, timeout, process-group, CPU/GPU/RSS/swap watchdog.
- [ ] Stage B uses the frozen behavior hash and passes realistic synchronous `1024 x H30 x 1000 <=5.0s`.
- [ ] Final behavior and Stage B reports come from the same final HEAD/config and are both fresh.
- [ ] Notes dashboard, branch page, log index, and per-gate evidence agree with code and reports.

## Stop Conditions

Stop implementation and return to design review instead of editing around the failure if any of these occurs:

- Nominal structure, gait timing, optimization variables, loss categories, line-search candidates/filters, or test thresholds appear insufficient and would need structural change.
- A new collision hard gate, recovery state, output projection, candidate repair, or scenario-specific loss seems necessary.
- Active KKT needs more than the inherited 32-row compile ceiling or pushes `BLOCK_N>32` / `BLOCK_R>64`.
- A long test cannot be placed under the approved watchdog or repeatedly triggers resource limits.
- Stage B appears to require reducing field footprint, geometry samples, candidates, horizon, environment count, or behavior coverage.

Only a documented, user-approved design amendment may cross these boundaries.
