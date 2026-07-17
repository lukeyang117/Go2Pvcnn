# Joint MPC RTI Root-Joint Coupled Gait Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a truly coupled H30 root-joint rolling RTI gait with adaptive per-leg touchdown and bounded root assistance, pass every inherited/new Stage A behavior metric, then make the realistic 1024-env synchronous field+H30 MPC complete 1000 updates in at most 5.0s and freshly rerun both gates on one final candidate.

**Architecture:** Keep `dt=0.02`, measured x0 and published x1, but fix production shape to `H=30`, nominal stance/swing to 15 frames each, and line-search alphas to `(1.0,0.5,0.25,0.1)`. Add per-leg contact/extension/recovery state, grounded-and-full-body-safe touchdown confirmation, complete root-joint/full-body GGN directions, continuously released root lateral/RPY assistance with per-frame and cumulative clamps, and one shared JointMetrics accumulator over the 275-command Cartesian matrix. Stage A must pass first; Stage B then freezes this exact H30 behavior and optimizes the realistic synchronous pipeline without relaxing the five-second gate; Stage C freshly reruns Stage A and Stage B on one final candidate.

**Tech Stack:** Python 3.10, PyTorch, torch.compile, CUDA/C++, SQP-RTI/GGN/Riccati, pytest, Isaac Lab in `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.

---

## File Map

- Create `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`: shared trace and metric accumulator.
- Create `Go2Pvcnn/tests/joint_mpc_rti/acceptance_thresholds.py`: applicability, thresholds and per-cell aggregation.
- Create `Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py`: flat/small/stop/large/step scenario rows.
- Create `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`: unified Stage A/Stage C runner.
- Create `Go2Pvcnn/tests/joint_mpc_rti/horizon_exploration.py`: deterministic H16-H50 candidate construction, screening reports and shortest-passing selection.
- Create `Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py` and `test_coupled_gait.py`.
- Modify `config.py`, `types.py`, `model/go2_kinematics.py`, `losses/command.py`, `losses/contact.py`, `planner.py`, `solver/primal_dual_ilqr.py`, `solver/sqp_rti.py`.
- Refactor existing crossing/stop/behavior probes to consume the shared metrics.
- Stage B may modify `terrain/csrc/work_efficient_edt_cuda.cu`, `terrain/cuda_edt.py` and the coupled solver only after fixed-H30 Stage A passes and H30 profiling identifies the largest component.

## Execution Status At Amendment

- Tasks 1-3 are committed as `b78d392`, `08a00a7`, `635eaad`.
- Tasks 4-5 have focused coupled tests passing but remain open until the inherited crossing/stop/packed-query gates recover.
- Tasks 6-7 and every Stage B/C task remain open.

## 2026-07-18 H30 Adaptive-Contact Amendment

This section supersedes the horizon-selection portions of Tasks 8-9 and every later reference to a variable `H_selected`. Tasks 1-7 remain inherited correctness work. The production and acceptance horizon is now exactly H30 with `half_cycle_steps=15`; H16 is diagnostic history only. Implementation must follow the approved Chinese HTML design at `docs/superpowers/specs/2026-07-18-joint-mpc-rti-h30-adaptive-contact-root-assist-design.html`.

The mandatory completion order is:

1. Stage A: make H30 behavior, safety, JointMetrics, real viewer and all inherited regressions pass together.
2. Stage B: only after Stage A is green, make realistic idle-GPU `1024 x H30 x 1000 <=5000ms` pass without changing Stage A contracts.
3. Stage C: on one final candidate, freshly rerun complete Stage A and complete Stage B; both latest results must pass.

### Task 8A: Fix The Production H30 Contract

**Files:** modify `Go2Pvcnn/extension/joint_mpc_rti/config.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_horizon_exploration.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_solver.py`.

- [ ] Write RED assertions:

```python
def test_production_horizon_is_h30_full_cycle():
    cfg = JointMpcRtiCfg()
    assert cfg.runtime.horizon_steps == 30
    assert cfg.gait.half_cycle_steps == 15
    assert cfg.runtime.horizon_steps == 2 * cfg.gait.half_cycle_steps
    assert cfg.solver.line_search_alphas == (1.0, 0.5, 0.25, 0.1)
```

- [ ] Run:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_horizon_exploration.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_solver.py -q
```

Expected before implementation: the H30 production-default assertion fails.

- [ ] Set the exact H30/15/15/four-alpha defaults. Remove production selection of shorter candidates; retain exploration helpers only as diagnostic utilities.
- [ ] Run the same tests and require pass. Commit `feat: fix joint mpc production horizon to h30`.

### Task 8B: Per-Leg Contact State And Fixed-Shape Schedule

**Files:** modify `Go2Pvcnn/extension/joint_mpc_rti/types.py`, `model/gait_schedule.py`, `planner.py`; create/modify `Go2Pvcnn/tests/joint_mpc_rti/test_coupled_gait.py`.

- [ ] Write RED tests that construct `[B,4]` contact, phase age, extension age, stance age and recovery tensors and require a fixed `[B,31,4]` mask. Verify one diagonal leg can confirm stance while its mate remains swing, and that a future liftoff is blocked when it would leave fewer than two reliable stance legs.
- [ ] Run `pytest Go2Pvcnn/tests/joint_mpc_rti/test_coupled_gait.py -k 'per_leg or liftoff_guard' -q`; expect failure because solver state and scheduler are global-phase only.
- [ ] Extend `JointMpcRtiSolverState` with:

```python
contact_state: Tensor | None = None
phase_age: Tensor | None = None
swing_extension_age: Tensor | None = None
stance_age: Tensor | None = None
recovery_state: Tensor | None = None
```

- [ ] Implement tensor-only schedule generation; no Python branching by environment. Preserve fixed shapes for CUDA Graph. Run focused tests and commit `feat: add per leg joint mpc contact scheduler`.

### Task 8C: Grounded And Full-Body-Safe Touchdown

**Files:** modify `planner.py`, `types.py`, `runtime/cuda_graph.py`, `model/go2_kinematics.py`; modify `test_coupled_gait.py`, `test_rolling_runtime.py`, `test_losses.py`.

- [ ] Write RED cases for: airborne scheduled touchdown; grounded foot on small semantic; foot safe but knee/calf/thigh collision; foot/leg safe but base collision; safe current touchdown with x1/x2 collision; and fully safe touchdown.
- [ ] Require the first five cases to remain non-stance without an anchor update. Require the safe case to set `anchor_xy=current_foot_xy` and `anchor_z=terrain_height+foot_contact_offset`.
- [ ] Require independent touchdown, extension ages `1..10`, no forced stance at age 10, grounded-but-unsafe recovery, and root progress scale `clamp(1-age/10,0,1)`.
- [ ] Add graph-safe clone/copy/reset handling for all new solver-state tensors.
- [ ] Run focused contact/runtime/CUDA Graph tests and commit `feat: confirm safe joint mpc touchdown`.

### Task 8D: Bounded Root Lateral And RPY Assistance

**Files:** modify `config.py`, `planner.py`, `losses/posture.py`; modify `test_coupled_gait.py`, `test_solver.py`.

- [ ] Write RED finite-difference/LQ tests proving small-obstacle residuals can generate root lateral, roll, pitch and yaw directions while root Z remains excluded. Verify early swing uses less root assistance than mid-swing and reduced joint margin continuously increases assistance.
- [ ] Add exact config bounds:

```python
root_lateral_offset_limit_m = 0.06
root_lateral_velocity_error_limit_mps = 0.20
root_roll_pitch_limit_rad = math.radians(6.0)
root_roll_pitch_rate_limit_rps = 0.6
root_yaw_error_limit_rad = math.radians(10.0)
root_yaw_rate_error_limit_rps = 0.8
```

- [ ] Replace pure command-axis root XY projection with `P_parallel + w_assist * P_lateral`; scale root RPY collision Jacobians by the same continuous SDF/phase/reachability weight.
- [ ] Project every line-search candidate through per-frame control clamps and integrated nominal-relative state clamps before nonlinear FK/merit evaluation. Run focused tests and commit `feat: plan bounded root obstacle assistance`.

### Task 8E: Full-Body Small-Obstacle LQ Directions

**Files:** modify `model/go2_kinematics.py`, `planner.py`, `losses/semantic.py`, `losses/rollout_objective.py`; modify `test_kinematics_gait.py`, `test_losses.py`, `test_solver.py`.

- [ ] Write RED tests for nonzero, finite and correctly directed foot/knee/calf/thigh/base signed-distance gradients and Gauss-Newton blocks. Verify each part changes the LQ direction, not only final merit.
- [ ] Add explicit knee and base sample Jacobians to the packed geometry query and small-obstacle linearization. Use the same radii/top-height overlap detector in LQ, merit, line-search violation and JointMetrics.
- [ ] Verify root-leg cross blocks remain symmetric and dense parity stays within `2e-5`. Commit `feat: optimize all joint mpc body clearances`.

### Task 8F: Complete JointMetrics And 275-Command Matrix

**Files:** modify `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`; create/modify `acceptance_thresholds.py`, `scenario_matrix.py`, `run_joint_acceptance.py`, `test_joint_metrics.py`, `small_obstacle_attitude_probe.py`.

- [ ] Write RED contract tests requiring exactly:

```python
VX = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8, -1.0, 1.0)
VY = (0.0, -0.3, 0.3, -0.5, 0.5)
YAW = (0.0, -0.5, 0.5, -1.0, 1.0)
assert len(tuple(product(VX, VY, YAW))) == 275
```

- [ ] Extend `JointMetricTrace` and the accumulator with every field in design section 7, including root roll/pitch mean/P95/abs-max, root lateral/yaw deviations, touchdown readiness/extension/recovery, reliable support/liftoff guard, foot/knee/calf/thigh/base phase collisions, joint/root clamp hits and alpha histograms/rejection reasons.
- [ ] Require small-obstacle per-cell root roll/pitch abs max `<=6deg`, every part collision rate `0`, airborne/unsafe/forced touchdown `0`, alpha=0 global `<=5%`, per-cell `<=10%`, consecutive run `<=2`, plus all inherited thresholds.
- [ ] Ensure reports carry raw `(vx,vy,yaw_rate)`, numerator, denominator, valid count, N/A reason and worst key. Run metric tests and commit `test: unify h30 adaptive contact metrics`.

### Task 9A: Close Stage A On Fixed H30

**Files:** only files implicated by focused RED failures; update `run_joint_acceptance.py` and create a Stage A evidence log.

- [ ] Run focused tests after each single-cause change, then run the entire fixed-H30 matrix. No H16/H20 fallback is allowed.
- [ ] Require all 275 commands in every applicable flat/small/stop/large/terrain matrix, all five small shapes, phase/placement/recovery coverage, and no omitted universal metric.
- [ ] Run all joint MPC tests, all inherited legacy MPC/reward/viewer tests, real Isaac nine-command plus high-speed/mixed playback, and manager construction for `1/40/512/1024`.
- [ ] Record every worst metric and failure cell. Stage A passes only when the complete latest run is green. Commit `fix: close h30 joint mpc stage a`.

### Task 10A: Stage B H30 Performance Gate

This task is blocked until Task 9A passes. It supersedes every Stage B reference to `H_selected` with H30 and preserves the recorded CUDA Graph/Triton progress.

- [ ] On idle GPU run realistic synchronous exact signed field + H30 MPC for 1024 environments and 1000 updates with 11x11/41x41 footprints, four alpha candidates and all Stage A geometry/losses.
- [ ] Require total `<=5000ms`, mean `<=5ms`, field version `+1000`, nonfinite `0`, stable memory. Optimize only measured bottlenecks and rerun Stage A after every accepted performance change.

### Task 14A: Final Same-Candidate Joint Verification

This task is blocked until Task 10A passes.

- [ ] Record final HEAD and fixed H30 configuration.
- [ ] Freshly rerun complete Stage A.
- [ ] Freshly rerun complete Stage B on the same HEAD.
- [ ] Mark completion only if both latest reports pass; otherwise return to the failing stage and later rerun both.

## Stage A — Behavior And Joint Metrics

### Task 1: Shared JointMetrics

**Files:** create `joint_metrics.py`, `acceptance_thresholds.py`, `test_joint_metrics.py`.

- [ ] Write RED tests: a trace with `root_step=4mm`, `stance_foot_step=4mm`, zero swing-relative motion must fail `stance_root_carry_ratio`, `swing_active_motion_ratio` and startup lead; a crossing-success trace with 2mm stance slip must still fail its scenario cell.
- [ ] Run RED:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py -q
```

- [ ] Implement these fixed contracts:

```python
@dataclass(frozen=True)
class JointMetricTrace:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    command_body: Tensor
    foot_height_w: Tensor
    foot_small_distance_m: Tensor
    part_collision: dict[str, Tensor]
    valid: Tensor

THRESHOLDS = {
    "stance_xy_slip_max_m": ("le", 0.0005),
    "stance_xy_slip_mean_m": ("le", 0.0002),
    "stance_stationary_ratio": ("ge", 1.0),
    "stance_root_carry_ratio_abs": ("le", 0.10),
    "swing_active_motion_ratio": ("ge", 0.50),
    "foot_root_lead_time_min_ms": ("ge", 20.0),
    "foot_root_lead_time_max_ms": ("le", 80.0),
    "root_leak_before_foot_m": ("le", 0.0005),
}
```

- [ ] Keep `None` for mathematically inapplicable events; require explicit opportunity coverage and never treat N/A as success.
- [ ] Run GREEN and commit `test: unify joint mpc gait metrics`.

### Task 2: Complete Root-Joint Foot Jacobian

**Files:** modify `model/go2_kinematics.py`, `test_kinematics_gait.py`.

- [ ] Write RED finite-difference tests for output `[B,4,3,18]`: translation block is identity, unrelated leg blocks are zero, root RPY and corresponding leg blocks match central differences.
- [ ] Run RED with `pytest .../test_kinematics_gait.py -k complete_foot_jacobian -q`.
- [ ] Implement:

```python
J = zeros(B, 4, 3, 18)
J[..., :3] = I3
J[..., 3:6] = d_rotation_times_point_d_rpy(root_rpy_w, foot_body)
J[:, leg, :, 6 + 3*leg : 9 + 3*leg] = foot_jacobian_leg(...)
```

- [ ] Run the complete kinematics suite and commit `feat: add complete root joint foot jacobian`.

### Task 3: Correct Dense Coupled LQ Baseline

**Files:** modify `solver/primal_dual_ilqr.py`, `solver/sqp_rti.py`, `config.py`, `test_solver.py`.

- [ ] Write RED random-SPD LQ tests with nonzero root-leg cross terms; compare state/control direction to a direct dense reference and require both root and leg movement.
- [ ] Run RED with `pytest .../test_solver.py -k coupled -q`; current root/diagonal solver must disagree.
- [ ] Replace diagonal division of dense control Hessians with a batched Cholesky solve:

```python
chol, info = torch.linalg.cholesky_ex(Huu_regularized)
K = torch.cholesky_solve(Hux, chol)
k = torch.cholesky_solve(gu.unsqueeze(-1), chol).squeeze(-1)
```

- [ ] Add explicit `coupled_state_riccati=True`; never select the old Go2 block solver when coupling is enabled.
- [ ] Run all solver tests and commit `fix: preserve coupled lq cross blocks`.

### Task 4: Scheduled Stance Equality

**Files:** modify `types.py`, `config.py`, `planner.py`; create `test_coupled_gait.py`.

- [ ] Write RED: inject a 4mm stance anchor error; require nonzero symmetric root-leg Hessian, simultaneous root/joint correction, reduced residual after one RTI, and max rolling stance slip at most 0.5mm.
- [ ] Extend solver state:

```python
stance_dual: Tensor | None = None
command_start_age: Tensor | None = None
command_start_origin_w: Tensor | None = None
previous_command_body: Tensor | None = None
```

- [ ] Add equality penalty/dual-step/tolerance config and assemble:

```python
c = foot_w - stance_anchor_surface_w
g += J.transpose(-1, -2) @ (dual + rho * c)
H += rho * J.transpose(-1, -2) @ J
dual_next = dual + dual_step * rho * c_x1
```

- [ ] Put the same constraint violation in nonlinear merit diagnostics; do not postprocess x1.
- [ ] Run `test_coupled_gait.py` plus `test_behavior.py`; commit `feat: enforce coupled stance equalities`.

### Task 5: Horizon Command And Touchdown Targets

**Files:** modify `losses/command.py`, `losses/contact.py`, `planner.py`, `test_coupled_gait.py`.

- [ ] Write RED for forward/back/left/right/diagonal at 0.1/0.2/0.4m/s: every complete swing has positive relative-root progress, correct touchdown direction, mean active ratio at least 0.5 and monotonic speed-conditioned lead.
- [ ] Change the primary translation task to horizon average progress:

```python
v_avg_body = Rz(-yaw0) @ (root_xy_H - root_xy_0) / (H * dt)
r_progress = v_avg_body - command_xy
```

- [ ] Build `[B,H+1,4,3]` swing targets from current feet, nominal footprint, command-world displacement over remaining swing time, terrain height and the existing clearance envelope. Use complete Jacobians in GGN and the same residual in merit.
- [ ] Lower per-node root velocity tracking to regularization strength; keep zero/yaw semantics.
- [ ] Run command/coupled tests and commit `feat: couple command touchdown and root progress`.

### Task 6: Foot Leads Root At Startup

**Files:** modify `types.py`, `config.py`, `planner.py`, `test_coupled_gait.py`.

- [ ] Write RED zero→move and stop→restart tests for forward/back/lateral/diagonal. Measure published poses and require 20–80ms lead, root leak at most 0.5mm, scheduled swing leader, correct direction and valid stance anchors.
- [ ] Detect per-env command onset with translational deadband/direction-change cosine.
- [ ] During startup age zero, constrain command-axis root x1 motion to at most 0.5mm while swing target remains active; release after 1mm relative foot motion and force release at 80ms. Do not re-arm every half cycle.
- [ ] Run coupled, zero and stop regressions; commit `feat: make swing feet lead root startup`.

### Task 7: Scenario × Metric Cartesian Gate

**Files:** create `scenario_matrix.py`, `run_joint_acceptance.py`; modify crossing/stop probes and `test_behavior.py`.

- [ ] Write RED proving a scenario cannot omit a universal metric: crossing succeeds and collision is zero but stance slip fails; stop recovers support but root-first restart fails.
- [ ] Refactor flat, native small crossing, native stop, large obstacle and up/down-step paths to return raw traces plus scenario metadata.
- [ ] Feed every trace through one accumulator. Preserve old convenience properties but derive them from shared values.
- [ ] Implement JSON per-cell/worst/global reporting and nonzero exit on any applicable failure.
- [ ] Run all `Go2Pvcnn/tests/joint_mpc_rti` tests; commit `test: joint all mpc scenario metrics`.

### Task 8: Full-Cycle Horizon And RTI Direction Exploration

> **Superseded audit section:** Task 8 through the original Task 14 below are retained only to explain the 2026-07-17 plan history. Do not execute their H16-H50 selection or variable `H_selected` instructions. Execute Tasks 8A-9A, then blocked Tasks 10A and 14A from the 2026-07-18 amendment instead.

**Files:** create `horizon_exploration.py`, `test_horizon_exploration.py`; modify `config.py`, `solver/line_search.py`, `solver/sqp_rti.py`, `test_solver.py` only when a focused RED identifies that component.

- [ ] Write RED tests for the exploration contract:

```python
def test_horizon_candidates_cover_one_full_fixed_trot_cycle():
    candidates = make_horizon_candidates((16, 20, 24, 30, 40, 50))
    assert [(c.horizon_steps, c.half_cycle_steps) for c in candidates] == [
        (16, 8), (20, 10), (24, 12), (30, 15), (40, 20), (50, 25)
    ]
    assert all(c.horizon_steps == 2 * c.half_cycle_steps for c in candidates)

def test_selection_uses_shortest_candidate_passing_every_metric():
    reports = [report(16, collision=1), report(20, collision=0), report(24, collision=0)]
    assert select_shortest_passing(reports).horizon_steps == 20
```

- [ ] Run RED:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_horizon_exploration.py -q
```

Expected: FAIL because `make_horizon_candidates` and `select_shortest_passing` do not exist.

- [ ] Implement immutable candidate/report types and reject odd horizons or mismatched full-cycle configurations:

```python
@dataclass(frozen=True)
class HorizonCandidate:
    horizon_steps: int
    half_cycle_steps: int

def make_horizon_candidates(horizons: tuple[int, ...]) -> tuple[HorizonCandidate, ...]:
    if any(h < 16 or h > 50 or h % 2 for h in horizons):
        raise ValueError("horizon candidates must be even and within [16, 50]")
    return tuple(HorizonCandidate(h, h // 2) for h in horizons)

def select_shortest_passing(reports: Sequence[HorizonReport]) -> HorizonReport:
    passing = [report for report in reports if report.all_applicable_metrics_pass]
    if not passing:
        raise RuntimeError("no horizon candidate passes the complete Stage A contract")
    return min(passing, key=lambda report: report.horizon_steps)
```

- [ ] Add trace fields `line_search_used_base`, `line_search_alpha`, `max_joint_velocity`, `max_joint_step` and `max_delta_control` to the shared report. Do not turn them into semantic gates until the existing design threshold defines one; use them to localize why a candidate fails.

- [ ] Run a screening matrix for H16/H20/H24/H30/H40/H50 with identical losses and solver settings. Each run must keep `dt=0.02`, measured x0, x1 publication and `H=2*half_cycle`; record every JointMetrics value rather than a composite score.

- [ ] For the shortest horizons whose only remaining failures contain `line_search_used_base`, write a solver RED before changing the old optimization direction:

```python
def test_parallel_line_search_can_accept_a_bounded_small_step():
    result = parallel_line_search(
        base_control=torch.zeros(1, 1, 1),
        delta_control=torch.full((1, 1, 1), 10.0),
        merit_fn=lambda u: (u - 0.5).square().flatten(1).mean(1),
        alphas=(1.0, 0.25, 0.10, 0.05),
        delta_limit=torch.tensor([1.0]),
    )
    assert result.alpha.item() > 0.0
    assert result.control.abs().max().item() <= 1.0
```

- [ ] Implement a continuous optimization trust region inside candidate construction, not a semantic gate or post-publish repair. Apply per-control-component limits to the SQP direction before all alpha candidates, keep every alpha evaluation batched, and expose the limit through solver config. Run one variable at a time: first horizon, then delta limit, then alpha set.

- [ ] Run focused failing shape/speed/stop cells after each single change, then rerun the complete screening matrix. Remove candidates that improve collision by breaking stance slip, command tracking, foot lead, support, grounding or active swing.

- [ ] Commit the exploration/report infrastructure and the accepted single-cause solver change separately:

```bash
git commit -m "test: compare full-cycle joint mpc horizons"
git commit -m "fix: bound joint mpc rti search direction"
```

### Task 9: Close Stage A On One Selected Horizon

**Files:** only files implicated by focused failures; create a Stage A log and update `scenario_matrix.py`, `run_joint_acceptance.py` to record `H_selected`.

- [ ] Run the complete behavior gate for every screening survivor, not only its previously failing cell:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py --stage behavior --horizon 20
```

Repeat with each surviving horizon. Expected: JSON includes `horizon_steps`, `half_cycle_steps`, every scenario-metric cell, worst values, opportunity counts and a nonzero exit for any failure.

- [ ] Select the shortest candidate for which every applicable inherited and new metric passes. Persist only that candidate as the production default `H_selected`; do not retain a runtime auto-selector in the planner.

- [ ] Run full joint and legacy pytest suites, real Isaac nine-command playback, and manager construction for `1/40/512/1024` using `H_selected`.

- [ ] Restore the packed query contract before accepting Stage A. The planner must perform one fixed-shape packed world query per RTI linearization; stance surface and measured/nominal link samples must be sections of that packed result rather than separate calls.

- [ ] For each failure, record one cell/metric, write or narrow one RED test, change one cause, rerun focused then unified tests. Never tune several weights in one attempt.

- [ ] Record `H_selected`, all worst-case behavior metrics, the rejected horizon reasons and current timing as non-gating evidence. Commit:

```bash
git commit -m "fix: close coupled joint mpc behavior gate"
```

## Stage B — Performance Gate

The frozen formal contract is `1024 x H_selected x 1000 <= 5000ms`; `H_selected` is the production horizon committed by Task 9, not a shorter benchmark substitute.

### Task 10: Profile Frozen Selected-Horizon Behavior

**Files:** modify `joint_mpc_rti_full_refresh_probe.py`, `test_performance.py`; create a profile log.

- [ ] Add a RED contract test requiring the probe to consume the production `H_selected` rather than H16:

```python
assert report.horizon_steps == JointMpcRtiCfg().runtime.horizon_steps
assert report.steps == 1000
assert report.num_envs == 1024
assert report.small_footprint == (11, 11)
assert report.large_footprint == (41, 41)
assert report.synchronous_field_updates == 1000
```

- [ ] On an idle GPU, record signed-field build, packed query/linearization, coupled solve, parallel line search, full P50/P95/P99/max and peak memory for 1024 environments and `H_selected`.

- [ ] Require field version `+1000`, nonfinite `0`, fixed realistic maps and complete Stage A losses. Commit probe/evidence:

```bash
git commit -m "test: profile selected horizon joint mpc pipeline"
```

### Task 11: Exact Batched Signed EDT If Field Dominates

**Files:** modify `terrain/csrc/work_efficient_edt_cuda.cu`, `.cpp`, `cuda_edt.py`, `test_terrain_fields.py` only when Task 10 shows field construction is the largest component.

- [ ] Write RED exactness/density tests for empty/full, single-cell, 11x11, 41x41, random and curved masks, including signed half-cell correction and interior gradients.

- [ ] Implement a true batched PBA-style exact transform: vertical bands, candidate merge, horizontal nearest-site propagation and exact squared-distance writeback for B×2 channels. Do not scan an occupied bbox per occupied cell and do not call a single-image API 2048 times.

- [ ] Keep fixed 151x151 workspaces; fuse outside/inside signed writeback only where CPU exact parity proves the result unchanged.

- [ ] Use 100-step screening, accept only candidates improving the realistic full refresh, remove rejected experiments, then commit:

```bash
git commit -m "perf: add batched exact signed edt"
```

### Task 12: MPX-Style Temporal And State-Space Parallel Solve

**Files:** modify `solver/primal_dual_ilqr.py`, create `solver/associative_tvlqr.py`, modify `solver/sqp_rti.py`, `test_solver.py`, `test_performance.py`.

- [ ] Read and map these reference functions before implementation:

```text
raw/mpx/mpx/jax_ocp_solvers/jax_ocp_solvers/primal_tvlqr.py:
  associative_scan based parallel backward TVLQR and forward rollout
raw/mpx/mpx/jax_ocp_solvers/jax_ocp_solvers/optimizers.py:
  multiple-shooting defects and parallel line search
raw/mpx/mpx/examples/multi_env.py:
  jit(vmap) multi-environment execution and warm-up timing
```

- [ ] Write RED dense-parity tests across `B=(1,40)`, `H=(16,H_selected)`: sequential dense Riccati, associative TVLQR and root-centered Schur must match `delta_state`, `delta_control` and merit improvement within `2e-5` for random SPD problems with nonzero root-leg cross blocks.

- [ ] Implement temporal composition with PyTorch's prototype operator:

```python
from torch._higher_order_ops.associative_scan import associative_scan

prefix = associative_scan(
    combine_conditional_value_factors,
    factor_elements,
    dim=1,
    reverse=True,
)
```

The combine operator must be mathematically associative and tested directly on three factors as `(a*b)*c == a*(b*c)`. Do not approximate or drop temporal/root-leg fill-in.

- [ ] Within each time factor, solve the root 6x6 plus four leg 3x3 arrowhead system by Cholesky leg factors, root Schur complement and back-substitution without explicit inverses. Route unsupported dense rows to the dense correctness path.

- [ ] Evaluate all line-search alphas in one `[B,A,H,18]` fixed-shape program, retain shifted warm starts and multiple-shooting defect diagnostics, and compile one static `H_selected` graph after Stage A.

- [ ] Benchmark isolated solve and full refresh. Accept the associative path only if parity passes, complete Stage A remains green and selected-horizon total improves. Commit:

```bash
git commit -m "perf: parallelize selected horizon joint mpc solve"
```

### Task 13: Close Stage B

**Files:** only measured bottlenecks; create final performance log.

- [ ] Use 100-step screening after each single-variable candidate, then run the formal idle-GPU probe:

```bash
H_SELECTED=$(PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -c \
  'from extension.joint_mpc_rti.config import JointMpcRtiCfg; print(JointMpcRtiCfg().runtime.horizon_steps)')
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=Go2Pvcnn \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_full_refresh_probe.py \
  --num-envs 1024 --horizon "$H_SELECTED" --steps 1000 \
  --small-footprint 11 --large-footprint 41
```

- [ ] Require total `<=5000ms`, mean `<=5ms`, version `+1000`, nonfinite `0`, stable memory and the frozen realistic maps. Do not substitute H16 when `H_selected` is longer.

- [ ] After every accepted performance commit, rerun the complete Stage A unified gate at `H_selected`.

- [ ] Commit only with fresh evidence:

```bash
git commit -m "perf: close selected horizon joint mpc gate"
```

## Stage C — Final Joint Verification

### Task 14: Freshly Rerun Both Gates On One Candidate

**Files:** create one final combined log; update T302v todo/log/index.

- [ ] Record final `HEAD`, `H_selected` and tracked cleanliness.

- [ ] From scratch rerun Stage A at `H_selected`: unified behavior matrix, all joint tests, all legacy regressions, real Isaac viewer and dynamic batches.

- [ ] On the same `HEAD` and same `H_selected`, from scratch rerun Stage B: idle-GPU 1024×`H_selected`×1000 realistic synchronous full-refresh probe.

- [ ] If either fresh gate fails, return to the responsible stage; historical passes do not count and the performance threshold remains five seconds.

- [ ] Update notes with both fresh result sets and commit:

```bash
git commit -m "test: verify selected horizon behavior and performance"
```

- [ ] Run `git diff --check HEAD^ HEAD` and `git status --short --branch`; only the pre-existing user-owned NvStreamer and `raw/mpx/` untracked paths may remain.
