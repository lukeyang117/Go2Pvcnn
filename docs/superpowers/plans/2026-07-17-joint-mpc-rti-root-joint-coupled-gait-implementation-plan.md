# Joint MPC RTI Root-Joint Coupled Gait Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a truly coupled root-joint rolling RTI gait, evaluate every scenario with all applicable old/new metrics, then reduce the realistic 1024-env synchronous field+MPC runtime from 7.4025s to at most 5.0s and freshly rerun both gates on one final candidate.

**Architecture:** Keep H16, dt=0.02, fixed trot, measured x0 and published x1. Add complete `3x18` FK Jacobians, augmented-Lagrangian stance equalities, command-conditioned touchdown, horizon command progress, startup foot lead, a correct dense coupled solver followed by a Schur production solver, and one shared JointMetrics accumulator.

**Tech Stack:** Python 3.10, PyTorch, torch.compile, CUDA/C++, SQP-RTI/GGN/Riccati, pytest, Isaac Lab in `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.

---

## File Map

- Create `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`: shared trace and metric accumulator.
- Create `Go2Pvcnn/tests/joint_mpc_rti/acceptance_thresholds.py`: applicability, thresholds and per-cell aggregation.
- Create `Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py`: flat/small/stop/large/step scenario rows.
- Create `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`: unified Stage A/Stage C runner.
- Create `Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py` and `test_coupled_gait.py`.
- Modify `config.py`, `types.py`, `model/go2_kinematics.py`, `losses/command.py`, `losses/contact.py`, `planner.py`, `solver/primal_dual_ilqr.py`, `solver/sqp_rti.py`.
- Refactor existing crossing/stop/behavior probes to consume the shared metrics.
- Stage B may modify `terrain/csrc/work_efficient_edt_cuda.cu`, `terrain/cuda_edt.py` and the coupled solver only after profiling.

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

### Task 8: Close Stage A

**Files:** only files implicated by focused failures; create a Stage A log.

- [ ] Run the unified behavior gate:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py --stage behavior
```

- [ ] Run full joint and legacy pytest suites, real Isaac nine-command playback, and manager construction for `1/40/512/1024`.
- [ ] For each failure, record one cell/metric, write or narrow one RED test, change one cause, rerun focused then unified tests. Never tune several weights in one attempt.
- [ ] Require all inherited cross/collision/support/grounding/viewer/rolling gates plus all new coupling/lead/slip gates.
- [ ] Record evidence and commit `fix: close coupled joint mpc behavior gate`.

## Stage B — Performance Gate

### Task 9: Profile Frozen Behavior

**Files:** modify `joint_mpc_rti_full_refresh_probe.py`, `test_performance.py`; create a profile log.

- [ ] Add contract tests retaining 1024 envs, H16, 1000 steps, small 11x11, large 41x41, exact synchronous field, coupled solver, version and nonfinite output.
- [ ] On an idle GPU, record field, linearization, coupled solve, line search, full P50/P95/P99/max and memory.
- [ ] Optimize only the largest measured component; commit probe/evidence `test: profile frozen coupled mpc pipeline`.

### Task 10: Exact Batched Signed EDT If Field Dominates

**Files:** modify `terrain/csrc/work_efficient_edt_cuda.cu`, `.cpp`, `cuda_edt.py`, `test_terrain_fields.py`.

- [ ] Write RED exactness/density tests for empty/full, single-cell, 11x11, 41x41, random and curved masks, including signed half-cell correction and interior gradients.
- [ ] Implement a true batched PBA-style exact transform: vertical bands, candidate merge, horizontal nearest-site propagation and exact squared-distance writeback for B×2 channels. Do not scan an occupied bbox per occupied cell and do not call a single-image API 2048 times.
- [ ] Keep fixed 151x151 workspaces; fuse outside/inside signed writeback where exactness permits.
- [ ] Accept only candidates that pass CPU exact parity and improve the realistic full refresh; remove rejected experiments.
- [ ] Commit `perf: add batched exact signed edt` only after exactness and measured improvement.

### Task 11: Coupled Schur/Riccati If MPC Dominates

**Files:** modify solver files and `test_solver.py`.

- [ ] Write RED dense-parity tests for root 6x6 + four leg 3x3 arrowhead Hessians across B/H.
- [ ] Factor each leg block, build the root Schur complement, solve root, then back-substitute legs without explicit inverses.
- [ ] Preserve temporal fill-in; route unsupported rows to dense correctness rather than dropping cross terms.
- [ ] Compile fixed shapes and benchmark isolated solve plus full refresh.
- [ ] Run solver/coupled/behavior tests and commit `perf: solve coupled joint mpc schur blocks`.

### Task 12: Close Stage B

**Files:** only measured bottlenecks; create final performance log.

- [ ] Use 100-step screening then the formal 1000-step probe for each single-variable performance candidate.
- [ ] Require full total `<=5000ms`, mean `<=5ms`, version `+1000`, nonfinite `0`, stable memory and the frozen realistic maps.
- [ ] After every accepted performance commit, rerun the complete Stage A unified gate.
- [ ] Commit `perf: close joint mpc full refresh gate` only with fresh evidence.

## Stage C — Final Joint Verification

### Task 13: Freshly Rerun Both Gates On One Candidate

**Files:** create one final combined log; update T302v todo/log/index.

- [ ] Record final `HEAD` and tracked cleanliness.
- [ ] From scratch rerun Stage A: unified behavior matrix, all joint tests, all legacy regressions, real Isaac viewer and dynamic batches.
- [ ] On the same `HEAD`, from scratch rerun Stage B: idle-GPU 1024×H16×1000 realistic synchronous full-refresh probe.
- [ ] If either fresh gate fails, return to the responsible stage; historical passes do not count.
- [ ] Update notes with both fresh result sets and commit `test: verify joint mpc behavior and performance gates`.
- [ ] Run `git diff --check HEAD^ HEAD` and `git status --short --branch`; only the pre-existing user-owned NvStreamer and `raw/mpx/` untracked paths may remain.
