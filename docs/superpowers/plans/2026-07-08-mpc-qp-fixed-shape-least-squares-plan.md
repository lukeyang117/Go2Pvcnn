# MPC-QP Fixed-Shape Least-Squares Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `mpc_qp` continuous path with a fixed-shape batched least-squares / QP-style step that removes semantic/low-small/risky gates and decode hard repairs while keeping MPC-style flat root-z locking and terrain-bound touchdown z.

**Architecture:** `mpc_qp` remains isolated in `Go2Pvcnn/extension/batch_mpc_qp_planner/`. The new solve path uses low-dimensional MPC-style root corrections plus Bezier foot controls, assembles fixed residuals, solves a small damped least-squares step for at most three iterations, and decodes without hard trajectory projection. The existing `mpc` backend is not modified.

**Tech Stack:** Python, PyTorch, pytest, IsaacLab via `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.

## Global Constraints

- Active workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim-mpc_qp`.
- Design source: `docs/superpowers/specs/2026-07-08-mpc-qp-fixed-shape-least-squares-design.html`.
- `mpc` and `mpc_qp` must stay isolated; do not modify `Go2Pvcnn/extension/batch_mpc_planner/` to make `mpc_qp` pass.
- Only two hard rules are allowed in the continuous `mpc_qp` main path: `plane_row` locks `root z`, and touchdown `P3.z = height_at(P3.xy)`.
- `plane_row` may only affect `root z`; it must not lock `root xy` or switch foot/root update gains.
- Viewer shows FK foot trajectory; planned-vs-FK consistency is enforced by residuals and tests, not by viewer overlay.
- No candidate endpoint search, ring search, nearby-cell lookup, fixed-offset touchdown selection, repair main path, semantic/low-small/risky update gate, decode projection, root readback, high semantic root lift, or conditional FK foot replacement.
- `qp_iterations` default is `3`; values above `3` must raise an error. Later tuning may reduce to `2` or `1`, never increase above `3`.
- If metrics fail, fix residuals, weights, damping, variable scaling, fixed sampling density, or root basis conditioning. Do not add hard constraints or gates.

---

## File Structure

- Modify `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - Set `qp_iterations=3`.
  - Add a max-iteration validation guard.
  - Add least-squares numeric weights/damping fields.
- Modify `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
  - Keep only P3 touchdown z binding in decode.
  - Remove foot terrain clamp, low-small z cap, root readback, high semantic root lift, and conditional FK replacement.
  - Ensure final `result.foot_pos` is FK foot for viewer/runtime while planned-vs-FK metrics remain available.
- Create `Go2Pvcnn/extension/batch_mpc_qp_planner/least_squares_solver.py`
  - Own the fixed-shape residual solve.
  - Expose `least_squares_qp_update(...)`.
  - Use fixed low-dimensional root basis and fixed Bezier foot variables.
- Modify `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - Route `continuous_qp_update()` to `least_squares_qp_update()`.
  - Remove dependency on `coupled_qp_update()` from the active path.
- Modify `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
  - Stop passing decode readback/root-z parameters.
  - Keep `qp_repair_main_path_active=0` for continuous path.
  - Preserve diagnostics required by tests.
- Modify `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`
  - Add/standardize planned-vs-FK metric names.
- Modify `Go2Pvcnn/tests/test_mpc_qp_backend.py`
  - Replace old repair/gate/iteration>3 expectations with new design-contract tests and focused behavior tests.

---

### Task 1: Contract Tests And Iteration Budget

**Files:**
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`

**Interfaces:**
- Consumes: `MpcQpPlannerCfg`, `validate_mpc_qp_config(cfg) -> None`, `plan_segment_qp(...)`.
- Produces: validation rule `1 <= cfg.runtime.qp_iterations <= 3`, default `qp_iterations == 3`, and structure tests guarding forbidden code patterns.

- [ ] **Step 1: Write failing tests for the new iteration budget**

Add tests in `Go2Pvcnn/tests/test_mpc_qp_backend.py`:

```python
def test_mpc_qp_default_qp_iterations_is_three_and_more_than_three_is_rejected() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg, validate_mpc_qp_config

    cfg = MpcQpPlannerCfg()
    assert cfg.runtime.qp_iterations == 3
    validate_mpc_qp_config(cfg)

    cfg.runtime.qp_iterations = 4
    try:
        validate_mpc_qp_config(cfg)
    except ValueError as exc:
        assert "qp_iterations" in str(exc)
        assert "3" in str(exc)
    else:
        raise AssertionError("qp_iterations > 3 must raise ValueError")
```

Update the earlier default-backend test so it expects `3` instead of `1`.

- [ ] **Step 2: Write failing structure tests for forbidden gates and decode hard bindings**

Add tests:

```python
def test_mpc_qp_continuous_decode_keeps_only_touchdown_z_hard_binding() -> None:
    source = Path("Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py").read_text()
    assert "controls_w[:, :, 3, 2] = touchdown_z" in source
    forbidden = [
        "foot_pos[..., 2] = torch.maximum",
        "low_small_swing",
        "root_z_readback",
        "high_crossing",
        "root_pos[..., 2] = root_pos[..., 2] +",
        "foot_pos = torch.where(crossing_readback_row, fk_foot, foot_pos)",
    ]
    for token in forbidden:
        assert token not in source


def test_mpc_qp_active_solver_source_has_no_semantic_low_small_or_risky_update_gate() -> None:
    source = Path("Go2Pvcnn/extension/batch_mpc_qp_planner/least_squares_solver.py").read_text()
    forbidden = [
        "semantic_present",
        "low_small_crossing_row",
        "risky_root",
        "semantic_root_support",
        "flat_root_xy_lock_row",
        "foot_semantic_row",
        "semantic_root_step_row",
        "negative_end_progress",
        "arc_active",
    ]
    for token in forbidden:
        assert token not in source
    assert "plane_row" in source
    assert "root_z" in source
```

The second test will fail until `least_squares_solver.py` exists.

- [ ] **Step 3: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_default_qp_iterations_is_three_and_more_than_three_is_rejected \
       Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_decode_keeps_only_touchdown_z_hard_binding \
       Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_active_solver_source_has_no_semantic_low_small_or_risky_update_gate -q
```

Expected: FAIL because default is `1`, `qp_iterations=4` is accepted, decode still has hard bindings, and `least_squares_solver.py` does not exist.

- [ ] **Step 4: Implement config budget**

In `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`:

```python
qp_iterations: int = 3
least_squares_damping: float = 1.0e-3
least_squares_step_scale: float = 1.0
least_squares_max_delta_norm: float = 0.12
root_basis_terminal_weight: float = 1.0
root_basis_mid_weight: float = 0.35
```

Update `validate_mpc_qp_config`:

```python
def validate_mpc_qp_config(cfg: MpcQpPlannerCfg) -> None:
    iterations = int(cfg.runtime.qp_iterations)
    if iterations <= 0:
        raise ValueError("runtime.qp_iterations must be positive")
    if iterations > 3:
        raise ValueError("runtime.qp_iterations must be <= 3 for mpc_qp")
```

- [ ] **Step 5: Create placeholder least-squares module for structure test**

Create `Go2Pvcnn/extension/batch_mpc_qp_planner/least_squares_solver.py` with a temporary function that delegates to current controls unchanged:

```python
from __future__ import annotations

from torch import Tensor

from extension.batch_mpc_planner.types import MpcPlannerTerrain

from .config import MpcQpPlannerCfg
from .continuous import ContinuousTrajectoryControls


def least_squares_qp_update(
    controls: ContinuousTrajectoryControls,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
    *,
    command: Tensor | None,
    contact_state: Tensor | None,
) -> tuple[ContinuousTrajectoryControls, dict[str, Tensor]]:
    plane_row = getattr(terrain, "is_plane_terrain", None)
    root_z = controls.root_pos_w[..., 2]
    return controls, {
        "qp_least_squares_solver_active": root_z.new_ones((root_z.shape[0],)),
        "qp_least_squares_plane_row_seen": root_z.new_zeros((root_z.shape[0],)) if plane_row is None else root_z.new_ones((root_z.shape[0],)),
    }
```

- [ ] **Step 6: Run partial GREEN for config and solver-file structure**

Run the same command from Step 3.

Expected: config and solver source tests pass; decode test still fails until Task 2.

---

### Task 2: Decode Contract And Planned-vs-FK Metrics

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Consumes: `decode_controls_to_result(result, terrain, controls, sample_count, contact_state) -> MpcPlannerResult`.
- Produces: decode that only hard-binds P3.z, returns FK foot for viewer/runtime, and reports planned-vs-FK consistency metrics through diagnostics.

- [ ] **Step 1: Write failing behavior tests for decode**

Add tests:

```python
def test_mpc_qp_decode_outputs_fk_feet_but_reports_planned_fk_error() -> None:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 1
    terrain = _flat_terrain(batch=1, size=51)
    state = _state(batch=1)
    command = torch.tensor([[0.18, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)

    torch.testing.assert_close(result.foot_pos, fk_foot, atol=1.0e-6, rtol=1.0e-6)
    assert "qp_continuous_fk_planned_foot_error_max" in result.loss_breakdown
    assert result.loss_breakdown["qp_continuous_fk_planned_foot_error_max"].item() <= 0.05
```

Add a second test asserting foot terrain penetration is measured by diagnostics, not fixed by decode clamp:

```python
def test_mpc_qp_decode_does_not_clamp_sampled_foot_z_to_terrain() -> None:
    source = Path("Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py").read_text()
    assert "foot_pos[..., 2] = torch.maximum" not in source
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_decode_outputs_fk_feet_but_reports_planned_fk_error \
       Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_decode_does_not_clamp_sampled_foot_z_to_terrain -q
```

Expected: FAIL because decode still has hard binding/readback logic and metrics use old names.

- [ ] **Step 3: Simplify decode**

In `continuous.py`, change `decode_controls_to_result` to:

- Keep:

```python
touchdown_z = height_at(terrain, controls_w[:, :, 3, :2]).to(dtype=controls_w.dtype, device=controls_w.device)
controls_w[:, :, 3, 2] = touchdown_z
foot_pos_planned = sample_controls_with_optional_gait(...)
root_pos = controls.root_pos_w.clone()
root_rpy = controls.root_rpy.clone()
joint_angles = solve_joint_angles_from_trajectory(root_pos, root_rpy, foot_pos_planned)
fk_foot = fk_feet_from_joint_angles(root_pos, root_rpy, joint_angles)
foot_pos = fk_foot
```

- Remove parameters and code paths for `low_small_swing_clearance_m`, `root_z_readback_gain`, `root_z_readback_max_step_m`, and `fk_readback_row_mask`.
- Keep planned touchdown sequence based on P3.
- Add planned foot to `loss_breakdown` only if the result type supports existing dictionaries; otherwise diagnostics are computed in `planner.py` using a helper.

- [ ] **Step 4: Update planner decode call**

In `planner.py`, change:

```python
result = decode_controls_to_result(
    result,
    terrain,
    controls,
    sample_count=sample_count,
    contact_state=fixed_gait.stance_mask,
)
```

Remove decode root-z delta diagnostics tied to readback.

- [ ] **Step 5: Add planned-vs-FK diagnostics**

In `losses.py` or `planner.py`, compute:

```python
fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
planned_error = torch.linalg.vector_norm(fk_foot - planned_foot_for_metrics, dim=-1)
```

If planned foot is not carried in `MpcPlannerResult`, recompute it from controls immediately before decode returns diagnostics. Add keys:

```python
"qp_continuous_fk_planned_foot_error_max"
"qp_continuous_fk_planned_foot_error_mean"
"qp_continuous_fk_planned_foot_terminal_error_max"
```

Keep old `qp_continuous_fk_readback_error_max` as an alias for compatibility during migration.

- [ ] **Step 6: Run GREEN for decode tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_decode_keeps_only_touchdown_z_hard_binding \
       Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_decode_outputs_fk_feet_but_reports_planned_fk_error \
       Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_decode_does_not_clamp_sampled_foot_z_to_terrain -q
```

Expected: PASS.

---

### Task 3: Fixed-Shape Least-Squares Solver

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/least_squares_solver.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Consumes: `ContinuousTrajectoryControls`, `MpcPlannerTerrain`, `MpcQpPlannerCfg`, `command`, `contact_state`.
- Produces: `least_squares_qp_update(...) -> (ContinuousTrajectoryControls, diagnostics)` using fixed residuals and no forbidden gates.

- [ ] **Step 1: Write focused tests for plane root-z lock and root-xy freedom**

Add:

```python
def test_mpc_qp_least_squares_plane_row_locks_root_z_only_not_root_xy() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 3
    terrain = _flat_terrain(batch=1, size=71)
    state = _state(batch=1)
    command = torch.tensor([[0.30, 0.08, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    root_z = result.root_pos[..., 2]
    root_xy_delta = torch.linalg.vector_norm(result.root_pos[:, -1, :2] - result.root_pos[:, 0, :2], dim=-1)

    assert float((root_z.amax(dim=1) - root_z.amin(dim=1)).max().item()) <= 1.0e-4
    assert root_xy_delta.item() > 0.01
    assert result.loss_breakdown["qp_least_squares_plane_root_z_lock_count"].item() >= 1
```

- [ ] **Step 2: Write focused tests for no iteration greater than three**

Update old tests that set `qp_iterations=4` or `6` to either expect `ValueError` or use `3`.

Add:

```python
def test_mpc_qp_plan_segment_rejects_qp_iterations_above_three() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 4
    try:
        plan_segment_qp(_flat_terrain(batch=1), _state(batch=1), torch.zeros((1, 3)), cfg=cfg)
    except ValueError as exc:
        assert "qp_iterations" in str(exc)
    else:
        raise AssertionError("plan_segment_qp must reject qp_iterations > 3")
```

- [ ] **Step 3: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_least_squares_plane_row_locks_root_z_only_not_root_xy \
       Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_plan_segment_rejects_qp_iterations_above_three -q
```

Expected: plane least-squares diagnostics fail until solver is implemented; iteration rejection should pass after Task 1.

- [ ] **Step 4: Route active solver to least-squares module**

In `solver.py`:

```python
from .least_squares_solver import least_squares_qp_update

def continuous_qp_update(...):
    return least_squares_qp_update(
        controls,
        terrain,
        cfg,
        command=command,
        contact_state=contact_state,
    )
```

Remove the active import/use of `coupled_qp_update`.

- [ ] **Step 5: Implement root basis helpers**

In `least_squares_solver.py`, add:

```python
def _plane_row(terrain: MpcPlannerTerrain, batch: int, device: torch.device) -> Tensor:
    plane = getattr(terrain, "is_plane_terrain", None)
    if plane is None:
        return torch.zeros((batch,), dtype=torch.bool, device=device)
    out = torch.as_tensor(plane, dtype=torch.bool, device=device).reshape(-1)
    if int(out.numel()) == 1 and batch > 1:
        out = out.expand(batch)
    return out[:batch]


def _smooth_basis(horizon: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    phase = torch.linspace(0.0, 1.0, int(horizon), dtype=dtype, device=device)
    terminal = phase * phase * (3.0 - 2.0 * phase)
    mid = 4.0 * phase * (1.0 - phase)
    return torch.stack((terminal, mid), dim=-1)
```

- [ ] **Step 6: Implement fixed residual solve**

Implement a compact Gauss-Newton/LM step using autograd Jacobian-vector-friendly dense assembly for the first version:

- Variables:
  - flattened foot delta for P1/P2/P3, with P3 z masked out after apply;
  - root low-dim corrections `[terminal_xy(2), mid_xy(2), z_terminal(1), z_mid(1), rpy_terminal(3), rpy_mid(3)]`.
- Decode inside solver:
  - apply root smooth basis to base root;
  - if `plane_row`, set root z to base first-frame z expanded over horizon;
  - bind P3.z with `height_at`.
- Residuals:
  - touchdown semantic risk and roughness using `build_qp_fields`;
  - swing terrain clearance residual;
  - FK planned-foot residual;
  - joint limit residual from unclamped IK;
  - reachability residual hip-to-foot range;
  - root/foot/joint first and second difference residuals;
  - nominal residual and low-weight progress residual.
- Solve:

```python
A = j.transpose(-1, -2).matmul(j) + damping * eye
b = -j.transpose(-1, -2).matmul(r.unsqueeze(-1)).squeeze(-1)
delta = torch.linalg.solve(A, b)
```

For the first implementation, use `torch.autograd.functional.jacobian` only if batch is small; for batched tests, prefer residual gradients per scalar block or a diagonal LM approximation if dense Jacobian is too slow. Keep fixed shapes and no Python per-env candidate search.

- [ ] **Step 7: Apply update without forbidden gates**

Apply `delta` uniformly:

```python
updated_foot = base_foot + foot_delta
updated_foot[:, :, 0, :] = base_foot[:, :, 0, :]
updated_foot[:, :, 3, 2] = height_at(terrain, updated_foot[:, :, 3, :2])
updated_root = decoded_root_from_basis
updated_root[plane_row, :, 2] = base_root[plane_row, :1, 2]
```

No semantic/low-small/risky conditional update gates are allowed.

- [ ] **Step 8: Run focused GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_active_solver_source_has_no_semantic_low_small_or_risky_update_gate \
       Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_least_squares_plane_row_locks_root_z_only_not_root_xy \
       Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_decode_outputs_fk_feet_but_reports_planned_fk_error -q
```

Expected: PASS.

---

### Task 4: Replace Conflicting Legacy Tests With Design Metrics

**Files:**
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- Modify: `notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md`
- Modify: `notes/todo.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-07-08-mpc-qp-fixed-shape-least-squares-contract.md`

**Interfaces:**
- Consumes: design metrics from `docs/superpowers/specs/2026-07-08-mpc-qp-fixed-shape-least-squares-design.html`.
- Produces: tests aligned with the no-gate/no-hard-binding contract and notes/log evidence.

- [ ] **Step 1: Update tests that require forbidden behavior**

In `Go2Pvcnn/tests/test_mpc_qp_backend.py`:

- Rename or replace tests with names containing `repair` if they target `continuous_trajectory_enabled=True`.
- For tests that set `qp_iterations=4` or `6`, change to `3` if still relevant, or replace with rejection tests.
- Remove expectations for:
  - `qp_flat_semantic_smooth_row_count`;
  - `qp_continuous_root_terrain_risk_reduces_progress`;
  - `qp_continuous_low_small_foot_over_update_count == 0` if it refers to old gate behavior;
  - root readback update metrics.
- Keep/replace expectations for:
  - semantic collision count zero;
  - touchdown-on-semantic zero;
  - planned-vs-FK metrics;
  - flat root z stability;
  - foot/joint/root continuity metrics;
  - `qp_iterations_configured == 3`.

- [ ] **Step 2: Add small-obstacle flat acceptance test under the new contract**

Add:

```python
def test_mpc_qp_fixed_ls_flat_small_obstacle_passes_without_decode_projection() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 3
    terrain = _flat_terrain(batch=1, size=151)
    terrain.height_map[:, 72:80, 88:96] = 0.05
    terrain.semantic_map[:, 72:80, 88:96] = 1
    state = _state(batch=1)
    command = torch.tensor([[0.35, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    loss = result.loss_breakdown

    assert loss["qp_iterations_configured"].item() == 3
    assert loss["qp_touchdown_on_small_count"].item() == 0
    assert loss["qp_fk_semantic_collision_count"].item() == 0
    assert loss["qp_continuous_fk_planned_foot_error_max"].item() <= 0.05
    root_z = result.root_pos[..., 2]
    assert float((root_z.amax(dim=1) - root_z.amin(dim=1)).max().item()) <= 1.0e-4
```

- [ ] **Step 3: Run focused test set**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Expected: all active `mpc_qp` unit tests pass or expose residual tuning failures. If failures are behavior quality issues, tune residual weights/damping only.

- [ ] **Step 4: Run static compile**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_qp_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py \
  Go2Pvcnn/extension/batch_mpc_qp_planner/least_squares_solver.py \
  Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py \
  Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py \
  Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py
```

Expected: exit `0`.

- [ ] **Step 5: Update notes and log**

Create `notes/log/2026-07-08-mpc-qp-fixed-shape-least-squares-contract.md` with:

- purpose;
- files changed;
- tests run;
- key metrics;
- whether real IsaacSim viewer was rerun;
- remaining risk.

Update `notes/todo.md`, `notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md`, and `notes/log/index.md` with links to the new design, plan, and log.

---

### Task 5: Viewer/Probe Smoke And Performance Guard

**Files:**
- Modify: `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py` only if metric names changed.
- Modify: `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py` only if metric names changed.
- Create: `notes/log/2026-07-08-mpc-qp-fixed-ls-viewer-smoke.md`

**Interfaces:**
- Consumes: `--planner-backend mpc_qp`, `--qp-iterations 3`, existing viewer/probe scripts.
- Produces: evidence for flat small obstacle and row 8 col 12 readiness, plus solve time/memory metrics.

- [ ] **Step 1: Run flat small obstacle probe**

Run the existing small obstacle probe command used by current tests. If no single command exists, run the focused pytest small-obstacle test from Task 4 and record its metrics.

Expected:

- semantic collision count `0`;
- touchdown-on-small count `0`;
- planned-vs-FK max error `<= 0.05`;
- flat root z range `<= 1e-4`.

- [ ] **Step 2: Run row 8 col 12 viewer command headless if available**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:0 \
  --num_envs 1 \
  --terrain task \
  --planner-backend mpc_qp \
  --n-frames 25 \
  --plan-dt 0.02 \
  --qp-iterations 3 \
  --terrain-row 8 \
  --terrain-col 12
```

Do not hard-code `CUDA_VISIBLE_DEVICES`; use the user's visible GPU setup.

Expected: no planner exception. If the run is manually interrupted after viewer startup, record that real acceptance metrics still need a probe run.

- [ ] **Step 3: Run 1024-env smoke if time/memory allow**

Run:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py \
  --num-envs 1024 \
  --mpc-num-envs 1024 \
  --steps 30 \
  --require-replan \
  --print-cuda-memory \
  --summary-path /tmp/mpc_qp_fixed_ls_1024.json \
  --planner-backend mpc_qp \
  --qp-iterations 3
```

Expected: no OOM, finite cache, at least one replan. Record CUDA memory and `qp_total_ms`.

- [ ] **Step 4: Record final smoke log**

Create `notes/log/2026-07-08-mpc-qp-fixed-ls-viewer-smoke.md` and update `notes/log/index.md`.

