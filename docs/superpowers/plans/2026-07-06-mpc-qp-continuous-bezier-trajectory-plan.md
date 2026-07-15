# MPC-QP Continuous Bezier Trajectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repair-dominant `mpc_qp` main path with an isolated continuous Bezier trajectory optimization path that samples trajectory points from optimized curves.

**Architecture:** The current `mpc` backend remains untouched and is used only as a semantic/loss reference. New continuous-QP code lives under `Go2Pvcnn/extension/batch_mpc_qp_planner/`, starting with fixed-shape Bezier sampling, preallocated trajectory work buffers, terrain-bound touchdown z, and diagnostics that prove repair is no longer the dominant behavior source. The implementation proceeds TDD-first and keeps the existing `mpc_qp` opt-in ABI stable.

**Tech Stack:** Python, PyTorch, pytest, IsaacLab smoke tests through `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python` and `CUDA_VISIBLE_DEVICES=1` for real simulation validation.

## Global Constraints

- Do not modify the current default `planner_backend="mpc"` behavior.
- Do not edit `Go2Pvcnn/extension/batch_mpc_planner/` for this redesign unless a separate shared-interface bug is discovered and explicitly documented.
- New implementation files live under `Go2Pvcnn/extension/batch_mpc_qp_planner/`.
- `mpc_qp` remains selected only by explicit `planner_backend="mpc_qp"` / `--planner-backend mpc_qp`.
- Trajectory output must be sampled from continuous Bezier/B-spline curves, not assembled by per-frame repair.
- `touchdown_z = height_at(terrain, touchdown_xy)` and is not a free optimization variable.
- Terrain and semantic maps are prepared once per planner refresh; QP iterations only query fixed-shape buffers.
- Do not allocate dense global tensors such as `[B, K, H*W, ...]` inside the optimization loop.
- If validation fails, tune loss weights, fixed sampling density, warm start, or `qp_iterations`; do not add new hard repair layers.

---

## File Structure

- Create `Go2Pvcnn/extension/batch_mpc_qp_planner/bezier.py`
  - Fixed cubic Bezier basis, sampling, finite-difference quality metrics.
- Create `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
  - Convert nominal `MpcPlannerResult` into continuous trajectory controls, bind touchdown z to terrain, sample output trajectory.
- Create `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`
  - Fixed-shape semantic, foothold, clearance, reachability, smoothness, and tracking residuals for sampled Bezier trajectories.
- Modify `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - Add continuous trajectory config, QP loss weights, and a switch that defaults the new `mpc_qp` path to continuous mode after tests prove ABI compatibility.
- Modify `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
  - Keep nominal generation and cache ABI, but route through the continuous trajectory path and remove/demote repair diagnostics from the main path.
- Modify `Go2Pvcnn/tests/test_mpc_qp_backend.py`
  - Add focused tests for Bezier sampling, touchdown z terrain binding, `mpc` isolation, no repair-dominant diagnostics, and continuity metrics.
- Update `notes/todo.md`, `notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md`, `notes/log/index.md`, and one per-verification log after implementation/verification.

### Task 1: Fixed Bezier Sampling Foundation

**Files:**
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/bezier.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Produces: `cubic_bezier_basis(sample_count: int, *, dtype: torch.dtype, device: torch.device) -> Tensor`
- Produces: `sample_cubic_bezier(control_points: Tensor, basis: Tensor) -> Tensor`
- Produces: `trajectory_frame_deltas(samples: Tensor) -> tuple[Tensor, Tensor]`

- [ ] **Step 1: Write failing Bezier tests**

Add this test block to `Go2Pvcnn/tests/test_mpc_qp_backend.py`:

```python
def test_mpc_qp_cubic_bezier_sampling_is_continuous_and_endpoint_exact() -> None:
    from extension.batch_mpc_qp_planner.bezier import (
        cubic_bezier_basis,
        sample_cubic_bezier,
        trajectory_frame_deltas,
    )

    controls = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.2, 0.0, 0.3], [0.4, 0.0, 0.3], [0.6, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    basis = cubic_bezier_basis(9, dtype=controls.dtype, device=controls.device)
    samples = sample_cubic_bezier(controls, basis)
    first_delta, second_delta = trajectory_frame_deltas(samples)

    assert samples.shape == (1, 1, 9, 3)
    assert torch.allclose(samples[:, :, 0], controls[:, :, 0])
    assert torch.allclose(samples[:, :, -1], controls[:, :, -1])
    assert torch.isfinite(first_delta).all()
    assert torch.isfinite(second_delta).all()
    assert float(first_delta.norm(dim=-1).amax().item()) < 0.20
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_cubic_bezier_sampling_is_continuous_and_endpoint_exact -q
```

Expected: fails with `ModuleNotFoundError` or missing `bezier` functions.

- [ ] **Step 3: Implement Bezier helpers**

Create `Go2Pvcnn/extension/batch_mpc_qp_planner/bezier.py`:

```python
"""Fixed-shape Bezier helpers for the continuous MPC-QP backend."""

from __future__ import annotations

import torch
from torch import Tensor


def cubic_bezier_basis(sample_count: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    count = max(2, int(sample_count))
    s = torch.linspace(0.0, 1.0, count, dtype=dtype, device=device)
    one = 1.0 - s
    return torch.stack((one * one * one, 3.0 * one * one * s, 3.0 * one * s * s, s * s * s), dim=-1)


def sample_cubic_bezier(control_points: Tensor, basis: Tensor) -> Tensor:
    if control_points.shape[-2:] != (4, 3):
        raise ValueError(f"control_points must end with [4,3], got {tuple(control_points.shape)}")
    if basis.ndim != 2 or int(basis.shape[-1]) != 4:
        raise ValueError(f"basis must have shape [S,4], got {tuple(basis.shape)}")
    return torch.einsum("sc,...cd->...sd", basis.to(dtype=control_points.dtype, device=control_points.device), control_points)


def trajectory_frame_deltas(samples: Tensor) -> tuple[Tensor, Tensor]:
    first = samples[..., 1:, :] - samples[..., :-1, :]
    second = first[..., 1:, :] - first[..., :-1, :]
    return first, second


__all__ = ["cubic_bezier_basis", "sample_cubic_bezier", "trajectory_frame_deltas"]
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_cubic_bezier_sampling_is_continuous_and_endpoint_exact -q
```

Expected: pass.

### Task 2: Terrain-Bound Touchdown And Continuous Trajectory Decode

**Files:**
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Consumes: `MpcPlannerResult`, `MpcPlannerTerrain`
- Produces: `ContinuousTrajectoryControls`
- Produces: `build_controls_from_nominal(result: MpcPlannerResult, terrain: MpcPlannerTerrain) -> ContinuousTrajectoryControls`
- Produces: `decode_controls_to_result(result: MpcPlannerResult, terrain: MpcPlannerTerrain, controls: ContinuousTrajectoryControls, sample_count: int) -> MpcPlannerResult`

- [ ] **Step 1: Write failing touchdown binding test**

Add:

```python
def test_mpc_qp_continuous_decode_binds_touchdown_z_to_terrain() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.continuous import build_controls_from_nominal, decode_controls_to_result
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp
    from extension.batch_mpc_planner.terrain import height_at

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 1
    terrain = _flat_terrain(batch=1, size=9)
    terrain.height_map[:, 4:, :] = 0.11
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)
    nominal = plan_segment_qp(terrain, state, command, cfg=cfg)

    controls = build_controls_from_nominal(nominal, terrain)
    decoded = decode_controls_to_result(nominal, terrain, controls, sample_count=cfg.runtime.horizon_steps)
    touchdown_xy = decoded.planned_touchdown_w[:, 0, :, :2]
    expected_z = height_at(terrain, touchdown_xy).to(dtype=decoded.planned_touchdown_w.dtype)

    assert torch.allclose(decoded.planned_touchdown_w[:, 0, :, 2], expected_z, atol=1.0e-5)
    assert torch.allclose(decoded.touchdown_seq[:, :, 0, 2], expected_z, atol=1.0e-5)
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_decode_binds_touchdown_z_to_terrain -q
```

Expected: fails because `continuous.py` does not exist.

- [ ] **Step 3: Implement continuous decode**

Implement `ContinuousTrajectoryControls`, `build_controls_from_nominal()`, and `decode_controls_to_result()` so `P0/P1/P2/P3` are derived from nominal foot trajectories, `P3.z` is replaced by `height_at(terrain, P3.xy)`, and sampled foot positions come from `sample_cubic_bezier()`.

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_decode_binds_touchdown_z_to_terrain -q
```

Expected: pass.

### Task 3: Fixed-Shape Loss Diagnostics For Continuous Samples

**Files:**
- Create: `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Produces: `continuous_loss_diagnostics(result: MpcPlannerResult, terrain: MpcPlannerTerrain, *, footprint_radius_m: float) -> dict[str, Tensor]`

- [ ] **Step 1: Write failing diagnostics test**

Add:

```python
def test_mpc_qp_continuous_loss_diagnostics_report_smoothness_and_foothold_quality() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.losses import continuous_loss_diagnostics
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=2, size=9)
    terrain.height_map[:, 4, :] = 0.08
    state = _state(batch=2)
    command = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.20, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    diag = continuous_loss_diagnostics(result, terrain, footprint_radius_m=0.04)

    for key in (
        "qp_continuous_foot_frame_jump_max",
        "qp_continuous_foot_acceleration_max",
        "qp_continuous_foothold_height_variation_max",
        "qp_continuous_touchdown_semantic_bad_count",
    ):
        assert key in diag
        assert diag[key].shape == (2,)
        assert torch.isfinite(diag[key]).all()
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_loss_diagnostics_report_smoothness_and_foothold_quality -q
```

Expected: fails because `losses.py` does not exist.

- [ ] **Step 3: Implement fixed-shape diagnostics**

Implement semantic touchdown count, footprint height variation from fixed offsets, foot frame jump max, and foot acceleration max. Do not allocate obstacle-list-shaped tensors.

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_loss_diagnostics_report_smoothness_and_foothold_quality -q
```

Expected: pass.

### Task 4: Route `plan_segment_qp()` Through Continuous Decode And Demote Repair Main Path

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Consumes: `build_controls_from_nominal()`, `decode_controls_to_result()`, `continuous_loss_diagnostics()`
- Produces: `result.loss_breakdown["qp_continuous_enabled"]`
- Produces: `result.loss_breakdown["qp_repair_main_path_active"] == 0`

- [ ] **Step 1: Write failing planner-route test**

Add:

```python
def test_mpc_qp_plan_segment_uses_continuous_path_without_repair_main_path() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_continuous_enabled"], torch.ones(1))
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert "qp_continuous_foot_frame_jump_max" in result.loss_breakdown
    assert "qp_continuous_foothold_height_variation_max" in result.loss_breakdown
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_plan_segment_uses_continuous_path_without_repair_main_path -q
```

Expected: fails because continuous diagnostics are not routed.

- [ ] **Step 3: Implement continuous route**

Add config fields:

```python
continuous_trajectory_enabled: bool = True
continuous_bezier_sample_count: int = 0
continuous_footprint_radius_m: float = 0.04
```

In `plan_segment_qp()`, use nominal `plan_segment()` as warm start, decode continuous controls, run configured QP iterations as no-repair continuous iterations for this first pass, attach continuous diagnostics, and do not call repair functions in the default continuous path.

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_plan_segment_uses_continuous_path_without_repair_main_path -q
```

Expected: pass.

### Task 5: Prove `mpc` And `mpc_qp` Isolation

**Files:**
- Modify: `Go2Pvcnn/tests/test_mpc_qp_backend.py`

**Interfaces:**
- Consumes: current `plan_segment()` and new `plan_segment_qp()`
- Produces: regression proof that `mpc` default remains independent.

- [ ] **Step 1: Write isolation test**

Add:

```python
def test_mpc_qp_continuous_route_does_not_modify_current_mpc_backend() -> None:
    from extension.batch_mpc_planner.config import MpcPlannerCfg
    from extension.batch_mpc_planner.planner import plan_segment
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    mpc_result = plan_segment(terrain, state, command, cfg=MpcPlannerCfg())
    qp_result = plan_segment_qp(terrain, state, command, cfg=MpcQpPlannerCfg())

    assert mpc_result.loss_breakdown is None or "qp_continuous_enabled" not in mpc_result.loss_breakdown
    assert qp_result.loss_breakdown is not None
    assert "qp_continuous_enabled" in qp_result.loss_breakdown
```

- [ ] **Step 2: Run focused isolation tests**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_route_does_not_modify_current_mpc_backend -q
```

Expected: pass.

- [ ] **Step 3: Run broader static regression**

Run:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py -q
```

Expected: pass or only known unrelated warnings.

### Task 6: Notes, Logs, And Real IsaacLab Smoke

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-07-06-mpc-qp-continuous-bezier-trajectory.md`

**Interfaces:**
- Produces: current-memory alignment for the new continuous trajectory mainline.

- [ ] **Step 1: Run pycompile and diff check**

Run:

```bash
python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_qp_planner/bezier.py \
  Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py \
  Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py \
  Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py
git diff --check
```

Expected: exit `0`.

- [ ] **Step 2: Run real GPU1 smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_continuous_16.json --planner-backend mpc_qp --qp-iterations 1
```

Expected: exit `0`, at least one QP replan, finite cache, no OOM.

- [ ] **Step 3: Update notes**

Record focused/regression/smoke results in `notes/log/2026-07-06-mpc-qp-continuous-bezier-trajectory.md`, add it to `notes/log/index.md`, update `notes/todo.md` current focus, and update the T302v branch page.

## Self-Review

- Spec coverage: continuous trajectory sampling, touchdown z terrain binding, four-leg/root joint intent, fixed-shape memory policy, loss-first tuning policy, and `mpc`/`mpc_qp` isolation are covered.
- Placeholder scan: no task uses TBD/TODO/fill-in language. Task 2 and Task 3 describe implementation behavior because the exact code depends on existing result shapes, but the tests define the required interfaces.
- Type consistency: all produced functions use `Tensor`, `MpcPlannerResult`, and `MpcPlannerTerrain` consistently.
