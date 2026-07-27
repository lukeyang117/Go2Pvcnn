# Parallelism Flat Foot Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Go2Pvcnn/extension/parallelism/`, a self-contained GPU/batched flat Go2 foot planner that emits 24-frame root/joint/foot/contact trajectories and can be viewed with `go2_foostep_planner.py --planner-backend parallelism`.

**Architecture:** The package is self-contained: no imports from non-viz `extension.*` modules. It owns terrain queries, Go2 FK/IK, root rollout, candidate generation, torch-mask hard filters, single-pass score/argmin selection, RL reference adaptation, and viewer adaptation. Core tensors keep a leading batch dimension and run as packed torch operations.

**Tech Stack:** Python 3.10, PyTorch tensor ops, pytest CPU unit tests, existing Isaac Lab viewer entrypoint for later smoke.

## Global Constraints

- Target package: `Go2Pvcnn/extension/parallelism/`.
- Only non-package extension integration allowed: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`.
- Do not import `extension.joint_mpc_rti`, `extension.batch_mpc_planner`, or other non-viz `extension` planner modules from `extension.parallelism`.
- Horizon is exactly 24 frames at `dt=0.02`.
- Gait is fixed trot: frames `0..11` swing `FL/RR`, frames `12..23` swing `FR/RL`.
- Candidate count is exactly 50 per leg, 200 total per environment, no `50^4` combination search.
- Command is body/root-frame `[vx, vy, vyaw]`, clamped to `[-1,1]`, `[-0.5,0.5]`, `[-1,1]`.
- Root is predicted, not searched; root XY/yaw use normalized smoothstep half-cycle increments.
- Root z is stance foot height mean plus `0.3m`.
- Terrain input is height map, semantic map, valid mask, origin, yaw, resolution; no gradient/SDF/occupancy/distance field.
- Filter and score use torch tensor conditions, `torch.where`/`masked_fill`, and one `argmin`; no CPU/Python candidate loop, retry, ranked fallback, or second search.
- First version filters only valid map, joint limit, landing grounded, and leg-height collision for foot/knee/calf/thigh. No body/base filter.
- First version score is only velocity tracking relative to the fixed hip frame, including `vyaw` via `R(vyaw*T)*hip_offset-r`.
- RL adapter is shape-only/reference adaptation; it must not replan.

---

## File Structure

- Create `Go2Pvcnn/extension/parallelism/__init__.py`: public exports.
- Create `Go2Pvcnn/extension/parallelism/config.py`: dataclass config and Go2 constants.
- Create `Go2Pvcnn/extension/parallelism/types.py`: state, terrain, diagnostics, trajectory, RL reference dataclasses.
- Create `Go2Pvcnn/extension/parallelism/terrain.py`: batched world-to-grid nearest/bilinear query helpers.
- Create `Go2Pvcnn/extension/parallelism/kinematics.py`: self-contained RPY rotation, FK, hip positions, collision sample geometry.
- Create `Go2Pvcnn/extension/parallelism/ik.py`: self-contained batched analytic IK.
- Create `Go2Pvcnn/extension/parallelism/root.py`: command clamp, smoothstep rollout, root z support rule.
- Create `Go2Pvcnn/extension/parallelism/candidates.py`: 50-point golden-angle disk and candidate centers/score targets.
- Create `Go2Pvcnn/extension/parallelism/swing.py`: batched 12-frame swing targets and 24-frame foot target assembly.
- Create `Go2Pvcnn/extension/parallelism/planner.py`: single-pass planning orchestration.
- Create `Go2Pvcnn/extension/parallelism/rl_adapter.py`: batched reference view.
- Create `Go2Pvcnn/extension/parallelism/viewer_adapter.py`: convert to viewer result shape.
- Modify `Go2Pvcnn/extension/viz/go2_foostep_planner.py`: add `parallelism` backend choice and adapter route.
- Create `Go2Pvcnn/tests/parallelism/` test package with focused tests.

---

### Task 1: Public Types, Config, Terrain Query, And Import Isolation

**Files:**
- Create: `Go2Pvcnn/extension/parallelism/__init__.py`
- Create: `Go2Pvcnn/extension/parallelism/config.py`
- Create: `Go2Pvcnn/extension/parallelism/types.py`
- Create: `Go2Pvcnn/extension/parallelism/terrain.py`
- Create: `Go2Pvcnn/tests/parallelism/test_contracts.py`

**Interfaces:**
- Produces: `ParallelismCfg`, `ParallelismState`, `ParallelismTerrain`, `ParallelismDiagnostics`, `ParallelismTrajectory`, `ParallelismReference`, `query_height_semantic_valid(terrain, points_w)`.
- Later tasks consume all these names.

- [ ] **Step 1: Write failing tests**

Add `Go2Pvcnn/tests/parallelism/test_contracts.py`:

```python
from __future__ import annotations

import importlib
import sys

import torch


def test_parallelism_import_isolation():
    before = set(sys.modules)
    module = importlib.import_module("extension.parallelism")
    after = set(sys.modules)
    newly_loaded = after - before
    forbidden = {
        name
        for name in newly_loaded
        if name.startswith("extension.joint_mpc_rti")
        or name.startswith("extension.batch_mpc_planner")
    }
    assert forbidden == set()
    assert hasattr(module, "ParallelismCfg")


def test_terrain_query_batched_points():
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.terrain import query_height_semantic_valid

    height = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
    semantic = torch.full((1, 5, 5), 7, dtype=torch.long)
    valid = torch.ones((1, 5, 5), dtype=torch.bool)
    terrain = ParallelismTerrain(
        height_w=height,
        semantic_id=semantic,
        valid_mask=valid,
        origin_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    points = torch.tensor([[[0.0, 0.0], [0.2, 0.2], [1.0, 1.0]]], dtype=torch.float32)
    result = query_height_semantic_valid(terrain, points)

    assert result.height.shape == (1, 3)
    assert result.semantic.shape == (1, 3)
    assert result.valid.tolist() == [[True, True, False]]
    assert result.semantic[0, 0].item() == 7
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism/test_contracts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'extension.parallelism'`.

- [ ] **Step 3: Implement minimal package, dataclasses, and terrain query**

Create `Go2Pvcnn/extension/parallelism/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelismCfg:
    horizon: int = 24
    dt: float = 0.02
    half_cycle: int = 12
    candidate_radius_m: float = 0.24
    candidates_per_leg: int = 50
    root_clearance_m: float = 0.30
    swing_height_m: float = 0.08
    landing_tolerance_m: float = 0.025
    collision_margin_m: float = 0.003
    vx_limit: float = 1.0
    vy_limit: float = 0.5
    vyaw_limit: float = 1.0
    foot_radius_m: float = 0.022
    knee_radius_m: float = 0.030
    calf_radius_m: float = 0.015
    thigh_radius_m: float = 0.035
    capsule_samples: int = 5
```

Create `Go2Pvcnn/extension/parallelism/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ParallelismState:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_pos: Tensor
    foot_pos_w: Tensor | None = None


@dataclass(frozen=True)
class ParallelismTerrain:
    height_w: Tensor
    semantic_id: Tensor
    valid_mask: Tensor
    origin_w: Tensor
    yaw_w: Tensor
    resolution: float


@dataclass(frozen=True)
class TerrainQueryResult:
    height: Tensor
    semantic: Tensor
    valid: Tensor


@dataclass(frozen=True)
class ParallelismDiagnostics:
    candidate_w: Tensor
    candidate_score: Tensor
    candidate_valid: Tensor
    candidate_reject_bits: Tensor
    candidate_semantic: Tensor
    selected_index: Tensor


@dataclass(frozen=True)
class ParallelismTrajectory:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_pos: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    valid: Tensor
    selected_foothold_w: Tensor
    selected_score: Tensor
    diagnostics: ParallelismDiagnostics


@dataclass(frozen=True)
class ParallelismReference:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_pos: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    valid: Tensor
```

Create `Go2Pvcnn/extension/parallelism/terrain.py`:

```python
from __future__ import annotations

import torch
from torch import Tensor

from extension.parallelism.types import ParallelismTerrain, TerrainQueryResult


def _world_to_grid(terrain: ParallelismTerrain, points_w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    points = torch.as_tensor(points_w, dtype=terrain.height_w.dtype, device=terrain.height_w.device)
    origin_xy = terrain.origin_w[:, None, :2].to(dtype=points.dtype, device=points.device)
    yaw = terrain.yaw_w[:, None].to(dtype=points.dtype, device=points.device)
    delta = points - origin_xy
    c = torch.cos(-yaw)
    s = torch.sin(-yaw)
    gx_m = c * delta[..., 0] - s * delta[..., 1]
    gy_m = s * delta[..., 0] + c * delta[..., 1]
    col = torch.round(gx_m / float(terrain.resolution)).to(torch.long)
    row = torch.round(gy_m / float(terrain.resolution)).to(torch.long)
    return row, col, points


def query_height_semantic_valid(terrain: ParallelismTerrain, points_w: Tensor) -> TerrainQueryResult:
    row, col, points = _world_to_grid(terrain, points_w)
    batch, height_count, width_count = terrain.height_w.shape
    batch_index = torch.arange(batch, device=points.device)[:, None].expand_as(row)
    inside = (row >= 0) & (row < height_count) & (col >= 0) & (col < width_count)
    safe_row = row.clamp(0, height_count - 1)
    safe_col = col.clamp(0, width_count - 1)
    height = terrain.height_w[batch_index, safe_row, safe_col]
    semantic = terrain.semantic_id[batch_index, safe_row, safe_col]
    valid = inside & terrain.valid_mask[batch_index, safe_row, safe_col]
    return TerrainQueryResult(height=height, semantic=semantic, valid=valid)
```

Create `Go2Pvcnn/extension/parallelism/__init__.py`:

```python
from extension.parallelism.config import ParallelismCfg
from extension.parallelism.types import (
    ParallelismDiagnostics,
    ParallelismReference,
    ParallelismState,
    ParallelismTerrain,
    ParallelismTrajectory,
    TerrainQueryResult,
)

__all__ = [
    "ParallelismCfg",
    "ParallelismDiagnostics",
    "ParallelismReference",
    "ParallelismState",
    "ParallelismTerrain",
    "ParallelismTrajectory",
    "TerrainQueryResult",
]
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism/test_contracts.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/parallelism Go2Pvcnn/tests/parallelism/test_contracts.py
git commit -m "feat: add parallelism contracts and terrain query"
```

---

### Task 2: Self-Contained Kinematics, IK, Root Rollout, And Candidates

**Files:**
- Create: `Go2Pvcnn/extension/parallelism/kinematics.py`
- Create: `Go2Pvcnn/extension/parallelism/ik.py`
- Create: `Go2Pvcnn/extension/parallelism/root.py`
- Create: `Go2Pvcnn/extension/parallelism/candidates.py`
- Create: `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`

**Interfaces:**
- Consumes: `ParallelismCfg`, `ParallelismState`, `ParallelismTerrain`, `query_height_semantic_valid`.
- Produces: `fk_go2(root_pos_w, root_rpy_w, joint_pos)`, `ik_go2(root_pos_w, root_rpy_w, foot_target_w)`, `rollout_root(state, command_body, terrain, cfg)`, `build_candidates(root, state, command, terrain, cfg)`.

- [ ] **Step 1: Write failing tests**

Add `Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py`:

```python
from __future__ import annotations

import torch


def _terrain(batch: int = 1):
    from extension.parallelism import ParallelismTerrain

    return ParallelismTerrain(
        height_w=torch.zeros(batch, 41, 41),
        semantic_id=torch.zeros(batch, 41, 41, dtype=torch.long),
        valid_mask=torch.ones(batch, 41, 41, dtype=torch.bool),
        origin_w=torch.tensor([[-2.0, -2.0, 0.0]], dtype=torch.float32).repeat(batch, 1),
        yaw_w=torch.zeros(batch),
        resolution=0.1,
    )


def _state(batch: int = 1):
    from extension.parallelism import ParallelismState

    return ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32).repeat(batch, 1),
        root_rpy_w=torch.zeros(batch, 3),
        joint_pos=torch.tensor([[0.0, 0.8, -1.5] * 4], dtype=torch.float32).repeat(batch, 1),
    )


def test_root_rollout_body_command_half_cycle_displacement():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import rollout_root

    cfg = ParallelismCfg()
    result = rollout_root(_state(), torch.tensor([[1.0, 0.0, 0.0]]), _terrain(), cfg)
    first_half = result.root_pos_w[0, 11, 0] - result.root_pos_w[0, 0, 0]
    full = result.root_pos_w[0, 23, 0] - result.root_pos_w[0, 0, 0]

    assert result.root_pos_w.shape == (1, 24, 3)
    assert torch.isclose(first_half, torch.tensor(12 * cfg.dt), atol=1e-5)
    assert torch.isclose(full, torch.tensor(24 * cfg.dt), atol=1e-5)
    assert torch.allclose(result.root_pos_w[..., 2], torch.full((1, 24), 0.30), atol=1e-6)


def test_candidate_shape_and_reference_hips():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.root import rollout_root

    cfg = ParallelismCfg()
    state = _state()
    terrain = _terrain()
    root = rollout_root(state, torch.zeros(1, 3), terrain, cfg)
    candidates = build_candidates(root, state, torch.zeros(1, 3), terrain, cfg)
    radius = torch.linalg.vector_norm(candidates.offset_body, dim=-1)

    assert candidates.candidate_w.shape == (1, 4, 50, 3)
    assert candidates.score_target_body.shape == (1, 4, 2)
    assert torch.all(radius <= cfg.candidate_radius_m + 1e-6)
    assert candidates.hip_ref_w.shape == (1, 4, 3)


def test_ik_fk_round_trip_default_pose():
    from extension.parallelism.kinematics import fk_go2
    from extension.parallelism.ik import ik_go2

    state = _state()
    geometry = fk_go2(state.root_pos_w, state.root_rpy_w, state.joint_pos)
    joint, reachable = ik_go2(state.root_pos_w, state.root_rpy_w, geometry.foot_pos_w)
    round_trip = fk_go2(state.root_pos_w, state.root_rpy_w, joint.reshape(1, 12))

    assert reachable.all()
    assert torch.allclose(round_trip.foot_pos_w, geometry.foot_pos_w, atol=1e-5)
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py -q
```

Expected: FAIL with missing modules/functions.

- [ ] **Step 3: Implement kinematics, IK, root, candidates**

Implement self-contained Go2 geometry in `kinematics.py` and `ik.py` using the formulas from the spec. Use constants:

```python
HIP_OFFSETS = ((0.1934, 0.0465, 0.0), (0.1934, -0.0465, 0.0), (-0.1934, 0.0465, 0.0), (-0.1934, -0.0465, 0.0))
LEG_SIDE_SIGNS = (1.0, -1.0, 1.0, -1.0)
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_OFFSET_Y = 0.0955
JOINT_LOWER = (-1.0472, -0.6632, -2.721)
JOINT_UPPER = (1.0472, 2.966, -0.837)
```

Implement `rollout_root()` with:

```python
weights = torch.linspace(0.0, 1.0, 12, device=device, dtype=dtype)
smooth = weights * weights * (3.0 - 2.0 * weights)
increments = smooth - torch.cat((smooth[:1].new_zeros(1), smooth[:-1]))
increments = increments / increments.sum().clamp_min(1e-8)
```

Build 24 increments by repeating the half-cycle weights twice. Use `query_height_semantic_valid()` on stance foot XY to set root z to stance mean height plus `cfg.root_clearance_m`.

Implement `build_candidates()` with a golden-angle disk:

```python
idx = torch.arange(cfg.candidates_per_leg, device=device, dtype=dtype)
radius = cfg.candidate_radius_m * torch.sqrt((idx + 0.5) / cfg.candidates_per_leg)
theta = idx * (math.pi * (3.0 - math.sqrt(5.0)))
offset = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py Go2Pvcnn/tests/parallelism/test_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/parallelism Go2Pvcnn/tests/parallelism
git commit -m "feat: add parallelism root candidates and kinematics"
```

---

### Task 3: Swing, Torch Filter/Score, Single-Pass Planner, And RL Adapter

**Files:**
- Create: `Go2Pvcnn/extension/parallelism/swing.py`
- Create: `Go2Pvcnn/extension/parallelism/planner.py`
- Create: `Go2Pvcnn/extension/parallelism/rl_adapter.py`
- Create: `Go2Pvcnn/tests/parallelism/test_planner.py`

**Interfaces:**
- Consumes: Task 1 and 2 functions.
- Produces: `plan_trajectory(state, command_body, terrain, cfg) -> ParallelismTrajectory`, `trajectory_to_reference(trajectory) -> ParallelismReference`.

- [ ] **Step 1: Write failing tests**

Add `Go2Pvcnn/tests/parallelism/test_planner.py`:

```python
from __future__ import annotations

import inspect

import torch


def _terrain(batch: int = 1, *, invalid: bool = False):
    from extension.parallelism import ParallelismTerrain

    valid = torch.ones(batch, 61, 61, dtype=torch.bool)
    if invalid:
        valid[:] = False
    return ParallelismTerrain(
        height_w=torch.zeros(batch, 61, 61),
        semantic_id=torch.zeros(batch, 61, 61, dtype=torch.long),
        valid_mask=valid,
        origin_w=torch.tensor([[-3.0, -3.0, 0.0]], dtype=torch.float32).repeat(batch, 1),
        yaw_w=torch.zeros(batch),
        resolution=0.1,
    )


def _state(batch: int = 1):
    from extension.parallelism import ParallelismState

    return ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32).repeat(batch, 1),
        root_rpy_w=torch.zeros(batch, 3),
        joint_pos=torch.tensor([[0.0, 0.8, -1.5] * 4], dtype=torch.float32).repeat(batch, 1),
    )


def test_full_flat_trajectory_contract():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    traj = plan_trajectory(_state(), torch.tensor([[0.2, 0.0, 0.0]]), _terrain(), ParallelismCfg())

    assert traj.root_pos_w.shape == (1, 24, 3)
    assert traj.joint_pos.shape == (1, 24, 12)
    assert traj.foot_pos_w.shape == (1, 24, 4, 3)
    assert traj.contact_state.shape == (1, 24, 4)
    assert traj.valid.shape == (1,)
    assert traj.diagnostics.candidate_w.shape == (1, 4, 50, 3)
    assert traj.diagnostics.candidate_reject_bits.shape == (1, 4, 50, 4)


def test_invalid_map_makes_trajectory_invalid_single_pass():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    traj = plan_trajectory(_state(), torch.zeros(1, 3), _terrain(invalid=True), ParallelismCfg())

    assert not bool(traj.valid[0])
    assert not traj.diagnostics.candidate_valid.any()


def test_filter_score_source_uses_torch_conditions():
    import extension.parallelism.planner as planner

    source = inspect.getsource(planner)
    assert "torch.where" in source
    assert ".argmin(" in source
    assert "reject_bits = torch.stack" in source
    assert "for candidate" not in source


def test_parallel_batch_contract():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    traj = plan_trajectory(_state(8), torch.zeros(8, 3), _terrain(8), ParallelismCfg())

    assert traj.root_pos_w.shape[0] == 8
    assert traj.diagnostics.candidate_score.shape == (8, 4, 50)


def test_rl_adapter_shape_contract():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.rl_adapter import trajectory_to_reference

    traj = plan_trajectory(_state(2), torch.zeros(2, 3), _terrain(2), ParallelismCfg())
    ref = trajectory_to_reference(traj)

    assert ref.root_pos_w.shape == (2, 24, 3)
    assert ref.foot_pos_w.shape == (2, 24, 4, 3)
    assert torch.equal(ref.valid, traj.valid)
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism/test_planner.py -q
```

Expected: FAIL with missing `plan_trajectory`.

- [ ] **Step 3: Implement swing and planner**

Implement `swing.py`:

```python
def swing_curve(start_w, touchdown_w, *, frames: int, height_m: float) -> Tensor:
    tau = torch.linspace(0.0, 1.0, frames, dtype=start_w.dtype, device=start_w.device)
    xyz = (1.0 - tau.view(1, 1, 1, -1, 1)) * start_w[..., None, :] + tau.view(1, 1, 1, -1, 1) * touchdown_w[..., None, :]
    xyz[..., 2] = xyz[..., 2] + float(height_m) * 4.0 * tau * (1.0 - tau)
    return xyz
```

Implement `planner.py` with this orchestration:

```python
command = clamp_command(command_body, cfg)
root = rollout_root(state, command, terrain, cfg)
candidates = build_candidates(root, state, command, terrain, cfg)
candidate foot targets -> ik_go2 -> fk_go2
valid_map_ok, joint_ok, landing_ok, collision_ok = torch bool masks
candidate_valid = valid_map_ok & joint_ok & landing_ok & collision_ok
reject_bits = torch.stack((~valid_map_ok, ~joint_ok, ~landing_ok, ~collision_ok), dim=-1)
score_raw = tracking_score(...)
score = torch.where(candidate_valid, score_raw, torch.full_like(score_raw, torch.inf))
selected_index = score.argmin(dim=-1)
selected_valid = torch.isfinite(selected_score).all(dim=-1)
assemble selected 24-frame targets, joint_pos, foot_pos, contact_state
```

Keep the implementation direct and deterministic. If collision sampling cannot be fully conservative in this task, implement the foot/knee/calf/thigh tensor path and keep margins configurable; do not add body/base or retry logic.

Implement `rl_adapter.py`:

```python
from extension.parallelism.types import ParallelismReference, ParallelismTrajectory


def trajectory_to_reference(trajectory: ParallelismTrajectory) -> ParallelismReference:
    return ParallelismReference(
        root_pos_w=trajectory.root_pos_w,
        root_rpy_w=trajectory.root_rpy_w,
        joint_pos=trajectory.joint_pos,
        foot_pos_w=trajectory.foot_pos_w,
        contact_state=trajectory.contact_state,
        valid=trajectory.valid,
    )
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism/test_planner.py Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py Go2Pvcnn/tests/parallelism/test_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/parallelism Go2Pvcnn/tests/parallelism
git commit -m "feat: add single pass parallelism planner"
```

---

### Task 4: Viewer Adapter And Backend CLI Route

**Files:**
- Create: `Go2Pvcnn/extension/parallelism/viewer_adapter.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py`
- Create: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Consumes: `ParallelismTrajectory`.
- Produces: `parallelism_trajectory_to_viewer_result(trajectory)`, viewer CLI accepts `--planner-backend parallelism`.

- [ ] **Step 1: Write failing tests**

Add `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`:

```python
from __future__ import annotations

import torch


def test_viewer_adapter_contract():
    from extension.parallelism import ParallelismDiagnostics, ParallelismTrajectory
    from extension.parallelism.viewer_adapter import parallelism_trajectory_to_viewer_result

    traj = ParallelismTrajectory(
        root_pos_w=torch.zeros(1, 24, 3),
        root_rpy_w=torch.zeros(1, 24, 3),
        joint_pos=torch.zeros(1, 24, 12),
        foot_pos_w=torch.zeros(1, 24, 4, 3),
        contact_state=torch.zeros(1, 24, 4, dtype=torch.bool),
        valid=torch.ones(1, dtype=torch.bool),
        selected_foothold_w=torch.zeros(1, 4, 3),
        selected_score=torch.zeros(1, 4),
        diagnostics=ParallelismDiagnostics(
            candidate_w=torch.zeros(1, 4, 50, 3),
            candidate_score=torch.zeros(1, 4, 50),
            candidate_valid=torch.ones(1, 4, 50, dtype=torch.bool),
            candidate_reject_bits=torch.zeros(1, 4, 50, 4, dtype=torch.bool),
            candidate_semantic=torch.zeros(1, 4, 50, dtype=torch.long),
            selected_index=torch.zeros(1, 4, dtype=torch.long),
        ),
    )
    result = parallelism_trajectory_to_viewer_result(traj)

    assert result.num_frames == 24
    assert result.root_pos_w.shape == (1, 24, 3)
    assert result.foot_pos_w.shape == (1, 24, 4, 3)
    assert result.planned_touchdown_w.shape == (1, 4, 3)
```

Modify `Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py` with:

```python
def test_viewer_exposes_parallelism_backend():
    source = VIEWER_FILE.read_text(encoding="utf-8")

    assert '"parallelism"' in source
    assert "parallelism_trajectory_to_viewer_result" in source
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py -q
```

Expected: FAIL for missing viewer adapter/backend string.

- [ ] **Step 3: Implement viewer adapter and CLI route**

Create `viewer_adapter.py` with a local quaternion helper and `SimpleNamespace` result:

```python
from __future__ import annotations

from types import SimpleNamespace

import torch

from extension.parallelism.types import ParallelismTrajectory


def _yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw * 0.5
    return torch.stack((torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)), dim=-1)


def parallelism_trajectory_to_viewer_result(trajectory: ParallelismTrajectory):
    root_quat_w = _yaw_to_quat_wxyz(trajectory.root_rpy_w[..., 2])
    return SimpleNamespace(
        num_frames=int(trajectory.root_pos_w.shape[1]),
        root_pos_w=trajectory.root_pos_w,
        root_quat_w=root_quat_w,
        joint_angles=trajectory.joint_pos,
        foot_pos_w=trajectory.foot_pos_w,
        foot_pos_root=trajectory.foot_pos_w - trajectory.root_pos_w.unsqueeze(2),
        contact_state=trajectory.contact_state,
        planned_touchdown_w=trajectory.selected_foothold_w,
        feasible=trajectory.valid,
        status=(~trajectory.valid).to(torch.long),
        safe_fallback=torch.zeros_like(trajectory.valid),
        parallelism_diagnostics=trajectory.diagnostics,
    )
```

Modify `go2_foostep_planner.py`:

- Add `"parallelism"` to `--planner-backend` choices.
- Import `parallelism_trajectory_to_viewer_result` only inside the parallelism branch.
- For first CLI route, if backend is `parallelism`, call the planner from current Isaac state and scanner terrain, then adapt to viewer result. If full Isaac scanner extraction is too broad for this task, wire the backend option and adapter tests first, then leave the live viewer smoke to Task 5.

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/parallelism/viewer_adapter.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/parallelism/test_viewer_adapter.py Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py
git commit -m "feat: expose parallelism viewer backend"
```

---

### Task 5: Focused Verification And Notes Alignment

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T302v-joint-mpc-rti-gpu.md` or create `notes/todo/T303-parallelism-flat-foot-planner.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-07-27-parallelism-flat-foot-planner-implementation.md`

**Interfaces:**
- Consumes: completed code and test evidence.
- Produces: repository memory for the new Parallelism branch.

- [ ] **Step 1: Run focused verification**

Run:

```bash
pytest Go2Pvcnn/tests/parallelism Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run import smoke**

Run:

```bash
python - <<'PY'
import torch
from extension.parallelism import ParallelismCfg
print(ParallelismCfg().horizon)
PY
```

Expected output: `24`.

- [ ] **Step 3: Write log**

Create `notes/log/2026-07-27-parallelism-flat-foot-planner-implementation.md` with:

```markdown
# Parallelism Flat Foot Planner Implementation

## Purpose

Implement the first flat/highmap Parallelism Go2 foot planner from the approved design.

## Stage

`extension/parallelism` flat foot planner.

## Related Todo

T303 Parallelism flat foot planner.

## Command

`pytest Go2Pvcnn/tests/parallelism Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py -q`

## Result

Record the exact passing test count here after running verification.

## Conclusion

Parallelism has a self-contained batched planner, torch single-pass candidate selection, RL adapter shape boundary, and viewer backend adapter. Real Isaac viewer smoke remains a follow-up unless run in this task.
```

- [ ] **Step 4: Update todo dashboard**

Add a compact active/verify line for Parallelism and link the log.

- [ ] **Step 5: Commit**

```bash
git add notes/todo.md notes/todo notes/log/index.md notes/log/2026-07-27-parallelism-flat-foot-planner-implementation.md
git commit -m "docs: record parallelism planner verification"
```

---

## Self-Review

- Spec coverage: tasks cover self-contained package, terrain height/semantic/valid query, root smoothstep rollout, four-leg 50-point candidates, self-contained IK/FK, torch single-pass filters/scores, no body/base, RL boundary, viewer backend, and tests.
- Placeholder scan: no unresolved markers are allowed during execution; task code blocks provide concrete file contents or exact implementation formulas.
- Type consistency: `ParallelismCfg`, `ParallelismState`, `ParallelismTerrain`, `ParallelismTrajectory`, `ParallelismDiagnostics`, and `ParallelismReference` are introduced in Task 1 and reused consistently.
