# Parallelism Terrain-Aware Swing Clearance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build terrain-aware parabolic swing for the parallelism Go2 foot planner so small-obstacle crossing can pass hard filters without changing touchdown score or relaxing collision checks.

**Architecture:** Add `swing_clearance_m` and `min_swing_apex_m` to `ParallelismCfg`, then add a terrain-aware swing function that samples the XY path height and solves a per-swing scalar parabola apex `A`. Replace planner collision-check and final trajectory assembly swing generation with the new function so validation and playback use the same trajectory. Update the viewer debug panel to expose only `swing_clearance_m`.

**Tech Stack:** Python 3.10, PyTorch tensor operations, IsaacLab runtime tests, existing `extension.parallelism` package and `extension.viz.go2_foostep_planner.py`.

## Global Constraints

- Do not reuse `extension.joint_mpc_rti` or `extension.batch_mpc_planner` code.
- Do not change touchdown candidate sampling, velocity tracking score, semantic touchdown margin, or official collision shape parameters.
- Do not relax hard filters: valid map, joint limits, landing, semantic touchdown, and collision remain hard constraints.
- Keep GPU parallelism: terrain height lookup and parabola apex computation must use torch tensor operations over candidate and frame dimensions.
- Viewer panel exposes `swing_clearance_m`; `min_swing_apex_m` remains config-only with default `0.08`.
- Tests must include small-obstacle IsaacLab coverage with `standstill_fallback_enabled=True`, root movement, no standstill, per-leg valid candidates, and root XY crossing through the small-obstacle region.

---

### Task 1: Terrain-Aware Swing Function

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/extension/parallelism/swing.py`
- Test: `Go2Pvcnn/tests/parallelism/test_swing.py`

**Interfaces:**
- Consumes: `extension.parallelism.terrain.query_height_semantic_valid(terrain, points_xy)`
- Produces: `terrain_aware_swing_curve(start_w: Tensor, touchdown_w: Tensor, terrain: ParallelismTerrain, *, frames: int, clearance_m: float, min_apex_m: float) -> Tensor`
- Produces: `ParallelismCfg.swing_clearance_m: float`
- Produces: `ParallelismCfg.min_swing_apex_m: float`

- [ ] **Step 1: Write failing swing tests**

Create `Go2Pvcnn/tests/parallelism/test_swing.py` with tests that prove the new function does not exist yet and defines the desired behavior:

```python
from __future__ import annotations

import torch


def _terrain_with_mid_obstacle():
    from extension.parallelism import ParallelismTerrain

    height = torch.zeros(1, 11, 11, dtype=torch.float32)
    height[:, 5, 10] = 0.16
    return ParallelismTerrain(
        height_w=height,
        semantic_id=torch.zeros(1, 11, 11, dtype=torch.long),
        valid_mask=torch.ones(1, 11, 11, dtype=torch.bool),
        origin_w=torch.tensor([[-0.5, -0.5, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )


def test_terrain_aware_swing_keeps_parabola_endpoints_and_clears_path_height():
    from extension.parallelism.swing import terrain_aware_swing_curve

    terrain = _terrain_with_mid_obstacle()
    start = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    touchdown = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)

    swing = terrain_aware_swing_curve(
        start,
        touchdown,
        terrain,
        frames=11,
        clearance_m=0.03,
        min_apex_m=0.08,
    )

    assert swing.shape == (1, 1, 11, 3)
    assert torch.allclose(swing[:, :, 0], start)
    assert torch.allclose(swing[:, :, -1], touchdown)
    assert torch.isclose(swing[0, 0, 5, 2], torch.tensor(0.19), atol=1e-5)

    tau = torch.linspace(0.0, 1.0, 11)
    shape = 4.0 * tau * (1.0 - tau)
    base_z = torch.zeros_like(shape)
    apex = torch.where(shape > 0.0, (swing[0, 0, :, 2] - base_z) / shape.clamp_min(1e-6), torch.zeros_like(shape))
    assert torch.allclose(apex[1:-1], torch.full((9,), apex[5]), atol=1e-5)


def test_terrain_aware_swing_uses_min_apex_on_flat_ground():
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.swing import terrain_aware_swing_curve

    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 7, 7),
        semantic_id=torch.zeros(1, 7, 7, dtype=torch.long),
        valid_mask=torch.ones(1, 7, 7, dtype=torch.bool),
        origin_w=torch.tensor([[-0.3, -0.3, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    start = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    touchdown = torch.tensor([[[0.6, 0.0, 0.0]]], dtype=torch.float32)

    swing = terrain_aware_swing_curve(
        start,
        touchdown,
        terrain,
        frames=7,
        clearance_m=0.0,
        min_apex_m=0.08,
    )

    assert torch.isclose(swing[0, 0, 3, 2], torch.tensor(0.08), atol=1e-6)


def test_parallelism_cfg_exposes_swing_clearance_and_min_apex():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()

    assert cfg.swing_clearance_m == 0.05
    assert cfg.min_swing_apex_m == 0.08
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism/test_swing.py -q
```

Expected: FAIL because `terrain_aware_swing_curve` and config fields are not implemented.

- [ ] **Step 3: Implement config fields and terrain-aware swing**

In `Go2Pvcnn/extension/parallelism/config.py`, replace `swing_height_m` with:

```python
swing_clearance_m: float = 0.05
min_swing_apex_m: float = 0.08
```

In `Go2Pvcnn/extension/parallelism/swing.py`, add:

```python
from extension.parallelism.terrain import query_height_semantic_valid
from extension.parallelism.types import ParallelismTerrain


def terrain_aware_swing_curve(
    start_w: Tensor,
    touchdown_w: Tensor,
    terrain: ParallelismTerrain,
    *,
    frames: int,
    clearance_m: float,
    min_apex_m: float,
) -> Tensor:
    start = torch.as_tensor(start_w)
    touchdown = torch.as_tensor(touchdown_w, dtype=start.dtype, device=start.device)
    tau = torch.linspace(0.0, 1.0, int(frames), dtype=start.dtype, device=start.device)
    tau_view = tau.view(*((1,) * (start.ndim - 1)), int(frames), 1)
    curve = (1.0 - tau_view) * start[..., None, :] + tau_view * touchdown[..., None, :]
    shape = 4.0 * tau * (1.0 - tau)
    xy = curve[..., :2]
    batch = int(start.shape[0])
    query = query_height_semantic_valid(terrain, xy.reshape(batch, -1, 2))
    terrain_z = query.height.reshape(*xy.shape[:-1])
    base_z = curve[..., 2]
    safe_z = terrain_z + float(clearance_m)
    shape_view = shape.view(*((1,) * (base_z.ndim - 1)), int(frames))
    interior = shape_view > 1.0e-6
    required = torch.where(
        interior,
        (safe_z - base_z) / shape_view.clamp_min(1.0e-6),
        torch.zeros_like(base_z),
    )
    apex = torch.amax(required, dim=-1).clamp_min(float(min_apex_m))
    curve = curve.clone()
    curve[..., 2] = base_z + apex[..., None] * shape_view
    curve[..., 0, 2] = start[..., 2]
    curve[..., -1, 2] = touchdown[..., 2]
    return curve
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism/test_swing.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add Go2Pvcnn/extension/parallelism/config.py Go2Pvcnn/extension/parallelism/swing.py Go2Pvcnn/tests/parallelism/test_swing.py
git commit -m "feat: add terrain-aware parallelism swing curve"
```

### Task 2: Planner and Viewer Integration

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Consumes: `terrain_aware_swing_curve(...)`
- Produces: planner collision checking and assembled output trajectories use the same terrain-aware swing.
- Produces: viewer panel writes `ViewerTestTerminalState.swing_clearance_m` into `ParallelismCfg.swing_clearance_m`.

- [ ] **Step 1: Write failing planner/viewer tests**

Add to `Go2Pvcnn/tests/parallelism/test_planner.py`:

```python
def test_planner_output_swing_uses_terrain_aware_clearance():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    terrain = _terrain()
    height = terrain.height_w.clone()
    height[:, 30, 30] = 0.16
    terrain = type(terrain)(
        height_w=height,
        semantic_id=terrain.semantic_id,
        valid_mask=terrain.valid_mask,
        origin_w=terrain.origin_w,
        yaw_w=terrain.yaw_w,
        resolution=terrain.resolution,
    )

    traj = plan_trajectory(
        _state(),
        torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float32),
        terrain,
        ParallelismCfg(swing_clearance_m=0.03, min_swing_apex_m=0.08, standstill_fallback_enabled=False),
    )

    assert traj.foot_pos_w[..., 2].amax() >= 0.08
```

Update `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py` by replacing the old swing-height cfg test with:

```python
def test_parallelism_cfg_from_viewer_uses_swing_clearance():
    from argparse import Namespace
    from extension.viz.go2_foostep_planner import ViewerTestTerminalState, _parallelism_cfg_from_viewer_args

    cfg = _parallelism_cfg_from_viewer_args(
        Namespace(plan_dt=0.02),
        ViewerTestTerminalState(swing_clearance_m=0.12),
    )

    assert cfg.swing_clearance_m == 0.12
    assert cfg.min_swing_apex_m == 0.08
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism/test_planner.py::test_planner_output_swing_uses_terrain_aware_clearance Go2Pvcnn/tests/parallelism/test_viewer_adapter.py::test_parallelism_cfg_from_viewer_uses_swing_clearance -q
```

Expected: FAIL because planner and viewer still use `swing_height_m`.

- [ ] **Step 3: Integrate terrain-aware swing into planner**

In `Go2Pvcnn/extension/parallelism/planner.py`:

```python
from extension.parallelism.swing import swing_curve, terrain_aware_swing_curve
```

Replace `_swing_collision_mask()` swing construction with:

```python
swing = terrain_aware_swing_curve(
    current_foot[:, None, leg_idx].expand(batch, candidate_count, 3),
    candidates.candidate_w[:, leg_idx],
    terrain,
    frames=half_cycle,
    clearance_m=cfg.swing_clearance_m,
    min_apex_m=cfg.min_swing_apex_m,
)
```

Change `_assemble_foot_targets(...)` signature to include `terrain: ParallelismTerrain`, and replace final trajectory swing calls with:

```python
first_swing = terrain_aware_swing_curve(
    foot0[:, (0, 3)],
    selected_foothold_w[:, (0, 3)],
    terrain,
    frames=int(cfg.half_cycle),
    clearance_m=cfg.swing_clearance_m,
    min_apex_m=cfg.min_swing_apex_m,
)
second_swing = terrain_aware_swing_curve(
    foot0[:, (1, 2)],
    selected_foothold_w[:, (1, 2)],
    terrain,
    frames=int(cfg.half_cycle),
    clearance_m=cfg.swing_clearance_m,
    min_apex_m=cfg.min_swing_apex_m,
)
```

Call it with:

```python
foot_targets = _assemble_foot_targets(state, root.root_pos_w, root.root_rpy_w, selected_foothold, terrain, cfg)
```

- [ ] **Step 4: Integrate viewer panel config**

In `Go2Pvcnn/extension/viz/go2_foostep_planner.py`:

```python
class ViewerTestTerminalState:
    swing_clearance_m: float = 0.05
```

Replace the panel slider:

```python
_slider("swing_clearance_m", "swing_clearance_m", 0.0, 0.25)
```

Replace cfg wiring:

```python
swing_clearance_m=float(test_terminal_state.swing_clearance_m)
if test_terminal_state is not None
else ParallelismCfg.swing_clearance_m,
```

- [ ] **Step 5: Run targeted tests to verify GREEN**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism/test_swing.py Go2Pvcnn/tests/parallelism/test_planner.py Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add Go2Pvcnn/extension/parallelism/planner.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/parallelism/test_planner.py Go2Pvcnn/tests/parallelism/test_viewer_adapter.py
git commit -m "feat: use terrain-aware swing clearance in parallelism planner"
```

### Task 3: IsaacLab Parallel Runtime Verification

**Files:**
- Modify: `Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py` only if a reusable helper is needed.
- Create: `Go2Pvcnn/tests/parallelism/parallelism_small_obstacle_runtime_probe.py`

**Interfaces:**
- Consumes: `RealViewerRuntimeFixture`
- Produces: command-line probe that runs several velocity commands and prints JSON metrics for `standstill`, `valid_env`, `per_leg_valid`, `root_delta`, and `root_crosses_small`.

- [ ] **Step 1: Create runtime probe**

Create a probe that:

```python
commands = [
    (0.2, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.5, -0.5, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, -1.0),
    (0.5, 0.0, 1.0),
]
```

For each command, reset to the selected small-obstacle tile, run several replan/playback cycles with `standstill_fallback_enabled=True`, `candidate_radius_m=0.35`, and `swing_clearance_m=0.05`. For each cycle, print:

```json
{
  "cmd": [0.5, 0.0, 0.0],
  "cycle": 2,
  "standstill": false,
  "valid_env": true,
  "per_leg_valid": [12, 18, 11, 17],
  "root_delta_xy_norm": 0.24,
  "root_crosses_small": true
}
```

- [ ] **Step 2: Run runtime probe**

Run:

```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/tests /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/parallelism/parallelism_small_obstacle_runtime_probe.py
```

Expected: JSON lines showing no standstill, all `per_leg_valid` values greater than 0, and `root_crosses_small=true` for forward small-obstacle cases.

- [ ] **Step 3: Run full parallelism test suite**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism -q
```

Expected: PASS.

- [ ] **Step 4: Commit runtime probe if kept**

```bash
git add Go2Pvcnn/tests/parallelism/parallelism_small_obstacle_runtime_probe.py
git commit -m "test: add parallelism small obstacle runtime probe"
```

### Task 4: Final Verification and Reporting

**Files:**
- No code files unless verification finds a defect.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: final user-facing metrics.

- [ ] **Step 1: Run py_compile**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/parallelism/config.py Go2Pvcnn/extension/parallelism/swing.py Go2Pvcnn/extension/parallelism/planner.py Go2Pvcnn/extension/viz/go2_foostep_planner.py
```

Expected: exit code 0.

- [ ] **Step 2: Check git diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only intended parallelism, viewer, tests, and plan/spec files are changed; unrelated dirty files remain untouched.

- [ ] **Step 3: Report metrics**

Final response must include:

```text
- unit/parallelism pytest result
- runtime probe command result
- small-obstacle metrics: no standstill, root movement, root crosses small obstacle, per_leg_valid > 0
- files changed
```
