# Parallelism Semantic Margin Debug Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add signed velocity sliders, a single adjustable touchdown semantic margin, and a working Go2 mesh visibility toggle for the Parallelism viewer backend.

**Architecture:** The planner keeps using high-resolution `ParallelismTerrain.semantic_id`; a new Torch helper builds an expanded obstacle mask with `max_pool2d`, and both nominal candidates and FK touchdowns query that mask. The viewer panel writes signed commands and `semantic_touchdown_margin_m` into `ParallelismCfg`, while `show mesh` controls USD visual prim visibility without affecting planner playback or marker visibility.

**Tech Stack:** Python 3.10, PyTorch/Torch tensor logic, IsaacSim/IsaacLab viewer APIs, pytest in `env_isaacsim`.

## Global Constraints

- Only touch `extension/parallelism`, `extension/viz/go2_foostep_planner.py`, parallelism tests, and this plan.
- Do not modify or reuse `extension/joint_mpc_rti` code.
- Keep active-leg-only collision behavior from the official collision design.
- Keep `standstill_fallback_enabled` behavior unchanged.
- Add only one semantic margin parameter: `semantic_touchdown_margin_m`.
- `semantic_touchdown_margin_m` default is `0.0 m`; viewer slider range is `[0.0, 0.12] m`.
- Velocity panel ranges are `vx [-1.0, 1.0]`, `vy [-0.5, 0.5]`, `vyaw [-1.0, 1.0]`.
- All semantic margin filtering must use Torch tensor logic and preserve env/leg/candidate parallelism.

---

### Task 1: Terrain Semantic Expansion Helper

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/config.py`
- Modify: `Go2Pvcnn/extension/parallelism/terrain.py`
- Test: `Go2Pvcnn/tests/parallelism/test_contracts.py`

**Interfaces:**
- Consumes: `ParallelismTerrain`, `ParallelismCfg.obstacle_semantic_ids`
- Produces: `expanded_obstacle_mask(terrain: ParallelismTerrain, obstacle_semantic_ids: tuple[int, ...], margin_m: float) -> torch.Tensor`
- Produces: `query_expanded_obstacle(terrain: ParallelismTerrain, points_w: Tensor, obstacle_mask: Tensor) -> Tensor`

- [ ] **Step 1: Write failing tests**

Add tests proving `semantic_touchdown_margin_m` exists and that a point adjacent to an obstacle becomes blocked after expansion:

```python
def test_parallelism_cfg_exposes_semantic_touchdown_margin():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()

    assert cfg.semantic_touchdown_margin_m == 0.0


def test_expanded_obstacle_mask_blocks_neighboring_touchdown():
    import torch
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.terrain import expanded_obstacle_mask, query_expanded_obstacle

    semantic = torch.zeros(1, 5, 5, dtype=torch.long)
    semantic[:, 2, 2] = 1
    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 5, 5),
        semantic_id=semantic,
        valid_mask=torch.ones(1, 5, 5, dtype=torch.bool),
        origin_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.01,
    )

    mask = expanded_obstacle_mask(terrain, (1, 2), margin_m=0.01)
    points = torch.tensor([[[0.01, 0.02], [0.04, 0.04]]], dtype=torch.float32)
    blocked = query_expanded_obstacle(terrain, points, mask)

    assert blocked.tolist() == [[True, False]]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism/test_contracts.py -q
```

Expected: FAIL because the config field and helper functions do not exist.

- [ ] **Step 3: Implement minimal helper**

Add `semantic_touchdown_margin_m: float = 0.0` to `ParallelismCfg`.

In `terrain.py`, import `torch.nn.functional as F`, reuse `_world_to_grid`, and add:

```python
def expanded_obstacle_mask(
    terrain: ParallelismTerrain,
    obstacle_semantic_ids: tuple[int, ...],
    *,
    margin_m: float,
) -> Tensor:
    semantic = torch.as_tensor(terrain.semantic_id, dtype=torch.long, device=terrain.semantic_id.device)
    ids = torch.tensor(tuple(obstacle_semantic_ids), dtype=semantic.dtype, device=semantic.device)
    if int(ids.numel()) == 0:
        return torch.zeros_like(semantic, dtype=torch.bool)
    obstacle = (semantic[..., None] == ids.view(*((1,) * semantic.ndim), -1)).any(dim=-1)
    radius = int(torch.ceil(torch.tensor(max(float(margin_m), 0.0) / max(float(terrain.resolution), 1.0e-6))).item())
    if radius <= 0:
        return obstacle
    pooled = F.max_pool2d(obstacle.to(dtype=torch.float32).unsqueeze(1), kernel_size=2 * radius + 1, stride=1, padding=radius)
    return pooled[:, 0] > 0.0


def query_expanded_obstacle(
    terrain: ParallelismTerrain,
    points_w: Tensor,
    obstacle_mask: Tensor,
) -> Tensor:
    row, col, points = _world_to_grid(terrain, points_w)
    batch, height_count, width_count = obstacle_mask.shape
    batch_index = torch.arange(batch, device=points.device)[:, None].expand_as(row)
    inside = (row >= 0) & (row < height_count) & (col >= 0) & (col < width_count)
    safe_row = row.clamp(0, height_count - 1)
    safe_col = col.clamp(0, width_count - 1)
    hit = obstacle_mask[batch_index, safe_row, safe_col]
    return inside & hit
```

- [ ] **Step 4: Run tests to verify pass**

Run the same test command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/parallelism/config.py Go2Pvcnn/extension/parallelism/terrain.py Go2Pvcnn/tests/parallelism/test_contracts.py
git commit -m "feat: add parallelism semantic touchdown margin mask"
```

### Task 2: Planner Uses Expanded Semantic Mask

**Files:**
- Modify: `Go2Pvcnn/extension/parallelism/planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_planner.py`

**Interfaces:**
- Consumes: `expanded_obstacle_mask(...)`, `query_expanded_obstacle(...)`
- Produces: candidate reject bit 4 and FK touchdown reject bit 5 driven by expanded semantic obstacle hits.

- [ ] **Step 1: Write failing tests**

Add a test where raw candidate/FK touchdown semantic can remain terrain, but a neighboring semantic obstacle invalidates candidates when margin is enabled:

```python
def test_semantic_touchdown_margin_rejects_nearby_obstacle_cells():
    import torch
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    terrain = _terrain()
    semantic = terrain.semantic_id.clone()
    semantic[:, 29:32, 29:32] = 1
    terrain = type(terrain)(
        height_w=terrain.height_w,
        semantic_id=semantic,
        valid_mask=terrain.valid_mask,
        origin_w=terrain.origin_w,
        yaw_w=terrain.yaw_w,
        resolution=terrain.resolution,
    )

    no_margin = plan_trajectory(_state(), torch.zeros(1, 3), terrain, ParallelismCfg(semantic_touchdown_margin_m=0.0))
    with_margin = plan_trajectory(_state(), torch.zeros(1, 3), terrain, ParallelismCfg(semantic_touchdown_margin_m=0.2))

    assert int(with_margin.diagnostics.candidate_reject_bits[..., 4].sum().item()) >= int(no_margin.diagnostics.candidate_reject_bits[..., 4].sum().item())
    assert int(with_margin.diagnostics.candidate_reject_bits[..., 5].sum().item()) >= int(no_margin.diagnostics.candidate_reject_bits[..., 5].sum().item())
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism/test_planner.py -q
```

Expected: FAIL until planner uses expanded mask.

- [ ] **Step 3: Implement planner wiring**

Import the new helpers:

```python
from extension.parallelism.terrain import expanded_obstacle_mask, query_expanded_obstacle, query_height_semantic_valid
```

In `plan_trajectory`, build one expanded mask after candidates and landing query are available:

```python
semantic_obstacle = expanded_obstacle_mask(
    terrain,
    tuple(cfg.obstacle_semantic_ids),
    margin_m=float(cfg.semantic_touchdown_margin_m),
)
candidate_semantic_ok = ~query_expanded_obstacle(
    terrain,
    candidates.candidate_w[..., :2].reshape(batch, leg_count * candidate_count, 2),
    semantic_obstacle,
).reshape(batch, leg_count, candidate_count)
fk_touchdown_semantic_ok = ~query_expanded_obstacle(
    terrain,
    fk_touchdown[..., :2].reshape(batch, leg_count * candidate_count, 2),
    semantic_obstacle,
).reshape(batch, leg_count, candidate_count)
```

Keep `candidate_semantic` and `fk_touchdown_semantic` diagnostics as raw queried semantic ids.

- [ ] **Step 4: Run tests to verify pass**

Run the same planner test command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/parallelism/planner.py Go2Pvcnn/tests/parallelism/test_planner.py
git commit -m "feat: filter touchdowns with semantic margin"
```

### Task 3: Viewer Panel Values And Mesh Visibility

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Test: `Go2Pvcnn/tests/parallelism/test_viewer_adapter.py`

**Interfaces:**
- Consumes: `ViewerTestTerminalState.semantic_touchdown_margin_m`
- Produces: `_parallelism_cfg_from_viewer_args(args_cli, test_terminal_state) -> ParallelismCfg`
- Produces: `_set_go2_mesh_visibility(base_env, visible: bool) -> None`

- [ ] **Step 1: Write failing tests**

Add tests for signed velocity output and config construction:

```python
def test_test_terminal_command_supports_signed_velocity():
    import torch
    from extension.viz.go2_foostep_planner import ViewerTestTerminalState, _apply_test_terminal_command

    state = ViewerTestTerminalState(vx=-0.4, vy=-0.2, vyaw=-0.7, enabled=True)
    command = _apply_test_terminal_command(torch.zeros(2, 3), state)

    assert torch.allclose(command, torch.tensor([[-0.4, -0.2, -0.7], [-0.4, -0.2, -0.7]]))


def test_parallelism_cfg_from_viewer_uses_semantic_margin():
    from argparse import Namespace
    from extension.viz.go2_foostep_planner import ViewerTestTerminalState, _parallelism_cfg_from_viewer_args

    cfg = _parallelism_cfg_from_viewer_args(
        Namespace(plan_dt=0.02),
        ViewerTestTerminalState(swing_height=0.11, semantic_touchdown_margin_m=0.04, standstill_fallback_enabled=False),
    )

    assert cfg.swing_height_m == 0.11
    assert cfg.semantic_touchdown_margin_m == 0.04
    assert cfg.standstill_fallback_enabled is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism/test_viewer_adapter.py -q
```

Expected: FAIL because `semantic_touchdown_margin_m` and `_parallelism_cfg_from_viewer_args` do not exist.

- [ ] **Step 3: Implement viewer wiring**

Add `semantic_touchdown_margin_m: float = 0.0` to `ViewerTestTerminalState`.

Update sliders:

```python
_slider("vx", "vx", -1.0, 1.0)
_slider("vy", "vy", -0.5, 0.5)
_slider("vyaw", "vyaw", -1.0, 1.0)
_slider("swing_height", "swing_height", 0.0, 0.25)
_slider("semantic_margin", "semantic_touchdown_margin_m", 0.0, 0.12)
```

Extract config construction:

```python
def _parallelism_cfg_from_viewer_args(args_cli: argparse.Namespace, test_terminal_state: ViewerTestTerminalState | None):
    from extension.parallelism import ParallelismCfg

    return ParallelismCfg(
        dt=float(args_cli.plan_dt),
        swing_height_m=float(test_terminal_state.swing_height) if test_terminal_state is not None else ParallelismCfg.swing_height_m,
        semantic_touchdown_margin_m=float(test_terminal_state.semantic_touchdown_margin_m)
        if test_terminal_state is not None
        else ParallelismCfg.semantic_touchdown_margin_m,
        standstill_fallback_enabled=bool(test_terminal_state.standstill_fallback_enabled)
        if test_terminal_state is not None
        else True,
    )
```

Use the helper in `_plan_parallelism_viewer_trajectory`.

Add `_set_go2_mesh_visibility(base_env, visible)` that locates the robot prim path from the scene articulation and applies USD `visibility` to mesh/imageable child prims. Call it when the checkbox changes or once per loop when flags are available.

- [ ] **Step 4: Run tests to verify pass**

Run the same viewer adapter test command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/parallelism/test_viewer_adapter.py
git commit -m "feat: expose semantic margin in parallelism viewer"
```

### Task 4: Full Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes all previous task outputs.
- Produces final confidence before reporting.

- [ ] **Step 1: Run syntax checks**

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/parallelism/config.py Go2Pvcnn/extension/parallelism/terrain.py Go2Pvcnn/extension/parallelism/planner.py Go2Pvcnn/extension/viz/go2_foostep_planner.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run parallelism tests**

```bash
PYTHONPATH=Go2Pvcnn /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/parallelism -q
```

Expected: all tests pass.

- [ ] **Step 3: Inspect git diff**

```bash
git status --short --branch
git diff --stat HEAD
```

Expected: only intended files changed after the last commit, with unrelated dirty `joint_mpc_rti` files ignored.
