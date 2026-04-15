# Batched Planner Vectorization + Viz Alignment + TDD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate serial Python bottlenecks in the batched planner (swing_targets 91% of 700ms), align Isaac Lab visualization with raw kinematic reference, and build a cross-validated TDD test suite with raw/kinematic_footsteps as gold standard.

**Architecture:** Phase 0 establishes golden references and cross-validation tests. Phases 1-3 vectorize serial code (swing → terrain → foothold) using PyTorch tensor ops, verified by regression tests after each phase. Phase 4 adds Triton kernels if needed. Phase 5 rewrites visualization to pure kinematic playback. Each phase produces self-contained, tested changes.

**Tech Stack:** PyTorch (tensor ops, scatter_reduce, grid_sample), Triton (optional fused kernels), pytest, Isaac Lab (viz only)

**Spec:** `docs/superpowers/specs/2026-04-15-batched-planner-vectorize-viz-tdd-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|----------------|
| `Go2Pvcnn/tests/conftest.py` | Shared fixtures: synthetic states, terrain, commands, config alignment |
| `Go2Pvcnn/tests/fixtures/__init__.py` | Package marker |
| `Go2Pvcnn/tests/fixtures/terrain_adapter.py` | Bridge raw `GlobalElevationTerrain` and batched `PlannerTerrain` from same heightmap |
| `Go2Pvcnn/tests/fixtures/golden/` | Directory for serialized `.pt` golden reference tensors |
| `Go2Pvcnn/tests/test_cross_validation_raw.py` | L1: raw ↔ batched output cross-validation |
| `Go2Pvcnn/tests/test_swing_vectorized.py` | L2: swing vectorization regression |
| `Go2Pvcnn/tests/test_terrain_vectorized.py` | L2: terrain vectorization regression |
| `Go2Pvcnn/tests/test_foothold_vectorized.py` | L2: foothold vectorization regression |
| `Go2Pvcnn/tests/fixtures/generate_golden.py` | Script to regenerate golden `.pt` references |
| `Go2Pvcnn/tests/benchmarks/__init__.py` | Package marker |
| `Go2Pvcnn/tests/benchmarks/bench_planner_scaling.py` | L3: performance scaling benchmark |
| `Go2Pvcnn/tests/test_viz_playback.py` | L4: visualization playback logic |

### Modified files

| File | Changes |
|------|---------|
| `Go2Pvcnn/extension/batched_planner/swing.py` | Full vectorization: eliminate for loops, .item() calls |
| `Go2Pvcnn/extension/batched_planner/terrain.py` | `max_height_along_segment` batch vectorization |
| `Go2Pvcnn/extension/batched_planner/foothold.py` | meshgrid spiral offsets, lazy .item() in evaluate |
| `Go2Pvcnn/extension/batched_planner/trajectory.py` | Merge 4× per-leg terrain calls into one |
| `Go2Pvcnn/extension/viz/go2_foostep_planner.py` | Pure kinematic playback, plan-once/replay loop |
| `Go2Pvcnn/tests/test_batched_manager.py` | Extend with selective replan + dynamic N tests |

---

## Phase 0: Baseline & Cross-Validation Tests

### Task 1: Create test fixtures infrastructure

**Files:**
- Create: `Go2Pvcnn/tests/conftest.py`
- Create: `Go2Pvcnn/tests/fixtures/__init__.py`
- Create: `Go2Pvcnn/tests/fixtures/terrain_adapter.py`

- [ ] **Step 1: Create fixtures package and terrain adapter**

```python
# Go2Pvcnn/tests/fixtures/__init__.py
# empty package marker

# Go2Pvcnn/tests/fixtures/terrain_adapter.py
"""Bridge raw GlobalElevationTerrain and batched PlannerTerrain from same heightmap."""
import sys
from pathlib import Path
import numpy as np
import torch

RAW_ROOT = Path(__file__).resolve().parents[3] / "raw" / "kinematic_footsteps"
GO2_ROOT = Path(__file__).resolve().parents[2]
for p in (str(RAW_ROOT), str(GO2_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.go2fp.terrain import GlobalElevationTerrain
from extension.batched_planner.terrain import PlannerTerrain


def make_flat_terrains(*, world_x_range=(-1.0, 1.0), world_y_range=(-1.0, 1.0),
                       resolution=0.05, height=0.0):
    """Create aligned flat terrains for both raw and batched."""
    nx = int((world_x_range[1] - world_x_range[0]) / resolution) + 1
    ny = int((world_y_range[1] - world_y_range[0]) / resolution) + 1
    heightmap = np.full((ny, nx), height, dtype=np.float64)
    xs = np.linspace(world_x_range[0], world_x_range[1], nx)
    ys = np.linspace(world_y_range[0], world_y_range[1], ny)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    zz = np.full_like(xx, height)
    ray_hits = torch.tensor(
        np.stack([xx, yy, zz], axis=-1).reshape(1, -1, 3), dtype=torch.float64
    )
    batched_terrain = PlannerTerrain.from_ray_hits(
        ray_hits, world_x_range=world_x_range, world_y_range=world_y_range
    )
    return heightmap, batched_terrain, world_x_range, world_y_range


def verify_terrain_height_at_consistency(batched_terrain, heightmap_2d, world_x_range, world_y_range,
                                         atol_interior=1e-6, atol_boundary=1e-4):
    """Assert batched terrain height_at matches source heightmap at grid-interior points.
    Raises AssertionError if parity fails. Used as L1 prerequisite gate."""
    nx, ny = heightmap_2d.shape[1], heightmap_2d.shape[0]
    margin = 2  # skip boundary cells
    xs = np.linspace(world_x_range[0], world_x_range[1], nx)[margin:-margin]
    ys = np.linspace(world_y_range[0], world_y_range[1], ny)[margin:-margin]
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=-1),
                       dtype=torch.float64).unsqueeze(0)
    sampled = batched_terrain.height_at(pts).squeeze(0)
    expected = torch.tensor(heightmap_2d[margin:-margin, margin:-margin].ravel(),
                            dtype=torch.float64)
    torch.testing.assert_close(sampled, expected, atol=atol_interior, rtol=0)


GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
```

- [ ] **Step 2: Create conftest.py with shared fixtures**

```python
# Go2Pvcnn/tests/conftest.py
import sys
from pathlib import Path
import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = TESTS_DIR / "fixtures" / "golden"
GO2_ROOT = TESTS_DIR.parent
RAW_ROOT = GO2_ROOT.parent / "raw" / "kinematic_footsteps"
for p in (str(RAW_ROOT), str(GO2_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.go2fp.config import TrajectoryConfig
from extension.batched_planner.config import BatchedTrajectoryConfig

GOLDEN_ALIGNMENT = {
    "gait_name": "trot",
    "step_freq": 2.0,
    "duty_factor": 0.55,
    "step_height": 0.08,
    "hip_height": 0.30,
    "body_clearance_margin": 0.012,
    "foothold_search_radius": 0.15,
    "foothold_search_step": 0.03,
    "max_foothold_step_down": 0.10,
    "max_touchdown_xy_reach": 0.22,
    "replan_stop_speed": 0.03,
}

@pytest.fixture
def aligned_configs():
    raw_cfg = TrajectoryConfig(**{k: v for k, v in GOLDEN_ALIGNMENT.items()
                                  if k in TrajectoryConfig.__dataclass_fields__})
    batched_cfg = BatchedTrajectoryConfig(**{k: v for k, v in GOLDEN_ALIGNMENT.items()
                                             if k in BatchedTrajectoryConfig.__dataclass_fields__})
    return raw_cfg, batched_cfg

@pytest.fixture
def flat_terrain_pair():
    from fixtures.terrain_adapter import make_flat_terrains
    return make_flat_terrains()

@pytest.fixture
def default_initial_state():
    root_pos = torch.zeros((1, 3), dtype=torch.float64)
    root_pos[0, 2] = 0.30
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    joint_pos = torch.zeros((1, 12), dtype=torch.float64)
    foot_pos = torch.zeros((1, 4, 3), dtype=torch.float64)
    # Default standing foot positions (approximate Go2)
    foot_pos[0, 0] = torch.tensor([0.19, 0.11, 0.0])
    foot_pos[0, 1] = torch.tensor([0.19, -0.11, 0.0])
    foot_pos[0, 2] = torch.tensor([-0.19, 0.11, 0.0])
    foot_pos[0, 3] = torch.tensor([-0.19, -0.11, 0.0])
    from extension.batched_planner.types import BatchedRobotState
    return BatchedRobotState(
        root_pos=root_pos, root_quat=root_quat,
        joint_angles=joint_pos, foot_pos=foot_pos, foot_vel=None,
    )
```

- [ ] **Step 3: Run to verify fixtures import correctly**

Run: `cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn && python -c "import sys; sys.path.insert(0,'.'); from tests.conftest import GOLDEN_ALIGNMENT; print('OK', GOLDEN_ALIGNMENT)"`
Expected: prints OK and the dict

- [ ] **Step 4: Commit**

```bash
git add Go2Pvcnn/tests/conftest.py Go2Pvcnn/tests/fixtures/
git commit -m "feat: add test fixtures infrastructure for cross-validation"
```

---

### Task 2: Write L1 gait cross-validation tests

**Files:**
- Create: `Go2Pvcnn/tests/test_cross_validation_raw.py`

- [ ] **Step 1: Write gait cross-validation test**

```python
# Tests that batched_gait_schedule matches raw gait_schedule output
# for aligned config params. contact_seq compared as float32 exact match.
# touchdown_times and stance_time compared with atol=1e-12.
class TestGaitCrossValidation:
    def test_contact_seq_trot_flat(self, aligned_configs):
        ...
    def test_touchdown_times_match(self, aligned_configs):
        ...
    def test_stance_time_match(self, aligned_configs):
        ...
```

- [ ] **Step 2: Run test to verify it passes (baseline)**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py::TestGaitCrossValidation -v`
Expected: PASS (gait is already vectorized, should match)

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/tests/test_cross_validation_raw.py
git commit -m "test: add L1 gait cross-validation against raw reference"
```

---

### Task 3: Write L1 swing cross-validation tests

**Files:**
- Modify: `Go2Pvcnn/tests/test_cross_validation_raw.py`

- [ ] **Step 1: Write swing cross-validation test**

```python
class TestSwingCrossValidation:
    def test_swing_targets_flat_forward(self, aligned_configs, flat_terrain_pair, default_initial_state):
        # Generate contact_seq with aligned params
        # Run raw swing computation
        # Run batched swing computation (N=1)
        # Compare foot_targets with atol=1e-8
        ...
```

- [ ] **Step 2: Run test to verify baseline**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py::TestSwingCrossValidation -v`
Expected: PASS (confirms current serial batched matches raw)

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/tests/test_cross_validation_raw.py
git commit -m "test: add L1 swing cross-validation against raw reference"
```

---

### Task 4: Write L1 end-to-end trajectory cross-validation

**Files:**
- Modify: `Go2Pvcnn/tests/test_cross_validation_raw.py`

- [ ] **Step 1: Write end-to-end cross-validation tests**

```python
class TestTrajectoryEndToEnd:
    @pytest.mark.parametrize("cmd,label", [
        (torch.tensor([[0.3, 0.0, 0.0]]), "forward"),
        (torch.tensor([[0.0, 0.2, 0.0]]), "lateral"),
        (torch.tensor([[0.0, 0.0, 0.5]]), "turn"),
        (torch.tensor([[0.0, 0.0, 0.0]]), "standstill"),
    ])
    def test_full_trajectory_flat(self, aligned_configs, flat_terrain_pair,
                                   default_initial_state, cmd, label):
        # Run raw generate_trajectory
        # Run batched batched_generate_trajectory (N=1)
        # Compare all output fields: root_pos_w, root_quat_w, joint_angles,
        #   foot_pos_w, contact_state, planned_touchdown_w
        # atol=1e-8, rtol=1e-6
        ...
```

- [ ] **Step 2: Run tests to verify baseline**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py::TestTrajectoryEndToEnd -v`
Expected: PASS (or document specific field differences if any)

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/tests/test_cross_validation_raw.py
git commit -m "test: add L1 end-to-end trajectory cross-validation"
```

---

### Task 4b: Write L1 terrain bridge gate test

**Files:**
- Modify: `Go2Pvcnn/tests/test_cross_validation_raw.py`

- [ ] **Step 1: Write terrain bridge consistency test as L1 prerequisite**

```python
class TestTerrainBridgeGate:
    def test_flat_height_at_parity(self, flat_terrain_pair):
        heightmap, batched_terrain, x_range, y_range = flat_terrain_pair
        from fixtures.terrain_adapter import verify_terrain_height_at_consistency
        verify_terrain_height_at_consistency(batched_terrain, heightmap, x_range, y_range)

    def test_nonflat_height_at_parity(self):
        # Create a sinusoidal heightmap and verify parity
        ...
```

All terrain-dependent L1 tests should `@pytest.mark.depends(on=["TestTerrainBridgeGate"])` or skip if bridge fails.

- [ ] **Step 2: Run test**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py::TestTerrainBridgeGate -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/tests/test_cross_validation_raw.py
git commit -m "test: add L1 terrain bridge gate test"
```

---

### Task 4c: Write L1 foothold, base solver, IK cross-validation

**Files:**
- Modify: `Go2Pvcnn/tests/test_cross_validation_raw.py`

- [ ] **Step 1: Write foothold cross-validation**

```python
class TestFootholdCrossValidation:
    def test_footholds_flat_forward(self, aligned_configs, flat_terrain_pair, default_initial_state):
        # Run raw compute_footholds and batched batched_compute_footholds (N=1)
        # Compare touchdown XYZ with atol=1e-8
        ...
```

- [ ] **Step 2: Write base solver cross-validation**

```python
class TestBaseSolverCrossValidation:
    def test_base_trajectory_flat_forward(self, aligned_configs, flat_terrain_pair, default_initial_state):
        # Run raw and batched base trajectory solvers
        # Compare root_pos_w, root_quat_w with atol=1e-8
        ...
```

- [ ] **Step 3: Write IK cross-validation**

```python
class TestIKCrossValidation:
    def test_ik_matches_raw(self, default_initial_state):
        # Same root_pos, root_quat, foot_targets → compare joint_angles
        # atol=1e-8
        ...
    def test_fk_matches_raw(self):
        # Same root_pos, root_quat, joint_angles → compare body_pos_w
        ...
```

- [ ] **Step 4: Run all new L1 tests**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/test_cross_validation_raw.py
git commit -m "test: add L1 foothold, base solver, IK cross-validation"
```

---

### Task 4d: Add stairs terrain E2E test

**Files:**
- Modify: `Go2Pvcnn/tests/fixtures/terrain_adapter.py`
- Modify: `Go2Pvcnn/tests/test_cross_validation_raw.py`

- [ ] **Step 1: Add stairs heightmap fixture to terrain_adapter**

```python
def make_stairs_terrains(*, n_steps=5, step_height=0.05, step_depth=0.3,
                          world_x_range=(-0.5, 2.0), world_y_range=(-1.0, 1.0),
                          resolution=0.05):
    """Create aligned staircase terrains for both raw and batched."""
    ...
```

- [ ] **Step 2: Add stairs E2E parametrized test**

Add to `TestTrajectoryEndToEnd`:
```python
def test_full_trajectory_stairs_forward(self, aligned_configs, default_initial_state):
    stairs_data = make_stairs_terrains()
    cmd = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
    # Run raw and batched, compare all fields
    ...
```

- [ ] **Step 3: Run test**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py::TestTrajectoryEndToEnd::test_full_trajectory_stairs_forward -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add Go2Pvcnn/tests/fixtures/terrain_adapter.py Go2Pvcnn/tests/test_cross_validation_raw.py
git commit -m "test: add L1 stairs terrain E2E cross-validation"
```

---

### Task 5: Generate and save golden references

**Files:**
- Create: `Go2Pvcnn/tests/fixtures/golden/` directory
- Create: `Go2Pvcnn/tests/fixtures/generate_golden.py` (script to regenerate)

- [ ] **Step 1: Write golden reference generator script**

Script that runs the current serial batched implementation on fixed inputs and saves all intermediate + final tensors to `.pt` files:
- `golden_swing_targets.pt`: swing inputs + outputs for N=1 and N=4
- `golden_trajectory.pt`: full trajectory outputs for 4 command scenarios
- `golden_terrain_segment.pt`: max_height_along_segment inputs + outputs

- [ ] **Step 2: Run generator to create golden files**

Run: `cd /home/lhy/testPvcnnWithIsaacsim && python Go2Pvcnn/tests/fixtures/generate_golden.py`
Expected: `.pt` files created in `Go2Pvcnn/tests/fixtures/golden/`

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/tests/fixtures/generate_golden.py Go2Pvcnn/tests/fixtures/golden/
git commit -m "test: generate golden reference tensors from serial implementation"
```

---

## Phase 1: Swing Vectorization (91% bottleneck)

### Task 6: Write L2 swing vectorization regression tests

**Files:**
- Create: `Go2Pvcnn/tests/test_swing_vectorized.py`

- [ ] **Step 1: Write regression tests loading golden references**

```python
class TestSwingProgressVectorized:
    def test_swing_progress_matches_golden(self):
        golden = torch.load(GOLDEN_DIR / "golden_swing_targets.pt", weights_only=True)
        # Will call new vectorized _batched_swing_progress (to be created)
        # Compare against golden["swing_progress"] with atol=1e-10
        ...

    def test_all_stance_returns_zero_progress(self):
        ...

    def test_all_swing_returns_0_to_1(self):
        ...

    def test_single_frame_swing(self):
        ...

    def test_alternating_stance_swing(self):
        ...

class TestSwingTargetsVectorized:
    def test_targets_match_golden_n1(self):
        ...
    def test_targets_match_golden_n4(self):
        ...
    def test_hermite_continuity_at_transitions(self):
        # z value at swing start/end should be close to lift_off/touchdown z
        ...
```

- [ ] **Step 2: Run tests to verify they fail (code not written yet)**

Run: `pytest Go2Pvcnn/tests/test_swing_vectorized.py -v`
Expected: FAIL (vectorized functions don't exist yet)

- [ ] **Step 3: Commit failing tests**

```bash
git add Go2Pvcnn/tests/test_swing_vectorized.py
git commit -m "test(red): add L2 swing vectorization regression tests"
```

---

### Task 7: Vectorize `_leg_swing_progress_and_stance_anchor`

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/swing.py`

- [ ] **Step 1: Implement `_batched_swing_progress` as new vectorized function**

Replace the per-frame Python loops with:
1. `torch.diff` for edge detection on `(N, T, 4)` stance tensor
2. `torch.cumsum` for swing run ID assignment
3. `scatter_reduce` / `scatter_add_` for run start frames and lengths
4. Vectorized progress = `idx_in_run / (run_length - 1).clamp(min=1)`

Keep the old `_leg_swing_progress_and_stance_anchor` temporarily (renamed to `_leg_swing_progress_serial`) for golden reference comparison.

New function signature:
```python
def _batched_swing_progress(stance_bool: Tensor) -> tuple[Tensor, Tensor]:
    """Vectorized swing progress for shape (N, T, 4).
    Returns (swing_progress, use_touchdown) both (N, T, 4)."""
```

- [ ] **Step 2: Run L2 swing progress tests**

Run: `pytest Go2Pvcnn/tests/test_swing_vectorized.py::TestSwingProgressVectorized -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/swing.py
git commit -m "feat: vectorize swing progress computation (eliminate for loops)"
```

---

### Task 8: Vectorize `_swing_phase_targets` (branchless)

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/swing.py`

- [ ] **Step 1: Rewrite `_swing_phase_targets` to use `torch.where` instead of `if torch.any` branches**

The current implementation has:
```python
if torch.any(mask_first):
    tau = swing_progress[mask_first] / 0.5
    ...
if torch.any(mask_second):
    ...
```

Rewrite as branchless `torch.where` operating on full `(N, T, 4)`:
```python
mask_first = swing_progress <= 0.5
tau_first = swing_progress / 0.5
tau_second = (swing_progress - 0.5) / 0.5
z_first = _hermite_cubic(tau_first, p0_first, p1_first, v0_first, v1_first)[..., 2]
z_second = _hermite_cubic(tau_second, p0_second, p1_second, v0_second, v1_second)[..., 2]
z = torch.where(mask_first, z_first, z_second)
```

- [ ] **Step 2: Run swing target tests**

Run: `pytest Go2Pvcnn/tests/test_swing_vectorized.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/swing.py
git commit -m "feat: branchless swing phase targets using torch.where"
```

---

### Task 9: Vectorize `batched_compute_swing_targets` (eliminate outer loops)

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/swing.py`

- [ ] **Step 1: Replace `for batch_idx × for leg` with full tensor ops**

New implementation operates on `(N, T, 4, 3)` directly:
1. `_compute_swing_apex` on `(N, 4)` lift_off/touchdown
2. `_batched_swing_progress` on `(N, T, 4)` stance bool
3. `_swing_phase_targets` on `(N, T, 4)` progress
4. `torch.where(stance, anchor, arc)` on `(N, T, 4, 3)`

Remove the old serial `_leg_swing_progress_serial` after confirming all tests pass.

- [ ] **Step 2: Run all swing tests + L1 cross-validation**

Run: `pytest Go2Pvcnn/tests/test_swing_vectorized.py Go2Pvcnn/tests/test_cross_validation_raw.py::TestSwingCrossValidation -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/swing.py
git commit -m "feat: fully vectorize batched_compute_swing_targets (N,T,4,3)"
```

---

### Task 10: Verify swing vectorization with full L1 end-to-end

**Files:** None (test-only)

- [ ] **Step 1: Run full L1 cross-validation**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py -v`
Expected: ALL PASS

- [ ] **Step 2: Run existing test suite to check nothing broke**

Run: `pytest Go2Pvcnn/tests/ -v --ignore=Go2Pvcnn/tests/benchmarks`
Expected: ALL PASS

- [ ] **Step 3: Commit any test adjustments needed**

---

## Phase 2: Terrain + Foothold Vectorization

### Task 11: Write L2 terrain vectorization tests

**Files:**
- Create: `Go2Pvcnn/tests/test_terrain_vectorized.py`

- [ ] **Step 1: Write tests for vectorized `max_height_along_segment`**

```python
class TestMaxHeightSegmentVectorized:
    def test_matches_golden(self):
        golden = torch.load(".../golden_terrain_segment.pt")
        # Compare new batch implementation against golden
        ...
    def test_varying_segment_lengths(self):
        # Different envs have different segment lengths
        ...
    def test_single_env(self):
        ...
    def test_batch_4legs_merged(self):
        # Test the merged 4-leg call from trajectory.py
        ...
```

- [ ] **Step 2: Run to verify tests fail**

Run: `pytest Go2Pvcnn/tests/test_terrain_vectorized.py -v`
Expected: FAIL

- [ ] **Step 3: Commit failing tests**

```bash
git add Go2Pvcnn/tests/test_terrain_vectorized.py
git commit -m "test(red): add L2 terrain vectorization regression tests"
```

---

### Task 12: Vectorize `max_height_along_segment`

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/terrain.py`

- [ ] **Step 1: Implement batch `max_height_along_segment`**

Replace `for idx in range(self.batch_size)` + `.item()` with:
1. Keep endpoints as tensor `(N, 2)`, no `.item()`
2. `t = torch.linspace(0, 1, n_samples).view(1, -1, 1)` → lerp → `(N, n_samples, 2)`
3. Normalize to grid_sample coords, one `F.grid_sample` call
4. Use `align_corners=True, mode='bilinear', padding_mode='border'` (match existing `_sample_map`)
5. `torch.amax(sampled, dim=1)` → `(N,)`

Use fixed `n_samples=32` upper bound; mask out-of-range samples.

- [ ] **Step 2: Run terrain tests + L1**

Run: `pytest Go2Pvcnn/tests/test_terrain_vectorized.py Go2Pvcnn/tests/test_cross_validation_raw.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/terrain.py
git commit -m "feat: vectorize max_height_along_segment (eliminate per-env loop)"
```

---

### Task 13: Merge 4× per-leg terrain calls in trajectory.py

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/trajectory.py`

- [ ] **Step 1: Replace 4× loop with single batched call**

Current:
```python
terrain_max_heights = torch.stack(
    [terrain.max_height_along_segment(states.foot_pos[:, leg_idx, :2], touchdowns[:, leg_idx, :2])
     for leg_idx in range(4)],
    dim=1,
)
```

New: stack all 4 legs' endpoints into `(N*4, 2)`, call once, reshape to `(N, 4)`.

- [ ] **Step 2: Run full L1 + L2 tests**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py Go2Pvcnn/tests/test_terrain_vectorized.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/trajectory.py
git commit -m "feat: merge 4× per-leg terrain segment calls into single batch"
```

---

### Task 13b: Write L2 foothold vectorization tests (TDD red)

**Files:**
- Create: `Go2Pvcnn/tests/test_foothold_vectorized.py`

- [ ] **Step 1: Write foothold regression tests**

```python
from conftest import GOLDEN_DIR

class TestSpiralOffsetsMeshgrid:
    def test_meshgrid_covers_all_loop_offsets(self):
        # Generate spiral with old loop, meshgrid with new
        # Assert new superset contains all old points
        ...

class TestEvaluateTouchdownsNoItem:
    def test_feasibility_matches_golden(self):
        golden = torch.load(GOLDEN_DIR / "golden_trajectory.pt", weights_only=True)
        # Compare feasibility result (bool tensor, no .item())
        ...

class TestFootholdDynamicN:
    def test_footholds_n1(self):
        ...
    def test_footholds_n64(self):
        ...
    def test_footholds_n2048(self):
        # Verify batched_compute_footholds works at N=2048
        ...
```

- [ ] **Step 2: Run tests to verify they define the contract**

Run: `pytest Go2Pvcnn/tests/test_foothold_vectorized.py -v`
Expected: PASS for existing code, sets baseline for regression

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/tests/test_foothold_vectorized.py
git commit -m "test(red): add L2 foothold vectorization regression tests"
```

---

### Task 13c: Add L2 dynamic-N tests for swing and terrain

**Files:**
- Modify: `Go2Pvcnn/tests/test_swing_vectorized.py`
- Modify: `Go2Pvcnn/tests/test_terrain_vectorized.py`

- [ ] **Step 1: Add N=1, N=32, N=2048 tests for swing**

```python
class TestSwingDynamicN:
    @pytest.mark.parametrize("n_envs", [1, 32, 2048])
    def test_swing_targets_arbitrary_n(self, n_envs):
        # Create synthetic (n_envs, T, 4) contact_seq
        # Run vectorized batched_compute_swing_targets
        # Verify output shape (n_envs, T, 4, 3) and no NaN/Inf
        ...
```

- [ ] **Step 2: Add N=1, N=32, N=2048 tests for terrain**

```python
class TestTerrainDynamicN:
    @pytest.mark.parametrize("n_envs", [1, 32, 2048])
    def test_max_height_segment_arbitrary_n(self, n_envs):
        ...
```

- [ ] **Step 3: Run tests**

Run: `pytest Go2Pvcnn/tests/test_swing_vectorized.py::TestSwingDynamicN Go2Pvcnn/tests/test_terrain_vectorized.py::TestTerrainDynamicN -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add Go2Pvcnn/tests/test_swing_vectorized.py Go2Pvcnn/tests/test_terrain_vectorized.py
git commit -m "test: add L2 dynamic-N tests for swing and terrain (N=1,32,2048)"
```

---

### Task 14: Vectorize foothold spiral offsets + evaluate

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/foothold.py`

- [ ] **Step 1: Replace `_precompute_spiral_offsets` with meshgrid**

Replace nested Python loops with `torch.meshgrid` for `[-R, R]` grid, sort by distance.

- [ ] **Step 2: Make `batched_evaluate_touchdowns` reasons lazy**

Move `.item()` calls inside `if verbose:` guard or use integer reason code tensor.

- [ ] **Step 3: Run L2 foothold tests + L1 cross-validation**

Run: `pytest Go2Pvcnn/tests/test_foothold_vectorized.py Go2Pvcnn/tests/test_cross_validation_raw.py::TestFootholdCrossValidation -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/foothold.py
git commit -m "feat: vectorize foothold spiral offsets + lazy evaluate reasons"
```

---

## Phase 3: Performance Benchmark & Evaluate

### Task 15: Create L3 performance benchmark

**Files:**
- Create: `Go2Pvcnn/tests/benchmarks/__init__.py`
- Create: `Go2Pvcnn/tests/benchmarks/bench_planner_scaling.py`

- [ ] **Step 1: Write benchmark script**

Reuse `_SyntheticEnv`/`_BenchRow` from `Go2Pvcnn/scripts/bench_batched_planner.py`. Add:
- Burst replan: N=[1, 64, 256, 1024, 2048], `replan_interval=1`
- Steady-state replan: N_total=2048, ~5-10% env per step, `replan_interval=10`
- Per-stage timing JSONL output
- `@pytest.mark.skipif(not torch.cuda.is_available())` guard
- **Performance threshold assertions** (initial values, to be refined after first bench run):

```python
PERF_THRESHOLDS_MS = {
    2048: {"plan": 50.0, "swing_targets": 15.0},
    1024: {"plan": 30.0},
    1: {"plan": 10.0},
}

def test_burst_replan_within_budget(self, num_envs, median_ms, stage_ms):
    thresholds = PERF_THRESHOLDS_MS.get(num_envs, {})
    for stage, limit in thresholds.items():
        assert stage_ms[stage] < limit, f"N={num_envs} {stage}={stage_ms[stage]:.1f}ms > {limit}ms"
```

- [ ] **Step 2: Run benchmark on CUDA**

Run: `pytest Go2Pvcnn/tests/benchmarks/bench_planner_scaling.py -v -s`
Expected: outputs timing table; record baseline numbers

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/tests/benchmarks/
git commit -m "test: add L3 performance scaling benchmark (burst + steady-state)"
```

---

### Task 16: Evaluate base_solver / terrain_estimator EMA

**Files:** None (benchmark-only evaluation)

- [ ] **Step 1: Check bench results for base_solve and terrain_est stages**

If `base_solve + terrain_est < 5ms` at N=2048: skip EMA optimization.
If > 5ms: create tasks for parallel scan rewrite (out of scope for this plan, would be a follow-up).

- [ ] **Step 2: Document decision in spec or notes**

Create `notes/ai/ai-14-batched-planner-bench-results.md` with the bench output and EMA decision.

- [ ] **Step 3: Commit benchmark results and decision doc**

```bash
git add notes/ai/ai-14-batched-planner-bench-results.md
git commit -m "bench: record Phase 3 results, EMA optimization decision"
```

---

## Phase 4: Triton Kernels (if needed)

### Task 17: Evaluate need for Triton based on bench results

- [ ] **Step 1: Check N=2048 burst replan timing**

If `swing_targets < 15ms` and total `plan < 50ms`: skip Triton.
If not: proceed with Task 18.

- [ ] **Step 2: Document decision**

---

### Task 18: Write fused Triton swing kernel (conditional)

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/triton_swing.py`

- [ ] **Step 1: Write Triton kernel**

Fused kernel: one block per `(env, leg)`, processes all T frames.
Grid: `(N * 4,)` — dynamic from input N.
Operations fused: swing progress + Hermite interpolation + stance/swing merge.

- [ ] **Step 2: Write regression test comparing Triton vs PyTorch**

File: `Go2Pvcnn/tests/test_triton_swing.py`

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonSwingParity:
    def test_triton_matches_pytorch_n1(self):
        ...
    def test_triton_matches_pytorch_n2048(self):
        ...
```

- [ ] **Step 3: Benchmark Triton vs PyTorch at N=2048**

- [ ] **Step 4: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/triton_swing.py Go2Pvcnn/tests/test_triton_swing.py
git commit -m "feat: fused Triton swing kernel for N=2048"
```

---

## Phase 5: Visualization — Pure Kinematic Playback

### Task 19: Write L4 visualization playback tests

**Files:**
- Create: `Go2Pvcnn/tests/test_viz_playback.py`

- [ ] **Step 1: Write playback logic tests (no Isaac rendering)**

```python
class TestKinematicPlayback:
    def test_state_chain_last_frame_to_next_input(self):
        # Verify: trajectory last frame state == next planning initial state
        ...
    def test_touchdown_visible_at_frame_zero(self):
        # Verify: touchdown markers set before first playback frame
        ...
    def test_command_update_triggers_replan(self):
        # Verify: changing teleop command → next generate_trajectory uses new cmd
        ...
    def test_playback_frame_counter_wraps(self):
        # Verify: after playing all T frames, counter resets and replans
        ...
```

- [ ] **Step 2: Run tests (expect fail — logic not implemented)**

Run: `pytest Go2Pvcnn/tests/test_viz_playback.py -v`
Expected: FAIL

- [ ] **Step 3: Commit failing tests**

```bash
git add Go2Pvcnn/tests/test_viz_playback.py
git commit -m "test(red): add L4 kinematic playback logic tests"
```

---

### Task 20: Rewrite viz to pure kinematic playback

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`

- [ ] **Step 1: Rewrite `main()` loop**

Replace:
```python
while simulation_app.is_running():
    env.step(zero_actions)
    result = rewards_reference.ensure_reference_cache(base_env)
    ...
```

With plan-once/replay-then-replan:
```python
while simulation_app.is_running():
    teleop_cmd = teleop.poll()
    if need_replan(teleop_cmd, playback_frame, result):
        state = build_state_from_last_frame(result, playback_frame)
        terrain = compute_terrain(scanner)
        result = batched_generate_trajectory(terrain, state, cmd, ...)
        update_touchdown_markers(result)  # visible immediately
        playback_frame = 0

    apply_kinematic_frame(robot, result, playback_frame)
    update_foot_trajectory_markers(result)
    update_camera(...)
    playback_frame += 1
```

No `env.step`. Robot pose set via `write_root_pose_to_sim` + `write_joint_state_to_sim`.

- [ ] **Step 2: Run L4 tests**

Run: `pytest Go2Pvcnn/tests/test_viz_playback.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py
git commit -m "feat: pure kinematic playback visualization (no physics step)"
```

---

## Phase 6: Manager Tests + Final Validation

### Task 21: Extend manager tests for selective replan + dynamic N

**Files:**
- Modify: `Go2Pvcnn/tests/test_batched_manager.py`

- [ ] **Step 1: Add selective replan tests**

```python
class TestSelectiveReplan:
    def test_subset_replan_preserves_non_replanned_cache(self):
        ...
    def test_full_replan_on_first_call(self):
        ...
    def test_command_change_triggers_per_env_replan(self):
        ...
    def test_horizon_change_triggers_full_replan(self):
        ...
    def test_mixed_reset_and_interval_replan(self):
        ...
```

- [ ] **Step 2: Run extended manager tests**

Run: `pytest Go2Pvcnn/tests/test_batched_manager.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/tests/test_batched_manager.py
git commit -m "test: extend manager tests for selective replan + dynamic N"
```

---

### Task 22: Final full validation

- [ ] **Step 1: Run ALL tests**

Run: `pytest Go2Pvcnn/tests/ -v --ignore=Go2Pvcnn/tests/benchmarks`
Expected: ALL PASS

- [ ] **Step 2: Run L3 benchmark on CUDA**

Run: `pytest Go2Pvcnn/tests/benchmarks/ -v -s`
Expected: N=2048 burst plan < 50ms

- [ ] **Step 3: Run L1 cross-validation to confirm raw parity maintained**

Run: `pytest Go2Pvcnn/tests/test_cross_validation_raw.py -v`
Expected: ALL PASS

- [ ] **Step 4: Tag milestone (no empty commit)**

If all tests pass and bench targets are met, this phase is complete. No additional commit needed unless there were test adjustments in Steps 1-3. If adjustments were made:

```bash
git add -u
git commit -m "milestone: Phase 6 complete - all L1-L4 tests pass, bench targets met"
```
