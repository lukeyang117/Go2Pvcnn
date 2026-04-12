# Batched GPU Kinematic Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `raw/kinematic_footsteps` trajectory generation as a batched PyTorch GPU pipeline under `Go2Pvcnn/extension/batched_planner/`, eliminating CPU process pool bottleneck.

**Architecture:** Each raw `go2fp` module is faithfully ported to PyTorch with an added batch dimension `(N, ...)`. All terrain queries use `F.grid_sample` on GPU heightmaps. Fixed-K candidate expansion uses `(N*K, ...)` reshaping. Tests verify N=1 output matches raw NumPy exactly.

**Tech Stack:** PyTorch (CUDA), Isaac Lab (RayCaster), unittest/pytest, raw `scripts.go2fp` as reference oracle.

**Spec:** `docs/specs/2026-04-12-batched-gpu-kinematic-planner-design.md`

**Conda envs:** `mujoco_env` for tests involving raw go2fp imports; `env_isaaclab` for Isaac Lab integration tests.

**PYTHONPATH:** Tests import `extension.*` and `go2_pvcnn.*`. All test commands assume `cwd = Go2Pvcnn/` which must be on `PYTHONPATH`. If not already configured by `pyproject.toml` or `conftest.py`, prepend: `PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:$PYTHONPATH` before pytest commands.

---

## File Structure

### New files (create)

| File | Responsibility |
|------|---------------|
| `Go2Pvcnn/extension/convention.py` | quat wxyz↔xyzw, Isaac↔planner state conversion |
| `Go2Pvcnn/extension/batched_planner/__init__.py` | Package init |
| `Go2Pvcnn/extension/batched_planner/config.py` | `BatchedTrajectoryConfig` dataclass |
| `Go2Pvcnn/extension/batched_planner/types.py` | `BatchedRobotState`, `BatchedTrajectoryResult`, constants |
| `Go2Pvcnn/extension/batched_planner/terrain.py` | `BatchedTerrain` with `F.grid_sample` queries |
| `Go2Pvcnn/extension/batched_planner/gait.py` | Batched gait schedule, touchdown times, swing events |
| `Go2Pvcnn/extension/batched_planner/foothold.py` | Batched spiral search, Raibert, candidate evaluation |
| `Go2Pvcnn/extension/batched_planner/swing.py` | Batched swing targets |
| `Go2Pvcnn/extension/batched_planner/terrain_estimator.py` | Batched terrain roll/pitch/height EMA |
| `Go2Pvcnn/extension/batched_planner/base_solver.py` | Batched base trajectory, body clearance |
| `Go2Pvcnn/extension/batched_planner/ik.py` | Batched IK/FK |
| `Go2Pvcnn/extension/batched_planner/trajectory.py` | `batched_generate_trajectory` main entry |
| `Go2Pvcnn/extension/batched_planner/manager.py` | `BatchedTrajectoryManager` for Isaac Lab integration |
| `Go2Pvcnn/extension/viz/compare_trajectories.py` | Numerical comparison tool (`--no-gui`) |
| `Go2Pvcnn/tests/test_batched_convention.py` | Convention tests |
| `Go2Pvcnn/tests/test_batched_terrain.py` | Terrain tests |
| `Go2Pvcnn/tests/test_batched_gait.py` | Gait tests |
| `Go2Pvcnn/tests/test_batched_foothold.py` | Foothold tests |
| `Go2Pvcnn/tests/test_batched_swing.py` | Swing tests |
| `Go2Pvcnn/tests/test_batched_terrain_estimator.py` | Terrain estimator tests |
| `Go2Pvcnn/tests/test_batched_base_solver.py` | Base solver tests |
| `Go2Pvcnn/tests/test_batched_ik.py` | IK/FK tests |
| `Go2Pvcnn/tests/test_batched_trajectory.py` | End-to-end trajectory tests |
| `Go2Pvcnn/tests/test_batched_trajectory_batch.py` | N=32 batch consistency tests |

### Modified files

| File | Change |
|------|--------|
| `Go2Pvcnn/extension/__init__.py` | Update imports |
| `Go2Pvcnn/extension/mdp/rewards_reference.py` | Adapt cache field access for new `BatchedTrajectoryResult` |
| `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py` | New file (move from extension/tasks/) |
| `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py` | Update import path |

### Deleted directories (after all tasks complete)

| Path | Reason |
|------|--------|
| `Go2Pvcnn/extension/planner/` | Replaced by `batched_planner/` |
| `Go2Pvcnn/extension/tasks/` | Moved to `go2_pvcnn/tasks/` |
| `Go2Pvcnn/extension/mdp/reference_trajectory_events.py` | Replaced by `manager.py` |

---

## Task 1: Convention & Types Foundation

**Files:**
- Create: `Go2Pvcnn/extension/convention.py`
- Create: `Go2Pvcnn/extension/batched_planner/__init__.py`
- Create: `Go2Pvcnn/extension/batched_planner/types.py`
- Create: `Go2Pvcnn/extension/batched_planner/config.py`
- Test: `Go2Pvcnn/tests/test_batched_convention.py`

- [ ] **Step 1: Write the failing test for convention.py**

```python
# Go2Pvcnn/tests/test_batched_convention.py
import unittest
import torch
import numpy as np

class TestConvention(unittest.TestCase):
    def test_quat_wxyz_to_xyzw_roundtrip(self):
        from extension.convention import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
        q_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0]])
        q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
        self.assertEqual(q_xyzw.shape, (2, 4))
        self.assertAlmostEqual(q_xyzw[0, 3].item(), 1.0, places=5)  # w last
        q_back = quat_xyzw_to_wxyz(q_xyzw)
        torch.testing.assert_close(q_back, q_wxyz)

    def test_quat_batch_dims(self):
        from extension.convention import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
        q = torch.randn(4, 8, 4)
        q_converted = quat_wxyz_to_xyzw(q)
        self.assertEqual(q_converted.shape, (4, 8, 4))
        q_back = quat_xyzw_to_wxyz(q_converted)
        torch.testing.assert_close(q_back, q)

    def test_extract_yaw_matches_raw(self):
        from extension.convention import extract_yaw_batch
        import sys, os
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_root = os.path.join(repo, "raw", "kinematic_footsteps")
        if raw_root not in sys.path:
            sys.path.insert(0, raw_root)
        from scripts.go2fp.trajectory import _extract_yaw
        q_np = np.array([0.924, 0.0, 0.383, 0.0], dtype=np.float64)  # ~44 deg yaw
        raw_yaw = _extract_yaw(q_np)
        q_torch = torch.tensor(q_np, dtype=torch.float64).unsqueeze(0)
        batched_yaw = extract_yaw_batch(q_torch)
        self.assertAlmostEqual(batched_yaw[0].item(), raw_yaw, places=10)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn && conda run -n mujoco_env python -m pytest tests/test_batched_convention.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement convention.py, types.py, config.py**

`Go2Pvcnn/extension/convention.py`:
```python
"""MuJoCo ↔ Isaac Lab convention alignment (quat ordering, coordinate frames)."""
from __future__ import annotations
import torch
from torch import Tensor

def quat_wxyz_to_xyzw(q: Tensor) -> Tensor:
    """(..., 4) wxyz → xyzw."""
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)

def quat_xyzw_to_wxyz(q: Tensor) -> Tensor:
    """(..., 4) xyzw → wxyz."""
    return torch.cat([q[..., 3:], q[..., :3]], dim=-1)

def extract_yaw_batch(quat_wxyz: Tensor) -> Tensor:
    """(..., 4) wxyz quaternion → (...,) yaw angle."""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def extract_roll_pitch_batch(quat_wxyz: Tensor) -> tuple[Tensor, Tensor]:
    """(..., 4) wxyz → roll (...,), pitch (...,)."""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    return roll, pitch

def isaac_state_to_planner_state(
    root_pos: Tensor, root_quat_xyzw: Tensor, joint_pos: Tensor, foot_pos: Tensor,
) -> "BatchedRobotState":
    """Convert Isaac Lab state tensors to planner's BatchedRobotState (wxyz quat)."""
    from extension.batched_planner.types import BatchedRobotState
    return BatchedRobotState(
        root_pos=root_pos,
        root_quat=quat_xyzw_to_wxyz(root_quat_xyzw),
        joint_angles=joint_pos,
        foot_pos=foot_pos,
    )

def planner_result_to_reference_cache(result: "BatchedTrajectoryResult") -> dict:
    """Convert BatchedTrajectoryResult to the dict format consumed by rewards_reference.py."""
    return result.gather_at_phase(
        torch.zeros(result.root_pos_w.shape[0], dtype=torch.long, device=result.root_pos_w.device)
    )

def euler_to_quat_batch(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    """(...,) roll, pitch, yaw → (..., 4) wxyz quaternion (ZYX convention)."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)

def yaw_rotation_matrix_batch(yaw: Tensor) -> Tensor:
    """(...,) yaw → (..., 3, 3) rotation matrix."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    row0 = torch.stack([c, -s, z], dim=-1)
    row1 = torch.stack([s, c, z], dim=-1)
    row2 = torch.stack([z, z, o], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
```

`Go2Pvcnn/extension/batched_planner/__init__.py`: empty.

`Go2Pvcnn/extension/batched_planner/types.py`:
```python
"""Batched data types mirroring raw/kinematic_footsteps/scripts/go2fp/types.py."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

GRAVITY = 9.81
HIP_HEIGHT = 0.30

# Leg order: FL=0, FR=1, RL=2, RR=3 (matches raw LEG_ORDER)
LEG_ORDER = ["FL", "FR", "RL", "RR"]
LEG_SIDE_SIGN = {"FL": -1.0, "FR": 1.0, "RL": -1.0, "RR": 1.0}

# Hip offsets in body frame [4, 3] — from raw types.py
HIP_OFFSETS_ARRAY = torch.tensor([
    [ 0.1934,  0.0465, 0.0],  # FL
    [ 0.1934, -0.0465, 0.0],  # FR
    [-0.1934,  0.0465, 0.0],  # RL
    [-0.1934, -0.0465, 0.0],  # RR
], dtype=torch.float64)

# Link lengths from raw types.py
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213

# Joint limits [12, 2] (min, max) from raw types.py
JOINT_LIMITS_MIN = torch.tensor([
    -1.047, -1.5, -2.721,  # FL
    -1.047, -1.5, -2.721,  # FR
    -1.047, -0.5, -2.721,  # RL
    -1.047, -0.5, -2.721,  # RR
], dtype=torch.float64)
JOINT_LIMITS_MAX = torch.tensor([
    1.047, 3.9, -0.611,  # FL
    1.047, 3.9, -0.611,  # FR
    1.047, 4.5, -0.611,  # RL
    1.047, 4.5, -0.611,  # RR
], dtype=torch.float64)

@dataclass
class BatchedRobotState:
    root_pos: Tensor      # (N, 3)
    root_quat: Tensor     # (N, 4) wxyz
    joint_angles: Tensor  # (N, 12)
    foot_pos: Tensor      # (N, 4, 3)

@dataclass
class BatchedTrajectoryResult:
    num_frames: int
    root_pos_w: Tensor          # (N, T, 3)
    root_quat_w: Tensor         # (N, T, 4) wxyz
    root_lin_vel_w: Tensor      # (N, T, 3)
    root_ang_vel_w: Tensor      # (N, T, 3)
    joint_angles: Tensor        # (N, T, 12)
    foot_pos_w: Tensor          # (N, T, 4, 3)
    foot_pos_root: Tensor       # (N, T, 4, 3)
    contact_state: Tensor       # (N, T, 4)
    body_pos_root: Tensor       # (N, T, 12, 3)
    planned_touchdown_w: Tensor # (N, 4, 3)

    def gather_at_phase(self, phase_idx: Tensor) -> dict[str, Tensor]:
        """phase_idx: (N,) long → dict of per-env reference tensors at that phase."""
        idx = phase_idx.clamp(max=self.num_frames - 1)
        N = idx.shape[0]
        bi = torch.arange(N, device=idx.device)
        return {
            "root_pos_w": self.root_pos_w[bi, idx],
            "root_quat_w": self.root_quat_w[bi, idx],
            "joint_angles": self.joint_angles[bi, idx],
            "foot_pos_root": self.foot_pos_root[bi, idx],
            "contact_state": self.contact_state[bi, idx],
            "planned_touchdown_w": self.planned_touchdown_w,
        }
```

`Go2Pvcnn/extension/batched_planner/config.py`:
```python
"""Batched trajectory config mirroring raw TrajectoryConfig."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class BatchedTrajectoryConfig:
    gait_name: str = "trot"
    step_freq: float = 2.0
    duty_factor: float = 0.6
    step_height: float = 0.08
    hip_height: float = 0.30
    body_clearance_margin: float = 0.012
    foothold_search_radius: float = 0.15
    foothold_search_step: float = 0.03
    max_foothold_step_down: float = float("inf")
    max_roughness: float = 0.5  # teacher override; raw default is 1.0
    max_touchdown_xy_reach: float = 0.15
    replan_stop_speed: float = 0.05
    replan_velocity_scales: list[float] = field(default_factory=lambda: [1.0, 0.8, 0.6])
    replan_yaw_biases: list[float] = field(default_factory=lambda: [0.0, 0.15, -0.15])
    replan_vy_biases: list[float] = field(default_factory=lambda: [0.0, 0.05, -0.05])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn && conda run -n mujoco_env python -m pytest tests/test_batched_convention.py -v`
Expected: 3 passed

- [ ] **Step 5: Verify types.py constants match raw and test convention bridges**

Add tests for `HIP_OFFSETS_ARRAY`, `isaac_state_to_planner_state`, `planner_result_to_reference_cache`:
```python
def test_hip_offsets_match_raw(self):
    from extension.batched_planner.types import HIP_OFFSETS_ARRAY
    import sys, os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_root = os.path.join(repo, "raw", "kinematic_footsteps")
    if raw_root not in sys.path:
        sys.path.insert(0, raw_root)
    from scripts.go2fp.types import HIP_OFFSETS_ARRAY as RAW_HIP_OFFSETS
    import numpy as np
    np.testing.assert_allclose(HIP_OFFSETS_ARRAY.numpy(), RAW_HIP_OFFSETS, atol=1e-6)

def test_isaac_state_to_planner_state(self):
    from extension.convention import isaac_state_to_planner_state
    pos = torch.zeros(2, 3)
    quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    joints = torch.zeros(2, 12)
    feet = torch.zeros(2, 4, 3)
    state = isaac_state_to_planner_state(pos, quat_xyzw, joints, feet)
    self.assertAlmostEqual(state.root_quat[0, 0].item(), 1.0, places=5)  # w first in wxyz
```

Run: same pytest command
Expected: 5+ passed

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/convention.py Go2Pvcnn/extension/batched_planner/ Go2Pvcnn/tests/test_batched_convention.py
git commit -m "feat: add convention, types, and config for batched planner"
```

---

## Task 2: Batched Terrain

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/terrain.py`
- Test: `Go2Pvcnn/tests/test_batched_terrain.py`

Reference: `raw/kinematic_footsteps/scripts/go2fp/terrain.py` — `_sample_metric_grid`, `_metric_slope_magnitude`, `_metric_max_height_along_segment`

- [ ] **Step 1: Write failing test**

Test constructs a known heightmap (e.g. a tilted plane `z = 0.1*x + 0.05*y`), queries `height_at`, `roughness_at`, `max_height_along_segment` on both raw `GlobalElevationTerrain` and `BatchedTerrain`, and asserts allclose.

Key test cases:
- `test_height_at_single_point_matches_raw`: N=1, M=1 point
- `test_height_at_multiple_points_matches_raw`: N=1, M=10 random points
- `test_roughness_at_matches_raw`: same grid, compare slope values
- `test_max_height_along_segment_matches_raw`: known segment across the grid
- `test_batch_consistency`: N=4 identical grids, verify all give same result

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn && conda run -n mujoco_env python -m pytest tests/test_batched_terrain.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BatchedTerrain**

Key implementation details:
- `__init__`: stores `(N, 1, H, W)` heightmap, precomputes roughness map via 4-neighbor central differences using `F.conv2d` with a hand-crafted kernel
- `height_at`: world coords → normalized grid coords → `F.grid_sample(mode='bilinear', align_corners=True)`
- `roughness_at`: same sampling but on precomputed roughness map
- `max_height_along_segment`: adaptive `sample_count = max(3, ceil(dist/step)*4 + 1)`, batched linspace, then `max(height_at(...))`
- `from_ray_hits`: reshape ray hits to `(N, H, W, 3)`, extract z component

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/terrain.py Go2Pvcnn/tests/test_batched_terrain.py
git commit -m "feat: add BatchedTerrain with F.grid_sample terrain queries"
```

---

## Task 3: Batched Gait

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/gait.py`
- Test: `Go2Pvcnn/tests/test_batched_gait.py`

Reference: `raw/kinematic_footsteps/scripts/go2fp/gait.py`

- [ ] **Step 1: Write failing test**

Test cases:
- `test_gait_schedule_matches_raw`: N=1, compare `batched_gait_schedule` vs raw `gait_schedule` for trot gait
- `test_next_touchdown_times_matches_raw`: N=1 vs raw
- `test_stance_time_matches_raw`: scalar comparison
- `test_legs_requiring_touchdown_matches_raw`: known contact sequence
- `test_detect_swing_events_matches_raw`: known contact sequence

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

Functions to implement: `batched_gait_schedule`, `batched_next_touchdown_times`, `batched_stance_time`, `batched_legs_requiring_touchdown`, `batched_detect_swing_events`. All follow raw logic with added `(N, ...)` dimension.

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/gait.py Go2Pvcnn/tests/test_batched_gait.py
git commit -m "feat: add batched gait schedule and swing event detection"
```

---

## Task 4: Batched IK/FK

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/ik.py`
- Test: `Go2Pvcnn/tests/test_batched_ik.py`

Reference: `raw/kinematic_footsteps/scripts/go2fp/ik.py`

- [ ] **Step 1: Write failing test**

Test cases:
- `test_inverse_kinematics_matches_raw`: N=1, T=1, known root+foot → joint angles
- `test_forward_kinematics_matches_raw`: N=1, T=1, known root+joints → body positions
- `test_ik_fk_roundtrip`: IK then FK should recover foot positions
- `test_body_pos_root_relative_matches_raw`

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

Port `_solve_leg_ik_batch` and `_forward_kinematics_leg_batch` from raw, replacing per-leg Python loop with tensor ops over the leg dimension using `HIP_OFFSETS_ARRAY` broadcasting.

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/ik.py Go2Pvcnn/tests/test_batched_ik.py
git commit -m "feat: add batched inverse/forward kinematics"
```

---

## Task 5: Batched Terrain Estimator

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/terrain_estimator.py`
- Test: `Go2Pvcnn/tests/test_batched_terrain_estimator.py`

Reference: `raw/kinematic_footsteps/scripts/go2fp/terrain_estimator.py`

- [ ] **Step 1: Write failing test**

Test: construct foot/base trajectories (N=1, T=25), run raw `estimate_terrain_batch` and batched version, compare roll/pitch/height per frame.

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

Vectorized raw value computation (einsum for yaw rotation), then time-serial EMA loop with `initial_roll`, `initial_pitch`, `initial_height`. Env dimension `(N,)` is parallel within each time step.

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/terrain_estimator.py Go2Pvcnn/tests/test_batched_terrain_estimator.py
git commit -m "feat: add batched terrain estimator with EMA filtering"
```

---

## Task 6: Batched Base Solver

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/base_solver.py`
- Test: `Go2Pvcnn/tests/test_batched_base_solver.py`

Reference: `raw/kinematic_footsteps/scripts/go2fp/base_solver.py`

- [ ] **Step 1: Write failing test**

Test cases:
- `test_integrate_base_planar_matches_raw`: N=1, known vx/vy/yaw_rate
- `test_solve_base_trajectory_matches_raw`: N=1, full solve with known terrain + foot targets + contact_seq

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

Key functions:
- `batched_integrate_base_planar`: cumsum of velocity rotated by yaw
- `batched_solve_base_height`: weighted average of foot z + EMA smoothing
- `batched_body_clearance`: expand 8 body sample points → `(N, T, 8, 2)` → batch `terrain.height_at`
- `batched_solve_base_trajectory`: wire all together

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/base_solver.py Go2Pvcnn/tests/test_batched_base_solver.py
git commit -m "feat: add batched base solver with body clearance"
```

---

## Task 7: Batched Foothold Search

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/foothold.py`
- Test: `Go2Pvcnn/tests/test_batched_foothold.py`

Reference: `raw/kinematic_footsteps/scripts/go2fp/foothold.py`

This is the most complex module. Implement in sub-steps.

- [ ] **Step 1: Write failing test for spiral offsets**

Test that `_precompute_spiral_offsets(0.15, 0.03)` generates the same set of `(ix, iy)` offsets as raw `_spiral_square_offsets`.

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement `_precompute_spiral_offsets`**

Port raw `_spiral_square_offsets` generator, convert to `(S, 2)` tensor.

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Write failing test for Raibert + spiral search**

Test N=1, single terrain, compare `batched_compute_footholds` output vs raw `compute_footholds`.

- [ ] **Step 6: Run to verify failure**

- [ ] **Step 7: Implement batched Raibert + spiral search**

Key: `_predict_planar_base_xy` → batched with `torch.where` for `|omega| < 1e-9`. Spiral candidates as `(N, 4, S, 2)`, batch `height_at`/`roughness_at`, scoring, argmin.

- [ ] **Step 8: Run test to verify pass**

- [ ] **Step 9: Write failing test for evaluate_touchdowns + candidate scoring**

- [ ] **Step 10: Run to verify failure**

- [ ] **Step 11: Implement batched touchdown evaluation + candidate scoring**

- [ ] **Step 12: Run test to verify pass**

- [ ] **Step 13: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/foothold.py Go2Pvcnn/tests/test_batched_foothold.py
git commit -m "feat: add batched foothold search with spiral pattern and candidate evaluation"
```

---

## Task 8: Batched Swing Targets

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/swing.py`
- Test: `Go2Pvcnn/tests/test_batched_swing.py`

Reference: `raw/kinematic_footsteps/scripts/go2fp/swing.py`

- [ ] **Step 1: Write failing test**

Test N=1, T=25, known contact_seq + footholds, compare vs raw `compute_swing_targets`.

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

Replace raw's per-leg run-length Python loop with `torch.cumsum` + mask for swing progress. Hermite z + linear xy interpolation as tensor ops.

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/swing.py Go2Pvcnn/tests/test_batched_swing.py
git commit -m "feat: add batched swing target computation"
```

---

## Task 9: Batched Trajectory Main Entry

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/trajectory.py`
- Test: `Go2Pvcnn/tests/test_batched_trajectory.py`
- Test: `Go2Pvcnn/tests/test_batched_trajectory_batch.py`

Reference: `raw/kinematic_footsteps/scripts/go2fp/trajectory.py`

- [ ] **Step 1: Write failing tests for standstill and horizon truncation**

Test cases:
- `test_standstill_zero_command`: zero command → all-stance trajectory, N=1 matches raw
- `test_standstill_below_stop_speed`: command with |v| < `replan_stop_speed` but > `_STANDSTILL_CMD_EPS` → should enter candidate loop, all candidates filtered, fallback to standstill
- `test_horizon_truncation`: `requested_n_frames=100`, `step_freq=2.0`, `dt=0.02` → `cycle_frames=25`, actual T should be 25, not 100
- `test_dual_standstill_thresholds`: verify `_STANDSTILL_CMD_EPS` (1e-5) and `cfg.replan_stop_speed` (0.05) are applied at the correct points

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement `batched_generate_trajectory` with standstill handling and horizon truncation**

Wire all modules: horizon clamp (`min(requested, cycle_frames)`), gait schedule, candidate expansion `(N*K,...)`, foothold search (skip standstill candidates), swing, terrain estimator, base solver, IK/FK, velocities, standstill merge with `torch.where`.

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Write failing test for motion trajectory (N=1 vs raw)**

Full end-to-end: construct terrain + state + command, run both raw and batched, compare all fields.

- [ ] **Step 6: Run to verify failure**

- [ ] **Step 7: Debug and fix any mismatches**

- [ ] **Step 8: Run test to verify pass**

- [ ] **Step 9: Write batch consistency test (N=32)**

Run N=32 with distinct inputs, verify each env's output matches N=1 run with same input.

- [ ] **Step 10: Run to verify pass**

- [ ] **Step 11: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/trajectory.py Go2Pvcnn/tests/test_batched_trajectory.py Go2Pvcnn/tests/test_batched_trajectory_batch.py
git commit -m "feat: add batched_generate_trajectory main entry with end-to-end tests"
```

---

## Task 10: Isaac Lab Integration

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/manager.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py` (new location)
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
- Modify: `Go2Pvcnn/extension/mdp/rewards_reference.py`
- Modify: `Go2Pvcnn/extension/__init__.py`

- [ ] **Step 1: Write failing test for BatchedTrajectoryManager**

```python
# Go2Pvcnn/tests/test_batched_manager.py
class TestBatchedTrajectoryManager(unittest.TestCase):
    def test_replan_at_interval(self):
        """Verify replan fires at step 0, interval, 2*interval, etc."""
        ...
    def test_phase_counter_advances(self):
        """Verify phase_counter increments each step, clamps at num_frames-1."""
        ...
    def test_reset_resets_phase_only(self):
        """Env reset resets phase_counter but not _step_counter."""
        ...
    def test_current_reference_shape(self):
        """Output dict has correct keys and shapes."""
        ...
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement BatchedTrajectoryManager**

```python
# Go2Pvcnn/extension/batched_planner/manager.py
class BatchedTrajectoryManager:
    def __init__(self, cfg, device): ...  # num_envs inferred from first step() call
    def step(self, terrain, states, commands): ...
    def current_reference(self) -> dict[str, Tensor]: ...
    def reset_envs(self, env_mask: Tensor): ...  # reset phase_counter for masked envs
```

Fixed interval global replan, `_step_counter` global (not per-env), `_phase_counter` per-env.
Writes to `env.unwrapped._trajectory_reference_cache` on each `step()`.

- [ ] **Step 4: Run test to verify pass**

- [ ] **Step 5: Move and rewrite env config**

Move `extension/tasks/teacher_elevation_trajectory_env_cfg.py` → `go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`. Remove EventTerm registrations, add batched planner config fields.

- [ ] **Step 3: Update register_envs.py import path**

Change `from extension.tasks.teacher_elevation_trajectory_env_cfg import ...` to `from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import ...`.

- [ ] **Step 4: Update rewards_reference.py**

Adapt `_select_reference_frame` and `_gather_reference_field` to work with new `BatchedTrajectoryResult.gather_at_phase()` output format.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/manager.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/register_envs.py \
  Go2Pvcnn/extension/mdp/rewards_reference.py \
  Go2Pvcnn/extension/__init__.py
git commit -m "feat: add BatchedTrajectoryManager and migrate env config to go2_pvcnn/tasks"
```

---

## Task 11: Comparison Tool & Cleanup

**Files:**
- Create: `Go2Pvcnn/extension/viz/compare_trajectories.py`
- Delete: `Go2Pvcnn/extension/planner/` (old code)
- Delete: `Go2Pvcnn/extension/tasks/` (moved)

- [ ] **Step 1: Implement compare_trajectories.py**

CLI tool: `--no-gui --seed 42 --command "0.5 0.0 0.3"`. Constructs terrain, runs raw + batched, prints per-field max error and PASS/FAIL.

- [ ] **Step 2: Run comparison**

```bash
cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn && conda run -n mujoco_env python extension/viz/compare_trajectories.py --no-gui --seed 42
```

Expected output:
```
=== Trajectory Alignment Report ===
root_pos     max_err: X.Xe-XX  PASS (< 1e-5)
...
=== ALL FIELDS ALIGNED ===
```

- [ ] **Step 3: Remove old extension code**

```bash
cd /home/lhy/testPvcnnWithIsaacsim
rm -rf Go2Pvcnn/extension/planner/
rm -rf Go2Pvcnn/extension/tasks/
rm -f Go2Pvcnn/extension/mdp/reference_trajectory_events.py
```

- [ ] **Step 4: Run all tests to verify nothing broke**

```bash
cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn && conda run -n mujoco_env python -m pytest tests/test_batched_*.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add comparison tool, remove old extension planner code"
```

---

## Task 12: Documentation Update

**Files:**
- Modify: `notes/human/human-09-extension-planner-mapping.md`
- Modify: `notes/human/human-10-extension-planner-runtime.md`
- Modify: `notes/human/human-11-extension-trajectory-reward.md`
- Modify: `notes/ai/ai-09-extension-planner-mapping.md`
- Modify: `notes/ai/ai-10-extension-planner-runtime.md`
- Modify: `notes/ai/ai-11-extension-trajectory-reward.md`

- [ ] **Step 1: Update human-09 and ai-09 mapping documents**

Replace the old module mapping table with `raw ↔ batched_planner` mapping. Keep human/ai pair synchronized.

- [ ] **Step 2: Update human-10 and ai-10 runtime documents**

Replace process pool / EventTerm description with `BatchedTrajectoryManager` GPU description.

- [ ] **Step 3: Update human-11 and ai-11 trajectory reward documents**

Remove "raw 参考重规划与并行" section, add note about fixed-interval GPU replan.

- [ ] **Step 4: Commit**

```bash
git add notes/
git commit -m "docs: update planner notes (human + ai pairs) for batched GPU architecture"
```

---

## Execution Order Summary

```
Task 1  → convention.py, types.py, config.py (foundation)
Task 2  → terrain.py (depends on convention)
Task 3  → gait.py (depends on types)
Task 4  → ik.py (depends on types)
Task 5  → terrain_estimator.py (depends on convention)
Task 6  → base_solver.py (depends on terrain, terrain_estimator, convention)
Task 7  → foothold.py (depends on terrain, gait, types, convention)
Task 8  → swing.py (depends on gait)
Task 9  → trajectory.py (depends on all above)
Task 10 → Isaac Lab integration (depends on trajectory)
Task 11 → comparison tool + cleanup
Task 12 → documentation
```

Tasks 2, 3, 4, 5 are independent of each other and can be parallelized.
Tasks 6, 7, 8 depend on 2-5 but are independent of each other.
