#!/usr/bin/env python3
"""Generate golden reference tensors from the current serial batched planner.

Run standalone:
    cd /home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn && python tests/fixtures/generate_golden.py

Produces .pt files in tests/fixtures/golden/ that L2 regression tests load
to guard against vectorization regressions.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ── path setup (mirrors conftest.py) ──
TESTS_DIR = Path(__file__).resolve().parent.parent
GO2_ROOT = TESTS_DIR.parent
REPO_ROOT = GO2_ROOT.parent
RAW_ROOT = REPO_ROOT / "raw" / "kinematic_footsteps"
GOLDEN_DIR = TESTS_DIR / "fixtures" / "golden"

for _path in (str(GO2_ROOT), str(RAW_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ── aligned config (mirrors conftest._build_golden_alignment) ──

def _build_golden_alignment() -> dict[str, Any]:
    from extension.batched_planner.config import BatchedTrajectoryConfig
    from scripts.go2fp.config import TrajectoryConfig

    raw = TrajectoryConfig()
    batched_defaults = BatchedTrajectoryConfig()
    raw_d = dataclasses.asdict(raw)
    batched_d = dataclasses.asdict(batched_defaults)
    out: dict[str, Any] = {}
    for name in BatchedTrajectoryConfig.__dataclass_fields__:
        out[name] = raw_d[name] if name in raw_d else batched_d[name]
    for name in TrajectoryConfig.__dataclass_fields__:
        if name not in out:
            out[name] = raw_d[name]
    return out


def _config_kwargs(cls: type, golden: dict[str, Any]) -> dict[str, Any]:
    names = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in golden.items() if k in names}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. golden_swing_targets.pt
# ═══════════════════════════════════════════════════════════════════════════════

def generate_swing_targets() -> dict[str, torch.Tensor]:
    from extension.batched_planner.config import BatchedTrajectoryConfig
    from extension.batched_planner.gait import GAIT_PARAMS, batched_gait_schedule
    from extension.batched_planner.swing import batched_compute_swing_targets

    golden = _build_golden_alignment()
    cfg = BatchedTrajectoryConfig(**_config_kwargs(BatchedTrajectoryConfig, golden))

    offsets = torch.as_tensor(GAIT_PARAMS["trot"]["offsets"], dtype=torch.float64)
    n_frames, dt = 25, 0.02

    # ── N=1 ──
    contact_seq_n1 = batched_gait_schedule(0.0, n_frames, dt, cfg.step_freq, cfg.duty_factor, offsets)
    foot_pos_n1 = torch.tensor(
        [[[0.19, 0.11, 0.0],
          [0.19, -0.11, 0.0],
          [-0.19, 0.11, 0.0],
          [-0.19, -0.11, 0.0]]],
        dtype=torch.float32,
    )
    touchdown_pos_n1 = foot_pos_n1.clone()
    touchdown_pos_n1[..., 0] += 0.05

    swing_targets_n1 = batched_compute_swing_targets(
        contact_seq_n1,
        foot_pos_n1,
        touchdown_pos_n1,
        cfg.step_height,
        terrain_max_heights=None,
    )

    # ── N=4 ──
    contact_seq_n4 = contact_seq_n1.expand(4, -1, -1).contiguous()
    foot_pos_n4 = foot_pos_n1.expand(4, -1, -1).clone()
    for i in range(4):
        foot_pos_n4[i, :, 0] += 0.01 * i
        foot_pos_n4[i, :, 1] += 0.005 * i
    touchdown_pos_n4 = foot_pos_n4.clone()
    touchdown_pos_n4[..., 0] += 0.05

    swing_targets_n4 = batched_compute_swing_targets(
        contact_seq_n4,
        foot_pos_n4,
        touchdown_pos_n4,
        cfg.step_height,
        terrain_max_heights=None,
    )

    return {
        "contact_seq_n1": contact_seq_n1,
        "foot_pos_n1": foot_pos_n1,
        "touchdown_pos_n1": touchdown_pos_n1,
        "swing_targets_n1": swing_targets_n1,
        "contact_seq_n4": contact_seq_n4,
        "foot_pos_n4": foot_pos_n4,
        "touchdown_pos_n4": touchdown_pos_n4,
        "swing_targets_n4": swing_targets_n4,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. golden_trajectory.pt
# ═══════════════════════════════════════════════════════════════════════════════

def generate_trajectory() -> dict[str, Any]:
    from extension.batched_planner.config import BatchedTrajectoryConfig
    from extension.batched_planner.terrain import PlannerTerrain
    from extension.batched_planner.trajectory import batched_generate_trajectory
    from extension.batched_planner.types import BatchedRobotState

    from tests.fixtures.terrain_adapter import make_flat_terrains

    golden = _build_golden_alignment()
    cfg = BatchedTrajectoryConfig(**_config_kwargs(BatchedTrajectoryConfig, golden))

    _, ray_hits, wx, wy = make_flat_terrains()
    terrain = PlannerTerrain.from_ray_hits(ray_hits, world_x_range=wx, world_y_range=wy)

    root_pos = torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float64)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    joint_angles = torch.zeros((1, 12), dtype=torch.float64)
    foot_pos = torch.tensor(
        [[[0.19, 0.11, 0.0],
          [0.19, -0.11, 0.0],
          [-0.19, 0.11, 0.0],
          [-0.19, -0.11, 0.0]]],
        dtype=torch.float64,
    )
    state = BatchedRobotState(root_pos=root_pos, root_quat=root_quat, joint_angles=joint_angles, foot_pos=foot_pos)

    scenarios = {
        "forward": [0.3, 0.0, 0.0],
        "lateral": [0.0, 0.2, 0.0],
        "turn": [0.0, 0.0, 0.5],
        "standstill": [0.0, 0.0, 0.0],
    }

    n_frames, dt = 25, 0.02
    out: dict[str, Any] = {}
    for label, cmd_vals in scenarios.items():
        cmd = torch.tensor([cmd_vals], dtype=torch.float64)
        result = batched_generate_trajectory(terrain, state, cmd, n_frames, dt, cfg=cfg)
        out[f"{label}_root_pos_w"] = result.root_pos_w
        out[f"{label}_root_quat_w"] = result.root_quat_w
        out[f"{label}_joint_angles"] = result.joint_angles
        out[f"{label}_foot_pos_w"] = result.foot_pos_w
        out[f"{label}_contact_state"] = result.contact_state
        out[f"{label}_planned_touchdown_w"] = result.planned_touchdown_w

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. golden_terrain_segment.pt
# ═══════════════════════════════════════════════════════════════════════════════

def generate_terrain_segment() -> dict[str, Any]:
    from extension.batched_planner.terrain import PlannerTerrain

    from tests.fixtures.terrain_adapter import make_flat_terrains

    _, ray_hits, wx, wy = make_flat_terrains(batch_size=4)
    terrain = PlannerTerrain.from_ray_hits(ray_hits, world_x_range=wx, world_y_range=wy)

    p0 = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [-0.5, 0.2], [0.3, -0.3]],
        dtype=torch.float32,
    )
    p1 = torch.tensor(
        [[0.2, 0.0], [0.3, 0.3], [-0.1, -0.1], [0.5, 0.1]],
        dtype=torch.float32,
    )
    max_heights = terrain.max_height_along_segment(p0, p1)

    return {
        "p0": p0,
        "p1": p1,
        "max_heights": max_heights,
        "heightmap": terrain.heightmaps.squeeze(1),
        "world_x_range": wx,
        "world_y_range": wy,
    }


# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating golden_swing_targets.pt ...")
    swing_data = generate_swing_targets()
    torch.save(swing_data, GOLDEN_DIR / "golden_swing_targets.pt")
    print(f"  Saved {len(swing_data)} tensors")

    print("Generating golden_trajectory.pt ...")
    traj_data = generate_trajectory()
    torch.save(traj_data, GOLDEN_DIR / "golden_trajectory.pt")
    print(f"  Saved {len(traj_data)} tensors")

    print("Generating golden_terrain_segment.pt ...")
    terrain_data = generate_terrain_segment()
    torch.save(terrain_data, GOLDEN_DIR / "golden_terrain_segment.pt")
    print(f"  Saved {len(terrain_data)} tensors")

    print(f"\nAll golden references written to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
