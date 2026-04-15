"""Pytest configuration: sys.path, golden alignment params, and shared fixtures."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parent
GO2_ROOT = TESTS_DIR.parent
REPO_ROOT = GO2_ROOT.parent
RAW_ROOT = REPO_ROOT / "raw" / "kinematic_footsteps"
GOLDEN_DIR = TESTS_DIR / "fixtures" / "golden"

for _path in (str(GO2_ROOT), str(RAW_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _build_golden_alignment() -> dict[str, Any]:
    """Unified planner params: shared fields follow raw ``TrajectoryConfig`` defaults."""
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


GOLDEN_ALIGNMENT: dict[str, Any] = _build_golden_alignment()


def _config_kwargs_from_golden(cls: type[Any], golden: dict[str, Any]) -> dict[str, Any]:
    names = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in golden.items() if k in names}


@pytest.fixture
def aligned_configs() -> tuple[Any, Any]:
    """``TrajectoryConfig`` and ``BatchedTrajectoryConfig`` built from ``GOLDEN_ALIGNMENT``."""
    from extension.batched_planner.config import BatchedTrajectoryConfig
    from scripts.go2fp.config import TrajectoryConfig

    raw_cfg = TrajectoryConfig(**_config_kwargs_from_golden(TrajectoryConfig, GOLDEN_ALIGNMENT))
    batched_cfg = BatchedTrajectoryConfig(**_config_kwargs_from_golden(BatchedTrajectoryConfig, GOLDEN_ALIGNMENT))
    return raw_cfg, batched_cfg


@pytest.fixture
def flat_terrain_pair() -> tuple[Any, Any, tuple[float, float], tuple[float, float]]:
    """Flat terrain: numpy heightmap, ``PlannerTerrain``, world ranges."""
    from extension.batched_planner.terrain import PlannerTerrain

    from tests.fixtures.terrain_adapter import make_flat_terrains

    heightmap_np, ray_hits, world_x_range, world_y_range = make_flat_terrains()
    terrain = PlannerTerrain.from_ray_hits(
        ray_hits,
        world_x_range=world_x_range,
        world_y_range=world_y_range,
    )
    return heightmap_np, terrain, world_x_range, world_y_range


@pytest.fixture
def default_initial_state() -> Any:
    """Single-env ``BatchedRobotState`` for a nominal standing pose on flat ground."""
    from extension.batched_planner.types import BatchedRobotState

    n = 1
    root_pos = torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    joint_angles = torch.zeros((n, 12), dtype=torch.float32)
    foot_pos = torch.tensor(
        [
            [
                [0.19, 0.11, 0.0],
                [0.19, -0.11, 0.0],
                [-0.19, 0.11, 0.0],
                [-0.19, -0.11, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    return BatchedRobotState(root_pos=root_pos, root_quat=root_quat, joint_angles=joint_angles, foot_pos=foot_pos)
