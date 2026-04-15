"""L2 terrain vectorization regression tests.

These tests load golden reference tensors produced by the current serial
implementation and verify that `max_height_along_segment` reproduces them
exactly.  After vectorization they become regression guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_GO2_ROOT = _TESTS_DIR.parent
_RAW_ROOT = _GO2_ROOT.parent / "raw" / "kinematic_footsteps"
for _p in (str(_GO2_ROOT), str(_RAW_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLDEN_DIR = _TESTS_DIR / "fixtures" / "golden"


def _load_terrain_golden() -> dict[str, torch.Tensor]:
    path = GOLDEN_DIR / "golden_terrain_segment.pt"
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}. Run generate_golden.py first.")
    return torch.load(path, weights_only=True)


def _make_terrain_from_golden(golden: dict) -> "BatchedTerrain":
    from extension.batched_planner.terrain import BatchedTerrain

    heightmap = golden["heightmap"]
    if heightmap.ndim == 3:
        heightmap = heightmap.unsqueeze(1)
    return BatchedTerrain(
        heightmap,
        world_x_range=tuple(golden["world_x_range"]),
        world_y_range=tuple(golden["world_y_range"]),
    )


class TestMaxHeightSegmentVectorized:
    """Regression tests for max_height_along_segment vectorization."""

    def test_matches_golden(self):
        """Load golden_terrain_segment.pt, recompute, compare."""
        golden = _load_terrain_golden()
        terrain = _make_terrain_from_golden(golden)

        recomputed = terrain.max_height_along_segment(golden["p0"], golden["p1"])

        torch.testing.assert_close(
            recomputed,
            golden["max_heights"],
            atol=1e-5,
            rtol=1e-5,
            msg="max_height_along_segment diverged from golden reference",
        )

    def test_single_env(self):
        """N=1 basic case: single environment terrain query."""
        from extension.batched_planner.terrain import BatchedTerrain

        heightmap = torch.rand(1, 1, 16, 16, dtype=torch.float32)
        terrain = BatchedTerrain(
            heightmap,
            world_x_range=(-1.0, 1.0),
            world_y_range=(-1.0, 1.0),
        )
        p0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        p1 = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

        result = terrain.max_height_along_segment(p0, p1)

        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert torch.isfinite(result).all(), "Result contains non-finite values"

    def test_batch_matches_serial_per_env(self):
        """N=8: verify batch result matches calling per-env serially."""
        from extension.batched_planner.terrain import BatchedTerrain

        N = 8
        torch.manual_seed(42)
        heightmaps = torch.rand(N, 1, 20, 20, dtype=torch.float32)
        terrain = BatchedTerrain(
            heightmaps,
            world_x_range=(-1.0, 1.0),
            world_y_range=(-1.0, 1.0),
        )

        p0 = torch.rand(N, 2, dtype=torch.float32) * 1.6 - 0.8
        p1 = torch.rand(N, 2, dtype=torch.float32) * 1.6 - 0.8

        batched_result = terrain.max_height_along_segment(p0, p1)

        serial_results = []
        for idx in range(N):
            single_terrain = BatchedTerrain(
                heightmaps[idx : idx + 1],
                world_x_range=(-1.0, 1.0),
                world_y_range=(-1.0, 1.0),
            )
            r = single_terrain.max_height_along_segment(
                p0[idx : idx + 1], p1[idx : idx + 1]
            )
            serial_results.append(r)
        serial_result = torch.cat(serial_results)

        torch.testing.assert_close(
            batched_result,
            serial_result,
            atol=1e-5,
            rtol=1e-5,
            msg="Batched result does not match serial per-env computation",
        )

    def test_flat_terrain_returns_zero(self):
        """All heights 0 -> max height = 0."""
        from extension.batched_planner.terrain import BatchedTerrain

        N = 4
        heightmaps = torch.zeros(N, 1, 16, 16, dtype=torch.float32)
        terrain = BatchedTerrain(
            heightmaps,
            world_x_range=(-1.0, 1.0),
            world_y_range=(-1.0, 1.0),
        )

        p0 = torch.tensor(
            [[0.0, 0.0], [0.1, 0.1], [-0.5, 0.2], [0.3, -0.3]],
            dtype=torch.float32,
        )
        p1 = torch.tensor(
            [[0.5, 0.5], [0.3, 0.3], [-0.1, -0.1], [0.5, 0.1]],
            dtype=torch.float32,
        )

        result = terrain.max_height_along_segment(p0, p1)

        torch.testing.assert_close(
            result,
            torch.zeros(N, dtype=torch.float32),
            atol=1e-7,
            rtol=0.0,
            msg="Flat terrain max_height should be 0",
        )

    def test_zero_length_segment(self):
        """When p0 == p1, should return height at that point."""
        from extension.batched_planner.terrain import BatchedTerrain

        heightmap = torch.ones(2, 1, 8, 8, dtype=torch.float32) * 3.0
        terrain = BatchedTerrain(
            heightmap,
            world_x_range=(-1.0, 1.0),
            world_y_range=(-1.0, 1.0),
        )

        p = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
        result = terrain.max_height_along_segment(p, p)

        torch.testing.assert_close(
            result,
            torch.tensor([3.0, 3.0], dtype=torch.float32),
            atol=1e-5,
            rtol=0.0,
            msg="Zero-length segment should return height at point",
        )

    def test_broadcast_single_endpoint(self):
        """Single (2,) endpoint should broadcast to all envs."""
        from extension.batched_planner.terrain import BatchedTerrain

        N = 3
        heightmap = torch.zeros(N, 1, 8, 8, dtype=torch.float32)
        terrain = BatchedTerrain(
            heightmap,
            world_x_range=(-1.0, 1.0),
            world_y_range=(-1.0, 1.0),
        )

        p0 = torch.tensor([0.0, 0.0], dtype=torch.float32)
        p1 = torch.tensor([0.5, 0.5], dtype=torch.float32)

        result = terrain.max_height_along_segment(p0, p1)
        assert result.shape == (N,), f"Expected shape ({N},), got {result.shape}"


class TestBatchMaxHeightMultiLeg:
    """Tests for the multi-leg batched terrain query (Task 13)."""

    def test_multi_leg_matches_per_leg_serial(self):
        """batch_max_height_along_segment(N, K, 2) matches K serial calls."""
        from extension.batched_planner.terrain import BatchedTerrain

        N, K = 4, 4
        torch.manual_seed(123)
        heightmaps = torch.rand(N, 1, 20, 20, dtype=torch.float32) * 0.5
        terrain = BatchedTerrain(
            heightmaps,
            world_x_range=(-1.0, 1.0),
            world_y_range=(-1.0, 1.0),
        )

        p0 = torch.rand(N, K, 2, dtype=torch.float32) * 1.6 - 0.8
        p1 = torch.rand(N, K, 2, dtype=torch.float32) * 1.6 - 0.8

        if not hasattr(terrain, "batch_max_height_along_segment"):
            pytest.skip("batch_max_height_along_segment not yet implemented")

        batched = terrain.batch_max_height_along_segment(p0, p1)
        assert batched.shape == (N, K)

        serial = torch.stack(
            [terrain.max_height_along_segment(p0[:, k, :], p1[:, k, :]) for k in range(K)],
            dim=1,
        )

        torch.testing.assert_close(
            batched,
            serial,
            atol=1e-5,
            rtol=1e-5,
            msg="Multi-leg batch does not match per-leg serial calls",
        )

    def test_multi_leg_shape(self):
        """batch_max_height_along_segment returns (N, K) shape."""
        from extension.batched_planner.terrain import BatchedTerrain

        N, K = 2, 4
        heightmaps = torch.zeros(N, 1, 8, 8, dtype=torch.float32)
        terrain = BatchedTerrain(
            heightmaps,
            world_x_range=(-1.0, 1.0),
            world_y_range=(-1.0, 1.0),
        )

        p0 = torch.zeros(N, K, 2, dtype=torch.float32)
        p1 = torch.ones(N, K, 2, dtype=torch.float32) * 0.5

        if not hasattr(terrain, "batch_max_height_along_segment"):
            pytest.skip("batch_max_height_along_segment not yet implemented")

        result = terrain.batch_max_height_along_segment(p0, p1)
        assert result.shape == (N, K), f"Expected ({N}, {K}), got {result.shape}"


class TestTerrainDynamicN:
    @pytest.mark.parametrize("n_envs", [1, 32, 256])
    def test_max_height_segment_arbitrary_n(self, n_envs):
        from extension.batched_planner.terrain import BatchedTerrain

        torch.manual_seed(7)
        heightmaps = torch.rand(n_envs, 1, 20, 20, dtype=torch.float32)
        terrain = BatchedTerrain(
            heightmaps,
            world_x_range=(-1.0, 1.0),
            world_y_range=(-1.0, 1.0),
        )
        p0 = torch.rand(n_envs, 2, dtype=torch.float32) * 1.6 - 0.8
        p1 = torch.rand(n_envs, 2, dtype=torch.float32) * 1.6 - 0.8

        result = terrain.max_height_along_segment(p0, p1)
        assert result.shape == (n_envs,)
        assert torch.isfinite(result).all()
