from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.reference.raw_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()
from scripts.go2fp.terrain import _metric_max_height_along_segment, _metric_slope_magnitude, _sample_metric_grid


class BatchedTerrainTest(unittest.TestCase):
    def setUp(self) -> None:
        from extension.batched_planner.terrain import BatchedTerrain, PlannerTerrain

        self.BatchedTerrain = BatchedTerrain
        self.PlannerTerrain = PlannerTerrain

    def _make_heightmaps(self) -> torch.Tensor:
        base = torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            dtype=torch.float32,
        )
        second = torch.tensor(
            [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [3.0, 4.0, 6.0]],
            dtype=torch.float32,
        )
        return torch.stack([base.unsqueeze(0), second.unsqueeze(0)])

    def _make_identical_heightmaps(self) -> torch.Tensor:
        base = torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            dtype=torch.float32,
        )
        return torch.stack([base.unsqueeze(0) for _ in range(4)])

    def _make_terrain(self, heightmaps: torch.Tensor | None = None):
        if heightmaps is None:
            heightmaps = self._make_heightmaps()
        return self.BatchedTerrain(heightmaps, world_x_range=(0.0, 2.0), world_y_range=(0.0, 2.0))

    def _make_ray_hits(self, heightmaps: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = heightmaps.shape
        hits = torch.zeros((batch_size, height * width, 3), dtype=heightmaps.dtype)
        for batch_idx in range(batch_size):
            idx = 0
            for row in range(height):
                for col in range(width):
                    hits[batch_idx, idx, 0] = float(col)
                    hits[batch_idx, idx, 1] = float(row)
                    hits[batch_idx, idx, 2] = float(heightmaps[batch_idx, 0, row, col])
                    idx += 1
        return hits

    def _make_offset_ray_hits(self, heightmaps: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = heightmaps.shape
        hits = torch.zeros((batch_size, height * width, 3), dtype=heightmaps.dtype)
        for batch_idx in range(batch_size):
            idx = 0
            for row in range(height):
                for col in range(width):
                    hits[batch_idx, idx, 0] = float(col + 10.0)
                    hits[batch_idx, idx, 1] = float(row + 20.0)
                    hits[batch_idx, idx, 2] = float(heightmaps[batch_idx, 0, row, col])
                    idx += 1
            hits[batch_idx, -1] = torch.tensor([float("inf"), float("nan"), 999.0], dtype=heightmaps.dtype)
        return hits

    def _make_non_square_heightmaps(self) -> torch.Tensor:
        base = torch.tensor(
            [
                [0.0, 0.5, 1.0],
                [1.0, 1.5, 2.0],
                [2.0, 2.5, 3.0],
                [3.0, 3.5, 4.0],
            ],
            dtype=torch.float32,
        )
        second = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 2.0],
                [3.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        return torch.stack([base.unsqueeze(0), second.unsqueeze(0)])

    def _make_segment_regression_heightmaps(self) -> torch.Tensor:
        base = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        second = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return torch.stack([base.unsqueeze(0), second.unsqueeze(0)])

    def test_height_at_single_point_matches_raw(self) -> None:
        heightmaps = self._make_heightmaps()
        terrain = self._make_terrain(heightmaps)
        point = torch.tensor([0.5, 1.25], dtype=torch.float32)

        actual = terrain.height_at(point)
        expected = torch.tensor(
            [
                _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 0.5, 1.25, (0.0, 2.0), (0.0, 2.0)),
                _sample_metric_grid(heightmaps[1, 0].cpu().numpy(), 0.5, 1.25, (0.0, 2.0), (0.0, 2.0)),
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(actual, expected)

    def test_plannerterrain_rejects_direct_heightmap_constructor(self) -> None:
        heightmaps = self._make_heightmaps()

        with self.assertRaisesRegex(TypeError, "from_ray_hits"):
            self.PlannerTerrain(heightmaps, world_x_range=(0.0, 2.0), world_y_range=(0.0, 2.0))

    def test_plannerterrain_from_ray_hits_sanitizes_invalid_hits_and_normalizes_dtype(self) -> None:
        ray_hits = torch.tensor(
            [
                [
                    [0.0, 0.0, 1.5],
                    [1.0, 0.0, float("inf")],
                    [0.0, 1.0, float("nan")],
                    [1.0, 1.0, -2.5],
                ]
            ],
            dtype=torch.float64,
        )

        terrain = self.PlannerTerrain.from_ray_hits(
            ray_hits,
            world_x_range=(0.0, 1.0),
            world_y_range=(0.0, 1.0),
        )

        self.assertEqual(terrain.heightmaps.dtype, torch.float64)
        self.assertTrue(torch.isfinite(terrain.heightmaps).all())
        torch.testing.assert_close(
            terrain.heightmaps,
            torch.tensor([[[[1.5, 0.0], [0.0, -2.5]]]], dtype=torch.float64),
        )

        sample = terrain.height_at(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.assertEqual(sample.shape, (1,))
        self.assertTrue(torch.isfinite(sample))

    def test_plannerterrain_from_ray_hits_derives_canonical_ranges_from_xy_layout(self) -> None:
        heightmaps = self._make_heightmaps()[:1]
        ray_hits = self._make_offset_ray_hits(heightmaps)

        terrain = self.PlannerTerrain.from_ray_hits(ray_hits)

        self.assertEqual(terrain.world_x_range, (10.0, 12.0))
        self.assertEqual(terrain.world_y_range, (20.0, 22.0))
        torch.testing.assert_close(
            terrain.heightmaps,
            self.PlannerTerrain.from_ray_hits(
                ray_hits,
                world_x_range=(10.0, 12.0),
                world_y_range=(20.0, 22.0),
            ).heightmaps,
        )

    def test_plannerterrain_from_ray_hits_allows_explicit_range_override(self) -> None:
        heightmaps = self._make_heightmaps()[:1]
        ray_hits = self._make_offset_ray_hits(heightmaps)

        terrain = self.PlannerTerrain.from_ray_hits(
            ray_hits,
            world_x_range=(0.0, 1.0),
            world_y_range=(0.0, 1.0),
        )

        self.assertEqual(terrain.world_x_range, (0.0, 1.0))
        self.assertEqual(terrain.world_y_range, (0.0, 1.0))

    def test_plannerterrain_query_shapes_follow_abi_rules(self) -> None:
        single_heightmaps = torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0]]]],
            dtype=torch.float32,
        )
        batched_heightmaps = torch.stack([single_heightmaps[0], single_heightmaps[0] + 10.0])

        single_terrain = self.PlannerTerrain.from_ray_hits(
            self._make_ray_hits(single_heightmaps),
            world_x_range=(0.0, 1.0),
            world_y_range=(0.0, 1.0),
        )
        batched_terrain = self.PlannerTerrain.from_ray_hits(
            self._make_ray_hits(batched_heightmaps),
            world_x_range=(0.0, 1.0),
            world_y_range=(0.0, 1.0),
        )

        single_point = torch.tensor([0.25, 0.75], dtype=torch.float32)
        single_point_2d = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
        multi_points = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float32)
        batched_points = torch.tensor(
            [
                [[0.25, 0.75], [0.75, 0.25]],
                [[0.75, 0.75], [0.25, 0.25]],
            ],
            dtype=torch.float32,
        )
        single_segment_p0 = torch.tensor([0.0, 0.0], dtype=torch.float32)
        single_segment_p1 = torch.tensor([1.0, 1.0], dtype=torch.float32)
        single_segment_p0_2d = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        single_segment_p1_2d = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        batched_segment_p0 = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        batched_segment_p1 = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

        self.assertEqual(single_terrain.height_at(single_point).shape, (1,))
        self.assertEqual(single_terrain.height_at(single_point_2d).shape, (1,))
        torch.testing.assert_close(single_terrain.height_at(single_point), single_terrain.height_at(single_point_2d))
        self.assertEqual(single_terrain.height_at(multi_points).shape, (1, 2))
        self.assertEqual(single_terrain.roughness_at(single_point).shape, (1,))
        self.assertEqual(single_terrain.roughness_at(single_point_2d).shape, (1,))
        torch.testing.assert_close(single_terrain.roughness_at(single_point), single_terrain.roughness_at(single_point_2d))
        self.assertEqual(single_terrain.roughness_at(multi_points).shape, (1, 2))
        self.assertEqual(single_terrain.max_height_along_segment(single_segment_p0, single_segment_p1).shape, (1,))
        self.assertEqual(
            single_terrain.max_height_along_segment(single_segment_p0_2d, single_segment_p1_2d).shape,
            (1,),
        )
        torch.testing.assert_close(
            single_terrain.max_height_along_segment(single_segment_p0, single_segment_p1),
            single_terrain.max_height_along_segment(single_segment_p0_2d, single_segment_p1_2d),
        )

        self.assertEqual(batched_terrain.height_at(single_point).shape, (2,))
        self.assertEqual(batched_terrain.height_at(batched_points).shape, (2, 2))
        self.assertEqual(batched_terrain.roughness_at(single_point).shape, (2,))
        self.assertEqual(batched_terrain.roughness_at(batched_points).shape, (2, 2))
        self.assertEqual(batched_terrain.max_height_along_segment(batched_segment_p0, batched_segment_p1).shape, (2,))

    def test_height_at_multiple_points_matches_raw(self) -> None:
        heightmaps = self._make_heightmaps()
        terrain = self._make_terrain(heightmaps[:1])
        points = torch.tensor([[0.25, 0.25], [1.5, 1.75], [0.75, 1.0]], dtype=torch.float32)

        actual = terrain.height_at(points)[0]
        expected = torch.tensor(
            [
                _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 0.25, 0.25, (0.0, 2.0), (0.0, 2.0)),
                _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.5, 1.75, (0.0, 2.0), (0.0, 2.0)),
                _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 0.75, 1.0, (0.0, 2.0), (0.0, 2.0)),
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(actual, expected)

    def test_roughness_at_off_center_matches_raw(self) -> None:
        heightmaps = self._make_non_square_heightmaps()
        terrain = self._make_terrain(heightmaps)
        point = torch.tensor([1.3, 0.7], dtype=torch.float32)

        actual = terrain.roughness_at(point)
        expected = torch.tensor(
            [
                _metric_slope_magnitude(heightmaps[0, 0].cpu().numpy(), 1.3, 0.7, (0.0, 2.0), (0.0, 2.0)),
                _metric_slope_magnitude(heightmaps[1, 0].cpu().numpy(), 1.3, 0.7, (0.0, 2.0), (0.0, 2.0)),
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(actual, expected)

    def test_max_height_along_segment_matches_raw_non_square_grid(self) -> None:
        heightmaps = self._make_segment_regression_heightmaps()
        terrain = self._make_terrain(heightmaps)
        p0 = torch.tensor([0.0, 0.2], dtype=torch.float32)
        p1 = torch.tensor([2.0, 1.6], dtype=torch.float32)
        step = (2.0 - 0.0) / (heightmaps.shape[-1] - 1)

        actual = terrain.max_height_along_segment(p0, p1)
        expected = torch.tensor(
            [
                _metric_max_height_along_segment(
                    heightmaps[0, 0].cpu().numpy(),
                    (0.0, 0.2),
                    (2.0, 1.6),
                    (0.0, 2.0),
                    (0.0, 2.0),
                    step,
                ),
                _metric_max_height_along_segment(
                    heightmaps[1, 0].cpu().numpy(),
                    (0.0, 0.2),
                    (2.0, 1.6),
                    (0.0, 2.0),
                    (0.0, 2.0),
                    step,
                ),
            ],
            dtype=torch.float32,
        )

        # Vectorized uses fixed sample count; small sampling-position
        # differences vs. variable-count raw are expected on steep heightmaps.
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)

    def test_shape_n_two_requires_batched_points_explicitly(self) -> None:
        terrain = self._make_terrain()
        points = torch.tensor([[0.25, 0.25], [1.25, 0.75]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "single terrain"):
            terrain.height_at(points)

    def test_batched_point_queries_route_per_terrain(self) -> None:
        heightmaps = self._make_identical_heightmaps()
        heightmaps[1] = heightmaps[1] + 10.0
        terrain = self._make_terrain(heightmaps)
        points = torch.tensor(
            [
                [[0.25, 0.25], [1.25, 0.75]],
                [[1.75, 1.25], [0.25, 1.75]],
                [[0.5, 1.5], [1.5, 0.5]],
                [[1.0, 1.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        actual = terrain.height_at(points)
        expected = torch.tensor(
            [
                [
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 0.25, 0.25, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.25, 0.75, (0.0, 2.0), (0.0, 2.0)),
                ],
                [
                    _sample_metric_grid(heightmaps[1, 0].cpu().numpy(), 1.75, 1.25, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[1, 0].cpu().numpy(), 0.25, 1.75, (0.0, 2.0), (0.0, 2.0)),
                ],
                [
                    _sample_metric_grid(heightmaps[2, 0].cpu().numpy(), 0.5, 1.5, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[2, 0].cpu().numpy(), 1.5, 0.5, (0.0, 2.0), (0.0, 2.0)),
                ],
                [
                    _sample_metric_grid(heightmaps[3, 0].cpu().numpy(), 1.0, 1.0, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[3, 0].cpu().numpy(), 0.0, 0.0, (0.0, 2.0), (0.0, 2.0)),
                ],
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(actual, expected)

    def test_nonfloating_heightmaps_are_coerced_to_float(self) -> None:
        heightmaps = torch.tensor(
            [
                [[[0, 1], [2, 3]]],
                [[[4, 5], [6, 7]]],
            ],
            dtype=torch.int32,
        )
        terrain = self.BatchedTerrain(heightmaps, world_x_range=(0.0, 1.0), world_y_range=(0.0, 1.0))

        self.assertTrue(torch.is_floating_point(terrain.heightmaps))

    def test_max_height_along_segment_uses_single_grid_sample_call(self) -> None:
        """Vectorized impl should use exactly one grid_sample call per invocation."""
        heightmaps = self._make_identical_heightmaps()
        terrain = self._make_terrain(heightmaps)
        p0 = torch.tensor([0.0, 0.2], dtype=torch.float32)
        p1 = torch.tensor([2.0, 1.6], dtype=torch.float32)

        from extension.batched_planner import terrain as terrain_module

        call_count = 0
        real_grid_sample = terrain_module.F.grid_sample

        def wrapped_grid_sample(input: torch.Tensor, grid: torch.Tensor, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return real_grid_sample(input, grid, *args, **kwargs)

        terrain_module.F.grid_sample = wrapped_grid_sample
        try:
            terrain.max_height_along_segment(p0, p1)
        finally:
            terrain_module.F.grid_sample = real_grid_sample

        self.assertEqual(call_count, 1, "Vectorized segment query should use exactly one grid_sample call")

    def test_from_ray_hits_reshapes_flattened_square_grid(self) -> None:
        heightmaps = self._make_heightmaps()

        def make_flat_hits(grid: torch.Tensor) -> torch.Tensor:
            hits = torch.zeros((grid.shape[-2] * grid.shape[-1], 3), dtype=torch.float32)
            idx = 0
            for row in range(grid.shape[-2]):
                for col in range(grid.shape[-1]):
                    hits[idx, 0] = float(col)
                    hits[idx, 1] = float(row)
                    hits[idx, 2] = float(grid[0, row, col])
                    idx += 1
            return hits

        ray_hits = torch.stack([make_flat_hits(heightmaps[0]), make_flat_hits(heightmaps[1])])
        terrain = self.BatchedTerrain.from_ray_hits(
            ray_hits,
            world_x_range=(0.0, 2.0),
            world_y_range=(0.0, 2.0),
        )

        torch.testing.assert_close(terrain.heightmaps, heightmaps)

    def test_from_ray_hits_rejects_non_square_flattened_grid(self) -> None:
        ray_hits = torch.zeros((2, 8, 3), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "square"):
            self.BatchedTerrain.from_ray_hits(
                ray_hits,
                world_x_range=(0.0, 2.0),
                world_y_range=(0.0, 2.0),
            )

    def test_batch_consistency(self) -> None:
        heightmaps = self._make_identical_heightmaps()
        terrain = self._make_terrain(heightmaps)
        points = torch.tensor(
            [
                [[0.25, 0.25], [1.25, 0.75], [1.75, 1.25]],
                [[0.25, 0.25], [1.25, 0.75], [1.75, 1.25]],
                [[0.25, 0.25], [1.25, 0.75], [1.75, 1.25]],
                [[0.25, 0.25], [1.25, 0.75], [1.75, 1.25]],
            ],
            dtype=torch.float32,
        )

        actual = terrain.height_at(points)
        expected = torch.tensor(
            [
                [
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 0.25, 0.25, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.25, 0.75, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.75, 1.25, (0.0, 2.0), (0.0, 2.0)),
                ],
                [
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 0.25, 0.25, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.25, 0.75, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.75, 1.25, (0.0, 2.0), (0.0, 2.0)),
                ],
                [
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 0.25, 0.25, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.25, 0.75, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.75, 1.25, (0.0, 2.0), (0.0, 2.0)),
                ],
                [
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 0.25, 0.25, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.25, 0.75, (0.0, 2.0), (0.0, 2.0)),
                    _sample_metric_grid(heightmaps[0, 0].cpu().numpy(), 1.75, 1.25, (0.0, 2.0), (0.0, 2.0)),
                ],
            ],
            dtype=torch.float32,
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0], actual[1])


if __name__ == "__main__":
    unittest.main()
