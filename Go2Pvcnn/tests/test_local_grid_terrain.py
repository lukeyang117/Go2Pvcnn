"""Tests for LocalGridTerrain (raw go2fp terrain adapter)."""

from __future__ import annotations

import unittest

import numpy as np

from Go2Pvcnn.extension.planner.adapters.isaac_heightmap import LocalGridTerrain


class LocalGridTerrainTest(unittest.TestCase):
    def test_height_at_center_flat(self):
        z = np.ones((5, 5), dtype=np.float64)
        t = LocalGridTerrain(z, (1.0, 1.0), (0.0, 0.0), 0.0)
        self.assertAlmostEqual(t.height_at(0.0, 0.0), 1.0, places=5)

    def test_from_world_ray_hits_flat_square(self):
        side = 4
        hits = np.zeros((side, side, 3), dtype=np.float64)
        for i in range(side):
            for j in range(side):
                hits[i, j, 0] = -0.5 + j / (side - 1)
                hits[i, j, 1] = -0.5 + i / (side - 1)
                hits[i, j, 2] = 0.25
        root = np.array([0.0, 0.0, 0.4], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        t = LocalGridTerrain.from_world_ray_hits(hits, root_pos_w=root, root_quat_w=quat, size_xy=(1.0, 1.0))
        self.assertAlmostEqual(t.height_at(0.0, 0.0), 0.25, places=4)

    def test_max_height_along_segment(self):
        z = np.full((5, 5), 0.1, dtype=np.float64)
        z[2, 2] = 2.0
        t = LocalGridTerrain(z, (2.0, 2.0), (0.0, 0.0), 0.0)
        m = t.max_height_along_segment((0.0, 0.0), (0.05, 0.05))
        self.assertGreaterEqual(m, 1.5)


if __name__ == "__main__":
    unittest.main()
