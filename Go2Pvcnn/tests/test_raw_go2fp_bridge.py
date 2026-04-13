"""Tests for raw kinematic_footsteps -> ReferenceTrajectoryCache bridge."""

from __future__ import annotations

import sys
import unittest

import numpy as np
import torch

from Go2Pvcnn.extension.planner.runtime.raw_go2fp_bridge import (
    ensure_kinematic_footsteps_on_syspath,
    generate_reference_cache_with_raw,
    kinematic_footsteps_repo_root,
    trajectory_result_to_reference_cache,
)


class RawGo2fpBridgeTest(unittest.TestCase):
    def test_kinematic_root_exists(self):
        root = kinematic_footsteps_repo_root()
        self.assertTrue((root / "scripts" / "go2fp").is_dir())

    def test_syspath_idempotent(self):
        ensure_kinematic_footsteps_on_syspath()
        ensure_kinematic_footsteps_on_syspath()
        root = str(kinematic_footsteps_repo_root())
        self.assertIn(root, sys.path[:5])

    def test_trajectory_result_to_cache_broadcasts_planned_touchdown(self):
        ensure_kinematic_footsteps_on_syspath()
        from scripts.go2fp.types import TrajectoryResult

        n = 6
        tr = TrajectoryResult(
            root_pos_w=np.zeros((n, 3)),
            root_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
            root_lin_vel_w=np.zeros((n, 3)),
            root_ang_vel_w=np.zeros((n, 3)),
            joint_angles=np.zeros((n, 12)),
            foot_pos_w=np.zeros((n, 4, 3)),
            foot_pos_root=np.zeros((n, 4, 3)),
            contact_state=np.ones((n, 4), dtype=np.float32),
            body_pos_root=np.zeros((n, 12, 3)),
            planned_touchdown_w=np.arange(12.0, dtype=np.float64).reshape(4, 3),
        )
        cache = trajectory_result_to_reference_cache(tr)
        self.assertTrue(cache.is_ready())
        self.assertEqual(cache.planned_touchdown_w.shape, (n, 4, 3))
        self.assertTrue(torch.all(cache.planned_touchdown_w[0] == cache.planned_touchdown_w[-1]))

    def test_generate_standstill_cache(self):
        cache = generate_reference_cache_with_raw(command=(0.0, 0.0, 0.0), n_frames=9, dt=0.02)
        self.assertTrue(cache.is_ready())
        self.assertEqual(cache.root_pos_w.shape[0], 9)
        self.assertEqual(cache.contact_state.dtype, torch.bool)

    def test_generate_forward_command_non_trivial_motion(self):
        cache = generate_reference_cache_with_raw(command=(0.12, 0.0, 0.0), n_frames=30, dt=0.02)
        self.assertTrue(cache.is_ready())
        # Raw planner clamps to roughly one gait cycle; root x should drift when feasible.
        dx = float(cache.root_pos_w[-1, 0] - cache.root_pos_w[0, 0])
        self.assertGreater(abs(dx), 1e-6)


if __name__ == "__main__":
    unittest.main()
