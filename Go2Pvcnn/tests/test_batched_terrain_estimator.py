import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.reference.raw_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()


def _sample_sequences(batch_size: int = 2, num_frames: int = 25):
    t = torch.arange(num_frames, dtype=torch.float64)

    base_pos = torch.zeros(batch_size, num_frames, 3, dtype=torch.float64)
    base_pos[:, :, 0] = 0.01 * t
    base_pos[:, :, 1] = -0.005 * t
    base_pos[:, :, 2] = 0.32 + 0.005 * torch.sin(0.2 * t)

    base_yaw = 0.03 * t.unsqueeze(0).expand(batch_size, -1)

    nominal = torch.tensor(
        [
            [0.22, 0.12, -0.30],
            [0.22, -0.12, -0.30],
            [-0.22, 0.12, -0.30],
            [-0.22, -0.12, -0.30],
        ],
        dtype=torch.float64,
    )
    gait_wave = 0.02 * torch.sin(0.25 * t).view(1, num_frames, 1, 1)
    lateral_wave = 0.01 * torch.cos(0.15 * t).view(1, num_frames, 1, 1)

    foot_pos = nominal.view(1, 1, 4, 3).expand(batch_size, num_frames, -1, -1).clone()
    foot_pos[..., 0] += gait_wave[..., 0]
    foot_pos[..., 1] += torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64).view(1, 1, 4) * lateral_wave[..., 0]
    foot_pos[..., 2] += 0.015 * torch.sin(0.35 * t).view(1, num_frames, 1)
    foot_pos = foot_pos + base_pos[:, :, None, :]

    if batch_size > 1:
        foot_pos[1] = foot_pos[1] + torch.tensor([0.01, -0.015, 0.0], dtype=torch.float64)
        base_pos[1, :, 2] += 0.01
        base_yaw[1] += 0.05

    return foot_pos, base_pos, base_yaw


class BatchedTerrainEstimatorTest(unittest.TestCase):
    def test_support_roll_pitch_matches_raw_support_plane_solver(self):
        from extension.batched_planner.terrain_estimator import batched_support_roll_pitch
        from scripts.go2fp.base_solver import solve_base_roll_pitch as raw_solve_base_roll_pitch

        foot_pos = torch.tensor(
            [
                [
                    [0.22, 0.12, -0.28],
                    [0.22, -0.12, -0.32],
                    [-0.22, 0.12, -0.26],
                    [-0.22, -0.12, -0.30],
                ],
                [
                    [0.20, 0.10, -0.31],
                    [0.20, -0.10, -0.29],
                    [-0.20, 0.10, -0.33],
                    [-0.20, -0.10, -0.31],
                ],
            ],
            dtype=torch.float64,
        )

        roll, pitch = batched_support_roll_pitch(foot_pos)

        for idx in range(int(foot_pos.shape[0])):
            expected_roll, expected_pitch = raw_solve_base_roll_pitch(foot_pos[idx].cpu().numpy())
            self.assertAlmostEqual(roll[idx].item(), expected_roll, places=7)
            self.assertAlmostEqual(pitch[idx].item(), expected_pitch, places=7)

    def test_estimate_terrain_matches_raw(self):
        from extension.batched_planner.terrain_estimator import batched_estimate_terrain
        from scripts.go2fp.terrain_estimator import estimate_terrain_batch as raw_estimate_terrain_batch

        foot_pos, base_pos, base_yaw = _sample_sequences(batch_size=1, num_frames=25)

        roll, pitch, height = batched_estimate_terrain(foot_pos, base_pos, base_yaw)
        expected = raw_estimate_terrain_batch(
            foot_pos[0].cpu().numpy(),
            base_pos[0].cpu().numpy(),
            base_yaw[0].cpu().numpy(),
        )

        self.assertEqual(tuple(roll.shape), (1, 25))
        self.assertEqual(tuple(pitch.shape), (1, 25))
        self.assertEqual(tuple(height.shape), (1, 25))
        torch.testing.assert_close(roll[0], torch.as_tensor(expected[0], dtype=roll.dtype), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(pitch[0], torch.as_tensor(expected[1], dtype=pitch.dtype), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(height[0], torch.as_tensor(expected[2], dtype=height.dtype), atol=1e-7, rtol=1e-7)

    def test_estimate_terrain_batch_consistency(self):
        from extension.batched_planner.terrain_estimator import batched_estimate_terrain
        from scripts.go2fp.terrain_estimator import estimate_terrain_batch as raw_estimate_terrain_batch

        foot_pos, base_pos, base_yaw = _sample_sequences(batch_size=2, num_frames=25)

        roll, pitch, height = batched_estimate_terrain(
            foot_pos,
            base_pos,
            base_yaw,
            alpha=0.1,
            initial_roll=torch.tensor([0.02, -0.03], dtype=torch.float64),
            initial_pitch=torch.tensor([-0.01, 0.04], dtype=torch.float64),
            initial_height=torch.tensor([0.02, -0.05], dtype=torch.float64),
        )

        for idx in range(2):
            expected = raw_estimate_terrain_batch(
                foot_pos[idx].cpu().numpy(),
                base_pos[idx].cpu().numpy(),
                base_yaw[idx].cpu().numpy(),
                alpha=0.1,
                initial_roll=float(torch.tensor([0.02, -0.03], dtype=torch.float64)[idx].item()),
                initial_pitch=float(torch.tensor([-0.01, 0.04], dtype=torch.float64)[idx].item()),
                initial_height=float(torch.tensor([0.02, -0.05], dtype=torch.float64)[idx].item()),
            )
            torch.testing.assert_close(roll[idx], torch.as_tensor(expected[0], dtype=roll.dtype), atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(pitch[idx], torch.as_tensor(expected[1], dtype=pitch.dtype), atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(height[idx], torch.as_tensor(expected[2], dtype=height.dtype), atol=1e-7, rtol=1e-7)

    def test_estimate_terrain_with_contact_state_ignores_swing_leg_lift(self):
        from extension.batched_planner.terrain_estimator import batched_estimate_terrain

        foot_pos = torch.tensor(
            [
                [
                    [
                        [0.22, 0.12, -0.18],
                        [0.22, -0.12, -0.30],
                        [-0.22, 0.12, -0.30],
                        [-0.22, -0.12, -0.18],
                    ]
                ]
            ],
            dtype=torch.float64,
        )
        base_pos = torch.zeros((1, 1, 3), dtype=torch.float64)
        base_yaw = torch.zeros((1, 1), dtype=torch.float64)
        contact_state = torch.tensor([[[0.0, 1.0, 1.0, 0.0]]], dtype=torch.float64)

        roll_free, pitch_free, height_free = batched_estimate_terrain(foot_pos, base_pos, base_yaw)
        roll_support, pitch_support, height_support = batched_estimate_terrain(
            foot_pos,
            base_pos,
            base_yaw,
            contact_state=contact_state,
        )

        self.assertGreater(abs(roll_free[0, 0].item()), 0.01)
        self.assertAlmostEqual(roll_support[0, 0].item(), 0.0, places=7)
        self.assertAlmostEqual(pitch_support[0, 0].item(), 0.0, places=7)
        self.assertAlmostEqual(height_support[0, 0].item(), -0.30, places=7)
        self.assertGreater(height_free[0, 0].item(), height_support[0, 0].item())


if __name__ == "__main__":
    unittest.main()
