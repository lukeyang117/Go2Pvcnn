import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.planner.runtime.raw_go2fp_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()


class BatchedSwingTest(unittest.TestCase):
    def test_compute_swing_targets_matches_raw(self):
        from extension.batched_planner.swing import batched_compute_swing_targets
        from scripts.go2fp.swing import compute_swing_targets as raw_compute_swing_targets

        contact_seq = torch.tensor(
            [[
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]],
            dtype=torch.float64,
        )
        lift_off_pos = torch.tensor(
            [[[0.22, 0.14, -0.31], [0.22, -0.14, -0.31], [-0.22, 0.14, -0.31], [-0.22, -0.14, -0.31]]],
            dtype=torch.float64,
        )
        touchdown_pos = torch.tensor(
            [[[0.28, 0.16, -0.29], [0.26, -0.13, -0.30], [-0.18, 0.15, -0.285], [-0.17, -0.12, -0.295]]],
            dtype=torch.float64,
        )
        terrain_max_heights = torch.tensor([[ -0.30, -0.305, -0.295, -0.298 ]], dtype=torch.float64)

        actual = batched_compute_swing_targets(
            contact_seq,
            lift_off_pos,
            touchdown_pos,
            step_height=0.08,
            terrain_max_heights=terrain_max_heights,
            clearance=0.02,
        )
        expected = raw_compute_swing_targets(
            contact_seq[0].cpu().numpy(),
            lift_off_pos[0].cpu().numpy(),
            touchdown_pos[0].cpu().numpy(),
            0.08,
            terrain_max_heights=terrain_max_heights[0].cpu().numpy(),
            clearance=0.02,
        )

        self.assertEqual(tuple(actual.shape), (1, contact_seq.shape[1], 4, 3))
        torch.testing.assert_close(actual[0], torch.as_tensor(expected, dtype=actual.dtype), atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
