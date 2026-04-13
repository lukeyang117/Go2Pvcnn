import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.planner.runtime.raw_go2fp_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()


class RawSmoothTerrain:
    def height_at(self, x: float, y: float) -> float:
        return 0.02 * float(x) - 0.01 * float(y) - 0.3

    def roughness_at(self, x: float, y: float) -> float:
        return 0.0


class BatchedSmoothTerrain:
    def height_at(self, points_xy):
        points = torch.as_tensor(points_xy, dtype=torch.float64)
        if points.ndim == 1:
            points = points.view(1, 2)
        if points.ndim == 2:
            return 0.02 * points[:, 0] - 0.01 * points[:, 1] - 0.3
        return 0.02 * points[..., 0] - 0.01 * points[..., 1] - 0.3

    def roughness_at(self, points_xy):
        points = torch.as_tensor(points_xy)
        return torch.zeros(points.shape[:-1], dtype=torch.float64, device=points.device)


class BatchedFootholdTest(unittest.TestCase):
    def test_precompute_spiral_offsets_matches_raw(self):
        from extension.batched_planner.foothold import _precompute_spiral_offsets
        from scripts.go2fp.foothold import _spiral_square_offsets

        actual = _precompute_spiral_offsets(0.15, 0.03)
        expected = torch.tensor(list(_spiral_square_offsets(5)), dtype=torch.int64)

        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.dtype, torch.int64)
        torch.testing.assert_close(actual, expected)

    def test_batched_compute_footholds_matches_raw(self):
        from extension.batched_planner.foothold import batched_compute_footholds
        from scripts.go2fp.foothold import compute_footholds as raw_compute_footholds
        from scripts.go2fp.foothold import compute_hip_positions as raw_compute_hip_positions

        terrain = BatchedSmoothTerrain()
        raw_terrain = RawSmoothTerrain()
        base_pos = torch.tensor([[0.0, 0.0, 0.32]], dtype=torch.float64)
        base_yaw = torch.tensor([0.15], dtype=torch.float64)
        base_lin_vel_xy = torch.tensor([[0.25, -0.05]], dtype=torch.float64)
        ref_lin_vel_xy = torch.tensor([[0.3, -0.02]], dtype=torch.float64)
        stance_time = torch.tensor([0.28], dtype=torch.float64)
        com_height = torch.tensor([0.33], dtype=torch.float64)
        touchdown_times = torch.tensor([[0.12, 0.18, 0.15, 0.2]], dtype=torch.float64)
        previous_footholds = torch.tensor(
            [[[0.23, 0.14, -0.31], [0.23, -0.14, -0.31], [-0.23, 0.14, -0.31], [-0.23, -0.14, -0.31]]],
            dtype=torch.float64,
        )
        hip_positions = torch.as_tensor(
            raw_compute_hip_positions(base_pos[0].cpu().numpy(), float(base_yaw[0].item())),
            dtype=torch.float64,
        ).unsqueeze(0)

        actual = batched_compute_footholds(
            base_pos=base_pos,
            base_yaw=base_yaw,
            base_lin_vel_xy=base_lin_vel_xy,
            ref_lin_vel_xy=ref_lin_vel_xy,
            hip_positions=hip_positions,
            stance_time=stance_time,
            com_height=com_height,
            terrain=terrain,
            previous_footholds=previous_footholds,
            touchdown_times=touchdown_times,
            yaw_rate=torch.tensor([0.1], dtype=torch.float64),
        )
        expected = raw_compute_footholds(
            base_pos=base_pos[0].cpu().numpy(),
            base_yaw=float(base_yaw[0].item()),
            base_lin_vel_xy=base_lin_vel_xy[0].cpu().numpy(),
            ref_lin_vel_xy=ref_lin_vel_xy[0].cpu().numpy(),
            hip_positions=hip_positions[0].cpu().numpy(),
            stance_time=float(stance_time[0].item()),
            com_height=float(com_height[0].item()),
            terrain=raw_terrain,
            previous_footholds=previous_footholds[0].cpu().numpy(),
            touchdown_times=touchdown_times[0].cpu().numpy(),
            yaw_rate=0.1,
        )

        self.assertEqual(tuple(actual.shape), (1, 4, 3))
        torch.testing.assert_close(actual[0], torch.as_tensor(expected, dtype=actual.dtype), atol=1e-7, rtol=1e-7)

    def test_batched_evaluate_touchdowns_and_candidate_score_match_raw(self):
        from extension.batched_planner.foothold import batched_candidate_total_score, batched_evaluate_touchdowns
        from scripts.go2fp.foothold import evaluate_touchdown_set as raw_evaluate_touchdown_set
        from scripts.go2fp.trajectory import _candidate_total_score as raw_candidate_total_score
        from scripts.go2fp.types import Command

        terrain = BatchedSmoothTerrain()
        raw_terrain = RawSmoothTerrain()
        touchdown_pos = torch.tensor(
            [[[0.24, 0.14, -0.31], [0.23, -0.13, -0.315], [-0.22, 0.15, -0.305], [-0.23, -0.14, -0.31]]],
            dtype=torch.float64,
        )
        liftoff_pos = torch.tensor(
            [[[0.21, 0.12, -0.31], [0.23, -0.14, -0.31], [-0.20, 0.13, -0.31], [-0.23, -0.14, -0.31]]],
            dtype=torch.float64,
        )
        previous_footholds = torch.tensor(
            [[[0.22, 0.13, -0.30], [0.23, -0.14, -0.31], [-0.21, 0.13, -0.30], [-0.23, -0.14, -0.31]]],
            dtype=torch.float64,
        )
        contact_seq = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            dtype=torch.float64,
        )
        touchdown_mask = torch.tensor([[True, False, True, False]])

        feasible, score, reason = batched_evaluate_touchdowns(
            touchdown_pos,
            liftoff_pos,
            contact_seq,
            touchdown_mask,
            terrain,
            previous_footholds,
            max_reach=0.15,
        )
        expected = raw_evaluate_touchdown_set(
            touchdowns=touchdown_pos[0].cpu().numpy(),
            lift_offs=liftoff_pos[0].cpu().numpy(),
            contact_seq=contact_seq[0].cpu().numpy(),
            touchdown_mask=touchdown_mask[0].cpu().numpy(),
            terrain=raw_terrain,
            previous_footholds=previous_footholds[0].cpu().numpy(),
            max_touchdown_xy_reach=0.15,
        )

        self.assertEqual(tuple(feasible.shape), (1,))
        self.assertEqual(tuple(score.shape), (1,))
        self.assertEqual(reason, [expected.reason])
        self.assertEqual(bool(feasible[0].item()), expected.feasible)
        self.assertAlmostEqual(float(score[0].item()), float(expected.score), places=7)

        total_score = batched_candidate_total_score(
            original_cmd=torch.tensor([[0.3, -0.02, 0.1]], dtype=torch.float64),
            candidate_cmd=torch.tensor([[0.24, 0.03, 0.08]], dtype=torch.float64),
            touchdown_scores=score,
            candidate_indices=torch.tensor([2], dtype=torch.int64),
        )
        expected_total = raw_candidate_total_score(
            Command(0.3, -0.02, 0.1),
            Command(0.24, 0.03, 0.08),
            float(expected.score),
            2,
        )
        self.assertEqual(tuple(total_score.shape), (1,))
        self.assertAlmostEqual(float(total_score[0].item()), float(expected_total), places=7)


if __name__ == "__main__":
    unittest.main()
