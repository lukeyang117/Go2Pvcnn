"""L2 foothold vectorization baseline tests (spiral offsets, touchdown eval, dynamic N)."""

from __future__ import annotations

import math

import pytest
import torch


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


class TestSpiralOffsetsMeshgrid:
    def test_meshgrid_covers_all_loop_offsets(self):
        from extension.batched_planner.foothold import _precompute_spiral_offsets
        from scripts.go2fp.foothold import _spiral_square_offsets

        search_radius, grid_step = 0.15, 0.03
        n_max = max(int(math.floor(float(search_radius) / float(grid_step) + 1e-9)), 0)
        offsets = _precompute_spiral_offsets(search_radius, grid_step)

        assert offsets.dtype == torch.int64
        assert offsets.shape == ((2 * n_max + 1) ** 2, 2)

        expected_set = {tuple(int(x) for x in row) for row in _spiral_square_offsets(n_max)}
        actual_set = {tuple(int(x) for x in row.tolist()) for row in offsets}
        assert actual_set == expected_set

        d2 = offsets[:, 0] * offsets[:, 0] + offsets[:, 1] * offsets[:, 1]
        assert torch.all(d2[:-1] <= d2[1:]), "offsets must be sorted by non-decreasing squared distance"


class TestEvaluateTouchdownsNoItem:
    def test_feasibility_result_is_tensor(self):
        from extension.batched_planner.foothold import batched_evaluate_touchdowns

        terrain = BatchedSmoothTerrain()
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

        feasible, score, reason_codes = batched_evaluate_touchdowns(
            touchdown_pos,
            liftoff_pos,
            contact_seq,
            touchdown_mask,
            terrain,
            previous_footholds,
            max_reach=0.15,
        )
        assert isinstance(feasible, torch.Tensor) and feasible.dtype == torch.bool
        assert isinstance(score, torch.Tensor) and score.dtype == torch.float64
        assert not isinstance(feasible, list) and not isinstance(score, list)
        assert isinstance(reason_codes, torch.Tensor) and reason_codes.dtype == torch.int64
        assert not isinstance(reason_codes, list)
        assert tuple(reason_codes.shape) == (1,)


class TestFootholdDynamicN:
    @pytest.mark.parametrize("n_envs", [1, 4, 64])
    def test_footholds_dynamic_n(self, n_envs):
        from extension.batched_planner.foothold import batched_compute_footholds
        from scripts.go2fp.foothold import compute_hip_positions as raw_compute_hip_positions

        terrain = BatchedSmoothTerrain()
        torch.manual_seed(0)
        base_pos = torch.randn(n_envs, 3, dtype=torch.float64) * 0.02
        base_pos[:, 2] = 0.32
        base_yaw = torch.randn(n_envs, dtype=torch.float64) * 0.05
        base_lin_vel_xy = torch.randn(n_envs, 2, dtype=torch.float64) * 0.1
        ref_lin_vel_xy = torch.randn(n_envs, 2, dtype=torch.float64) * 0.1
        stance_time = torch.full((n_envs,), 0.28, dtype=torch.float64)
        com_height = torch.full((n_envs,), 0.33, dtype=torch.float64)
        touchdown_times = torch.rand(n_envs, 4, dtype=torch.float64) * 0.1 + 0.12
        previous_footholds = torch.randn(n_envs, 4, 3, dtype=torch.float64) * 0.02
        previous_footholds[..., 2] = -0.31
        yaw_rate = torch.randn(n_envs, dtype=torch.float64) * 0.05

        hip_list = [
            torch.as_tensor(
                raw_compute_hip_positions(base_pos[i].cpu().numpy(), float(base_yaw[i].item())),
                dtype=torch.float64,
            )
            for i in range(n_envs)
        ]
        hip_positions = torch.stack(hip_list, dim=0)

        out = batched_compute_footholds(
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
            yaw_rate=yaw_rate,
        )
        assert tuple(out.shape) == (n_envs, 4, 3)
        assert torch.isfinite(out).all()
