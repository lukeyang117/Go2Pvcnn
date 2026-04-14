import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.reference.raw_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()


class RawPlaneTerrain:
    def __init__(self, ax: float, ay: float, bias: float):
        self.ax = float(ax)
        self.ay = float(ay)
        self.bias = float(bias)

    def height_at(self, x: float, y: float) -> float:
        return self.ax * float(x) + self.ay * float(y) + self.bias


class BatchedPlaneTerrain:
    def __init__(self, ax: float, ay: float, bias: float, batch_size: int):
        self.ax = float(ax)
        self.ay = float(ay)
        self.bias = float(bias)
        self.batch_size = int(batch_size)

    def height_at(self, points_xy):
        points = torch.as_tensor(points_xy, dtype=torch.float64)
        if points.ndim == 1:
            points = points.view(1, 2)
        if points.ndim == 2 and points.shape[-1] == 2:
            if points.shape[0] == self.batch_size:
                return self.ax * points[:, 0] + self.ay * points[:, 1] + self.bias
            points = points.unsqueeze(0)
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError(f"points_xy must have shape (2,), (N, 2), or (N, P, 2); got {tuple(points.shape)}")
        if points.shape[0] != self.batch_size:
            raise ValueError("terrain batch size mismatch")
        return self.ax * points[..., 0] + self.ay * points[..., 1] + self.bias


def _sample_base_inputs(num_frames: int = 12):
    t = torch.arange(num_frames, dtype=torch.float64)
    foot_targets = torch.zeros(1, num_frames, 4, 3, dtype=torch.float64)
    foot_targets[0, :, 0] = torch.stack([0.22 + 0.01 * torch.sin(0.2 * t), 0.12 + 0.01 * torch.cos(0.1 * t), -0.31 + 0.005 * torch.sin(0.15 * t)], dim=-1)
    foot_targets[0, :, 1] = torch.stack([0.22 + 0.01 * torch.sin(0.2 * t), -0.12 - 0.01 * torch.cos(0.1 * t), -0.31 + 0.005 * torch.cos(0.15 * t)], dim=-1)
    foot_targets[0, :, 2] = torch.stack([-0.22 + 0.01 * torch.cos(0.2 * t), 0.12 + 0.01 * torch.sin(0.1 * t), -0.31 - 0.004 * torch.sin(0.12 * t)], dim=-1)
    foot_targets[0, :, 3] = torch.stack([-0.22 + 0.01 * torch.cos(0.2 * t), -0.12 - 0.01 * torch.sin(0.1 * t), -0.31 - 0.004 * torch.cos(0.12 * t)], dim=-1)

    contact_seq = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)

    terrain_roll = 0.03 * torch.sin(0.15 * t).unsqueeze(0)
    terrain_pitch = -0.04 * torch.cos(0.2 * t).unsqueeze(0)
    terrain_height = (-0.31 + 0.01 * torch.sin(0.1 * t)).unsqueeze(0)
    return foot_targets, contact_seq, terrain_roll, terrain_pitch, terrain_height


class BatchedBaseSolverTest(unittest.TestCase):
    def test_integrate_base_planar_rejects_mixed_device_inputs_with_clear_error(self):
        from extension.batched_planner import base_solver

        initial_pos_xy = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        initial_yaw = torch.tensor([0.25], dtype=torch.float64, device="meta")
        vx = torch.tensor([0.4], dtype=torch.float64)
        vy = torch.tensor([-0.1], dtype=torch.float64)
        yaw_rate = torch.tensor([0.2], dtype=torch.float64)

        def fail_if_coerced(*args, **kwargs):
            raise AssertionError("mixed-device inputs should be rejected before coercion")

        with patch.object(base_solver, "_coerce_tensor", side_effect=fail_if_coerced):
            with self.assertRaisesRegex(ValueError, "mixed-device.*cpu.*meta"):
                base_solver.batched_integrate_base_planar(initial_pos_xy, initial_yaw, vx, vy, yaw_rate, n_frames=15, dt=0.02)

    def test_solve_base_trajectory_rejects_terrain_device_mismatch_without_sync(self):
        from extension.batched_planner import base_solver

        foot_targets, contact_seq, terrain_roll, terrain_pitch, terrain_height = _sample_base_inputs()

        terrain = SimpleNamespace(
            heightmaps=torch.empty((1, 1, 2, 2), dtype=torch.float64, device="meta"),
            height_at=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("terrain should not be sampled after a device mismatch")),
        )

        with self.assertRaisesRegex(ValueError, "terrain and tensor inputs on the same device"):
            base_solver.batched_solve_base_trajectory(
                initial_pos=torch.tensor([[0.0, 0.0, 0.34]], dtype=torch.float64),
                initial_yaw=torch.tensor([0.1], dtype=torch.float64),
                vx=torch.tensor([0.35], dtype=torch.float64),
                vy=torch.tensor([-0.05], dtype=torch.float64),
                yaw_rate=torch.tensor([0.12], dtype=torch.float64),
                n_frames=foot_targets.shape[1],
                dt=0.02,
                terrain=terrain,
                foot_targets=foot_targets,
                contact_seq=contact_seq,
                terrain_roll=terrain_roll,
                terrain_pitch=terrain_pitch,
                terrain_height=terrain_height,
            )

    def test_integrate_base_planar_matches_raw(self):
        from extension.batched_planner.base_solver import batched_integrate_base_planar
        from scripts.go2fp.base_solver import integrate_base_planar as raw_integrate_base_planar

        initial_pos_xy = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        initial_yaw = torch.tensor([0.25], dtype=torch.float64)
        vx = torch.tensor([0.4], dtype=torch.float64)
        vy = torch.tensor([-0.1], dtype=torch.float64)
        yaw_rate = torch.tensor([0.2], dtype=torch.float64)

        pos_xy, yaw = batched_integrate_base_planar(initial_pos_xy, initial_yaw, vx, vy, yaw_rate, n_frames=15, dt=0.02)
        expected_pos_xy, expected_yaw = raw_integrate_base_planar(
            initial_pos_xy[0].cpu().numpy(),
            float(initial_yaw[0].item()),
            float(vx[0].item()),
            float(vy[0].item()),
            float(yaw_rate[0].item()),
            15,
            0.02,
        )

        self.assertEqual(tuple(pos_xy.shape), (1, 15, 2))
        self.assertEqual(tuple(yaw.shape), (1, 15))
        torch.testing.assert_close(pos_xy[0], torch.as_tensor(expected_pos_xy, dtype=pos_xy.dtype), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(yaw[0], torch.as_tensor(expected_yaw, dtype=yaw.dtype), atol=1e-7, rtol=1e-7)

    def test_solve_base_trajectory_outputs_canonical_float64_tensors(self):
        from extension.batched_planner.base_solver import batched_solve_base_trajectory

        foot_targets, contact_seq, terrain_roll, terrain_pitch, terrain_height = _sample_base_inputs()
        batched_terrain = BatchedPlaneTerrain(ax=0.02, ay=-0.01, bias=-0.33, batch_size=1)

        root_pos, root_quat = batched_solve_base_trajectory(
            initial_pos=torch.tensor([[0.0, 0.0, 0.34]], dtype=torch.float64),
            initial_yaw=torch.tensor([0.1], dtype=torch.float64),
            vx=torch.tensor([0.35], dtype=torch.float64),
            vy=torch.tensor([-0.05], dtype=torch.float64),
            yaw_rate=torch.tensor([0.12], dtype=torch.float64),
            n_frames=foot_targets.shape[1],
            dt=0.02,
            terrain=batched_terrain,
            foot_targets=foot_targets,
            contact_seq=contact_seq,
            terrain_roll=terrain_roll,
            terrain_pitch=terrain_pitch,
            terrain_height=terrain_height,
        )

        self.assertEqual(root_pos.dtype, torch.float64)
        self.assertEqual(root_quat.dtype, torch.float64)
        self.assertEqual(root_pos.device.type, "cpu")
        self.assertEqual(root_quat.device.type, "cpu")

    def test_solve_base_trajectory_matches_raw(self):
        from extension.batched_planner.base_solver import batched_solve_base_trajectory
        from scripts.go2fp.base_solver import solve_base_trajectory as raw_solve_base_trajectory

        foot_targets, contact_seq, terrain_roll, terrain_pitch, terrain_height = _sample_base_inputs()
        raw_terrain = RawPlaneTerrain(ax=0.02, ay=-0.01, bias=-0.33)
        batched_terrain = BatchedPlaneTerrain(ax=0.02, ay=-0.01, bias=-0.33, batch_size=1)

        root_pos, root_quat = batched_solve_base_trajectory(
            initial_pos=torch.tensor([[0.0, 0.0, 0.34]], dtype=torch.float64),
            initial_yaw=torch.tensor([0.1], dtype=torch.float64),
            vx=torch.tensor([0.35], dtype=torch.float64),
            vy=torch.tensor([-0.05], dtype=torch.float64),
            yaw_rate=torch.tensor([0.12], dtype=torch.float64),
            n_frames=foot_targets.shape[1],
            dt=0.02,
            terrain=batched_terrain,
            foot_targets=foot_targets,
            contact_seq=contact_seq,
            terrain_roll=terrain_roll,
            terrain_pitch=terrain_pitch,
            terrain_height=terrain_height,
        )

        expected_pos, expected_quat = raw_solve_base_trajectory(
            initial_pos=torch.tensor([0.0, 0.0, 0.34], dtype=torch.float64).cpu().numpy(),
            initial_yaw=0.1,
            vx=0.35,
            vy=-0.05,
            yaw_rate=0.12,
            n_frames=foot_targets.shape[1],
            dt=0.02,
            terrain=raw_terrain,
            foot_targets=foot_targets[0].cpu().numpy(),
            contact_seq=contact_seq[0].cpu().numpy(),
            terrain_roll=terrain_roll[0].cpu().numpy(),
            terrain_pitch=terrain_pitch[0].cpu().numpy(),
            terrain_height=terrain_height[0].cpu().numpy(),
        )

        self.assertEqual(tuple(root_pos.shape), (1, foot_targets.shape[1], 3))
        self.assertEqual(tuple(root_quat.shape), (1, foot_targets.shape[1], 4))
        torch.testing.assert_close(root_pos[0], torch.as_tensor(expected_pos, dtype=root_pos.dtype), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(root_quat[0], torch.as_tensor(expected_quat, dtype=root_quat.dtype), atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
