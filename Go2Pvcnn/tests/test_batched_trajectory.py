import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.planner.runtime.raw_go2fp_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()


class FlatTerrain:
    def height_at(self, x, y=None):
        if y is None:
            points = torch.as_tensor(x, dtype=torch.float64)
            return torch.zeros(points.shape[:-1], dtype=torch.float64, device=points.device)
        return 0.0

    def roughness_at(self, x, y=None):
        if y is None:
            points = torch.as_tensor(x, dtype=torch.float64)
            return torch.zeros(points.shape[:-1], dtype=torch.float64, device=points.device)
        return 0.0

    def max_height_along_segment(self, p0, p1=None):
        if p1 is None:
            p0_t = torch.as_tensor(p0, dtype=torch.float64)
            return torch.zeros(p0_t.shape[0], dtype=torch.float64, device=p0_t.device)
        if not isinstance(p0, tuple):
            p0_t = torch.as_tensor(p0, dtype=torch.float64)
            return torch.zeros(p0_t.shape[0], dtype=torch.float64, device=p0_t.device)
        return 0.0


class BatchedTrajectoryTest(unittest.TestCase):
    def _default_batched_state(self):
        from extension.batched_planner.types import BatchedRobotState
        from scripts.go2fp.trajectory import default_initial_state

        raw_state = default_initial_state(None, x=0.0, y=0.0)
        return BatchedRobotState(
            root_pos=torch.as_tensor(raw_state.root_pos, dtype=torch.float64).unsqueeze(0),
            root_quat=torch.as_tensor(raw_state.root_quat, dtype=torch.float64).unsqueeze(0),
            joint_angles=torch.as_tensor(raw_state.joint_angles, dtype=torch.float64).unsqueeze(0),
            foot_pos=torch.as_tensor(raw_state.foot_pos, dtype=torch.float64).unsqueeze(0),
        ), raw_state

    def test_standstill_zero_command_matches_raw(self):
        from extension.batched_planner.trajectory import batched_generate_trajectory
        from scripts.go2fp.trajectory import generate_trajectory
        from scripts.go2fp.types import Command

        state, raw_state = self._default_batched_state()
        cmd = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)

        actual = batched_generate_trajectory(FlatTerrain(), state, cmd, requested_n_frames=12, dt=0.02)
        expected = generate_trajectory(FlatTerrain(), raw_state, Command(0.0, 0.0, 0.0), 12, dt=0.02)

        self.assertEqual(actual.num_frames, 12)
        torch.testing.assert_close(actual.root_pos_w[0], torch.as_tensor(expected.root_pos_w, dtype=actual.root_pos_w.dtype))
        torch.testing.assert_close(actual.root_quat_w[0], torch.as_tensor(expected.root_quat_w, dtype=actual.root_quat_w.dtype))
        torch.testing.assert_close(actual.contact_state[0], torch.as_tensor(expected.contact_state, dtype=actual.contact_state.dtype))
        torch.testing.assert_close(actual.planned_touchdown_w[0], torch.as_tensor(expected.planned_touchdown_w, dtype=actual.planned_touchdown_w.dtype))

    def test_standstill_below_stop_speed_falls_back_to_standstill(self):
        from extension.batched_planner.config import BatchedTrajectoryConfig
        from extension.batched_planner.trajectory import batched_generate_trajectory

        state, _ = self._default_batched_state()
        cmd = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64)
        cfg = BatchedTrajectoryConfig(replan_stop_speed=0.05)

        actual = batched_generate_trajectory(FlatTerrain(), state, cmd, requested_n_frames=10, dt=0.02, cfg=cfg)

        self.assertEqual(actual.num_frames, 10)
        self.assertTrue(torch.all(actual.contact_state[0] == 1.0))
        self.assertTrue(torch.allclose(actual.root_pos_w[0], actual.root_pos_w[0, :1].expand_as(actual.root_pos_w[0])))
        self.assertTrue(torch.allclose(actual.root_quat_w[0], actual.root_quat_w[0, :1].expand_as(actual.root_quat_w[0])))

    def test_horizon_truncation_clamps_to_cycle_frames(self):
        from extension.batched_planner.config import BatchedTrajectoryConfig
        from extension.batched_planner.trajectory import batched_generate_trajectory

        state, _ = self._default_batched_state()
        cmd = torch.tensor([[0.4, 0.0, 0.2]], dtype=torch.float64)
        cfg = BatchedTrajectoryConfig(step_freq=2.0, duty_factor=0.6)

        actual = batched_generate_trajectory(FlatTerrain(), state, cmd, requested_n_frames=100, dt=0.02, cfg=cfg)

        self.assertEqual(actual.num_frames, 25)
        self.assertEqual(tuple(actual.root_pos_w.shape), (1, 25, 3))
        self.assertEqual(tuple(actual.contact_state.shape), (1, 25, 4))

    def test_motion_trajectory_matches_raw_single_env(self):
        from extension.batched_planner.config import BatchedTrajectoryConfig
        from extension.batched_planner.trajectory import batched_generate_trajectory
        from scripts.go2fp.config import TrajectoryConfig
        from scripts.go2fp.trajectory import generate_trajectory
        from scripts.go2fp.types import Command

        state, raw_state = self._default_batched_state()
        cmd = torch.tensor([[0.35, 0.0, 0.1]], dtype=torch.float64)
        cfg = BatchedTrajectoryConfig(step_freq=2.0, duty_factor=0.6)

        actual = batched_generate_trajectory(FlatTerrain(), state, cmd, requested_n_frames=20, dt=0.02, cfg=cfg)
        expected = generate_trajectory(
            FlatTerrain(),
            raw_state,
            Command(0.35, 0.0, 0.1),
            20,
            dt=0.02,
            config=TrajectoryConfig(
                gait_name=cfg.gait_name,
                step_freq=cfg.step_freq,
                duty_factor=cfg.duty_factor,
                step_height=cfg.step_height,
                hip_height=cfg.hip_height,
                body_clearance_margin=cfg.body_clearance_margin,
                foothold_search_radius=cfg.foothold_search_radius,
                foothold_search_step=cfg.foothold_search_step,
                max_foothold_step_down=cfg.max_foothold_step_down,
                max_touchdown_xy_reach=cfg.max_touchdown_xy_reach,
                replan_stop_speed=cfg.replan_stop_speed,
                replan_velocity_scales=tuple(cfg.replan_velocity_scales),
                replan_yaw_biases=tuple(cfg.replan_yaw_biases),
                replan_vy_biases=tuple(cfg.replan_vy_biases),
            ),
        )

        self.assertEqual(actual.num_frames, expected.root_pos_w.shape[0])
        torch.testing.assert_close(actual.root_pos_w[0], torch.as_tensor(expected.root_pos_w, dtype=actual.root_pos_w.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual.root_quat_w[0], torch.as_tensor(expected.root_quat_w, dtype=actual.root_quat_w.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual.joint_angles[0], torch.as_tensor(expected.joint_angles, dtype=actual.joint_angles.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual.foot_pos_w[0], torch.as_tensor(expected.foot_pos_w, dtype=actual.foot_pos_w.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual.contact_state[0], torch.as_tensor(expected.contact_state, dtype=actual.contact_state.dtype))
        torch.testing.assert_close(actual.planned_touchdown_w[0], torch.as_tensor(expected.planned_touchdown_w, dtype=actual.planned_touchdown_w.dtype), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
