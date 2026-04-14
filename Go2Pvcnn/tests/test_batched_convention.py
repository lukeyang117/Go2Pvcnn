import math
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class BatchedConventionTest(unittest.TestCase):
    def test_batched_trajectory_config_matches_approved_defaults(self):
        from Go2Pvcnn.extension.batched_planner.config import BatchedTrajectoryConfig

        cfg = BatchedTrajectoryConfig()
        self.assertEqual(cfg.duty_factor, 0.6)
        self.assertEqual(cfg.max_roughness, 0.5)
        self.assertEqual(cfg.max_touchdown_xy_reach, 0.15)
        self.assertEqual(cfg.replan_stop_speed, 0.05)
        cfg.replan_stop_speed = 0.07
        self.assertEqual(cfg.replan_stop_speed, 0.07)

    def test_compare_trajectories_builds_matching_configs_without_recovery_fields(self):
        from Go2Pvcnn.extension.viz.compare_trajectories import _build_matching_configs

        batched_cfg, raw_cfg = _build_matching_configs()

        self.assertEqual(raw_cfg.replan_stop_speed, batched_cfg.replan_stop_speed)
        self.assertFalse(hasattr(batched_cfg, "replan_velocity_scales"))

    def test_compare_trajectories_import_is_lazy_about_raw_bridge_setup(self):
        module_name = "Go2Pvcnn.extension.viz.compare_trajectories"
        go2_root = REPO_ROOT / "Go2Pvcnn"
        if str(go2_root) not in sys.path:
            sys.path.insert(0, str(go2_root))
        sys.modules.pop(module_name, None)

        with patch("extension.reference.raw_bridge.ensure_kinematic_footsteps_on_syspath", side_effect=AssertionError("eager raw bridge setup")):
            module = importlib.import_module(module_name)

        self.assertIsNotNone(module)

    def test_viewer_planner_config_builder_uses_single_shot_fields(self):
        from Go2Pvcnn.extension.viz.go2_foostep_planner import _build_planner_cfg

        env_cfg = SimpleNamespace(
            gait_name="trot",
            step_freq=2.0,
            duty_factor=0.6,
            step_height=0.08,
            foothold_search_radius=0.15,
            foothold_search_step=0.03,
            max_step_down=0.10,
            max_roughness=0.5,
            replan_stop_speed=0.05,
        )

        planner_cfg = _build_planner_cfg(env_cfg)

        self.assertEqual(planner_cfg.replan_stop_speed, env_cfg.replan_stop_speed)
        self.assertFalse(hasattr(planner_cfg, "replan_velocity_scales"))

    def test_hip_offsets_array_matches_raw(self):
        from Go2Pvcnn.extension.batched_planner.types import HIP_OFFSETS_ARRAY
        from Go2Pvcnn.extension.reference.raw_bridge import ensure_kinematic_footsteps_on_syspath

        ensure_kinematic_footsteps_on_syspath()
        from scripts.go2fp.types import HIP_OFFSETS_ARRAY as RAW_HIP_OFFSETS_ARRAY

        self.assertEqual(HIP_OFFSETS_ARRAY.shape, RAW_HIP_OFFSETS_ARRAY.shape)
        torch.testing.assert_close(
            HIP_OFFSETS_ARRAY,
            torch.as_tensor(RAW_HIP_OFFSETS_ARRAY, dtype=HIP_OFFSETS_ARRAY.dtype),
        )

    def test_quat_wxyz_xyzw_roundtrip(self):
        from Go2Pvcnn.extension.convention import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

        q_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0]])
        q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
        self.assertEqual(q_xyzw.shape, (2, 4))
        self.assertAlmostEqual(q_xyzw[0, 3].item(), 1.0, places=5)

        q_back = quat_xyzw_to_wxyz(q_xyzw)
        torch.testing.assert_close(q_back, q_wxyz)

    def test_isaac_state_to_planner_state_converts_quaternion_order(self):
        from Go2Pvcnn.extension.convention import isaac_state_to_planner_state

        state = isaac_state_to_planner_state(
            root_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
            root_quat_xyzw=torch.tensor([[0.0, 0.0, 0.70710678, 0.70710678]]),
            joint_pos=torch.zeros((1, 12)),
            foot_pos_w=torch.zeros((1, 4, 3)),
        )

        self.assertEqual(tuple(state.root_pos.shape), (1, 3))
        self.assertEqual(tuple(state.root_quat.shape), (1, 4))
        torch.testing.assert_close(
            state.root_quat,
            torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]]),
        )
        self.assertEqual(tuple(state.foot_vel.shape), (1, 4, 3))
        torch.testing.assert_close(state.foot_vel, torch.zeros((1, 4, 3)))
        torch.testing.assert_close(state.root_pos, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_batched_robot_state_normalizes_omitted_foot_vel(self):
        from Go2Pvcnn.extension.batched_planner.types import BatchedRobotState

        state = BatchedRobotState(
            root_pos=torch.zeros((2, 3)),
            root_quat=torch.zeros((2, 4)),
            joint_angles=torch.zeros((2, 12)),
            foot_pos=torch.zeros((2, 4, 3)),
        )

        self.assertEqual(tuple(state.foot_vel.shape), (2, 4, 3))
        torch.testing.assert_close(state.foot_vel, torch.zeros((2, 4, 3)))

    def test_planner_result_to_reference_cache_wraps_batched_result(self):
        from Go2Pvcnn.extension.batched_planner.types import BatchedTrajectoryResult
        from Go2Pvcnn.extension.convention import planner_result_to_reference_cache

        result = BatchedTrajectoryResult(
            num_frames=3,
            root_pos_w=torch.zeros((2, 3, 3)),
            root_quat_w=torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.0, 0.0, 0.70710678], [1.0, 0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.0, 0.0, 0.70710678], [1.0, 0.0, 0.0, 0.0]],
                ]
            ),
            root_lin_vel_w=torch.zeros((2, 3, 3)),
            root_ang_vel_w=torch.zeros((2, 3, 3)),
            joint_angles=torch.zeros((2, 3, 12)),
            foot_pos_w=torch.zeros((2, 3, 4, 3)),
            foot_pos_root=torch.zeros((2, 3, 4, 3)),
            contact_state=torch.ones((2, 3, 4), dtype=torch.bool),
            body_pos_root=torch.zeros((2, 3, 12, 3)),
            planned_touchdown_w=torch.zeros((2, 4, 3)),
        )

        cache = planner_result_to_reference_cache(result)
        self.assertTrue(cache.is_ready())
        self.assertEqual(tuple(cache.root_pos_w.shape), (2, 3, 3))
        self.assertEqual(tuple(cache.phase_index.shape), (2, 3))
        self.assertEqual(tuple(cache.planned_touchdown_w.shape), (2, 3, 4, 3))
        self.assertTrue(torch.all(cache.valid_mask))
        torch.testing.assert_close(cache.planned_touchdown_w[:, 0], result.planned_touchdown_w)

    def test_planner_result_to_reference_cache_rejects_num_frames_mismatch(self):
        from Go2Pvcnn.extension.batched_planner.types import BatchedTrajectoryResult
        from Go2Pvcnn.extension.convention import planner_result_to_reference_cache

        result = BatchedTrajectoryResult(
            num_frames=4,
            root_pos_w=torch.zeros((2, 3, 3)),
            root_quat_w=torch.zeros((2, 3, 4)),
            root_lin_vel_w=torch.zeros((2, 3, 3)),
            root_ang_vel_w=torch.zeros((2, 3, 3)),
            joint_angles=torch.zeros((2, 3, 12)),
            foot_pos_w=torch.zeros((2, 3, 4, 3)),
            foot_pos_root=torch.zeros((2, 3, 4, 3)),
            contact_state=torch.ones((2, 3, 4), dtype=torch.bool),
            body_pos_root=torch.zeros((2, 3, 12, 3)),
            planned_touchdown_w=torch.zeros((2, 4, 3)),
        )

        with self.assertRaisesRegex(ValueError, "num_frames"):
            planner_result_to_reference_cache(result)

    def test_extract_yaw_batch_matches_raw_formula(self):
        from Go2Pvcnn.extension.convention import extract_yaw_batch

        quat = torch.tensor(
            [
                [0.92387953, 0.0, 0.0, 0.38268343],
                [0.70710678, 0.0, 0.0, 0.70710678],
            ],
            dtype=torch.float64,
        )
        yaw = extract_yaw_batch(quat)
        self.assertAlmostEqual(yaw[0].item(), math.pi / 4.0, places=6)
        self.assertAlmostEqual(yaw[1].item(), math.pi / 2.0, places=6)

    def test_extract_roll_pitch_batch_matches_known_rotations(self):
        from Go2Pvcnn.extension.convention import extract_roll_pitch_batch

        quat = torch.tensor(
            [
                [0.92387953, 0.38268343, 0.0, 0.0],
                [0.96592583, 0.0, 0.25881905, 0.0],
            ],
            dtype=torch.float64,
        )
        roll, pitch = extract_roll_pitch_batch(quat)
        self.assertAlmostEqual(roll[0].item(), math.pi / 4.0, places=6)
        self.assertAlmostEqual(pitch[1].item(), math.pi / 6.0, places=6)

    def test_euler_to_quat_batch_roundtrip_known_angles(self):
        from Go2Pvcnn.extension.convention import euler_to_quat_batch, extract_roll_pitch_batch, extract_yaw_batch

        roll = torch.tensor([0.0, math.pi / 6.0], dtype=torch.float64)
        pitch = torch.tensor([0.0, -math.pi / 8.0], dtype=torch.float64)
        yaw = torch.tensor([math.pi / 4.0, -math.pi / 3.0], dtype=torch.float64)
        quat = euler_to_quat_batch(roll, pitch, yaw)
        self.assertEqual(tuple(quat.shape), (2, 4))
        roll_out, pitch_out = extract_roll_pitch_batch(quat)
        yaw_out = extract_yaw_batch(quat)
        torch.testing.assert_close(roll_out, roll, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(pitch_out, pitch, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(yaw_out, yaw, atol=1e-6, rtol=1e-6)

    def test_yaw_rotation_matrix_batch_rotates_xy_plane(self):
        from Go2Pvcnn.extension.convention import yaw_rotation_matrix_batch

        yaw = torch.tensor([math.pi / 2.0], dtype=torch.float64)
        rot = yaw_rotation_matrix_batch(yaw)
        self.assertEqual(tuple(rot.shape), (1, 3, 3))
        v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        rotated = torch.einsum("bij,bj->bi", rot, v)
        torch.testing.assert_close(rotated, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
