import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fake_result(num_envs: int, num_frames: int):
    from extension.batched_planner.types import BatchedTrajectoryResult

    shape_env_frame = (num_envs, num_frames)
    root_pos = torch.arange(num_envs * num_frames * 3, dtype=torch.float64).reshape(num_envs, num_frames, 3)
    root_quat = torch.zeros(num_envs, num_frames, 4, dtype=torch.float64)
    root_quat[..., 0] = 1.0
    joint_angles = torch.arange(num_envs * num_frames * 12, dtype=torch.float64).reshape(num_envs, num_frames, 12)
    foot_pos_root = torch.arange(num_envs * num_frames * 4 * 3, dtype=torch.float64).reshape(num_envs, num_frames, 4, 3)
    contact_state = torch.ones(num_envs, num_frames, 4, dtype=torch.float32)
    body_pos_root = torch.arange(num_envs * num_frames * 12 * 3, dtype=torch.float64).reshape(num_envs, num_frames, 12, 3)
    touchdown = torch.arange(num_envs * 4 * 3, dtype=torch.float64).reshape(num_envs, 4, 3)
    zeros = torch.zeros(num_envs, num_frames, 3, dtype=torch.float64)
    return BatchedTrajectoryResult(
        num_frames=num_frames,
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        root_lin_vel_w=zeros.clone(),
        root_ang_vel_w=zeros.clone(),
        joint_angles=joint_angles,
        foot_pos_w=foot_pos_root.clone(),
        foot_pos_root=foot_pos_root,
        contact_state=contact_state,
        body_pos_root=body_pos_root,
        planned_touchdown_w=touchdown,
    )


class BatchedManagerTest(unittest.TestCase):
    def _cfg(self):
        return SimpleNamespace(reference_replan_interval_steps=3, reference_trajectory_horizon=5, dt=0.02)

    def test_replan_at_interval(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = self._cfg()
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        states = SimpleNamespace(root_pos=torch.zeros(2, 3, dtype=torch.float64))
        commands = torch.zeros(2, 3, dtype=torch.float64)

        with patch("extension.batched_planner.manager.batched_generate_trajectory", return_value=_fake_result(2, 5)) as gen:
            for _ in range(7):
                manager.step("terrain", states, commands)
            self.assertEqual(gen.call_count, 3)

    def test_phase_counter_advances_and_clamps(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = self._cfg()
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        states = SimpleNamespace(root_pos=torch.zeros(2, 3, dtype=torch.float64))
        commands = torch.zeros(2, 3, dtype=torch.float64)

        with patch("extension.batched_planner.manager.batched_generate_trajectory", return_value=_fake_result(2, 4)):
            for _ in range(6):
                manager.step("terrain", states, commands)
        self.assertTrue(torch.equal(manager._phase_counter, torch.tensor([3, 3], dtype=torch.long)))

    def test_reset_resets_phase_only(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = self._cfg()
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        states = SimpleNamespace(root_pos=torch.zeros(3, 3, dtype=torch.float64))
        commands = torch.zeros(3, 3, dtype=torch.float64)

        with patch("extension.batched_planner.manager.batched_generate_trajectory", return_value=_fake_result(3, 5)):
            manager.step("terrain", states, commands)
            manager.step("terrain", states, commands)
        manager.reset_envs(torch.tensor([False, True, False]))
        self.assertEqual(manager._step_counter, 2)
        self.assertTrue(torch.equal(manager._phase_counter, torch.tensor([2, 0, 2], dtype=torch.long)))

    def test_current_reference_shape(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = self._cfg()
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        states = SimpleNamespace(root_pos=torch.zeros(2, 3, dtype=torch.float64))
        commands = torch.zeros(2, 3, dtype=torch.float64)

        with patch("extension.batched_planner.manager.batched_generate_trajectory", return_value=_fake_result(2, 5)):
            manager.step("terrain", states, commands)
            manager.step("terrain", states, commands)
        ref = manager.current_reference()
        self.assertEqual(tuple(ref["root_pos_w"].shape), (2, 3))
        self.assertEqual(tuple(ref["root_quat_w"].shape), (2, 4))
        self.assertEqual(tuple(ref["joint_angles"].shape), (2, 12))
        self.assertEqual(tuple(ref["foot_pos_root"].shape), (2, 4, 3))
        self.assertEqual(tuple(ref["contact_state"].shape), (2, 4))
        self.assertEqual(tuple(ref["planned_touchdown_w"].shape), (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
