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


class BatchedReferenceIntegrationTest(unittest.TestCase):
    def test_planner_result_is_normalized_into_canonical_reference_cache_layout(self):
        from extension.convention import planner_result_to_reference_cache

        cache = planner_result_to_reference_cache(_fake_result(2, 5))

        self.assertEqual(tuple(cache.root_pos_w.shape), (2, 5, 3))
        self.assertEqual(cache.root_pos_w.dtype, torch.float32)
        self.assertEqual(cache.root_pos_w.device.type, "cpu")
        self.assertEqual(tuple(cache.root_quat_w.shape), (2, 5, 4))
        self.assertEqual(cache.root_quat_w.dtype, torch.float32)
        self.assertEqual(cache.contact_state.dtype, torch.bool)
        self.assertEqual(cache.phase_index.dtype, torch.long)
        self.assertEqual(cache.valid_mask.dtype, torch.bool)
        self.assertEqual(tuple(cache.planned_touchdown_w.shape), (2, 5, 4, 3))
        self.assertEqual(cache.planned_touchdown_w.dtype, torch.float32)

    def test_manager_cache_is_compatible_with_reference_gather(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager
        from extension.mdp import rewards_reference

        cfg = SimpleNamespace(reference_replan_interval_steps=3, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        states = SimpleNamespace(root_pos=torch.zeros(2, 3, dtype=torch.float64))
        commands = torch.zeros(2, 3, dtype=torch.float64)

        with patch("extension.batched_planner.manager.batched_generate_trajectory", return_value=_fake_result(2, 5)):
            cache = manager.step("terrain", states, commands)

        env = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=2,
            episode_length_buf=torch.tensor([0, 3], dtype=torch.long),
            cfg=SimpleNamespace(reference_trajectory_horizon=5),
            unwrapped=SimpleNamespace(_trajectory_reference_cache=cache),
        )
        _, frame_ids = rewards_reference._select_reference_frame(env)
        gathered = rewards_reference._gather_reference_field(cache, "root_pos_w", frame_ids, env)

        self.assertEqual(tuple(gathered.shape), (2, 3))
        torch.testing.assert_close(gathered[0], cache.root_pos_w[0, 0])
        torch.testing.assert_close(gathered[1], cache.root_pos_w[1, 3])


if __name__ == "__main__":
    unittest.main()
