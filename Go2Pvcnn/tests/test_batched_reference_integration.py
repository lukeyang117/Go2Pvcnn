import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
        self.assertEqual(cache.canonical_issues(), ())
        self.assertTrue(cache.is_canonical())

    def test_expand_reference_cache_preserves_canonical_layout_and_dtype(self):
        from extension.reference.cache import expand_reference_cache_to_num_envs
        from extension.reference.generator import ReferenceGenerator, ReferenceGeneratorConfig

        cache = ReferenceGenerator(ReferenceGeneratorConfig(horizon_steps=4)).generate()
        expanded = expand_reference_cache_to_num_envs(cache, 3)

        self.assertEqual(tuple(expanded.root_pos_w.shape), (3, 4, 3))
        self.assertEqual(expanded.root_pos_w.dtype, torch.float32)
        self.assertEqual(expanded.root_pos_w.device.type, "cpu")
        self.assertEqual(tuple(expanded.root_quat_w.shape), (3, 4, 4))
        self.assertEqual(expanded.root_quat_w.dtype, torch.float32)
        self.assertEqual(expanded.contact_state.dtype, torch.bool)
        self.assertEqual(expanded.phase_index.dtype, torch.long)
        self.assertEqual(expanded.valid_mask.dtype, torch.bool)
        self.assertTrue(expanded.is_ready())

    def test_shape_issues_rejects_noncanonical_dtype_and_device(self):
        from extension.convention import planner_result_to_reference_cache

        cache = planner_result_to_reference_cache(_fake_result(2, 5))
        cache.root_pos_w = cache.root_pos_w.to(dtype=torch.float64)
        cache.contact_state = cache.contact_state.to(dtype=torch.float32)
        cache.phase_index = cache.phase_index.to(dtype=torch.float32)
        cache.valid_mask = cache.valid_mask.to(device="meta")

        issues = cache.shape_issues()
        self.assertIn("contact_state:dtype=torch.float32", issues)
        self.assertIn("phase_index:dtype=torch.float32", issues)
        self.assertIn("valid_mask:device=meta", issues)
        self.assertFalse(cache.is_ready())

        canonical_issues = cache.canonical_issues()
        self.assertIn("root_pos_w:dtype=torch.float64", canonical_issues)
        self.assertIn("contact_state:dtype=torch.float32", canonical_issues)
        self.assertIn("phase_index:dtype=torch.float32", canonical_issues)
        self.assertIn("valid_mask:device=meta", canonical_issues)

    def test_consumed_cache_stays_ready_after_device_migration_but_loses_canonical_status(self):
        from extension.convention import planner_result_to_reference_cache

        cache = planner_result_to_reference_cache(_fake_result(2, 5)).to(device="meta")

        self.assertEqual(cache.shape_issues(), ())
        self.assertTrue(cache.is_ready())
        issues = cache.canonical_issues()
        self.assertIn("root_pos_w:device=meta", issues)
        self.assertIn("phase_index:device=meta", issues)
        self.assertIn("valid_mask:device=meta", issues)
        self.assertFalse(cache.is_canonical())

    def test_manager_cache_is_compatible_with_reference_gather(self):
        from extension.mdp import rewards_reference
        from extension.convention import planner_result_to_reference_cache

        cache = planner_result_to_reference_cache(_fake_result(2, 5))
        manager = SimpleNamespace(refresh_from_env=Mock(return_value=cache))

        env = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=2,
            episode_length_buf=torch.tensor([0, 3], dtype=torch.long),
            cfg=SimpleNamespace(reference_trajectory_horizon=5),
            unwrapped=SimpleNamespace(_trajectory_manager=manager),
        )
        ensured = rewards_reference.ensure_reference_cache(env)
        self.assertIs(ensured, cache)
        manager.refresh_from_env.assert_called_once_with(env)
        frame_ids = rewards_reference._reference_indices(env, ensured.horizon_length())
        gathered = rewards_reference._gather_reference_field(cache, "root_pos_w", frame_ids, env)

        self.assertEqual(tuple(gathered.shape), (2, 3))
        torch.testing.assert_close(gathered[0], cache.root_pos_w[0, 0])
        torch.testing.assert_close(gathered[1], cache.root_pos_w[1, 3])

    def test_reference_gather_accepts_gpu_frame_indices_for_cpu_canonical_cache(self):
        from extension.mdp import rewards_reference
        from extension.convention import planner_result_to_reference_cache

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to verify mixed CPU-cache/GPU-index gather")

        cache = planner_result_to_reference_cache(_fake_result(2, 5))
        env = SimpleNamespace(device=torch.device("cuda:0"), num_envs=2)
        frame_ids = torch.tensor([1, 4], dtype=torch.long, device=env.device)

        gathered = rewards_reference._gather_reference_field(cache, "root_pos_w", frame_ids, env)

        self.assertEqual(gathered.device.type, "cuda")
        torch.testing.assert_close(gathered.cpu()[0], cache.root_pos_w[0, 1])
        torch.testing.assert_close(gathered.cpu()[1], cache.root_pos_w[1, 4])

    def test_ensure_reference_cache_requires_manager_owned_cache(self):
        from extension.mdp import rewards_reference

        env = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=1,
            episode_length_buf=torch.tensor([0], dtype=torch.long),
            cfg=SimpleNamespace(reference_trajectory_horizon=5),
            unwrapped=SimpleNamespace(),
        )

        with self.assertRaisesRegex(RuntimeError, "planner-owned reference cache"):
            rewards_reference.ensure_reference_cache(env)


if __name__ == "__main__":
    unittest.main()
