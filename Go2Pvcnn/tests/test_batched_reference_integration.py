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

def _fake_result_with_offset(num_envs: int, num_frames: int, *, offset: float):
    result = _fake_result(num_envs, num_frames)
    return result.__class__(
        num_frames=result.num_frames,
        root_pos_w=result.root_pos_w + offset,
        root_quat_w=result.root_quat_w,
        root_lin_vel_w=result.root_lin_vel_w,
        root_ang_vel_w=result.root_ang_vel_w,
        joint_angles=result.joint_angles + offset,
        foot_pos_w=result.foot_pos_w + offset,
        foot_pos_root=result.foot_pos_root + offset,
        contact_state=result.contact_state,
        body_pos_root=result.body_pos_root + offset,
        planned_touchdown_w=result.planned_touchdown_w + offset,
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

    def test_reward_facing_reference_cache_remains_full_canonical_after_partial_replan(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager
        from extension.mdp import rewards_reference

        cfg = SimpleNamespace(reference_replan_interval_steps=50, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))

        num_envs = 3
        ray_hits = torch.zeros(num_envs, 16, 3, dtype=torch.float64)

        class _FakeScene:
            def __init__(self, robot, sensors):
                self.robot = robot
                self.sensors = sensors

            def __getitem__(self, name):
                return getattr(self, name)

        class _FakeRobot:
            def __init__(self):
                self.data = SimpleNamespace(
                    root_pos_w=torch.zeros(num_envs, 3, dtype=torch.float64),
                    root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=torch.float64),
                    joint_pos=torch.zeros(num_envs, 12, dtype=torch.float64),
                    body_pos_w=torch.zeros(num_envs, 4, 3, dtype=torch.float64),
                )

            def find_bodies(self, pattern):
                return torch.tensor([0, 1, 2, 3], dtype=torch.long), ["FL", "FR", "RL", "RR"]

        class _FakeCommandManager:
            def __init__(self, command):
                self.command = command

            def get_command(self, name):
                return self.command

        class _FakeEnv:
            def __init__(self):
                robot = _FakeRobot()
                scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=ray_hits))
                self.scene = _FakeScene(robot, {"height_scanner": scanner})
                self.command_manager = _FakeCommandManager(torch.zeros(num_envs, 3, dtype=torch.float64))
                self.episode_length_buf = torch.tensor([5, 5, 5], dtype=torch.long)
                self.device = torch.device("cpu")
                self.num_envs = num_envs
                self.cfg = SimpleNamespace(reference_trajectory_horizon=5)
                self.unwrapped = self
                self._trajectory_manager = manager

        env = _FakeEnv()
        planner_batch_sizes: list[int] = []

        def gen_side_effect(terrain, states, commands, requested_n_frames, dt):
            n = int(commands.shape[0])
            planner_batch_sizes.append(n)
            offset = 1000.0 if n == 1 else 0.0
            return _fake_result_with_offset(n, requested_n_frames, offset=offset)

        # Initial full cache; then trigger a partial reset on env1 only.
        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            side_effect=gen_side_effect,
        ):
            cache0 = rewards_reference.ensure_reference_cache(env)
            root_pos0 = cache0.root_pos_w.clone()

            env.episode_length_buf = torch.tensor([6, 0, 6], dtype=torch.long)
            cache1 = rewards_reference.ensure_reference_cache(env)

        self.assertEqual(planner_batch_sizes, [3, 1])
        self.assertTrue(cache1.is_ready())
        self.assertTrue(cache1.is_canonical())
        self.assertEqual(tuple(cache1.root_pos_w.shape), (num_envs, 5, 3))

        # Reward gather still works for full batch, and non-replanned rows keep their values.
        frame_ids = rewards_reference._reference_indices(env, cache1.horizon_length())
        gathered = rewards_reference._gather_reference_field(cache1, "root_pos_w", frame_ids, env)
        self.assertEqual(tuple(gathered.shape), (num_envs, 3))
        torch.testing.assert_close(cache1.root_pos_w[0], root_pos0[0])
        torch.testing.assert_close(cache1.root_pos_w[2], root_pos0[2])
        self.assertFalse(torch.equal(cache1.root_pos_w[1], root_pos0[1]))

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
