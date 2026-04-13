import unittest

import torch

from Go2Pvcnn.extension.mdp.rewards_reference import ensure_reference_cache
from Go2Pvcnn.extension.planner.runtime.reference_cache import (
    ReferenceTrajectoryCache,
    expand_reference_cache_to_num_envs,
)
from Go2Pvcnn.extension.planner.runtime.reference_generator import (
    ReferenceGenerator,
    ReferenceGeneratorConfig,
)
from Go2Pvcnn.extension.planner.runtime.replanning_policy import ReplanningPolicy
from Go2Pvcnn.extension.planner.viz.kinematic_player import (
    KinematicPlayerConfig,
    KinematicTrajectoryPlayer,
)


class ExtensionPlannerRuntimeTest(unittest.TestCase):
    def test_ensure_reference_cache_moves_cache_to_env_device(self):
        class DummyCfg:
            reference_trajectory_horizon = 50

        class DummyEnv:
            def __init__(self, device: str):
                self.device = device
                self.unwrapped = self
                self._trajectory_reference_cache = None
                self.num_envs = 1
                self.cfg = DummyCfg()

        env = DummyEnv("cpu")
        cache = ensure_reference_cache(env)
        self.assertIs(cache, env._trajectory_reference_cache)
        self.assertEqual(cache.root_pos_w.device.type, "cpu")

        class Cfg5:
            reference_trajectory_horizon = 5

        env.cfg = Cfg5()
        env._trajectory_reference_cache = expand_reference_cache_to_num_envs(
            ReferenceGenerator(ReferenceGeneratorConfig(horizon_steps=5)).generate(),
            1,
        )
        cache = ensure_reference_cache(env)
        self.assertEqual(cache.root_pos_w.device.type, "cpu")
        self.assertTrue(cache.is_ready())
        self.assertEqual(cache.root_pos_w.ndim, 3)

    def test_reference_cache_not_ready_without_required_fields(self):
        cache = ReferenceTrajectoryCache()
        self.assertFalse(cache.is_ready())

    def test_reference_generator_returns_empty_cache_scaffold(self):
        generator = ReferenceGenerator(ReferenceGeneratorConfig(horizon_steps=10, dt=0.05))
        cache = generator.generate()
        self.assertIsInstance(cache, ReferenceTrajectoryCache)
        self.assertTrue(cache.is_ready())
        self.assertEqual(cache.root_pos_w.shape, (10, 3))
        self.assertEqual(cache.root_quat_w.shape, (10, 4))
        self.assertEqual(cache.joint_angles.shape, (10, 12))
        self.assertEqual(cache.foot_pos_root.shape, (10, 4, 3))
        self.assertEqual(cache.contact_state.shape, (10, 4))
        self.assertEqual(cache.planned_touchdown_w.shape, (10, 4, 3))
        self.assertEqual(cache.phase_index.shape, (10,))
        self.assertEqual(cache.valid_mask.shape, (10,))
        self.assertTrue(torch.all(cache.valid_mask))
        self.assertGreater(cache.root_pos_w[-1, 0].item(), cache.root_pos_w[0, 0].item())

    def test_expand_reference_cache_batched_shapes(self):
        base = ReferenceGenerator(ReferenceGeneratorConfig(horizon_steps=4)).generate()
        batched = expand_reference_cache_to_num_envs(base, 3)
        self.assertTrue(batched.is_ready())
        self.assertEqual(batched.root_pos_w.shape, (3, 4, 3))
        self.assertEqual(batched.phase_index.shape, (3, 4))

    def test_reference_cache_rejects_mismatched_horizon_lengths(self):
        cache = ReferenceTrajectoryCache(
            root_pos_w=torch.zeros(3, 3),
            root_quat_w=torch.zeros(4, 4),
            joint_angles=torch.zeros(3, 12),
            foot_pos_root=torch.zeros(3, 4, 3),
            contact_state=torch.zeros(3, 4),
            planned_touchdown_w=torch.zeros(3, 4, 3),
            phase_index=torch.zeros(3, dtype=torch.long),
            valid_mask=torch.ones(3, dtype=torch.bool),
        )
        self.assertFalse(cache.is_ready())

    def test_replanning_policy_triggers_on_thresholds(self):
        policy = ReplanningPolicy(max_command_delta=0.2, max_tracking_error=0.4)
        self.assertFalse(policy.should_replan(command_delta=0.1, tracking_error=0.3, reset=False))
        self.assertTrue(policy.should_replan(command_delta=0.3, tracking_error=0.3, reset=False))
        self.assertTrue(policy.should_replan(command_delta=0.1, tracking_error=0.5, reset=False))
        self.assertTrue(policy.should_replan(command_delta=0.0, tracking_error=0.0, reset=True))

    def test_kinematic_player_describe_includes_config(self):
        player = KinematicTrajectoryPlayer(KinematicPlayerConfig(terrain_name="demo", n_frames=12, dt=0.1))
        description = player.describe()
        self.assertIn("demo", description)
        self.assertIn("12", description)
        self.assertIn("0.1", description)


if __name__ == "__main__":
    unittest.main()
