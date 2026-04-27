import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fake_result(num_envs: int, num_frames: int, *, offset: float = 0.0, feasible=None, safe_fallback=None, device=None):
    device = torch.device("cpu") if device is None else torch.device(device)
    root_pos = torch.arange(num_envs * num_frames * 3, dtype=torch.float64, device=device).reshape(num_envs, num_frames, 3)
    root_quat = torch.zeros(num_envs, num_frames, 4, dtype=torch.float64, device=device)
    root_quat[..., 0] = 1.0
    joint_angles = torch.arange(num_envs * num_frames * 12, dtype=torch.float64, device=device).reshape(num_envs, num_frames, 12)
    foot_pos_root = torch.arange(num_envs * num_frames * 4 * 3, dtype=torch.float64, device=device).reshape(num_envs, num_frames, 4, 3)
    contact_state = torch.ones(num_envs, num_frames, 4, dtype=torch.bool, device=device)
    touchdown = torch.arange(num_envs * 4 * 3, dtype=torch.float64, device=device).reshape(num_envs, 4, 3)
    result = SimpleNamespace(
        num_frames=num_frames,
        root_pos_w=root_pos + float(offset),
        root_quat_w=root_quat,
        joint_angles=joint_angles + float(offset),
        foot_pos_root=foot_pos_root + float(offset),
        contact_state=contact_state,
        planned_touchdown_w=touchdown + float(offset),
    )
    if feasible is not None:
        result.feasible = torch.as_tensor(feasible, dtype=torch.bool, device=device)
    if safe_fallback is not None:
        result.safe_fallback = torch.as_tensor(safe_fallback, dtype=torch.bool, device=device)
    return result


class _FakeScene:
    def __init__(self, robot, sensors):
        self.robot = robot
        self.sensors = sensors

    def __getitem__(self, name):
        return getattr(self, name)


class _FakeRobot:
    def __init__(self, num_envs: int, *, device):
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros(num_envs, 3, dtype=torch.float64, device=device),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64, device=device).expand(num_envs, 4).clone(),
            joint_pos=torch.zeros(num_envs, 12, dtype=torch.float64, device=device),
            body_pos_w=torch.zeros(num_envs, 4, 3, dtype=torch.float64, device=device),
        )

    def find_bodies(self, pattern):
        return torch.tensor([0, 1, 2, 3], dtype=torch.long), ["FL", "FR", "RL", "RR"]


class _FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_command(self, name):
        return self.command


class _FakeEnv:
    def __init__(self, *, num_envs: int, device=torch.device("cpu"), horizon: int = 5):
        device = torch.device(device)
        robot = _FakeRobot(num_envs, device=device)
        scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=torch.zeros(num_envs, 16, 3, dtype=torch.float64, device=device)))
        self.scene = _FakeScene(robot, {"height_scanner": scanner})
        self.command_manager = _FakeCommandManager(torch.zeros(num_envs, 3, dtype=torch.float64, device=device))
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.device = device
        self.num_envs = num_envs
        self.common_step_counter = 0
        self.cfg = SimpleNamespace(reference_trajectory_horizon=horizon)
        self.unwrapped = self


def _cfg(*, backend="together", horizon=5, interval=35):
    return SimpleNamespace(
        planner_backend=backend,
        planner_owned_reference_cache=True,
        reference_replan_interval_steps=interval,
        reference_trajectory_horizon=horizon,
        plan_dt=0.02,
        sim=SimpleNamespace(device="cpu"),
    )


class TogetherManagerTest(unittest.TestCase):
    def test_factory_selects_together_and_legacy(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager
        from extension.batched_together_planner.manager import TogetherTrajectoryManager
        from extension.trajectory_manager_factory import create_trajectory_manager

        self.assertIsInstance(create_trajectory_manager(_cfg(backend="together"), device="cpu"), TogetherTrajectoryManager)
        self.assertIsInstance(create_trajectory_manager(_cfg(backend="legacy"), device="cpu"), BatchedTrajectoryManager)

    def test_factory_rejects_invalid_backend(self):
        from extension.trajectory_manager_factory import create_trajectory_manager

        with self.assertRaisesRegex(ValueError, "Invalid planner_backend"):
            create_trajectory_manager(_cfg(backend="banana"), device="cpu")

    def test_full_n_planner_call_and_no_trigger_skip(self):
        from extension.batched_together_planner.manager import TogetherTrajectoryManager

        env = _FakeEnv(num_envs=4, horizon=5)
        manager = TogetherTrajectoryManager(_cfg(horizon=5, interval=35), device="cpu")
        batch_sizes = []

        def generate(terrain, states, commands, cfg):
            batch_sizes.append(int(commands.shape[0]))
            return _fake_result(int(commands.shape[0]), int(cfg.horizon_steps))

        with patch("extension.batched_together_planner.manager.TogetherPlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_together_planner.manager.plan_segment",
            side_effect=generate,
        ) as gen:
            manager.refresh_from_env(env)
            env.common_step_counter = 1
            manager.refresh_from_env(env)

        self.assertEqual(batch_sizes, [4])
        self.assertEqual(gen.call_count, 1)
        self.assertTrue(torch.equal(manager.current_frame_ids(), torch.ones(4, dtype=torch.long)))

    def test_planner_cfg_ignores_legacy_training_fields(self):
        from extension.batched_together_planner.config import TogetherPlannerConfig
        from extension.batched_together_planner.manager import TogetherTrajectoryManager

        raw_defaults = TogetherPlannerConfig()
        env = _FakeEnv(num_envs=2, horizon=35)
        cfg = _cfg(horizon=35, interval=35)
        cfg.step_freq = 3.0
        cfg.step_height = 0.11
        cfg.duty_factor = 0.6
        cfg.replan_stop_speed = 0.05
        cfg.foothold_search_radius = 0.15
        cfg.foothold_search_step = 0.03
        manager = TogetherTrajectoryManager(cfg, device="cpu")

        def generate(terrain, states, commands, cfg):
            return _fake_result(int(commands.shape[0]), int(cfg.horizon_steps))

        with patch("extension.batched_together_planner.manager.TogetherPlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_together_planner.manager.plan_segment",
            side_effect=generate,
        ) as gen:
            manager.refresh_from_env(env)

        planner_cfg = gen.call_args.kwargs["cfg"]
        self.assertEqual(planner_cfg.horizon_steps, 35)
        self.assertEqual(planner_cfg.dt, raw_defaults.dt)
        self.assertEqual(planner_cfg.step_freq, 3.0)
        self.assertEqual(planner_cfg.swing_height, 0.11)
        self.assertEqual(planner_cfg.duty_factor, raw_defaults.duty_factor)
        self.assertEqual(planner_cfg.idle_command_eps, raw_defaults.idle_command_eps)
        self.assertEqual(planner_cfg.support_search_radius, raw_defaults.support_search_radius)
        self.assertEqual(planner_cfg.support_search_step, raw_defaults.support_search_step)

    def test_command_dirty_token_triggers_once_full_n(self):
        from extension.batched_together_planner.manager import TogetherTrajectoryManager

        env = _FakeEnv(num_envs=3, horizon=5)
        manager = TogetherTrajectoryManager(_cfg(horizon=5, interval=35), device="cpu")
        with patch("extension.batched_together_planner.manager.TogetherPlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_together_planner.manager.plan_segment",
            side_effect=lambda terrain, states, commands, cfg: _fake_result(int(commands.shape[0]), int(cfg.horizon_steps)),
        ) as gen:
            manager.refresh_from_env(env)
            env.common_step_counter = 1
            manager.mark_command_changed(torch.tensor([False, True, False]))
            manager.refresh_from_env(env)
            env.common_step_counter = 2
            manager.refresh_from_env(env)

        self.assertEqual(gen.call_count, 2)
        self.assertEqual(int(gen.call_args.args[2].shape[0]), 3)

    def test_reset_triggers_full_n_attempt(self):
        from extension.batched_together_planner.manager import TogetherTrajectoryManager

        env = _FakeEnv(num_envs=2, horizon=5)
        manager = TogetherTrajectoryManager(_cfg(horizon=5, interval=35), device="cpu")
        with patch("extension.batched_together_planner.manager.TogetherPlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_together_planner.manager.plan_segment",
            return_value=_fake_result(2, 5),
        ) as gen:
            manager.refresh_from_env(env)
            env.common_step_counter = 1
            manager.reset_envs(torch.tensor([True, False]))
            manager.refresh_from_env(env)

        self.assertEqual(gen.call_count, 2)
        self.assertEqual(int(gen.call_args.args[2].shape[0]), 2)

    def test_same_step_reward_calls_are_idempotent(self):
        from extension.batched_together_planner.manager import TogetherTrajectoryManager
        from extension.mdp import rewards_reference

        env = _FakeEnv(num_envs=2, horizon=5)
        manager = TogetherTrajectoryManager(_cfg(horizon=5, interval=35), device="cpu")
        env._trajectory_manager = manager
        with patch("extension.batched_together_planner.manager.TogetherPlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_together_planner.manager.plan_segment",
            return_value=_fake_result(2, 5),
        ) as gen:
            rewards_reference.ensure_reference_cache(env)
            first_frame = manager.current_frame_ids().clone()
            rewards_reference.ensure_reference_cache(env)
            second_frame = manager.current_frame_ids().clone()

        self.assertEqual(gen.call_count, 1)
        self.assertTrue(torch.equal(first_frame, second_frame))

    def test_old_new_fallback_truth_table(self):
        from extension.batched_together_planner.manager import TogetherTrajectoryManager

        env = _FakeEnv(num_envs=6, horizon=5)
        manager = TogetherTrajectoryManager(_cfg(horizon=5, interval=35), device="cpu")
        results = [
            _fake_result(6, 5, offset=0.0, feasible=[True, True, True, True, True, True]),
            _fake_result(
                6,
                5,
                offset=1000.0,
                feasible=[True, False, True, False, True, False],
                safe_fallback=[False, False, False, True, False, False],
            ),
        ]

        with patch("extension.batched_together_planner.manager.TogetherPlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_together_planner.manager.plan_segment",
            side_effect=results,
        ):
            cache0 = manager.refresh_from_env(env)
            old_root = cache0.root_pos_w.clone()
            env.common_step_counter = 1
            manager.reset_envs(torch.tensor([True, True, False, False, False, False]))
            manager.mark_command_changed(torch.tensor([False, False, True, True, False, False]))
            cache1 = manager.refresh_from_env(env)

        torch.testing.assert_close(cache1.root_pos_w[0], old_root[0] + 1000.0)
        torch.testing.assert_close(cache1.root_pos_w[1], torch.zeros_like(cache1.root_pos_w[1]))
        torch.testing.assert_close(cache1.root_pos_w[2], old_root[2] + 1000.0)
        torch.testing.assert_close(cache1.root_pos_w[3], old_root[3] + 1000.0)
        torch.testing.assert_close(cache1.root_pos_w[4], old_root[4])
        torch.testing.assert_close(cache1.root_pos_w[5], old_root[5])
        self.assertTrue(torch.equal(manager.current_frame_ids(), torch.tensor([0, 0, 0, 0, 1, 1])))

    def test_together_cache_preserves_result_device(self):
        from extension.batched_together_planner.adapter import together_result_to_reference_cache

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cache = together_result_to_reference_cache(_fake_result(2, 5, device=device))

        self.assertEqual(cache.root_pos_w.device, device)
        self.assertEqual(cache.root_quat_w.device, device)
        self.assertEqual(cache.phase_index.device, device)
        self.assertTrue(cache.is_ready())

    def test_attach_helper_wraps_command_hook(self):
        from extension.trajectory_manager_factory import attach_trajectory_manager

        class Term:
            def _resample_command(self, env_ids):
                return "ok"

        term = Term()
        env = _FakeEnv(num_envs=2, horizon=5)
        env.command_manager.get_term = Mock(return_value=term)
        manager = attach_trajectory_manager(env, _cfg(horizon=5), device="cpu")
        term._resample_command(torch.tensor([1]))

        self.assertTrue(manager._pending_command_dirty)
        self.assertTrue(torch.equal(manager._pending_command_mask, torch.tensor([False, True])))


if __name__ == "__main__":
    unittest.main()
