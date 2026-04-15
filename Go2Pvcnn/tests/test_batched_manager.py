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

def _fake_result_with_offset(num_envs: int, num_frames: int, *, offset: float):
    result = _fake_result(num_envs, num_frames)
    # Make it easy to detect which env rows were replanned by shifting numeric fields.
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


class _FakeScene:
    def __init__(self, robot, sensors):
        self.robot = robot
        self.sensors = sensors

    def __getitem__(self, name):
        return getattr(self, name)


class _FakeRobot:
    def __init__(self, root_pos, root_quat, joint_pos, foot_pos):
        self.data = SimpleNamespace(
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            joint_pos=joint_pos,
            body_pos_w=foot_pos,
        )

    def find_bodies(self, pattern):
        return torch.tensor([0, 1, 2, 3], dtype=torch.long), ["FL", "FR", "RL", "RR"]


class _FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_command(self, name):
        return self.command


class _FakeEnv:
    def __init__(self, *, episode_length_buf, command, ray_hits):
        num_envs = int(episode_length_buf.shape[0])
        root_pos = torch.zeros(num_envs, 3, dtype=torch.float64)
        root_quat = torch.zeros(num_envs, 4, dtype=torch.float64)
        root_quat[..., 0] = 1.0
        joint_pos = torch.zeros(num_envs, 12, dtype=torch.float64)
        foot_pos = torch.zeros(num_envs, 4, 3, dtype=torch.float64)
        robot = _FakeRobot(root_pos, root_quat, joint_pos, foot_pos)
        scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=ray_hits))
        self.scene = _FakeScene(robot, {"height_scanner": scanner})
        self.command_manager = _FakeCommandManager(command)
        self.episode_length_buf = episode_length_buf
        self.device = torch.device("cpu")
        self.num_envs = int(episode_length_buf.shape[0])
        self.unwrapped = self


class BatchedManagerTest(unittest.TestCase):
    def _cfg(self):
        return SimpleNamespace(reference_replan_interval_steps=3, reference_trajectory_horizon=5, dt=0.02)

    def _cfg_from_sim(self):
        return SimpleNamespace(
            reference_replan_interval_steps=3,
            reference_trajectory_horizon=5,
            decimation=4,
            sim=SimpleNamespace(dt=0.005),
        )

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

    def test_refresh_from_env_replans_on_reset_and_command_change(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(reference_replan_interval_steps=50, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        ray_hits = torch.zeros(2, 16, 3, dtype=torch.float64)
        env = _FakeEnv(
            episode_length_buf=torch.tensor([5, 5], dtype=torch.long),
            command=torch.zeros(2, 3, dtype=torch.float64),
            ray_hits=ray_hits,
        )

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()) as from_hits, patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            return_value=_fake_result(2, 5),
        ) as gen:
            manager.refresh_from_env(env)
            manager.refresh_from_env(env)
            env.episode_length_buf = torch.tensor([0, 0], dtype=torch.long)
            manager.refresh_from_env(env)
            env.episode_length_buf = torch.tensor([1, 1], dtype=torch.long)
            env.command_manager.command = torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float64)
            manager.refresh_from_env(env)

        self.assertEqual(from_hits.call_count, 3)
        self.assertEqual(gen.call_count, 3)
        self.assertTrue(torch.equal(manager._last_episode_length_buf, torch.tensor([1, 1], dtype=torch.long)))
        self.assertTrue(torch.allclose(manager._last_commands, torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float64)))

    def test_refresh_from_env_replans_when_command_changes_without_episode_progress(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(reference_replan_interval_steps=50, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        ray_hits = torch.zeros(2, 16, 3, dtype=torch.float64)
        env = _FakeEnv(
            episode_length_buf=torch.tensor([7, 7], dtype=torch.long),
            command=torch.zeros(2, 3, dtype=torch.float64),
            ray_hits=ray_hits,
        )

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            return_value=_fake_result(2, 5),
        ) as gen:
            manager.refresh_from_env(env)
            env.command_manager.command = torch.tensor([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float64)
            manager.refresh_from_env(env)

        self.assertEqual(gen.call_count, 2)
        self.assertTrue(torch.allclose(manager._last_commands, torch.tensor([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float64)))

    def test_step_uses_cfg_sim_dt_times_decimation_when_cfg_dt_is_missing(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = self._cfg_from_sim()
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        states = SimpleNamespace(root_pos=torch.zeros(2, 3, dtype=torch.float64))
        commands = torch.zeros(2, 3, dtype=torch.float64)

        with patch("extension.batched_planner.manager.batched_generate_trajectory", return_value=_fake_result(2, 5)) as gen:
            manager.step("terrain", states, commands)

        self.assertEqual(gen.call_args.kwargs["dt"], 0.02)

    def test_refresh_from_env_prefers_runtime_step_dt(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = self._cfg_from_sim()
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        ray_hits = torch.zeros(2, 16, 3, dtype=torch.float64)
        env = _FakeEnv(
            episode_length_buf=torch.tensor([2, 2], dtype=torch.long),
            command=torch.zeros(2, 3, dtype=torch.float64),
            ray_hits=ray_hits,
        )
        env.step_dt = 0.03

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            return_value=_fake_result(2, 5),
        ) as gen:
            manager.refresh_from_env(env)

        self.assertEqual(gen.call_args.kwargs["dt"], 0.03)

    def test_refresh_from_env_reset_replans_only_selected_env_rows(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(reference_replan_interval_steps=50, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        num_envs = 3
        ray_hits = torch.zeros(num_envs, 16, 3, dtype=torch.float64)
        env = _FakeEnv(
            episode_length_buf=torch.tensor([5, 5, 5], dtype=torch.long),
            command=torch.zeros(num_envs, 3, dtype=torch.float64),
            ray_hits=ray_hits,
        )

        planner_batch_sizes = []

        def gen_side_effect(terrain, states, commands, requested_n_frames, dt):
            n = int(commands.shape[0])
            planner_batch_sizes.append(n)
            offset = 1000.0 if n == 1 else 0.0
            return _fake_result_with_offset(n, requested_n_frames, offset=offset)

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            side_effect=gen_side_effect,
        ):
            cache0 = manager.refresh_from_env(env)
            root_pos0 = cache0.root_pos_w.clone()
            env.episode_length_buf = torch.tensor([6, 0, 6], dtype=torch.long)
            cache1 = manager.refresh_from_env(env)

        self.assertEqual(planner_batch_sizes, [3, 1])
        self.assertTrue(cache1.is_ready())
        self.assertTrue(cache1.is_canonical())
        self.assertEqual(tuple(cache1.root_pos_w.shape), (num_envs, 5, 3))

        torch.testing.assert_close(cache1.root_pos_w[0], root_pos0[0])
        torch.testing.assert_close(cache1.root_pos_w[2], root_pos0[2])
        self.assertFalse(torch.equal(cache1.root_pos_w[1], root_pos0[1]))
        self.assertTrue(torch.equal(manager._phase_counter, torch.tensor([1, 0, 1], dtype=torch.long)))

    def test_refresh_from_env_command_change_replans_only_changed_env_rows(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(reference_replan_interval_steps=50, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        num_envs = 3
        ray_hits = torch.zeros(num_envs, 16, 3, dtype=torch.float64)
        env = _FakeEnv(
            episode_length_buf=torch.tensor([5, 5, 5], dtype=torch.long),
            command=torch.zeros(num_envs, 3, dtype=torch.float64),
            ray_hits=ray_hits,
        )

        planner_batch_sizes = []

        def gen_side_effect(terrain, states, commands, requested_n_frames, dt):
            n = int(commands.shape[0])
            planner_batch_sizes.append(n)
            offset = 2000.0 if n == 1 else 0.0
            return _fake_result_with_offset(n, requested_n_frames, offset=offset)

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            side_effect=gen_side_effect,
        ):
            cache0 = manager.refresh_from_env(env)
            root_pos0 = cache0.root_pos_w.clone()
            env.episode_length_buf = torch.tensor([6, 6, 6], dtype=torch.long)
            env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
            cache1 = manager.refresh_from_env(env)

        self.assertEqual(planner_batch_sizes, [3, 1])
        torch.testing.assert_close(cache1.root_pos_w[1], root_pos0[1])
        torch.testing.assert_close(cache1.root_pos_w[2], root_pos0[2])
        self.assertFalse(torch.equal(cache1.root_pos_w[0], root_pos0[0]))

    def test_refresh_from_env_interval_replans_only_elapsed_env_rows_and_keeps_full_cache_contract(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(reference_replan_interval_steps=3, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        num_envs = 3
        ray_hits = torch.zeros(num_envs, 16, 3, dtype=torch.float64)
        env = _FakeEnv(
            episode_length_buf=torch.tensor([0, 0, 0], dtype=torch.long),
            command=torch.zeros(num_envs, 3, dtype=torch.float64),
            ray_hits=ray_hits,
        )

        planner_batch_sizes: list[int] = []

        def gen_side_effect(terrain, states, commands, requested_n_frames, dt):
            n = int(commands.shape[0])
            planner_batch_sizes.append(n)
            # Offset only for the subset replans to make it obvious which rows changed.
            offset = 1000.0 if n == 1 else 0.0
            return _fake_result_with_offset(n, requested_n_frames, offset=offset)

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            side_effect=gen_side_effect,
        ):
            cache0 = manager.refresh_from_env(env)
            root_pos0 = cache0.root_pos_w.clone()

            # No interval elapsed yet, so no replans should happen here.
            env.episode_length_buf = torch.tensor([1, 1, 1], dtype=torch.long)
            cache1 = manager.refresh_from_env(env)
            torch.testing.assert_close(cache1.root_pos_w, root_pos0)

            # Only env0 hits the interval; env1/env2 stay on cached rows.
            env.episode_length_buf = torch.tensor([3, 2, 2], dtype=torch.long)
            cache2 = manager.refresh_from_env(env)

        self.assertEqual(planner_batch_sizes, [3, 1])
        self.assertTrue(cache2.is_ready())
        self.assertTrue(cache2.is_canonical())
        self.assertEqual(tuple(cache2.root_pos_w.shape), (num_envs, 5, 3))

        self.assertFalse(torch.equal(cache2.root_pos_w[0], root_pos0[0]))
        torch.testing.assert_close(cache2.root_pos_w[1], root_pos0[1])
        torch.testing.assert_close(cache2.root_pos_w[2], root_pos0[2])

        # Replanned env resets phase; non-replanned envs advance.
        self.assertTrue(torch.equal(manager._phase_counter, torch.tensor([0, 2, 2], dtype=torch.long)))

        # Reward-facing accessor should remain batch-shaped even after a partial replan.
        ref = manager.current_reference()
        self.assertEqual(tuple(ref["root_pos_w"].shape), (num_envs, 3))
        self.assertEqual(tuple(ref["root_quat_w"].shape), (num_envs, 4))
        self.assertEqual(tuple(ref["joint_angles"].shape), (num_envs, 12))
        self.assertEqual(tuple(ref["foot_pos_root"].shape), (num_envs, 4, 3))
        self.assertEqual(tuple(ref["contact_state"].shape), (num_envs, 4))
        self.assertEqual(tuple(ref["planned_touchdown_w"].shape), (num_envs, 4, 3))
        self.assertEqual(tuple(ref["phase_index"].shape), (num_envs,))
        self.assertEqual(tuple(ref["valid_mask"].shape), (num_envs,))

    def test_failed_env_keeps_standstill_cache_until_its_own_next_replan(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(reference_replan_interval_steps=50, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        num_envs = 3
        ray_hits = torch.zeros(num_envs, 16, 3, dtype=torch.float64)
        env = _FakeEnv(
            episode_length_buf=torch.tensor([5, 5, 5], dtype=torch.long),
            command=torch.zeros(num_envs, 3, dtype=torch.float64),
            ray_hits=ray_hits,
        )

        def gen_side_effect(terrain, states, commands, requested_n_frames, dt):
            n = int(commands.shape[0])
            if n == 1 and float(commands[0, 0].item()) == 0.0:
                raise RuntimeError("planner failed for this env")
            if n == 1 and float(commands[0, 0].item()) > 0.0:
                return _fake_result_with_offset(n, requested_n_frames, offset=2000.0)
            if n == 1 and float(commands[0, 0].item()) < 0.0:
                return _fake_result_with_offset(n, requested_n_frames, offset=3000.0)
            return _fake_result_with_offset(n, requested_n_frames, offset=0.0)

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            side_effect=gen_side_effect,
        ):
            cache0 = manager.refresh_from_env(env)
            root_pos0 = cache0.root_pos_w.clone()

            # Env2 resets; replanning that env fails. Manager should keep a standstill row for env2.
            env.episode_length_buf = torch.tensor([6, 6, 0], dtype=torch.long)
            cache1 = manager.refresh_from_env(env)
            env2_row = cache1.root_pos_w[2]
            env2_expected = env2_row[0].unsqueeze(0).expand_as(env2_row)
            torch.testing.assert_close(env2_row, env2_expected)

            # Env0 changes command; only env0 replans, env2 stays standstill.
            env.episode_length_buf = torch.tensor([7, 7, 1], dtype=torch.long)
            env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
            cache2 = manager.refresh_from_env(env)
            env2_row2 = cache2.root_pos_w[2]
            env2_expected2 = env2_row2[0].unsqueeze(0).expand_as(env2_row2)
            torch.testing.assert_close(env2_row2, env2_expected2)

            # Env2 changes command; now env2 replans successfully and is no longer standstill.
            env.episode_length_buf = torch.tensor([8, 8, 2], dtype=torch.long)
            env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=torch.float64)
            cache3 = manager.refresh_from_env(env)
            self.assertFalse(torch.equal(cache3.root_pos_w[2, 0], cache3.root_pos_w[2, 1]))

        # Sanity: env1 never replanned after the initial call in this scenario.
        torch.testing.assert_close(cache3.root_pos_w[1], root_pos0[1])

    def test_failed_interval_replan_does_not_retry_every_step_due_to_interval_bookkeeping(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(reference_replan_interval_steps=3, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        num_envs = 2
        ray_hits = torch.zeros(num_envs, 16, 3, dtype=torch.float64)
        env = _FakeEnv(
            episode_length_buf=torch.tensor([0, 0], dtype=torch.long),
            command=torch.zeros(num_envs, 3, dtype=torch.float64),
            ray_hits=ray_hits,
        )

        def gen_side_effect(terrain, states, commands, requested_n_frames, dt):
            n = int(commands.shape[0])
            if n == 1:
                raise RuntimeError("planner failed for interval subset")
            return _fake_result(n, requested_n_frames)

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            side_effect=gen_side_effect,
        ) as gen:
            # Initial full plan.
            manager.refresh_from_env(env)
            self.assertEqual(gen.call_count, 1)

            # Env0 hits the interval trigger (3 steps since last replan); env1 does not.
            env.episode_length_buf = torch.tensor([3, 1], dtype=torch.long)
            manager.refresh_from_env(env)
            self.assertEqual(gen.call_count, 2)

            # The failed env should record the attempted "replan time" so the interval trigger
            # doesn't keep firing every subsequent step.
            self.assertTrue(torch.equal(manager._last_replan_episode_length_buf, torch.tensor([3, 0], dtype=torch.long)))

            # Next step: should not retry replanning env0 purely due to interval bookkeeping.
            env.episode_length_buf = torch.tensor([4, 2], dtype=torch.long)
            manager.refresh_from_env(env)
            self.assertEqual(gen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
