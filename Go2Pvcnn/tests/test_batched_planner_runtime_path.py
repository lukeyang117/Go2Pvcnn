import importlib
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


def _fresh_import(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@contextmanager
def _fake_isaaclab_app(flag_name: str):
    fake_isaaclab = ModuleType("isaaclab")
    fake_app = ModuleType("isaaclab.app")

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument(flag_name, action="store_true", default=False)

    fake_app.AppLauncher = FakeAppLauncher
    fake_isaaclab.app = fake_app

    with patch.dict(sys.modules, {"isaaclab": fake_isaaclab, "isaaclab.app": fake_app}):
        yield


class BatchedPlannerRuntimePathTest(unittest.TestCase):
    class _FakeCommandManager:
        def __init__(self, command: torch.Tensor):
            self.command = command

        def get_command(self, name: str):
            return self.command

    def _make_fake_scanner(
        self,
        hits: torch.Tensor,
        *,
        pos: tuple[float, float, float] = (10.0, 20.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        size: tuple[float, float] = (1.5, 1.5),
    ):
        data = SimpleNamespace(
            ray_hits_w=hits,
            pos_w=torch.tensor([pos], dtype=hits.dtype),
            quat_w=torch.tensor([quat], dtype=hits.dtype),
        )
        cfg = SimpleNamespace(pattern_cfg=SimpleNamespace(size=size))
        return SimpleNamespace(data=data, cfg=cfg)

    def _make_fake_viewer_env(self, *, episode_length_buf: torch.Tensor, command: torch.Tensor, ray_hits: torch.Tensor):
        num_envs = int(episode_length_buf.shape[0])
        root_pos = torch.zeros((num_envs, 3), dtype=torch.float64)
        root_quat = torch.zeros((num_envs, 4), dtype=torch.float64)
        root_quat[..., 0] = 1.0
        joint_pos = torch.zeros((num_envs, 12), dtype=torch.float64)
        foot_pos = torch.zeros((num_envs, 4, 3), dtype=torch.float64)
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=root_pos,
                root_quat_w=root_quat,
                joint_pos=joint_pos,
                body_pos_w=foot_pos,
            ),
            find_bodies=lambda pattern: (torch.tensor([0, 1, 2, 3], dtype=torch.long), ["FL", "FR", "RL", "RR"]),
        )
        scanner = self._make_fake_scanner(ray_hits)
        scene = SimpleNamespace(robot=robot, sensors={"height_scanner": scanner})
        env = SimpleNamespace(
            scene=scene,
            command_manager=self._FakeCommandManager(command),
            episode_length_buf=episode_length_buf,
            device=torch.device("cpu"),
            num_envs=num_envs,
        )
        env.unwrapped = env
        return env

    def test_train_module_is_import_safe(self):
        module = _fresh_import("Go2Pvcnn.scripts.train")
        self.assertTrue(hasattr(module, "build_arg_parser"))

    def test_train_parser_includes_app_launcher_args(self):
        with _fake_isaaclab_app("--train-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.scripts.train")
            parser = module.build_arg_parser()
            parsed = parser.parse_args(["--train-launcher-flag"])

        self.assertTrue(parsed.train_launcher_flag)

    def test_train_parser_rejects_legacy_raw_reference_flag(self):
        with _fake_isaaclab_app("--train-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.scripts.train")
            parser = module.build_arg_parser()

            with self.assertRaises(SystemExit):
                parser.parse_args(["--use-raw-reference-trajectory"])

    def test_train_parser_accepts_verbose_planner_flag(self):
        with _fake_isaaclab_app("--train-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.scripts.train")
            parser = module.build_arg_parser()
            parsed = parser.parse_args(["--verbose-planner", "--train-launcher-flag"])

        self.assertTrue(parsed.verbose_planner)

    def test_train_runtime_prep_sets_allocator_and_disables_livestream_for_distributed(self):
        module = _fresh_import("Go2Pvcnn.scripts.train")
        with patch.dict(module.os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "original"}, clear=False):
            args = SimpleNamespace(
                distributed=True,
                livestream=1,
                enable_cameras=False,
                device="cuda:0",
            )
            prepared = module._prepare_runtime_args(args)
            self.assertEqual(module.os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

        self.assertIs(prepared, args)
        self.assertEqual(prepared.livestream, 0)
        self.assertFalse(prepared.enable_cameras)

    def test_train_attaches_planner_owned_manager_for_teacher_elevation_trajectory(self):
        module = _fresh_import("Go2Pvcnn.scripts.train")
        env = SimpleNamespace(device=torch.device("cpu"), unwrapped=SimpleNamespace())
        env_cfg = SimpleNamespace(
            sim=SimpleNamespace(device="cpu"),
            planner_owned_reference_cache=True,
            planner_backend="together",
        )
        sentinel_manager = SimpleNamespace(planner_backend="together", refresh_from_env=SimpleNamespace())

        with patch("extension.trajectory_manager_factory.create_trajectory_manager", return_value=sentinel_manager) as factory:
            module._attach_reference_manager_if_enabled(env, env_cfg, "teacher_elevation_trajectory")

        factory.assert_called_once_with(env_cfg, device=torch.device("cpu"))
        self.assertIs(env.unwrapped._trajectory_manager, sentinel_manager)
        self.assertIsNone(env.unwrapped._trajectory_reference_cache)

    def test_train_rejects_nonplanner_reference_mode(self):
        module = _fresh_import("Go2Pvcnn.scripts.train")
        env = SimpleNamespace(device=torch.device("cpu"), unwrapped=SimpleNamespace())
        env_cfg = SimpleNamespace(sim=SimpleNamespace(device="cpu"), planner_owned_reference_cache=False)

        with self.assertRaisesRegex(RuntimeError, "planner_owned_reference_cache"):
            module._attach_reference_manager_if_enabled(env, env_cfg, "teacher_elevation_trajectory")

    def test_viewer_module_is_import_safe(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        self.assertTrue(hasattr(module, "build_arg_parser"))

    def test_play_module_is_import_safe(self):
        module = _fresh_import("Go2Pvcnn.scripts.play")
        self.assertTrue(hasattr(module, "build_arg_parser"))

    def test_play_parser_includes_app_launcher_args_and_debug_flag(self):
        with _fake_isaaclab_app("--play-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.scripts.play")
            parser = module.build_arg_parser()
            parsed = parser.parse_args(
                [
                    "--run_dir",
                    "2026-04-16_11-53-48",
                    "--checkpoint",
                    "model_5300.pt",
                    "--debug-livestream",
                    "--play-launcher-flag",
                ]
            )

        self.assertTrue(parsed.debug_livestream)
        self.assertTrue(parsed.play_launcher_flag)
        self.assertIsNone(parsed.planner_backend)

    def test_play_prepare_runtime_args_enables_cameras_for_livestream(self):
        module = _fresh_import("Go2Pvcnn.scripts.play")
        args = SimpleNamespace(
            livestream=2,
            enable_cameras=False,
            headless=True,
            video=False,
            debug_livestream=False,
        )

        prepared = module._prepare_runtime_args(args)

        self.assertIs(prepared, args)
        self.assertTrue(prepared.enable_cameras)

    def test_play_render_mode_uses_rgb_array_for_livestream(self):
        module = _fresh_import("Go2Pvcnn.scripts.play")
        args = SimpleNamespace(video=False, livestream=2)

        render_mode = module._resolve_render_mode(args)

        self.assertEqual(render_mode, "rgb_array")

    def test_play_camera_update_policy_throttles_livestream(self):
        module = _fresh_import("Go2Pvcnn.scripts.play")

        self.assertTrue(module._should_update_follow_camera(timestep=8, num_envs=1, livestream=2, interval=4))
        self.assertFalse(module._should_update_follow_camera(timestep=9, num_envs=1, livestream=2, interval=4))
        self.assertTrue(module._should_update_follow_camera(timestep=9, num_envs=1, livestream=0, interval=4))
        self.assertFalse(module._should_update_follow_camera(timestep=8, num_envs=2, livestream=2, interval=4))

    def test_play_debug_runtime_snapshot_reports_effective_launch_settings(self):
        module = _fresh_import("Go2Pvcnn.scripts.play")
        args = SimpleNamespace(
            livestream=2,
            headless=True,
            enable_cameras=True,
            device="cuda:0",
            debug_livestream=True,
        )

        snapshot = module._collect_runtime_debug_snapshot(args, argv=["python", "play.py", "--livestream", "2"])

        self.assertEqual(snapshot["argv"], ["python", "play.py", "--livestream", "2"])
        self.assertEqual(snapshot["args"]["livestream"], 2)
        self.assertEqual(snapshot["args"]["headless"], True)
        self.assertEqual(snapshot["args"]["enable_cameras"], True)

    def test_play_step_probe_snapshot_resets_accumulators(self):
        module = _fresh_import("Go2Pvcnn.scripts.play")
        probe = module._StepProbe(enabled=True)
        probe.accumulators["sim_step_s"] = 1.25
        probe.accumulators["obs_compute_s"] = 0.5

        snapshot = probe.snapshot_and_reset()

        self.assertEqual(snapshot["sim_step_s"], 1.25)
        self.assertEqual(snapshot["obs_compute_s"], 0.5)
        self.assertEqual(probe.accumulators["sim_step_s"], 0.0)
        self.assertEqual(probe.accumulators["obs_compute_s"], 0.0)

    def test_play_configures_batched_reference_runtime_and_warns_legacy_raw_flag(self):
        module = _fresh_import("Go2Pvcnn.scripts.play")
        env_cfg = SimpleNamespace(use_batched_reference_trajectory=False)

        with patch("builtins.print") as print_mock:
            module._configure_reference_trajectory(env_cfg, use_raw_reference_trajectory=True)

        self.assertTrue(env_cfg.use_batched_reference_trajectory)
        print_mock.assert_called_once()
        self.assertIn("legacy-only", print_mock.call_args.args[0])

    def test_viewer_attaches_factory_owned_manager_before_warmup(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        env = SimpleNamespace(device=torch.device("cpu"), unwrapped=SimpleNamespace())
        env_cfg = SimpleNamespace(
            sim=SimpleNamespace(device="cpu"),
            planner_owned_reference_cache=True,
            planner_backend="together",
        )
        sentinel_manager = SimpleNamespace(planner_backend="together", refresh_from_env=SimpleNamespace())

        with patch("extension.trajectory_manager_factory.create_trajectory_manager", return_value=sentinel_manager) as factory:
            module._attach_reference_manager_if_enabled(env, env_cfg)

        factory.assert_called_once_with(env_cfg, device=torch.device("cpu"))
        self.assertIs(env.unwrapped._trajectory_manager, sentinel_manager)
        self.assertIsNone(env.unwrapped._trajectory_reference_cache)

    def test_viewer_parser_includes_app_launcher_args(self):
        with _fake_isaaclab_app("--viewer-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
            parser = module.build_arg_parser()
            parsed = parser.parse_args(["--viewer-launcher-flag"])

        self.assertTrue(parsed.viewer_launcher_flag)

    def test_viewer_parser_defaults_match_validated_diagnostics_regime(self):
        with _fake_isaaclab_app("--viewer-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
            parser = module.build_arg_parser()
            parsed = parser.parse_args([])

        self.assertEqual(parsed.terrain, "task")
        self.assertEqual(parsed.planner_backend, "together")
        self.assertEqual(parsed.n_frames, 35)
        self.assertEqual(parsed.plan_dt, 0.02)
        self.assertEqual(parsed.vx_scale, 0.4)
        self.assertEqual(parsed.yaw_scale, 0.3)

    def test_viewer_build_planner_cfg_preserves_touchdown_reach_for_translation_commands(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        env_cfg = SimpleNamespace(
            gait_name="trot",
            step_freq=2.0,
            duty_factor=0.6,
            step_height=0.08,
            foothold_search_radius=0.15,
            foothold_search_step=0.03,
            max_step_down=float("inf"),
            max_roughness=0.5,
            replan_stop_speed=0.05,
            max_touchdown_xy_reach=0.22,
        )

        planner_cfg = module._build_planner_cfg(env_cfg)

        self.assertEqual(planner_cfg.max_touchdown_xy_reach, 0.22)

    def test_viewer_direct_playback_can_drive_robot_pose_from_planner_result(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        self.assertTrue(hasattr(module, "_apply_direct_playback_to_robot"))

        class FakeRobot:
            def __init__(self):
                self.root_pose_xyzw = None
                self.joint_pos = None
                self.joint_vel = None

            def write_root_pose_to_sim(self, root_pose_xyzw, env_ids=None):
                self.root_pose_xyzw = root_pose_xyzw.clone()

            def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
                self.joint_pos = joint_pos.clone()
                self.joint_vel = joint_vel.clone()

        robot = FakeRobot()
        fake_result = SimpleNamespace(
            root_pos_w=torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64),
            root_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64),  # wxyz identity
            joint_angles=torch.ones((1, 1, 12), dtype=torch.float64),
        )

        module._apply_direct_playback_to_robot(robot, fake_result, frame_idx=0)

        # Isaac Lab write_root_pose_to_sim expects quaternion in wxyz order.
        expected_root_pose = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        torch.testing.assert_close(robot.root_pose_xyzw, expected_root_pose)
        torch.testing.assert_close(robot.joint_pos, torch.ones((1, 12), dtype=torch.float32))
        torch.testing.assert_close(robot.joint_vel, torch.zeros((1, 12), dtype=torch.float32))

    def test_viewer_uses_plannerterrain_with_stable_scan_window(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")

        raw_ray_hits = torch.tensor(
            [
                [
                    [float("inf"), float("nan"), 1.0],
                    [10.0, 19.25, 1.1],
                    [10.75, 19.25, 1.2],
                    [float("inf"), float("nan"), 1.3],
                    [10.0, 20.0, 1.4],
                    [10.75, 20.0, 1.5],
                    [float("inf"), float("nan"), 1.6],
                    [10.0, 20.75, 1.7],
                    [10.75, 20.75, 1.8],
                ]
            ],
            dtype=torch.float32,
        )
        scanner = self._make_fake_scanner(raw_ray_hits)

        captured = {}
        sentinel_terrain = object()

        def fake_from_ray_hits(ray_hits, *, world_x_range=None, world_y_range=None):
            captured["ray_hits"] = ray_hits.clone()
            captured["world_x_range"] = world_x_range
            captured["world_y_range"] = world_y_range
            self.assertTrue(torch.isinf(ray_hits[0, 0, 0]))
            self.assertTrue(torch.isnan(ray_hits[0, 0, 1]))
            return sentinel_terrain

        with patch("extension.batched_planner.terrain.PlannerTerrain.from_ray_hits", side_effect=fake_from_ray_hits):
            terrain, returned_hits = module._compute_local_terrain(scanner, env_id=0)

        self.assertFalse(hasattr(module, "SingleTerrainAdapter"))
        self.assertIs(terrain, sentinel_terrain)
        self.assertEqual(returned_hits.shape, (9, 3))
        self.assertTrue(torch.isinf(returned_hits[0, 0]))
        self.assertTrue(torch.isnan(returned_hits[0, 1]))
        expected_hits = raw_ray_hits.to(torch.float64)
        finite_mask = torch.isfinite(expected_hits)
        self.assertTrue(torch.equal(torch.isfinite(captured["ray_hits"]), finite_mask))
        torch.testing.assert_close(captured["ray_hits"][finite_mask], expected_hits[finite_mask])
        self.assertEqual(captured["world_x_range"], (9.25, 10.75))
        self.assertEqual(captured["world_y_range"], (19.25, 20.75))

    def test_viewer_kinematic_planner_state_slices_reference_result(self):
        """Kinematic viewer replans from the last displayed frame via _planner_state_from_reference_result."""
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        self.assertTrue(hasattr(module, "_planner_state_from_reference_result"))

        fake_result = SimpleNamespace(
            root_pos_w=torch.tensor([[[0.0, 0.0, 0.1], [1.0, 2.0, 3.0]]], dtype=torch.float64),
            root_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64),
            joint_angles=torch.arange(24, dtype=torch.float64).reshape(1, 2, 12),
            foot_pos_w=torch.arange(24, dtype=torch.float64).reshape(1, 2, 4, 3) + 10.0,
        )
        state = module._planner_state_from_reference_result(fake_result, frame_idx=1)
        torch.testing.assert_close(state.root_pos, fake_result.root_pos_w[:, 1])
        torch.testing.assert_close(state.root_quat, fake_result.root_quat_w[:, 1])
        torch.testing.assert_close(state.joint_angles, fake_result.joint_angles[:, 1])
        torch.testing.assert_close(state.foot_pos, fake_result.foot_pos_w[:, 1])
        self.assertEqual(tuple(state.foot_vel.shape), tuple(state.foot_pos.shape))
        self.assertTrue(torch.all(state.foot_vel == 0))

    def test_viewer_planner_state_from_env_reorders_robot_joint_order_back_to_planner_order(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        robot = SimpleNamespace(
            joint_names=[
                "FL_hip_joint",
                "FR_hip_joint",
                "RL_hip_joint",
                "RR_hip_joint",
                "FL_thigh_joint",
                "FR_thigh_joint",
                "RL_thigh_joint",
                "RR_thigh_joint",
                "FL_calf_joint",
                "FR_calf_joint",
                "RL_calf_joint",
                "RR_calf_joint",
            ],
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float64),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
                joint_pos=torch.tensor([[0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]], dtype=torch.float64),
                body_pos_w=torch.zeros((1, 4, 3), dtype=torch.float64),
                body_lin_vel_w=torch.zeros((1, 4, 3), dtype=torch.float64),
            ),
        )
        env = SimpleNamespace(scene={"robot": robot})

        state = module._planner_state_from_env(env, [0, 1, 2, 3])

        torch.testing.assert_close(
            state.joint_angles,
            torch.arange(12, dtype=torch.float64).reshape(1, 12),
        )

    def test_reference_cache_requires_manager_owned_runtime_path(self):
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

    def test_reference_cache_does_not_fallback_to_existing_cache_without_manager(self):
        from extension.mdp import rewards_reference

        fake_cache = SimpleNamespace(is_ready=lambda: True)
        env = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=1,
            episode_length_buf=torch.tensor([0], dtype=torch.long),
            cfg=SimpleNamespace(reference_trajectory_horizon=5),
            unwrapped=SimpleNamespace(_trajectory_reference_cache=fake_cache),
        )

        with self.assertRaisesRegex(RuntimeError, "planner-owned reference cache"):
            rewards_reference.ensure_reference_cache(env)

    def test_reference_cache_is_filled_from_manager_not_placeholder_generator(self):
        from extension.mdp import rewards_reference

        cache = SimpleNamespace(is_ready=lambda: True)
        manager = SimpleNamespace(refresh_from_env=Mock(return_value=cache))
        env = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=1,
            episode_length_buf=torch.tensor([0], dtype=torch.long),
            cfg=SimpleNamespace(reference_trajectory_horizon=5),
            unwrapped=SimpleNamespace(_trajectory_manager=manager),
        )

        ensured = rewards_reference.ensure_reference_cache(env)

        self.assertIs(ensured, cache)
        manager.refresh_from_env.assert_called_once_with(env)

    def test_shared_runtime_manager_uses_single_batched_replan_for_multi_env(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        base_hits = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
                [2.0, 3.0, 0.0],
                [3.0, 3.0, 0.0],
            ],
            dtype=torch.float64,
        )
        raw_ray_hits = torch.stack([base_hits, base_hits + torch.tensor([10.0, 20.0, 0.0], dtype=torch.float64)])
        env = self._make_fake_viewer_env(
            episode_length_buf=torch.tensor([0, 0], dtype=torch.long),
            command=torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64),
            ray_hits=raw_ray_hits,
        )
        cfg = SimpleNamespace(
            reference_replan_interval_steps=50,
            reference_trajectory_horizon=5,
            dt=0.02,
            reference_command_name="base_velocity",
        )
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))

        with patch("extension.batched_planner.manager.batched_generate_trajectory", return_value=SimpleNamespace(
            num_frames=5,
            root_pos_w=torch.zeros((2, 5, 3), dtype=torch.float64),
            root_quat_w=torch.zeros((2, 5, 4), dtype=torch.float64),
            root_lin_vel_w=torch.zeros((2, 5, 3), dtype=torch.float64),
            root_ang_vel_w=torch.zeros((2, 5, 3), dtype=torch.float64),
            joint_angles=torch.zeros((2, 5, 12), dtype=torch.float64),
            foot_pos_w=torch.zeros((2, 5, 4, 3), dtype=torch.float64),
            foot_pos_root=torch.zeros((2, 5, 4, 3), dtype=torch.float64),
            contact_state=torch.zeros((2, 5, 4), dtype=torch.bool),
            body_pos_root=torch.zeros((2, 5, 12, 3), dtype=torch.float64),
            planned_touchdown_w=torch.zeros((2, 4, 3), dtype=torch.float64),
        )) as gen:
            cache = manager.refresh_from_env(env)

        self.assertTrue(cache.is_ready())
        gen.assert_called_once()

    def test_shared_runtime_manager_partial_replan_is_masked_and_cache_stays_full_shaped(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager
        from extension.mdp import rewards_reference

        num_envs = 2
        ray_hits = torch.zeros((num_envs, 16, 3), dtype=torch.float64)
        env = self._make_fake_viewer_env(
            episode_length_buf=torch.tensor([0, 0], dtype=torch.long),
            command=torch.zeros((num_envs, 3), dtype=torch.float64),
            ray_hits=ray_hits,
        )
        cfg = SimpleNamespace(
            reference_replan_interval_steps=50,
            reference_trajectory_horizon=5,
            dt=0.02,
            reference_command_name="base_velocity",
        )
        env.unwrapped._trajectory_manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))

        planner_batch_sizes: list[int] = []

        def make_result(n: int, h: int, *, offset: float):
            root_pos_w = torch.zeros((n, h, 3), dtype=torch.float64) + offset
            root_quat_w = torch.zeros((n, h, 4), dtype=torch.float64)
            root_quat_w[..., 0] = 1.0
            zeros = torch.zeros((n, h, 3), dtype=torch.float64)
            joint_angles = torch.zeros((n, h, 12), dtype=torch.float64) + offset
            foot_pos = torch.zeros((n, h, 4, 3), dtype=torch.float64) + offset
            contact_state = torch.zeros((n, h, 4), dtype=torch.bool)
            body_pos_root = torch.zeros((n, h, 12, 3), dtype=torch.float64) + offset
            touchdown = torch.zeros((n, 4, 3), dtype=torch.float64) + offset
            return SimpleNamespace(
                num_frames=h,
                root_pos_w=root_pos_w,
                root_quat_w=root_quat_w,
                root_lin_vel_w=zeros.clone(),
                root_ang_vel_w=zeros.clone(),
                joint_angles=joint_angles,
                foot_pos_w=foot_pos.clone(),
                foot_pos_root=foot_pos,
                contact_state=contact_state,
                body_pos_root=body_pos_root,
                planned_touchdown_w=touchdown,
            )

        def gen_side_effect(terrain, states, commands, requested_n_frames, dt, **kwargs):
            n = int(commands.shape[0])
            planner_batch_sizes.append(n)
            offset = 2000.0 if n == 1 else 0.0
            return make_result(n, int(requested_n_frames), offset=offset)

        with patch("extension.batched_planner.manager.PlannerTerrain.from_ray_hits", return_value=SimpleNamespace()), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            side_effect=gen_side_effect,
        ):
            cache0 = rewards_reference.ensure_reference_cache(env)
            cache0_root = cache0.root_pos_w.clone()

            # No changes: should not replan or mutate the cache.
            cache1 = rewards_reference.ensure_reference_cache(env)
            torch.testing.assert_close(cache1.root_pos_w, cache0_root)

            # Change only env0 command: replan should be masked to a single row but keep the full cache shape.
            env.episode_length_buf = torch.tensor([1, 1], dtype=torch.long)
            env.command_manager.command = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
            cache2 = rewards_reference.ensure_reference_cache(env)

        self.assertEqual(planner_batch_sizes, [2, 1])
        self.assertTrue(cache2.is_ready())
        self.assertTrue(cache2.is_canonical())
        self.assertEqual(tuple(cache2.root_pos_w.shape), (num_envs, 5, 3))
        self.assertFalse(torch.equal(cache2.root_pos_w[0], cache0_root[0]))
        torch.testing.assert_close(cache2.root_pos_w[1], cache0_root[1])


if __name__ == "__main__":
    unittest.main()
