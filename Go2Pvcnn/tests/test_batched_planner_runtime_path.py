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
        env_cfg = SimpleNamespace(sim=SimpleNamespace(device="cpu"), planner_owned_reference_cache=True)
        sentinel_manager = SimpleNamespace(refresh_from_env=SimpleNamespace())

        with patch("extension.batched_planner.manager.BatchedTrajectoryManager", return_value=sentinel_manager) as ctor:
            module._attach_reference_manager_if_enabled(env, env_cfg, "teacher_elevation_trajectory")

        ctor.assert_called_once_with(env_cfg, device=torch.device("cpu"))
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

    def test_viewer_parser_includes_app_launcher_args(self):
        with _fake_isaaclab_app("--viewer-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
            parser = module.build_arg_parser()
            parsed = parser.parse_args(["--viewer-launcher-flag"])

        self.assertTrue(parsed.viewer_launcher_flag)

    def test_viewer_parser_accepts_explicit_planner_playback_mode_flag(self):
        with _fake_isaaclab_app("--viewer-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
            parser = module.build_arg_parser()
            try:
                parsed = parser.parse_args(
                    ["--planner-playback-mode", "direct", "--viewer-launcher-flag"]
                )
            except SystemExit as exc:  # argparse uses SystemExit for parse errors
                self.fail(f"viewer parser rejected --planner-playback-mode direct: {exc}")

        self.assertEqual(parsed.planner_playback_mode, "direct")

    def test_viewer_planner_playback_mode_resolves_distinctly_from_physics_default(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        self.assertTrue(hasattr(module, "_resolve_planner_playback_mode"))

        direct = module._resolve_planner_playback_mode(SimpleNamespace(planner_playback_mode="direct"))
        physics = module._resolve_planner_playback_mode(SimpleNamespace(planner_playback_mode="physics"))
        default = module._resolve_planner_playback_mode(SimpleNamespace())

        self.assertEqual(direct, "direct")
        self.assertEqual(physics, "physics")
        self.assertEqual(default, "physics")

    def test_viewer_direct_playback_uses_reference_cache_state_not_env_state(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        self.assertTrue(hasattr(module, "_get_viewer_planner_state"))

        fake_result = SimpleNamespace(
            root_pos_w=torch.tensor([[[0.0, 0.0, 0.25], [1.0, 2.0, 3.0]]], dtype=torch.float64),
            root_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64),
            joint_angles=torch.zeros((1, 2, 12), dtype=torch.float64),
            foot_pos_w=torch.zeros((1, 2, 4, 3), dtype=torch.float64),
        )

        with patch.object(
            module,
            "_planner_state_from_env",
            side_effect=AssertionError("direct playback should not read env state"),
        ):
            planner_state = module._get_viewer_planner_state(
                SimpleNamespace(),
                foot_ids=[0, 1, 2, 3],
                playback_mode="direct",
                result=fake_result,
                frame_idx=1,
            )

        torch.testing.assert_close(planner_state.root_pos, fake_result.root_pos_w[:, 1])
        torch.testing.assert_close(planner_state.root_quat, fake_result.root_quat_w[:, 1])

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

    def test_viewer_shared_runtime_manager_replans_on_reset_and_command_change(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        from extension.mdp import rewards_reference

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
        env = self._make_fake_viewer_env(
            episode_length_buf=torch.tensor([4], dtype=torch.long),
            command=torch.zeros((1, 3), dtype=torch.float64),
            ray_hits=raw_ray_hits,
        )
        planner_cfg = SimpleNamespace(
            reference_replan_interval_steps=50,
            reference_trajectory_horizon=5,
            dt=0.02,
            reference_command_name="base_velocity",
        )
        module._attach_viewer_reference_manager(env, planner_cfg)

        fake_cache = SimpleNamespace(
            is_ready=lambda: True,
            horizon_length=lambda: 5,
            root_pos_w=torch.zeros((1, 5, 3), dtype=torch.float64),
            root_quat_w=torch.zeros((1, 5, 4), dtype=torch.float64),
            joint_angles=torch.zeros((1, 5, 12), dtype=torch.float64),
            foot_pos_root=torch.zeros((1, 5, 4, 3), dtype=torch.float64),
            foot_pos_w=torch.zeros((1, 5, 4, 3), dtype=torch.float64),
            contact_state=torch.zeros((1, 5, 4), dtype=torch.bool),
            body_pos_root=torch.zeros((1, 5, 12, 3), dtype=torch.float64),
            planned_touchdown_w=torch.zeros((1, 4, 3), dtype=torch.float64),
            phase_index=torch.zeros((1, 5), dtype=torch.int64),
            valid_mask=torch.ones((1, 5), dtype=torch.bool),
        )
        fake_result = SimpleNamespace(
            num_frames=5,
            root_pos_w=fake_cache.root_pos_w,
            root_quat_w=fake_cache.root_quat_w,
            root_lin_vel_w=torch.zeros((1, 5, 3), dtype=torch.float64),
            root_ang_vel_w=torch.zeros((1, 5, 3), dtype=torch.float64),
            joint_angles=fake_cache.joint_angles,
            foot_pos_w=fake_cache.foot_pos_w,
            foot_pos_root=fake_cache.foot_pos_root,
            contact_state=fake_cache.contact_state,
            body_pos_root=fake_cache.body_pos_root,
            planned_touchdown_w=fake_cache.planned_touchdown_w,
        )

        with patch("extension.batched_planner.trajectory.batched_generate_trajectory", side_effect=AssertionError("viewer should not call the raw trajectory entrypoint")), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            return_value=fake_result,
        ) as gen:
            ensured_1 = rewards_reference.ensure_reference_cache(env)
            ensured_2 = rewards_reference.ensure_reference_cache(env)
            env.command_manager.command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float64)
            ensured_3 = rewards_reference.ensure_reference_cache(env)
            env.episode_length_buf = torch.tensor([0], dtype=torch.long)
            ensured_4 = rewards_reference.ensure_reference_cache(env)

        self.assertIs(ensured_1, ensured_2)
        self.assertIs(env.unwrapped._trajectory_reference_cache, ensured_4)
        self.assertEqual(gen.call_count, 3)
        self.assertTrue(torch.allclose(env.command_manager.get_command("base_velocity"), torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float64)))
        self.assertTrue(ensured_4.is_ready())

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


if __name__ == "__main__":
    unittest.main()
