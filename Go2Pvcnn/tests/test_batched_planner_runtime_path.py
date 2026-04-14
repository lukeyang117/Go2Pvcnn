import importlib
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

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

    def test_viewer_module_is_import_safe(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        self.assertTrue(hasattr(module, "build_arg_parser"))

    def test_viewer_parser_includes_app_launcher_args(self):
        with _fake_isaaclab_app("--viewer-launcher-flag"):
            module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
            parser = module.build_arg_parser()
            parsed = parser.parse_args(["--viewer-launcher-flag"])

        self.assertTrue(parsed.viewer_launcher_flag)

    def test_viewer_uses_plannerterrain_for_local_terrain(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")

        raw_ray_hits = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, float("inf")],
                [0.0, 1.0, float("nan")],
                [1.0, 1.0, 2.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        class FakeScanner:
            def __init__(self, hits):
                self.data = SimpleNamespace(ray_hits_w=hits)

        captured = {}
        sentinel_terrain = object()

        def fake_from_ray_hits(ray_hits, *, world_x_range=None, world_y_range=None):
            captured["ray_hits"] = ray_hits.clone()
            captured["world_x_range"] = world_x_range
            captured["world_y_range"] = world_y_range
            self.assertTrue(torch.isnan(ray_hits[0, 2, 2]))
            self.assertTrue(torch.isinf(ray_hits[0, 1, 2]))
            return sentinel_terrain

        with patch("extension.batched_planner.terrain.PlannerTerrain.from_ray_hits", side_effect=fake_from_ray_hits):
            terrain, returned_hits = module._compute_local_terrain(FakeScanner(raw_ray_hits), env_id=0)

        self.assertFalse(hasattr(module, "SingleTerrainAdapter"))
        self.assertIs(terrain, sentinel_terrain)
        self.assertEqual(returned_hits.shape, (4, 3))
        self.assertTrue(torch.isnan(returned_hits[2, 2]))
        self.assertTrue(torch.isinf(returned_hits[1, 2]))
        expected_hits = raw_ray_hits.to(torch.float64)
        finite_mask = torch.isfinite(expected_hits)
        self.assertTrue(torch.equal(torch.isfinite(captured["ray_hits"]), finite_mask))
        torch.testing.assert_close(captured["ray_hits"][finite_mask], expected_hits[finite_mask])
        self.assertIsNone(captured["world_x_range"])
        self.assertIsNone(captured["world_y_range"])


if __name__ == "__main__":
    unittest.main()
