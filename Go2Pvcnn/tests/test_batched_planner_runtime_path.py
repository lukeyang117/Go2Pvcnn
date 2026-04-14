import importlib
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fresh_import(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class BatchedPlannerRuntimePathTest(unittest.TestCase):
    def test_train_module_is_import_safe(self):
        module = _fresh_import("Go2Pvcnn.scripts.train")
        self.assertTrue(hasattr(module, "build_arg_parser"))

    def test_viewer_module_is_import_safe(self):
        module = _fresh_import("Go2Pvcnn.extension.viz.go2_foostep_planner")
        self.assertTrue(hasattr(module, "build_arg_parser"))


if __name__ == "__main__":
    unittest.main()
