from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWER_FILE = REPO_ROOT / "Go2Pvcnn" / "extension" / "viz" / "go2_foostep_planner.py"


def test_viewer_adds_go2pvcnn_root_before_extension_imports():
    source = VIEWER_FILE.read_text(encoding="utf-8")
    path_insert = "sys.path.insert(0, str(GO2PVCNN_ROOT))"
    extension_import = "from extension.batch_mpc_planner.planner import plan_segment"

    assert path_insert in source
    assert extension_import in source
    assert source.index(path_insert) < source.index(extension_import)
