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


def test_joint_mpc_viewer_exposes_nominal_only_mode():
    source = VIEWER_FILE.read_text(encoding="utf-8")

    assert '"--joint-mpc-nominal-only"' in source
    assert "runtime.nominal_only" in source


def test_viewer_exposes_parallelism_backend():
    source = VIEWER_FILE.read_text(encoding="utf-8")

    assert '"parallelism"' in source
    assert "parallelism_trajectory_to_viewer_result" in source


def test_parallelism_viewer_draws_candidate_circle_markers():
    source = VIEWER_FILE.read_text(encoding="utf-8")

    assert "parallelism_candidate_circle" in source
    assert "/Visuals/Parallelism/candidate_circle_" in source


def test_parallelism_viewer_prints_reject_diagnostics():
    source = VIEWER_FILE.read_text(encoding="utf-8")

    assert "_format_parallelism_reject_diagnostics" in source
    assert "parallelism_reject(" in source
    assert "collision_detail(" in source
    assert "print(\n                        _format_viewer_plan_line(" in source
