from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = REPO_ROOT / "Go2Pvcnn" / "go2_pvcnn" / "tasks" / "teacher_elevation_trajectory_semantic_viewer_env_cfg.py"


def _parse_module() -> ast.Module:
    return ast.parse(CFG_PATH.read_text(encoding="utf-8"))


def _class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Missing class {name} in {CFG_PATH}.")


def _assignment_value(class_def: ast.ClassDef, target_name: str):
    for node in class_def.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == target_name:
            return node.value
    raise AssertionError(f"Missing assignment for {target_name} in class {class_def.name}.")


def test_scene_replaces_height_scanner_with_semantic_scanner_and_disables_replication():
    tree = _parse_module()
    scene_cls = _class_def(tree, "TeacherElevationTrajectorySemanticViewerSceneCfg")

    replicate_value = _assignment_value(scene_cls, "replicate_physics")
    assert isinstance(replicate_value, ast.Constant)
    assert replicate_value.value is False

    height_scanner_value = _assignment_value(scene_cls, "height_scanner")
    assert isinstance(height_scanner_value, ast.Constant)
    assert height_scanner_value.value is None

    semantic_scanner_value = _assignment_value(scene_cls, "semantic_height_scanner")
    assert isinstance(semantic_scanner_value, ast.Call)
    assert isinstance(semantic_scanner_value.func, ast.Name)
    assert semantic_scanner_value.func.id == "SemanticGridRayCasterCfg"


def test_semantic_scanner_roots_and_ids_match_course_contract():
    source = CFG_PATH.read_text(encoding="utf-8")

    assert 'mesh_prim_paths=["/World/ground", SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT]' in source
    assert '"/World/ground": 0' in source
    assert "SEMANTIC_COURSE_SMALL_ROOT: 1" in source
    assert "SEMANTIC_COURSE_LARGE_ROOT: 2" in source


def test_observation_and_planner_references_point_to_semantic_height_scanner():
    source = CFG_PATH.read_text(encoding="utf-8")

    assert source.count('SceneEntityCfg("semantic_height_scanner")') >= 2
    assert 'reference_height_scanner_name: str = "semantic_height_scanner"' in source


def test_prestartup_semantic_course_event_is_wired():
    source = CFG_PATH.read_text(encoding="utf-8")

    assert "spawn_semantic_course_prestartup" in source
    assert 'mode="prestartup"' in source
    assert 'params={"default_stage": DEFAULT_VIEWER_REPRESENTATIVE_STAGE.value}' in source
