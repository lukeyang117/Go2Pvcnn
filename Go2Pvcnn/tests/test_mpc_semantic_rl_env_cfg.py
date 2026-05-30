from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = REPO_ROOT / "Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py"


def test_mpc_semantic_cfg_has_one_body_filtered_contact_sensors() -> None:
    source = CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    body_names = (
        "FL_foot",
        "FR_foot",
        "RL_foot",
        "RR_foot",
        "FL_calf",
        "FR_calf",
        "RL_calf",
        "RR_calf",
        "FL_thigh",
        "FR_thigh",
        "RL_thigh",
        "RR_thigh",
        "base",
    )

    assert "SEMANTIC_CONTACT_BODY_NAMES" in source
    assert "SEMANTIC_CONTACT_BODY_WEIGHTS" in source
    assert "semantic_filtered_contact_collision_reward" in source
    scene_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TeacherElevationTrajectoryMpcSemanticSceneCfg")
    assignments = {
        target.id
        for node in scene_class.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for body in body_names:
        assert f"semantic_contact_{body}_small" in assignments
        assert f"semantic_contact_{body}_large" in assignments
        assert f'_semantic_contact_sensor("{body}", SEMANTIC_COURSE_SMALL_ROOT)' in source
        assert f'_semantic_contact_sensor("{body}", SEMANTIC_COURSE_LARGE_ROOT)' in source
    assert 'prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}"' in source
    assert 'filter_prim_paths_expr=[f"{semantic_root}/.*"]' in source
    assert "swing_leg_collision = None" in source
