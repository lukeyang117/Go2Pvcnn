from __future__ import annotations

import ast
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
CFG_PATH = REPO_ROOT / "Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


def _install_fake_isaaclab_contact_sensor(monkeypatch):
    for name in (
        "go2_pvcnn.sensor.semantic_contacter",
        "go2_pvcnn.sensor.semantic_contacter.semantic_global_contact_sensor",
    ):
        sys.modules.pop(name, None)

    class ContactSensor:
        pass

    isaaclab_module = types.ModuleType("isaaclab")
    sensors_module = types.ModuleType("isaaclab.sensors")
    sensors_module.ContactSensor = ContactSensor
    isaaclab_module.sensors = sensors_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors_module)
    return ContactSensor


def test_semantic_global_contact_sensor_importable(monkeypatch) -> None:
    ContactSensor = _install_fake_isaaclab_contact_sensor(monkeypatch)

    from go2_pvcnn.sensor.semantic_contacter import SemanticGlobalContactSensor

    assert issubclass(SemanticGlobalContactSensor, ContactSensor)


def test_semantic_global_contact_leaf_filter_keeps_only_slots(monkeypatch) -> None:
    _install_fake_isaaclab_contact_sensor(monkeypatch)
    from go2_pvcnn.sensor.semantic_contacter.semantic_global_contact_sensor import filter_semantic_leaf_obstacle_paths

    paths = [
        "/World/semantic_course/small/row_00",
        "/World/semantic_course/small/row_00/col_00",
        "/World/semantic_course/small/row_00/col_00/slot_00",
        "/World/semantic_course/small/row_00/col_01/slot_01",
        "/World/semantic_course/large/row_00/col_00/slot_00",
    ]

    assert filter_semantic_leaf_obstacle_paths(paths, "/World/semantic_course/small") == [
        "/World/semantic_course/small/row_00/col_00/slot_00",
        "/World/semantic_course/small/row_00/col_01/slot_01",
    ]


def test_mpc_semantic_cfg_uses_two_global_semantic_contact_sensors() -> None:
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
    assert "SemanticGlobalContactSensor" in source
    assert "semantic_global_contact_collision_reward" in source
    scene_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TeacherElevationTrajectoryMpcSemanticSceneCfg")
    assignments = {
        target.id
        for node in scene_class.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "semantic_contact_small" in assignments
    assert "semantic_contact_large" in assignments
    for body in body_names:
        assert f"semantic_contact_{body}_small" not in assignments
        assert f"semantic_contact_{body}_large" not in assignments
    assert "class_type=SemanticGlobalContactSensor" in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/.*"' in source
    assert 'filter_prim_paths_expr=[f"{semantic_root}/.*"]' in source
    assert '"small_sensor_cfg": SceneEntityCfg("semantic_contact_small")' in source
    assert '"large_sensor_cfg": SceneEntityCfg("semantic_contact_large")' in source
    assert "swing_leg_collision = None" in source
