from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.parallelism.offline_obstacle_diagnostics import (  # noqa: E402
    build_terrain,
    load_scene,
    obstacle_geometry,
    run_scene,
    sweep_scene,
)
from extension.parallelism.terrain import query_height_semantic_valid  # noqa: E402


SCENE_FILE = Path(__file__).with_name("offline_obstacle_scenes.yaml")


def test_yaml_scene_loads_failure_case():
    scene = load_scene(SCENE_FILE, "front_center_high_small")
    assert scene.obstacle.center_w == pytest.approx((0.10, -0.16))
    assert scene.obstacle.radius_m == pytest.approx(0.12)
    assert scene.obstacle.height_m == pytest.approx(0.20)
    assert scene.root.rpy_w == pytest.approx((0.0, 0.0, 0.0))
    assert len(scene.joint_pos) == 12


def test_scene_builds_known_semantic_and_height_map():
    scene = load_scene(SCENE_FILE, "front_center_high_small")
    terrain = build_terrain(scene, torch.device("cpu"))
    query = query_height_semantic_valid(
        terrain,
        torch.tensor([[[0.10, -0.16]]], dtype=torch.float32),
    )
    assert query.semantic.item() == 1
    assert query.height.item() == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("command_index", "expected_per_leg_valid"),
    ((0, [0, 28, 41, 0]), (1, [0, 35, 39, 0])),
)
def test_front_center_obstacle_reproduces_parallelism_standstill(command_index, expected_per_leg_valid):
    scene = load_scene(SCENE_FILE, "front_center_high_small")
    report = run_scene(scene, command_index=command_index, device=torch.device("cpu"))
    assert report["standstill"] is True
    assert report["per_leg_valid"] == expected_per_leg_valid
    assert report["obstacle"]["center_root"] == pytest.approx([0.10, -0.16, -0.40], abs=1e-5)


def test_obstacle_geometry_and_pose_are_reported_in_root_frame():
    scene = load_scene(SCENE_FILE, "front_left_offset")
    report = run_scene(scene, device=torch.device("cpu"))
    assert report["obstacle"]["center_root"][0] == pytest.approx(0.220, abs=1e-3)
    assert report["obstacle"]["center_root"][1] == pytest.approx(0.078, abs=1e-3)
    assert len(report["current_foot_pos_w"]) == 4
    assert len(report["joint_pos"]) == 12


def test_scene_sweep_accepts_different_obstacle_positions_and_root_yaw():
    scene = load_scene(SCENE_FILE, "front_center_high_small")
    reports = sweep_scene(
        scene,
        {
            "obstacle.center_w": [[0.20, 0.0], [0.20, 0.12]],
            "root.rpy_w": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
        },
        device=torch.device("cpu"),
    )
    assert len(reports) == 4
    assert all("center_root" in report["obstacle"] for report in reports)


def test_failure_snapshot_records_full_scene(tmp_path):
    scene = load_scene(SCENE_FILE, "front_center_high_small")
    report = run_scene(scene, device=torch.device("cpu"), snapshot_dir=tmp_path)
    snapshot = Path(report["snapshot_path"])
    assert snapshot.parent == tmp_path
    payload = snapshot.read_text()
    assert "center_w:" in payload
    assert "joint_pos:" in payload
    assert "per_leg_valid:" in payload
