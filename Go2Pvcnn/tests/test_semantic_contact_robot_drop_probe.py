from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


def test_summarize_semantic_contact_step_reports_active_envs() -> None:
    from Go2Pvcnn.tests.semantic_contact_robot_drop_probe import summarize_semantic_contact_step

    small = torch.zeros((3, 2, 4, 3), dtype=torch.float32)
    large = torch.zeros((3, 2, 5, 3), dtype=torch.float32)
    small[0, 1, 2, 0] = 6.0
    large[2, 0, 3, 1] = 11.0

    rows = summarize_semantic_contact_step(
        step=7,
        case_by_env=("small_drop", "empty", "large_drop"),
        body_names=("base", "FL_foot"),
        small_force_matrix_w=small,
        large_force_matrix_w=large,
        reward=torch.tensor([-0.1, 0.0, -0.2], dtype=torch.float32),
        force_threshold=1.0,
    )

    assert rows[0]["case"] == "small_drop"
    assert rows[0]["small_active_count"] == 1
    assert rows[0]["large_active_count"] == 0
    assert rows[0]["active_body_names"] == ["FL_foot"]
    assert rows[1]["case"] == "empty"
    assert rows[1]["small_active_count"] == 0
    assert rows[1]["large_active_count"] == 0
    assert rows[2]["case"] == "large_drop"
    assert rows[2]["small_active_count"] == 0
    assert rows[2]["large_active_count"] == 1
    assert rows[2]["active_body_names"] == ["base"]
    assert all(row["has_nan"] is False and row["has_inf"] is False for row in rows)


def test_semantic_contact_robot_drop_probe_real_isaaclab_small() -> None:
    from isaaclab.app import AppLauncher
    from Go2Pvcnn.tests.semantic_contact_robot_drop_probe import run_probe

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        summary = run_probe(num_envs=8, steps=80, drop_height=0.35, output=None)
    finally:
        simulation_app.close()

    assert summary["has_nan"] is False
    assert summary["has_inf"] is False
    assert summary["small_drop_hit_small"] is True
    assert summary["small_drop_hit_large"] is False
    assert summary["large_drop_hit_large"] is True
    assert summary["large_drop_hit_small"] is False
    assert summary["empty_hit_any"] is False
    assert summary["min_reward"] < 0.0
