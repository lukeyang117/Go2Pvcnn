from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_m1_asset_and_viewer_cfg_use_m1_names():
    asset = (ROOT / "go2_pvcnn/assets/m1.py").read_text()
    env = (ROOT / "tracking/m1_parallelism_viewer_env_cfg.py").read_text()

    assert "ZJ_V3_URDF_V1_0.usd" in asset
    assert "FBL_ABAD" in asset and "FBL_FOOT_JOINT" in asset
    assert "BASE_LINK" in env
    assert all(name in env for name in ("FBL_FOOT_LINK", "FAR_FOOT_LINK", "RBL_FOOT_LINK", "RAR_FOOT_LINK"))
    assert "reference_robot" not in env
    assert "M1_CFG" in env
    assert "SemanticCourseTerrainImporter" in env
    assert "class_type = SemanticCourseTerrainImporter" in env


def test_m1_viewer_cfg_declares_16_joint_action_and_no_go2_body_selectors():
    source = (ROOT / "tracking/m1_parallelism_viewer_env_cfg.py").read_text()

    assert "joint_names=[\".*\"]" in source
    assert "action_dim" not in source
    assert "body_names=\".*_foot\"" not in source
    assert "body_names=\"base\"" not in source
