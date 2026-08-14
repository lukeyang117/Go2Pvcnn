import ast
from pathlib import Path


def test_distillation_env_cfg_static_contract():
    source = Path("tracking/parallelism_cross_large_complex_distillation_env_cfg.py").read_text()

    assert "ParallelismTrackingCrossLargeComplexEnvCfg" in source
    assert "parallelism_tracking_cross_large_complex_distillation" in source
    assert "parallelism_consecutive_standstill" in source
    assert "parallelism_geometry_collision" in source
    assert "student_state" in source
    assert "teacher_state" in source
    assert "parallelism_ref_joint_pos = None" in source
    assert "parallelism_ref_joint_vel = None" in source
    assert "parallelism_ref_root_pos = None" in source
    assert "parallelism_ref_root_rot = None" in source


def test_distillation_experiment_registered_static():
    expected = "parallelism_tracking_cross_large_complex_distillation"

    assert expected in Path("tracking/register_envs.py").read_text()
    assert expected in Path("scripts/train.py").read_text()
    assert expected in Path("scripts/play.py").read_text()
    assert expected in Path("agent/train_cfg.py").read_text()


def test_distillation_train_cfg_static():
    tree = ast.parse(Path("agent/train_cfg.py").read_text())
    source = ast.unparse(tree)

    assert "Distillation" in source
    assert "StudentTeacherCNN" in source
    assert "student_elevation_semantic_map" in source
    assert "teacher_elevation_semantic_map" in source
