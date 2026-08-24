import ast
from pathlib import Path


def test_distillation_env_cfg_static_contract():
    source = Path("Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py").read_text()

    assert "ParallelismTrackingCrossLargeComplexEnvCfg" in source
    assert "parallelism_tracking_cross_large_complex_distillation" in source
    assert "parallelism_consecutive_standstill" in source
    assert "parallelism_geometry_collision" in source
    assert "ParallelismTrackingObservationsCfg.PolicyStateCfg" in source
    assert "student_state" in source
    assert "teacher_state" in source
    assert "DistillationContextCfg" in source
    assert "distillation_context" in source
    assert "parallelism_plan_valid = ObsTerm" in source
    assert "parallelism_plan_valid = None" in source
    assert "parallelism_ref_joint_pos = None" in source
    assert "parallelism_ref_joint_vel = None" in source
    assert "parallelism_ref_root_pos = None" in source
    assert "parallelism_ref_root_rot = None" in source
    assert "base_lin_vel = None" in source
    assert "track_lin_vel_xy" in source
    assert "track_ang_vel_z" in source
    assert "resampling_time_range = (10.0, 10.0)" in source
    assert "resampling_time_range = (100.0, 100.0)" not in source
    assert "ranges.lin_vel_y = (-0.1, 0.1)" in source
    assert "ranges.ang_vel_z = (-1.0, 1.0)" in source


def test_distillation_experiment_registered_static():
    expected = "parallelism_tracking_cross_large_complex_distillation"

    assert expected in Path("Go2Pvcnn/tracking/register_envs.py").read_text()
    assert expected in Path("Go2Pvcnn/scripts/train.py").read_text()
    assert expected in Path("Go2Pvcnn/scripts/play.py").read_text()
    assert expected in Path("Go2Pvcnn/agent/train_cfg.py").read_text()


def test_distillation_train_cfg_static():
    tree = ast.parse(Path("Go2Pvcnn/agent/train_cfg.py").read_text())
    source = ast.unparse(tree)

    assert "HybridDistillationPPO" in source
    assert "StudentTeacherCNN" in source
    assert "student_elevation_semantic_map" in source
    assert "teacher_elevation_semantic_map" in source
    assert "ppo_coef" in source
    assert "teacher_coef" in source
    assert "entropy_coef" in source
    assert "init_noise_std" in source
