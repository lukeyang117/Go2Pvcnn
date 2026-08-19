from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_distillation_wrapper_exports_separate_teacher_and_critic_observations() -> None:
    source = (ROOT / "scripts/train.py").read_text()
    assert '"teacher": teacher_obs' in source
    assert '"critic": critic_obs' in source


def test_distillation_config_defines_legacy_critic_groups() -> None:
    source = (ROOT / "tracking/parallelism_cross_large_complex_distillation_env_cfg.py").read_text()
    assert "CriticElevationSemanticMapCfg" in source
    assert "CriticStateCfg" in source
    assert "velocity_commands = None" in source or "velocity_commands" in source


def test_resume_launcher_explicitly_selects_student_and_teacher_checkpoints() -> None:
    source = (ROOT / "scripts/train_parallelism_large_obstacles_rl_headless_distilation_resume.sh").read_text()
    assert "--load_run 2026-08-18_21-38-36/6def073" in source
    assert "--load_checkpoint model_5499.pt" in source
    assert "--teacher_checkpoint" in source
    assert "2026-08-18_20-30-59/6def073/model_4999.pt" in source


def test_fresh_launcher_uses_the_new_teacher_checkpoint() -> None:
    source = (ROOT / "scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh").read_text()
    assert "2026-08-18_20-30-59/6def073/model_4999.pt" in source


def test_resume_flow_prefers_explicit_teacher_checkpoint() -> None:
    source = (ROOT / "scripts/train.py").read_text()
    assert "load_student_checkpoint" in source
    assert "args_cli.teacher_checkpoint" in source
    assert "runner.load_teacher(teacher_checkpoint" in source
