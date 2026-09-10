from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_shell_has_requested_warm_start_and_defaults():
    source = (ROOT / "scripts/train_cross_large_complex_ame_amp_headless.sh").read_text()
    assert "model_19999.pt" in source
    assert "MAX_ITERATIONS:-3500" in source
    assert "NUM_ENVS:-1024" in source
    assert "parallelism_tracking_cross_large_complex_ame_amp" in source
    assert "--keep_std" in source


def test_train_script_registers_only_runtime_experiment():
    source = (ROOT / "scripts/train_cross_large_complex_ame_amp.py").read_text()
    assert "gym.register" in source
    assert "AmeParallelismAmpCrossLargeComplexEnvCfg" in source
    assert "AmeAmpOnPolicyRunner" in source
    assert "attach_trajectory_manager_if_enabled" in source
    assert "finally:" in source and "simulation_app.close()" in source


def test_play_script_requires_complete_checkpoint():
    source = (ROOT / "scripts/play_cross_large_complex_ame_amp.py").read_text()
    assert "load_amp_checkpoint" in source
    assert "full_amp_resume" in source
    assert "torch.no_grad" in source
    assert "finally:" in source
