from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_smoke_uses_1024_envs_and_four_iterations():
    source = (PACKAGE_ROOT / "tests/ame_baseline/run_ame_amp_isaac_smoke.sh").read_text()
    assert "NUM_ENVS=1024" in source
    assert "MAX_ITERATIONS=4" in source
    assert "train_cross_large_complex_ame_amp_headless.sh" in source
    assert "AME_AMP_OUTPUT_ROOT" in source


def test_training_script_supports_isolated_output_root():
    source = (PACKAGE_ROOT / "scripts/train_cross_large_complex_ame_amp.py").read_text()
    assert "AME_AMP_OUTPUT_ROOT" in source
    assert "source_checkpoint.json" in source
