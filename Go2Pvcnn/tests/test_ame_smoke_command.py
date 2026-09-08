from pathlib import Path


def test_ame_smoke_command_is_documented():
    source = Path("scripts/train_cross_large_complex_ame_headless.sh").read_text()
    assert "NUM_ENVS" in source
    assert "MAX_ITERATIONS" in source
