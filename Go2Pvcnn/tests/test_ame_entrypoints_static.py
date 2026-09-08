from pathlib import Path


def test_ame_train_entrypoint_isolated_from_existing_experiments():
    source = Path("scripts/train_cross_large_complex_ame.py").read_text()
    assert "cross_large_complex_ame" in source
    assert "ActorCriticAME" in source
    assert "OnPolicyRunner" in source


def test_ame_headless_script_contains_headless_entrypoint():
    source = Path("scripts/train_cross_large_complex_ame_headless.sh").read_text()
    assert "train_cross_large_complex_ame.py" in source
    assert "--headless" in source
