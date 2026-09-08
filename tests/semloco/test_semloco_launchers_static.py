from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_train_launcher_selects_isolated_experiment():
    source = (ROOT / "Go2Pvcnn/scripts/train_cross_large_complex_semloco.py").read_text()
    assert '"--experiment", "cross_large_complex_semloco"' in source


def test_headless_launcher_forwards_runtime_controls():
    source = (ROOT / "Go2Pvcnn/scripts/train_cross_large_complex_semloco_headless.sh").read_text()
    assert "--num_envs" in source and "--max_iterations" in source and "--headless" in source


def test_play_launcher_uses_semloco_experiment():
    source = (ROOT / "Go2Pvcnn/scripts/play_cross_large_complex_semloco.py").read_text()
    assert '"--experiment", "cross_large_complex_semloco"' in source
