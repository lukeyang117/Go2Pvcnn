from pathlib import Path


def test_launcher_requires_checkpoint_and_forwards_resume_flags():
    text = Path("Go2Pvcnn/scripts/train_parallelism_amp_cross_large_complex_headless.sh").read_text()
    assert "--resume" in text and "--keep_std" in text and "--load_checkpoint" in text
    assert "NUM_ENVS" in text and "MAX_ITERATIONS" in text
