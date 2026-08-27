from pathlib import Path


def test_amp_runtime_probe_targets_real_1024_env_launcher():
    source = Path("Go2Pvcnn/tests/tracking/parallelism_amp_training_smoke_probe.py").read_text()
    assert "train_parallelism_amp_cross_large_complex_headless.sh" in source
    assert "NUM_ENVS" in source and "MAX_ITERATIONS" in source
    assert "set(range(args.max_iterations))" in source
