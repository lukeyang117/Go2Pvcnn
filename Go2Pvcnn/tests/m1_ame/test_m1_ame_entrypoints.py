from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_m1_train_entrypoint_isolated_from_parallelism():
    source = (ROOT / "scripts/train_m1_cross_large_complex_ame.py").read_text(encoding="utf-8")

    assert "M1AmeCrossLargeComplexEnvCfg" in source
    assert "M1AmeOnPolicyRunner" in source
    assert "get_m1_ame_train_cfg" in source
    assert "Isaac-M1-Cross-Large-Complex-AME-v0" in source
    assert "AmeRslRlEnvWrapper" in source
    assert "attach_trajectory_manager" not in source
    assert "ParallelismReferenceManager" not in source


def test_m1_play_entrypoint_uses_m1_checkpoint_runner():
    source = (ROOT / "scripts/play_m1_cross_large_complex_ame.py").read_text(encoding="utf-8")

    assert "M1AmeCrossLargeComplexEnvCfg_PLAY" in source
    assert "M1AmeOnPolicyRunner" in source
    assert "get_m1_ame_train_cfg" in source
    assert "Isaac-M1-Cross-Large-Complex-AME-Play-v0" in source
    assert "act_inference" in source or "get_inference_policy" in source


def test_m1_entrypoints_do_not_mutate_go2_ame_files():
    train_source = (ROOT / "scripts/train_m1_cross_large_complex_ame.py").read_text(encoding="utf-8")
    play_source = (ROOT / "scripts/play_m1_cross_large_complex_ame.py").read_text(encoding="utf-8")

    assert "from ame_baseline.ame_env_cfg" not in train_source
    assert "runner = AmeOnPolicyRunner(" not in train_source
    assert "from ame_baseline.ame_env_cfg" not in play_source
    assert "runner = AmeOnPolicyRunner(" not in play_source
