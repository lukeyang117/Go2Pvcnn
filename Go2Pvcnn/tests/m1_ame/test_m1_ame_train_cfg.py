from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "m1_ame"


def test_m1_train_cfg_matches_go2_ame_algorithm_and_network_values():
    from ame_baseline.ame_train_cfg import get_ame_train_cfg
    from m1_ame.m1_ame_train_cfg import get_m1_ame_train_cfg

    assert get_m1_ame_train_cfg() == get_ame_train_cfg()


def test_m1_runner_uses_m1_dimension_signature():
    source = (PACKAGE / "m1_ame_runner.py").read_text(encoding="utf-8")

    assert "m1_ame_xyz_semantic_v1_c6_s16_mha64_h16" in source
    assert "M1_AME_ACTION_DIM" in source
    assert "M1_AME_POLICY_OBS_DIM" in source
    assert "M1_AME_CRITIC_OBS_DIM" in source
    assert "expected_dims" in source
    assert "ParallelismAMPPPO" not in source


def test_m1_train_cfg_does_not_change_algorithm_family():
    source = (PACKAGE / "m1_ame_train_cfg.py").read_text(encoding="utf-8")

    assert "get_ame_train_cfg" in source
    assert "class_name" not in source
    assert "AMP" not in source
    assert "parallelism" not in source.lower()
