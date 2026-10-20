from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "m1_ame"


def test_m1_train_cfg_matches_go2_ame_algorithm_and_network_values():
    from ame_baseline.ame_train_cfg import get_ame_train_cfg
    from m1_ame.m1_ame_train_cfg import get_m1_ame_train_cfg

    go2 = get_ame_train_cfg()
    m1 = get_m1_ame_train_cfg()
    assert {key: value for key, value in m1["algorithm"].items() if key != "normalize_kl_by_action_dim"} == go2[
        "algorithm"
    ]
    assert {key: value for key, value in m1["policy"].items() if key != "init_noise_std"} == {
        key: value for key, value in go2["policy"].items() if key != "init_noise_std"
    }
    assert m1["algorithm"]["normalize_kl_by_action_dim"] is True
    assert len(m1["policy"]["init_noise_std"]) == 16
    assert m1["policy"]["init_noise_std"][3] == 0.10


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
