from agent import get_train_cfg


def test_amp_config_is_additive_and_old_configs_keep_classes():
    amp = get_train_cfg("parallelism_tracking_cross_large_complex_amp")
    pure = get_train_cfg("cross_large_complex_ppo")
    distill = get_train_cfg("parallelism_tracking_cross_large_complex_distillation")
    assert amp["algorithm"]["class_name"] == "ParallelismAMPPPO"
    assert amp["policy"]["class_name"] == "AmpActorCriticCNN"
    assert pure["algorithm"]["class_name"] == "PPO"
    assert distill["algorithm"]["class_name"] == "HybridDistillationPPO"


def test_amp_experiment_uses_new_name_only():
    amp = get_train_cfg("parallelism_tracking_cross_large_complex_amp")
    assert amp["experiment_name"] == "parallelism_tracking_cross_large_complex_amp"

