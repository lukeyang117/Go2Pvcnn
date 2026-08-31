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


def test_amp_ppo_settings_match_pure_ppo_and_add_only_amp_channels():
    amp = get_train_cfg("parallelism_tracking_cross_large_complex_amp")
    pure = get_train_cfg("cross_large_complex_ppo")
    shared = (
        "num_learning_epochs",
        "num_mini_batches",
        "learning_rate",
        "clip_param",
        "gamma",
        "lam",
        "value_loss_coef",
        "entropy_coef",
        "max_grad_norm",
        "use_clipped_value_loss",
        "schedule",
        "desired_kl",
    )
    assert all(amp["algorithm"][key] == pure["algorithm"][key] for key in shared)
    assert amp["algorithm"]["amp_value_loss_coef"] == 1.0
    assert amp["algorithm"]["amp_warmup_iterations"] == 500
    assert amp["algorithm"]["amp_weight_ramp_iterations"] == 100
