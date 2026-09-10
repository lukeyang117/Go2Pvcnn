from __future__ import annotations

from pathlib import Path

from ame_baseline.ame_amp_train_cfg import get_ame_amp_train_cfg
from ame_baseline.ame_train_cfg import get_ame_train_cfg


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_config_preserves_ame_observations_and_amp_fields():
    source = (PACKAGE_ROOT / "ame_baseline/ame_amp_env_cfg.py").read_text()
    assert "class AmeParallelismAmpCrossLargeComplexEnvCfg(AmeCrossLargeComplexEnvCfg)" in source
    assert 'AME_AMP_EXPERIMENT_NAME = "parallelism_tracking_cross_large_complex_ame_amp"' in source
    assert "planner_owned_reference_cache: bool = True" in source
    assert "parallelism_plan_batch_size: int = 1024" in source
    assert "amp_window_frames: int = 24" in source
    assert "amp_dt: float = 0.02" in source
    assert 'AME_AMP_ENV_ID = "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0"' in source

    train = get_ame_amp_train_cfg()
    assert train["policy"]["class_name"] == "AmpActorCriticAME"
    assert train["algorithm"]["class_name"] == "ParallelismAMPPPO"
    assert train["num_steps_per_env"] == 40
    assert train["algorithm"]["schedule"] == "adaptive"
    assert train["algorithm"]["amp_window_frames"] == 24


def test_amp_config_does_not_mutate_ame_config():
    assert get_ame_train_cfg()["policy"]["class_name"] == "ActorCriticAME"
    assert get_ame_train_cfg()["algorithm"]["class_name"] == "PPO"


def test_shared_ppo_values_match_ame_baseline():
    amp = get_ame_amp_train_cfg()
    ame = get_ame_train_cfg()
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
    assert all(amp["algorithm"][key] == ame["algorithm"][key] for key in shared)
