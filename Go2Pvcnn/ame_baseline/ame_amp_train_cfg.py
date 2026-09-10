"""Training configuration for the AME + Parallelism AMP experiment."""

from __future__ import annotations

import copy

from .ame_train_cfg import get_ame_train_cfg


def get_ame_amp_train_cfg() -> dict:
    """Return an independent AME-AMP config without mutating the AME baseline."""

    cfg = copy.deepcopy(get_ame_train_cfg())
    cfg["experiment_name"] = "parallelism_tracking_cross_large_complex_ame_amp"
    cfg["algorithm"].update(
        {
            "class_name": "ParallelismAMPPPO",
            "amp_window_frames": 24,
            "amp_reward_weight": 0.1,
            "amp_value_loss_coef": 1.0,
            "amp_warmup_iterations": 500,
            "amp_weight_ramp_iterations": 100,
            "disc_learning_rate": 1.0e-4,
            "disc_epochs": 2,
            "disc_batch_size": 4096,
            "disc_replay_capacity": 32768,
        }
    )
    cfg["policy"]["class_name"] = "AmpActorCriticAME"
    cfg["policy"]["amp_value_hidden_dims"] = [256, 128]
    return cfg


__all__ = ["get_ame_amp_train_cfg"]
