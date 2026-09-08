"""PPO configuration for the isolated AME baseline."""

from __future__ import annotations


def get_ame_train_cfg() -> dict:
    return {
        "num_steps_per_env": 40,
        "save_interval": 100,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "policy": {
            "class_name": "ActorCriticAME",
            "map_channels": 6,
            "map_size": 16,
            "mha_dim": 64,
            "num_heads": 16,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
        },
    }


__all__ = ["get_ame_train_cfg"]
