"""Training configuration for Go2 teacher experiments.

Split from train.py - provides train_cfg dict by experiment name.
"""


def get_train_cfg(experiment_name: str) -> dict:
    """Get training configuration for the given experiment.

    Args:
        experiment_name: One of "teacher_semantic", "teacher_without_semantic"

    Returns:
        train_cfg dict for OnPolicyRunner
    """
    if experiment_name == "teacher_semantic":
        return _teacher_semantic_train_cfg()
    if experiment_name == "teacher_without_semantic":
        return _teacher_without_semantic_train_cfg()
    raise ValueError(f"Unknown experiment: {experiment_name}")


def _teacher_semantic_train_cfg() -> dict:
    """Training config for teacher_semantic (CNN cost map + state)."""
    return {
        "num_steps_per_env": 40,
        "save_interval": 100,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "policy": {
            "class_name": "ActorCriticCNN",
            "init_noise_std": 1.0,
            "noise_std_type": "log",
            "state_dependent_std": False,
            "actor_cnn_cfg": {
                "output_channels": [32, 64],
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": "zeros",
                "max_pool": [True, True],
                "activation": "elu",
                "flatten": True,
            },
            "critic_cnn_cfg": {
                "output_channels": [32, 64],
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": "zeros",
                "max_pool": [True, True],
                "activation": "elu",
                "flatten": True,
            },
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "policy": ["policy_cost_map", "policy_state"],
            "critic": ["critic_cost_map", "critic_state"],
        },
    }


def _teacher_without_semantic_train_cfg() -> dict:
    """Training config for teacher_without_semantic (state-only, no CNN)."""
    return {
        "num_steps_per_env": 40,
        "save_interval": 100,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["critic"],
        },
    }


