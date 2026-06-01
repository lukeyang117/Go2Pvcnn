"""Training configuration for the active semantic MPC teacher experiment."""


def get_train_cfg(experiment_name: str) -> dict:
    """Return the RSL-RL config for the active semantic MPC experiment."""

    if experiment_name != "teacher_elevation_trajectory_mpc_semantic":
        raise ValueError(f"Unknown experiment: {experiment_name}")
    return _teacher_elevation_trajectory_mpc_semantic_train_cfg()


def _teacher_elevation_trajectory_mpc_semantic_train_cfg() -> dict:
    """Training config for MPC semantic trajectory imitation."""

    return {
        "num_steps_per_env": 40,
        "save_interval": 100,
        "empirical_normalization": False,
        "cost_map_channels": 2,
        "cost_map_size": 16,
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
            "policy": ["policy_elevation_semantic_map", "policy_state"],
            "critic": ["critic_elevation_semantic_map", "critic_state"],
        },
    }
