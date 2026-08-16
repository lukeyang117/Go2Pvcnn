"""Training configuration for the active semantic MPC teacher experiment."""


def get_train_cfg(experiment_name: str) -> dict:
    """Return the RSL-RL config for the active semantic MPC experiment."""

    supported = {
        "teacher_elevation_trajectory_mpc_semantic",
        "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance",
        "parallelism_tracking_flat",
        "parallelism_tracking_small_obstacles",
        "parallelism_tracking_ladder",
        "parallelism_tracking_cross_large_complex",
        "parallelism_tracking_cross_large_complex_distillation",
    }
    if experiment_name not in supported:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    if experiment_name == "parallelism_tracking_cross_large_complex_distillation":
        return _parallelism_distillation_train_cfg()
    return _teacher_elevation_trajectory_mpc_semantic_train_cfg()


def _parallelism_distillation_train_cfg() -> dict:
    return {
        "num_steps_per_env": 40,
        "save_interval": 100,
        "empirical_normalization": False,
        "cost_map_channels": 2,
        "cost_map_size": 16,
        "algorithm": {
            "class_name": "Distillation",
            "num_learning_epochs": 1,
            "num_mini_batches": 4,
            "gradient_length": 1,
            "learning_rate": 1e-3,
            "loss_type": "mse",
            "teacher_ratio_warmup_pct": 0.10,
            "teacher_ratio_decay_end_pct": 0.80,
            "teacher_ratio_min": 0.0,
            "student_action_start_ratio": 0.30,
        },
        "policy": {
            "class_name": "StudentTeacherCNN",
            "init_noise_std": 0.1,
            "cost_map_channels": 2,
            "cost_map_size": 16,
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
            "student_hidden_dims": [256, 128],
            "teacher_hidden_dims": [256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "student": ["student_elevation_semantic_map", "student_state"],
            "teacher": ["teacher_elevation_semantic_map", "teacher_state"],
        },
    }


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
