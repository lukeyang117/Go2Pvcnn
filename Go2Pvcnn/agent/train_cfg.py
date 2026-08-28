"""Training configuration for the active semantic MPC teacher experiment."""

import copy


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
        "cross_large_complex_ppo",
        "parallelism_tracking_cross_large_complex_amp",
    }
    if experiment_name not in supported:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    if experiment_name == "parallelism_tracking_cross_large_complex_distillation":
        return _parallelism_distillation_train_cfg()
    if experiment_name == "cross_large_complex_ppo":
        return _cross_large_complex_ppo_train_cfg()
    if experiment_name == "parallelism_tracking_cross_large_complex_amp":
        return _parallelism_amp_train_cfg()
    return _teacher_elevation_trajectory_mpc_semantic_train_cfg()


def _parallelism_distillation_train_cfg() -> dict:
    return {
        "num_steps_per_env": 40,
        "save_interval": 100,
        "empirical_normalization": False,
        "cost_map_channels": 2,
        "cost_map_size": 16,
        "algorithm": {
            "class_name": "HybridDistillationPPO",
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
            "ppo_coef": 1.0,
            "teacher_coef": 0.1,
            "teacher_ratio_start": 0.0,
            "teacher_ratio_end": 0.0,
            "teacher_ratio_warmup_pct": 0.0,
            "teacher_ratio_decay_end_pct": 0.0,
            "teacher_ratio_min": 0.0,
        },
        "policy": {
            "class_name": "StudentTeacherCNN",
            "init_noise_std": 1.0,
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
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "student": ["student_elevation_semantic_map", "student_state"],
            "teacher": ["teacher_elevation_semantic_map", "teacher_state"],
        },
    }


def _cross_large_complex_ppo_train_cfg() -> dict:
    """Pure PPO config aligned with the PPO-side distillation settings."""

    cnn_cfg = {
        "output_channels": [32, 64],
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": "zeros",
        "max_pool": [True, True],
        "activation": "elu",
        "flatten": True,
    }
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
            "cost_map_channels": 2,
            "cost_map_size": 16,
            "actor_cnn_cfg": dict(cnn_cfg),
            "critic_cnn_cfg": dict(cnn_cfg),
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "policy": ["policy_elevation_semantic_map", "policy_state"],
            "critic": ["critic_elevation_semantic_map", "critic_state"],
        },
    }


def _parallelism_amp_train_cfg() -> dict:
    cfg = copy.deepcopy(_cross_large_complex_ppo_train_cfg())
    cfg["experiment_name"] = "parallelism_tracking_cross_large_complex_amp"
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
    cfg["policy"]["class_name"] = "AmpActorCriticCNN"
    cfg["policy"]["amp_value_hidden_dims"] = [256, 128]
    return cfg


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
