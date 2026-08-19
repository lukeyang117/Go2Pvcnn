from pathlib import Path

import torch


def test_project_rsl_rl_exports_distillation():
    from rsl_rl.algorithms import Distillation

    assert Distillation.__name__ == "Distillation"


def test_student_teacher_cnn_builds_with_different_obs_dims():
    from rsl_rl.modules import StudentTeacherCNN

    model = StudentTeacherCNN(
        num_student_obs=560,
        num_teacher_obs=620,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        student_hidden_dims=[32],
        teacher_hidden_dims=[32],
        activation="elu",
    )

    assert model.act_inference(torch.zeros(2, 560)).shape == (2, 12)
    assert model.evaluate(torch.zeros(2, 620)).shape == (2, 12)
    assert model.evaluate_value(torch.zeros(2, 620)).shape == (2, 1)


def test_student_teacher_cnn_keeps_critic_observation_dimension_independent():
    from rsl_rl.modules import StudentTeacherCNN

    model = StudentTeacherCNN(
        num_student_obs=560,
        num_teacher_obs=620,
        num_critic_obs=600,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        critic_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        critic_hidden_dims=[16],
        activation="elu",
    )

    assert model.student_critic.critic[0].in_features == 600 - 2 * 16 * 16 + 256


def test_student_only_checkpoint_loading_does_not_replace_teacher():
    from rsl_rl.modules import StudentTeacherCNN

    model = StudentTeacherCNN(
        num_student_obs=560,
        num_teacher_obs=620,
        num_critic_obs=600,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        critic_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        critic_hidden_dims=[16],
        activation="elu",
    )
    original_teacher = model.teacher.actor[0].weight.detach().clone()
    state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state_dict["student.actor.0.weight"] = state_dict["student.actor.0.weight"] + 1.0
    state_dict["student_critic.critic.0.weight"] = state_dict["student_critic.critic.0.weight"] + 1.0
    state_dict["teacher.actor.0.weight"] = state_dict["teacher.actor.0.weight"] + 2.0

    model.load_student_state_dict(state_dict)

    assert torch.allclose(model.student.actor[0].weight, state_dict["student.actor.0.weight"])
    assert torch.allclose(
        model.student_critic.critic[0].weight,
        state_dict["student_critic.critic.0.weight"],
    )
    assert torch.allclose(model.teacher.actor[0].weight, original_teacher)


def test_hybrid_distillation_ppo_runs_one_update():
    from rsl_rl.algorithms import HybridDistillationPPO
    from rsl_rl.modules import StudentTeacherCNN

    num_envs = 2
    num_steps = 3
    policy = StudentTeacherCNN(
        num_student_obs=560,
        num_teacher_obs=620,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        critic_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        critic_hidden_dims=[16],
        activation="elu",
        init_noise_std=0.3,
    )
    alg = HybridDistillationPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        device="cpu",
    )
    alg.init_storage("hybrid_distillation", num_envs, num_steps, [560], [620], [12])

    for _ in range(num_steps):
        obs = torch.randn(num_envs, 560)
        teacher_obs = torch.randn(num_envs, 620)
        actions = alg.act(obs, teacher_obs)
        assert actions.shape == (num_envs, 12)
        alg.process_env_step(
            torch.zeros(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            {},
        )

    alg.compute_returns(torch.randn(num_envs, 620))
    losses = alg.update()
    assert losses["ppo_coef"] == 1.0
    assert losses["teacher_coef"] > 0.0
    assert torch.isfinite(torch.tensor(losses["imitation_loss"]))


def test_hybrid_distillation_ppo_uses_separate_teacher_and_critic_observations():
    from rsl_rl.algorithms import HybridDistillationPPO
    from rsl_rl.modules import StudentTeacherCNN

    num_envs = 2
    num_steps = 2
    policy = StudentTeacherCNN(
        num_student_obs=560,
        num_teacher_obs=620,
        num_critic_obs=600,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        critic_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        critic_hidden_dims=[16],
        activation="elu",
    )
    alg = HybridDistillationPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        device="cpu",
    )
    alg.init_storage("hybrid_distillation", num_envs, num_steps, [560], [600], [12])
    for _ in range(num_steps):
        actions = alg.act(
            torch.randn(num_envs, 560),
            torch.randn(num_envs, 620),
            torch.randn(num_envs, 600),
        )
        assert actions.shape == (num_envs, 12)
        assert alg.transition.critic_observations.shape == (num_envs, 600)
        alg.process_env_step(
            torch.zeros(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            {},
        )
    alg.compute_returns(torch.randn(num_envs, 600))
    losses = alg.update()
    assert torch.isfinite(torch.tensor(losses["value_loss"]))


def _make_small_student_teacher_policy():
    from rsl_rl.modules import StudentTeacherCNN

    return StudentTeacherCNN(
        num_student_obs=560,
        num_teacher_obs=620,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        critic_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        critic_hidden_dims=[16],
        activation="elu",
        init_noise_std=0.3,
    )


def test_hybrid_teacher_student_ratio_and_action_source_mask():
    from rsl_rl.algorithms import HybridDistillationPPO

    num_envs = 10
    policy = _make_small_student_teacher_policy()
    alg = HybridDistillationPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
        teacher_ratio_warmup_pct=0.30,
        teacher_ratio_decay_end_pct=0.80,
        teacher_ratio_min=0.0,
    )

    alg.set_iteration(0, 100)
    assert alg._compute_teacher_ratio() == 1.0
    actions = alg.act(torch.zeros(num_envs, 560), torch.zeros(num_envs, 620))
    assert actions.shape == (num_envs, 12)
    assert torch.equal(alg.transition.ppo_active, torch.zeros(num_envs))
    assert alg.last_teacher_action_share == 1.0

    mid_alg = HybridDistillationPPO(
        _make_small_student_teacher_policy(),
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
        teacher_ratio_warmup_pct=0.30,
        teacher_ratio_decay_end_pct=0.80,
        teacher_ratio_min=0.0,
    )
    mid_alg.set_iteration(55, 100)
    assert abs(mid_alg._compute_teacher_ratio() - 0.5) < 1e-6
    mid_alg.act(torch.zeros(num_envs, 560), torch.zeros(num_envs, 620))
    assert mid_alg.transition.ppo_active.sum().item() == 5
    assert mid_alg.last_teacher_action_share == 0.5

    late_alg = HybridDistillationPPO(
        _make_small_student_teacher_policy(),
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
        teacher_ratio_warmup_pct=0.30,
        teacher_ratio_decay_end_pct=0.80,
        teacher_ratio_min=0.0,
    )
    late_alg.set_iteration(80, 100)
    assert late_alg._compute_teacher_ratio() == 0.0
    late_alg.act(torch.zeros(num_envs, 560), torch.zeros(num_envs, 620))
    assert torch.equal(late_alg.transition.ppo_active, torch.ones(num_envs))
    assert late_alg.last_teacher_action_share == 0.0


def test_hybrid_controller_assignment_stays_fixed_until_episode_reset():
    from rsl_rl.algorithms import HybridDistillationPPO

    num_envs = 10
    policy = _make_small_student_teacher_policy()
    alg = HybridDistillationPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
        teacher_ratio_warmup_pct=0.30,
        teacher_ratio_decay_end_pct=0.80,
        teacher_ratio_min=0.0,
    )
    alg.init_storage("hybrid_distillation", num_envs, 2, [560], [620], [12])
    alg.set_iteration(55, 100)
    obs = torch.zeros(num_envs, 560)
    teacher_obs = torch.zeros(num_envs, 620)
    alg.act(obs, teacher_obs)
    initial_mask = alg._teacher_control_mask.clone()

    alg.process_env_step(
        torch.zeros(num_envs),
        torch.zeros(num_envs, dtype=torch.bool),
        {},
    )
    alg.act(obs, teacher_obs)
    assert torch.equal(alg._teacher_control_mask, initial_mask)

    done_ids = torch.tensor([0, 1])
    dones = torch.zeros(num_envs, dtype=torch.bool)
    dones[done_ids] = True
    alg.process_env_step(torch.zeros(num_envs), dones, {})
    alg.set_iteration(80, 100)
    alg.act(obs, teacher_obs)
    assert torch.equal(alg._teacher_control_mask[2:], initial_mask[2:])
    assert torch.equal(
        alg._teacher_control_mask[done_ids],
        torch.zeros(done_ids.numel(), dtype=torch.bool),
    )


def test_rollout_storage_preserves_ppo_active_mask():
    from rsl_rl.storage.rollout_storage import RolloutStorage

    storage = RolloutStorage(
        num_envs=2,
        num_transitions_per_env=1,
        obs_shape=[3],
        privileged_obs_shape=[4],
        actions_shape=[2],
        device="cpu",
    )
    transition = RolloutStorage.Transition()
    transition.observations = torch.zeros(2, 3)
    transition.critic_observations = torch.zeros(2, 4)
    transition.actions = torch.zeros(2, 2)
    transition.privileged_actions = torch.zeros(2, 2)
    transition.rewards = torch.zeros(2)
    transition.dones = torch.zeros(2, dtype=torch.bool)
    transition.ppo_active = torch.tensor([0.0, 1.0])
    storage.add_transitions(transition)

    batch = next(
        storage.mini_batch_generator(
            num_mini_batches=1,
            num_epochs=1,
            include_privileged_actions=True,
            include_ppo_mask=True,
        )
    )
    assert torch.equal(
        batch[-1].flatten().sort().values,
        torch.tensor([0.0, 1.0]),
    )


def test_runner_source_recognizes_distillation():
    source = Path("Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text()

    assert '"Distillation"' in source
    assert '("distillation", "hybrid_distillation")' in source
    assert "load_teacher" in source
    assert 'extras["observations"].get("teacher"' in source
    assert "load_student_checkpoint" in source


def test_distillation_teacher_ratio_schedule():
    from rsl_rl.algorithms import Distillation
    from rsl_rl.modules import StudentTeacherCNN

    policy = StudentTeacherCNN(
        num_student_obs=560,
        num_teacher_obs=620,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        activation="elu",
    )
    alg = Distillation(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        gradient_length=1,
        learning_rate=1e-3,
        loss_type="mse",
        device="cpu",
        teacher_ratio_warmup_pct=0.10,
        teacher_ratio_decay_end_pct=0.80,
        teacher_ratio_min=0.0,
    )

    alg.set_iteration(0, 100)
    assert alg._compute_teacher_ratio() == 1.0
    alg.set_iteration(10, 100)
    assert alg._compute_teacher_ratio() == 1.0
    alg.set_iteration(45, 100)
    assert alg._compute_teacher_ratio() == 0.5
    assert alg._compute_env_teacher_ratio() == 0.5
    alg.set_iteration(80, 100)
    assert alg._compute_teacher_ratio() == 0.0
    alg.set_iteration(100, 100)
    assert alg._compute_teacher_ratio() == 0.0


def test_distillation_student_action_waits_for_start_ratio():
    from rsl_rl.algorithms import Distillation
    from rsl_rl.modules import StudentTeacherCNN

    policy = StudentTeacherCNN(
        num_student_obs=560,
        num_teacher_obs=620,
        num_actions=12,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg={
            "output_channels": [8, 16],
            "kernel_size": [3, 3],
            "max_pool": [True, True],
            "activation": "elu",
        },
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        activation="elu",
    )
    alg = Distillation(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        gradient_length=1,
        learning_rate=1e-3,
        loss_type="mse",
        device="cpu",
        teacher_ratio_warmup_pct=0.10,
        teacher_ratio_decay_end_pct=0.80,
        teacher_ratio_min=0.0,
        student_action_start_ratio=0.30,
    )

    alg.set_iteration(20, 100)
    assert alg._compute_teacher_ratio() > 0.70
    assert alg._compute_env_teacher_ratio() == 1.0
    alg.set_iteration(32, 100)
    assert alg._compute_teacher_ratio() <= 0.70
    assert alg._compute_env_teacher_ratio() == alg._compute_teacher_ratio()


def test_hybrid_teacher_coef_is_fixed_across_iterations():
    from rsl_rl.algorithms import HybridDistillationPPO

    alg = HybridDistillationPPO(
        _make_small_student_teacher_policy(),
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        device="cpu",
        teacher_coef=0.7,
        teacher_coef_min=0.1,
        teacher_coef_decay_end_pct=0.2,
    )

    for iteration in (0, 10, 50, 100):
        alg.set_iteration(iteration, 100)
        assert alg._compute_teacher_coef() == 0.7


def test_hybrid_schedule_config_and_fresh_launcher():
    from agent.train_cfg import get_train_cfg

    cfg = get_train_cfg("parallelism_tracking_cross_large_complex_distillation")
    algorithm = cfg["algorithm"]
    assert algorithm["teacher_ratio_warmup_pct"] == 0.10
    assert algorithm["teacher_ratio_decay_end_pct"] == 0.80
    assert algorithm["teacher_ratio_min"] == 0.0

    launcher = Path(
        "Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh"
    ).read_text()
    assert "--teacher_checkpoint" in launcher
    assert "--resume" not in launcher
    assert "--load_run" not in launcher
    assert "--load_checkpoint" not in launcher
