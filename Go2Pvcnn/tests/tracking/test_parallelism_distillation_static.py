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


def test_runner_source_recognizes_distillation():
    source = Path("Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text()

    assert '"Distillation"' in source
    assert "load_teacher" in source
    assert 'extras["observations"].get("teacher"' in source


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
    alg.set_iteration(80, 100)
    assert alg._compute_teacher_ratio() == 0.0
    alg.set_iteration(100, 100)
    assert alg._compute_teacher_ratio() == 0.0
