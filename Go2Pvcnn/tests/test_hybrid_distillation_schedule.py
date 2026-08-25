import pytest
import torch

from rsl_rl.algorithms.hybrid_distillation_ppo import HybridDistillationPPO


class _Optimizer:
    def __init__(self, learning_rate):
        self.param_groups = [{"lr": learning_rate}]


def _algorithm(**kwargs):
    # Bypass network construction: these tests cover only schedule math.
    algorithm = HybridDistillationPPO.__new__(HybridDistillationPPO)
    algorithm.teacher_ratio_start = kwargs.get("teacher_ratio_start")
    algorithm.teacher_ratio_end = kwargs.get("teacher_ratio_end")
    algorithm.teacher_ratio_warmup_pct = kwargs.get("teacher_ratio_warmup_pct", 0.1)
    algorithm.teacher_ratio_decay_end_pct = kwargs.get("teacher_ratio_decay_end_pct", 0.8)
    algorithm.teacher_ratio_min = kwargs.get("teacher_ratio_min", 0.0)
    algorithm.current_iteration = kwargs.get("iteration", 0)
    algorithm.total_iterations = kwargs.get("total_iterations", 100)
    algorithm.schedule_start_iteration = kwargs.get("schedule_start_iteration", 0)
    return algorithm


def test_explicit_ratio_schedule_is_relative_to_resume_segment():
    algorithm = _algorithm(
        teacher_ratio_start=0.0,
        teacher_ratio_end=0.0,
        iteration=800,
        total_iterations=4800,
        schedule_start_iteration=800,
    )
    assert algorithm._compute_teacher_ratio() == pytest.approx(0.0)


def test_explicit_ratio_schedule_interpolates_between_start_and_end():
    algorithm = _algorithm(
        teacher_ratio_start=0.8,
        teacher_ratio_end=0.2,
        iteration=250,
        total_iterations=500,
        schedule_start_iteration=0,
    )
    assert algorithm._compute_teacher_ratio() == pytest.approx(0.5)


def test_legacy_ratio_schedule_remains_unchanged():
    algorithm = _algorithm(iteration=20, total_iterations=100)
    assert algorithm._compute_teacher_ratio() == pytest.approx(1.0 - 0.1 / 0.7)


def test_adaptive_learning_rate_matches_ppo_thresholds():
    algorithm = HybridDistillationPPO.__new__(HybridDistillationPPO)
    algorithm.desired_kl = 0.01
    algorithm.schedule = "adaptive"
    algorithm.learning_rate = 1.0e-3
    algorithm.optimizer = _Optimizer(algorithm.learning_rate)
    algorithm.distributed = False

    old_mu = torch.zeros(2, 3)
    old_sigma = torch.ones(2, 3)
    high_kl_mu = torch.full((2, 3), 0.3)
    low_kl_mu = torch.full((2, 3), 1.0e-3)

    # For unit variance, the first case is above desired_kl * 2.
    algorithm._adapt_learning_rate_from_kl(high_kl_mu, old_sigma, old_mu, old_sigma)
    assert algorithm.learning_rate == pytest.approx(1.0e-3 / 1.5)
    assert algorithm.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3 / 1.5)

    algorithm.learning_rate = 1.0e-3
    algorithm.optimizer.param_groups[0]["lr"] = 1.0e-3
    algorithm._adapt_learning_rate_from_kl(low_kl_mu, old_sigma, old_mu, old_sigma)
    assert algorithm.learning_rate == pytest.approx(1.0e-3 * 1.5)


def test_fixed_schedule_does_not_change_learning_rate():
    algorithm = HybridDistillationPPO.__new__(HybridDistillationPPO)
    algorithm.desired_kl = 0.01
    algorithm.schedule = "fixed"
    algorithm.learning_rate = 1.0e-3
    algorithm.optimizer = _Optimizer(algorithm.learning_rate)
    algorithm.distributed = False

    values = torch.zeros(2, 3)
    algorithm._adapt_learning_rate_from_kl(values, torch.ones(2, 3), values, torch.ones(2, 3))
    assert algorithm.learning_rate == pytest.approx(1.0e-3)
