import pytest

from rsl_rl.algorithms.hybrid_distillation_ppo import HybridDistillationPPO


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
