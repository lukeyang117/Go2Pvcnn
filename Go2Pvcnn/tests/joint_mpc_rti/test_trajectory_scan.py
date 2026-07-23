from __future__ import annotations

import inspect

import pytest
import torch

from extension.joint_mpc_rti.solver.associative_scan import (
    combine_conditional_value_factors,
)
from extension.joint_mpc_rti.solver.trajectory_scan import (
    factor_tree_shapes,
    pad_h30_factors,
    solve_trajectory_qp_scan,
)
from .test_trajectory_qp import _problem


def _random_factor(batch: int, dimension: int, *, generator: torch.Generator):
    matrix_a = 0.15 * torch.randn(batch, dimension, dimension, generator=generator, dtype=torch.float64)
    vector_c = 0.1 * torch.randn(batch, dimension, generator=generator, dtype=torch.float64)
    c_root = 0.05 * torch.randn(batch, dimension, dimension, generator=generator, dtype=torch.float64)
    matrix_c = c_root @ c_root.transpose(-1, -2)
    vector_p = 0.1 * torch.randn(batch, dimension, generator=generator, dtype=torch.float64)
    p_root = 0.05 * torch.randn(batch, dimension, dimension, generator=generator, dtype=torch.float64)
    matrix_p = p_root @ p_root.transpose(-1, -2)
    return matrix_a, vector_c, matrix_c, vector_p, matrix_p


def test_conditional_factor_combine_is_associative() -> None:
    generator = torch.Generator().manual_seed(71)
    first = _random_factor(3, 6, generator=generator)
    second = _random_factor(3, 6, generator=generator)
    third = _random_factor(3, 6, generator=generator)

    left = combine_conditional_value_factors(
        combine_conditional_value_factors(first, second), third
    )
    right = combine_conditional_value_factors(
        first, combine_conditional_value_factors(second, third)
    )

    for actual, expected in zip(left, right):
        torch.testing.assert_close(actual, expected, atol=1.0e-10, rtol=1.0e-10)


def test_h30_padding_and_tree_have_two_neutral_factors_and_five_levels() -> None:
    generator = torch.Generator().manual_seed(73)
    factors = tuple(
        torch.stack(
            [_random_factor(1, 4, generator=generator)[index][0] for _ in range(30)]
        )
        for index in range(5)
    )
    padded, valid = pad_h30_factors(factors)

    assert all(value.shape[0] == 32 for value in padded)
    assert valid.tolist() == [True] * 30 + [False, False]
    assert factor_tree_shapes() == (32, 16, 8, 4, 2, 1)
    torch.testing.assert_close(
        padded[0][-2:], torch.eye(4, dtype=torch.float64).expand(2, -1, -1)
    )
    for value in padded[1:]:
        torch.testing.assert_close(value[-2:], torch.zeros_like(value[-2:]))


@pytest.mark.parametrize("batch", (1, 8, 40))
def test_scan_matches_dense_constrained_solution(batch: int) -> None:
    problem, _, _, _ = _problem(batch=batch)
    problem = type(problem)(
        **{
            **vars(problem),
            "lower": torch.full_like(problem.lower, -10.0),
            "upper": torch.full_like(problem.upper, 10.0),
            "rate_lower": torch.full_like(problem.rate_lower, -10.0),
            "rate_upper": torch.full_like(problem.rate_upper, 10.0),
        }
    )
    problem.lower[:, 0] = 0.0
    problem.upper[:, 0] = 0.0

    scan = solve_trajectory_qp_scan(problem)

    assert scan.direction.shape == (batch, 31, 18)
    assert scan.kkt_primal_residual.max() <= 1.0e-4
    assert scan.kkt_dual_residual.max() <= 1.0e-4
    assert scan.dense_parity_error.max() < 2.0e-5
    torch.testing.assert_close(
        scan.direction[:, 0], torch.zeros_like(scan.direction[:, 0])
    )


def test_active_refinement_does_not_rebuild_lq_problem() -> None:
    import extension.joint_mpc_rti.solver.trajectory_scan as module

    source = inspect.getsource(module.solve_trajectory_qp_scan)
    assert "build_lq_problem" not in source
    assert source.count("refinement") >= 1
