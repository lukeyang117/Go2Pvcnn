from __future__ import annotations

import inspect

import pytest
import torch

from extension.joint_mpc_rti.solver.associative_scan import (
    combine_conditional_value_factors,
)
from extension.joint_mpc_rti.solver.trajectory_qp import (
    ActiveConstraints,
    TrajectoryQp,
    solve_dense_active_kkt,
)
from extension.joint_mpc_rti.solver.trajectory_scan import (
    pad_h30_factors,
    solve_active_trajectory_qp_scan,
    solve_trajectory_qp_scan,
)


def _random_factor(batch: int, dimension: int, *, generator: torch.Generator):
    matrix_a = 0.15 * torch.randn(batch, dimension, dimension, generator=generator, dtype=torch.float64)
    vector_c = 0.1 * torch.randn(batch, dimension, generator=generator, dtype=torch.float64)
    c_root = 0.05 * torch.randn(batch, dimension, dimension, generator=generator, dtype=torch.float64)
    matrix_c = c_root @ c_root.transpose(-1, -2)
    vector_p = 0.1 * torch.randn(batch, dimension, generator=generator, dtype=torch.float64)
    p_root = 0.05 * torch.randn(batch, dimension, dimension, generator=generator, dtype=torch.float64)
    matrix_p = p_root @ p_root.transpose(-1, -2)
    return matrix_a, vector_c, matrix_c, vector_p, matrix_p


def _random_h30_qp(
    batch: int,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> TrajectoryQp:
    generator = torch.Generator(device=device).manual_seed(900 + batch)
    dtype = dtype or (torch.float64 if device == "cpu" else torch.float32)
    nodes = 31
    state_dim = 18
    diagonal = 5.0 * torch.eye(state_dim, dtype=dtype, device=device).expand(batch, nodes, -1, -1).clone()
    diagonal += 0.05 * torch.randn(batch, nodes, state_dim, state_dim, generator=generator, dtype=dtype, device=device)
    diagonal = 0.5 * (diagonal + diagonal.transpose(-1, -2))
    first = 0.025 * torch.randn(
        batch, nodes - 1, state_dim, state_dim, generator=generator, dtype=dtype, device=device
    )
    second = 0.01 * torch.randn(
        batch, nodes - 2, state_dim, state_dim, generator=generator, dtype=dtype, device=device
    )
    gradient = 0.2 * torch.randn(batch, nodes, state_dim, generator=generator, dtype=dtype, device=device)
    lower = torch.full_like(gradient, -10.0)
    upper = torch.full_like(gradient, 10.0)
    lower[:, 0] = 0.0
    upper[:, 0] = 0.0
    difference_lower = torch.full((batch, nodes - 1, 12), -10.0, dtype=dtype, device=device)
    difference_upper = torch.full_like(difference_lower, 10.0)
    support_jacobian = 0.05 * torch.randn(
        batch, 6, state_dim, generator=generator, dtype=dtype, device=device
    )
    support_target = 1.0e-3 * torch.randn(
        batch, 6, generator=generator, dtype=dtype, device=device
    )
    return TrajectoryQp(
        diagonal=diagonal,
        first_offdiag=first,
        second_offdiag=second,
        gradient=gradient,
        lower=lower,
        upper=upper,
        joint_difference_lower=difference_lower,
        joint_difference_upper=difference_upper,
        support_jacobian=support_jacobian,
        support_target=support_target,
    )


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


def test_h30_padding_appends_two_identity_no_cost_factors() -> None:
    generator = torch.Generator().manual_seed(73)
    factors = tuple(
        torch.stack([_random_factor(1, 4, generator=generator)[index][0] for _ in range(30)], dim=0)
        for index in range(5)
    )
    padded, valid = pad_h30_factors(factors)

    assert all(value.shape[0] == 32 for value in padded)
    assert valid.tolist() == [True] * 30 + [False, False]
    torch.testing.assert_close(padded[0][-2:], torch.eye(4, dtype=torch.float64).expand(2, -1, -1))
    for value in padded[1:]:
        torch.testing.assert_close(value[-2:], torch.zeros_like(value[-2:]))


@pytest.mark.parametrize("batch", [1, 7, 40])
def test_h30_scan_matches_dense_active_kkt(batch: int) -> None:
    qp = _random_h30_qp(batch)
    scan = solve_trajectory_qp_scan(qp)
    dense = solve_dense_active_kkt(qp, scan.active)

    assert scan.direction.shape == (batch, 31, 18)
    assert torch.isfinite(scan.direction).all()
    assert (scan.direction - dense).abs().max() < 2.0e-5


def test_h30_float32_scan_matches_dense_active_kkt() -> None:
    qp = _random_h30_qp(7, dtype=torch.float32)
    scan = solve_trajectory_qp_scan(qp)
    dense = solve_dense_active_kkt(qp, scan.active)
    assert (scan.direction - dense).abs().max() < 2.0e-5


def test_scan_matches_dense_with_local_velocity_and_box_constraints() -> None:
    qp = _random_h30_qp(2)
    active = ActiveConstraints.empty(qp)
    box_low = active.box_low.clone()
    velocity_high = active.velocity_high.clone()
    box_low[:, 0] = True
    box_low[0, 9, 2] = True
    velocity_high[0, 14, 3] = True
    constrained = ActiveConstraints(
        box_low=box_low,
        box_high=active.box_high,
        velocity_low=active.velocity_low,
        velocity_high=velocity_high,
    )

    scan = solve_active_trajectory_qp_scan(qp, constrained)
    dense = solve_dense_active_kkt(qp, constrained)
    torch.testing.assert_close(scan, dense, atol=2.0e-5, rtol=2.0e-5)


def test_scan_preserves_velocity_equality_when_its_destination_is_box_fixed() -> None:
    qp = _random_h30_qp(1)
    active = ActiveConstraints.empty(qp)
    box_low = active.box_low.clone()
    velocity_high = active.velocity_high.clone()
    box_low[:, 0] = True
    box_low[0, 15, 9] = True
    velocity_high[0, 14, 3] = True
    constrained = ActiveConstraints(
        box_low=box_low,
        box_high=active.box_high,
        velocity_low=active.velocity_low,
        velocity_high=velocity_high,
    )

    scan = solve_active_trajectory_qp_scan(qp, constrained)
    dense = solve_dense_active_kkt(qp, constrained)
    torch.testing.assert_close(scan, dense, atol=2.0e-5, rtol=2.0e-5)


def test_scan_ignores_velocity_edge_when_both_endpoint_boxes_are_fixed() -> None:
    qp = _random_h30_qp(1)
    active = ActiveConstraints.empty(qp)
    box_low = active.box_low.clone()
    velocity_high = active.velocity_high.clone()
    box_low[:, 0] = True
    box_low[0, 14, 9] = True
    box_low[0, 15, 9] = True
    velocity_high[0, 14, 3] = True
    constrained = ActiveConstraints(
        box_low=box_low,
        box_high=active.box_high,
        velocity_low=active.velocity_low,
        velocity_high=velocity_high,
    )

    scan = solve_active_trajectory_qp_scan(qp, constrained)
    dense = solve_dense_active_kkt(qp, constrained)
    torch.testing.assert_close(scan, dense, atol=2.0e-5, rtol=2.0e-5)


def test_scan_matches_dense_for_random_feasible_active_components() -> None:
    qp = _random_h30_qp(3)
    generator = torch.Generator().manual_seed(79)
    witness = 0.2 * torch.randn(3, 31, 18, generator=generator, dtype=torch.float64)
    witness[:, 0] = 0.0
    box_low = torch.rand(3, 31, 18, generator=generator) < 0.04
    box_high = (torch.rand(3, 31, 18, generator=generator) < 0.04) & ~box_low
    box_low[:, 0] = True
    box_high[:, 0] = False
    velocity_low = torch.rand(3, 30, 12, generator=generator) < 0.06
    velocity_high = (torch.rand(3, 30, 12, generator=generator) < 0.06) & ~velocity_low
    difference = witness[:, 1:, 6:] - witness[:, :-1, 6:]
    constrained_qp = TrajectoryQp(
        diagonal=qp.diagonal,
        first_offdiag=qp.first_offdiag,
        second_offdiag=qp.second_offdiag,
        gradient=qp.gradient,
        lower=torch.where(box_low, witness, qp.lower),
        upper=torch.where(box_high, witness, qp.upper),
        joint_difference_lower=torch.where(velocity_low, difference, qp.joint_difference_lower),
        joint_difference_upper=torch.where(velocity_high, difference, qp.joint_difference_upper),
        support_jacobian=qp.support_jacobian,
        support_target=qp.support_target,
    )
    active = ActiveConstraints(box_low, box_high, velocity_low, velocity_high)

    scan = solve_active_trajectory_qp_scan(constrained_qp, active)
    dense = solve_dense_active_kkt(constrained_qp, active)
    torch.testing.assert_close(scan, dense, atol=2.0e-5, rtol=2.0e-5)


def test_solver_source_uses_fixed_tree_not_generic_associative_scan() -> None:
    import extension.joint_mpc_rti.solver.associative_scan as scan_module
    import extension.joint_mpc_rti.solver.trajectory_scan as trajectory_module

    source = inspect.getsource(scan_module) + inspect.getsource(trajectory_module)
    assert "torch._higher_order_ops.associative_scan" not in source
    assert "combine_level_1" in source
    assert "combine_level_5" in source


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA compile smoke requires a visible GPU",
)
def test_cuda_b1_compile() -> None:
    qp = _random_h30_qp(1, device="cuda")
    compiled = torch.compile(solve_trajectory_qp_scan, fullgraph=True, dynamic=False)
    result = compiled(qp)
    assert result.direction.shape == (1, 31, 18)
    assert torch.isfinite(result.direction).all()
