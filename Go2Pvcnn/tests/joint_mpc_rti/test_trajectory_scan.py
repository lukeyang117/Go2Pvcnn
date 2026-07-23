from __future__ import annotations

import inspect
from dataclasses import fields

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
from extension.joint_mpc_rti.solver.trajectory_qp import solve_dense_qp
from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti import planner
from .helpers import make_command, make_flat_field, make_state
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

    dense = solve_dense_qp(problem)
    scan = solve_trajectory_qp_scan(problem)

    assert scan.direction.shape == (batch, 31, 18)
    assert scan.kkt_primal_residual.max() <= 1.0e-4
    assert scan.kkt_dual_residual.max() <= 1.0e-4
    torch.testing.assert_close(
        scan.direction, dense.direction, atol=2.0e-5, rtol=2.0e-5
    )
    torch.testing.assert_close(
        scan.direction[:, 0], torch.zeros_like(scan.direction[:, 0])
    )


@pytest.mark.parametrize("vx,command_scale", ((0.4, 0.45), (0.8, 0.60)))
def test_production_scan_satisfies_flat_full_horizon_stance_equalities(
    vx: float, command_scale: float
) -> None:
    cfg = JointMpcRtiCfg()
    cfg.nominal.command_scale = command_scale
    result = planner.step(
        make_state(1),
        make_command(1, vx=vx),
        make_flat_field(1),
        None,
        cfg,
    )

    assert result.diagnostics.kkt_primal_residual.max() <= 1.0e-4


def test_production_scan_reports_scaled_dual_kkt_residual() -> None:
    result = planner.step(
        make_state(1),
        make_command(1, vx=0.2),
        make_flat_field(1),
        None,
        JointMpcRtiCfg(),
    )

    assert result.diagnostics.kkt_dual_residual.max() <= 1.0e-4


def test_warm_scan_eliminates_fixed_published_root_before_stance_solve() -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.4)
    field = make_flat_field(1)
    solver_state = None
    result = None
    for _ in range(7):
        origin = field.origin_w.clone()
        origin[:, :2] = measured.root_pos_w[:, :2]
        result = planner.step(
            measured,
            command,
            replace(field, origin_w=origin),
            solver_state,
            cfg,
        )
        state = result.full_trajectory.state_nodes[:, 1]
        velocity = result.full_trajectory.derived_velocity[:, 0]
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        solver_state = result.solver_state

    assert result is not None
    assert result.diagnostics.kkt_primal_residual.max() <= 1.0e-4


def test_warm_scan_keeps_high_translation_stance_residual_below_one_mm() -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti.types import JointMpcRtiState

    cfg = JointMpcRtiCfg()
    cfg.nominal.command_scale = 0.6
    measured = make_state(1)
    command = make_command(1, vx=0.8)
    field = make_flat_field(1)
    solver_state = None
    residuals = []
    for _ in range(9):
        origin = field.origin_w.clone()
        origin[:, :2] = measured.root_pos_w[:, :2]
        result = planner.step(
            measured,
            command,
            replace(field, origin_w=origin),
            solver_state,
            cfg,
        )
        residuals.append(result.diagnostics.kkt_primal_residual)
        state = result.full_trajectory.state_nodes[:, 1]
        velocity = result.full_trajectory.derived_velocity[:, 0]
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        solver_state = result.solver_state

    assert torch.stack(residuals).max() <= 1.0e-3


def test_associative_scan_matches_dense_warm_augmented_system(monkeypatch) -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti.solver import trajectory_scan
    from extension.joint_mpc_rti.types import JointMpcRtiState

    associative_solve = trajectory_scan._solve_augmented_associative
    comparisons: list[tuple[torch.Tensor, torch.Tensor]] = []

    def compare_with_dense(diagonal, first, second, gradient):
        direction = associative_solve(diagonal, first, second, gradient)
        batch, nodes, state_dim = gradient.shape
        dense = gradient.new_zeros(batch, nodes * state_dim, nodes * state_dim)
        for node in range(nodes):
            row = slice(node * state_dim, (node + 1) * state_dim)
            dense[:, row, row] = diagonal[:, node]
        for node in range(nodes - 1):
            row = slice(node * state_dim, (node + 1) * state_dim)
            column = slice((node + 1) * state_dim, (node + 2) * state_dim)
            dense[:, row, column] = first[:, node]
            dense[:, column, row] = first[:, node].transpose(-1, -2)
        for node in range(nodes - 2):
            row = slice(node * state_dim, (node + 1) * state_dim)
            column = slice((node + 2) * state_dim, (node + 3) * state_dim)
            dense[:, row, column] = second[:, node]
            dense[:, column, row] = second[:, node].transpose(-1, -2)
        dense_direction = torch.linalg.solve(dense, -gradient.flatten(1)).reshape_as(
            gradient
        )
        comparisons.append((direction.detach(), dense_direction.detach()))
        return direction

    monkeypatch.setattr(
        trajectory_scan, "_solve_augmented_associative", compare_with_dense
    )
    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.4)
    field = make_flat_field(1)
    solver_state = None
    for _ in range(7):
        origin = field.origin_w.clone()
        origin[:, :2] = measured.root_pos_w[:, :2]
        result = planner.step(
            measured,
            command,
            replace(field, origin_w=origin),
            solver_state,
            cfg,
        )
        state = result.full_trajectory.state_nodes[:, 1]
        velocity = result.full_trajectory.derived_velocity[:, 0]
        measured = JointMpcRtiState(
            root_pos_w=state[:, :3],
            root_rpy_w=state[:, 3:6],
            joint_pos=state[:, 6:],
            root_lin_vel_b=velocity[:, :3],
            root_ang_vel_b=velocity[:, 3:6],
            joint_vel=velocity[:, 6:],
        )
        solver_state = result.solver_state

    scan_direction, dense_direction = comparisons[-1]
    torch.testing.assert_close(
        scan_direction, dense_direction, atol=2.0e-5, rtol=2.0e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires a GPU")
def test_cuda_scan_matches_cpu_for_same_lq_problem() -> None:
    problem, _, _, _ = _problem(batch=3)
    problem = type(problem)(
        **{
            item.name: (
                getattr(problem, item.name).float()
                if isinstance(getattr(problem, item.name), torch.Tensor)
                and getattr(problem, item.name).is_floating_point()
                else getattr(problem, item.name)
            )
            for item in fields(problem)
        }
    )
    cpu = solve_trajectory_qp_scan(problem)
    cuda_problem = type(problem)(
        **{
            item.name: (
                getattr(problem, item.name).cuda()
                if isinstance(getattr(problem, item.name), torch.Tensor)
                else getattr(problem, item.name)
            )
            for item in fields(problem)
        }
    )

    cuda = solve_trajectory_qp_scan(cuda_problem)

    torch.testing.assert_close(
        cuda.direction.cpu(), cpu.direction, atol=2.0e-5, rtol=2.0e-5
    )


def test_active_refinement_does_not_rebuild_lq_problem() -> None:
    import extension.joint_mpc_rti.solver.trajectory_scan as module

    source = inspect.getsource(module.solve_trajectory_qp_scan)
    assert "build_lq_problem" not in source
    assert source.count("refinement") >= 1


def test_production_scan_does_not_call_dense_reference() -> None:
    import extension.joint_mpc_rti.solver.trajectory_scan as module

    source = inspect.getsource(module.solve_trajectory_qp_scan)
    assert "solve_dense_qp" not in source


def test_lq_problem_uses_five_level_recovery_not_sequential_block_solve() -> None:
    import extension.joint_mpc_rti.solver.trajectory_scan as module

    source = inspect.getsource(module._solve_lq_problem)
    assert "_solve_augmented_associative" in source
    assert "_solve_block_pentadiagonal" not in inspect.getsource(module)
