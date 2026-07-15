from __future__ import annotations

import torch

from .helpers import make_state


def test_shift_warm_start_injects_measured_x0_and_shifts_controls() -> None:
    from extension.joint_mpc_rti.runtime.warm_start import shift_warm_start

    state = torch.arange(1 * 17 * 18, dtype=torch.float32).reshape(1, 17, 18)
    control = torch.arange(1 * 16 * 18, dtype=torch.float32).reshape(1, 16, 18)
    measured = torch.full((1, 18), -2.0)

    shifted = shift_warm_start(state, control, measured)

    torch.testing.assert_close(shifted.state[:, 0], measured)
    torch.testing.assert_close(shifted.state[:, 1:-1], state[:, 2:])
    torch.testing.assert_close(shifted.control[:, :-1], control[:, 1:])


def test_rollout_geometry_is_derived_from_root_and_joint_only() -> None:
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    result = rollout_controls(make_state(batch=2), torch.zeros(2, 16, 18), dt=0.02)

    assert result.state.shape == (2, 17, 18)
    assert result.foot_pos_w.shape == (2, 17, 4, 3)
    assert result.knee_pos_w.shape == (2, 17, 4, 3)
    assert result.shank_samples_w.shape == (2, 17, 4, 3, 3)
    assert result.body_samples_w.shape[:2] == (2, 17)
    assert not hasattr(result, "independent_foot_state")


def test_associative_scan_matches_sequential_affine_composition() -> None:
    from extension.joint_mpc_rti.solver.associative_scan import affine_scan

    generator = torch.Generator().manual_seed(7)
    matrix = 0.1 * torch.randn(4, 16, 2, 2, generator=generator)
    matrix = matrix + torch.eye(2).reshape(1, 1, 2, 2)
    bias = torch.randn(4, 16, 2, generator=generator)

    parallel = affine_scan(matrix, bias)
    state = torch.zeros(4, 2)
    sequence = []
    for index in range(16):
        state = torch.einsum("bij,bj->bi", matrix[:, index], state) + bias[:, index]
        sequence.append(state)
    sequential = torch.stack(sequence, dim=1)

    torch.testing.assert_close(parallel, sequential, atol=1.0e-5, rtol=1.0e-5)


def test_ilqr_matches_dense_scalar_integrator_solution() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem, solve_lq_subproblem

    batch = 3
    horizon = 16
    matrix_a = torch.ones(batch, horizon, 1, 1)
    matrix_b = torch.ones(batch, horizon, 1, 1)
    matrix_q = torch.ones(batch, horizon, 1, 1)
    matrix_r = 0.1 * torch.ones(batch, horizon, 1, 1)
    vector_q = torch.zeros(batch, horizon, 1)
    vector_r = torch.zeros(batch, horizon, 1)
    terminal_q = 4.0 * torch.ones(batch, 1, 1)
    terminal_vector = torch.zeros(batch, 1)
    initial_state = torch.tensor([[1.0], [0.5], [-0.75]])
    problem = LqProblem(
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        matrix_q=matrix_q,
        matrix_r=matrix_r,
        vector_q=vector_q,
        vector_r=vector_r,
        terminal_q=terminal_q,
        terminal_vector=terminal_vector,
        initial_state=initial_state,
        affine_dynamics=torch.zeros(batch, horizon, 1),
        matrix_s=None,
    )

    result = solve_lq_subproblem(problem, regularization=1.0e-8)

    lower = torch.tril(torch.ones(horizon, horizon))
    state_weight = torch.diag(torch.tensor([1.0] * (horizon - 1) + [4.0]))
    hessian = 0.1 * torch.eye(horizon) + lower.T @ state_weight @ lower
    gradient_map = lower.T @ state_weight @ torch.ones(horizon)
    expected = torch.stack(
        [torch.linalg.solve(hessian, -gradient_map * value) for value in initial_state[:, 0]],
        dim=0,
    ).unsqueeze(-1)

    torch.testing.assert_close(result.delta_control, expected, atol=3.0e-4, rtol=3.0e-4)


def test_analytic_dynamics_jacobian_matches_central_difference() -> None:
    from extension.joint_mpc_rti.model.dynamics import kinematic_step
    from extension.joint_mpc_rti.solver.linearization import dynamics_jacobians

    state = torch.tensor([[0.1, -0.2, 0.3, 0.1, -0.15, 0.4] + [0.2] * 12], dtype=torch.float64)
    control = torch.tensor([[0.3, -0.1, 0.05, 0.2, -0.1, 0.3] + [0.4] * 12], dtype=torch.float64)
    matrix_a, matrix_b = dynamics_jacobians(state, control, dt=0.02)
    epsilon = 1.0e-6
    finite_a = torch.empty_like(matrix_a)
    finite_b = torch.empty_like(matrix_b)
    for index in range(18):
        plus = state.clone(); plus[:, index] += epsilon
        minus = state.clone(); minus[:, index] -= epsilon
        finite_a[:, :, index] = (kinematic_step(plus, control, dt=0.02) - kinematic_step(minus, control, dt=0.02)) / (2 * epsilon)
        plus_u = control.clone(); plus_u[:, index] += epsilon
        minus_u = control.clone(); minus_u[:, index] -= epsilon
        finite_b[:, :, index] = (kinematic_step(state, plus_u, dt=0.02) - kinematic_step(state, minus_u, dt=0.02)) / (2 * epsilon)

    torch.testing.assert_close(matrix_a, finite_a, atol=2.0e-5, rtol=2.0e-4)
    torch.testing.assert_close(matrix_b, finite_b, atol=2.0e-5, rtol=2.0e-4)


def test_gauss_newton_blocks_match_linear_residual() -> None:
    from extension.joint_mpc_rti.solver.gauss_newton import ResidualLinearization, build_gauss_newton_lq

    residual = torch.tensor([[[-3.0]]])
    jacobian_x = torch.tensor([[[[1.0]]]])
    jacobian_u = torch.tensor([[[[2.0]]]])
    linearization = ResidualLinearization(
        residual=residual,
        jacobian_x=jacobian_x,
        jacobian_u=jacobian_u,
        terminal_residual=torch.zeros(1, 1),
        terminal_jacobian_x=torch.zeros(1, 1, 1),
    )

    problem = build_gauss_newton_lq(
        linearization,
        matrix_a=torch.ones(1, 1, 1, 1),
        matrix_b=torch.ones(1, 1, 1, 1),
        affine_dynamics=torch.zeros(1, 1, 1),
        initial_delta_state=torch.zeros(1, 1),
        regularization=0.0,
    )

    torch.testing.assert_close(problem.matrix_q, torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(problem.matrix_r, torch.tensor([[[[4.0]]]]))
    torch.testing.assert_close(problem.matrix_s, torch.tensor([[[[2.0]]]]))
    torch.testing.assert_close(problem.vector_q, torch.tensor([[[-3.0]]]))
    torch.testing.assert_close(problem.vector_r, torch.tensor([[[-6.0]]]))


def test_parallel_line_search_selects_lowest_improving_candidate() -> None:
    from extension.joint_mpc_rti.solver.line_search import parallel_line_search

    base = torch.zeros(2, 4, 1)
    delta = torch.ones_like(base)

    def merit(control: torch.Tensor) -> torch.Tensor:
        return ((control - 0.5) ** 2).mean(dim=(1, 2))

    result = parallel_line_search(base, delta, merit, alphas=(1.0, 0.5, 0.25))

    torch.testing.assert_close(result.alpha, torch.full((2,), 0.5))
    torch.testing.assert_close(result.control, torch.full_like(base, 0.5))


def test_sqp_rti_update_reduces_merit_with_one_lq_solve() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem
    from extension.joint_mpc_rti.solver.sqp_rti import sqp_rti_update

    batch = 2
    horizon = 16
    base = torch.zeros(batch, horizon, 1)
    problem = LqProblem(
        matrix_a=torch.ones(batch, horizon, 1, 1),
        matrix_b=torch.zeros(batch, horizon, 1, 1),
        matrix_q=torch.zeros(batch, horizon, 1, 1),
        matrix_r=torch.ones(batch, horizon, 1, 1),
        vector_q=torch.zeros(batch, horizon, 1),
        vector_r=-torch.ones(batch, horizon, 1),
        terminal_q=torch.zeros(batch, 1, 1),
        terminal_vector=torch.zeros(batch, 1),
        initial_state=torch.zeros(batch, 1),
        affine_dynamics=torch.zeros(batch, horizon, 1),
        matrix_s=torch.zeros(batch, horizon, 1, 1),
    )

    def merit(control: torch.Tensor) -> torch.Tensor:
        return ((control - 0.5) ** 2).mean(dim=(1, 2))

    result = sqp_rti_update(
        base_control=base,
        lq_problem=problem,
        merit_fn=merit,
        regularization=1.0e-8,
        alphas=(1.0, 0.5, 0.25),
    )

    assert torch.all(result.merit_after < result.merit_before)
    torch.testing.assert_close(result.alpha, torch.full((batch,), 0.5))
