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


def test_go2_block_ilqr_matches_generic_structured_solution() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import (
        LqProblem,
        solve_go2_block_lq_subproblem,
        solve_lq_subproblem,
    )

    generator = torch.Generator().manual_seed(19)
    batch, horizon = 2, 16
    matrix_a = torch.eye(18).view(1, 1, 18, 18).expand(batch, horizon, -1, -1).clone()
    matrix_b = 0.02 * torch.eye(18).view(1, 1, 18, 18).expand(batch, horizon, -1, -1).clone()
    matrix_a[:, :, :6, :6] += 0.01 * torch.randn(batch, horizon, 6, 6, generator=generator)
    matrix_b[:, :, :6, :6] += 0.01 * torch.randn(batch, horizon, 6, 6, generator=generator)
    matrix_q = torch.diag_embed(0.1 + torch.rand(batch, horizon, 18, generator=generator))
    matrix_r = torch.diag_embed(0.2 + torch.rand(batch, horizon, 18, generator=generator))
    terminal_q = torch.diag_embed(0.1 + torch.rand(batch, 18, generator=generator))
    problem = LqProblem(
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        matrix_q=matrix_q,
        matrix_r=matrix_r,
        vector_q=torch.randn(batch, horizon, 18, generator=generator),
        vector_r=torch.randn(batch, horizon, 18, generator=generator),
        terminal_q=terminal_q,
        terminal_vector=torch.randn(batch, 18, generator=generator),
        initial_state=torch.randn(batch, 18, generator=generator),
        affine_dynamics=0.01 * torch.randn(batch, horizon, 18, generator=generator),
        matrix_s=torch.zeros(batch, horizon, 18, 18),
    )

    generic = solve_lq_subproblem(problem, regularization=1.0e-4)
    blocked = solve_go2_block_lq_subproblem(problem, regularization=1.0e-4)

    torch.testing.assert_close(blocked.delta_state, generic.delta_state, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(blocked.delta_control, generic.delta_control, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(blocked.dual, generic.dual, atol=2.0e-5, rtol=2.0e-5)


def test_diagonal_ilqr_matches_generic_diagonal_solution() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import (
        LqProblem,
        solve_diagonal_lq_subproblem,
        solve_lq_subproblem,
    )

    generator = torch.Generator().manual_seed(23)
    batch, horizon, dimension = 3, 16, 18
    problem = LqProblem(
        matrix_a=torch.diag_embed(0.9 + 0.1 * torch.rand(batch, horizon, dimension, generator=generator)),
        matrix_b=torch.diag_embed(0.01 + 0.02 * torch.rand(batch, horizon, dimension, generator=generator)),
        matrix_q=torch.diag_embed(0.1 + torch.rand(batch, horizon, dimension, generator=generator)),
        matrix_r=torch.diag_embed(0.2 + torch.rand(batch, horizon, dimension, generator=generator)),
        vector_q=torch.randn(batch, horizon, dimension, generator=generator),
        vector_r=torch.randn(batch, horizon, dimension, generator=generator),
        terminal_q=torch.diag_embed(0.1 + torch.rand(batch, dimension, generator=generator)),
        terminal_vector=torch.randn(batch, dimension, generator=generator),
        initial_state=torch.randn(batch, dimension, generator=generator),
        affine_dynamics=0.01 * torch.randn(batch, horizon, dimension, generator=generator),
        matrix_s=torch.zeros(batch, horizon, dimension, dimension),
    )

    generic = solve_lq_subproblem(problem, regularization=1.0e-4)
    diagonal = solve_diagonal_lq_subproblem(problem, regularization=1.0e-4)

    torch.testing.assert_close(diagonal.delta_state, generic.delta_state, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(diagonal.delta_control, generic.delta_control, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(diagonal.dual, generic.dual, atol=2.0e-5, rtol=2.0e-5)


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


def test_parallel_line_search_evaluates_base_merit_once() -> None:
    from extension.joint_mpc_rti.solver.line_search import parallel_line_search

    calls: list[int] = []

    def merit(control: torch.Tensor) -> torch.Tensor:
        calls.append(int(control.shape[0]))
        return control.square().sum(dim=(1, 2))

    base = torch.zeros(2, 16, 18)
    delta = torch.ones_like(base)
    result = parallel_line_search(base, delta, merit, alphas=(1.0, 0.5, 0.25))

    assert calls == [8]
    torch.testing.assert_close(result.base_merit, torch.zeros(2))


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


def test_planner_packs_linearization_geometry_into_one_world_query(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from .helpers import make_command, make_flat_field

    calls: list[tuple[int, ...]] = []
    original = planner.query_world_maybe_compiled

    def counted_query(field, points_w, *, enabled):
        calls.append(tuple(points_w.shape))
        return original(field, points_w, enabled=enabled)

    monkeypatch.setattr(planner, "query_world_maybe_compiled", counted_query)
    planner.step(make_state(1), make_command(1), make_flat_field(1), None, JointMpcRtiCfg())

    assert calls == [(1, 17 * (9 + 4 + 4 + 12 + 12 + 1), 3)]


def test_planner_skips_final_named_diagnostics_when_disabled(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from .helpers import make_command, make_flat_field

    calls: list[int] = []
    original = planner.rollout_loss_breakdown_maybe_compiled

    def counted_objective(**kwargs):
        calls.append(int(kwargs["rollout"].state.shape[0]))
        return original(**kwargs)

    cfg = JointMpcRtiCfg()
    cfg.solver.emit_loss_breakdown = False
    monkeypatch.setattr(planner, "rollout_loss_breakdown_maybe_compiled", counted_objective)
    result = planner.step(make_state(1), make_command(1), make_flat_field(1), None, cfg)

    assert calls == [1, 3]
    assert set(result.full_trajectory.loss_breakdown) == {
        "merit_before",
        "merit_after",
        "line_search_alpha",
    }


def test_small_foot_calf_thigh_clearance_each_changes_lq_joint_gradient() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery
    from .helpers import make_state

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, 16, 18)
    rollout = rollout_controls(state, control, dt=0.02)
    joint_target = rollout.state[..., 6:].clone()
    base = planner._build_lq_problem(rollout, control, joint_target, state, cfg)
    nodes = int(rollout.state.shape[1])

    def query(point_count: int) -> JointMpcTerrainQuery:
        shape = (1, nodes, point_count)
        gradient = torch.zeros(*shape, 2)
        gradient[..., 0] = 1.0
        return JointMpcTerrainQuery(
            height_w=torch.full(shape, 0.16),
            small_distance_m=torch.full(shape, -0.02),
            large_distance_m=torch.full(shape, 1.0),
            small_gradient_w=gradient,
            large_gradient_w=torch.zeros_like(gradient),
            valid=torch.ones(shape, dtype=torch.bool),
        )

    queries = planner._LinearizationQueries(
        body=query(9),
        foot=query(4),
        knee=query(4),
        shank=query(12),
        thigh=query(12),
        root=query(1),
    )

    for active_name in (
        "small_object_foot_clearance",
        "small_object_calf_clearance",
        "small_object_thigh_clearance",
    ):
        local_cfg = JointMpcRtiCfg()
        local_cfg.losses.small_object_foot_clearance = 0.0
        local_cfg.losses.small_object_calf_clearance = 0.0
        local_cfg.losses.small_object_thigh_clearance = 0.0
        setattr(local_cfg.losses, active_name, 10.0)
        changed = planner._add_small_obstacle_linearization(base, rollout, queries, local_cfg)
        assert torch.count_nonzero(changed.vector_q[..., 6:]) > 0
        torch.testing.assert_close(changed.vector_q[..., :2], base.vector_q[..., :2])
