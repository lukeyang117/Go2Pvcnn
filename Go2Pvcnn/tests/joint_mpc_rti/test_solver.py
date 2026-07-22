from __future__ import annotations

import pytest
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


def test_dense_ilqr_preserves_coupled_control_hessian_direction() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem, solve_lq_subproblem

    terminal_q = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]])
    matrix_r = 0.5 * torch.eye(2).reshape(1, 1, 2, 2)
    vector_r = torch.tensor([[[-1.0, 0.0]]])
    problem = LqProblem(
        matrix_a=torch.zeros(1, 1, 2, 2),
        matrix_b=torch.eye(2).reshape(1, 1, 2, 2),
        matrix_q=torch.zeros(1, 1, 2, 2),
        matrix_r=matrix_r,
        vector_q=torch.zeros(1, 1, 2),
        vector_r=vector_r,
        terminal_q=terminal_q,
        terminal_vector=torch.zeros(1, 2),
        initial_state=torch.zeros(1, 2),
        affine_dynamics=torch.zeros(1, 1, 2),
        matrix_s=torch.zeros(1, 1, 2, 2),
    )

    result = solve_lq_subproblem(problem, regularization=1.0e-8)
    expected = -torch.linalg.solve(matrix_r[0, 0] + terminal_q[0], vector_r[0, 0])

    torch.testing.assert_close(result.delta_control[0, 0], expected, atol=1.0e-6, rtol=1.0e-6)
    assert abs(float(result.delta_control[0, 0, 1])) > 0.1


def test_dense_ilqr_regularizes_an_indefinite_control_hessian_per_batch() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem, solve_lq_subproblem

    problem = LqProblem(
        matrix_a=torch.zeros(2, 1, 2, 2),
        matrix_b=torch.eye(2).reshape(1, 1, 2, 2).expand(2, -1, -1, -1).clone(),
        matrix_q=torch.zeros(2, 1, 2, 2),
        matrix_r=torch.stack((torch.eye(2), -torch.eye(2))).unsqueeze(1),
        vector_q=torch.zeros(2, 1, 2),
        vector_r=torch.ones(2, 1, 2),
        terminal_q=torch.zeros(2, 2, 2),
        terminal_vector=torch.zeros(2, 2),
        initial_state=torch.zeros(2, 2),
        affine_dynamics=torch.zeros(2, 1, 2),
        matrix_s=torch.zeros(2, 1, 2, 2),
    )

    result = solve_lq_subproblem(problem, regularization=1.0e-4)

    assert torch.isfinite(result.delta_control).all()


def test_fixed_spd_solve_matches_dense_batched_solution() -> None:
    from extension.joint_mpc_rti.solver.fixed_spd import fixed_spd_solve

    generator = torch.Generator().manual_seed(37)
    factor = torch.randn(3, 5, 5, generator=generator)
    matrix = factor @ factor.transpose(-1, -2) + 0.5 * torch.eye(5)
    rhs = torch.randn(3, 5, 4, generator=generator)

    actual = fixed_spd_solve(matrix, rhs)
    expected = torch.linalg.solve(matrix, rhs)

    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_joint_kkt_compile_budget_keeps_triton_blocks_below_incident_shape() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import joint_kkt_compile_budget

    budget = joint_kkt_compile_budget(constraint_rows=32, state_dim=18)

    assert budget.constraint_rows == 32
    assert budget.constraint_block == 32
    assert budget.combined_rhs_columns == 51
    assert budget.rhs_block == 64


def test_joint_kkt_compile_budget_rejects_more_than_32_constraint_rows() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import joint_kkt_compile_budget

    with pytest.raises(ValueError, match="constraint_rows must be <= 32"):
        joint_kkt_compile_budget(constraint_rows=33, state_dim=18)


def test_lq_solver_rejects_unsafe_kkt_shape_before_compilation() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem, solve_lq_subproblem

    problem = LqProblem(
        matrix_a=torch.zeros(1, 1, 18, 18),
        matrix_b=torch.zeros(1, 1, 18, 18),
        matrix_q=torch.zeros(1, 1, 18, 18),
        matrix_r=torch.eye(18).reshape(1, 1, 18, 18),
        vector_q=torch.zeros(1, 1, 18),
        vector_r=torch.zeros(1, 1, 18),
        terminal_q=torch.eye(18).reshape(1, 18, 18),
        terminal_vector=torch.zeros(1, 18),
        initial_state=torch.zeros(1, 18),
        affine_dynamics=torch.zeros(1, 1, 18),
        constraint_control=torch.zeros(1, 1, 33, 18),
        constraint_state=torch.zeros(1, 1, 33, 18),
        constraint_residual=torch.zeros(1, 1, 33),
    )

    with pytest.raises(ValueError, match="constraint_rows must be <= 32"):
        solve_lq_subproblem(problem, regularization=1.0e-4)


def test_fixed_general_solve_matches_dense_batched_solution() -> None:
    from extension.joint_mpc_rti.solver.fixed_general import fixed_general_solve

    generator = torch.Generator().manual_seed(41)
    matrix = 0.1 * torch.randn(3, 5, 5, generator=generator) + 2.0 * torch.eye(5)
    rhs = torch.randn(3, 5, 4, generator=generator)

    actual = fixed_general_solve(matrix, rhs)
    expected = torch.linalg.solve(matrix, rhs)

    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fixed solve requires a GPU")
def test_fixed_general_cuda_solve_matches_dense_batched_solution() -> None:
    from extension.joint_mpc_rti.solver.fixed_general import fixed_general_solve

    generator = torch.Generator(device="cuda").manual_seed(43)
    matrix = 0.1 * torch.randn(8, 5, 5, generator=generator, device="cuda")
    matrix = matrix + 2.0 * torch.eye(5, device="cuda")
    rhs = torch.randn(8, 5, 4, generator=generator, device="cuda")

    actual = fixed_general_solve(matrix, rhs)
    expected = torch.linalg.solve(matrix, rhs)

    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_conditional_value_factor_composition_is_associative() -> None:
    from extension.joint_mpc_rti.solver.associative_tvlqr import (
        combine_conditional_value_factors,
    )

    generator = torch.Generator().manual_seed(47)

    def factor():
        matrix_a = 0.2 * torch.randn(2, 4, 4, generator=generator)
        vector_c = 0.2 * torch.randn(2, 4, generator=generator)
        c_root = 0.1 * torch.randn(2, 4, 4, generator=generator)
        matrix_c = c_root @ c_root.transpose(-1, -2)
        vector_p = 0.2 * torch.randn(2, 4, generator=generator)
        p_root = 0.1 * torch.randn(2, 4, 4, generator=generator)
        matrix_p = p_root @ p_root.transpose(-1, -2)
        return matrix_a, vector_c, matrix_c, vector_p, matrix_p

    first, second, third = factor(), factor(), factor()
    left_grouped = combine_conditional_value_factors(
        combine_conditional_value_factors(first, second),
        third,
    )
    right_grouped = combine_conditional_value_factors(
        first,
        combine_conditional_value_factors(second, third),
    )

    for actual, expected in zip(left_grouped, right_grouped):
        torch.testing.assert_close(actual, expected, atol=3.0e-5, rtol=3.0e-5)


def test_control_constraint_parameterization_satisfies_linearized_equality() -> None:
    from extension.joint_mpc_rti import planner

    constraint_control = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    constraint_state = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]])
    feedback, projector = planner._control_constraint_parameterization(
        constraint_control,
        constraint_state,
    )
    free_feedback = torch.tensor([[[[0.4, -0.2], [0.1, 0.3]]]])
    recovered = feedback + torch.matmul(projector, free_feedback)

    residual = torch.matmul(constraint_control, recovered) + constraint_state
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1.0e-5, rtol=0.0)


def test_affine_control_constraint_parameterization_restores_anchor_residual() -> None:
    from extension.joint_mpc_rti import planner

    constraint_control = torch.eye(2).reshape(1, 1, 2, 2)
    constraint_state = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]])
    constraint_residual = torch.tensor([[[0.004, -0.006]]])
    feedback, feedforward, projector = planner._affine_control_constraint_parameterization(
        constraint_control,
        constraint_state,
        constraint_residual,
    )
    delta_state = torch.tensor([[[0.1, -0.2]]])
    free_control = torch.tensor([[[0.4, 0.3]]])
    recovered = (
        torch.matmul(feedback, delta_state.unsqueeze(-1)).squeeze(-1)
        + feedforward
        + torch.matmul(projector, free_control.unsqueeze(-1)).squeeze(-1)
    )

    residual = (
        torch.matmul(constraint_control, recovered.unsqueeze(-1)).squeeze(-1)
        + torch.matmul(constraint_state, delta_state.unsqueeze(-1)).squeeze(-1)
        + constraint_residual
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1.0e-5, rtol=0.0)


def test_eliminated_stance_constraints_hold_after_lq_direction_recovery() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.go2_kinematics import complete_foot_jacobian
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import solve_lq_subproblem

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    command = torch.tensor([[0.2, 0.05, 0.0]])
    phase = torch.zeros(1, dtype=torch.long)
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    contact[:, 1, 0] = False
    desired, joint_target = planner._desired_control(state, command, contact, phase, cfg)
    rollout = rollout_controls(state, desired, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(rollout, desired, joint_target, state, command, cfg)

    stance_anchor = rollout.foot_pos_w.clone()
    continuing = torch.logical_and(contact[:, :-1], contact[:, 1:])
    stance_anchor[:, 1:, :, 0] = torch.where(
        continuing,
        stance_anchor[:, 1:, :, 0] + 0.004,
        stance_anchor[:, 1:, :, 0],
    )
    transformed, feedback, feedforward, projector = planner._eliminate_stance_control_constraints(
        problem,
        rollout,
        contact,
        stance_anchor,
    )
    free_solution = solve_lq_subproblem(transformed, regularization=cfg.solver.regularization)
    recovered_control = (
        torch.matmul(feedback, free_solution.delta_state[:, :-1].unsqueeze(-1)).squeeze(-1)
        + feedforward
        + torch.matmul(projector, free_solution.delta_control.unsqueeze(-1)).squeeze(-1)
    )

    batch, horizon = problem.matrix_a.shape[:2]
    state_next = rollout.state[:, 1:].reshape(batch * horizon, 18)
    foot_jacobian = complete_foot_jacobian(
        state_next[:, :3], state_next[:, 3:6], state_next[:, 6:]
    ).reshape(batch, horizon, 4, 3, 18)[..., :2, :]
    constraint_jacobian = (
        foot_jacobian * continuing[..., None, None]
    ).reshape(batch, horizon, 8, 18)
    constraint_control = torch.matmul(constraint_jacobian, problem.matrix_b)
    constraint_state = torch.matmul(constraint_jacobian, problem.matrix_a)
    constraint_residual = (
        (rollout.foot_pos_w[:, 1:, :, :2] - stance_anchor[:, 1:, :, :2])
        * continuing[..., None]
    ).reshape(batch, horizon, 8)
    residual = (
        torch.matmul(constraint_control, recovered_control.unsqueeze(-1)).squeeze(-1)
        + torch.matmul(
            constraint_state,
            free_solution.delta_state[:, :-1].unsqueeze(-1),
        ).squeeze(-1)
        + constraint_residual
    )

    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2.0e-5, rtol=0.0)


def test_sqp_line_search_uses_recovered_control_direction() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem, LqSolution
    from extension.joint_mpc_rti.solver.sqp_rti import sqp_rti_update

    problem = LqProblem(
        matrix_a=torch.zeros(1, 1, 1, 1),
        matrix_b=torch.ones(1, 1, 1, 1),
        matrix_q=torch.zeros(1, 1, 1, 1),
        matrix_r=torch.ones(1, 1, 1, 1),
        vector_q=torch.zeros(1, 1, 1),
        vector_r=torch.tensor([[[-1.0]]]),
        terminal_q=torch.zeros(1, 1, 1),
        terminal_vector=torch.zeros(1, 1),
        initial_state=torch.zeros(1, 1),
        affine_dynamics=torch.zeros(1, 1, 1),
    )

    def recover(solution: LqSolution) -> Tensor:
        return 0.25 * torch.ones_like(solution.delta_control)

    result = sqp_rti_update(
        base_control=torch.zeros(1, 1, 1),
        lq_problem=problem,
        merit_fn=lambda control: (control - 0.25).square().flatten(1).mean(1),
        regularization=1.0e-6,
        alphas=(1.0,),
        coupled_state_riccati=True,
        recover_control_direction=recover,
    )

    torch.testing.assert_close(result.delta_control, torch.full((1, 1, 1), 0.25))
    torch.testing.assert_close(result.control, torch.full((1, 1, 1), 0.25))


def test_constrained_riccati_satisfies_stage_control_equalities() -> None:
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem, solve_lq_subproblem

    horizon = 3
    matrix_a = torch.eye(2).reshape(1, 1, 2, 2).expand(1, horizon, -1, -1).clone()
    matrix_b = torch.eye(2).reshape(1, 1, 2, 2).expand(1, horizon, -1, -1).clone()
    constraint_control = torch.tensor([[[[1.0, 0.0]]]]).expand(1, horizon, -1, -1).clone()
    constraint_state = torch.tensor([[[[0.5, 0.0]]]]).expand(1, horizon, -1, -1).clone()
    constraint_residual = torch.tensor([[[-0.25], [0.10], [-0.05]]])
    problem = LqProblem(
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        matrix_q=torch.eye(2).reshape(1, 1, 2, 2).expand(1, horizon, -1, -1).clone(),
        matrix_r=0.2 * torch.eye(2).reshape(1, 1, 2, 2).expand(1, horizon, -1, -1).clone(),
        vector_q=torch.zeros(1, horizon, 2),
        vector_r=torch.tensor([[[-1.0, -0.2], [-0.5, 0.3], [-0.1, -0.4]]]),
        terminal_q=torch.eye(2).reshape(1, 2, 2),
        terminal_vector=torch.zeros(1, 2),
        initial_state=torch.tensor([[0.4, -0.2]]),
        affine_dynamics=torch.zeros(1, horizon, 2),
        constraint_control=constraint_control,
        constraint_state=constraint_state,
        constraint_residual=constraint_residual,
    )

    solution = solve_lq_subproblem(problem, regularization=1.0e-6)
    residual = (
        torch.matmul(constraint_control, solution.delta_control.unsqueeze(-1)).squeeze(-1)
        + torch.matmul(
            constraint_state,
            solution.delta_state[:, :-1].unsqueeze(-1),
        ).squeeze(-1)
        + constraint_residual
    )

    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2.0e-5, rtol=0.0)
    assert torch.count_nonzero(solution.delta_control[..., 1]) == horizon


def test_default_solver_keeps_full_coupled_state_blocks() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()

    assert cfg.solver.coupled_state_riccati
    assert not cfg.solver.diagonal_state_riccati


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


def test_parallel_line_search_keeps_required_correction_at_every_alpha() -> None:
    from extension.joint_mpc_rti.solver.line_search import parallel_line_search

    base = torch.zeros(1, 1, 1)
    required = torch.ones_like(base)
    free = torch.full_like(base, 2.0)
    evaluated: list[torch.Tensor] = []

    def merit(control: torch.Tensor) -> torch.Tensor:
        evaluated.append(control.detach().clone())
        return (control.flatten(1).mean(dim=1) - 1.5).square()

    result = parallel_line_search(
        base,
        free,
        merit,
        alphas=(1.0, 0.5, 0.25),
        required_control=required,
    )

    candidates = evaluated[0].reshape(1, 4, 1, 1)[0, :3, 0, 0]
    torch.testing.assert_close(candidates, torch.tensor([3.0, 2.0, 1.5]))
    torch.testing.assert_close(result.control, torch.tensor([[[1.5]]]))
    torch.testing.assert_close(result.alpha, torch.tensor([0.25]))


def test_parallel_line_search_bounds_each_control_direction_component() -> None:
    from extension.joint_mpc_rti.solver.line_search import parallel_line_search

    base = torch.zeros(1, 1, 2)
    delta = torch.full_like(base, 10.0)
    limit = torch.tensor([0.5, 2.0])

    result = parallel_line_search(
        base,
        delta,
        lambda control: (control - limit).square().flatten(1).mean(dim=1),
        alphas=(1.0, 0.25),
        delta_limit=limit,
    )

    torch.testing.assert_close(result.alpha, torch.ones(1))
    torch.testing.assert_close(result.control, limit.reshape(1, 1, 2))


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


def test_parallel_line_search_restores_constraint_before_comparing_merit() -> None:
    from extension.joint_mpc_rti.solver.line_search import parallel_line_search

    base = torch.zeros(1, 1, 1)
    delta = torch.ones_like(base)

    def merit_and_violation(control: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = control.flatten(1).mean(dim=1)
        merit = value.square()
        violation = (1.0 - value).clamp_min(0.0)
        return merit, violation

    result = parallel_line_search(
        base,
        delta,
        merit_and_violation,
        alphas=(1.0, 0.5, 0.25),
    )

    torch.testing.assert_close(result.alpha, torch.ones(1))
    torch.testing.assert_close(result.constraint_violation, torch.zeros(1))
    torch.testing.assert_close(result.base_constraint_violation, torch.ones(1))


def test_parallel_line_search_rejects_trade_between_constraint_components() -> None:
    from extension.joint_mpc_rti.solver.line_search import parallel_line_search

    base = torch.zeros(1, 1, 1)
    delta = torch.ones_like(base)

    def merit_and_violation(control: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = control.flatten(1).mean(dim=1)
        collision = torch.where(
            value > 0.75,
            value.new_full(value.shape, 0.090),
            torch.where(value > 0.25, value.new_full(value.shape, 0.095), value.new_full(value.shape, 0.100)),
        )
        stance = torch.where(value > 0.75, value.new_full(value.shape, 0.080), torch.zeros_like(value))
        return value.square(), torch.stack((collision, stance), dim=1)

    result = parallel_line_search(
        base,
        delta,
        merit_and_violation,
        alphas=(1.0, 0.5),
    )

    torch.testing.assert_close(result.alpha, torch.full((1,), 0.5))
    torch.testing.assert_close(result.constraint_violation, torch.full((1,), 0.095))
    torch.testing.assert_close(result.base_constraint_violation, torch.full((1,), 0.100))


def test_parallel_line_search_accepts_float32_ground_recovery_tolerance() -> None:
    from extension.joint_mpc_rti.solver.line_search import parallel_line_search

    base = torch.zeros(1, 1, 1)
    result = parallel_line_search(
        base,
        torch.ones_like(base),
        lambda candidate: (
            torch.zeros(candidate.shape[0]),
            torch.tensor(
                [[0.0, 0.0, 0.00892482, 0.03356878]],
                dtype=candidate.dtype,
            ).expand(candidate.shape[0], -1),
        ),
        alphas=(0.1,),
        base_merit=torch.ones(1),
        base_constraint_violation=torch.tensor([[0.0, 0.0, 0.00893692, 0.03356810]]),
        constraint_tolerance=torch.tensor([0.0, 0.0, 1.0e-5, 1.0e-5]),
    )

    assert not result.used_base.item()
    torch.testing.assert_close(result.alpha, torch.tensor([0.1]))


def test_x1_ground_violation_uses_signed_stage_a_gap_and_penetration_limits(
    monkeypatch,
) -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(batch=3)
    rollout = rollout_controls(
        state,
        torch.zeros(3, cfg.runtime.horizon_steps, 18),
        dt=cfg.runtime.dt,
    )
    foot = rollout.foot_pos_w.clone()
    signed_error = foot.new_tensor(
        [
            [0.0119, -0.0009, 0.0, 0.0],
            [0.0125, 0.0, 0.0, 0.0],
            [-0.0014, 0.0, 0.0, 0.0],
        ]
    )
    foot[:, 1, :, 2] = float(cfg.gait.foot_contact_offset) + signed_error
    rollout = replace(rollout, foot_pos_w=foot)

    def flat_query(_field, points, _cfg):
        shape = points.shape[:-1]
        scalar = points.new_zeros(shape)
        gradient = points.new_zeros(*shape, 2)
        return JointMpcTerrainQuery(
            height_w=scalar,
            small_distance_m=scalar.new_full(shape, 1.0),
            large_distance_m=scalar.new_full(shape, 1.0),
            small_gradient_w=gradient,
            large_gradient_w=gradient,
            valid=torch.ones(shape, dtype=torch.bool),
            height_gradient_w=gradient,
        )

    monkeypatch.setattr(planner, "_query_world", flat_query)
    monkeypatch.setattr(
        planner,
        "_small_link_collision_violation",
        lambda *_args, **_kwargs: torch.zeros(3),
    )
    monkeypatch.setattr(
        planner,
        "_root_attitude_violation_components",
        lambda *_args, **_kwargs: torch.zeros(3, 2),
    )
    monkeypatch.setattr(
        planner,
        "_recovery_landing_constraint_violation",
        lambda *_args, **_kwargs: torch.zeros(3),
    )
    monkeypatch.setattr(
        planner,
        "_leg_small_horizontal_clearance",
        lambda *_args, **_kwargs: torch.ones(3, 4),
    )

    contact = torch.ones(3, cfg.runtime.horizon_steps + 1, 4, dtype=torch.bool)
    components = planner._x1_constraint_violation_components(
        rollout,
        object(),
        cfg,
        contact,
        foot.clone(),
        torch.zeros(3, 4, dtype=torch.bool),
    )

    torch.testing.assert_close(
        components[:, 2],
        torch.tensor([0.0, 0.0005, 0.0004]),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert planner._line_search_constraint_tolerance(components, cfg)[2].item() == pytest.approx(
        1.0e-5
    )


def test_planner_accepts_recovery_step_with_ground_drift_inside_stance_contract() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import _line_search_constraint_tolerance
    from extension.joint_mpc_rti.solver.line_search import parallel_line_search

    cfg = JointMpcRtiCfg()
    base = torch.zeros(1, 1, 1)
    tolerance = _line_search_constraint_tolerance(base.new_zeros(1, 6), cfg)

    result = parallel_line_search(
        base,
        torch.ones_like(base),
        lambda candidate: (
                torch.zeros(candidate.shape[0]),
                candidate.new_tensor(
                    [[0.045684, 0.087703, 0.000007782, 0.0, 0.0, 0.080206]]
                ).expand(candidate.shape[0], -1),
        ),
        alphas=(0.5,),
        base_merit=torch.ones(1),
        base_constraint_violation=base.new_tensor(
            [[0.049301, 0.092938, 0.000007770, 0.0, 0.0, 0.083427]]
        ),
        constraint_tolerance=tolerance,
    )

    assert tolerance[2].item() == pytest.approx(1.0e-5)
    assert not result.used_base.item()
    torch.testing.assert_close(result.alpha, torch.tensor([0.5]))


def test_safer_rti_base_selection_uses_hold_without_constraint_trade() -> None:
    from extension.joint_mpc_rti.planner import _select_safer_base_control

    shaped = torch.ones(2, 1, 1)
    hold = torch.zeros_like(shaped)
    shaped_violation = torch.tensor([[0.0, 0.050, 0.020, 0.0], [0.0, 0.0, 0.0, 0.0]])
    hold_violation = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.010, 0.0, 0.0, 0.0]])

    selected = _select_safer_base_control(
        shaped,
        shaped_violation,
        hold,
        hold_violation,
    )

    torch.testing.assert_close(selected[0], hold[0])
    torch.testing.assert_close(selected[1], shaped[1])


def test_safer_rti_base_selection_protects_support_before_swing_recovery() -> None:
    from extension.joint_mpc_rti.planner import _select_safer_base_control

    shaped = torch.ones(1, 2, 3)
    hold = torch.zeros_like(shaped)
    selected = _select_safer_base_control(
        shaped,
        torch.tensor([[0.0, 0.028, 0.007, 0.108]]),
        hold,
        torch.tensor([[0.0, 0.0, 0.0, 0.137]]),
    )

    torch.testing.assert_close(selected, hold)


def test_hold_base_keeps_only_ground_safe_recovery_leg_control() -> None:
    from extension.joint_mpc_rti.planner import _hold_control_with_ground_safe_recovery

    shaped = torch.arange(36, dtype=torch.float32).reshape(1, 2, 18)
    ground_safe_recovery = torch.tensor([[False, False, True, False]])

    hold = _hold_control_with_ground_safe_recovery(
        shaped,
        ground_safe_recovery,
    )

    torch.testing.assert_close(hold[0, 0, :6], torch.zeros(6))
    shaped_joint = shaped[0, 0, 6:].reshape(4, 3)
    hold_joint = hold[0, 0, 6:].reshape(4, 3)
    torch.testing.assert_close(hold_joint[2], shaped_joint[2])
    torch.testing.assert_close(hold_joint[(0, 1, 3), :], torch.zeros(3, 3))
    torch.testing.assert_close(hold[:, 1:], shaped[:, 1:])


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

    nodes = JointMpcRtiCfg().runtime.horizon_steps + 1
    assert calls == [(1, nodes * (9 + 4 + 4 + 12 + 12 + 1), 3)]


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

    assert calls == [1, len(cfg.solver.line_search_alphas)]
    assert set(result.full_trajectory.loss_breakdown) == {
        "merit_before",
        "merit_after",
        "line_search_alpha",
        "collision_violation_before",
        "collision_violation_after",
    }


def test_warm_start_rebases_root_controls_to_current_command_reference() -> None:
    from extension.joint_mpc_rti.planner import _initial_control
    from extension.joint_mpc_rti.types import JointMpcRtiSolverState

    desired = torch.zeros(1, 16, 18)
    desired[..., 0] = -0.1
    desired[..., 1] = 0.2
    desired[..., 5] = 0.3
    previous = torch.arange(16 * 18, dtype=torch.float32).reshape(1, 16, 18)
    solver_state = JointMpcRtiSolverState(
        state=torch.zeros(1, 17, 18),
        control=previous,
        dual=None,
        previous_control=torch.zeros(1, 18),
    )

    warm = _initial_control(desired, solver_state)

    torch.testing.assert_close(warm[..., :6], desired[..., :6])
    torch.testing.assert_close(warm[:, :-1, 6:], previous[:, 1:, 6:])
    torch.testing.assert_close(warm[:, -1, 6:], previous[:, -1, 6:])


def test_warm_start_bounds_shifted_joint_control_around_current_reference() -> None:
    from extension.joint_mpc_rti.planner import _initial_control
    from extension.joint_mpc_rti.types import JointMpcRtiSolverState

    desired = torch.zeros(1, 16, 18)
    previous = torch.zeros_like(desired)
    previous[..., 6:] = 30.0
    solver_state = JointMpcRtiSolverState(
        state=torch.zeros(1, 17, 18),
        control=previous,
        dual=None,
        previous_control=torch.zeros(1, 18),
    )

    warm = _initial_control(desired, solver_state, joint_delta_limit=10.0)

    torch.testing.assert_close(warm[..., :6], desired[..., :6])
    torch.testing.assert_close(warm[..., 6:], torch.full_like(warm[..., 6:], 10.0))


def test_small_foot_calf_thigh_clearance_each_changes_coupled_lq_direction() -> None:
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
    base = planner._build_lq_problem(rollout, control, joint_target, state, torch.zeros(1, 3), cfg)
    nodes = int(rollout.state.shape[1])

    def query(point_count: int) -> JointMpcTerrainQuery:
        shape = (1, nodes, point_count)
        gradient = torch.zeros(*shape, 2)
        gradient[..., 0] = 1.0
        return JointMpcTerrainQuery(
                height_w=torch.full(shape, 1.0),
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
        local_cfg.losses.small_object_knee_clearance = 0.0
        local_cfg.losses.small_object_calf_clearance = 0.0
        local_cfg.losses.small_object_thigh_clearance = 0.0
        local_cfg.losses.small_object_base_clearance = 0.0
        setattr(local_cfg.losses, active_name, 10.0)
        changed = planner._add_small_obstacle_linearization(
            base,
            rollout,
            queries,
            local_cfg,
            torch.tensor([[0.2, 0.0, 0.0]]),
        )
        gradient_delta = changed.vector_q[..., 6:] - base.vector_q[..., 6:]
        root_gradient_delta = changed.vector_q[..., :6] - base.vector_q[..., :6]
        cross_delta = changed.matrix_q[..., :6, 6:] - base.matrix_q[..., :6, 6:]
        assert torch.count_nonzero(gradient_delta) > 0
        assert torch.count_nonzero(root_gradient_delta) > 0
        assert torch.count_nonzero(cross_delta) > 0
        torch.testing.assert_close(root_gradient_delta[..., 2], torch.zeros_like(root_gradient_delta[..., 2]))
        assert torch.linalg.vector_norm(gradient_delta[:, 1]) > 5.0 * torch.linalg.vector_norm(
            gradient_delta[:, 2]
        )


def test_small_collision_root_direction_can_use_bounded_lateral_and_rpy_assist() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    base = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    nodes = int(rollout.state.shape[1])

    def query(point_count: int) -> JointMpcTerrainQuery:
        shape = (1, nodes, point_count)
        gradient = torch.zeros(*shape, 2)
        gradient[..., 1] = 1.0
        return JointMpcTerrainQuery(
            height_w=torch.ones(shape),
            small_distance_m=torch.full(shape, -0.02),
            large_distance_m=torch.ones(shape),
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
    changed = planner._add_small_obstacle_linearization(
        base,
        rollout,
        queries,
        cfg,
        torch.tensor([[0.2, 0.0, 0.0]]),
    )

    assert torch.count_nonzero(changed.vector_q[..., 1] - base.vector_q[..., 1]) > 0
    torch.testing.assert_close(changed.vector_q[..., 2], base.vector_q[..., 2])
    assert torch.count_nonzero(changed.matrix_q[..., 3:6, 3:6] - base.matrix_q[..., 3:6, 3:6]) > 0
    assert torch.count_nonzero(changed.vector_q[..., 6:] - base.vector_q[..., 6:]) > 0


def test_small_knee_and_base_clearance_each_changes_the_lq_direction() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, 2, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    base = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    nodes = int(rollout.state.shape[1])

    def query(point_count: int) -> JointMpcTerrainQuery:
        shape = (1, nodes, point_count)
        gradient = torch.zeros(*shape, 2)
        gradient[..., 1] = 1.0
        return JointMpcTerrainQuery(
            height_w=torch.ones(shape),
            small_distance_m=torch.full(shape, -0.02),
            large_distance_m=torch.ones(shape),
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

    knee_cfg = JointMpcRtiCfg()
    knee_cfg.losses.small_object_foot_clearance = 0.0
    knee_cfg.losses.small_object_knee_clearance = 10.0
    knee_cfg.losses.small_object_calf_clearance = 0.0
    knee_cfg.losses.small_object_thigh_clearance = 0.0
    knee_cfg.losses.small_object_base_clearance = 0.0
    knee = planner._add_small_obstacle_linearization(
        base, rollout, queries, knee_cfg, torch.tensor([[0.2, 0.0, 0.0]])
    )
    assert torch.count_nonzero(knee.vector_q[..., :6] - base.vector_q[..., :6]) > 0
    assert torch.count_nonzero(knee.vector_q[..., 6:] - base.vector_q[..., 6:]) > 0

    base_cfg = JointMpcRtiCfg()
    base_cfg.losses.small_object_foot_clearance = 0.0
    base_cfg.losses.small_object_knee_clearance = 0.0
    base_cfg.losses.small_object_calf_clearance = 0.0
    base_cfg.losses.small_object_thigh_clearance = 0.0
    base_cfg.losses.small_object_base_clearance = 10.0
    body = planner._add_small_obstacle_linearization(
        base, rollout, queries, base_cfg, torch.tensor([[0.2, 0.0, 0.0]])
    )
    assert torch.count_nonzero(body.vector_q[..., :6] - base.vector_q[..., :6]) > 0
    torch.testing.assert_close(body.vector_q[..., 6:], base.vector_q[..., 6:])


def test_small_foot_clearance_lq_keeps_an_upward_direction_when_below_object_top() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, 2, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    base = planner._build_lq_problem(
        rollout, control, rollout.state[..., 6:], state, torch.zeros(1, 3), cfg
    )
    nodes = int(rollout.state.shape[1])

    def query(point_count: int) -> JointMpcTerrainQuery:
        shape = (1, nodes, point_count)
        return JointMpcTerrainQuery(
            height_w=torch.full(shape, 0.50),
            small_distance_m=torch.full(shape, -0.02),
            large_distance_m=torch.ones(shape),
            small_gradient_w=torch.zeros(*shape, 2),
            large_gradient_w=torch.zeros(*shape, 2),
            valid=torch.ones(shape, dtype=torch.bool),
        )

    queries = planner._LinearizationQueries(
        body=query(9), foot=query(4), knee=query(4), shank=query(12), thigh=query(12), root=query(1)
    )
    cfg.losses.small_object_knee_clearance = 0.0
    cfg.losses.small_object_calf_clearance = 0.0
    cfg.losses.small_object_thigh_clearance = 0.0
    cfg.losses.small_object_base_clearance = 0.0
    changed = planner._add_small_obstacle_linearization(base, rollout, queries, cfg)

    assert torch.linalg.vector_norm(changed.vector_q[..., 6:] - base.vector_q[..., 6:]) > 1.0


def test_sdf_corrected_swing_target_stays_above_small_object_until_safe_touchdown() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    target = torch.zeros(1, 2, 4, 3)
    target[..., 2] = 0.022
    shape = (1, 2, 4)
    query = JointMpcTerrainQuery(
        height_w=torch.full(shape, 0.08),
        small_distance_m=torch.full(shape, -0.02),
        large_distance_m=torch.ones(shape),
        small_gradient_w=torch.zeros(*shape, 2),
        large_gradient_w=torch.zeros(*shape, 2),
        valid=torch.ones(shape, dtype=torch.bool),
    )
    contact = torch.tensor([[[True, False, False, True], [True, False, False, True]]])

    corrected = planner._sdf_corrected_foot_targets(target, contact, query, cfg)

    assert torch.all(corrected[:, :, 1:3, 2] >= 0.095)
    torch.testing.assert_close(corrected[:, :, (0, 3), 2], target[:, :, (0, 3), 2])


def test_candidate_touchdown_requires_signed_distance_margin(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    rollout = rollout_controls(make_state(1), torch.zeros(1, 16, 18), dt=cfg.runtime.dt)

    def fixed_query(field, points_w, local_cfg):
        shape = points_w.shape[:-1]
        gradient = torch.zeros(*shape, 2)
        gradient[..., 0] = 1.0
        return JointMpcTerrainQuery(
            height_w=torch.zeros(shape),
            small_distance_m=torch.full(shape, 0.01),
            large_distance_m=torch.ones(shape),
            small_gradient_w=gradient,
            large_gradient_w=torch.zeros_like(gradient),
            valid=torch.ones(shape, dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "_query_world", fixed_query)
    clearance, jacobian = planner._candidate_x1_collision_constraints(
        rollout,
        object(),
        cfg,
        torch.zeros(1, 4, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
    )

    expected = 0.01 - cfg.gait.small_touchdown_margin
    assert clearance.shape == (1, 4, 9)
    assert jacobian.shape == (1, 4, 9, 3)
    torch.testing.assert_close(clearance[..., -1], torch.full((1, 4), expected))
    assert torch.all(jacobian[:, :, -1].abs().sum(dim=-1) > 0.0)


def test_candidate_collision_restoration_includes_knee_samples(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    rollout = rollout_controls(make_state(1), torch.zeros(1, 2, 18), dt=cfg.runtime.dt)

    def knee_only_collision(field, points_w, local_cfg):
        assert points_w.shape == (1, 32, 3)
        distance = torch.ones(1, 4, 8)
        distance[:, :, 1] = -0.02
        distance = distance.reshape(1, 32)
        return JointMpcTerrainQuery(
            height_w=torch.ones_like(distance),
            small_distance_m=distance,
            large_distance_m=torch.ones_like(distance),
            small_gradient_w=torch.ones(1, 32, 2),
            large_gradient_w=torch.zeros(1, 32, 2),
            valid=torch.ones(1, 32, dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "_query_world", knee_only_collision)
    clearance, jacobian = planner._candidate_x1_collision_constraints(
        rollout,
        object(),
        cfg,
        torch.zeros(1, 4, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert clearance.shape == (1, 4, 9)
    assert torch.all(clearance[:, :, 1] < 0.0)
    assert torch.all(jacobian[:, :, 1].abs().sum(dim=-1) > 0.0)


def test_candidate_collision_restoration_leaves_swing_to_lq_and_merit(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    rollout = rollout_controls(
        make_state(1),
        torch.zeros(1, cfg.runtime.horizon_steps, 18),
        dt=cfg.runtime.dt,
    )

    def horizontal_less_violated_than_vertical(field, points_w, local_cfg):
        shape = points_w.shape[:-1]
        gradient = torch.zeros(*shape, 2)
        gradient[..., 0] = 1.0
        return JointMpcTerrainQuery(
            height_w=torch.ones(shape),
            small_distance_m=torch.full(shape, 0.02),
            large_distance_m=torch.ones(shape),
            small_gradient_w=gradient,
            large_gradient_w=torch.zeros_like(gradient),
            valid=torch.ones(shape, dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "_query_world", horizontal_less_violated_than_vertical)
    clearance, jacobian = planner._candidate_x1_collision_constraints(
        rollout,
        object(),
        cfg,
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.bool),
    )

    torch.testing.assert_close(clearance[..., :8], torch.ones_like(clearance[..., :8]))
    torch.testing.assert_close(jacobian[..., :8, :], torch.zeros_like(jacobian[..., :8, :]))


def test_line_search_collision_violation_includes_knee_and_base(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    rollout = rollout_controls(make_state(1), torch.zeros(1, 2, 18), dt=cfg.runtime.dt)
    active_slice = [slice(13, 17)]

    def collision_query(field, points_w, local_cfg):
        assert points_w.shape == (1, 41, 3)
        shape = points_w.shape[:-1]
        distance = torch.ones(shape)
        distance[:, active_slice[0]] = -0.02
        return JointMpcTerrainQuery(
            height_w=torch.ones(shape),
            small_distance_m=distance,
            large_distance_m=torch.ones(shape),
            small_gradient_w=torch.zeros(*shape, 2),
            large_gradient_w=torch.zeros(*shape, 2),
            valid=torch.ones(shape, dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "_query_world", collision_query)
    knee_violation = planner._small_link_collision_violation(
        rollout, object(), cfg, torch.zeros(1, 4, dtype=torch.bool)
    )
    assert knee_violation.item() > 0.0

    active_slice[0] = slice(0, 9)
    base_violation = planner._small_link_collision_violation(
        rollout, object(), cfg, torch.zeros(1, 4, dtype=torch.bool)
    )
    assert base_violation.item() > 0.0


def test_line_search_collision_violation_allows_safe_stance_near_small_object(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    rollout = rollout_controls(make_state(1), torch.zeros(1, 2, 18), dt=cfg.runtime.dt)

    def safe_nearby_query(field, points_w, local_cfg):
        assert points_w.shape == (1, 41, 3)
        shape = points_w.shape[:-1]
        distance = torch.ones(shape)
        distance[:, 9:13] = 0.033
        return JointMpcTerrainQuery(
            height_w=torch.full(shape, -1.0),
            small_distance_m=distance,
            large_distance_m=torch.ones(shape),
            small_gradient_w=torch.zeros(*shape, 2),
            large_gradient_w=torch.zeros(*shape, 2),
            valid=torch.ones(shape, dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "_query_world", safe_nearby_query)
    violation = planner._small_link_collision_violation(
        rollout, object(), cfg, torch.ones(1, 4, dtype=torch.bool)
    )

    torch.testing.assert_close(violation, torch.zeros(1))


def test_recovery_landing_violation_covers_safe_grounding_and_sdf_exit() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    foot_pos = torch.tensor(
        [[[0.0, 0.0, 0.120], [0.0, 0.0, 0.200], [0.0, 0.0, 0.300], [0.0, 0.0, 0.400]]]
    )
    query = JointMpcTerrainQuery(
        height_w=torch.zeros(1, 4),
        small_distance_m=torch.tensor([[0.200, 0.010, 0.200, 0.200]]),
        large_distance_m=torch.ones(1, 4),
        small_gradient_w=torch.zeros(1, 4, 2),
        large_gradient_w=torch.zeros(1, 4, 2),
        valid=torch.tensor([[True, True, True, False]]),
    )

    violation = planner._recovery_landing_constraint_violation(
        foot_pos,
        query,
        contact_x1=torch.tensor([[False, False, True, False]]),
        recovery_state=torch.tensor([[True, True, True, True]]),
        cfg=cfg,
    )

    expected_ground_gap = 0.120 - cfg.gait.foot_contact_offset
    assert torch.allclose(violation, torch.tensor([expected_ground_gap]))


def test_recovery_sdf_exit_remains_active_above_small_obstacle() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    foot_pos = torch.tensor([[[0.0, 0.0, 0.300]] * 4])
    distance = torch.full((1, 4), 0.200)
    ground_safe_distance = cfg.gait.foot_collision_radius + cfg.gait.small_collision_margin_xy
    midpoint = 0.5 * (ground_safe_distance + cfg.gait.small_touchdown_margin)
    distance[:, 0] = midpoint
    query = JointMpcTerrainQuery(
        height_w=torch.zeros(1, 4),
        small_distance_m=distance,
        large_distance_m=torch.ones(1, 4),
        small_gradient_w=torch.zeros(1, 4, 2),
        large_gradient_w=torch.zeros(1, 4, 2),
        valid=torch.ones(1, 4, dtype=torch.bool),
    )

    violation = planner._recovery_landing_constraint_violation(
        foot_pos,
        query,
        contact_x1=torch.zeros(1, 4, dtype=torch.bool),
        recovery_state=torch.tensor([[True, False, False, False]]),
        cfg=cfg,
    )

    torch.testing.assert_close(
        violation,
        torch.tensor([cfg.gait.small_touchdown_margin - midpoint]),
    )


def test_recovery_vertical_clearance_decreases_as_sdf_exit_improves() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    foot_pos = torch.zeros(2, 4, 3)
    foot_pos[..., 2] = 0.180
    distance = torch.full((2, 4), 0.200)
    distance[:, 0] = torch.tensor([0.020, 0.040])
    query = JointMpcTerrainQuery(
        height_w=torch.zeros(2, 4),
        small_distance_m=distance,
        large_distance_m=torch.ones(2, 4),
        small_gradient_w=torch.zeros(2, 4, 2),
        large_gradient_w=torch.zeros(2, 4, 2),
        valid=torch.ones(2, 4, dtype=torch.bool),
    )

    violation = planner._recovery_landing_constraint_violation(
        foot_pos,
        query,
        contact_x1=torch.zeros(2, 4, dtype=torch.bool),
        recovery_state=torch.tensor(
            [[True, False, False, False], [True, False, False, False]]
        ),
        cfg=cfg,
    )

    assert violation[1] < violation[0]


def test_recovery_grounding_requires_full_leg_landing_clearance() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    active = planner._recovery_grounding_active_mask(
        recovery_state=torch.tensor([[True, True, True, False]]),
        contact_state=torch.tensor([[False, False, True, False]]),
        map_valid=torch.ones(1, 4, dtype=torch.bool),
        foot_small_distance_m=torch.tensor([[0.060, 0.060, 0.060, 0.060]]),
        leg_landing_clearance_m=torch.tensor([[0.020, -0.001, 0.020, 0.020]]),
        cfg=cfg,
    )

    torch.testing.assert_close(
        active,
        torch.tensor([[True, False, False, False]]),
    )


def test_recovery_grounding_uses_measured_clearance_not_unsafe_nominal_preview() -> None:
    from extension.joint_mpc_rti import planner

    stance, recovery = planner._split_stance_and_recovery_leg_clearance(
        measured_leg_clearance=torch.tensor([[0.020, 0.030]]),
        nominal_leg_clearance=torch.tensor([[-0.040, 0.010]]),
        stance_lookahead_margin=0.005,
    )

    torch.testing.assert_close(stance, torch.tensor([[-0.035, 0.015]]))
    torch.testing.assert_close(recovery, torch.tensor([[0.020, 0.030]]))


def test_recovery_exit_clearance_keeps_worst_leg_link_active() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    clearance = planner._recovery_exit_clearance(
        foot_small_distance_m=torch.tensor([[0.060, 0.060]]),
        leg_landing_clearance_m=torch.tensor([[0.020, -0.001]]),
        cfg=cfg,
    )

    torch.testing.assert_close(clearance, torch.tensor([[0.008, -0.011]]))


def test_candidate_collision_restoration_respects_joint_control_bound(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()

    def penetrating_constraints(rollout, field, local_cfg, contact_x0, contact_x1):
        clearance = torch.full((1, 4, 2), -1.0)
        jacobian = torch.zeros(1, 4, 2, 3)
        jacobian[..., 0] = 1.0
        return clearance, jacobian

    monkeypatch.setattr(planner, "_candidate_x1_collision_constraints", penetrating_constraints)
    monkeypatch.setattr(
        planner,
        "_small_link_collision_violation",
        lambda rollout, field, local_cfg, contact_x1: rollout.control.new_zeros((rollout.control.shape[0],)),
    )
    control, _ = planner._restore_candidate_collision_feasibility(
        make_state(1),
        torch.zeros(1, cfg.runtime.horizon_steps, 18),
        object(),
        cfg,
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.bool),
    )

    limit = cfg.gait.max_nominal_joint_velocity + cfg.solver.joint_direction_limit
    assert float(control[:, 0, 6:].abs().max()) <= limit


def test_minimum_norm_leg_correction_is_constraint_order_invariant() -> None:
    from extension.joint_mpc_rti import planner

    clearance = torch.full((1, 4, 2), -0.1)
    jacobian = torch.zeros(1, 4, 2, 3)
    jacobian[:, :, 0, 0] = 1.0
    jacobian[:, :, 1, :2] = 1.0

    forward = planner._minimum_norm_leg_correction(clearance, jacobian, max_norm=1.0)
    reverse = planner._minimum_norm_leg_correction(
        clearance.flip(2),
        jacobian.flip(2),
        max_norm=1.0,
    )

    torch.testing.assert_close(forward, reverse, atol=1.0e-6, rtol=1.0e-6)


def test_candidate_collision_restoration_keeps_lowest_violation_iterate(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    call_index = 0

    def oscillating_constraints(rollout, field, local_cfg, contact_x0, contact_x1):
        nonlocal call_index
        values = (-0.1, 0.1, -0.2, 0.2, -0.3, 0.3)
        gradients = (1.0, 1.0, -1.0, -1.0, 1.0, 1.0)
        clearance = torch.full((1, 4, 1), values[call_index])
        jacobian = torch.zeros(1, 4, 1, 3)
        jacobian[..., 0] = gradients[call_index]
        call_index += 1
        return clearance, jacobian

    monkeypatch.setattr(planner, "_candidate_x1_collision_constraints", oscillating_constraints)
    monkeypatch.setattr(
        planner,
        "_small_link_collision_violation",
        lambda rollout, field, local_cfg, contact_x1: rollout.control.new_zeros((rollout.control.shape[0],)),
    )
    control, _ = planner._restore_candidate_collision_feasibility(
        make_state(1),
        torch.zeros(1, cfg.runtime.horizon_steps, 18),
        object(),
        cfg,
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.bool),
    )

    torch.testing.assert_close(control[:, 0, 6::3], torch.full((1, 4), 5.0))


def test_candidate_collision_restoration_falls_back_to_safer_x1_hold(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()

    def unresolved_constraints(rollout, field, local_cfg, contact_x0, contact_x1):
        clearance = torch.full((1, 4, 1), -0.1)
        jacobian = torch.zeros(1, 4, 1, 3)
        jacobian[..., 0] = 1.0
        return clearance, jacobian

    def control_violation(rollout, field, local_cfg, contact_x1):
        return rollout.control[:, 0].abs().amax(dim=1)

    monkeypatch.setattr(planner, "_candidate_x1_collision_constraints", unresolved_constraints)
    monkeypatch.setattr(planner, "_small_link_collision_violation", control_violation)
    candidate = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    candidate[:, 0, 6::3] = 2.0
    control, _ = planner._restore_candidate_collision_feasibility(
        make_state(1),
        candidate,
        object(),
        cfg,
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.bool),
    )

    torch.testing.assert_close(control[:, 0], torch.zeros_like(control[:, 0]))


def test_candidate_collision_restoration_uses_exact_violation_for_hold_fallback(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()

    def approximate_constraints_report_safe(rollout, field, local_cfg, contact_x0, contact_x1):
        return torch.ones(1, 4, 1), torch.zeros(1, 4, 1, 3)

    def exact_control_violation(rollout, field, local_cfg, contact_x1):
        return rollout.control[:, 0].abs().amax(dim=1)

    monkeypatch.setattr(planner, "_candidate_x1_collision_constraints", approximate_constraints_report_safe)
    monkeypatch.setattr(planner, "_small_link_collision_violation", exact_control_violation)
    candidate = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    candidate[:, 0, 6] = 2.0

    control, _ = planner._restore_candidate_collision_feasibility(
        make_state(1),
        candidate,
        object(),
        cfg,
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.bool),
    )

    torch.testing.assert_close(control[:, 0], torch.zeros_like(control[:, 0]))


def test_line_search_candidates_use_fk_kkt_before_rollout() -> None:
    import inspect

    from extension.joint_mpc_rti import planner

    source = inspect.getsource(planner.step)
    candidate_path = source.split("def merit_fn(candidate_control: Tensor)", maxsplit=1)[1]
    candidate_path = candidate_path.split("update = sqp_rti_update", maxsplit=1)[0]

    assert "candidate_control, candidate_rollout = _apply_fk_contact_kkt(" in candidate_path


def test_production_step_uses_fk_kkt_without_local_ik_projection() -> None:
    import inspect

    from extension.joint_mpc_rti import planner

    source = inspect.getsource(planner.step)
    for helper in (
        "_enforce_first_stance_equality(",
        "_enforce_recovery_landing(",
        "_minimum_norm_leg_correction(",
        "_restore_candidate_collision_feasibility(",
    ):
        assert helper not in source, f"production step still calls local projection {helper}"


def test_production_fk_kkt_helper_has_no_local_ik_or_collision_repair() -> None:
    import inspect

    from extension.joint_mpc_rti import planner

    source = inspect.getsource(planner._apply_fk_contact_kkt)
    for helper in (
        "_enforce_startup_foot_lead(",
        "_enforce_joint_position_limits(",
        "_enforce_root_assist_limits(",
        "_enforce_first_stance_equality(",
        "_enforce_recovery_landing(",
        "_minimum_norm_leg_correction(",
        "_restore_candidate_collision_feasibility(",
    ):
        assert helper not in source, f"production FK/KKT helper still calls local repair {helper}"


def test_fk_stance_kkt_couples_root_and_joints_to_hold_world_anchors() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos
    from extension.joint_mpc_rti.model.dynamics import kinematic_step

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[:, 0, 0] = 0.8
    anchor = go2_foot_pos(
        measured.root_pos_w,
        measured.root_rpy_w,
        measured.joint_pos,
    )

    corrected = planner._apply_fk_stance_kkt_constraint(
        measured,
        control,
        torch.tensor([[0.8, 0.0, 0.0]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
        anchor,
        cfg,
    )
    x1 = kinematic_step(measured.as_vector(), corrected[:, 0], dt=cfg.runtime.dt)
    foot_x1 = go2_foot_pos(x1[:, :3], x1[:, 3:6], x1[:, 6:])

    assert corrected[0, 0, 0] < 0.8
    assert corrected[0, 0, 6:].abs().sum() > 0.0
    assert torch.linalg.vector_norm(foot_x1 - anchor, dim=-1).max() <= 0.0005


def test_candidate_joint_trajectory_is_projected_inside_kinematic_limits() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[..., 6:] = 30.0

    projected = planner._enforce_joint_position_limits(state, control, cfg)
    joint = state.joint_pos[:, None] + torch.cumsum(
        projected[..., 6:] * cfg.runtime.dt,
        dim=1,
    )
    lower = joint.new_tensor((-1.0472, -0.6632, -2.721) * 4)
    upper = joint.new_tensor((1.0472, 2.966, -0.837) * 4)
    margin = cfg.solver.joint_position_safety_margin_rad

    assert torch.all(joint >= lower + margin)
    assert torch.all(joint <= upper - margin)
    torch.testing.assert_close(projected[..., :6], control[..., :6])


def test_recovery_joint_reference_uses_nominal_stance_only_for_recovery_leg() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    target = torch.full((1, cfg.runtime.horizon_steps + 1, 12), 3.0)
    recovery = torch.tensor([[True, False, False, False]])

    corrected = planner._recovery_joint_targets(target, recovery, cfg)
    nominal = torch.tensor(cfg.gait.nominal_joint_pos).reshape(4, 3)

    torch.testing.assert_close(
        corrected.reshape(1, cfg.runtime.horizon_steps + 1, 4, 3)[:, :, 0],
        nominal[0].reshape(1, 1, 3).expand(1, cfg.runtime.horizon_steps + 1, 3),
    )
    torch.testing.assert_close(corrected[..., 3:], target[..., 3:])


def test_recovery_joint_reference_keeps_swing_shape_near_small_obstacle() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    nodes = cfg.runtime.horizon_steps + 1
    target = torch.full((1, nodes, 12), 3.0)
    recovery = torch.tensor([[True, False, True, False]])
    near_small = torch.zeros(1, nodes, 4, dtype=torch.bool)
    near_small[:, :, 0] = True

    corrected = planner._recovery_joint_targets(
        target,
        recovery,
        cfg,
        near_small=near_small,
    ).reshape(1, nodes, 4, 3)
    nominal = torch.tensor(cfg.gait.nominal_joint_pos).reshape(4, 3)

    torch.testing.assert_close(corrected[:, :, 0], torch.full_like(corrected[:, :, 0], 3.0))
    torch.testing.assert_close(
        corrected[:, :, 2],
        nominal[2].reshape(1, 1, 3).expand(1, nodes, 3),
    )
    torch.testing.assert_close(corrected[:, :, (1, 3)], torch.full_like(corrected[:, :, (1, 3)], 3.0))


def test_planner_defers_joint_bounds_until_preview_predicts_a_violation() -> None:
    from dataclasses import replace

    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    joint = state.joint_pos.clone()
    joint[:, 2::3] = -0.837
    state = replace(state, joint_pos=joint)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.zeros(1, 3),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
    )

    assert constrained.constraint_control.shape == (1, cfg.runtime.horizon_steps, 32, 18)
    active_joint_rows = constrained.constraint_control[..., -12:, :].abs().sum(dim=-1) > 0.0
    assert not torch.any(active_joint_rows)


def test_constrained_lq_full_step_preserves_nonlinear_x1_stance_anchor() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.go2_kinematics import go2_foot_pos
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.solver.primal_dual_ilqr import solve_lq_subproblem

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[..., 0] = 0.2
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    joint_target = rollout.state[..., 6:].clone()
    problem = planner._build_lq_problem(
        rollout,
        control,
        joint_target,
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    measured_foot = go2_foot_pos(state.root_pos_w, state.root_rpy_w, state.joint_pos)
    anchor = measured_foot[:, None].expand_as(rollout.foot_pos_w).clone()
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        anchor,
        cfg,
    )

    solution = solve_lq_subproblem(constrained, regularization=cfg.solver.regularization)
    corrected = rollout_controls(
        state,
        control + solution.delta_control,
        dt=cfg.runtime.dt,
    )
    stance = contact[:, 1]
    error = torch.linalg.vector_norm(corrected.foot_pos_w[:, 1, :, :2] - measured_foot[..., :2], dim=-1)

    assert torch.all(error[stance] <= cfg.solver.stance_equality_tolerance_m)


def test_stance_kkt_adds_active_full_body_collision_rows(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.zeros(1, 3),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )

    def calf_collision_query(field, points_w, local_cfg):
        assert points_w.shape == (1, 41, 3)
        distance = torch.ones(1, 41)
        distance[:, 17] = -0.01
        height = torch.zeros(1, 41)
        height[:, 17] = 0.16
        gradient = torch.zeros(1, 41, 2)
        gradient[:, 17, 0] = 1.0
        return JointMpcTerrainQuery(
            height_w=height,
            small_distance_m=distance,
            large_distance_m=torch.ones(1, 41),
            small_gradient_w=gradient,
            large_gradient_w=torch.zeros(1, 41, 2),
            valid=torch.ones(1, 41, dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "_query_world", calf_collision_query)
    query_shape = (1, cfg.runtime.horizon_steps + 1, 4)
    foot_query = JointMpcTerrainQuery(
        height_w=torch.zeros(query_shape),
        small_distance_m=torch.ones(query_shape),
        large_distance_m=torch.ones(query_shape),
        small_gradient_w=torch.zeros(*query_shape, 2),
        large_gradient_w=torch.zeros(*query_shape, 2),
        valid=torch.ones(query_shape, dtype=torch.bool),
    )
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        terrain_field=object(),
        foot_query=foot_query,
    )

    row = 12 + 1
    assert constrained.constraint_control.shape == (1, cfg.runtime.horizon_steps, 32, 18)
    assert constrained.constraint_residual[0, 0, row] < 0.0
    assert constrained.constraint_control[0, 0, row, :6].abs().sum() > 0.0
    assert constrained.constraint_control[0, 0, row, 6:].abs().sum() > 0.0


def test_dedicated_swing_foot_clearance_suppresses_duplicate_aggregated_foot_row(
    monkeypatch,
) -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.4, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    contact[:, 1, 0] = False

    def swing_foot_collision_query(field, points_w, local_cfg):
        assert points_w.shape == (1, 41, 3)
        distance = torch.ones(1, 41)
        distance[:, 9] = 0.0
        height = torch.zeros(1, 41)
        height[:, 9] = 0.20
        gradient = torch.zeros(1, 41, 2)
        gradient[:, 9, 0] = 1.0
        return JointMpcTerrainQuery(
            height_w=height,
            small_distance_m=distance,
            large_distance_m=torch.ones(1, 41),
            small_gradient_w=gradient,
            large_gradient_w=torch.zeros(1, 41, 2),
            valid=torch.ones(1, 41, dtype=torch.bool),
        )

    monkeypatch.setattr(planner, "_query_world", swing_foot_collision_query)
    query_shape = (1, cfg.runtime.horizon_steps + 1, 4)
    distance = torch.ones(query_shape)
    height = torch.zeros(query_shape)
    gradient = torch.zeros(*query_shape, 2)
    distance[:, 1, 0] = 0.0
    height[:, 1, 0] = 0.20
    gradient[:, 1, 0, 0] = 1.0
    foot_query = JointMpcTerrainQuery(
        height_w=height,
        small_distance_m=distance,
        large_distance_m=torch.ones(query_shape),
        small_gradient_w=gradient,
        large_gradient_w=torch.zeros_like(gradient),
        valid=torch.ones(query_shape, dtype=torch.bool),
    )

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
        foot_query=foot_query,
        swing_target_w=rollout.foot_pos_w,
        terrain_field=object(),
        startup_mask=torch.zeros(1, dtype=torch.bool),
        command_body=torch.tensor([[0.4, 0.0, 0.0]]),
    )

    assert constrained.constraint_control[0, 0, 2].abs().sum() > 0.0
    torch.testing.assert_close(
        constrained.constraint_control[0, 0, 13],
        torch.zeros_like(constrained.constraint_control[0, 0, 13]),
    )
    torch.testing.assert_close(constrained.constraint_residual[0, 0, 13], torch.tensor(0.0))


def test_collision_kkt_root_columns_follow_bounded_assist_projection() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    jacobian = torch.ones(1, 5, 18)
    state_x1 = make_state(1).as_vector()
    projected = planner._project_collision_kkt_root_assist(
        jacobian,
        selected_distance_m=torch.zeros(1, 5),
        state_x1=state_x1,
        command_body=torch.tensor([[0.4, 0.0, 0.0]]),
        cfg=cfg,
    )

    torch.testing.assert_close(projected[..., 0], torch.ones_like(projected[..., 0]))
    assert torch.all(projected[..., 1] > 0.0)
    assert torch.all(projected[..., 1] < 1.0)
    torch.testing.assert_close(projected[..., 2], torch.zeros_like(projected[..., 2]))
    assert torch.all(projected[..., 3:6] > 0.0)
    assert torch.all(projected[..., 3:6] < 1.0)
    torch.testing.assert_close(projected[..., 6:], jacobian[..., 6:])


def test_small_obstacle_envelope_floor_prevents_late_swing_descent() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule

    cfg = JointMpcRtiCfg()
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    floor = torch.zeros_like(contact, dtype=torch.float32)
    swing_leg = int(torch.logical_not(contact[0, -2]).nonzero()[0])
    floor[0, -2, swing_leg] = 1.0

    target = planner._nominal_joint_target(
        contact,
        torch.zeros(1, dtype=torch.long),
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
        dtype=torch.float32,
        swing_envelope_floor=floor,
    ).reshape(1, contact.shape[1], 4, 3)

    assert target[0, -2, swing_leg, 1] == pytest.approx(cfg.gait.swing_thigh_angle)
    assert target[0, -2, swing_leg, 2] == pytest.approx(cfg.gait.swing_calf_angle)


def test_recovery_near_small_mask_uses_touchdown_margin_not_influence_radius() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    distance = torch.tensor([[cfg.gait.small_touchdown_margin - 0.001, cfg.gait.small_touchdown_margin + 0.001]])

    near = planner._recovery_near_small_mask(distance, cfg)

    assert torch.equal(near, torch.tensor([[True, False]]))


def test_control_from_joint_target_carries_clipped_error_into_later_frames() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    target = state.joint_pos[:, None].expand(1, cfg.runtime.horizon_steps + 1, 12).clone()
    target[:, 1:, 0] += 0.40

    control = planner._control_from_joint_target(
        state,
        torch.zeros(1, 3),
        target,
        cfg,
        carry_clipped_error=True,
    )

    torch.testing.assert_close(
        control[0, :3, 6],
        torch.tensor([9.0, 9.0, 2.0]),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_startup_root_leak_reuses_fixed_collision_reserve_row() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[:, 0, 0] = 0.2
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        startup_mask=torch.ones(1, dtype=torch.bool),
        command_body=torch.tensor([[0.2, 0.0, 0.0]]),
    )

    reserve_row = 12 + 5
    assert constrained.constraint_residual[0, 0, reserve_row] > 0.0
    assert constrained.constraint_control[0, 0, reserve_row, :2].abs().sum() > 0.0
    torch.testing.assert_close(
        constrained.constraint_control[0, 1:, reserve_row],
        torch.zeros_like(constrained.constraint_control[0, 1:, reserve_row]),
    )


def test_startup_root_reserve_row_is_active_before_preview_exceeds_limit() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        startup_mask=torch.ones(1, dtype=torch.bool),
        command_body=torch.tensor([[0.2, 0.0, 0.0]]),
    )

    reserve_row = 12 + 5
    assert constrained.constraint_control[0, 0, reserve_row, :2].abs().sum() > 0.0
    torch.testing.assert_close(
        constrained.constraint_residual[0, 0, reserve_row],
        torch.tensor(-cfg.gait.startup_root_leak_limit_m),
    )


def test_root_attitude_uses_two_fixed_rows_within_32_row_budget() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
    )

    assert constrained.constraint_control.shape == (1, cfg.runtime.horizon_steps, 32, 18)
    attitude_rows = constrained.constraint_control[0, 0, 18:20]
    assert attitude_rows[:, 3:5].abs().sum() > 0.0


def test_root_attitude_uses_two_fixed_rows_within_32_row_budget() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
    )

    assert constrained.constraint_control.shape == (1, cfg.runtime.horizon_steps, 32, 18)
    attitude_rows = constrained.constraint_control[0, 0, 12 + 6 : 12 + 8]
    assert attitude_rows[:, 3:5].abs().sum() > 0.0
    torch.testing.assert_close(
        constrained.constraint_control[0, 1:, 12 + 6 : 12 + 8],
        torch.zeros_like(constrained.constraint_control[0, 1:, 12 + 6 : 12 + 8]),
    )


def test_recovery_reuses_inactive_stance_row_for_fk_sdf_direction() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.zeros(1, 3),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    contact[:, 1, 0] = False
    query_shape = (1, cfg.runtime.horizon_steps + 1, 4)
    gradient = torch.zeros(*query_shape, 2)
    gradient[..., 0] = 1.0
    ground_safe_distance = cfg.gait.foot_collision_radius + cfg.gait.small_collision_margin_xy
    midpoint = 0.5 * (ground_safe_distance + cfg.gait.small_touchdown_margin)
    foot_query = JointMpcTerrainQuery(
        height_w=torch.zeros(query_shape),
        small_distance_m=torch.full(query_shape, midpoint),
        large_distance_m=torch.ones(query_shape),
        small_gradient_w=gradient,
        large_gradient_w=torch.zeros_like(gradient),
        valid=torch.ones(query_shape, dtype=torch.bool),
    )
    swing_target = rollout.foot_pos_w.clone()
    swing_target[:, 1, 0, 1] += 0.05

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        recovery_state=torch.tensor([[True, False, False, False]]),
        foot_query=foot_query,
        swing_target_w=swing_target,
    )

    assert constrained.constraint_control.shape == (1, cfg.runtime.horizon_steps, 32, 18)
    assert constrained.constraint_residual[0, 0, 0] < 0.0
    assert constrained.constraint_control[0, 0, 0].abs().sum() > 0.0
    torch.testing.assert_close(
        constrained.constraint_control[0, 0, 2],
        torch.zeros_like(constrained.constraint_control[0, 0, 2]),
    )
    torch.testing.assert_close(constrained.constraint_residual[0, 0, 2], torch.tensor(0.0))
    torch.testing.assert_close(
        constrained.constraint_control[0, 0, 1],
        torch.zeros_like(constrained.constraint_control[0, 0, 1]),
    )
    torch.testing.assert_close(constrained.constraint_residual[0, 0, 1], torch.tensor(0.0))
    torch.testing.assert_close(
        constrained.constraint_control[0, 1:, :12],
        torch.zeros_like(constrained.constraint_control[0, 1:, :12]),
    )


def test_recovery_sdf_uses_independent_reach_fraction() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    assert cfg.solver.constraint_reach_fraction == 0.015
    assert cfg.solver.recovery_sdf_reach_fraction > cfg.solver.constraint_reach_fraction

    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout, control, rollout.state[..., 6:], state, torch.zeros(1, 3), cfg
    )
    contact = fixed_trot_schedule(
        1, cfg.runtime.horizon_steps, "cpu", half_cycle_steps=cfg.gait.half_cycle_steps
    )
    contact[:, 1, 0] = False
    query_shape = (1, cfg.runtime.horizon_steps + 1, 4)
    gradient = torch.zeros(*query_shape, 2)
    gradient[..., 0] = 1.0
    ground_safe_distance = cfg.gait.foot_collision_radius + cfg.gait.small_collision_margin_xy
    midpoint = 0.5 * (ground_safe_distance + cfg.gait.small_touchdown_margin)
    foot_query = JointMpcTerrainQuery(
        height_w=torch.zeros(query_shape),
        small_distance_m=torch.full(query_shape, midpoint),
        large_distance_m=torch.ones(query_shape),
        small_gradient_w=gradient,
        large_gradient_w=torch.zeros_like(gradient),
        valid=torch.ones(query_shape, dtype=torch.bool),
    )

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        recovery_state=torch.tensor([[True, False, False, False]]),
        foot_query=foot_query,
    )
    recovery_control = constrained.constraint_control[0, 0, 0, 6:]
    recovery_capacity = recovery_control.abs().sum() * cfg.solver.joint_direction_limit
    expected_capacity = recovery_control.abs().sum() * (
        cfg.solver.joint_direction_limit * cfg.solver.recovery_sdf_reach_fraction
    )
    assert recovery_capacity > 0.0
    assert constrained.constraint_residual[0, 0, 0].abs() <= expected_capacity + 1.0e-7
    assert constrained.constraint_residual[0, 0, 0].abs() > recovery_capacity * cfg.solver.constraint_reach_fraction
    torch.testing.assert_close(
        constrained.constraint_residual[0, 0, 2],
        torch.tensor(0.0),
    )


def test_recovery_sdf_constraint_targets_clearance_beyond_readiness_boundary() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    distance = torch.tensor(
        [[cfg.gait.small_touchdown_margin + 0.5 * cfg.gait.recovery_sdf_exit_buffer_m]]
    )

    clearance = planner._recovery_sdf_constraint_clearance(distance, cfg)

    assert clearance.item() < 0.0
    torch.testing.assert_close(
        clearance,
        distance
        - cfg.gait.small_touchdown_margin
        - cfg.gait.recovery_sdf_exit_buffer_m,
    )


def test_safe_recovery_combines_bounded_progress_with_full_grounding() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    state.root_pos_w[:, 2] = 0.42
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.zeros(1, 3),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    contact[:, 1, 0] = False
    query_shape = (1, cfg.runtime.horizon_steps + 1, 4)
    gradient = torch.zeros(*query_shape, 2)
    gradient[..., 0] = 1.0
    foot_query = JointMpcTerrainQuery(
        height_w=torch.zeros(query_shape),
        small_distance_m=torch.full(query_shape, 0.06),
        large_distance_m=torch.ones(query_shape),
        small_gradient_w=gradient,
        large_gradient_w=torch.zeros(*query_shape, 2),
        valid=torch.ones(query_shape, dtype=torch.bool),
    )
    swing_target = rollout.foot_pos_w.clone()
    swing_target[:, 1, 0, 0] += 0.05

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        recovery_state=torch.tensor([[True, False, False, False]]),
        foot_query=foot_query,
        swing_target_w=swing_target,
        startup_mask=torch.zeros(1, dtype=torch.bool),
        command_body=torch.tensor([[0.2, 0.0, 0.0]]),
    )

    torch.testing.assert_close(constrained.constraint_residual[0, 0, 0], torch.tensor(0.0))
    torch.testing.assert_close(constrained.constraint_residual[0, 0, 1], torch.tensor(-0.001))
    assert 0.0 < constrained.constraint_residual[0, 0, 2]
    assert constrained.constraint_residual[0, 0, 2] <= cfg.gait.stance_ground_recovery_step_m
    assert constrained.constraint_control[0, 0, 2].abs().sum() > 0.0
    recovery_capacity = (
        constrained.constraint_control[0, 0, 2, 6:].abs().sum()
        * cfg.solver.joint_direction_limit
    )
    assert constrained.constraint_residual[0, 0, 2] > 0.5 * recovery_capacity
    torch.testing.assert_close(
        constrained.constraint_state[0, 0, 2, :6],
        torch.zeros_like(constrained.constraint_state[0, 0, 2, :6]),
    )
    assert constrained.constraint_state[0, 0, 2, 6:9].abs().sum() > 0.0
    assert constrained.constraint_control[0, 0, 1].abs().sum() > 0.0


def test_colliding_swing_x1_reuses_inactive_stance_rows_for_fk_clearance() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.zeros(1, 3),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    contact[:, 1, 0] = False
    query_shape = (1, cfg.runtime.horizon_steps + 1, 4)
    foot_query = JointMpcTerrainQuery(
        height_w=torch.full(query_shape, 0.08),
        small_distance_m=torch.zeros(query_shape),
        large_distance_m=torch.ones(query_shape),
        small_gradient_w=torch.zeros(*query_shape, 2),
        large_gradient_w=torch.zeros(*query_shape, 2),
        valid=torch.ones(query_shape, dtype=torch.bool),
    )
    swing_target = rollout.foot_pos_w.clone()
    swing_target[:, 1, 0, 0] += 0.05

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
        foot_query=foot_query,
        swing_target_w=swing_target,
    )

    assert torch.all(constrained.constraint_control[0, 0, :3].abs().sum(dim=-1) > 0.0)
    assert constrained.constraint_residual[0, 0, 0].abs() > 0.0
    assert constrained.constraint_residual[0, 0, 2] < 0.0
    effective_height = 0.08 + cfg.gait.small_semantic_height * 0.5
    raw_clearance_residual = (
        rollout.foot_pos_w[0, 1, 0, 2]
        - effective_height
        - cfg.gait.foot_collision_radius
        - cfg.gait.small_collision_margin_z
    )
    full_reach = (
        constrained.constraint_control[0, 0, 2, 6:].abs().sum()
        * cfg.solver.joint_direction_limit
    )
    swing_reach = full_reach * cfg.solver.swing_clearance_reach_fraction
    expected = torch.clamp(
        raw_clearance_residual - cfg.gait.swing_clearance_kkt_buffer_m,
        min=-swing_reach,
        max=swing_reach,
    )
    torch.testing.assert_close(constrained.constraint_residual[0, 0, 2], expected)
    assert constrained.constraint_residual[0, 0, 2].abs() > (
        full_reach * cfg.solver.constraint_reach_fraction
    )
    assert constrained.constraint_residual[0, 0, 2].abs() < full_reach
    torch.testing.assert_close(
        constrained.constraint_state[0, 0, :3, :6],
        torch.zeros_like(constrained.constraint_state[0, 0, :3, :6]),
    )


def test_free_swing_x1_reuses_inactive_stance_xy_rows_for_target_motion() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls
    from extension.joint_mpc_rti.terrain.query import JointMpcTerrainQuery

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.zeros(1, 3),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    contact[:, 1, 0] = False
    query_shape = (1, cfg.runtime.horizon_steps + 1, 4)
    foot_query = JointMpcTerrainQuery(
        height_w=torch.zeros(query_shape),
        small_distance_m=torch.ones(query_shape),
        large_distance_m=torch.ones(query_shape),
        small_gradient_w=torch.zeros(*query_shape, 2),
        large_gradient_w=torch.zeros(*query_shape, 2),
        valid=torch.ones(query_shape, dtype=torch.bool),
    )
    swing_target = rollout.foot_pos_w.clone()
    swing_target[:, 1, 0, 0] += 0.05

    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        recovery_state=torch.zeros(1, 4, dtype=torch.bool),
        foot_query=foot_query,
        swing_target_w=swing_target,
        startup_mask=torch.ones(1, dtype=torch.bool),
        command_body=torch.tensor([[0.2, 0.0, 0.0]]),
    )

    assert constrained.constraint_residual[0, 0, 0].abs() > 0.0
    assert constrained.constraint_residual[0, 0, 0].abs() <= cfg.gait.startup_foot_lead_target_m
    assert constrained.constraint_control[0, 0, 0].abs().sum() > 0.0
    assert constrained.constraint_control[0, 0, 1].abs().sum() > 0.0
    torch.testing.assert_close(
        constrained.constraint_state[0, 0, :2, :6],
        torch.zeros_like(constrained.constraint_state[0, 0, :2, :6]),
    )


def test_preview_direction_activates_predicted_joint_bound_rows() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.zeros(1, 3),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
    )
    preview_delta_state = torch.zeros_like(rollout.state)
    preview_delta_state[:, 1:, 6] = 2.0
    preview_delta_control = torch.zeros_like(control)

    refined = planner._refine_predicted_joint_bound_constraints(
        constrained,
        rollout,
        preview_delta_state,
        cfg,
    )

    active_joint_rows = refined.constraint_control[..., -12:, :].abs().sum(dim=-1) > 0.0
    assert torch.all(active_joint_rows[..., 0])
    assert refined.constraint_control.shape == (1, cfg.runtime.horizon_steps, 32, 18)
    torch.testing.assert_close(
        refined.constraint_control[..., :20, :],
        constrained.constraint_control[..., :20, :],
    )
    torch.testing.assert_close(
        refined.constraint_residual[..., :20],
        constrained.constraint_residual[..., :20],
    )


def test_preview_joint_bounds_ignore_unreachable_raw_state_direction() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.zeros(1, 3),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
    )
    raw_preview_state = torch.zeros_like(rollout.state)
    raw_preview_state[:, 1:, 6] = 2.0

    refined = planner._refine_predicted_joint_bound_constraints(
        constrained,
        rollout,
        raw_preview_state,
        cfg,
        preview_delta_control=torch.zeros_like(control),
        base_control=control,
    )

    torch.testing.assert_close(
        refined.constraint_control[..., -12:, :],
        torch.zeros_like(refined.constraint_control[..., -12:, :]),
    )
    torch.testing.assert_close(
        refined.constraint_residual[..., -12:],
        torch.zeros_like(refined.constraint_residual[..., -12:]),
    )


def test_preview_joint_bound_rows_take_priority_over_same_leg_swing_equalities() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    state.joint_pos[:, 0] = 1.0472 - cfg.solver.joint_position_safety_margin_rad - 0.01
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.4, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    contact[:, :, 0] = False
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
    )
    preview_control = torch.zeros_like(control)
    preview_control[..., 6] = cfg.solver.joint_direction_limit

    refined = planner._refine_predicted_joint_bound_constraints(
        constrained,
        rollout,
        torch.zeros_like(rollout.state),
        cfg,
        preview_delta_control=preview_control,
        base_control=control,
        contact_state=contact,
    )

    joint_active = refined.constraint_control[..., 20, :].abs().sum(dim=-1) > 0.0
    assert torch.any(joint_active)
    swing_leg_rows = refined.constraint_control[..., :3, :].abs().sum(dim=-1)
    torch.testing.assert_close(
        swing_leg_rows[joint_active],
        torch.zeros_like(swing_leg_rows[joint_active]),
    )


def test_preview_refinement_does_not_add_hard_root_progress_floor() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
    )
    reserve_control = constrained.constraint_control[:, :, 17].clone()
    reserve_residual = constrained.constraint_residual[:, :, 17].clone()
    preview_delta_state = torch.zeros_like(rollout.state)
    preview_delta_control = torch.zeros_like(control)
    preview_delta_control[:, 0, 0] = -1.0

    refined = planner._refine_predicted_joint_bound_constraints(
        constrained,
        rollout,
        preview_delta_state,
        cfg,
        preview_delta_control=preview_delta_control,
        base_control=control,
        command_body=torch.tensor([[0.2, 0.0, 0.0]]),
    )

    torch.testing.assert_close(refined.constraint_control[:, :, 17], reserve_control)
    torch.testing.assert_close(refined.constraint_residual[:, :, 17], reserve_residual)


def test_startup_leak_row_is_not_overwritten_by_preview_progress_floor() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    problem = planner._build_lq_problem(
        rollout,
        control,
        rollout.state[..., 6:],
        state,
        torch.tensor([[0.2, 0.0, 0.0]]),
        cfg,
    )
    contact = fixed_trot_schedule(
        1,
        cfg.runtime.horizon_steps,
        "cpu",
        half_cycle_steps=cfg.gait.half_cycle_steps,
    )
    constrained = planner._add_stance_control_constraints(
        problem,
        rollout,
        contact,
        rollout.foot_pos_w,
        cfg,
        startup_mask=torch.ones(1, dtype=torch.bool),
        command_body=torch.tensor([[0.2, 0.0, 0.0]]),
    )
    startup_control = constrained.constraint_control[:, :, 17].clone()
    startup_residual = constrained.constraint_residual[:, :, 17].clone()
    preview_delta_control = torch.zeros_like(control)
    preview_delta_control[:, 0, 0] = -1.0

    refined = planner._refine_predicted_joint_bound_constraints(
        constrained,
        rollout,
        torch.zeros_like(rollout.state),
        cfg,
        preview_delta_control=preview_delta_control,
        base_control=control,
        command_body=torch.tensor([[0.2, 0.0, 0.0]]),
        startup_mask=torch.ones(1, dtype=torch.bool),
    )

    torch.testing.assert_close(refined.constraint_control[:, :, 17], startup_control)
    torch.testing.assert_close(refined.constraint_residual[:, :, 17], startup_residual)


def test_root_attitude_violation_components_cover_state_and_rate() -> None:
    from extension.joint_mpc_rti.planner import _root_attitude_violation_components
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    cfg = JointMpcRtiCfg()
    state = make_state(1)
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    rollout = rollout_controls(state, control, dt=cfg.runtime.dt)
    rollout.state[:, 1, 3] = cfg.solver.root_roll_pitch_limit_rad + 0.1
    rollout.control[:, 0, 4] = cfg.solver.root_roll_pitch_rate_limit_rps + 0.2

    violation = _root_attitude_violation_components(rollout, cfg)

    torch.testing.assert_close(violation, torch.tensor([[0.1, 0.2]]))


def test_control_direction_scaling_preserves_direction_and_limits() -> None:
    from extension.joint_mpc_rti.planner import _scale_control_direction_to_limits
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    limits = torch.tensor(
        [cfg.solver.root_linear_direction_limit] * 3
        + [cfg.solver.root_angular_direction_limit] * 3
        + [cfg.solver.joint_direction_limit] * 12
    )
    direction = (0.5 * limits).view(1, 1, 18).expand(1, 3, -1).clone()
    direction[0, 0, 0] = 2.0 * cfg.solver.root_linear_direction_limit
    direction[0, 1, 6] = 4.0 * cfg.solver.joint_direction_limit

    scaled = _scale_control_direction_to_limits(
        direction,
        root_linear_limit=cfg.solver.root_linear_direction_limit,
        root_angular_limit=cfg.solver.root_angular_direction_limit,
        joint_limit=cfg.solver.joint_direction_limit,
    )

    torch.testing.assert_close(scaled[:, 0], 0.5 * direction[:, 0])
    torch.testing.assert_close(scaled[:, 1], 0.25 * direction[:, 1])
    torch.testing.assert_close(scaled[:, 2], direction[:, 2])
    assert torch.all(scaled[..., :3].abs() <= cfg.solver.root_linear_direction_limit)
    assert torch.all(scaled[..., 3:6].abs() <= cfg.solver.root_angular_direction_limit)
    assert torch.all(scaled[..., 6:].abs() <= cfg.solver.joint_direction_limit)


def test_default_joint_direction_trust_region_caps_candidate_velocity_at_ten_rps() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()

    assert cfg.gait.max_nominal_joint_velocity + cfg.solver.joint_direction_limit <= 10.0
    assert planner._joint_candidate_absolute_limit(cfg) == pytest.approx(10.0)


def test_joint_candidate_base_control_preserves_one_rps_correction_reserve() -> None:
    from extension.joint_mpc_rti import planner
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    control = torch.zeros(1, cfg.runtime.horizon_steps, 18)
    control[..., 6:] = torch.linspace(-12.0, 12.0, 12)

    reserved = planner._reserve_joint_candidate_direction_capacity(control, cfg)

    assert reserved[..., 6:].abs().max() == pytest.approx(
        cfg.gait.max_nominal_joint_velocity
    )
    assert (
        reserved[..., 6:].abs() + cfg.solver.joint_direction_limit
        <= planner._joint_candidate_absolute_limit(cfg)
    ).all()


def test_constrained_direction_scaling_preserves_affine_equality() -> None:
    from extension.joint_mpc_rti.planner import _scale_constrained_control_direction

    delta_control = torch.tensor([[[2.0, -1.0]]])
    delta_state = torch.zeros(1, 2, 2)
    constraint_control = torch.tensor([[[[1.0, 1.0]]]])
    constraint_state = torch.zeros(1, 1, 1, 2)
    constraint_residual = torch.tensor([[[-1.0]]])

    scaled = _scale_constrained_control_direction(
        delta_control,
        delta_state,
        constraint_control,
        constraint_state,
        constraint_residual,
        limits=torch.ones(2),
    )

    torch.testing.assert_close(scaled, torch.tensor([[[1.0, 0.0]]]), atol=2.0e-5, rtol=0.0)
    residual = (
        torch.matmul(constraint_control, scaled.unsqueeze(-1)).squeeze(-1)
        + constraint_residual
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2.0e-5, rtol=0.0)


def test_constrained_direction_scaling_redistributes_saturated_required_control() -> None:
    from extension.joint_mpc_rti.planner import _scale_constrained_control_direction

    constraint_control = torch.tensor([[[[1.0, 1.0]]]])
    constraint_residual = torch.tensor([[[-2.0]]])

    scaled = _scale_constrained_control_direction(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2, 2),
        constraint_control,
        torch.zeros(1, 1, 1, 2),
        constraint_residual,
        limits=torch.tensor([0.5, 2.0]),
    )

    assert scaled[0, 0, 0].abs() <= 0.5
    assert scaled[0, 0, 1].abs() <= 2.0
    residual = (
        torch.matmul(constraint_control, scaled.unsqueeze(-1)).squeeze(-1)
        + constraint_residual
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2.0e-5, rtol=0.0)


def test_constrained_direction_scaling_converges_for_coupled_feasible_box() -> None:
    from extension.joint_mpc_rti.planner import _scale_constrained_control_direction

    torch.manual_seed(373)
    constraint_control = torch.randn(6, 8)
    constraint_control[:, 0] *= 3.0
    limits = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    feasible = (torch.rand(8) * 2.0 - 1.0) * limits * 0.7
    constraint_residual = -(constraint_control @ feasible)

    scaled = _scale_constrained_control_direction(
        torch.zeros(1, 1, 8),
        torch.zeros(1, 2, 8),
        constraint_control.view(1, 1, 6, 8),
        torch.zeros(1, 1, 6, 8),
        constraint_residual.view(1, 1, 6),
        limits=limits,
    )

    assert torch.all(scaled.abs() <= limits.view(1, 1, -1) + 1.0e-6)
    residual = constraint_control @ scaled[0, 0] + constraint_residual
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2.0e-5, rtol=0.0)


def test_constrained_direction_scaling_propagates_recovered_state_between_nodes() -> None:
    from extension.joint_mpc_rti.planner import _scale_constrained_control_direction

    constraint_control = torch.ones(1, 2, 1, 1)
    constraint_state = torch.tensor([[[[0.0]], [[1.0]]]])
    constraint_residual = torch.tensor([[[-1.0], [0.0]]])
    matrix_a = torch.ones(1, 2, 1, 1)
    matrix_b = torch.ones(1, 2, 1, 1)

    scaled = _scale_constrained_control_direction(
        torch.zeros(1, 2, 1),
        torch.zeros(1, 3, 1),
        constraint_control,
        constraint_state,
        constraint_residual,
        limits=torch.full((1,), 2.0),
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        affine_dynamics=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1),
    )

    torch.testing.assert_close(scaled, torch.tensor([[[1.0], [-1.0]]]), atol=2.0e-5, rtol=0.0)
    state_x1 = scaled[:, 0]
    residual_x1 = constraint_control[:, 1, 0] * scaled[:, 1]
    residual_x1 = residual_x1 + constraint_state[:, 1, 0] * state_x1
    torch.testing.assert_close(
        residual_x1 + constraint_residual[:, 1],
        torch.zeros_like(constraint_residual[:, 1]),
        atol=2.0e-5,
        rtol=0.0,
    )


def test_constrained_direction_scaling_bounds_unreachable_required_control() -> None:
    from extension.joint_mpc_rti.planner import _scale_constrained_control_direction

    scaled = _scale_constrained_control_direction(
        torch.tensor([[[0.0, 0.5]]]),
        torch.zeros(1, 2, 2),
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.zeros(1, 1, 1, 2),
        torch.tensor([[[-100.0]]]),
        limits=torch.ones(2),
    )

    assert scaled.abs().max() <= 1.0
    torch.testing.assert_close(scaled[0, 0, 1], torch.tensor(0.5))


def test_constrained_direction_scaling_preserves_unreachable_required_ratio() -> None:
    from extension.joint_mpc_rti.planner import _scale_constrained_control_direction

    scaled = _scale_constrained_control_direction(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2, 2),
        torch.eye(2).reshape(1, 1, 2, 2),
        torch.zeros(1, 1, 2, 2),
        torch.tensor([[[-2.0, -1.0]]]),
        limits=torch.ones(2),
    )

    torch.testing.assert_close(scaled, torch.tensor([[[1.0, 0.5]]]))


def test_constrained_direction_scaling_uses_separate_required_capacity() -> None:
    from extension.joint_mpc_rti.planner import _scale_constrained_control_direction

    scaled = _scale_constrained_control_direction(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2, 2),
        torch.eye(2).reshape(1, 1, 2, 2),
        torch.zeros(1, 1, 2, 2),
        torch.tensor([[[-2.0, -1.0]]]),
        limits=torch.ones(2),
        required_limits=torch.full((1, 1, 2), 3.0),
    )

    torch.testing.assert_close(scaled, torch.tensor([[[2.0, 1.0]]]))


def test_constrained_required_capacity_respects_absolute_candidate_control_bounds() -> None:
    from extension.joint_mpc_rti.planner import _scale_constrained_control_direction

    base = torch.tensor([[[-9.0]], [[9.0]]])
    scaled = _scale_constrained_control_direction(
        torch.zeros(2, 1, 1),
        torch.zeros(2, 2, 1),
        torch.ones(2, 1, 1, 1),
        torch.zeros(2, 1, 1, 1),
        torch.full((2, 1, 1), -18.0),
        limits=torch.ones(1),
        base_control=base,
        required_absolute_limits=torch.full((1,), 10.0),
    )

    torch.testing.assert_close(scaled[:, 0, 0], torch.tensor([18.0, 1.0]))
    assert torch.all((base + scaled).abs() <= 10.0)


def test_recovery_ground_error_is_limited_by_joint_control_reach() -> None:
    from extension.joint_mpc_rti.planner import _clamp_recovery_ground_error_to_control_reach

    error = torch.tensor([[[0.02, -0.02]]])
    control_jacobian = torch.zeros(1, 1, 2, 18)
    control_jacobian[0, 0, 0, 6] = 5.0e-4

    limited = _clamp_recovery_ground_error_to_control_reach(
        error,
        control_jacobian,
        joint_direction_limit=10.0,
    )

    torch.testing.assert_close(limited, torch.tensor([[[0.005, 0.0]]]))


def test_constraint_reach_fraction_keeps_stance_correction_conservative() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    assert JointMpcRtiCfg().solver.constraint_reach_fraction == 0.015
