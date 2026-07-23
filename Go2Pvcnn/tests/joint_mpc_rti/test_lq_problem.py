from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.model.nominal import NominalTrajectory
from extension.joint_mpc_rti.solver.context import LossContext
from extension.joint_mpc_rti.solver.lq_problem import build_lq_problem, lq_residuals
from .helpers import make_flat_field, make_state_nodes


EXPECTED_FAMILIES = {
    "velocity",
    "posture",
    "root",
    "swing",
    "touchdown",
    "smooth",
    "warm",
    "slack",
}


def _nominal_and_context(
    *, batch: int = 1, dtype: torch.dtype = torch.float64
) -> tuple[NominalTrajectory, LossContext]:
    state = make_state_nodes(batch, dtype=dtype)
    state[:, 1:, 0] += torch.linspace(0.0, 0.12, 30, dtype=dtype)
    state[:, 1:, 5] += torch.linspace(0.0, 0.03, 30, dtype=dtype)
    nodes = int(state.shape[1])
    flat = state.reshape(batch * nodes, 18)
    foot = go2_fk(flat[:, :3], flat[:, 3:6], flat[:, 6:]).foot_pos_w.reshape(
        batch, nodes, 4, 3
    )
    schedule = fixed_trot_schedule(torch.arange(batch, dtype=torch.long) % 24)
    touchdown = foot.detach().clone()
    touchdown[..., 0] += 0.015
    rebased = state.detach().clone()
    rebased[:, 1:, 1] -= 0.004
    nominal = NominalTrajectory(
        state=state,
        foot_reference_w=foot.detach().clone(),
        touchdown_reference_w=touchdown,
        contact_state=schedule.stance,
        used_cold_start=torch.ones(batch, dtype=torch.bool),
        used_warm_start=torch.zeros(batch, dtype=torch.bool),
        valid=torch.ones(batch, dtype=torch.bool),
        current_stance_anchor_w=foot[:, 0].detach().clone(),
        rebased_state=rebased,
    )
    context = LossContext(
        command_body=state.new_tensor((0.2, 0.0, 0.1)).expand(batch, -1),
        touchdown_reference_w=touchdown,
        schedule=schedule,
        terrain=make_flat_field(batch),
        stance_anchor_w=foot.detach().clone(),
        support_height=state.new_zeros(batch, nodes),
    )
    return nominal, context


def test_lq_problem_has_exact_residual_families_and_h30_bands() -> None:
    nominal, context = _nominal_and_context(batch=2)

    problem = build_lq_problem(nominal, context, JointMpcRtiCfg())

    assert set(problem.residuals) == EXPECTED_FAMILIES
    assert set(problem.cost_breakdown) == EXPECTED_FAMILIES
    assert problem.diagonal.shape == (2, 31, 18, 18)
    assert problem.first_offdiag.shape == (2, 30, 18, 18)
    assert problem.second_offdiag.shape == (2, 29, 18, 18)
    assert problem.gradient.shape == (2, 31, 18)
    assert problem.residuals["slack"].shape == (2, 0)


def test_lq_gradient_matches_finite_difference() -> None:
    cfg = JointMpcRtiCfg()
    cfg.solver.regularization = 0.0
    nominal, context = _nominal_and_context()
    problem = build_lq_problem(nominal, context, cfg)
    state = nominal.state.detach()
    epsilon = 1.0e-6
    probes = ((0, 3), (1, 0), (7, 5), (12, 8), (20, 14), (30, 17))

    errors = []
    for node, coordinate in probes:
        plus = state.clone()
        minus = state.clone()
        plus[:, node, coordinate] += epsilon
        minus[:, node, coordinate] -= epsilon
        plus_cost = sum(
            0.5 * value.square().sum(dim=1)
            for value in lq_residuals(plus, nominal, context, cfg).values()
        )
        minus_cost = sum(
            0.5 * value.square().sum(dim=1)
            for value in lq_residuals(minus, nominal, context, cfg).values()
        )
        finite_difference = (plus_cost - minus_cost) / (2.0 * epsilon)
        errors.append(
            (finite_difference - problem.gradient[:, node, coordinate]).abs()
        )

    assert torch.stack(errors).max() < 2.0e-3


def test_low_speed_posture_gate_and_warm_reference_are_explicit() -> None:
    cfg = JointMpcRtiCfg()
    nominal, context = _nominal_and_context()
    moved = nominal.state.detach().clone()
    moved[:, :, 6:] += 0.1
    stopped = LossContext(
        command_body=torch.zeros_like(context.command_body),
        touchdown_reference_w=context.touchdown_reference_w,
        schedule=context.schedule,
        terrain=context.terrain,
        stance_anchor_w=context.stance_anchor_w,
        support_height=context.support_height,
    )
    fast = LossContext(
        command_body=context.command_body.new_tensor((1.0, 0.0, 0.0)).expand(1, -1),
        touchdown_reference_w=context.touchdown_reference_w,
        schedule=context.schedule,
        terrain=context.terrain,
        stance_anchor_w=context.stance_anchor_w,
        support_height=context.support_height,
    )
    rough_height = context.support_height.clone()
    rough_height[:, 1::2] = 0.08
    rough = LossContext(
        command_body=torch.zeros_like(context.command_body),
        touchdown_reference_w=context.touchdown_reference_w,
        schedule=context.schedule,
        terrain=context.terrain,
        stance_anchor_w=context.stance_anchor_w,
        support_height=rough_height,
    )

    stopped_residuals = lq_residuals(moved, nominal, stopped, cfg)
    fast_residuals = lq_residuals(moved, nominal, fast, cfg)
    rough_residuals = lq_residuals(moved, nominal, rough, cfg)

    assert stopped_residuals["posture"].square().sum() > 100.0 * fast_residuals[
        "posture"
    ].square().sum()
    assert stopped_residuals["posture"].square().sum() > 100.0 * rough_residuals[
        "posture"
    ].square().sum()
    expected_warm = cfg.lq_cost.warm * (
        moved - nominal.rebased_state
    ).square().sum()
    torch.testing.assert_close(
        stopped_residuals["warm"].square().sum(), expected_warm
    )


def test_lq_bands_equal_dense_autograd_gauss_newton_on_active_bands() -> None:
    cfg = JointMpcRtiCfg()
    cfg.solver.regularization = 0.0
    nominal, context = _nominal_and_context()
    problem = build_lq_problem(nominal, context, cfg)
    state = nominal.state.detach().requires_grad_(True)

    jacobian = torch.autograd.functional.jacobian(
        lambda value: torch.cat(
            tuple(lq_residuals(value, nominal, context, cfg).values()), dim=1
        ),
        state,
        vectorize=True,
    )[0, :, 0]
    dense = jacobian.reshape(jacobian.shape[0], -1)
    hessian = dense.transpose(0, 1) @ dense
    blocks = hessian.reshape(31, 18, 31, 18).permute(0, 2, 1, 3)
    node = torch.arange(31)

    torch.testing.assert_close(problem.diagonal[0], blocks[node, node], atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(
        problem.first_offdiag[0], blocks[node[:-1], node[1:]], atol=2e-7, rtol=2e-7
    )
    torch.testing.assert_close(
        problem.second_offdiag[0], blocks[node[:-2], node[2:]], atol=2e-7, rtol=2e-7
    )
