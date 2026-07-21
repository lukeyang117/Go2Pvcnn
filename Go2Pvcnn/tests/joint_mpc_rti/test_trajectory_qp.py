from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import total_trajectory_loss, weighted_trajectory_residual
from extension.joint_mpc_rti.model.nominal import build_nominal
from extension.joint_mpc_rti.planner import _stance_anchors_from_state, build_loss_context
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.solver.linearization import linearize_trajectory
from extension.joint_mpc_rti.solver.trajectory_scan import solve_trajectory_qp_scan
from extension.joint_mpc_rti.solver.trajectory_qp import (
    ActiveConstraints,
    refine_active_set,
    select_active_constraints,
    solve_dense_active_kkt,
)
from extension.joint_mpc_rti.types import JointMpcRtiSolverState
from .helpers import make_state
from .test_trajectory_losses import _context, _state
from .test_trajectory_losses import _flat_field


def test_stance_anchor_uses_current_fk_then_future_touchdown_reference() -> None:
    state = _state(batch=1)
    state[0, :, 0] = torch.arange(31) * 0.01
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))
    touchdown = torch.full((1, 31, 4, 3), 7.0)

    anchor = _stance_anchors_from_state(state, touchdown, schedule)

    assert not torch.equal(anchor[:, 0, 1], touchdown[:, 0, 1])
    torch.testing.assert_close(anchor[:, :12, 1], anchor[:, :1, 1].expand(-1, 12, -1))
    torch.testing.assert_close(anchor[:, 12:24, 0], touchdown[:, 12:13, 0].expand(-1, 12, -1))
    torch.testing.assert_close(anchor[:, 24:, 1], touchdown[:, 24:25, 1].expand(-1, 7, -1))


def test_direct_z_qp_matches_autograd_gradient_and_gauss_newton_hessian() -> None:
    cfg = JointMpcRtiCfg()
    cfg.solver.regularization = 1.0e-6
    state = _state(batch=1).to(torch.float64).requires_grad_(True)
    context = _context(state.detach())

    qp = linearize_trajectory(state, context, cfg)
    dense_h, dense_g = qp.to_dense()
    auto_g = torch.autograd.grad(total_trajectory_loss(state, context, cfg).sum(), state)[0]
    jacobian = torch.func.jacrev(
        lambda value: weighted_trajectory_residual(value, context, cfg)[0]
    )(state).squeeze(1)
    jacobian = jacobian.reshape(jacobian.shape[0], -1)
    expected_h = jacobian.transpose(0, 1) @ jacobian
    expected_h = expected_h + cfg.solver.regularization * torch.eye(expected_h.shape[0], dtype=expected_h.dtype)

    torch.testing.assert_close(dense_g, auto_g.flatten(1), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(dense_h[0], expected_h, atol=5e-4, rtol=5e-4)


def test_qp_fixes_delta_z0_and_builds_joint_position_velocity_trust_bounds() -> None:
    cfg = JointMpcRtiCfg()
    state = _state(batch=2)

    qp = linearize_trajectory(state, _context(state), cfg)

    assert torch.count_nonzero(qp.lower[:, 0]) == 0
    assert torch.count_nonzero(qp.upper[:, 0]) == 0
    assert qp.lower.shape == (2, 31, 18)
    assert qp.upper.shape == (2, 31, 18)
    assert qp.joint_difference_lower.shape == (2, 30, 12)
    assert qp.joint_difference_upper.shape == (2, 30, 12)
    assert torch.all(qp.lower <= qp.upper)


def test_qp_contains_only_two_temporal_offdiagonal_bands() -> None:
    state = _state(batch=1)
    qp = linearize_trajectory(state, _context(state), JointMpcRtiCfg())

    assert qp.diagonal.shape == (1, 31, 18, 18)
    assert qp.first_offdiag.shape == (1, 30, 18, 18)
    assert qp.second_offdiag.shape == (1, 29, 18, 18)


def test_active_set_selects_merged_box_and_joint_velocity_boundaries() -> None:
    state = _state(batch=2)
    qp = linearize_trajectory(state, _context(state), JointMpcRtiCfg())
    direction = torch.zeros_like(state)
    direction[:, 5, 6] = qp.upper[:, 5, 6] + 0.1
    direction[:, 10, 7] = qp.lower[:, 10, 7] - 0.1
    direction[:, 15, 8] = 0.5
    direction[:, 14, 8] = -0.5

    active = select_active_constraints(qp, direction)

    assert active.box_mask.shape == (2, 31, 18)
    assert active.velocity_mask.shape == (2, 30, 12)
    assert active.box_mask[:, 5, 6].all()
    assert active.box_mask[:, 10, 7].all()
    assert active.velocity_mask[:, 14, 2].all()
    assert active.max_rows_per_interval <= 30


def test_two_unrolled_refinements_match_dense_active_kkt_and_hold_constraints() -> None:
    cfg = JointMpcRtiCfg()
    state = _state(batch=1)
    qp = linearize_trajectory(state, _context(state), cfg)

    dense = refine_active_set(qp, solve_dense_active_kkt, refinements=2)
    scan = solve_trajectory_qp_scan(qp)
    difference = scan.direction[:, 1:, 6:] - scan.direction[:, :-1, 6:]

    torch.testing.assert_close(scan.direction, dense.direction, atol=2e-5, rtol=2e-5)
    assert torch.all(scan.direction >= qp.lower - 2e-5)
    assert torch.all(scan.direction <= qp.upper + 2e-5)
    assert torch.all(difference >= qp.joint_difference_lower - 2e-5)
    assert torch.all(difference <= qp.joint_difference_upper + 2e-5)


def test_joint_kkt_compile_budget_rejects_more_than_32_local_rows() -> None:
    import pytest

    with pytest.raises(ValueError, match="constraint rows"):
        ActiveConstraints.validate_compile_budget(constraint_rows=33)


def test_refined_cold_command_direction_does_not_increase_convex_qp_model() -> None:
    cfg = JointMpcRtiCfg()
    cfg.nominal.command_scale = 0.45
    cfg.solver.root_position_trust = 0.01
    measured = make_state(1)
    phase = torch.zeros(1, dtype=torch.long)
    previous = JointMpcRtiSolverState(
        trajectory=measured.as_vector()[:, None].expand(-1, 31, -1).clone(),
        gait_phase=phase,
        valid=torch.zeros(1, dtype=torch.bool),
    )
    command = torch.tensor(((1.0, 0.5, 1.0),))
    field = _flat_field(1)
    nominal = build_nominal(measured, command, field, phase, previous=previous, cfg=cfg)
    context = build_loss_context(nominal, command, field, phase, cfg)
    qp = linearize_trajectory(nominal.state, context, cfg)

    direction = solve_trajectory_qp_scan(qp).direction
    difference = direction[:, 1:, 6:] - direction[:, :-1, 6:]
    box_violation = torch.maximum(
        (qp.lower - direction).amax(), (direction - qp.upper).amax()
    )
    velocity_violation = torch.maximum(
        (qp.joint_difference_lower - difference).amax(),
        (difference - qp.joint_difference_upper).amax(),
    )
    assert direction.abs().amax().item() > 0.0
    hessian, gradient = qp.to_dense()
    flat = direction.flatten(1)
    model_change = (gradient * flat).sum(dim=1) + 0.5 * torch.einsum(
        "bi,bij,bj->b", flat, hessian, flat
    )

    assert box_violation.item() <= 2.0e-5, (box_violation.item(), model_change.item())
    assert velocity_violation.item() <= 2.0e-5, (
        velocity_violation.item(),
        model_change.item(),
    )
    assert model_change.item() <= 1.0e-6
