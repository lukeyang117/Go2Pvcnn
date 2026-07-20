from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import total_trajectory_loss, weighted_trajectory_residual
from extension.joint_mpc_rti.solver.linearization import linearize_trajectory
from .test_trajectory_losses import _context, _state


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
