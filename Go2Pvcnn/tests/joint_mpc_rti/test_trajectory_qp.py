from __future__ import annotations

import torch
from dataclasses import replace

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import total_trajectory_loss, weighted_trajectory_residual
from extension.joint_mpc_rti.model.nominal import build_nominal
from extension.joint_mpc_rti.model.nominal import NominalTrajectory
from extension.joint_mpc_rti.planner import (
    _foot_positions,
    _stance_anchors_from_state,
    build_loss_context,
)
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.solver.linearization import linearize_trajectory
from extension.joint_mpc_rti.solver import linearization as linearization_module
from extension.joint_mpc_rti.solver.trajectory_scan import solve_trajectory_qp_scan
from extension.joint_mpc_rti.solver.trajectory_qp import (
    ActiveConstraints,
    TrajectoryQp,
    refine_active_set,
    select_active_constraints,
    solve_dense_active_kkt,
    trajectory_bounds,
)
from extension.joint_mpc_rti.types import JointMpcRtiSolverState
from .helpers import make_state
from .test_trajectory_losses import _context, _state
from .test_trajectory_losses import _flat_field


def test_stance_anchor_context_repeats_only_the_persistent_anchor() -> None:
    state = _state(batch=1)
    state[0, :, 0] = torch.arange(31) * 0.01
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))
    touchdown = torch.full((1, 31, 4, 3), 7.0)

    current_anchor = _foot_positions(state)[:, 0]
    anchor = _stance_anchors_from_state(state, touchdown, current_anchor, schedule)

    torch.testing.assert_close(
        anchor,
        current_anchor[:, None].expand(-1, state.shape[1], -1, -1),
    )
    assert not torch.any(anchor == 7.0)


def test_loss_context_grounds_persistent_anchor_z_at_contact_surface() -> None:
    state = _state(batch=1)
    foot = _foot_positions(state)
    anchor = foot[:, 0].clone()
    anchor[..., 2] = -0.1
    nominal = NominalTrajectory(
        state=state,
        foot_reference_w=foot,
        touchdown_reference_w=foot,
        contact_state=fixed_trot_schedule(torch.tensor((0,))).stance,
        used_cold_start=torch.ones(1, dtype=torch.bool),
        used_warm_start=torch.zeros(1, dtype=torch.bool),
        valid=torch.ones(1, dtype=torch.bool),
        current_stance_anchor_w=anchor,
    )
    cfg = JointMpcRtiCfg()

    context = build_loss_context(
        nominal, torch.zeros(1, 3), _flat_field(1), torch.tensor((0,)), cfg
    )

    torch.testing.assert_close(
        context.stance_anchor_w[..., 2],
        torch.full_like(context.stance_anchor_w[..., 2], cfg.gait.foot_contact_offset),
    )


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


def test_root_position_trust_fixes_published_xy_and_accumulates_after_x1() -> None:
    cfg = JointMpcRtiCfg()
    cfg.solver.root_position_trust = 0.01
    cfg.solver.root_roll_pitch_trust = 0.08
    cfg.solver.root_yaw_trust = 0.02
    nominal = _state(batch=2)

    lower, upper, _, _ = trajectory_bounds(nominal, cfg)

    torch.testing.assert_close(lower[:, 0], torch.zeros_like(lower[:, 0]))
    torch.testing.assert_close(upper[:, 0], torch.zeros_like(upper[:, 0]))
    torch.testing.assert_close(
        upper[:, 1, :2],
        torch.zeros_like(upper[:, 1, :2]),
    )
    torch.testing.assert_close(
        lower[:, 1, :2],
        torch.zeros_like(lower[:, 1, :2]),
    )
    torch.testing.assert_close(
        upper[:, 1, 2],
        torch.full_like(upper[:, 1, 2], cfg.solver.root_position_trust),
    )
    torch.testing.assert_close(
        lower[:, 1, 2],
        torch.full_like(lower[:, 1, 2], -cfg.solver.root_position_trust),
    )
    torch.testing.assert_close(
        upper[:, 2, :3],
        torch.full_like(upper[:, 2, :3], 2.0 * cfg.solver.root_position_trust),
    )
    torch.testing.assert_close(
        upper[:, 30, :3],
        torch.full_like(upper[:, 30, :3], 30.0 * cfg.solver.root_position_trust),
    )
    torch.testing.assert_close(
        upper[:, 1:, 3:5],
        torch.full_like(upper[:, 1:, 3:5], cfg.solver.root_roll_pitch_trust),
    )
    torch.testing.assert_close(
        upper[:, 1:, 5],
        torch.full_like(upper[:, 1:, 5], cfg.solver.root_yaw_trust),
    )
    torch.testing.assert_close(
        upper[:, 1:, 6:],
        torch.full_like(upper[:, 1:, 6:], cfg.solver.joint_trust),
    )


def test_qp_contains_only_two_temporal_offdiagonal_bands() -> None:
    state = _state(batch=1)
    qp = linearize_trajectory(state, _context(state), JointMpcRtiCfg())

    assert qp.diagonal.shape == (1, 31, 18, 18)
    assert qp.first_offdiag.shape == (1, 30, 18, 18)
    assert qp.second_offdiag.shape == (1, 29, 18, 18)


def test_published_kinematic_jacobian_has_stance_xy_and_swing_z_rows() -> None:
    state = _state(batch=2).to(torch.float64)
    schedule = fixed_trot_schedule(torch.tensor((0, 7), dtype=torch.long))
    build = getattr(linearization_module, "published_kinematic_jacobian", None)

    assert callable(build)
    support = build(state, schedule)
    assert support.shape == (2, 6, 18)

    delta = torch.tensor(
        (
            (0.2, -0.1, 0.05, 0.03, -0.02, 0.04) + (0.01,) * 12,
            (-0.1, 0.2, -0.04, -0.02, 0.03, -0.01) + (-0.015,) * 12,
        ),
        dtype=state.dtype,
    )
    epsilon = 1.0e-6
    before = go2_fk(state[:, 1, :3], state[:, 1, 3:6], state[:, 1, 6:]).foot_pos_w
    moved = state[:, 1] + epsilon * delta
    after = go2_fk(moved[:, :3], moved[:, 3:6], moved[:, 6:]).foot_pos_w
    stance_index = torch.topk(schedule.stance[:, 1].to(torch.int64), k=2, dim=1).indices
    stance_actual = torch.gather(
        (after - before) / epsilon,
        1,
        stance_index[..., None].expand(-1, -1, 3),
    )[..., :2].reshape(2, 4)
    swing_index = torch.topk(schedule.swing[:, 1].to(torch.int64), k=2, dim=1).indices
    swing_actual = torch.gather(
        (after - before) / epsilon,
        1,
        swing_index[..., None].expand(-1, -1, 3),
    )[..., 2]
    actual = torch.cat((stance_actual, swing_actual), dim=1)
    predicted = torch.einsum("bri,bi->br", support, delta)

    torch.testing.assert_close(predicted, actual, atol=2.0e-5, rtol=2.0e-5)


def test_qp_support_target_corrects_nominal_x1_to_persistent_stance_anchor() -> None:
    state = _state(batch=2).to(torch.float64)
    context = _context(state)
    stance_index = torch.topk(
        context.schedule.stance[:, 1].to(torch.int64), k=2, dim=1
    ).indices
    offset = state.new_tensor((0.0004, -0.0003, 0.0002))
    anchors = context.stance_anchor_w.clone()
    selected_anchor = torch.gather(
        anchors[:, 1], 1, stance_index[..., None].expand(-1, -1, 3)
    )
    selected_anchor = selected_anchor + offset
    anchors[:, 1].scatter_(
        1, stance_index[..., None].expand(-1, -1, 3), selected_anchor
    )

    qp = linearize_trajectory(state, replace(context, stance_anchor_w=anchors), JointMpcRtiCfg())

    expected = offset[:2].expand(2, 2, 2).reshape(2, 4)
    torch.testing.assert_close(qp.support_target[:, :4], expected)


def test_warm_manifold_initialization_zeros_published_continuing_stance_target() -> None:
    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    phase = torch.tensor((12,), dtype=torch.long)
    previous_trajectory = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    previous_trajectory[:, 2:, 0] += 0.006
    persistent_anchor = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w
    previous = JointMpcRtiSolverState(
        trajectory=previous_trajectory,
        gait_phase=phase,
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=persistent_anchor,
    )
    field = _flat_field(1)
    nominal = build_nominal(
        measured,
        torch.zeros(1, 3),
        field,
        phase,
        previous=previous,
        cfg=cfg,
    )
    context = build_loss_context(nominal, torch.zeros(1, 3), field, phase, cfg)

    qp = linearize_trajectory(nominal.state, context, cfg)

    torch.testing.assert_close(
        qp.support_target[:, :4],
        torch.zeros(1, 4),
        atol=2.0e-5,
        rtol=0.0,
    )


def test_qp_support_target_is_zero_for_new_x1_stance() -> None:
    state = _state(batch=1).to(torch.float64)
    context = _context(state)
    schedule = fixed_trot_schedule(torch.tensor((11,), dtype=torch.long))
    foot = go2_fk(state[:, 1, :3], state[:, 1, 3:6], state[:, 1, 6:]).foot_pos_w
    touchdown = context.touchdown_reference_w.clone()
    touchdown[:, 1] = foot + state.new_tensor((0.0002, -0.0001, 0.0003))
    stale_anchor = context.stance_anchor_w.clone()
    stale_anchor[:, 1] = foot + 0.1
    onset_context = replace(
        context,
        schedule=schedule,
        touchdown_reference_w=touchdown,
        stance_anchor_w=stale_anchor,
    )

    qp = linearize_trajectory(state, onset_context, JointMpcRtiCfg())

    torch.testing.assert_close(qp.support_target[:, :4], torch.zeros(1, 4, dtype=state.dtype))


def test_qp_swing_target_raises_below_floor_and_preserves_safe_nominal() -> None:
    cfg = JointMpcRtiCfg()
    low = _state(batch=1).to(torch.float64)
    low[..., 2] -= 0.05
    low_qp = linearize_trajectory(low, _context(low), cfg)
    high = low.clone()
    high[..., 2] += 0.1
    high_qp = linearize_trajectory(high, _context(high), cfg)

    assert low_qp.support_target.shape == (1, 6)
    assert torch.all(low_qp.support_target[:, 4:] > 0.0)
    torch.testing.assert_close(
        high_qp.support_target[:, 4:],
        torch.zeros(1, 2, dtype=high.dtype),
    )


def test_dense_kkt_satisfies_affine_published_stance_target() -> None:
    state = _state(batch=2).to(torch.float64)
    context = _context(state)
    stance_index = torch.topk(
        context.schedule.stance[:, 1].to(torch.int64), k=2, dim=1
    ).indices
    anchors = context.stance_anchor_w.clone()
    selected_anchor = torch.gather(
        anchors[:, 1], 1, stance_index[..., None].expand(-1, -1, 3)
    )
    selected_anchor[..., 0] += 0.0004
    anchors[:, 1].scatter_(
        1, stance_index[..., None].expand(-1, -1, 3), selected_anchor
    )
    qp = linearize_trajectory(state, replace(context, stance_anchor_w=anchors), JointMpcRtiCfg())

    assert qp.support_jacobian.shape == (2, 6, 18)
    assert torch.any(qp.support_target != 0.0)
    direction = solve_dense_active_kkt(qp, ActiveConstraints.empty(qp))
    support_motion = torch.einsum(
        "bri,bi->br", qp.support_jacobian, direction[:, 1]
    )

    torch.testing.assert_close(
        support_motion,
        qp.support_target,
        atol=2.0e-8,
        rtol=0.0,
    )


def test_dense_and_scan_satisfy_positive_swing_floor_with_original_bounds() -> None:
    state = _state(batch=1)
    state[..., 2] -= 0.01
    qp = linearize_trajectory(state, _context(state), JointMpcRtiCfg())

    assert torch.all(qp.support_target[:, 4:] > 0.0)
    dense = refine_active_set(qp, solve_dense_active_kkt, refinements=2)
    scan = solve_trajectory_qp_scan(qp)
    for direction in (dense.direction, scan.direction):
        support_motion = torch.einsum(
            "bri,bi->br", qp.support_jacobian, direction[:, 1]
        )
        difference = direction[:, 1:, 6:] - direction[:, :-1, 6:]
        torch.testing.assert_close(
            support_motion, qp.support_target, atol=2.0e-5, rtol=0.0
        )
        torch.testing.assert_close(
            direction[:, 1, :2], torch.zeros_like(direction[:, 1, :2])
        )
        assert torch.all(direction >= qp.lower - 2.0e-5)
        assert torch.all(direction <= qp.upper + 2.0e-5)
        assert torch.all(difference >= qp.joint_difference_lower - 2.0e-5)
        assert torch.all(difference <= qp.joint_difference_upper + 2.0e-5)
    torch.testing.assert_close(scan.direction, dense.direction, atol=2.0e-5, rtol=2.0e-5)


def test_affine_seed_respects_fixed_published_xy_before_bound_refinement() -> None:
    dtype = torch.float64
    diagonal = torch.eye(18, dtype=dtype).expand(1, 31, -1, -1).clone()
    gradient = torch.zeros(1, 31, 18, dtype=dtype)
    gradient[0, 2, :3] = -1.0
    lower = torch.full_like(gradient, -10.0)
    upper = torch.full_like(gradient, 10.0)
    lower[:, 0] = 0.0
    upper[:, 0] = 0.0
    lower[:, 1, :2] = 0.0
    upper[:, 1, :2] = 0.0
    upper[0, 2, :3] = torch.tensor((0.1, 0.2, 0.3), dtype=dtype)
    support = torch.zeros(1, 6, 18, dtype=dtype)
    support[0, 0, (0, 6)] = 1.0
    support[0, 1, (1, 7)] = 1.0
    support[0, 2, 8] = 1.0
    support[0, 3, 9] = 1.0
    support[0, 4, 10] = 1.0
    support[0, 5, 11] = 1.0
    qp = TrajectoryQp(
        diagonal=diagonal,
        first_offdiag=torch.zeros(1, 30, 18, 18, dtype=dtype),
        second_offdiag=torch.zeros(1, 29, 18, 18, dtype=dtype),
        gradient=gradient,
        lower=lower,
        upper=upper,
        joint_difference_lower=torch.full((1, 30, 12), -10.0, dtype=dtype),
        joint_difference_upper=torch.full((1, 30, 12), 10.0, dtype=dtype),
        support_jacobian=support,
        support_target=torch.full((1, 6), 0.06, dtype=dtype),
    )

    dense = refine_active_set(qp, solve_dense_active_kkt, refinements=2)
    scan = solve_trajectory_qp_scan(qp)

    for direction in (dense.direction, scan.direction):
        torch.testing.assert_close(
            direction[:, 1, :2], torch.zeros_like(direction[:, 1, :2]), atol=1.0e-12, rtol=0.0
        )
        torch.testing.assert_close(
            torch.einsum("bri,bi->br", support, direction[:, 1]),
            qp.support_target,
            atol=1.0e-10,
            rtol=0.0,
        )
        assert torch.all(direction >= lower - 1.0e-10)
        assert torch.all(direction <= upper + 1.0e-10)


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
        initialized=torch.zeros(1, dtype=torch.bool),
        stance_anchor_w=torch.zeros(1, 4, 3),
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
