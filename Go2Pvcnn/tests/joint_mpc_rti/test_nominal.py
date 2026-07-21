from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.nominal import build_nominal
from extension.joint_mpc_rti.runtime.warm_start import shift_rebase_trajectory
from extension.joint_mpc_rti.types import JointMpcRtiSolverState, JointMpcRtiState, JointMpcTerrainField


def _measured(batch: int) -> JointMpcRtiState:
    root_pos = torch.zeros(batch, 3)
    root_pos[:, 2] = 0.34
    return JointMpcRtiState(
        root_pos_w=root_pos,
        root_rpy_w=torch.zeros(batch, 3),
        joint_pos=torch.tensor((0.0, 0.8, -1.5) * 4).expand(batch, -1).clone(),
        root_lin_vel_b=torch.zeros(batch, 3),
        root_ang_vel_b=torch.zeros(batch, 3),
        joint_vel=torch.zeros(batch, 12),
    )


def _field(batch: int, *, semantic_value: int = 0) -> JointMpcTerrainField:
    shape = (batch, 3, 3)
    return JointMpcTerrainField(
        height_w=torch.zeros(shape),
        semantic_id=torch.full(shape, semantic_value, dtype=torch.long),
        small_distance_m=torch.ones(shape),
        large_distance_m=torch.ones(shape),
        small_gradient_xy=torch.zeros(*shape, 2),
        large_gradient_xy=torch.zeros(*shape, 2),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        origin_w=torch.zeros(batch, 3),
        yaw_w=torch.zeros(batch),
        timestamp=torch.zeros(batch),
        version=torch.ones(batch, dtype=torch.long),
        resolution=1.0,
    )


def _invalid_previous(measured: JointMpcRtiState, phase: torch.Tensor) -> JointMpcRtiSolverState:
    return JointMpcRtiSolverState(
        trajectory=measured.as_vector()[:, None].expand(-1, 31, -1).clone(),
        gait_phase=phase,
        valid=torch.zeros(measured.batch_size, dtype=torch.bool),
    )


@pytest.mark.parametrize("batch", (1, 40, 512, 1024))
def test_nominal_builder_returns_complete_b31_state_in_one_call(batch: int) -> None:
    measured = _measured(batch)
    phase = torch.arange(batch) % 24

    result = build_nominal(
        measured,
        torch.tensor((0.2, 0.05, 0.1)).expand(batch, -1),
        _field(batch),
        phase,
        previous=_invalid_previous(measured, phase),
        cfg=JointMpcRtiCfg(),
    )

    assert result.state.shape == (batch, 31, 18)
    assert result.foot_reference_w.shape == (batch, 31, 4, 3)
    assert result.touchdown_reference_w.shape == (batch, 31, 4, 3)
    assert result.contact_state.shape == (batch, 31, 4)
    assert torch.isfinite(result.state).all()
    torch.testing.assert_close(result.state[:, 0], measured.as_vector())


def test_warm_nominal_is_shift_rebase_and_measurement_decay() -> None:
    cfg = JointMpcRtiCfg()
    measured = _measured(2)
    phase = torch.tensor([0, 7])
    cold = build_nominal(
        measured,
        torch.tensor((0.25, 0.0, 0.2)).expand(2, -1),
        _field(2),
        phase,
        previous=_invalid_previous(measured, phase),
        cfg=cfg,
    )
    previous = JointMpcRtiSolverState(cold.state, phase, torch.ones(2, dtype=torch.bool))
    measured_next = JointMpcRtiState(
        root_pos_w=measured.root_pos_w + torch.tensor((0.01, -0.02, 0.0)),
        root_rpy_w=measured.root_rpy_w + torch.tensor((0.0, 0.0, 0.05)),
        joint_pos=measured.joint_pos + 0.02,
        root_lin_vel_b=measured.root_lin_vel_b,
        root_ang_vel_b=measured.root_ang_vel_b,
        joint_vel=measured.joint_vel,
    )

    result = build_nominal(
        measured_next,
        torch.tensor((0.25, 0.0, 0.2)).expand(2, -1),
        _field(2),
        phase + 1,
        previous=previous,
        cfg=cfg,
    )

    assert result.used_warm_start.all()
    torch.testing.assert_close(result.state[:, 0], measured_next.as_vector())
    torch.testing.assert_close(result.state[:, 30, 6:], previous.trajectory[:, 30, 6:], atol=1e-6, rtol=0.0)
    delta_yaw = measured_next.root_rpy_w[:, 2] - cold.state[:, 1, 5]
    relative = cold.state[:, 6, :2] - cold.state[:, 1, :2]
    expected_xy = measured_next.root_pos_w[:, :2] + torch.stack(
        (
            torch.cos(delta_yaw) * relative[:, 0] - torch.sin(delta_yaw) * relative[:, 1],
            torch.sin(delta_yaw) * relative[:, 0] + torch.cos(delta_yaw) * relative[:, 1],
        ),
        dim=-1,
    )
    torch.testing.assert_close(result.state[:, 5, :2], expected_xy)


def test_warm_shift_does_not_create_a_terminal_joint_velocity_violation() -> None:
    measured = _measured(1)
    previous = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    previous[:, 6, 6] += 0.35
    previous[:, 7, 6] += 0.70
    previous[:, 8, 6] += 0.35
    velocity_step_limit = 30.0 * 0.02

    assert (previous[:, 1:, 6:] - previous[:, :-1, 6:]).abs().amax() <= velocity_step_limit

    shifted = shift_rebase_trajectory(previous, previous[:, 1], decay_nodes=6)

    assert (shifted[:, 1:, 6:] - shifted[:, :-1, 6:]).abs().amax() <= velocity_step_limit


def test_warm_nominal_uses_current_measured_stance_reference() -> None:
    cfg = JointMpcRtiCfg()
    measured = _measured(1)
    phase = torch.zeros(1, dtype=torch.long)
    cold = build_nominal(
        measured,
        torch.tensor((0.25, 0.0, 0.2)).view(1, 3),
        _field(1),
        phase,
        previous=_invalid_previous(measured, phase),
        cfg=cfg,
    )
    measured_next = JointMpcRtiState(
        root_pos_w=measured.root_pos_w,
        root_rpy_w=measured.root_rpy_w,
        joint_pos=measured.joint_pos + 0.02,
        root_lin_vel_b=measured.root_lin_vel_b,
        root_ang_vel_b=measured.root_ang_vel_b,
        joint_vel=measured.joint_vel,
    )
    previous = JointMpcRtiSolverState(cold.state, phase, torch.ones(1, dtype=torch.bool))
    result = build_nominal(
        measured_next,
        torch.tensor((0.25, 0.0, 0.2)).view(1, 3),
        _field(1),
        phase + 1,
        previous=previous,
        cfg=cfg,
    )

    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    measured_foot = go2_fk(
        measured_next.root_pos_w, measured_next.root_rpy_w, measured_next.joint_pos
    ).foot_pos_w
    # At phase one, legs 1/2 remain in the current stance through the next lift.
    expected = measured_foot[:, None, 1:3, :2].expand(-1, 11, -1, -1)
    torch.testing.assert_close(result.foot_reference_w[:, :11, 1:3, :2], expected)


def test_cold_nominal_does_not_use_semantics_to_modify_xy() -> None:
    cfg = JointMpcRtiCfg()
    measured = _measured(3)
    phase = torch.tensor([0, 4, 19])
    previous = _invalid_previous(measured, phase)
    command = torch.tensor((0.3, 0.1, 0.0)).expand(3, -1)

    flat = build_nominal(measured, command, _field(3, semantic_value=0), phase, previous=previous, cfg=cfg)
    semantic = build_nominal(measured, command, _field(3, semantic_value=2), phase, previous=previous, cfg=cfg)

    torch.testing.assert_close(flat.state[..., :2], semantic.state[..., :2])
    torch.testing.assert_close(flat.foot_reference_w[..., :2], semantic.foot_reference_w[..., :2])
    torch.testing.assert_close(flat.touchdown_reference_w[..., :2], semantic.touchdown_reference_w[..., :2])


def test_current_stance_nominal_anchor_starts_at_measured_foot() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    cfg = JointMpcRtiCfg()
    measured = _measured(1)
    phase = torch.zeros(1, dtype=torch.long)
    result = build_nominal(
        measured,
        torch.tensor((1.0, 0.5, 1.0)).view(1, 3),
        _field(1),
        phase,
        previous=_invalid_previous(measured, phase),
        cfg=cfg,
    )
    measured_foot = go2_fk(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos).foot_pos_w

    # Phase zero has diagonal swing legs 0/3 and stance legs 1/2.
    expected = measured_foot[:, None, 1:3, :2].expand(-1, 12, -1, -1)
    torch.testing.assert_close(result.foot_reference_w[:, :12, 1:3, :2], expected)


def test_phase_zero_liftoff_reference_starts_at_measured_foot() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    measured = _measured(1)
    phase = torch.zeros(1, dtype=torch.long)
    result = build_nominal(
        measured,
        torch.tensor((1.0, 0.5, 1.0)).view(1, 3),
        _field(1),
        phase,
        previous=_invalid_previous(measured, phase),
        cfg=JointMpcRtiCfg(),
    )
    measured_foot = go2_fk(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos).foot_pos_w

    # At phase zero, diagonal legs 0/3 lift from the measured x0 pose.
    torch.testing.assert_close(result.foot_reference_w[:, 0, (0, 3)], measured_foot[:, (0, 3)])


def test_future_liftoff_reference_is_continuous_with_previous_stance() -> None:
    measured = _measured(1)
    phase = torch.zeros(1, dtype=torch.long)
    result = build_nominal(
        measured,
        torch.tensor((1.0, 0.5, 1.0)).view(1, 3),
        _field(1),
        phase,
        previous=_invalid_previous(measured, phase),
        cfg=JointMpcRtiCfg(),
    )
    liftoff = result.contact_state[:, :-1] & ~result.contact_state[:, 1:]
    before = result.foot_reference_w[:, :-1][liftoff]
    after = result.foot_reference_w[:, 1:][liftoff]

    assert before.numel() > 0
    torch.testing.assert_close(after, before, atol=1.0e-6, rtol=0.0)


def test_nominal_source_has_no_for_or_while() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(build_nominal)))

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
