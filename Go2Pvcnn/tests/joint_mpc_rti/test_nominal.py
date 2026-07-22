from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model import nominal as nominal_module
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
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    return JointMpcRtiSolverState(
        trajectory=measured.as_vector()[:, None].expand(-1, 31, -1).clone(),
        gait_phase=phase,
        initialized=torch.zeros(measured.batch_size, dtype=torch.bool),
        stance_anchor_w=go2_fk(
            measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
        ).foot_pos_w,
    )


def test_nominal_joint_reference_uses_cached_constant_for_cuda_graph_capture() -> None:
    source = inspect.getsource(nominal_module._build_foot_references)

    assert "constant_like" in source
    assert "cfg.gait.nominal_joint_pos" in source


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
    previous = JointMpcRtiSolverState(
        cold.state,
        phase,
        torch.ones(2, dtype=torch.bool),
        cold.current_stance_anchor_w,
    )
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


def test_each_environment_cold_starts_once_then_uses_warm_until_reset() -> None:
    cfg = JointMpcRtiCfg()
    measured = _measured(2)
    phase = torch.tensor([0, 7])
    first = build_nominal(
        measured,
        torch.zeros(2, 3),
        _field(2),
        phase,
        previous=_invalid_previous(measured, phase),
        cfg=cfg,
    )

    assert first.used_cold_start.all()
    assert not first.used_warm_start.any()

    initialized = JointMpcRtiSolverState(
        trajectory=first.state,
        gait_phase=phase + 1,
        initialized=torch.ones(2, dtype=torch.bool),
        stance_anchor_w=first.current_stance_anchor_w,
    )
    second = build_nominal(
        measured,
        torch.zeros(2, 3),
        _field(2),
        phase + 1,
        previous=initialized,
        cfg=cfg,
    )

    assert not second.used_cold_start.any()
    assert second.used_warm_start.all()

    reset_one = JointMpcRtiSolverState(
        trajectory=second.state,
        gait_phase=phase + 2,
        initialized=torch.tensor([True, False]),
        stance_anchor_w=second.current_stance_anchor_w,
    )
    third = build_nominal(
        measured,
        torch.zeros(2, 3),
        _field(2),
        phase + 2,
        previous=reset_one,
        cfg=cfg,
    )

    assert torch.equal(third.used_cold_start, torch.tensor([False, True]))
    assert torch.equal(third.used_warm_start, torch.tensor([True, False]))


def test_initialized_nonfinite_warm_cache_raises_instead_of_cold_start() -> None:
    cfg = JointMpcRtiCfg()
    measured = _measured(1)
    phase = torch.zeros(1, dtype=torch.long)
    corrupt = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    corrupt[:, 4, 7] = torch.nan
    previous = JointMpcRtiSolverState(
        trajectory=corrupt,
        gait_phase=phase,
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=torch.zeros(1, 4, 3),
    )

    with pytest.raises(nominal_module.WarmStartInvariantError, match="initialized warm cache"):
        build_nominal(
            measured,
            torch.zeros(1, 3),
            _field(1),
            phase,
            previous=previous,
            cfg=cfg,
        )


def test_initialized_warm_nominal_preserves_persistent_stance_anchor() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    measured = _measured(1)
    phase = torch.tensor([12])
    measured_foot = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w
    persistent_anchor = measured_foot.clone()
    persistent_anchor[:, (0, 3), 0] -= 0.01
    previous = JointMpcRtiSolverState(
        trajectory=measured.as_vector()[:, None].expand(-1, 31, -1).clone(),
        gait_phase=phase,
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=persistent_anchor,
    )

    result = build_nominal(
        measured,
        torch.tensor((0.2, 0.0, 0.0)).view(1, 3),
        _field(1),
        phase,
        previous=previous,
        cfg=JointMpcRtiCfg(),
    )

    assert result.used_warm_start.all()
    torch.testing.assert_close(result.current_stance_anchor_w, persistent_anchor)


@pytest.mark.parametrize("phase_value", (11, 12))
def test_warm_published_x1_initializes_all_stance_on_raw_ground_manifold(
    phase_value: int,
) -> None:
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    cfg = JointMpcRtiCfg()
    measured = _measured(1)
    phase = torch.tensor([phase_value])
    measured_vector = measured.as_vector()
    previous_trajectory = measured_vector[:, None].expand(-1, 31, -1).clone()
    previous_trajectory[:, 2:, 0] += 0.006
    previous_trajectory[:, 2:, 2] += 0.030
    persistent_anchor = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w
    previous = JointMpcRtiSolverState(
        trajectory=previous_trajectory,
        gait_phase=phase,
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=persistent_anchor,
    )
    shifted = shift_rebase_trajectory(
        previous_trajectory,
        measured_vector,
        decay_nodes=int(cfg.nominal.measurement_decay_nodes),
    )
    shifted_foot = go2_fk(
        shifted[:, 1, :3], shifted[:, 1, 3:6], shifted[:, 1, 6:]
    ).foot_pos_w
    schedule = fixed_trot_schedule(phase, horizon_steps=cfg.runtime.horizon_steps)
    continuing = schedule.stance[:, 0] & schedule.stance[:, 1]
    onset = ~schedule.stance[:, 0] & schedule.stance[:, 1]
    published_stance = schedule.stance[:, 1]
    before_error = torch.linalg.vector_norm(
        shifted_foot[..., :2] - persistent_anchor[..., :2], dim=-1
    )
    if continuing.any():
        torch.testing.assert_close(
            before_error[continuing],
            torch.full_like(before_error[continuing], 0.006),
            atol=2.0e-5,
            rtol=0.0,
        )

    result = build_nominal(
        measured,
        torch.tensor((0.2, 0.0, 0.0)).view(1, 3),
        _field(1),
        phase,
        previous=previous,
        cfg=cfg,
    )

    corrected_foot = go2_fk(
        result.state[:, 1, :3], result.state[:, 1, 3:6], result.state[:, 1, 6:]
    ).foot_pos_w
    target_xy = torch.where(
        continuing[..., None], persistent_anchor[..., :2], shifted_foot[..., :2]
    )
    if continuing.any():
        torch.testing.assert_close(
            corrected_foot[..., :2][continuing],
            persistent_anchor[..., :2][continuing],
            atol=2.0e-5,
            rtol=0.0,
        )
    if onset.any():
        torch.testing.assert_close(
            corrected_foot[..., :2][onset],
            shifted_foot[..., :2][onset],
            atol=2.0e-5,
            rtol=0.0,
        )
    torch.testing.assert_close(
        corrected_foot[..., :2][published_stance],
        target_xy[published_stance],
        atol=2.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        corrected_foot[..., 2][published_stance],
        torch.full_like(corrected_foot[..., 2][published_stance], cfg.gait.foot_contact_offset),
        atol=2.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(result.state[:, 0], shifted[:, 0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.state[:, 1, :6], shifted[:, 1, :6], atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.state[:, 2:], shifted[:, 2:], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        result.state[:, 1, 6:].reshape(1, 4, 3)[~published_stance],
        shifted[:, 1, 6:].reshape(1, 4, 3)[~published_stance],
        atol=0.0,
        rtol=0.0,
    )
    assert result.valid.all()
    assert result.used_warm_start.all()
    assert not result.used_cold_start.any()


@pytest.mark.parametrize(
    ("root_shift_x", "failure"),
    ((0.30, "reachability"), (-0.20, "velocity"), (-0.30, "position")),
)
def test_warm_published_x1_manifold_failure_stays_warm_and_marks_invalid(
    root_shift_x: float,
    failure: str,
) -> None:
    from extension.joint_mpc_rti.model.analytic_ik import go2_analytic_ik
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.solver.trajectory_qp import JOINT_LOWER, JOINT_UPPER

    cfg = JointMpcRtiCfg()
    measured = _measured(1)
    phase = torch.tensor([12])
    measured_vector = measured.as_vector()
    previous_trajectory = measured_vector[:, None].expand(-1, 31, -1).clone()
    previous_trajectory[:, 2:, 0] += root_shift_x
    persistent_anchor = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w
    previous = JointMpcRtiSolverState(
        trajectory=previous_trajectory,
        gait_phase=phase,
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=persistent_anchor,
    )
    shifted = shift_rebase_trajectory(
        previous_trajectory,
        measured_vector,
        decay_nodes=int(cfg.nominal.measurement_decay_nodes),
    )
    shifted_foot = go2_fk(
        shifted[:, 1, :3], shifted[:, 1, 3:6], shifted[:, 1, 6:]
    ).foot_pos_w
    target = torch.cat((persistent_anchor[..., :2], shifted_foot[..., 2:3]), dim=-1)
    ik_joint, reachable = go2_analytic_ik(
        shifted[:, 1, :3], shifted[:, 1, 3:6], target
    )
    continuing = (
        fixed_trot_schedule(phase, horizon_steps=cfg.runtime.horizon_steps).stance[:, :2]
    ).all(dim=1)
    position_ok = (
        (ik_joint.flatten(1) >= ik_joint.new_tensor(JOINT_LOWER))
        & (ik_joint.flatten(1) <= ik_joint.new_tensor(JOINT_UPPER))
    ).all(dim=-1)
    velocity_ok = (
        (ik_joint.flatten(1) - measured.joint_pos).abs()
        <= float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt)
    ).all(dim=-1)
    if failure == "reachability":
        assert not reachable[continuing].all()
    elif failure == "velocity":
        assert reachable[continuing].all() and position_ok.all() and not velocity_ok.all()
    else:
        assert reachable[continuing].all() and not position_ok.all()

    result = build_nominal(
        measured,
        torch.zeros(1, 3),
        _field(1),
        phase,
        previous=previous,
        cfg=cfg,
    )

    assert not result.valid.any()
    assert result.used_warm_start.all()
    assert not result.used_cold_start.any()


@pytest.mark.parametrize("batch", (1, 40, 512, 1024))
def test_warm_published_x1_manifold_preserves_batched_shape_and_finite_state(
    batch: int,
) -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk

    measured = _measured(batch)
    phase = torch.full((batch,), 12, dtype=torch.long)
    previous_trajectory = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    previous_trajectory[:, 2:, 0] += 0.006
    previous = JointMpcRtiSolverState(
        trajectory=previous_trajectory,
        gait_phase=phase,
        initialized=torch.ones(batch, dtype=torch.bool),
        stance_anchor_w=go2_fk(
            measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
        ).foot_pos_w,
    )

    result = build_nominal(
        measured,
        torch.zeros(batch, 3),
        _field(batch),
        phase,
        previous=previous,
        cfg=JointMpcRtiCfg(),
    )

    assert result.state.shape == (batch, 31, 18)
    assert result.valid.shape == (batch,)
    assert result.valid.all()
    assert torch.isfinite(result.state).all()


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


def test_warm_shift_keeps_yaw_continuous_across_pi() -> None:
    measured = _measured(1).as_vector()
    previous = measured[:, None].expand(-1, 31, -1).clone()
    previous[0, :, 5] = 3.12 + 0.02 * torch.arange(31)
    measured[0, 5] = torch.remainder(previous[0, 1, 5] + torch.pi, 2.0 * torch.pi) - torch.pi

    shifted = shift_rebase_trajectory(previous, measured, decay_nodes=6)

    yaw_step = shifted[0, 1:, 5] - shifted[0, :-1, 5]
    torch.testing.assert_close(shifted[0, 0, 5], measured[0, 5])
    torch.testing.assert_close(yaw_step, torch.full_like(yaw_step, 0.02), atol=1.0e-6, rtol=0.0)
    assert shifted[0, -1, 5] > torch.pi


def test_warm_shift_moves_the_whole_yaw_horizon_to_the_measured_branch() -> None:
    measured = _measured(1).as_vector()
    previous = measured[:, None].expand(-1, 31, -1).clone()
    previous[0, :, 5] = 3.14 + 0.02 * torch.arange(31)
    measured[0, 5] = torch.remainder(previous[0, 1, 5] + torch.pi, 2.0 * torch.pi) - torch.pi

    shifted = shift_rebase_trajectory(previous, measured, decay_nodes=6)

    yaw_step = shifted[0, 1:, 5] - shifted[0, :-1, 5]
    torch.testing.assert_close(shifted[0, 0, 5], measured[0, 5])
    torch.testing.assert_close(yaw_step, torch.full_like(yaw_step, 0.02), atol=1.0e-6, rtol=0.0)
    assert shifted[0, -1, 5] < -2.0


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
    previous = JointMpcRtiSolverState(
        cold.state,
        phase,
        torch.ones(1, dtype=torch.bool),
        cold.current_stance_anchor_w,
    )
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


def test_cold_nominal_holds_root_pose_on_the_relaxed_first_command_edge() -> None:
    cfg = JointMpcRtiCfg()
    measured = _measured(1)
    command = torch.tensor((1.0, 0.5, 1.0)).view(1, 3)
    phase = torch.zeros(1, dtype=torch.long)

    result = build_nominal(
        measured,
        command,
        _field(1),
        phase,
        previous=_invalid_previous(measured, phase),
        cfg=cfg,
    )

    torch.testing.assert_close(result.state[:, 1, :3], measured.root_pos_w)
    torch.testing.assert_close(result.state[:, 1, 3:6], measured.root_rpy_w)
    expected_xy = command[:, :2] * float(cfg.nominal.command_scale) * float(cfg.runtime.dt)
    expected_yaw = command[:, 2] * float(cfg.nominal.command_scale) * float(cfg.runtime.dt)
    torch.testing.assert_close(result.state[:, 2, :2] - result.state[:, 1, :2], expected_xy)
    torch.testing.assert_close(result.state[:, 2, 5] - result.state[:, 1, 5], expected_yaw)


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


def test_warm_phase_eleven_endpoint_does_not_query_unobservable_liftoff() -> None:
    measured = _measured(1)
    phase = torch.tensor([23])
    previous = JointMpcRtiSolverState(
        trajectory=measured.as_vector()[:, None].expand(-1, 31, -1).clone(),
        gait_phase=phase,
        initialized=torch.ones(1, dtype=torch.bool),
        stance_anchor_w=torch.zeros(1, 4, 3),
    )

    result = build_nominal(
        measured,
        torch.tensor((0.4, 0.0, 1.0)).view(1, 3),
        _field(1),
        phase,
        previous=previous,
        cfg=JointMpcRtiCfg(),
    )

    assert result.used_warm_start.all()
    assert not result.used_cold_start.any()
    assert result.valid.all()
    assert torch.isfinite(result.foot_reference_w).all()


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
    source = "\n".join(
        (
            inspect.getsource(build_nominal),
            inspect.getsource(nominal_module._initialize_published_stance_manifold),
        )
    )
    tree = ast.parse(textwrap.dedent(source))

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
