from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import go2_collision_geometry, go2_fk
from extension.joint_mpc_rti.model.analytic_ik import go2_analytic_ik
from extension.joint_mpc_rti.model.nominal import build_nominal
from extension.joint_mpc_rti.model.perceptive_plan import TouchdownPlan, select_touchdowns
from extension.joint_mpc_rti.runtime.warm_start import shift_rebase_trajectory
from extension.joint_mpc_rti.types import (
    JointMpcPerceptiveField,
    JointMpcRtiSolverState,
    JointMpcRtiState,
)

from .helpers import make_command, make_state
from .test_perceptive_plan import _field


def _to_device(field: JointMpcPerceptiveField, device: str) -> JointMpcPerceptiveField:
    return replace(
        field,
        **{
            item.name: getattr(field, item.name).to(device)
            for item in fields(field)
            if isinstance(getattr(field, item.name), torch.Tensor)
        },
    )


def _seed(
    measured: JointMpcRtiState,
    phase: torch.Tensor,
    *,
    initialized: bool | torch.Tensor,
    trajectory: torch.Tensor | None = None,
    anchor: torch.Tensor | None = None,
    preview_tail_state: torch.Tensor | None = None,
) -> JointMpcRtiSolverState:
    batch = measured.batch_size
    initialized_mask = torch.as_tensor(
        initialized, dtype=torch.bool, device=measured.device
    ).expand(batch).clone()
    if trajectory is None:
        trajectory = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    if anchor is None:
        anchor = go2_collision_geometry(
            measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
        ).foot_center_w
    return JointMpcRtiSolverState(
        trajectory=trajectory,
        gait_phase=phase,
        initialized=initialized_mask,
        stance_anchor_w=anchor,
        preview_tail_state=preview_tail_state,
    )


def _plan(
    measured: JointMpcRtiState,
    command: torch.Tensor,
    phase: torch.Tensor,
    field: JointMpcPerceptiveField,
    warm: torch.Tensor,
    *,
    previous: TouchdownPlan | None = None,
) -> TouchdownPlan:
    return select_touchdowns(
        measured,
        command,
        fixed_trot_schedule(phase),
        warm,
        field,
        JointMpcRtiCfg(),
        previous_plan=previous,
    )


def _build(
    measured: JointMpcRtiState,
    command: torch.Tensor,
    field: JointMpcPerceptiveField,
    phase: torch.Tensor,
    plan: TouchdownPlan,
    previous: JointMpcRtiSolverState,
):
    return build_nominal(
        measured,
        command,
        field,
        phase,
        perceptive_plan=plan,
        previous=previous,
        cfg=JointMpcRtiCfg(),
    )


def test_only_first_optimize_after_reset_is_cold() -> None:
    measured = make_state(2)
    command = make_command(2)
    phase = torch.tensor([0, 7])
    field = _field()
    seed = _seed(measured, phase, initialized=False)
    first_plan = _plan(measured, command, phase, field, seed.trajectory)

    first = _build(measured, command, field, phase, first_plan, seed)
    accepted = _seed(
        measured,
        phase + 1,
        initialized=True,
        trajectory=first.state,
        anchor=first.current_stance_anchor_w,
    )
    second_plan = _plan(
        measured, command, phase + 1, field, first.state, previous=first_plan
    )
    second = _build(measured, command, field, phase + 1, second_plan, accepted)

    assert first.state.shape == (2, 31, 18)
    assert first.foot_reference_w.shape == (2, 31, 4, 3)
    assert first.used_cold_start.all()
    assert not first.used_warm_start.any()
    assert second.used_warm_start.all()
    assert not second.used_cold_start.any()
    torch.testing.assert_close(first.state[:, 0], measured.as_vector())
    torch.testing.assert_close(second.state[:, 0], measured.as_vector())


def test_initialized_nonfinite_cache_is_a_warm_fault_not_a_cold_fallback() -> None:
    measured = make_state(2)
    command = make_command(2)
    phase = torch.tensor([0, 6])
    field = _field()
    trajectory = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    trajectory[1, 5, 9] = torch.nan
    previous = _seed(measured, phase, initialized=True, trajectory=trajectory)
    plan = _plan(measured, command, phase, field, torch.nan_to_num(trajectory))

    result = _build(measured, command, field, phase, plan, previous)

    assert torch.equal(result.warm_cache_invariant_fault, torch.tensor([False, True]))
    assert result.used_warm_start.all()
    assert not result.used_cold_start.any()
    assert not result.nominal_safe[1]


@pytest.mark.parametrize("fault", ("missing_trajectory", "trajectory_shape", "anchor_shape", "initialized_shape"))
def test_initialized_cache_contract_fault_stops_without_cold_fallback(
    fault: str,
) -> None:
    measured = make_state(1)
    command = make_command(1)
    phase = torch.tensor([0])
    field = _field()
    valid = _seed(measured, phase, initialized=True)
    plan = _plan(measured, command, phase, field, valid.trajectory)
    values = {
        "trajectory": valid.trajectory,
        "gait_phase": valid.gait_phase,
        "initialized": valid.initialized,
        "stance_anchor_w": valid.stance_anchor_w,
    }
    if fault == "missing_trajectory":
        values["trajectory"] = None
    elif fault == "trajectory_shape":
        values["trajectory"] = valid.trajectory[:, :-1]
    elif fault == "anchor_shape":
        values["stance_anchor_w"] = valid.stance_anchor_w[..., :2]
    else:
        values["initialized"] = valid.initialized[:, None]
    previous = JointMpcRtiSolverState(**values)

    result = _build(measured, command, field, phase, plan, previous)

    assert result.warm_cache_invariant_fault.all()
    assert result.used_warm_start.all()
    assert not result.used_cold_start.any()
    assert not result.nominal_safe.any()


def test_shift_rebase_preserves_trend_and_injects_measured_z0_exactly() -> None:
    measured = make_state(1)
    previous = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    node = torch.arange(31, dtype=previous.dtype)
    previous[..., 0] += 0.01 * node
    previous[..., 5] += 0.02 * node
    previous[..., 7] += 0.001 * node
    measured_vector = measured.as_vector().clone()
    measured_vector[:, :2] += torch.tensor((0.03, -0.02))
    measured_vector[:, 5] += 0.1

    rebased = shift_rebase_trajectory(
        previous,
        measured_vector,
        decay_nodes=6,
        command_body=torch.tensor([[0.3, 0.0, 0.1]]),
        dt=0.02,
    )

    torch.testing.assert_close(rebased[:, 0], measured_vector, atol=0.0, rtol=0.0)
    assert torch.all(rebased[:, 2:, 0] > rebased[:, 1:-1, 0])
    assert torch.isfinite(rebased).all()
    assert (rebased[:, -1, 7] - rebased[:, -2, 7]).abs().max() <= 0.6


def test_nominal_exposes_single_apex_and_warm_retarget_tuning_parameters() -> None:
    cfg = JointMpcRtiCfg()

    assert cfg.nominal.swing_outward_offset_m >= 0.0
    assert cfg.nominal.swing_apex_margin_m > 0.0
    assert 0.0 <= cfg.nominal.terminal_command_fill_scale <= 1.0
    assert 0.0 < cfg.nominal.ik_blend_scale <= 1.0


def test_warm_selector_change_retargets_without_becoming_cold() -> None:
    measured = make_state(1)
    command = make_command(1)
    phase = torch.tensor([0])
    field = _field()
    seed = _seed(measured, phase, initialized=False)
    first_plan = _plan(measured, command, phase, field, seed.trajectory)
    cold = _build(measured, command, field, phase, first_plan, seed)
    previous = _seed(
        measured,
        phase,
        initialized=True,
        trajectory=cold.state,
        anchor=cold.current_stance_anchor_w,
    )
    shifted_target = first_plan.target_w.clone()
    shifted_target[..., 0] += 0.02
    changed_plan = replace(first_plan, target_w=shifted_target)

    result = _build(measured, command, field, phase, changed_plan, previous)

    assert result.used_warm_start.all()
    assert not result.used_cold_start.any()
    assert result.retarget_change.item() > 0.0
    assert torch.linalg.vector_norm(result.state - result.rebased_state) > 0.0


def test_warm_current_swing_preserves_shifted_boundary_velocity() -> None:
    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.3)
    phase = torch.tensor([5])
    field = _field()
    seed = _seed(measured, phase, initialized=False)
    first_plan = _plan(measured, command, phase, field, seed.trajectory)
    first = _build(measured, command, field, phase, first_plan, seed)
    measured_next = JointMpcRtiState(
        root_pos_w=first.state[:, 1, :3],
        root_rpy_w=first.state[:, 1, 3:6],
        joint_pos=first.state[:, 1, 6:],
        root_lin_vel_b=(first.state[:, 1, :3] - first.state[:, 0, :3])
        / cfg.runtime.dt,
        root_ang_vel_b=(first.state[:, 1, 3:6] - first.state[:, 0, 3:6])
        / cfg.runtime.dt,
        joint_vel=(first.state[:, 1, 6:] - first.state[:, 0, 6:])
        / cfg.runtime.dt,
    )
    previous = _seed(
        measured_next,
        phase + 1,
        initialized=True,
        trajectory=first.state,
        anchor=first.current_stance_anchor_w,
        preview_tail_state=first.preview_tail_state,
    )
    second_plan = _plan(
        measured_next,
        command,
        phase + 1,
        field,
        first.state,
        previous=first_plan,
    )

    second = _build(
        measured_next, command, field, phase + 1, second_plan, previous
    )

    swing = fixed_trot_schedule(phase + 1).swing[:, 0]
    first_actual = go2_fk(
        first.state[..., :3], first.state[..., 3:6], first.state[..., 6:]
    ).foot_pos_w
    second_actual = go2_fk(
        second.state[..., :3], second.state[..., 3:6], second.state[..., 6:]
    ).foot_pos_w
    old_velocity = (
        first_actual[:, 2] - first_actual[:, 1]
    ) / cfg.runtime.dt
    new_velocity = (
        second_actual[:, 1] - second_actual[:, 0]
    ) / cfg.runtime.dt
    mismatch = torch.linalg.vector_norm(old_velocity - new_velocity, dim=-1)
    assert mismatch[swing].max() <= 0.05
    assert second.nominal_safe.all()


def test_warm_swing_ik_uses_phase_dependent_quintic_blend() -> None:
    measured = make_state(1)
    command = make_command(1, vx=0.2)
    phase = torch.tensor([0])
    field = _field()
    previous = _seed(measured, phase, initialized=True)
    plan = _plan(measured, command, phase, field, previous.trajectory)

    result = _build(measured, command, field, phase, plan, previous)
    ik_joint, _ = go2_analytic_ik(
        result.rebased_state[..., :3],
        result.rebased_state[..., 3:6],
        result.foot_reference_w,
    )
    rebased_joint = result.rebased_state[..., 6:].reshape(1, 31, 4, 3)
    actual_joint = result.state[..., 6:].reshape(1, 31, 4, 3)

    def quintic(value: float) -> float:
        return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5

    for node, tau, legs in ((1, 1.0 / 12.0, (0, 3)), (25, 1.0 / 12.0, (0, 3))):
        expected = rebased_joint[:, node, legs] + quintic(tau) * (
            ik_joint[:, node, legs] - rebased_joint[:, node, legs]
        )
        torch.testing.assert_close(
            actual_joint[:, node, legs], expected, atol=2.0e-5, rtol=0.0
        )


def test_phase12_node_reaches_touchdown_without_phase11_freeze() -> None:
    measured = make_state(1)
    command = make_command(1, vx=0.3)
    phase = torch.tensor([0])
    field = _field()
    previous = _seed(measured, phase, initialized=False)
    plan = _plan(measured, command, phase, field, previous.trajectory)

    result = _build(measured, command, field, phase, plan, previous)
    foot = result.foot_reference_w[0, :, 0]

    torch.testing.assert_close(foot[12], plan.target_w[0, 0], atol=2.0e-5, rtol=0.0)
    assert torch.linalg.vector_norm(foot[12] - foot[11]) > 0.0
    assert result.contact_state[0, 12, 0]


def test_continuing_stance_uses_full_xyz_persistent_anchor_over_horizon_segment() -> None:
    measured = make_state(1)
    command = make_command(1, vx=0.05)
    phase = torch.tensor([12])
    field = _field()
    anchor = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w
    anchor = anchor.clone()
    anchor[:, (0, 3), 0] -= 0.005
    previous = _seed(measured, phase, initialized=True, anchor=anchor)
    plan = _plan(measured, command, phase, field, previous.trajectory)

    result = _build(measured, command, field, phase, plan, previous)
    schedule = fixed_trot_schedule(phase)
    future = torch.arange(31)[None, :, None] > 0
    continuing = (
        schedule.stance
        & future
        & (torch.arange(31)[None, :, None] < plan.event_step[:, None])
    )
    reference = anchor[:, None].expand_as(result.foot_reference_w)

    torch.testing.assert_close(
        result.foot_reference_w[continuing], reference[continuing], atol=2.0e-5, rtol=0.0
    )
    actual = go2_fk(
        result.state[..., :3], result.state[..., 3:6], result.state[..., 6:]
    ).foot_pos_w
    torch.testing.assert_close(actual[continuing], reference[continuing], atol=2.0e-4, rtol=0.0)


def test_h30_preview_tail_keeps_moving_toward_touchdown_outside_horizon() -> None:
    measured = make_state(1)
    command = make_command(1, vx=0.35)
    phase = torch.tensor([6])
    field = _field()
    previous = _seed(measured, phase, initialized=False)
    plan = _plan(measured, command, phase, field, previous.trajectory)

    result = _build(measured, command, field, phase, plan, previous)
    schedule = fixed_trot_schedule(phase)
    late_swing = schedule.swing[:, -3:, :]
    delta = result.foot_reference_w[:, 1:] - result.foot_reference_w[:, :-1]

    assert plan.preview_touchdown_step.gt(30).any()
    assert torch.linalg.vector_norm(delta[:, -2:], dim=-1)[late_swing[:, 1:]].max() > 0.0


def test_h30_preview_tail_runs_whole_leg_safety_to_real_touchdown() -> None:
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.terrain.swept_safety import (
        evaluate_nodes,
        evaluate_swept_intervals,
    )
    from extension.joint_mpc_rti.types import JointMpcFieldFrame

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=0.6)
    phase = torch.tensor([6])
    flat = _field()
    previous = _seed(measured, phase, initialized=False)
    selector_warm = previous.trajectory.clone()
    selector_warm[..., 0] += (
        torch.arange(31, dtype=selector_warm.dtype)[None]
        * command[:, :1]
        * cfg.nominal.command_scale
        * cfg.runtime.dt
    )
    flat_plan = _plan(measured, command, phase, flat, selector_warm)
    flat_result = _build(measured, command, flat, phase, flat_plan, previous)
    assert flat_result.preview_tail_state.shape == (1, 13, 18)
    preview_leg = torch.nonzero(
        flat_plan.preview_touchdown_step[0] > 30, as_tuple=False
    )[0, 0]
    obstacle_xy = (
        0.25 * flat_plan.target_w[0, preview_leg, :2]
        + 0.75 * flat_result.preview_touchdown_reference_w[0, preview_leg, :2]
    )
    size = 151
    center = (size - 1) // 2
    index_x = center + int(round(float(obstacle_xy[0]) / 0.01))
    index_y = center + int(round(float(obstacle_xy[1]) / 0.01))
    height = torch.zeros(1, size, size)
    semantic = torch.zeros(1, size, size, dtype=torch.long)
    height[:, index_x - 1 : index_x + 2, index_y - 1 : index_y + 2] = 0.10
    semantic[:, index_x - 1 : index_x + 2, index_y - 1 : index_y + 2] = 1
    field = build_perceptive_field(
        height,
        semantic,
        torch.ones_like(semantic, dtype=torch.bool),
        JointMpcFieldFrame(
            origin_w=torch.zeros(1, 3),
            yaw_w=torch.zeros(1),
            timestamp=torch.zeros(1),
            refresh_id=torch.zeros(1, dtype=torch.long),
        ),
        cfg,
    )
    plan = _plan(measured, command, phase, field, selector_warm)

    result = _build(measured, command, field, phase, plan, previous)
    schedule = fixed_trot_schedule(phase)
    nodes = evaluate_nodes(result.state, field, cfg, contact_state=schedule.stance)
    swept = evaluate_swept_intervals(
        result.state, field, cfg, contact_state=schedule.stance
    )

    assert plan.valid.all() and plan.preview_valid.all()
    assert nodes.safe.all() and swept.safe.all()
    assert result.nominal_safe.all()
    assert torch.linalg.vector_norm(
        result.preview_touchdown_reference_w[0, preview_leg, :2]
        - flat_result.preview_touchdown_reference_w[0, preview_leg, :2]
    ) > 0.0


def test_flat_nominal_is_finite_and_hard_safe_before_lq() -> None:
    measured = make_state(1)
    command = make_command(1, vx=0.2)
    phase = torch.tensor([0])
    field = _field()
    previous = _seed(measured, phase, initialized=False)
    plan = _plan(measured, command, phase, field, previous.trajectory)

    result = _build(measured, command, field, phase, plan, previous)

    assert torch.isfinite(result.state).all()
    assert result.nominal_safe.all()
    assert result.valid.all()
    assert result.minimum_clearance_by_part.shape == (1, 5)


def test_flat_nominal_is_hard_safe_for_all_24_start_phases() -> None:
    batch = 24
    measured = make_state(batch)
    command = make_command(batch, vx=0.2)
    phase = torch.arange(24)
    field = _field()
    previous = _seed(measured, phase, initialized=False)
    plan = _plan(measured, command, phase, field, previous.trajectory)

    result = _build(measured, command, field, phase, plan, previous)

    assert plan.valid.all()
    assert result.nominal_safe.all(), phase[~result.nominal_safe]


def test_selector_swing_path_allows_multiple_moving_flat_candidates() -> None:
    measured = make_state(1)
    command = make_command(1, vx=0.3)
    phase = torch.tensor([0])
    field = _field()
    warm = measured.as_vector()[:, None].expand(-1, 31, -1).clone()

    plan = _plan(measured, command, phase, field, warm)

    assert (plan.valid_components["sweep"].sum(dim=-1) > 1).all()
    assert plan.valid_components["sweep_resolved"].shape == (1, 4, 25)
    assert plan.valid_components["sweep_resolved"][plan.safe_mask].all()


def test_nominal_retries_ranked_candidates_without_rebuilding_selector() -> None:
    measured = make_state(1)
    command = make_command(1, vx=0.2)
    phase = torch.tensor([0])
    field = _field()
    previous = _seed(measured, phase, initialized=False)
    plan = _plan(measured, command, phase, field, previous.trajectory)
    selected = torch.nn.functional.one_hot(
        plan.selected_index, num_classes=25
    ).to(torch.bool)
    bad_candidate_w = torch.where(
        selected[..., None],
        plan.candidate_w + plan.candidate_w.new_tensor((0.8, 0.0, 0.0)),
        plan.candidate_w,
    )
    bad_target = torch.gather(
        bad_candidate_w,
        2,
        plan.selected_index[..., None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    bad_plan = replace(plan, candidate_w=bad_candidate_w, target_w=bad_target)

    result = _build(measured, command, field, phase, bad_plan, previous)

    assert result.nominal_safe.all()
    assert result.candidate_retry_rank.gt(0).all()
    assert result.perceptive_plan is not None
    assert not torch.equal(
        result.perceptive_plan.selected_index, bad_plan.selected_index
    )


def test_nominal_retry_rank_is_independent_for_each_leg() -> None:
    measured = make_state(1)
    command = make_command(1, vx=0.2)
    phase = torch.tensor([0])
    field = _field()
    previous = _seed(measured, phase, initialized=False)
    plan = _plan(measured, command, phase, field, previous.trajectory)
    corrupt_index = torch.stack(
        (
            plan.selected_index[:, 0],
            plan.selected_index[:, 1],
            plan.ranked_index[:, 1, 1],
        ),
        dim=-1,
    )
    corrupt_leg = torch.tensor((0, 1, 1))
    corrupt = torch.zeros_like(plan.safe_mask)
    corrupt[:, corrupt_leg, corrupt_index] = True
    bad_candidate_w = torch.where(
        corrupt[..., None],
        plan.candidate_w + plan.candidate_w.new_tensor((0.8, 0.0, 0.0)),
        plan.candidate_w,
    )
    bad_target = torch.gather(
        bad_candidate_w,
        2,
        plan.selected_index[..., None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    bad_plan = replace(plan, candidate_w=bad_candidate_w, target_w=bad_target)

    result = _build(measured, command, field, phase, bad_plan, previous)

    assert result.nominal_safe.all()
    assert result.candidate_retry_rank.shape == (1, 4)
    assert result.candidate_retry_rank.tolist() == [[1, 2, 0, 0]]


@pytest.mark.parametrize("batch", (1, 40))
def test_nominal_fixed_shapes_for_training_batches(batch: int) -> None:
    measured = make_state(batch)
    command = make_command(batch)
    phase = torch.arange(batch) % 24
    field = _field()
    previous = _seed(measured, phase, initialized=False)
    plan = _plan(measured, command, phase, field, previous.trajectory)

    result = _build(measured, command, field, phase, plan, previous)

    assert result.state.shape == (batch, 31, 18)
    assert result.nominal_safe.shape == (batch,)
    assert result.warm_cache_invariant_fault.shape == (batch,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="large nominal shape gate requires CUDA")
@pytest.mark.parametrize("batch", (512, 1024))
def test_nominal_large_fixed_shapes_on_cuda(batch: int) -> None:
    measured = make_state(batch, device="cuda")
    command = make_command(batch, device="cuda")
    phase = torch.arange(batch, device="cuda") % 24
    field = _to_device(_field(), "cuda")
    previous = _seed(measured, phase, initialized=False)
    plan = _plan(measured, command, phase, field, previous.trajectory)

    result = _build(measured, command, field, phase, plan, previous)

    assert result.state.shape == (batch, 31, 18)
    assert torch.isfinite(result.state).all()
