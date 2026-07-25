from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule

from .helpers import make_command, make_state


def _field(
    *,
    small_x: float | None = None,
    small_y: float = 0.0,
    obstacle_id: int = 1,
):
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.types import JointMpcFieldFrame

    size = 151
    height = torch.zeros(1, size, size)
    semantic = torch.zeros(1, size, size, dtype=torch.long)
    if small_x is not None:
        center = (size - 1) // 2
        index_x = center + int(round(small_x / 0.01))
        index_y = center + int(round(small_y / 0.01))
        height[:, index_x - 1 : index_x + 2, index_y - 2 : index_y + 3] = 0.08
        semantic[:, index_x - 1 : index_x + 2, index_y - 2 : index_y + 3] = int(obstacle_id)
    frame = JointMpcFieldFrame(
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.zeros(1),
        timestamp=torch.zeros(1),
        refresh_id=torch.zeros(1, dtype=torch.long),
    )
    return build_perceptive_field(
        height,
        semantic,
        torch.ones_like(semantic, dtype=torch.bool),
        frame,
        JointMpcRtiCfg(),
    )


def _warm(batch: int) -> torch.Tensor:
    state = make_state(batch).as_vector()
    return state[:, None].expand(-1, 31, -1).clone()


def test_selector_uses_25_candidates_per_leg_without_four_leg_product() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    batch = 2
    measured = make_state(batch)
    schedule = fixed_trot_schedule(torch.tensor([0, 7]))
    plan = select_touchdowns(
        measured,
        make_command(batch),
        schedule,
        _warm(batch),
        _field(),
        JointMpcRtiCfg(),
    )

    assert plan.candidate_w.shape == (batch, 4, 25, 3)
    assert plan.safe_mask.shape == (batch, 4, 25)
    assert plan.selected_index.shape == (batch, 4)
    assert plan.target_w.shape == (batch, 4, 3)
    assert plan.event_step.shape == (batch, 4)
    assert ((plan.event_step >= 1) & (plan.event_step <= 24)).all()
    assert plan.valid.all()
    assert plan.selected_sweep_safe.all()


def test_touchdown_longitudinal_domain_can_clear_small_obstacle_and_margins() -> None:
    cfg = JointMpcRtiCfg()
    required_span = 0.12 + 2.0 * cfg.touchdown.landing_after_margin_m

    assert cfg.touchdown.small_cross_candidate_extent_m >= required_span
    assert max(cfg.touchdown.candidate_x_m) < cfg.touchdown.small_cross_candidate_extent_m


def test_small_map_expands_only_the_outer_candidate_ring() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    measured = make_state(1)
    command = make_command(1)
    schedule = fixed_trot_schedule(torch.tensor([0]))
    warm = _warm(1)
    cfg = JointMpcRtiCfg()
    flat = select_touchdowns(measured, command, schedule, warm, _field(), cfg)
    small = select_touchdowns(
        measured, command, schedule, warm, _field(small_x=0.10), cfg
    )

    flat_span = flat.candidate_w[..., :2].amax(dim=2) - flat.candidate_w[..., :2].amin(dim=2)
    small_span = small.candidate_w[..., :2].amax(dim=2) - small.candidate_w[..., :2].amin(dim=2)

    torch.testing.assert_close(flat_span, torch.full_like(flat_span, 0.24))
    torch.testing.assert_close(
        small_span[..., 0],
        torch.full_like(
            small_span[..., 0], 2.0 * cfg.touchdown.small_cross_candidate_extent_m
        ),
    )
    torch.testing.assert_close(small_span[..., 1], torch.full_like(small_span[..., 1], 0.24))
    expected_small_x = torch.tensor(
        (
            -cfg.touchdown.small_cross_candidate_extent_m,
            -cfg.touchdown.small_cross_candidate_inner_m,
            0.0,
            cfg.touchdown.small_cross_candidate_inner_m,
            cfg.touchdown.small_cross_candidate_extent_m,
        )
    )
    torch.testing.assert_close(
        torch.unique(small.candidate_w[0, 0, :, 0] - small.candidate_w[0, 0, 12, 0]),
        expected_small_x,
    )


@pytest.mark.parametrize("vx", (-0.2, 0.2))
def test_cold_flat_touchdown_leads_predicted_hip_in_command_direction(
    vx: float,
) -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_collision_geometry
    from extension.joint_mpc_rti.model.nominal import build_rebased_seed
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns
    from extension.joint_mpc_rti.types import JointMpcRtiSolverState

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = make_command(1, vx=vx)
    phase = torch.tensor([0])
    measured_foot = go2_collision_geometry(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_center_w
    previous = JointMpcRtiSolverState(
        trajectory=_warm(1),
        gait_phase=phase,
        initialized=torch.zeros(1, dtype=torch.bool),
        stance_anchor_w=measured_foot,
    )
    warm = build_rebased_seed(measured, command, phase, previous, cfg)
    plan = select_touchdowns(
        measured,
        command,
        fixed_trot_schedule(phase),
        warm,
        _field(),
        cfg,
    )

    root_event = torch.gather(
        warm[..., :2].unsqueeze(2).expand(-1, -1, 4, -1),
        1,
        plan.event_step[:, None, :, None].expand(-1, 1, -1, 2),
    ).squeeze(1)
    command_axis = torch.tensor((1.0 if vx > 0.0 else -1.0, 0.0))
    relative_change = (
        plan.target_w[..., :2]
        - root_event
        - (measured_foot[..., :2] - measured.root_pos_w[:, None, :2])
    )
    lead = (relative_change * command_axis).sum(-1)

    assert (lead >= 0.01).all()


def test_small_cross_candidates_that_intersect_corridor_must_land_after_obstacle() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.types import JointMpcFieldFrame
    from .run_joint_acceptance import build_small_obstacle_field

    measured = make_state(1)
    terrain, _, _ = build_small_obstacle_field(
        commands=torch.tensor([[0.4, 0.0, 0.0]]),
        shapes=("cuboid",),
        offsets=torch.tensor([0.0]),
        obstacle_center_xy_w=torch.tensor([[0.40, 0.0]]),
        device="cpu",
        terrain_cfg=JointMpcRtiCfg().terrain,
    )
    field = build_perceptive_field(
        terrain.height_w,
        terrain.semantic_id,
        terrain.valid_mask,
        JointMpcFieldFrame(
            origin_w=terrain.origin_w,
            yaw_w=terrain.yaw_w,
            timestamp=terrain.timestamp,
            refresh_id=terrain.version,
        ),
        JointMpcRtiCfg(),
    )
    plan = select_touchdowns(
        measured,
        make_command(1, vx=0.4),
        fixed_trot_schedule(torch.tensor([0])),
        _warm(1),
        field,
        JointMpcRtiCfg(),
        terrain_field=terrain,
    )

    required = plan.small_cross_required
    assert required.any()
    assert not plan.safe_mask[required & ~plan.small_after_mask].any()
    selected_after = torch.gather(
        plan.small_after_mask, 2, plan.selected_index[..., None]
    )[..., 0]
    assert selected_after[required.any(dim=-1)].all()
    assert plan.selected_sweep_safe[plan.valid].all()
    assert "sweep_joint_rate" in plan.valid_components
    assert plan.valid_components["sweep_joint_rate"][plan.safe_mask].all()
    for name in (
        "sweep_joint_linear_foot",
        "sweep_joint_linear_knee",
        "sweep_joint_linear_calf",
        "sweep_joint_linear_thigh",
    ):
        assert name in plan.valid_components
        assert plan.valid_components[name][plan.safe_mask].all()


def test_candidate_support_rejects_future_stance_thigh_collision() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import (
        go2_collision_geometry,
    )
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _candidate_support_safe,
    )

    cfg = JointMpcRtiCfg()
    warm = _warm(1)
    geometry = go2_collision_geometry(
        warm[:, 0, :3], warm[:, 0, 3:6], warm[:, 0, 6:]
    )
    candidate = geometry.foot_center_w[:, :, None].expand(-1, -1, 25, -1).clone()
    field = _field()

    # Keep the touchdown cell landable while placing collision height under the
    # middle of FL's future support thigh.
    thigh_midpoint = geometry.thigh_endpoints_w[0, 0].mean(dim=0)
    center = (field.inflated_height_w.shape[-1] - 1) // 2
    index_x = center + int(round(float(thigh_midpoint[0]) / field.resolution))
    index_y = center + int(round(float(thigh_midpoint[1]) / field.resolution))
    inflated = field.inflated_height_w.clone()
    inflated[:, 3, index_x - 1 : index_x + 2, index_y - 1 : index_y + 2] = 0.30
    field = replace(field, inflated_height_w=inflated)

    safe, components = _candidate_support_safe(
        warm,
        candidate,
        torch.ones((1, 4), dtype=torch.long),
        field,
        cfg,
    )

    assert not components["support_thigh"][0, 0, 0]
    assert not safe[0, 0, 0]


def test_small_sdf_guides_swing_bump_over_obstacle_without_moving_endpoints() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _small_crossing_offsets,
    )
    from extension.joint_mpc_rti.model.swing_profile import swing_xy_profile
    from extension.joint_mpc_rti.terrain.query import query_world
    from .run_joint_acceptance import build_small_obstacle_field

    cfg = JointMpcRtiCfg()
    terrain, _, _ = build_small_obstacle_field(
        commands=torch.tensor([[0.2, 0.0, 0.0]]),
        shapes=("cuboid",),
        offsets=torch.tensor([0.0]),
        obstacle_center_xy_w=torch.tensor([[0.40, 0.0]]),
        device="cpu",
        terrain_cfg=cfg.terrain,
    )
    lift = torch.tensor([[[0.20, 0.14]]])
    touchdown = torch.tensor([[[[0.52, 0.14]]]])
    axis = torch.tensor([[[1.0, 0.0]]])
    offset, opportunity, obstacle_in, obstacle_out = _small_crossing_offsets(
        lift, touchdown, axis, terrain, cfg
    )
    tau = torch.linspace(0.0, 1.0, 65).view(1, 1, 1, 65, 1)
    path = swing_xy_profile(
        lift[:, :, None, None],
        touchdown[..., None, :],
        axis[:, :, None, None],
        tau,
        crossing=opportunity[..., None, None],
        outward=offset[..., None, :],
        cfg=cfg,
    )
    distance = query_world(terrain, path.reshape(1, -1, 2)).small_distance_m

    assert opportunity.item()
    assert torch.isfinite(obstacle_in).item()
    assert offset[0, 0, 0, 1] < 0.0
    torch.testing.assert_close(path[..., 0, :], lift[:, :, None], atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(
        path[..., -1, :], touchdown, atol=1.0e-7, rtol=0.0
    )
    assert distance.amin() >= 0.0
    assert distance.amin() <= (
        cfg.touchdown.small_cross_foot_overlap_fraction
        * cfg.terrain.foot_radius_m
        + 1.0e-4
    )
    assert obstacle_in.item() <= 0.14
    assert obstacle_out.item() >= 0.26


def test_committed_small_staging_is_revalidated_after_current_map_recentering() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _small_crossing_offsets,
        _small_staging_mask,
    )
    from .run_joint_acceptance import build_small_obstacle_field

    cfg = JointMpcRtiCfg()
    command = torch.tensor([[0.2, 0.0, 0.0]])
    obstacle_center = torch.tensor([[0.40, 0.0]])
    lift = torch.tensor([[[0.2434434, -0.1413063]]])
    touchdown = torch.tensor([[[[0.4842444, -0.20]]]])
    axis = torch.tensor([[[1.0, 1.1810415e-5]]])
    staging_rows = []

    for refresh, origin_x in enumerate((0.1500417, 0.1534417)):
        terrain, _, _ = build_small_obstacle_field(
            commands=command,
            shapes=("cuboid",),
            offsets=torch.tensor([0.0]),
            origin_xy_w=torch.tensor([[origin_x, 0.0]]),
            obstacle_center_xy_w=obstacle_center,
            device="cpu",
            terrain_cfg=cfg.terrain,
        )
        _, opportunity, obstacle_in, obstacle_out = _small_crossing_offsets(
            lift, touchdown, axis, terrain, cfg
        )
        progress = ((touchdown - lift[:, :, None]) * axis[:, :, None]).sum(dim=-1)
        staging_rows.append(
            _small_staging_mask(
                progress,
                obstacle_in,
                obstacle_out,
                before_margin_m=cfg.touchdown.landing_before_margin_m,
                after_margin_m=cfg.touchdown.landing_after_margin_m,
                continued_candidate=torch.full_like(
                    progress, refresh > 0, dtype=torch.bool
                ),
            )
        )
        assert opportunity.item()

    assert torch.equal(
        torch.stack(staging_rows).reshape(-1), torch.tensor((True, False))
    )


def test_diagonal_crossing_uses_swing_corridor_for_after_obstacle() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import _crossing_after_mask

    lift_xy = torch.tensor([[[0.0, 0.13]]])
    candidate_xy = torch.tensor([[[[0.50, -0.05], [0.41, -0.05]]]])
    command_axis = torch.tensor([[[1.0, 0.0]]])
    corridor_xy = torch.tensor(
        [[[[[0.0, 0.13], [0.30, 0.0], [0.40, 0.0], [0.50, -0.05]],
           [[0.0, 0.13], [0.30, 0.0], [0.40, 0.0], [0.41, -0.05]]]]]
    )
    small_corridor = torch.tensor([[[[False, True, True, False], [False, True, True, False]]]])
    crossing = small_corridor.any(dim=-1)

    after = _crossing_after_mask(
        lift_xy,
        candidate_xy,
        command_axis,
        corridor_xy,
        small_corridor,
        crossing,
        torch.zeros_like(crossing),
        margin_m=0.025,
    )

    assert after.tolist() == [[[True, False]]]


def test_sdf_footprint_overlap_can_establish_after_without_center_occupancy() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import _crossing_after_mask

    lift_xy = torch.tensor([[[0.20, 0.14]]])
    candidate_xy = torch.tensor([[[[0.50, 0.14]]]])
    command_axis = torch.tensor([[[1.0, 0.0]]])
    corridor_xy = torch.tensor([[[[[0.20, 0.14], [0.40, 0.08], [0.50, 0.14]]]]])
    small_corridor = torch.zeros((1, 1, 1, 3), dtype=torch.bool)
    crossing = torch.ones((1, 1, 1), dtype=torch.bool)

    after = _crossing_after_mask(
        lift_xy,
        candidate_xy,
        command_axis,
        corridor_xy,
        small_corridor,
        crossing,
        torch.zeros_like(crossing),
        margin_m=0.025,
        sdf_obstacle_out=torch.tensor([[[0.26]]]),
    )

    assert after.item()


def test_small_approach_grid_contains_a_selected_lateral_lane_hold() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.types import JointMpcFieldFrame
    from .run_joint_acceptance import build_small_obstacle_field

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    command = torch.tensor([[0.2, 0.0, 0.0]])
    terrain, _, _ = build_small_obstacle_field(
        commands=command,
        shapes=("cuboid",),
        offsets=torch.tensor([0.16]),
        origin_xy_w=measured.root_pos_w[:, :2],
        device="cpu",
        terrain_cfg=cfg.terrain,
    )
    field = build_perceptive_field(
        terrain.height_w,
        terrain.semantic_id,
        terrain.valid_mask,
        JointMpcFieldFrame(
            origin_w=terrain.origin_w,
            yaw_w=terrain.yaw_w,
            timestamp=terrain.timestamp,
            refresh_id=terrain.version,
        ),
        cfg,
    )
    plan = select_touchdowns(
        measured,
        command,
        fixed_trot_schedule(torch.zeros(1, dtype=torch.long)),
        _warm(1),
        field,
        cfg,
    )
    measured_foot = go2_fk(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    ).foot_pos_w

    torch.testing.assert_close(
        plan.target_w[:, 0, 1], measured_foot[:, 0, 1], atol=1.0e-5, rtol=0.0
    )


def test_small_candidate_longitudinal_rows_share_the_warm_foot_lane() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    measured = make_state(1)
    warm = _warm(1)
    plan = select_touchdowns(
        measured,
        make_command(1, vx=0.2),
        fixed_trot_schedule(torch.tensor([0])),
        warm,
        _field(small_x=0.10),
        JointMpcRtiCfg(),
    )
    warm_foot = go2_fk(
        warm[..., :3], warm[..., 3:6], warm[..., 6:]
    ).foot_pos_w
    event = plan.event_step
    event_foot = torch.gather(
        warm_foot,
        1,
        event[:, None, :, None].expand(-1, 1, -1, 3),
    ).squeeze(1)
    zero_lateral_slots = torch.tensor((2, 7, 12, 17, 22))

    torch.testing.assert_close(
        plan.candidate_w[:, :, zero_lateral_slots, 1],
        event_foot[..., 1, None].expand(-1, -1, 5),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_small_candidate_center_uses_predicted_hip_command_progress() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import HIP_OFFSETS
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    measured = make_state(1)
    warm = _warm(1)
    plan = select_touchdowns(
        measured,
        make_command(1, vx=0.2),
        fixed_trot_schedule(torch.tensor([0])),
        warm,
        _field(small_x=0.10),
        JointMpcRtiCfg(),
    )
    event_root_x = torch.gather(
        warm[..., 0], 1, plan.event_step
    )
    expected_hip_x = event_root_x + torch.tensor(HIP_OFFSETS)[:, 0]

    torch.testing.assert_close(
        plan.candidate_w[:, :, 12, 0],
        expected_hip_x,
        atol=1.0e-5,
        rtol=0.0,
    )


def test_selector_rerun_is_warm_and_latched_safe_target_does_not_drift() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    measured = make_state(1)
    schedule = fixed_trot_schedule(torch.tensor([6]))
    first = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        _warm(1),
        _field(),
        JointMpcRtiCfg(),
    )
    second = select_touchdowns(
        measured,
        make_command(1, vx=0.21),
        schedule,
        _warm(1),
        _field(),
        JointMpcRtiCfg(),
        previous_plan=first,
    )

    keep = first.latched & first.valid
    assert keep.any()
    torch.testing.assert_close(second.target_w[keep], first.target_w[keep])


def test_solver_state_crossing_commitment_survives_current_swing() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    measured = make_state(1)
    cfg = JointMpcRtiCfg()
    field = _field()
    warm = _warm(1)
    first = select_touchdowns(
        measured,
        make_command(1, vx=0.4),
        fixed_trot_schedule(torch.tensor([0])),
        warm,
        field,
        cfg,
    )
    first_crossing = torch.zeros_like(first.selected_index, dtype=torch.bool)
    first_crossing[:, 0] = True
    first_offset = torch.zeros_like(first.target_w[..., :2])
    first_offset[:, 0, 1] = -0.06

    second = select_touchdowns(
        measured,
        make_command(1, vx=0.4),
        fixed_trot_schedule(torch.tensor([1])),
        warm,
        field,
        cfg,
        previous_target_w=first.target_w,
        previous_selected_index=first.selected_index,
        previous_crossing=first_crossing,
        previous_swing_offset_w=first_offset,
        previous_lift_w=go2_fk(
            measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
        ).foot_pos_w,
    )
    second_crossing = torch.gather(
        second.small_cross_required, 2, second.selected_index[..., None]
    ).squeeze(-1)

    torch.testing.assert_close(
        second.target_w[first_crossing], first.target_w[first_crossing]
    )
    assert second_crossing[first_crossing].all()
    torch.testing.assert_close(
        second.selected_swing_offset_w[first_crossing],
        first_offset[first_crossing],
    )


def test_future_crossing_commitment_survives_stance_until_its_touchdown() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    measured = make_state(1)
    cfg = JointMpcRtiCfg()
    phase = torch.tensor([13])
    schedule = fixed_trot_schedule(phase)
    first = select_touchdowns(
        measured,
        make_command(1, vx=0.2),
        schedule,
        _warm(1),
        _field(),
        cfg,
    )
    crossing = torch.zeros_like(first.selected_index, dtype=torch.bool)
    crossing[:, 0] = True

    second = select_touchdowns(
        measured,
        make_command(1, vx=0.2),
        schedule,
        _warm(1),
        _field(),
        cfg,
        previous_target_w=first.target_w,
        previous_selected_index=first.selected_index,
        previous_crossing=crossing,
        previous_remaining_steps=torch.full_like(first.selected_index, 23),
        previous_lift_w=go2_fk(
            measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
        ).foot_pos_w,
    )
    selected_crossing = torch.gather(
        second.small_cross_required, 2, second.selected_index[..., None]
    ).squeeze(-1)

    torch.testing.assert_close(second.target_w[crossing], first.target_w[crossing])
    assert selected_crossing[crossing].all()
    assert torch.gather(
        second.valid_components["sweep"],
        2,
        first.selected_index[..., None],
    ).squeeze(-1)[crossing].all()


def test_future_stance_crossing_does_not_resume_at_a_swing_phase_above_one() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _crossing_sweep_continuation,
    )

    phase0 = torch.tensor([[3, 15, 15, 3]])
    continued = torch.ones((1, 4, 25), dtype=torch.bool)

    mask, start_tau = _crossing_sweep_continuation(
        phase0,
        continued,
        swing_steps=12,
        dtype=torch.float32,
    )

    assert torch.equal(mask[:, :, 0], torch.tensor([[True, False, False, True]]))
    torch.testing.assert_close(
        start_tau,
        torch.tensor([[0.25, 0.0, 0.0, 0.25]]),
    )


def test_small_staging_band_accepts_only_before_or_after_touchdowns() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import _small_staging_mask

    safe = _small_staging_mask(
        candidate_progress=torch.tensor((0.15, 0.20, 0.35)),
        obstacle_in=torch.tensor((0.20, 0.20, 0.20)),
        obstacle_out=torch.tensor((0.32, 0.32, 0.32)),
        before_margin_m=0.025,
        after_margin_m=0.025,
    )

    assert torch.equal(safe, torch.tensor((True, False, True)))


def test_continued_crossing_expands_bounded_outward_retarget_candidates() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _continuation_retarget_candidates,
    )

    cfg = JointMpcRtiCfg()
    candidate = torch.zeros((1, 4, 25, 2))
    prior_target = torch.tensor(
        [[[0.40, 0.16], [0.40, -0.16], [0.00, 0.16], [0.00, -0.16]]]
    )
    prior_index = torch.tensor([[7, 17, 7, 17]])
    continued = torch.tensor([[False, True, False, False]])

    retargeted, injected, exact = _continuation_retarget_candidates(
        candidate,
        prior_target,
        prior_index,
        continued,
        current_swing=torch.zeros((1, 4), dtype=torch.bool),
        event_yaw=torch.zeros((1, 4)),
        cfg=cfg,
    )

    slots = torch.remainder(prior_index[0, 1] + torch.arange(5), 25)
    expected_y = prior_target[0, 1, 1] - torch.tensor(
        cfg.touchdown.continuation_outward_retarget_m
    )
    torch.testing.assert_close(retargeted[0, 1, slots, 0], torch.full((5,), 0.40))
    torch.testing.assert_close(retargeted[0, 1, slots, 1], expected_y)
    assert injected[0, 1, slots].all()
    assert exact[0, 1, prior_index[0, 1]]
    assert injected.sum().item() == 5


def test_continued_candidate_still_requires_current_map_after_margin() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import _crossing_after_mask

    after = _crossing_after_mask(
        lift_xy=torch.tensor([[[0.0, 0.0]]]),
        candidate_xy=torch.tensor([[[[0.15, 0.0]]]]),
        command_axis=torch.tensor([[[1.0, 0.0]]]),
        corridor_xy=torch.tensor([[[[[0.0, 0.0], [0.10, 0.0], [0.20, 0.0]]]]]),
        small_corridor=torch.tensor([[[[False, True, True]]]]),
        crossing_required=torch.ones((1, 1, 1), dtype=torch.bool),
        continued_candidate=torch.ones((1, 1, 1), dtype=torch.bool),
        margin_m=0.025,
    )

    assert not after.item()


def test_preview_sweep_starts_at_future_liftoff_with_primary_foot_anchored() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _preview_sweep_pose_path,
        _preview_sweep_start,
    )

    cfg = JointMpcRtiCfg()
    warm = _warm(1)
    warm[..., 0] += 0.004 * torch.arange(31)
    preview_step = torch.tensor([[-1, 37, 37, -1]])
    liftoff = (preview_step - cfg.gait.swing_steps).clamp(0, 30)
    batch = torch.arange(1)[:, None]
    leg = torch.arange(4)[None]
    liftoff_state = warm[batch, liftoff]
    liftoff_geometry = go2_fk(
        liftoff_state[..., :3],
        liftoff_state[..., 3:6],
        liftoff_state[..., 6:],
    )
    primary_target = liftoff_geometry.foot_pos_w[batch, leg, leg]

    root, rpy, joint = _preview_sweep_start(
        warm, primary_target, preview_step, cfg
    )
    root_path, _ = _preview_sweep_pose_path(warm, preview_step, cfg)
    geometry = go2_fk(root, rpy, joint)
    anchored = geometry.foot_pos_w[batch, leg, leg]

    torch.testing.assert_close(root[:, 1], warm[:, 25, :3])
    torch.testing.assert_close(root[:, 2], warm[:, 25, :3])
    torch.testing.assert_close(anchored, primary_target, atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(root_path[:, 1, :6], warm[:, 25:31, :3])
    torch.testing.assert_close(
        root_path[:, 1, 6:], warm[:, 30:31, :3].expand(-1, 7, -1)
    )


def test_safe_crossing_commitment_excludes_other_crossing_candidates() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _committed_crossing_selection,
    )

    safe_cross = torch.tensor([[[True, True, False]]])
    ordinary = torch.tensor([[[False, False, True]]])
    selected_cross_leg = torch.tensor([[True]])
    continued_crossing = torch.tensor([[True]])
    continued_candidate = torch.tensor([[[False, True, False]]])

    selection = _committed_crossing_selection(
        safe_cross,
        ordinary,
        selected_cross_leg,
        continued_crossing,
        continued_candidate,
    )

    assert torch.equal(selection, continued_candidate)


def test_current_swing_uses_post_obstacle_candidates_when_one_safe_before_remains() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _prefer_current_swing_post_obstacle,
    )

    base = torch.ones((1, 2, 3), dtype=torch.bool)
    progress = torch.tensor(
        [[[0.10, 0.15, 0.50], [0.10, 0.30, 0.50]]]
    )
    obstacle_in = torch.full_like(progress, 0.20)
    obstacle_out = torch.full_like(progress, 0.35)

    selection = _prefer_current_swing_post_obstacle(
        base,
        base,
        progress,
        obstacle_in,
        obstacle_out,
        current_swing=torch.ones((1, 2), dtype=torch.bool),
        selected_cross_leg=torch.zeros((1, 2), dtype=torch.bool),
        continued_crossing=torch.zeros((1, 2), dtype=torch.bool),
        before_margin_m=0.025,
        after_margin_m=0.025,
    )

    assert torch.equal(selection[0, 0], torch.tensor([True, True, True]))
    assert torch.equal(selection[0, 1], torch.tensor([False, False, True]))


def test_only_current_swing_post_obstacle_target_gets_ordinary_continuation() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import (
        _post_obstacle_continuation,
    )

    continued = _post_obstacle_continuation(
        current_swing=torch.tensor([[True, True, False, True]]),
        candidate_progress=torch.tensor([[0.40, 0.20, 0.40, 0.40]]),
        obstacle_out=torch.tensor([[0.35, 0.35, 0.35, torch.inf]]),
        margin_m=0.025,
    )

    assert torch.equal(continued, torch.tensor([[True, False, False, False]]))


def test_large_obstacle_corridor_candidates_are_hard_invalid() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    plan = select_touchdowns(
        make_state(1),
        make_command(1, vx=0.4),
        fixed_trot_schedule(torch.tensor([0])),
        _warm(1),
        _field(small_x=0.10, obstacle_id=2),
        JointMpcRtiCfg(),
    )
    corridor_safe = plan.valid_components["corridor"]

    assert (~corridor_safe).any()
    assert not plan.safe_mask[~corridor_safe].any()


def test_event_preview_keeps_touchdown_after_h30_when_horizon_ends_in_swing() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import touchdown_event_steps

    schedule = fixed_trot_schedule(torch.arange(24))
    first, preview = touchdown_event_steps(schedule)

    assert ((first >= 1) & (first <= 24)).all()
    assert ((preview == -1) | ((preview > 30) & (preview <= 42))).all()
    assert (preview > 30).any()


def test_preview_touchdown_runs_full_candidate_region_and_sweep_selection() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    plan = select_touchdowns(
        make_state(1),
        make_command(1, vx=0.3),
        fixed_trot_schedule(torch.tensor([6])),
        _warm(1),
        _field(),
        JointMpcRtiCfg(),
    )

    assert plan.preview_candidate_w.shape == (1, 4, 25, 3)
    assert plan.preview_safe_mask.shape == (1, 4, 25)
    assert plan.preview_ranked_index.shape == (1, 4, 25)
    assert plan.preview_target_w.shape == (1, 4, 3)
    assert plan.preview_region.A.shape == (1, 4, 4, 2)
    containment = torch.einsum(
        "blij,blj->bli", plan.preview_region.A, plan.preview_target_w[..., :2]
    ) + plan.preview_region.b
    assert (containment >= -1.0e-6).all()
    assert plan.preview_valid.all()
    assert plan.preview_selected_sweep_safe.all()


def test_selected_touchdown_has_a_containing_safe_region_and_local_plane() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns
    from extension.joint_mpc_rti.terrain.query import query_perceptive_world

    cfg = JointMpcRtiCfg()
    field = _field()
    plan = select_touchdowns(
        make_state(1),
        make_command(1),
        fixed_trot_schedule(torch.tensor([0])),
        _warm(1),
        field,
        cfg,
    )

    assert plan.region_A.shape == (1, 4, 4, 2)
    assert plan.region_b.shape == (1, 4, 4)
    assert plan.region_half_extent.shape == (1, 4, 4)
    assert plan.region_corners_w.shape == (1, 4, 4, 3)
    assert plan.region_plane.shape == (1, 4, 3)
    assert plan.region_normal_w.shape == (1, 4, 3)
    containment = torch.einsum("blij,blj->bli", plan.region_A, plan.target_w[..., :2])
    assert torch.all(containment + plan.region_b >= -1.0e-6)
    assert plan.region_valid.all()
    assert torch.all(plan.region_half_extent >= cfg.region.min_half_extent_m)
    assert torch.all(plan.region_area > 0.0)
    assert torch.all(plan.region_plane_residual <= cfg.region.max_plane_residual_m)
    torch.testing.assert_close(
        torch.linalg.vector_norm(plan.region_normal_w, dim=-1),
        torch.ones_like(plan.region_area),
    )

    corner_query = query_perceptive_world(field, plan.region_corners_w.reshape(1, 16, 3))
    assert corner_query.valid.all()
    assert corner_query.landing_safe.all()
    torch.testing.assert_close(
        plan.target_w[..., 2],
        plan.region_plane[..., 0] + cfg.gait.foot_contact_offset,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_every_retry_candidate_keeps_its_maximal_safe_region() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    cfg = JointMpcRtiCfg()
    plan = select_touchdowns(
        make_state(1),
        make_command(1),
        fixed_trot_schedule(torch.tensor([0])),
        _warm(1),
        _field(),
        cfg,
    )

    expected = cfg.region.cap_m - cfg.region.margin_m - 1.0e-6
    assert (plan.candidate_region.half_extent[plan.safe_mask] >= expected).all()


def test_region_never_expands_through_forbidden_cells_and_invalid_candidates_fallback() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns
    from extension.joint_mpc_rti.terrain.query import query_perceptive_world

    cfg = JointMpcRtiCfg()
    field = _field(small_x=0.10)
    plan = select_touchdowns(
        make_state(1),
        make_command(1, vx=0.4),
        fixed_trot_schedule(torch.tensor([0])),
        _warm(1),
        field,
        cfg,
    )

    assert "region" in plan.valid_components
    assert torch.equal(plan.safe_mask, plan.valid_components["pre_region"] & plan.valid_components["region"])
    selected_region_valid = torch.gather(
        plan.valid_components["region"], 2, plan.selected_index[..., None]
    )[..., 0]
    assert selected_region_valid[plan.valid].all()
    assert plan.region_valid[plan.valid].all()

    # Sampling the complete rectangle catches corner-cutting that four axial rays miss.
    u = torch.linspace(0.0, 1.0, 7)
    s = u.view(1, 1, 7, 1, 1)
    t = u.view(1, 1, 1, 7, 1)
    c0, c1, c2, c3 = plan.region_corners_w.unbind(dim=2)
    samples = (
        (1.0 - s) * ((1.0 - t) * c0[:, :, None, None] + t * c1[:, :, None, None])
        + s * ((1.0 - t) * c3[:, :, None, None] + t * c2[:, :, None, None])
    )
    samples = samples.reshape(1, 4 * 49, 3)
    query = query_perceptive_world(field, samples)
    valid_samples = query.valid.reshape(1, 4, 49)
    safe_samples = query.landing_safe.reshape(1, 4, 49)
    assert valid_samples[plan.region_valid].all()
    assert safe_samples[plan.region_valid].all()
    sole_offset = torch.tensor(
        ((0.03, 0.02), (0.03, -0.02), (-0.03, 0.02), (-0.03, -0.02))
    )
    sole_points = samples[..., None, :2] + sole_offset
    sole_query = query_perceptive_world(field, sole_points.reshape(1, 4 * 49 * 4, 2))
    assert not sole_query.small_mask.reshape(1, 4, 49, 4)[plan.region_valid].any()
    assert not sole_query.large_mask.reshape(1, 4, 49, 4)[plan.region_valid].any()
    assert not sole_query.unknown_mask.reshape(1, 4, 49, 4)[plan.region_valid].any()


def test_region_builder_selects_the_maximum_safe_rectangle_containing_center() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import _build_candidate_regions

    field = _field()
    landing_safe = field.landing_safe.clone()
    center = (landing_safe.shape[-1] - 1) // 2
    landing_safe[:, center + 2, center + 5] = False
    field = replace(field, landing_safe=landing_safe)
    candidate = torch.tensor((0.0, 0.0, 0.022)).view(1, 1, 1, 3)

    region = _build_candidate_regions(
        candidate,
        torch.zeros(1, 1),
        field,
        JointMpcRtiCfg(),
    )

    assert region.valid.item()
    assert region.half_extent[..., 1].item() == pytest.approx(0.055)
    assert region.half_extent[..., 3].item() == pytest.approx(0.035)


def test_region_builder_rejects_forbidden_cell_on_the_center_cross_section() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import _build_candidate_regions

    field = _field()
    landing_safe = field.landing_safe.clone()
    center = (landing_safe.shape[-1] - 1) // 2
    landing_safe[:, center + 2, center] = False
    field = replace(field, landing_safe=landing_safe)

    region = _build_candidate_regions(
        torch.tensor((0.0, 0.0, 0.022)).view(1, 1, 1, 3),
        torch.zeros(1, 1),
        field,
        JointMpcRtiCfg(),
    )

    assert not region.valid.item()


def test_local_plane_is_constrained_to_the_current_height_at_region_center() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import _build_candidate_regions
    from extension.joint_mpc_rti.terrain.query import query_landing_region_world

    field = _field()
    center = (field.height_w.shape[-1] - 1) // 2
    coordinate = torch.arange(field.height_w.shape[-1], dtype=field.height_w.dtype) - center
    curved_height = 0.008 * (coordinate / 6.0).square()
    height = curved_height.view(1, -1, 1).expand_as(field.height_w).clone()
    field = replace(field, height_w=height)
    candidate = torch.tensor((0.0, 0.0, 0.022)).view(1, 1, 1, 3)

    region = _build_candidate_regions(
        candidate,
        torch.zeros(1, 1),
        field,
        JointMpcRtiCfg(),
    )
    center_height = query_landing_region_world(field, candidate[..., :2].reshape(1, 1, 2)).height_w

    torch.testing.assert_close(region.plane[..., 0], center_height.reshape(1, 1, 1))


def test_best_center_with_no_minimum_region_falls_back_to_next_ranked_candidate() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    schedule = fixed_trot_schedule(torch.tensor([0]))
    flat_field = _field()
    flat = replace(
        flat_field,
        inflated_height_w=flat_field.inflated_height_w - 0.20,
    )
    first = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        _warm(1),
        flat,
        cfg,
    )
    landing_safe = flat.landing_safe.clone()
    center = (landing_safe.shape[-1] - 1) // 2
    for target in first.target_w[0]:
        index_x = int(torch.floor(target[0] / flat.resolution + center))
        index_y = int(torch.floor(target[1] / flat.resolution + center))
        landing_safe[:, index_x - 1 : index_x + 3, index_y - 1 : index_y + 3] = False
        landing_safe[:, index_x : index_x + 2, index_y : index_y + 2] = True
    pinched = replace(flat, landing_safe=landing_safe)

    second = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        _warm(1),
        pinched,
        cfg,
    )
    first_pre_region = torch.gather(
        second.valid_components["pre_region"], 2, first.selected_index[..., None]
    )[..., 0]
    first_region = torch.gather(
        second.valid_components["region"], 2, first.selected_index[..., None]
    )[..., 0]

    assert first_pre_region.all()
    assert not first_region.any()
    assert torch.all(second.selected_index != first.selected_index)
    assert second.region_valid.all()


def test_latched_safe_target_stays_while_current_region_is_rebuilt() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    schedule = fixed_trot_schedule(torch.tensor([6]))
    flat_field = _field()
    flat = replace(
        flat_field,
        inflated_height_w=flat_field.inflated_height_w - 0.20,
    )
    first = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        _warm(1),
        flat,
        cfg,
    )
    landing_safe = flat.landing_safe.clone()
    center = (landing_safe.shape[-1] - 1) // 2
    for target in first.target_w[0]:
        index_x = center + int(round(float(target[0]) / flat.resolution))
        index_y = center + int(round(float(target[1]) / flat.resolution))
        landing_safe[:, index_x + 4, index_y + 4] = False
    changed = replace(flat, landing_safe=landing_safe)

    second = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        _warm(1),
        changed,
        cfg,
        previous_plan=first,
    )

    assert first.latched.any()
    torch.testing.assert_close(second.target_w[first.latched], first.target_w[first.latched])
    assert torch.any(
        second.region_half_extent[first.latched] != first.region_half_extent[first.latched]
    )
    assert second.region_valid[first.latched].all()


def test_latched_target_reselects_when_current_map_makes_its_center_unsafe() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    schedule = fixed_trot_schedule(torch.tensor([6]))
    flat_field = _field()
    flat = replace(
        flat_field,
        inflated_height_w=flat_field.inflated_height_w - 0.20,
    )
    first = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        _warm(1),
        flat,
        cfg,
    )
    landing_safe = flat.landing_safe.clone()
    center = (landing_safe.shape[-1] - 1) // 2
    for target in first.target_w[0, first.latched[0]]:
        index_x = int(torch.floor(target[0] / flat.resolution + center))
        index_y = int(torch.floor(target[1] / flat.resolution + center))
        landing_safe[:, index_x : index_x + 2, index_y : index_y + 2] = False
    changed = replace(flat, landing_safe=landing_safe)

    second = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        _warm(1),
        changed,
        cfg,
        previous_plan=first,
    )

    assert torch.all(second.selected_index[first.latched] != first.selected_index[first.latched])
    assert second.valid[first.latched].all()


def test_latched_world_target_survives_warm_hip_grid_translation() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    cfg = JointMpcRtiCfg()
    measured = make_state(1)
    schedule = fixed_trot_schedule(torch.tensor([6]))
    flat_field = _field()
    flat = replace(
        flat_field,
        inflated_height_w=flat_field.inflated_height_w - 0.20,
    )
    warm = _warm(1)
    first = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        warm,
        flat,
        cfg,
    )
    translated_warm = warm.clone()
    translated_warm[..., 0] += 0.01

    second = select_touchdowns(
        measured,
        make_command(1),
        schedule,
        translated_warm,
        flat,
        cfg,
        previous_plan=first,
    )

    torch.testing.assert_close(second.target_w[first.latched], first.target_w[first.latched])
    assert second.region_valid[first.latched].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph test requires CUDA")
def test_region_plane_solve_captures_and_replays_on_cuda() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    field = _field()
    field = replace(
        field,
        **{
            item.name: getattr(field, item.name).cuda()
            for item in fields(field)
            if isinstance(getattr(field, item.name), torch.Tensor)
        },
    )
    measured = make_state(1, device="cuda")
    command = make_command(1, device="cuda")
    schedule = fixed_trot_schedule(torch.full((1,), 6, dtype=torch.long, device="cuda"))
    warm = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    first = select_touchdowns(measured, command, schedule, warm, field, JointMpcRtiCfg())
    select_touchdowns(
        measured,
        command,
        schedule,
        warm,
        field,
        JointMpcRtiCfg(),
        previous_plan=first,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = select_touchdowns(
            measured,
            command,
            schedule,
            warm,
            field,
            JointMpcRtiCfg(),
            previous_plan=first,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(captured.region_plane).all()
    assert captured.region_valid.all()


@pytest.mark.parametrize("batch", (1, 40, 512, 1024))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="large selector shape gate requires CUDA")
def test_selector_supports_all_frozen_batch_shapes_on_cuda(batch: int) -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    field = _field()
    field = replace(
        field,
        **{
            item.name: getattr(field, item.name).cuda()
            for item in fields(field)
            if isinstance(getattr(field, item.name), torch.Tensor)
        },
    )
    measured = make_state(batch, device="cuda")
    warm = measured.as_vector()[:, None].expand(-1, 31, -1).clone()
    plan = select_touchdowns(
        measured,
        make_command(batch, device="cuda"),
        fixed_trot_schedule(torch.arange(batch, device="cuda") % 24),
        warm,
        field,
        JointMpcRtiCfg(),
    )

    assert plan.candidate_w.shape == (batch, 4, 25, 3)
    assert plan.safe_mask.shape == (batch, 4, 25)
    assert plan.valid.shape == (batch, 4)
    assert plan.valid.all()
