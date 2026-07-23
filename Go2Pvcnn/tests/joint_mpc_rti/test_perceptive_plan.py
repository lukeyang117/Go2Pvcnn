from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule

from .helpers import make_command, make_state


def _field(*, small_x: float | None = None, obstacle_id: int = 1):
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.types import JointMpcFieldFrame

    size = 151
    height = torch.zeros(1, size, size)
    semantic = torch.zeros(1, size, size, dtype=torch.long)
    if small_x is not None:
        center = (size - 1) // 2
        index_x = center + int(round(small_x / 0.01))
        height[:, index_x - 1 : index_x + 2, center - 2 : center + 3] = 0.08
        semantic[:, index_x - 1 : index_x + 2, center - 2 : center + 3] = int(obstacle_id)
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


def test_small_cross_candidates_that_intersect_corridor_must_land_after_obstacle() -> None:
    from extension.joint_mpc_rti.model.perceptive_plan import select_touchdowns

    measured = make_state(1)
    plan = select_touchdowns(
        measured,
        make_command(1, vx=0.4),
        fixed_trot_schedule(torch.tensor([0])),
        _warm(1),
        _field(small_x=0.10),
        JointMpcRtiCfg(),
    )

    required = plan.small_cross_required
    assert required.any()
    assert not plan.safe_mask[required & ~plan.small_after_mask].any()
    selected_after = torch.gather(
        plan.small_after_mask, 2, plan.selected_index[..., None]
    )[..., 0]
    assert selected_after[required.any(dim=-1)].all()
    assert plan.selected_sweep_safe[plan.valid].all()


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
        plan.region_plane[..., 0] + cfg.terrain.foot_radius_m,
        atol=1.0e-6,
        rtol=0.0,
    )


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
