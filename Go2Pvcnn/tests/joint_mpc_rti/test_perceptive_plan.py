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
