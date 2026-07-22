from __future__ import annotations

import inspect
from dataclasses import replace

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.command import command_residual
from extension.joint_mpc_rti.losses.contact import contact_residual
from extension.joint_mpc_rti.losses import objective as objective_module
from extension.joint_mpc_rti.losses.objective import LossContext, total_trajectory_loss, trajectory_loss_breakdown
from extension.joint_mpc_rti.losses.step import step_residual
from extension.joint_mpc_rti.losses.smoothness import smooth_loss
from extension.joint_mpc_rti.losses import swing_speed as swing_speed_module
from extension.joint_mpc_rti.losses.swing_speed import swing_speed_penalty, swing_speed_residual
from extension.joint_mpc_rti.losses import terrain as terrain_module
from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.terrain.field_builder import build_field_batch


def _state(batch: int = 2) -> torch.Tensor:
    state = torch.zeros(batch, 31, 18)
    state[..., 2] = 0.34
    state[..., 6:] = torch.tensor((0.0, 0.8, -1.5) * 4)
    return state


def _flat_field(batch: int):
    return build_field_batch(
        height_w=torch.zeros(batch, 51, 51),
        semantic_id=torch.zeros(batch, 51, 51, dtype=torch.long),
        origin_w=torch.zeros(batch, 3),
        yaw_w=torch.zeros(batch),
        timestamp=torch.zeros(batch),
        version=torch.ones(batch, dtype=torch.long),
        resolution=0.02,
        small_ids=(1,),
        large_ids=(2,),
        terrain_cfg=JointMpcRtiCfg().terrain,
    )


def _context(state: torch.Tensor) -> LossContext:
    batch = state.shape[0]
    schedule = fixed_trot_schedule(torch.arange(batch) % 24)
    foot = go2_fk(state[..., :3], state[..., 3:6], state[..., 6:]).foot_pos_w
    return LossContext(
        command_body=torch.zeros(batch, 3),
        touchdown_reference_w=foot.detach().clone(),
        schedule=schedule,
        terrain=_flat_field(batch),
        stance_anchor_w=foot.detach().clone(),
        support_height=torch.zeros(batch, 31),
    )


def test_objective_has_exactly_seven_losses() -> None:
    state = _state()

    breakdown = trajectory_loss_breakdown(state, _context(state), JointMpcRtiCfg())

    assert tuple(breakdown) == (
        "command",
        "step",
        "contact",
        "swing_speed",
        "terrain",
        "posture",
        "smooth",
    )
    assert all(value.shape == (2,) for value in breakdown.values())
    assert all(torch.isfinite(value).all() for value in breakdown.values())


def test_effective_foot_surface_distinguishes_flat_small_stance_and_large() -> None:
    cfg = JointMpcRtiCfg()
    build = getattr(terrain_module, "effective_foot_surface_height", None)
    assert callable(build)
    raw = torch.zeros(1, 4)
    small_occupancy = torch.tensor(((0.0, 1.0, 1.0, 0.0),))
    large_occupancy = torch.tensor(((0.0, 0.0, 0.0, 1.0),))
    small_height = torch.full((1, 4), 0.12)
    stance = torch.tensor(((False, False, True, False),))

    surface = build(
        raw,
        small_occupancy,
        large_occupancy,
        small_height,
        stance=stance,
        h_wall=float(cfg.terrain.h_wall),
    )

    torch.testing.assert_close(
        surface,
        torch.tensor(((0.0, 0.12, cfg.terrain.h_wall, cfg.terrain.h_wall),)),
    )


def test_node_loss_diagnostics_sum_to_existing_family_losses() -> None:
    cfg = JointMpcRtiCfg()
    state = _state()
    context = _context(state)
    diagnostic_fn = getattr(objective_module, "trajectory_node_loss_breakdown", None)

    assert callable(diagnostic_fn)
    node_breakdown = diagnostic_fn(state, context, cfg)
    family_breakdown = trajectory_loss_breakdown(state, context, cfg)

    assert tuple(node_breakdown) == ("step", "terrain", "smooth")
    assert all(value.shape == (2, 31) for value in node_breakdown.values())
    for name, node_energy in node_breakdown.items():
        torch.testing.assert_close(node_energy.sum(dim=1), family_breakdown[name])


def test_step_event_targets_tau_one_swing_endpoint_not_first_stance_node() -> None:
    cfg = JointMpcRtiCfg()
    state = _state(batch=1)
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))
    target = go2_fk(state[..., :3], state[..., 3:6], state[..., 6:]).foot_pos_w.detach()
    endpoint = state.clone()
    endpoint[:, 11, 0] += 0.01
    first_stance = state.clone()
    first_stance[:, 12, 0] += 0.01

    endpoint_residual = step_residual(endpoint, target, schedule, cfg)
    first_stance_residual = step_residual(first_stance, target, schedule, cfg)

    assert endpoint_residual.square().sum() > 0.0
    assert first_stance_residual.square().sum() == 0.0


def test_contact_keeps_current_stance_on_the_persistent_anchor() -> None:
    cfg = JointMpcRtiCfg()
    state = _state(batch=1)
    context = _context(state)
    shifted = state.clone()
    shifted[:, 1, 0] += 0.01

    baseline = contact_residual(state, context, cfg).square().sum()
    moved = contact_residual(shifted, context, cfg).square().sum()

    assert moved > baseline


def test_future_contact_rows_do_not_dilute_current_anchor_energy() -> None:
    cfg = JointMpcRtiCfg()
    state = _state(batch=1)
    context = _context(state)
    current_only = torch.zeros_like(context.schedule.stance)
    current_only[:, :2, 0] = True
    with_future = current_only.clone()
    with_future[:, 10:, :] = True

    def anchor_energy(stance: torch.Tensor) -> torch.Tensor:
        schedule = replace(context.schedule, stance=stance, swing=~stance)
        scoped_context = replace(context, schedule=schedule)
        shifted = state.clone()
        shifted[:, 1, 0] += 0.01
        baseline = contact_residual(state, scoped_context, cfg).square().sum()
        moved = contact_residual(shifted, scoped_context, cfg).square().sum()
        return moved - baseline

    torch.testing.assert_close(anchor_energy(with_future), anchor_energy(current_only))


def test_contact_uses_a_weak_reference_only_at_future_stance_onset() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.contact_future_onset_xy = 1.0
    state = _state(batch=1)
    context = _context(state)
    future_shifted = state.clone()
    future_shifted[:, 12:24, 0] += 0.01
    current_shifted = state.clone()
    current_shifted[:, 1, 0] += 0.01

    baseline = contact_residual(state, context, cfg).square().sum()
    future_cost = contact_residual(future_shifted, context, cfg).square().sum() - baseline
    current_cost = contact_residual(current_shifted, context, cfg).square().sum() - baseline

    assert future_cost > 0.0
    assert current_cost > 100.0 * future_cost


def test_contact_penalizes_slip_inside_a_future_stance_segment() -> None:
    cfg = JointMpcRtiCfg()
    state = _state(batch=1)
    context = _context(state)
    slipped = state.clone()
    slipped[:, 13, 0] += 0.01

    baseline = contact_residual(state, context, cfg).square().sum()
    moved = contact_residual(slipped, context, cfg).square().sum()

    assert moved > baseline


def test_swing_speed_penalizes_foot_not_faster_than_root() -> None:
    slow = swing_speed_penalty(
        foot_step=torch.tensor(0.01), root_step=torch.tensor(0.02), margin=0.002
    )
    fast = swing_speed_penalty(
        foot_step=torch.tensor(0.03), root_step=torch.tensor(0.02), margin=0.002
    )

    assert slow > fast


def test_swing_speed_margin_scales_with_translation_command_magnitude() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.swing_speed_margin = 0.02
    cfg.loss_terms.swing_speed_command_scale = 0.4
    state = _state(batch=2)
    schedule = fixed_trot_schedule(torch.zeros(2, dtype=torch.long))
    command = torch.tensor(((0.2, 0.0, 0.0), (0.4, 0.0, 0.0)))

    energy = swing_speed_residual(state, command, schedule, cfg).square().sum(dim=1)

    torch.testing.assert_close(energy[1], 4.0 * energy[0], rtol=1.0e-5, atol=1.0e-7)


def test_swing_progress_is_signed_along_the_command_axis() -> None:
    step = torch.tensor([[[0.03, 0.0]]])
    yaw = torch.zeros(1, 1)

    forward = swing_speed_module.directional_progress(
        step, torch.tensor([[1.0, 0.0, 0.0]]), yaw, activity_scale=0.01
    )
    backward = swing_speed_module.directional_progress(
        step, torch.tensor([[-1.0, 0.0, 0.0]]), yaw, activity_scale=0.01
    )

    assert forward.item() > 0.0
    assert backward.item() < 0.0


def test_command_early_swing_subweight_reduces_early_root_pressure() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.command_early_swing = 0.0
    state = _state(batch=1)
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))

    residual = command_residual(
        state,
        torch.tensor(((1.0, 0.0, 0.0),)),
        schedule,
        cfg,
    )
    per_edge = residual.reshape(1, 30, 3).square().sum(dim=-1)

    assert per_edge[0, 0] == 0.0
    assert per_edge[0, 1] > per_edge[0, 0]


def test_command_early_swing_keeps_future_transition_pressure_while_moving() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.command_early_swing = 0.0
    state = _state(batch=1)
    state[0, :, 0] = torch.arange(31) * cfg.runtime.dt
    state[0, 13:, 0] -= cfg.runtime.dt
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))

    residual = command_residual(
        state,
        torch.tensor(((1.0, 0.0, 0.0),)),
        schedule,
        cfg,
    ).reshape(1, 30, 3)

    assert residual[0, 12].square().sum() > 0.0


def test_command_early_swing_does_not_relax_zero_command_hold() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.command_early_swing = 0.0
    state = _state(batch=1)
    state[0, 1:, 0] = 0.01
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))

    residual = command_residual(
        state,
        torch.zeros(1, 3),
        schedule,
        cfg,
    ).reshape(1, 30, 3)

    assert residual[0, 0].square().sum() > 0.0


def test_command_early_swing_does_not_relax_pure_yaw_linear_hold() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.command_early_swing = 0.0
    state = _state(batch=1)
    state[0, 1:, 0] = 0.01
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))

    residual = command_residual(
        state,
        torch.tensor(((0.0, 0.0, 0.2),)),
        schedule,
        cfg,
    ).reshape(1, 30, 3)

    assert residual[0, 0, :2].square().sum() > 0.0
    assert residual[0, 0, 2].square() > 0.0


def test_command_hold_multiplier_scales_zero_command_residual_energy() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.command_early_swing = 1.0
    cfg.loss_terms.command_hold_multiplier = 9.0
    state = _state(batch=1)
    state[0, 1:, 0] = 0.01
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))

    held = command_residual(state, torch.zeros(1, 3), schedule, cfg)
    cfg.loss_terms.command_hold_multiplier = 1.0
    baseline = command_residual(state, torch.zeros(1, 3), schedule, cfg)

    torch.testing.assert_close(held.square().sum(), 9.0 * baseline.square().sum())


def test_command_hold_multiplier_converges_to_one_for_formal_nonzero_command() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.command_early_swing = 1.0
    cfg.loss_terms.command_hold_multiplier = 100.0
    state = _state(batch=1)
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))
    command = torch.tensor(((0.2, 0.0, 0.0),))

    held = command_residual(state, command, schedule, cfg)
    cfg.loss_terms.command_hold_multiplier = 1.0
    baseline = command_residual(state, command, schedule, cfg)

    torch.testing.assert_close(held, baseline, atol=1.0e-6, rtol=1.0e-6)


def test_command_hold_multiplier_is_continuous_between_zero_and_motion() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.command_early_swing = 1.0
    cfg.loss_terms.command_hold_multiplier = 9.0
    state = _state(batch=3)
    state[:, 1:, 0] = 0.01
    schedule = fixed_trot_schedule(torch.zeros(3, dtype=torch.long))
    commands = torch.tensor(((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.2, 0.0, 0.0)))

    held = command_residual(state, commands, schedule, cfg).square().sum(dim=1)
    cfg.loss_terms.command_hold_multiplier = 1.0
    baseline = command_residual(state, commands, schedule, cfg).square().sum(dim=1)
    ratio = held / baseline.clamp_min(1.0e-12)

    assert ratio[0] > ratio[1] > ratio[2]
    torch.testing.assert_close(ratio[0], torch.tensor(9.0))
    torch.testing.assert_close(ratio[2], torch.tensor(1.0), atol=1.0e-6, rtol=1.0e-6)


def test_command_hold_multiplier_treats_linear_and_yaw_components_independently() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.command_early_swing = 1.0
    cfg.loss_terms.command_hold_multiplier = 9.0
    state = _state(batch=2)
    state[:, 1:, 0] = torch.arange(1, 31) * 0.01
    state[:, 1:, 5] = torch.arange(1, 31) * 0.01
    schedule = fixed_trot_schedule(torch.zeros(2, dtype=torch.long))
    commands = torch.tensor(((0.0, 0.0, 0.2), (0.2, 0.0, 0.0)))

    held = command_residual(state, commands, schedule, cfg).reshape(2, 30, 3)
    cfg.loss_terms.command_hold_multiplier = 1.0
    baseline = command_residual(state, commands, schedule, cfg).reshape(2, 30, 3)

    torch.testing.assert_close(held[0, :, :2], 3.0 * baseline[0, :, :2])
    torch.testing.assert_close(held[0, :, 2], baseline[0, :, 2])
    torch.testing.assert_close(held[1, :, :2], baseline[1, :, :2])
    torch.testing.assert_close(held[1, :, 2], 3.0 * baseline[1, :, 2])


def test_swing_speed_early_subweight_increases_early_foot_pressure() -> None:
    cfg = JointMpcRtiCfg()
    cfg.loss_terms.swing_speed_early = 4.0
    state = _state(batch=1)
    state[:, :, 0] = torch.linspace(0.0, 0.3, 31)
    schedule = fixed_trot_schedule(torch.zeros(1, dtype=torch.long))

    residual = swing_speed_residual(
        state, torch.tensor([[0.2, 0.0, 0.0]]), schedule, cfg
    ).reshape(1, 30, 4)

    assert residual[0, 0, 0] > residual[0, 10, 0]


def test_smooth_loss_contains_first_and_second_state_differences() -> None:
    cfg = JointMpcRtiCfg()
    straight = _state(batch=1)
    straight[:, :, 0] = torch.linspace(0.0, 0.3, 31)
    kinked = straight.clone()
    kinked[:, 15, 6] += 0.1

    assert smooth_loss(kinked, cfg) > smooth_loss(straight, cfg)


def test_terrain_source_does_not_branch_on_raw_semantic_ids() -> None:
    from extension.joint_mpc_rti.losses import terrain

    source = inspect.getsource(terrain)

    assert "semantic_id ==" not in source
    assert "semantic_id.eq" not in source
    assert ".semantic_id" not in source


def test_total_loss_has_finite_state_gradient_and_one_packed_terrain_query(monkeypatch) -> None:
    from extension.joint_mpc_rti.losses import terrain as terrain_module

    state = _state(batch=1).requires_grad_(True)
    context = _context(state.detach())
    calls: list[tuple[int, ...]] = []
    original = terrain_module.query_world

    def counted_query(field, points):
        calls.append(tuple(points.shape))
        return original(field, points)

    monkeypatch.setattr(terrain_module, "query_world", counted_query)
    total = total_trajectory_loss(state, context, JointMpcRtiCfg())
    total.sum().backward()

    assert calls == [(1, 31 * 41, 3)]
    assert torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 0.0
