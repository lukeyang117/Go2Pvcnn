from __future__ import annotations

import inspect

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import LossContext, total_trajectory_loss, trajectory_loss_breakdown
from extension.joint_mpc_rti.losses.smoothness import smooth_loss
from extension.joint_mpc_rti.losses.swing_speed import swing_speed_penalty
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


def test_swing_speed_penalizes_foot_not_faster_than_root() -> None:
    slow = swing_speed_penalty(
        foot_step=torch.tensor(0.01), root_step=torch.tensor(0.02), margin=0.002
    )
    fast = swing_speed_penalty(
        foot_step=torch.tensor(0.03), root_step=torch.tensor(0.02), margin=0.002
    )

    assert slow > fast


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
