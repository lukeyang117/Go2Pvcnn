from __future__ import annotations

import torch

from extension.batched_together_planner import (
    TogetherPlannerConfig,
    TogetherPlannerTerrain,
    TogetherRobotState,
    plan_segment,
)
from extension.batched_together_planner.schedule import build_fixed_schedule
from extension.batched_together_planner.types import HIP_OFFSETS_ARRAY


def _flat_fixture(device: torch.device) -> tuple[TogetherPlannerTerrain, TogetherRobotState]:
    dtype = torch.float32
    heightmap = torch.zeros((3, 33, 33), device=device, dtype=dtype)
    terrain = TogetherPlannerTerrain.from_heightmap(
        heightmap,
        world_x_range=(-0.8, 0.8),
        world_y_range=(-0.8, 0.8),
    )
    root_pos = torch.tensor(
        [[0.0, 0.0, 0.30], [0.0, 0.0, 0.30], [0.0, 0.0, 0.30]],
        device=device,
        dtype=dtype,
    )
    root_rpy = torch.zeros((3, 3), device=device, dtype=dtype)
    foot_pos = torch.tensor(
        [
            [0.1934, 0.0465, 0.0],
            [0.1934, -0.0465, 0.0],
            [-0.1934, 0.0465, 0.0],
            [-0.1934, -0.0465, 0.0],
        ],
        device=device,
        dtype=dtype,
    ).expand(3, -1, -1)
    joint_angles = torch.zeros((3, 12), device=device, dtype=dtype)
    return terrain, TogetherRobotState(root_pos=root_pos, root_rpy=root_rpy, foot_pos=foot_pos, joint_angles=joint_angles)


def test_plan_segment_schema_shape_and_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TogetherPlannerConfig()
    terrain, state = _flat_fixture(device)
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.2, 0.6]], device=device)

    result = plan_segment(terrain, state, commands, cfg)

    assert result.root_pos.shape == (3, 35, 3)
    assert result.root_rpy.shape == (3, 35, 3)
    assert result.foot_pos.shape == (3, 35, 4, 3)
    assert result.joint_angles.shape == (3, 35, 12)
    assert result.contact_state.shape == (3, 35, 4)
    assert result.touchdown_seq.shape == (3, 4, 2, 3)
    assert result.touchdown_mask.shape == (3, 4, 2)
    assert result.cost_total.shape == (3,)
    assert result.status.shape == (3,)
    assert result.feasible.shape == (3,)
    assert result.safe_fallback.shape == (3,)
    assert result.joint_limit_violation.shape == (3, 35, 12)
    assert result.workspace_margin.shape == (3, 35, 4)
    assert result.support_xy.shape == (3, 35, 4, 2)
    assert result.support_height.shape == (3, 35, 4)
    assert result.support_slope.shape == (3, 35, 4)
    expected_device = commands.device
    assert result.root_pos.device == expected_device
    assert result.support_height.device == expected_device
    assert set(result.cost_breakdown) == {"J_td", "J_swing", "J_ik", "J_base", "J_vel"}


def test_zero_command_standstill_rehome_is_training_safe() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TogetherPlannerConfig()
    terrain, state = _flat_fixture(device)
    commands = torch.zeros((3, 3), device=device)

    result = plan_segment(terrain, state, commands, cfg)

    assert torch.allclose(result.root_pos, state.root_pos[:, None, :].expand_as(result.root_pos))
    assert torch.allclose(result.foot_pos, state.foot_pos[:, None, :, :].expand_as(result.foot_pos))
    assert torch.all(result.contact_state == 1.0)
    assert not torch.any(result.touchdown_mask)
    assert torch.all(result.feasible)
    assert not torch.any(result.safe_fallback)


def test_zero_command_root_frame_rehome_moves_terminal_feet_toward_nominal() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TogetherPlannerConfig()
    terrain, state = _flat_fixture(device)
    perturb = torch.tensor(
        [[0.04, -0.02], [-0.03, 0.03], [0.02, 0.01], [-0.04, -0.01]],
        device=device,
    )
    root_pos = state.root_pos.clone()
    root_pos[:, 2] = torch.tensor([0.18, 0.42, 0.24], device=device)
    foot_pos = state.foot_pos.clone()
    foot_pos[..., :2] = foot_pos[..., :2] + perturb.view(1, 4, 2)
    root_rpy = state.root_rpy.clone()
    root_rpy[:, 0] = torch.tensor([0.22, -0.18, 0.08], device=device)
    root_rpy[:, 1] = torch.tensor([-0.16, 0.14, -0.10], device=device)
    root_rpy[:, 2] = torch.tensor([0.35, -0.70, 1.10], device=device)
    state = TogetherRobotState(root_pos=root_pos, root_rpy=root_rpy, foot_pos=foot_pos, joint_angles=state.joint_angles)
    commands = torch.zeros((3, 3), device=device)

    result = plan_segment(terrain, state, commands, cfg)

    nominal_xy = HIP_OFFSETS_ARRAY.to(device=device, dtype=result.root_pos.dtype)[:, :2]
    yaw = state.root_rpy[:, 2]
    cos_yaw = torch.cos(yaw).view(3, 1)
    sin_yaw = torch.sin(yaw).view(3, 1)
    initial_offset_w = state.foot_pos[:, :, :2] - state.root_pos[:, None, :2]
    terminal_offset_w = result.foot_pos[:, -1, :, :2] - result.root_pos[:, -1, None, :2]
    initial_root_xy = torch.stack(
        (
            cos_yaw * initial_offset_w[..., 0] + sin_yaw * initial_offset_w[..., 1],
            -sin_yaw * initial_offset_w[..., 0] + cos_yaw * initial_offset_w[..., 1],
        ),
        dim=-1,
    )
    terminal_root_xy = torch.stack(
        (
            cos_yaw * terminal_offset_w[..., 0] + sin_yaw * terminal_offset_w[..., 1],
            -sin_yaw * terminal_offset_w[..., 0] + cos_yaw * terminal_offset_w[..., 1],
        ),
        dim=-1,
    )
    assert torch.linalg.vector_norm(terminal_root_xy - nominal_xy, dim=(1, 2)).amax() < torch.linalg.vector_norm(initial_root_xy - nominal_xy, dim=(1, 2)).amin()
    torch.testing.assert_close(result.root_pos[:, -1, :2], state.root_pos[:, :2], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(result.root_rpy[:, -1, 2], state.root_rpy[:, 2], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(result.root_pos[:, -1, 2], torch.full((3,), float(cfg.hip_height), device=device), atol=1e-5, rtol=0.0)
    assert torch.all(torch.linalg.vector_norm(result.root_rpy[:, -1, :2], dim=-1) < torch.linalg.vector_norm(state.root_rpy[:, :2], dim=-1) * 0.25)


def test_mixed_command_batch_keeps_zero_row_and_moves_commanded_rows() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TogetherPlannerConfig()
    terrain, state = _flat_fixture(device)
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.25, 0.5]], device=device)

    result = plan_segment(terrain, state, commands, cfg)

    assert torch.allclose(result.root_pos[0], state.root_pos[0].expand_as(result.root_pos[0]))
    assert result.root_pos[1, -1, 0] > result.root_pos[1, 0, 0]
    assert result.root_pos[2, -1, 1] > result.root_pos[2, 0, 1]
    assert result.root_rpy[2, -1, 2] > result.root_rpy[2, 0, 2]
    assert torch.all(result.contact_state[0] == 1.0)
    assert torch.any(result.touchdown_mask[1])


def test_safe_fallback_is_only_non_feasible_training_safe_rows() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terrain, state = _flat_fixture(device)
    commands = torch.zeros((3, 3), device=device)

    feasible_result = plan_segment(terrain, state, commands, TogetherPlannerConfig())
    fallback_result = plan_segment(
        terrain,
        state,
        commands,
        TogetherPlannerConfig(feasible_workspace_margin_min=0.20, safe_workspace_margin_min=-1.0),
    )

    assert torch.all(feasible_result.feasible)
    assert not torch.any(feasible_result.safe_fallback)
    assert not torch.any(fallback_result.feasible)
    assert torch.all(fallback_result.safe_fallback)


def test_raw_fixed_schedule_exact_contact_and_touchdown_mask() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TogetherPlannerConfig()
    commands = torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)

    schedule = build_fixed_schedule(2, cfg.horizon_steps, cfg.dt, device, torch.float32, commands, cfg)
    times = torch.arange(35, device=device, dtype=torch.float32) * 0.02
    offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=device)
    expected_contact = (torch.remainder(times[:, None] * 2.0 + offsets[None, :], 1.0) < 0.55).to(torch.float32)
    expected_td = torch.tensor(
        [[True, False], [True, False], [True, False], [True, False]],
        device=device,
    )

    assert torch.equal(schedule.contact_state[0], expected_contact)
    assert torch.equal(schedule.touchdown_mask[0], expected_td)
    assert torch.all(schedule.contact_state[1] == 1.0)
    assert not torch.any(schedule.touchdown_mask[1])
