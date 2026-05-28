import torch

from extension.batch_mpc_planner.parametric import (
    MpcParametricVariables,
    bounded_unit_interval,
    command_frame_axes,
    cubic_bezier,
    decode_parametric_trajectory,
    init_parametric_variables,
)
from extension.batch_mpc_planner.config import MpcPlannerCfg
from extension.batch_mpc_planner.semantic_policy import build_parametric_nominal
from extension.batch_mpc_planner.types import MpcPlannerTerrain, MpcRobotState


def test_command_frame_axes_uses_translation_direction() -> None:
    command = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    yaw = torch.tensor([0.0], dtype=torch.float32)

    forward, left, active = command_frame_axes(command, yaw, linear_eps=1.0e-4)

    assert active.tolist() == [True]
    torch.testing.assert_close(forward, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(left, torch.tensor([[-1.0, 0.0]]))


def test_command_frame_axes_falls_back_to_root_yaw_for_pure_yaw() -> None:
    command = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    yaw = torch.tensor([1.5707964], dtype=torch.float32)

    forward, left, active = command_frame_axes(command, yaw, linear_eps=1.0e-4)

    assert active.tolist() == [False]
    torch.testing.assert_close(forward, torch.tensor([[0.0, 1.0]]), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(left, torch.tensor([[-1.0, 0.0]]), atol=1.0e-6, rtol=1.0e-6)


def test_cubic_bezier_starts_and_ends_at_control_points() -> None:
    p0 = torch.tensor([[[0.0, 0.0]]])
    p1 = torch.tensor([[[0.3, 0.1]]])
    p2 = torch.tensor([[[0.7, 0.1]]])
    p3 = torch.tensor([[[1.0, 0.0]]])
    phase = torch.tensor([0.0, 0.5, 1.0])

    curve = cubic_bezier(p0, p1, p2, p3, phase)

    torch.testing.assert_close(curve[:, 0], p0)
    torch.testing.assert_close(curve[:, -1], p3)


def test_bounded_unit_interval_maps_zero_raw_to_midpoint() -> None:
    raw = torch.zeros((2, 4), dtype=torch.float32)

    value = bounded_unit_interval(raw, low=0.15, high=0.85)

    torch.testing.assert_close(value, torch.full((2, 4), 0.5))


def _flat_terrain(batch: int = 1, height: float = 0.2) -> MpcPlannerTerrain:
    return MpcPlannerTerrain(
        height_map=torch.full((batch, 5, 5), height, dtype=torch.float32),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )


def _terrain_with_large_obstacle() -> MpcPlannerTerrain:
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    height[:, 4, 6] = 0.45
    semantic[:, 4, 6] = 2
    return MpcPlannerTerrain(
        height_map=height,
        semantic_map=semantic,
        world_x_range=(-0.8, 0.8),
        world_y_range=(-0.8, 0.8),
    )


def _state(batch: int = 1) -> MpcRobotState:
    return MpcRobotState(
        root_pos=torch.tensor([[0.0, 0.0, 0.35]], dtype=torch.float32).expand(batch, 3).clone(),
        root_rpy=torch.zeros((batch, 3), dtype=torch.float32),
        foot_pos=torch.tensor(
            [
                [
                    [0.25, 0.12, 0.2],
                    [0.25, -0.12, 0.2],
                    [-0.25, 0.12, 0.2],
                    [-0.25, -0.12, 0.2],
                ]
            ],
            dtype=torch.float32,
        ).expand(batch, 4, 3).clone(),
        joint_angles=torch.zeros((batch, 12), dtype=torch.float32),
    )


def _nominal(state: MpcRobotState, terrain: MpcPlannerTerrain, command: torch.Tensor, *, horizon: int = 25):
    return build_parametric_nominal(state, terrain, command, MpcPlannerCfg(), horizon=horizon)


def test_build_parametric_nominal_shapes_high_large_command() -> None:
    terrain = _terrain_with_large_obstacle()
    state = _state()
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)

    nominal = build_parametric_nominal(state, terrain, command, MpcPlannerCfg(), horizon=25)

    assert nominal.command.shape == (1, 3)
    assert nominal.shape_diagnostics.command_shaped.item() is True
    assert abs(float(nominal.command[0, 1])) > 0.0
    assert nominal.root_goal_delta.shape == (1, 2)
    assert nominal.terminal_rel_xy.shape == (1, 4, 2)


def test_decode_parametric_consumes_nominal_without_command_shaping() -> None:
    state = _state()
    terrain = _flat_terrain()
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    nominal = _nominal(state, terrain, command)
    variables = init_parametric_variables(state, nominal.command, horizon=25)

    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)

    assert decoded.root_pos.shape == (1, 25, 3)
    assert decoded.target_foot_pos.shape == (1, 25, 4, 3)


def test_init_parametric_variables_has_expected_shapes() -> None:
    variables = init_parametric_variables(_state(), torch.tensor([[0.5, 0.0, 0.0]]), horizon=25)

    assert isinstance(variables, MpcParametricVariables)
    assert variables.touchdown_delta_raw.shape == (1, 4, 2)
    assert variables.swing_clearance_raw.shape == (1, 4)
    assert variables.bezier_ab_raw.shape == (1, 4, 2)
    assert variables.root_goal_delta_raw.shape == (1, 2)
    assert variables.root_bezier_raw.shape == (1, 2)
    assert variables.diagonal_phase_raw.shape == (1,)


def test_parametric_variables_parameters_are_optimizable() -> None:
    variables = init_parametric_variables(_state(), torch.tensor([[0.5, 0.0, 0.0]]), horizon=25)

    params = variables.parameters()

    assert len(params) >= 8
    assert all(param.requires_grad for param in params)


def test_decode_parametric_trajectory_starts_from_current_state_and_has_25_frames() -> None:
    state = _state()
    terrain = _flat_terrain()
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)

    decoded = decode_parametric_trajectory(state, terrain, _nominal(state, terrain, command), variables, horizon=25)

    assert decoded.root_pos.shape == (1, 25, 3)
    assert decoded.target_foot_pos.shape == (1, 25, 4, 3)
    torch.testing.assert_close(decoded.root_pos[:, 0], state.root_pos)
    torch.testing.assert_close(decoded.target_foot_pos[:, 0], state.foot_pos)


def test_decode_parametric_touchdown_z_comes_from_height_map() -> None:
    state = _state()
    terrain = _flat_terrain(height=0.42)
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)

    decoded = decode_parametric_trajectory(state, terrain, _nominal(state, terrain, command), variables, horizon=25)

    torch.testing.assert_close(decoded.touchdown_w[..., 2], torch.full((1, 4), 0.42))


def test_decode_parametric_root_curve_bypasses_high_large_semantic_margin_without_projection() -> None:
    state = _state()
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    height[:, 4, 6] = 0.45
    semantic[:, 4, 6] = 2
    terrain = MpcPlannerTerrain(
        height_map=height,
        semantic_map=semantic,
        world_x_range=(-0.8, 0.8),
        world_y_range=(-0.8, 0.8),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)

    decoded = decode_parametric_trajectory(state, terrain, _nominal(state, terrain, command), variables, horizon=25)

    obstacle_xy = torch.tensor([0.4, 0.0], dtype=torch.float32)
    distance = torch.linalg.vector_norm(decoded.root_pos[0, :, :2] - obstacle_xy, dim=-1)
    assert distance.amin().item() >= 0.305


def test_decode_parametric_foot_curves_move_trot_pairs_in_alternating_windows() -> None:
    state = _state()
    terrain = _flat_terrain()
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)
    with torch.no_grad():
        variables.touchdown_delta_raw[:, :, 0].fill_(1.0)

    decoded = decode_parametric_trajectory(state, terrain, _nominal(state, terrain, command), variables, horizon=25)

    xy_step = torch.linalg.vector_norm(
        decoded.target_foot_pos[0, 1:, :, :2] - decoded.target_foot_pos[0, :-1, :, :2],
        dim=-1,
    )
    moving = xy_step > 0.005
    pair_a = moving[:, [0, 3]]
    pair_b = moving[:, [1, 2]]

    assert not torch.any(pair_a.all(dim=1) & pair_b.all(dim=1)).item()
    assert torch.any(pair_a.all(dim=1) & ~pair_b.any(dim=1)).item()
    assert torch.any(pair_b.all(dim=1) & ~pair_a.any(dim=1)).item()


def test_decode_parametric_replan_starts_from_current_isaaclab_foot_positions() -> None:
    state = _state()
    state.foot_pos[:, :, :2] += torch.tensor(
        [[[0.03, 0.01], [-0.04, 0.02], [0.02, -0.03], [-0.01, -0.02]]],
        dtype=torch.float32,
    )
    terrain = _flat_terrain()
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)
    with torch.no_grad():
        variables.touchdown_delta_raw[:, :, 0].fill_(1.0)

    decoded = decode_parametric_trajectory(state, terrain, _nominal(state, terrain, command), variables, horizon=25)
    torch.testing.assert_close(decoded.target_foot_pos[:, 0], state.foot_pos, atol=1.0e-6, rtol=1.0e-6)


def test_decode_parametric_full_cycle_replan_does_not_accumulate_root_relative_foot_drift() -> None:
    terrain = _flat_terrain()
    command = torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float32)
    state0 = _state()
    variables0 = init_parametric_variables(state0, command, horizon=25)
    with torch.no_grad():
        variables0.touchdown_delta_raw[:, :, 0].fill_(1.0)

    first = decode_parametric_trajectory(state0, terrain, _nominal(state0, terrain, command), variables0, horizon=25)
    state1 = MpcRobotState(
        root_pos=first.root_pos[:, -1].detach(),
        root_rpy=first.root_rpy[:, -1].detach(),
        foot_pos=first.target_foot_pos[:, -1].detach(),
        joint_angles=state0.joint_angles.clone(),
    )
    variables1 = init_parametric_variables(state1, command, horizon=25)
    with torch.no_grad():
        variables1.touchdown_delta_raw[:, :, 0].fill_(1.0)

    second = decode_parametric_trajectory(state1, terrain, _nominal(state1, terrain, command), variables1, horizon=25)

    initial_rel_xy = state0.foot_pos[..., :2] - state0.root_pos[:, None, :2]
    first_terminal_rel_xy = first.target_foot_pos[:, -1, :, :2] - first.root_pos[:, -1:, :2]
    second_terminal_rel_xy = second.target_foot_pos[:, -1, :, :2] - second.root_pos[:, -1:, :2]
    first_drift = torch.linalg.vector_norm(first_terminal_rel_xy - initial_rel_xy, dim=-1).amax()
    second_drift = torch.linalg.vector_norm(second_terminal_rel_xy - initial_rel_xy, dim=-1).amax()

    assert second_drift.item() <= first_drift.item() + 1.0e-5


def test_decode_parametric_root_pitch_follows_support_plane_after_first_frame() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    x = torch.linspace(-0.5, 0.5, 5, dtype=torch.float32)
    height[0] = (0.18 - 0.24 * x).view(1, 5).expand(5, 5)
    terrain = MpcPlannerTerrain(height_map=height, world_x_range=(-0.5, 0.5), world_y_range=(-0.5, 0.5))
    state = MpcRobotState(
        root_pos=torch.tensor([[0.0, 0.0, 0.35]], dtype=torch.float32),
        root_rpy=torch.zeros((1, 3), dtype=torch.float32),
        foot_pos=torch.tensor(
            [[[0.25, 0.12, 0.12], [0.25, -0.12, 0.12], [-0.25, 0.12, 0.24], [-0.25, -0.12, 0.24]]],
            dtype=torch.float32,
        ),
        joint_angles=torch.zeros((1, 12), dtype=torch.float32),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)

    decoded = decode_parametric_trajectory(state, terrain, _nominal(state, terrain, command), variables, horizon=25)

    torch.testing.assert_close(decoded.root_rpy[:, 0, :2], state.root_rpy[:, :2], atol=1.0e-6, rtol=1.0e-6)
    assert decoded.root_rpy[0, -1, 1].item() > 0.05
