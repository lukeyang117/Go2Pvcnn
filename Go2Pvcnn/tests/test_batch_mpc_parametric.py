import torch

from extension.batch_mpc_planner.parametric import (
    MpcParametricVariables,
    bounded_unit_interval,
    command_frame_axes,
    cubic_bezier,
    decode_parametric_trajectory,
    init_parametric_variables,
)
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

    decoded = decode_parametric_trajectory(state, terrain, command, variables, horizon=25)

    assert decoded.root_pos.shape == (1, 25, 3)
    assert decoded.target_foot_pos.shape == (1, 25, 4, 3)
    torch.testing.assert_close(decoded.root_pos[:, 0], state.root_pos)
    torch.testing.assert_close(decoded.target_foot_pos[:, 0], state.foot_pos)


def test_decode_parametric_touchdown_z_comes_from_height_map() -> None:
    state = _state()
    terrain = _flat_terrain(height=0.42)
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)

    decoded = decode_parametric_trajectory(state, terrain, command, variables, horizon=25)

    torch.testing.assert_close(decoded.touchdown_w[..., 2], torch.full((1, 4), 0.42))
