from __future__ import annotations

import torch
import pytest

from .helpers import make_command, make_flat_field, make_state


def _realtime_cfg():
    from extension.joint_mpc_rti.config import JointMpcRtiCfg

    cfg = JointMpcRtiCfg()
    cfg.solver.line_search_alphas = (1.0, 0.25)
    return cfg


def _field_with_box(*, semantic_id: int, height_m: float, x_range: tuple[float, float], y_range: tuple[float, float]):
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch

    height = torch.zeros(1, 151, 151)
    semantic = torch.zeros(1, 151, 151, dtype=torch.long)
    x0, x1 = (int(round(value / 0.01)) + 75 for value in x_range)
    y0, y1 = (int(round(value / 0.01)) + 75 for value in y_range)
    height[:, x0 : x1 + 1, y0 : y1 + 1] = float(height_m)
    semantic[:, x0 : x1 + 1, y0 : y1 + 1] = int(semantic_id)
    return build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.zeros(1),
        timestamp=torch.zeros(1),
        version=torch.zeros(1, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )


@pytest.mark.parametrize(
    ("vx", "vy", "yaw_rate"),
    ((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.5), (0.15, 0.1, 0.2)),
)
def test_flat_commands_track_direction_with_grounded_stance(
    vx: float,
    vy: float,
    yaw_rate: float,
) -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step

    trajectory = step(
        make_state(1),
        make_command(1, vx=vx, vy=vy, yaw=yaw_rate),
        make_flat_field(1),
        None,
        _realtime_cfg(),
    ).full_trajectory

    assert torch.isfinite(trajectory.state).all()
    expected = torch.tensor([vx, vy, yaw_rate]) * (16 * 0.02)
    actual = trajectory.state[0, -1, [0, 1, 5]]
    for index in range(3):
        if abs(float(expected[index])) > 1.0e-6:
            assert torch.sign(actual[index]) == torch.sign(expected[index])
    stance = trajectory.contact_state[:, 1:]
    stance_height_error = trajectory.foot_pos_w[:, 1:, :, 2].abs()
    assert stance_height_error[stance].max() <= 0.01
    assert trajectory.foot_pos_w[..., 2].min() >= -1.0e-4


def test_rolling_x1_keeps_stance_grounded_for_zero_direction_magnitude_yaw_and_mixed_commands() -> None:
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.terrain.query import query_world
    from extension.joint_mpc_rti.types import JointMpcRtiState

    commands = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.40, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, -0.25, 0.0],
            [0.0, 0.0, 0.50],
            [0.20, 0.15, 0.30],
            [0.35, -0.20, -0.35],
        ],
        dtype=torch.float32,
    )
    batch = int(commands.shape[0])
    field = make_flat_field(batch)
    measured = make_state(batch)
    initial_root = measured.root_pos_w.clone()
    initial_yaw = measured.root_rpy_w[:, 2].clone()
    solver_state = None
    max_stance_gap = torch.zeros(batch)
    max_stance_xy_step = torch.zeros(batch)
    min_foot_gap = torch.full((batch,), torch.inf)
    previous_foot = None
    previous_contact = None

    for _ in range(32):
        result = step(measured, commands, field, solver_state, _realtime_cfg())
        trajectory = result.full_trajectory
        foot = trajectory.foot_pos_w[:, 1]
        contact = trajectory.contact_state[:, 1]
        query = query_world(field, foot)
        gap = foot[..., 2] - query.height_w
        max_stance_gap = torch.maximum(
            max_stance_gap,
            torch.where(contact, torch.abs(gap), torch.zeros_like(gap)).amax(dim=1),
        )
        if previous_foot is not None and previous_contact is not None:
            consecutive_stance = torch.logical_and(contact, previous_contact)
            xy_step = torch.linalg.vector_norm(foot[..., :2] - previous_foot[..., :2], dim=-1)
            max_stance_xy_step = torch.maximum(
                max_stance_xy_step,
                torch.where(consecutive_stance, xy_step, torch.zeros_like(xy_step)).amax(dim=1),
            )
        min_foot_gap = torch.minimum(min_foot_gap, gap.amin(dim=1))
        first_control = trajectory.control[:, 0]
        measured = JointMpcRtiState(
            root_pos_w=trajectory.state[:, 1, :3],
            root_rpy_w=trajectory.state[:, 1, 3:6],
            joint_pos=trajectory.state[:, 1, 6:],
            root_lin_vel_b=first_control[:, :3],
            root_ang_vel_b=first_control[:, 3:6],
            joint_vel=first_control[:, 6:],
        )
        solver_state = result.solver_state
        previous_foot = foot
        previous_contact = contact

    zero_root_xy_drift = torch.linalg.vector_norm(measured.root_pos_w[0, :2] - initial_root[0, :2])
    zero_root_yaw_drift = torch.abs(measured.root_rpy_w[0, 2] - initial_yaw[0])
    assert zero_root_xy_drift <= 1.0e-5
    assert zero_root_yaw_drift <= 1.0e-5
    assert max_stance_gap.max() <= 0.01
    assert max_stance_xy_step.max() <= 0.012
    assert min_foot_gap.min() >= -1.0e-4
    assert torch.isfinite(measured.as_vector()).all()


@pytest.mark.parametrize(
    ("y_range", "expected_lateral_sign"),
    (((0.02, 0.18), -1.0), ((-0.18, -0.02), 1.0)),
)
def test_large_obstacle_continuous_loss_changes_route_away_from_risk(
    y_range: tuple[float, float],
    expected_lateral_sign: float,
) -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step

    state = make_state(1)
    command = make_command(1, vx=0.2)
    flat = step(state, command, make_flat_field(1), None, _realtime_cfg()).full_trajectory
    obstacle = _field_with_box(
        semantic_id=2,
        height_m=0.30,
        x_range=(0.35, 0.48),
        y_range=y_range,
    )
    avoided = step(state, command, obstacle, None, _realtime_cfg()).full_trajectory

    assert "large_root_footprint_barrier" in avoided.loss_breakdown
    assert avoided.loss_breakdown["large_root_footprint_barrier"] > flat.loss_breakdown[
        "large_root_footprint_barrier"
    ]
    lateral_change = avoided.state[0, -1, 1] - flat.state[0, -1, 1]
    assert expected_lateral_sign * lateral_change > 0.005


@pytest.mark.parametrize("y_range", ((0.05, 0.23), (-0.23, -0.05)))
def test_small_object_is_crossed_by_swing_clearance_without_lifting_root(
    y_range: tuple[float, float],
) -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.terrain.query import query_world

    field = _field_with_box(
        semantic_id=1,
        height_m=0.08,
        x_range=(0.20, 0.32),
        y_range=y_range,
    )
    trajectory = step(make_state(1), make_command(1, vx=0.2), field, None, _realtime_cfg()).full_trajectory
    foot_query = query_world(field, trajectory.foot_pos_w.reshape(1, -1, 3))
    small_distance = foot_query.small_distance_m.reshape(1, 17, 4)
    surface_height = foot_query.height_w.reshape(1, 17, 4)
    swing = torch.logical_not(trajectory.contact_state)
    near_small_swing = torch.logical_and(swing, small_distance < 0.02)

    assert near_small_swing.any()
    clearance = trajectory.foot_pos_w[..., 2] - surface_height
    assert clearance[near_small_swing].min() >= 0.015
    assert trajectory.state[..., 2].amax() - trajectory.state[..., 2].amin() < 0.03


def test_step_height_field_lifts_supporting_root_without_penetration() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    height = torch.zeros(1, 151, 151)
    height[:, 95:, :] = 0.06
    field = build_field_batch(
        height_w=height,
        semantic_id=torch.zeros(1, 151, 151, dtype=torch.long),
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.zeros(1),
        timestamp=torch.zeros(1),
        version=torch.zeros(1, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )
    trajectory = step(make_state(1), make_command(1, vx=0.2), field, None, _realtime_cfg()).full_trajectory
    query = query_world(field, trajectory.foot_pos_w.reshape(1, -1, 3))
    foot_height = query.height_w.reshape(1, 17, 4)
    foot_gap = trajectory.foot_pos_w[..., 2] - foot_height

    assert foot_gap.min() >= -1.0e-4
    assert trajectory.state[0, -1, 2] > 0.33
    assert torch.diff(trajectory.state[0, :, 2]).abs().max() < 0.03


def test_down_step_lowers_root_continuously_without_penetration() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    height = torch.zeros(1, 151, 151)
    height[:, :95, :] = 0.06
    field = build_field_batch(
        height_w=height,
        semantic_id=torch.zeros(1, 151, 151, dtype=torch.long),
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.zeros(1),
        timestamp=torch.zeros(1),
        version=torch.zeros(1, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )
    state = make_state(1)
    state.root_pos_w[:, 2] = 0.38
    trajectory = step(state, make_command(1, vx=0.2), field, None, _realtime_cfg()).full_trajectory
    query = query_world(field, trajectory.foot_pos_w.reshape(1, -1, 3))
    foot_gap = trajectory.foot_pos_w[..., 2] - query.height_w.reshape(1, 17, 4)

    assert foot_gap.min() >= -1.0e-4
    assert trajectory.state[0, -1, 2] < 0.379
    assert torch.diff(trajectory.state[0, :, 2]).abs().max() < 0.03
