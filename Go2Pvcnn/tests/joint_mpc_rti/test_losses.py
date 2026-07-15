from __future__ import annotations

import torch


def test_relaxed_barrier_is_finite_and_increases_toward_violation() -> None:
    from extension.joint_mpc_rti.losses.barriers import relaxed_barrier

    margin = torch.tensor([0.2, 0.02, 0.0, -0.02], requires_grad=True)
    value = relaxed_barrier(margin, relaxation=0.01)

    assert torch.isfinite(value).all()
    assert torch.all(value[1:] > value[:-1])
    value.sum().backward()
    assert torch.isfinite(margin.grad).all()


def test_command_losses_prefer_matching_body_velocity_and_progress() -> None:
    from extension.joint_mpc_rti.losses.command import command_losses

    command = torch.tensor([[0.2, 0.0, 0.0]])
    root_pos = torch.zeros(1, 17, 3)
    root_pos[:, :, 0] = torch.linspace(0.0, 0.064, 17)
    root_rpy = torch.zeros_like(root_pos)
    matching_control = torch.zeros(1, 16, 18)
    matching_control[:, :, 0] = 0.2
    stopped_control = torch.zeros_like(matching_control)

    matching = command_losses(root_pos, root_rpy, matching_control, command, dt=0.02)
    stopped = command_losses(root_pos * 0.0, root_rpy, stopped_control, command, dt=0.02)

    assert matching["command_linear_velocity"] < stopped["command_linear_velocity"]
    assert matching["command_progress"] < stopped["command_progress"]


def test_stance_losses_separate_ground_contact_from_slip() -> None:
    from extension.joint_mpc_rti.losses.contact import stance_losses

    foot = torch.zeros(1, 3, 4, 3)
    height = torch.zeros(1, 3, 4)
    contact = torch.ones(1, 3, 4, dtype=torch.bool)
    stable = stance_losses(foot, height, contact, dt=0.02)
    drifting_foot = foot.clone()
    drifting_foot[:, 1:, :, 0] = 0.02
    drifting = stance_losses(drifting_foot, height, contact, dt=0.02)

    assert stable["stance_ground_contact"].item() == 0.0
    assert stable["stance_slip_velocity"] < drifting["stance_slip_velocity"]
    assert stable["stance_xy_lock"] < drifting["stance_xy_lock"]


def test_small_object_loss_prefers_foot_over_or_bypass_without_root_gate() -> None:
    from extension.joint_mpc_rti.losses.semantic import small_object_losses

    common = dict(
        small_top_height=torch.tensor([[[0.08]]]),
        small_distance_touchdown=torch.tensor([[0.20]]),
        link_pos_w=torch.tensor([[[[0.0, 0.0, 0.20]]]]),
        link_small_distance=torch.tensor([[[0.0]]]),
        swing_mask=torch.tensor([[[True]]]),
        stance_mask=torch.tensor([[[False]]]),
        extra_margin=0.03,
    )
    over = dict(
        common,
        foot_pos_w=torch.tensor([[[[0.0, 0.0, 0.20]]]]),
        foot_small_distance=torch.tensor([[[0.0]]]),
    )
    low = dict(
        common,
        foot_pos_w=torch.tensor([[[[0.0, 0.0, 0.03]]]]),
        foot_small_distance=torch.tensor([[[0.0]]]),
    )
    bypass = dict(
        common,
        foot_pos_w=torch.tensor([[[[0.2, 0.0, 0.03]]]]),
        foot_small_distance=torch.tensor([[[0.20]]]),
    )

    over_loss = small_object_losses(**over)
    low_loss = small_object_losses(**low)
    bypass_loss = small_object_losses(**bypass)

    assert over_loss["small_object_foot_over"] < low_loss["small_object_foot_over"]
    assert bypass_loss["small_object_foot_over"] < low_loss["small_object_foot_over"]
    assert "small_object_root_avoidance" not in over_loss


def test_touchdown_on_small_is_penalized_even_when_height_matches_surface() -> None:
    from extension.joint_mpc_rti.losses.contact import touchdown_losses
    from extension.joint_mpc_rti.losses.semantic import small_object_losses

    foot = torch.tensor([[[[0.0, 0.0, 0.08]]]])
    contact = touchdown_losses(
        touchdown_pos_w=foot[:, 0],
        queried_height_w=torch.tensor([[0.08]]),
        queried_valid=torch.tensor([[True]]),
    )
    semantic = small_object_losses(
        foot_pos_w=foot,
        foot_small_distance=torch.tensor([[[-0.01]]]),
        small_top_height=torch.tensor([[[0.08]]]),
        small_distance_touchdown=torch.tensor([[-0.01]]),
        link_pos_w=foot,
        link_small_distance=torch.tensor([[[-0.01]]]),
        swing_mask=torch.tensor([[[False]]]),
        stance_mask=torch.tensor([[[True]]]),
        extra_margin=0.03,
    )

    assert contact["touchdown_ground_height"] < 1.0e-6
    assert semantic["small_object_touchdown_avoidance"] > 0.0


def test_large_obstacle_barrier_increases_when_body_is_closer() -> None:
    from extension.joint_mpc_rti.losses.semantic import large_obstacle_losses

    far = large_obstacle_losses(
        root_footprint_distance=torch.full((1, 4), 0.4),
        body_distance=torch.full((1, 4), 0.4),
        foot_distance=torch.full((1, 4), 0.4),
        knee_shank_distance=torch.full((1, 8), 0.4),
        terminal_distance=torch.tensor([0.4]),
        terminal_approach_speed=torch.tensor([0.0]),
    )
    near = large_obstacle_losses(
        root_footprint_distance=torch.full((1, 4), 0.02),
        body_distance=torch.full((1, 4), 0.02),
        foot_distance=torch.full((1, 4), 0.02),
        knee_shank_distance=torch.full((1, 8), 0.02),
        terminal_distance=torch.tensor([0.02]),
        terminal_approach_speed=torch.tensor([-0.2]),
    )

    assert near["large_root_footprint_barrier"] > far["large_root_footprint_barrier"]
    assert near["large_terminal_risk"] > far["large_terminal_risk"]


def test_smoothness_losses_penalize_control_jump() -> None:
    from extension.joint_mpc_rti.losses.smoothness import smoothness_losses

    stable = torch.zeros(1, 16, 18)
    jump = stable.clone()
    jump[:, 8:, 6:] = 1.0

    stable_loss = smoothness_losses(stable, previous_control=torch.zeros(1, 18), dt=0.02)
    jump_loss = smoothness_losses(jump, previous_control=torch.zeros(1, 18), dt=0.02)

    assert jump_loss["control_rate"] > stable_loss["control_rate"]
    assert jump_loss["joint_acceleration"] > stable_loss["joint_acceleration"]


def test_posture_losses_penalize_root_height_and_joint_limit_margins() -> None:
    from extension.joint_mpc_rti.losses.posture import posture_losses

    root_pos = torch.zeros(1, 3, 3)
    root_pos[..., 2] = 0.32
    root_rpy = torch.zeros(1, 3, 3)
    joint_nominal = torch.tensor([0.0, 0.8, -1.5] * 4)
    joint = joint_nominal.reshape(1, 1, 12).expand(1, 3, 12).clone()
    joint_velocity = torch.zeros(1, 2, 12)
    lower = torch.full((12,), -2.0)
    upper = torch.full((12,), 2.0)
    nominal = posture_losses(
        root_pos_w=root_pos,
        root_rpy_w=root_rpy,
        joint_pos=joint,
        joint_velocity=joint_velocity,
        support_height=torch.zeros(1, 3),
        nominal_root_clearance=0.32,
        nominal_joint_pos=joint_nominal,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=torch.full((12,), 20.0),
    )
    near_limit_joint = joint.clone()
    near_limit_joint[..., 0] = 1.99
    near_limit = posture_losses(
        root_pos_w=root_pos + torch.tensor([0.0, 0.0, 0.08]),
        root_rpy_w=root_rpy,
        joint_pos=near_limit_joint,
        joint_velocity=joint_velocity,
        support_height=torch.zeros(1, 3),
        nominal_root_clearance=0.32,
        nominal_joint_pos=joint_nominal,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=torch.full((12,), 20.0),
    )

    assert near_limit["root_support_height"] > nominal["root_support_height"]
    assert near_limit["joint_position_limit_barrier"] > nominal["joint_position_limit_barrier"]


def test_clearance_losses_penalize_penetrating_foot_knee_shank_and_body() -> None:
    from extension.joint_mpc_rti.losses.clearance import clearance_losses

    safe = clearance_losses(
        foot_pos_w=torch.full((1, 2, 4, 3), 0.20),
        foot_height_w=torch.zeros(1, 2, 4),
        knee_pos_w=torch.full((1, 2, 4, 3), 0.20),
        knee_height_w=torch.zeros(1, 2, 4),
        shank_pos_w=torch.full((1, 2, 4, 3, 3), 0.20),
        shank_height_w=torch.zeros(1, 2, 4, 3),
        body_pos_w=torch.full((1, 2, 9, 3), 0.20),
        body_height_w=torch.zeros(1, 2, 9),
        swing_mask=torch.ones(1, 2, 4, dtype=torch.bool),
    )
    penetrating = clearance_losses(
        foot_pos_w=torch.full((1, 2, 4, 3), -0.02),
        foot_height_w=torch.zeros(1, 2, 4),
        knee_pos_w=torch.full((1, 2, 4, 3), -0.02),
        knee_height_w=torch.zeros(1, 2, 4),
        shank_pos_w=torch.full((1, 2, 4, 3, 3), -0.02),
        shank_height_w=torch.zeros(1, 2, 4, 3),
        body_pos_w=torch.full((1, 2, 9, 3), -0.02),
        body_height_w=torch.zeros(1, 2, 9),
        swing_mask=torch.ones(1, 2, 4, dtype=torch.bool),
    )

    for name in ("foot_ground_penetration", "knee_ground_clearance", "shank_ground_clearance", "body_ground_clearance"):
        assert penetrating[name] > safe[name]


def test_swing_losses_prefer_clear_smooth_foot_and_soft_touchdown() -> None:
    from extension.joint_mpc_rti.losses.contact import swing_losses

    nominal = torch.zeros(1, 3, 4, 3)
    nominal[..., 2] = 0.10
    swing_mask = torch.ones(1, 3, 4, dtype=torch.bool)
    good = swing_losses(
        foot_pos_w=nominal,
        nominal_foot_pos_w=nominal,
        queried_height_w=torch.zeros(1, 3, 4),
        swing_mask=swing_mask,
        dt=0.02,
    )
    low_and_jumpy = nominal.clone()
    low_and_jumpy[:, 1, :, 0] = 0.20
    low_and_jumpy[..., 2] = 0.0
    bad = swing_losses(
        foot_pos_w=low_and_jumpy,
        nominal_foot_pos_w=nominal,
        queried_height_w=torch.zeros(1, 3, 4),
        swing_mask=swing_mask,
        dt=0.02,
    )

    assert bad["swing_nominal_shape"] > good["swing_nominal_shape"]
    assert bad["terrain_swing_clearance"] > good["terrain_swing_clearance"]
    assert bad["swing_velocity_smoothness"] > good["swing_velocity_smoothness"]


def test_weighted_objective_sums_named_terms_per_environment() -> None:
    from extension.joint_mpc_rti.losses.objective import weighted_objective

    losses = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    total = weighted_objective(losses, {"a": 2.0, "b": 0.5})

    torch.testing.assert_close(total, torch.tensor([3.5, 6.0]))


def test_touchdown_geometry_penalizes_workspace_edge_and_crossed_feet() -> None:
    from extension.joint_mpc_rti.losses.contact import touchdown_geometry_losses

    nominal = torch.tensor(
        [[[0.25, 0.15, -0.30], [0.25, -0.15, -0.30], [-0.25, 0.15, -0.30], [-0.25, -0.15, -0.30]]]
    )
    unsafe = nominal.clone()
    unsafe[:, 0] = torch.tensor([0.60, -0.01, -0.05])
    unsafe[:, 1, 1] = 0.0

    nominal_loss = touchdown_geometry_losses(nominal, min_reach=0.20, max_reach=0.48, min_left_right_separation=0.12)
    unsafe_loss = touchdown_geometry_losses(unsafe, min_reach=0.20, max_reach=0.48, min_left_right_separation=0.12)

    assert unsafe_loss["touchdown_reach_margin"] > nominal_loss["touchdown_reach_margin"]
    assert unsafe_loss["touchdown_foot_separation"] > nominal_loss["touchdown_foot_separation"]


def test_terminal_losses_penalize_stopping_at_obstacle_in_extreme_posture() -> None:
    from extension.joint_mpc_rti.losses.objective import terminal_losses

    command = torch.tensor([[0.2, 0.0, 0.0]])
    nominal_joint = torch.tensor([0.0, 0.8, -1.5] * 4)
    good = terminal_losses(
        terminal_control=torch.tensor([[0.2, 0.0, 0.0] + [0.0] * 15]),
        command_body=command,
        terminal_root_rpy=torch.zeros(1, 3),
        terminal_joint_pos=nominal_joint.unsqueeze(0),
        nominal_joint_pos=nominal_joint,
        obstacle_distance=torch.tensor([0.5]),
        obstacle_approach_speed=torch.tensor([0.0]),
        contact_viability=torch.tensor([1.0]),
    )
    bad = terminal_losses(
        terminal_control=torch.zeros(1, 18),
        command_body=command,
        terminal_root_rpy=torch.tensor([[0.4, -0.4, 0.0]]),
        terminal_joint_pos=(nominal_joint + 0.5).unsqueeze(0),
        nominal_joint_pos=nominal_joint,
        obstacle_distance=torch.tensor([0.02]),
        obstacle_approach_speed=torch.tensor([-0.3]),
        contact_viability=torch.tensor([0.0]),
    )

    for name in ("terminal_command_velocity", "terminal_obstacle_safety", "terminal_posture", "terminal_contact_viability"):
        assert bad[name] > good[name]
