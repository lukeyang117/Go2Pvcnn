from __future__ import annotations

from types import SimpleNamespace

import torch


def test_m1_backend_exposes_12_planner_joints_and_16_asset_joints():
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")

    assert len(backend.planner_joint_names) == 12
    assert len(backend.asset_joint_names) == 16
    assert backend.wheel_joint_names == (
        "FBL_FOOT_JOINT",
        "FAR_FOOT_JOINT",
        "RBL_FOOT_JOINT",
        "RAR_FOOT_JOINT",
    )


def test_m1_fk_neutral_pose_has_symmetric_wheel_centers():
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    geometry = backend.fk(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 12))
    expected = torch.tensor(
        [
            [
                [0.3295, 0.2074, -0.5398],
                [0.3295, -0.2074, -0.5398],
                [-0.3295, 0.2074, -0.5398],
                [-0.3295, -0.2074, -0.5398],
            ]
        ]
    )

    torch.testing.assert_close(geometry.foot_pos_w, expected, atol=2e-4, rtol=0)


def test_m1_collision_geometry_has_16_shapes_and_464_points():
    from extension.parallelism.m1_kinematics import M1_CFG
    from extension.parallelism.m1_collision import build_m1_surface_points_l

    points, mask = build_m1_surface_points_l(
        M1_CFG.official_collision_shapes,
        M1_CFG,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert len(M1_CFG.official_collision_shapes) == 16
    assert points.shape == (16, 38, 3)
    assert int(mask.sum().item()) == 464
    assert sum(spec.shape_type == "cylinder" for spec in M1_CFG.official_collision_shapes) == 4
    assert not any(spec.link_type == "base" for spec in M1_CFG.official_collision_shapes)


def test_m1_joint_limits_keep_wheel_joints_out_of_planner():
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    assert backend.joint_limit_mask(torch.zeros(2, 12)).all()
    assert all("FOOT_JOINT" not in name for name in backend.planner_joint_names)


def test_m1_planner_mapping_matches_usd_joint_order():
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.viz.go2_foostep_planner import _backend_planner_joint_indices, merge_planner_joints_into_robot

    backend = get_robot_backend("m1")
    robot = SimpleNamespace(
        joint_names=(
            "FAR_ABAD_JOINT",
            "FBL_ABAD_JOINT",
            "RAR_ABAD_JOINT",
            "RBL_ABAD_JOINT",
            "FAR_HIP_JOINT",
            "FBL_HIP_JOINT",
            "RAR_HIP_JOINT",
            "RBL_HIP_JOINT",
            "FAR_KNEE_JOINT",
            "FBL_KNEE_JOINT",
            "RAR_KNEE_JOINT",
            "RBL_KNEE_JOINT",
            "FAR_FOOT_JOINT",
            "FBL_FOOT_JOINT",
            "RAR_FOOT_JOINT",
            "RBL_FOOT_JOINT",
        )
    )

    indices = _backend_planner_joint_indices(robot, backend)

    assert indices.tolist() == [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

    current = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    planned = torch.full((1, 12), -1.0)
    merged = merge_planner_joints_into_robot(current, planned, backend, robot=robot)

    assert merged[0, [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]].tolist() == [-1.0] * 12
    assert merged[0, [12, 13, 14, 15]].tolist() == [12.0, 13.0, 14.0, 15.0]


def test_m1_ik_round_trips_the_neutral_wheel_centers():
    from extension.parallelism.m1_kinematics import M1_ROOT_Z_M
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    root = torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]])
    rpy = torch.zeros(1, 3)
    target = backend.fk(root, rpy, torch.zeros(1, 12)).foot_pos_w
    joint, reachable = backend.ik(root, rpy, target)

    assert reachable.all()
    torch.testing.assert_close(joint, torch.zeros(1, 4, 3), atol=2e-3, rtol=0)


def test_m1_ik_reaches_a_small_forward_foothold_displacement():
    from extension.parallelism.m1_kinematics import M1_ROOT_Z_M
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    root = torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]])
    rpy = torch.zeros(1, 3)
    neutral = torch.zeros(1, 12)
    target = backend.fk(root, rpy, neutral).foot_pos_w
    target[..., 0] += 0.03

    joint, reachable = backend.ik(root, rpy, target)
    residual = torch.linalg.vector_norm(
        backend.fk(root, rpy, joint.reshape(1, 12)).foot_pos_w - target,
        dim=-1,
    )

    assert reachable.all()
    assert torch.all(residual <= 2.0e-3)


def test_m1_default_stance_is_nonzero_and_keeps_wheels_at_contact_height():
    from extension.parallelism.m1_kinematics import (
        M1_DEFAULT_JOINT_POS,
        M1_DEFAULT_KNEE_ANGLE_RAD,
        M1_ROOT_Z_M,
        M1_WHEEL_RADIUS_M,
    )
    from extension.parallelism.robot_backend import get_robot_backend

    default_joint = torch.tensor([M1_DEFAULT_JOINT_POS], dtype=torch.float32)
    assert default_joint.shape == (1, 12)
    assert not torch.allclose(default_joint, torch.zeros_like(default_joint))
    assert torch.allclose(default_joint.reshape(1, 4, 3)[..., 0], torch.zeros(1, 4))
    assert torch.all(default_joint.reshape(1, 4, 3)[..., 1] < 0.0)
    assert torch.all(default_joint.reshape(1, 4, 3)[..., 2] > 0.0)
    assert M1_DEFAULT_KNEE_ANGLE_RAD == 1.10
    assert M1_ROOT_Z_M < 0.63

    geometry = get_robot_backend("m1").fk(
        torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        torch.zeros(1, 3),
        default_joint,
    )
    torch.testing.assert_close(
        geometry.foot_pos_w[..., 2],
        torch.full((1, 4), M1_WHEEL_RADIUS_M),
        atol=2.0e-4,
        rtol=0.0,
    )


def test_m1_asset_default_stance_inserts_a_zero_wheel_joint_per_leg():
    from extension.parallelism import m1_kinematics

    asset_default = tuple(getattr(m1_kinematics, "M1_DEFAULT_ASSET_JOINT_POS", ()))

    assert asset_default == (
        0.0,
        -0.15,
        1.10,
        0.0,
        0.0,
        -0.15,
        1.10,
        0.0,
        0.0,
        -0.15,
        1.10,
        0.0,
        0.0,
        -0.15,
        1.10,
        0.0,
    )


def test_m1_ik_round_trips_the_default_stance_without_branch_jump():
    from extension.parallelism.m1_kinematics import M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    root = torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]])
    rpy = torch.zeros(1, 3)
    default_joint = torch.tensor([M1_DEFAULT_JOINT_POS], dtype=torch.float32)
    target = backend.fk(root, rpy, default_joint).foot_pos_w

    joint, reachable = backend.ik(root, rpy, target)

    assert reachable.all()
    torch.testing.assert_close(joint.reshape(1, 12), default_joint, atol=2.0e-3, rtol=0.0)


def test_m1_ik_is_analytic_and_does_not_iterate_through_fk(monkeypatch):
    from extension.parallelism import m1_kinematics
    from extension.parallelism.m1_kinematics import M1_ROOT_Z_M
    from extension.parallelism.robot_backend import get_robot_backend

    root = torch.tensor([[0.10, -0.20, M1_ROOT_Z_M]], dtype=torch.float64)
    rpy = torch.tensor([[0.05, -0.04, 0.30]], dtype=torch.float64)
    expected = torch.tensor(
        [[0.0, -0.20, 0.40, 0.03, -0.15, 0.30, -0.04, -0.25, 0.50, 0.02, -0.10, 0.20]],
        dtype=torch.float64,
    )
    target = get_robot_backend("m1").fk(root, rpy, expected).foot_pos_w

    original_fk = m1_kinematics.m1_fk
    fk_call_count = 0

    def count_fk_calls(*args, **kwargs):
        nonlocal fk_call_count
        fk_call_count += 1
        return original_fk(*args, **kwargs)

    monkeypatch.setattr(m1_kinematics, "m1_fk", count_fk_calls)
    joint, reachable = get_robot_backend("m1").ik(root, rpy, target)

    assert reachable.all()
    assert fk_call_count <= 1
    torch.testing.assert_close(joint.reshape(1, 12), expected, atol=2.0e-4, rtol=0.0)


def test_m1_rolling_swing_has_no_nominal_lift_but_clears_obstacles():
    from extension.parallelism.m1_kinematics import M1_CFG
    from extension.parallelism.swing import terrain_aware_swing_curve
    from extension.parallelism.types import ParallelismTerrain

    flat = ParallelismTerrain(
        height_w=torch.zeros(1, 11, 11),
        semantic_id=torch.zeros(1, 11, 11, dtype=torch.long),
        valid_mask=torch.ones(1, 11, 11, dtype=torch.bool),
        origin_w=torch.tensor([[-0.5, -0.5, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    obstacle_height = flat.height_w.clone()
    obstacle_height[:, 5, 10] = 0.08
    obstacle = ParallelismTerrain(
        height_w=obstacle_height,
        semantic_id=flat.semantic_id,
        valid_mask=flat.valid_mask,
        origin_w=flat.origin_w,
        yaw_w=flat.yaw_w,
        resolution=flat.resolution,
    )
    start = torch.tensor([[[0.0, 0.0, 0.0]]])
    touchdown = torch.tensor([[[1.0, 0.0, 0.0]]])

    flat_swing = terrain_aware_swing_curve(
        start,
        touchdown,
        flat,
        frames=11,
        clearance_m=0.0,
        min_apex_m=0.0,
    )
    obstacle_swing = terrain_aware_swing_curve(
        start,
        touchdown,
        obstacle,
        frames=11,
        clearance_m=M1_CFG.swing_clearance_m,
        min_apex_m=M1_CFG.min_swing_apex_m,
        terrain_query_radius_m=M1_CFG.swing_terrain_query_radius_m,
    )

    assert M1_CFG.swing_clearance_m >= 0.095958 + M1_CFG.collision_margin_m
    assert M1_CFG.min_swing_apex_m == 0.0
    assert torch.allclose(flat_swing[..., 2], torch.zeros_like(flat_swing[..., 2]))
    assert float(obstacle_swing[..., 2].max().item()) >= 0.08


def test_m1_swing_clearance_includes_wheel_radius_and_collision_margin():
    from extension.parallelism.m1_kinematics import M1_CFG, M1_WHEEL_RADIUS_M

    assert M1_CFG.swing_clearance_m >= M1_WHEEL_RADIUS_M + M1_CFG.collision_margin_m


def test_m1_swing_apex_clears_wheel_envelope_over_local_obstacle():
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_WHEEL_RADIUS_M
    from extension.parallelism.swing import terrain_aware_swing_curve

    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 11, 11),
        semantic_id=torch.zeros(1, 11, 11, dtype=torch.long),
        valid_mask=torch.ones(1, 11, 11, dtype=torch.bool),
        origin_w=torch.tensor([[-0.5, -0.5, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    terrain.height_w[:, 5, 5] = 0.08
    swing = terrain_aware_swing_curve(
        torch.tensor([[[0.0, 0.0, 0.0]]]),
        torch.tensor([[[0.6, 0.0, 0.0]]]),
        terrain,
        frames=11,
        clearance_m=M1_CFG.swing_clearance_m,
        min_apex_m=0.0,
        terrain_query_radius_m=M1_CFG.swing_terrain_query_radius_m,
    )

    assert float(swing[..., 2].max()) >= 0.08 + M1_WHEEL_RADIUS_M + M1_CFG.collision_margin_m


def test_m1_minimum_clearance_swing_does_not_amplify_edge_obstacle_apex():
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_WHEEL_RADIUS_M, M1_WHEEL_THICKNESS_M
    from extension.parallelism.swing import minimum_clearance_swing_curve

    height_m = 0.16
    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 401, 401),
        semantic_id=torch.zeros(1, 401, 401, dtype=torch.long),
        valid_mask=torch.ones(1, 401, 401, dtype=torch.bool),
        origin_w=torch.tensor([[-1.0, -1.0, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.01,
    )
    terrain.height_w[:, 60:81, 100:121] = height_m
    terrain.semantic_id[:, 60:81, 100:121] = 1

    swing = minimum_clearance_swing_curve(
        torch.tensor([[[0.0, -0.30, M1_WHEEL_RADIUS_M]]]),
        torch.tensor([[[0.30, -0.30, M1_WHEEL_RADIUS_M]]]),
        terrain,
        frames=12,
        clearance_m=M1_CFG.swing_clearance_m,
        min_apex_m=M1_CFG.min_swing_apex_m,
        terrain_query_radius_m=M1_CFG.swing_terrain_query_radius_m,
    )

    expected_apex = height_m + M1_CFG.swing_clearance_m
    assert float(swing[..., 2].max()) <= expected_apex + 1.0e-5
    assert float(swing[..., 2].max()) >= expected_apex - 3.0e-4


def test_m1_wheel_side_collision_is_not_treated_as_support_contact():
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.m1_collision import build_m1_surface_points_l
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    root = torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]])
    geometry = backend.fk(root, torch.zeros(1, 3), torch.tensor([M1_DEFAULT_JOINT_POS]))
    wheel_spec = (M1_CFG.official_collision_shapes[3],)
    wheel_points, _ = build_m1_surface_points_l(
        wheel_spec,
        M1_CFG,
        dtype=root.dtype,
        device=root.device,
    )
    side_xy = geometry.foot_pos_w[0, 0, :2] + wheel_points[0, 14, :2]
    origin = torch.tensor([[-1.2, -1.2, 0.0]])
    resolution = 0.02
    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 121, 121),
        semantic_id=torch.zeros(1, 121, 121, dtype=torch.long),
        valid_mask=torch.ones(1, 121, 121, dtype=torch.bool),
        origin_w=origin,
        yaw_w=torch.zeros(1),
        resolution=resolution,
    )
    row = int(torch.round((side_xy[1] - origin[0, 1]) / resolution).item())
    col = int(torch.round((side_xy[0] - origin[0, 0]) / resolution).item())
    terrain.height_w[:, row - 1 : row + 2, col - 1 : col + 2] = 0.20

    def batch_geometry(value):
        if value.ndim == 3:
            return value.unsqueeze(1).unsqueeze(2)
        return value.unsqueeze(1).unsqueeze(2)

    batched_geometry = type(geometry)(
        **{name: batch_geometry(getattr(geometry, name)) for name in geometry.__dataclass_fields__}
    )
    _ok, bits = backend.collision_mask(terrain, batched_geometry, M1_CFG)

    assert bool(bits[0, 0, 0, 3])
