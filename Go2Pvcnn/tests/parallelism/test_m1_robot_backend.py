from __future__ import annotations

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
