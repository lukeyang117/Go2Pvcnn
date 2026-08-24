from __future__ import annotations

import inspect

import torch


def _terrain():
    from extension.parallelism import ParallelismTerrain

    return ParallelismTerrain(
        height_w=torch.zeros(1, 121, 121),
        semantic_id=torch.zeros(1, 121, 121, dtype=torch.long),
        valid_mask=torch.ones(1, 121, 121, dtype=torch.bool),
        origin_w=torch.tensor([[-6.0, -6.0, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )


def test_go2_planner_default_backend_remains_go2():
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    assert inspect.signature(plan_trajectory).parameters["robot_backend"].default is None
    assert get_robot_backend().name == "go2"


def test_m1_planner_uses_m1_shape_names_and_12_dof_output():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 12),
    )
    result = plan_trajectory(
        state,
        torch.zeros(1, 3),
        _terrain(),
        M1_CFG,
        robot_backend=get_robot_backend("m1"),
    )

    assert result.joint_pos.shape[-1] == 12
    assert result.diagnostics.collision_shape_names[0].startswith("FBL_")
