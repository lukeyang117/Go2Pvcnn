from __future__ import annotations

from types import SimpleNamespace

import torch


def test_viewer_source_has_robot_selector_and_bounded_cycle_option():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "extension/viz/go2_foostep_planner.py").read_text()
    assert 'choices=["go2", "m1"]' in source
    assert "max_plan_cycles" in source
    assert "M1ParallelismViewerEnvCfg" in source
    registration = (Path(__file__).resolve().parents[1] / "go2_pvcnn/tasks/register_envs.py").read_text()
    assert "Isaac-M1-Parallelism-Viewer-v0" in registration


def test_m1_playback_overwrites_only_12_leg_joints():
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.viz import go2_foostep_planner as viewer

    current = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    planner = torch.full((1, 12), 9.0)
    merged = viewer.merge_planner_joints_into_robot(current, planner, get_robot_backend("m1"))

    torch.testing.assert_close(merged[:, [3, 7, 11, 15]], current[:, [3, 7, 11, 15]])
    assert torch.all(merged[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]] == 9.0)


def test_m1_direct_playback_preserves_current_wheel_values():
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.viz import go2_foostep_planner as viewer

    class Robot:
        joint_names = list(get_robot_backend("m1").asset_joint_names)

        def __init__(self):
            self.data = SimpleNamespace(joint_pos=torch.arange(16, dtype=torch.float32).reshape(1, 16))
            self.written = None

        def write_joint_state_to_sim(self, joint_pos, joint_vel):
            self.written = (joint_pos.clone(), joint_vel.clone())

    robot = Robot()
    result = SimpleNamespace(
        root_pos_w=torch.zeros(1, 1, 3),
        root_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        joint_angles=torch.full((1, 1, 12), 9.0),
    )
    viewer._apply_direct_playback_to_robot(robot, result, frame_idx=0, robot_backend=get_robot_backend("m1"))

    assert robot.written is not None
    torch.testing.assert_close(robot.written[0][:, [3, 7, 11, 15]], torch.tensor([[3.0, 7.0, 11.0, 15.0]]))
