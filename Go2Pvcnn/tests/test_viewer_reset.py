from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.batch_mpc_planner.types import MpcPlannerTerrain
from extension.viz import go2_foostep_planner as viewer


class _FakeRobot:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            root_lin_vel_w=torch.tensor([[0.2, 0.0, 0.1]], dtype=torch.float32),
            root_ang_vel_w=torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32),
            joint_pos=torch.full((1, 12), 0.25, dtype=torch.float32),
            joint_vel=torch.full((1, 12), 0.4, dtype=torch.float32),
            body_pos_w=torch.tensor(
                [
                    [
                        [0.2, 0.1, 0.2],
                        [0.2, -0.1, 0.2],
                        [-0.2, 0.1, 0.2],
                        [-0.2, -0.1, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ),
        )
        self.last_root_pose = None
        self.last_root_vel = None
        self.last_joint_pos = None
        self.last_joint_vel = None

    def write_root_pose_to_sim(self, root_pose):
        root_pose = torch.as_tensor(root_pose, dtype=torch.float32).clone()
        self.last_root_pose = root_pose
        self.data.root_pos_w = root_pose[:, :3]
        self.data.root_quat_w = root_pose[:, 3:7]

    def write_root_velocity_to_sim(self, root_vel):
        root_vel = torch.as_tensor(root_vel, dtype=torch.float32).clone()
        self.last_root_vel = root_vel
        self.data.root_lin_vel_w = root_vel[:, :3]
        self.data.root_ang_vel_w = root_vel[:, 3:6]

    def write_joint_state_to_sim(self, joint_pos, joint_vel):
        self.last_joint_pos = torch.as_tensor(joint_pos, dtype=torch.float32).clone()
        self.last_joint_vel = torch.as_tensor(joint_vel, dtype=torch.float32).clone()
        self.data.joint_pos = self.last_joint_pos
        self.data.joint_vel = self.last_joint_vel


class _FakeScene:
    def __init__(self, robot: _FakeRobot) -> None:
        self.robot = robot
        self.write_count = 0
        self.update_count = 0

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot

    def write_data_to_sim(self):
        self.write_count += 1

    def update(self, _dt: float):
        self.update_count += 1


class _FakeSim:
    def __init__(self) -> None:
        self.render_count = 0

    def render(self):
        self.render_count += 1


def _fake_base_env(command: torch.Tensor | None = None):
    robot = _FakeRobot()
    scene = _FakeScene(robot)
    sim = _FakeSim()
    command_manager = None
    if command is not None:
        command_manager = SimpleNamespace(get_command=lambda _name: command)
    return SimpleNamespace(scene=scene, sim=sim, physics_dt=0.02, command_manager=command_manager), robot, scene, sim


def test_viewer_zero_base_command_clears_command_tensor() -> None:
    command = torch.tensor([[0.3, -0.1, 0.4]], dtype=torch.float32)
    base_env, _, _, _ = _fake_base_env(command)

    viewer._viewer_zero_base_command(base_env)

    torch.testing.assert_close(command, torch.zeros_like(command))


def test_viewer_apply_reset_snapshot_restores_root_and_joint_state() -> None:
    base_env, robot, scene, sim = _fake_base_env()
    snapshot = viewer.ViewerResetSnapshot(
        joint_pos=torch.zeros((1, 12), dtype=torch.float32),
        joint_vel=torch.zeros((1, 12), dtype=torch.float32),
    )
    root_pos = torch.tensor([[1.0, 2.0, 0.6]], dtype=torch.float32)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    viewer._viewer_apply_joint_reset_snapshot(base_env, snapshot, root_pos_w=root_pos, root_quat_w=root_quat)

    torch.testing.assert_close(robot.data.root_pos_w, root_pos)
    torch.testing.assert_close(robot.data.joint_pos, snapshot.joint_pos)
    torch.testing.assert_close(robot.data.root_lin_vel_w, torch.zeros_like(robot.data.root_lin_vel_w))
    assert scene.write_count == 1
    assert scene.update_count == 1
    assert sim.render_count == 1


def test_viewer_ground_robot_from_scanner_shifts_root_z_to_match_ground(monkeypatch) -> None:
    base_env, robot, scene, sim = _fake_base_env()
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    monkeypatch.setattr(viewer, "_compute_mpc_local_terrain", lambda scanner, env_id=0: (terrain, None))

    z_shift = viewer._viewer_ground_robot_from_scanner(base_env, object(), [0, 1, 2, 3])

    assert z_shift == pytest.approx(-0.2)
    torch.testing.assert_close(robot.data.root_pos_w[:, 2], torch.tensor([0.3], dtype=torch.float32))
    torch.testing.assert_close(robot.data.root_lin_vel_w, torch.zeros_like(robot.data.root_lin_vel_w))
    assert scene.write_count == 1
    assert scene.update_count == 1
    assert sim.render_count == 1
