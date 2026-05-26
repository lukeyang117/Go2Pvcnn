from __future__ import annotations

import sys
import signal
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
from scripts import play


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


def test_viewer_main_builds_only_selected_backend_planner_cfgs(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(viewer, "_parse_args", lambda: SimpleNamespace(planner_backend="mpc", plan_dt=0.02, n_frames=25, livestream=-1))
    monkeypatch.setattr(viewer, "_prepare_runtime_args", lambda args: args)
    monkeypatch.setattr(viewer, "_launch_app", lambda _args: (None, SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(viewer, "_build_env_cfg", lambda _args: SimpleNamespace())
    monkeypatch.setattr(viewer, "_build_planner_cfg", lambda _env_cfg: calls.append("legacy") or SimpleNamespace())
    monkeypatch.setattr(viewer, "_build_together_planner_cfg", lambda _env_cfg: calls.append("together") or SimpleNamespace())
    monkeypatch.setattr(viewer, "_build_mpc_planner_cfg", lambda _env_cfg, args_cli=None: calls.append("mpc") or SimpleNamespace())

    class _Stop(Exception):
        pass

    def _stop_make(*_args, **_kwargs):
        raise _Stop()

    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=_stop_make))
    monkeypatch.setitem(sys.modules, "go2_pvcnn.tasks.register_envs", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.envs",
        SimpleNamespace(ManagerBasedRLEnv=object),
    )

    with pytest.raises(_Stop):
        viewer.main()

    assert calls == ["legacy", "mpc"]


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


def test_viewer_step_mode_defers_command_replan_until_current_trajectory_finishes() -> None:
    result = SimpleNamespace(num_frames=5)
    previous = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
    changed = torch.tensor([[0.0, 0.3, 0.0]], dtype=torch.float64)

    assert not viewer._viewer_loop_need_replan(
        result=result,
        playback_frame=3,
        reset_requested=False,
        teleop_values=changed,
        last_cmd=previous,
        defer_command_replan_until_trajectory_end=True,
    )
    assert viewer._viewer_loop_need_replan(
        result=result,
        playback_frame=5,
        reset_requested=False,
        teleop_values=changed,
        last_cmd=previous,
        defer_command_replan_until_trajectory_end=True,
    )


def test_viewer_step_gate_requires_space_for_each_frame() -> None:
    gate = viewer.ViewerStepGate(enabled=True)

    assert not gate.consume_frame_permission(step_requested=False)
    assert gate.consume_frame_permission(step_requested=True)
    assert not gate.consume_frame_permission(step_requested=False)


def test_viewer_step_gate_can_toggle_runtime_mode() -> None:
    gate = viewer.ViewerStepGate(enabled=False)

    assert gate.consume_frame_permission(step_requested=False)
    assert gate.toggle_enabled()
    assert not gate.consume_frame_permission(step_requested=False)
    assert gate.consume_frame_permission(step_requested=True)
    assert not gate.toggle_enabled()
    assert gate.consume_frame_permission(step_requested=False)


def test_viewer_teleop_signal_handler_removes_guards_before_interrupt(monkeypatch) -> None:
    teleop = viewer.TerminalTeleop(
        device=torch.device("cpu"),
        vx_scale=1.0,
        vy_scale=1.0,
        yaw_scale=1.0,
        timeout_s=0.1,
    )
    calls: list[str] = []
    monkeypatch.setattr(teleop, "_remove_cleanup_guards", lambda: calls.append("remove"))
    monkeypatch.setattr(teleop, "_restore_terminal_state", lambda: calls.append("restore"))

    with pytest.raises(KeyboardInterrupt):
        teleop._handle_signal(signal.SIGINT, None)

    assert calls == ["remove", "restore"]


def test_viewer_step_mode_defers_command_replan_only_while_enabled() -> None:
    result = SimpleNamespace(num_frames=5)
    previous = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
    changed = torch.tensor([[0.0, 0.3, 0.0]], dtype=torch.float64)

    assert viewer._viewer_loop_need_replan(
        result=result,
        playback_frame=3,
        reset_requested=False,
        teleop_values=changed,
        last_cmd=previous,
        defer_command_replan_until_trajectory_end=False,
    )


def test_viewer_selects_latched_command_while_step_mode_enabled() -> None:
    live = torch.zeros((1, 3), dtype=torch.float64)
    latched = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)

    selected = viewer._viewer_select_active_teleop_values(
        live_values=live,
        latched_values=latched,
        step_mode_enabled=True,
    )

    torch.testing.assert_close(selected, latched)


def test_viewer_rotates_mpc_body_frame_command_by_root_yaw() -> None:
    command = torch.tensor([[0.4, 0.0, 0.2]], dtype=torch.float64)
    state = SimpleNamespace(root_rpy=torch.tensor([[0.0, 0.0, torch.pi / 2.0]], dtype=torch.float64))

    world_command = viewer._viewer_mpc_world_command_from_root_frame(command, state)

    torch.testing.assert_close(
        world_command,
        torch.tensor([[0.0, 0.4, 0.2]], dtype=torch.float64),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_viewer_mpc_cfg_keeps_fixed_cycle_horizon_when_viewer_requests_long_playback() -> None:
    env_cfg = SimpleNamespace(
        plan_dt=0.02,
        reference_trajectory_horizon=300,
        reference_replan_interval_steps=300,
    )

    cfg = viewer._build_mpc_planner_cfg(env_cfg)

    assert cfg.runtime.horizon_steps == 25


def test_play_step_gate_disabled_does_not_block() -> None:
    gate = play._TerminalStepGate(enabled=False)

    assert gate.wait_for_step()


def test_viewer_step_mode_paused_loop_keeps_rendering_window() -> None:
    base_env, _, scene, sim = _fake_base_env()

    viewer._viewer_pump_paused_window(base_env, sleep_s=0.0)

    assert sim.render_count == 1
    assert scene.update_count == 1


def test_viewer_step_mode_updates_visualizer_only_when_frame_is_permitted() -> None:
    calls = []

    def record_update() -> None:
        calls.append("update")

    viewer._viewer_update_visualizer_when_permitted(frame_permitted=False, update_fn=record_update)
    viewer._viewer_update_visualizer_when_permitted(frame_permitted=True, update_fn=record_update)

    assert calls == ["update"]
