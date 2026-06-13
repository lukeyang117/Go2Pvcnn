from __future__ import annotations

import math
import importlib.util
import sys
import types
import copy
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


def _install_fake_isaaclab(monkeypatch) -> None:
    isaaclab_module = types.ModuleType("isaaclab")
    envs_module = types.ModuleType("isaaclab.envs")
    mdp_module = types.ModuleType("isaaclab.envs.mdp")
    managers_module = types.ModuleType("isaaclab.managers")
    utils_module = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")

    def configclass(cls):
        def __init__(self, **kwargs):
            for base in reversed(cls.mro()):
                for name, value in base.__dict__.items():
                    if name.startswith("__") or callable(value) or isinstance(
                        value, (staticmethod, classmethod, property, type)
                    ):
                        continue
                    setattr(self, name, copy.deepcopy(value))
            for key, value in kwargs.items():
                setattr(self, key, value)

        cls.__init__ = __init__
        return cls

    def wrap_to_pi(value):
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    class CommandTerm:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.metrics = {}
            self.time_left = torch.zeros(env.num_envs, device=env.device)
            self.command_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        @property
        def num_envs(self):
            return self._env.num_envs

        @property
        def device(self):
            return self._env.device

        def reset(self, env_ids=None):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self._resample_command(env_ids)

        def compute(self, _dt):
            self._update_metrics()
            self._update_command()

    class CommandTermCfg:
        class_type = None
        resampling_time_range = (100.0, 100.0)
        debug_vis = False

    class UniformVelocityCommandCfg:
        class Ranges:
            def __init__(self, lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=None):
                self.lin_vel_x = lin_vel_x
                self.lin_vel_y = lin_vel_y
                self.ang_vel_z = ang_vel_z
                self.heading = heading

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    mdp_module.UniformVelocityCommandCfg = UniformVelocityCommandCfg
    managers_module.CommandTerm = CommandTerm
    managers_module.CommandTermCfg = CommandTermCfg
    utils_module.configclass = configclass
    math_module.wrap_to_pi = wrap_to_pi
    isaaclab_module.envs = envs_module
    envs_module.mdp = mdp_module
    isaaclab_module.managers = managers_module
    isaaclab_module.utils = utils_module
    utils_module.math = math_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs_module)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", mdp_module)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)


class _FakeRobotData:
    def __init__(self, *, device: str, num_envs: int):
        self.root_pos_w = torch.zeros(num_envs, 3, device=device)
        self.heading_w = torch.zeros(num_envs, device=device)
        self.root_lin_vel_b = torch.zeros(num_envs, 3, device=device)
        self.root_ang_vel_b = torch.zeros(num_envs, 3, device=device)


class _FakeRobot:
    def __init__(self, *, device: str = "cpu", num_envs: int = 2):
        self.data = _FakeRobotData(device=device, num_envs=num_envs)


class _FakeEnv:
    def __init__(self, *, device: str = "cpu", num_envs: int = 2):
        self.device = device
        self.num_envs = num_envs
        self.step_dt = 0.02
        self.scene = {"robot": _FakeRobot(device=device, num_envs=num_envs)}


def _make_command(monkeypatch, *, num_envs: int = 2):
    _install_fake_isaaclab(monkeypatch)
    module_path = GO2PVCNN_ROOT / "go2_pvcnn/mdp/commands/velocity_command.py"
    spec = importlib.util.spec_from_file_location("goal_anchored_velocity_command_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    GoalAnchoredVelocityCommand = module.GoalAnchoredVelocityCommand
    GoalAnchoredVelocityCommandCfg = module.GoalAnchoredVelocityCommandCfg

    env = _FakeEnv(num_envs=num_envs)
    cfg = GoalAnchoredVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(100.0, 100.0),
        goal_distance=10.0,
        goal_reached_threshold=1.0,
        vx_abs_range=(0.6, 1.0),
        vy_abs_range=(0.6, 1.0),
        yaw_stiffness=0.5,
        yaw_range=(-0.8, 0.8),
        rel_standing_envs=0.0,
    )
    return env, GoalAnchoredVelocityCommand(cfg, env)


def test_goal_anchored_command_resample_initializes_goal_and_speed(monkeypatch) -> None:
    env, command = _make_command(monkeypatch, num_envs=4)

    command._resample_command(torch.arange(4))

    assert command.command.shape == (4, 3)
    goal_offsets = command.goal_xy_w - env.scene["robot"].data.root_pos_w[:, :2]
    assert torch.linalg.norm(goal_offsets, dim=1).tolist() == pytest.approx([10.0] * 4)
    assert torch.all(command.vx_abs >= 0.6)
    assert torch.all(command.vx_abs <= 1.0)
    assert torch.all(command.vy_abs >= 0.6)
    assert torch.all(command.vy_abs <= 1.0)


def test_goal_anchored_command_updates_xy_signs_from_body_quadrant(monkeypatch) -> None:
    env, command = _make_command(monkeypatch, num_envs=1)
    robot = env.scene["robot"]
    command.goal_xy_w[0] = torch.tensor([10.0, 10.0])
    command.vx_abs[0] = 0.7
    command.vy_abs[0] = 0.9

    robot.data.heading_w[0] = 0.0
    command._update_command()
    assert command.command[0, 0].item() == pytest.approx(0.7)
    assert command.command[0, 1].item() == pytest.approx(0.9)

    robot.data.heading_w[0] = math.pi
    command._update_command()
    assert command.command[0, 0].item() == pytest.approx(-0.7)
    assert command.command[0, 1].item() == pytest.approx(-0.9)


def test_goal_anchored_command_clamps_yaw_rate(monkeypatch) -> None:
    env, command = _make_command(monkeypatch, num_envs=1)
    robot = env.scene["robot"]
    command.goal_xy_w[0] = torch.tensor([0.0, 10.0])
    command.vx_abs[0] = 0.8
    command.vy_abs[0] = 0.8
    command.cfg.yaw_stiffness = 10.0
    command.cfg.yaw_range = (-0.8, 0.8)
    robot.data.heading_w[0] = 0.0

    command._update_command()

    assert command.command[0, 2].item() == pytest.approx(0.8)


def test_goal_anchored_command_extends_reached_goal(monkeypatch) -> None:
    env, command = _make_command(monkeypatch, num_envs=1)
    robot = env.scene["robot"]
    robot.data.root_pos_w[0, :2] = torch.tensor([9.5, 0.0])
    command.goal_xy_w[0] = torch.tensor([10.0, 0.0])
    command.vx_abs[0] = 0.8
    command.vy_abs[0] = 0.8

    command._update_command()

    assert command.goal_xy_w[0].tolist() == pytest.approx([19.5, 0.0])
