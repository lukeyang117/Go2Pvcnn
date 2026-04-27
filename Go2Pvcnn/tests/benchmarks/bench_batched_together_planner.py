"""P0 smoke/benchmark coverage for the fixed-shape together planner runtime."""

from __future__ import annotations

import importlib
import time
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import pytest
import torch


def _require_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == name or exc.name.startswith(name + ".")):
            pytest.fail(f"missing required together benchmark module {name!r}")
        raise


def _cfg(**overrides):
    values = {
        "planner_backend": "together",
        "reference_trajectory_horizon": 35,
        "reference_replan_interval_steps": 35,
        "dt": 0.02,
        "planner_instrumentation": True,
        "verbose_planner": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_together_result(num_envs: int, horizon: int, *, device: torch.device):
    types_mod = _require_module("extension.batched_together_planner.types")
    result_cls = getattr(types_mod, "TogetherPlannerResult", None)
    root_pos = torch.zeros(num_envs, horizon, 3, dtype=torch.float32, device=device)
    root_pos[..., 0] = torch.arange(horizon, dtype=torch.float32, device=device).view(1, horizon) * 0.01
    root_rpy = torch.zeros_like(root_pos)
    foot_pos = torch.zeros(num_envs, horizon, 4, 3, dtype=torch.float32, device=device)
    joint_angles = torch.zeros(num_envs, horizon, 12, dtype=torch.float32, device=device)
    contact_state = torch.ones(num_envs, horizon, 4, dtype=torch.bool, device=device)
    event_cap = 2
    payload = {
        "root_pos": root_pos,
        "root_rpy": root_rpy,
        "foot_pos": foot_pos,
        "joint_angles": joint_angles,
        "contact_state": contact_state,
        "touchdown_seq": torch.zeros(num_envs, 4, event_cap, 3, dtype=torch.float32, device=device),
        "touchdown_mask": torch.zeros(num_envs, 4, event_cap, dtype=torch.bool, device=device),
        "cost_total": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "cost_breakdown": {
            "J_td": torch.zeros(num_envs, dtype=torch.float32, device=device),
            "J_swing": torch.zeros(num_envs, dtype=torch.float32, device=device),
            "J_ik": torch.zeros(num_envs, dtype=torch.float32, device=device),
            "J_base": torch.zeros(num_envs, dtype=torch.float32, device=device),
            "J_vel": torch.zeros(num_envs, dtype=torch.float32, device=device),
        },
        "status": torch.zeros(num_envs, dtype=torch.long, device=device),
        "feasible": torch.ones(num_envs, dtype=torch.bool, device=device),
        "safe_fallback": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "joint_limit_violation": torch.zeros(num_envs, horizon, 12, dtype=torch.float32, device=device),
        "workspace_margin": torch.ones(num_envs, horizon, 4, dtype=torch.float32, device=device),
        "support_xy": torch.zeros(num_envs, horizon, 4, 2, dtype=torch.float32, device=device),
        "support_height": torch.zeros(num_envs, horizon, 4, dtype=torch.float32, device=device),
        "support_slope": torch.zeros(num_envs, horizon, 4, dtype=torch.float32, device=device),
    }
    if result_cls is None:
        return SimpleNamespace(**payload)
    if is_dataclass(result_cls):
        names = {field.name for field in fields(result_cls)}
        return result_cls(**{name: value for name, value in payload.items() if name in names})
    return result_cls(**payload)


def _make_ray_hits(num_envs: int, *, device: torch.device) -> torch.Tensor:
    side = 4
    xs = torch.linspace(-0.75, 0.75, side, dtype=torch.float32, device=device)
    ys = torch.linspace(-0.75, 0.75, side, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    zz = torch.zeros_like(xx)
    grid = torch.stack((xx, yy, zz), dim=-1)
    return grid.unsqueeze(0).expand(num_envs, -1, -1, -1).contiguous()


class _FakeScene:
    def __init__(self, robot, scanner):
        self.robot = robot
        self.sensors = {"height_scanner": scanner}

    def __getitem__(self, name):
        return getattr(self, name)


class _FakeRobot:
    def __init__(self, num_envs: int, *, device: torch.device):
        root_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        root_pos[:, 2] = 0.30
        root_quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        root_quat[..., 0] = 1.0
        body_pos = torch.zeros(num_envs, 4, 3, dtype=torch.float32, device=device)
        body_pos[:, :, 2] = 0.0
        self.data = SimpleNamespace(
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            joint_pos=torch.zeros(num_envs, 12, dtype=torch.float32, device=device),
            body_pos_w=body_pos,
        )

    def find_bodies(self, pattern):
        return torch.tensor([0, 1, 2, 3], dtype=torch.long, device=self.data.root_pos_w.device), ["FL", "FR", "RL", "RR"]


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command
        self.version = 0

    def get_command(self, name):
        return self.command

    def set_command(self, command: torch.Tensor):
        self.command = command
        self.version += 1


class _FakeEnv:
    def __init__(self, num_envs: int, *, device: torch.device):
        command = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        command[:, 0] = 0.30
        self.command_manager = _FakeCommandManager(command)
        scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=_make_ray_hits(num_envs, device=device)))
        self.scene = _FakeScene(_FakeRobot(num_envs, device=device), scanner)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.common_step_counter = 0
        self.num_envs = num_envs
        self.device = device
        self.step_dt = 0.02
        self.unwrapped = self
        self._trajectory_reference_cache = None


def _patch_planner(monkeypatch: pytest.MonkeyPatch, *, device: torch.device, call_sizes: list[int]) -> None:
    manager_mod = _require_module("extension.batched_together_planner.manager")
    planner_mod = _require_module("extension.batched_together_planner.planner")

    def fake_plan_segment(terrain, states, commands, planner_cfg=None, *args, **kwargs):
        call_sizes.append(int(commands.shape[0]))
        horizon = int(getattr(planner_cfg, "horizon_steps", 35)) if planner_cfg is not None else 35
        return _fake_together_result(int(commands.shape[0]), horizon, device=device)

    patched = False
    for module in (manager_mod, planner_mod):
        for name in ("plan_segment", "generate_together_trajectory", "batched_generate_together_trajectory"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, fake_plan_segment)
                patched = True
    if not patched:
        pytest.fail("together manager/planner must expose a patchable full-batch planner entrypoint")


def _manager_cache(manager, env):
    return getattr(env.unwrapped, "_trajectory_reference_cache", None) or getattr(manager, "_trajectory_reference_cache", None) or getattr(manager, "_cache", None)


def _current_frame_ids(manager):
    if not hasattr(manager, "current_frame_ids"):
        pytest.fail("TogetherTrajectoryManager must expose current_frame_ids()")
    return manager.current_frame_ids()


@pytest.mark.parametrize("num_envs", [1, 32, 128])
def test_p0_together_smoke_full_batch_cache_and_phase(num_envs: int, monkeypatch: pytest.MonkeyPatch):
    manager_mod = _require_module("extension.batched_together_planner.manager")
    device = torch.device("cpu")
    call_sizes: list[int] = []
    _patch_planner(monkeypatch, device=device, call_sizes=call_sizes)

    env = _FakeEnv(num_envs, device=device)
    manager = manager_mod.TogetherTrajectoryManager(_cfg(), device=device)

    manager.refresh_from_env(env)
    assert call_sizes == [num_envs], f"initial planner call must be full batch N={num_envs}"
    cache = _manager_cache(manager, env)
    assert cache is not None and cache.is_ready()
    assert tuple(cache.root_pos_w.shape) == (num_envs, 35, 3)
    assert cache.root_pos_w.device == device
    torch.testing.assert_close(_current_frame_ids(manager), torch.zeros(num_envs, dtype=torch.long, device=device), atol=0, rtol=0)

    manager.refresh_from_env(env)
    assert call_sizes == [num_envs], "same env step must not launch another planner call"
    torch.testing.assert_close(_current_frame_ids(manager), torch.zeros(num_envs, dtype=torch.long, device=device), atol=0, rtol=0)

    env.episode_length_buf += 1
    env.common_step_counter += 1
    manager.refresh_from_env(env)
    assert call_sizes == [num_envs], "no host trigger before 0.7s interval should avoid planner call"
    torch.testing.assert_close(_current_frame_ids(manager), torch.ones(num_envs, dtype=torch.long, device=device), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark optional; skipped when CUDA is unavailable")
@pytest.mark.parametrize("num_envs", [1024, 4096])
def test_optional_cuda_smoke_records_full_batch_size(num_envs: int, monkeypatch: pytest.MonkeyPatch):
    manager_mod = _require_module("extension.batched_together_planner.manager")
    device = torch.device("cuda:0")
    call_sizes: list[int] = []
    _patch_planner(monkeypatch, device=device, call_sizes=call_sizes)

    env = _FakeEnv(num_envs, device=device)
    manager = manager_mod.TogetherTrajectoryManager(_cfg(), device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    manager.refresh_from_env(env)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    cache = _manager_cache(manager, env)
    allocated_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    assert call_sizes == [num_envs]
    assert tuple(cache.root_pos_w.shape) == (num_envs, 35, 3)
    print(f"[together-smoke] N={num_envs} elapsed_ms={elapsed_ms:.3f} cuda_allocated_mb={allocated_mb:.2f}")
