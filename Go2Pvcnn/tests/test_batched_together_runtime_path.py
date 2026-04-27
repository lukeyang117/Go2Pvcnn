"""Runtime path tests for the together trajectory manager contract."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from extension.reference.cache import ReferenceTrajectoryCache


def _require_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == name or exc.name.startswith(name + ".")):
            pytest.fail(f"missing required runtime module {name!r}")
        raise


def _manager_factory() -> Callable:
    factory = _require_module("extension.trajectory_manager_factory")
    for name in ("create_trajectory_manager", "make_trajectory_manager", "build_trajectory_manager"):
        fn = getattr(factory, name, None)
        if callable(fn):
            return fn
    pytest.fail("extension.trajectory_manager_factory must expose create_trajectory_manager(cfg, device)")


def _create_manager(cfg, *, device: torch.device):
    create = _manager_factory()
    try:
        return create(cfg, device=device)
    except TypeError:
        return create(cfg, device)


def _fake_cache(num_envs: int, horizon: int, *, device: torch.device | str = "cpu") -> ReferenceTrajectoryCache:
    device = torch.device(device)
    root_pos = torch.arange(num_envs * horizon * 3, dtype=torch.float32, device=device).reshape(num_envs, horizon, 3)
    root_quat = torch.zeros(num_envs, horizon, 4, dtype=torch.float32, device=device)
    root_quat[..., 0] = 1.0
    joint_angles = torch.arange(num_envs * horizon * 12, dtype=torch.float32, device=device).reshape(num_envs, horizon, 12)
    foot_pos_root = torch.arange(num_envs * horizon * 4 * 3, dtype=torch.float32, device=device).reshape(num_envs, horizon, 4, 3)
    contact_state = torch.ones(num_envs, horizon, 4, dtype=torch.bool, device=device)
    touchdown = torch.arange(num_envs * horizon * 4 * 3, dtype=torch.float32, device=device).reshape(num_envs, horizon, 4, 3)
    phase_index = torch.arange(horizon, dtype=torch.long, device=device).unsqueeze(0).expand(num_envs, -1).clone()
    valid_mask = torch.ones(num_envs, horizon, dtype=torch.bool, device=device)
    return ReferenceTrajectoryCache(
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        joint_angles=joint_angles,
        foot_pos_root=foot_pos_root,
        contact_state=contact_state,
        planned_touchdown_w=touchdown,
        phase_index=phase_index,
        valid_mask=valid_mask,
    )


class _FrameManager:
    planner_backend = "together"

    def __init__(self, cache: ReferenceTrajectoryCache, frame_ids: torch.Tensor):
        self.cache = cache
        self.frame_ids = frame_ids.clone()
        self.refresh_calls = 0
        self._trajectory_reference_cache = cache

    def refresh_from_env(self, env):
        self.refresh_calls += 1
        env.unwrapped._trajectory_reference_cache = self.cache
        return self.cache

    def current_frame_ids(self):
        return self.frame_ids.clone()

    def current_reference(self):
        env_idx = torch.arange(self.frame_ids.shape[0], device=self.frame_ids.device)
        idx = self.frame_ids
        return {
            "root_pos_w": self.cache.root_pos_w[env_idx, idx],
            "root_quat_w": self.cache.root_quat_w[env_idx, idx],
            "joint_angles": self.cache.joint_angles[env_idx, idx],
            "foot_pos_root": self.cache.foot_pos_root[env_idx, idx],
            "contact_state": self.cache.contact_state[env_idx, idx],
            "planned_touchdown_w": self.cache.planned_touchdown_w[env_idx, idx],
            "phase_index": self.cache.phase_index[env_idx, idx],
            "valid_mask": self.cache.valid_mask[env_idx, idx],
        }


class _IdempotentFrameManager(_FrameManager):
    def __init__(self, cache: ReferenceTrajectoryCache, num_envs: int):
        super().__init__(cache, torch.zeros(num_envs, dtype=torch.long, device=cache.root_pos_w.device))
        self._last_step_token = None
        self.advance_count = 0

    def refresh_from_env(self, env):
        self.refresh_calls += 1
        token = getattr(env, "step_token", None)
        if token != self._last_step_token:
            self.frame_ids = torch.clamp(self.frame_ids + 1, max=self.cache.horizon_length() - 1)
            self.advance_count += 1
            self._last_step_token = token
        env.unwrapped._trajectory_reference_cache = self.cache
        return self.cache


class _FakeEnv:
    def __init__(self, manager, *, episode_length_buf: torch.Tensor):
        self.unwrapped = self
        self._trajectory_manager = manager
        self._trajectory_reference_cache = manager.cache
        self.episode_length_buf = episode_length_buf
        self.num_envs = int(episode_length_buf.shape[0])
        self.device = episode_length_buf.device
        self.cfg = SimpleNamespace(reference_trajectory_horizon=int(manager.cache.horizon_length()))
        self.step_token = 0


def _cfg(**overrides):
    values = {
        "reference_trajectory_horizon": 35,
        "reference_replan_interval_steps": 35,
        "dt": 0.02,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_backend_factory_defaults_to_together_manager():
    together_manager_mod = _require_module("extension.batched_together_planner.manager")
    manager = _create_manager(_cfg(), device=torch.device("cpu"))
    assert isinstance(manager, together_manager_mod.TogetherTrajectoryManager)


def test_backend_factory_legacy_rollback_uses_existing_manager():
    from extension.batched_planner.manager import BatchedTrajectoryManager

    manager = _create_manager(_cfg(planner_backend="legacy"), device=torch.device("cpu"))
    assert isinstance(manager, BatchedTrajectoryManager)


def test_backend_factory_rejects_unknown_backend():
    with pytest.raises((KeyError, ValueError), match="planner_backend|backend|unknown|invalid"):
        _create_manager(_cfg(planner_backend="does-not-exist"), device=torch.device("cpu"))


def test_reward_frame_ids_come_from_manager_not_episode_modulo():
    rewards_reference = _require_module("extension.mdp.rewards_reference")
    cache = _fake_cache(num_envs=2, horizon=5)
    manager = _FrameManager(cache, torch.tensor([3, 1], dtype=torch.long))
    env = _FakeEnv(manager, episode_length_buf=torch.tensor([0, 0], dtype=torch.long))

    _cache, frame_ids = rewards_reference._select_reference_frame(env)

    torch.testing.assert_close(frame_ids, torch.tensor([3, 1], dtype=torch.long), atol=0, rtol=0)


def test_multiple_reward_terms_same_step_do_not_advance_phase_twice():
    rewards_reference = _require_module("extension.mdp.rewards_reference")
    cache = _fake_cache(num_envs=2, horizon=5)
    manager = _IdempotentFrameManager(cache, num_envs=2)
    env = _FakeEnv(manager, episode_length_buf=torch.tensor([99, 99], dtype=torch.long))
    env.step_token = 42

    _cache0, frame_ids0 = rewards_reference._select_reference_frame(env)
    _cache1, frame_ids1 = rewards_reference._select_reference_frame(env)

    torch.testing.assert_close(frame_ids0, torch.tensor([1, 1], dtype=torch.long), atol=0, rtol=0)
    torch.testing.assert_close(frame_ids1, frame_ids0, atol=0, rtol=0)
    assert manager.advance_count == 1


def test_viewer_together_cfg_ignores_legacy_tuning_fields():
    viewer = _require_module("extension.viz.go2_foostep_planner")
    from extension.batched_together_planner.config import TogetherPlannerConfig

    raw_defaults = TogetherPlannerConfig()
    cfg = viewer._build_together_planner_cfg(
        SimpleNamespace(
            reference_trajectory_horizon=raw_defaults.horizon_steps,
            plan_dt=raw_defaults.dt,
            step_freq=2.25,
            step_height=0.11,
            duty_factor=0.60,
            foothold_search_radius=0.15,
            foothold_search_step=0.03,
            replan_stop_speed=0.05,
        )
    )

    assert cfg.step_freq == 2.25
    assert cfg.swing_height == 0.11
    assert cfg.duty_factor == raw_defaults.duty_factor == 0.55
    assert cfg.support_search_radius == raw_defaults.support_search_radius == 0.04
    assert cfg.support_search_step == raw_defaults.support_search_step == 0.02


def test_viewer_together_cfg_allows_explicit_together_fields():
    viewer = _require_module("extension.viz.go2_foostep_planner")
    from extension.batched_together_planner.config import TogetherPlannerConfig

    raw_defaults = TogetherPlannerConfig()
    cfg = viewer._build_together_planner_cfg(
        SimpleNamespace(
            reference_trajectory_horizon=raw_defaults.horizon_steps,
            plan_dt=raw_defaults.dt,
            step_freq=raw_defaults.step_freq,
            step_height=raw_defaults.swing_height,
            together_duty_factor=0.62,
            idle_command_eps=0.004,
            support_search_radius=0.07,
            support_search_step=0.025,
            duty_factor=0.60,
            foothold_search_radius=0.15,
            foothold_search_step=0.03,
        )
    )

    assert cfg.duty_factor == 0.62
    assert cfg.idle_command_eps == 0.004
    assert cfg.support_search_radius == 0.07
    assert cfg.support_search_step == 0.025


def _fake_together_result(command: torch.Tensor, *, horizon: int = 5):
    num_envs = int(command.shape[0])
    root_pos = torch.zeros((num_envs, horizon, 3), dtype=torch.float64, device=command.device)
    root_rpy = torch.zeros_like(root_pos)
    root_pos[:, -1, 0] = command[:, 0]
    root_pos[:, -1, 1] = command[:, 1]
    root_rpy[:, -1, 2] = command[:, 2]
    return SimpleNamespace(
        root_pos=root_pos,
        root_rpy=root_rpy,
        foot_pos=torch.zeros((num_envs, horizon, 4, 3), dtype=torch.float64, device=command.device),
        joint_angles=torch.zeros((num_envs, horizon, 12), dtype=torch.float64, device=command.device),
        contact_state=torch.ones((num_envs, horizon, 4), dtype=torch.bool, device=command.device),
        touchdown_seq=torch.zeros((num_envs, 4, 2, 3), dtype=torch.float64, device=command.device),
    )


def test_viewer_together_backend_calls_plan_segment_with_active_command(monkeypatch):
    viewer = _require_module("extension.viz.go2_foostep_planner")
    together_planner = _require_module("extension.batched_together_planner.planner")
    legacy_planner = _require_module("extension.batched_planner.trajectory")
    command = torch.tensor([[0.31, -0.12, 0.07]], dtype=torch.float64)
    calls = {}

    def fake_plan_segment(terrain, state, command_batch, cfg=None):
        calls["terrain"] = terrain
        calls["state"] = state
        calls["command"] = command_batch.clone()
        calls["cfg"] = cfg
        return _fake_together_result(command_batch, horizon=5)

    def fail_legacy(*args, **kwargs):
        pytest.fail("together viewer backend must not call legacy batched_generate_trajectory")

    monkeypatch.setattr(together_planner, "plan_segment", fake_plan_segment)
    monkeypatch.setattr(legacy_planner, "batched_generate_trajectory", fail_legacy)

    terrain = object()
    state = object()
    cfg = object()
    result = viewer._plan_viewer_trajectory(
        backend="together",
        terrain=terrain,
        state=state,
        command=command,
        requested_n_frames=5,
        dt=0.02,
        legacy_cfg=object(),
        together_cfg=cfg,
    )

    assert calls["terrain"] is terrain
    assert calls["state"] is state
    assert calls["cfg"] is cfg
    torch.testing.assert_close(calls["command"], command)
    assert result.num_frames == 5
    torch.testing.assert_close(result.root_pos_w[:, -1, :2], command[:, :2])
    torch.testing.assert_close(result.root_quat_w[:, 0], torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64))
    assert result.foot_pos_w.shape == (1, 5, 4, 3)
    assert result.planned_touchdown_w.shape == (1, 4, 3)


def test_viewer_legacy_backend_preserves_batched_generate_path(monkeypatch):
    viewer = _require_module("extension.viz.go2_foostep_planner")
    together_planner = _require_module("extension.batched_together_planner.planner")
    legacy_planner = _require_module("extension.batched_planner.trajectory")
    command = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
    sentinel_result = SimpleNamespace(num_frames=3)
    calls = {}

    def fail_together(*args, **kwargs):
        pytest.fail("legacy viewer backend must not call together plan_segment")

    def fake_batched_generate(terrain, state, commands, requested_n_frames, dt=0.02, cfg=None, **kwargs):
        calls["terrain"] = terrain
        calls["state"] = state
        calls["command"] = commands.clone()
        calls["requested_n_frames"] = requested_n_frames
        calls["dt"] = dt
        calls["cfg"] = cfg
        return sentinel_result

    monkeypatch.setattr(together_planner, "plan_segment", fail_together)
    monkeypatch.setattr(legacy_planner, "batched_generate_trajectory", fake_batched_generate)

    terrain = object()
    state = object()
    cfg = object()
    result = viewer._plan_viewer_trajectory(
        backend="legacy",
        terrain=terrain,
        state=state,
        command=command,
        requested_n_frames=7,
        dt=0.03,
        legacy_cfg=cfg,
        together_cfg=object(),
    )

    assert result is sentinel_result
    assert calls["terrain"] is terrain
    assert calls["state"] is state
    assert calls["requested_n_frames"] == 7
    assert calls["dt"] == 0.03
    assert calls["cfg"] is cfg
    torch.testing.assert_close(calls["command"], command)
