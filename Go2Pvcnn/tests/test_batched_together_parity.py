"""A/B behavior and raw parity tests for ``batched_together_planner``.

These tests intentionally define the T106 contract before the together backend
exists.  Raw ``batch_planner`` is used only as a CPU reference fixture here; the
together training path must stay native torch/GPU.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from scripts.go2fp.trajectory import default_initial_state as raw_default_initial_state
from scripts.go2fp.types import HIP_HEIGHT, HIP_OFFSETS_ARRAY
from scripts.go2fp.batch_planner.config import BatchPlannerConfig as RawBatchPlannerConfig
from scripts.go2fp.batch_planner.planner import plan_segment as raw_plan_segment
from scripts.go2fp.batch_planner.schedule import build_fixed_schedule as raw_build_fixed_schedule
from scripts.go2fp.batch_planner.terrain import TerrainBatch as RawTerrainBatch
from scripts.go2fp.batch_planner.types import PlannerStatus as RawPlannerStatus


def _require_together_modules() -> SimpleNamespace:
    modules = {}
    missing = []
    for name in ("config", "planner", "schedule", "terrain", "types"):
        module_name = f"extension.batched_together_planner.{name}"
        try:
            modules[name] = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("extension.batched_together_planner"):
                missing.append(module_name)
            else:
                raise
    if missing:
        pytest.fail(
            "batched_together_planner modules are missing: "
            + ", ".join(missing)
            + ". T106 expects a raw-aligned together backend under extension/batched_together_planner/."
        )
    return SimpleNamespace(**modules)


def _cfg(together: SimpleNamespace):
    cls = getattr(together.config, "BatchPlannerConfig", None) or getattr(together.config, "TogetherPlannerConfig", None)
    if cls is None:
        pytest.fail("together config module must expose BatchPlannerConfig or TogetherPlannerConfig")
    return cls()


def _terrain_cls(together: SimpleNamespace):
    cls = getattr(together.terrain, "TerrainBatch", None) or getattr(together.terrain, "TogetherPlannerTerrain", None)
    if cls is None:
        pytest.fail("together terrain module must expose TerrainBatch or TogetherPlannerTerrain")
    return cls


def _state_for_together(together: SimpleNamespace, state: dict[str, torch.Tensor]):
    state_cls = getattr(together.types, "TogetherRobotState", None)
    if state_cls is None:
        return state
    return state_cls(
        root_pos=state["root_pos"],
        root_rpy=state["root_rpy"],
        foot_pos=state["foot_pos"],
        joint_angles=state["joint_angles"],
    )


def _dataclass_kwargs(cls, kwargs: dict):
    if not is_dataclass(cls):
        return kwargs
    names = {field.name for field in fields(cls)}
    return {name: value for name, value in kwargs.items() if name in names}


def _make_terrain_batch(
    terrain_cls,
    root_xy: torch.Tensor,
    *,
    height_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    resolution: float = 0.05,
    extent_xy: tuple[float, float] = (1.5, 1.5),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    if hasattr(terrain_cls, "from_heightmap"):
        root_xy = torch.as_tensor(root_xy, device=device, dtype=dtype)
        if root_xy.ndim == 1:
            root_xy = root_xy.unsqueeze(0)
        batch_size = int(root_xy.shape[0])
        extent_x, extent_y = extent_xy
        width = max(1, int(round(extent_x / resolution)))
        height = max(1, int(round(extent_y / resolution)))
        xs = torch.linspace(-0.5 * extent_x, 0.5 * extent_x, width, device=root_xy.device, dtype=dtype)
        ys = torch.linspace(-0.5 * extent_y, 0.5 * extent_y, height, device=root_xy.device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xx = xx.unsqueeze(0).expand(batch_size, -1, -1)
        yy = yy.unsqueeze(0).expand(batch_size, -1, -1)
        if height_fn is None:
            heightmap = torch.zeros((batch_size, height, width), device=root_xy.device, dtype=dtype)
        else:
            heightmap = height_fn(xx, yy).to(device=root_xy.device, dtype=dtype)
        return terrain_cls.from_heightmap(
            heightmap,
            world_x_range=(-0.5 * extent_x, 0.5 * extent_x),
            world_y_range=(-0.5 * extent_y, 0.5 * extent_y),
        )

    root_xy = torch.as_tensor(root_xy, device=device, dtype=dtype)
    if root_xy.ndim == 1:
        root_xy = root_xy.unsqueeze(0)
    batch_size = int(root_xy.shape[0])
    extent_x, extent_y = extent_xy
    width = max(1, int(round(extent_x / resolution)))
    height = max(1, int(round(extent_y / resolution)))
    origin = root_xy - torch.tensor([0.5 * extent_x, 0.5 * extent_y], device=root_xy.device, dtype=dtype)
    xs = origin[:, 0:1] + (torch.arange(width, device=root_xy.device, dtype=dtype) + 0.5) * resolution
    ys = origin[:, 1:2] + (torch.arange(height, device=root_xy.device, dtype=dtype) + 0.5) * resolution
    yy, xx = torch.meshgrid(torch.arange(height, device=root_xy.device), torch.arange(width, device=root_xy.device), indexing="ij")
    sample_x = xs[:, xx.reshape(-1)].reshape(batch_size, height, width)
    sample_y = ys[:, yy.reshape(-1)].reshape(batch_size, height, width)
    if height_fn is None:
        patch = torch.zeros((batch_size, height, width), device=root_xy.device, dtype=dtype)
    else:
        patch = height_fn(sample_x, sample_y).to(device=root_xy.device, dtype=dtype)
    kwargs = {
        "terrain_id": "synthetic",
        "patch": patch,
        "center_xy": root_xy,
        "patch_origin_xy": origin,
        "patch_extent_xy": extent_xy,
        "patch_resolution": float(resolution),
        "device": torch.device(device),
        "dtype": dtype,
        "cache_keys": tuple(("synthetic", idx) for idx in range(batch_size)),
    }
    return terrain_cls(**_dataclass_kwargs(terrain_cls, kwargs))


def _state_batch(
    *,
    num_envs: int,
    root_yaw: torch.Tensor | None = None,
    root_xy: torch.Tensor | None = None,
    roll_pitch: tuple[float, float] = (0.0, 0.0),
    foot_xy_offset: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    raw_state = raw_default_initial_state(terrain=None)
    root_pos = torch.as_tensor(raw_state.root_pos, dtype=torch.float32).repeat(num_envs, 1)
    if root_xy is None:
        root_pos[:, 0] = torch.arange(num_envs, dtype=torch.float32) * 0.05
        root_pos[:, 1] = torch.arange(num_envs, dtype=torch.float32) * -0.03
    else:
        root_pos[:, :2] = torch.as_tensor(root_xy, dtype=torch.float32)
    root_rpy = torch.zeros((num_envs, 3), dtype=torch.float32)
    root_rpy[:, 0] = float(roll_pitch[0])
    root_rpy[:, 1] = float(roll_pitch[1])
    if root_yaw is not None:
        root_rpy[:, 2] = torch.as_tensor(root_yaw, dtype=torch.float32)

    foot_pos0 = torch.as_tensor(raw_state.foot_pos, dtype=torch.float32).reshape(4, 3)
    rel0 = foot_pos0 - torch.as_tensor(raw_state.root_pos, dtype=torch.float32).view(1, 3)
    yaw = root_rpy[:, 2]
    cos_yaw = torch.cos(yaw).view(num_envs, 1)
    sin_yaw = torch.sin(yaw).view(num_envs, 1)
    x = rel0[:, 0].view(1, 4)
    y = rel0[:, 1].view(1, 4)
    rel_xy = torch.stack((cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y), dim=-1)
    foot_pos = root_pos[:, None, :].repeat(1, 4, 1)
    foot_pos[..., :2] = root_pos[:, None, :2] + rel_xy
    foot_pos[..., 2] = 0.0
    if foot_xy_offset is not None:
        foot_pos[..., :2] += foot_xy_offset.to(dtype=foot_pos.dtype).view(1, 4, 2)

    joint_angles = torch.as_tensor(raw_state.joint_angles, dtype=torch.float32).repeat(num_envs, 1)
    return {
        "root_pos": root_pos,
        "root_rpy": root_rpy,
        "foot_pos": foot_pos,
        "joint_angles": joint_angles,
    }


def _plan_together(
    command: torch.Tensor,
    *,
    height_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    root_yaw: torch.Tensor | None = None,
    root_xy: torch.Tensor | None = None,
    roll_pitch: tuple[float, float] = (0.0, 0.0),
    foot_xy_offset: torch.Tensor | None = None,
):
    together = _require_together_modules()
    cfg = _cfg(together)
    command = torch.as_tensor(command, dtype=torch.float32)
    if command.ndim == 1:
        command = command.unsqueeze(0)
    state = _state_batch(
        num_envs=int(command.shape[0]),
        root_yaw=root_yaw,
        root_xy=root_xy,
        roll_pitch=roll_pitch,
        foot_xy_offset=foot_xy_offset,
    )
    terrain = _make_terrain_batch(
        _terrain_cls(together),
        state["root_pos"][:, :2],
        height_fn=height_fn,
        resolution=float(getattr(cfg, "default_terrain_local_resolution", 0.05)),
    )
    return together.planner.plan_segment(terrain, _state_for_together(together, state), command, cfg), cfg, terrain, state


def _plan_raw_and_together(command: torch.Tensor, *, root_yaw: torch.Tensor | None = None):
    together = _require_together_modules()
    raw_cfg = RawBatchPlannerConfig()
    together_cfg = _cfg(together)
    command = torch.as_tensor(command, dtype=torch.float32)
    if command.ndim == 1:
        command = command.unsqueeze(0)
    state = _state_batch(num_envs=int(command.shape[0]), root_yaw=root_yaw)
    raw_terrain = _make_terrain_batch(RawTerrainBatch, state["root_pos"][:, :2], resolution=float(raw_cfg.default_terrain_local_resolution))
    together_terrain = _make_terrain_batch(
        _terrain_cls(together),
        state["root_pos"][:, :2],
        resolution=float(getattr(together_cfg, "default_terrain_local_resolution", raw_cfg.default_terrain_local_resolution)),
    )
    raw_result = raw_plan_segment(raw_terrain, state, command, raw_cfg)
    together_result = together.planner.plan_segment(together_terrain, _state_for_together(together, state), command, together_cfg)
    return raw_result, together_result, raw_cfg, together_cfg


def _assert_result_schema(result, *, num_envs: int, horizon: int, event_cap: int) -> None:
    expected_shapes = {
        "root_pos": (num_envs, horizon, 3),
        "root_rpy": (num_envs, horizon, 3),
        "foot_pos": (num_envs, horizon, 4, 3),
        "joint_angles": (num_envs, horizon, 12),
        "contact_state": (num_envs, horizon, 4),
        "touchdown_seq": (num_envs, 4, event_cap, 3),
        "touchdown_mask": (num_envs, 4, event_cap),
        "cost_total": (num_envs,),
        "status": (num_envs,),
        "feasible": (num_envs,),
        "safe_fallback": (num_envs,),
    }
    for name, shape in expected_shapes.items():
        assert hasattr(result, name), f"TogetherPlannerResult missing field {name!r}"
        assert tuple(getattr(result, name).shape) == shape, f"{name} shape mismatch"
    for key in ("J_td", "J_swing", "J_ik", "J_base", "J_vel"):
        assert key in result.cost_breakdown
        assert tuple(result.cost_breakdown[key].shape) == (num_envs,)
    for diagnostic in ("joint_limit_violation", "workspace_margin"):
        assert hasattr(result, diagnostic), f"together result must expose IK diagnostic {diagnostic}"


def _assert_close_field(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    torch.testing.assert_close(
        actual.detach().cpu().to(dtype=torch.float32),
        expected.detach().cpu().to(dtype=torch.float32),
        atol=atol,
        rtol=rtol,
        msg=f"raw/together mismatch on {name}",
    )


class TestBatchedTogetherBehavior:
    @pytest.mark.parametrize(
        "command,axis,minimum,label",
        [
            ([0.35, 0.0, 0.0], 0, 0.12, "forward"),
            ([0.0, 0.25, 0.0], 1, 0.08, "lateral"),
            ([0.0, 0.0, 0.60], 2, 0.25, "yaw"),
        ],
    )
    def test_forward_lateral_yaw_command_response(self, command, axis, minimum, label):
        result, _cfg, _terrain, _state = _plan_together(torch.tensor(command))
        if axis < 2:
            delta = result.root_pos[0, -1, axis] - result.root_pos[0, 0, axis]
        else:
            delta = result.root_rpy[0, -1, 2] - result.root_rpy[0, 0, 2]
        assert float(delta) > minimum, f"{label} command did not move expected axis enough: {float(delta):.4f}"

    def test_yaw_pi_over_2_base_frame_forward_command_moves_world_y(self):
        result, _cfg, _terrain, _state = _plan_together(
            torch.tensor([[0.35, 0.0, 0.0]]),
            root_yaw=torch.tensor([math.pi / 2.0]),
        )
        delta_xy = result.root_pos[0, -1, :2] - result.root_pos[0, 0, :2]
        assert float(delta_xy[1]) > 0.12
        assert abs(float(delta_xy[0])) < 0.06

    def test_zero_command_uses_root_frame_template_rehome(self):
        perturb = torch.tensor([[0.04, -0.02], [-0.03, 0.03], [0.02, 0.01], [-0.04, -0.01]], dtype=torch.float32)
        result, _cfg, _terrain, state = _plan_together(
            torch.tensor([[0.0, 0.0, 0.0]]),
            roll_pitch=(0.22, -0.16),
            foot_xy_offset=perturb,
        )
        assert torch.all(result.contact_state[0] == 1)
        assert torch.all(result.touchdown_mask[0] == 0)
        nominal_root = torch.as_tensor(HIP_OFFSETS_ARRAY[:, :2], dtype=torch.float32)
        initial_root_xy = state["foot_pos"][0, :, :2] - state["root_pos"][0, :2]
        terminal_root_xy = result.foot_pos[0, -1, :, :2] - result.root_pos[0, -1, :2]
        assert torch.linalg.norm(terminal_root_xy - nominal_root) < torch.linalg.norm(initial_root_xy - nominal_root)
        assert torch.linalg.norm(result.root_rpy[0, -1, :2]) < torch.linalg.norm(state["root_rpy"][0, :2]) * 0.25

    def test_mixed_full_batch_per_row_commands_are_independent(self):
        commands = torch.tensor(
            [
                [0.35, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.60],
                [0.25, -0.10, 0.45],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        result, cfg, _terrain, _state = _plan_together(commands)
        _assert_result_schema(result, num_envs=5, horizon=int(cfg.horizon_steps), event_cap=int(cfg.event_cap))
        assert torch.all(result.root_pos[0, -1, 0] - result.root_pos[0, 0, 0] > 0.12)
        assert torch.all(result.root_pos[1, -1, 1] - result.root_pos[1, 0, 1] > 0.08)
        assert torch.all(result.root_rpy[2, -1, 2] - result.root_rpy[2, 0, 2] > 0.25)
        assert torch.all(result.contact_state[4] == 1)

    def test_synthetic_no_low_ground_fallback_smoke(self):
        def pit_height(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            pit = (x > 0.12) & (x < 0.42) & (y.abs() < 0.35)
            side_step = (x >= 0.42).to(dtype=x.dtype) * 0.08
            return torch.where(pit, torch.full_like(x, -0.25), side_step)

        result, _cfg, terrain, _state = _plan_together(torch.tensor([[0.45, 0.0, 0.0]]), height_fn=pit_height)
        terrain_mod = _require_together_modules().terrain
        foot_xy = result.foot_pos[..., :2].reshape(1, -1, 2)
        if hasattr(terrain_mod, "sample_patch_heights_at_xy"):
            sampled = terrain_mod.sample_patch_heights_at_xy(terrain, foot_xy).reshape(result.foot_pos.shape[:-1])
        else:
            sampled = terrain.height_at(foot_xy).reshape(result.foot_pos.shape[:-1])
        assert bool(torch.all(result.foot_pos[..., 2] >= sampled - 1e-4))
        assert not bool(result.safe_fallback[0]), "flat synthetic pit should not need training fallback"


class TestBatchedTogetherRawParity:
    @pytest.mark.parametrize(
        "commands",
        [
            torch.tensor([[0.0, 0.0, 0.0], [0.35, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.60], [0.25, -0.10, 0.45]]),
        ],
    )
    def test_schedule_contact_state_and_touchdown_mask_exact(self, commands):
        together = _require_together_modules()
        raw_cfg = RawBatchPlannerConfig()
        together_cfg = _cfg(together)
        raw_schedule = raw_build_fixed_schedule(
            batch_size=int(commands.shape[0]),
            horizon_steps=int(raw_cfg.horizon_steps),
            dt=float(raw_cfg.dt),
            device=torch.device("cpu"),
            dtype=torch.float32,
            command_batch=commands,
            planner_cfg=raw_cfg,
        )
        together_schedule = together.schedule.build_fixed_schedule(
            batch_size=int(commands.shape[0]),
            horizon_steps=int(together_cfg.horizon_steps),
            dt=float(together_cfg.dt),
            device=torch.device("cpu"),
            dtype=torch.float32,
            command_batch=commands,
            planner_cfg=together_cfg,
        )
        torch.testing.assert_close(together_schedule.contact_state, raw_schedule.contact_state, atol=0.0, rtol=0.0)
        torch.testing.assert_close(together_schedule.touchdown_mask, raw_schedule.touchdown_mask, atol=0.0, rtol=0.0)

    def test_result_schema_shapes_exact(self):
        commands = torch.tensor([[0.35, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
        result, cfg, _terrain, _state = _plan_together(commands)
        _assert_result_schema(result, num_envs=3, horizon=int(cfg.horizon_steps), event_cap=int(cfg.event_cap))

    @pytest.mark.parametrize(
        "command,root_yaw,label",
        [
            ([0.0, 0.0, 0.0], None, "zero"),
            ([0.35, 0.0, 0.0], None, "forward"),
            ([0.0, 0.25, 0.0], None, "lateral"),
            ([0.0, 0.0, 0.60], None, "yaw"),
            ([0.25, -0.10, 0.45], None, "combo"),
            ([0.35, 0.0, 0.0], torch.tensor([math.pi / 2.0]), "base_frame_yaw_pi_over_2"),
        ],
    )
    def test_flat_terrain_raw_together_fields_match(self, command, root_yaw, label):
        raw_result, together_result, raw_cfg, _together_cfg = _plan_raw_and_together(torch.tensor([command]), root_yaw=root_yaw)
        horizon = int(raw_cfg.horizon_steps)
        assert tuple(together_result.root_pos.shape) == (1, horizon, 3), label

        exact_fields = ("contact_state", "touchdown_mask", "status", "feasible")
        for name in exact_fields:
            torch.testing.assert_close(getattr(together_result, name).detach().cpu(), getattr(raw_result, name).detach().cpu(), atol=0.0, rtol=0.0)

        assert hasattr(together_result, "safe_fallback")
        expected_safe_fallback = raw_result.status != int(RawPlannerStatus.OK)
        torch.testing.assert_close(together_result.safe_fallback.detach().cpu().to(torch.bool), torch.zeros_like(expected_safe_fallback, dtype=torch.bool))

        for name in ("root_pos", "root_rpy", "foot_pos", "joint_angles", "touchdown_seq", "cost_total"):
            _assert_close_field(name, getattr(together_result, name), getattr(raw_result, name), atol=1e-4, rtol=1e-4)

        assert set(together_result.cost_breakdown) == set(raw_result.cost_breakdown)
        for name in sorted(raw_result.cost_breakdown):
            _assert_close_field(f"cost_breakdown.{name}", together_result.cost_breakdown[name], raw_result.cost_breakdown[name], atol=1e-4, rtol=1e-4)
