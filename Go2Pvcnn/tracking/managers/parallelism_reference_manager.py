"""Reference cache manager for flat parallelism RL tracking."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from extension.convention import extract_roll_pitch_batch, extract_yaw_batch
from extension.parallelism.config import ParallelismCfg
from extension.parallelism.planner import plan_trajectory
from extension.parallelism.types import ParallelismState, ParallelismTerrain


def _env_root(env):
    return getattr(env, "unwrapped", env)


def _as_env_ids(env_ids, *, num_envs: int, device: torch.device) -> Tensor:
    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.long, device=device)
    tensor = torch.as_tensor(env_ids, device=device)
    if tensor.dtype == torch.bool:
        return tensor.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
    return tensor.to(dtype=torch.long).flatten()


class ParallelismReferenceManager:
    """Owns 24-frame parallelism references and exposes the current phase frame."""

    def __init__(
        self,
        env,
        cfg: ParallelismCfg | None = None,
        *,
        command_name: str = "base_velocity",
        plan_batch_size: int | None = None,
        terrain_grid_size: int = 151,
        terrain_resolution: float = 0.01,
        autostart: bool = True,
    ) -> None:
        self.env = _env_root(env)
        self.cfg = cfg or ParallelismCfg()
        self.command_name = str(command_name)
        self.plan_batch_size = int(plan_batch_size or getattr(getattr(self.env, "cfg", None), "parallelism_plan_batch_size", 64))
        self.device = torch.device(getattr(self.env, "device", "cpu"))
        self.num_envs = int(getattr(self.env, "num_envs"))
        self.horizon = int(self.cfg.horizon)
        self.dt = float(self.cfg.dt)
        self.terrain_grid_size = int(terrain_grid_size)
        self.terrain_resolution = float(terrain_resolution)

        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cached_cycle = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.plan_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._manual_episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.root_pos_w = torch.zeros(self.num_envs, self.horizon, 3, dtype=torch.float32, device=self.device)
        self.root_rpy_w = torch.zeros_like(self.root_pos_w)
        self.joint_pos = torch.zeros(self.num_envs, self.horizon, 12, dtype=torch.float32, device=self.device)
        self.foot_pos_w = torch.zeros(self.num_envs, self.horizon, 4, 3, dtype=torch.float32, device=self.device)
        self.contact_state = torch.ones(self.num_envs, self.horizon, 4, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros(self.num_envs, self.horizon, dtype=torch.bool, device=self.device)

        if autostart:
            self.reset()

    def refresh(self) -> None:
        episode_length = self._episode_length()
        cycle = torch.div(episode_length, self.horizon, rounding_mode="floor")
        phase = torch.remainder(episode_length, self.horizon)
        reset_mask = (episode_length == 0) & ((~self._initialized) | (self._cached_cycle != 0))
        needs_plan = (~self._initialized) | reset_mask | (cycle != self._cached_cycle)
        env_ids = needs_plan.nonzero(as_tuple=False).flatten()
        if int(env_ids.numel()) > 0:
            self._plan(env_ids, cycle.index_select(0, env_ids))
        self.phase.copy_(phase.to(dtype=torch.long))

    def reset(self, env_ids: Sequence[int] | Tensor | None = None) -> None:
        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        if int(ids.numel()) == 0:
            return
        cycle = torch.zeros_like(ids, dtype=torch.long, device=self.device)
        self._manual_episode_length[ids] = 0
        self.phase[ids] = 0
        self._plan(ids, cycle)

    def step(self) -> None:
        self._manual_episode_length += 1
        self.refresh()

    @property
    def current_joint_pos(self) -> Tensor:
        self.refresh()
        return self._current_take(self.joint_pos)

    @property
    def current_joint_vel(self) -> Tensor:
        self.refresh()
        next_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        current = self._take(self.joint_pos, self.phase)
        nxt = self._take(self.joint_pos, next_phase)
        return (nxt - current) / max(self.dt, 1.0e-6)

    @property
    def current_root_pos_w(self) -> Tensor:
        self.refresh()
        return self._current_take(self.root_pos_w)

    @property
    def current_root_rpy_w(self) -> Tensor:
        self.refresh()
        return self._current_take(self.root_rpy_w)

    @property
    def current_foot_pos_w(self) -> Tensor:
        self.refresh()
        return self._current_take(self.foot_pos_w)

    @property
    def current_root_lin_vel_b(self) -> Tensor:
        self.refresh()
        next_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        pos = self._take(self.root_pos_w, self.phase)
        nxt = self._take(self.root_pos_w, next_phase)
        vel_w = (nxt - pos) / max(self.dt, 1.0e-6)
        yaw = self._take(self.root_rpy_w, self.phase)[:, 2]
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        vel_b = torch.zeros_like(vel_w)
        vel_b[:, 0] = cosine * vel_w[:, 0] + sine * vel_w[:, 1]
        vel_b[:, 1] = -sine * vel_w[:, 0] + cosine * vel_w[:, 1]
        vel_b[:, 2] = vel_w[:, 2]
        return vel_b

    @property
    def current_root_ang_vel_b(self) -> Tensor:
        self.refresh()
        next_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        rpy = self._take(self.root_rpy_w, self.phase)
        nxt = self._take(self.root_rpy_w, next_phase)
        return (nxt - rpy) / max(self.dt, 1.0e-6)

    def _episode_length(self) -> Tensor:
        value = getattr(self.env, "episode_length_buf", None)
        if value is None:
            return self._manual_episode_length
        return torch.as_tensor(value, dtype=torch.long, device=self.device).reshape(self.num_envs)

    def _robot(self):
        return self.env.scene["robot"]

    def _command(self, env_ids: Tensor) -> Tensor:
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is None:
            return torch.zeros(int(env_ids.numel()), 3, dtype=torch.float32, device=self.device)
        if hasattr(command_manager, "get_command"):
            command = command_manager.get_command(self.command_name)
        else:
            command = getattr(command_manager, self.command_name)
        return torch.as_tensor(command, dtype=torch.float32, device=self.device).index_select(0, env_ids)

    def _state(self, env_ids: Tensor) -> ParallelismState:
        robot = self._robot()
        root_pos = torch.as_tensor(robot.data.root_pos_w, dtype=torch.float32, device=self.device).index_select(0, env_ids)
        root_quat = torch.as_tensor(robot.data.root_quat_w, dtype=torch.float32, device=self.device).index_select(0, env_ids)
        roll, pitch = extract_roll_pitch_batch(root_quat)
        yaw = extract_yaw_batch(root_quat)
        root_rpy = torch.stack((roll, pitch, yaw), dim=-1)
        joint = torch.as_tensor(robot.data.joint_pos, dtype=torch.float32, device=self.device).index_select(0, env_ids)
        return ParallelismState(root_pos_w=root_pos, root_rpy_w=root_rpy, joint_pos=joint, foot_pos_w=None)

    def _terrain(self, root_pos: Tensor, env_ids: Tensor | None = None) -> ParallelismTerrain:
        n = int(root_pos.shape[0])
        ids = (
            torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
            if env_ids is not None
            else None
        )
        scanner = self._semantic_height_scanner()
        data = getattr(scanner, "data", None) if scanner is not None else None
        ray_hits_source = getattr(data, "ray_hits_w", None)
        height_source = getattr(data, "elevation_map", None)
        semantic_source = getattr(getattr(scanner, "data", None), "semantic_map", None) if scanner is not None else None
        valid_source = getattr(getattr(scanner, "data", None), "valid_mask", None) if scanner is not None else None
        if (ray_hits_source is not None or height_source is not None) and semantic_source is not None:
            origin_xy = None
            yaw = None
            if ray_hits_source is not None:
                ray_hits = torch.as_tensor(ray_hits_source, dtype=torch.float32, device=self.device)
                if ray_hits.ndim != 3 or int(ray_hits.shape[-1]) != 3:
                    raise ValueError(f"semantic_height_scanner ray_hits_w must have shape [B,H*W,3], got {tuple(ray_hits.shape)}")
                if ids is not None and int(ray_hits.shape[0]) == self.num_envs:
                    ray_hits = ray_hits.index_select(0, ids)
                side = int(round(float(ray_hits.shape[1]) ** 0.5))
                if side * side != int(ray_hits.shape[1]):
                    raise ValueError(f"semantic_height_scanner ray count {int(ray_hits.shape[1])} is not a square grid")
                ray_grid = ray_hits.reshape(int(ray_hits.shape[0]), side, side, 3)
                finite_ray = torch.isfinite(ray_grid).all(dim=-1)
                height = ray_grid[..., 2]
                origin_xy = ray_grid[:, 0, 0, :2]
                if side > 1:
                    step_xy = ray_grid[:, 0, 1, :2] - ray_grid[:, 0, 0, :2]
                    yaw = torch.atan2(step_xy[:, 1], step_xy[:, 0])
            else:
                height = torch.as_tensor(height_source, dtype=torch.float32, device=self.device)
                finite_ray = torch.isfinite(height)
            semantic = torch.as_tensor(semantic_source, dtype=torch.long, device=self.device)
            if ids is not None and int(height.shape[0]) == self.num_envs:
                height = height.index_select(0, ids)
                finite_ray = finite_ray.index_select(0, ids)
            if ids is not None and int(semantic.shape[0]) == self.num_envs:
                semantic = semantic.index_select(0, ids)
            if valid_source is not None:
                valid_source = torch.as_tensor(valid_source, dtype=torch.bool, device=self.device)
                if ids is not None and int(valid_source.shape[0]) == self.num_envs:
                    valid_source = valid_source.index_select(0, ids)
            if height.ndim == 2:
                height = height.unsqueeze(0).expand(n, -1, -1)
            if semantic.ndim == 2:
                semantic = semantic.unsqueeze(0).expand(n, -1, -1)
            side = int(height.shape[-1])
            resolution = self._scanner_resolution(scanner, fallback=self.terrain_resolution)
            valid = (
                torch.as_tensor(valid_source, dtype=torch.bool, device=self.device)
                if valid_source is not None
                else finite_ray
            )
            if valid.ndim == 2:
                valid = valid.unsqueeze(0).expand(n, -1, -1)
            height = torch.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            side = self.terrain_grid_size
            resolution = self.terrain_resolution
            height = torch.zeros(n, side, side, dtype=torch.float32, device=self.device)
            semantic = torch.zeros(n, side, side, dtype=torch.long, device=self.device)
            valid = torch.ones(n, side, side, dtype=torch.bool, device=self.device)
            origin_xy = None
            yaw = None
        half_extent = 0.5 * float(side - 1) * resolution
        origin = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        if origin_xy is None or int(origin_xy.shape[0]) != n:
            origin[:, 0] = root_pos[:, 0] - half_extent
            origin[:, 1] = root_pos[:, 1] - half_extent
        else:
            origin[:, :2] = origin_xy.to(dtype=origin.dtype, device=origin.device)
        if yaw is None or int(yaw.shape[0]) != n:
            yaw = torch.zeros(n, dtype=torch.float32, device=self.device)
        else:
            yaw = yaw.to(dtype=torch.float32, device=self.device)
        return ParallelismTerrain(
            height_w=height,
            semantic_id=semantic,
            valid_mask=valid,
            origin_w=origin,
            yaw_w=yaw,
            resolution=resolution,
        )

    def _plan(self, env_ids: Tensor, cycle: Tensor) -> None:
        batch_size = max(int(self.plan_batch_size), 1)
        for start in range(0, int(env_ids.numel()), batch_size):
            subset = env_ids[start : start + batch_size]
            subset_cycle = cycle[start : start + batch_size]
            state = self._state(subset)
            trajectory = plan_trajectory(state, self._command(subset), self._terrain(state.root_pos_w, subset), self.cfg)
            self.root_pos_w[subset] = trajectory.root_pos_w
            self.root_rpy_w[subset] = trajectory.root_rpy_w
            self.joint_pos[subset] = trajectory.joint_pos
            self.foot_pos_w[subset] = trajectory.foot_pos_w
            self.contact_state[subset] = trajectory.contact_state
            self.valid[subset] = trajectory.valid[:, None].expand(-1, self.horizon)
            self._cached_cycle[subset] = subset_cycle
            self._initialized[subset] = True
            self.plan_count[subset] += 1

    def _semantic_height_scanner(self):
        scene = getattr(self.env, "scene", None)
        if scene is None:
            return None
        sensors = getattr(scene, "sensors", None)
        if sensors is not None:
            try:
                return sensors["semantic_height_scanner"]
            except Exception:  # noqa: BLE001 - Isaac containers and test doubles are both duck-typed.
                scanner = getattr(sensors, "semantic_height_scanner", None)
                if scanner is not None:
                    return scanner
        try:
            return scene["semantic_height_scanner"]
        except Exception:  # noqa: BLE001
            return getattr(scene, "semantic_height_scanner", None)

    def _scanner_resolution(self, scanner, *, fallback: float) -> float:
        pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
        resolution = getattr(pattern_cfg, "resolution", None)
        if resolution is None:
            return float(fallback)
        return float(resolution)

    def _take(self, values: Tensor, phase: Tensor) -> Tensor:
        batch = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return values[batch, phase.to(dtype=torch.long, device=self.device)]

    def _current_take(self, values: Tensor) -> Tensor:
        return self._take(values, self.phase)


def get_parallelism_reference_manager(env) -> ParallelismReferenceManager:
    root = _env_root(env)
    manager = getattr(root, "parallelism_reference_manager", None)
    if manager is None:
        manager = ParallelismReferenceManager(root)
        root.parallelism_reference_manager = manager
    return manager
