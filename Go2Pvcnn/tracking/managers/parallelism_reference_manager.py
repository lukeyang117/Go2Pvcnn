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

    def _terrain(self, root_pos: Tensor) -> ParallelismTerrain:
        n = int(root_pos.shape[0])
        side = self.terrain_grid_size
        resolution = self.terrain_resolution
        half_extent = 0.5 * float(side - 1) * resolution
        origin = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        origin[:, 0] = root_pos[:, 0] - half_extent
        origin[:, 1] = root_pos[:, 1] - half_extent
        height = torch.zeros(n, side, side, dtype=torch.float32, device=self.device)
        semantic = torch.zeros(n, side, side, dtype=torch.long, device=self.device)
        valid = torch.ones(n, side, side, dtype=torch.bool, device=self.device)
        yaw = torch.zeros(n, dtype=torch.float32, device=self.device)
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
            trajectory = plan_trajectory(state, self._command(subset), self._terrain(state.root_pos_w), self.cfg)
            self.root_pos_w[subset] = trajectory.root_pos_w
            self.root_rpy_w[subset] = trajectory.root_rpy_w
            self.joint_pos[subset] = trajectory.joint_pos
            self.foot_pos_w[subset] = trajectory.foot_pos_w
            self.contact_state[subset] = trajectory.contact_state
            self.valid[subset] = trajectory.valid[:, None].expand(-1, self.horizon)
            self._cached_cycle[subset] = subset_cycle
            self._initialized[subset] = True
            self.plan_count[subset] += 1

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
