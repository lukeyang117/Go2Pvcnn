"""Planner-owned batched trajectory manager."""

from __future__ import annotations

import torch
from torch import Tensor

from ..convention import planner_result_to_reference_cache
from ..reference.cache import ReferenceTrajectoryCache, fill_reference_cache_standstill_rows, masked_write_reference_cache_rows
from .instrumentation import PlannerInstrumentation, PlannerTimingSummary
from .terrain import PlannerTerrain
from .types import BatchedRobotState
from .trajectory import batched_generate_trajectory


class BatchedTrajectoryManager:
    def __init__(self, cfg, device):
        self._cfg = cfg
        self._device = torch.device(device)
        self._cache = None
        self._phase_counter: Tensor | None = None
        self._step_counter = 0
        self._planner_instrumentation = PlannerInstrumentation(
            enabled=bool(getattr(cfg, "planner_instrumentation", False) or getattr(cfg, "verbose_planner", False))
        )
        self._verbose_planner = bool(getattr(cfg, "verbose_planner", False))
        self._verbose_planner_interval_steps = max(1, int(getattr(cfg, "verbose_planner_interval_steps", 250)))
        self._last_episode_length_buf: Tensor | None = None
        self._last_replan_episode_length_buf: Tensor | None = None
        self._last_commands: Tensor | None = None
        self._pending_reset_mask: Tensor | None = None
        self._foot_body_ids: Tensor | None = None

    @staticmethod
    def _cfg_dt(cfg) -> float:
        for attr in ("dt", "step_dt"):
            value = getattr(cfg, attr, None)
            if value is not None:
                return float(value)
        sim = getattr(cfg, "sim", None)
        sim_dt = getattr(sim, "dt", None) if sim is not None else None
        decimation = getattr(cfg, "decimation", None)
        if sim_dt is not None and decimation is not None:
            return float(sim_dt) * float(decimation)
        raise AttributeError("planner dt is unavailable; expected env.step_dt, cfg.dt, cfg.step_dt, or cfg.decimation * cfg.sim.dt")

    def _ensure_phase_counter(self, num_envs: int) -> None:
        if self._phase_counter is None or int(self._phase_counter.shape[0]) != num_envs:
            self._phase_counter = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        if self._pending_reset_mask is None or int(self._pending_reset_mask.shape[0]) != num_envs:
            self._pending_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)

    @staticmethod
    def _named_get(container, name: str):
        try:
            return container[name]
        except Exception:  # noqa: BLE001 - tiny compatibility shim for fake/test scenes
            return getattr(container, name)

    def _env_root(self, env):
        return getattr(env, "unwrapped", env)

    def _command_name(self) -> str:
        return str(getattr(self._cfg, "reference_command_name", "base_velocity"))

    def _scanner_name(self) -> str:
        return str(getattr(self._cfg, "reference_height_scanner_name", "height_scanner"))

    def _foot_ids(self, robot) -> Tensor:
        if self._foot_body_ids is None:
            body_ids, _ = robot.find_bodies(".*_foot")
            self._foot_body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=self._device)
        return self._foot_body_ids

    def _batched_state_from_env(self, env) -> BatchedRobotState:
        root = self._env_root(env)
        robot = self._named_get(root.scene, "robot")
        data = robot.data
        foot_ids = self._foot_ids(robot)
        return BatchedRobotState(
            root_pos=torch.as_tensor(data.root_pos_w, dtype=torch.float64, device=self._device),
            root_quat=torch.as_tensor(data.root_quat_w, dtype=torch.float64, device=self._device),
            joint_angles=torch.as_tensor(data.joint_pos, dtype=torch.float64, device=self._device),
            foot_pos=torch.as_tensor(data.body_pos_w[:, foot_ids, :], dtype=torch.float64, device=self._device),
            foot_vel=None,
        )

    def _terrain_from_env(self, env) -> PlannerTerrain:
        root = self._env_root(env)
        scanner = self._named_get(root.scene.sensors, self._scanner_name())
        ray_hits = torch.as_tensor(scanner.data.ray_hits_w, dtype=torch.float64, device=self._device)
        return PlannerTerrain.from_ray_hits(ray_hits)

    def _commands_from_env(self, env) -> Tensor:
        root = self._env_root(env)
        command = root.command_manager.get_command(self._command_name())
        return torch.as_tensor(command, dtype=torch.float64, device=self._device)

    def _episode_length_buf_from_env(self, env) -> Tensor:
        root = self._env_root(env)
        return torch.as_tensor(root.episode_length_buf, dtype=torch.long, device=self._device)

    def _cache_from_result(self, result):
        cache = planner_result_to_reference_cache(result)
        self._cache = cache
        return cache

    @staticmethod
    def _clone_reference_cache(cache: ReferenceTrajectoryCache) -> ReferenceTrajectoryCache:
        """Clone tensors to avoid aliasing issues during in-place masked writes."""

        def _clone(t: Tensor | None) -> Tensor | None:
            if t is None:
                return None
            return t.clone()

        return ReferenceTrajectoryCache(
            root_pos_w=_clone(cache.root_pos_w),
            root_quat_w=_clone(cache.root_quat_w),
            joint_angles=_clone(cache.joint_angles),
            foot_pos_root=_clone(cache.foot_pos_root),
            contact_state=_clone(cache.contact_state),
            planned_touchdown_w=_clone(cache.planned_touchdown_w),
            phase_index=_clone(cache.phase_index),
            valid_mask=_clone(cache.valid_mask),
        )

    def planner_timing_summary(self, *, window: bool = True, reset_window: bool = False) -> PlannerTimingSummary:
        """Return the planner-owned timing summary collected by this manager."""

        return self._planner_instrumentation.summary(window=window, reset_window=reset_window)

    def _maybe_print_planner_diagnostics(self, *, step: int | None) -> None:
        if not self._verbose_planner:
            return
        if step is None:
            return
        if int(step) % int(self._verbose_planner_interval_steps) != 0:
            return
        summary = self.planner_timing_summary(window=True, reset_window=True)
        print(summary.format_compact())

    def _planner_call_kwargs(self, *, env) -> dict[str, object]:
        """Extra kwargs for the planner entrypoint, only when enabled.

        Important: avoid always passing the kwarg so unit tests that patch
        `batched_generate_trajectory` with a strict signature keep working.
        """

        if not self._planner_instrumentation.enabled:
            return {}
        return {"instrumentation": self._planner_instrumentation}

    def _planner_dt(self, env=None) -> float:
        if env is not None:
            root = self._env_root(env)
            step_dt = getattr(root, "step_dt", None)
            if step_dt is not None:
                return float(step_dt)
        return self._cfg_dt(self._cfg)

    def _needs_replan(self, episode_length_buf: Tensor, commands: Tensor) -> bool:
        if self._cache is None:
            return True
        if self._last_episode_length_buf is None or self._last_replan_episode_length_buf is None:
            return True
        if self._pending_reset_mask is not None and torch.any(self._pending_reset_mask):
            return True
        if episode_length_buf.shape != self._last_episode_length_buf.shape:
            return True
        if torch.any(episode_length_buf < self._last_episode_length_buf):
            return True
        if self._last_commands is None or self._last_commands.shape != commands.shape:
            return True
        if not torch.allclose(commands, self._last_commands, atol=1e-6, rtol=1e-6):
            return True
        interval = int(self._cfg.reference_replan_interval_steps)
        if torch.any(episode_length_buf - self._last_replan_episode_length_buf >= interval):
            return True
        horizon = int(self._cfg.reference_trajectory_horizon)
        if self._cache.horizon_length() != horizon:
            return True
        return False

    def _compute_replan_mask(self, episode_length_buf: Tensor, commands: Tensor) -> Tensor:
        """Return a per-env boolean mask indicating which envs need replanning."""
        episode_length_buf = torch.as_tensor(episode_length_buf, device=self._device, dtype=torch.long)
        commands = torch.as_tensor(commands, device=self._device, dtype=torch.float64)
        num_envs = int(episode_length_buf.shape[0])

        # If we cannot reason about shape compatibility, fall back to a full replan.
        if self._cache is None:
            return torch.ones(num_envs, dtype=torch.bool, device=self._device)
        if self._last_episode_length_buf is None or self._last_commands is None or self._last_replan_episode_length_buf is None:
            return torch.ones(num_envs, dtype=torch.bool, device=self._device)
        if episode_length_buf.shape != self._last_episode_length_buf.shape:
            return torch.ones(num_envs, dtype=torch.bool, device=self._device)
        if commands.shape != self._last_commands.shape:
            return torch.ones(num_envs, dtype=torch.bool, device=self._device)
        horizon = int(self._cfg.reference_trajectory_horizon)
        if self._cache.horizon_length() != horizon:
            return torch.ones(num_envs, dtype=torch.bool, device=self._device)
        if self._last_replan_episode_length_buf.shape != episode_length_buf.shape:
            return torch.ones(num_envs, dtype=torch.bool, device=self._device)

        replan_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        replan_mask |= episode_length_buf < self._last_episode_length_buf

        if self._pending_reset_mask is not None and self._pending_reset_mask.shape == replan_mask.shape:
            replan_mask |= self._pending_reset_mask

        # Per-env command delta (avoid a single any() turning this into a global trigger).
        cmd_delta = torch.abs(commands - self._last_commands)
        if cmd_delta.ndim == 2:
            replan_mask |= torch.any(cmd_delta > 1e-6, dim=-1)
        else:
            replan_mask |= torch.any(cmd_delta.reshape(num_envs, -1) > 1e-6, dim=-1)

        interval = int(self._cfg.reference_replan_interval_steps)
        replan_mask |= (episode_length_buf - self._last_replan_episode_length_buf) >= interval
        return replan_mask

    @staticmethod
    def _subset_state(states: BatchedRobotState, env_ids: Tensor) -> BatchedRobotState:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=states.root_pos.device)
        foot_vel = states.foot_vel
        return BatchedRobotState(
            root_pos=states.root_pos.index_select(0, env_ids),
            root_quat=states.root_quat.index_select(0, env_ids),
            joint_angles=states.joint_angles.index_select(0, env_ids),
            foot_pos=states.foot_pos.index_select(0, env_ids),
            foot_vel=None if foot_vel is None else foot_vel.index_select(0, env_ids),
        )

    @staticmethod
    def _subset_terrain(terrain: PlannerTerrain, env_ids: Tensor) -> PlannerTerrain:
        # Unit tests sometimes stub terrain construction; accept those objects by
        # returning them unchanged. Real runtime uses PlannerTerrain instances.
        if not hasattr(terrain, "heightmaps") or not hasattr(terrain, "world_x_range") or not hasattr(terrain, "world_y_range"):
            return terrain
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=terrain.heightmaps.device)
        return PlannerTerrain._from_heightmaps(
            terrain.heightmaps.index_select(0, env_ids),
            world_x_range=terrain.world_x_range,
            world_y_range=terrain.world_y_range,
        )

    def _run_replan(self, terrain, states, commands, episode_length_buf: Tensor) -> None:
        result = batched_generate_trajectory(
            terrain,
            states,
            commands,
            requested_n_frames=int(self._cfg.reference_trajectory_horizon),
            dt=self._planner_dt(),
        )
        self._cache_from_result(result)
        self._ensure_phase_counter(int(states.root_pos.shape[0]))
        assert self._phase_counter is not None
        self._phase_counter.zero_()
        self._last_replan_episode_length_buf = episode_length_buf.clone()
        self._pending_reset_mask = torch.zeros_like(episode_length_buf, dtype=torch.bool)

    def refresh_from_env(self, env):
        # Reset the rolling window so each replan attempt reports its own timings.
        if self._planner_instrumentation.enabled:
            self._planner_instrumentation.summary(window=True, reset_window=True)

        episode_length_buf = self._episode_length_buf_from_env(env)
        commands = self._commands_from_env(env)
        num_envs = int(episode_length_buf.shape[0])
        self._ensure_phase_counter(num_envs)
        assert self._pending_reset_mask is not None

        same_step = (
            self._last_episode_length_buf is not None
            and self._last_commands is not None
            and episode_length_buf.shape == self._last_episode_length_buf.shape
            and commands.shape == self._last_commands.shape
            and torch.equal(episode_length_buf, self._last_episode_length_buf)
            and torch.allclose(commands, self._last_commands, atol=1e-6, rtol=1e-6)
        )
        if same_step and self._cache is not None and not torch.any(self._pending_reset_mask):
            return self._cache

        with self._planner_instrumentation.stage("terrain"):
            terrain = self._terrain_from_env(env)
        with self._planner_instrumentation.stage("state"):
            states = self._batched_state_from_env(env)

        with self._planner_instrumentation.stage("replan_mask"):
            replan_mask = self._compute_replan_mask(episode_length_buf, commands)
        assert self._phase_counter is not None

        if torch.any(replan_mask):
            env_ids = torch.nonzero(replan_mask, as_tuple=False).squeeze(-1)
            sub_states = self._subset_state(states, env_ids)
            sub_commands = commands.index_select(0, env_ids)
            sub_terrain = self._subset_terrain(terrain, env_ids)

            try:
                with self._planner_instrumentation.stage("plan"):
                    result = batched_generate_trajectory(
                        sub_terrain,
                        sub_states,
                        sub_commands,
                        requested_n_frames=int(self._cfg.reference_trajectory_horizon),
                        dt=self._planner_dt(env),
                        **self._planner_call_kwargs(env=env),
                    )
            except Exception:  # noqa: BLE001 - replanning should degrade gracefully when a cache exists
                if self._cache is None:
                    raise
                env_ids_cpu = env_ids.to(device=self._cache.root_pos_w.device, dtype=torch.long)  # type: ignore[union-attr]
                with self._planner_instrumentation.stage("standstill_fill"):
                    fill_reference_cache_standstill_rows(self._cache, env_ids_cpu)
                # Clear pending resets so the env stays standstill until it hits a new trigger.
                self._pending_reset_mask = torch.where(replan_mask, torch.zeros_like(self._pending_reset_mask), self._pending_reset_mask)
                # Record the attempted "replan time" for the failed envs so interval-based replans
                # don't immediately retrigger every subsequent step once the interval has elapsed.
                if (
                    self._last_replan_episode_length_buf is None
                    or self._last_replan_episode_length_buf.shape != episode_length_buf.shape
                ):
                    # Without compatible history, fall back to a full overwrite. In this case,
                    # _compute_replan_mask will already have requested replanning for all envs.
                    self._last_replan_episode_length_buf = episode_length_buf.clone()
                else:
                    self._last_replan_episode_length_buf = torch.where(
                        replan_mask,
                        episode_length_buf,
                        self._last_replan_episode_length_buf,
                    )

                # Reset phases for failed envs; advance others.
                max_phase = int(self._cache.root_pos_w.shape[1]) - 1  # type: ignore[union-attr]
                advanced = torch.clamp(self._phase_counter + 1, max=max_phase)
                self._phase_counter = torch.where(replan_mask, torch.zeros_like(self._phase_counter), advanced)
                self._maybe_print_planner_diagnostics(step=int(torch.max(episode_length_buf).item()))
            else:
                with self._planner_instrumentation.stage("cache_convert"):
                    sub_cache = planner_result_to_reference_cache(result)
                if self._cache is None or self._cache.root_pos_w is None:
                    self._cache = sub_cache
                elif torch.all(replan_mask):
                    # For a full replan, swapping the cache is simpler and avoids in-place masked
                    # writes that can trigger aliasing errors with some tensor backends.
                    self._cache = sub_cache
                else:
                    env_ids_cpu = env_ids.to(device=self._cache.root_pos_w.device, dtype=torch.long)
                    with self._planner_instrumentation.stage("cache_write"):
                        try:
                            masked_write_reference_cache_rows(self._cache, sub_cache, env_ids_cpu)
                        except RuntimeError as exc:
                            # In tests (and some backend edge cases) index_copy_ may fail if
                            # src/dst tensors alias. Clone the incoming cache and retry.
                            if "single memory location" not in str(exc):
                                raise
                            masked_write_reference_cache_rows(self._cache, self._clone_reference_cache(sub_cache), env_ids_cpu)

                # Reset phases for replanned envs; advance others.
                max_phase = int(self._cache.root_pos_w.shape[1]) - 1  # type: ignore[union-attr]
                advanced = torch.clamp(self._phase_counter + 1, max=max_phase)
                self._phase_counter = torch.where(replan_mask, torch.zeros_like(self._phase_counter), advanced)

                # Update interval accounting only for the envs we actually replanned.
                if self._last_replan_episode_length_buf is None or self._last_replan_episode_length_buf.shape != episode_length_buf.shape:
                    self._last_replan_episode_length_buf = episode_length_buf.clone()
                else:
                    self._last_replan_episode_length_buf = torch.where(replan_mask, episode_length_buf, self._last_replan_episode_length_buf)

                # Clear pending reset flags for replanned envs.
                self._pending_reset_mask = torch.where(replan_mask, torch.zeros_like(self._pending_reset_mask), self._pending_reset_mask)
                self._maybe_print_planner_diagnostics(step=int(torch.max(episode_length_buf).item()))
        else:
            assert self._cache is not None
            self._phase_counter = torch.clamp(self._phase_counter + 1, max=int(self._cache.root_pos_w.shape[1]) - 1)

        self._last_episode_length_buf = episode_length_buf.clone()
        self._last_commands = commands.clone()
        root = self._env_root(env)
        root._trajectory_reference_cache = self._cache
        return self._cache

    def step(self, terrain, states, commands):
        num_envs = int(states.root_pos.shape[0])
        self._ensure_phase_counter(num_envs)

        if self._step_counter % int(self._cfg.reference_replan_interval_steps) == 0:
            # Reset rolling window for each replan.
            if self._planner_instrumentation.enabled:
                self._planner_instrumentation.summary(window=True, reset_window=True)
            with self._planner_instrumentation.stage("plan"):
                result = batched_generate_trajectory(
                    terrain,
                    states,
                    commands,
                    requested_n_frames=int(self._cfg.reference_trajectory_horizon),
                    dt=self._planner_dt(),
                    **self._planner_call_kwargs(env=None),
                )
            with self._planner_instrumentation.stage("cache_convert"):
                self._cache = planner_result_to_reference_cache(result)
            self._phase_counter.zero_()
            self._maybe_print_planner_diagnostics(step=int(self._step_counter))

        assert self._cache is not None
        self._phase_counter = torch.clamp(self._phase_counter + 1, max=int(self._cache.root_pos_w.shape[1]) - 1)
        self._step_counter += 1
        return self._cache

    def current_reference(self) -> dict[str, Tensor]:
        if self._cache is None or self._phase_counter is None:
            raise RuntimeError("trajectory manager has no cached trajectory; call step() first")

        idx = self._phase_counter.clamp(max=int(self._cache.root_pos_w.shape[1]) - 1)
        env_idx = torch.arange(idx.shape[0], device=idx.device)
        return {
            "root_pos_w": self._cache.root_pos_w[env_idx, idx],
            "root_quat_w": self._cache.root_quat_w[env_idx, idx],
            "joint_angles": self._cache.joint_angles[env_idx, idx],
            "foot_pos_root": self._cache.foot_pos_root[env_idx, idx],
            "contact_state": self._cache.contact_state[env_idx, idx],
            "planned_touchdown_w": self._cache.planned_touchdown_w[env_idx, idx],
            "phase_index": self._cache.phase_index[env_idx, idx],
            "valid_mask": self._cache.valid_mask[env_idx, idx],
        }

    def reset_envs(self, env_mask: Tensor) -> None:
        if self._phase_counter is None:
            return
        mask = torch.as_tensor(env_mask, device=self._phase_counter.device, dtype=torch.bool)
        if mask.shape != self._phase_counter.shape:
            raise ValueError(f"env_mask must have shape {tuple(self._phase_counter.shape)}, got {tuple(mask.shape)}")
        self._phase_counter = torch.where(mask, torch.zeros_like(self._phase_counter), self._phase_counter)
        if self._pending_reset_mask is None or self._pending_reset_mask.shape != mask.shape:
            self._pending_reset_mask = torch.zeros_like(mask, dtype=torch.bool)
        self._pending_reset_mask = torch.logical_or(self._pending_reset_mask, mask)


__all__ = ["BatchedTrajectoryManager"]
