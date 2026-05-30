"""Full-batch trajectory manager for the together planner backend."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from extension.convention import extract_roll_pitch_batch, extract_yaw_batch
from extension.reference.cache import ReferenceTrajectoryCache

from .adapter import (
    blend_reference_caches,
    result_new_ok_mask,
    standstill_cache_from_state,
    together_result_to_reference_cache,
)
from .config import TogetherPlannerConfig
from .planner import build_together_terrain_from_scanner, plan_segment
from .terrain import TogetherPlannerTerrain
from .types import TogetherRobotState


class TogetherTrajectoryManager:
    """Manager-owned cadence/cache adapter for fixed-shape full-batch planning."""

    planner_backend = "together"

    def __init__(self, cfg, device):
        self._cfg = cfg
        self._device = torch.device(device)
        self._cache: ReferenceTrajectoryCache | None = None
        self._phase_counter: Tensor | None = None
        self._pending_reset_mask: Tensor | None = None
        self._pending_command_mask: Tensor | None = None
        self._pending_reset_dirty = False
        self._pending_command_dirty = False
        self._pending_all_commands_dirty = False
        self._foot_body_ids: Tensor | None = None
        self._last_refresh_step_token = None
        self._steps_since_attempt = 0

    @staticmethod
    def _named_get(container, name: str):
        try:
            return container[name]
        except Exception:  # noqa: BLE001 - duck-typed tests and Isaac containers differ here
            return getattr(container, name)

    @staticmethod
    def _env_root(env):
        return getattr(env, "unwrapped", env)

    @staticmethod
    def _cfg_dt(cfg) -> float:
        value = getattr(cfg, "plan_dt", None)
        if value is not None:
            return float(value)
        value = getattr(cfg, "dt", None)
        if value is not None:
            return float(value)
        value = getattr(cfg, "step_dt", None)
        if value is not None:
            return float(value)
        sim = getattr(cfg, "sim", None)
        sim_dt = getattr(sim, "dt", None) if sim is not None else None
        decimation = getattr(cfg, "decimation", None)
        if sim_dt is not None and decimation is not None:
            return float(sim_dt) * float(decimation)
        return 0.02

    def _horizon(self) -> int:
        return int(getattr(self._cfg, "reference_trajectory_horizon", TogetherPlannerConfig().horizon_steps))

    def _interval(self) -> int:
        return int(getattr(self._cfg, "reference_replan_interval_steps", self._horizon()))

    def _command_name(self) -> str:
        return str(getattr(self._cfg, "reference_command_name", "base_velocity"))

    def _scanner_name(self) -> str:
        return str(getattr(self._cfg, "reference_height_scanner_name", "height_scanner"))

    def _planner_cfg(self) -> TogetherPlannerConfig:
        base = TogetherPlannerConfig()
        return replace(
            base,
            dt=float(getattr(self._cfg, "plan_dt", base.dt)),
            horizon_s=float(self._horizon()) * float(getattr(self._cfg, "plan_dt", base.dt)),
            horizon_steps=self._horizon(),
            step_freq=float(getattr(self._cfg, "step_freq", base.step_freq)),
            swing_height=float(getattr(self._cfg, "step_height", base.swing_height)),
            duty_factor=float(getattr(self._cfg, "together_duty_factor", base.duty_factor)),
            idle_command_eps=float(getattr(self._cfg, "idle_command_eps", base.idle_command_eps)),
            support_search_radius=float(getattr(self._cfg, "support_search_radius", base.support_search_radius)),
            support_search_step=float(getattr(self._cfg, "support_search_step", base.support_search_step)),
        )

    def _ensure_masks(self, num_envs: int) -> None:
        if self._phase_counter is None or int(self._phase_counter.shape[0]) != num_envs:
            self._phase_counter = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        if self._pending_reset_mask is None or int(self._pending_reset_mask.shape[0]) != num_envs:
            self._pending_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        if self._pending_command_mask is None or int(self._pending_command_mask.shape[0]) != num_envs:
            self._pending_command_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        if self._pending_all_commands_dirty:
            self._pending_command_mask = torch.ones(num_envs, dtype=torch.bool, device=self._device)
            self._pending_all_commands_dirty = False

    def _foot_ids(self, robot) -> Tensor:
        if self._foot_body_ids is None:
            body_ids, _ = robot.find_bodies(".*_foot")
            self._foot_body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=self._device)
        return self._foot_body_ids

    def _batched_state_from_env(self, env) -> TogetherRobotState:
        root = self._env_root(env)
        robot = self._named_get(root.scene, "robot")
        data = robot.data
        foot_ids = self._foot_ids(robot)
        root_quat = torch.as_tensor(data.root_quat_w, dtype=torch.float64, device=self._device)
        roll, pitch = extract_roll_pitch_batch(root_quat)
        yaw = extract_yaw_batch(root_quat)
        return TogetherRobotState(
            root_pos=torch.as_tensor(data.root_pos_w, dtype=torch.float64, device=self._device),
            root_rpy=torch.stack((roll, pitch, yaw), dim=-1),
            joint_angles=torch.as_tensor(data.joint_pos, dtype=torch.float64, device=self._device),
            foot_pos=torch.as_tensor(data.body_pos_w[:, foot_ids, :], dtype=torch.float64, device=self._device),
        )

    @staticmethod
    def _terrain_ranges_from_scanner(scanner) -> tuple[tuple[float, float], tuple[float, float]]:
        pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
        size = getattr(pattern_cfg, "size", None)
        if size is None:
            return (-0.75, 0.75), (-0.75, 0.75)
        half_x = 0.5 * float(size[0])
        half_y = 0.5 * float(size[1])
        return (-half_x, half_x), (-half_y, half_y)

    def _terrain_from_env(self, env) -> TogetherPlannerTerrain:
        root = self._env_root(env)
        scanner = self._named_get(root.scene.sensors, self._scanner_name())
        ray_hits = torch.as_tensor(scanner.data.ray_hits_w, dtype=torch.float64, device=self._device)
        semantic_map_value = getattr(scanner.data, "semantic_map", None)
        semantic_map = None
        if semantic_map_value is not None:
            semantic_map = torch.as_tensor(semantic_map_value, dtype=torch.long, device=self._device)
        world_x_range, world_y_range = self._terrain_ranges_from_scanner(scanner)
        return build_together_terrain_from_scanner(
            ray_hits,
            world_x_range=world_x_range,
            world_y_range=world_y_range,
            semantic_map=semantic_map,
        )

    def _commands_from_env(self, env) -> Tensor:
        root = self._env_root(env)
        command = root.command_manager.get_command(self._command_name())
        return torch.as_tensor(command, dtype=torch.float64, device=self._device)

    def _episode_length_buf_from_env(self, env) -> Tensor:
        root = self._env_root(env)
        return torch.as_tensor(root.episode_length_buf, dtype=torch.long, device=self._device)

    @staticmethod
    def _host_step_token(root):
        return getattr(root, "common_step_counter", getattr(root, "_trajectory_step_token", None))

    def _cache_shape_valid(self, *, num_envs: int) -> bool:
        if self._cache is None or self._cache.root_pos_w is None:
            return False
        if not self._cache.is_ready():
            return False
        if self._cache.horizon_length() != self._horizon():
            return False
        return self._cache.root_pos_w.ndim == 3 and int(self._cache.root_pos_w.shape[0]) == num_envs

    def _consume_pending_masks(self, *, num_envs: int, device: torch.device) -> tuple[Tensor, Tensor]:
        self._ensure_masks(num_envs)
        assert self._pending_reset_mask is not None
        assert self._pending_command_mask is not None
        return (
            self._pending_reset_mask.to(device=device),
            self._pending_command_mask.to(device=device),
        )

    def _clear_pending_masks(self) -> None:
        if self._pending_reset_mask is not None:
            self._pending_reset_mask = torch.zeros_like(self._pending_reset_mask)
        if self._pending_command_mask is not None:
            self._pending_command_mask = torch.zeros_like(self._pending_command_mask)
        self._pending_reset_dirty = False
        self._pending_command_dirty = False

    def refresh_from_env(self, env):
        root = self._env_root(env)
        step_token = self._host_step_token(root)
        if self._cache is not None and self._last_refresh_step_token == step_token:
            root._trajectory_reference_cache = self._cache
            return self._cache

        episode_length_buf = self._episode_length_buf_from_env(env)
        num_envs = int(episode_length_buf.shape[0])
        self._ensure_masks(num_envs)
        assert self._phase_counter is not None

        cache_valid = self._cache_shape_valid(num_envs=num_envs)
        first_cache = not cache_valid
        if cache_valid:
            self._steps_since_attempt += 1
        interval_due = cache_valid and self._steps_since_attempt >= self._interval()
        trigger_attempt = first_cache or self._pending_reset_dirty or self._pending_command_dirty or interval_due

        if not trigger_attempt:
            max_phase = int(self._cache.root_pos_w.shape[1]) - 1  # type: ignore[union-attr]
            self._phase_counter = torch.clamp(self._phase_counter + 1, max=max_phase)
            self._last_refresh_step_token = step_token
            root._trajectory_reference_cache = self._cache
            return self._cache

        terrain = self._terrain_from_env(env)
        states = self._batched_state_from_env(env)
        commands = self._commands_from_env(env)
        result = plan_segment(
            terrain,
            states,
            commands,
            cfg=self._planner_cfg(),
        )
        new_cache = together_result_to_reference_cache(result)
        fallback_cache = standstill_cache_from_state(states, horizon=self._horizon())
        old_cache = self._cache if cache_valid else fallback_cache
        pending_reset, pending_command = self._consume_pending_masks(num_envs=num_envs, device=new_cache.root_pos_w.device)
        all_rows = torch.ones(num_envs, dtype=torch.bool, device=new_cache.root_pos_w.device)
        first_or_invalid = all_rows if first_cache else torch.zeros(num_envs, dtype=torch.bool, device=new_cache.root_pos_w.device)
        interval_mask = all_rows if interval_due else torch.zeros(num_envs, dtype=torch.bool, device=new_cache.root_pos_w.device)
        replan_mask = torch.logical_or(torch.logical_or(pending_reset, pending_command), torch.logical_or(interval_mask, first_or_invalid))
        must_replace = torch.logical_or(first_or_invalid, pending_reset)
        soft_replan = torch.logical_and(replan_mask, torch.logical_not(must_replace))
        new_ok = result_new_ok_mask(result, num_envs=num_envs, device=new_cache.root_pos_w.device)
        replace = torch.logical_or(torch.logical_and(must_replace, new_ok), torch.logical_and(soft_replan, new_ok))
        fallback = torch.logical_and(must_replace, torch.logical_not(new_ok))

        self._cache = blend_reference_caches(
            old_cache=old_cache,
            new_cache=new_cache,
            fallback_cache=fallback_cache,
            replace_mask=replace,
            fallback_mask=fallback,
        )

        max_phase = int(self._cache.root_pos_w.shape[1]) - 1
        advanced = torch.clamp(self._phase_counter.to(device=new_cache.root_pos_w.device) + 1, max=max_phase)
        reset_phase = torch.logical_or(replace, fallback).to(device=advanced.device)
        self._phase_counter = torch.where(reset_phase, torch.zeros_like(advanced), advanced)
        self._steps_since_attempt = 0
        self._clear_pending_masks()
        self._last_refresh_step_token = step_token
        root._trajectory_reference_cache = self._cache
        return self._cache

    def current_reference(self) -> dict[str, Tensor]:
        if self._cache is None or self._phase_counter is None:
            raise RuntimeError("trajectory manager has no cached trajectory; call refresh_from_env() first")

        idx = self.current_frame_ids()
        env_idx = torch.arange(idx.shape[0], device=idx.device)
        return {
            "root_pos_w": self._cache.root_pos_w[env_idx, idx],
            "root_quat_w": self._cache.root_quat_w[env_idx, idx],
            "joint_angles": self._cache.joint_angles[env_idx, idx],
            "foot_pos_w": self._cache.foot_pos_w[env_idx, idx],
            "foot_pos_root": self._cache.foot_pos_root[env_idx, idx],
            "contact_state": self._cache.contact_state[env_idx, idx],
            "planned_touchdown_w": self._cache.planned_touchdown_w[env_idx, idx],
            "phase_index": self._cache.phase_index[env_idx, idx],
            "valid_mask": self._cache.valid_mask[env_idx, idx],
        }

    def current_frame_ids(self) -> Tensor:
        if self._cache is None or self._phase_counter is None:
            raise RuntimeError("trajectory manager has no cached trajectory; call refresh_from_env() first")
        return self._phase_counter.clamp(max=int(self._cache.root_pos_w.shape[1]) - 1)

    def reset_envs(self, env_mask: Tensor) -> None:
        if self._phase_counter is None:
            mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._device)
            self._pending_reset_mask = mask.clone()
            self._pending_reset_dirty = True
            return
        mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._phase_counter.device)
        if mask.shape != self._phase_counter.shape:
            raise ValueError(f"env_mask must have shape {tuple(self._phase_counter.shape)}, got {tuple(mask.shape)}")
        if self._pending_reset_mask is None or self._pending_reset_mask.shape != mask.shape:
            self._pending_reset_mask = torch.zeros_like(mask)
        self._pending_reset_mask = torch.logical_or(self._pending_reset_mask, mask)
        self._phase_counter = torch.where(mask, torch.zeros_like(self._phase_counter), self._phase_counter)
        self._pending_reset_dirty = True

    def mark_command_changed(self, env_mask: Tensor | None = None, *_, **__) -> None:
        self._pending_command_dirty = True
        if env_mask is None:
            if self._pending_command_mask is None:
                self._pending_all_commands_dirty = True
            else:
                self._pending_command_mask = torch.ones_like(self._pending_command_mask)
            return
        mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._device)
        if self._pending_command_mask is None or self._pending_command_mask.shape != mask.shape:
            self._pending_command_mask = mask.clone()
        else:
            self._pending_command_mask = torch.logical_or(self._pending_command_mask, mask)


__all__ = ["TogetherTrajectoryManager"]
