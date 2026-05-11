"""Asynchronous fixed-budget trajectory manager for MPC backend."""

from __future__ import annotations

import time

import torch
from torch import Tensor

from extension.convention import extract_roll_pitch_batch, extract_yaw_batch
from extension.reference.cache import ReferenceTrajectoryCache

from .adapter import (
    blend_reference_caches,
    clone_reference_cache,
    mpc_result_to_reference_cache,
    result_new_ok_mask,
    scatter_cache_rows,
    standstill_cache_from_state,
)
from .config import MpcPlannerCfg, planner_cfg_from_task_cfg
from .planner import plan_segment
from .terrain import build_mpc_terrain_from_scanner, subset_mpc_terrain
from .types import MpcRobotState


def _normalize_body_name(name: str) -> str:
    normalized = str(name).split("/")[-1]
    normalized = normalized.split(":")[-1]
    return normalized.lower()


class MpcTrajectoryManager:
    """Planner-owned cache manager with asynchronous per-env dirty scheduling."""

    planner_backend = "mpc"

    def __init__(self, cfg, device):
        self._cfg = cfg
        self._device = torch.device(device)
        self._cache: ReferenceTrajectoryCache | None = None
        self._phase_counter: Tensor | None = None
        self._pending_reset_mask: Tensor | None = None
        self._pending_command_mask: Tensor | None = None
        self._last_replan_step: Tensor | None = None
        self._foot_body_ids: Tensor | None = None
        self._last_refresh_step_token = None
        self._manager_step = 0
        self._runtime_counters: dict[str, float | int] = {}
        self._max_stale_observed = 0

    @staticmethod
    def _named_get(container, name: str):
        try:
            return container[name]
        except Exception:  # noqa: BLE001 - Isaac containers are duck-typed
            return getattr(container, name)

    @staticmethod
    def _env_root(env):
        return getattr(env, "unwrapped", env)

    @staticmethod
    def _host_step_token(root):
        return getattr(root, "common_step_counter", getattr(root, "_trajectory_step_token", None))

    def _planner_cfg(self) -> MpcPlannerCfg:
        return planner_cfg_from_task_cfg(self._cfg)

    def horizon_steps(self) -> int:
        return int(self._planner_cfg().runtime.horizon_steps)

    def _command_name(self) -> str:
        return str(getattr(self._cfg, "reference_command_name", "base_velocity"))

    def _scanner_name(self) -> str:
        return str(getattr(self._cfg, "reference_height_scanner_name", "height_scanner"))

    def _ensure_state(self, num_envs: int) -> None:
        if self._phase_counter is None or int(self._phase_counter.shape[0]) != num_envs:
            self._phase_counter = torch.zeros(num_envs, dtype=torch.long, device=self._device)
        if self._pending_reset_mask is None or int(self._pending_reset_mask.shape[0]) != num_envs:
            self._pending_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        if self._pending_command_mask is None or int(self._pending_command_mask.shape[0]) != num_envs:
            self._pending_command_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        if self._last_replan_step is None or int(self._last_replan_step.shape[0]) != num_envs:
            self._last_replan_step = torch.full((num_envs,), -10_000, dtype=torch.long, device=self._device)

    def _foot_ids(self, robot) -> Tensor:
        if self._foot_body_ids is None:
            body_ids, body_names = robot.find_bodies(".*_foot")
            ids = torch.as_tensor(body_ids, dtype=torch.long, device=self._device)
            if body_names:
                name_to_id = {
                    _normalize_body_name(name): int(body_id)
                    for name, body_id in zip(body_names, body_ids)
                }
                planner_ids: list[int] = []
                for planner_name in ("fl_foot", "fr_foot", "rl_foot", "rr_foot"):
                    body_id = name_to_id.get(planner_name)
                    if body_id is None:
                        planner_ids = []
                        break
                    planner_ids.append(int(body_id))
                if planner_ids:
                    ids = torch.as_tensor(planner_ids, dtype=torch.long, device=self._device)
            self._foot_body_ids = ids
        return self._foot_body_ids

    def _state_from_env(self, env) -> MpcRobotState:
        root = self._env_root(env)
        robot = self._named_get(root.scene, "robot")
        data = robot.data
        foot_ids = self._foot_ids(robot)
        root_quat = torch.as_tensor(data.root_quat_w, dtype=torch.float32, device=self._device)
        roll, pitch = extract_roll_pitch_batch(root_quat)
        yaw = extract_yaw_batch(root_quat)
        joint_pos = torch.as_tensor(data.joint_pos, dtype=torch.float32, device=self._device)
        return MpcRobotState(
            root_pos=torch.as_tensor(data.root_pos_w, dtype=torch.float32, device=self._device),
            root_rpy=torch.stack((roll, pitch, yaw), dim=-1),
            joint_angles=joint_pos,
            foot_pos=torch.as_tensor(data.body_pos_w[:, foot_ids, :], dtype=torch.float32, device=self._device),
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

    def _terrain_from_env(self, env):
        root = self._env_root(env)
        scanner = self._named_get(root.scene.sensors, self._scanner_name())
        ray_hits = torch.as_tensor(scanner.data.ray_hits_w, dtype=torch.float32, device=self._device)
        semantic_map_value = getattr(scanner.data, "semantic_map", None)
        semantic_map = None
        if semantic_map_value is not None:
            semantic_map = torch.as_tensor(semantic_map_value, dtype=torch.long, device=self._device)
        world_x_range, world_y_range = self._terrain_ranges_from_scanner(scanner)
        return build_mpc_terrain_from_scanner(
            ray_hits,
            world_x_range=world_x_range,
            world_y_range=world_y_range,
            semantic_map=semantic_map,
        )

    def _commands_from_env(self, env) -> Tensor:
        root = self._env_root(env)
        command = root.command_manager.get_command(self._command_name())
        return torch.as_tensor(command, dtype=torch.float32, device=self._device)

    def _episode_length_buf_from_env(self, env) -> Tensor:
        root = self._env_root(env)
        return torch.as_tensor(root.episode_length_buf, dtype=torch.long, device=self._device)

    def _cache_shape_valid(self, *, num_envs: int, horizon: int) -> bool:
        if self._cache is None or self._cache.root_pos_w is None:
            return False
        if not self._cache.is_ready():
            return False
        return (
            self._cache.root_pos_w.ndim == 3
            and int(self._cache.root_pos_w.shape[0]) == num_envs
            and int(self._cache.root_pos_w.shape[1]) == horizon
        )

    @staticmethod
    def _select_dirty_rows(score: Tensor, budget: int) -> tuple[Tensor, Tensor]:
        if int(score.shape[0]) == 0:
            empty_ids = torch.empty(0, dtype=torch.long, device=score.device)
            return empty_ids, torch.zeros_like(score, dtype=torch.bool)
        k = min(max(1, int(budget)), int(score.shape[0]))
        values, idx = torch.topk(score, k=k, dim=0, largest=True, sorted=False)
        valid = values > 0
        selected = torch.zeros_like(score, dtype=torch.bool)
        selected.scatter_(0, idx, valid)
        return idx, selected

    def _profile_now(self, *, sync: bool) -> float:
        if sync and self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
        return time.perf_counter()

    def _record_runtime_counters(
        self,
        *,
        cfg: MpcPlannerCfg,
        num_envs: int,
        dirty_count: int,
        selected_dirty_count: int,
        dirty_backlog: int,
        max_stale_observed: int,
        planner_ms: float,
        cache_ms: float,
    ) -> None:
        if not bool(cfg.diagnostics.emit_runtime_counters):
            return
        self._runtime_counters = {
            "num_envs": int(num_envs),
            "dirty_count": int(dirty_count),
            "selected_dirty_count": int(selected_dirty_count),
            "dirty_backlog": int(dirty_backlog),
            "max_stale_observed": int(max_stale_observed),
            "planner_ms": float(planner_ms),
            "cache_ms": float(cache_ms),
        }

    def runtime_counters(self) -> dict[str, float | int]:
        return dict(self._runtime_counters)

    def refresh_from_env(self, env):
        root = self._env_root(env)
        step_token = self._host_step_token(root)
        if step_token is not None and self._cache is not None and self._last_refresh_step_token == step_token:
            root._trajectory_reference_cache = self._cache
            return self._cache

        cfg = self._planner_cfg()
        counters_enabled = bool(cfg.diagnostics.emit_runtime_counters)
        timing_sync = counters_enabled and bool(cfg.diagnostics.profile_cuda_sync)
        refresh_t0 = self._profile_now(sync=timing_sync) if counters_enabled else 0.0
        planner_ms = 0.0
        horizon = int(cfg.runtime.horizon_steps)
        self._manager_step += 1

        episode_length_buf = self._episode_length_buf_from_env(env)
        num_envs = int(episode_length_buf.shape[0])
        self._ensure_state(num_envs)
        assert self._phase_counter is not None
        assert self._pending_reset_mask is not None
        assert self._pending_command_mask is not None
        assert self._last_replan_step is not None

        cache_valid = self._cache_shape_valid(num_envs=num_envs, horizon=horizon)
        first_mask = torch.ones(num_envs, dtype=torch.bool, device=self._device) if not cache_valid else torch.zeros(
            num_envs, dtype=torch.bool, device=self._device
        )
        age = torch.full_like(self._last_replan_step, self._manager_step) - self._last_replan_step
        interval_mask = age >= int(cfg.runtime.replan_interval_steps)
        stale_mask = age >= int(cfg.runtime.max_stale_steps)

        score = torch.zeros(num_envs, dtype=torch.float32, device=self._device)
        score = torch.where(interval_mask, torch.full_like(score, 1.0), score)
        score = torch.where(stale_mask, torch.full_like(score, 2.0), score)
        score = torch.where(self._pending_command_mask, torch.full_like(score, 3.0), score)
        score = torch.where(self._pending_reset_mask, torch.full_like(score, 4.0), score)
        score = torch.where(first_mask, torch.full_like(score, 5.0), score)
        _, selected = self._select_dirty_rows(score, int(cfg.runtime.max_dirty_envs_per_step))
        selected_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
        dirty_count = int(torch.count_nonzero(score > 0).item())
        selected_dirty_count = int(torch.count_nonzero(selected).item())
        dirty_backlog = max(0, dirty_count - selected_dirty_count)
        max_stale_now = int(torch.amax(age).item()) if int(age.numel()) > 0 else 0
        self._max_stale_observed = max(self._max_stale_observed, max_stale_now)

        if not cache_valid:
            states_full = self._state_from_env(env)
            self._cache = standstill_cache_from_state(states_full, horizon=horizon)
        assert self._cache is not None

        replace_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        fallback_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        old_cache = self._cache
        if int(selected_ids.numel()) > 0:
            plan_t0 = self._profile_now(sync=timing_sync) if counters_enabled else 0.0
            states = self._state_from_env(env)
            command = self._commands_from_env(env)
            terrain = self._terrain_from_env(env)
            sub_states = MpcRobotState(
                root_pos=states.root_pos.index_select(0, selected_ids),
                root_rpy=states.root_rpy.index_select(0, selected_ids),
                foot_pos=states.foot_pos.index_select(0, selected_ids),
                joint_angles=states.joint_angles.index_select(0, selected_ids),
                foot_vel=states.foot_vel.index_select(0, selected_ids) if states.foot_vel is not None else None,
            )
            sub_command = command.index_select(0, selected_ids)
            sub_terrain = subset_mpc_terrain(terrain, selected_ids)
            result = plan_segment(sub_terrain, sub_states, sub_command, cfg=cfg)

            sub_new_cache = mpc_result_to_reference_cache(result)
            sub_fallback_cache = standstill_cache_from_state(sub_states, horizon=horizon)
            full_new_cache = clone_reference_cache(old_cache)
            full_fallback_cache = clone_reference_cache(old_cache)
            scatter_cache_rows(full_new_cache, sub_new_cache, selected_ids)
            scatter_cache_rows(full_fallback_cache, sub_fallback_cache, selected_ids)

            ok_sub = result_new_ok_mask(result, num_envs=int(selected_ids.shape[0]), device=self._device)
            gated_ok_sub = torch.logical_and(ok_sub, selected.index_select(0, selected_ids))
            replace_mask.scatter_(0, selected_ids, gated_ok_sub)
            fallback_mask.scatter_(
                0,
                selected_ids,
                torch.logical_and(selected.index_select(0, selected_ids), torch.logical_not(gated_ok_sub)),
            )
            self._cache = blend_reference_caches(
                old_cache=old_cache,
                new_cache=full_new_cache,
                fallback_cache=full_fallback_cache,
                replace_mask=replace_mask,
                fallback_mask=fallback_mask,
            )
            if counters_enabled:
                planner_ms = (self._profile_now(sync=timing_sync) - plan_t0) * 1000.0
        else:
            self._cache = old_cache

        selected_any = selected

        if step_token is None:
            self._last_refresh_step_token = object()
        else:
            self._last_refresh_step_token = step_token

        assert self._cache is not None
        max_phase = int(self._cache.root_pos_w.shape[1]) - 1
        advanced = torch.clamp(self._phase_counter + 1, max=max_phase)
        # First-cache rows that miss the fixed replan budget still receive a
        # standstill fallback cache; keep their phase at the cache origin.
        reset_phase = torch.logical_or(torch.logical_or(replace_mask, fallback_mask), first_mask)
        self._phase_counter = torch.where(reset_phase, torch.zeros_like(advanced), advanced)
        self._last_replan_step = torch.where(selected_any, torch.full_like(self._last_replan_step, self._manager_step), self._last_replan_step)
        self._pending_reset_mask = torch.logical_and(self._pending_reset_mask, torch.logical_not(selected_any))
        self._pending_command_mask = torch.logical_and(self._pending_command_mask, torch.logical_not(selected_any))
        root._trajectory_reference_cache = self._cache
        if counters_enabled:
            total_ms = (self._profile_now(sync=timing_sync) - refresh_t0) * 1000.0
            cache_ms = max(0.0, total_ms - planner_ms)
            self._record_runtime_counters(
                cfg=cfg,
                num_envs=num_envs,
                dirty_count=dirty_count,
                selected_dirty_count=selected_dirty_count,
                dirty_backlog=dirty_backlog,
                max_stale_observed=self._max_stale_observed,
                planner_ms=planner_ms,
                cache_ms=cache_ms,
            )
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
            self._pending_reset_mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._device).clone()
            return
        mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._phase_counter.device)
        if mask.shape != self._phase_counter.shape:
            raise ValueError(f"env_mask must have shape {tuple(self._phase_counter.shape)}, got {tuple(mask.shape)}")
        if self._pending_reset_mask is None or self._pending_reset_mask.shape != mask.shape:
            self._pending_reset_mask = torch.zeros_like(mask)
        self._pending_reset_mask = torch.logical_or(self._pending_reset_mask, mask)
        self._phase_counter = torch.where(mask, torch.zeros_like(self._phase_counter), self._phase_counter)

    def mark_command_changed(self, env_mask: Tensor | None = None, *_, **__) -> None:
        if env_mask is None:
            if self._pending_command_mask is None:
                return
            self._pending_command_mask = torch.ones_like(self._pending_command_mask)
            return
        mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._device)
        if self._pending_command_mask is None or self._pending_command_mask.shape != mask.shape:
            self._pending_command_mask = mask.clone()
        else:
            self._pending_command_mask = torch.logical_or(self._pending_command_mask, mask)


__all__ = ["MpcTrajectoryManager"]
