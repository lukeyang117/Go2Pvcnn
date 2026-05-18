"""Fixed-interval global-sync trajectory manager for MPC backend."""

from __future__ import annotations

import os
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
from .terrain import build_mpc_terrain_from_scanner
from .types import MpcRobotState


def _normalize_body_name(name: str) -> str:
    normalized = str(name).split("/")[-1]
    normalized = normalized.split(":")[-1]
    return normalized.lower()


class MpcTrajectoryManager:
    """Planner-owned cache manager for MPC reference trajectories."""

    planner_backend = "mpc"

    def __init__(self, cfg, device):
        self._cfg = cfg
        self._device = torch.device(device)
        self._cache: ReferenceTrajectoryCache | None = None
        self._phase_counter: Tensor | None = None
        self._foot_body_ids: Tensor | None = None
        self._last_refresh_step_token = None
        self._manager_step = 0
        self._last_global_replan_step = -10_000
        self._runtime_counters: dict[str, float | int] = {}
        self._max_stale_observed = 0
        self._reference_reward_mask: Tensor | None = None

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
        if self._reference_reward_mask is None or int(self._reference_reward_mask.shape[0]) != num_envs:
            self._reference_reward_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)

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
        sensor_pos = torch.as_tensor(scanner.data.pos_w, dtype=torch.float32, device=self._device)
        sensor_quat = torch.as_tensor(scanner.data.quat_w, dtype=torch.float32, device=self._device)
        return build_mpc_terrain_from_scanner(
            ray_hits,
            world_x_range=world_x_range,
            world_y_range=world_y_range,
            semantic_map=semantic_map,
            sensor_pos_w=sensor_pos,
            sensor_yaw=extract_yaw_batch(sensor_quat),
        )

    def _terrain_subset_from_env(self, env, env_ids: Tensor):
        root = self._env_root(env)
        scanner = self._named_get(root.scene.sensors, self._scanner_name())
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device).reshape(-1)
        if hasattr(scanner, "update_env_ids"):
            data = scanner.update_env_ids(ids)
        else:
            data = scanner.data
        ray_hits = torch.as_tensor(data.ray_hits_w, dtype=torch.float32, device=self._device).index_select(0, ids)
        semantic_map_value = getattr(data, "semantic_map", None)
        semantic_map = None
        if semantic_map_value is not None:
            semantic_map = torch.as_tensor(semantic_map_value, dtype=torch.long, device=self._device).index_select(0, ids)
        world_x_range, world_y_range = self._terrain_ranges_from_scanner(scanner)
        sensor_pos = torch.as_tensor(data.pos_w, dtype=torch.float32, device=self._device).index_select(0, ids)
        sensor_quat = torch.as_tensor(data.quat_w, dtype=torch.float32, device=self._device).index_select(0, ids)
        return build_mpc_terrain_from_scanner(
            ray_hits,
            world_x_range=world_x_range,
            world_y_range=world_y_range,
            semantic_map=semantic_map,
            sensor_pos_w=sensor_pos,
            sensor_yaw=extract_yaw_batch(sensor_quat),
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
    def _sample_global_rows(num_envs: int, sample_count: int, *, device: torch.device) -> tuple[Tensor, Tensor]:
        if num_envs <= 0:
            empty_ids = torch.empty(0, dtype=torch.long, device=device)
            return empty_ids, torch.zeros(0, dtype=torch.bool, device=device)
        count = min(max(1, int(sample_count)), int(num_envs))
        if count >= int(num_envs):
            ids = torch.arange(num_envs, dtype=torch.long, device=device)
        else:
            ids = torch.randperm(num_envs, device=device)[:count]
        selected = torch.zeros(num_envs, dtype=torch.bool, device=device)
        selected[ids] = True
        return ids, selected

    @staticmethod
    def _optional_count(value: Tensor, *, enabled: bool) -> int:
        return int(torch.count_nonzero(value).item()) if enabled else 0

    @staticmethod
    def _optional_max(value: Tensor, *, enabled: bool) -> int:
        return int(torch.amax(value).item()) if enabled and int(value.numel()) > 0 else 0

    def _profile_now(self, *, sync: bool) -> float:
        if sync and self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
        return time.perf_counter()

    def _record_runtime_counters(
        self,
        *,
        cfg: MpcPlannerCfg,
        num_envs: int,
        global_due_count: int,
        sampled_plan_count: int,
        max_stale_observed: int,
        global_due: bool,
        planner_ms: float,
        cache_ms: float,
    ) -> None:
        if not bool(cfg.diagnostics.emit_runtime_counters):
            return
        self._runtime_counters = {
            "num_envs": int(num_envs),
            "global_due": bool(global_due),
            "global_due_count": int(global_due_count),
            "sampled_plan_count": int(sampled_plan_count),
            "max_stale_observed": int(max_stale_observed),
            "planner_ms": float(planner_ms),
            "cache_ms": float(cache_ms),
        }

    def runtime_counters(self) -> dict[str, float | int]:
        return dict(self._runtime_counters)

    def _debug_nonfinite_result(self, result, selected_ids: Tensor) -> None:
        if os.environ.get("T302G_DEBUG_MPC_FINITE", "0").strip() != "1":
            return

        fields = {
            "root_pos": result.root_pos,
            "root_rpy": result.root_rpy,
            "foot_pos": result.foot_pos,
            "joint_angles": result.joint_angles,
            "planned_touchdown_w": result.planned_touchdown_w,
            "cost_total": result.cost_total,
        }
        bad_fields: list[str] = []
        for name, value in fields.items():
            tensor = torch.as_tensor(value)
            if torch.any(~torch.isfinite(tensor)):
                bad_fields.append(name)
        if not bad_fields:
            return

        feasible = torch.as_tensor(getattr(result, "feasible", None), dtype=torch.bool, device=self._device)
        safe_fallback = torch.as_tensor(getattr(result, "safe_fallback", None), dtype=torch.bool, device=self._device)
        print(
            "[MPC][NonFiniteResult] "
            f"selected_env_ids={selected_ids.detach().cpu().tolist()} "
            f"bad_fields={bad_fields} "
            f"feasible={feasible.detach().cpu().tolist()} "
            f"safe_fallback={safe_fallback.detach().cpu().tolist()}",
            flush=True,
        )

    def _finite_result_row_mask(self, result, *, num_envs: int) -> Tensor:
        mask = torch.ones(num_envs, dtype=torch.bool, device=self._device)
        fields = (
            result.root_pos,
            result.root_rpy,
            result.foot_pos,
            result.joint_angles,
            result.planned_touchdown_w,
            result.cost_total,
        )
        for value in fields:
            tensor = torch.as_tensor(value, device=self._device)
            if tensor.ndim == 0:
                mask = torch.logical_and(mask, torch.isfinite(tensor).expand_as(mask))
                continue
            reduce_dims = tuple(range(1, tensor.ndim))
            row_finite = torch.all(torch.isfinite(tensor), dim=reduce_dims) if reduce_dims else torch.isfinite(tensor)
            mask = torch.logical_and(mask, row_finite.to(dtype=torch.bool, device=self._device))
        return mask

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
        assert self._reference_reward_mask is not None

        cache_valid = self._cache_shape_valid(num_envs=num_envs, horizon=horizon)
        first_mask = torch.ones(num_envs, dtype=torch.bool, device=self._device) if not cache_valid else torch.zeros(
            num_envs, dtype=torch.bool, device=self._device
        )
        global_age = self._manager_step - int(self._last_global_replan_step)
        global_due = bool(not cache_valid) or global_age >= int(cfg.runtime.replan_interval_steps)
        if global_due:
            _, selected = self._sample_global_rows(
                num_envs, int(cfg.runtime.parallel_plan_batch_size), device=self._device
            )
        else:
            selected = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        selected_ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
        global_due_count = int(num_envs) if counters_enabled and global_due else 0
        sampled_plan_count = self._optional_count(selected, enabled=counters_enabled)
        max_stale_now = int(global_age) if counters_enabled else 0
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
            sub_states = MpcRobotState(
                root_pos=states.root_pos.index_select(0, selected_ids),
                root_rpy=states.root_rpy.index_select(0, selected_ids),
                foot_pos=states.foot_pos.index_select(0, selected_ids),
                joint_angles=states.joint_angles.index_select(0, selected_ids),
                foot_vel=states.foot_vel.index_select(0, selected_ids) if states.foot_vel is not None else None,
            )
            sub_command = command.index_select(0, selected_ids)
            sub_terrain = self._terrain_subset_from_env(env, selected_ids)
            result = plan_segment(sub_terrain, sub_states, sub_command, cfg=cfg)
            self._debug_nonfinite_result(result, selected_ids)

            sub_new_cache = mpc_result_to_reference_cache(result)
            sub_fallback_cache = standstill_cache_from_state(sub_states, horizon=horizon)
            full_new_cache = clone_reference_cache(old_cache)
            full_fallback_cache = clone_reference_cache(old_cache)
            scatter_cache_rows(full_new_cache, sub_new_cache, selected_ids)
            scatter_cache_rows(full_fallback_cache, sub_fallback_cache, selected_ids)

            ok_sub = result_new_ok_mask(result, num_envs=int(selected_ids.shape[0]), device=self._device)
            finite_row_mask = self._finite_result_row_mask(result, num_envs=int(selected_ids.shape[0]))
            gated_ok_sub = torch.logical_and(
                torch.logical_and(ok_sub, finite_row_mask),
                selected.index_select(0, selected_ids),
            )
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
        if int(selected_ids.numel()) > 0:
            self._reference_reward_mask = replace_mask
            self._last_global_replan_step = self._manager_step
        else:
            self._reference_reward_mask = self._reference_reward_mask
        root._trajectory_reference_cache = self._cache
        if counters_enabled:
            total_ms = (self._profile_now(sync=timing_sync) - refresh_t0) * 1000.0
            cache_ms = max(0.0, total_ms - planner_ms)
            self._record_runtime_counters(
                cfg=cfg,
                num_envs=num_envs,
                global_due_count=global_due_count,
                sampled_plan_count=sampled_plan_count,
                max_stale_observed=self._max_stale_observed,
                global_due=global_due,
                planner_ms=planner_ms,
                cache_ms=cache_ms,
            )
        return self._cache

    def reference_reward_mask(self) -> Tensor:
        if self._reference_reward_mask is None:
            raise RuntimeError("trajectory manager has no reference reward mask; call refresh_from_env() first")
        return self._reference_reward_mask

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
            return
        mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._phase_counter.device)
        if mask.shape != self._phase_counter.shape:
            raise ValueError(f"env_mask must have shape {tuple(self._phase_counter.shape)}, got {tuple(mask.shape)}")
        self._phase_counter = torch.where(mask, torch.zeros_like(self._phase_counter), self._phase_counter)
        if self._reference_reward_mask is not None and self._reference_reward_mask.shape == mask.shape:
            self._reference_reward_mask = torch.logical_and(self._reference_reward_mask, torch.logical_not(mask))

    def mark_command_changed(self, env_mask: Tensor | None = None, *_, **__) -> None:
        if env_mask is None:
            if self._reference_reward_mask is not None:
                self._reference_reward_mask = torch.zeros_like(self._reference_reward_mask)
            return
        mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._device)
        if self._reference_reward_mask is not None and self._reference_reward_mask.shape == mask.shape:
            self._reference_reward_mask = torch.logical_and(self._reference_reward_mask, torch.logical_not(mask))


__all__ = ["MpcTrajectoryManager"]
