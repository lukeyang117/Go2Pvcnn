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
from .types import MpcFootholdMemory, MpcRobotState


def _yaw_dominance(command: Tensor) -> Tensor:
    cmd = torch.as_tensor(command, dtype=torch.float32, device=command.device)
    lin = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    yaw = torch.abs(cmd[:, 2])
    return yaw / torch.clamp(lin + yaw, min=1.0e-6)


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
        self._stance_anchor_w: Tensor | None = None
        self._running_foot_rel_body: Tensor | None = None
        self._prev_contact_state: Tensor | None = None
        self._prev_yaw_dominance: Tensor | None = None
        self._yaw_entry_steps: Tensor | None = None
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
        if self._stance_anchor_w is None or int(self._stance_anchor_w.shape[0]) != num_envs:
            self._stance_anchor_w = torch.zeros((num_envs, 4, 3), dtype=torch.float32, device=self._device)
        if self._running_foot_rel_body is None or int(self._running_foot_rel_body.shape[0]) != num_envs:
            self._running_foot_rel_body = torch.zeros((num_envs, 4, 3), dtype=torch.float32, device=self._device)
        if self._prev_contact_state is None or int(self._prev_contact_state.shape[0]) != num_envs:
            self._prev_contact_state = torch.ones((num_envs, 4), dtype=torch.bool, device=self._device)
        if self._prev_yaw_dominance is None or int(self._prev_yaw_dominance.shape[0]) != num_envs:
            self._prev_yaw_dominance = torch.zeros(num_envs, dtype=torch.float32, device=self._device)
        if self._yaw_entry_steps is None or int(self._yaw_entry_steps.shape[0]) != num_envs:
            self._yaw_entry_steps = torch.full((num_envs,), 10_000, dtype=torch.long, device=self._device)

    @staticmethod
    def _foot_rel_body(state: MpcRobotState) -> Tensor:
        root = torch.as_tensor(state.root_pos, dtype=torch.float32, device=state.foot_pos.device)
        rpy = torch.as_tensor(state.root_rpy, dtype=torch.float32, device=state.foot_pos.device)
        foot = torch.as_tensor(state.foot_pos, dtype=torch.float32, device=state.foot_pos.device)
        rel = foot - root.unsqueeze(1)
        yaw = rpy[:, 2]
        cy = torch.cos(yaw).unsqueeze(-1)
        sy = torch.sin(yaw).unsqueeze(-1)
        rel_body_xy = torch.stack(
            (
                cy * rel[..., 0] + sy * rel[..., 1],
                -sy * rel[..., 0] + cy * rel[..., 1],
            ),
            dim=-1,
        )
        return torch.cat((rel_body_xy, rel[..., 2:3]), dim=-1)

    def _initialize_foothold_memory(self, states: MpcRobotState, env_ids: Tensor | None = None) -> None:
        if self._stance_anchor_w is None or self._running_foot_rel_body is None or self._prev_contact_state is None:
            return
        foot = torch.as_tensor(states.foot_pos, dtype=torch.float32, device=self._device)
        rel_body = self._foot_rel_body(states).to(dtype=torch.float32, device=self._device)
        contact = torch.ones((foot.shape[0], 4), dtype=torch.bool, device=self._device)
        if env_ids is None:
            self._stance_anchor_w = foot.clone()
            self._running_foot_rel_body = rel_body.clone()
            self._prev_contact_state = contact
            if self._prev_yaw_dominance is not None:
                self._prev_yaw_dominance.zero_()
            if self._yaw_entry_steps is not None:
                self._yaw_entry_steps.fill_(10_000)
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        self._stance_anchor_w.index_copy_(0, ids, foot)
        self._running_foot_rel_body.index_copy_(0, ids, rel_body)
        self._prev_contact_state.index_copy_(0, ids, contact)
        if self._prev_yaw_dominance is not None:
            self._prev_yaw_dominance.index_fill_(0, ids, 0.0)
        if self._yaw_entry_steps is not None:
            self._yaw_entry_steps.index_fill_(0, ids, 10_000)

    def _foothold_memory_for(self, env_ids: Tensor, command: Tensor, cfg: MpcPlannerCfg) -> MpcFootholdMemory | None:
        if not bool(cfg.runtime.foothold_memory_enabled):
            return None
        if self._running_foot_rel_body is None or self._stance_anchor_w is None:
            return None
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        yaw_dom = _yaw_dominance(command).to(dtype=torch.float32, device=self._device)
        if self._prev_yaw_dominance is None or self._yaw_entry_steps is None:
            ramp = torch.ones_like(yaw_dom)
        else:
            prev = self._prev_yaw_dominance.index_select(0, ids)
            old_steps = self._yaw_entry_steps.index_select(0, ids)
            enter = torch.logical_and(
                yaw_dom > float(cfg.runtime.foothold_yaw_entry_enter_threshold),
                prev <= float(cfg.runtime.foothold_yaw_entry_exit_threshold),
            )
            stay = yaw_dom > float(cfg.runtime.foothold_yaw_entry_enter_threshold)
            new_steps = torch.where(enter, torch.zeros_like(old_steps), torch.where(stay, old_steps + 1, torch.full_like(old_steps, 10_000)))
            self._yaw_entry_steps.index_copy_(0, ids, new_steps)
            self._prev_yaw_dominance.index_copy_(0, ids, yaw_dom)
            ramp = torch.clamp(
                (new_steps.to(dtype=torch.float32) + 1.0) / float(cfg.runtime.foothold_yaw_entry_ramp_steps),
                min=0.0,
                max=1.0,
            )
            ramp = torch.where(new_steps >= 10_000, torch.ones_like(ramp), ramp)
        return MpcFootholdMemory(
            foot_rel_body_seed=self._running_foot_rel_body.index_select(0, ids),
            stance_anchor_w=self._stance_anchor_w.index_select(0, ids),
            yaw_entry_ramp=ramp,
        )

    def _update_foothold_memory(self, states: MpcRobotState, result, env_ids: Tensor, cfg: MpcPlannerCfg) -> None:
        if not bool(cfg.runtime.foothold_memory_enabled):
            return
        if self._stance_anchor_w is None or self._running_foot_rel_body is None or self._prev_contact_state is None:
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        frame_idx = int(result.foot_pos.shape[1]) - 1
        last_contact = torch.as_tensor(result.contact_state[:, frame_idx], dtype=torch.bool, device=self._device)
        prior_contact = self._prev_contact_state.index_select(0, ids)
        touchdown_last = torch.logical_and(last_contact, torch.logical_not(prior_contact))
        last_foot = torch.as_tensor(states.foot_pos, dtype=torch.float32, device=self._device)
        anchor_prev = self._stance_anchor_w.index_select(0, ids)
        update_anchor = torch.logical_or(last_contact, touchdown_last).unsqueeze(-1)
        anchor_new = torch.where(update_anchor, last_foot, anchor_prev)
        self._stance_anchor_w.index_copy_(0, ids, anchor_new)
        self._prev_contact_state.index_copy_(0, ids, last_contact.detach().clone())

        current_rel = self._foot_rel_body(states).to(dtype=torch.float32, device=self._device)
        running_prev = self._running_foot_rel_body.index_select(0, ids)
        touchdown_mask = touchdown_last.unsqueeze(-1).to(dtype=current_rel.dtype)
        contact_mask = last_contact.unsqueeze(-1).to(dtype=current_rel.dtype)
        blended = torch.lerp(running_prev, current_rel, float(cfg.runtime.foothold_touchdown_blend) * touchdown_mask)
        blended = torch.lerp(blended, current_rel, float(cfg.runtime.foothold_contact_blend) * contact_mask)
        self._running_foot_rel_body.index_copy_(0, ids, blended.detach().clone())

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
            self._initialize_foothold_memory(states_full)
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
            reset_or_first_sub = torch.logical_or(
                self._pending_reset_mask.index_select(0, selected_ids),
                first_mask.index_select(0, selected_ids),
            )
            if bool(torch.any(reset_or_first_sub).item()):
                reset_ids = selected_ids[reset_or_first_sub]
                reset_sub_states = MpcRobotState(
                    root_pos=states.root_pos.index_select(0, reset_ids),
                    root_rpy=states.root_rpy.index_select(0, reset_ids),
                    foot_pos=states.foot_pos.index_select(0, reset_ids),
                    joint_angles=states.joint_angles.index_select(0, reset_ids),
                    foot_vel=states.foot_vel.index_select(0, reset_ids) if states.foot_vel is not None else None,
                )
                self._initialize_foothold_memory(reset_sub_states, reset_ids)
            memory = self._foothold_memory_for(selected_ids, sub_command, cfg)
            result = plan_segment(sub_terrain, sub_states, sub_command, cfg=cfg, memory=memory)

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
            self._update_foothold_memory(sub_states, result, selected_ids, cfg)
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
        if self._stance_anchor_w is not None:
            self._stance_anchor_w = torch.where(mask[:, None, None], torch.zeros_like(self._stance_anchor_w), self._stance_anchor_w)
        if self._running_foot_rel_body is not None:
            self._running_foot_rel_body = torch.where(
                mask[:, None, None],
                torch.zeros_like(self._running_foot_rel_body),
                self._running_foot_rel_body,
            )
        if self._prev_contact_state is not None:
            self._prev_contact_state = torch.where(mask[:, None], torch.ones_like(self._prev_contact_state), self._prev_contact_state)
        if self._prev_yaw_dominance is not None:
            self._prev_yaw_dominance = torch.where(mask, torch.zeros_like(self._prev_yaw_dominance), self._prev_yaw_dominance)
        if self._yaw_entry_steps is not None:
            self._yaw_entry_steps = torch.where(mask, torch.full_like(self._yaw_entry_steps, 10_000), self._yaw_entry_steps)

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
