"""Fixed-interval batched trajectory manager."""

from __future__ import annotations

import torch
from torch import Tensor

from ..convention import planner_result_to_reference_cache
from .trajectory import batched_generate_trajectory


class BatchedTrajectoryManager:
    def __init__(self, cfg, device):
        self._cfg = cfg
        self._device = torch.device(device)
        self._cache = None
        self._phase_counter: Tensor | None = None
        self._step_counter = 0

    def _ensure_phase_counter(self, num_envs: int) -> None:
        if self._phase_counter is None or int(self._phase_counter.shape[0]) != num_envs:
            self._phase_counter = torch.zeros(num_envs, dtype=torch.long, device=self._device)

    def step(self, terrain, states, commands):
        num_envs = int(states.root_pos.shape[0])
        self._ensure_phase_counter(num_envs)

        if self._step_counter % int(self._cfg.reference_replan_interval_steps) == 0:
            result = batched_generate_trajectory(
                terrain,
                states,
                commands,
                requested_n_frames=int(self._cfg.reference_trajectory_horizon),
                dt=float(self._cfg.dt),
            )
            self._cache = planner_result_to_reference_cache(result)
            self._phase_counter.zero_()

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


__all__ = ["BatchedTrajectoryManager"]
