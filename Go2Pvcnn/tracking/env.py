"""Tracking-specific Isaac Lab environment hooks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from tracking.mdp.rewards import parallelism_tracking_episode_errors


class ParallelismTrackingEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with episode-level Parallelism tracking diagnostics."""

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        tracking_log: dict[str, torch.Tensor] = {}
        if hasattr(self, "_parallelism_tracking_joint_mean_sum"):
            if isinstance(env_ids, slice):
                ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            else:
                ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
            stats = parallelism_tracking_episode_errors(self)
            metric_names = (
                ("episode_joint_mean_error", "Episode_Tracking/episode_joint_mean_error"),
                ("episode_joint_max_error", "Episode_Tracking/episode_joint_max_error"),
                ("episode_foot_mean_error", "Episode_Tracking/episode_foot_mean_error"),
                ("episode_foot_max_error", "Episode_Tracking/episode_foot_max_error"),
                ("episode_lin_vel_error", "Episode_Tracking/episode_reference_root_lin_vel_error"),
                ("episode_ang_vel_error", "Episode_Tracking/episode_reference_root_ang_vel_error"),
            )
            for source_name, log_name in metric_names:
                tracking_log[log_name] = stats[source_name].index_select(0, ids).mean().detach()

        super()._reset_idx(env_ids)
        if tracking_log:
            self.extras.setdefault("log", {}).update(tracking_log)
