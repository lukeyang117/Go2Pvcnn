"""Tracking-specific Isaac Lab environment hooks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager
from tracking.mdp.rewards import (
    parallelism_obstacle_episode_metrics,
    parallelism_tracking_episode_errors,
    reset_parallelism_obstacle_stats,
    reset_parallelism_tracking_error_stats,
)


_LEG_NAMES = ("FL", "FR", "RL", "RR")


class ParallelismTrackingEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with episode-level Parallelism tracking diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Tracking metrics used to be updated as a side effect of reference
        # rewards. Keep that diagnostic stream when the distillation task
        # removes those rewards from the optimization objective.
        original_compute = self.reward_manager.compute

        def compute_with_tracking(*compute_args, **compute_kwargs):
            reward = original_compute(*compute_args, **compute_kwargs)
            update_parallelism_tracking_error_stats(self)
            return reward

        self.reward_manager.compute = compute_with_tracking

    def step(self, action):
        get_parallelism_reference_manager(self).prepare_step_reference()
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        tracking_log: dict[str, torch.Tensor] = {}
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        violation_count = getattr(self, "_parallelism_joint_violation_count", None)
        if violation_count is not None:
            violation_count[ids] = 0
        if hasattr(self, "_parallelism_tracking_joint_mean_sum"):
            stats = parallelism_tracking_episode_errors(self)
            metric_names = (
                ("episode_joint_mean_error", "Episode_Tracking/episode_joint_mean_error"),
                ("episode_joint_max_error", "Episode_Tracking/episode_joint_max_error"),
                ("episode_foot_mean_error", "Episode_Tracking/episode_foot_mean_error"),
                ("episode_foot_max_error", "Episode_Tracking/episode_foot_max_error"),
                ("episode_root_pos_error", "Episode_Tracking/episode_reference_root_pos_error"),
                ("episode_root_rot_error", "Episode_Tracking/episode_reference_root_rot_error"),
                (
                    "episode_active_swing_foot_mean_error",
                    "Episode_Tracking/episode_active_swing_foot_mean_error",
                ),
                (
                    "episode_active_swing_foot_max_error",
                    "Episode_Tracking/episode_active_swing_foot_max_error",
                ),
                (
                    "episode_active_swing_foot_z_mean_error",
                    "Episode_Tracking/episode_active_swing_foot_z_mean_error",
                ),
                (
                    "episode_active_swing_foot_z_max_error",
                    "Episode_Tracking/episode_active_swing_foot_z_max_error",
                ),
            )
            for source_name, log_name in metric_names:
                tracking_log[log_name] = stats[source_name].index_select(0, ids).mean().detach()
            per_leg_metrics = (
                (
                    "episode_swing_foot_mean_error_per_leg",
                    "Episode_Tracking/episode_swing_foot_{leg_name}_mean_error",
                ),
                (
                    "episode_swing_foot_max_error_per_leg",
                    "Episode_Tracking/episode_swing_foot_{leg_name}_max_error",
                ),
                (
                    "episode_swing_foot_z_mean_error_per_leg",
                    "Episode_Tracking/episode_swing_foot_{leg_name}_z_mean_error",
                ),
                (
                    "episode_joint_max_error_per_leg",
                    "Episode_Tracking/episode_joint_{leg_name}_max_error",
                ),
            )
            for source_name, log_name in per_leg_metrics:
                values = stats[source_name].index_select(0, ids)
                for leg_index, leg_name in enumerate(_LEG_NAMES):
                    tracking_log[log_name.format(leg_name=leg_name)] = values[:, leg_index].mean().detach()
            obstacle_stats = parallelism_obstacle_episode_metrics(self)
            for source_name, log_name in (
                ("geometry_collision_ratio", "Episode_Obstacle/geometry_collision_ratio"),
                ("active_swing_foot_on_small_ratio", "Episode_Obstacle/active_swing_foot_on_small_ratio"),
                (
                    "active_swing_foot_on_small_no_collision_ratio",
                    "Episode_Obstacle/active_swing_foot_on_small_no_collision_ratio",
                ),
                ("standstill_ratio", "Episode_Obstacle/standstill_ratio"),
                ("reference_valid_ratio", "Episode_Obstacle/reference_valid_ratio"),
            ):
                if source_name in obstacle_stats:
                    tracking_log[log_name] = obstacle_stats[source_name].index_select(0, ids).mean().detach()

        super()._reset_idx(env_ids)
        get_parallelism_reference_manager(self).on_environment_reset(ids)
        if tracking_log:
            self.extras.setdefault("log", {}).update(tracking_log)
            reset_parallelism_tracking_error_stats(self, ids)
            reset_parallelism_obstacle_stats(self, ids)
