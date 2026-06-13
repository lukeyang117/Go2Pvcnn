"""Velocity command with curriculum support (ranges -> limit_ranges)."""

from __future__ import annotations

from dataclasses import MISSING
from collections.abc import Sequence
import math

import torch

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command with curriculum: ranges expand toward limit_ranges."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class GoalAnchoredVelocityCommand(CommandTerm):
    """Body-frame velocity command anchored to a per-episode world target direction."""

    cfg: "GoalAnchoredVelocityCommandCfg"

    def __init__(self, cfg: "GoalAnchoredVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.vx_abs = torch.zeros(self.num_envs, device=self.device)
        self.vy_abs = torch.zeros(self.num_envs, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired base velocity command in the robot base frame. Shape is ``[num_envs, 3]``."""
        return self.vel_command_b

    def _update_metrics(self):
        max_command_time = float(self.cfg.resampling_time_range[1])
        max_command_step = max(max_command_time / float(self._env.step_dt), 1.0)
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        root_xy = self.robot.data.root_pos_w[ids, :2]
        random_values = torch.empty(ids.numel(), device=self.device)
        theta = random_values.uniform_(-math.pi, math.pi)
        direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        self.goal_xy_w[ids] = root_xy + float(self.cfg.goal_distance) * direction
        self.vx_abs[ids] = random_values.uniform_(*self.cfg.vx_abs_range)
        self.vy_abs[ids] = random_values.uniform_(*self.cfg.vy_abs_range)
        self.is_standing_env[ids] = random_values.uniform_(0.0, 1.0) <= float(self.cfg.rel_standing_envs)

    def _update_command(self):
        root_xy = self.robot.data.root_pos_w[:, :2]
        to_goal = self.goal_xy_w - root_xy
        distance = torch.linalg.norm(to_goal, dim=1).clamp_min(1.0e-6)
        dir_world = to_goal / distance.unsqueeze(-1)

        reached = distance < float(self.cfg.goal_reached_threshold)
        if torch.any(reached):
            self.goal_xy_w[reached] = root_xy[reached] + float(self.cfg.goal_distance) * dir_world[reached]
            to_goal = self.goal_xy_w - root_xy
            distance = torch.linalg.norm(to_goal, dim=1).clamp_min(1.0e-6)
            dir_world = to_goal / distance.unsqueeze(-1)

        heading = self.robot.data.heading_w
        cos_yaw = torch.cos(heading)
        sin_yaw = torch.sin(heading)
        dir_body_x = cos_yaw * dir_world[:, 0] + sin_yaw * dir_world[:, 1]
        dir_body_y = -sin_yaw * dir_world[:, 0] + cos_yaw * dir_world[:, 1]

        self.vel_command_b[:, 0] = torch.where(dir_body_x >= 0.0, self.vx_abs, -self.vx_abs)
        self.vel_command_b[:, 1] = torch.where(dir_body_y >= 0.0, self.vy_abs, -self.vy_abs)

        target_heading = torch.atan2(dir_world[:, 1], dir_world[:, 0])
        heading_error = math_utils.wrap_to_pi(target_heading - heading)
        yaw_min, yaw_max = self.cfg.yaw_range
        self.vel_command_b[:, 2] = torch.clamp(float(self.cfg.yaw_stiffness) * heading_error, yaw_min, yaw_max)

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class GoalAnchoredVelocityCommandCfg(CommandTermCfg):
    """Config for goal-anchored body-frame velocity commands."""

    class_type: type = GoalAnchoredVelocityCommand
    asset_name: str = MISSING
    resampling_time_range: tuple[float, float] = (100.0, 100.0)
    debug_vis: bool = False
    goal_distance: float = 10.0
    goal_reached_threshold: float = 1.0
    vx_abs_range: tuple[float, float] = (0.6, 1.0)
    vy_abs_range: tuple[float, float] = (0.6, 1.0)
    yaw_stiffness: float = 0.5
    yaw_range: tuple[float, float] = (-0.8, 0.8)
    rel_standing_envs: float = 0.0
