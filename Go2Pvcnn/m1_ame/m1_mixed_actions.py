"""Isaac Lab action term for M1 leg position and wheel velocity control."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from collections.abc import Sequence

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from m1_ame.m1_mixed_action_math import split_m1_mixed_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class M1MixedJointAction(ActionTerm):
    """Keep a canonical 16-D action while sending legs and wheels different targets."""

    cfg: "M1MixedJointActionCfg"

    def __init__(self, cfg: "M1MixedJointActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._asset = self._env.scene[cfg.asset_name]
        self._joint_ids, resolved_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._joint_names = tuple(resolved_names)
        if len(self._joint_names) != len(cfg.joint_names) or len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError(
                "M1 mixed action requires every configured joint name to resolve exactly once: "
                f"configured={cfg.joint_names}, resolved={self._joint_names}"
            )

        name_to_action_index = {name: index for index, name in enumerate(self._joint_names)}
        if set(cfg.leg_joint_names) | set(cfg.wheel_joint_names) != set(self._joint_names):
            raise ValueError("M1 leg and wheel action selectors must cover exactly the full action joint list")
        if set(cfg.leg_joint_names) & set(cfg.wheel_joint_names):
            raise ValueError("M1 leg and wheel action selectors must be disjoint")

        self._leg_action_indices = tuple(name_to_action_index[name] for name in cfg.leg_joint_names)
        self._wheel_action_indices = tuple(name_to_action_index[name] for name in cfg.wheel_joint_names)
        self._leg_joint_ids, leg_names = self._asset.find_joints(cfg.leg_joint_names, preserve_order=True)
        self._wheel_joint_ids, wheel_names = self._asset.find_joints(cfg.wheel_joint_names, preserve_order=True)
        if tuple(leg_names) != tuple(cfg.leg_joint_names) or tuple(wheel_names) != tuple(cfg.wheel_joint_names):
            raise ValueError("M1 mixed action joint resolution did not preserve configured leg/wheel order")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._position_targets = self._asset.data.default_joint_pos[:, self._leg_joint_ids].clone()
        self._velocity_targets = torch.zeros(self.num_envs, len(self._wheel_joint_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return len(self._joint_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Invalid M1 mixed action shape: expected {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}")
        self._raw_actions[:] = torch.clamp(
            actions.to(self.device),
            min=-abs(float(self.cfg.normalized_action_limit)),
            max=abs(float(self.cfg.normalized_action_limit)),
        )
        default_leg_pos = self._asset.data.default_joint_pos[:, self._leg_joint_ids]
        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._leg_joint_ids]
        self._position_targets, self._velocity_targets = split_m1_mixed_action(
            self._raw_actions,
            self._leg_action_indices,
            self._wheel_action_indices,
            self.cfg.leg_scale,
            self.cfg.wheel_scale,
            default_leg_pos,
            self.cfg.wheel_velocity_limit,
            soft_limits[..., 0],
            soft_limits[..., 1],
            self.cfg.normalized_action_limit,
            self._position_targets,
            self.cfg.leg_target_delta_limit,
        )
        self._processed_actions.zero_()
        self._processed_actions[:, self._leg_action_indices] = self._position_targets
        self._processed_actions[:, self._wheel_action_indices] = self._velocity_targets

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._position_targets, joint_ids=self._leg_joint_ids)
        self._asset.set_joint_velocity_target(self._velocity_targets, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        default_leg_pos = self._asset.data.default_joint_pos[:, self._leg_joint_ids]
        self._position_targets[env_ids] = default_leg_pos[env_ids]
        self._velocity_targets[env_ids] = 0.0


@configclass
class M1MixedJointActionCfg(ActionTermCfg):
    """Configuration for the canonical M1 12-leg-position/4-wheel-velocity action."""

    class_type: type[ActionTerm] = M1MixedJointAction
    joint_names: list[str] = MISSING
    leg_joint_names: list[str] = MISSING
    wheel_joint_names: list[str] = MISSING
    leg_scale: float = 0.25
    wheel_scale: float = 8.0
    wheel_velocity_limit: float = 8.0
    leg_position_lower: tuple[float, ...] | None = None
    leg_position_upper: tuple[float, ...] | None = None
    normalized_action_limit: float = 1.0
    leg_target_delta_limit: float = 0.15


__all__ = ["M1MixedJointAction", "M1MixedJointActionCfg", "split_m1_mixed_action"]
