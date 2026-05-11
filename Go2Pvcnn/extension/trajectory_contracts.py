"""Backend-agnostic trajectory contracts used by manager/reward/viewer wiring."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from extension.reference.cache import ReferenceTrajectoryCache


class TrajectoryManagerProtocol(Protocol):
    """Minimal contract expected by reward/viewer consumers."""

    planner_backend: str

    def refresh_from_env(self, env) -> ReferenceTrajectoryCache:
        """Refresh internal cache from env state and return ready cache."""

    def current_reference(self) -> dict[str, Tensor]:
        """Return current-frame reference fields."""

    def current_frame_ids(self) -> Tensor:
        """Return current frame index for each environment row."""

    def reset_envs(self, env_mask: Tensor) -> None:
        """Mark reset rows and reset per-env phase counters."""

    def mark_command_changed(self, env_mask: Tensor | None = None, *_, **__) -> None:
        """Mark command-changed rows for asynchronous replanning."""

    def horizon_steps(self) -> int:
        """Return planning horizon in frames."""


class PlannerResultProtocol(Protocol):
    """Core planner output ABI required by cache adapter."""

    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    joint_angles: Tensor
    contact_state: Tensor
    touchdown_seq: Tensor
    planned_touchdown_w: Tensor
    cost_total: Tensor
    status: Tensor
    feasible: Tensor
    safe_fallback: Tensor


def manager_supports_current_reference(manager) -> bool:
    """Return whether manager exposes current-frame gather API."""
    return callable(getattr(manager, "current_reference", None)) and callable(getattr(manager, "current_frame_ids", None))


__all__ = [
    "PlannerResultProtocol",
    "TrajectoryManagerProtocol",
    "manager_supports_current_reference",
]
