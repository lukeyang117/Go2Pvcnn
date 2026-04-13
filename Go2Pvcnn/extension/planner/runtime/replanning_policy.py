"""Replanning trigger scaffolding for planner-guided training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplanningPolicy:
    """Simple trigger thresholds for future reference regeneration."""

    max_command_delta: float = 0.25
    max_tracking_error: float = 0.5
    replan_on_reset: bool = True

    def should_replan(self, *, command_delta: float, tracking_error: float, reset: bool = False) -> bool:
        """Return whether a new reference should be generated."""
        return bool(
            (reset and self.replan_on_reset)
            or command_delta > self.max_command_delta
            or tracking_error > self.max_tracking_error
        )
