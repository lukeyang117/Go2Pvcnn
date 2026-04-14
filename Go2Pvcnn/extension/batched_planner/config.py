"""Batched trajectory config mirroring the raw planner defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatchedTrajectoryConfig:
    gait_name: str = "trot"
    step_freq: float = 2.0
    duty_factor: float = 0.6
    step_height: float = 0.08
    hip_height: float = 0.30
    body_clearance_margin: float = 0.012
    foothold_search_radius: float = 0.15
    foothold_search_step: float = 0.03
    max_foothold_step_down: float = 0.10
    max_roughness: float = 0.5
    max_touchdown_xy_reach: float = 0.15
    replan_stop_speed: float = 0.05


__all__ = ["BatchedTrajectoryConfig"]
