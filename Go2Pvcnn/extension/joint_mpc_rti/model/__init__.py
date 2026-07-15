"""Kinematic model and fixed gait schedule for joint MPC RTI."""

from .dynamics import kinematic_step
from .gait_schedule import fixed_trot_schedule

__all__ = ["fixed_trot_schedule", "kinematic_step"]
