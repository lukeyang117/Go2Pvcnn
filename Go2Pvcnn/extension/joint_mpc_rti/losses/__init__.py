"""Continuous residual and relaxed-barrier terms for joint MPC RTI."""

from .barriers import relaxed_barrier
from .command import command_losses
from .clearance import clearance_losses
from .contact import stance_losses, swing_losses, touchdown_geometry_losses, touchdown_losses
from .objective import terminal_losses, weighted_objective
from .posture import posture_losses
from .semantic import large_obstacle_losses, small_object_losses
from .smoothness import smoothness_losses

__all__ = [
    "command_losses",
    "clearance_losses",
    "large_obstacle_losses",
    "relaxed_barrier",
    "posture_losses",
    "small_object_losses",
    "smoothness_losses",
    "stance_losses",
    "swing_losses",
    "terminal_losses",
    "touchdown_geometry_losses",
    "touchdown_losses",
    "weighted_objective",
]
from .rollout_objective import rollout_loss_breakdown

__all__ = ["rollout_loss_breakdown"]
