"""Extension modules for trajectory-guided teacher experiments."""

from .batched_planner.manager import BatchedTrajectoryManager
from .batched_together_planner.manager import TogetherTrajectoryManager
from .trajectory_manager_factory import attach_trajectory_manager, create_trajectory_manager

__all__ = [
    "BatchedTrajectoryManager",
    "TogetherTrajectoryManager",
    "attach_trajectory_manager",
    "create_trajectory_manager",
]
