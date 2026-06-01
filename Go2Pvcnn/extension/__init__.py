"""Extension modules for trajectory-guided teacher experiments."""

from .trajectory_manager_factory import attach_trajectory_manager, create_trajectory_manager

__all__ = [
    "attach_trajectory_manager",
    "create_trajectory_manager",
]
