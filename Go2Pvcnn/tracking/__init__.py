"""Parallelism RL tracking task package."""

from __future__ import annotations

try:
    import tracking.register_envs  # noqa: F401
except Exception:
    # IsaacLab/Gym may be unavailable during static imports.
    pass
