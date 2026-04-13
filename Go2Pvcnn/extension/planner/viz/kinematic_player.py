"""Pure kinematic Isaac Lab playback scaffolding for the extension planner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KinematicPlayerConfig:
    """Configuration scaffold for future Isaac Lab kinematic playback."""

    terrain_name: str = "default"
    n_frames: int = 50
    dt: float = 0.02


class KinematicTrajectoryPlayer:
    """Minimal scaffold for a future pure-kinematic playback tool."""

    def __init__(self, config: KinematicPlayerConfig | None = None):
        self.config = config or KinematicPlayerConfig()

    def describe(self) -> str:
        """Return a short human-readable summary of the current scaffold config."""
        return (
            f"KinematicTrajectoryPlayer(terrain_name={self.config.terrain_name!r}, "
            f"n_frames={self.config.n_frames}, dt={self.config.dt})"
        )
