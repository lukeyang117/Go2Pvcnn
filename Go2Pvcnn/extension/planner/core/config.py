from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PlannerTerrainConfig:
    heightmap_path: Path
    terrain_name: str = "sample_heightmap"
    elevation_path: Path | None = None
    local_window_extent_xy: tuple[float, float] = (1.5, 1.5)
    local_window_resolution: float = 0.01
    scene_variant: str = "scene.xml"
    world_x_range: tuple[float, float] = (-1.5, 4.5)
    world_y_range: tuple[float, float] = (-2.0, 2.0)
    height_range: tuple[float, float] = (0.0, 0.45)

    def resolved_elevation_path(self) -> Path:
        if self.elevation_path is not None:
            return self.elevation_path
        project_root = Path(__file__).resolve().parents[3]
        return (
            project_root
            / "scripts"
            / "terrain"
            / self.terrain_name
            / f"{self.terrain_name}.npz"
        )


@dataclass(frozen=True)
class TrajectoryConfig:
    gait_name: str = "trot"
    step_freq: float = 2.0
    duty_factor: float = 0.55
    step_height: float = 0.08
    hip_height: float = 0.30
    body_clearance_margin: float = 0.012
    max_base_roll: float = 0.35
    max_base_pitch: float = 0.45
    foothold_search_radius: float = 0.15
    foothold_search_step: float = 0.03
    max_foothold_step_down: float = 0.10
    replan_velocity_scales: tuple[float, ...] = (1.0, 0.7, 0.4, 0.2, 0.0)
    replan_yaw_biases: tuple[float, ...] = (0.0, 0.20, -0.20)
    replan_vy_biases: tuple[float, ...] = (0.0, 0.08, -0.08)
    max_touchdown_xy_reach: float = 0.22
    replan_stop_speed: float = 0.03


@dataclass(frozen=True)
class PlannerConfig:
    gait_name: str = "trot"
    overlay_ttl_sec: float = 1.5
    terrain: PlannerTerrainConfig = field(
        default_factory=lambda: PlannerTerrainConfig(
            heightmap_path=Path(__file__).resolve().parents[3]
            / "assets"
            / "terrain"
            / "sample_heightmap.png"
        )
    )
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)


__all__ = ["PlannerConfig", "PlannerTerrainConfig", "TrajectoryConfig"]
