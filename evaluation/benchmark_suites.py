"""Suite-specific configuration and goal-command helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import torch
from torch import Tensor

from .manifest import Condition


SUITE_NAMES = ("complex_mixed", "large_runway", "small_runway")


def goal_command(root_pos_w: Tensor, root_yaw: Tensor, goal_xy: Tensor, vx: float, gain: float = 1.5) -> Tensor:
    root = torch.as_tensor(root_pos_w, dtype=torch.float32)
    yaw = torch.as_tensor(root_yaw, dtype=torch.float32, device=root.device).reshape(-1)
    goal = torch.as_tensor(goal_xy, dtype=torch.float32, device=root.device)
    if root.ndim != 2 or root.shape[-1] < 2 or goal.ndim != 2 or goal.shape[-1] != 2 or goal.shape[0] not in (1, root.shape[0]):
        raise ValueError("root_pos_w must be [N,3+] and goal_xy must be [1|N,2]")
    if goal.shape[0] == 1:
        goal = goal.expand(root.shape[0], -1)
    if yaw.numel() != root.shape[0]:
        raise ValueError("root_yaw must have one value per environment")
    delta = goal - root[:, :2]
    target = torch.atan2(delta[:, 1], delta[:, 0])
    error = torch.atan2(torch.sin(target - yaw), torch.cos(target - yaw))
    command = torch.zeros(root.shape[0], 3, dtype=root.dtype, device=root.device)
    command[:, 0] = float(vx)
    command[:, 2] = torch.clamp(float(gain) * error, -1.0, 1.0)
    return command


def runway_large_points(count: int) -> tuple[tuple[float, float], ...]:
    """Place large obstacles in the central 8x2 m runway region."""

    number = int(count)
    if number not in (4, 6, 8, 10):
        raise ValueError("large runway obstacle count must be one of 4, 6, 8, 10")
    columns = max(1, math.ceil(number / 2))
    points = []
    for index in range(number):
        column = index % columns
        row = index // columns
        x = -4.0 + 8.0 * (column + 0.5) / columns
        y = -0.45 if row == 0 else 0.45
        points.append((float(x), float(y)))
    return tuple(points)


def runway_small_points(count: int) -> tuple[tuple[float, float], ...]:
    """Place small obstacles throughout the central 8x2 m runway region."""

    number = int(count)
    if number not in {40, 80, 120, 160}:
        raise ValueError("small runway obstacle count must be one of 40, 80, 120, 160")
    columns = int(math.ceil(math.sqrt(number * 4.0)))
    rows = int(math.ceil(number / columns))
    points = []
    for index in range(number):
        column = index % columns
        row = index // columns
        x = -4.0 + 8.0 * (column + 0.5) / columns
        y = -0.85 + 1.7 * (row + 0.5) / rows
        points.append((float(x), float(y)))
    return tuple(points)


def build_suite_config(suite: str, base_cfg, condition: Condition):
    """Copy a PLAY config and apply one frozen benchmark condition."""

    if str(suite) not in SUITE_NAMES or condition.suite != str(suite):
        raise ValueError("suite and condition.suite must match a supported suite")
    cfg = deepcopy(base_cfg)
    cfg.benchmark_suite = str(suite)
    cfg.benchmark_condition_id = condition.condition_id
    cfg.benchmark_goal_xy = tuple(condition.goal_xy)
    cfg.benchmark_layout_id = int(condition.layout_id)
    cfg.benchmark_seed = int(condition.seed)
    cfg.benchmark_large_obstacle_count = int(condition.large_obstacle_count)
    cfg.benchmark_small_obstacle_count = int(condition.small_obstacle_count)
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.scene.terrain.terrain_generator.curriculum = False
    if hasattr(cfg, "curriculum"):
        for name in ("terrain_levels", "lin_vel_cmd_levels", "parallelism_velocity"):
            if hasattr(cfg.curriculum, name):
                setattr(cfg.curriculum, name, None)

    from extension.semantic_course import SemanticCourseLayoutCfg

    current_layout = getattr(cfg.scene.terrain, "semantic_course_layout_cfg", SemanticCourseLayoutCfg())
    large_points = runway_large_points(condition.large_obstacle_count) if suite == "large_runway" else None
    small_points = runway_small_points(condition.small_obstacle_count) if suite == "small_runway" else None
    cfg.scene.terrain.semantic_course_layout_cfg = replace(
        current_layout,
        fixed_large_obstacle_local_xy=large_points,
        fixed_small_obstacle_local_xy=small_points,
    )
    if suite in {"large_runway", "small_runway"}:
        from extension.semantic_curriculum import SemanticObstacleCount
        import isaaclab.terrains as terrain_gen

        counts = SemanticObstacleCount(
            small=int(condition.small_obstacle_count),
            large=int(condition.large_obstacle_count),
        )
        cfg.scene.terrain.semantic_obstacle_curriculum.plane_terrain_names = ("flat",)
        cfg.scene.terrain.semantic_obstacle_curriculum.plane_counts = (counts,)
        cfg.scene.terrain.semantic_obstacle_curriculum.non_plane_counts = (counts,)
        cfg.scene.terrain.semantic_obstacle_curriculum.terrain_obstacle_count_overrides = {"flat": counts}
        cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(10.0, 2.0),
            border_width=20.0,
            num_rows=1,
            num_cols=max(1, int(condition.terrain_col) + 1),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 0.0),
            curriculum=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
        )
    return cfg
