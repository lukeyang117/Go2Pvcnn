"""Velocity command curriculum (aligned with unitree_rl_lab)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from extension.semantic_curriculum import (
    SemanticObstacleCurriculumState,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel_unitree_rl_lab(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terrain curriculum based on velocity tracking distance.

    Increases terrain difficulty when robot walks far enough; decreases when
    robot walks less than half of commanded distance.
    Implemented from unitree_rl_lab / isaaclab_tasks.velocity.mdp.

    Returns:
        Mean terrain level for the given env_ids.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Expand lin_vel command ranges when velocity tracking reward is high enough.

    When mean reward > weight * 0.8, expand ranges toward limit_ranges.
    Requires base_velocity to use UniformLevelVelocityCommandCfg with limit_ranges.
    Aligned with unitree_rl_lab velocity_env_cfg.
    """
    command_term = env.command_manager.get_term("base_velocity")
    if not hasattr(command_term.cfg, "limit_ranges"):
        return torch.tensor(0.0, device=env.device)

    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def semantic_collision_mask_from_force_matrices(
    small_force_matrix_w: torch.Tensor,
    large_force_matrix_w: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return per-env semantic collision mask from small/large force matrices."""
    small = torch.as_tensor(small_force_matrix_w, dtype=torch.float32)
    large = torch.as_tensor(large_force_matrix_w, dtype=torch.float32, device=small.device)
    if small.ndim != 4 or large.ndim != 4 or int(small.shape[-1]) != 3 or int(large.shape[-1]) != 3:
        raise ValueError("small/large force matrices must have shape [N, B, O, 3]")
    if int(small.shape[0]) != int(large.shape[0]):
        raise ValueError("small and large force matrices must share env dimension")
    small_hit = torch.linalg.vector_norm(small, dim=-1) > float(threshold)
    large_hit = torch.linalg.vector_norm(large, dim=-1) > float(threshold)
    return torch.logical_or(small_hit.any(dim=(1, 2)), large_hit.any(dim=(1, 2)))


def plane_env_mask_from_terrain(
    terrain_types: torch.Tensor,
    terrain_names: tuple[str, ...] | list[str],
    plane_terrain_names: tuple[str, ...],
) -> torch.Tensor:
    """Return env mask whose terrain column name is in ``plane_terrain_names``."""
    types = torch.as_tensor(terrain_types, dtype=torch.long)
    wanted = {str(name) for name in plane_terrain_names}
    out = torch.zeros_like(types, dtype=torch.bool)
    for col, name in enumerate(terrain_names):
        if str(name) in wanted:
            out = torch.logical_or(out, types == int(col))
    return out


def _scene_sensor(env, name: str):
    sensors = getattr(env.scene, "sensors", None)
    if sensors is not None:
        try:
            return sensors[name]
        except Exception:  # noqa: BLE001 - Isaac containers are duck-typed.
            if hasattr(sensors, name):
                return getattr(sensors, name)
    return env.scene[name]


def _terrain_names_from_env(env) -> tuple[str, ...]:
    terrain = env.scene.terrain
    terrain_generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    sub_terrains = getattr(terrain_generator, "sub_terrains", None)
    if isinstance(sub_terrains, dict):
        return tuple(str(name) for name in sub_terrains.keys())
    return ()


def _flat_semantic_gate_info(
    env: ManagerBasedRLEnv,
    cfg_name: str = "semantic_obstacle_curriculum",
) -> tuple[object | None, dict[str, float | int | bool]]:
    device = torch.device(getattr(env, "device", "cpu"))
    cfg = getattr(env.cfg, cfg_name, None)
    root = getattr(env, "unwrapped", env)
    state = getattr(root, "_semantic_obstacle_curriculum_state", None)
    if state is None:
        state = SemanticObstacleCurriculumState()
        root._semantic_obstacle_curriculum_state = state

    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return cfg, {
            "consecutive_success_count": int(state.consecutive_success_count),
            "plane_collision_rate": 0.0,
            "plane_env_count": 0,
            "gate_pass": False,
            "enabled": False,
        }

    terrain = env.scene.terrain
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_names = _terrain_names_from_env(env)
    if terrain_types is None or len(terrain_names) == 0:
        plane_mask = torch.zeros(int(env.num_envs), dtype=torch.bool, device=device)
    else:
        plane_mask = plane_env_mask_from_terrain(
            torch.as_tensor(terrain_types, dtype=torch.long, device=device),
            terrain_names,
            tuple(getattr(cfg, "plane_terrain_names", ("flat",))),
        ).to(device=device)

    plane_env_count = int(plane_mask.sum().item())
    if plane_env_count == 0:
        rate = torch.tensor(0.0, dtype=torch.float32, device=device)
    else:
        small_sensor = _scene_sensor(env, "semantic_contact_small")
        large_sensor = _scene_sensor(env, "semantic_contact_large")
        collision = semantic_collision_mask_from_force_matrices(
            torch.as_tensor(small_sensor.data.force_matrix_w, dtype=torch.float32, device=device),
            torch.as_tensor(large_sensor.data.force_matrix_w, dtype=torch.float32, device=device),
            float(cfg.collision_force_threshold),
        )
        rate = (collision & plane_mask).to(dtype=torch.float32).sum() / float(plane_env_count)

    return cfg, state.update_gate_from_plane_collision_rate(rate, cfg, plane_env_count=plane_env_count)


def terrain_levels_vel_semantic_plane_gate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cfg_name: str = "semantic_obstacle_curriculum",
) -> dict[str, torch.Tensor]:
    """Terrain curriculum with semantic collision gate applied only to flat env move-up."""
    device = torch.device(getattr(env, "device", "cpu"))
    if isinstance(env_ids, slice):
        env_ids_t = torch.arange(int(env.num_envs), dtype=torch.long, device=device)[env_ids]
    else:
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids_t, :2] - env.scene.env_origins[env_ids_t, :2], dim=1)
    terrain_move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    terrain_move_down = distance < torch.norm(command[env_ids_t, :2], dim=1) * env.max_episode_length_s * 0.5
    terrain_move_down = torch.logical_and(terrain_move_down, torch.logical_not(terrain_move_up))

    cfg, info = _flat_semantic_gate_info(env, cfg_name=cfg_name)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_names = _terrain_names_from_env(env)
    if cfg is None or terrain_types is None or len(terrain_names) == 0:
        is_plane_env = torch.zeros_like(terrain_move_up, dtype=torch.bool, device=device)
    else:
        is_plane_all = plane_env_mask_from_terrain(
            torch.as_tensor(terrain_types, dtype=torch.long, device=device),
            terrain_names,
            tuple(getattr(cfg, "plane_terrain_names", ("flat",))),
        ).to(device=device)
        is_plane_env = is_plane_all[env_ids_t]

    semantic_gate_pass = bool(info["gate_pass"])
    semantic_gate_tensor = torch.full_like(terrain_move_up, semantic_gate_pass, dtype=torch.bool, device=device)
    move_up = torch.where(is_plane_env, torch.logical_and(terrain_move_up, semantic_gate_tensor), terrain_move_up)

    terrain.update_env_origins(env_ids_t, move_up, terrain_move_down)
    flat_move_up_count = torch.logical_and(is_plane_env, move_up).sum()
    non_flat_move_up_count = torch.logical_and(torch.logical_not(is_plane_env), move_up).sum()

    return {
        "mean_terrain_level": torch.mean(terrain.terrain_levels.float()),
        "plane_collision_rate": torch.tensor(float(info["plane_collision_rate"]), device=device),
        "plane_env_count": torch.tensor(float(info["plane_env_count"]), device=device),
        "consecutive_success_count": torch.tensor(float(info["consecutive_success_count"]), device=device),
        "semantic_gate_pass": torch.tensor(1.0 if semantic_gate_pass else 0.0, device=device),
        "flat_move_up_count": flat_move_up_count.to(dtype=torch.float32),
        "non_flat_move_up_count": non_flat_move_up_count.to(dtype=torch.float32),
        "enabled": torch.tensor(1.0 if bool(info["enabled"]) else 0.0, device=device),
    }
