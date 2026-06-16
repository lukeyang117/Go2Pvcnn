"""Velocity command curriculum (aligned with unitree_rl_lab)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from extension.semantic_curriculum import (
    SemanticObstacleCurriculumState,
    update_episode_small_collision_from_forces,
    update_episode_small_collision_from_map_contacts,
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
    Works with UniformLevelVelocityCommandCfg and GoalAnchoredVelocityCommandCfg.
    Aligned with unitree_rl_lab velocity_env_cfg.
    """
    command_term = env.command_manager.get_term("base_velocity")
    if not hasattr(command_term.cfg, "ranges") or not hasattr(command_term.cfg, "limit_ranges"):
        return torch.tensor(0.0, device=env.device)
    if command_term.cfg.ranges is None or command_term.cfg.limit_ranges is None:
        return torch.tensor(0.0, device=env.device)

    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    interval = max(int(env.max_episode_length // 50), 1)
    if env.common_step_counter % interval == 0:
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


def small_semantic_collision_mask_from_force_matrix(
    small_force_matrix_w: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return per-env small-obstacle collision mask from the real small contact matrix."""
    small = torch.as_tensor(small_force_matrix_w, dtype=torch.float32)
    if small.ndim != 4 or int(small.shape[-1]) != 3:
        raise ValueError("small force matrix must have shape [N, B, O, 3]")
    return torch.linalg.vector_norm(small, dim=-1).gt(float(threshold)).any(dim=(1, 2))


def plane_env_mask_from_terrain(
    terrain_types: torch.Tensor,
    terrain_names: tuple[str, ...] | list[str],
    plane_terrain_names: tuple[str, ...],
) -> torch.Tensor:
    """Return env mask whose terrain column name is in ``plane_terrain_names``."""
    types = torch.as_tensor(terrain_types, dtype=torch.long)
    wanted = {str(name) for name in plane_terrain_names}
    if len(terrain_names) == 1 and str(terrain_names[0]) in wanted:
        return torch.ones_like(types, dtype=torch.bool)
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


def _flat_episode_curriculum_info(
    env: ManagerBasedRLEnv,
    cfg_name: str = "semantic_obstacle_curriculum",
    completed_env_ids: torch.Tensor | None = None,
) -> tuple[object | None, dict[str, torch.Tensor | bool]]:
    device = torch.device(getattr(env, "device", "cpu"))
    cfg = getattr(env.cfg, cfg_name, None)
    root = getattr(env, "unwrapped", env)
    state = getattr(root, "_semantic_obstacle_curriculum_state", None)
    if state is None:
        state = SemanticObstacleCurriculumState()
        root._semantic_obstacle_curriculum_state = state

    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return cfg, {
            "plane_mask": torch.zeros(int(env.num_envs), dtype=torch.bool, device=device),
            "episode_success": torch.zeros(int(env.num_envs), dtype=torch.bool, device=device),
            "base_contact": _env_bool_buffer(env, "base_contact", device=device),
            "bad_orientation": _env_bool_buffer(env, "bad_orientation", device=device),
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

    if bool(plane_mask.any().item()):
        try:
            from extension.mdp.semantic_body_part_clearance import infer_current_small_semantic_contact

            small_hit = infer_current_small_semantic_contact(
                env,
                asset_cfg=SceneEntityCfg("robot"),
                scanner_cfg=SceneEntityCfg("semantic_height_scanner"),
                contact_sensor_cfg=SceneEntityCfg("contact_forces"),
                small_semantic_ids=(1,),
                force_threshold=float(cfg.collision_force_threshold),
            )
            update_episode_small_collision_from_map_contacts(state, small_hit.to(device=device))
        except Exception:  # noqa: BLE001 - unit tests may use force-matrix-only fake scenes.
            try:
                small_sensor = _scene_sensor(env, "semantic_contact_small")
            except Exception:
                raise
            else:
                small_force = torch.as_tensor(small_sensor.data.force_matrix_w, dtype=torch.float32, device=device)
                update_episode_small_collision_from_forces(
                    state,
                    small_force,
                    float(cfg.collision_force_threshold),
                )
    else:
        update_episode_small_collision_from_map_contacts(
            state,
            torch.zeros((int(env.num_envs),), dtype=torch.bool, device=device),
        )

    reset_env_ids = (
        torch.as_tensor(completed_env_ids, dtype=torch.long, device=device).reshape(-1)
        if completed_env_ids is not None
        else _completed_episode_env_ids(env, device=device)
    )
    flags = state.episode_had_small_collision
    if flags is None or int(flags.numel()) != int(env.num_envs) or flags.device != device:
        flags = torch.zeros(int(env.num_envs), dtype=torch.bool, device=device)
        state.episode_had_small_collision = flags
    time_out = _env_bool_buffer(env, "time_out", device=device)
    base_contact = _env_bool_buffer(env, "base_contact", device=device)
    bad_orientation = _env_bool_buffer(env, "bad_orientation", device=device)
    episode_success = torch.zeros(int(env.num_envs), dtype=torch.bool, device=device)
    if reset_env_ids.numel() > 0:
        episode_success[reset_env_ids] = (
            plane_mask.index_select(0, reset_env_ids)
            & time_out.index_select(0, reset_env_ids)
            & torch.logical_not(flags.index_select(0, reset_env_ids))
            & torch.logical_not(base_contact.index_select(0, reset_env_ids))
            & torch.logical_not(bad_orientation.index_select(0, reset_env_ids))
        )
        flags[reset_env_ids] = False

    return cfg, {
        "plane_mask": plane_mask,
        "episode_success": episode_success,
        "base_contact": base_contact,
        "bad_orientation": bad_orientation,
        "enabled": bool(cfg.enabled),
    }


def _completed_episode_env_ids(env, *, device: torch.device) -> torch.Tensor:
    for name in ("reset_env_ids", "_reset_env_ids", "done_env_ids"):
        value = getattr(env, name, None)
        if value is not None:
            return torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
    value = getattr(env, "reset_buf", None)
    if value is not None:
        return torch.nonzero(torch.as_tensor(value, dtype=torch.bool, device=device), as_tuple=False).flatten()
    value = getattr(env, "terminated_buf", None)
    truncated = getattr(env, "time_out_buf", None)
    base_contact = getattr(env, "base_contact_buf", None)
    bad_orientation = getattr(env, "bad_orientation_buf", None)
    if value is not None or truncated is not None or base_contact is not None or bad_orientation is not None:
        done = torch.zeros(int(env.num_envs), dtype=torch.bool, device=device)
        if value is not None:
            done |= torch.as_tensor(value, dtype=torch.bool, device=device)
        if truncated is not None:
            done |= torch.as_tensor(truncated, dtype=torch.bool, device=device)
        if base_contact is not None:
            done |= torch.as_tensor(base_contact, dtype=torch.bool, device=device)
        if bad_orientation is not None:
            done |= torch.as_tensor(bad_orientation, dtype=torch.bool, device=device)
        return torch.nonzero(done, as_tuple=False).flatten()
    return torch.empty(0, dtype=torch.long, device=device)


def _env_bool_buffer(env, kind: str, *, device: torch.device) -> torch.Tensor:
    names_by_kind = {
        "time_out": ("time_out_buf", "time_outs", "truncated_buf"),
        "base_contact": ("base_contact_buf",),
        "bad_orientation": ("bad_orientation_buf",),
    }
    for name in names_by_kind[kind]:
        value = getattr(env, name, None)
        if value is not None:
            return torch.as_tensor(value, dtype=torch.bool, device=device).reshape(-1)
    termination_manager = getattr(env, "termination_manager", None)
    if termination_manager is not None:
        terminations = getattr(termination_manager, "_term_dones", None)
        if isinstance(terminations, dict):
            if kind == "time_out" and "time_out" in terminations:
                return torch.as_tensor(terminations["time_out"], dtype=torch.bool, device=device).reshape(-1)
            if kind in terminations:
                return torch.as_tensor(terminations[kind], dtype=torch.bool, device=device).reshape(-1)
    default = kind == "time_out"
    return torch.full((int(env.num_envs),), default, dtype=torch.bool, device=device)


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

    cfg, info = _flat_episode_curriculum_info(env, cfg_name=cfg_name, completed_env_ids=env_ids_t)
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

    episode_success_all = torch.as_tensor(info["episode_success"], dtype=torch.bool, device=device)
    base_contact_all = torch.as_tensor(info["base_contact"], dtype=torch.bool, device=device)
    bad_orientation_all = torch.as_tensor(info["bad_orientation"], dtype=torch.bool, device=device)
    flat_episode_success = episode_success_all.index_select(0, env_ids_t)
    base_contact = base_contact_all.index_select(0, env_ids_t)
    bad_orientation = bad_orientation_all.index_select(0, env_ids_t)

    flat_move_up = torch.logical_and(terrain_move_up, flat_episode_success)
    move_up = torch.where(is_plane_env, flat_move_up, terrain_move_up)
    flat_failure_move_down = torch.logical_or(base_contact, bad_orientation)
    move_down = torch.where(is_plane_env, torch.logical_or(terrain_move_down, flat_failure_move_down), terrain_move_down)

    terrain.update_env_origins(env_ids_t, move_up, move_down)

    return {
        "mean_terrain_level": torch.mean(terrain.terrain_levels.float()),
    }
