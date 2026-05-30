"""Rewards based on IsaacLab filtered contact sensors for semantic objects."""

from __future__ import annotations

import torch
from torch import Tensor


def filtered_contact_penalty_from_force_matrix(
    force_matrix_w: Tensor,
    *,
    force_threshold: float,
    force_scale: float,
    force_clip: float,
) -> Tensor:
    """Aggregate one filtered contact sensor matrix into a per-env penalty."""

    force = torch.as_tensor(force_matrix_w, dtype=torch.float32)
    if force.ndim != 4 or int(force.shape[-1]) != 3:
        raise ValueError(f"force_matrix_w must have shape [N,B,F,3], got {tuple(force.shape)}")
    per_filter = torch.linalg.vector_norm(force, dim=-1)
    total_excess = torch.relu(per_filter - float(force_threshold)).sum(dim=(1, 2))
    scaled = total_excess / max(float(force_scale), 1.0e-6)
    return scaled.clamp(0.0, float(force_clip))


def _scene_sensor(env, name: str):
    sensors = getattr(env.scene, "sensors", None)
    if sensors is not None:
        try:
            return sensors[name]
        except Exception:  # noqa: BLE001 - Isaac scene containers are duck-typed.
            return getattr(sensors, name)
    return env.scene[name]


def global_semantic_contact_penalty_from_matrices(
    small_force_matrix_w: Tensor,
    large_force_matrix_w: Tensor,
    *,
    body_weights: tuple[float, ...],
    force_threshold: float,
    force_scale: float,
    force_clip: float,
    small_weight: float,
    large_weight: float,
) -> Tensor:
    """Aggregate two global semantic contact matrices into a per-env penalty."""

    small = torch.as_tensor(small_force_matrix_w, dtype=torch.float32)
    large = torch.as_tensor(large_force_matrix_w, dtype=torch.float32, device=small.device)
    if small.ndim != 4 or large.ndim != 4 or int(small.shape[-1]) != 3 or int(large.shape[-1]) != 3:
        raise ValueError("force matrices must have shape [N,B,O,3]")
    if int(small.shape[0]) != int(large.shape[0]) or int(small.shape[1]) != int(large.shape[1]):
        raise ValueError("small and large matrices must share [N,B]")

    weights = torch.as_tensor(body_weights, dtype=torch.float32, device=small.device)
    if int(weights.numel()) != int(small.shape[1]):
        raise ValueError("body_weights length must match body dimension")

    small_excess = torch.relu(torch.linalg.vector_norm(small, dim=-1) - float(force_threshold)).sum(dim=-1)
    large_excess = torch.relu(torch.linalg.vector_norm(large, dim=-1) - float(force_threshold)).sum(dim=-1)
    total = (weights[None, :] * (float(small_weight) * small_excess + float(large_weight) * large_excess)).sum(dim=-1)
    return (total / max(float(force_scale), 1.0e-6)).clamp(0.0, float(force_clip))


def semantic_global_contact_collision_reward(
    env,
    small_sensor_cfg,
    large_sensor_cfg,
    body_names: tuple[str, ...],
    body_weights: tuple[float, ...],
    force_threshold: float = 1.0,
    force_scale: float = 50.0,
    force_clip: float = 1.0,
    small_weight: float = 1.0,
    large_weight: float = 2.0,
) -> Tensor:
    """Return negative semantic collision penalty from two global contact sensors."""

    device = torch.device(getattr(env, "device", "cpu"))
    small_sensor = _scene_sensor(env, small_sensor_cfg.name)
    large_sensor = _scene_sensor(env, large_sensor_cfg.name)
    if tuple(getattr(small_sensor, "body_names", ())) != tuple(body_names):
        raise ValueError("small semantic contact sensor body_names do not match reward body_names")
    if tuple(getattr(large_sensor, "body_names", ())) != tuple(body_names):
        raise ValueError("large semantic contact sensor body_names do not match reward body_names")

    penalty = global_semantic_contact_penalty_from_matrices(
        torch.as_tensor(small_sensor.data.force_matrix_w, dtype=torch.float32, device=device),
        torch.as_tensor(large_sensor.data.force_matrix_w, dtype=torch.float32, device=device),
        body_weights=body_weights,
        force_threshold=force_threshold,
        force_scale=force_scale,
        force_clip=force_clip,
        small_weight=small_weight,
        large_weight=large_weight,
    )
    return -penalty.to(device=device)


def semantic_filtered_contact_collision_reward(
    env,
    small_sensor_names: tuple[str, ...],
    large_sensor_names: tuple[str, ...],
    body_weights: tuple[float, ...],
    force_threshold: float = 1.0,
    force_scale: float = 50.0,
    force_clip: float = 1.0,
    small_weight: float = 1.0,
    large_weight: float = 2.0,
) -> Tensor:
    """Return negative contact penalty for semantic small/large object contacts."""

    device = torch.device(getattr(env, "device", "cpu"))
    out = torch.zeros(int(env.num_envs), dtype=torch.float32, device=device)
    weights = torch.as_tensor(body_weights, dtype=torch.float32, device=device)
    if len(small_sensor_names) != int(weights.numel()) or len(large_sensor_names) != int(weights.numel()):
        raise ValueError("sensor name counts must match body_weights")

    for idx, name in enumerate(small_sensor_names):
        sensor = _scene_sensor(env, name)
        matrix = torch.as_tensor(sensor.data.force_matrix_w, dtype=torch.float32, device=device)
        out = out + weights[idx] * float(small_weight) * filtered_contact_penalty_from_force_matrix(
            matrix,
            force_threshold=force_threshold,
            force_scale=force_scale,
            force_clip=force_clip,
        ).to(device=device)
    for idx, name in enumerate(large_sensor_names):
        sensor = _scene_sensor(env, name)
        matrix = torch.as_tensor(sensor.data.force_matrix_w, dtype=torch.float32, device=device)
        out = out + weights[idx] * float(large_weight) * filtered_contact_penalty_from_force_matrix(
            matrix,
            force_threshold=force_threshold,
            force_scale=force_scale,
            force_clip=force_clip,
        ).to(device=device)
    return -out


__all__ = [
    "filtered_contact_penalty_from_force_matrix",
    "global_semantic_contact_penalty_from_matrices",
    "semantic_filtered_contact_collision_reward",
    "semantic_global_contact_collision_reward",
]
