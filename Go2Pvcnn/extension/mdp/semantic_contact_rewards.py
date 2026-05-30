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
    "semantic_filtered_contact_collision_reward",
]
