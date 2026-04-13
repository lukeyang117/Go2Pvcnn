"""Observation helpers for trajectory-guided teacher experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def downsample_height_map(height_map: torch.Tensor, target_size: int) -> torch.Tensor:
    """Downsample a batch of height maps with area pooling.

    Args:
        height_map: Tensor shaped ``(batch, H, W)``.
        target_size: The square output size.

    Returns:
        Tensor shaped ``(batch, target_size, target_size)``.
    """
    if height_map.ndim != 3:
        raise ValueError(f"Expected (batch, H, W), got shape {tuple(height_map.shape)}")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    return F.adaptive_avg_pool2d(height_map.unsqueeze(1), (target_size, target_size)).squeeze(1)


def downsampled_height_scan(env, sensor_cfg, target_size: int = 16, offset: float = 0.5) -> torch.Tensor:
    """Read a high-resolution Isaac Lab height scan and downsample it for CNN input."""
    from isaaclab.envs import mdp as isaac_mdp

    flat = isaac_mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    n_rays = flat.shape[1]
    side = math.isqrt(n_rays)
    if side * side != n_rays:
        raise ValueError(f"height_scan length {n_rays} is not a perfect square")
    high_res = flat.reshape(env.num_envs, side, side)
    return downsample_height_map(high_res, target_size=target_size).unsqueeze(1)
