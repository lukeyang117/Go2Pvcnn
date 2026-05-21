"""Optimization variables and decode helpers for batch MPC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import MpcRuntimeCfg
from .terrain import height_at
from .types import MpcPlannerTerrain


@dataclass
class MpcOptimizationVariables:
    root_pos_residual: Tensor
    root_rpy_residual: Tensor
    foot_pos_residual: Tensor
    swing_center_raw: Tensor
    swing_width_raw: Tensor

    def parameters(self) -> list[Tensor]:
        return [
            self.root_pos_residual,
            self.root_rpy_residual,
            self.foot_pos_residual,
            self.swing_center_raw,
            self.swing_width_raw,
        ]


@dataclass(frozen=True)
class DecodedMpcTrajectory:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    swing_center: Tensor
    swing_width: Tensor
    swing_start: Tensor
    swing_end: Tensor
    swing_prob: Tensor
    contact_prob: Tensor


def _clone_or_zeros(value: Tensor | None, like: Tensor) -> Tensor:
    if value is None:
        return torch.zeros_like(like)
    out = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if tuple(out.shape) != tuple(like.shape):
        return torch.zeros_like(like)
    return out.clone()


def init_optimization_variables(
    nominal: dict[str, Tensor],
    runtime_cfg: MpcRuntimeCfg,
    *,
    warm_start: MpcOptimizationVariables | None = None,
) -> MpcOptimizationVariables:
    # Rollout can invoke planner under torch.inference_mode(); create optimizer
    # variables in normal autograd mode so Adam can update them in-place.
    with torch.inference_mode(False):
        root_pos_like = nominal["root_pos"]
        root_rpy_like = nominal["root_rpy"]
        foot_pos_like = nominal["foot_pos"]
        center_like = nominal["swing_center"]
        width_like = nominal["swing_width"]
        if warm_start is not None and runtime_cfg.warm_start_from_previous_plan:
            root_pos_residual = _clone_or_zeros(warm_start.root_pos_residual, root_pos_like)
            root_rpy_residual = _clone_or_zeros(warm_start.root_rpy_residual, root_rpy_like)
            foot_pos_residual = _clone_or_zeros(warm_start.foot_pos_residual, foot_pos_like)
            swing_center_raw = _clone_or_zeros(warm_start.swing_center_raw, center_like)
            swing_width_raw = _clone_or_zeros(warm_start.swing_width_raw, width_like)
        else:
            root_pos_residual = torch.zeros_like(root_pos_like)
            root_rpy_residual = torch.zeros_like(root_rpy_like)
            foot_pos_residual = torch.zeros_like(foot_pos_like)
            swing_center_raw = torch.zeros_like(center_like)
            swing_width_raw = _width_prior_to_raw(width_like, runtime_cfg)

        return MpcOptimizationVariables(
            root_pos_residual=root_pos_residual.detach().clone().requires_grad_(True),
            root_rpy_residual=root_rpy_residual.detach().clone().requires_grad_(True),
            foot_pos_residual=foot_pos_residual.detach().clone().requires_grad_(True),
            swing_center_raw=swing_center_raw.detach().clone().requires_grad_(True),
            swing_width_raw=swing_width_raw.detach().clone().requires_grad_(True),
        )


def _width_prior_to_raw(width: Tensor, runtime_cfg: MpcRuntimeCfg) -> Tensor:
    width_min = float(runtime_cfg.swing_window_min_width)
    width_max = float(runtime_cfg.swing_window_max_width)
    normalized = (width - width_min) / max(width_max - width_min, 1.0e-6)
    normalized = normalized.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.logit(normalized)


def _circular_abs_distance(a: Tensor, b: Tensor) -> Tensor:
    diff = torch.remainder(a - b + 0.5, 1.0) - 0.5
    return torch.abs(diff)


def _sample_time_noncyclic(values: Tensor, phase: Tensor) -> Tensor:
    batch, horizon, legs, *tail = values.shape
    pos = torch.clamp(phase, 0.0, 1.0) * float(max(horizon - 1, 1))
    i0 = torch.floor(pos).to(dtype=torch.long).clamp(0, horizon - 1)
    i1 = (i0 + 1).clamp(0, horizon - 1)
    alpha = (pos - torch.floor(pos)).to(dtype=values.dtype)
    b = torch.arange(batch, device=values.device).view(batch, 1).expand(batch, legs)
    l = torch.arange(legs, device=values.device).view(1, legs).expand(batch, legs)
    v0 = values[b, i0, l]
    v1 = values[b, i1, l]
    return torch.lerp(v0, v1, alpha.view(batch, legs, *([1] * len(tail))))


def _ground_touchdowns_and_lock_stance(
    terrain: MpcPlannerTerrain,
    foot_pos: Tensor,
    touchdown_phase: Tensor,
) -> Tensor:
    batch, horizon, legs, _ = foot_pos.shape
    sampled_touchdown = _sample_time_noncyclic(foot_pos, touchdown_phase)
    touchdown_xy = sampled_touchdown[..., :2]
    touchdown_z = height_at(terrain, touchdown_xy).to(dtype=foot_pos.dtype, device=foot_pos.device)
    grounded_touchdown = torch.cat((touchdown_xy, touchdown_z.unsqueeze(-1)), dim=-1)

    touchdown_pos = torch.clamp(touchdown_phase, 0.0, 1.0) * float(max(horizon - 1, 1))
    touchdown_frame = torch.floor(touchdown_pos).to(dtype=torch.long).clamp(0, horizon - 1)
    frame_ids = torch.arange(horizon, dtype=torch.long, device=foot_pos.device).view(1, horizon, 1)
    post_touchdown = frame_ids >= touchdown_frame.view(batch, 1, legs)
    return torch.where(
        post_touchdown.unsqueeze(-1),
        grounded_touchdown[:, None, :, :].expand(batch, horizon, legs, 3),
        foot_pos,
    )


def decode_trajectory(
    nominal: dict[str, Tensor],
    variables: MpcOptimizationVariables,
    runtime_cfg: MpcRuntimeCfg,
    *,
    terrain: MpcPlannerTerrain | None = None,
) -> DecodedMpcTrajectory:
    root_pos = nominal["root_pos"] + variables.root_pos_residual
    root_rpy = nominal["root_rpy"] + variables.root_rpy_residual
    foot_pos = nominal["foot_pos"] + variables.foot_pos_residual
    center_prior = nominal["swing_center"]
    width_min = float(runtime_cfg.swing_window_min_width)
    width_max = float(runtime_cfg.swing_window_max_width)
    swing_center = torch.remainder(
        center_prior + float(runtime_cfg.swing_window_center_scale) * torch.tanh(variables.swing_center_raw),
        1.0,
    )
    swing_width = width_min + (width_max - width_min) * torch.sigmoid(variables.swing_width_raw)
    swing_start = torch.remainder(swing_center - 0.5 * swing_width, 1.0)
    swing_end = torch.remainder(swing_center + 0.5 * swing_width, 1.0)
    touchdown_phase = torch.clamp(swing_center + 0.5 * swing_width, min=0.0, max=1.0)
    if terrain is not None:
        foot_pos = _ground_touchdowns_and_lock_stance(terrain, foot_pos, touchdown_phase)

    horizon = int(root_pos.shape[1])
    frame_phase = torch.arange(horizon, dtype=root_pos.dtype, device=root_pos.device).view(1, horizon, 1) / float(horizon)
    dist = _circular_abs_distance(frame_phase, swing_center.unsqueeze(1))
    swing_prob = torch.sigmoid(float(runtime_cfg.swing_window_temperature) * (0.5 * swing_width.unsqueeze(1) - dist))
    contact_prob = 1.0 - swing_prob
    return DecodedMpcTrajectory(
        root_pos=root_pos,
        root_rpy=root_rpy,
        foot_pos=foot_pos,
        swing_center=swing_center,
        swing_width=swing_width,
        swing_start=swing_start,
        swing_end=swing_end,
        swing_prob=swing_prob,
        contact_prob=contact_prob,
    )


__all__ = [
    "DecodedMpcTrajectory",
    "MpcOptimizationVariables",
    "decode_trajectory",
    "init_optimization_variables",
]
