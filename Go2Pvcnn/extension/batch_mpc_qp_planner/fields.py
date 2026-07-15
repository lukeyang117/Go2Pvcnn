"""Differentiable terrain and semantic fields for the MPC-QP backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.batch_mpc_planner.terrain import height_at
from extension.batch_mpc_planner.types import MpcPlannerTerrain


@dataclass(frozen=True)
class QpFieldSample:
    height: Tensor
    height_grad_xy: Tensor
    semantic_risk: Tensor
    semantic_grad_xy: Tensor
    roughness: Tensor
    roughness_grad_xy: Tensor


@dataclass(frozen=True)
class QpDifferentiableFields:
    terrain: MpcPlannerTerrain
    semantic_risk_terrain: MpcPlannerTerrain
    roughness_terrain: MpcPlannerTerrain
    eps_m: float

    def query(self, points_xy: Tensor) -> QpFieldSample:
        points = torch.as_tensor(points_xy, dtype=torch.float32, device=self.terrain.height_map.device)
        dx = torch.tensor([self.eps_m, 0.0], dtype=points.dtype, device=points.device)
        dy = torch.tensor([0.0, self.eps_m], dtype=points.dtype, device=points.device)

        height = height_at(self.terrain, points).to(dtype=points.dtype, device=points.device)
        height_x0 = height_at(self.terrain, points - dx).to(dtype=points.dtype, device=points.device)
        height_x1 = height_at(self.terrain, points + dx).to(dtype=points.dtype, device=points.device)
        height_y0 = height_at(self.terrain, points - dy).to(dtype=points.dtype, device=points.device)
        height_y1 = height_at(self.terrain, points + dy).to(dtype=points.dtype, device=points.device)
        height_grad = torch.stack(
            ((height_x1 - height_x0) / (2.0 * self.eps_m), (height_y1 - height_y0) / (2.0 * self.eps_m)),
            dim=-1,
        )

        semantic = height_at(self.semantic_risk_terrain, points).to(dtype=points.dtype, device=points.device)
        semantic_x0 = height_at(self.semantic_risk_terrain, points - dx).to(dtype=points.dtype, device=points.device)
        semantic_x1 = height_at(self.semantic_risk_terrain, points + dx).to(dtype=points.dtype, device=points.device)
        semantic_y0 = height_at(self.semantic_risk_terrain, points - dy).to(dtype=points.dtype, device=points.device)
        semantic_y1 = height_at(self.semantic_risk_terrain, points + dy).to(dtype=points.dtype, device=points.device)
        semantic_grad = torch.stack(
            ((semantic_x1 - semantic_x0) / (2.0 * self.eps_m), (semantic_y1 - semantic_y0) / (2.0 * self.eps_m)),
            dim=-1,
        )

        roughness = height_at(self.roughness_terrain, points).to(dtype=points.dtype, device=points.device)
        rough_x0 = height_at(self.roughness_terrain, points - dx).to(dtype=points.dtype, device=points.device)
        rough_x1 = height_at(self.roughness_terrain, points + dx).to(dtype=points.dtype, device=points.device)
        rough_y0 = height_at(self.roughness_terrain, points - dy).to(dtype=points.dtype, device=points.device)
        rough_y1 = height_at(self.roughness_terrain, points + dy).to(dtype=points.dtype, device=points.device)
        rough_grad = torch.stack(
            ((rough_x1 - rough_x0) / (2.0 * self.eps_m), (rough_y1 - rough_y0) / (2.0 * self.eps_m)),
            dim=-1,
        )

        return QpFieldSample(
            height=torch.nan_to_num(height),
            height_grad_xy=torch.nan_to_num(height_grad),
            semantic_risk=torch.nan_to_num(semantic),
            semantic_grad_xy=torch.nan_to_num(semantic_grad),
            roughness=torch.nan_to_num(roughness),
            roughness_grad_xy=torch.nan_to_num(rough_grad),
        )


def _terrain_with_height_like(terrain: MpcPlannerTerrain, height_map: Tensor) -> MpcPlannerTerrain:
    return MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=None,
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=terrain.sensor_pos_w,
        sensor_yaw=terrain.sensor_yaw,
        is_plane_terrain=terrain.is_plane_terrain,
    )


def _roughness_from_height(height: Tensor) -> Tensor:
    padded = torch.nn.functional.pad(height.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    center = padded[:, 1:-1, 1:-1]
    dx = 0.5 * (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2])
    dy = 0.5 * (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1])
    edge = torch.maximum(
        torch.abs(center - padded[:, 1:-1, 2:]),
        torch.abs(center - padded[:, 2:, 1:-1]),
    )
    return torch.sqrt(dx.square() + dy.square() + 1.0e-12) + edge


def build_qp_fields(terrain: MpcPlannerTerrain, *, eps_m: float = 0.025) -> QpDifferentiableFields:
    height = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    if terrain.semantic_map is None:
        semantic_risk = torch.zeros_like(height)
    else:
        semantic = torch.as_tensor(terrain.semantic_map, device=height.device)
        if semantic.ndim == 2:
            semantic = semantic.unsqueeze(0)
        semantic_risk = (semantic != 0).to(dtype=height.dtype, device=height.device)
        semantic_risk = torch.nn.functional.avg_pool2d(
            semantic_risk.unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1).clamp(0.0, 1.0)
    roughness = _roughness_from_height(height)
    return QpDifferentiableFields(
        terrain=_terrain_with_height_like(terrain, height),
        semantic_risk_terrain=_terrain_with_height_like(terrain, semantic_risk),
        roughness_terrain=_terrain_with_height_like(terrain, roughness),
        eps_m=max(float(eps_m), 1.0e-4),
    )


__all__ = ["QpDifferentiableFields", "QpFieldSample", "build_qp_fields"]
