"""Batched semantic Raibert foothold planner used by the SemLoco baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SemanticRaibertPlannerCfg:
    search_size: int = 5
    search_resolution: float = 0.04
    foot_radius: float = 0.04
    stance_time: float = 0.25
    w_forward: float = 1.0
    w_backward: float = 0.5
    w_lateral: float = 8.0
    w_collision: float = 100.0


@dataclass(frozen=True)
class PlannerOutput:
    target_foothold_w: Tensor
    nominal_foothold_w: Tensor
    valid: Tensor
    collision_free: Tensor


class SemanticRaibertPlanner:
    """Select collision-free footholds around a Raibert prediction in batches."""

    def __init__(self, cfg: SemanticRaibertPlannerCfg | None = None):
        self.cfg = cfg or SemanticRaibertPlannerCfg()
        if self.cfg.search_size < 1 or self.cfg.search_size % 2 == 0:
            raise ValueError("search_size must be a positive odd integer")

    @staticmethod
    def _yaw(quat: Tensor) -> Tensor:
        w, x, y, z = quat.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def plan(
        self,
        root_pos_w: Tensor,
        root_quat_w: Tensor,
        root_lin_vel_b: Tensor,
        root_ang_vel_b: Tensor,
        foot_pos_w: Tensor,
        velocity_command: Tensor,
        gait_phase: Tensor,
        semantic_height_scanner: Tensor | dict[str, Tensor],
    ) -> PlannerOutput:
        del root_ang_vel_b, gait_phase
        root = torch.as_tensor(root_pos_w)
        feet = torch.as_tensor(foot_pos_w, dtype=root.dtype, device=root.device)
        command = torch.as_tensor(velocity_command, dtype=root.dtype, device=root.device)
        if root.ndim != 2 or root.shape[-1] != 3 or feet.shape != (root.shape[0], 4, 3):
            raise ValueError("root_pos_w must be [N,3] and foot_pos_w must be [N,4,3]")
        if command.shape[-1] < 3:
            raise ValueError("velocity_command must contain vx, vy, vyaw")

        # The current foot pose is the nominal foothold supplied by the existing
        # kinematics pipeline; only the commanded Raibert displacement is added.
        nominal = feet.clone()
        dt = 0.5 * self.cfg.stance_time
        yaw = self._yaw(torch.as_tensor(root_quat_w, dtype=root.dtype, device=root.device))
        c, s = torch.cos(yaw), torch.sin(yaw)
        vx, vy, vyaw = command[:, 0], command[:, 1], command[:, 2]
        body_delta = torch.zeros(root.shape[0], 4, 3, dtype=root.dtype, device=root.device)
        body_delta[..., 0] = dt * vx[:, None]
        body_delta[..., 1] = dt * vy[:, None]
        # Front/rear x position determines the yaw-induced lateral displacement.
        x_nominal = nominal[..., 0] - root[:, None, 0]
        body_delta[..., 1] += dt * vyaw[:, None] * x_nominal
        world_delta = torch.zeros_like(body_delta)
        world_delta[..., 0] = c[:, None] * body_delta[..., 0] - s[:, None] * body_delta[..., 1]
        world_delta[..., 1] = s[:, None] * body_delta[..., 0] + c[:, None] * body_delta[..., 1]
        raibert = nominal + world_delta

        height, semantic, origin, resolution = self._scanner_tensors(semantic_height_scanner, root)
        offsets = torch.arange(
            -(self.cfg.search_size // 2), self.cfg.search_size // 2 + 1,
            device=root.device, dtype=root.dtype,
        ) * self.cfg.search_resolution
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_xy = torch.stack((ox.reshape(-1), oy.reshape(-1)), dim=-1)
        candidates = raibert[:, :, None, :].expand(-1, -1, offset_xy.shape[0], -1).clone()
        candidates[..., :2] += offset_xy[None, None]
        gx = ((candidates[..., 0] - origin[:, None, None, 0]) / resolution).round().long()
        gy = ((candidates[..., 1] - origin[:, None, None, 1]) / resolution).round().long()
        n, h, w = height.shape
        inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        gx_safe, gy_safe = gx.clamp(0, w - 1), gy.clamp(0, h - 1)
        flat_index = (gy_safe * w + gx_safe).reshape(n, -1)
        cell_sem = semantic.reshape(n, -1).gather(1, flat_index).reshape_as(gx)
        cell_height = height.reshape(n, -1).gather(1, flat_index).reshape_as(gx)
        collision = (cell_sem > 0) | ~torch.isfinite(cell_height) | ~inside
        # Keep targets on the scanned surface where a finite height is available.
        candidates[..., 2] = torch.where(torch.isfinite(cell_height), cell_height, candidates[..., 2])
        delta = candidates - raibert[:, :, None]
        cost = self.cfg.w_forward * delta[..., 0].clamp_min(0) + self.cfg.w_backward * (-delta[..., 0]).clamp_min(0)
        cost = cost + self.cfg.w_lateral * delta[..., 1].abs() + self.cfg.w_collision * collision.to(cost.dtype)
        feasible = ~collision
        cost = cost.masked_fill(~feasible, float("inf"))
        best_cost, best_idx = cost.min(dim=-1)
        valid = torch.isfinite(best_cost)
        best = candidates.gather(2, best_idx[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        target = torch.where(valid[..., None], best, nominal)
        return PlannerOutput(target, nominal, valid, valid)

    @staticmethod
    def _scanner_tensors(scanner: Tensor | dict[str, Tensor], root: Tensor) -> tuple[Tensor, Tensor, Tensor, float]:
        if isinstance(scanner, dict):
            height = scanner["height"]
            semantic = scanner["semantic"]
            origin = scanner.get("origin", root[:, :3] - 0.5 * (height.shape[-1] - 1) * scanner.get("resolution", 0.01))
            resolution = float(scanner.get("resolution", 0.01))
        else:
            height = scanner
            semantic = torch.zeros_like(height, dtype=torch.long)
            resolution = 0.01
            origin = root[:, :3] - 0.5 * (height.shape[-1] - 1) * resolution
        height = torch.as_tensor(height, dtype=root.dtype, device=root.device)
        semantic = torch.as_tensor(semantic, dtype=torch.long, device=root.device)
        if height.ndim != 3 or height.shape[-1] != height.shape[-2]:
            raise ValueError("scanner height must be [N,H,W] square")
        if semantic.ndim == 2:
            semantic = semantic.unsqueeze(0).expand(height.shape[0], -1, -1)
        return height, semantic, torch.as_tensor(origin, dtype=root.dtype, device=root.device), resolution


__all__ = ["PlannerOutput", "SemanticRaibertPlanner", "SemanticRaibertPlannerCfg"]
