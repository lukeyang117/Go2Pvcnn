"""Dense MPC planner core (gradient-based trajectory optimization)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import MpcPlannerCfg, validate_mpc_config
from .diagnostics import evaluate_hard_reasons, status_from_hard_reasons
from .kinematics import solve_joint_angles_from_trajectory
from .nominal import build_nominal_trajectory
from .optimizer import optimize_variables
from .types import MPC_HARD_REASON_COUNT, MpcFootholdMemory, MpcPlannerResult, MpcPlannerStatus, MpcPlannerTerrain, MpcRobotState
from .variables import MpcOptimizationVariables, init_optimization_variables


def _extract_touchdown_seq(
    foot_pos: Tensor,
    contact_state: Tensor,
    *,
    event_cap: int,
) -> tuple[Tensor, Tensor]:
    """Extract touchdown events and frame-expanded touchdown targets."""
    batch, horizon, legs, _ = foot_pos.shape
    prev = torch.cat((contact_state[:, :1], contact_state[:, :-1]), dim=1)
    rises = torch.logical_and(contact_state, torch.logical_not(prev))
    frame_ids = torch.arange(horizon, dtype=torch.long, device=foot_pos.device).view(1, horizon, 1).expand(batch, horizon, legs)
    fill = torch.full_like(frame_ids, horizon)
    remaining = rises
    seq_pos: list[Tensor] = []
    first_touchdown = foot_pos[:, 0]

    for _ in range(int(event_cap)):
        masked = torch.where(remaining, frame_ids, fill)
        chosen = masked.amin(dim=1)  # [B, 4]
        valid = chosen < horizon
        chosen_idx = chosen.clamp(max=horizon - 1)
        b_idx = torch.arange(batch, device=foot_pos.device).view(batch, 1).expand(batch, legs)
        l_idx = torch.arange(legs, device=foot_pos.device).view(1, legs).expand(batch, legs)
        chosen_pos = foot_pos[b_idx, chosen_idx, l_idx]
        default_pos = foot_pos[:, 0]
        chosen_pos = torch.where(valid.unsqueeze(-1), chosen_pos, default_pos)
        seq_pos.append(chosen_pos)
        remove = torch.logical_and(remaining, frame_ids == chosen.unsqueeze(1))
        remaining = torch.logical_and(remaining, torch.logical_not(remove))
        first_touchdown = torch.where(valid.unsqueeze(-1), chosen_pos, first_touchdown)

    touchdown_seq = torch.stack(seq_pos, dim=2)  # [B, 4, E, 3]
    planned_touchdown_w = first_touchdown.unsqueeze(1).expand(batch, horizon, legs, 3).contiguous()
    return touchdown_seq, planned_touchdown_w


def _terrain_height_at(terrain: MpcPlannerTerrain, points_xy: Tensor) -> Tensor:
    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32, device=points_xy.device)
    if height_map.ndim == 2:
        height_map = height_map.unsqueeze(0)
    points_xy = torch.as_tensor(points_xy, dtype=torch.float32, device=height_map.device)
    batch = int(height_map.shape[0])
    if points_xy.ndim == 2:
        points_xy = points_xy.unsqueeze(0)
    if int(points_xy.shape[0]) == 1 and batch > 1:
        points_xy = points_xy.expand(batch, -1, -1)
    x0, x1 = terrain.world_x_range
    y0, y1 = terrain.world_y_range
    xs = points_xy[..., 0].clamp(float(x0), float(x1))
    ys = points_xy[..., 1].clamp(float(y0), float(y1))
    x_norm = (xs - float(x0)) / max(float(x1) - float(x0), 1.0e-6) * 2.0 - 1.0
    y_norm = (float(y1) - ys) / max(float(y1) - float(y0), 1.0e-6) * 2.0 - 1.0
    sample_grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        height_map.unsqueeze(1),
        sample_grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return sampled[:, 0, :, 0]


def _ground_contact_feet_to_terrain(terrain: MpcPlannerTerrain, foot_pos: Tensor, contact_state: Tensor) -> Tensor:
    batch = int(foot_pos.shape[0])
    terrain_z = _terrain_height_at(terrain, foot_pos[..., :2].reshape(batch, -1, 2)).reshape(foot_pos.shape[:3])
    grounded = foot_pos.clone()
    grounded_z = torch.where(contact_state, terrain_z.to(dtype=grounded.dtype, device=grounded.device), grounded[..., 2])
    grounded[..., 2] = grounded_z
    return grounded


def plan_segment(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    *,
    cfg: MpcPlannerCfg,
    warm_start: MpcOptimizationVariables | None = None,
    memory: MpcFootholdMemory | None = None,
) -> MpcPlannerResult:
    """Plan one horizon for a batch of environments."""
    validate_mpc_config(cfg)
    nominal = build_nominal_trajectory(state, command, cfg.runtime, memory=memory)
    joint_seed = torch.as_tensor(state.joint_angles, dtype=nominal["root_pos"].dtype, device=nominal["root_pos"].device)
    joint_seed_seq = joint_seed.unsqueeze(1).expand(joint_seed.shape[0], cfg.runtime.horizon_steps, joint_seed.shape[1]).contiguous()
    variables = init_optimization_variables(nominal, cfg.runtime, warm_start=warm_start)
    decoded, cost_total, loss_breakdown, finite_ok = optimize_variables(nominal, variables, joint_seed_seq, command, cfg)

    contact_state = decoded.contact_prob > float(cfg.runtime.contact_threshold)
    foot_pos = _ground_contact_feet_to_terrain(terrain, decoded.foot_pos, contact_state)
    joint_seq = solve_joint_angles_from_trajectory(decoded.root_pos, decoded.root_rpy, foot_pos)
    touchdown_seq, planned_touchdown_w = _extract_touchdown_seq(
        foot_pos,
        contact_state,
        event_cap=cfg.runtime.touchdown_event_cap,
    )
    cost_breakdown = {"cost_total": cost_total}
    status = torch.full(
        (decoded.root_pos.shape[0],),
        int(MpcPlannerStatus.OK),
        dtype=torch.long,
        device=decoded.root_pos.device,
    )
    feasible = torch.ones_like(status, dtype=torch.bool)
    safe_fallback = torch.zeros_like(status, dtype=torch.bool)
    hard_reason_mask = torch.zeros(
        (decoded.root_pos.shape[0], MPC_HARD_REASON_COUNT),
        dtype=torch.bool,
        device=decoded.root_pos.device,
    )

    if cfg.diagnostics.enabled:
        hard_reason_mask = evaluate_hard_reasons(
            root_pos=decoded.root_pos,
            foot_pos=foot_pos,
            joint_angles=joint_seq,
            contact_state=contact_state,
            command=torch.as_tensor(command, dtype=decoded.root_pos.dtype, device=decoded.root_pos.device),
        )
        status, feasible, safe_fallback = status_from_hard_reasons(hard_reason_mask)
    finite_ok = torch.as_tensor(finite_ok, dtype=torch.bool, device=decoded.root_pos.device)
    status = torch.where(
        finite_ok,
        status,
        torch.full_like(status, int(MpcPlannerStatus.ALL_INFEASIBLE)),
    )
    feasible = torch.logical_and(feasible, finite_ok)
    safe_fallback = torch.logical_or(safe_fallback, torch.logical_not(finite_ok))

    return MpcPlannerResult(
        root_pos=decoded.root_pos,
        root_rpy=decoded.root_rpy,
        foot_pos=foot_pos,
        joint_angles=joint_seq,
        contact_state=contact_state,
        touchdown_seq=touchdown_seq,
        planned_touchdown_w=planned_touchdown_w,
        cost_total=cost_total,
        cost_breakdown=cost_breakdown,
        status=status,
        feasible=feasible,
        safe_fallback=safe_fallback,
        loss_breakdown=loss_breakdown if cfg.diagnostics.enabled else None,
        hard_reason_mask=hard_reason_mask if cfg.diagnostics.enabled else None,
    )


__all__ = ["plan_segment"]
