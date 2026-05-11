"""Dense MPC planner core (gradient-based trajectory optimization)."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import MpcPlannerCfg, validate_mpc_config
from .diagnostics import evaluate_hard_reasons, status_from_hard_reasons
from .kinematics import solve_joint_angles_from_trajectory
from .nominal import build_nominal_trajectory
from .optimizer import optimize_variables
from .types import MPC_HARD_REASON_COUNT, MpcPlannerResult, MpcPlannerStatus, MpcPlannerTerrain, MpcRobotState
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


def plan_segment(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    *,
    cfg: MpcPlannerCfg,
    warm_start: MpcOptimizationVariables | None = None,
) -> MpcPlannerResult:
    """Plan one horizon for a batch of environments."""
    del terrain  # scaffold: terrain losses are wired through loss registry placeholders.
    validate_mpc_config(cfg)
    nominal = build_nominal_trajectory(state, command, cfg.runtime)
    joint_seed = torch.as_tensor(state.joint_angles, dtype=nominal["root_pos"].dtype, device=nominal["root_pos"].device)
    joint_seed_seq = joint_seed.unsqueeze(1).expand(joint_seed.shape[0], cfg.runtime.horizon_steps, joint_seed.shape[1]).contiguous()
    variables = init_optimization_variables(nominal, cfg.runtime, warm_start=warm_start)
    decoded, cost_total, loss_breakdown, finite_ok = optimize_variables(nominal, variables, joint_seed_seq, command, cfg)
    joint_seq = solve_joint_angles_from_trajectory(decoded.root_pos, decoded.root_rpy, decoded.foot_pos)

    contact_state = decoded.contact_prob > float(cfg.runtime.contact_threshold)
    touchdown_seq, planned_touchdown_w = _extract_touchdown_seq(
        decoded.foot_pos,
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
            foot_pos=decoded.foot_pos,
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
        foot_pos=decoded.foot_pos,
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
