"""Dense MPC planner core (gradient-based trajectory optimization)."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import MpcPlannerCfg, validate_mpc_config
from .diagnostics import evaluate_hard_reasons, status_from_hard_reasons
from .kinematics import solve_joint_angles_from_trajectory
from .losses.terrain_clearance import finite_horizon_touchdown_phase, sample_time
from .nominal import build_nominal_trajectory
from .optimizer import optimize_variables
from .types import MPC_HARD_REASON_COUNT, MpcPlannerResult, MpcPlannerStatus, MpcPlannerTerrain, MpcRobotState
from .variables import MpcOptimizationVariables, init_optimization_variables


def sample_touchdown_positions(foot_pos: Tensor, swing_center: Tensor, swing_width: Tensor) -> Tensor:
    touchdown_phase = finite_horizon_touchdown_phase(swing_center, swing_width)
    return sample_time(foot_pos, touchdown_phase, cyclic=False)


def _touchdown_export(
    foot_pos: Tensor,
    swing_center: Tensor,
    swing_width: Tensor,
    *,
    event_cap: int,
) -> tuple[Tensor, Tensor]:
    batch, horizon, legs, _ = foot_pos.shape
    touchdown_w = sample_touchdown_positions(foot_pos, swing_center, swing_width)
    touchdown_seq = touchdown_w.unsqueeze(2).expand(batch, legs, int(event_cap), 3).contiguous()
    planned_touchdown_w = touchdown_w.unsqueeze(1).expand(batch, horizon, legs, 3).contiguous()
    return touchdown_seq, planned_touchdown_w


def _zero_command_mask(command: Tensor, *, batch: int, device: torch.device) -> Tensor:
    cmd = torch.as_tensor(command, dtype=torch.float32, device=device)
    if cmd.ndim != 2:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    if int(cmd.shape[0]) != batch:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((int(cmd.shape[0]), 3 - int(cmd.shape[-1])), dtype=cmd.dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    return torch.linalg.vector_norm(cmd[:, :3], dim=-1) <= 1.0e-5


def plan_segment(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    *,
    cfg: MpcPlannerCfg,
    warm_start: MpcOptimizationVariables | None = None,
) -> MpcPlannerResult:
    """Plan one horizon for a batch of environments."""
    validate_mpc_config(cfg)
    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)
    variables = init_optimization_variables(nominal, cfg.runtime, warm_start=warm_start)
    decoded, cost_total, loss_breakdown, finite_ok = optimize_variables(nominal, variables, state, command, terrain, cfg)

    batch, horizon = int(decoded.root_pos.shape[0]), int(decoded.root_pos.shape[1])
    zero_mask = _zero_command_mask(command, batch=batch, device=decoded.root_pos.device)
    root_pos = decoded.root_pos
    root_rpy = decoded.root_rpy
    foot_pos = decoded.foot_pos
    contact_state = decoded.contact_prob >= float(cfg.runtime.contact_threshold)
    row_3 = zero_mask.view(batch, 1, 1)
    row_4 = zero_mask.view(batch, 1, 1, 1)
    state_root = torch.as_tensor(state.root_pos, dtype=root_pos.dtype, device=root_pos.device)[:, None, :].expand(batch, horizon, 3)
    state_rpy = torch.as_tensor(state.root_rpy, dtype=root_rpy.dtype, device=root_rpy.device)[:, None, :].expand(batch, horizon, 3)
    state_foot = torch.as_tensor(state.foot_pos, dtype=foot_pos.dtype, device=foot_pos.device)[:, None, :, :].expand(batch, horizon, 4, 3)
    root_pos = torch.where(row_3, state_root, root_pos)
    root_rpy = torch.where(row_3, state_rpy, root_rpy)
    foot_pos = torch.where(row_4, state_foot, foot_pos)
    contact_state = torch.where(row_3, torch.ones_like(contact_state), contact_state)
    joint_seq = solve_joint_angles_from_trajectory(root_pos, root_rpy, foot_pos)
    state_joints = torch.as_tensor(state.joint_angles, dtype=joint_seq.dtype, device=joint_seq.device)[:, None, :].expand(batch, horizon, 12)
    joint_seq = torch.where(row_3, state_joints, joint_seq)
    touchdown_seq, planned_touchdown_w = _touchdown_export(
        foot_pos,
        decoded.swing_center,
        decoded.swing_width,
        event_cap=cfg.runtime.touchdown_event_cap,
    )
    state_touchdown = torch.as_tensor(state.foot_pos, dtype=planned_touchdown_w.dtype, device=planned_touchdown_w.device)
    state_touchdown_w = state_touchdown[:, None, :, :].expand_as(planned_touchdown_w)
    planned_touchdown_w = torch.where(row_4, state_touchdown_w, planned_touchdown_w)
    touchdown_state = state_touchdown.unsqueeze(2).expand(batch, 4, int(cfg.runtime.touchdown_event_cap), 3)
    touchdown_seq = torch.where(row_4, touchdown_state, touchdown_seq)
    cost_breakdown = {"cost_total": cost_total}
    cost_breakdown.update({str(name): value.detach() for name, value in loss_breakdown.items()})
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
            root_pos=root_pos,
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
        root_pos=root_pos,
        root_rpy=root_rpy,
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


__all__ = ["plan_segment", "sample_touchdown_positions"]
