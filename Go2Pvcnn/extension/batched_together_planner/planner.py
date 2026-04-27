"""Native torch-only P0 together planner core."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import TogetherPlannerConfig, validate_config
from .costs import compute_costs
from .kinematics import evaluate_kinematics
from .parameterization import expand_segment
from .schedule import build_fixed_schedule
from .terrain import TogetherPlannerTerrain
from .types import TogetherPlannerResult, TogetherPlannerStatus, TogetherRobotState


def _as_batch_tensor(value: Tensor, *, device: torch.device, dtype: torch.dtype, suffix: tuple[int, ...], name: str) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == len(suffix):
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != len(suffix) + 1 or tuple(tensor.shape[1:]) != suffix:
        raise ValueError(f"{name} must have shape [B, ...] with suffix {suffix}")
    return tensor


def _coerce_state(state: TogetherRobotState, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    root_pos = _as_batch_tensor(state.root_pos, device=device, dtype=dtype, suffix=(3,), name="root_pos")
    root_rpy = _as_batch_tensor(state.root_rpy, device=device, dtype=dtype, suffix=(3,), name="root_rpy")
    foot_pos = _as_batch_tensor(state.foot_pos, device=device, dtype=dtype, suffix=(4, 3), name="foot_pos")
    if state.joint_angles is None:
        joint_angles = torch.zeros((root_pos.shape[0], 12), device=device, dtype=dtype)
    else:
        joint_angles = _as_batch_tensor(state.joint_angles, device=device, dtype=dtype, suffix=(12,), name="joint_angles")
    if root_rpy.shape[0] != root_pos.shape[0] or foot_pos.shape[0] != root_pos.shape[0] or joint_angles.shape[0] != root_pos.shape[0]:
        raise ValueError("state tensors must share batch dimension")
    return root_pos, root_rpy, foot_pos, joint_angles


def _coerce_command(command_batch: Tensor, *, device: torch.device, dtype: torch.dtype, batch_size: int) -> Tensor:
    command = torch.as_tensor(command_batch, device=device, dtype=dtype)
    if command.ndim == 1:
        command = command.unsqueeze(0)
    if command.shape != (batch_size, 3):
        raise ValueError("command_batch must have shape [B, 3]")
    return command


def plan_segment(
    terrain: TogetherPlannerTerrain,
    state: TogetherRobotState,
    command_batch: Tensor,
    cfg: TogetherPlannerConfig | None = None,
) -> TogetherPlannerResult:
    planner_cfg = cfg or TogetherPlannerConfig()
    validate_config(planner_cfg)
    if not isinstance(terrain, TogetherPlannerTerrain):
        raise TypeError("terrain must be a TogetherPlannerTerrain")
    device = terrain.device
    dtype = torch.float32 if terrain.dtype not in (torch.float32, torch.float64) else terrain.dtype
    root_pos, root_rpy, foot_pos, _ = _coerce_state(state, device=device, dtype=dtype)
    batch_size = root_pos.shape[0]
    if terrain.batch_size != batch_size:
        raise ValueError("terrain and state must share batch dimension")
    command = _coerce_command(command_batch, device=device, dtype=dtype, batch_size=batch_size)
    schedule = build_fixed_schedule(
        batch_size,
        int(planner_cfg.horizon_steps),
        float(planner_cfg.dt),
        device,
        dtype,
        command,
        planner_cfg,
    )
    rollout = expand_segment(
        terrain,
        root_pos,
        root_rpy,
        foot_pos,
        command,
        schedule,
        planner_cfg,
    )
    kinematics = evaluate_kinematics(rollout.root_pos, rollout.root_rpy, rollout.foot_pos)
    costs = compute_costs(terrain, rollout, kinematics, command, planner_cfg)
    ok_status = torch.full((batch_size,), int(TogetherPlannerStatus.OK), device=device, dtype=torch.int64)
    infeasible_status = torch.full((batch_size,), int(TogetherPlannerStatus.ALL_INFEASIBLE), device=device, dtype=torch.int64)
    status = torch.where(costs.feasible, ok_status, infeasible_status)
    return TogetherPlannerResult(
        root_pos=rollout.root_pos,
        root_rpy=rollout.root_rpy,
        foot_pos=rollout.foot_pos,
        joint_angles=kinematics.joint_angles,
        contact_state=rollout.contact_state,
        touchdown_seq=rollout.touchdown_seq,
        touchdown_mask=rollout.touchdown_mask,
        cost_total=costs.total,
        cost_breakdown=costs.breakdown,
        status=status,
        feasible=costs.feasible,
        safe_fallback=costs.safe_fallback,
        joint_limit_violation=kinematics.joint_limit_violation,
        workspace_margin=kinematics.workspace_margin,
        support_xy=rollout.support_xy,
        support_height=rollout.support_height,
        support_slope=rollout.support_slope,
    )


__all__ = ["plan_segment"]
