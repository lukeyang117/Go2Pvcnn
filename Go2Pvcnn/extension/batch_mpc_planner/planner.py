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
from .profiling import MpcProfile, maybe_print_mpc_profile, should_profile_mpc
from .terrain import height_at
from .types import MPC_HARD_REASON_COUNT, MpcPlannerResult, MpcPlannerStatus, MpcPlannerTerrain, MpcRobotState
from .variables import MpcOptimizationVariables, init_optimization_variables


def _normal_tensor(value: Tensor | None) -> Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value).detach().clone()


def _normal_state(state: MpcRobotState) -> MpcRobotState:
    return MpcRobotState(
        root_pos=_normal_tensor(state.root_pos),
        root_rpy=_normal_tensor(state.root_rpy),
        foot_pos=_normal_tensor(state.foot_pos),
        joint_angles=_normal_tensor(state.joint_angles),
        foot_vel=_normal_tensor(state.foot_vel),
    )


def _normal_terrain(terrain: MpcPlannerTerrain) -> MpcPlannerTerrain:
    return MpcPlannerTerrain(
        height_map=_normal_tensor(terrain.height_map),
        semantic_map=_normal_tensor(terrain.semantic_map),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=_normal_tensor(terrain.sensor_pos_w),
        sensor_yaw=_normal_tensor(terrain.sensor_yaw),
    )


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


def _standstill_result_from_state(
    state: MpcRobotState,
    *,
    horizon: int,
    cfg: MpcPlannerCfg,
    terrain: MpcPlannerTerrain | None = None,
) -> MpcPlannerResult:
    root_pos0 = torch.as_tensor(state.root_pos)
    device = root_pos0.device
    dtype = root_pos0.dtype
    batch = int(root_pos0.shape[0])
    root_rpy0 = torch.as_tensor(state.root_rpy, dtype=dtype, device=device)
    foot0 = torch.as_tensor(state.foot_pos, dtype=dtype, device=device)
    joint0 = torch.as_tensor(state.joint_angles, dtype=dtype, device=device)
    foot_standstill = foot0
    if terrain is not None:
        terrain_z = height_at(terrain, foot0[..., :2]).to(dtype=dtype, device=device)
        foot_standstill = torch.cat((foot0[..., :2], terrain_z.unsqueeze(-1)), dim=-1)
    root_pos = root_pos0[:, None, :].expand(batch, int(horizon), 3).contiguous()
    root_rpy = root_rpy0[:, None, :].expand(batch, int(horizon), 3).contiguous()
    foot_pos = foot_standstill[:, None, :, :].expand(batch, int(horizon), 4, 3).contiguous()
    joint_angles = joint0[:, None, :].expand(batch, int(horizon), 12).contiguous()
    contact_state = torch.ones((batch, int(horizon), 4), dtype=torch.bool, device=device)
    touchdown_seq = foot_standstill.unsqueeze(2).expand(batch, 4, int(cfg.runtime.touchdown_event_cap), 3).contiguous()
    planned_touchdown_w = foot_standstill[:, None, :, :].expand(batch, int(horizon), 4, 3).contiguous()
    cost_total = torch.zeros((batch,), dtype=dtype, device=device)
    status = torch.full((batch,), int(MpcPlannerStatus.OK), dtype=torch.long, device=device)
    feasible = torch.ones((batch,), dtype=torch.bool, device=device)
    safe_fallback = torch.zeros((batch,), dtype=torch.bool, device=device)
    hard_reason_mask = torch.zeros((batch, MPC_HARD_REASON_COUNT), dtype=torch.bool, device=device)
    return MpcPlannerResult(
        root_pos=root_pos,
        root_rpy=root_rpy,
        foot_pos=foot_pos,
        joint_angles=joint_angles,
        contact_state=contact_state,
        touchdown_seq=touchdown_seq,
        planned_touchdown_w=planned_touchdown_w,
        cost_total=cost_total,
        cost_breakdown={"cost_total": cost_total},
        status=status,
        feasible=feasible,
        safe_fallback=safe_fallback,
        loss_breakdown={} if cfg.diagnostics.enabled else None,
        hard_reason_mask=hard_reason_mask if cfg.diagnostics.enabled else None,
    )


def _subset_state(state: MpcRobotState, ids: Tensor) -> MpcRobotState:
    return MpcRobotState(
        root_pos=state.root_pos.index_select(0, ids),
        root_rpy=state.root_rpy.index_select(0, ids),
        foot_pos=state.foot_pos.index_select(0, ids),
        joint_angles=state.joint_angles.index_select(0, ids),
        foot_vel=state.foot_vel.index_select(0, ids) if state.foot_vel is not None else None,
    )


def _subset_terrain(terrain: MpcPlannerTerrain, ids: Tensor) -> MpcPlannerTerrain:
    return MpcPlannerTerrain(
        height_map=terrain.height_map.index_select(0, ids),
        semantic_map=terrain.semantic_map.index_select(0, ids) if terrain.semantic_map is not None else None,
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        sensor_pos_w=terrain.sensor_pos_w.index_select(0, ids) if terrain.sensor_pos_w is not None else None,
        sensor_yaw=terrain.sensor_yaw.index_select(0, ids) if terrain.sensor_yaw is not None else None,
    )


def _merge_subset_result(base: MpcPlannerResult, subset: MpcPlannerResult, ids: Tensor, *, diagnostics_enabled: bool) -> MpcPlannerResult:
    def _scatter(field: str):
        dst = getattr(base, field).clone()
        dst.index_copy_(0, ids, getattr(subset, field))
        return dst

    cost_breakdown = {name: value.clone() for name, value in base.cost_breakdown.items()}
    for name, value in subset.cost_breakdown.items():
        target = cost_breakdown.get(name)
        if target is None:
            target = torch.zeros_like(base.cost_total)
        else:
            target = target.clone()
        target.index_copy_(0, ids, value)
        cost_breakdown[name] = target

    loss_breakdown = None
    if diagnostics_enabled:
        loss_breakdown = {}
        base_loss = base.loss_breakdown or {}
        sub_loss = subset.loss_breakdown or {}
        for name in sorted(set(base_loss) | set(sub_loss)):
            if name in base_loss:
                target = base_loss[name].clone()
            else:
                target = torch.zeros_like(base.cost_total)
            if name in sub_loss:
                target.index_copy_(0, ids, sub_loss[name])
            loss_breakdown[name] = target

    hard_reason_mask = None
    if diagnostics_enabled:
        hard_reason_mask = base.hard_reason_mask.clone() if base.hard_reason_mask is not None else None
        if hard_reason_mask is not None and subset.hard_reason_mask is not None:
            hard_reason_mask.index_copy_(0, ids, subset.hard_reason_mask)

    return MpcPlannerResult(
        root_pos=_scatter("root_pos"),
        root_rpy=_scatter("root_rpy"),
        foot_pos=_scatter("foot_pos"),
        joint_angles=_scatter("joint_angles"),
        contact_state=_scatter("contact_state"),
        touchdown_seq=_scatter("touchdown_seq"),
        planned_touchdown_w=_scatter("planned_touchdown_w"),
        cost_total=_scatter("cost_total"),
        cost_breakdown=cost_breakdown,
        status=_scatter("status"),
        feasible=_scatter("feasible"),
        safe_fallback=_scatter("safe_fallback"),
        loss_breakdown=loss_breakdown,
        hard_reason_mask=hard_reason_mask,
    )


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
    profile = (
        MpcProfile(sync_cuda=bool(cfg.diagnostics.profile_cuda_sync))
        if should_profile_mpc(cfg)
        else None
    )
    plan_t0 = profile.now() if profile is not None else 0.0
    with torch.inference_mode(False):
        normalize_t0 = profile.now() if profile is not None else 0.0
        terrain = _normal_terrain(terrain)
        state = _normal_state(state)
        command = _normal_tensor(command)
        if profile is not None:
            profile.add_stage("plan.normalize", (profile.now() - normalize_t0) * 1000.0)
    batch_for_zero = int(state.root_pos.shape[0])
    horizon_for_zero = int(cfg.runtime.horizon_steps)
    zero_mask_pre = _zero_command_mask(command, batch=batch_for_zero, device=state.root_pos.device)
    if bool(torch.all(zero_mask_pre)):
        if profile is not None:
            profile.batch_size = batch_for_zero
            profile.horizon = horizon_for_zero
            profile.add_stage("plan.zero_command_standstill", (profile.now() - plan_t0) * 1000.0)
            profile.add_stage("plan.total", (profile.now() - plan_t0) * 1000.0)
            maybe_print_mpc_profile(profile, cfg=cfg)
        return _standstill_result_from_state(state, horizon=horizon_for_zero, cfg=cfg, terrain=terrain)
    if bool(torch.any(zero_mask_pre)):
        nonzero_ids = torch.nonzero(torch.logical_not(zero_mask_pre), as_tuple=False).squeeze(-1)
        base = _standstill_result_from_state(state, horizon=horizon_for_zero, cfg=cfg, terrain=terrain)
        subset = plan_segment(
            _subset_terrain(terrain, nonzero_ids),
            _subset_state(state, nonzero_ids),
            command.index_select(0, nonzero_ids),
            cfg=cfg,
            warm_start=None,
        )
        merged = _merge_subset_result(base, subset, nonzero_ids, diagnostics_enabled=bool(cfg.diagnostics.enabled))
        if profile is not None:
            profile.batch_size = batch_for_zero
            profile.horizon = horizon_for_zero
            profile.add_stage("plan.mixed_zero_split", (profile.now() - plan_t0) * 1000.0)
            profile.add_stage("plan.total", (profile.now() - plan_t0) * 1000.0)
            maybe_print_mpc_profile(profile, cfg=cfg)
        return merged
    nominal_t0 = profile.now() if profile is not None else 0.0
    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)
    if profile is not None:
        profile.add_stage("plan.nominal", (profile.now() - nominal_t0) * 1000.0)
    variables_t0 = profile.now() if profile is not None else 0.0
    variables = init_optimization_variables(nominal, cfg.runtime, warm_start=warm_start)
    if profile is not None:
        profile.add_stage("plan.init_variables", (profile.now() - variables_t0) * 1000.0)
        profile.batch_size = int(nominal["root_pos"].shape[0])
        profile.horizon = int(nominal["root_pos"].shape[1])
    decoded, cost_total, loss_breakdown, finite_ok = optimize_variables(
        nominal,
        variables,
        state,
        command,
        terrain,
        cfg,
        profile=profile,
    )

    post_t0 = profile.now() if profile is not None else 0.0
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
    if horizon > 0:
        joint_seq = joint_seq.clone()
        joint_seq[:, 0, :] = state_joints[:, 0, :]
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
    if profile is not None:
        profile.add_stage("plan.postprocess", (profile.now() - post_t0) * 1000.0)
        profile.add_stage("plan.total", (profile.now() - plan_t0) * 1000.0)
        maybe_print_mpc_profile(profile, cfg=cfg)

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
