"""Batched trajectory generation main entry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..convention import euler_to_quat_batch, extract_roll_pitch_batch, extract_yaw_batch
from .base_solver import batched_integrate_base_planar, batched_solve_base_trajectory
from .config import BatchedTrajectoryConfig
from .foothold import batched_candidate_total_score, batched_compute_footholds, batched_evaluate_touchdowns
from .gait import GAIT_PARAMS, batched_gait_schedule, batched_legs_requiring_touchdown, batched_next_touchdown_times, batched_stance_time
from .ik import batch_forward_kinematics, batch_inverse_kinematics
from .swing import batched_compute_swing_targets
from .terrain_estimator import batched_estimate_terrain
from .types import BatchedRobotState, BatchedTrajectoryResult

_STANDSTILL_CMD_EPS = 1e-5


@dataclass(frozen=True)
class _CandidatePlan:
    command: Tensor
    touchdowns: Tensor
    score: Tensor


def _resolve_input_device(*values) -> torch.device:
    devices = [value.device for value in values if isinstance(value, Tensor)]
    if not devices:
        return torch.device("cpu")
    first = devices[0]
    for device in devices[1:]:
        if device != first:
            raise ValueError("batched trajectory helpers do not accept tensor inputs on multiple devices")
    return first


def _coerce_tensor(value, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64, device=device)


def _command_is_standstill(cmd: Tensor, eps: float = _STANDSTILL_CMD_EPS) -> Tensor:
    return torch.all(torch.abs(cmd) <= float(eps), dim=-1)


def _iter_replan_commands(command: Tensor, cfg: BatchedTrajectoryConfig) -> list[Tensor]:
    out = [command]
    for scale in cfg.replan_velocity_scales[1:]:
        base = command * float(scale)
        out.append(base)
        for bias in cfg.replan_yaw_biases[1:]:
            candidate = base.clone()
            candidate[:, 2] = candidate[:, 2] + float(bias)
            out.append(candidate)
        for bias in cfg.replan_vy_biases[1:]:
            candidate = base.clone()
            candidate[:, 1] = candidate[:, 1] + float(bias)
            out.append(candidate)
    return out


def _world_to_root_frame(root_pos: Tensor, root_quat: Tensor, points_w: Tensor) -> Tensor:
    delta = points_w - root_pos.unsqueeze(-2)
    w, x, y, z = root_quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    rot = torch.stack([row0, row1, row2], dim=-2)
    return torch.einsum("ntji,ntmi->ntmj", rot, delta)


def _standstill_trajectory(initial_state: BatchedRobotState, n_frames: int, dt: float) -> BatchedTrajectoryResult:
    batch_size = int(initial_state.root_pos.shape[0])
    root_pos = initial_state.root_pos[:, None, :].expand(batch_size, n_frames, 3).clone()
    root_quat = initial_state.root_quat[:, None, :].expand(batch_size, n_frames, 4).clone()
    foot_targets = initial_state.foot_pos[:, None, :, :].expand(batch_size, n_frames, 4, 3).clone()
    contact_seq = torch.ones((batch_size, n_frames, 4), dtype=torch.float32, device=root_pos.device)

    root_pos_flat = root_pos.reshape(batch_size * n_frames, 3)
    root_quat_flat = root_quat.reshape(batch_size * n_frames, 4)
    foot_targets_flat = foot_targets.reshape(batch_size * n_frames, 4, 3)
    joint_angles = batch_inverse_kinematics(root_pos_flat, root_quat_flat, foot_targets_flat).reshape(batch_size, n_frames, 12)
    body_pos_w = batch_forward_kinematics(root_pos_flat, root_quat_flat, joint_angles.reshape(batch_size * n_frames, 12)).reshape(batch_size, n_frames, 12, 3)
    body_pos_root = _world_to_root_frame(root_pos, root_quat, body_pos_w)
    foot_pos_root = _world_to_root_frame(root_pos, root_quat, foot_targets)
    root_lin_vel = torch.zeros((batch_size, n_frames, 3), dtype=torch.float64, device=root_pos.device)
    root_ang_vel = torch.zeros((batch_size, n_frames, 3), dtype=torch.float64, device=root_pos.device)
    planned_td = initial_state.foot_pos.clone()
    return BatchedTrajectoryResult(
        num_frames=n_frames,
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        root_lin_vel_w=root_lin_vel,
        root_ang_vel_w=root_ang_vel,
        joint_angles=joint_angles,
        foot_pos_w=foot_targets,
        foot_pos_root=foot_pos_root,
        contact_state=contact_seq,
        body_pos_root=body_pos_root,
        planned_touchdown_w=planned_td,
    )


def batched_generate_trajectory(
    terrain,
    states: BatchedRobotState,
    commands,
    requested_n_frames: int,
    dt: float = 0.02,
    cfg: BatchedTrajectoryConfig | None = None,
) -> BatchedTrajectoryResult:
    cfg = cfg or BatchedTrajectoryConfig()
    device = _resolve_input_device(states.root_pos, states.root_quat, states.joint_angles, states.foot_pos, commands)
    commands_t = _coerce_tensor(commands, device=device)
    if commands_t.ndim != 2 or commands_t.shape[1] != 3:
        raise ValueError(f"commands must have shape (N, 3); got {tuple(commands_t.shape)}")

    batch_size = int(states.root_pos.shape[0])
    if commands_t.shape[0] != batch_size:
        raise ValueError("states and commands must share batch size")

    if batch_size > 1:
        parts = []
        for idx in range(batch_size):
            part_state = BatchedRobotState(
                root_pos=states.root_pos[idx : idx + 1],
                root_quat=states.root_quat[idx : idx + 1],
                joint_angles=states.joint_angles[idx : idx + 1],
                foot_pos=states.foot_pos[idx : idx + 1],
                foot_vel=states.foot_vel[idx : idx + 1] if states.foot_vel is not None else None,
            )
            parts.append(
                batched_generate_trajectory(
                    terrain,
                    part_state,
                    commands_t[idx : idx + 1],
                    requested_n_frames=requested_n_frames,
                    dt=dt,
                    cfg=cfg,
                )
            )
        num_frames = parts[0].num_frames
        return BatchedTrajectoryResult(
            num_frames=num_frames,
            root_pos_w=torch.cat([part.root_pos_w for part in parts], dim=0),
            root_quat_w=torch.cat([part.root_quat_w for part in parts], dim=0),
            root_lin_vel_w=torch.cat([part.root_lin_vel_w for part in parts], dim=0),
            root_ang_vel_w=torch.cat([part.root_ang_vel_w for part in parts], dim=0),
            joint_angles=torch.cat([part.joint_angles for part in parts], dim=0),
            foot_pos_w=torch.cat([part.foot_pos_w for part in parts], dim=0),
            foot_pos_root=torch.cat([part.foot_pos_root for part in parts], dim=0),
            contact_state=torch.cat([part.contact_state for part in parts], dim=0),
            body_pos_root=torch.cat([part.body_pos_root for part in parts], dim=0),
            planned_touchdown_w=torch.cat([part.planned_touchdown_w for part in parts], dim=0),
        )

    requested_n_frames = int(requested_n_frames)
    cycle_frames = max(1, round(1.0 / (cfg.step_freq * dt)))
    n_frames = min(requested_n_frames, cycle_frames)

    if torch.all(_command_is_standstill(commands_t)):
        return _standstill_trajectory(states, requested_n_frames, dt)
    if torch.all(torch.linalg.norm(commands_t, dim=-1) < float(cfg.replan_stop_speed)):
        return _standstill_trajectory(states, requested_n_frames, dt)

    phase_offsets = torch.as_tensor(GAIT_PARAMS[cfg.gait_name]["offsets"], dtype=torch.float64, device=device)
    contact_seq = batched_gait_schedule(
        torch.zeros(batch_size, dtype=torch.float64, device=device),
        n_frames,
        dt,
        torch.full((batch_size,), float(cfg.step_freq), dtype=torch.float64, device=device),
        torch.full((batch_size,), float(cfg.duty_factor), dtype=torch.float64, device=device),
        phase_offsets,
    )
    touchdown_times = batched_next_touchdown_times(
        torch.full((batch_size,), float(cfg.step_freq), dtype=torch.float64, device=device),
        phase_offsets,
    )
    st = batched_stance_time(
        torch.full((batch_size,), float(cfg.step_freq), dtype=torch.float64, device=device),
        torch.full((batch_size,), float(cfg.duty_factor), dtype=torch.float64, device=device),
    )
    initial_yaw = extract_yaw_batch(states.root_quat)
    touchdown_mask = batched_legs_requiring_touchdown(contact_seq)

    candidates = _iter_replan_commands(commands_t, cfg)
    best_plan: _CandidatePlan | None = None
    best_score = torch.full((batch_size,), float("inf"), dtype=torch.float64, device=device)
    for candidate_index, candidate in enumerate(candidates):
        touchdowns = batched_compute_footholds(
            base_pos=states.root_pos,
            base_yaw=initial_yaw,
            base_lin_vel_xy=candidate[:, :2],
            ref_lin_vel_xy=candidate[:, :2],
            hip_positions=torch.zeros((batch_size, 4, 3), dtype=torch.float64, device=device),
            stance_time=st,
            com_height=torch.full((batch_size,), float(cfg.hip_height), dtype=torch.float64, device=device),
            terrain=terrain,
            previous_footholds=states.foot_pos,
            touchdown_times=touchdown_times,
            yaw_rate=candidate[:, 2],
            search_radius=cfg.foothold_search_radius,
            search_step=cfg.foothold_search_step,
            max_step_down=cfg.max_foothold_step_down,
        )
        feasible, td_score, _ = batched_evaluate_touchdowns(
            touchdowns,
            states.foot_pos,
            contact_seq,
            touchdown_mask,
            terrain,
            states.foot_pos,
            max_reach=cfg.max_touchdown_xy_reach,
        )
        candidate_standstill = _command_is_standstill(candidate, eps=cfg.replan_stop_speed)
        total_score = batched_candidate_total_score(commands_t, candidate, td_score, torch.full((batch_size,), candidate_index, dtype=torch.int64, device=device))
        valid = feasible & ~candidate_standstill & (total_score < best_score)
        best_score = torch.where(valid, total_score, best_score)
        if best_plan is None:
            best_plan = _CandidatePlan(candidate.clone(), touchdowns.clone(), total_score.clone())
        else:
            best_plan = _CandidatePlan(
                command=torch.where(valid[:, None], candidate, best_plan.command),
                touchdowns=torch.where(valid[:, None, None], touchdowns, best_plan.touchdowns),
                score=torch.where(valid, total_score, best_plan.score),
            )

    if best_plan is None or torch.all(~torch.isfinite(best_score)):
        return _standstill_trajectory(states, requested_n_frames, dt)

    touchdowns = best_plan.touchdowns
    terrain_max_heights = terrain.max_height_along_segment(states.foot_pos[..., :2].reshape(-1, 2), touchdowns[..., :2].reshape(-1, 2)).reshape(batch_size, 4)
    foot_targets = batched_compute_swing_targets(contact_seq, states.foot_pos, touchdowns, cfg.step_height, terrain_max_heights=terrain_max_heights)

    pos_xy_approx, yaw_approx = batched_integrate_base_planar(states.root_pos[:, :2], initial_yaw, best_plan.command[:, 0], best_plan.command[:, 1], best_plan.command[:, 2], n_frames, dt)
    base_pos_approx = torch.cat([pos_xy_approx, states.root_pos[:, 2:3].unsqueeze(1).expand(-1, n_frames, -1)], dim=-1)
    initial_roll, initial_pitch = extract_roll_pitch_batch(states.root_quat)
    initial_height = states.foot_pos[..., 2].mean(dim=-1)
    roll, pitch, height = batched_estimate_terrain(
        foot_targets,
        base_pos_approx,
        yaw_approx,
        alpha=0.05,
        initial_roll=initial_roll,
        initial_pitch=initial_pitch,
        initial_height=initial_height,
    )
    root_pos, root_quat = batched_solve_base_trajectory(
        states.root_pos,
        initial_yaw,
        best_plan.command[:, 0],
        best_plan.command[:, 1],
        best_plan.command[:, 2],
        n_frames,
        dt,
        terrain,
        foot_targets,
        contact_seq,
        roll,
        pitch,
        height,
        hip_height=cfg.hip_height,
        body_clearance_margin=cfg.body_clearance_margin,
    )

    root_pos_flat = root_pos.reshape(batch_size * n_frames, 3)
    root_quat_flat = root_quat.reshape(batch_size * n_frames, 4)
    foot_targets_flat = foot_targets.reshape(batch_size * n_frames, 4, 3)
    joint_angles = batch_inverse_kinematics(root_pos_flat, root_quat_flat, foot_targets_flat).reshape(batch_size, n_frames, 12)
    body_pos_w = batch_forward_kinematics(root_pos_flat, root_quat_flat, joint_angles.reshape(batch_size * n_frames, 12)).reshape(batch_size, n_frames, 12, 3)
    body_pos_root = _world_to_root_frame(root_pos, root_quat, body_pos_w)
    foot_pos_root = _world_to_root_frame(root_pos, root_quat, foot_targets)
    root_lin_vel = torch.diff(root_pos, dim=1, prepend=root_pos[:, :1]) / float(dt)
    roll_rate = torch.diff(roll, dim=1, prepend=roll[:, :1]) / float(dt)
    pitch_rate = torch.diff(pitch, dim=1, prepend=pitch[:, :1]) / float(dt)
    root_ang_vel = torch.zeros((batch_size, n_frames, 3), dtype=torch.float64, device=device)
    root_ang_vel[..., 0] = roll_rate
    root_ang_vel[..., 1] = pitch_rate
    root_ang_vel[..., 2] = best_plan.command[:, 2].unsqueeze(1)

    return BatchedTrajectoryResult(
        num_frames=n_frames,
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        root_lin_vel_w=root_lin_vel,
        root_ang_vel_w=root_ang_vel,
        joint_angles=joint_angles,
        foot_pos_w=foot_targets,
        foot_pos_root=foot_pos_root,
        contact_state=contact_seq,
        body_pos_root=body_pos_root,
        planned_touchdown_w=touchdowns,
    )


__all__ = ["batched_generate_trajectory"]
