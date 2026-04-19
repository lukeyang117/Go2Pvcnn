"""Batched trajectory generation main entry."""

from __future__ import annotations

import torch
from torch import Tensor

from ..convention import extract_yaw_batch, yaw_rotation_matrix_batch
from .base_solver import batched_integrate_base_planar, batched_solve_base_trajectory
from .config import BatchedTrajectoryConfig
from .foothold import batched_compute_footholds, batched_evaluate_touchdowns
from .gait import GAIT_PARAMS, batched_gait_schedule, batched_legs_requiring_touchdown, batched_next_touchdown_times, batched_stance_time
from .ik import batch_forward_kinematics, batch_inverse_kinematics
from .instrumentation import PlannerInstrumentation
from .swing import batched_compute_swing_targets
from .terrain_estimator import batched_estimate_terrain, batched_support_roll_pitch
from .types import BatchedRobotState, BatchedTrajectoryResult, HIP_OFFSETS_ARRAY

_STANDSTILL_CMD_EPS = 1e-5
_NOOP_PLANNER_INSTRUMENTATION = PlannerInstrumentation(enabled=False)


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


def _compute_hip_positions(base_pos: Tensor, base_yaw: Tensor) -> Tensor:
    base_pos_t = torch.as_tensor(base_pos, dtype=torch.float64)
    base_yaw_t = torch.as_tensor(base_yaw, dtype=torch.float64)
    if base_pos_t.ndim != 2 or base_pos_t.shape[-1] != 3:
        raise ValueError(f"base_pos must have shape (N, 3); got {tuple(base_pos_t.shape)}")
    if base_yaw_t.ndim != 1 or base_yaw_t.shape[0] != base_pos_t.shape[0]:
        raise ValueError("base_pos and base_yaw must share batch size")
    rot = yaw_rotation_matrix_batch(base_yaw_t).to(dtype=torch.float64)
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=base_pos_t.device, dtype=torch.float64)
    return base_pos_t[:, None, :] + torch.einsum("nij,kj->nki", rot, hip_offsets)


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


def _mix_trajectory_results(
    motion: BatchedTrajectoryResult,
    standstill: BatchedTrajectoryResult,
    standstill_mask: Tensor,
) -> BatchedTrajectoryResult:
    mask3 = standstill_mask[:, None, None]
    mask4 = standstill_mask[:, None, None, None]
    return BatchedTrajectoryResult(
        num_frames=motion.num_frames,
        root_pos_w=torch.where(mask3, standstill.root_pos_w, motion.root_pos_w),
        root_quat_w=torch.where(mask3, standstill.root_quat_w, motion.root_quat_w),
        root_lin_vel_w=torch.where(mask3, standstill.root_lin_vel_w, motion.root_lin_vel_w),
        root_ang_vel_w=torch.where(mask3, standstill.root_ang_vel_w, motion.root_ang_vel_w),
        joint_angles=torch.where(mask3, standstill.joint_angles, motion.joint_angles),
        foot_pos_w=torch.where(mask4, standstill.foot_pos_w, motion.foot_pos_w),
        foot_pos_root=torch.where(mask4, standstill.foot_pos_root, motion.foot_pos_root),
        contact_state=torch.where(mask3, standstill.contact_state, motion.contact_state),
        body_pos_root=torch.where(mask4, standstill.body_pos_root, motion.body_pos_root),
        planned_touchdown_w=torch.where(mask3, standstill.planned_touchdown_w, motion.planned_touchdown_w),
    )


def _resolve_stage_diagnostics_handler(stage_diagnostics):
    if stage_diagnostics is None:
        return None
    if callable(stage_diagnostics):
        return stage_diagnostics
    capture = getattr(stage_diagnostics, "capture", None)
    if callable(capture):
        return capture
    raise TypeError("stage_diagnostics must be callable or expose a callable 'capture' attribute")


def _clone_stage_value(value):
    if isinstance(value, Tensor):
        return value.detach().clone()
    return value


def _emit_stage_diagnostics(stage_diagnostics, name: str, **payload) -> None:
    handler = _resolve_stage_diagnostics_handler(stage_diagnostics)
    if handler is None:
        return
    handler(str(name), {key: _clone_stage_value(value) for key, value in payload.items()})


def batched_generate_trajectory(
    terrain,
    states: BatchedRobotState,
    commands,
    requested_n_frames: int,
    dt: float = 0.02,
    cfg: BatchedTrajectoryConfig | None = None,
    instrumentation: PlannerInstrumentation | None = None,
    stage_diagnostics=None,
) -> BatchedTrajectoryResult:
    cfg = cfg or BatchedTrajectoryConfig()
    instr = instrumentation if instrumentation is not None else _NOOP_PLANNER_INSTRUMENTATION
    device = _resolve_input_device(states.root_pos, states.root_quat, states.joint_angles, states.foot_pos, commands)
    batch_size = int(states.root_pos.shape[0])
    with instr.stage("input"):
        commands_t = _coerce_tensor(commands, device=device)
        if commands_t.ndim != 2 or commands_t.shape[1] != 3:
            raise ValueError(f"commands must have shape (N, 3); got {tuple(commands_t.shape)}")
        _emit_stage_diagnostics(
            stage_diagnostics,
            "input",
            commands=commands_t,
            root_pos=states.root_pos,
            root_quat=states.root_quat,
            foot_pos=states.foot_pos,
            joint_angles=states.joint_angles,
        )

    if commands_t.shape[0] != batch_size:
        raise ValueError("states and commands must share batch size")

    requested_n_frames = int(requested_n_frames)
    cycle_frames = max(1, round(1.0 / (cfg.step_freq * dt)))
    n_frames = min(requested_n_frames, cycle_frames)
    # Align with raw semantics: componentwise thresholding (not vector norm).
    standstill_mask = _command_is_standstill(commands_t) | _command_is_standstill(commands_t, eps=cfg.replan_stop_speed)

    if torch.all(standstill_mask):
        with instr.stage("standstill"):
            result = _standstill_trajectory(states, n_frames, dt)
            _emit_stage_diagnostics(
                stage_diagnostics,
                "standstill",
                root_pos_w=result.root_pos_w,
                root_quat_w=result.root_quat_w,
                joint_angles=result.joint_angles,
                foot_pos_w=result.foot_pos_w,
                contact_state=result.contact_state,
                planned_touchdown_w=result.planned_touchdown_w,
                standstill_mask=standstill_mask,
            )
        _emit_stage_diagnostics(
            stage_diagnostics,
            "result",
            root_pos_w=result.root_pos_w,
            root_quat_w=result.root_quat_w,
            joint_angles=result.joint_angles,
            foot_pos_w=result.foot_pos_w,
            contact_state=result.contact_state,
            planned_touchdown_w=result.planned_touchdown_w,
        )
        return result

    with instr.stage("gait"):
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
        _emit_stage_diagnostics(
            stage_diagnostics,
            "gait",
            contact_seq=contact_seq,
            touchdown_times=touchdown_times,
            stance_time=st,
        )
    initial_yaw = extract_yaw_batch(states.root_quat)
    hip_positions = _compute_hip_positions(states.root_pos, initial_yaw)
    touchdown_mask = batched_legs_requiring_touchdown(contact_seq)

    standstill_command_mask = _command_is_standstill(commands_t) | _command_is_standstill(commands_t, eps=cfg.replan_stop_speed)
    if torch.all(standstill_command_mask):
        with instr.stage("standstill"):
            result = _standstill_trajectory(states, n_frames, dt)
            _emit_stage_diagnostics(
                stage_diagnostics,
                "standstill",
                root_pos_w=result.root_pos_w,
                root_quat_w=result.root_quat_w,
                joint_angles=result.joint_angles,
                foot_pos_w=result.foot_pos_w,
                contact_state=result.contact_state,
                planned_touchdown_w=result.planned_touchdown_w,
                standstill_mask=standstill_command_mask,
            )
        _emit_stage_diagnostics(
            stage_diagnostics,
            "result",
            root_pos_w=result.root_pos_w,
            root_quat_w=result.root_quat_w,
            joint_angles=result.joint_angles,
            foot_pos_w=result.foot_pos_w,
            contact_state=result.contact_state,
            planned_touchdown_w=result.planned_touchdown_w,
        )
        return result

    with instr.stage("footholds"):
        touchdowns = batched_compute_footholds(
            base_pos=states.root_pos,
            base_yaw=initial_yaw,
            base_lin_vel_xy=commands_t[:, :2],
            ref_lin_vel_xy=commands_t[:, :2],
            hip_positions=hip_positions,
            stance_time=st,
            com_height=torch.full((batch_size,), float(cfg.hip_height), dtype=torch.float64, device=device),
            terrain=terrain,
            previous_footholds=states.foot_pos,
            touchdown_times=touchdown_times,
            yaw_rate=commands_t[:, 2],
            search_radius=cfg.foothold_search_radius,
            search_step=cfg.foothold_search_step,
            max_step_down=cfg.max_foothold_step_down,
        )
        _emit_stage_diagnostics(
            stage_diagnostics,
            "footholds",
            hip_positions=hip_positions,
            touchdowns=touchdowns,
            touchdown_mask=touchdown_mask,
        )
    with instr.stage("touchdown_eval"):
        feasible, _, _ = batched_evaluate_touchdowns(
            touchdowns,
            states.foot_pos,
            contact_seq,
            touchdown_mask,
            terrain,
            states.foot_pos,
            max_reach=cfg.max_touchdown_xy_reach,
        )
        _emit_stage_diagnostics(
            stage_diagnostics,
            "touchdown_eval",
            feasible=feasible,
            touchdown_mask=touchdown_mask,
            touchdowns=touchdowns,
        )

    standstill_mask = standstill_command_mask | ~feasible
    if torch.all(standstill_mask):
        with instr.stage("standstill"):
            result = _standstill_trajectory(states, n_frames, dt)
            _emit_stage_diagnostics(
                stage_diagnostics,
                "standstill",
                root_pos_w=result.root_pos_w,
                root_quat_w=result.root_quat_w,
                joint_angles=result.joint_angles,
                foot_pos_w=result.foot_pos_w,
                contact_state=result.contact_state,
                planned_touchdown_w=result.planned_touchdown_w,
                standstill_mask=standstill_mask,
            )
        _emit_stage_diagnostics(
            stage_diagnostics,
            "result",
            root_pos_w=result.root_pos_w,
            root_quat_w=result.root_quat_w,
            joint_angles=result.joint_angles,
            foot_pos_w=result.foot_pos_w,
            contact_state=result.contact_state,
            planned_touchdown_w=result.planned_touchdown_w,
        )
        return result
    terrain_max_heights = terrain.batch_max_height_along_segment(
        states.foot_pos[:, :, :2], touchdowns[:, :, :2],
    )
    with instr.stage("swing_targets"):
        foot_targets = batched_compute_swing_targets(
            contact_seq,
            states.foot_pos,
            touchdowns,
            cfg.step_height,
            terrain_max_heights=terrain_max_heights,
        )
        _emit_stage_diagnostics(
            stage_diagnostics,
            "swing_targets",
            foot_targets=foot_targets,
            terrain_max_heights=terrain_max_heights,
        )

    with instr.stage("base_approx"):
        pos_xy_approx, yaw_approx = batched_integrate_base_planar(
            states.root_pos[:, :2],
            initial_yaw,
            commands_t[:, 0],
            commands_t[:, 1],
            commands_t[:, 2],
            n_frames,
            dt,
        )
    base_pos_approx = torch.cat([pos_xy_approx, states.root_pos[:, 2:3].unsqueeze(1).expand(-1, n_frames, -1)], dim=-1)
    _emit_stage_diagnostics(
        stage_diagnostics,
        "base_approx",
        base_pos_approx=base_pos_approx,
        yaw_approx=yaw_approx,
    )
    initial_roll, initial_pitch = batched_support_roll_pitch(states.foot_pos)
    initial_height = states.foot_pos[..., 2].mean(dim=-1)
    with instr.stage("terrain_est"):
        roll, pitch, height = batched_estimate_terrain(
            foot_targets,
            base_pos_approx,
            yaw_approx,
            alpha=0.05,
            initial_roll=initial_roll,
            initial_pitch=initial_pitch,
            initial_height=initial_height,
            contact_state=contact_seq if cfg.use_support_contact_terrain_estimator else None,
        )
        _emit_stage_diagnostics(
            stage_diagnostics,
            "terrain_est",
            roll=roll,
            pitch=pitch,
            height=height,
        )
    with instr.stage("base_solve"):
        root_pos, root_quat = batched_solve_base_trajectory(
            states.root_pos,
            initial_yaw,
            commands_t[:, 0],
            commands_t[:, 1],
            commands_t[:, 2],
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
        _emit_stage_diagnostics(
            stage_diagnostics,
            "base_solve",
            root_pos_w=root_pos,
            root_quat_w=root_quat,
        )

    root_pos_flat = root_pos.reshape(batch_size * n_frames, 3)
    root_quat_flat = root_quat.reshape(batch_size * n_frames, 4)
    foot_targets_flat = foot_targets.reshape(batch_size * n_frames, 4, 3)
    with instr.stage("ik"):
        joint_angles = batch_inverse_kinematics(root_pos_flat, root_quat_flat, foot_targets_flat).reshape(batch_size, n_frames, 12)
        _emit_stage_diagnostics(
            stage_diagnostics,
            "ik",
            joint_angles=joint_angles,
        )
    with instr.stage("fk"):
        body_pos_w = batch_forward_kinematics(
            root_pos_flat,
            root_quat_flat,
            joint_angles.reshape(batch_size * n_frames, 12),
        ).reshape(batch_size, n_frames, 12, 3)
        _emit_stage_diagnostics(
            stage_diagnostics,
            "fk",
            body_pos_w=body_pos_w,
        )
    body_pos_root = _world_to_root_frame(root_pos, root_quat, body_pos_w)
    foot_pos_root = _world_to_root_frame(root_pos, root_quat, foot_targets)
    root_lin_vel = torch.diff(root_pos, dim=1, prepend=root_pos[:, :1]) / float(dt)
    roll_rate = torch.diff(roll, dim=1, prepend=roll[:, :1]) / float(dt)
    pitch_rate = torch.diff(pitch, dim=1, prepend=pitch[:, :1]) / float(dt)
    root_ang_vel = torch.zeros((batch_size, n_frames, 3), dtype=torch.float64, device=device)
    root_ang_vel[..., 0] = roll_rate
    root_ang_vel[..., 1] = pitch_rate
    root_ang_vel[..., 2] = commands_t[:, 2].unsqueeze(1)
    motion_result = BatchedTrajectoryResult(
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

    if torch.any(standstill_mask):
        with instr.stage("mix"):
            standstill_result = _standstill_trajectory(states, n_frames, dt)
            _emit_stage_diagnostics(
                stage_diagnostics,
                "standstill",
                root_pos_w=standstill_result.root_pos_w,
                root_quat_w=standstill_result.root_quat_w,
                joint_angles=standstill_result.joint_angles,
                foot_pos_w=standstill_result.foot_pos_w,
                contact_state=standstill_result.contact_state,
                planned_touchdown_w=standstill_result.planned_touchdown_w,
                standstill_mask=standstill_mask,
            )
            mixed_result = _mix_trajectory_results(motion_result, standstill_result, standstill_mask)
            _emit_stage_diagnostics(
                stage_diagnostics,
                "mix",
                standstill_mask=standstill_mask,
                root_pos_w=mixed_result.root_pos_w,
                root_quat_w=mixed_result.root_quat_w,
                joint_angles=mixed_result.joint_angles,
                foot_pos_w=mixed_result.foot_pos_w,
                contact_state=mixed_result.contact_state,
                planned_touchdown_w=mixed_result.planned_touchdown_w,
            )
        _emit_stage_diagnostics(
            stage_diagnostics,
            "result",
            root_pos_w=mixed_result.root_pos_w,
            root_quat_w=mixed_result.root_quat_w,
            joint_angles=mixed_result.joint_angles,
            foot_pos_w=mixed_result.foot_pos_w,
            contact_state=mixed_result.contact_state,
            planned_touchdown_w=mixed_result.planned_touchdown_w,
        )
        return mixed_result

    _emit_stage_diagnostics(
        stage_diagnostics,
        "result",
        root_pos_w=motion_result.root_pos_w,
        root_quat_w=motion_result.root_quat_w,
        joint_angles=motion_result.joint_angles,
        foot_pos_w=motion_result.foot_pos_w,
        contact_state=motion_result.contact_state,
        planned_touchdown_w=motion_result.planned_touchdown_w,
    )
    return motion_result


__all__ = ["batched_generate_trajectory"]
