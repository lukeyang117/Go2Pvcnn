from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT, GO2PVCNN_ROOT / "tests"):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture, refresh_targeted_scanner_pose  # noqa: E402
from mpc_low_small_reachable_crossing_probe import _parse_command, _result_metrics  # noqa: E402
from mpc_qp_viewer_crossing_probe import _playback_readback_error  # noqa: E402


DEFAULT_COMMANDS: tuple[str, ...] = (
    "forward_v050:0.50,0.00,0.00",
    "diag_v050:0.35,0.20,0.00",
)

DEFAULT_SEQUENCE = "move_v050:0.50,0.00,0.00x2;stop:0.00,0.00,0.00x4"


def _parse_command_list(value: str) -> tuple[str, ...]:
    text = str(value)
    if ";" in text:
        return tuple(item.strip() for item in text.split(";") if item.strip())
    if ":" in text:
        return (text.strip(),) if text.strip() else ()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _parse_offsets(value: str) -> tuple[tuple[float, float], ...]:
    offsets: list[tuple[float, float]] = []
    for item in str(value).replace(";", ",").split(","):
        token = item.strip()
        if not token:
            continue
        if ":" in token:
            x_s, y_s = token.split(":", 1)
        elif "/" in token:
            x_s, y_s = token.split("/", 1)
        else:
            parts = token.split()
            if len(parts) != 2:
                raise ValueError(f"Offset must look like x:y, got {token!r}")
            x_s, y_s = parts
        offsets.append((float(x_s), float(y_s)))
    return tuple(offsets)


def _parse_sequence(value: str) -> tuple[tuple[str, tuple[float, float, float], int], ...]:
    entries: list[tuple[str, tuple[float, float, float], int]] = []
    for item in str(value).split(";"):
        token = item.strip()
        if not token:
            continue
        count = 1
        if "x" in token.rsplit(":", 1)[-1]:
            prefix, count_s = token.rsplit("x", 1)
            token = prefix.strip()
            count = int(count_s.strip())
        name, command = _parse_command(token)
        entries.append((name, command, max(0, int(count))))
    return tuple(entries)


def _terrain_height_summary(terrain) -> dict[str, float]:
    height = torch.as_tensor(terrain.height_map[0], dtype=torch.float64)
    finite = torch.isfinite(height)
    if not bool(finite.any().item()):
        return {"terrain_height_min_m": float("nan"), "terrain_height_max_m": float("nan"), "terrain_height_range_m": float("nan")}
    hmin = float(height[finite].amin().item())
    hmax = float(height[finite].amax().item())
    return {
        "terrain_height_min_m": hmin,
        "terrain_height_max_m": hmax,
        "terrain_height_range_m": hmax - hmin,
    }


def _loss_breakdown_summary(result, *, top_k: int = 12) -> dict[str, object]:
    loss_breakdown = getattr(result, "loss_breakdown", None)
    if not loss_breakdown:
        return {"loss_breakdown_top": [], "loss_breakdown": {}}
    scalars: dict[str, float] = {}
    for name, value in loss_breakdown.items():
        tensor = torch.as_tensor(value)
        if int(tensor.numel()) == 0:
            scalars[str(name)] = 0.0
        else:
            scalars[str(name)] = float(tensor.detach().reshape(-1).abs().amax().item())
    ordered = sorted(scalars.items(), key=lambda item: item[1], reverse=True)
    return {
        "loss_breakdown_top": [{"name": name, "abs_max": value} for name, value in ordered[: max(0, int(top_k))]],
        "loss_breakdown": scalars,
    }


def _planned_fk_error_detail(result) -> dict[str, object]:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
    from mpc_low_small_reachable_crossing_probe import _root_rpy_from_viewer_result

    root = torch.as_tensor(result.root_pos_w, dtype=torch.float32)
    rpy = _root_rpy_from_viewer_result(result).to(dtype=root.dtype, device=root.device)
    planned = torch.as_tensor(result.foot_pos_w, dtype=root.dtype, device=root.device)
    joint = torch.as_tensor(result.joint_angles, dtype=root.dtype, device=root.device)
    fk = fk_feet_from_joint_angles(root, rpy, joint)
    error = torch.linalg.vector_norm(planned - fk, dim=-1)
    flat_idx = int(torch.argmax(error.reshape(-1)).item()) if int(error.numel()) else 0
    env_idx, frame_idx, leg_idx = torch.unravel_index(torch.tensor(flat_idx, device=error.device), error.shape)
    env_i = int(env_idx.item())
    frame_i = int(frame_idx.item())
    leg_i = int(leg_idx.item())
    return {
        "planned_fk_worst_env": env_i,
        "planned_fk_worst_frame": frame_i,
        "planned_fk_worst_leg": leg_i,
        "planned_fk_worst_error_m": float(error[env_i, frame_i, leg_i].item()),
        "planned_fk_frame0_error_max_m": float(error[:, 0].max().item()) if error.shape[1] > 0 else 0.0,
        "planned_fk_after_frame0_error_max_m": float(error[:, 1:].max().item()) if error.shape[1] > 1 else 0.0,
        "planned_fk_worst_planned_xyz": [
            float(v) for v in planned[env_i, frame_i, leg_i].detach().cpu().reshape(-1).tolist()
        ],
        "planned_fk_worst_fk_xyz": [
            float(v) for v in fk[env_i, frame_i, leg_i].detach().cpu().reshape(-1).tolist()
        ],
    }


def _touchdown_alignment_detail(result, *, current_foot_w: torch.Tensor | None = None) -> dict[str, object]:
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32)
    touchdown = torch.as_tensor(getattr(result, "planned_touchdown_w", foot), dtype=foot.dtype, device=foot.device)
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
    if touchdown.ndim == 4:
        td = touchdown[:, 0]
    elif touchdown.ndim == 3:
        td = touchdown
    else:
        return {}
    td_seq = td[:, None, :, :].expand_as(foot)
    dist = torch.linalg.vector_norm(foot - td_seq, dim=-1)
    closest = dist.min(dim=1)
    closest_dist = closest.values
    closest_frame = closest.indices
    contact_dist = []
    contact_frames = []
    for leg_idx in range(int(foot.shape[2])):
        leg_contact = contact[0, :, leg_idx]
        rising = torch.logical_and(leg_contact[1:], torch.logical_not(leg_contact[:-1]))
        candidates = torch.nonzero(rising, as_tuple=False).reshape(-1) + 1
        if int(candidates.numel()) == 0:
            candidates = torch.nonzero(leg_contact, as_tuple=False).reshape(-1)
        if int(candidates.numel()) == 0:
            frame = int(closest_frame[0, leg_idx].item())
        else:
            frame = int(candidates[0].item())
        contact_frames.append(frame)
        contact_dist.append(float(dist[0, frame, leg_idx].item()))
    frame0 = torch.linalg.vector_norm(foot[:, 0] - td, dim=-1)
    terminal = torch.linalg.vector_norm(foot[:, -1] - td, dim=-1)
    current_err = None
    if current_foot_w is not None:
        current = torch.as_tensor(current_foot_w, dtype=foot.dtype, device=foot.device)
        if current.ndim == 3:
            current_err = torch.linalg.vector_norm(current - td, dim=-1)
    out: dict[str, object] = {
        "touchdown_to_frame0_foot_error_max_m": float(frame0.max().item()),
        "touchdown_to_frame0_foot_error_mean_m": float(frame0.mean().item()),
        "touchdown_to_terminal_foot_error_max_m": float(terminal.max().item()),
        "touchdown_to_terminal_foot_error_mean_m": float(terminal.mean().item()),
        "touchdown_to_closest_foot_error_max_m": float(closest_dist.max().item()),
        "touchdown_to_closest_foot_error_mean_m": float(closest_dist.mean().item()),
        "touchdown_closest_frame_by_leg": [int(v) for v in closest_frame[0].detach().cpu().tolist()],
        "touchdown_to_contact_frame_foot_error_max_m": float(max(contact_dist)) if contact_dist else 0.0,
        "touchdown_to_contact_frame_foot_error_mean_m": float(sum(contact_dist) / max(1, len(contact_dist))),
        "touchdown_contact_frame_by_leg": contact_frames,
    }
    if current_err is not None:
        out["touchdown_to_current_actual_foot_error_max_m"] = float(current_err.max().item())
        out["touchdown_to_current_actual_foot_error_mean_m"] = float(current_err.mean().item())
    return out


def _heightfield_collision_detail(result, terrain) -> dict[str, object]:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles, fk_leg_points_from_joint_angles
    from extension.batch_mpc_planner.terrain import height_at
    from mpc_low_small_reachable_crossing_probe import _root_rpy_from_viewer_result

    root = torch.as_tensor(result.root_pos_w, dtype=torch.float32)
    device = root.device
    dtype = root.dtype
    rpy = _root_rpy_from_viewer_result(result).to(dtype=dtype, device=device)
    planned = torch.as_tensor(result.foot_pos_w, dtype=dtype, device=device)
    joint = torch.as_tensor(result.joint_angles, dtype=dtype, device=device)
    fk_foot = fk_feet_from_joint_angles(root, rpy, joint)
    fk_points = fk_leg_points_from_joint_angles(root, rpy, joint, shank_sample_count=2)

    def _clearance(points: torch.Tensor) -> torch.Tensor:
        return points[..., 2] - height_at(terrain, points[..., :2]).to(dtype=dtype, device=device)

    planned_foot_clearance = _clearance(planned)
    fk_foot_clearance = _clearance(fk_foot)
    fk_knee_clearance = _clearance(fk_points.knee_pos_world)
    fk_shank_clearance = _clearance(fk_points.shank_sample_world)

    yaw = rpy[..., 2]
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    offsets = torch.as_tensor(
        ((0.22, 0.10), (0.22, -0.10), (-0.22, 0.10), (-0.22, -0.10), (0.0, 0.0)),
        dtype=dtype,
        device=device,
    )
    ox = offsets[:, 0].view(1, 1, -1)
    oy = offsets[:, 1].view(1, 1, -1)
    body_xy = torch.stack((cy * ox - sy * oy, sy * ox + cy * oy), dim=-1) + root[..., None, :2]
    body_z = root[..., None, 2] - 0.18
    body_points = torch.cat((body_xy, body_z.unsqueeze(-1).expand_as(body_xy[..., :1])), dim=-1)
    body_clearance = _clearance(body_points)
    root_step = torch.linalg.vector_norm(root[:, 1:, :2] - root[:, :-1, :2], dim=-1) if int(root.shape[1]) > 1 else torch.zeros((root.shape[0], 0), dtype=dtype, device=device)
    root_z_step = torch.abs(root[:, 1:, 2] - root[:, :-1, 2]) if int(root.shape[1]) > 1 else torch.zeros((root.shape[0], 0), dtype=dtype, device=device)
    return {
        "planned_foot_ground_clearance_min_m": float(planned_foot_clearance.min().item()),
        "planned_foot_ground_penetration_count": int((planned_foot_clearance < 0.0).sum().item()),
        "fk_foot_ground_clearance_min_m": float(fk_foot_clearance.min().item()),
        "fk_foot_ground_penetration_count": int((fk_foot_clearance < 0.0).sum().item()),
        "fk_knee_ground_clearance_min_m": float(fk_knee_clearance.min().item()),
        "fk_knee_ground_penetration_count": int((fk_knee_clearance < 0.0).sum().item()),
        "fk_shank_ground_clearance_min_m": float(fk_shank_clearance.min().item()),
        "fk_shank_ground_penetration_count": int((fk_shank_clearance < 0.0).sum().item()),
        "body_ground_clearance_min_m": float(body_clearance.min().item()),
        "body_ground_penetration_count": int((body_clearance < 0.0).sum().item()),
        "root_step_xy_max_m": float(root_step.max().item()) if int(root_step.numel()) else 0.0,
        "root_step_z_max_m": float(root_z_step.max().item()) if int(root_z_step.numel()) else 0.0,
    }


def _plan_cycle(
    *,
    runtime: RealViewerRuntimeFixture,
    terrain_row: int,
    terrain_col: int,
    offset_xy_m: tuple[float, float],
    command_name: str,
    command_tuple: tuple[float, float, float],
    cycle: int,
    playback_frames: int,
) -> dict[str, object]:
    terrain = runtime._single_env_terrain()
    state = runtime._single_env_state()
    command = torch.tensor([command_tuple], dtype=torch.float64, device=runtime.base_env.device)
    result = runtime._viewer._plan_viewer_trajectory(
        terrain=terrain,
        state=state,
        command=command,
        mpc_cfg=runtime.mpc_planner_cfg,
    )
    internal_rpy = getattr(result, "root_rpy", None)
    viewer_rpy = None
    if internal_rpy is not None:
        from mpc_low_small_reachable_crossing_probe import _root_rpy_from_viewer_result

        viewer_rpy = _root_rpy_from_viewer_result(result).to(
            dtype=torch.as_tensor(internal_rpy).dtype,
            device=torch.as_tensor(internal_rpy).device,
        )
    selected = runtime.select_terrain_tile(terrain_row=int(terrain_row), terrain_col=int(terrain_col))
    selected_xy = torch.as_tensor(selected[:2], dtype=torch.float32, device=runtime.base_env.device)
    obstacle_xy = selected_xy + torch.tensor(offset_xy_m, dtype=torch.float32, device=runtime.base_env.device)

    row: dict[str, object] = {
        "type": "mpc_row8_col12_cycle",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "planner_backend": "mpc",
        "terrain_row": int(terrain_row),
        "terrain_col": int(terrain_col),
        "offset_x_m": float(offset_xy_m[0]),
        "offset_y_m": float(offset_xy_m[1]),
        "command": command_name,
        "command_vx": float(command_tuple[0]),
        "command_vy": float(command_tuple[1]),
        "command_wz": float(command_tuple[2]),
        "cycle": int(cycle),
        "horizon": int(result.root_pos_w.shape[1]),
        "playback_readback_error_max_m": _playback_readback_error(runtime, result, frames=int(playback_frames)),
    }
    if internal_rpy is not None and viewer_rpy is not None:
        internal = torch.as_tensor(internal_rpy)
        row["internal_vs_viewer_rpy_error_max_rad"] = float(torch.abs(internal - viewer_rpy).max().item())
        row["internal_roll_pitch_abs_max_rad"] = float(torch.abs(internal[..., :2]).max().item())
        row["viewer_roll_pitch_abs_max_rad"] = float(torch.abs(viewer_rpy[..., :2]).max().item())
    row.update(_terrain_height_summary(terrain))
    row.update(_result_metrics(result, terrain, obstacle_xy, command_tuple, obstacle_height=float(row["terrain_height_max_m"])))
    row.update(_planned_fk_error_detail(result))
    row.update(_touchdown_alignment_detail(result, current_foot_w=state.foot_pos))
    row.update(_heightfield_collision_detail(result, terrain))
    row.update(_loss_breakdown_summary(result))
    runtime._viewer._viewer_direct_playback_step(
        runtime.base_env,
        result,
        frame_idx=min(int(playback_frames) - 1, int(result.num_frames) - 1),
    )
    refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
    return row


def _run_sequence(
    *,
    runtime: RealViewerRuntimeFixture,
    terrain_row: int,
    terrain_col: int,
    offset_xy_m: tuple[float, float],
    sequence: tuple[tuple[str, tuple[float, float, float], int], ...],
    playback_frames: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seq_idx = 0
    for phase_name, command_tuple, count in sequence:
        for phase_cycle in range(int(count)):
            row = _plan_cycle(
                runtime=runtime,
                terrain_row=int(terrain_row),
                terrain_col=int(terrain_col),
                offset_xy_m=offset_xy_m,
                command_name=phase_name,
                command_tuple=command_tuple,
                cycle=seq_idx,
                playback_frames=int(playback_frames),
            )
            row["sequence_phase"] = phase_name
            row["sequence_phase_cycle"] = int(phase_cycle)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            seq_idx += 1
    return rows


def run_probe(
    *,
    device: str,
    terrain_row: int,
    terrain_col: int,
    offsets: tuple[tuple[float, float], ...],
    commands: tuple[str, ...],
    cycles: int,
    requested_n_frames: int,
    warmup_steps: int,
    playback_frames: int,
    ik_fk_weight_mult: float,
    kinematics_weight_mult: float,
    root_foot_center_weight_mult: float,
    fk_body_leg_collision_weight_mult: float,
    touchdown_endpoint_weight_mult: float,
    progress_weight_mult: float,
    swing_direction_weight_mult: float,
    optimize_steps: int | None,
    sequence: tuple[tuple[str, tuple[float, float, float], int], ...] | None,
) -> int:
    runtime = RealViewerRuntimeFixture(
        num_envs=1,
        device=device,
        planner_backend="mpc",
        requested_n_frames=requested_n_frames,
        warmup_steps=warmup_steps,
        task_id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
        env_cfg_entry_point=(
            "go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg:"
            "TeacherElevationTrajectoryMpcSemanticEnvCfg"
        ),
        compact_semantic_grid=False,
    )
    rows: list[dict[str, object]] = []
    try:
        runtime.mpc_planner_cfg.runtime.horizon_steps = int(requested_n_frames)
        runtime.mpc_planner_cfg.runtime.replan_interval_steps = int(requested_n_frames)
        runtime.mpc_planner_cfg.runtime.dt = 0.02
        if optimize_steps is not None:
            runtime.mpc_planner_cfg.runtime.optimize_steps = int(optimize_steps)
        runtime.mpc_planner_cfg.losses.ik_fk_residual.weight *= float(ik_fk_weight_mult)
        runtime.mpc_planner_cfg.losses.kinematics.weight *= float(kinematics_weight_mult)
        runtime.mpc_planner_cfg.losses.root_foot_center.weight *= float(root_foot_center_weight_mult)
        runtime.mpc_planner_cfg.losses.fk_body_leg_collision.weight *= float(fk_body_leg_collision_weight_mult)
        touchdown_endpoint_cfg = getattr(runtime.mpc_planner_cfg.losses, "touchdown_endpoint", None)
        if touchdown_endpoint_cfg is not None:
            touchdown_endpoint_cfg.weight *= float(touchdown_endpoint_weight_mult)
        runtime.mpc_planner_cfg.losses.progress.weight *= float(progress_weight_mult)
        runtime.mpc_planner_cfg.losses.swing_direction.weight *= float(swing_direction_weight_mult)
        print(
            json.dumps(
                {
                    "type": "mpc_row8_col12_header",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "device": device,
                    "planner_backend": "mpc",
                    "terrain_row": int(terrain_row),
                    "terrain_col": int(terrain_col),
                    "offsets": [list(offset) for offset in offsets],
                    "commands": list(commands),
                    "sequence": None
                    if sequence is None
                    else [{"name": name, "command": list(command), "cycles": count} for name, command, count in sequence],
                    "cycles": int(cycles),
                    "requested_n_frames": int(requested_n_frames),
                    "warmup_steps": int(warmup_steps),
                    "playback_frames": int(playback_frames),
                    "ik_fk_weight": float(runtime.mpc_planner_cfg.losses.ik_fk_residual.weight),
                    "kinematics_weight": float(runtime.mpc_planner_cfg.losses.kinematics.weight),
                    "root_foot_center_weight": float(runtime.mpc_planner_cfg.losses.root_foot_center.weight),
                    "fk_body_leg_collision_weight": float(runtime.mpc_planner_cfg.losses.fk_body_leg_collision.weight),
                    "touchdown_endpoint_weight": None
                    if touchdown_endpoint_cfg is None
                    else float(touchdown_endpoint_cfg.weight),
                    "progress_weight": float(runtime.mpc_planner_cfg.losses.progress.weight),
                    "swing_direction_weight": float(runtime.mpc_planner_cfg.losses.swing_direction.weight),
                    "optimize_steps": int(runtime.mpc_planner_cfg.runtime.optimize_steps),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for offset_xy in offsets:
            runtime.reset()
            runtime.move_env0_to_terrain_tile(
                terrain_row=int(terrain_row),
                terrain_col=int(terrain_col),
                z_clearance=0.85,
                offset_xy_m=offset_xy,
                ground_robot=True,
                root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=runtime.base_env.device),
            )
            if sequence is not None:
                rows.extend(
                    _run_sequence(
                        runtime=runtime,
                        terrain_row=int(terrain_row),
                        terrain_col=int(terrain_col),
                        offset_xy_m=offset_xy,
                        sequence=sequence,
                        playback_frames=int(playback_frames),
                    )
                )
                continue
            for command_text in commands:
                command_name, command_tuple = _parse_command(command_text)
                for cycle in range(int(cycles)):
                    row = _plan_cycle(
                        runtime=runtime,
                        terrain_row=int(terrain_row),
                        terrain_col=int(terrain_col),
                        offset_xy_m=offset_xy,
                        command_name=command_name,
                        command_tuple=command_tuple,
                        cycle=cycle,
                        playback_frames=int(playback_frames),
                    )
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
        if rows:
            summary = {
                "type": "mpc_row8_col12_summary",
                "cycle_count": int(len(rows)),
                "planner_backend": "mpc",
                "max_playback_readback_error_m": max(float(row["playback_readback_error_max_m"]) for row in rows),
                "max_terminal_planned_vs_fk_foot_error_m": max(
                    float(row["terminal_planned_vs_fk_foot_error_max"]) for row in rows
                ),
                "max_planned_vs_fk_foot_error_crossing_leg_m": max(
                    float(row.get("planned_vs_fk_foot_error_crossing_leg_max_m", 0.0) or 0.0) for row in rows
                ),
                "max_fk_semantic_collision_count": max(
                    int(row.get("fk_semantic_collision_count", 0) or 0) for row in rows
                ),
                "max_fk_touchdown_on_small_rate": max(float(row.get("fk_touchdown_on_small_rate", 0.0) or 0.0) for row in rows),
                "max_fk_stance_on_small_rate": max(float(row.get("fk_stance_on_small_rate", 0.0) or 0.0) for row in rows),
                "max_touchdown_to_frame0_foot_error_m": max(
                    float(row.get("touchdown_to_frame0_foot_error_max_m", 0.0) or 0.0) for row in rows
                ),
                "max_touchdown_to_contact_frame_foot_error_m": max(
                    float(row.get("touchdown_to_contact_frame_foot_error_max_m", 0.0) or 0.0) for row in rows
                ),
                "max_touchdown_to_current_actual_foot_error_m": max(
                    float(row.get("touchdown_to_current_actual_foot_error_max_m", 0.0) or 0.0) for row in rows
                ),
                "max_raw_ik_joint_limit_violation": max(
                    float(row.get("raw_ik_joint_limit_violation_max", 0.0) or 0.0) for row in rows
                ),
                "max_calf_upper_saturation": max(
                    float(row.get("calf_upper_saturation_max", 0.0) or 0.0) for row in rows
                ),
                "max_speed_magnitude_tracking_error": max(
                    float(row.get("speed_magnitude_tracking_error", 0.0) or 0.0) for row in rows
                ),
                "min_planned_foot_ground_clearance_m": min(
                    float(row.get("planned_foot_ground_clearance_min_m", 0.0) or 0.0) for row in rows
                ),
                "max_planned_foot_ground_penetration_count": max(
                    int(row.get("planned_foot_ground_penetration_count", 0) or 0) for row in rows
                ),
                "min_fk_foot_ground_clearance_m": min(
                    float(row.get("fk_foot_ground_clearance_min_m", 0.0) or 0.0) for row in rows
                ),
                "max_fk_foot_ground_penetration_count": max(
                    int(row.get("fk_foot_ground_penetration_count", 0) or 0) for row in rows
                ),
                "min_fk_knee_ground_clearance_m": min(
                    float(row.get("fk_knee_ground_clearance_min_m", 0.0) or 0.0) for row in rows
                ),
                "max_fk_knee_ground_penetration_count": max(
                    int(row.get("fk_knee_ground_penetration_count", 0) or 0) for row in rows
                ),
                "min_fk_shank_ground_clearance_m": min(
                    float(row.get("fk_shank_ground_clearance_min_m", 0.0) or 0.0) for row in rows
                ),
                "max_fk_shank_ground_penetration_count": max(
                    int(row.get("fk_shank_ground_penetration_count", 0) or 0) for row in rows
                ),
                "min_body_ground_clearance_m": min(
                    float(row.get("body_ground_clearance_min_m", 0.0) or 0.0) for row in rows
                ),
                "max_body_ground_penetration_count": max(
                    int(row.get("body_ground_penetration_count", 0) or 0) for row in rows
                ),
                "max_root_step_xy_m": max(float(row.get("root_step_xy_max_m", 0.0) or 0.0) for row in rows),
                "max_root_step_z_m": max(float(row.get("root_step_z_max_m", 0.0) or 0.0) for row in rows),
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        runtime.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--terrain-row", type=int, default=8)
    parser.add_argument("--terrain-col", type=int, default=12)
    parser.add_argument("--offsets", default="0.0:0.0")
    parser.add_argument("--commands", default=";".join(DEFAULT_COMMANDS))
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--requested-n-frames", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--playback-frames", type=int, default=25)
    parser.add_argument("--ik-fk-weight-mult", type=float, default=1.0)
    parser.add_argument("--kinematics-weight-mult", type=float, default=1.0)
    parser.add_argument("--root-foot-center-weight-mult", type=float, default=1.0)
    parser.add_argument("--fk-body-leg-collision-weight-mult", type=float, default=1.0)
    parser.add_argument("--touchdown-endpoint-weight-mult", type=float, default=1.0)
    parser.add_argument("--progress-weight-mult", type=float, default=1.0)
    parser.add_argument("--swing-direction-weight-mult", type=float, default=1.0)
    parser.add_argument("--optimize-steps", type=int, default=None)
    parser.add_argument(
        "--sequence",
        default=None,
        help=f"Continuous no-reset command sequence, e.g. {DEFAULT_SEQUENCE!r}. Use name:vx,vy,wzxN entries separated by ';'.",
    )
    args = parser.parse_args()
    return run_probe(
        device=str(args.device),
        terrain_row=int(args.terrain_row),
        terrain_col=int(args.terrain_col),
        offsets=_parse_offsets(str(args.offsets)),
        commands=_parse_command_list(str(args.commands)),
        cycles=int(args.cycles),
        requested_n_frames=int(args.requested_n_frames),
        warmup_steps=int(args.warmup_steps),
        playback_frames=int(args.playback_frames),
        ik_fk_weight_mult=float(args.ik_fk_weight_mult),
        kinematics_weight_mult=float(args.kinematics_weight_mult),
        root_foot_center_weight_mult=float(args.root_foot_center_weight_mult),
        fk_body_leg_collision_weight_mult=float(args.fk_body_leg_collision_weight_mult),
        touchdown_endpoint_weight_mult=float(args.touchdown_endpoint_weight_mult),
        progress_weight_mult=float(args.progress_weight_mult),
        swing_direction_weight_mult=float(args.swing_direction_weight_mult),
        optimize_steps=args.optimize_steps,
        sequence=None if args.sequence is None else _parse_sequence(str(args.sequence)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
