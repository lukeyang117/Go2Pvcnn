from __future__ import annotations

import argparse
import json
import math
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
from extension.batch_mpc_planner.kinematics import (  # noqa: E402
    HIP_OFFSETS_ARRAY,
    _JOINT_LIMITS,
    _rpy_to_rot_matrix,
    fk_feet_from_joint_angles,
)
from extension.viz.go2_foostep_planner import _quat_wxyz_to_rpy  # noqa: E402
from mpc_qp_viewer_crossing_probe import _playback_readback_error, _qp_diag_metrics  # noqa: E402
from mpc_swing_trajectory_quality_probe import _parse_command, _trajectory_summary  # noqa: E402


DEFAULT_COMMANDS: tuple[str, ...] = (
    "forward:0.35,0.0,0.0",
    "diag_left:0.30,0.12,0.0",
)
DEFAULT_TILES: tuple[tuple[int, int], ...] = ((9, 19),)
DEFAULT_OFFSETS: tuple[tuple[float, float], ...] = ((0.0, 0.0),)


def _parse_command_list(value: str) -> tuple[str, ...]:
    text = str(value)
    if ";" in text:
        return tuple(item.strip() for item in text.split(";") if item.strip())
    if ":" in text:
        return (text.strip(),) if text.strip() else ()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _parse_tiles(value: str) -> tuple[tuple[int, int], ...]:
    tiles: list[tuple[int, int]] = []
    for item in str(value).replace(";", ",").split(","):
        token = item.strip()
        if not token:
            continue
        if ":" in token:
            row_s, col_s = token.split(":", 1)
        elif "/" in token:
            row_s, col_s = token.split("/", 1)
        else:
            parts = token.split()
            if len(parts) != 2:
                raise ValueError(f"Tile must look like row:col, got {token!r}")
            row_s, col_s = parts
        tiles.append((int(row_s), int(col_s)))
    return tuple(tiles)


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


def _loss_scalar(result, name: str) -> float:
    loss_breakdown = getattr(result, "loss_breakdown", None)
    if not loss_breakdown or name not in loss_breakdown:
        return 0.0
    value = torch.as_tensor(loss_breakdown[name])
    if int(value.numel()) == 0:
        return 0.0
    return float(value.reshape(-1).amax().item())


def _terrain_height_summary(terrain) -> dict[str, float]:
    height = torch.as_tensor(terrain.height_map[0], dtype=torch.float64)
    finite = torch.isfinite(height)
    if not bool(finite.any().item()):
        return {"terrain_height_min_m": math.nan, "terrain_height_max_m": math.nan, "terrain_height_range_m": math.nan}
    hmin = float(height[finite].amin().item())
    hmax = float(height[finite].amax().item())
    return {
        "terrain_height_min_m": hmin,
        "terrain_height_max_m": hmax,
        "terrain_height_range_m": hmax - hmin,
    }


def _result_readback_detail(result) -> dict[str, object]:
    root_value = getattr(result, "root_pos", None)
    if root_value is None:
        root_value = result.root_pos_w
    rpy_value = getattr(result, "root_rpy", None)
    if rpy_value is None:
        rpy_value = _quat_wxyz_to_rpy(torch.as_tensor(result.root_quat_w))
    root = torch.as_tensor(root_value, dtype=torch.float64)
    rpy = torch.as_tensor(rpy_value, dtype=torch.float64, device=root.device)
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64, device=root.device)
    joint = torch.as_tensor(result.joint_angles, dtype=torch.float64, device=root.device)
    fk_foot = fk_feet_from_joint_angles(root, rpy, joint)
    error = torch.linalg.vector_norm(fk_foot - foot, dim=-1)
    flat_idx = int(torch.argmax(error.reshape(-1)).item()) if int(error.numel()) else 0
    _, frame_idx, leg_idx = torch.unravel_index(torch.tensor(flat_idx, device=error.device), error.shape)
    frame_i = int(frame_idx.item())
    leg_i = int(leg_idx.item())

    rot_world_to_body = _rpy_to_rot_matrix(rpy).transpose(-1, -2)
    foot_delta_w = foot - root.unsqueeze(2)
    foot_body = torch.einsum("btij,btkj->btki", rot_world_to_body, foot_delta_w)
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=root.device, dtype=root.dtype).view(1, 1, 4, 3)
    foot_hip = foot_body - hip_offsets
    reach_xz = torch.linalg.vector_norm(foot_hip[..., (0, 2)], dim=-1)
    reach_xyz = torch.linalg.vector_norm(foot_hip, dim=-1)

    limits = _JOINT_LIMITS.to(dtype=joint.dtype, device=joint.device)
    lower = limits[:, 0].view(1, 1, 12)
    upper = limits[:, 1].view(1, 1, 12)
    limit_margin = torch.minimum(joint - lower, upper - joint)
    saturated = torch.logical_or(
        torch.isclose(joint, lower, atol=1.0e-5, rtol=0.0),
        torch.isclose(joint, upper, atol=1.0e-5, rtol=0.0),
    )
    leg_joint = joint.reshape(joint.shape[0], joint.shape[1], 4, 3)
    leg_margin = limit_margin.reshape(limit_margin.shape[0], limit_margin.shape[1], 4, 3)
    leg_saturated = saturated.reshape(saturated.shape[0], saturated.shape[1], 4, 3)

    return {
        "qp_readback_worst_frame": frame_i,
        "qp_readback_worst_leg": leg_i,
        "qp_readback_detail_error_m": float(error[0, frame_i, leg_i].item()),
        "qp_readback_detail_root_z_m": float(root[0, frame_i, 2].item()),
        "qp_readback_detail_target_foot_z_m": float(foot[0, frame_i, leg_i, 2].item()),
        "qp_readback_detail_fk_foot_z_m": float(fk_foot[0, frame_i, leg_i, 2].item()),
        "qp_readback_detail_foot_body_x_m": float(foot_body[0, frame_i, leg_i, 0].item()),
        "qp_readback_detail_foot_body_y_m": float(foot_body[0, frame_i, leg_i, 1].item()),
        "qp_readback_detail_foot_body_z_m": float(foot_body[0, frame_i, leg_i, 2].item()),
        "qp_readback_detail_foot_hip_x_m": float(foot_hip[0, frame_i, leg_i, 0].item()),
        "qp_readback_detail_foot_hip_y_m": float(foot_hip[0, frame_i, leg_i, 1].item()),
        "qp_readback_detail_foot_hip_z_m": float(foot_hip[0, frame_i, leg_i, 2].item()),
        "qp_readback_detail_reach_xz_m": float(reach_xz[0, frame_i, leg_i].item()),
        "qp_readback_detail_reach_xyz_m": float(reach_xyz[0, frame_i, leg_i].item()),
        "qp_readback_detail_leg_joint_min_margin_rad": float(leg_margin[0, frame_i, leg_i].amin().item()),
        "qp_readback_detail_leg_joint_saturated_count": int(torch.count_nonzero(leg_saturated[0, frame_i, leg_i]).item()),
        "qp_readback_global_joint_saturated_count": int(torch.count_nonzero(saturated).item()),
        "qp_readback_global_joint_min_margin_rad": float(limit_margin.amin().item()),
        "qp_readback_detail_leg_joints": [
            float(v) for v in leg_joint[0, frame_i, leg_i].detach().cpu().reshape(-1).tolist()
        ],
    }


def _scan_tile_height_range(
    runtime: RealViewerRuntimeFixture,
    *,
    terrain_row: int,
    terrain_col: int,
    offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> dict[str, object]:
    selected = runtime.move_env0_to_terrain_tile(
        terrain_row=int(terrain_row),
        terrain_col=int(terrain_col),
        z_clearance=0.85,
        offset_xy_m=offset_xy_m,
        ground_robot=True,
    )
    terrain = runtime._single_env_terrain()
    scanner_xy = torch.as_tensor(runtime.scanner.data.pos_w[0, :2], dtype=torch.float64).detach().cpu()
    selected_xy = torch.as_tensor(selected[:2], dtype=torch.float64).detach().cpu()
    offset_xy = torch.tensor(offset_xy_m, dtype=torch.float64)
    target_xy = selected_xy + offset_xy
    summary: dict[str, object] = {
        "terrain_row": int(terrain_row),
        "terrain_col": int(terrain_col),
        "offset_x_m": float(offset_xy[0].item()),
        "offset_y_m": float(offset_xy[1].item()),
        "selected_x": float(selected_xy[0].item()),
        "selected_y": float(selected_xy[1].item()),
        "target_x": float(target_xy[0].item()),
        "target_y": float(target_xy[1].item()),
        "scanner_selected_xy_error_m": float(torch.linalg.vector_norm(scanner_xy - target_xy).item()),
    }
    summary.update(_terrain_height_summary(terrain))
    return summary


def _auto_scan_tiles(runtime: RealViewerRuntimeFixture, *, top_k: int) -> tuple[tuple[int, int], ...]:
    origins = torch.as_tensor(runtime.base_env.scene.terrain.terrain_origins)
    rows, cols = int(origins.shape[0]), int(origins.shape[1])
    scored: list[tuple[float, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            summary = _scan_tile_height_range(runtime, terrain_row=row, terrain_col=col)
            height_range = float(summary["terrain_height_range_m"])
            if math.isfinite(height_range):
                scored.append((height_range, row, col))
            print(json.dumps({"type": "mpc_qp_hard_terrain_scan", **summary}, sort_keys=True), flush=True)
    scored.sort(reverse=True)
    return tuple((row, col) for _score, row, col in scored[: max(1, int(top_k))])


def _cycle_row(
    *,
    runtime: RealViewerRuntimeFixture,
    terrain_row: int,
    terrain_col: int,
    offset_xy_m: tuple[float, float],
    command_name: str,
    command_tuple: tuple[float, float, float],
    cycle: int,
    qp_iterations: int,
    playback_frames: int,
):
    terrain = runtime._single_env_terrain()
    state = runtime._single_env_state()
    command = torch.tensor([command_tuple], dtype=torch.float64, device=runtime.base_env.device)
    result = runtime._viewer._plan_viewer_trajectory(
        terrain=terrain,
        state=state,
        command=command,
        mpc_cfg=runtime.mpc_planner_cfg,
    )
    quality_rows = _trajectory_summary(
        command_name=command_name,
        cycle=int(cycle),
        result=result,
        variant="mpc_qp",
        terrain_case=f"row{terrain_row}_col{terrain_col}",
    )
    summary = dict(quality_rows[0])
    terrain_summary = _terrain_height_summary(terrain)
    row: dict[str, object] = {
        "type": "mpc_qp_hard_terrain_cycle",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "terrain_row": int(terrain_row),
        "terrain_col": int(terrain_col),
        "offset_x_m": float(offset_xy_m[0]),
        "offset_y_m": float(offset_xy_m[1]),
        "command": command_name,
        "cycle": int(cycle),
        "qp_iterations": int(qp_iterations),
        "horizon": int(result.root_pos_w.shape[1]),
        "playback_readback_error_max_m": _playback_readback_error(runtime, result, frames=int(playback_frames)),
        "root_terrain_risk_reduces_progress": _loss_scalar(
            result,
            "qp_continuous_root_terrain_risk_reduces_progress",
        ),
        "root_height_variation_max_m": _loss_scalar(
            result,
            "qp_continuous_root_height_variation_max",
        ),
    }
    row.update(terrain_summary)
    row.update(_qp_diag_metrics(result))
    row.update(_result_readback_detail(result))
    row.update(
        {
            "worst_max_to_median_step": float(summary["worst_max_to_median_step"]),
            "worst_boundary_to_median_step": float(summary["worst_boundary_to_median_step"]),
            "worst_z_unimodal_violation_ratio": float(summary["worst_z_unimodal_violation_ratio"]),
            "min_z_quadratic_r2": float(summary["min_z_quadratic_r2"]),
        }
    )
    frame_idx = min(int(playback_frames) - 1, int(result.num_frames) - 1)
    runtime._viewer._viewer_direct_playback_step(runtime.base_env, result, frame_idx=frame_idx)
    refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
    return row


def summarize_hard_terrain_rows(rows: list[dict[str, object]], *, high_range_threshold_m: float) -> dict[str, object]:
    max_fk_collision = max((int(row.get("qp_fk_semantic_collision_count", 0) or 0) for row in rows), default=0)
    max_touchdown_small = max((float(row.get("qp_touchdown_on_small_count", 0.0) or 0.0) for row in rows), default=0.0)
    max_readback = max((float(row.get("playback_readback_error_max_m", 0.0) or 0.0) for row in rows), default=0.0)
    max_foot_jump = max((float(row.get("qp_continuous_foot_frame_jump_max", 0.0) or 0.0) for row in rows), default=0.0)
    max_joint_jump = max((float(row.get("qp_continuous_joint_frame_jump_max", 0.0) or 0.0) for row in rows), default=0.0)
    max_planned_penetration = max(
        (float(row.get("qp_continuous_planned_foot_terrain_penetration_count", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    max_fk_penetration = max(
        (float(row.get("qp_continuous_fk_foot_terrain_penetration_count", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    min_planned_clearance = min(
        (float(row.get("qp_continuous_planned_foot_terrain_clearance_min", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    min_fk_clearance = min(
        (float(row.get("qp_continuous_fk_foot_terrain_clearance_min", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    max_height_range = max((float(row.get("terrain_height_range_m", 0.0) or 0.0) for row in rows), default=0.0)
    max_root_risk_reduce = max(
        (float(row.get("root_terrain_risk_reduces_progress", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    max_path_height_variation = max(
        (float(row.get("root_height_variation_max_m", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    high_range_rows = [row for row in rows if float(row.get("terrain_height_range_m", 0.0) or 0.0) >= high_range_threshold_m]
    high_path_risk_rows = [
        row for row in rows if float(row.get("root_height_variation_max_m", 0.0) or 0.0) >= high_range_threshold_m
    ]
    accepted = (
        len(rows) > 0
        and max_fk_collision == 0
        and max_touchdown_small <= 1.0e-6
        and max_readback <= 0.05
        and max_foot_jump <= 0.25
        and max_joint_jump <= 1.25
        and max_planned_penetration <= 1.0e-6
        and min_fk_clearance >= -0.005
        and (not high_path_risk_rows or max_root_risk_reduce > 0.0)
    )
    return {
        "type": "mpc_qp_hard_terrain_summary",
        "cycle_count": int(len(rows)),
        "high_range_threshold_m": float(high_range_threshold_m),
        "high_range_cycle_count": int(len(high_range_rows)),
        "high_path_risk_cycle_count": int(len(high_path_risk_rows)),
        "max_terrain_height_range_m": float(max_height_range),
        "max_qp_continuous_root_height_variation_m": float(max_path_height_variation),
        "max_fk_semantic_collision_count": int(max_fk_collision),
        "max_qp_touchdown_on_small_count": float(max_touchdown_small),
        "max_playback_readback_error_m": float(max_readback),
        "max_qp_continuous_foot_frame_jump_m": float(max_foot_jump),
        "max_qp_continuous_joint_frame_jump_rad": float(max_joint_jump),
        "min_qp_continuous_planned_foot_terrain_clearance_m": float(min_planned_clearance),
        "min_qp_continuous_fk_foot_terrain_clearance_m": float(min_fk_clearance),
        "max_qp_continuous_planned_foot_terrain_penetration_count": float(max_planned_penetration),
        "max_qp_continuous_fk_foot_terrain_penetration_count": float(max_fk_penetration),
        "max_qp_continuous_root_terrain_risk_reduces_progress": float(max_root_risk_reduce),
        "viewer_hard_terrain_acceptance_passed": bool(accepted),
    }


def run_probe(
    *,
    device: str,
    commands: tuple[str, ...],
    tiles: tuple[tuple[int, int], ...],
    offsets: tuple[tuple[float, float], ...],
    auto_scan_top_k: int,
    cycles: int,
    requested_n_frames: int,
    warmup_steps: int,
    qp_iterations: int,
    playback_frames: int,
    high_range_threshold_m: float,
    reset_each_case: bool,
) -> int:
    runtime = RealViewerRuntimeFixture(
        num_envs=1,
        device=device,
        planner_backend="mpc_qp",
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
        runtime.mpc_planner_cfg.runtime.qp_iterations = int(qp_iterations)
        runtime.mpc_planner_cfg.runtime.horizon_steps = int(requested_n_frames)
        runtime.mpc_planner_cfg.runtime.replan_interval_steps = int(requested_n_frames)
        selected_tiles = _auto_scan_tiles(runtime, top_k=int(auto_scan_top_k)) if int(auto_scan_top_k) > 0 else tiles
        header = {
            "type": "mpc_qp_hard_terrain_header",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": device,
            "commands": list(commands),
            "tiles": [list(tile) for tile in selected_tiles],
            "offsets": [list(offset) for offset in offsets],
            "cycles": int(cycles),
            "requested_n_frames": int(requested_n_frames),
            "warmup_steps": int(warmup_steps),
            "qp_iterations": int(qp_iterations),
            "playback_frames": int(playback_frames),
            "reset_each_case": bool(reset_each_case),
        }
        print(json.dumps(header, sort_keys=True), flush=True)
        for terrain_row, terrain_col in selected_tiles:
            for offset_xy in offsets:
                tile_scan = _scan_tile_height_range(
                    runtime,
                    terrain_row=int(terrain_row),
                    terrain_col=int(terrain_col),
                    offset_xy_m=offset_xy,
                )
                print(json.dumps({"type": "mpc_qp_hard_terrain_selected_tile", **tile_scan}, sort_keys=True), flush=True)
                for command_text in commands:
                    command_name, command_tuple = _parse_command(command_text)
                    if bool(reset_each_case):
                        runtime.reset()
                    root_quat_w = (
                        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=runtime.base_env.device)
                        if bool(reset_each_case)
                        else None
                    )
                    runtime.move_env0_to_terrain_tile(
                        terrain_row=int(terrain_row),
                        terrain_col=int(terrain_col),
                        z_clearance=0.85,
                        offset_xy_m=offset_xy,
                        ground_robot=True,
                        root_quat_w=root_quat_w,
                    )
                    for cycle in range(int(cycles)):
                        row = _cycle_row(
                            runtime=runtime,
                            terrain_row=int(terrain_row),
                            terrain_col=int(terrain_col),
                            offset_xy_m=offset_xy,
                            command_name=command_name,
                            command_tuple=command_tuple,
                            cycle=cycle,
                            qp_iterations=int(qp_iterations),
                            playback_frames=int(playback_frames),
                        )
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
        summary = summarize_hard_terrain_rows(rows, high_range_threshold_m=float(high_range_threshold_m))
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if bool(summary["viewer_hard_terrain_acceptance_passed"]) else 2
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--commands", default=";".join(DEFAULT_COMMANDS))
    parser.add_argument("--tiles", default=",".join(f"{row}:{col}" for row, col in DEFAULT_TILES))
    parser.add_argument("--offsets", default=",".join(f"{x}:{y}" for x, y in DEFAULT_OFFSETS))
    parser.add_argument("--auto-scan-top-k", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--requested-n-frames", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--qp-iterations", type=int, default=1)
    parser.add_argument("--playback-frames", type=int, default=25)
    parser.add_argument("--high-range-threshold-m", type=float, default=0.15)
    parser.add_argument("--reset-each-case", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    commands = _parse_command_list(str(args.commands))
    tiles = _parse_tiles(str(args.tiles))
    offsets = _parse_offsets(str(args.offsets))
    return run_probe(
        device=str(args.device),
        commands=commands,
        tiles=tiles,
        offsets=offsets,
        auto_scan_top_k=int(args.auto_scan_top_k),
        cycles=int(args.cycles),
        requested_n_frames=int(args.requested_n_frames),
        warmup_steps=int(args.warmup_steps),
        qp_iterations=int(args.qp_iterations),
        playback_frames=int(args.playback_frames),
        high_range_threshold_m=float(args.high_range_threshold_m),
        reset_each_case=bool(args.reset_each_case),
    )


if __name__ == "__main__":
    raise SystemExit(main())
