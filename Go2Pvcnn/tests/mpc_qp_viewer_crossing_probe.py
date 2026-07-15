from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT, GO2PVCNN_ROOT / "tests"):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture, refresh_targeted_scanner_pose  # noqa: E402
from mpc_low_small_reachable_crossing_probe import (  # noqa: E402
    _command_heading_yaw,
    _command_relative_xy,
    _parse_command,
    _result_metrics,
    _set_env0_yaw,
)


DEFAULT_COMMANDS: tuple[str, ...] = ("forward:0.45,0.0,0.0",)


def _loss_scalar(result, name: str) -> float:
    loss_breakdown = getattr(result, "loss_breakdown", None)
    if not loss_breakdown or name not in loss_breakdown:
        return 0.0
    value = torch.as_tensor(loss_breakdown[name])
    if int(value.numel()) == 0:
        return 0.0
    return float(value.reshape(-1).amax().item())


def _qp_diag_metrics(result) -> dict[str, float]:
    keys = (
        "qp_low_small_contact_over_repair_count",
        "qp_low_small_swing_over_repair_count",
        "qp_low_small_crossing_root_lift_count",
        "qp_fk_body_leg_xy_repair_count",
        "qp_fk_body_leg_root_lift_count",
        "qp_fk_shank_clearance_lift_count",
        "qp_fk_low_small_contact_suppressed_count",
        "qp_touchdown_on_small_count",
        "qp_fk_semantic_collision_count",
        "qp_fk_body_leg_collision_count",
        "qp_crossing_leg_count",
        "qp_continuous_foot_frame_jump_max",
        "qp_continuous_joint_frame_jump_max",
        "qp_continuous_fk_readback_error_max",
        "qp_continuous_fk_readback_start_error_max",
        "qp_continuous_fk_readback_mid_error_max",
        "qp_continuous_fk_readback_end_error_max",
        "qp_continuous_low_small_clearance_deficit_max",
        "qp_continuous_planned_foot_terrain_clearance_min",
        "qp_continuous_fk_foot_terrain_clearance_min",
        "qp_continuous_planned_foot_terrain_penetration_count",
        "qp_continuous_fk_foot_terrain_penetration_count",
        "qp_continuous_swing_height_over_terrain_max",
        "qp_continuous_low_small_swing_height_over_terrain_max",
        "qp_continuous_fk_swing_height_over_terrain_max",
        "qp_continuous_fk_low_small_swing_height_over_terrain_max",
        "qp_continuous_root_terrain_risk_reduces_progress",
        "qp_continuous_low_small_progress_update_count",
        "qp_continuous_low_small_progress_deficit_before_max",
        "qp_continuous_low_small_foot_over_update_count",
        "qp_continuous_low_small_foot_over_lateral_deficit_max",
        "qp_continuous_low_small_foot_over_reach_reject_count",
        "qp_continuous_solver_reachability_update_count",
        "qp_continuous_solver_reachability_excess_before_max",
        "qp_continuous_solver_reachability_delta_max",
        "qp_continuous_solver_fk_readback_update_count",
        "qp_continuous_solver_fk_readback_error_before_max",
        "qp_continuous_solver_fk_endpoint_update_count",
        "qp_continuous_solver_fk_endpoint_delta_max",
        "qp_continuous_solver_fk_root_z_update_count",
        "qp_continuous_solver_fk_root_z_delta_max",
        "qp_continuous_solver_body_leg_clearance_update_count",
        "qp_continuous_solver_body_leg_lateral_update_count",
        "qp_continuous_solver_body_leg_clearance_deficit_before_max",
        "qp_continuous_solver_joint_limit_readback_update_count",
        "qp_continuous_solver_joint_limit_violation_before_max",
    )
    return {key: _loss_scalar(result, key) for key in keys}


def parse_command_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(";") if item.strip())


def summarize_viewer_crossing_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    cycle_count = len(rows)
    foot_over_success = sum(int(row.get("fk_foot_over_low_small_success", 0) or 0) for row in rows)
    opportunity_rows = [row for row in rows if int(row.get("crossing_leg_count", 0) or 0) > 0]
    opportunity_count = len(opportunity_rows)
    required_foot_over_success = sum(
        int(row.get("fk_foot_over_low_small_success", 0) or 0) for row in opportunity_rows
    )
    max_fk_collision = max((int(row.get("fk_semantic_collision_count", 0) or 0) for row in rows), default=0)
    max_penetration = max((float(row.get("fk_foot_small_penetration_rate", 0.0) or 0.0) for row in rows), default=0.0)
    max_touchdown_small = max((float(row.get("fk_touchdown_on_small_rate", 0.0) or 0.0) for row in rows), default=0.0)
    max_stance_small = max((float(row.get("fk_stance_on_small_rate", 0.0) or 0.0) for row in rows), default=0.0)
    max_readback = max((float(row.get("playback_readback_error_max_m", 0.0) or 0.0) for row in rows), default=0.0)
    max_foot_jump = max((float(row.get("qp_continuous_foot_frame_jump_max", 0.0) or 0.0) for row in rows), default=0.0)
    max_joint_jump = max((float(row.get("qp_continuous_joint_frame_jump_max", 0.0) or 0.0) for row in rows), default=0.0)
    max_fk_readback = max((float(row.get("qp_continuous_fk_readback_error_max", 0.0) or 0.0) for row in rows), default=0.0)
    max_clearance_deficit = max(
        (float(row.get("qp_continuous_low_small_clearance_deficit_max", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    max_planned_penetration = max(
        (float(row.get("qp_continuous_planned_foot_terrain_penetration_count", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    max_fk_penetration = max(
        (float(row.get("qp_continuous_fk_foot_terrain_penetration_count", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    max_low_small_swing_height = max(
        (float(row.get("qp_continuous_low_small_swing_height_over_terrain_max", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    max_fk_low_small_swing_height = max(
        (float(row.get("qp_continuous_fk_low_small_swing_height_over_terrain_max", 0.0) or 0.0) for row in rows),
        default=0.0,
    )
    accepted = (
        cycle_count > 0
        and opportunity_count > 0
        and required_foot_over_success == opportunity_count
        and max_fk_collision == 0
        and max_penetration <= 1.0e-6
        and max_touchdown_small <= 1.0e-6
        and max_stance_small <= 1.0e-6
        and max_readback <= 0.05
        and max_foot_jump <= 0.25
        and max_joint_jump <= 1.25
        and max_fk_readback <= 0.05
        and max_clearance_deficit <= 1.0e-5
        and max_planned_penetration <= 1.0e-6
        and max_fk_penetration <= 1.0e-6
        and max_low_small_swing_height <= 0.18
        and max_fk_low_small_swing_height <= 0.18
    )
    return {
        "type": "mpc_qp_viewer_crossing_summary",
        "cycle_count": int(cycle_count),
        "crossing_opportunity_count": int(opportunity_count),
        "fk_foot_over_low_small_success_count": int(foot_over_success),
        "fk_foot_over_low_small_required_success_count": int(required_foot_over_success),
        "max_fk_semantic_collision_count": int(max_fk_collision),
        "max_fk_foot_small_penetration_rate": float(max_penetration),
        "max_fk_stance_on_small_rate": float(max_stance_small),
        "max_fk_touchdown_on_small_rate": float(max_touchdown_small),
        "max_playback_readback_error_m": float(max_readback),
        "max_qp_continuous_foot_frame_jump_m": float(max_foot_jump),
        "max_qp_continuous_joint_frame_jump_rad": float(max_joint_jump),
        "max_qp_continuous_fk_readback_error_m": float(max_fk_readback),
        "max_qp_continuous_low_small_clearance_deficit_m": float(max_clearance_deficit),
        "max_qp_continuous_planned_foot_terrain_penetration_count": float(max_planned_penetration),
        "max_qp_continuous_fk_foot_terrain_penetration_count": float(max_fk_penetration),
        "max_qp_continuous_low_small_swing_height_over_terrain_m": float(max_low_small_swing_height),
        "max_qp_continuous_fk_low_small_swing_height_over_terrain_m": float(max_fk_low_small_swing_height),
        "viewer_crossing_acceptance_passed": bool(accepted),
    }


def _playback_readback_error(runtime: RealViewerRuntimeFixture, result, *, frames: int) -> float:
    viewer = runtime._viewer
    max_error = 0.0
    frame_count = min(int(frames), int(result.num_frames))
    for frame_idx in range(frame_count):
        viewer._viewer_direct_playback_step(runtime.base_env, result, frame_idx=frame_idx)
        actual = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
        planned = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64, device=runtime.base_env.device)
        actual_foot = torch.as_tensor(actual["foot_pos_w"], dtype=torch.float64, device=runtime.base_env.device)
        error = torch.linalg.vector_norm(actual_foot - planned, dim=-1)
        max_error = max(max_error, float(error.max().item()) if int(error.numel()) else 0.0)
    return float(max_error)


def run_probe(
    *,
    device: str,
    commands: tuple[str, ...],
    cycles: int,
    requested_n_frames: int,
    warmup_steps: int,
    longitudinal_offset_m: float,
    lateral_offset_m: float,
    semantic_small_height_m: float | None,
    semantic_small_diameter_m: float | None,
    qp_iterations: int,
    playback_frames: int,
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
        semantic_small_height_m=semantic_small_height_m,
        semantic_small_diameter_m=semantic_small_diameter_m,
    )
    rows: list[dict[str, object]] = []
    try:
        runtime.mpc_planner_cfg.runtime.qp_iterations = int(qp_iterations)
        runtime.mpc_planner_cfg.runtime.horizon_steps = int(requested_n_frames)
        runtime.mpc_planner_cfg.runtime.replan_interval_steps = int(requested_n_frames)
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        header = {
            "type": "mpc_qp_viewer_crossing_header",
            "cuda_visible_devices": cuda_visible_devices,
            "device": device,
            "commands": list(commands),
            "cycles": int(cycles),
            "requested_n_frames": int(requested_n_frames),
            "warmup_steps": int(warmup_steps),
            "qp_iterations": int(qp_iterations),
            "playback_frames": int(playback_frames),
            "semantic_small_height_m": None if semantic_small_height_m is None else float(semantic_small_height_m),
            "semantic_small_diameter_m": None if semantic_small_diameter_m is None else float(semantic_small_diameter_m),
        }
        print(json.dumps(header, sort_keys=True), flush=True)
        anchor = runtime.s4_semantic_course_anchor("small")
        obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=runtime.base_env.device)
        for command_text in commands:
            command_name, command_tuple = _parse_command(command_text)
            runtime.reset()
            start_xy = _command_relative_xy(
                anchor.world_xy,
                command_tuple,
                longitudinal_offset_m=longitudinal_offset_m,
                lateral_offset_m=lateral_offset_m,
                device=runtime.base_env.device,
            )
            runtime._write_env0_root_xy(start_xy)
            _set_env0_yaw(runtime, _command_heading_yaw(command_tuple))
            runtime._sync_targeted_scan_pose()
            state = runtime._single_env_state()
            for cycle in range(int(cycles)):
                terrain = runtime._single_env_terrain()
                command = torch.tensor([command_tuple], dtype=torch.float64, device=runtime.base_env.device)
                result = runtime._viewer._plan_viewer_trajectory(
                    terrain=terrain,
                    state=state,
                    command=command,
                    mpc_cfg=runtime.mpc_planner_cfg,
                )
                row: dict[str, object] = {
                    "type": "mpc_qp_viewer_crossing_cycle",
                    "cuda_visible_devices": cuda_visible_devices,
                    "device": device,
                    "command": command_name,
                    "cycle": int(cycle),
                    "qp_iterations": int(qp_iterations),
                    "horizon": int(result.root_pos_w.shape[1]),
                    "semantic_anchor_x": float(anchor.world_xy[0]),
                    "semantic_anchor_y": float(anchor.world_xy[1]),
                    "semantic_target_diameter": float(anchor.target_diameter),
                    "semantic_target_height": float(anchor.target_height),
                }
                row.update(
                    _result_metrics(
                        result,
                        terrain,
                        obstacle_xy,
                        command_tuple,
                        obstacle_height=float(anchor.target_height),
                    )
                )
                row.update(_qp_diag_metrics(result))
                row["playback_readback_error_max_m"] = _playback_readback_error(
                    runtime,
                    result,
                    frames=int(playback_frames),
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
                state = runtime._viewer._planner_state_from_reference_result(result, frame_idx=int(result.num_frames) - 1)
        summary = summarize_viewer_crossing_rows(rows)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if bool(summary["viewer_crossing_acceptance_passed"]) else 2
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--commands", default=";".join(DEFAULT_COMMANDS))
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--requested-n-frames", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--longitudinal-offset-m", type=float, default=-0.35)
    parser.add_argument("--lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--semantic-small-height-m", type=float, default=None)
    parser.add_argument("--semantic-small-diameter-m", type=float, default=None)
    parser.add_argument("--qp-iterations", type=int, default=1)
    parser.add_argument("--playback-frames", type=int, default=50)
    args = parser.parse_args()
    commands = parse_command_list(str(args.commands))
    return run_probe(
        device=str(args.device),
        commands=commands,
        cycles=int(args.cycles),
        requested_n_frames=int(args.requested_n_frames),
        warmup_steps=int(args.warmup_steps),
        longitudinal_offset_m=float(args.longitudinal_offset_m),
        lateral_offset_m=float(args.lateral_offset_m),
        semantic_small_height_m=args.semantic_small_height_m,
        semantic_small_diameter_m=args.semantic_small_diameter_m,
        qp_iterations=int(args.qp_iterations),
        playback_frames=int(args.playback_frames),
    )


if __name__ == "__main__":
    raise SystemExit(main())
