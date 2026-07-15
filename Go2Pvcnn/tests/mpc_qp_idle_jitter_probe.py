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


def _max_step_norm(values: torch.Tensor) -> float:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim < 2 or int(tensor.shape[1]) < 2:
        return 0.0
    delta = tensor[:, 1:] - tensor[:, :-1]
    if delta.ndim == 3:
        norm = torch.linalg.vector_norm(delta, dim=-1)
    elif delta.ndim == 4:
        norm = torch.linalg.vector_norm(delta, dim=-1)
    else:
        norm = torch.abs(delta)
    return float(norm.amax().item()) if int(norm.numel()) else 0.0


def _max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    lhs = torch.as_tensor(a, dtype=torch.float64)
    rhs = torch.as_tensor(b, dtype=torch.float64, device=lhs.device)
    err = torch.linalg.vector_norm(lhs - rhs, dim=-1)
    return float(err.amax().item()) if int(err.numel()) else 0.0


def _playback_readback_sequences(
    runtime: RealViewerRuntimeFixture,
    result,
    *,
    frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    viewer = runtime._viewer
    frame_count = min(int(frames), int(result.num_frames))
    roots: list[torch.Tensor] = []
    feet: list[torch.Tensor] = []
    joints: list[torch.Tensor] = []
    for frame_idx in range(frame_count):
        viewer._viewer_direct_playback_step(runtime.base_env, result, frame_idx=frame_idx)
        actual = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
        roots.append(
            torch.as_tensor(
                runtime.robot.data.root_pos_w[:1],
                dtype=torch.float64,
                device=runtime.base_env.device,
            ).clone()
        )
        feet.append(torch.as_tensor(actual["foot_pos_w"], dtype=torch.float64, device=runtime.base_env.device).clone())
        joints.append(
            torch.as_tensor(
                actual["joint_pos_planner"],
                dtype=torch.float64,
                device=runtime.base_env.device,
            ).clone()
        )
    return torch.stack(roots, dim=1), torch.stack(feet, dim=1), torch.stack(joints, dim=1)


def _loss_scalar(result, name: str) -> float:
    loss_breakdown = getattr(result, "loss_breakdown", None)
    if not loss_breakdown or name not in loss_breakdown:
        return 0.0
    value = torch.as_tensor(loss_breakdown[name], dtype=torch.float64)
    return float(value.reshape(-1).amax().item()) if int(value.numel()) else 0.0


def _actual_root_and_foot(runtime: RealViewerRuntimeFixture) -> tuple[torch.Tensor, torch.Tensor]:
    actual = runtime._viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
    root = torch.as_tensor(
        runtime.robot.data.root_pos_w[:1],
        dtype=torch.float64,
        device=runtime.base_env.device,
    ).clone()
    foot = torch.as_tensor(actual["foot_pos_w"], dtype=torch.float64, device=runtime.base_env.device).clone()
    return root, foot


def _run_env_step_probe(
    runtime: RealViewerRuntimeFixture,
    *,
    frames: int,
) -> dict[str, object]:
    roots: list[torch.Tensor] = []
    feet: list[torch.Tensor] = []
    for _ in range(max(1, int(frames))):
        runtime.env.step(runtime.zero_actions)
        root, foot = _actual_root_and_foot(runtime)
        roots.append(root)
        feet.append(foot)
    root_seq = torch.stack(roots, dim=1)
    foot_seq = torch.stack(feet, dim=1)
    root_total = torch.linalg.vector_norm(root_seq[:, -1] - root_seq[:, 0], dim=-1)
    foot_total = torch.linalg.vector_norm(foot_seq[:, -1] - foot_seq[:, 0], dim=-1)
    return {
        "actual_root_step_max_m": _max_step_norm(root_seq),
        "actual_foot_step_max_m": _max_step_norm(foot_seq),
        "actual_root_total_delta_m": float(root_total.amax().item()) if int(root_total.numel()) else 0.0,
        "actual_foot_total_delta_m": float(foot_total.amax().item()) if int(foot_total.numel()) else 0.0,
    }


def run_probe(
    *,
    device: str,
    planner_backend: str,
    cycles: int,
    requested_n_frames: int,
    playback_frames: int,
    warmup_steps: int,
    qp_iterations: int,
    terrain: str,
    terrain_row: int | None,
    terrain_col: int | None,
    step_env_between_cycles: int,
    mode: str,
    pre_command: tuple[float, float, float],
    pre_cycles: int,
) -> int:
    runtime = RealViewerRuntimeFixture(
        num_envs=1,
        device=device,
        terrain=terrain,
        planner_backend=planner_backend,
        requested_n_frames=requested_n_frames,
        warmup_steps=warmup_steps,
        task_id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
        compact_semantic_grid=not (terrain_row is not None and terrain_col is not None),
        env_cfg_entry_point=(
            "go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg:"
            "TeacherElevationTrajectoryMpcSemanticEnvCfg"
        ),
    )
    rows: list[dict[str, object]] = []
    try:
        runtime_cfg = getattr(runtime.mpc_planner_cfg, "runtime", None)
        if runtime_cfg is not None:
            if hasattr(runtime_cfg, "qp_iterations"):
                runtime_cfg.qp_iterations = int(qp_iterations)
            if hasattr(runtime_cfg, "horizon_steps"):
                runtime_cfg.horizon_steps = int(requested_n_frames)
            if hasattr(runtime_cfg, "replan_interval_steps"):
                runtime_cfg.replan_interval_steps = int(requested_n_frames)
        if terrain_row is not None and terrain_col is not None:
            runtime.move_env0_to_terrain_tile(
                terrain_row=int(terrain_row),
                terrain_col=int(terrain_col),
                ground_robot=True,
            )
        else:
            runtime.reset()
        idle_command = torch.zeros((1, 3), dtype=torch.float64, device=runtime.base_env.device)
        move_command = torch.tensor([pre_command], dtype=torch.float64, device=runtime.base_env.device)
        state = runtime._single_env_state()
        previous_actual_root: torch.Tensor | None = None
        previous_actual_foot: torch.Tensor | None = None
        previous_actual_joint: torch.Tensor | None = None
        previous_planned_root: torch.Tensor | None = None
        previous_planned_foot: torch.Tensor | None = None
        previous_planned_joint: torch.Tensor | None = None
        previous_planned_root_seq: torch.Tensor | None = None
        previous_planned_foot_seq: torch.Tensor | None = None
        previous_planned_joint_seq: torch.Tensor | None = None
        header = {
            "type": "mpc_qp_idle_jitter_header",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": device,
            "planner_backend": str(planner_backend),
            "cycles": int(cycles),
            "requested_n_frames": int(requested_n_frames),
            "playback_frames": int(playback_frames),
            "warmup_steps": int(warmup_steps),
            "qp_iterations": int(qp_iterations),
            "terrain": str(terrain),
            "terrain_row": None if terrain_row is None else int(terrain_row),
            "terrain_col": None if terrain_col is None else int(terrain_col),
            "mode": str(mode),
            "pre_command": [float(v) for v in pre_command],
            "pre_cycles": int(pre_cycles),
        }
        print(json.dumps(header, sort_keys=True), flush=True)
        for cycle in range(int(cycles)):
            command = move_command if int(cycle) < int(pre_cycles) else idle_command
            if str(mode) == "env-step":
                row = {
                    "type": "mpc_qp_idle_jitter_cycle",
                    "cycle": int(cycle),
                    "frame_count": int(playback_frames),
                    "mode": "env-step",
                }
                row.update(_run_env_step_probe(runtime, frames=int(playback_frames)))
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                continue
            terrain_obj = runtime._single_env_terrain()
            result = runtime._viewer._plan_viewer_trajectory(
                terrain=terrain_obj,
                state=state,
                command=command,
                mpc_cfg=runtime.mpc_planner_cfg,
            )
            frame_count = min(int(playback_frames), int(result.num_frames))
            planned_root = torch.as_tensor(result.root_pos_w[:, :frame_count], dtype=torch.float64)
            planned_foot = torch.as_tensor(result.foot_pos_w[:, :frame_count], dtype=torch.float64)
            planned_joint = torch.as_tensor(result.joint_angles[:, :frame_count], dtype=torch.float64)
            actual_root, actual_foot, actual_joint = _playback_readback_sequences(runtime, result, frames=frame_count)
            row: dict[str, object] = {
                "type": "mpc_qp_idle_jitter_cycle",
                "planner_backend": str(planner_backend),
                "cycle": int(cycle),
                "command_vx": float(command[0, 0].item()),
                "command_vy": float(command[0, 1].item()),
                "command_yaw": float(command[0, 2].item()),
                "is_idle_cycle": bool(int(cycle) >= int(pre_cycles)),
                "horizon": int(result.num_frames),
                "frame_count": int(frame_count),
                "qp_idle_all_stance_active": float(_loss_scalar(result, "qp_idle_all_stance_active")),
                "planned_root_step_max_m": _max_step_norm(planned_root),
                "planned_foot_step_max_m": _max_step_norm(planned_foot),
                "planned_joint_step_max_rad": _max_step_norm(planned_joint),
                "actual_root_step_max_m": _max_step_norm(actual_root),
                "actual_foot_step_max_m": _max_step_norm(actual_foot),
                "actual_joint_step_max_rad": _max_step_norm(actual_joint),
                "root_planned_vs_actual_max_m": _max_error(planned_root, actual_root),
                "foot_planned_vs_actual_max_m": _max_error(planned_foot, actual_foot),
                "joint_planned_vs_actual_max_rad": _max_error(planned_joint, actual_joint),
                "qp_continuous_root_frame_jump_max": _loss_scalar(result, "qp_continuous_root_frame_jump_max"),
                "qp_continuous_foot_frame_jump_max": _loss_scalar(result, "qp_continuous_foot_frame_jump_max"),
                "qp_continuous_joint_frame_jump_max": _loss_scalar(result, "qp_continuous_joint_frame_jump_max"),
            }
            if previous_actual_root is not None and previous_actual_foot is not None:
                row["actual_root_replan_boundary_delta_m"] = _max_error(previous_actual_root, actual_root[:, :1])
                row["actual_foot_replan_boundary_delta_m"] = _max_error(previous_actual_foot, actual_foot[:, :1])
                row["actual_joint_replan_boundary_delta_rad"] = _max_error(previous_actual_joint, actual_joint[:, :1])
            if previous_planned_root is not None and previous_planned_foot is not None:
                row["planned_root_replan_boundary_delta_m"] = _max_error(previous_planned_root, planned_root[:, :1])
                row["planned_foot_replan_boundary_delta_m"] = _max_error(previous_planned_foot, planned_foot[:, :1])
                row["planned_joint_replan_boundary_delta_rad"] = _max_error(previous_planned_joint, planned_joint[:, :1])
            if previous_planned_root_seq is not None and previous_planned_foot_seq is not None:
                common_frames = min(int(previous_planned_root_seq.shape[1]), int(planned_root.shape[1]))
                row["planned_root_trajectory_replan_delta_m"] = _max_error(
                    previous_planned_root_seq[:, :common_frames],
                    planned_root[:, :common_frames],
                )
                row["planned_foot_trajectory_replan_delta_m"] = _max_error(
                    previous_planned_foot_seq[:, :common_frames],
                    planned_foot[:, :common_frames],
                )
                row["planned_joint_trajectory_replan_delta_rad"] = _max_error(
                    previous_planned_joint_seq[:, :common_frames],
                    planned_joint[:, :common_frames],
                )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            previous_actual_root = actual_root[:, -1:].clone()
            previous_actual_foot = actual_foot[:, -1:].clone()
            previous_actual_joint = actual_joint[:, -1:].clone()
            previous_planned_root = planned_root[:, -1:].clone()
            previous_planned_foot = planned_foot[:, -1:].clone()
            previous_planned_joint = planned_joint[:, -1:].clone()
            previous_planned_root_seq = planned_root.clone()
            previous_planned_foot_seq = planned_foot.clone()
            previous_planned_joint_seq = planned_joint.clone()
            refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
            state = runtime._viewer._planner_state_from_reference_result(result, frame_idx=int(result.num_frames) - 1)
            for _ in range(max(0, int(step_env_between_cycles))):
                runtime.env.step(runtime.zero_actions)
        summary = {
            "type": "mpc_qp_idle_jitter_summary",
            "planner_backend": str(planner_backend),
            "cycle_count": int(len(rows)),
            "mode": str(mode),
            "max_planned_root_step_m": max(
                (float(row.get("planned_root_step_max_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_planned_foot_step_m": max(
                (float(row.get("planned_foot_step_max_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_planned_joint_step_rad": max(
                (float(row.get("planned_joint_step_max_rad", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_actual_root_step_m": max((float(row["actual_root_step_max_m"]) for row in rows), default=0.0),
            "max_actual_foot_step_m": max((float(row["actual_foot_step_max_m"]) for row in rows), default=0.0),
            "max_actual_joint_step_rad": max(
                (float(row.get("actual_joint_step_max_rad", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_actual_root_total_delta_m": max(
                (float(row.get("actual_root_total_delta_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_actual_foot_total_delta_m": max(
                (float(row.get("actual_foot_total_delta_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_root_planned_vs_actual_m": max(
                (float(row.get("root_planned_vs_actual_max_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_foot_planned_vs_actual_m": max(
                (float(row.get("foot_planned_vs_actual_max_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_joint_planned_vs_actual_rad": max(
                (float(row.get("joint_planned_vs_actual_max_rad", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_actual_root_replan_boundary_delta_m": max(
                (float(row.get("actual_root_replan_boundary_delta_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_actual_foot_replan_boundary_delta_m": max(
                (float(row.get("actual_foot_replan_boundary_delta_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_actual_joint_replan_boundary_delta_rad": max(
                (float(row.get("actual_joint_replan_boundary_delta_rad", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_planned_root_replan_boundary_delta_m": max(
                (float(row.get("planned_root_replan_boundary_delta_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_planned_foot_replan_boundary_delta_m": max(
                (float(row.get("planned_foot_replan_boundary_delta_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_planned_joint_replan_boundary_delta_rad": max(
                (float(row.get("planned_joint_replan_boundary_delta_rad", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_planned_root_trajectory_replan_delta_m": max(
                (float(row.get("planned_root_trajectory_replan_delta_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_planned_foot_trajectory_replan_delta_m": max(
                (float(row.get("planned_foot_trajectory_replan_delta_m", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_planned_joint_trajectory_replan_delta_rad": max(
                (float(row.get("planned_joint_trajectory_replan_delta_rad", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planner-backend", choices=("mpc", "mpc_qp"), default="mpc_qp")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--requested-n-frames", type=int, default=25)
    parser.add_argument("--playback-frames", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--qp-iterations", type=int, default=1)
    parser.add_argument("--terrain", default="task")
    parser.add_argument("--terrain-row", type=int, default=None)
    parser.add_argument("--terrain-col", type=int, default=None)
    parser.add_argument("--step-env-between-cycles", type=int, default=0)
    parser.add_argument("--mode", choices=("playback", "env-step"), default="playback")
    parser.add_argument("--pre-command", default="0,0,0")
    parser.add_argument("--pre-cycles", type=int, default=0)
    args = parser.parse_args()
    pre_values = tuple(float(part.strip()) for part in str(args.pre_command).replace(" ", ",").split(",") if part.strip())
    if len(pre_values) != 3:
        raise ValueError("--pre-command must contain exactly three values, e.g. 0.45,0,0")
    return run_probe(
        device=str(args.device),
        planner_backend=str(args.planner_backend),
        cycles=int(args.cycles),
        requested_n_frames=int(args.requested_n_frames),
        playback_frames=int(args.playback_frames),
        warmup_steps=int(args.warmup_steps),
        qp_iterations=int(args.qp_iterations),
        terrain=str(args.terrain),
        terrain_row=args.terrain_row,
        terrain_col=args.terrain_col,
        step_env_between_cycles=int(args.step_env_between_cycles),
        mode=str(args.mode),
        pre_command=(float(pre_values[0]), float(pre_values[1]), float(pre_values[2])),
        pre_cycles=int(args.pre_cycles),
    )


if __name__ == "__main__":
    raise SystemExit(main())
