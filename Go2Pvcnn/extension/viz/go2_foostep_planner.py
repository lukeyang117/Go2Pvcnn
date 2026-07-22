"""Isaac Lab livestream viewer for the batched Go2 footstep planner.

Pure kinematic playback: plan once, replay frame-by-frame, replan when
the horizon is exhausted or the teleop command changes.  No physics step.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import math
import os
import select
import signal
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch

THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parents[2]
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


@dataclass(frozen=True)
class ViewerTrajectoryResult:
    num_frames: int
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_angles: torch.Tensor
    foot_pos_w: torch.Tensor
    foot_pos_root: torch.Tensor
    contact_state: torch.Tensor
    planned_touchdown_w: torch.Tensor
    touchdown_seq: torch.Tensor | None = None
    root_lin_vel_w: torch.Tensor | None = None
    root_ang_vel_w: torch.Tensor | None = None
    status: torch.Tensor | None = None
    feasible: torch.Tensor | None = None
    safe_fallback: torch.Tensor | None = None
    state_mode: torch.Tensor | None = None
    small_strategy_outcome: torch.Tensor | None = None
    mode: torch.Tensor | None = None
    selected_beta: torch.Tensor | None = None
    selected_route: torch.Tensor | None = None
    direction_id: torch.Tensor | None = None
    small_front_s: torch.Tensor | None = None
    small_back_s: torch.Tensor | None = None
    small_top_z: torch.Tensor | None = None
    command_direction_violation: torch.Tensor | None = None
    cross_small_success: torch.Tensor | None = None
    body_min_clearance: torch.Tensor | None = None
    leg_min_clearance: torch.Tensor | None = None
    base_min_clearance_to_small: torch.Tensor | None = None
    per_leg_touchdown_on_small_count: torch.Tensor | None = None
    per_leg_foot_small_collision_count: torch.Tensor | None = None
    per_leg_min_clearance_to_small: torch.Tensor | None = None
    per_leg_touchdown_beyond_small_back_edge: torch.Tensor | None = None
    touchdown_ground_gap_by_leg: torch.Tensor | None = None
    touchdown_semantic_by_leg: torch.Tensor | None = None
    touchdown_frame_by_leg: torch.Tensor | None = None
    front_touchdown_ground_gap: torch.Tensor | None = None
    rear_touchdown_ground_gap: torch.Tensor | None = None
    touchdown_on_small_count: torch.Tensor | None = None
    front_foot_small_collision_count: torch.Tensor | None = None
    rear_foot_small_collision_count: torch.Tensor | None = None
    base_small_penetration_count: torch.Tensor | None = None
    base_path_crosses_small_flag: torch.Tensor | None = None
    candidate_hard_reason_mask: torch.Tensor | None = None
    selected_hard_reason_mask: torch.Tensor | None = None
    candidate_hard_rank_cost: torch.Tensor | None = None
    selected_hard_rank_cost: torch.Tensor | None = None
    selected_candidate_index: torch.Tensor | None = None
    hard_reason_mask: torch.Tensor | None = None
    hard_reason_names: tuple[str, ...] | None = None
    status_names: tuple[str, ...] | None = None
    loss_breakdown: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class ViewerResetSnapshot:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


@dataclass
class ViewerStepGate:
    enabled: bool

    def toggle_enabled(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def consume_frame_permission(self, *, step_requested: bool) -> bool:
        if not self.enabled:
            return True
        return bool(step_requested)


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Visualize batched Go2 footstep planning in Isaac Lab.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
    parser.add_argument(
        "--terrain",
        type=str,
        default="task",
        choices=["task"],
        help="Use the active semantic MPC task terrain.",
    )
    parser.add_argument("--n-frames", type=int, default=30, help="Planner horizon in frames.")
    parser.add_argument("--plan-dt", type=float, default=0.02, help="Planner integration step.")
    parser.add_argument(
        "--planner-backend",
        type=str,
        default="mpc",
        choices=["mpc", "joint_mpc_rti"],
        help="Trajectory manager backend used by the task attachment path.",
    )
    parser.add_argument(
        "--mpc-debug-variant",
        type=str,
        default=None,
        help="Optional MPC debug loss variant for reproducing probe behavior, e.g. reachable_fk_cross_v9.",
    )
    parser.add_argument("--vx-scale", type=float, default=0.5, help="Teleop forward/backward speed.")
    parser.add_argument("--vy-scale", type=float, default=0.4, help="Teleop lateral speed.")
    parser.add_argument("--yaw-scale", type=float, default=1, help="Teleop yaw-rate command.")
    parser.add_argument("--key-hold-timeout", type=float, default=0.18, help="Seconds before a key press expires.")
    parser.add_argument("--heightmap-viz-stride", type=int, default=10, help="Subsample stride for heightmap markers.")
    parser.add_argument("--camera-distance", type=float, default=3.2, help="Follow-camera distance behind the robot.")
    parser.add_argument("--camera-height", type=float, default=1.6, help="Follow-camera height offset.")
    parser.add_argument("--warmup-steps", type=int, default=6, help="Number of zero-action warmup steps before visualization.")
    parser.add_argument(
        "--scripted-command",
        type=str,
        default=None,
        help='Optional fixed body-frame command as "vx vy yaw_rate" for deterministic diagnostics.',
    )
    parser.add_argument(
        "--scripted-command-cycles",
        type=int,
        default=0,
        help="How many replan cycles to apply --scripted-command for (0 disables scripted playback).",
    )
    parser.add_argument(
        "--terrain-row",
        type=int,
        default=0,
        help="Generated terrain row used for env0 reset/spawn targeting.",
    )
    parser.add_argument(
        "--terrain-col",
        type=int,
        default=0,
        help="Generated terrain column used for env0 reset/spawn targeting.",
    )
    parser.add_argument(
        "--webrtc-public-ip",
        type=str,
        default=None,
        help="Public endpoint address advertised by Isaac WebRTC livestream. Defaults to PUBLIC_IP or SSH server IP.",
    )
    parser.add_argument(
        "--webrtc-port",
        type=int,
        default=49100,
        help="WebRTC livestream port advertised by Isaac Kit.",
    )
    parser.add_argument(
        "--no-webrtc-auto-public-ip",
        action="store_true",
        default=False,
        help="Disable SSH-based PUBLIC_IP inference for remote WebRTC livestream.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _append_kit_arg(args_cli: argparse.Namespace, kit_arg: str) -> None:
    existing = str(getattr(args_cli, "kit_args", "") or "").strip()
    args_cli.kit_args = f"{existing} {kit_arg}".strip() if existing else kit_arg


def _ssh_server_ip_from_env() -> str | None:
    # SSH_CONNECTION: "<client-ip> <client-port> <server-ip> <server-port>"
    parts = os.environ.get("SSH_CONNECTION", "").split()
    if len(parts) >= 3:
        candidate = parts[2].strip()
        if candidate and not candidate.startswith("127.") and candidate not in {"::1", "localhost"}:
            return candidate
    return None


def _configure_webrtc_endpoint(args_cli: argparse.Namespace) -> None:
    if getattr(args_cli, "livestream", -1) != 2:
        return

    explicit_public_ip = (getattr(args_cli, "webrtc_public_ip", None) or "").strip()
    existing_public_ip = os.environ.get("PUBLIC_IP", "").strip()
    inferred_public_ip = None
    source = None

    if explicit_public_ip:
        inferred_public_ip = explicit_public_ip
        source = "cli"
    elif existing_public_ip:
        inferred_public_ip = existing_public_ip
        source = "env"
    elif not getattr(args_cli, "no_webrtc_auto_public_ip", False):
        inferred_public_ip = _ssh_server_ip_from_env()
        if inferred_public_ip:
            source = "ssh"

    if inferred_public_ip:
        os.environ["PUBLIC_IP"] = inferred_public_ip
        print(
            "[INFO][go2_foostep_planner.py] WebRTC public endpoint "
            f"PUBLIC_IP={inferred_public_ip} source={source}.",
            flush=True,
        )
    else:
        print(
            "[WARN][go2_foostep_planner.py] WebRTC PUBLIC_IP is not set; IsaacLab will advertise 127.0.0.1. "
            "Remote browsers usually need PUBLIC_IP=<server-ip> or --webrtc-public-ip <server-ip>.",
            flush=True,
        )

    webrtc_port = int(getattr(args_cli, "webrtc_port", 49100))
    if webrtc_port != 49100:
        _append_kit_arg(args_cli, f"--/app/livestream/port={webrtc_port}")
        print(
            f"[INFO][go2_foostep_planner.py] WebRTC livestream port override: {webrtc_port}.",
            flush=True,
        )


def _prepare_runtime_args(args_cli: argparse.Namespace) -> argparse.Namespace:
    if getattr(args_cli, "livestream", -1) in (1, 2) and not args_cli.enable_cameras:
        args_cli.enable_cameras = True
        print(
            "[INFO][go2_foostep_planner.py] livestream: enabled AppLauncher --enable_cameras "
            "for WebRTC rendering.",
            flush=True,
        )
    _configure_webrtc_endpoint(args_cli)
    return args_cli


def _planner_state_from_reference_result(result, *, frame_idx: int):
    """Build a planner-facing state snapshot from a trajectory result.

    The planner uses wxyz convention; result stores wxyz quaternions.
    """
    from extension.batch_mpc_planner.types import MpcRobotState

    frame = int(frame_idx)
    root_pos = torch.as_tensor(result.root_pos_w[:, frame], dtype=torch.float64)
    root_quat = torch.as_tensor(result.root_quat_w[:, frame], dtype=torch.float64)
    joint_angles = torch.as_tensor(result.joint_angles[:, frame], dtype=torch.float64)
    foot_pos = torch.as_tensor(result.foot_pos_w[:, frame], dtype=torch.float64)
    return MpcRobotState(
        root_pos=root_pos,
        root_rpy=_quat_wxyz_to_rpy(root_quat),
        joint_angles=joint_angles,
        foot_pos=foot_pos,
        foot_vel=torch.zeros_like(foot_pos),
    )


def _quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _trajectory_motion_summary(result) -> dict[str, float | bool]:
    root_pos = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
    root_quat = torch.as_tensor(result.root_quat_w, dtype=torch.float64)
    first_pos = root_pos[:, 0]
    last_pos = root_pos[:, -1]
    delta_pos = last_pos - first_pos
    first_yaw = _quat_wxyz_to_yaw(root_quat[:, 0])
    last_yaw = _quat_wxyz_to_yaw(root_quat[:, -1])
    delta_yaw = last_yaw - first_yaw
    standstill = bool(torch.allclose(root_pos, root_pos[:, :1], atol=1e-6, rtol=1e-6) and torch.allclose(root_quat, root_quat[:, :1], atol=1e-6, rtol=1e-6))
    return {
        "dx": float(delta_pos[0, 0].item()),
        "dy": float(delta_pos[0, 1].item()),
        "dz": float(delta_pos[0, 2].item()),
        "dyaw": float(delta_yaw[0].item()),
        "standstill": standstill,
    }


def _format_command_values(values: torch.Tensor) -> str:
    command = torch.as_tensor(values, dtype=torch.float64)
    return f"({command[0,0]:+0.2f}, {command[0,1]:+0.2f}, {command[0,2]:+0.2f})"


def _format_hard_reason_mask(mask: torch.Tensor, *, names: tuple[str, ...] | None = None) -> str:
    values = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    reason_names = tuple(f"reason_{idx}" for idx in range(values.numel())) if names is None else tuple(names)
    enabled_names = [name for name, enabled in zip(reason_names, values.tolist()) if enabled]
    return "|".join(enabled_names) if enabled_names else "none"


def _parse_scripted_command(spec: str | None, *, device: torch.device) -> torch.Tensor | None:
    if spec is None:
        return None
    parts = str(spec).split()
    if len(parts) != 3:
        raise ValueError("--scripted-command must contain exactly three floats: vx vy yaw_rate")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--scripted-command must contain exactly three floats: vx vy yaw_rate") from exc
    return torch.tensor([values], dtype=torch.float64, device=device)


def _viewer_mpc_world_command_from_root_frame(command: torch.Tensor, state) -> torch.Tensor:
    body_command = torch.as_tensor(command)
    if body_command.ndim != 2 or int(body_command.shape[-1]) < 3:
        raise ValueError("viewer MPC command must have shape [B, 3]")
    root_rpy = torch.as_tensor(state.root_rpy, dtype=body_command.dtype, device=body_command.device)
    yaw = root_rpy[:, 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    vx_body = body_command[:, 0]
    vy_body = body_command[:, 1]
    world_command = body_command.clone()
    world_command[:, 0] = cos_yaw * vx_body - sin_yaw * vy_body
    world_command[:, 1] = sin_yaw * vx_body + cos_yaw * vy_body
    return world_command


def _quat_wxyz_to_rpy(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = torch.as_tensor(quat_wxyz, dtype=torch.float64)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


def _format_xyz(values: torch.Tensor) -> str:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return f"({tensor[0]:+0.3f}, {tensor[1]:+0.3f}, {tensor[2]:+0.3f})"


def _format_quat(values: torch.Tensor) -> str:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return f"({tensor[0]:+0.4f}, {tensor[1]:+0.4f}, {tensor[2]:+0.4f}, {tensor[3]:+0.4f})"


def _foot_id_list(foot_ids) -> list[int]:
    if isinstance(foot_ids, torch.Tensor):
        return [int(value) for value in foot_ids.detach().cpu().tolist()]
    return [int(value) for value in foot_ids]


def _read_actual_base_state(base_env) -> dict[str, torch.Tensor]:
    from extension.convention import quat_xyzw_to_wxyz

    robot = base_env.scene["robot"]
    root_pos_w = torch.as_tensor(robot.data.root_pos_w[:1], dtype=torch.float64).clone()
    root_quat_raw = torch.as_tensor(robot.data.root_quat_w[:1], dtype=torch.float64).clone()
    rpy_if_wxyz = _quat_wxyz_to_rpy(root_quat_raw)
    rpy_if_xyzw = _quat_wxyz_to_rpy(quat_xyzw_to_wxyz(root_quat_raw))
    return {
        "root_pos_w": root_pos_w,
        "root_quat_raw": root_quat_raw,
        "rpy_if_wxyz": rpy_if_wxyz,
        "rpy_if_xyzw": rpy_if_xyzw,
    }


def _reorder_feet_by_quadrant(foot_pos_w: torch.Tensor, root_pos_w: torch.Tensor) -> torch.Tensor:
    rel = torch.as_tensor(foot_pos_w, dtype=torch.float64) - torch.as_tensor(root_pos_w, dtype=torch.float64).unsqueeze(1)
    order = torch.empty((rel.shape[0], 4), dtype=torch.long, device=rel.device)
    selectors = (
        torch.tensor([1.0, 1.0], dtype=torch.float64, device=rel.device),
        torch.tensor([1.0, -1.0], dtype=torch.float64, device=rel.device),
        torch.tensor([-1.0, 1.0], dtype=torch.float64, device=rel.device),
        torch.tensor([-1.0, -1.0], dtype=torch.float64, device=rel.device),
    )
    xy = rel[..., :2]
    selected = torch.zeros((rel.shape[0], rel.shape[1]), dtype=torch.bool, device=rel.device)
    large_negative = torch.finfo(torch.float64).min
    for target_idx, selector in enumerate(selectors):
        scores = (xy * selector).sum(dim=-1)
        scores = torch.where(selected, torch.full_like(scores, large_negative), scores)
        chosen = scores.argmax(dim=-1)
        order[:, target_idx] = chosen
        selected.scatter_(1, chosen.unsqueeze(-1), True)
    gather_index = order.unsqueeze(-1).expand(-1, -1, foot_pos_w.shape[-1])
    return foot_pos_w.gather(1, gather_index)


def _reorder_feet_to_planner_order(robot, foot_ids_t: torch.Tensor, foot_pos_w: torch.Tensor) -> torch.Tensor:
    body_names = getattr(robot, "body_names", None)
    if not body_names:
        return foot_pos_w
    name_to_local_index: dict[str, int] = {}
    for local_idx, body_id in enumerate(foot_ids_t.detach().cpu().tolist()):
        if not (0 <= int(body_id) < len(body_names)):
            return foot_pos_w
        body_name = normalize_articulation_name(body_names[int(body_id)])
        name_to_local_index[body_name] = int(local_idx)
    order: list[int] = []
    for planner_name in ("fl_foot", "fr_foot", "rl_foot", "rr_foot"):
        local_idx = name_to_local_index.get(planner_name)
        if local_idx is None:
            return foot_pos_w
        order.append(local_idx)
    order_t = torch.as_tensor(order, dtype=torch.long, device=foot_pos_w.device)
    return foot_pos_w.index_select(1, order_t)


def _read_actual_kinematic_state(
    base_env,
    foot_ids: list[int] | torch.Tensor,
    *,
    reorder_by_quadrant: bool = False,
) -> dict[str, torch.Tensor]:
    robot = base_env.scene["robot"]
    joint_pos_planner = _joint_pos_robot_to_planner(
        robot,
        torch.as_tensor(robot.data.joint_pos[:1], dtype=torch.float64).clone(),
    )
    body_pos_w = torch.as_tensor(robot.data.body_pos_w[:1], dtype=torch.float64).clone()
    foot_ids_t = torch.as_tensor(_foot_id_list(foot_ids), dtype=torch.long, device=body_pos_w.device)
    foot_pos_w = body_pos_w.index_select(1, foot_ids_t)
    foot_pos_w = _reorder_feet_to_planner_order(robot, foot_ids_t, foot_pos_w)
    if reorder_by_quadrant:
        root_pos_w = torch.as_tensor(robot.data.root_pos_w[:1], dtype=torch.float64).clone()
        foot_pos_w = _reorder_feet_by_quadrant(foot_pos_w, root_pos_w)
    return {
        "joint_pos_planner": joint_pos_planner,
        "foot_pos_w": foot_pos_w,
    }


def _viewer_loop_need_replan(
    *,
    result,
    playback_frame: int,
    reset_requested: bool,
    teleop_values: torch.Tensor,
    last_cmd: torch.Tensor | None,
    defer_command_replan_until_trajectory_end: bool = False,
    atol: float = 1e-3,
) -> bool:
    if result is None:
        return True
    if playback_frame >= result.num_frames:
        return True
    if reset_requested:
        return True
    if (
        not defer_command_replan_until_trajectory_end
        and last_cmd is not None
        and not torch.allclose(teleop_values, last_cmd, atol=atol)
    ):
        return True
    return False


def _viewer_playback_frame_index(backend: str, *, playback_frame: int) -> int:
    if str(backend).lower() == "joint_mpc_rti":
        return 1
    return int(playback_frame)


def _viewer_command_is_zero(command: torch.Tensor, *, atol: float = 1.0e-5) -> bool:
    values = torch.as_tensor(command, dtype=torch.float64)
    if values.ndim != 2 or int(values.shape[-1]) < 3:
        return False
    return bool(torch.linalg.vector_norm(values[:, :3], dim=-1).max().item() <= float(atol))


def _viewer_plan_has_motion(result, *, atol: float = 1.0e-5) -> bool:
    if result is None:
        return False
    root_pos = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
    root_quat = torch.as_tensor(result.root_quat_w, dtype=torch.float64)
    return not bool(
        torch.allclose(root_pos, root_pos[:, :1], atol=atol, rtol=0.0)
        and torch.allclose(root_quat, root_quat[:, :1], atol=atol, rtol=0.0)
    )


def _viewer_find_grounded_all_feet_frame(
    result,
    terrain,
    *,
    start_frame: int,
    tol_m: float = 0.02,
) -> int | None:
    if result is None:
        return None
    from extension.batch_mpc_planner.terrain import height_at

    foot_pos = torch.as_tensor(result.foot_pos_w, dtype=torch.float32)
    if foot_pos.ndim != 4:
        return None
    start = max(0, min(int(start_frame), int(foot_pos.shape[1]) - 1))
    terrain_z = height_at(terrain, foot_pos[..., :2].reshape(int(foot_pos.shape[0]), -1, 2)).to(
        dtype=foot_pos.dtype,
        device=foot_pos.device,
    ).reshape(foot_pos.shape[:3])
    grounded = torch.abs(foot_pos[..., 2] - terrain_z) <= float(tol_m)
    all_landed = grounded.all(dim=(0, 2))
    future = torch.nonzero(all_landed[start:], as_tuple=False)
    if int(future.numel()) == 0:
        return None
    return int(start + future[0, 0].item())


def _viewer_should_drain_before_zero_replan(
    *,
    backend: str,
    result,
    playback_frame: int,
    teleop_values: torch.Tensor,
    last_cmd: torch.Tensor | None,
    atol: float = 1.0e-3,
) -> bool:
    if str(backend).lower() != "mpc":
        return False
    if result is None or last_cmd is None:
        return False
    if int(playback_frame) >= int(result.num_frames):
        return False
    if not _viewer_command_is_zero(teleop_values):
        return False
    if _viewer_command_is_zero(last_cmd):
        return False
    if torch.allclose(torch.as_tensor(teleop_values), torch.as_tensor(last_cmd), atol=atol):
        return False
    return _viewer_plan_has_motion(result)


def _apply_direct_playback_to_robot(robot, result, *, frame_idx: int) -> None:
    """Write the planner frame pose/joints into the displayed robot.

    Isaac Lab is not available in unit tests, so we keep this duck-typed and
    only call common "write_*_to_sim" methods when present.
    """
    frame = int(frame_idx)
    root_pos_w = torch.as_tensor(result.root_pos_w[:, frame], dtype=torch.float32)
    root_quat_wxyz = torch.as_tensor(result.root_quat_w[:, frame], dtype=torch.float32)
    root_pose_wxyz = torch.cat([root_pos_w, root_quat_wxyz], dim=-1)
    joint_pos = torch.as_tensor(result.joint_angles[:, frame], dtype=torch.float32)
    joint_pos = _joint_pos_planner_to_robot(robot, joint_pos)
    joint_vel = torch.zeros_like(joint_pos)

    if hasattr(robot, "write_root_pose_to_sim"):
        robot.write_root_pose_to_sim(root_pose_wxyz)
    elif hasattr(robot, "write_root_state_to_sim"):
        zeros = torch.zeros((root_pos_w.shape[0], 6), dtype=root_pos_w.dtype, device=root_pos_w.device)
        robot.write_root_state_to_sim(torch.cat([root_pose_wxyz, zeros], dim=-1))

    if hasattr(robot, "write_joint_state_to_sim"):
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
    elif hasattr(robot, "write_joint_pos_to_sim"):
        robot.write_joint_pos_to_sim(joint_pos)
    elif hasattr(robot, "write_joint_position_to_sim"):
        robot.write_joint_position_to_sim(joint_pos)


def _viewer_capture_reset_snapshot(base_env) -> ViewerResetSnapshot:
    robot = base_env.scene["robot"]
    return ViewerResetSnapshot(
        joint_pos=torch.as_tensor(robot.data.joint_pos[:1], dtype=torch.float32).clone(),
        joint_vel=torch.as_tensor(robot.data.joint_vel[:1], dtype=torch.float32).clone(),
    )


def _viewer_zero_base_command(base_env, *, command_name: str = "base_velocity") -> None:
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_command"):
        return
    try:
        command = command_manager.get_command(command_name)
    except Exception:
        return
    if command is None:
        return
    command.zero_()


def _viewer_apply_joint_reset_snapshot(
    base_env,
    snapshot: ViewerResetSnapshot,
    *,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> None:
    robot = base_env.scene["robot"]
    root_pose = torch.cat([root_pos_w, root_quat_w], dim=-1)
    root_vel = torch.zeros((root_pose.shape[0], 6), dtype=root_pose.dtype, device=root_pose.device)
    if hasattr(robot, "write_root_pose_to_sim"):
        robot.write_root_pose_to_sim(root_pose)
    elif hasattr(robot, "write_root_state_to_sim"):
        robot.write_root_state_to_sim(torch.cat([root_pose, root_vel], dim=-1))
    if hasattr(robot, "write_root_velocity_to_sim"):
        robot.write_root_velocity_to_sim(root_vel)
    if hasattr(robot, "write_joint_state_to_sim"):
        robot.write_joint_state_to_sim(snapshot.joint_pos, snapshot.joint_vel)
    elif hasattr(robot, "write_joint_pos_to_sim"):
        robot.write_joint_pos_to_sim(snapshot.joint_pos)
    elif hasattr(robot, "write_joint_position_to_sim"):
        robot.write_joint_position_to_sim(snapshot.joint_pos)
    if hasattr(base_env.scene, "write_data_to_sim"):
        base_env.scene.write_data_to_sim()
    base_env.sim.render()
    if hasattr(base_env.scene, "update"):
        base_env.scene.update(float(base_env.physics_dt))


def _viewer_ground_robot_from_scanner(
    base_env,
    scanner,
    foot_ids,
    *,
    root_pos_xy: torch.Tensor | None = None,
    root_quat_w: torch.Tensor | None = None,
) -> float:
    from extension.batch_mpc_planner.terrain import height_at

    robot = base_env.scene["robot"]
    if root_pos_xy is not None or root_quat_w is not None:
        current_root_pos = torch.as_tensor(robot.data.root_pos_w[:1], dtype=torch.float32).clone()
        current_root_quat = torch.as_tensor(robot.data.root_quat_w[:1], dtype=torch.float32).clone()
        if root_pos_xy is not None:
            current_root_pos[:, :2] = torch.as_tensor(root_pos_xy, dtype=current_root_pos.dtype, device=current_root_pos.device)
        if root_quat_w is not None:
            current_root_quat = torch.as_tensor(root_quat_w, dtype=current_root_quat.dtype, device=current_root_quat.device).clone()
        root_pose = torch.cat([current_root_pos, current_root_quat], dim=-1)
        if hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root_pose)
        elif hasattr(robot, "write_root_state_to_sim"):
            zero_vel = torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device)
            robot.write_root_state_to_sim(torch.cat([root_pose, zero_vel], dim=-1))
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device))
        if hasattr(base_env.scene, "write_data_to_sim"):
            base_env.scene.write_data_to_sim()
        base_env.sim.render()
        if hasattr(base_env.scene, "update"):
            base_env.scene.update(float(base_env.physics_dt))
    foot_ids_t = torch.as_tensor(_foot_id_list(foot_ids), dtype=torch.long, device=robot.data.body_pos_w.device)
    foot_pos_w = torch.as_tensor(robot.data.body_pos_w[:1], dtype=torch.float32).index_select(1, foot_ids_t)
    foot_xy = foot_pos_w[..., :2]
    foot_z = foot_pos_w[..., 2]
    terrain, _ = _compute_mpc_local_terrain(scanner, env_id=0)
    terrain_z = height_at(terrain, foot_xy).to(dtype=foot_z.dtype, device=foot_z.device)
    z_shift = (terrain_z - foot_z).mean(dim=1, keepdim=True)
    if torch.allclose(z_shift, torch.zeros_like(z_shift), atol=1.0e-5, rtol=0.0):
        return 0.0
    root_pos = torch.as_tensor(robot.data.root_pos_w[:1], dtype=torch.float32).clone()
    root_quat = torch.as_tensor(robot.data.root_quat_w[:1], dtype=torch.float32).clone()
    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    root_pose[:, 2] += z_shift[:, 0]
    if hasattr(robot, "write_root_pose_to_sim"):
        robot.write_root_pose_to_sim(root_pose)
    elif hasattr(robot, "write_root_state_to_sim"):
        zero_vel = torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device)
        robot.write_root_state_to_sim(torch.cat([root_pose, zero_vel], dim=-1))
    if hasattr(robot, "write_root_velocity_to_sim"):
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device))
    if hasattr(base_env.scene, "write_data_to_sim"):
        base_env.scene.write_data_to_sim()
    base_env.sim.render()
    if hasattr(base_env.scene, "update"):
        base_env.scene.update(float(base_env.physics_dt))
    return float(z_shift[0, 0].item())


def _viewer_direct_playback_step(base_env, result, *, frame_idx: int, sync_scene: bool = True) -> str:
    _apply_direct_playback_to_robot(base_env.scene["robot"], result, frame_idx=int(frame_idx))
    if sync_scene and hasattr(base_env.scene, "write_data_to_sim"):
        base_env.scene.write_data_to_sim()
    base_env.sim.render()
    if sync_scene and hasattr(base_env.scene, "update"):
        base_env.scene.update(float(base_env.physics_dt))
    return "render+scene_sync" if sync_scene else "render-only"


def _viewer_pump_paused_window(base_env, *, sleep_s: float = 0.01) -> None:
    base_env.sim.render()
    if hasattr(base_env.scene, "update"):
        base_env.scene.update(float(base_env.physics_dt))
    if sleep_s > 0.0:
        time.sleep(float(sleep_s))


def _viewer_update_visualizer_when_permitted(*, frame_permitted: bool, update_fn) -> None:
    if frame_permitted:
        update_fn()


def _viewer_select_active_teleop_values(
    *,
    live_values: torch.Tensor,
    latched_values: torch.Tensor,
    step_mode_enabled: bool,
) -> torch.Tensor:
    if step_mode_enabled:
        return torch.as_tensor(latched_values).clone()
    return torch.as_tensor(live_values).clone()


def _launch_app(args_cli: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    return app_launcher, app_launcher.app


def _attach_reference_manager_if_enabled(env, env_cfg):
    from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

    manager_device = getattr(env, "device", env_cfg.sim.device)
    manager = attach_trajectory_manager_if_enabled(env, env_cfg, device=manager_device)
    if manager is not None:
        print(
            f"[Viewer] Attached {getattr(manager, 'planner_backend', 'mpc')} trajectory manager",
            flush=True,
        )
    return manager

LEG_COLORS = (
    (1.0, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.4, 1.0),
    (1.0, 0.8, 0.2),
)

SEMANTIC_TERRAIN_ID = 0
SEMANTIC_SMALL_ID = 1
SEMANTIC_LARGE_ID = 2

SEMANTIC_MARKER_COLORS = {
    SEMANTIC_TERRAIN_ID: (1.0, 1.0, 1.0),
    SEMANTIC_SMALL_ID: (0.2, 0.9, 0.2),
    SEMANTIC_LARGE_ID: (1.0, 0.2, 0.2),
}

from extension.joint_mpc_rti.integration.joint_order import (
    PLANNER_JOINT_ORDER,
    joint_order_indices,
    normalize_articulation_name,
    planner_to_robot_joints,
    robot_to_planner_joints,
)


def _joint_order_indices(*, source_order: tuple[str, ...], target_order: tuple[str, ...]) -> torch.Tensor | None:
    try:
        return joint_order_indices(source_order=source_order, target_order=target_order)
    except ValueError:
        return None


def _joint_pos_planner_to_robot(robot, joint_pos: torch.Tensor) -> torch.Tensor:
    return planner_to_robot_joints(joint_pos, getattr(robot, "joint_names", None))


def _joint_pos_robot_to_planner(robot, joint_pos: torch.Tensor) -> torch.Tensor:
    return robot_to_planner_joints(joint_pos, getattr(robot, "joint_names", None))


@dataclass
class TeleopCommand:
    values: torch.Tensor
    reset_requested: bool = False
    step_requested: bool = False
    mode_toggle_requested: bool = False


class TerminalTeleop:
    """Minimal raw-terminal teleop with key-repeat based hold semantics."""

    _KEY_AXIS = {
        "w": (0, 1.0),
        "s": (0, -1.0),
        "a": (1, 1.0),
        "d": (1, -1.0),
        "q": (2, 1.0),
        "e": (2, -1.0),
    }

    def __init__(
        self,
        *,
        device: torch.device,
        vx_scale: float,
        vy_scale: float,
        yaw_scale: float,
        timeout_s: float,
    ):
        self._device = device
        self._scales = torch.tensor([vx_scale, vy_scale, yaw_scale], dtype=torch.float64, device=device)
        self._timeout_s = float(timeout_s)
        self._latched_values = torch.zeros((1, 3), dtype=torch.float64, device=device)
        self._last_seen: dict[str, float] = {}
        self._old_termios = None
        self._old_flags = None
        self._enabled = False
        self._stdin_fd = None
        self._old_signal_handlers: dict[int, object] = {}
        self._atexit_registered = False

    def __enter__(self) -> "TerminalTeleop":
        if not sys.stdin.isatty():
            print("[WARN] stdin is not a TTY; teleop keys are disabled.", flush=True)
            return self
        import fcntl
        import termios
        import tty

        self._stdin_fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._stdin_fd)
        self._old_flags = fcntl.fcntl(self._stdin_fd, fcntl.F_GETFL)
        tty.setcbreak(self._stdin_fd)
        fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, self._old_flags | os.O_NONBLOCK)
        self._enabled = True
        self._install_cleanup_guards()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._remove_cleanup_guards()
        self._restore_terminal_state()

    def _restore_terminal_state(self) -> None:
        if not self._enabled:
            return
        import fcntl
        import termios

        assert self._stdin_fd is not None
        self._enabled = False
        if self._old_termios is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_termios)
        if self._old_flags is not None:
            fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, self._old_flags)

    def _install_cleanup_guards(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._restore_terminal_state)
            self._atexit_registered = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._old_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _remove_cleanup_guards(self) -> None:
        if self._atexit_registered:
            try:
                atexit.unregister(self._restore_terminal_state)
            except Exception:
                pass
            self._atexit_registered = False
        for signum, handler in self._old_signal_handlers.items():
            signal.signal(signum, handler)
        self._old_signal_handlers.clear()

    def _handle_signal(self, signum, frame) -> None:
        self._remove_cleanup_guards()
        self._restore_terminal_state()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    def poll(self) -> TeleopCommand:
        reset_requested = False
        step_requested = False
        mode_toggle_requested = False
        if self._enabled:
            while True:
                readable, _, _ = select.select([sys.stdin], [], [], 0.0)
                if not readable:
                    break
                char = sys.stdin.read(1)
                if not char:
                    break
                key = char.lower()
                now = time.monotonic()
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == "x":
                    self._last_seen.clear()
                    self._latched_values.zero_()
                    continue
                if key == "r":
                    reset_requested = True
                    self._last_seen.clear()
                    self._latched_values.zero_()
                    continue
                if key == " ":
                    step_requested = True
                    continue
                if key == "m":
                    mode_toggle_requested = True
                    continue
                if key in self._KEY_AXIS:
                    axis, sign = self._KEY_AXIS[key]
                    self._latched_values[0, axis] = sign * self._scales[axis]
                    self._last_seen[key] = now
        now = time.monotonic()
        values = torch.zeros((1, 3), dtype=torch.float64, device=self._device)
        for key, (axis, sign) in self._KEY_AXIS.items():
            last_seen = self._last_seen.get(key)
            if last_seen is not None and now - last_seen <= self._timeout_s:
                values[0, axis] += sign * self._scales[axis]
        if mode_toggle_requested:
            self._last_seen.clear()
            values = self._latched_values.clone()
        return TeleopCommand(
            values=values,
            reset_requested=reset_requested,
            step_requested=step_requested,
            mode_toggle_requested=mode_toggle_requested,
        )

    def latched_command(self) -> torch.Tensor:
        return self._latched_values.clone()


def _make_marker_cfg(prim_path: str, *, radius: float, color: tuple[float, float, float]):
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg

    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "marker": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


def _make_cuboid_cfg(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
):
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg

    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "marker": sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


class PlannerVisualizer:
    def __init__(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG

        self.root_traj = VisualizationMarkers(
            _make_marker_cfg("/Visuals/BatchedPlanner/root_traj", radius=0.03, color=(1.0, 0.6, 0.1))
        )
        self.heightmap = {
            semantic_id: VisualizationMarkers(
                _make_marker_cfg(
                    f"/Visuals/BatchedPlanner/heightmap_{semantic_id}",
                    radius=0.012,
                    color=color,
                )
            )
            for semantic_id, color in SEMANTIC_MARKER_COLORS.items()
        }
        self.command_arrow = VisualizationMarkers(
            copy.deepcopy(GREEN_ARROW_X_MARKER_CFG).replace(prim_path="/Visuals/BatchedPlanner/command_arrow")
        )
        self.command_arrow.set_visibility(False)
        self.foot_traj = []
        self.touchdowns = []
        for leg_idx, color in enumerate(LEG_COLORS):
            self.foot_traj.append(
                VisualizationMarkers(
                    _make_marker_cfg(
                        f"/Visuals/BatchedPlanner/foot_traj_{leg_idx}",
                        radius=0.02,
                        color=color,
                    )
                )
            )
            self.touchdowns.append(
                VisualizationMarkers(
                    _make_cuboid_cfg(
                        f"/Visuals/BatchedPlanner/touchdown_{leg_idx}",
                        size=(0.05, 0.05, 0.03),
                        color=color,
                    )
                )
            )

    @staticmethod
    def _foot_positions_world(trajectory) -> torch.Tensor:
        foot_pos_w = getattr(trajectory, "foot_pos_w", None)
        if foot_pos_w is not None:
            return foot_pos_w

        from isaaclab.utils import math as math_utils

        root_pos_w = trajectory.root_pos_w
        root_quat_w = trajectory.root_quat_w
        foot_pos_root = trajectory.foot_pos_root
        num_envs, num_frames, num_legs, _ = foot_pos_root.shape
        rotated = math_utils.quat_apply(
            root_quat_w.unsqueeze(2).expand(-1, -1, num_legs, -1).reshape(-1, 4),
            foot_pos_root.reshape(-1, 3),
        ).reshape(num_envs, num_frames, num_legs, 3)
        return rotated + root_pos_w.unsqueeze(2)

    @staticmethod
    def _touchdown_markers_world(trajectory) -> torch.Tensor:
        touchdowns = trajectory.planned_touchdown_w
        if touchdowns.ndim == 4:
            return touchdowns[:, 0]
        return touchdowns

    def update(
        self,
        *,
        result,
        command: torch.Tensor,
        root_yaw: torch.Tensor,
        height_points_by_class: dict[int, torch.Tensor],
    ) -> None:
        from extension.convention import quat_wxyz_to_xyzw

        foot_pos_w = self._foot_positions_world(result)
        touchdown_w = self._touchdown_markers_world(result)
        self.root_traj.visualize(translations=result.root_pos_w[0].to(torch.float32))
        for leg_idx in range(4):
            self.foot_traj[leg_idx].visualize(translations=foot_pos_w[0, :, leg_idx].to(torch.float32))
            self.touchdowns[leg_idx].visualize(translations=touchdown_w[0, leg_idx : leg_idx + 1].to(torch.float32))

        for semantic_id, markers in self.heightmap.items():
            points = height_points_by_class.get(semantic_id)
            if points is None or points.numel() == 0:
                markers.set_visibility(False)
            else:
                markers.set_visibility(True)
                markers.visualize(translations=points.to(torch.float32))

        cmd_xy = command[0, :2]
        speed = float(torch.linalg.norm(cmd_xy).item())
        if speed < 1e-6:
            self.command_arrow.set_visibility(False)
            return

        self.command_arrow.set_visibility(True)
        arrow_yaw = root_yaw + torch.atan2(command[:, 1], command[:, 0])
        arrow_quat_wxyz = torch.stack(
            [
                torch.cos(0.5 * arrow_yaw),
                torch.zeros_like(arrow_yaw),
                torch.zeros_like(arrow_yaw),
                torch.sin(0.5 * arrow_yaw),
            ],
            dim=-1,
        )
        arrow_quat_xyzw = quat_wxyz_to_xyzw(arrow_quat_wxyz).to(torch.float32)
        arrow_pos = result.root_pos_w[0, :1].to(torch.float32).clone()
        arrow_pos[:, 2] = arrow_pos[:, 2] + 0.32
        arrow_scale = torch.tensor([[max(0.25, speed), 0.12, 0.12]], dtype=torch.float32)
        self.command_arrow.visualize(
            translations=arrow_pos,
            orientations=arrow_quat_xyzw,
            scales=arrow_scale,
        )


def _build_env_cfg(args_cli: argparse.Namespace):
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER,
    )

    env_cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = 6.0
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    env_cfg.planner_backend = str(args_cli.planner_backend)
    if env_cfg.planner_backend == "joint_mpc_rti":
        env_cfg.joint_mpc_rti_cfg.runtime.horizon_steps = int(args_cli.n_frames)
        env_cfg.joint_mpc_rti_cfg.runtime.dt = float(args_cli.plan_dt)
    else:
        env_cfg.mpc_planner_cfg.runtime.horizon_steps = int(args_cli.n_frames)
        env_cfg.mpc_planner_cfg.runtime.replan_interval_steps = int(args_cli.n_frames)
        env_cfg.mpc_planner_cfg.runtime.dt = float(args_cli.plan_dt)
        env_cfg.mpc_planner_cfg.runtime.optimize_steps = 16
    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"]["x"] = (0.0, 0.0)
    reset_base.params["pose_range"]["y"] = (0.0, 0.0)
    reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    return env_cfg


def _build_mpc_planner_cfg(env_cfg, args_cli: argparse.Namespace | None = None):
    from extension.batch_mpc_planner.config import planner_cfg_from_task_cfg

    cfg = planner_cfg_from_task_cfg(env_cfg)
    variant = None if args_cli is None else getattr(args_cli, "mpc_debug_variant", None)
    if variant not in (None, "", "baseline"):
        raise ValueError("MPC debug variants were removed from the production planner package.")
    return cfg


def _selected_terrain_origin(scene, *, terrain_row: int, terrain_col: int) -> torch.Tensor:
    terrain = getattr(scene, "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None) if terrain is not None else None
    if terrain_origins is None:
        raise RuntimeError("Viewer terrain-row/terrain-col targeting requires generated terrain origins.")

    origins = torch.as_tensor(terrain_origins)
    if origins.ndim != 3 or origins.shape[-1] < 3:
        raise RuntimeError(f"Expected terrain_origins shape [rows, cols, 3], got {tuple(origins.shape)}")

    num_rows = int(origins.shape[0])
    num_cols = int(origins.shape[1])
    row = int(terrain_row)
    col = int(terrain_col)
    if row < 0 or row >= num_rows:
        raise ValueError(f"--terrain-row must be in [0, {num_rows}), got {row}")
    if col < 0 or col >= num_cols:
        raise ValueError(f"--terrain-col must be in [0, {num_cols}), got {col}")
    return origins[row, col].clone()


def _apply_viewer_terrain_selection(scene, *, env_id: int, terrain_row: int, terrain_col: int) -> torch.Tensor:
    terrain = getattr(scene, "terrain", None)
    selected_origin = _selected_terrain_origin(scene, terrain_row=terrain_row, terrain_col=terrain_col)

    if terrain is not None:
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is not None:
            terrain_levels[env_id] = int(terrain_row)

        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is not None:
            terrain_types[env_id] = int(terrain_col)

        env_origins = getattr(terrain, "env_origins", None)
        if env_origins is not None:
            env_origins[env_id] = selected_origin.to(device=env_origins.device, dtype=env_origins.dtype)

    scene_env_origins = getattr(scene, "env_origins", None)
    if scene_env_origins is not None:
        scene_env_origins[env_id] = selected_origin.to(device=scene_env_origins.device, dtype=scene_env_origins.dtype)

    return selected_origin


def _viewer_scanner_refresh_steps(
    scanner,
    *,
    physics_dt: float,
    minimum_steps: int = 1,
    extra_steps: int = 4,
) -> int:
    scanner_cfg = getattr(scanner, "cfg", None)
    update_period = float(getattr(scanner_cfg, "update_period", 0.0))
    dt = float(physics_dt)
    if not math.isfinite(update_period) or not math.isfinite(dt) or dt <= 0.0:
        return max(1, int(minimum_steps))
    return max(int(minimum_steps), int(math.ceil(update_period / dt)) + max(1, int(extra_steps)))


def _refresh_viewer_scanner(
    base_env,
    scanner,
    *,
    minimum_steps: int = 1,
    extra_steps: int = 4,
) -> int:
    steps = _viewer_scanner_refresh_steps(
        scanner,
        physics_dt=float(base_env.physics_dt),
        minimum_steps=minimum_steps,
        extra_steps=extra_steps,
    )
    for _ in range(steps):
        base_env.sim.render()
        if hasattr(base_env.scene, "update"):
            base_env.scene.update(float(base_env.physics_dt))
    return steps


def _reset_viewer_env(
    env,
    *,
    base_env,
    zero_actions: torch.Tensor,
    warmup_steps: int,
    terrain_row: int,
    terrain_col: int,
    scanner=None,
    reset_snapshot: ViewerResetSnapshot | None = None,
    foot_ids=None,
) -> torch.Tensor:
    preserved_root_pos = None
    preserved_root_quat = None
    if reset_snapshot is not None:
        robot = base_env.scene["robot"]
        preserved_root_pos = torch.as_tensor(robot.data.root_pos_w[:1], dtype=torch.float32).clone()
        preserved_root_quat = torch.as_tensor(robot.data.root_quat_w[:1], dtype=torch.float32).clone()
    selected_origin = _apply_viewer_terrain_selection(
        base_env.scene,
        env_id=0,
        terrain_row=terrain_row,
        terrain_col=terrain_col,
    )
    env.reset()
    _viewer_zero_base_command(base_env)
    selected_origin = _apply_viewer_terrain_selection(
        base_env.scene,
        env_id=0,
        terrain_row=terrain_row,
        terrain_col=terrain_col,
    )
    warmup_step_count = max(0, int(warmup_steps))
    for _ in range(warmup_step_count):
        env.step(zero_actions)
    if reset_snapshot is not None and preserved_root_pos is not None and preserved_root_quat is not None:
        _viewer_apply_joint_reset_snapshot(
            base_env,
            reset_snapshot,
            root_pos_w=preserved_root_pos,
            root_quat_w=preserved_root_quat,
        )
    if scanner is not None:
        _refresh_viewer_scanner(
            base_env,
            scanner,
            minimum_steps=max(1, warmup_step_count),
        )
        if (
            reset_snapshot is not None
            and foot_ids is not None
            and preserved_root_pos is not None
            and preserved_root_quat is not None
        ):
            _viewer_ground_robot_from_scanner(
                base_env,
                scanner,
                foot_ids,
                root_pos_xy=preserved_root_pos[:, :2],
                root_quat_w=preserved_root_quat,
            )
            _refresh_viewer_scanner(
                base_env,
                scanner,
                minimum_steps=1,
            )
    return selected_origin


def _compute_stable_scan_ranges(scanner, *, env_id: int = 0) -> tuple[tuple[float, float], tuple[float, float]]:
    from extension.convention import extract_yaw_batch

    pattern_cfg = scanner.cfg.pattern_cfg
    if not hasattr(pattern_cfg, "size"):
        raise ValueError("scanner.cfg.pattern_cfg must expose a size for stable terrain windows")

    sensor_pos = torch.as_tensor(scanner.data.pos_w[env_id], dtype=torch.float64)
    sensor_quat = torch.as_tensor(scanner.data.quat_w[env_id], dtype=torch.float64).unsqueeze(0)
    yaw = extract_yaw_batch(sensor_quat)[0]

    half_x = 0.5 * float(pattern_cfg.size[0])
    half_y = 0.5 * float(pattern_cfg.size[1])
    local_corners = torch.tensor(
        [
            [-half_x, -half_y],
            [-half_x, half_y],
            [half_x, -half_y],
            [half_x, half_y],
        ],
        dtype=torch.float64,
        device=sensor_pos.device,
    )
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    world_x = sensor_pos[0] + local_corners[:, 0] * cos_yaw - local_corners[:, 1] * sin_yaw
    world_y = sensor_pos[1] + local_corners[:, 0] * sin_yaw + local_corners[:, 1] * cos_yaw
    return (float(world_x.min().item()), float(world_x.max().item())), (float(world_y.min().item()), float(world_y.max().item()))


def _compute_mpc_local_terrain(scanner, *, env_id: int = 0):
    from extension.convention import extract_yaw_batch
    from extension.batch_mpc_planner.terrain import build_mpc_terrain_from_scanner

    ray_hits = scanner.data.ray_hits_w[env_id].to(dtype=torch.float32)
    semantic_map = _scanner_semantic_map(scanner, env_id=env_id)
    pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
    size = getattr(pattern_cfg, "size", None)
    if size is None:
        world_x_range, world_y_range = (-0.75, 0.75), (-0.75, 0.75)
    else:
        world_x_range = (-0.5 * float(size[0]), 0.5 * float(size[0]))
        world_y_range = (-0.5 * float(size[1]), 0.5 * float(size[1]))
    sensor_pos = torch.as_tensor(scanner.data.pos_w[env_id], dtype=torch.float32, device=ray_hits.device).unsqueeze(0)
    sensor_quat = torch.as_tensor(scanner.data.quat_w[env_id], dtype=torch.float32, device=ray_hits.device).unsqueeze(0)
    terrain = build_mpc_terrain_from_scanner(
        ray_hits.unsqueeze(0),
        world_x_range=world_x_range,
        world_y_range=world_y_range,
        semantic_map=semantic_map,
        sensor_pos_w=sensor_pos,
        sensor_yaw=extract_yaw_batch(sensor_quat),
    )
    return terrain, ray_hits


def _reshape_scanner_grid(ray_hits: torch.Tensor) -> tuple[torch.Tensor, int]:
    if ray_hits.ndim != 2 or ray_hits.shape[-1] != 3:
        raise ValueError("ray_hits must have shape (H*W, 3)")
    side = int(round(math.sqrt(int(ray_hits.shape[0]))))
    if side * side != int(ray_hits.shape[0]):
        raise ValueError(f"ray hit count {int(ray_hits.shape[0])} is not a perfect square")
    return ray_hits.reshape(side, side, 3), side


def _subsample_semantic_height_points(
    ray_hits: torch.Tensor,
    semantic_map: torch.Tensor | None,
    stride: int,
) -> tuple[dict[int, torch.Tensor], dict[str, float | int]]:
    grid, side = _reshape_scanner_grid(torch.as_tensor(ray_hits))
    if semantic_map is None:
        semantic_grid = torch.full((side, side), SEMANTIC_TERRAIN_ID, dtype=torch.long, device=grid.device)
    else:
        semantic_grid = torch.as_tensor(semantic_map, device=grid.device)
        if semantic_grid.ndim != 2 or tuple(semantic_grid.shape) != (side, side):
            raise ValueError(f"semantic_map must have shape {(side, side)}, got {tuple(semantic_grid.shape)}")
        semantic_grid = semantic_grid.to(dtype=torch.long)

    step = max(1, int(stride))
    sampled_hits = grid[::step, ::step].reshape(-1, 3)
    sampled_semantic = semantic_grid[::step, ::step].reshape(-1)
    valid_mask = torch.isfinite(sampled_hits).all(dim=-1)

    points_by_class: dict[int, torch.Tensor] = {}
    for semantic_id in (SEMANTIC_TERRAIN_ID, SEMANTIC_SMALL_ID, SEMANTIC_LARGE_ID):
        class_mask = valid_mask & (sampled_semantic == semantic_id)
        points_by_class[semantic_id] = sampled_hits[class_mask]

    valid_hits = sampled_hits[valid_mask]
    terrain_hits = points_by_class[SEMANTIC_TERRAIN_ID]
    if terrain_hits.shape[0] > 0:
        baseline_z = terrain_hits[:, 2].median()
    elif valid_hits.shape[0] > 0:
        baseline_z = valid_hits[:, 2].amin()
    else:
        baseline_z = torch.tensor(0.0, dtype=sampled_hits.dtype, device=sampled_hits.device)

    height_lift_max = 0.0
    if valid_hits.shape[0] > 0:
        height_lift_max = float(torch.clamp(valid_hits[:, 2] - baseline_z, min=0.0).amax().item())

    diagnostics: dict[str, float | int] = {
        "terrain_hit_count": int(points_by_class[SEMANTIC_TERRAIN_ID].shape[0]),
        "small_hit_count": int(points_by_class[SEMANTIC_SMALL_ID].shape[0]),
        "large_hit_count": int(points_by_class[SEMANTIC_LARGE_ID].shape[0]),
        "valid_sample_count": int(valid_hits.shape[0]),
        "height_lift_max": height_lift_max,
    }
    return points_by_class, diagnostics


def _subsample_height_points(ray_hits: torch.Tensor, stride: int) -> torch.Tensor:
    try:
        return _subsample_semantic_height_points(ray_hits, None, stride)[0][SEMANTIC_TERRAIN_ID]
    except ValueError:
        sampled = torch.as_tensor(ray_hits)[:: max(1, int(stride))]
        valid = torch.isfinite(sampled).all(dim=-1)
        return sampled[valid]


def _reference_height_scanner_name(env_cfg) -> str:
    return str(getattr(env_cfg, "reference_height_scanner_name", "height_scanner"))


def _reference_height_scanner(base_env, env_cfg):
    return base_env.scene.sensors[_reference_height_scanner_name(env_cfg)]


def _scanner_semantic_map(scanner, *, env_id: int = 0) -> torch.Tensor | None:
    semantic_map = getattr(scanner.data, "semantic_map", None)
    if semantic_map is None:
        return None
    return torch.as_tensor(semantic_map[env_id])


def _format_semantic_diagnostics(diagnostics: dict[str, float | int]) -> str:
    return (
        "semantic("
        f"terrain={int(diagnostics['terrain_hit_count'])} "
        f"small={int(diagnostics['small_hit_count'])} "
        f"large={int(diagnostics['large_hit_count'])} "
        f"valid={int(diagnostics['valid_sample_count'])} "
        f"height_lift_max={float(diagnostics['height_lift_max']):0.3f}"
        ")"
    )


def _format_viewer_status(result) -> str:
    status = getattr(result, "status", None)
    if status is None:
        return ""
    status_value = int(torch.as_tensor(status).reshape(-1)[0].item())
    status_names = getattr(result, "status_names", None)
    if status_names is not None and 0 <= status_value < len(status_names):
        return f"status={status_names[status_value]}"
    return f"status={status_value}"


def _format_viewer_hard_reason_diagnostics(result) -> str:
    hard_reason_mask = getattr(result, "hard_reason_mask", None)
    if hard_reason_mask is not None:
        hard_reason_t = torch.as_tensor(hard_reason_mask, dtype=torch.bool)
        if hard_reason_t.ndim == 1:
            hard_reason_t = hard_reason_t.unsqueeze(0)
        status = getattr(result, "status", None)
        all_infeasible = False
        if status is not None:
            status_value = int(torch.as_tensor(status).reshape(-1)[0].item())
            status_names = getattr(result, "status_names", None)
            if status_names is not None and 0 <= status_value < len(status_names):
                all_infeasible = str(status_names[status_value]) == "ALL_INFEASIBLE"
            else:
                all_infeasible = False
        selected = hard_reason_t.reshape(-1, hard_reason_t.shape[-1])[0]
        selected_nonempty = bool(torch.any(selected).item())
        if not all_infeasible and not selected_nonempty:
            return ""
        reason_names = tuple(getattr(result, "hard_reason_names", ()))
        return f"hard_reasons={_format_hard_reason_mask(selected, names=reason_names)}"

    selected_mask = getattr(result, "selected_hard_reason_mask", None)
    candidate_mask = getattr(result, "candidate_hard_reason_mask", None)
    candidate_rank = getattr(result, "candidate_hard_rank_cost", None)
    selected_rank = getattr(result, "selected_hard_rank_cost", None)
    if selected_mask is None or candidate_mask is None or candidate_rank is None or selected_rank is None:
        return ""

    selected_mask_t = torch.as_tensor(selected_mask, dtype=torch.bool)
    candidate_mask_t = torch.as_tensor(candidate_mask, dtype=torch.bool)
    candidate_rank_t = torch.as_tensor(candidate_rank, dtype=torch.float64)
    selected_rank_t = torch.as_tensor(selected_rank, dtype=torch.float64)
    status = getattr(result, "status", None)
    all_infeasible = False
    if status is not None:
        status_names = getattr(result, "status_names", None)
        status_value = int(torch.as_tensor(status).reshape(-1)[0].item())
        all_infeasible = bool(status_names is not None and 0 <= status_value < len(status_names) and str(status_names[status_value]) == "ALL_INFEASIBLE")
    selected_nonempty = bool(torch.any(selected_mask_t.reshape(-1, selected_mask_t.shape[-1])[0]).item())
    if not all_infeasible and not selected_nonempty:
        return ""

    reason_names = tuple(getattr(result, "hard_reason_names", ()))
    selected_reasons = _format_hard_reason_mask(selected_mask_t.reshape(-1, selected_mask_t.shape[-1])[0], names=reason_names)
    candidate_reasons = [
        _format_hard_reason_mask(
            candidate_mask_t.reshape(-1, candidate_mask_t.shape[-2], candidate_mask_t.shape[-1])[0, idx],
            names=reason_names,
        )
        for idx in range(candidate_mask_t.shape[-2])
    ]
    rank_values = [f"{float(value):0.3f}" for value in candidate_rank_t.reshape(-1, candidate_rank_t.shape[-1])[0].tolist()]
    return (
        f"selected_hard_reasons={selected_reasons} "
        f"selected_hard_rank_cost={float(selected_rank_t.reshape(-1)[0].item()):0.3f} "
        f"candidate_hard_rank=[{','.join(rank_values)}] "
        f"candidate_hard_reasons=[{';'.join(candidate_reasons)}]"
    )


def _format_viewer_plan_line(
    *,
    backend: str,
    cycle: int,
    command: torch.Tensor,
    result,
    semantic_diagnostics: dict[str, float | int],
) -> str:
    summary = _trajectory_motion_summary(result)
    parts = [
        "[Viewer][Plan]",
        f"backend={backend}",
        f"cycle={cycle}",
        f"cmd={_format_command_values(command)}",
        f"delta=({summary['dx']:+0.2f}, {summary['dy']:+0.2f}, {summary['dz']:+0.2f})",
        f"dyaw={summary['dyaw']:+0.2f}",
        f"standstill={summary['standstill']}",
    ]
    status_text = _format_viewer_status(result)
    if status_text:
        parts.append(status_text)
    if semantic_diagnostics:
        parts.append(_format_semantic_diagnostics(semantic_diagnostics))
    hard_text = _format_viewer_hard_reason_diagnostics(result)
    if hard_text:
        parts.append(hard_text)
    return " ".join(parts)


def _make_zero_actions(env) -> torch.Tensor:
    import gymnasium as gym

    if hasattr(env, "action_manager"):
        action_dim = int(env.action_manager.total_action_dim)
    else:
        action_dim = int(gym.spaces.flatdim(env.single_action_space))
    return torch.zeros((env.num_envs, action_dim), dtype=torch.float32, device=env.device)


def _update_camera(env, *, root_pos: torch.Tensor, root_yaw: torch.Tensor, distance: float, height: float) -> None:
    yaw_val = float(root_yaw[0].item())
    camera_offset = torch.tensor(
        [-distance * math.cos(yaw_val), -distance * math.sin(yaw_val), height],
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    camera_position = (root_pos + camera_offset).detach().cpu().numpy()
    target_position = (root_pos + torch.tensor([0.0, 0.0, 0.35], device=root_pos.device)).detach().cpu().numpy()
    env.sim.set_camera_view(camera_position, target_position)


def _mpc_state_from_env(env, foot_ids: list[int]):
    from extension.batch_mpc_planner.types import MpcRobotState
    from extension.convention import extract_roll_pitch_batch, extract_yaw_batch

    robot = env.scene["robot"]
    root_quat = torch.as_tensor(robot.data.root_quat_w[:1], dtype=torch.float64)
    roll, pitch = extract_roll_pitch_batch(root_quat)
    yaw = extract_yaw_batch(root_quat)
    # Keep viewer MPC joint ordering consistent with legacy/together playback
    # contract so direct playback does not scramble leg pose.
    joint_pos_planner = _joint_pos_robot_to_planner(
        robot,
        torch.as_tensor(robot.data.joint_pos[:1], dtype=torch.float64),
    )
    foot_ids_t = torch.as_tensor(_foot_id_list(foot_ids), dtype=torch.long, device=robot.data.body_pos_w.device)
    foot_pos = torch.as_tensor(robot.data.body_pos_w[:1], dtype=torch.float64).index_select(1, foot_ids_t)
    foot_vel = torch.as_tensor(robot.data.body_lin_vel_w[:1], dtype=torch.float64).index_select(1, foot_ids_t)
    foot_pos = _reorder_feet_to_planner_order(robot, foot_ids_t, foot_pos)
    foot_vel = _reorder_feet_to_planner_order(robot, foot_ids_t, foot_vel)
    return MpcRobotState(
        root_pos=torch.as_tensor(robot.data.root_pos_w[:1], dtype=torch.float64),
        root_rpy=torch.stack((roll, pitch, yaw), dim=-1),
        joint_angles=joint_pos_planner,
        foot_pos=foot_pos,
        foot_vel=foot_vel,
    )


def _adapt_mpc_result_for_viewer(result) -> ViewerTrajectoryResult:
    from extension.batch_mpc_planner.types import MPC_HARD_REASON_NAMES, MpcPlannerStatus
    from extension.convention import euler_to_quat_batch

    root_pos_w = torch.as_tensor(result.root_pos).detach().contiguous()
    root_rpy = torch.as_tensor(result.root_rpy, device=root_pos_w.device, dtype=root_pos_w.dtype).detach()
    root_quat_w = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2]).contiguous()
    foot_pos_w = torch.as_tensor(result.foot_pos, device=root_pos_w.device, dtype=root_pos_w.dtype).detach().contiguous()
    foot_pos_root = (foot_pos_w - root_pos_w.unsqueeze(2)).contiguous()
    touchdown_w = torch.as_tensor(result.planned_touchdown_w, device=root_pos_w.device, dtype=root_pos_w.dtype).detach()
    if touchdown_w.ndim == 4:
        touchdown_w = touchdown_w[:, 0]
    elif not (touchdown_w.ndim == 3 and tuple(touchdown_w.shape[-2:]) == (4, 3)):
        touchdown_w = torch.as_tensor(result.touchdown_seq[:, :, 0, :], device=root_pos_w.device, dtype=root_pos_w.dtype).detach()
    hard_mask = getattr(result, "hard_reason_mask", None)
    hard_reason_mask = None if hard_mask is None else torch.as_tensor(hard_mask, device=root_pos_w.device, dtype=torch.bool).detach()
    loss_breakdown = getattr(result, "loss_breakdown", None)
    if loss_breakdown is None:
        loss_breakdown = getattr(result, "cost_breakdown", None)
    if loss_breakdown is not None:
        loss_breakdown = {
            str(name): torch.as_tensor(value, device=root_pos_w.device).detach().contiguous()
            for name, value in loss_breakdown.items()
        }
    zeros_vel = torch.zeros_like(root_pos_w)
    return ViewerTrajectoryResult(
        num_frames=int(root_pos_w.shape[1]),
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        joint_angles=torch.as_tensor(result.joint_angles, device=root_pos_w.device, dtype=root_pos_w.dtype).detach().contiguous(),
        foot_pos_w=foot_pos_w,
        foot_pos_root=foot_pos_root,
        contact_state=torch.as_tensor(result.contact_state, device=root_pos_w.device).detach().contiguous(),
        planned_touchdown_w=touchdown_w.contiguous(),
        touchdown_seq=(
            None
            if getattr(result, "touchdown_seq", None) is None
            else torch.as_tensor(result.touchdown_seq, device=root_pos_w.device, dtype=root_pos_w.dtype).detach().contiguous()
        ),
        root_lin_vel_w=zeros_vel,
        root_ang_vel_w=zeros_vel.clone(),
        status=torch.as_tensor(result.status, device=root_pos_w.device).detach(),
        feasible=torch.as_tensor(result.feasible, device=root_pos_w.device, dtype=torch.bool).detach(),
        safe_fallback=torch.as_tensor(result.safe_fallback, device=root_pos_w.device, dtype=torch.bool).detach(),
        hard_reason_mask=hard_reason_mask,
        hard_reason_names=tuple(MPC_HARD_REASON_NAMES),
        status_names=tuple(status.name for status in MpcPlannerStatus),
        loss_breakdown=loss_breakdown,
    )


def _adapt_joint_mpc_rti_result_for_viewer(trajectory) -> ViewerTrajectoryResult:
    from extension.convention import euler_to_quat_batch
    from extension.joint_mpc_rti.integration.reference_adapter import trajectory_to_reference_cache

    state = torch.as_tensor(trajectory.state).detach().contiguous()
    root_pos_w = state[..., :3]
    root_rpy = state[..., 3:6]
    root_quat_w = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2]).contiguous()
    foot_pos_w = torch.as_tensor(trajectory.foot_pos_w, dtype=state.dtype, device=state.device).detach().contiguous()
    reference_cache = trajectory_to_reference_cache(trajectory)
    zeros_vel = torch.zeros_like(root_pos_w)
    return ViewerTrajectoryResult(
        num_frames=int(state.shape[1]),
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        joint_angles=state[..., 6:],
        foot_pos_w=foot_pos_w,
        foot_pos_root=reference_cache.foot_pos_root,
        contact_state=torch.as_tensor(trajectory.contact_state, device=state.device).detach().contiguous(),
        planned_touchdown_w=foot_pos_w,
        root_lin_vel_w=zeros_vel,
        root_ang_vel_w=zeros_vel.clone(),
        status=torch.as_tensor(trajectory.status, device=state.device).detach(),
        feasible=torch.as_tensor(trajectory.valid, dtype=torch.bool, device=state.device).detach(),
        safe_fallback=torch.as_tensor(trajectory.fallback, dtype=torch.bool, device=state.device).detach(),
        status_names=("SUCCESS", "FALLBACK", "INVALID"),
        loss_breakdown={
            str(name): torch.as_tensor(value, device=state.device).detach().contiguous()
            for name, value in trajectory.loss_breakdown.items()
        },
    )


def _plan_joint_viewer_trajectory(*, manager, env, command: torch.Tensor) -> ViewerTrajectoryResult:
    manager.refresh_from_env(env, command_body=command, force=True)
    return _adapt_joint_mpc_rti_result_for_viewer(manager.latest_trajectory())


def _plan_viewer_trajectory(
    *,
    terrain,
    state,
    command: torch.Tensor,
    mpc_cfg,
):
    from extension.batch_mpc_planner.planner import plan_segment

    return _adapt_mpc_result_for_viewer(
        plan_segment(
            terrain,
            state,
            command,
            cfg=mpc_cfg,
        )
    )


def _print_help() -> None:
    print("\nTerminal teleop (hold keys with repeat):", flush=True)
    print("  W/S : forward/backward", flush=True)
    print("  A/D : lateral left/right", flush=True)
    print("  Q/E : yaw left/right", flush=True)
    print("  X   : clear command", flush=True)
    print("  R   : reset environment", flush=True)
    print("  M   : toggle step mode", flush=True)
    print("  Space : step one frame when step mode is enabled", flush=True)
    print("  Ctrl-C : quit\n", flush=True)


def main() -> int:
    args_cli = _prepare_runtime_args(_parse_args())
    _, simulation_app = _launch_app(args_cli)

    import gymnasium as gym

    import go2_pvcnn.tasks.register_envs  # noqa: F401
    from extension.convention import extract_yaw_batch
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = _build_env_cfg(args_cli)
    joint_backend = str(args_cli.planner_backend).lower() == "joint_mpc_rti"
    mpc_planner_cfg = None if joint_backend else _build_mpc_planner_cfg(env_cfg, args_cli=args_cli)

    env = gym.make(
        "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if getattr(args_cli, "livestream", -1) in (1, 2) else None,
    )
    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env = env.unwrapped
    trajectory_manager = _attach_reference_manager_if_enabled(base_env, env_cfg)
    zero_actions = _make_zero_actions(base_env)
    foot_ids, _ = base_env.scene["robot"].find_bodies(".*_foot")
    scanner = _reference_height_scanner(base_env, env_cfg)
    visualizer = PlannerVisualizer()

    print("[Viewer] Terrain source: teacher_elevation_trajectory env config", flush=True)
    print(f"[Viewer] Planner horizon: {args_cli.n_frames} frames @ dt={args_cli.plan_dt:.3f}s", flush=True)
    print("[Viewer] Playback mode: kinematic (no physics)", flush=True)
    print("[Viewer] Step mode: disabled; press M to toggle, Space advances one frame while enabled.", flush=True)
    _print_help()

    selected_origin = _reset_viewer_env(
        env,
        base_env=base_env,
        zero_actions=zero_actions,
        warmup_steps=int(args_cli.warmup_steps),
        terrain_row=int(args_cli.terrain_row),
        terrain_col=int(args_cli.terrain_col),
        scanner=scanner,
        foot_ids=foot_ids,
    )
    reset_snapshot = _viewer_capture_reset_snapshot(base_env)
    print(
        "[Viewer] Terrain tile override: "
        f"row={int(args_cli.terrain_row)} "
        f"col={int(args_cli.terrain_col)} "
        f"origin={_format_xyz(selected_origin)}",
        flush=True,
    )

    result = None
    playback_frame = 0
    last_cmd = None
    plan_cycle = 0
    scripted_cycles_remaining = max(0, int(args_cli.scripted_command_cycles))
    scripted_command = _parse_scripted_command(args_cli.scripted_command, device=base_env.device)
    step_gate = ViewerStepGate(enabled=False)

    with TerminalTeleop(
        device=base_env.device,
        vx_scale=float(args_cli.vx_scale),
        vy_scale=float(args_cli.vy_scale),
        yaw_scale=float(args_cli.yaw_scale),
        timeout_s=float(args_cli.key_hold_timeout),
    ) as teleop:
        last_status = None
        last_loop_diag = None
        last_playback_path = None
        last_actual_summary = None
        last_kinematic_summary = None
        try:
            while True:
                teleop_cmd = teleop.poll()
                active_cmd = teleop_cmd
                if scripted_command is not None and scripted_cycles_remaining > 0:
                    active_cmd = TeleopCommand(
                        values=scripted_command.clone(),
                        reset_requested=teleop_cmd.reset_requested,
                        step_requested=teleop_cmd.step_requested,
                        mode_toggle_requested=teleop_cmd.mode_toggle_requested,
                    )

                if active_cmd.mode_toggle_requested:
                    enabled = step_gate.toggle_enabled()
                    print(f"[Viewer][StepMode] {'enabled' if enabled else 'disabled'}", flush=True)

                active_cmd = replace(
                    active_cmd,
                    values=_viewer_select_active_teleop_values(
                        live_values=active_cmd.values,
                        latched_values=teleop.latched_command(),
                        step_mode_enabled=step_gate.enabled,
                    ),
                )

                if active_cmd.reset_requested:
                    selected_origin = _reset_viewer_env(
                        env,
                        base_env=base_env,
                        zero_actions=zero_actions,
                        warmup_steps=int(args_cli.warmup_steps),
                        terrain_row=int(args_cli.terrain_row),
                        terrain_col=int(args_cli.terrain_col),
                        scanner=scanner,
                        reset_snapshot=reset_snapshot,
                        foot_ids=foot_ids,
                    )
                    print(
                        "[Viewer][Reset] "
                        f"row={int(args_cli.terrain_row)} "
                        f"col={int(args_cli.terrain_col)} "
                        f"origin={_format_xyz(selected_origin)}",
                        flush=True,
                    )
                    result = None
                    playback_frame = 0
                    last_cmd = None
                    plan_cycle = 0

                need_replan = joint_backend or _viewer_loop_need_replan(
                    result=result,
                    playback_frame=playback_frame,
                    reset_requested=active_cmd.reset_requested,
                    teleop_values=active_cmd.values,
                    last_cmd=last_cmd,
                    defer_command_replan_until_trajectory_end=step_gate.enabled,
                )
                drain_zero_replan = False
                if need_replan and _viewer_should_drain_before_zero_replan(
                    backend=args_cli.planner_backend,
                    result=result,
                    playback_frame=playback_frame,
                    teleop_values=active_cmd.values,
                    last_cmd=last_cmd,
                ):
                    drain_terrain, _ = _compute_mpc_local_terrain(scanner)
                    landing_frame = _viewer_find_grounded_all_feet_frame(
                        result,
                        drain_terrain,
                        start_frame=playback_frame,
                    )
                    if landing_frame is None:
                        drain_zero_replan = True
                    elif playback_frame <= landing_frame:
                        drain_zero_replan = True
                    if drain_zero_replan:
                        need_replan = False
                loop_diag = (_format_command_values(active_cmd.values), need_replan)
                if loop_diag != last_loop_diag:
                    # print(
                    #     "[Viewer][Loop] "
                    #     f"teleop_cmd={loop_diag[0]} "
                    #     f"need_replan={need_replan} "
                    #     f"playback_frame={playback_frame} "
                    #     f"cycle={plan_cycle}",
                    #     flush=True,
                    # )
                    last_loop_diag = loop_diag

                frame_permitted = step_gate.consume_frame_permission(step_requested=active_cmd.step_requested)
                if step_gate.enabled and not frame_permitted and not active_cmd.reset_requested:
                    _viewer_pump_paused_window(base_env)
                    continue

                if need_replan:
                    if joint_backend:
                        if trajectory_manager is None:
                            raise RuntimeError("joint_mpc_rti viewer requires an attached trajectory manager")
                        result = _plan_joint_viewer_trajectory(
                            manager=trajectory_manager,
                            env=base_env,
                            command=active_cmd.values,
                        )
                        ray_hits = torch.as_tensor(scanner.data.ray_hits_w[0], dtype=torch.float32)
                    else:
                        state = _mpc_state_from_env(base_env, foot_ids)
                        terrain, ray_hits = _compute_mpc_local_terrain(scanner)
                        result = _plan_viewer_trajectory(
                            terrain=terrain,
                            state=state,
                            command=active_cmd.values,
                            mpc_cfg=mpc_planner_cfg,
                        )
                    summary = _trajectory_motion_summary(result)
                    semantic_map = _scanner_semantic_map(scanner)
                    height_points_by_class, semantic_diag = _subsample_semantic_height_points(
                        ray_hits,
                        semantic_map,
                        int(args_cli.heightmap_viz_stride),
                    )
                    # print(
                    #     _format_viewer_plan_line(
                    #         backend=args_cli.planner_backend,
                    #         cycle=plan_cycle,
                    #         command=active_cmd.values,
                    #         result=result,
                    #         semantic_diagnostics=semantic_diag,
                    #     ),
                    #     flush=True,
                    # )
                    playback_frame = 0

                    def _update_step_visualizer() -> None:
                        planner_state = _planner_state_from_reference_result(result, frame_idx=0)
                        root_yaw = planner_state.root_rpy[:, 2]
                        visualizer.update(
                            result=result,
                            command=active_cmd.values,
                            root_yaw=root_yaw,
                            height_points_by_class=height_points_by_class,
                        )

                    _viewer_update_visualizer_when_permitted(
                        frame_permitted=(frame_permitted or not step_gate.enabled),
                        update_fn=_update_step_visualizer,
                    )
                    plan_cycle += 1
                    if scripted_command is not None and scripted_cycles_remaining > 0:
                        scripted_cycles_remaining = max(0, scripted_cycles_remaining - 1)

                if result is not None and playback_frame < result.num_frames and frame_permitted:
                    applied_frame = _viewer_playback_frame_index(
                        args_cli.planner_backend,
                        playback_frame=playback_frame,
                    )
                    playback_path = _viewer_direct_playback_step(base_env, result, frame_idx=applied_frame)
                    if playback_path != last_playback_path:
                        print(
                            f"[Viewer][Playback] path={playback_path}",
                            flush=True,
                        )
                        last_playback_path = playback_path
                    actual = _read_actual_base_state(base_env)
                    planner_frame = _planner_state_from_reference_result(result, frame_idx=applied_frame)
                    actual_summary = (
                        _format_xyz(actual["root_pos_w"][0]),
                        _format_quat(actual["root_quat_raw"][0]),
                        _format_xyz(actual["rpy_if_wxyz"][0]),
                        _format_xyz(actual["rpy_if_xyzw"][0]),
                        _format_xyz(planner_frame.root_pos[0]),
                        _format_xyz(planner_frame.root_rpy[0]),
                    )
                    if actual_summary != last_actual_summary:
                        # print(
                        #     "[Viewer][ActualBase] "
                        #     f"cycle={max(plan_cycle - 1, 0)} "
                        #     f"actual_pos={actual_summary[0]} "
                        #     f"actual_quat_raw={actual_summary[1]} "
                        #     f"actual_rpy_if_wxyz={actual_summary[2]} "
                        #     f"actual_rpy_if_xyzw={actual_summary[3]} "
                        #     f"plan_pos={actual_summary[4]} "
                        #     f"plan_rpy={actual_summary[5]}",
                        #     flush=True,
                        # )
                        last_actual_summary = actual_summary
                    actual_kin = _read_actual_kinematic_state(base_env, foot_ids)
                    joint_err = actual_kin["joint_pos_planner"] - planner_frame.joint_angles
                    foot_err = actual_kin["foot_pos_w"] - planner_frame.foot_pos
                    foot_err_norm = torch.linalg.vector_norm(foot_err, dim=-1)
                    kinematic_summary = (
                        float(joint_err.abs().max().item()),
                        float(joint_err.abs().mean().item()),
                        float(foot_err_norm.max().item()),
                        float(foot_err_norm.mean().item()),
                    )
                    if kinematic_summary != last_kinematic_summary:
                        # print(
                        #     "[Viewer][ActualKinematics] "
                        #     f"cycle={max(plan_cycle - 1, 0)} "
                        #     f"joint_err_max={kinematic_summary[0]:0.6f} "
                        #     f"joint_err_mean={kinematic_summary[1]:0.6f} "
                        #     f"foot_err_max={kinematic_summary[2]:0.6f} "
                        #     f"foot_err_mean={kinematic_summary[3]:0.6f}",
                        #     flush=True,
                        # )
                        last_kinematic_summary = kinematic_summary
                    playback_frame += 1
                    if drain_zero_replan and landing_frame is not None and playback_frame > landing_frame:
                        last_cmd = active_cmd.values.clone()
                        result = None
                        continue
                    if drain_zero_replan:
                        continue
                elif result is not None and playback_frame < result.num_frames and step_gate.enabled:
                    time.sleep(0.01)

                last_cmd = active_cmd.values.clone()

                if result is not None:
                    display_frame = _viewer_playback_frame_index(
                        args_cli.planner_backend,
                        playback_frame=(min(playback_frame - 1, result.num_frames - 1) if playback_frame > 0 else 0),
                    )
                    planner_state = _planner_state_from_reference_result(result, frame_idx=display_frame)
                    root_yaw = planner_state.root_rpy[:, 2]
                    _update_camera(
                        base_env,
                        root_pos=planner_state.root_pos[0],
                        root_yaw=root_yaw,
                        distance=float(args_cli.camera_distance),
                        height=float(args_cli.camera_height),
                    )

                    root_pos = planner_state.root_pos[0]
                    yaw_rate = float(active_cmd.values[0, 2].item())
                    actual = _read_actual_base_state(base_env)
                    actual_pos = actual["root_pos_w"][0]
                    actual_rpy_wxyz = actual["rpy_if_wxyz"][0]
                    actual_rpy_xyzw = actual["rpy_if_xyzw"][0]
                    status = (
                        f"\rcycle={max(plan_cycle - 1, 0)} "
                        f"cmd vx={active_cmd.values[0,0]:+0.2f} "
                        f"vy={active_cmd.values[0,1]:+0.2f} "
                        f"yaw={yaw_rate:+0.2f} | "
                        f"plan=({root_pos[0]:+0.2f}, {root_pos[1]:+0.2f}, {root_pos[2]:+0.2f}) "
                        f"actual=({actual_pos[0]:+0.2f}, {actual_pos[1]:+0.2f}, {actual_pos[2]:+0.2f}) "
                        f"actual_rpy_wxyz=({actual_rpy_wxyz[0]:+0.2f}, {actual_rpy_wxyz[1]:+0.2f}, {actual_rpy_wxyz[2]:+0.2f}) "
                        f"actual_rpy_xyzw_dbg=({actual_rpy_xyzw[0]:+0.2f}, {actual_rpy_xyzw[1]:+0.2f}, {actual_rpy_xyzw[2]:+0.2f}) "
                        f"frame={display_frame}/{result.num_frames}"
                    )
                    # if status != last_status:
                    #     sys.stdout.write(status)
                    #     sys.stdout.flush()
                    #     last_status = status
        except KeyboardInterrupt:
            print("\n[Viewer] Ctrl-C received; shutting down.", flush=True)
        finally:
            print()
            env.close()

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
