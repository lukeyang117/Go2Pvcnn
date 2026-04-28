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
    root_lin_vel_w: torch.Tensor | None = None
    root_ang_vel_w: torch.Tensor | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Visualize batched Go2 footstep planning in Isaac Lab.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
    parser.add_argument(
        "--terrain",
        type=str,
        default="task",
        choices=["task"],
        help="Use the terrain generator exactly as defined by teacher_elevation_trajectory env config.",
    )
    parser.add_argument("--n-frames", type=int, default=35, help="Planner horizon in frames.")
    parser.add_argument("--plan-dt", type=float, default=0.02, help="Planner integration step.")
    parser.add_argument(
        "--planner-backend",
        type=str,
        default="together",
        choices=["together", "legacy"],
        help="Trajectory manager backend used by the task attachment path.",
    )
    parser.add_argument("--vx-scale", type=float, default=0.4, help="Teleop forward/backward speed.")
    parser.add_argument("--vy-scale", type=float, default=0.4, help="Teleop lateral speed.")
    parser.add_argument("--yaw-scale", type=float, default=0.3, help="Teleop yaw-rate command.")
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
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _prepare_runtime_args(args_cli: argparse.Namespace) -> argparse.Namespace:
    if getattr(args_cli, "livestream", -1) in (1, 2) and not args_cli.enable_cameras:
        args_cli.enable_cameras = True
        print(
            "[INFO][go2_foostep_planner.py] livestream: enabled AppLauncher --enable_cameras "
            "for WebRTC rendering.",
            flush=True,
        )
    return args_cli


def _planner_state_from_reference_result(result, *, frame_idx: int):
    """Build a planner-facing state snapshot from a trajectory result.

    The planner uses wxyz convention; result stores wxyz quaternions.
    """
    from extension.batched_planner.types import BatchedRobotState

    frame = int(frame_idx)
    root_pos = torch.as_tensor(result.root_pos_w[:, frame], dtype=torch.float64)
    root_quat = torch.as_tensor(result.root_quat_w[:, frame], dtype=torch.float64)
    joint_angles = torch.as_tensor(result.joint_angles[:, frame], dtype=torch.float64)
    foot_pos = torch.as_tensor(result.foot_pos_w[:, frame], dtype=torch.float64)
    return BatchedRobotState(
        root_pos=root_pos,
        root_quat=root_quat,
        joint_angles=joint_angles,
        foot_pos=foot_pos,
        foot_vel=torch.zeros_like(foot_pos),
    )


def _together_handoff_root_pos(result, *, frame_idx: int) -> torch.Tensor:
    root_pos_all = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
    root_pos = root_pos_all[:, int(frame_idx)].clone()
    contact_state = getattr(result, "contact_state", None)
    foot_pos_w = getattr(result, "foot_pos_w", None)
    if contact_state is None or foot_pos_w is None:
        return root_pos

    foot_pos = torch.as_tensor(foot_pos_w, device=root_pos.device, dtype=root_pos.dtype)
    contact = torch.as_tensor(contact_state, device=root_pos.device)
    if foot_pos.ndim != 4 or contact.ndim != 3 or foot_pos.shape[:3] != contact.shape:
        return root_pos

    frame = int(frame_idx)
    root_quat_w = getattr(result, "root_quat_w", None)
    hold_like = torch.zeros((root_pos.shape[0],), dtype=torch.bool, device=root_pos.device)
    if root_quat_w is not None:
        root_quat = torch.as_tensor(root_quat_w, device=root_pos.device, dtype=root_pos.dtype)
        if root_quat.ndim == 3 and root_quat.shape[:2] == root_pos_all.shape[:2]:
            # Hold-like segments should hand off their settled base height directly;
            # reconstructing support clearance would replay the recovery each replan.
            full_contact = (contact > 0.5).all(dim=2).all(dim=1)
            planar_delta = torch.linalg.vector_norm(root_pos_all[:, frame, :2] - root_pos_all[:, 0, :2], dim=-1)
            yaw_delta = torch.abs(_quat_wxyz_to_yaw(root_quat[:, frame]) - _quat_wxyz_to_yaw(root_quat[:, 0]))
            hold_like = full_contact & (planar_delta <= 1e-6) & (yaw_delta <= 1e-6)

    frame_contact = contact[:, frame].to(dtype=root_pos.dtype)
    initial_contact = contact[:, 0].to(dtype=root_pos.dtype)
    frame_contact_count = frame_contact.sum(dim=-1)
    initial_contact_count = initial_contact.sum(dim=-1)
    frame_support_z = torch.where(
        frame_contact_count > 0.0,
        (foot_pos[:, frame, :, 2] * frame_contact).sum(dim=-1) / frame_contact_count.clamp_min(1.0),
        foot_pos[:, frame, :, 2].mean(dim=-1),
    )
    initial_support_z = torch.where(
        initial_contact_count > 0.0,
        (foot_pos[:, 0, :, 2] * initial_contact).sum(dim=-1) / initial_contact_count.clamp_min(1.0),
        foot_pos[:, 0, :, 2].mean(dim=-1),
    )
    initial_clearance = root_pos_all[:, 0, 2].to(device=root_pos.device, dtype=root_pos.dtype) - initial_support_z
    reconstructed_z = frame_support_z + initial_clearance
    root_pos[:, 2] = torch.where(hold_like, root_pos_all[:, frame, 2], reconstructed_z)
    return root_pos


def _together_state_from_reference_result(result, *, frame_idx: int):
    from extension.batched_together_planner.types import TogetherRobotState

    frame = int(frame_idx)
    root_pos = _together_handoff_root_pos(result, frame_idx=frame)
    root_quat = torch.as_tensor(result.root_quat_w[:, frame], dtype=torch.float64)
    root_rpy = _quat_wxyz_to_rpy(root_quat)
    joint_angles = torch.as_tensor(result.joint_angles[:, frame], dtype=torch.float64)
    foot_pos = torch.as_tensor(result.foot_pos_w[:, frame], dtype=torch.float64)
    return TogetherRobotState(
        root_pos=root_pos,
        root_rpy=root_rpy,
        joint_angles=joint_angles,
        foot_pos=foot_pos,
        foot_vel=torch.zeros_like(foot_pos),
    )


def _legacy_state_to_together_state(state):
    from extension.batched_together_planner.types import TogetherRobotState

    root_pos = torch.as_tensor(state.root_pos, dtype=torch.float64)
    root_quat = torch.as_tensor(state.root_quat, dtype=torch.float64)
    return TogetherRobotState(
        root_pos=root_pos,
        root_rpy=_quat_wxyz_to_rpy(root_quat),
        joint_angles=torch.as_tensor(state.joint_angles, dtype=torch.float64),
        foot_pos=torch.as_tensor(state.foot_pos, dtype=torch.float64),
        foot_vel=torch.zeros_like(torch.as_tensor(state.foot_pos, dtype=torch.float64)),
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


def _read_actual_kinematic_state(base_env, foot_ids: list[int] | torch.Tensor) -> dict[str, torch.Tensor]:
    robot = base_env.scene["robot"]
    joint_pos_planner = _joint_pos_robot_to_planner(
        robot,
        torch.as_tensor(robot.data.joint_pos[:1], dtype=torch.float64).clone(),
    )
    body_pos_w = torch.as_tensor(robot.data.body_pos_w[:1], dtype=torch.float64).clone()
    foot_ids_t = torch.as_tensor(_foot_id_list(foot_ids), dtype=torch.long, device=body_pos_w.device)
    foot_pos_w = body_pos_w.index_select(1, foot_ids_t)
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
    atol: float = 1e-3,
) -> bool:
    if result is None:
        return True
    if playback_frame >= result.num_frames:
        return True
    if reset_requested:
        return True
    if last_cmd is not None and not torch.allclose(teleop_values, last_cmd, atol=atol):
        return True
    return False


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


def _viewer_direct_playback_step(base_env, result, *, frame_idx: int, sync_scene: bool = True) -> str:
    _apply_direct_playback_to_robot(base_env.scene["robot"], result, frame_idx=int(frame_idx))
    if sync_scene and hasattr(base_env.scene, "write_data_to_sim"):
        base_env.scene.write_data_to_sim()
    base_env.sim.render()
    if sync_scene and hasattr(base_env.scene, "update"):
        base_env.scene.update(float(base_env.physics_dt))
    return "render+scene_sync" if sync_scene else "render-only"


def _launch_app(args_cli: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    return app_launcher, app_launcher.app


def _attach_reference_manager_if_enabled(env, env_cfg) -> None:
    from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

    manager_device = getattr(env, "device", env_cfg.sim.device)
    manager = attach_trajectory_manager_if_enabled(env, env_cfg, device=manager_device)
    if manager is not None:
        print(
            f"[Viewer] Attached {getattr(manager, 'planner_backend', 'legacy')} trajectory manager",
            flush=True,
        )

LEG_COLORS = (
    (1.0, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.4, 1.0),
    (1.0, 0.8, 0.2),
)

PLANNER_JOINT_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


def _normalize_joint_name(name: str) -> str:
    normalized = str(name).split("/")[-1]
    normalized = normalized.split(":")[-1]
    return normalized.lower()


def _joint_order_indices(*, source_order: tuple[str, ...], target_order: tuple[str, ...]) -> torch.Tensor | None:
    source_to_index = {_normalize_joint_name(name): idx for idx, name in enumerate(source_order)}
    indices: list[int] = []
    for target_name in target_order:
        source_idx = source_to_index.get(_normalize_joint_name(target_name))
        if source_idx is None:
            return None
        indices.append(int(source_idx))
    return torch.tensor(indices, dtype=torch.long)


def _joint_pos_planner_to_robot(robot, joint_pos: torch.Tensor) -> torch.Tensor:
    joint_names = getattr(robot, "joint_names", None)
    if not joint_names:
        return joint_pos
    indices = _joint_order_indices(source_order=PLANNER_JOINT_ORDER, target_order=tuple(joint_names))
    if indices is None:
        return joint_pos
    return joint_pos.index_select(-1, indices.to(device=joint_pos.device))


def _joint_pos_robot_to_planner(robot, joint_pos: torch.Tensor) -> torch.Tensor:
    joint_names = getattr(robot, "joint_names", None)
    if not joint_names:
        return joint_pos
    indices = _joint_order_indices(source_order=tuple(joint_names), target_order=PLANNER_JOINT_ORDER)
    if indices is None:
        return joint_pos
    return joint_pos.index_select(-1, indices.to(device=joint_pos.device))


@dataclass
class TeleopCommand:
    values: torch.Tensor
    reset_requested: bool = False


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

    def __init__(self, *, device: torch.device, vx_scale: float, vy_scale: float, yaw_scale: float, timeout_s: float):
        self._device = device
        self._scales = torch.tensor([vx_scale, vy_scale, yaw_scale], dtype=torch.float64, device=device)
        self._timeout_s = float(timeout_s)
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
        if self._old_termios is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_termios)
        if self._old_flags is not None:
            fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, self._old_flags)
        self._enabled = False

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
        self._restore_terminal_state()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    def poll(self) -> TeleopCommand:
        reset_requested = False
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
                    continue
                if key == "r":
                    reset_requested = True
                    self._last_seen.clear()
                    continue
                if key in self._KEY_AXIS:
                    self._last_seen[key] = now
        now = time.monotonic()
        values = torch.zeros((1, 3), dtype=torch.float64, device=self._device)
        for key, (axis, sign) in self._KEY_AXIS.items():
            last_seen = self._last_seen.get(key)
            if last_seen is not None and now - last_seen <= self._timeout_s:
                values[0, axis] += sign * self._scales[axis]
        return TeleopCommand(values=values, reset_requested=reset_requested)


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
        self.heightmap = VisualizationMarkers(
            _make_marker_cfg("/Visuals/BatchedPlanner/heightmap", radius=0.012, color=(0.7, 0.9, 1.0))
        )
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

    def update(self, *, result, command: torch.Tensor, root_yaw: torch.Tensor, height_points: torch.Tensor) -> None:
        from extension.convention import quat_wxyz_to_xyzw

        foot_pos_w = self._foot_positions_world(result)
        touchdown_w = self._touchdown_markers_world(result)
        self.root_traj.visualize(translations=result.root_pos_w[0].to(torch.float32))
        for leg_idx in range(4):
            self.foot_traj[leg_idx].visualize(translations=foot_pos_w[0, :, leg_idx].to(torch.float32))
            self.touchdowns[leg_idx].visualize(translations=touchdown_w[0, leg_idx : leg_idx + 1].to(torch.float32))

        if height_points.numel() > 0:
            self.heightmap.visualize(translations=height_points.to(torch.float32))
        else:
            self.heightmap.visualize(translations=torch.empty((0, 3), dtype=torch.float32))

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
    from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import TeacherElevationTrajectoryEnvCfg_PLAY

    env_cfg = TeacherElevationTrajectoryEnvCfg_PLAY()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = 6.0
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    env_cfg.planner_backend = str(args_cli.planner_backend)
    env_cfg.reference_trajectory_horizon = int(args_cli.n_frames)
    env_cfg.reference_replan_interval_steps = int(args_cli.n_frames)
    env_cfg.plan_dt = float(args_cli.plan_dt)
    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"]["x"] = (0.0, 0.0)
    reset_base.params["pose_range"]["y"] = (0.0, 0.0)
    reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    return env_cfg


def _build_planner_cfg(env_cfg):
    from extension.batched_planner.config import BatchedTrajectoryConfig

    return BatchedTrajectoryConfig(
        gait_name=env_cfg.gait_name,
        step_freq=float(env_cfg.step_freq),
        duty_factor=float(env_cfg.duty_factor),
        step_height=float(env_cfg.step_height),
        foothold_search_radius=float(env_cfg.foothold_search_radius),
        foothold_search_step=float(env_cfg.foothold_search_step),
        max_foothold_step_down=float(env_cfg.max_step_down),
        max_roughness=float(env_cfg.max_roughness),
        max_touchdown_xy_reach=float(getattr(env_cfg, "max_touchdown_xy_reach", 0.22)),
        replan_stop_speed=float(env_cfg.replan_stop_speed),
        use_support_contact_terrain_estimator=True,
    )


def _build_together_planner_cfg(env_cfg):
    from extension.batched_together_planner.config import TogetherPlannerConfig

    base = TogetherPlannerConfig()
    horizon_steps = int(getattr(env_cfg, "reference_trajectory_horizon", base.horizon_steps))
    dt = float(getattr(env_cfg, "plan_dt", base.dt))
    return replace(
        base,
        horizon_s=float(horizon_steps) * dt,
        dt=dt,
        horizon_steps=horizon_steps,
        step_freq=float(getattr(env_cfg, "step_freq", base.step_freq)),
        duty_factor=float(getattr(env_cfg, "together_duty_factor", base.duty_factor)),
        idle_command_eps=float(getattr(env_cfg, "idle_command_eps", base.idle_command_eps)),
        swing_height=float(getattr(env_cfg, "step_height", base.swing_height)),
        support_search_radius=float(getattr(env_cfg, "support_search_radius", base.support_search_radius)),
        support_search_step=float(getattr(env_cfg, "support_search_step", base.support_search_step)),
    )


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


def _compute_local_terrain(scanner, *, env_id: int = 0):
    from extension.batched_planner.terrain import PlannerTerrain

    ray_hits = scanner.data.ray_hits_w[env_id].to(dtype=torch.float64)
    world_x_range, world_y_range = _compute_stable_scan_ranges(scanner, env_id=env_id)
    terrain = PlannerTerrain.from_ray_hits(
        ray_hits.unsqueeze(0),
        world_x_range=world_x_range,
        world_y_range=world_y_range,
    )
    return terrain, ray_hits


def _compute_together_local_terrain(scanner, *, env_id: int = 0):
    from extension.batched_together_planner.terrain import TogetherPlannerTerrain

    ray_hits = scanner.data.ray_hits_w[env_id].to(dtype=torch.float64)
    world_x_range, world_y_range = _compute_stable_scan_ranges(scanner, env_id=env_id)
    terrain = TogetherPlannerTerrain.from_ray_hits(
        ray_hits.unsqueeze(0),
        world_x_range=world_x_range,
        world_y_range=world_y_range,
    )
    return terrain, ray_hits


def _subsample_height_points(ray_hits: torch.Tensor, stride: int) -> torch.Tensor:
    if ray_hits.ndim != 2 or ray_hits.shape[-1] != 3:
        return torch.empty((0, 3), dtype=ray_hits.dtype, device=ray_hits.device)
    side = int(round(math.sqrt(int(ray_hits.shape[0]))))
    if side * side != int(ray_hits.shape[0]):
        sampled = ray_hits[:: max(1, stride)]
        valid = torch.isfinite(sampled).all(dim=-1)
        return sampled[valid]
    grid = ray_hits.reshape(side, side, 3)
    sampled = grid[:: max(1, stride), :: max(1, stride)].reshape(-1, 3)
    valid = torch.isfinite(sampled).all(dim=-1)
    return sampled[valid]


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
    camera_position = (root_pos + camera_offset).cpu().numpy()
    target_position = (root_pos + torch.tensor([0.0, 0.0, 0.35], device=root_pos.device)).cpu().numpy()
    env.sim.set_camera_view(camera_position, target_position)


def _planner_state_from_env(env, foot_ids: list[int]):
    from extension.convention import isaac_state_to_planner_state, quat_wxyz_to_xyzw

    robot = env.scene["robot"]
    joint_pos = _joint_pos_robot_to_planner(robot, robot.data.joint_pos[:1].to(dtype=torch.float64))
    return isaac_state_to_planner_state(
        root_pos_w=robot.data.root_pos_w[:1].to(dtype=torch.float64),
        root_quat_xyzw=quat_wxyz_to_xyzw(robot.data.root_quat_w[:1]).to(dtype=torch.float64),
        joint_pos=joint_pos,
        foot_pos_w=robot.data.body_pos_w[:1, foot_ids, :].to(dtype=torch.float64),
        foot_vel_w=robot.data.body_lin_vel_w[:1, foot_ids, :].to(dtype=torch.float64),
    )


def _together_state_from_env(env, foot_ids: list[int]):
    return _legacy_state_to_together_state(_planner_state_from_env(env, foot_ids))


def _adapt_together_result_for_viewer(result) -> ViewerTrajectoryResult:
    from extension.convention import euler_to_quat_batch

    root_pos_w = torch.as_tensor(result.root_pos).contiguous()
    root_rpy = torch.as_tensor(result.root_rpy, device=root_pos_w.device, dtype=root_pos_w.dtype)
    root_quat_w = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2]).contiguous()
    foot_pos_w = torch.as_tensor(result.foot_pos, device=root_pos_w.device, dtype=root_pos_w.dtype).contiguous()
    foot_pos_root = (foot_pos_w - root_pos_w.unsqueeze(2)).contiguous()
    planned_touchdown_w = torch.as_tensor(result.touchdown_seq[:, :, 0, :], device=root_pos_w.device, dtype=root_pos_w.dtype).contiguous()
    num_frames = int(root_pos_w.shape[1])
    zeros_vel = torch.zeros_like(root_pos_w)
    return ViewerTrajectoryResult(
        num_frames=num_frames,
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        joint_angles=torch.as_tensor(result.joint_angles, device=root_pos_w.device, dtype=root_pos_w.dtype).contiguous(),
        foot_pos_w=foot_pos_w,
        foot_pos_root=foot_pos_root,
        contact_state=torch.as_tensor(result.contact_state, device=root_pos_w.device).contiguous(),
        planned_touchdown_w=planned_touchdown_w,
        root_lin_vel_w=zeros_vel,
        root_ang_vel_w=zeros_vel.clone(),
    )


def _plan_viewer_trajectory(
    *,
    backend: str,
    terrain,
    state,
    command: torch.Tensor,
    requested_n_frames: int,
    dt: float,
    legacy_cfg,
    together_cfg,
):
    backend_name = str(backend).lower()
    if backend_name == "legacy":
        from extension.batched_planner.trajectory import batched_generate_trajectory

        return batched_generate_trajectory(
            terrain,
            state,
            command,
            requested_n_frames=requested_n_frames,
            dt=dt,
            cfg=legacy_cfg,
        )
    if backend_name == "together":
        from extension.batched_together_planner.planner import plan_segment

        return _adapt_together_result_for_viewer(
            plan_segment(
                terrain,
                state,
                command,
                cfg=together_cfg,
            )
        )
    raise ValueError(f"Unsupported planner backend: {backend!r}")


def _print_help() -> None:
    print("\nTerminal teleop (hold keys with repeat):", flush=True)
    print("  W/S : forward/backward", flush=True)
    print("  A/D : lateral left/right", flush=True)
    print("  Q/E : yaw left/right", flush=True)
    print("  X   : clear command", flush=True)
    print("  R   : reset environment", flush=True)
    print("  Ctrl-C : quit\n", flush=True)


def main() -> int:
    args_cli = _prepare_runtime_args(_parse_args())
    _, simulation_app = _launch_app(args_cli)

    import gymnasium as gym

    import go2_pvcnn.tasks.register_envs  # noqa: F401,F403
    from extension.convention import extract_yaw_batch
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = _build_env_cfg(args_cli)
    planner_cfg = _build_planner_cfg(env_cfg)
    planner_cfg.dt = float(args_cli.plan_dt)
    planner_cfg.reference_trajectory_horizon = int(args_cli.n_frames)
    together_planner_cfg = _build_together_planner_cfg(env_cfg)

    env = gym.make(
        "Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if getattr(args_cli, "livestream", -1) in (1, 2) else None,
    )
    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env = env.unwrapped
    _attach_reference_manager_if_enabled(base_env, env_cfg)
    zero_actions = _make_zero_actions(base_env)
    foot_ids, _ = base_env.scene["robot"].find_bodies(".*_foot")
    scanner = base_env.scene.sensors["height_scanner"]
    visualizer = PlannerVisualizer()

    print("[Viewer] Terrain source: teacher_elevation_trajectory env config", flush=True)
    print(f"[Viewer] Planner horizon: {args_cli.n_frames} frames @ dt={args_cli.plan_dt:.3f}s", flush=True)
    print("[Viewer] Playback mode: kinematic (no physics)", flush=True)
    _print_help()

    env.reset()
    for _ in range(max(0, int(args_cli.warmup_steps))):
        env.step(zero_actions)

    result = None
    playback_frame = 0
    last_cmd = None
    plan_cycle = 0
    scripted_cycles_remaining = max(0, int(args_cli.scripted_command_cycles))
    scripted_command = _parse_scripted_command(args_cli.scripted_command, device=base_env.device)

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
            while simulation_app.is_running():
                teleop_cmd = teleop.poll()
                active_cmd = teleop_cmd
                if scripted_command is not None and scripted_cycles_remaining > 0:
                    active_cmd = TeleopCommand(
                        values=scripted_command.clone(),
                        reset_requested=teleop_cmd.reset_requested,
                    )

                if active_cmd.reset_requested:
                    env.reset()
                    for _ in range(max(0, int(args_cli.warmup_steps))):
                        env.step(zero_actions)
                    result = None
                    playback_frame = 0
                    last_cmd = None
                    plan_cycle = 0

                need_replan = _viewer_loop_need_replan(
                    result=result,
                    playback_frame=playback_frame,
                    reset_requested=active_cmd.reset_requested,
                    teleop_values=active_cmd.values,
                    last_cmd=last_cmd,
                )
                loop_diag = (_format_command_values(active_cmd.values), need_replan)
                if loop_diag != last_loop_diag:
                    print(
                        "[Viewer][Loop] "
                        f"teleop_cmd={loop_diag[0]} "
                        f"need_replan={need_replan} "
                        f"playback_frame={playback_frame} "
                        f"cycle={plan_cycle}",
                        flush=True,
                    )
                    last_loop_diag = loop_diag

                if need_replan:
                    if result is not None and playback_frame > 0:
                        frame = min(playback_frame - 1, result.num_frames - 1)
                        if args_cli.planner_backend == "together":
                            state = _together_state_from_reference_result(result, frame_idx=frame)
                        else:
                            state = _planner_state_from_reference_result(result, frame_idx=frame)
                    else:
                        if args_cli.planner_backend == "together":
                            state = _together_state_from_env(base_env, foot_ids)
                        else:
                            state = _planner_state_from_env(base_env, foot_ids)

                    if args_cli.planner_backend == "together":
                        terrain, ray_hits = _compute_together_local_terrain(scanner)
                    else:
                        terrain, ray_hits = _compute_local_terrain(scanner)

                    result = _plan_viewer_trajectory(
                        backend=args_cli.planner_backend,
                        terrain=terrain,
                        state=state,
                        command=active_cmd.values,
                        requested_n_frames=args_cli.n_frames,
                        dt=args_cli.plan_dt,
                        legacy_cfg=planner_cfg,
                        together_cfg=together_planner_cfg,
                    )
                    summary = _trajectory_motion_summary(result)
                    print(
                        "[Viewer][Plan] "
                        f"backend={args_cli.planner_backend} "
                        f"cycle={plan_cycle} "
                        f"cmd={_format_command_values(active_cmd.values)} "
                        f"delta=({summary['dx']:+0.2f}, {summary['dy']:+0.2f}, {summary['dz']:+0.2f}) "
                        f"dyaw={summary['dyaw']:+0.2f} "
                        f"standstill={summary['standstill']}",
                        flush=True,
                    )
                    playback_frame = 0

                    planner_state = _planner_state_from_reference_result(result, frame_idx=0)
                    root_yaw = extract_yaw_batch(planner_state.root_quat)
                    height_points = _subsample_height_points(ray_hits, int(args_cli.heightmap_viz_stride))
                    visualizer.update(
                        result=result,
                        command=active_cmd.values,
                        root_yaw=root_yaw,
                        height_points=height_points,
                    )
                    plan_cycle += 1
                    if scripted_command is not None and scripted_cycles_remaining > 0:
                        scripted_cycles_remaining = max(0, scripted_cycles_remaining - 1)

                if result is not None and playback_frame < result.num_frames:
                    playback_path = _viewer_direct_playback_step(base_env, result, frame_idx=playback_frame)
                    if playback_path != last_playback_path:
                        print(
                            f"[Viewer][Playback] path={playback_path}",
                            flush=True,
                        )
                        last_playback_path = playback_path
                    actual = _read_actual_base_state(base_env)
                    planner_frame = _planner_state_from_reference_result(result, frame_idx=playback_frame)
                    actual_summary = (
                        _format_xyz(actual["root_pos_w"][0]),
                        _format_quat(actual["root_quat_raw"][0]),
                        _format_xyz(actual["rpy_if_wxyz"][0]),
                        _format_xyz(actual["rpy_if_xyzw"][0]),
                        _format_xyz(planner_frame.root_pos[0]),
                        _format_xyz(_quat_wxyz_to_rpy(planner_frame.root_quat[0])),
                    )
                    if actual_summary != last_actual_summary:
                        print(
                            "[Viewer][ActualBase] "
                            f"cycle={max(plan_cycle - 1, 0)} "
                            f"actual_pos={actual_summary[0]} "
                            f"actual_quat_raw={actual_summary[1]} "
                            f"actual_rpy_if_wxyz={actual_summary[2]} "
                            f"actual_rpy_if_xyzw={actual_summary[3]} "
                            f"plan_pos={actual_summary[4]} "
                            f"plan_rpy={actual_summary[5]}",
                            flush=True,
                        )
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
                        print(
                            "[Viewer][ActualKinematics] "
                            f"cycle={max(plan_cycle - 1, 0)} "
                            f"joint_err_max={kinematic_summary[0]:0.6f} "
                            f"joint_err_mean={kinematic_summary[1]:0.6f} "
                            f"foot_err_max={kinematic_summary[2]:0.6f} "
                            f"foot_err_mean={kinematic_summary[3]:0.6f}",
                            flush=True,
                        )
                        last_kinematic_summary = kinematic_summary
                    playback_frame += 1

                last_cmd = active_cmd.values.clone()

                if result is not None:
                    display_frame = min(playback_frame - 1, result.num_frames - 1) if playback_frame > 0 else 0
                    planner_state = _planner_state_from_reference_result(result, frame_idx=display_frame)
                    root_yaw = extract_yaw_batch(planner_state.root_quat)
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
                    actual_rpy_xyzw = actual["rpy_if_xyzw"][0]
                    status = (
                        f"\rcycle={max(plan_cycle - 1, 0)} "
                        f"cmd vx={active_cmd.values[0,0]:+0.2f} "
                        f"vy={active_cmd.values[0,1]:+0.2f} "
                        f"yaw={yaw_rate:+0.2f} | "
                        f"plan=({root_pos[0]:+0.2f}, {root_pos[1]:+0.2f}, {root_pos[2]:+0.2f}) "
                        f"actual=({actual_pos[0]:+0.2f}, {actual_pos[1]:+0.2f}, {actual_pos[2]:+0.2f}) "
                        f"actual_rpy_xyzw=({actual_rpy_xyzw[0]:+0.2f}, {actual_rpy_xyzw[1]:+0.2f}, {actual_rpy_xyzw[2]:+0.2f}) "
                        f"frame={display_frame}/{result.num_frames}"
                    )
                    if status != last_status:
                        sys.stdout.write(status)
                        sys.stdout.flush()
                        last_status = status
        finally:
            print()
            env.close()

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
