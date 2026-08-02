"""Script to play a trained teacher policy."""

from __future__ import annotations

import argparse
import atexit
import os
import select
import signal
import sys
import termios
import threading
import tty
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch


PARALLELISM_TERMINATION_NAMES = (
    "time_out",
    "base_contact",
    "bad_orientation",
    "parallelism_ref_root_z_too_far",
    "parallelism_ref_projected_gravity_too_far",
    "parallelism_ref_foot_z_too_far",
    "parallelism_ref_joint_pos_too_far",
)


@dataclass
class ParallelismPlayPanelState:
    """UI-owned command and diagnostic-only termination controls."""

    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    suppress_termination: dict[str, bool] = field(
        default_factory=lambda: dict.fromkeys(PARALLELISM_TERMINATION_NAMES, True)
    )


def _panel_command_tensor(state: ParallelismPlayPanelState, command: torch.Tensor) -> torch.Tensor:
    """Return one clamped body-frame command in the command tensor's dtype/device."""

    values = command.new_tensor(((state.vx, state.vy, state.vyaw),))
    lower = command.new_tensor((-1.0, -0.5, -1.0))
    upper = command.new_tensor((1.0, 0.5, 1.0))
    return torch.maximum(torch.minimum(values, upper), lower)


def _filter_termination_masks(
    raw_masks: dict[str, torch.Tensor],
    suppress_termination: dict[str, bool],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep raw diagnostics while OR-ing only termination terms allowed to reset."""

    if not raw_masks:
        raise ValueError("raw_masks must contain at least one termination term")
    first = next(iter(raw_masks.values()))
    effective_done = torch.zeros_like(torch.as_tensor(first), dtype=torch.bool)
    diagnostics: dict[str, torch.Tensor] = {}
    for name, value in raw_masks.items():
        raw = torch.as_tensor(value, dtype=torch.bool, device=effective_done.device)
        diagnostics[name] = raw
        if not bool(suppress_termination.get(name, False)):
            effective_done |= raw
    return effective_done, diagnostics


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for _path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play a trained teacher policy")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
    parser.add_argument("--video_length", type=int, default=2000000, help="Length of recorded video (steps)")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (steps)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
    parser.add_argument("--checkpoint", type=str, default="model_1600.pt", help="Checkpoint file name")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory name")
    parser.add_argument(
        "--experiment",
        type=str,
        default="teacher_elevation_trajectory_mpc_semantic",
        choices=[
            "teacher_elevation_trajectory_mpc_semantic",
            "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance",
        ],
        help="Experiment/task: semantic MPC teacher or flat-small avoidance continuation",
    )
    parser.add_argument("--sample", action="store_true", default=False, help="Sample actions with std instead of using policy")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many play steps; 0 means run until the app exits.")
    parser.add_argument(
        "--use-raw-reference-trajectory",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug-livestream",
        action="store_true",
        default=False,
        help="Print startup and loop timing diagnostics for WebRTC livestream bottlenecks.",
    )
    parser.add_argument(
        "--step-mode",
        action="store_true",
        default=False,
        help="Pause play loop and advance exactly one env/render step for each Space key press.",
    )
    parser.add_argument(
        "--keyboard-control",
        action="store_true",
        default=False,
        help="Use terminal hold-to-move keyboard velocity commands.",
    )
    parser.add_argument("--keyboard-linear-speed", type=float, default=0.5, help="Keyboard forward/backward speed.")
    parser.add_argument("--keyboard-lateral-speed", type=float, default=0.25, help="Keyboard left/right speed.")
    parser.add_argument("--keyboard-yaw-speed", type=float, default=0.5, help="Keyboard yaw-rate speed.")
    parser.add_argument("--keyboard-speed-step", type=float, default=0.1, help="Keyboard +/- speed increment.")
    parser.add_argument("--terrain-row", type=int, default=None, help="Initial terrain row for env0; omit for default.")
    parser.add_argument("--terrain-col", type=int, default=None, help="Initial terrain column for env0; omit for default.")
    parser.add_argument(
        "--planner-backend",
        type=str,
        default="mpc",
        choices=["mpc"],
        help="Trajectory planner backend. Cleanup build supports only mpc.",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _prepare_runtime_args(args_cli: argparse.Namespace) -> argparse.Namespace:
    if getattr(args_cli, "livestream", -1) in (1, 2) and not args_cli.enable_cameras:
        args_cli.enable_cameras = True
        print(
            "[INFO][play.py] livestream: enabled AppLauncher --enable_cameras so the simulator "
            "uses a rendering experience (works without X11; WebRTC client on another machine).",
            flush=True,
        )
    return args_cli


def _launch_app(args_cli: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    return app_launcher, app_launcher.app


def _resolve_render_mode(args_cli: argparse.Namespace) -> str | None:
    if args_cli.video or getattr(args_cli, "livestream", -1) in (1, 2):
        return "rgb_array"
    return None


def _livestream_camera_update_interval(livestream: int) -> int:
    return 4 if livestream in (1, 2) else 1


def _should_update_follow_camera(*, timestep: int, num_envs: int, livestream: int, interval: int) -> bool:
    if num_envs != 1:
        return False
    if livestream in (1, 2):
        return timestep % max(1, interval) == 0
    return True


def _play_loop_should_continue(simulation_app, *, timestep: int, max_steps: int) -> bool:
    if max_steps > 0:
        return timestep < max_steps
    return bool(simulation_app.is_running())


@dataclass
class _KeyboardVelocityController:
    enabled: bool
    linear_speed: float
    lateral_speed: float
    yaw_speed: float
    speed_step: float
    max_linear_speed: float = 1.0
    max_lateral_speed: float = 0.5
    max_yaw_speed: float = 1.0
    hold_timeout_s: float = 0.15
    poll_interval_s: float = 0.02

    def __post_init__(self) -> None:
        self._pressed: set[str] = set()
        self._last_pressed_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._terminal_fd: int | None = None
        self._terminal_original_attrs = None

    def press(self, key: str) -> None:
        key = str(key).lower()
        with self._lock:
            if key in {"+", "="}:
                self.linear_speed = min(self.max_linear_speed, self.linear_speed + self.speed_step)
                self.lateral_speed = min(self.max_lateral_speed, self.lateral_speed + self.speed_step)
                self.yaw_speed = min(self.max_yaw_speed, self.yaw_speed + self.speed_step)
                return
            if key in {"-", "_"}:
                self.linear_speed = max(0.0, self.linear_speed - self.speed_step)
                self.lateral_speed = max(0.0, self.lateral_speed - self.speed_step)
                self.yaw_speed = max(0.0, self.yaw_speed - self.speed_step)
                return
            if key in {" ", "space", "x"}:
                self._pressed.clear()
                self._last_pressed_at.clear()
                return
            self._pressed.add(key)
            self._last_pressed_at[key] = perf_counter()

    def release(self, key: str) -> None:
        key = str(key).lower()
        with self._lock:
            self._pressed.discard(key)

    def command_values(self) -> tuple[float, float, float]:
        with self._lock:
            self._expire_stale_keys_locked(perf_counter())
            keys = set(self._pressed)
            linear_speed = float(self.linear_speed)
            lateral_speed = float(self.lateral_speed)
            yaw_speed = float(self.yaw_speed)

        vx = (1.0 if "w" in keys else 0.0) + (-1.0 if "s" in keys else 0.0)
        vy = (1.0 if "a" in keys else 0.0) + (-1.0 if "d" in keys else 0.0)
        yaw = (1.0 if "q" in keys else 0.0) + (-1.0 if "e" in keys else 0.0)
        return (
            float(np.clip(vx * linear_speed, -self.max_linear_speed, self.max_linear_speed)),
            float(np.clip(vy * lateral_speed, -self.max_lateral_speed, self.max_lateral_speed)),
            float(np.clip(yaw * yaw_speed, -self.max_yaw_speed, self.max_yaw_speed)),
        )

    def command_tensor(self, *, device, dtype, num_envs: int) -> torch.Tensor:
        values = self.command_values()
        command = torch.tensor(values, device=device, dtype=dtype).view(1, 3)
        return command.repeat(int(num_envs), 1)

    @staticmethod
    def _key_to_name(key: str) -> str | None:
        if key == "\x1b":
            return "esc"
        if key in {" ", "\r", "\n"}:
            return "space"
        text = str(key).lower()
        if text == "key.space":
            return "space"
        if text == "key.esc":
            return "esc"
        if len(text) == 1:
            return text
        return None

    def _expire_stale_keys_locked(self, now_s: float) -> None:
        stale = [
            key
            for key in self._pressed
            if key in {"w", "s", "a", "d", "q", "e"}
            and now_s - self._last_pressed_at.get(key, 0.0) > self.hold_timeout_s
        ]
        for key in stale:
            self._pressed.discard(key)
            self._last_pressed_at.pop(key, None)

    def _terminal_read_loop(self) -> None:
        assert self._terminal_fd is not None
        while not self._stop_event.is_set():
            readable, _, _ = select.select([self._terminal_fd], [], [], self.poll_interval_s)
            if not readable:
                continue
            try:
                char = os.read(self._terminal_fd, 1).decode(errors="ignore")
            except OSError:
                break
            name = self._key_to_name(char)
            if name == "esc":
                self._stop_event.set()
                break
            if name is not None:
                self.press(name)

    def __enter__(self) -> "_KeyboardVelocityController":
        if not self.enabled:
            return self
        try:
            if not sys.stdin.isatty():
                raise RuntimeError("stdin is not a TTY")
            self._terminal_fd = sys.stdin.fileno()
            self._terminal_original_attrs = termios.tcgetattr(self._terminal_fd)
            tty.setcbreak(self._terminal_fd)
            self._reader_thread = threading.Thread(target=self._terminal_read_loop, name="play-terminal-keyboard", daemon=True)
            self._reader_thread.start()
            print(
                "[play.py] Terminal keyboard control enabled: hold W/S/A/D/Q/E, +/- speed, Space or X stop, Esc stop.",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - terminal availability varies in headless launchers.
            print(f"[WARN][play.py] --keyboard-control disabled: failed to start terminal keyboard reader: {exc}", flush=True)
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.5)
            self._reader_thread = None
        if self._terminal_fd is not None and self._terminal_original_attrs is not None:
            try:
                termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_original_attrs)
            except Exception:
                pass
        self._terminal_fd = None
        self._terminal_original_attrs = None


def _apply_keyboard_velocity_command(base_env, controller: _KeyboardVelocityController) -> torch.Tensor | None:
    if not controller.enabled:
        return None
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_command"):
        return None
    command = command_manager.get_command("base_velocity")
    if command is None:
        return None
    target = controller.command_tensor(device=command.device, dtype=command.dtype, num_envs=int(command.shape[0]))
    command[:, :3] = target
    return command


def _apply_initial_terrain_selection(base_env, *, terrain_row: int | None, terrain_col: int | None, env_id: int = 0) -> None:
    if terrain_row is None and terrain_col is None:
        return
    terrain = getattr(getattr(base_env, "scene", None), "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain is None or terrain_origins is None:
        raise RuntimeError("--terrain-row/--terrain-col require curriculum terrain_origins.")

    origins = torch.as_tensor(terrain_origins)
    if origins.ndim != 3 or int(origins.shape[-1]) != 3:
        raise RuntimeError(f"Expected terrain_origins shape [rows, cols, 3], got {tuple(origins.shape)}")
    num_rows, num_cols = int(origins.shape[0]), int(origins.shape[1])
    row = int(terrain_row) if terrain_row is not None else int(getattr(terrain, "terrain_levels")[env_id])
    col = int(terrain_col) if terrain_col is not None else int(getattr(terrain, "terrain_types")[env_id])
    if not (0 <= row < num_rows):
        raise ValueError(f"--terrain-row must be in [0, {num_rows - 1}], got {row}")
    if not (0 <= col < num_cols):
        raise ValueError(f"--terrain-col must be in [0, {num_cols - 1}], got {col}")

    selected_origin = origins[row, col]
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is not None:
        terrain_levels[env_id] = row
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is not None:
        terrain_types[env_id] = col
    env_origins = getattr(terrain, "env_origins", None)
    if env_origins is not None:
        env_origins[env_id] = selected_origin.to(device=env_origins.device, dtype=env_origins.dtype)
    scene_env_origins = getattr(base_env.scene, "env_origins", None)
    if scene_env_origins is not None:
        scene_env_origins[env_id] = selected_origin.to(device=scene_env_origins.device, dtype=scene_env_origins.dtype)
    print(f"[play.py] Initial terrain env{env_id}: row={row}, col={col}", flush=True)


@dataclass
class _TerminalStepGate:
    enabled: bool

    def __post_init__(self) -> None:
        self._stdin_fd = None
        self._old_termios = None
        self._old_flags = None
        self._raw_enabled = False
        self._old_signal_handlers: dict[int, object] = {}
        self._atexit_registered = False

    def __enter__(self) -> "_TerminalStepGate":
        if not self.enabled:
            return self
        if not sys.stdin.isatty():
            print("[WARN][play.py] stdin is not a TTY; --step-mode cannot receive Space key presses.", flush=True)
            return self
        import fcntl
        import termios
        import tty

        self._stdin_fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._stdin_fd)
        self._old_flags = fcntl.fcntl(self._stdin_fd, fcntl.F_GETFL)
        tty.setcbreak(self._stdin_fd)
        fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, self._old_flags | os.O_NONBLOCK)
        self._raw_enabled = True
        self._install_cleanup_guards()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._remove_cleanup_guards()
        self._restore_terminal_state()

    def wait_for_step(self) -> bool:
        if not self.enabled:
            return True
        if not self._raw_enabled:
            sleep(0.05)
            return False
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                return False
            char = sys.stdin.read(1)
            if not char:
                return False
            if char == "\x03":
                raise KeyboardInterrupt
            if char == " ":
                return True

    def _restore_terminal_state(self) -> None:
        if not self._raw_enabled:
            return
        import fcntl
        import termios

        assert self._stdin_fd is not None
        self._raw_enabled = False
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
        self._restore_terminal_state()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))


def _compute_follow_camera_pose(robot_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_direction = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    camera_position = robot_pos - camera_direction + np.array([0.0, 0.0, 1.5], dtype=np.float64)
    return camera_position, robot_pos


def _collect_runtime_debug_snapshot(args_cli: argparse.Namespace, *, argv: list[str] | None = None) -> dict[str, object]:
    return {
        "argv": list(sys.argv if argv is None else argv),
        "env": {
            "LIVESTREAM": os.environ.get("LIVESTREAM"),
            "HEADLESS": os.environ.get("HEADLESS"),
            "ENABLE_CAMERAS": os.environ.get("ENABLE_CAMERAS"),
        },
        "args": {
            "livestream": getattr(args_cli, "livestream", None),
            "headless": getattr(args_cli, "headless", None),
            "enable_cameras": getattr(args_cli, "enable_cameras", None),
            "device": getattr(args_cli, "device", None),
            "debug_livestream": getattr(args_cli, "debug_livestream", None),
        },
    }


def _print_runtime_debug_snapshot(args_cli: argparse.Namespace) -> None:
    snapshot = _collect_runtime_debug_snapshot(args_cli)
    print("[debug-livestream] runtime launch snapshot:", flush=True)
    print(f"[debug-livestream]   argv={snapshot['argv']}", flush=True)
    print(f"[debug-livestream]   env={snapshot['env']}", flush=True)
    print(f"[debug-livestream]   args={snapshot['args']}", flush=True)
    if snapshot["args"]["livestream"] == 0 and snapshot["args"]["headless"]:
        print(
            "[debug-livestream] warning: effective livestream=0 while headless=True; "
            "WebRTC is not actually enabled in this run.",
            flush=True,
        )


@dataclass(slots=True)
class _LivestreamDebug:
    enabled: bool
    startup_marks: list[tuple[str, float]] = field(default_factory=list)
    loop_samples: list[dict[str, float]] = field(default_factory=list)
    _startup_last: float = field(default_factory=perf_counter)

    def mark_startup(self, label: str) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        self.startup_marks.append((label, now - self._startup_last))
        self._startup_last = now

    def add_loop_sample(
        self,
        *,
        policy_s: float,
        env_step_s: float,
        camera_s: float,
        total_s: float,
        timestep: int,
        step_probe: dict[str, float] | None = None,
    ) -> None:
        if not self.enabled:
            return
        sample = {
            "policy_s": policy_s,
            "env_step_s": env_step_s,
            "camera_s": camera_s,
            "total_s": total_s,
            "timestep": float(timestep),
        }
        if step_probe is not None:
            sample.update(step_probe)
        self.loop_samples.append(sample)
        if len(self.loop_samples) in {1, 10, 30}:
            self.print_loop_summary(prefix=f"[debug-livestream][sample={len(self.loop_samples)}]")

    def print_startup_summary(self) -> None:
        if not self.enabled or not self.startup_marks:
            return
        print("[debug-livestream] startup timing summary:", flush=True)
        for label, dt_s in self.startup_marks:
            print(f"[debug-livestream]   {label:<24} {dt_s * 1000.0:8.1f} ms", flush=True)

    def print_loop_summary(self, *, prefix: str = "[debug-livestream]") -> None:
        if not self.enabled or not self.loop_samples:
            return
        count = len(self.loop_samples)
        totals = {"policy_s": 0.0, "env_step_s": 0.0, "camera_s": 0.0, "total_s": 0.0}
        for sample in self.loop_samples:
            for key in totals:
                totals[key] += sample[key]
        mean_total_ms = totals["total_s"] * 1000.0 / count
        fps = 1.0 / (totals["total_s"] / count) if totals["total_s"] > 0.0 else float("inf")
        print(
            f"{prefix} mean step={mean_total_ms:0.2f} ms "
            f"(policy={totals['policy_s'] * 1000.0 / count:0.2f} ms, "
            f"env={totals['env_step_s'] * 1000.0 / count:0.2f} ms, "
            f"camera={totals['camera_s'] * 1000.0 / count:0.2f} ms) "
            f"approx_fps={fps:0.2f}",
            flush=True,
        )
        detail_keys = [
            "action_process_s",
            "action_apply_s",
            "sim_step_s",
            "sim_render_s",
            "scene_update_s",
            "obs_compute_s",
            "reward_compute_s",
            "termination_compute_s",
            "command_compute_s",
        ]
        detail_parts = []
        for key in detail_keys:
            if key in self.loop_samples[0]:
                value_ms = sum(sample.get(key, 0.0) for sample in self.loop_samples) * 1000.0 / count
                detail_parts.append(f"{key.removesuffix('_s')}={value_ms:0.2f} ms")
        if detail_parts:
            print(f"{prefix} env breakdown: " + ", ".join(detail_parts), flush=True)


@dataclass(slots=True)
class _StepProbe:
    enabled: bool
    accumulators: dict[str, float] = field(
        default_factory=lambda: {
            "action_process_s": 0.0,
            "action_apply_s": 0.0,
            "sim_step_s": 0.0,
            "sim_render_s": 0.0,
            "scene_update_s": 0.0,
            "obs_compute_s": 0.0,
            "reward_compute_s": 0.0,
            "termination_compute_s": 0.0,
            "command_compute_s": 0.0,
        }
    )

    def wrap_method(self, obj, attr_name: str, metric_key: str) -> None:
        if not self.enabled or not hasattr(obj, attr_name):
            return
        original = getattr(obj, attr_name)
        if not callable(original):
            return

        def wrapped(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.accumulators[metric_key] += perf_counter() - start

        setattr(obj, attr_name, wrapped)

    def snapshot_and_reset(self) -> dict[str, float]:
        snapshot = dict(self.accumulators)
        for key in self.accumulators:
            self.accumulators[key] = 0.0
        return snapshot


def _install_env_step_probes(base_env, *, enabled: bool) -> _StepProbe:
    probe = _StepProbe(enabled=enabled)
    if not enabled:
        return probe

    probe.wrap_method(base_env.action_manager, "process_action", "action_process_s")
    probe.wrap_method(base_env.action_manager, "apply_action", "action_apply_s")
    probe.wrap_method(base_env.sim, "step", "sim_step_s")
    probe.wrap_method(base_env.sim, "render", "sim_render_s")
    probe.wrap_method(base_env.scene, "update", "scene_update_s")
    probe.wrap_method(base_env.observation_manager, "compute", "obs_compute_s")
    probe.wrap_method(base_env.reward_manager, "compute", "reward_compute_s")
    probe.wrap_method(base_env.termination_manager, "compute", "termination_compute_s")
    probe.wrap_method(base_env.command_manager, "compute", "command_compute_s")
    return probe


def _pump_play_paused_window(base_env, *, sleep_s: float = 0.01) -> None:
    base_env.sim.render()
    if hasattr(base_env.scene, "update"):
        base_env.scene.update(float(base_env.physics_dt))
    if sleep_s > 0.0:
        sleep(float(sleep_s))


def _make_env_wrapper(env, *, gym_module, vec_env_cls, tensor_dict_cls, clip_actions: float | None = None):
    class SimpleRslRlEnvWrapper(vec_env_cls):
        """Simple wrapper for RSL-RL without PVCNN."""

        def __init__(self, env, clip_actions: float | None = None):
            self.env = env
            self.clip_actions = clip_actions
            self.num_envs = env.num_envs
            self.device = env.device
            self.max_episode_length = env.max_episode_length

            if hasattr(env, "action_manager"):
                self.num_actions = env.action_manager.total_action_dim
            else:
                self.num_actions = gym_module.spaces.flatdim(env.single_action_space)

            if clip_actions is not None:
                self.env.action_space = gym_module.spaces.Box(
                    low=-clip_actions,
                    high=clip_actions,
                    shape=(self.num_actions,),
                    dtype=env.action_space.dtype,
                )

            obs_dict, _ = self.env.reset()
            self._initial_observations = self._format_observations(obs_dict)

        @property
        def unwrapped(self):
            return self.env.unwrapped

        @property
        def cfg(self):
            return self.env.unwrapped.cfg

        @property
        def episode_length_buf(self):
            return self.env.unwrapped.episode_length_buf

        @episode_length_buf.setter
        def episode_length_buf(self, value):
            self.env.unwrapped.episode_length_buf = value

        @property
        def observation_space(self):
            return self.env.observation_space

        @property
        def action_space(self):
            return self.env.action_space

        def _flatten_group(self, obs_dict, group_names: list[str]) -> torch.Tensor:
            values = []
            for name in group_names:
                value = obs_dict[name]
                values.append(value.reshape(value.shape[0], -1))
            return torch.cat(values, dim=-1)

        def _format_observations(self, obs_dict) -> tuple[torch.Tensor, dict]:
            policy_obs = self._flatten_group(obs_dict, ["policy_elevation_semantic_map", "policy_state"])
            critic_obs = self._flatten_group(obs_dict, ["critic_elevation_semantic_map", "critic_state"])
            return policy_obs, {"observations": {"critic": critic_obs}}

        def get_observations(self):
            obs_dict = self.env.unwrapped.observation_manager.compute()
            return self._format_observations(obs_dict)

        def reset(self):
            obs_dict, _ = self.env.reset()
            return self._format_observations(obs_dict)

        def consume_initial_observations(self):
            observations = self._initial_observations
            self._initial_observations = None
            if observations is not None:
                return observations
            return self.get_observations()

        def step(self, actions):
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

            obs_dict, rewards, dones, truncated, extras = self.env.step(actions)
            dones = dones | truncated

            obs, obs_extras = self._format_observations(obs_dict)
            extras.update(obs_extras)
            return obs, rewards, dones, extras

    return SimpleRslRlEnvWrapper(env, clip_actions=clip_actions)


def _configure_reference_trajectory(env_cfg, *, use_raw_reference_trajectory: bool) -> None:
    if hasattr(env_cfg, "use_batched_reference_trajectory"):
        env_cfg.use_batched_reference_trajectory = False
        if hasattr(env_cfg, "planner_owned_reference_cache"):
            env_cfg.planner_owned_reference_cache = False
        if use_raw_reference_trajectory:
            print(
                "[play.py] Warning: --use-raw-reference-trajectory is legacy-only and is ignored; "
                "policy playback runs without MPC reference trajectory.",
                flush=True,
            )
        return

    if hasattr(env_cfg, "use_raw_reference_trajectory"):
        env_cfg.use_raw_reference_trajectory = bool(use_raw_reference_trajectory)


def _attach_reference_manager_if_enabled(env, env_cfg, experiment_name: str) -> None:
    if not getattr(env_cfg, "planner_owned_reference_cache", False):
        return

    from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

    manager_device = getattr(env, "device", env_cfg.sim.device)
    manager = attach_trajectory_manager_if_enabled(
        env,
        env_cfg,
        experiment_name=experiment_name,
        device=manager_device,
    )
    if manager is not None:
        print(
            f"[Planner] Attached {getattr(manager, 'planner_backend', 'mpc')} trajectory manager "
            f"for {experiment_name}",
            flush=True,
        )


def main() -> int:
    args_cli = _prepare_runtime_args(_parse_args())
    debug = _LivestreamDebug(enabled=bool(args_cli.debug_livestream))
    if args_cli.debug_livestream:
        _print_runtime_debug_snapshot(args_cli)

    _, simulation_app = _launch_app(args_cli)
    debug.mark_startup("app launch")

    import gymnasium as gym

    from agent import get_train_cfg
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY,
        TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
    )
    import go2_pvcnn.tasks.register_envs  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.dict import print_dict
    from rsl_rl.env import VecEnv
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    debug.mark_startup("python imports")

    experiment_play_map = {
        "teacher_elevation_trajectory_mpc_semantic": (
            TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
            "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
        ),
        "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance": (
            TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY,
            "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-Play-v0",
        ),
    }

    experiment_name = args_cli.experiment
    env_cfg_cls, task_id = experiment_play_map[experiment_name]

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    log_dir = os.path.join(log_root_path, args_cli.run_dir)
    checkpoint_path = os.path.join(log_dir, args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n{'=' * 80}")
    print(f"Playing - {experiment_name}")
    print(f"{'=' * 80}")
    print(f"Task: {task_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Livestream mode: {getattr(args_cli, 'livestream', 0)}")
    print(f"Debug livestream: {args_cli.debug_livestream}")
    print(f"Step mode: {args_cli.step_mode}")
    print(f"Keyboard control: {args_cli.keyboard_control}")
    print(f"Initial terrain row/col: {args_cli.terrain_row}/{args_cli.terrain_col}")
    print(f"{'=' * 80}\n")

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    _configure_reference_trajectory(
        env_cfg,
        use_raw_reference_trajectory=bool(args_cli.use_raw_reference_trajectory),
    )
    env_cfg.planner_backend = str(args_cli.planner_backend)

    render_mode = _resolve_render_mode(args_cli)
    if render_mode is not None:
        env_cfg.sim.enable_cameras = True
    if args_cli.video:
        print(f"[Video] Recording enabled (length={args_cli.video_length})", flush=True)
    debug.mark_startup("env cfg setup")

    print(f"[INFO][play.py] gym.make({task_id!r}) ... (scene build can take several minutes)", flush=True)
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    print("[INFO][play.py] gym.make done.", flush=True)
    debug.mark_startup("gym.make")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{args_cli.checkpoint.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording video during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        debug.mark_startup("video wrapper")

    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env = env.unwrapped
    _apply_initial_terrain_selection(
        base_env,
        terrain_row=args_cli.terrain_row,
        terrain_col=args_cli.terrain_col,
        env_id=0,
    )
    _attach_reference_manager_if_enabled(base_env, env_cfg, experiment_name)
    step_probe = _install_env_step_probes(base_env, enabled=bool(args_cli.debug_livestream))

    print("\n[Wrapper] Creating RSL-RL environment wrapper...", flush=True)
    wrapped_env = _make_env_wrapper(
        base_env,
        gym_module=gym,
        vec_env_cls=VecEnv,
        tensor_dict_cls=TensorDict,
        clip_actions=100.0,
    )
    debug.mark_startup("wrapper init")

    print("\n[Environment] Created successfully", flush=True)
    print(f"  - Observation space: {wrapped_env.observation_space}", flush=True)
    print(f"  - Action space: {wrapped_env.action_space}", flush=True)
    print(f"  - Device: {wrapped_env.device}", flush=True)
    print(f"  - Render mode: {render_mode}", flush=True)
    print(f"  - Render interval: {base_env.cfg.sim.render_interval}", flush=True)

    train_cfg = get_train_cfg(experiment_name)
    print("\n[Runner] Creating OnPolicyRunner...", flush=True)
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=env_cfg.sim.device)
    debug.mark_startup("runner init")

    print(f"\n[Checkpoint] Loading model from: {checkpoint_path}", flush=True)
    runner.load(checkpoint_path, load_optimizer=False)
    print("[Policy] Loaded successfully", flush=True)
    debug.mark_startup("checkpoint load")

    if args_cli.sample:
        policy = runner.alg.policy.act
    else:
        policy = runner.get_inference_policy(device=wrapped_env.device)
    print(f"[Policy] Using {'sampling' if args_cli.sample else 'inference'} mode", flush=True)

    obs, _ = wrapped_env.consume_initial_observations()
    timestep = 0
    camera_interval = _livestream_camera_update_interval(getattr(args_cli, "livestream", 0))
    debug.mark_startup("first observations")
    debug.print_startup_summary()
    if args_cli.debug_livestream:
        print(
            f"[debug-livestream] camera follow interval={camera_interval} "
            f"(livestream={getattr(args_cli, 'livestream', 0)}, num_envs={args_cli.num_envs})",
            flush=True,
        )

    print(f"\n{'=' * 80}")
    print("Starting Play Loop")
    if args_cli.step_mode:
        print("Step mode enabled: press Space to advance one env/render step.", flush=True)
    print(f"{'=' * 80}\n")

    keyboard_controller = _KeyboardVelocityController(
        enabled=bool(args_cli.keyboard_control),
        linear_speed=float(args_cli.keyboard_linear_speed),
        lateral_speed=float(args_cli.keyboard_lateral_speed),
        yaw_speed=float(args_cli.keyboard_yaw_speed),
        speed_step=float(args_cli.keyboard_speed_step),
    )

    try:
        with _TerminalStepGate(enabled=bool(args_cli.step_mode)) as step_gate, keyboard_controller:
            while _play_loop_should_continue(simulation_app, timestep=timestep, max_steps=args_cli.max_steps):
                if not step_gate.wait_for_step():
                    if args_cli.step_mode:
                        _pump_play_paused_window(base_env)
                    continue
                step_start = perf_counter()
                with torch.inference_mode():
                    _apply_keyboard_velocity_command(base_env, keyboard_controller)
                    obs, _ = wrapped_env.get_observations()
                    policy_start = perf_counter()
                    actions = policy(obs)
                    policy_s = perf_counter() - policy_start

                    _apply_keyboard_velocity_command(base_env, keyboard_controller)
                    env_start = perf_counter()
                    obs, rewards, dones, extras = wrapped_env.step(actions)
                    env_step_s = perf_counter() - env_start
                    _apply_keyboard_velocity_command(base_env, keyboard_controller)

                timestep += 1

                camera_s = 0.0
                if _should_update_follow_camera(
                    timestep=timestep,
                    num_envs=args_cli.num_envs,
                    livestream=getattr(args_cli, "livestream", 0),
                    interval=camera_interval,
                ):
                    camera_start = perf_counter()
                    robot_pos = base_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                    camera_position, target_position = _compute_follow_camera_pose(robot_pos)
                    base_env.sim.set_camera_view(camera_position, target_position)
                    camera_s = perf_counter() - camera_start

                total_s = perf_counter() - step_start
                debug.add_loop_sample(
                    policy_s=policy_s,
                    env_step_s=env_step_s,
                    camera_s=camera_s,
                    total_s=total_s,
                    timestep=timestep,
                    step_probe=step_probe.snapshot_and_reset() if args_cli.debug_livestream else None,
                )

                if args_cli.video and timestep == args_cli.video_length:
                    break
                if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
                    break

    except KeyboardInterrupt:
        print("\n[Play] Interrupted by user")

    finally:
        wrapped_env.env.close()
        debug.print_loop_summary(prefix="[debug-livestream][final]")
        print(f"\n{'=' * 80}")
        print(f"Play Complete - Timesteps: {timestep}")
        print(f"{'=' * 80}\n")
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
