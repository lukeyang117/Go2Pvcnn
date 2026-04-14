"""Isaac Lab livestream viewer for the batched Go2 footstep planner.

This script keeps ``extension.batched_planner`` read-only and visualizes its
predicted trajectory inside Isaac Lab with terminal teleop controls.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parents[2]
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Visualize batched Go2 footstep planning in Isaac Lab.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac Lab environments.")
    parser.add_argument("--terrain", type=str, default="mixed", choices=["flat", "stairs", "mixed"])
    parser.add_argument("--n-frames", type=int, default=50, help="Planner horizon in frames.")
    parser.add_argument("--plan-dt", type=float, default=0.02, help="Planner integration step.")
    parser.add_argument("--vx-scale", type=float, default=0.8, help="Teleop forward/backward speed.")
    parser.add_argument("--vy-scale", type=float, default=0.4, help="Teleop lateral speed.")
    parser.add_argument("--yaw-scale", type=float, default=1.0, help="Teleop yaw-rate command.")
    parser.add_argument("--key-hold-timeout", type=float, default=0.18, help="Seconds before a key press expires.")
    parser.add_argument("--heightmap-viz-stride", type=int, default=10, help="Subsample stride for heightmap markers.")
    parser.add_argument("--camera-distance", type=float, default=3.2, help="Follow-camera distance behind the robot.")
    parser.add_argument("--camera-height", type=float, default=1.6, help="Follow-camera height offset.")
    parser.add_argument("--warmup-steps", type=int, default=6, help="Number of zero-action warmup steps before visualization.")
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


def _launch_app(args_cli: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    return app_launcher, app_launcher.app

LEG_COLORS = (
    (1.0, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.4, 1.0),
    (1.0, 0.8, 0.2),
)


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
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
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


def _make_marker_cfg(prim_path: str, *, radius: float, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
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
) -> VisualizationMarkersCfg:
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


class _ViewerCommandManagerProxy:
    """Expose teleop commands through the env's shared command manager interface."""

    def __init__(self, command_manager, *, command_name: str, num_envs: int, device: torch.device):
        self._command_manager = command_manager
        self._command_name = str(command_name)
        self._num_envs = int(num_envs)
        self._device = torch.device(device)
        self._command: torch.Tensor | None = None

    def set_command(self, command: torch.Tensor) -> None:
        command_t = torch.as_tensor(command, dtype=torch.float64, device=self._device)
        if command_t.ndim == 1:
            command_t = command_t.unsqueeze(0)
        if command_t.ndim != 2 or int(command_t.shape[-1]) != 3:
            raise ValueError(f"command must have shape (3,) or (num_envs, 3), got {tuple(command_t.shape)}")
        if int(command_t.shape[0]) == 1 and self._num_envs > 1:
            command_t = command_t.expand(self._num_envs, -1).clone()
        elif int(command_t.shape[0]) != self._num_envs:
            raise ValueError(f"command must have shape (1, 3) or ({self._num_envs}, 3), got {tuple(command_t.shape)}")
        self._command = command_t

    @property
    def command(self) -> torch.Tensor | None:
        return self._command

    @command.setter
    def command(self, value: torch.Tensor) -> None:
        self.set_command(value)

    def get_command(self, name: str):
        if name == self._command_name and self._command is not None:
            return self._command
        return self._command_manager.get_command(name)

    def __getattr__(self, name: str):
        return getattr(self._command_manager, name)


def _apply_terrain_mode(env_cfg: TeacherElevationTrajectoryEnvCfg_PLAY, terrain_mode: str) -> None:
    tg = env_cfg.scene.terrain.terrain_generator
    if tg is None:
        return

    for name, sub_terrain in tg.sub_terrains.items():
        sub_terrain.proportion = 0.0

    if terrain_mode == "flat":
        tg.num_rows = 1
        tg.num_cols = 1
        tg.sub_terrains["flat"].proportion = 1.0
    elif terrain_mode == "stairs":
        tg.num_rows = 1
        tg.num_cols = 1
        tg.sub_terrains["pyramid_stairs"].proportion = 1.0
    else:
        tg.num_rows = 1
        tg.num_cols = 2
        tg.sub_terrains["flat"].proportion = 0.5
        tg.sub_terrains["pyramid_stairs"].proportion = 0.5

    env_cfg.scene.terrain.max_init_terrain_level = 0


def _build_env_cfg(args_cli: argparse.Namespace) -> TeacherElevationTrajectoryEnvCfg_PLAY:
    from go2_pvcnn.tasks.teacher_elevation_trajectory_env_cfg import TeacherElevationTrajectoryEnvCfg_PLAY

    env_cfg = TeacherElevationTrajectoryEnvCfg_PLAY()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = 6.0
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"]["x"] = (0.0, 0.0)
    reset_base.params["pose_range"]["y"] = (0.0, 0.0)
    reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    _apply_terrain_mode(env_cfg, args_cli.terrain)
    return env_cfg


def _build_planner_cfg(env_cfg: TeacherElevationTrajectoryEnvCfg_PLAY) -> BatchedTrajectoryConfig:
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
        replan_stop_speed=float(env_cfg.replan_stop_speed),
        replan_velocity_scales=list(env_cfg.replan_velocity_scales),
        replan_yaw_biases=list(env_cfg.replan_yaw_biases),
        replan_vy_biases=list(env_cfg.replan_vy_biases),
    )


def _attach_viewer_reference_manager(base_env: ManagerBasedRLEnv, planner_cfg: BatchedTrajectoryConfig):
    from extension.batched_planner.manager import BatchedTrajectoryManager

    command_name = str(getattr(planner_cfg, "reference_command_name", "base_velocity"))
    proxy = _ViewerCommandManagerProxy(
        base_env.command_manager,
        command_name=command_name,
        num_envs=int(base_env.num_envs),
        device=base_env.device,
    )
    base_env.command_manager = proxy
    manager = BatchedTrajectoryManager(planner_cfg, device=base_env.device)
    base_env._trajectory_manager = manager
    base_env._trajectory_reference_cache = None
    return manager, proxy


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


def _make_zero_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    import gymnasium as gym

    if hasattr(env, "action_manager"):
        action_dim = int(env.action_manager.total_action_dim)
    else:
        action_dim = int(gym.spaces.flatdim(env.single_action_space))
    return torch.zeros((env.num_envs, action_dim), dtype=torch.float32, device=env.device)


def _update_camera(env: ManagerBasedRLEnv, *, root_pos: torch.Tensor, root_yaw: torch.Tensor, distance: float, height: float) -> None:
    yaw_val = float(root_yaw[0].item())
    camera_offset = torch.tensor(
        [-distance * math.cos(yaw_val), -distance * math.sin(yaw_val), height],
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    camera_position = (root_pos + camera_offset).cpu().numpy()
    target_position = (root_pos + torch.tensor([0.0, 0.0, 0.35], device=root_pos.device)).cpu().numpy()
    env.sim.set_camera_view(camera_position, target_position)


def _planner_state_from_env(env: ManagerBasedRLEnv, foot_ids: list[int]):
    from extension.convention import isaac_state_to_planner_state, quat_wxyz_to_xyzw

    robot = env.scene["robot"]
    return isaac_state_to_planner_state(
        root_pos_w=robot.data.root_pos_w[:1].to(dtype=torch.float64),
        root_quat_xyzw=quat_wxyz_to_xyzw(robot.data.root_quat_w[:1]).to(dtype=torch.float64),
        joint_pos=robot.data.joint_pos[:1].to(dtype=torch.float64),
        foot_pos_w=robot.data.body_pos_w[:1, foot_ids, :].to(dtype=torch.float64),
        foot_vel_w=robot.data.body_lin_vel_w[:1, foot_ids, :].to(dtype=torch.float64),
    )


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
    from extension.mdp import rewards_reference
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = _build_env_cfg(args_cli)
    planner_cfg = _build_planner_cfg(env_cfg)
    planner_cfg.dt = float(args_cli.plan_dt)
    planner_cfg.reference_trajectory_horizon = int(args_cli.n_frames)
    planner_cfg.reference_replan_interval_steps = int(env_cfg.reference_replan_interval_steps)

    env = gym.make(
        "Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if getattr(args_cli, "livestream", -1) in (1, 2) else None,
    )
    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env = env.unwrapped
    zero_actions = _make_zero_actions(base_env)
    foot_ids, _ = base_env.scene["robot"].find_bodies(".*_foot")
    scanner = base_env.scene.sensors["height_scanner"]
    visualizer = PlannerVisualizer()
    _, command_proxy = _attach_viewer_reference_manager(base_env, planner_cfg)

    print(f"[Viewer] Terrain mode: {args_cli.terrain}", flush=True)
    print(f"[Viewer] Planner horizon: {args_cli.n_frames} frames @ dt={args_cli.plan_dt:.3f}s", flush=True)
    _print_help()

    env.reset()
    for _ in range(max(0, int(args_cli.warmup_steps))):
        env.step(zero_actions)

    with TerminalTeleop(
        device=base_env.device,
        vx_scale=float(args_cli.vx_scale),
        vy_scale=float(args_cli.vy_scale),
        yaw_scale=float(args_cli.yaw_scale),
        timeout_s=float(args_cli.key_hold_timeout),
    ) as teleop:
        last_status = None
        try:
            while simulation_app.is_running():
                teleop_cmd = teleop.poll()
                command_proxy.command = teleop_cmd.values
                if teleop_cmd.reset_requested:
                    env.reset()
                    for _ in range(max(0, int(args_cli.warmup_steps))):
                        env.step(zero_actions)

                env.step(zero_actions)

                planner_state = _planner_state_from_env(base_env, foot_ids)
                result = rewards_reference.ensure_reference_cache(base_env)
                root_yaw = extract_yaw_batch(planner_state.root_quat)
                ray_hits = scanner.data.ray_hits_w[0].to(dtype=torch.float64)
                height_points = _subsample_height_points(ray_hits, int(args_cli.heightmap_viz_stride))
                visualizer.update(
                    result=result,
                    command=teleop_cmd.values,
                    root_yaw=root_yaw,
                    height_points=height_points,
                )
                _update_camera(
                    base_env,
                    root_pos=planner_state.root_pos[0],
                    root_yaw=root_yaw,
                    distance=float(args_cli.camera_distance),
                    height=float(args_cli.camera_height),
                )

                root_pos = planner_state.root_pos[0]
                yaw_rate = float(teleop_cmd.values[0, 2].item())
                status = (
                    f"\rcmd vx={teleop_cmd.values[0,0]:+0.2f} "
                    f"vy={teleop_cmd.values[0,1]:+0.2f} "
                    f"yaw={yaw_rate:+0.2f} | "
                    f"root=({root_pos[0]:+0.2f}, {root_pos[1]:+0.2f}, {root_pos[2]:+0.2f})"
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
