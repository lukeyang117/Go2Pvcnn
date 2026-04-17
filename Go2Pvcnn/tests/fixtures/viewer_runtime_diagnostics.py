from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import torch


@dataclass(frozen=True, slots=True)
class CommandCase:
    name: str
    command: tuple[float, float, float]


COMMAND_CASES: tuple[CommandCase, ...] = (
    CommandCase("standstill", (0.0, 0.0, 0.0)),
    CommandCase("forward", (0.3, 0.0, 0.0)),
    CommandCase("backward", (-0.3, 0.0, 0.0)),
    CommandCase("lateral_left", (0.0, 0.25, 0.0)),
    CommandCase("lateral_right", (0.0, -0.25, 0.0)),
    CommandCase("yaw_left", (0.0, 0.0, 0.3)),
    CommandCase("yaw_right", (0.0, 0.0, -0.3)),
    CommandCase("batched_forward", (0.1, 0.0, 0.0)),
    CommandCase("batched_lateral_left", (0.0, 0.08, 0.0)),
)


def build_command_cases(*, device: torch.device, num_envs: int) -> dict[str, torch.Tensor]:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")

    return {
        case.name: torch.tensor(case.command, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1).clone()
        for case in COMMAND_CASES
    }


REPO_ROOT = Path(__file__).resolve().parents[3]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@dataclass(slots=True)
class _RuntimeAppState:
    launcher: object
    app: object
    device: str


@dataclass(frozen=True, slots=True)
class PlaybackReadback:
    root_pos_w: torch.Tensor
    joint_pos: torch.Tensor


@dataclass(frozen=True, slots=True)
class RuntimePlanDiagnostics:
    name: str
    command: torch.Tensor
    result: object
    summary: dict[str, float | bool]
    touchdown_xy_deltas: torch.Tensor
    touchdown_xy_delta_norms: torch.Tensor
    left_touchdown_mean_y: float
    right_touchdown_mean_y: float


@dataclass(frozen=True, slots=True)
class BatchedRuntimeCacheDiagnostics:
    cache: object
    root_pos_w: torch.Tensor
    path_deltas: torch.Tensor


@dataclass(frozen=True, slots=True)
class BatchedRuntimePlanDiagnostics:
    result: object
    root_pos_w: torch.Tensor
    path_deltas: torch.Tensor


_APP_STATE: _RuntimeAppState | None = None


def _close_runtime_app() -> None:
    global _APP_STATE

    if _APP_STATE is None:
        return

    try:
        _APP_STATE.app.close()
    except Exception:
        pass
    _APP_STATE = None


def _ensure_runtime_app(*, device: str) -> _RuntimeAppState:
    global _APP_STATE

    if _APP_STATE is not None:
        if _APP_STATE.device != device:
            raise RuntimeError(f"real runtime app already launched on {_APP_STATE.device}, cannot switch to {device}")
        return _APP_STATE

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(["--headless", "--device", device])
    args_cli.enable_cameras = False
    args_cli.livestream = 0
    launcher = _construct_runtime_launcher(AppLauncher, args_cli)
    _APP_STATE = _RuntimeAppState(launcher=launcher, app=launcher.app, device=device)
    atexit.register(_close_runtime_app)
    return _APP_STATE


def _construct_runtime_launcher(launcher_cls, args_cli):
    launcher = launcher_cls.__new__(launcher_cls)
    try:
        launcher.__init__(args_cli)
    except Exception:
        app = getattr(launcher, "app", None)
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
        raise
    return launcher


def _candidate_runtime_devices(requested_device: str | None) -> list[str]:
    env_device = os.environ.get("VIEWER_RUNTIME_DIAGNOSTICS_DEVICE")
    if env_device:
        return [env_device]
    if requested_device:
        return [requested_device]
    if not torch.cuda.is_available():
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception:
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]

    scored: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        try:
            free_mb = int(parts[1])
        except ValueError:
            continue
        scored.append((free_mb, f"cuda:{parts[0]}"))
    if not scored:
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [device for _, device in scored]


def _is_runtime_resource_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "unable to allocate memory",
            "out of memory",
            "mgpucontactpairsdev",
        )
    )


class RealViewerRuntimeFixture:
    def __init__(
        self,
        *,
        num_envs: int,
        device: str = "cuda:0",
        terrain: str = "flat",
        warmup_steps: int = 6,
        requested_n_frames: int = 20,
        planner_max_touchdown_xy_reach: float = 0.22,
    ) -> None:
        self._closed = False
        self.env = None
        try:
            _ensure_runtime_app(device=device)

            import gymnasium as gym

            import go2_pvcnn.tasks.register_envs  # noqa: F401
            from extension.batched_planner.trajectory import batched_generate_trajectory
            from extension.viz import go2_foostep_planner as viewer_module

            self._gym = gym
            self._viewer = viewer_module
            self._batched_generate_trajectory = batched_generate_trajectory
            self.num_envs = int(num_envs)
            self.device = str(device)
            self.terrain = str(terrain)
            self.warmup_steps = int(warmup_steps)
            self.requested_n_frames = int(requested_n_frames)

            args_cli = SimpleNamespace(num_envs=self.num_envs, device=self.device, terrain=self.terrain)
            self.env_cfg = self._viewer._build_env_cfg(args_cli)
            self.planner_cfg = self._viewer._build_planner_cfg(self.env_cfg)
            self.planner_cfg.max_touchdown_xy_reach = float(planner_max_touchdown_xy_reach)
            self.plan_dt = float(getattr(self.env_cfg, "dt", self.env_cfg.decimation * self.env_cfg.sim.dt))

            self.env = self._gym.make(
                "Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0",
                cfg=self.env_cfg,
                render_mode=None,
            )
            self.base_env = self.env.unwrapped
            self._viewer._attach_reference_manager_if_enabled(self.base_env, self.env_cfg)
            self.zero_actions = self._viewer._make_zero_actions(self.base_env)
            self.robot = self.base_env.scene["robot"]
            foot_ids, foot_names = self.robot.find_bodies(".*_foot")
            self.foot_ids = torch.as_tensor(foot_ids, dtype=torch.long, device=self.base_env.device)
            self.foot_names = tuple(name.replace("_foot", "") for name in foot_names)
            self.command_cases = build_command_cases(device=self.base_env.device, num_envs=self.num_envs)
            self.reset()
        except Exception:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
            _close_runtime_app()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self.env.close()
        self._closed = True
        _close_runtime_app()

    def reset(self) -> None:
        self.env.reset()
        for _ in range(self.warmup_steps):
            self.env.step(self.zero_actions)

    def _command_tensor(self, name: str) -> torch.Tensor:
        command = self.command_cases[name]
        if command.shape[0] != self.num_envs:
            raise RuntimeError(f"command case batch mismatch: expected {self.num_envs}, got {command.shape[0]}")
        return command.to(device=self.base_env.device, dtype=torch.float64)

    def _single_env_state(self):
        return self._viewer._planner_state_from_env(self.base_env, self.foot_ids.tolist())

    def _single_env_terrain(self):
        terrain, _ = self._viewer._compute_local_terrain(self.base_env.scene.sensors["height_scanner"], env_id=0)
        return terrain

    def plan_case(self, name: str) -> RuntimePlanDiagnostics:
        self.reset()
        state = self._single_env_state()
        terrain = self._single_env_terrain()
        command = self._command_tensor(name)[:1]
        result = self._batched_generate_trajectory(
            terrain,
            state,
            command,
            requested_n_frames=self.requested_n_frames,
            dt=self.plan_dt,
            cfg=self.planner_cfg,
        )
        summary = self._viewer._trajectory_motion_summary(result)
        touchdown_xy_deltas = torch.as_tensor(
            result.planned_touchdown_w[:, :, :2] - state.foot_pos[:, :, :2],
            dtype=torch.float64,
        ).clone()
        touchdown_xy_delta_norms = torch.linalg.vector_norm(touchdown_xy_deltas[0], dim=-1)
        left_touchdown_mean_y = float(touchdown_xy_deltas[0, (0, 2), 1].mean().item())
        right_touchdown_mean_y = float(touchdown_xy_deltas[0, (1, 3), 1].mean().item())
        return RuntimePlanDiagnostics(
            name=name,
            command=command.clone(),
            result=result,
            summary=summary,
            touchdown_xy_deltas=touchdown_xy_deltas,
            touchdown_xy_delta_norms=touchdown_xy_delta_norms,
            left_touchdown_mean_y=left_touchdown_mean_y,
            right_touchdown_mean_y=right_touchdown_mean_y,
        )

    def playback_sync_authoritative_readback(self, result, *, frame_idx: int) -> PlaybackReadback:
        self._viewer._apply_direct_playback_to_robot(self.robot, result, frame_idx=int(frame_idx))
        self.base_env.scene.write_data_to_sim()
        self.base_env.sim.render()
        self.base_env.scene.update(float(self.base_env.physics_dt))
        batch = int(result.root_pos_w.shape[0])
        return PlaybackReadback(
            root_pos_w=torch.as_tensor(self.robot.data.root_pos_w[:batch], dtype=torch.float64).clone(),
            joint_pos=torch.as_tensor(self.robot.data.joint_pos[:batch], dtype=torch.float64).clone(),
        )

    def refresh_manager_cache(self, case_names: list[str], *, episode_length_step: int) -> BatchedRuntimeCacheDiagnostics:
        if len(case_names) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} case names, got {len(case_names)}")

        self.reset()
        manager = self.base_env._trajectory_manager
        manager._cache = None
        manager._last_episode_length_buf = None
        manager._last_replan_episode_length_buf = None
        manager._last_commands = None
        commands = torch.stack([self._command_tensor(name)[env_id] for env_id, name in enumerate(case_names)], dim=0)
        command_buffer = self.base_env.command_manager.get_command("base_velocity")
        command_buffer[:] = commands.to(device=command_buffer.device, dtype=command_buffer.dtype)
        self.base_env.episode_length_buf[:] = int(episode_length_step)
        cache = manager.refresh_from_env(self.base_env)
        root_pos_w = torch.as_tensor(cache.root_pos_w, dtype=torch.float64).clone()
        path_deltas = (root_pos_w[:, -1] - root_pos_w[:, 0]).clone()
        return BatchedRuntimeCacheDiagnostics(
            cache=cache,
            root_pos_w=root_pos_w,
            path_deltas=path_deltas,
        )

    def plan_batched_cases(self, case_names: list[str]) -> BatchedRuntimePlanDiagnostics:
        if len(case_names) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} case names, got {len(case_names)}")

        self.reset()
        manager = self.base_env._trajectory_manager
        commands = torch.stack([self._command_tensor(name)[env_id] for env_id, name in enumerate(case_names)], dim=0)
        states = manager._batched_state_from_env(self.base_env)
        terrain = manager._terrain_from_env(self.base_env)
        result = self._batched_generate_trajectory(
            terrain,
            states,
            commands,
            requested_n_frames=self.requested_n_frames,
            dt=self.plan_dt,
            cfg=self.planner_cfg,
        )
        root_pos_w = torch.as_tensor(result.root_pos_w, dtype=torch.float64).clone()
        path_deltas = (root_pos_w[:, -1] - root_pos_w[:, 0]).clone()
        return BatchedRuntimePlanDiagnostics(
            result=result,
            root_pos_w=root_pos_w,
            path_deltas=path_deltas,
        )


def make_real_runtime_fixture(**kwargs) -> RealViewerRuntimeFixture:
    import pytest

    requested_device = kwargs.pop("device", None)
    candidates = _candidate_runtime_devices(requested_device)
    if not candidates:
        pytest.skip("real Isaac runtime requires CUDA, but no CUDA device is available")

    failures: list[str] = []
    for device in candidates:
        try:
            return RealViewerRuntimeFixture(device=device, **kwargs)
        except Exception as exc:
            _close_runtime_app()
            if not _is_runtime_resource_error(exc):
                raise
            failures.append(f"{device}: {type(exc).__name__}: {exc}")

    joined = "; ".join(failures)
    pytest.skip(
        "real Isaac runtime unavailable after trying GPU candidates "
        f"{candidates}; resource-related init failures: {joined}"
    )


__all__ = [
    "BatchedRuntimeCacheDiagnostics",
    "BatchedRuntimePlanDiagnostics",
    "COMMAND_CASES",
    "CommandCase",
    "PlaybackReadback",
    "RealViewerRuntimeFixture",
    "RuntimePlanDiagnostics",
    "build_command_cases",
    "make_real_runtime_fixture",
]
