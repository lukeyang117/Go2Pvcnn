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
    semantic_diagnostics: dict[str, float | int]
    touchdown_xy_deltas: torch.Tensor
    touchdown_xy_delta_norms: torch.Tensor
    left_touchdown_mean_y: float
    right_touchdown_mean_y: float


@dataclass(frozen=True, slots=True)
class PlannerStageSnapshot:
    name: str
    tensors: dict[str, torch.Tensor]
    scalars: dict[str, float | bool | int]

    @property
    def primary_tensor(self) -> torch.Tensor:
        for value in self.tensors.values():
            return value
        raise RuntimeError(f"stage '{self.name}' does not contain tensor diagnostics")


@dataclass(frozen=True, slots=True)
class RuntimePlanStageDiagnostics:
    plan: RuntimePlanDiagnostics
    stages: dict[str, PlannerStageSnapshot]
    stage_order: tuple[str, ...]
    stage_summaries: dict[str, dict[str, float]]


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


@dataclass(frozen=True, slots=True)
class BatchedRuntimePlanStageDiagnostics:
    plan: BatchedRuntimePlanDiagnostics
    stages: dict[str, PlannerStageSnapshot]
    stage_order: tuple[str, ...]
    stage_summaries: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class PlaybackDivergenceReport:
    frame_idx: int
    root_pos_max_abs: float
    root_pos_mean_abs: float
    joint_pos_max_abs: float
    joint_pos_mean_abs: float
    plan: RuntimePlanStageDiagnostics


@dataclass(frozen=True, slots=True)
class ViewerStyleReplanReport:
    command_name: str
    cycle_summaries: tuple[dict[str, float | bool], ...]
    cycle_stage_summaries: tuple[dict[str, dict[str, float]], ...]


_APP_STATE: _RuntimeAppState | None = None


def _quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = torch.as_tensor(quat_wxyz, dtype=torch.float64)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _batch_size_from_tensors(tensors: dict[str, torch.Tensor]) -> float:
    for value in tensors.values():
        if value.ndim > 0:
            return float(value.shape[0])
    return 0.0


def _constant_over_time_ratio(values: torch.Tensor, *, tol: float = 1e-6) -> float:
    if values.ndim < 2:
        return 0.0
    reduce_dims = tuple(range(1, values.ndim))
    delta = torch.amax(torch.abs(values - values[:, :1]), dim=reduce_dims)
    return float((delta <= float(tol)).to(torch.float64).mean().item())


def _summarize_stage_snapshot(
    snapshot: PlannerStageSnapshot,
    *,
    input_foot_pos: torch.Tensor | None,
) -> dict[str, float]:
    summary: dict[str, float] = {"batch_size": _batch_size_from_tensors(snapshot.tensors)}
    tensors = snapshot.tensors

    commands = tensors.get("commands")
    if commands is not None:
        summary["command_vx_mean"] = float(commands[:, 0].mean().item())
        summary["command_vy_mean"] = float(commands[:, 1].mean().item())
        summary["command_yaw_mean"] = float(commands[:, 2].mean().item())

    contact = tensors.get("contact_state", tensors.get("contact_seq"))
    if contact is not None:
        summary["contact_mean"] = float(contact.to(torch.float64).mean().item())

    standstill_mask = tensors.get("standstill_mask")
    if standstill_mask is not None:
        summary["standstill_ratio"] = float(standstill_mask.to(torch.float64).mean().item())

    path = tensors.get("root_pos_w", tensors.get("base_pos_approx"))
    if path is not None and path.ndim >= 3:
        delta = path[:, -1] - path[:, 0]
        summary["path_dx_mean"] = float(delta[:, 0].mean().item())
        summary["path_dy_mean"] = float(delta[:, 1].mean().item())
        summary["path_dz_mean"] = float(delta[:, 2].mean().item())
        path_standstill_ratio = _constant_over_time_ratio(path)
    else:
        path_standstill_ratio = None

    yaw = tensors.get("yaw_approx")
    if yaw is not None and yaw.ndim >= 2:
        yaw_delta = yaw[:, -1] - yaw[:, 0]
        summary["yaw_delta_mean"] = float(yaw_delta.mean().item())
        summary["yaw_delta_abs_mean"] = float(yaw_delta.abs().mean().item())
        if "standstill_ratio" not in summary:
            yaw_standstill_ratio = _constant_over_time_ratio(yaw)
            if path_standstill_ratio is None:
                summary["standstill_ratio"] = yaw_standstill_ratio
            else:
                summary["standstill_ratio"] = min(path_standstill_ratio, yaw_standstill_ratio)
    else:
        root_quat = tensors.get("root_quat_w")
        if root_quat is not None and root_quat.ndim >= 3:
            yaw_values = _quat_wxyz_to_yaw(root_quat)
            yaw_delta = yaw_values[:, -1] - yaw_values[:, 0]
            summary["yaw_delta_mean"] = float(yaw_delta.mean().item())
            summary["yaw_delta_abs_mean"] = float(yaw_delta.abs().mean().item())
            if "standstill_ratio" not in summary:
                quat_standstill_ratio = _constant_over_time_ratio(root_quat)
                if path_standstill_ratio is None:
                    summary["standstill_ratio"] = quat_standstill_ratio
                else:
                    summary["standstill_ratio"] = min(path_standstill_ratio, quat_standstill_ratio)
        elif path_standstill_ratio is not None and "standstill_ratio" not in summary:
            summary["standstill_ratio"] = path_standstill_ratio

    touchdowns = tensors.get("planned_touchdown_w", tensors.get("touchdowns"))
    if touchdowns is not None and input_foot_pos is not None:
        touchdown_xy_delta = touchdowns[:, :, :2] - input_foot_pos[:, :, :2]
        touchdown_norms = torch.linalg.vector_norm(touchdown_xy_delta, dim=-1)
        summary["touchdown_dx_mean"] = float(touchdown_xy_delta[..., 0].mean().item())
        summary["touchdown_delta_norm_max"] = float(touchdown_norms.max().item())
        summary["touchdown_delta_norm_span"] = float((touchdown_norms.max() - touchdown_norms.min()).item())
        summary["left_touchdown_mean_y"] = float(touchdown_xy_delta[:, (0, 2), 1].mean().item())
        summary["right_touchdown_mean_y"] = float(touchdown_xy_delta[:, (1, 3), 1].mean().item())

    feasible = tensors.get("feasible")
    if feasible is not None:
        summary["feasible_ratio"] = float(feasible.to(torch.float64).mean().item())

    return summary


def _summarize_stage_snapshots(
    stages: dict[str, PlannerStageSnapshot],
    *,
    input_foot_pos: torch.Tensor | None,
) -> dict[str, dict[str, float]]:
    return {
        name: _summarize_stage_snapshot(snapshot, input_foot_pos=input_foot_pos)
        for name, snapshot in stages.items()
    }


def format_stage_summary_report(name: str, stage_summaries: dict[str, dict[str, float]], *, stage_order: tuple[str, ...] | None = None) -> str:
    ordered_names = list(stage_order) if stage_order is not None else list(stage_summaries.keys())
    seen: set[str] = set()
    lines = [f"[planner-diag] case={name}"]
    for stage_name in ordered_names:
        if stage_name in seen or stage_name not in stage_summaries:
            continue
        seen.add(stage_name)
        summary = stage_summaries[stage_name]
        parts: list[str] = []
        for key in (
            "command_vx_mean",
            "command_vy_mean",
            "command_yaw_mean",
            "path_dx_mean",
            "path_dy_mean",
            "yaw_delta_mean",
            "standstill_ratio",
            "touchdown_dx_mean",
            "touchdown_delta_norm_max",
            "feasible_ratio",
            "contact_mean",
        ):
            if key in summary:
                parts.append(f"{key}={summary[key]:+.4f}")
        if not parts:
            parts.append("no_numeric_summary")
        lines.append(f"  - {stage_name}: " + " ".join(parts))
    return "\n".join(lines)


def format_playback_divergence_report(report: PlaybackDivergenceReport) -> str:
    result_summary = report.plan.stage_summaries.get("result", {})
    parts = [
        f"[playback-diag] case={report.plan.plan.name}",
        f"frame_idx={report.frame_idx}",
        f"root_pos_max_abs={report.root_pos_max_abs:.6f}",
        f"root_pos_mean_abs={report.root_pos_mean_abs:.6f}",
        f"joint_pos_max_abs={report.joint_pos_max_abs:.6f}",
        f"joint_pos_mean_abs={report.joint_pos_mean_abs:.6f}",
    ]
    for key in ("path_dx_mean", "path_dy_mean", "yaw_delta_mean", "standstill_ratio"):
        if key in result_summary:
            parts.append(f"{key}={result_summary[key]:+.6f}")
    return " ".join(parts)


class _StageSnapshotCollector:
    def __init__(self) -> None:
        self._stage_order: list[str] = []
        self._stages: dict[str, PlannerStageSnapshot] = {}

    def capture(self, name: str, payload: dict[str, object]) -> None:
        tensors: dict[str, torch.Tensor] = {}
        scalars: dict[str, float | bool | int] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = torch.as_tensor(value)
            elif isinstance(value, (bool, int, float)):
                scalars[key] = value
        self._stage_order.append(str(name))
        self._stages[str(name)] = PlannerStageSnapshot(name=str(name), tensors=tensors, scalars=scalars)

    def finish(self) -> tuple[tuple[str, ...], dict[str, PlannerStageSnapshot]]:
        return tuple(self._stage_order), dict(self._stages)


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
        planner_backend: str = "legacy",
        heightmap_viz_stride: int = 10,
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
            self.planner_backend = str(planner_backend)
            self.heightmap_viz_stride = int(heightmap_viz_stride)

            args_cli = SimpleNamespace(
                num_envs=self.num_envs,
                device=self.device,
                terrain=self.terrain,
                planner_backend=self.planner_backend,
                n_frames=self.requested_n_frames,
                plan_dt=0.02,
            )
            self.env_cfg = self._viewer._build_env_cfg(args_cli)
            self._configure_compact_semantic_runtime_grid()
            self.planner_cfg = self._viewer._build_planner_cfg(self.env_cfg)
            self.planner_cfg.max_touchdown_xy_reach = float(planner_max_touchdown_xy_reach)
            self.together_planner_cfg = self._viewer._build_together_planner_cfg(self.env_cfg)
            self.plan_dt = float(getattr(self.env_cfg, "plan_dt", self.env_cfg.decimation * self.env_cfg.sim.dt))

            self.env = self._gym.make(
                "Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0",
                cfg=self.env_cfg,
                render_mode=None,
            )
            self.base_env = self.env.unwrapped
            self._viewer._attach_reference_manager_if_enabled(self.base_env, self.env_cfg)
            self.zero_actions = self._viewer._make_zero_actions(self.base_env)
            self.robot = self.base_env.scene["robot"]
            self.scanner_name = self._viewer._reference_height_scanner_name(self.env_cfg)
            self.scanner = self.base_env.scene.sensors[self.scanner_name]
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

    def _configure_compact_semantic_runtime_grid(self) -> None:
        """Shrink semantic-course runtime smoke to a 4x1 terrain grid.

        The feature config keeps training-aligned terrain dimensions. For real headless
        runtime diagnostics we only need one representative tile per semantic stage,
        so reducing the terrain grid keeps Isaac startup bounded while preserving
        `S1..S4` coverage.
        """
        scene = getattr(self.env_cfg, "scene", None)
        if scene is None:
            return
        if not hasattr(scene, "semantic_height_scanner") or getattr(scene, "semantic_height_scanner") is None:
            return
        terrain_cfg = getattr(scene, "terrain", None)
        terrain_gen = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg is not None else None
        if terrain_gen is None:
            return
        terrain_gen.num_rows = 4
        terrain_gen.num_cols = 1
        if hasattr(terrain_cfg, "max_init_terrain_level"):
            terrain_cfg.max_init_terrain_level = 3

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
        if self.planner_backend == "together":
            return self._viewer._together_state_from_env(self.base_env, self.foot_ids.tolist())
        return self._viewer._planner_state_from_env(self.base_env, self.foot_ids.tolist())

    def _single_env_terrain_and_hits(self):
        if self.planner_backend == "together":
            return self._viewer._compute_together_local_terrain(self.scanner, env_id=0)
        return self._viewer._compute_local_terrain(self.scanner, env_id=0)

    def _single_env_terrain(self):
        terrain, _ = self._single_env_terrain_and_hits()
        return terrain

    def _semantic_scan_diagnostics(self, ray_hits: torch.Tensor) -> dict[str, float | int]:
        semantic_map = self._viewer._scanner_semantic_map(self.scanner, env_id=0)
        _, diagnostics = self._viewer._subsample_semantic_height_points(
            ray_hits,
            semantic_map,
            self.heightmap_viz_stride,
        )
        return diagnostics

    def _build_runtime_plan_diagnostics(self, *, name: str, command: torch.Tensor, state, result) -> RuntimePlanDiagnostics:
        summary = self._viewer._trajectory_motion_summary(result)
        touchdown_xy_deltas = torch.as_tensor(
            result.planned_touchdown_w[:, :, :2] - state.foot_pos[:, :, :2],
            dtype=torch.float64,
        ).clone()
        touchdown_xy_delta_norms = torch.linalg.vector_norm(touchdown_xy_deltas[0], dim=-1)
        left_touchdown_mean_y = float(touchdown_xy_deltas[0, (0, 2), 1].mean().item())
        right_touchdown_mean_y = float(touchdown_xy_deltas[0, (1, 3), 1].mean().item())
        _terrain, ray_hits = self._single_env_terrain_and_hits()
        return RuntimePlanDiagnostics(
            name=name,
            command=command.clone(),
            result=result,
            summary=summary,
            semantic_diagnostics=self._semantic_scan_diagnostics(ray_hits),
            touchdown_xy_deltas=touchdown_xy_deltas,
            touchdown_xy_delta_norms=touchdown_xy_delta_norms,
            left_touchdown_mean_y=left_touchdown_mean_y,
            right_touchdown_mean_y=right_touchdown_mean_y,
        )

    def _build_stage_diagnostics(self, collector: _StageSnapshotCollector):
        stage_order, stages = collector.finish()
        input_stage = stages.get("input")
        input_foot_pos = None if input_stage is None else input_stage.tensors.get("foot_pos")
        stage_summaries = _summarize_stage_snapshots(stages, input_foot_pos=input_foot_pos)
        return stage_order, stages, stage_summaries

    def plan_case(self, name: str) -> RuntimePlanDiagnostics:
        self.reset()
        state = self._single_env_state()
        terrain = self._single_env_terrain()
        command = self._command_tensor(name)[:1]
        result = self._viewer._plan_viewer_trajectory(
            backend=self.planner_backend,
            terrain=terrain,
            state=state,
            command=command,
            requested_n_frames=self.requested_n_frames,
            dt=self.plan_dt,
            legacy_cfg=self.planner_cfg,
            together_cfg=self.together_planner_cfg,
        )
        return self._build_runtime_plan_diagnostics(name=name, command=command, state=state, result=result)

    def plan_case_with_stage_diagnostics(self, name: str) -> RuntimePlanStageDiagnostics:
        if self.planner_backend != "legacy":
            raise RuntimeError("stage diagnostics are only available for the legacy batched planner path")
        self.reset()
        state = self._single_env_state()
        terrain = self._single_env_terrain()
        command = self._command_tensor(name)[:1]
        collector = _StageSnapshotCollector()
        result = self._batched_generate_trajectory(
            terrain,
            state,
            command,
            requested_n_frames=self.requested_n_frames,
            dt=self.plan_dt,
            cfg=self.planner_cfg,
            stage_diagnostics=collector.capture,
        )
        plan = self._build_runtime_plan_diagnostics(name=name, command=command, state=state, result=result)
        stage_order, stages, stage_summaries = self._build_stage_diagnostics(collector)
        return RuntimePlanStageDiagnostics(
            plan=plan,
            stages=stages,
            stage_order=stage_order,
            stage_summaries=stage_summaries,
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

    def plan_batched_cases_with_stage_diagnostics(self, case_names: list[str]) -> BatchedRuntimePlanStageDiagnostics:
        if len(case_names) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} case names, got {len(case_names)}")

        self.reset()
        manager = self.base_env._trajectory_manager
        commands = torch.stack([self._command_tensor(name)[env_id] for env_id, name in enumerate(case_names)], dim=0)
        states = manager._batched_state_from_env(self.base_env)
        terrain = manager._terrain_from_env(self.base_env)
        collector = _StageSnapshotCollector()
        result = self._batched_generate_trajectory(
            terrain,
            states,
            commands,
            requested_n_frames=self.requested_n_frames,
            dt=self.plan_dt,
            cfg=self.planner_cfg,
            stage_diagnostics=collector.capture,
        )
        root_pos_w = torch.as_tensor(result.root_pos_w, dtype=torch.float64).clone()
        path_deltas = (root_pos_w[:, -1] - root_pos_w[:, 0]).clone()
        plan = BatchedRuntimePlanDiagnostics(
            result=result,
            root_pos_w=root_pos_w,
            path_deltas=path_deltas,
        )
        stage_order, stages, stage_summaries = self._build_stage_diagnostics(collector)
        return BatchedRuntimePlanStageDiagnostics(
            plan=plan,
            stages=stages,
            stage_order=stage_order,
            stage_summaries=stage_summaries,
        )

    def planner_output_vs_playback_divergence(self, name: str, *, frame_idx: int | None = None) -> PlaybackDivergenceReport:
        plan = self.plan_case_with_stage_diagnostics(name)
        actual_frame_idx = min(7, plan.plan.result.num_frames - 1) if frame_idx is None else int(frame_idx)
        readback = self.playback_sync_authoritative_readback(plan.plan.result, frame_idx=actual_frame_idx)
        root_pos_delta = readback.root_pos_w - plan.plan.result.root_pos_w[:, actual_frame_idx]
        joint_pos_delta = readback.joint_pos - plan.plan.result.joint_angles[:, actual_frame_idx]
        return PlaybackDivergenceReport(
            frame_idx=actual_frame_idx,
            root_pos_max_abs=float(root_pos_delta.abs().max().item()),
            root_pos_mean_abs=float(root_pos_delta.abs().mean().item()),
            joint_pos_max_abs=float(joint_pos_delta.abs().max().item()),
            joint_pos_mean_abs=float(joint_pos_delta.abs().mean().item()),
            plan=plan,
        )

    def viewer_style_replan_sequence(self, name: str, *, num_cycles: int = 3) -> ViewerStyleReplanReport:
        if num_cycles < 1:
            raise ValueError("num_cycles must be >= 1")

        self.reset()
        terrain = self._single_env_terrain()
        command = self._command_tensor(name)[:1]
        state = self._single_env_state()
        cycle_summaries: list[dict[str, float | bool]] = []
        cycle_stage_summaries: list[dict[str, dict[str, float]]] = []

        for cycle_idx in range(num_cycles):
            collector = _StageSnapshotCollector()
            result = self._batched_generate_trajectory(
                terrain,
                state,
                command,
                requested_n_frames=self.requested_n_frames,
                dt=self.plan_dt,
                cfg=self.planner_cfg,
                stage_diagnostics=collector.capture,
            )
            summary = self._viewer._trajectory_motion_summary(result)
            _, stages, stage_summaries = self._build_stage_diagnostics(collector)
            cycle_summaries.append(dict(summary))
            cycle_stage_summaries.append(stage_summaries)

            frame_idx = min(result.num_frames - 1, max(0, result.num_frames - 1))
            state = self._viewer._planner_state_from_reference_result(result, frame_idx=frame_idx)

        return ViewerStyleReplanReport(
            command_name=name,
            cycle_summaries=tuple(cycle_summaries),
            cycle_stage_summaries=tuple(cycle_stage_summaries),
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
    "BatchedRuntimePlanStageDiagnostics",
    "COMMAND_CASES",
    "CommandCase",
    "PlaybackDivergenceReport",
    "PlaybackReadback",
    "PlannerStageSnapshot",
    "RealViewerRuntimeFixture",
    "RuntimePlanDiagnostics",
    "RuntimePlanStageDiagnostics",
    "ViewerStyleReplanReport",
    "build_command_cases",
    "format_playback_divergence_report",
    "format_stage_summary_report",
    "make_real_runtime_fixture",
]
