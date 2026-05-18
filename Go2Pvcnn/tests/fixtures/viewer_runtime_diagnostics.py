from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
import math
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
    CommandCase("forward_yaw_left", (0.25, 0.0, 0.25)),
    CommandCase("forward_yaw_right", (0.25, 0.0, -0.25)),
    CommandCase("diagonal_forward_left", (0.22, 0.18, 0.0)),
    CommandCase("diagonal_forward_right", (0.22, -0.18, 0.0)),
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


def scanner_sync_steps(
    *,
    scanner_update_period: float,
    physics_dt: float,
    minimum_steps: int = 1,
    extra_steps: int = 4,
) -> int:
    """Number of scene updates needed before a post-teleport scanner read."""
    update_period = max(0.0, float(scanner_update_period))
    dt = float(physics_dt)
    if not math.isfinite(update_period) or not math.isfinite(dt) or dt <= 0.0:
        return max(1, int(minimum_steps))
    return max(int(minimum_steps), int(math.ceil(update_period / dt)) + max(1, int(extra_steps)))


def refresh_targeted_scanner_pose(base_env, scanner, *, minimum_steps: int, extra_steps: int = 4) -> int:
    from extension.viz import go2_foostep_planner as viewer_module

    return viewer_module._refresh_viewer_scanner(
        base_env,
        scanner,
        minimum_steps=int(minimum_steps),
        extra_steps=int(extra_steps),
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from extension.batched_together_planner.parameterization import (
    T116_MODE_APPROACH_SMALL,
    T116_MODE_BYPASS_OBSTACLE,
    T116_MODE_CROSS_SMALL,
    T116_MODE_CRUISE,
)
from extension.batched_together_planner.types import HARD_REASON_NAMES


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
    grounded_crossing: "GroundedCrossingDiagnostics | None"
    grounded_crossing_summary: dict[str, object] | None
    touchdown_xy_deltas: torch.Tensor
    touchdown_xy_delta_norms: torch.Tensor
    left_touchdown_mean_y: float
    right_touchdown_mean_y: float


@dataclass(frozen=True, slots=True)
class GroundedCrossingDiagnostics:
    mode: torch.Tensor
    status: torch.Tensor
    feasible: torch.Tensor
    safe_fallback: torch.Tensor
    selected_beta: torch.Tensor
    selected_route: torch.Tensor
    direction_id: torch.Tensor
    command_direction_violation: torch.Tensor
    cross_small_success: torch.Tensor
    body_min_clearance: torch.Tensor
    leg_min_clearance: torch.Tensor
    base_min_clearance_to_small: torch.Tensor
    per_leg_touchdown_on_small_count: torch.Tensor
    per_leg_foot_small_collision_count: torch.Tensor
    per_leg_min_clearance_to_small: torch.Tensor
    per_leg_touchdown_beyond_small_back_edge: torch.Tensor
    touchdown_ground_gap_by_leg: torch.Tensor
    touchdown_semantic_by_leg: torch.Tensor
    state_mode: torch.Tensor
    small_strategy_outcome: torch.Tensor
    front_touchdown_ground_gap: torch.Tensor
    rear_touchdown_ground_gap: torch.Tensor
    touchdown_on_small_count: torch.Tensor
    front_foot_small_collision_count: torch.Tensor
    rear_foot_small_collision_count: torch.Tensor
    base_small_penetration_count: torch.Tensor
    base_path_crosses_small_flag: torch.Tensor
    candidate_hard_reason_mask: torch.Tensor | None = None
    selected_hard_reason_mask: torch.Tensor | None = None
    candidate_hard_rank_cost: torch.Tensor | None = None
    selected_hard_rank_cost: torch.Tensor | None = None
    selected_candidate_index: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class GroundedCrossingRuntimeReport:
    mode_sequence: tuple[int, ...]
    state_sequence: tuple[int, ...]
    small_strategy_sequence: tuple[int, ...]
    status_sequence: tuple[int, ...]
    feasible_sequence: tuple[bool, ...]
    safe_fallback_sequence: tuple[bool, ...]
    selected_beta_sequence: tuple[float, ...]
    selected_route_sequence: tuple[int, ...]
    direction_id_sequence: tuple[int, ...]
    front_touchdown_ground_gap_abs_m: float
    rear_touchdown_ground_gap_abs_m: float
    touchdown_ground_gap_by_leg_abs_m: float
    rear_touchdown_airborne_count: int
    touchdown_on_small_count: int
    foot_small_collision_count: int
    front_foot_small_collision_count: int
    rear_foot_small_collision_count: int
    base_small_penetration_count: int
    base_path_crosses_small_flag: int
    body_min_clearance_m: float
    leg_min_clearance_m: float
    base_min_clearance_to_small_m: float
    per_leg_touchdown_on_small_count: tuple[int, ...]
    per_leg_foot_small_collision_count: tuple[int, ...]
    per_leg_min_clearance_to_small_m: tuple[float, ...]
    per_leg_touchdown_beyond_small_back_edge: tuple[bool, ...]
    touchdown_semantic_by_leg: tuple[int, ...]
    command_direction_violation_count: int
    cross_small_success_count: int
    cross_phase_progression_valid: int
    cross_outcome_grounded: int
    sampled_plans: tuple[RuntimePlanDiagnostics, ...]


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


def _format_hard_reason_mask(mask: torch.Tensor) -> str:
    values = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    names = [name for name, enabled in zip(HARD_REASON_NAMES, values.tolist()) if enabled]
    return "|".join(names) if names else "none"


def format_hard_reason_summary(result) -> str:
    selected_mask = getattr(result, "selected_hard_reason_mask", None)
    candidate_mask = getattr(result, "candidate_hard_reason_mask", None)
    candidate_rank = getattr(result, "candidate_hard_rank_cost", None)
    selected_rank = getattr(result, "selected_hard_rank_cost", None)
    if selected_mask is None or candidate_mask is None or candidate_rank is None:
        return ""
    selected_mask_t = torch.as_tensor(selected_mask, dtype=torch.bool)
    candidate_mask_t = torch.as_tensor(candidate_mask, dtype=torch.bool)
    candidate_rank_t = torch.as_tensor(candidate_rank, dtype=torch.float64)
    selected_rank_t = None if selected_rank is None else torch.as_tensor(selected_rank, dtype=torch.float64)
    selected_reasons = _format_hard_reason_mask(selected_mask_t.reshape(-1, selected_mask_t.shape[-1])[0])
    candidate_reasons = [
        _format_hard_reason_mask(candidate_mask_t.reshape(-1, candidate_mask_t.shape[-2], candidate_mask_t.shape[-1])[0, idx])
        for idx in range(candidate_mask_t.shape[-2])
    ]
    rank_values = [f"{float(value):0.3f}" for value in candidate_rank_t.reshape(-1, candidate_rank_t.shape[-1])[0].tolist()]
    parts = [
        f"selected_hard_reasons={selected_reasons}",
        f"candidate_hard_rank=[{','.join(rank_values)}]",
        f"candidate_hard_reasons=[{';'.join(candidate_reasons)}]",
    ]
    if selected_rank_t is not None:
        parts.insert(1, f"selected_hard_rank_cost={float(selected_rank_t.reshape(-1)[0].item()):0.3f}")
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


def _grounded_crossing_diagnostics_from_result(result) -> GroundedCrossingDiagnostics | None:
    required_names = (
        "mode",
        "status",
        "feasible",
        "safe_fallback",
        "selected_beta",
        "selected_route",
        "direction_id",
        "command_direction_violation",
        "cross_small_success",
        "body_min_clearance",
        "leg_min_clearance",
        "base_min_clearance_to_small",
        "per_leg_touchdown_on_small_count",
        "per_leg_foot_small_collision_count",
        "per_leg_min_clearance_to_small",
        "per_leg_touchdown_beyond_small_back_edge",
        "touchdown_ground_gap_by_leg",
        "touchdown_semantic_by_leg",
    )
    if any(getattr(result, name, None) is None for name in required_names):
        return None
    mode = torch.as_tensor(result.mode).clone()
    touchdown_ground_gap_by_leg = torch.as_tensor(result.touchdown_ground_gap_by_leg, dtype=torch.float64).clone()
    per_leg_touchdown_on_small_count = torch.as_tensor(result.per_leg_touchdown_on_small_count).clone()
    per_leg_foot_small_collision_count = torch.as_tensor(result.per_leg_foot_small_collision_count).clone()
    base_min_clearance = torch.as_tensor(result.base_min_clearance_to_small, dtype=torch.float64).clone()
    front_touchdown_ground_gap = getattr(result, "front_touchdown_ground_gap", None)
    rear_touchdown_ground_gap = getattr(result, "rear_touchdown_ground_gap", None)
    touchdown_on_small_count = getattr(result, "touchdown_on_small_count", None)
    front_foot_small_collision_count = getattr(result, "front_foot_small_collision_count", None)
    rear_foot_small_collision_count = getattr(result, "rear_foot_small_collision_count", None)
    base_small_penetration_count = getattr(result, "base_small_penetration_count", None)
    base_path_crosses_small_flag = getattr(result, "base_path_crosses_small_flag", None)
    return GroundedCrossingDiagnostics(
        mode=mode,
        status=torch.as_tensor(result.status).clone(),
        feasible=torch.as_tensor(result.feasible, dtype=torch.bool).clone(),
        safe_fallback=torch.as_tensor(result.safe_fallback, dtype=torch.bool).clone(),
        selected_beta=torch.as_tensor(result.selected_beta, dtype=torch.float64).clone(),
        selected_route=torch.as_tensor(result.selected_route).clone(),
        direction_id=torch.as_tensor(result.direction_id).clone(),
        command_direction_violation=torch.as_tensor(result.command_direction_violation, dtype=torch.bool).clone(),
        cross_small_success=torch.as_tensor(result.cross_small_success, dtype=torch.bool).clone(),
        body_min_clearance=torch.as_tensor(result.body_min_clearance, dtype=torch.float64).clone(),
        leg_min_clearance=torch.as_tensor(result.leg_min_clearance, dtype=torch.float64).clone(),
        base_min_clearance_to_small=base_min_clearance,
        per_leg_touchdown_on_small_count=per_leg_touchdown_on_small_count,
        per_leg_foot_small_collision_count=per_leg_foot_small_collision_count,
        per_leg_min_clearance_to_small=torch.as_tensor(result.per_leg_min_clearance_to_small, dtype=torch.float64).clone(),
        per_leg_touchdown_beyond_small_back_edge=torch.as_tensor(
            result.per_leg_touchdown_beyond_small_back_edge,
            dtype=torch.bool,
        ).clone(),
        touchdown_ground_gap_by_leg=touchdown_ground_gap_by_leg,
        touchdown_semantic_by_leg=torch.as_tensor(result.touchdown_semantic_by_leg).clone(),
        state_mode=torch.as_tensor(getattr(result, "state_mode", mode)).clone(),
        small_strategy_outcome=torch.as_tensor(getattr(result, "small_strategy_outcome", mode)).clone(),
        front_touchdown_ground_gap=(
            touchdown_ground_gap_by_leg[:, :2].clone()
            if front_touchdown_ground_gap is None
            else torch.as_tensor(front_touchdown_ground_gap, dtype=torch.float64).clone()
        ),
        rear_touchdown_ground_gap=(
            touchdown_ground_gap_by_leg[:, 2:].clone()
            if rear_touchdown_ground_gap is None
            else torch.as_tensor(rear_touchdown_ground_gap, dtype=torch.float64).clone()
        ),
        touchdown_on_small_count=(
            per_leg_touchdown_on_small_count.sum(dim=-1).clone()
            if touchdown_on_small_count is None
            else torch.as_tensor(touchdown_on_small_count).clone()
        ),
        front_foot_small_collision_count=(
            per_leg_foot_small_collision_count[:, :2].sum(dim=-1).clone()
            if front_foot_small_collision_count is None
            else torch.as_tensor(front_foot_small_collision_count).clone()
        ),
        rear_foot_small_collision_count=(
            per_leg_foot_small_collision_count[:, 2:].sum(dim=-1).clone()
            if rear_foot_small_collision_count is None
            else torch.as_tensor(rear_foot_small_collision_count).clone()
        ),
        base_small_penetration_count=(
            (base_min_clearance < 0.0).to(torch.int64)
            if base_small_penetration_count is None
            else torch.as_tensor(base_small_penetration_count).clone()
        ),
        base_path_crosses_small_flag=(
            (base_min_clearance < 0.0).to(torch.bool)
            if base_path_crosses_small_flag is None
            else torch.as_tensor(base_path_crosses_small_flag, dtype=torch.bool).clone()
        ),
        candidate_hard_reason_mask=(
            None
            if getattr(result, "candidate_hard_reason_mask", None) is None
            else torch.as_tensor(result.candidate_hard_reason_mask, dtype=torch.bool).clone()
        ),
        selected_hard_reason_mask=(
            None
            if getattr(result, "selected_hard_reason_mask", None) is None
            else torch.as_tensor(result.selected_hard_reason_mask, dtype=torch.bool).clone()
        ),
        candidate_hard_rank_cost=(
            None
            if getattr(result, "candidate_hard_rank_cost", None) is None
            else torch.as_tensor(result.candidate_hard_rank_cost, dtype=torch.float64).clone()
        ),
        selected_hard_rank_cost=(
            None
            if getattr(result, "selected_hard_rank_cost", None) is None
            else torch.as_tensor(result.selected_hard_rank_cost, dtype=torch.float64).clone()
        ),
        selected_candidate_index=(
            None
            if getattr(result, "selected_candidate_index", None) is None
            else torch.as_tensor(result.selected_candidate_index).clone()
        ),
    )


def _grounded_crossing_summary(diag: GroundedCrossingDiagnostics | None) -> dict[str, object] | None:
    if diag is None:
        return None
    summary = {
        "mode": int(diag.mode.reshape(-1)[0].item()),
        "status": int(diag.status.reshape(-1)[0].item()),
        "feasible": bool(diag.feasible.reshape(-1)[0].item()),
        "safe_fallback": bool(diag.safe_fallback.reshape(-1)[0].item()),
        "selected_beta": float(diag.selected_beta.reshape(-1)[0].item()),
        "selected_route": int(diag.selected_route.reshape(-1)[0].item()),
        "direction_id": int(diag.direction_id.reshape(-1)[0].item()),
        "state_mode": int(diag.state_mode.reshape(-1)[0].item()),
        "small_strategy_outcome": int(diag.small_strategy_outcome.reshape(-1)[0].item()),
        "command_direction_violation": bool(diag.command_direction_violation.reshape(-1)[0].item()),
        "cross_small_success": bool(diag.cross_small_success.reshape(-1)[0].item()),
        "body_min_clearance": float(diag.body_min_clearance.reshape(-1)[0].item()),
        "leg_min_clearance": float(diag.leg_min_clearance.reshape(-1)[0].item()),
        "base_min_clearance_to_small": float(diag.base_min_clearance_to_small.reshape(-1)[0].item()),
        "per_leg_touchdown_on_small_count": tuple(int(value) for value in diag.per_leg_touchdown_on_small_count[0].tolist()),
        "per_leg_foot_small_collision_count": tuple(int(value) for value in diag.per_leg_foot_small_collision_count[0].tolist()),
        "per_leg_min_clearance_to_small": tuple(float(value) for value in diag.per_leg_min_clearance_to_small[0].tolist()),
        "per_leg_touchdown_beyond_small_back_edge": tuple(
            bool(value) for value in diag.per_leg_touchdown_beyond_small_back_edge[0].tolist()
        ),
        "touchdown_ground_gap_by_leg": tuple(float(value) for value in diag.touchdown_ground_gap_by_leg[0].tolist()),
        "touchdown_semantic_by_leg": tuple(int(value) for value in diag.touchdown_semantic_by_leg[0].tolist()),
        "front_touchdown_ground_gap": tuple(float(value) for value in diag.front_touchdown_ground_gap[0].tolist()),
        "rear_touchdown_ground_gap": tuple(float(value) for value in diag.rear_touchdown_ground_gap[0].tolist()),
        "touchdown_on_small_count": int(diag.touchdown_on_small_count.reshape(-1)[0].item()),
        "front_foot_small_collision_count": int(diag.front_foot_small_collision_count.reshape(-1)[0].item()),
        "rear_foot_small_collision_count": int(diag.rear_foot_small_collision_count.reshape(-1)[0].item()),
        "base_small_penetration_count": int(diag.base_small_penetration_count.reshape(-1)[0].item()),
        "base_path_crosses_small_flag": bool(diag.base_path_crosses_small_flag.reshape(-1)[0].item()),
    }
    if diag.selected_candidate_index is not None:
        summary["selected_candidate_index"] = int(diag.selected_candidate_index.reshape(-1)[0].item())
    if diag.selected_hard_rank_cost is not None:
        summary["selected_hard_rank_cost"] = float(diag.selected_hard_rank_cost.reshape(-1)[0].item())
    if diag.selected_hard_reason_mask is not None:
        summary["selected_hard_reasons"] = _format_hard_reason_mask(diag.selected_hard_reason_mask[0])
    return summary


def _grounded_crossing_state_value(value: torch.Tensor) -> int:
    return int(torch.as_tensor(value).reshape(-1)[0].item())


def _grounded_crossing_gap_abs_max(value: torch.Tensor) -> float:
    return float(torch.amax(torch.abs(torch.as_tensor(value, dtype=torch.float64))).item())


def _grounded_crossing_count(value: torch.Tensor) -> int:
    return int(torch.as_tensor(value).reshape(-1)[0].item())


def _grounded_crossing_bool(value: torch.Tensor) -> bool:
    return bool(torch.as_tensor(value, dtype=torch.bool).reshape(-1)[0].item())


def _grounded_crossing_float(value: torch.Tensor) -> float:
    return float(torch.as_tensor(value, dtype=torch.float64).reshape(-1)[0].item())


def _grounded_crossing_first_row(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor.reshape(tensor.shape[0], -1)[0]


def _grounded_crossing_max_per_leg_int(diagnostics: list[GroundedCrossingDiagnostics], name: str) -> tuple[int, ...]:
    stacked = torch.stack(
        [_grounded_crossing_first_row(getattr(diag, name)).to(dtype=torch.int64) for diag in diagnostics],
        dim=0,
    )
    return tuple(int(value) for value in stacked.max(dim=0).values.tolist())


def _grounded_crossing_min_per_leg_float(diagnostics: list[GroundedCrossingDiagnostics], name: str) -> tuple[float, ...]:
    stacked = torch.stack(
        [_grounded_crossing_first_row(getattr(diag, name)).to(dtype=torch.float64) for diag in diagnostics],
        dim=0,
    )
    return tuple(float(value) for value in stacked.min(dim=0).values.tolist())


def _grounded_crossing_any_per_leg_bool(diagnostics: list[GroundedCrossingDiagnostics], name: str) -> tuple[bool, ...]:
    stacked = torch.stack(
        [_grounded_crossing_first_row(getattr(diag, name)).to(dtype=torch.bool) for diag in diagnostics],
        dim=0,
    )
    return tuple(bool(value) for value in stacked.any(dim=0).tolist())


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
        requested_n_frames: int = 50,
        planner_max_touchdown_xy_reach: float = 0.22,
        planner_backend: str = "legacy",
        heightmap_viz_stride: int = 10,
        semantic_small_height_m: float | None = None,
        cobblestone_num_rows: int | None = None,
        cobblestone_num_cols: int | None = None,
        task_id: str | None = None,
        env_cfg_cls=None,
        env_cfg_entry_point: str | None = None,
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
            if env_cfg_cls is None and env_cfg_entry_point is not None:
                module_name, class_name = env_cfg_entry_point.rsplit(":", 1)
                module = __import__(module_name, fromlist=[class_name])
                env_cfg_cls = getattr(module, class_name)
            self.num_envs = int(num_envs)
            self.device = str(device)
            self.terrain = str(terrain)
            self.warmup_steps = int(warmup_steps)
            self.requested_n_frames = int(requested_n_frames)
            self.planner_backend = str(planner_backend)
            self.heightmap_viz_stride = int(heightmap_viz_stride)
            self._cobblestone_num_rows = cobblestone_num_rows
            self._cobblestone_num_cols = cobblestone_num_cols
            self.terrain_row = 0
            self.terrain_col = 0

            self.task_id = task_id or "Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0"
            if env_cfg_cls is None:
                args_cli = SimpleNamespace(
                    num_envs=self.num_envs,
                    device=self.device,
                    terrain=self.terrain,
                    planner_backend=self.planner_backend,
                    n_frames=self.requested_n_frames,
                    plan_dt=0.02,
                )
                self.env_cfg = self._viewer._build_env_cfg(args_cli)
            else:
                self.env_cfg = env_cfg_cls()
                self.env_cfg.scene.num_envs = self.num_envs
                self.env_cfg.scene.env_spacing = 6.0
                self.env_cfg.sim.device = self.device
                self.env_cfg.sim.render_interval = self.env_cfg.decimation
                if hasattr(self.env_cfg.events, "push_robot"):
                    self.env_cfg.events.push_robot = None
                self.env_cfg.commands.base_velocity.debug_vis = False
                self.env_cfg.commands.base_velocity.ranges = self.env_cfg.commands.base_velocity.limit_ranges
                self.env_cfg.planner_backend = self.planner_backend
                self.env_cfg.reference_trajectory_horizon = self.requested_n_frames
            if semantic_small_height_m is not None:
                from extension.semantic_course import SMALL_OBSTACLE_DIAMETER

                event = getattr(self.env_cfg.events, "generate_semantic_course", None)
                if event is None:
                    raise RuntimeError("semantic_small_height_m requires semantic-course event support")
                event.params["scale_profile_overrides"] = {
                    "small": (float(SMALL_OBSTACLE_DIAMETER), float(semantic_small_height_m)),
                }
            self._configure_compact_semantic_runtime_grid()
            self._configure_compact_cobblestone_runtime_grid()
            self._configure_large_runtime_physx_buffers()
            self.planner_cfg = self._viewer._build_planner_cfg(self.env_cfg)
            self.planner_cfg.max_touchdown_xy_reach = float(planner_max_touchdown_xy_reach)
            self.together_planner_cfg = (
                self._viewer._build_together_planner_cfg(self.env_cfg) if self.planner_backend == "together" else None
            )
            self.mpc_planner_cfg = self._viewer._build_mpc_planner_cfg(self.env_cfg)
            self.plan_dt = float(getattr(self.env_cfg, "plan_dt", self.env_cfg.decimation * self.env_cfg.sim.dt))

            self.env = self._gym.make(
                self.task_id,
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

    def compact_semantic_shape_kinds(self) -> set[str]:
        from extension.semantic_course import build_course_anchors

        terrain = getattr(self.base_env.scene, "terrain", None)
        terrain_origins = getattr(terrain, "terrain_origins", None) if terrain is not None else None
        if terrain_origins is None:
            raise RuntimeError("semantic runtime fixture requires terrain origins to inspect compact shape coverage")
        anchors = build_course_anchors(terrain_origins.tolist())
        return {str(anchor.shape_kind) for anchor in anchors}

    def _semantic_course_anchors(self):
        from extension.semantic_course import build_course_anchors

        terrain = getattr(self.base_env.scene, "terrain", None)
        terrain_origins = getattr(terrain, "terrain_origins", None) if terrain is not None else None
        if terrain_origins is None:
            raise RuntimeError("semantic runtime fixture requires terrain origins to select targeted anchors")
        terrain_cfg = getattr(terrain, "cfg", None)
        terrain_generator = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg is not None else None
        if hasattr(terrain_origins, "tolist"):
            terrain_origins = terrain_origins.tolist()
        return build_course_anchors(terrain_origins, terrain_generator=terrain_generator)

    def s4_semantic_course_anchor(self, semantic_class: str):
        from extension.semantic_course import SemanticCourseStage

        if semantic_class not in {"small", "large"}:
            raise ValueError(f"semantic_class must be 'small' or 'large', got {semantic_class!r}")
        anchors = [
            anchor
            for anchor in self._semantic_course_anchors()
            if anchor.stage is SemanticCourseStage.S4 and anchor.semantic_class == semantic_class
        ]
        if not anchors:
            raise RuntimeError(f"semantic runtime fixture found no S4 {semantic_class} anchors")
        return sorted(anchors, key=lambda anchor: (anchor.row, anchor.col, anchor.slot_index))[0]

    def semantic_stage_origin_xy(self, stage: str) -> tuple[float, float]:
        from extension.semantic_course import SemanticCourseStage, representative_rows

        requested_stage = SemanticCourseStage(stage)
        terrain = getattr(self.base_env.scene, "terrain", None)
        terrain_origins = getattr(terrain, "terrain_origins", None) if terrain is not None else None
        if terrain_origins is None:
            raise RuntimeError("semantic runtime fixture requires terrain origins to select stage origins")
        row = representative_rows(len(terrain_origins))[requested_stage]
        terrain_types = getattr(terrain, "terrain_types", None) if terrain is not None else None
        col = 0
        if terrain_types is not None:
            col_value = terrain_types[0]
            col = int(col_value.item()) if hasattr(col_value, "item") else int(col_value)
        origin = terrain_origins[row, col]
        return (float(origin[0]), float(origin[1]))

    def _command_relative_xy(
        self,
        origin_xy: tuple[float, float],
        *,
        command_name: str,
        longitudinal_offset_m: float,
        lateral_offset_m: float,
    ) -> tuple[float, float]:
        command_xy = self._command_tensor(command_name)[0, :2]
        norm = torch.linalg.vector_norm(command_xy)
        if float(norm.item()) <= 1.0e-6:
            forward = torch.tensor((1.0, 0.0), device=command_xy.device, dtype=torch.float64)
        else:
            forward = command_xy / norm
        left = torch.stack((-forward[1], forward[0]))
        origin = torch.tensor(origin_xy, device=command_xy.device, dtype=torch.float64)
        xy = origin + forward * float(longitudinal_offset_m) + left * float(lateral_offset_m)
        return (float(xy[0].item()), float(xy[1].item()))

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

    def _configure_compact_cobblestone_runtime_grid(self) -> None:
        if self.terrain != "cobblestone":
            return
        scene = getattr(self.env_cfg, "scene", None)
        terrain_cfg = getattr(scene, "terrain", None) if scene is not None else None
        terrain_gen = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg is not None else None
        if terrain_gen is None:
            return
        num_rows = 2 if self._cobblestone_num_rows is None else int(self._cobblestone_num_rows)
        num_cols = 1 if self._cobblestone_num_cols is None else int(self._cobblestone_num_cols)
        terrain_gen.num_rows = num_rows
        terrain_gen.num_cols = num_cols
        terrain_gen.curriculum = False
        if hasattr(terrain_cfg, "max_init_terrain_level"):
            terrain_cfg.max_init_terrain_level = max(0, num_rows - 1)

    def _configure_large_runtime_physx_buffers(self) -> None:
        """Increase PhysX GPU pair capacities for very large headless env counts.

        At `num_envs=4096`, default capacities can trigger found/lost pair overflow
        during startup. This keeps the runtime acceptance path deterministic.
        """
        if int(self.num_envs) < 2048:
            return
        sim_cfg = getattr(self.env_cfg, "sim", None)
        physx_cfg = getattr(sim_cfg, "physx", None) if sim_cfg is not None else None
        if physx_cfg is None:
            return
        capacity_overrides = {
            "gpu_found_lost_pairs_capacity": 3_500_000,
            "gpu_found_lost_aggregate_pairs_capacity": 200_000_000,
            "gpu_total_aggregate_pairs_capacity": 3_000_000,
        }
        for field_name, required_min in capacity_overrides.items():
            if not hasattr(physx_cfg, field_name):
                continue
            current = getattr(physx_cfg, field_name)
            try:
                current_i = int(current)
            except Exception:  # noqa: BLE001 - Isaac config containers are duck-typed
                continue
            setattr(physx_cfg, field_name, max(current_i, int(required_min)))

    def close(self) -> None:
        if self._closed:
            return
        self.env.close()
        self._closed = True

    def reset(self) -> None:
        self.select_terrain_tile(terrain_row=self.terrain_row, terrain_col=self.terrain_col)
        self.env.reset()
        self.select_terrain_tile(terrain_row=self.terrain_row, terrain_col=self.terrain_col)
        for _ in range(self.warmup_steps):
            self.env.step(self.zero_actions)

    def select_terrain_tile(self, *, terrain_row: int, terrain_col: int) -> torch.Tensor:
        selected = self._viewer._apply_viewer_terrain_selection(
            self.base_env.scene,
            env_id=0,
            terrain_row=int(terrain_row),
            terrain_col=int(terrain_col),
        )
        self.terrain_row = int(terrain_row)
        self.terrain_col = int(terrain_col)
        return selected

    def _write_env0_root_xy(self, world_xy: tuple[float, float], *, z_clearance: float = 0.65) -> None:
        root_pose = torch.cat(
            [
                torch.as_tensor(self.robot.data.root_pos_w, device=self.base_env.device, dtype=torch.float32),
                torch.as_tensor(self.robot.data.root_quat_w, device=self.base_env.device, dtype=torch.float32),
            ],
            dim=-1,
        ).clone()
        root_pose[0, 0] = float(world_xy[0])
        root_pose[0, 1] = float(world_xy[1])
        root_pose[0, 2] = torch.clamp(root_pose[0, 2], min=float(z_clearance))
        env_ids = torch.tensor([0], dtype=torch.long, device=self.base_env.device)
        self.robot.write_root_pose_to_sim(root_pose[:1], env_ids=env_ids)
        if hasattr(self.robot, "write_root_velocity_to_sim"):
            zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=self.base_env.device)
            self.robot.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        self.base_env.scene.write_data_to_sim()

    def _sync_targeted_scan_pose(self) -> None:
        refresh_targeted_scanner_pose(
            self.base_env,
            self.scanner,
            minimum_steps=max(1, self.warmup_steps),
        )

    def semantic_scan_near_s4_anchor(self, semantic_class: str) -> dict[str, float | int]:
        anchor = self.s4_semantic_course_anchor(semantic_class)
        self.reset()
        self._write_env0_root_xy(anchor.world_xy)
        self._sync_targeted_scan_pose()
        scanner_xy = torch.as_tensor(self.scanner.data.pos_w[0, :2], dtype=torch.float64)
        anchor_xy = torch.tensor(anchor.world_xy, dtype=torch.float64, device=scanner_xy.device)
        torch.testing.assert_close(scanner_xy, anchor_xy, atol=0.1, rtol=0.0)
        _terrain, ray_hits = self._single_env_terrain_and_hits()
        return self._semantic_scan_diagnostics(ray_hits, stride=1)

    def plan_case_near_s4_anchor(
        self,
        semantic_class: str,
        *,
        command_name: str = "forward",
        x_offset_m: float = 0.0,
        y_offset_m: float = 0.0,
        z_clearance: float = 0.65,
    ) -> RuntimePlanDiagnostics:
        anchor = self.s4_semantic_course_anchor(semantic_class)
        self.reset()
        self._write_env0_root_xy(
            (anchor.world_xy[0] + float(x_offset_m), anchor.world_xy[1] + float(y_offset_m)),
            z_clearance=float(z_clearance),
        )
        self._sync_targeted_scan_pose()
        state = self._single_env_state()
        terrain = self._single_env_terrain()
        command = self._command_tensor(command_name)[:1]
        result = self._viewer._plan_viewer_trajectory(
            backend=self.planner_backend,
            terrain=terrain,
            state=state,
            command=command,
            requested_n_frames=self.requested_n_frames,
            dt=self.plan_dt,
            legacy_cfg=self.planner_cfg,
            together_cfg=self.together_planner_cfg,
            mpc_cfg=self.mpc_planner_cfg,
        )
        return self._build_runtime_plan_diagnostics(name=command_name, command=command, state=state, result=result)

    def plan_case_near_s4_anchor_command_relative(
        self,
        semantic_class: str,
        *,
        command_name: str = "forward",
        longitudinal_offset_m: float = 0.0,
        lateral_offset_m: float = 0.0,
        z_clearance: float = 0.65,
    ) -> RuntimePlanDiagnostics:
        anchor = self.s4_semantic_course_anchor(semantic_class)
        self.reset()
        self._write_env0_root_xy(
            self._command_relative_xy(
                anchor.world_xy,
                command_name=command_name,
                longitudinal_offset_m=float(longitudinal_offset_m),
                lateral_offset_m=float(lateral_offset_m),
            ),
            z_clearance=float(z_clearance),
        )
        self._sync_targeted_scan_pose()
        state = self._single_env_state()
        terrain = self._single_env_terrain()
        command = self._command_tensor(command_name)[:1]
        result = self._viewer._plan_viewer_trajectory(
            backend=self.planner_backend,
            terrain=terrain,
            state=state,
            command=command,
            requested_n_frames=self.requested_n_frames,
            dt=self.plan_dt,
            legacy_cfg=self.planner_cfg,
            together_cfg=self.together_planner_cfg,
            mpc_cfg=self.mpc_planner_cfg,
        )
        return self._build_runtime_plan_diagnostics(name=command_name, command=command, state=state, result=result)

    def plan_case_near_semantic_stage(
        self,
        stage: str,
        *,
        command_name: str = "forward",
        longitudinal_offset_m: float = 0.0,
        lateral_offset_m: float = 0.0,
        z_clearance: float = 0.65,
    ) -> RuntimePlanDiagnostics:
        self.reset()
        self._write_env0_root_xy(
            self._command_relative_xy(
                self.semantic_stage_origin_xy(stage),
                command_name=command_name,
                longitudinal_offset_m=float(longitudinal_offset_m),
                lateral_offset_m=float(lateral_offset_m),
            ),
            z_clearance=float(z_clearance),
        )
        self._sync_targeted_scan_pose()
        state = self._single_env_state()
        terrain = self._single_env_terrain()
        command = self._command_tensor(command_name)[:1]
        result = self._viewer._plan_viewer_trajectory(
            backend=self.planner_backend,
            terrain=terrain,
            state=state,
            command=command,
            requested_n_frames=self.requested_n_frames,
            dt=self.plan_dt,
            legacy_cfg=self.planner_cfg,
            together_cfg=self.together_planner_cfg,
            mpc_cfg=self.mpc_planner_cfg,
        )
        return self._build_runtime_plan_diagnostics(name=command_name, command=command, state=state, result=result)

    def grounded_crossing_runtime_sequence(
        self,
        *,
        semantic_class: str = "small",
        command_name: str = "forward",
        x_offsets_m: tuple[float, ...] = (-0.18, 0.04, 0.28),
        y_offset_m: float = 0.0,
        z_clearances: tuple[float, ...] | None = None,
    ) -> GroundedCrossingRuntimeReport:
        if not x_offsets_m:
            raise ValueError("x_offsets_m must be non-empty")
        if z_clearances is None:
            z_clearances = tuple(0.65 for _ in x_offsets_m)
        if len(z_clearances) != len(x_offsets_m):
            raise ValueError("z_clearances must match x_offsets_m length")

        sampled = tuple(
            self.plan_case_near_s4_anchor_command_relative(
                semantic_class,
                command_name=command_name,
                longitudinal_offset_m=float(x_offset_m),
                lateral_offset_m=float(y_offset_m),
                z_clearance=float(z_clearance),
            )
            for x_offset_m, z_clearance in zip(x_offsets_m, z_clearances)
        )
        grounded = [plan.grounded_crossing for plan in sampled]
        if any(diag is None for diag in grounded):
            raise RuntimeError("grounded crossing runtime sequence requires grounded_crossing diagnostics on all sampled plans")
        grounded = [diag for diag in grounded if diag is not None]
        mode_sequence = tuple(_grounded_crossing_state_value(diag.mode) for diag in grounded)
        state_sequence = tuple(_grounded_crossing_state_value(diag.state_mode) for diag in grounded)
        small_strategy_sequence = tuple(_grounded_crossing_state_value(diag.small_strategy_outcome) for diag in grounded)
        status_sequence = tuple(_grounded_crossing_state_value(diag.status) for diag in grounded)
        feasible_sequence = tuple(_grounded_crossing_bool(diag.feasible) for diag in grounded)
        safe_fallback_sequence = tuple(_grounded_crossing_bool(diag.safe_fallback) for diag in grounded)
        selected_beta_sequence = tuple(_grounded_crossing_float(diag.selected_beta) for diag in grounded)
        selected_route_sequence = tuple(_grounded_crossing_state_value(diag.selected_route) for diag in grounded)
        direction_id_sequence = tuple(_grounded_crossing_state_value(diag.direction_id) for diag in grounded)
        front_gap_abs = max(_grounded_crossing_gap_abs_max(diag.front_touchdown_ground_gap) for diag in grounded)
        rear_gap_abs = max(_grounded_crossing_gap_abs_max(diag.rear_touchdown_ground_gap) for diag in grounded)
        touchdown_ground_gap_by_leg_abs = max(
            _grounded_crossing_gap_abs_max(diag.touchdown_ground_gap_by_leg) for diag in grounded
        )
        rear_touchdown_airborne_count = sum(
            int(_grounded_crossing_gap_abs_max(diag.rear_touchdown_ground_gap) > 0.02) for diag in grounded
        )
        touchdown_on_small_count = max(_grounded_crossing_count(diag.touchdown_on_small_count) for diag in grounded)
        foot_small_collision_count = max(
            int(torch.as_tensor(diag.per_leg_foot_small_collision_count).reshape(-1).sum().item()) for diag in grounded
        )
        front_foot_small_collision_count = max(
            _grounded_crossing_count(diag.front_foot_small_collision_count) for diag in grounded
        )
        rear_foot_small_collision_count = max(
            _grounded_crossing_count(diag.rear_foot_small_collision_count) for diag in grounded
        )
        base_small_penetration_count = max(_grounded_crossing_count(diag.base_small_penetration_count) for diag in grounded)
        base_path_crosses_small_flag = int(
            any(bool(torch.as_tensor(diag.base_path_crosses_small_flag).reshape(-1)[0].item()) for diag in grounded)
        )
        body_min_clearance = min(_grounded_crossing_float(diag.body_min_clearance) for diag in grounded)
        leg_min_clearance = min(_grounded_crossing_float(diag.leg_min_clearance) for diag in grounded)
        base_min_clearance_to_small = min(_grounded_crossing_float(diag.base_min_clearance_to_small) for diag in grounded)
        per_leg_touchdown_on_small_count = _grounded_crossing_max_per_leg_int(
            grounded,
            "per_leg_touchdown_on_small_count",
        )
        per_leg_foot_small_collision_count = _grounded_crossing_max_per_leg_int(
            grounded,
            "per_leg_foot_small_collision_count",
        )
        per_leg_min_clearance_to_small = _grounded_crossing_min_per_leg_float(
            grounded,
            "per_leg_min_clearance_to_small",
        )
        per_leg_touchdown_beyond_small_back_edge = _grounded_crossing_any_per_leg_bool(
            grounded,
            "per_leg_touchdown_beyond_small_back_edge",
        )
        touchdown_semantic_by_leg = tuple(
            int(value)
            for value in torch.stack(
                [_grounded_crossing_first_row(diag.touchdown_semantic_by_leg).to(dtype=torch.int64) for diag in grounded],
                dim=0,
            )
            .max(dim=0)
            .values.tolist()
        )
        command_direction_violation_count = sum(int(_grounded_crossing_bool(diag.command_direction_violation)) for diag in grounded)
        cross_small_success_count = sum(int(_grounded_crossing_bool(diag.cross_small_success)) for diag in grounded)
        cross_phase_progression_valid = int(T116_MODE_CROSS_SMALL in mode_sequence)
        cross_outcome_grounded = int(
            cross_small_success_count > 0
            and touchdown_on_small_count == 0
            and foot_small_collision_count == 0
            and base_small_penetration_count == 0
            and command_direction_violation_count == 0
        )
        return GroundedCrossingRuntimeReport(
            mode_sequence=mode_sequence,
            state_sequence=state_sequence,
            small_strategy_sequence=small_strategy_sequence,
            status_sequence=status_sequence,
            feasible_sequence=feasible_sequence,
            safe_fallback_sequence=safe_fallback_sequence,
            selected_beta_sequence=selected_beta_sequence,
            selected_route_sequence=selected_route_sequence,
            direction_id_sequence=direction_id_sequence,
            front_touchdown_ground_gap_abs_m=front_gap_abs,
            rear_touchdown_ground_gap_abs_m=rear_gap_abs,
            touchdown_ground_gap_by_leg_abs_m=touchdown_ground_gap_by_leg_abs,
            rear_touchdown_airborne_count=rear_touchdown_airborne_count,
            touchdown_on_small_count=touchdown_on_small_count,
            foot_small_collision_count=foot_small_collision_count,
            front_foot_small_collision_count=front_foot_small_collision_count,
            rear_foot_small_collision_count=rear_foot_small_collision_count,
            base_small_penetration_count=base_small_penetration_count,
            base_path_crosses_small_flag=base_path_crosses_small_flag,
            body_min_clearance_m=body_min_clearance,
            leg_min_clearance_m=leg_min_clearance,
            base_min_clearance_to_small_m=base_min_clearance_to_small,
            per_leg_touchdown_on_small_count=per_leg_touchdown_on_small_count,
            per_leg_foot_small_collision_count=per_leg_foot_small_collision_count,
            per_leg_min_clearance_to_small_m=per_leg_min_clearance_to_small,
            per_leg_touchdown_beyond_small_back_edge=per_leg_touchdown_beyond_small_back_edge,
            touchdown_semantic_by_leg=touchdown_semantic_by_leg,
            command_direction_violation_count=command_direction_violation_count,
            cross_small_success_count=cross_small_success_count,
            cross_phase_progression_valid=cross_phase_progression_valid,
            cross_outcome_grounded=cross_outcome_grounded,
            sampled_plans=sampled,
        )

    def grounded_crossing_runtime_sequences_by_command(
        self,
        *,
        command_names: tuple[str, ...] = ("forward", "backward", "lateral_left", "lateral_right"),
        semantic_class: str = "small",
        x_offsets_m: tuple[float, ...] = (-0.18, 0.04, 0.28),
        y_offset_m: float = 0.0,
        z_clearances: tuple[float, ...] | None = None,
    ) -> dict[str, GroundedCrossingRuntimeReport]:
        return {
            command_name: self.grounded_crossing_runtime_sequence(
                semantic_class=semantic_class,
                command_name=command_name,
                x_offsets_m=x_offsets_m,
                y_offset_m=y_offset_m,
                z_clearances=z_clearances,
            )
            for command_name in command_names
        }

    def _command_tensor(self, name: str) -> torch.Tensor:
        command = self.command_cases[name]
        if command.shape[0] != self.num_envs:
            raise RuntimeError(f"command case batch mismatch: expected {self.num_envs}, got {command.shape[0]}")
        return command.to(device=self.base_env.device, dtype=torch.float64)

    def _single_env_state(self):
        if self.planner_backend == "together":
            return self._viewer._together_state_from_env(self.base_env, self.foot_ids.tolist())
        if self.planner_backend == "mpc":
            return self._viewer._mpc_state_from_env(self.base_env, self.foot_ids.tolist())
        return self._viewer._planner_state_from_env(self.base_env, self.foot_ids.tolist())

    def _single_env_terrain_and_hits(self):
        if self.planner_backend == "together":
            return self._viewer._compute_together_local_terrain(self.scanner, env_id=0)
        if self.planner_backend == "mpc":
            return self._viewer._compute_mpc_local_terrain(self.scanner, env_id=0)
        return self._viewer._compute_local_terrain(self.scanner, env_id=0)

    def _single_env_terrain(self):
        terrain, _ = self._single_env_terrain_and_hits()
        return terrain

    def _semantic_scan_diagnostics(self, ray_hits: torch.Tensor, *, stride: int | None = None) -> dict[str, float | int]:
        semantic_map = self._viewer._scanner_semantic_map(self.scanner, env_id=0)
        _, diagnostics = self._viewer._subsample_semantic_height_points(
            ray_hits,
            semantic_map,
            self.heightmap_viz_stride if stride is None else int(stride),
        )
        return diagnostics

    def _build_runtime_plan_diagnostics(self, *, name: str, command: torch.Tensor, state, result) -> RuntimePlanDiagnostics:
        summary = self._viewer._trajectory_motion_summary(result)
        grounded_crossing = _grounded_crossing_diagnostics_from_result(result)
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
            grounded_crossing=grounded_crossing,
            grounded_crossing_summary=_grounded_crossing_summary(grounded_crossing),
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
            mpc_cfg=self.mpc_planner_cfg,
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
        joint_pos = self._viewer._joint_pos_robot_to_planner(
            self.robot,
            torch.as_tensor(self.robot.data.joint_pos[:batch], dtype=torch.float64).clone(),
        )
        return PlaybackReadback(
            root_pos_w=torch.as_tensor(self.robot.data.root_pos_w[:batch], dtype=torch.float64).clone(),
            joint_pos=joint_pos,
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
    "GroundedCrossingDiagnostics",
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
