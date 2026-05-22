from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT, GO2PVCNN_ROOT / "tests"):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture, refresh_targeted_scanner_pose  # noqa: E402
from extension.batch_mpc_planner.nominal import build_nominal_trajectory  # noqa: E402
from extension.batch_mpc_planner.optimizer import optimize_variables  # noqa: E402
from extension.batch_mpc_planner.variables import decode_trajectory, init_optimization_variables  # noqa: E402


COMMAND_ALIASES: dict[str, tuple[float, float, float]] = {
    "standstill": (0.0, 0.0, 0.0),
    "forward": (0.3, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "lateral_left": (0.0, 0.25, 0.0),
    "lateral_right": (0.0, -0.25, 0.0),
    "yaw_left": (0.0, 0.0, 0.3),
    "yaw_right": (0.0, 0.0, -0.3),
    "forward_yaw_left": (0.25, 0.0, 0.25),
    "forward_yaw_right": (0.25, 0.0, -0.25),
}

LEG_NAMES = ("FL", "FR", "RL", "RR")
COBBLESTONE_SUBTERRAINS = (
    "flat",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "boxes",
    "pyramid_stairs",
    "pyramid_stairs_inv",
)


def _parse_command(value: str) -> tuple[str, tuple[float, float, float]]:
    raw = value.strip()
    if raw in COMMAND_ALIASES:
        return raw, COMMAND_ALIASES[raw]
    if ":" in raw:
        name, values = raw.split(":", 1)
        parts = [float(item) for item in values.replace(",", " ").split()]
        if len(parts) != 3:
            raise ValueError(f"Command must have three floats after ':', got {value!r}")
        return name.strip(), (parts[0], parts[1], parts[2])
    parts = [float(item) for item in raw.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Command must be a known name or three floats, got {value!r}")
    name = f"cmd_{parts[0]:+.2f}_{parts[1]:+.2f}_{parts[2]:+.2f}"
    return name, (parts[0], parts[1], parts[2])


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def _build_speed_grid_commands(*, vx_values: tuple[float, ...], vy_values: tuple[float, ...], yaw: float) -> tuple[str, ...]:
    commands: list[str] = []
    for vx in vx_values:
        for vy in vy_values:
            commands.append(f"grid_vx{vx:.2f}_vy{vy:.2f}_yaw{yaw:.2f}:{vx:.6f} {vy:.6f} {yaw:.6f}")
    return tuple(commands)


def _iter_true_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    values = torch.as_tensor(mask, dtype=torch.bool).reshape(-1).tolist()
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1.0e-9:
        return float("nan")
    return num / den


def _quadratic_r2(z: torch.Tensor) -> float:
    z = torch.as_tensor(z, dtype=torch.float64).reshape(-1)
    n = int(z.numel())
    if n < 4:
        return float("nan")
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64, device=z.device)
    design = torch.stack((torch.ones_like(t), t, t * t), dim=-1)
    try:
        coeff = torch.linalg.lstsq(design, z).solution
    except RuntimeError:
        return float("nan")
    pred = design @ coeff
    ss_res = torch.square(z - pred).sum()
    ss_tot = torch.square(z - z.mean()).sum()
    if float(ss_tot.item()) <= 1.0e-12:
        return float("nan")
    return float((1.0 - ss_res / ss_tot).item())


def _unimodal_z_violation_ratio(z: torch.Tensor) -> tuple[float, int]:
    z = torch.as_tensor(z, dtype=torch.float64).reshape(-1)
    n = int(z.numel())
    if n < 4:
        return (float("nan"), -1)
    peak_idx = int(torch.argmax(z).item())
    dz = z[1:] - z[:-1]
    tol = 2.0e-3
    rising_bad = dz[:peak_idx] < -tol
    falling_bad = dz[peak_idx:] > tol
    denom = max(int(dz.numel()), 1)
    bad = int(torch.count_nonzero(rising_bad).item() + torch.count_nonzero(falling_bad).item())
    return (float(bad) / float(denom), peak_idx)


def _swing_run_metrics(
    *,
    command_name: str,
    cycle: int,
    leg_idx: int,
    start: int,
    end: int,
    foot: torch.Tensor,
    contact: torch.Tensor,
) -> dict[str, float | int | str]:
    segment = foot[start : end + 1, leg_idx].to(dtype=torch.float64)
    z = segment[:, 2]
    steps = torch.linalg.vector_norm(segment[1:] - segment[:-1], dim=-1) if int(segment.shape[0]) > 1 else torch.empty(0)
    xy_steps = (
        torch.linalg.vector_norm(segment[1:, :2] - segment[:-1, :2], dim=-1)
        if int(segment.shape[0]) > 1
        else torch.empty(0)
    )
    prev_boundary = float("nan")
    next_boundary = float("nan")
    if start > 0:
        prev_boundary = float(torch.linalg.vector_norm(foot[start, leg_idx] - foot[start - 1, leg_idx]).item())
    if end + 1 < int(foot.shape[0]):
        next_boundary = float(torch.linalg.vector_norm(foot[end + 1, leg_idx] - foot[end, leg_idx]).item())
    max_step = float(steps.max().item()) if int(steps.numel()) else 0.0
    min_step = float(steps.min().item()) if int(steps.numel()) else 0.0
    mean_step = float(steps.mean().item()) if int(steps.numel()) else 0.0
    median_step = float(steps.median().item()) if int(steps.numel()) else 0.0
    early_count = max(1, int(math.ceil(float(steps.numel()) * 0.25))) if int(steps.numel()) else 0
    late_count = early_count
    early_step_mean = float(steps[:early_count].mean().item()) if early_count else float("nan")
    late_step_mean = float(steps[-late_count:].mean().item()) if late_count else float("nan")
    z_unimodal_violation_ratio, peak_idx = _unimodal_z_violation_ratio(z)
    peak_phase = float(peak_idx) / float(max(int(z.numel()) - 1, 1)) if peak_idx >= 0 else float("nan")
    contact_values = contact[start : end + 1, leg_idx]
    return {
        "type": "swing_run",
        "command": command_name,
        "cycle": int(cycle),
        "leg": LEG_NAMES[leg_idx],
        "leg_idx": int(leg_idx),
        "start": int(start),
        "end": int(end),
        "frames": int(end - start + 1),
        "contact_true_inside": int(torch.count_nonzero(contact_values).item()),
        "path_len": float(steps.sum().item()) if int(steps.numel()) else 0.0,
        "xy_path_len": float(xy_steps.sum().item()) if int(xy_steps.numel()) else 0.0,
        "chord_len": float(torch.linalg.vector_norm(segment[-1] - segment[0]).item()),
        "max_step": max_step,
        "mean_step": mean_step,
        "median_step": median_step,
        "min_step": min_step,
        "max_to_median_step": _safe_ratio(max_step, median_step),
        "max_to_mean_step": _safe_ratio(max_step, mean_step),
        "early_to_late_step_mean": _safe_ratio(early_step_mean, late_step_mean),
        "prev_boundary_step": prev_boundary,
        "next_boundary_step": next_boundary,
        "boundary_to_median_step": _safe_ratio(max(prev_boundary, next_boundary), median_step),
        "z_start": float(z[0].item()),
        "z_peak": float(z.max().item()),
        "z_end": float(z[-1].item()),
        "z_lift": float((z.max() - torch.maximum(z[0], z[-1])).item()),
        "z_peak_phase": peak_phase,
        "z_unimodal_violation_ratio": z_unimodal_violation_ratio,
        "z_quadratic_r2": _quadratic_r2(z),
    }


def _trajectory_summary(
    *,
    command_name: str,
    cycle: int,
    result,
    layer: str = "result",
    variant: str = "baseline",
    terrain_case: str = "",
) -> list[dict[str, float | int | str]]:
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)[0]
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool)[0]
    rows: list[dict[str, float | int | str]] = []
    for leg_idx in range(4):
        swing_mask = torch.logical_not(contact[:, leg_idx])
        for start, end in _iter_true_runs(swing_mask):
            if end - start + 1 < 3:
                continue
            rows.append(
                _swing_run_metrics(
                    command_name=command_name,
                    cycle=cycle,
                    leg_idx=leg_idx,
                    start=start,
                    end=end,
                    foot=foot,
                    contact=contact,
                )
            )
    if rows:
        worst_jump = max(float(row["max_to_median_step"]) for row in rows if math.isfinite(float(row["max_to_median_step"])))
        worst_boundary = max(
            float(row["boundary_to_median_step"])
            for row in rows
            if math.isfinite(float(row["boundary_to_median_step"]))
        )
        worst_unimodal = max(
            float(row["z_unimodal_violation_ratio"])
            for row in rows
            if math.isfinite(float(row["z_unimodal_violation_ratio"]))
        )
        min_r2 = min(float(row["z_quadratic_r2"]) for row in rows if math.isfinite(float(row["z_quadratic_r2"])))
    else:
        worst_jump = float("nan")
        worst_boundary = float("nan")
        worst_unimodal = float("nan")
        min_r2 = float("nan")
    root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
    rows.insert(
        0,
        {
            "type": "cycle_summary",
            "layer": layer,
            "variant": variant,
            "terrain_case": terrain_case,
            "command": command_name,
            "cycle": int(cycle),
            "swing_run_count": len(rows),
            "worst_max_to_median_step": worst_jump,
            "worst_boundary_to_median_step": worst_boundary,
            "worst_z_unimodal_violation_ratio": worst_unimodal,
            "min_z_quadratic_r2": min_r2,
            "root_dx": float((root[0, -1, 0] - root[0, 0, 0]).item()),
            "root_dy": float((root[0, -1, 1] - root[0, 0, 1]).item()),
        },
    )
    for row in rows[1:]:
        row["layer"] = layer
        row["variant"] = variant
        row["terrain_case"] = terrain_case
    return rows


def _result_like(*, root_pos: torch.Tensor, foot_pos: torch.Tensor, contact_state: torch.Tensor):
    return SimpleNamespace(
        root_pos_w=torch.as_tensor(root_pos).detach(),
        foot_pos_w=torch.as_tensor(foot_pos).detach(),
        contact_state=torch.as_tensor(contact_state).detach(),
    )


def _trace_decode_layers(
    *,
    command_name: str,
    cycle: int,
    state,
    command: torch.Tensor,
    terrain,
    cfg,
    variant: str,
    terrain_case: str,
) -> list[dict[str, float | int | str]]:
    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)
    variables = init_optimization_variables(nominal, cfg.runtime)
    initial_unlocked = decode_trajectory(nominal, variables, cfg.runtime, terrain=None)
    initial_locked = decode_trajectory(nominal, variables, cfg.runtime, terrain=terrain)
    optimized_locked, _cost_total, _loss_breakdown, _finite_ok = optimize_variables(
        nominal,
        variables,
        state,
        command,
        terrain,
        cfg,
    )
    optimized_unlocked = decode_trajectory(nominal, variables, cfg.runtime, terrain=None)
    layers = (
        (
            "nominal",
            _result_like(
                root_pos=nominal["root_pos"],
                foot_pos=nominal["foot_pos"],
                contact_state=nominal["contact_prior"] >= float(cfg.runtime.contact_threshold),
            ),
        ),
        (
            "initial_decode_unlocked",
            _result_like(
                root_pos=initial_unlocked.root_pos,
                foot_pos=initial_unlocked.foot_pos,
                contact_state=initial_unlocked.contact_prob >= float(cfg.runtime.contact_threshold),
            ),
        ),
        (
            "initial_decode_locked",
            _result_like(
                root_pos=initial_locked.root_pos,
                foot_pos=initial_locked.foot_pos,
                contact_state=initial_locked.contact_prob >= float(cfg.runtime.contact_threshold),
            ),
        ),
        (
            "optimized_decode_unlocked",
            _result_like(
                root_pos=optimized_unlocked.root_pos,
                foot_pos=optimized_unlocked.foot_pos,
                contact_state=optimized_unlocked.contact_prob >= float(cfg.runtime.contact_threshold),
            ),
        ),
        (
            "optimized_decode_locked",
            _result_like(
                root_pos=optimized_locked.root_pos,
                foot_pos=optimized_locked.foot_pos,
                contact_state=optimized_locked.contact_prob >= float(cfg.runtime.contact_threshold),
            ),
        ),
    )
    rows: list[dict[str, float | int | str]] = []
    for layer_name, layer_result in layers:
        rows.extend(
            _trajectory_summary(
                command_name=command_name,
                cycle=cycle,
                result=layer_result,
                layer=layer_name,
                variant=variant,
                terrain_case=terrain_case,
            )
        )
    return rows


def _variant_cfg(base_cfg, name: str):
    cfg = copy.deepcopy(base_cfg)
    variant = str(name)
    if variant == "baseline":
        return cfg
    if variant == "smooth8":
        cfg.losses.smoothness.weight *= 8.0
        cfg.losses.smoothness.foot_weight *= 8.0
        return cfg
    if variant == "smooth24":
        cfg.losses.smoothness.weight *= 24.0
        cfg.losses.smoothness.foot_weight *= 24.0
        return cfg
    if variant == "lr_half":
        cfg.runtime.lr *= 0.5
        return cfg
    if variant == "lr_quarter":
        cfg.runtime.lr *= 0.25
        return cfg
    if variant == "steps12":
        cfg.runtime.optimize_steps = min(int(cfg.runtime.optimize_steps), 12)
        return cfg
    if variant == "smooth8_lr_half":
        cfg.losses.smoothness.weight *= 8.0
        cfg.losses.smoothness.foot_weight *= 8.0
        cfg.runtime.lr *= 0.5
        return cfg
    if variant == "smooth24_lr_half":
        cfg.losses.smoothness.weight *= 24.0
        cfg.losses.smoothness.foot_weight *= 24.0
        cfg.runtime.lr *= 0.5
        return cfg
    if variant == "smooth8_steps12":
        cfg.losses.smoothness.weight *= 8.0
        cfg.losses.smoothness.foot_weight *= 8.0
        cfg.runtime.optimize_steps = min(int(cfg.runtime.optimize_steps), 12)
        return cfg
    raise ValueError(f"Unknown variant {name!r}")


def _summary_score(row: dict[str, float | int | str]) -> float:
    jump = float(row["worst_max_to_median_step"])
    boundary = float(row["worst_boundary_to_median_step"])
    violation = float(row["worst_z_unimodal_violation_ratio"])
    r2 = float(row["min_z_quadratic_r2"])
    if not math.isfinite(jump):
        jump = 100.0
    if not math.isfinite(boundary):
        boundary = 100.0
    if not math.isfinite(violation):
        violation = 1.0
    if not math.isfinite(r2):
        r2 = 0.0
    return jump + boundary + 10.0 * violation + 10.0 * max(0.0, 1.0 - r2)


def _aggregate_variant_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    out: list[dict[str, float | int | str]] = []
    for variant, values in grouped.items():
        jumps = [float(row["worst_max_to_median_step"]) for row in values if math.isfinite(float(row["worst_max_to_median_step"]))]
        boundaries = [
            float(row["worst_boundary_to_median_step"])
            for row in values
            if math.isfinite(float(row["worst_boundary_to_median_step"]))
        ]
        violations = [
            float(row["worst_z_unimodal_violation_ratio"])
            for row in values
            if math.isfinite(float(row["worst_z_unimodal_violation_ratio"]))
        ]
        r2s = [float(row["min_z_quadratic_r2"]) for row in values if math.isfinite(float(row["min_z_quadratic_r2"]))]
        scores = [_summary_score(row) for row in values]
        out.append(
            {
                "type": "variant_summary",
                "variant": variant,
                "cycle_count": len(values),
                "score_mean": sum(scores) / max(len(scores), 1),
                "score_max": max(scores) if scores else float("nan"),
                "max_worst_max_to_median_step": max(jumps) if jumps else float("nan"),
                "max_worst_boundary_to_median_step": max(boundaries) if boundaries else float("nan"),
                "max_worst_z_unimodal_violation_ratio": max(violations) if violations else float("nan"),
                "min_z_quadratic_r2": min(r2s) if r2s else float("nan"),
            }
        )
    out.sort(key=lambda row: (float(row["score_mean"]), float(row["score_max"]), str(row["variant"])))
    return out


def _aggregate_terrain_case_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("terrain_case", "")), []).append(row)
    out: list[dict[str, float | int | str]] = []
    for terrain_case, values in grouped.items():
        scores = [_summary_score(row) for row in values]
        jumps = [float(row["worst_max_to_median_step"]) for row in values if math.isfinite(float(row["worst_max_to_median_step"]))]
        boundaries = [
            float(row["worst_boundary_to_median_step"])
            for row in values
            if math.isfinite(float(row["worst_boundary_to_median_step"]))
        ]
        violations = [
            float(row["worst_z_unimodal_violation_ratio"])
            for row in values
            if math.isfinite(float(row["worst_z_unimodal_violation_ratio"]))
        ]
        r2s = [float(row["min_z_quadratic_r2"]) for row in values if math.isfinite(float(row["min_z_quadratic_r2"]))]
        out.append(
            {
                "type": "terrain_case_summary",
                "terrain_case": terrain_case,
                "cycle_count": len(values),
                "score_mean": sum(scores) / max(len(scores), 1),
                "score_max": max(scores) if scores else float("nan"),
                "max_worst_max_to_median_step": max(jumps) if jumps else float("nan"),
                "max_worst_boundary_to_median_step": max(boundaries) if boundaries else float("nan"),
                "max_worst_z_unimodal_violation_ratio": max(violations) if violations else float("nan"),
                "min_z_quadratic_r2": min(r2s) if r2s else float("nan"),
            }
        )
    out.sort(key=lambda row: (float(row["score_mean"]), float(row["score_max"]), str(row["terrain_case"])))
    return out


def run_probe(
    *,
    device: str,
    terrain: str,
    cycles: int,
    playback_frame: int,
    commands: tuple[str, ...],
    requested_n_frames: int,
    warmup_steps: int,
    trace_decode_layers: bool,
    variants: tuple[str, ...],
    terrain_cases: tuple[str, ...],
) -> int:
    all_summaries: list[dict[str, float | int | str]] = []
    if terrain_cases == ("default",):
        terrain_cases = (terrain,)
    for terrain_case in terrain_cases:
        runtime_kwargs = {
            "num_envs": 1,
            "device": device,
            "terrain": terrain,
            "warmup_steps": warmup_steps,
            "requested_n_frames": requested_n_frames,
            "planner_backend": "mpc",
        }
        if terrain == "cobblestone":
            runtime_kwargs.update(
                {
                    "task_id": "Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0",
                    "cobblestone_num_rows": 1,
                    "cobblestone_num_cols": 1,
                }
            )
            if terrain_case not in {"cobblestone", "all", "all_cobblestone"}:
                runtime_kwargs["cobblestone_subterrain"] = terrain_case
        else:
            runtime_kwargs.update(
                {
                    "task_id": "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
                    "env_cfg_entry_point": (
                        "go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg:"
                        "TeacherElevationTrajectoryMpcSemanticEnvCfg"
                    ),
                }
            )
        runtime = RealViewerRuntimeFixture(**runtime_kwargs)
        try:
            case_summaries: list[dict[str, float | int | str]] = []
            print(
                json.dumps(
                    {
                        "type": "terrain_case_header",
                        "terrain": terrain,
                        "terrain_case": terrain_case,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if terrain == "cobblestone":
                runtime.select_terrain_tile(terrain_row=0, terrain_col=0)
            _run_probe_case(
                runtime=runtime,
                device=device,
                terrain=terrain,
                terrain_case=terrain_case,
                cycles=cycles,
                playback_frame=playback_frame,
                commands=commands,
                requested_n_frames=requested_n_frames,
                warmup_steps=warmup_steps,
                trace_decode_layers=trace_decode_layers,
                variants=variants,
                all_summaries=case_summaries,
            )
            all_summaries.extend(case_summaries)
            for row in _aggregate_terrain_case_rows(case_summaries):
                print(json.dumps(row, sort_keys=True), flush=True)
        finally:
            runtime.close()
    variant_summaries = _aggregate_variant_rows(all_summaries)
    for row in variant_summaries:
        print(json.dumps(row, sort_keys=True), flush=True)
    for row in _aggregate_terrain_case_rows(all_summaries):
        print(json.dumps({**row, "type": "terrain_case_overall_summary"}, sort_keys=True), flush=True)
    finite_jump = [
        float(row["worst_max_to_median_step"])
        for row in all_summaries
        if math.isfinite(float(row["worst_max_to_median_step"]))
    ]
    finite_boundary = [
        float(row["worst_boundary_to_median_step"])
        for row in all_summaries
        if math.isfinite(float(row["worst_boundary_to_median_step"]))
    ]
    finite_unimodal = [
        float(row["worst_z_unimodal_violation_ratio"])
        for row in all_summaries
        if math.isfinite(float(row["worst_z_unimodal_violation_ratio"]))
    ]
    finite_r2 = [
        float(row["min_z_quadratic_r2"])
        for row in all_summaries
        if math.isfinite(float(row["min_z_quadratic_r2"]))
    ]
    footer = {
        "type": "probe_footer",
        "cycle_count": len(all_summaries),
        "best_variant": str(variant_summaries[0]["variant"]) if variant_summaries else "",
        "best_variant_score_mean": float(variant_summaries[0]["score_mean"]) if variant_summaries else float("nan"),
        "max_worst_max_to_median_step": max(finite_jump) if finite_jump else float("nan"),
        "max_worst_boundary_to_median_step": max(finite_boundary) if finite_boundary else float("nan"),
        "max_worst_z_unimodal_violation_ratio": max(finite_unimodal) if finite_unimodal else float("nan"),
        "min_z_quadratic_r2": min(finite_r2) if finite_r2 else float("nan"),
    }
    print(json.dumps(footer, sort_keys=True), flush=True)
    return 0


def _run_probe_case(
    *,
    runtime,
    device: str,
    terrain: str,
    terrain_case: str,
    cycles: int,
    playback_frame: int,
    commands: tuple[str, ...],
    requested_n_frames: int,
    warmup_steps: int,
    trace_decode_layers: bool,
    variants: tuple[str, ...],
    all_summaries: list[dict[str, float | int | str]],
) -> None:
    print(
        json.dumps(
            {
                "type": "probe_header",
                "device": device,
                "terrain": terrain,
                "terrain_case": terrain_case,
                "cycles": int(cycles),
                "playback_frame": int(playback_frame),
                "requested_n_frames": int(requested_n_frames),
                "trace_decode_layers": bool(trace_decode_layers),
                "variants": list(variants),
                "warmup_steps": int(warmup_steps),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for command_text in commands:
        command_name, command_tuple = _parse_command(command_text)
        for variant in variants:
            runtime.reset()
            command = torch.tensor([command_tuple], dtype=torch.float64, device=runtime.base_env.device)
            state = runtime._single_env_state()
            mpc_cfg = _variant_cfg(runtime.mpc_planner_cfg, variant)
            for cycle in range(int(cycles)):
                terrain_obj = runtime._single_env_terrain()
                result = runtime._viewer._plan_viewer_trajectory(
                    backend=runtime.planner_backend,
                    terrain=terrain_obj,
                    state=state,
                    command=command,
                    requested_n_frames=runtime.requested_n_frames,
                    dt=runtime.plan_dt,
                    legacy_cfg=runtime.planner_cfg,
                    together_cfg=runtime.together_planner_cfg,
                    mpc_cfg=mpc_cfg,
                )
                rows = _trajectory_summary(
                    command_name=command_name,
                    cycle=cycle,
                    result=result,
                    variant=variant,
                    terrain_case=terrain_case,
                )
                all_summaries.append(rows[0])
                for row in rows:
                    print(json.dumps(row, sort_keys=True), flush=True)
                if trace_decode_layers:
                    trace_rows = _trace_decode_layers(
                        command_name=command_name,
                        cycle=cycle,
                        state=state,
                        command=command,
                        terrain=terrain_obj,
                        cfg=mpc_cfg,
                        variant=variant,
                        terrain_case=terrain_case,
                    )
                    for row in trace_rows:
                        print(json.dumps(row, sort_keys=True), flush=True)
                frame_idx = min(int(playback_frame), int(result.num_frames) - 1)
                runtime._viewer._viewer_direct_playback_step(runtime.base_env, result, frame_idx=frame_idx)
                refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
                state = runtime._single_env_state()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--terrain", default="task", choices=("flat", "task", "cobblestone"))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--playback-frame", type=int, default=49)
    parser.add_argument("--requested-n-frames", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument(
        "--trace-decode-layers",
        action="store_true",
        help="Also print nominal/pre-lock/post-lock MPC decode quality metrics for root-cause tracing.",
    )
    parser.add_argument(
        "--variants",
        default="baseline",
        help=(
            "Comma-separated test-only MPC cfg variants. Known: baseline, smooth8, smooth24, "
            "lr_half, lr_quarter, steps12, smooth8_lr_half, smooth24_lr_half, smooth8_steps12."
        ),
    )
    parser.add_argument(
        "--commands",
        default="forward,yaw_left,forward_yaw_left",
        help="Comma-separated command aliases or semicolon-separated vx vy yaw triples.",
    )
    parser.add_argument(
        "--speed-grid",
        action="store_true",
        help="Ignore --commands and generate vx/vy/yaw command grid.",
    )
    parser.add_argument("--vx-values", default="0.0,0.5,1.0")
    parser.add_argument("--vy-values", default="0.0,0.25,0.5")
    parser.add_argument("--yaw-value", type=float, default=1.0)
    parser.add_argument(
        "--terrain-cases",
        default="default",
        help=(
            "Comma-separated terrain cases. For cobblestone use all_cobblestone or one/more of: "
            + ",".join(COBBLESTONE_SUBTERRAINS)
        ),
    )
    args = parser.parse_args()
    if bool(args.speed_grid):
        commands = _build_speed_grid_commands(
            vx_values=_parse_float_list(str(args.vx_values)),
            vy_values=_parse_float_list(str(args.vy_values)),
            yaw=float(args.yaw_value),
        )
    else:
        command_text = str(args.commands).replace(";", ",")
        commands = tuple(item.strip() for item in command_text.split(",") if item.strip())
    variants = tuple(item.strip() for item in str(args.variants).split(",") if item.strip())
    terrain_cases = tuple(item.strip() for item in str(args.terrain_cases).split(",") if item.strip())
    if terrain_cases == ("all_cobblestone",):
        terrain_cases = COBBLESTONE_SUBTERRAINS
    return run_probe(
        device=str(args.device),
        terrain=str(args.terrain),
        cycles=int(args.cycles),
        playback_frame=int(args.playback_frame),
        commands=commands,
        requested_n_frames=int(args.requested_n_frames),
        warmup_steps=int(args.warmup_steps),
        trace_decode_layers=bool(args.trace_decode_layers),
        variants=variants,
        terrain_cases=terrain_cases,
    )


if __name__ == "__main__":
    raise SystemExit(main())
