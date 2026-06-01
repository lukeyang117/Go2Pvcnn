from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from Go2Pvcnn.tests.fixtures.viewer_runtime_diagnostics import make_real_runtime_fixture


LEG_NAMES = ("FL", "FR", "RL", "RR")
LEFT_LEGS = (0, 2)
RIGHT_LEGS = (1, 3)
DIAGONAL_PAIRS = ((0, 3), (1, 2))
CUSTOM_COMMANDS: dict[str, tuple[float, float, float]] = {
    "forward_slow": (0.15, 0.0, 0.0),
    "forward_fast": (0.45, 0.0, 0.0),
    "backward_slow": (-0.15, 0.0, 0.0),
    "backward_fast": (-0.45, 0.0, 0.0),
    "lateral_left_slow": (0.0, 0.12, 0.0),
    "lateral_left_fast": (0.0, 0.38, 0.0),
    "lateral_right_slow": (0.0, -0.12, 0.0),
    "lateral_right_fast": (0.0, -0.38, 0.0),
    "yaw_left_slow": (0.0, 0.0, 0.15),
    "yaw_left_fast": (0.0, 0.0, 0.45),
    "yaw_right_slow": (0.0, 0.0, -0.15),
    "yaw_right_fast": (0.0, 0.0, -0.45),
    "forward_yaw_left": (0.25, 0.0, 0.25),
    "forward_yaw_right": (0.25, 0.0, -0.25),
    "backward_yaw_left": (-0.25, 0.0, 0.25),
    "backward_yaw_right": (-0.25, 0.0, -0.25),
    "lateral_left_yaw_left": (0.0, 0.20, 0.25),
    "lateral_left_yaw_right": (0.0, 0.20, -0.25),
    "lateral_right_yaw_left": (0.0, -0.20, 0.25),
    "lateral_right_yaw_right": (0.0, -0.20, -0.25),
}


def _command_sequences() -> tuple[tuple[str, tuple[str, ...]], ...]:
    default = (
        "yaw_left_only:yaw_left;"
        "yaw_right_only:yaw_right;"
        "lateral_left_yaw_right_lateral_left:lateral_left,yaw_right,lateral_left;"
        "lateral_right_yaw_left_lateral_right:lateral_right,yaw_left,lateral_right"
    )
    requested = os.environ.get("MPC_YAW_GAIT_SEQUENCES", default)
    out: list[tuple[str, tuple[str, ...]]] = []
    for raw in requested.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        name, sep, segments = raw.partition(":")
        if not sep:
            raise ValueError(f"bad sequence {raw!r}; expected name:a,b,c")
        parts = tuple(part.strip() for part in segments.split(",") if part.strip())
        if not parts:
            raise ValueError(f"bad sequence {raw!r}; no segments")
        out.append((name.strip(), parts))
    return tuple(out)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(quat, dtype=torch.float64)
    w, x, y, z = q.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _body_relative_foot(root_pos: torch.Tensor, root_quat: torch.Tensor, foot_pos: torch.Tensor) -> torch.Tensor:
    root = torch.as_tensor(root_pos, dtype=torch.float64)
    foot = torch.as_tensor(foot_pos, dtype=torch.float64)
    yaw = _yaw_from_quat_wxyz(torch.as_tensor(root_quat, dtype=torch.float64))
    rel = foot - root.unsqueeze(2)
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    rel_x = cy * rel[..., 0] + sy * rel[..., 1]
    rel_y = -sy * rel[..., 0] + cy * rel[..., 1]
    return torch.stack((rel_x, rel_y, rel[..., 2]), dim=-1)


def _sample_ground_height(terrain, xy: torch.Tensor) -> torch.Tensor:
    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float64, device=xy.device)
    if height_map.ndim == 2:
        height_map = height_map.unsqueeze(0)
    xy = torch.as_tensor(xy, dtype=torch.float64, device=height_map.device)
    flat_xy = xy.reshape(xy.shape[0], -1, 2)
    x0, x1 = terrain.world_x_range
    y0, y1 = terrain.world_y_range
    x_norm = (flat_xy[..., 0] - float(x0)) / max(float(x1) - float(x0), 1.0e-6) * 2.0 - 1.0
    y_norm = (flat_xy[..., 1] - float(y0)) / max(float(y1) - float(y0), 1.0e-6) * 2.0 - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        height_map.unsqueeze(1),
        grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return sampled[:, 0, :, 0].reshape(xy.shape[:-1])


def _plan_with_viewer_memory(runtime, terrain, state, command, memory):
    viewer = runtime._viewer
    result = viewer._plan_viewer_trajectory(
                terrain=terrain,
                state=state,
                command=command,
                mpc_cfg=runtime.mpc_planner_cfg,
    )
    return result, memory


def _command_tensor(runtime, name: str) -> torch.Tensor:
    if name in runtime.command_cases:
        return runtime._command_tensor(name)[:1]
    command = CUSTOM_COMMANDS.get(name)
    if command is None:
        known = ", ".join(sorted([*runtime.command_cases.keys(), *CUSTOM_COMMANDS.keys()]))
        raise KeyError(f"unknown command segment {name!r}; known commands: {known}")
    return torch.tensor(command, dtype=torch.float64, device=runtime.base_env.device).view(1, 3)


def _stance_ground_metrics(terrain, result, *, tol: float) -> dict[str, float]:
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
    ground = _sample_ground_height(terrain, foot[..., :2])
    gap = foot[..., 2] - ground
    stance_gap = gap[contact]
    if int(stance_gap.numel()) == 0:
        return {
            "stance_gap_mean": 0.0,
            "stance_gap_abs_mean": 0.0,
            "stance_gap_max": 0.0,
            "stance_airborne_ratio": 0.0,
            "stance_airborne_max_gap": 0.0,
        }
    positive = torch.relu(stance_gap)
    airborne = stance_gap > float(tol)
    return {
        "stance_gap_mean": float(stance_gap.mean().item()),
        "stance_gap_abs_mean": float(torch.abs(stance_gap).mean().item()),
        "stance_gap_max": float(stance_gap.max().item()),
        "stance_airborne_ratio": float(airborne.to(dtype=torch.float64).mean().item()),
        "stance_airborne_max_gap": float(positive.max().item()),
    }


def _stance_ground_metrics_from_frame(
    terrain,
    foot_pos_w: torch.Tensor,
    contact_state: torch.Tensor,
    *,
    tol: float,
    prefix: str,
) -> dict[str, float]:
    foot = torch.as_tensor(foot_pos_w, dtype=torch.float64)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    ground = _sample_ground_height(terrain, foot[..., :2])
    gap = foot[..., 2] - ground
    stance_gap = gap[contact]
    if int(stance_gap.numel()) == 0:
        return {
            f"{prefix}_stance_gap_mean": 0.0,
            f"{prefix}_stance_gap_abs_mean": 0.0,
            f"{prefix}_stance_gap_max": 0.0,
            f"{prefix}_stance_airborne_ratio": 0.0,
            f"{prefix}_stance_airborne_max_gap": 0.0,
        }
    positive = torch.relu(stance_gap)
    airborne = stance_gap > float(tol)
    return {
        f"{prefix}_stance_gap_mean": float(stance_gap.mean().item()),
        f"{prefix}_stance_gap_abs_mean": float(torch.abs(stance_gap).mean().item()),
        f"{prefix}_stance_gap_max": float(stance_gap.max().item()),
        f"{prefix}_stance_airborne_ratio": float(airborne.to(dtype=torch.float64).mean().item()),
        f"{prefix}_stance_airborne_max_gap": float(positive.max().item()),
    }


def _leg_swing_stats(rel_body: torch.Tensor, contact: torch.Tensor, *, front_eps: float) -> dict[str, float]:
    swing = torch.logical_not(contact)
    rel_x = rel_body[..., 0]
    stats: dict[str, float] = {}
    front_occupancy = torch.zeros_like(rel_x, dtype=torch.float64)
    for leg_idx, leg_name in enumerate(LEG_NAMES):
        mask = swing[..., leg_idx]
        values = rel_x[..., leg_idx][mask]
        if int(values.numel()) == 0:
            stats[f"{leg_name}_swing_count"] = 0.0
            stats[f"{leg_name}_front_ratio"] = 0.0
            stats[f"{leg_name}_rear_ratio"] = 0.0
            stats[f"{leg_name}_rel_x_mean"] = 0.0
            stats[f"{leg_name}_rel_x_min"] = 0.0
            stats[f"{leg_name}_rel_x_max"] = 0.0
            stats[f"{leg_name}_rel_x_span"] = 0.0
            stats[f"{leg_name}_front_rear_switches"] = 0.0
            continue
        signs = torch.where(values > front_eps, torch.ones_like(values), torch.where(values < -front_eps, -torch.ones_like(values), torch.zeros_like(values)))
        nonzero = signs[signs != 0]
        switches = 0
        if int(nonzero.numel()) > 1:
            switches = int(torch.count_nonzero(nonzero[1:] != nonzero[:-1]).item())
        stats[f"{leg_name}_swing_count"] = float(values.numel())
        stats[f"{leg_name}_front_ratio"] = float((values > front_eps).to(dtype=torch.float64).mean().item())
        stats[f"{leg_name}_rear_ratio"] = float((values < -front_eps).to(dtype=torch.float64).mean().item())
        stats[f"{leg_name}_rel_x_mean"] = float(values.mean().item())
        stats[f"{leg_name}_rel_x_min"] = float(values.min().item())
        stats[f"{leg_name}_rel_x_max"] = float(values.max().item())
        stats[f"{leg_name}_rel_x_span"] = float((values.max() - values.min()).item())
        stats[f"{leg_name}_front_rear_switches"] = float(switches)
        front_occupancy[..., leg_idx] = (swing[..., leg_idx] & (rel_x[..., leg_idx] > front_eps)).to(dtype=torch.float64)

    left_front = front_occupancy[..., LEFT_LEGS].sum(dim=-1)
    right_front = front_occupancy[..., RIGHT_LEGS].sum(dim=-1)
    active_front = (left_front + right_front) > 0
    if bool(active_front.any().item()):
        side_bias = torch.abs(left_front[active_front] - right_front[active_front])
        stats["front_side_imbalance_mean"] = float(side_bias.mean().item())
        stats["left_front_frame_ratio"] = float((left_front[active_front] > 0).to(dtype=torch.float64).mean().item())
        stats["right_front_frame_ratio"] = float((right_front[active_front] > 0).to(dtype=torch.float64).mean().item())
    else:
        stats["front_side_imbalance_mean"] = 0.0
        stats["left_front_frame_ratio"] = 0.0
        stats["right_front_frame_ratio"] = 0.0

    diag_front_counts = []
    for a, b in DIAGONAL_PAIRS:
        diag_front_counts.append((front_occupancy[..., a] + front_occupancy[..., b]).reshape(-1))
    diag_front = torch.stack(diag_front_counts, dim=0)
    diag_total = diag_front.sum(dim=0)
    diag_active = diag_total > 0
    if bool(diag_active.any().item()):
        stats["diagonal_front_imbalance_mean"] = float(torch.abs(diag_front[0, diag_active] - diag_front[1, diag_active]).mean().item())
        stats["diag_FL_RR_front_ratio"] = float((diag_front[0, diag_active] > 0).to(dtype=torch.float64).mean().item())
        stats["diag_FR_RL_front_ratio"] = float((diag_front[1, diag_active] > 0).to(dtype=torch.float64).mean().item())
    else:
        stats["diagonal_front_imbalance_mean"] = 0.0
        stats["diag_FL_RR_front_ratio"] = 0.0
        stats["diag_FR_RL_front_ratio"] = 0.0

    front_extents = []
    for leg_idx in range(4):
        mask = swing[..., leg_idx] & (rel_x[..., leg_idx] > front_eps)
        values = rel_x[..., leg_idx][mask]
        if int(values.numel()) > 0:
            front_extents.append(values.mean())
    if front_extents:
        ext = torch.stack(front_extents)
        stats["front_extent_mean"] = float(ext.mean().item())
        stats["front_extent_std"] = float(ext.std(unbiased=False).item())
        stats["front_extent_cv"] = float((ext.std(unbiased=False) / torch.clamp(torch.abs(ext.mean()), min=1.0e-6)).item())
    else:
        stats["front_extent_mean"] = 0.0
        stats["front_extent_std"] = 0.0
        stats["front_extent_cv"] = 0.0
    return stats


def _pair_left_right_alternation_stats(
    rel_body: torch.Tensor,
    contact: torch.Tensor,
    *,
    front_eps: float,
) -> dict[str, float]:
    rel_x = rel_body[..., 0]
    swing = torch.logical_not(contact)
    stats: dict[str, float] = {}
    for prefix, left_idx, right_idx in (("front_pair", 0, 1), ("rear_pair", 2, 3)):
        pair_active = swing[..., left_idx] | swing[..., right_idx]
        lead_delta = rel_x[..., left_idx] - rel_x[..., right_idx]
        values = lead_delta[pair_active]
        if int(values.numel()) == 0:
            stats[f"{prefix}_active_ratio"] = 0.0
            stats[f"{prefix}_left_ahead_ratio"] = 0.0
            stats[f"{prefix}_right_ahead_ratio"] = 0.0
            stats[f"{prefix}_lead_switches"] = 0.0
            stats[f"{prefix}_lead_abs_mean"] = 0.0
            stats[f"{prefix}_lead_signed_mean"] = 0.0
            stats[f"{prefix}_both_swing_ratio"] = 0.0
            continue
        signs = torch.where(
            values > front_eps,
            torch.ones_like(values),
            torch.where(values < -front_eps, -torch.ones_like(values), torch.zeros_like(values)),
        )
        nonzero = signs[signs != 0]
        switches = 0
        if int(nonzero.numel()) > 1:
            switches = int(torch.count_nonzero(nonzero[1:] != nonzero[:-1]).item())
        both_swing = swing[..., left_idx] & swing[..., right_idx]
        stats[f"{prefix}_active_ratio"] = float(pair_active.to(dtype=torch.float64).mean().item())
        stats[f"{prefix}_left_ahead_ratio"] = float((values > front_eps).to(dtype=torch.float64).mean().item())
        stats[f"{prefix}_right_ahead_ratio"] = float((values < -front_eps).to(dtype=torch.float64).mean().item())
        stats[f"{prefix}_lead_switches"] = float(switches)
        stats[f"{prefix}_lead_abs_mean"] = float(torch.abs(values).mean().item())
        stats[f"{prefix}_lead_signed_mean"] = float(values.mean().item())
        stats[f"{prefix}_both_swing_ratio"] = float(both_swing.to(dtype=torch.float64).mean().item())
    return stats


def _summarize_cycle(runtime, terrain, result, *, tol: float, front_eps: float) -> dict[str, float]:
    root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
    quat = torch.as_tensor(result.root_quat_w, dtype=torch.float64)
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
    rel_body = _body_relative_foot(root, quat, foot)
    out = _stance_ground_metrics(terrain, result, tol=tol)
    out.update(_leg_swing_stats(rel_body, contact, front_eps=front_eps))
    out.update(_pair_left_right_alternation_stats(rel_body, contact, front_eps=front_eps))
    out["contact_ratio"] = float(contact.to(dtype=torch.float64).mean().item())
    out["swing_count_mean"] = float(torch.logical_not(contact).sum(dim=-1).to(dtype=torch.float64).mean().item())
    return out


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _segment_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    summary: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        summary[f"{key}_mean"] = _mean(vals)
        summary[f"{key}_max"] = _max(vals)
    for leg in LEG_NAMES:
        summary[f"{leg}_front_rear_switches_total"] = sum(float(row.get(f"{leg}_front_rear_switches", 0.0)) for row in rows)
    return summary


def main() -> int:
    output_path = Path(os.environ.get("MPC_YAW_GAIT_OUTPUT", "/tmp/mpc_yaw_gait_failure_probe.jsonl"))
    cycles = int(os.environ.get("MPC_YAW_GAIT_CYCLES", "24"))
    tol = float(os.environ.get("MPC_YAW_GAIT_STANCE_AIR_TOL", "0.02"))
    front_eps = float(os.environ.get("MPC_YAW_GAIT_FRONT_EPS", "0.01"))
    device = os.environ.get("MPC_TEST_DEVICE", "cuda:2")
    runtime = None
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            runtime = make_real_runtime_fixture(num_envs=2, planner_backend="mpc", device=device)
            viewer = runtime._viewer
            terrain = runtime._single_env_terrain()
            sequences = _command_sequences()
            handle.write(json.dumps({"kind": "startup", "cycles": cycles, "device": device, "sequences": sequences}) + "\n")
            for seq_name, segments in sequences:
                runtime.reset()
                state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
                memory = None
                seq_summaries = []
                for segment in segments:
                    command = _command_tensor(runtime, segment)
                    cycle_rows: list[dict[str, float]] = []
                    for cycle_idx in range(cycles):
                        result, memory = _plan_with_viewer_memory(runtime, terrain, state, command, memory)
                        metrics = _summarize_cycle(runtime, terrain, result, tol=tol, front_eps=front_eps)
                        frame_idx = int(result.num_frames) - 1
                        viewer._apply_direct_playback_to_robot(runtime.robot, result, frame_idx=frame_idx)
                        runtime.base_env.scene.write_data_to_sim()
                        runtime.base_env.sim.render()
                        runtime.base_env.scene.update(float(runtime.base_env.physics_dt))
                        actual_kin = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
                        last_contact = torch.as_tensor(result.contact_state[:, frame_idx], dtype=torch.bool)
                        actual_foot = torch.as_tensor(actual_kin["foot_pos_w"], dtype=torch.float64)
                        metrics.update(
                            _stance_ground_metrics_from_frame(
                                terrain,
                                actual_foot,
                                last_contact,
                                tol=tol,
                                prefix="actual_last",
                            )
                        )
                        plan_last_foot = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64)
                        metrics.update(
                            _stance_ground_metrics_from_frame(
                                terrain,
                                plan_last_foot,
                                last_contact,
                                tol=tol,
                                prefix="plan_last",
                            )
                        )
                        metrics["actual_plan_last_foot_err_mean"] = float(
                            torch.linalg.vector_norm(actual_foot - plan_last_foot, dim=-1).mean().item()
                        )
                        cycle_rows.append(metrics)
                        handle.write(json.dumps({
                            "kind": "cycle",
                            "seq": seq_name,
                            "segment": segment,
                            "cycle": cycle_idx,
                            **metrics,
                        }, ensure_ascii=False) + "\n")
                        handle.write(json.dumps({
                            "kind": "actual_cycle",
                            "seq": seq_name,
                            "segment": segment,
                            "cycle": cycle_idx,
                            **{
                                key: value
                                for key, value in metrics.items()
                                if key.startswith("actual_last_")
                                or key.startswith("plan_last_")
                                or key == "actual_plan_last_foot_err_mean"
                            },
                        }, ensure_ascii=False) + "\n")
                        state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
                    summary = _segment_summary(cycle_rows)
                    seq_summaries.append(summary)
                    handle.write(json.dumps({
                        "kind": "segment",
                        "seq": seq_name,
                        "segment": segment,
                        "cycles": cycles,
                        **summary,
                    }, ensure_ascii=False) + "\n")
                combined = _segment_summary(seq_summaries)
                handle.write(json.dumps({"kind": "summary", "seq": seq_name, **combined}, ensure_ascii=False) + "\n")
                handle.flush()
        print(output_path)
        return 0
    except Exception as exc:  # noqa: BLE001
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "kind": "exception",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False) + "\n")
        print(output_path)
        return 1
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
