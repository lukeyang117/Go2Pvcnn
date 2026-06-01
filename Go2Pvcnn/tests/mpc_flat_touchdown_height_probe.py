from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT, GO2PVCNN_ROOT / "tests"):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from extension.batch_mpc_planner.terrain import height_at  # noqa: E402
from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture, refresh_targeted_scanner_pose  # noqa: E402


def _world_to_scanner_local(terrain, xy_w: torch.Tensor) -> torch.Tensor:
    xy = torch.as_tensor(xy_w, dtype=torch.float32, device=terrain.height_map.device)
    sensor_pos = torch.as_tensor(terrain.sensor_pos_w, dtype=torch.float32, device=xy.device)
    sensor_yaw = torch.as_tensor(terrain.sensor_yaw, dtype=torch.float32, device=xy.device).reshape(-1)
    if sensor_pos.ndim == 1:
        sensor_pos = sensor_pos.view(1, -1)
    if xy.ndim == 2:
        xy = xy.unsqueeze(0)
    delta = xy - sensor_pos[:, None, :2]
    cy = torch.cos(sensor_yaw).view(-1, 1)
    sy = torch.sin(sensor_yaw).view(-1, 1)
    return torch.stack(
        (cy * delta[..., 0] + sy * delta[..., 1], -sy * delta[..., 0] + cy * delta[..., 1]),
        dim=-1,
    )


def _nearest_scanner_height(terrain, xy_w: torch.Tensor) -> torch.Tensor:
    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    if height_map.ndim == 2:
        height_map = height_map.unsqueeze(0)
    local = _world_to_scanner_local(terrain, xy_w)
    x0, x1 = terrain.world_x_range
    y0, y1 = terrain.world_y_range
    height = int(height_map.shape[1])
    width = int(height_map.shape[2])
    col = torch.round((local[..., 0] - float(x0)) / max(float(x1) - float(x0), 1.0e-6) * float(width - 1))
    row = torch.round((local[..., 1] - float(y0)) / max(float(y1) - float(y0), 1.0e-6) * float(height - 1))
    col = col.to(dtype=torch.long).clamp(0, width - 1)
    row = row.to(dtype=torch.long).clamp(0, height - 1)
    b = torch.arange(int(height_map.shape[0]), device=height_map.device).view(-1, 1).expand_as(row)
    return height_map[b, row, col]


def _manual_bilinear_scanner_height(terrain, xy_w: torch.Tensor) -> torch.Tensor:
    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32)
    if height_map.ndim == 2:
        height_map = height_map.unsqueeze(0)
    local = _world_to_scanner_local(terrain, xy_w)
    x0, x1 = terrain.world_x_range
    y0, y1 = terrain.world_y_range
    height = int(height_map.shape[1])
    width = int(height_map.shape[2])
    gx = ((local[..., 0] - float(x0)) / max(float(x1) - float(x0), 1.0e-6) * float(width - 1)).clamp(0.0, float(width - 1))
    gy = ((local[..., 1] - float(y0)) / max(float(y1) - float(y0), 1.0e-6) * float(height - 1)).clamp(0.0, float(height - 1))
    x_floor = torch.floor(gx)
    y_floor = torch.floor(gy)
    x0i = x_floor.to(dtype=torch.long).clamp(0, width - 1)
    y0i = y_floor.to(dtype=torch.long).clamp(0, height - 1)
    x1i = (x0i + 1).clamp(0, width - 1)
    y1i = (y0i + 1).clamp(0, height - 1)
    ax = (gx - x_floor).to(dtype=height_map.dtype)
    ay = (gy - y_floor).to(dtype=height_map.dtype)
    b = torch.arange(int(height_map.shape[0]), device=height_map.device).view(-1, 1).expand_as(x0i)
    h00 = height_map[b, y0i, x0i]
    h10 = height_map[b, y0i, x1i]
    h01 = height_map[b, y1i, x0i]
    h11 = height_map[b, y1i, x1i]
    return (1.0 - ax) * (1.0 - ay) * h00 + ax * (1.0 - ay) * h10 + (1.0 - ax) * ay * h01 + ax * ay * h11


def _format_vec(values: torch.Tensor) -> str:
    flat = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    return "[" + ", ".join(f"{float(v.item()):+.5f}" for v in flat) + "]"


def _summarize_gap(prefix: str, marker: torch.Tensor, terrain) -> dict[str, object]:
    xy = torch.as_tensor(marker, dtype=torch.float32, device=terrain.height_map.device)[..., :2]
    z = torch.as_tensor(marker, dtype=torch.float32, device=terrain.height_map.device)[..., 2]
    mpc_bilinear = height_at(terrain, xy, mode="bilinear")
    nearest = _nearest_scanner_height(terrain, xy)
    return {
        "prefix": prefix,
        "z": z.detach().clone(),
        "mpc_bilinear": mpc_bilinear.detach().clone(),
        "nearest": nearest.detach().clone(),
        "td_minus_mpc": z - mpc_bilinear,
        "td_minus_nearest": z - nearest,
    }


def run_probe(
    *,
    device: str,
    cycles: int,
    playback_frame: int,
    speeds: tuple[float, ...],
    zero_after_forward_frame: int | None,
) -> int:
    runtime = RealViewerRuntimeFixture(
        num_envs=1,
        device=device,
        terrain="flat",
        warmup_steps=6,
        requested_n_frames=50,
        planner_backend="mpc",
        task_id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
        env_cfg_entry_point=(
            "go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg:"
            "TeacherElevationTrajectoryMpcSemanticEnvCfg"
        ),
    )
    try:
        rows = []
        for speed in speeds:
            runtime.reset()
            command = torch.tensor(
                [[float(speed), 0.0, 0.0]],
                dtype=torch.float64,
                device=runtime.base_env.device,
            )
            for cycle in range(int(cycles)):
                state = runtime._single_env_state()
                terrain = runtime._single_env_terrain()
                result = runtime._viewer._plan_viewer_trajectory(
                    terrain=terrain,
                    state=state,
                    command=command,
                    mpc_cfg=runtime.mpc_planner_cfg,
                )
                touchdown = torch.as_tensor(
                    runtime._viewer.PlannerVisualizer._touchdown_markers_world(result),
                    dtype=torch.float32,
                    device=terrain.height_map.device,
                )
                xy = touchdown[..., :2]
                touchdown_z = touchdown[..., 2]
                mpc_bilinear = height_at(terrain, xy, mode="bilinear")
                manual_bilinear = _manual_bilinear_scanner_height(terrain, xy)
                scanner_nearest = _nearest_scanner_height(terrain, xy)
                frame_idx = min(int(playback_frame), int(result.num_frames) - 1)
                root0 = torch.as_tensor(result.root_pos_w[:, 0], dtype=torch.float64)
                rootf = torch.as_tensor(result.root_pos_w[:, frame_idx], dtype=torch.float64)
                rows.append(
                    {
                        "speed": float(speed),
                        "cycle": cycle,
                        "frame": frame_idx,
                        "dx": float((rootf[0, 0] - root0[0, 0]).item()),
                        "td_minus_mpc": touchdown_z - mpc_bilinear,
                        "td_minus_manual": touchdown_z - manual_bilinear,
                        "td_minus_nearest": touchdown_z - scanner_nearest,
                        "mpc_minus_manual": mpc_bilinear - manual_bilinear,
                        "mpc_minus_nearest": mpc_bilinear - scanner_nearest,
                        "height_min": float(torch.as_tensor(terrain.height_map).min().item()),
                        "height_max": float(torch.as_tensor(terrain.height_map).max().item()),
                        "touchdown_z": touchdown_z.detach().clone(),
                        "mpc_bilinear": mpc_bilinear.detach().clone(),
                        "scanner_nearest": scanner_nearest.detach().clone(),
                        "viz_touchdown": touchdown.detach().clone(),
                    }
                )
                runtime._viewer._viewer_direct_playback_step(runtime.base_env, result, frame_idx=frame_idx)
                refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)

        print(
            "[flat-forward-touchdown-height-probe] "
            f"device={device} speeds={speeds} cycles_per_speed={cycles} playback_frame={playback_frame} "
            f"scanner={runtime.scanner_name} horizon={runtime.requested_n_frames}",
            flush=True,
        )
        for row in rows:
            print(
                f"speed={row['speed']:+.2f} cycle={row['cycle']} frame={row['frame']} dx={row['dx']:+.4f} "
                f"height_range=[{row['height_min']:+.5f},{row['height_max']:+.5f}] "
                f"viz_td_z={_format_vec(row['touchdown_z'])} "
                f"mpc_bilin={_format_vec(row['mpc_bilinear'])} "
                f"scanner_nearest={_format_vec(row['scanner_nearest'])} "
                f"td-mpc={_format_vec(row['td_minus_mpc'])} "
                f"td-nearest={_format_vec(row['td_minus_nearest'])} "
                f"mpc-manual={_format_vec(row['mpc_minus_manual'])} "
                f"mpc-nearest={_format_vec(row['mpc_minus_nearest'])}",
                flush=True,
            )
        max_abs_td_mpc = max(float(row["td_minus_mpc"].abs().max().item()) for row in rows)
        max_abs_td_nearest = max(float(row["td_minus_nearest"].abs().max().item()) for row in rows)
        max_abs_mpc_manual = max(float(row["mpc_minus_manual"].abs().max().item()) for row in rows)
        max_abs_mpc_nearest = max(float(row["mpc_minus_nearest"].abs().max().item()) for row in rows)
        print(
            "[flat-forward-touchdown-height-summary] "
            f"max_abs_td_minus_mpc={max_abs_td_mpc:.6f} "
            f"max_abs_td_minus_scanner_nearest={max_abs_td_nearest:.6f} "
            f"max_abs_mpc_minus_manual_bilinear={max_abs_mpc_manual:.6f} "
            f"max_abs_mpc_minus_scanner_nearest={max_abs_mpc_nearest:.6f}",
            flush=True,
        )
        if zero_after_forward_frame is not None and int(zero_after_forward_frame) >= 0:
            runtime.reset()
            forward_command = torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float64, device=runtime.base_env.device)
            zero_command = torch.zeros_like(forward_command)
            state = runtime._single_env_state()
            terrain = runtime._single_env_terrain()
            forward_result = runtime._viewer._plan_viewer_trajectory(
                terrain=terrain,
                state=state,
                command=forward_command,
                mpc_cfg=runtime.mpc_planner_cfg,
            )
            frame_idx = min(int(zero_after_forward_frame), int(forward_result.num_frames) - 1)
            landing_frame = runtime._viewer._viewer_find_grounded_all_feet_frame(
                forward_result,
                terrain,
                start_frame=frame_idx,
            )
            if landing_frame is not None:
                landing_foot = torch.as_tensor(
                    forward_result.foot_pos_w[:, landing_frame],
                    dtype=torch.float32,
                    device=terrain.height_map.device,
                )
                landing_gap = _summarize_gap("forward_landing_foot", landing_foot, terrain)
                print(
                    "[flat-forward-drain-target] "
                    f"start_frame={frame_idx} landing_frame={landing_frame} "
                    f"landing_foot_minus_mpc={_format_vec(landing_gap['td_minus_mpc'])}",
                    flush=True,
                )
            runtime._viewer._viewer_direct_playback_step(runtime.base_env, forward_result, frame_idx=frame_idx)
            refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
            zero_state = runtime._single_env_state()
            zero_terrain = runtime._single_env_terrain()
            zero_result = runtime._viewer._plan_viewer_trajectory(
                terrain=zero_terrain,
                state=zero_state,
                command=zero_command,
                mpc_cfg=runtime.mpc_planner_cfg,
            )
            zero_viz_td = runtime._viewer.PlannerVisualizer._touchdown_markers_world(zero_result)
            foot_now = torch.as_tensor(zero_state.foot_pos, dtype=torch.float32, device=zero_terrain.height_map.device)
            zero_td = _summarize_gap("zero_replan_viz_touchdown", zero_viz_td, zero_terrain)
            foot_gap = _summarize_gap("zero_replan_state_foot", foot_now, zero_terrain)
            print(
                "[flat-zero-after-forward-repro] "
                f"forward_frame={frame_idx} "
                f"viz_td_z={_format_vec(zero_td['z'])} "
                f"viz_td_minus_mpc={_format_vec(zero_td['td_minus_mpc'])} "
                f"state_foot_z={_format_vec(foot_gap['z'])} "
                f"state_foot_minus_mpc={_format_vec(foot_gap['td_minus_mpc'])} "
                f"td_equals_state_foot_max_abs={float((torch.as_tensor(zero_viz_td, device=foot_now.device, dtype=foot_now.dtype) - foot_now).abs().max().item()):.6f}",
                flush=True,
            )
        if not math.isfinite(max_abs_td_mpc):
            return 2
        return 0
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--playback-frame", type=int, default=49)
    parser.add_argument("--speeds", default="0.10,0.30,0.50")
    parser.add_argument("--zero-after-forward-frame", type=int, default=-1)
    args = parser.parse_args()
    speeds = tuple(float(item.strip()) for item in str(args.speeds).split(",") if item.strip())
    return run_probe(
        device=str(args.device),
        cycles=int(args.cycles),
        playback_frame=int(args.playback_frame),
        speeds=speeds,
        zero_after_forward_frame=int(args.zero_after_forward_frame),
    )


if __name__ == "__main__":
    raise SystemExit(main())
