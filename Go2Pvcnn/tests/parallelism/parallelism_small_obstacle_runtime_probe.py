from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT, GO2PVCNN_ROOT / "tests"):
    path = str(_path)
    if path not in sys.path:
        sys.path.insert(0, path)

from tests.fixtures.viewer_runtime_diagnostics import make_real_runtime_fixture  # noqa: E402


DEFAULT_COMMANDS: tuple[tuple[float, float, float], ...] = (
    (0.2, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.5, -0.5, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, -1.0),
    (0.5, 0.0, 1.0),
)


def _command_basis(command: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    xy = torch.tensor(command[:2], dtype=torch.float64)
    if float(torch.linalg.vector_norm(xy).item()) <= 1.0e-6:
        xy = torch.tensor((1.0, 0.0), dtype=torch.float64)
    forward = xy / torch.linalg.vector_norm(xy)
    left = torch.stack((-forward[1], forward[0]))
    return forward, left


def _path_crossing_metrics(root_xy: torch.Tensor, obstacle_xy: torch.Tensor, command: tuple[float, float, float]) -> dict[str, float | int]:
    forward, left = _command_basis(command)
    rel = torch.as_tensor(root_xy, dtype=torch.float64) - torch.as_tensor(obstacle_xy, dtype=torch.float64).view(1, 2)
    along = rel @ forward
    lateral = rel @ left
    return {
        "root_crosses_small": int(bool((along[0] < 0.0 and along[-1] > 0.0).item())),
        "root_start_along_small_m": float(along[0].item()),
        "root_end_along_small_m": float(along[-1].item()),
        "root_min_abs_lateral_small_m": float(torch.abs(lateral).min().item()),
        "root_delta_xy_norm_m": float(torch.linalg.vector_norm(root_xy[-1] - root_xy[0]).item()),
    }


def _diagnostic_counts(result) -> dict[str, object]:
    diagnostics = result.parallelism_diagnostics
    valid = torch.as_tensor(diagnostics.candidate_valid, dtype=torch.bool)
    reject = torch.as_tensor(diagnostics.candidate_reject_bits, dtype=torch.bool)
    collision = torch.as_tensor(diagnostics.candidate_collision_bits, dtype=torch.bool)
    per_leg_valid = valid[0].sum(dim=-1).to(dtype=torch.long)
    reject_names = ("valid_map", "joint", "landing", "collision", "candidate_semantic", "fk_touchdown_semantic")
    return {
        "valid_env": bool(torch.as_tensor(result.feasible, dtype=torch.bool).reshape(-1)[0].item()),
        "standstill": bool(torch.allclose(result.root_pos_w, result.root_pos_w[:, :1], atol=1.0e-6, rtol=0.0)),
        "valid_count": int(valid[0].sum().item()),
        "per_leg_valid": [int(v) for v in per_leg_valid.tolist()],
        "reject": {name: int(reject[0, :, :, idx].any(dim=0).sum().item()) for idx, name in enumerate(reject_names)},
        "per_leg_reject": {
            name: [int(v) for v in reject[0, :, :, idx].sum(dim=-1).to(dtype=torch.long).tolist()]
            for idx, name in enumerate(reject_names)
        },
        "per_leg_collision_shape": {
            str(name): [int(v) for v in collision[0, :, :, idx].sum(dim=-1).to(dtype=torch.long).tolist()]
            for idx, name in enumerate(tuple(diagnostics.collision_shape_names))
        },
    }


def _reset_before_small(runtime, command: tuple[float, float, float], *, start_offset_m: float) -> torch.Tensor:
    anchor = runtime.s4_semantic_course_anchor("small")
    runtime.reset()
    forward, _left = _command_basis(command)
    obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float64, device=runtime.base_env.device)
    start_xy = obstacle_xy - forward.to(device=runtime.base_env.device) * float(start_offset_m)
    runtime._write_env0_root_xy((float(start_xy[0].item()), float(start_xy[1].item())), z_clearance=0.65)
    runtime._viewer._viewer_ground_robot_from_scanner(
        runtime.base_env,
        runtime.scanner,
        runtime.foot_ids.tolist(),
        root_pos_xy=start_xy.view(1, 2).to(dtype=torch.float32),
    )
    runtime._sync_targeted_scan_pose()
    return obstacle_xy.cpu()


def _commands_from_arg(value: str | None) -> tuple[tuple[float, float, float], ...]:
    if value is None or str(value).strip() == "":
        return DEFAULT_COMMANDS
    commands = []
    for chunk in str(value).split(";"):
        parts = [float(part.strip()) for part in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError("--commands entries must be vx,vy,vyaw triples separated by ';'")
        commands.append((parts[0], parts[1], parts[2]))
    return tuple(commands)


def run_probe(args: argparse.Namespace, commands: tuple[tuple[float, float, float], ...]) -> list[dict[str, object]]:
    runtime = make_real_runtime_fixture(
        num_envs=1,
        terrain="task",
        warmup_steps=0,
        requested_n_frames=int(args.n_frames),
        planner_backend="parallelism",
        task_id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
        semantic_small_height_m=0.16,
    )
    rows: list[dict[str, object]] = []
    try:
        args_cli = SimpleNamespace(plan_dt=float(args.plan_dt))
        state = runtime._viewer.ViewerTestTerminalState(
            swing_clearance_m=float(args.swing_clearance_m),
            semantic_touchdown_margin_m=float(args.semantic_touchdown_margin_m),
            candidate_radius_m=float(args.candidate_radius_m),
            standstill_fallback_enabled=True,
        )
        for command in commands:
            obstacle_xy = _reset_before_small(runtime, command, start_offset_m=float(args.start_offset_m))
            command_t = torch.tensor(command, dtype=torch.float32, device=runtime.base_env.device).view(1, 3)
            path_xy = [torch.as_tensor(runtime.robot.data.root_pos_w[0, :2], dtype=torch.float64).cpu()]
            for cycle in range(int(args.cycles)):
                result, _ray_hits = runtime._viewer._plan_parallelism_viewer_trajectory(
                    base_env=runtime.base_env,
                    scanner=runtime.scanner,
                    foot_ids=runtime.foot_ids.tolist(),
                    command=command_t,
                    args_cli=args_cli,
                    test_terminal_state=state,
                )
                runtime._viewer._viewer_direct_playback_step(runtime.base_env, result, frame_idx=int(result.num_frames) - 1)
                runtime._sync_targeted_scan_pose()
                path_xy.append(torch.as_tensor(runtime.robot.data.root_pos_w[0, :2], dtype=torch.float64).cpu())
                row = {
                    "cmd": [float(v) for v in command],
                    "cycle": int(cycle),
                    **_diagnostic_counts(result),
                    **_path_crossing_metrics(torch.stack(path_xy), obstacle_xy, command),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        return rows
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallelism small-obstacle rolling runtime probe.")
    parser.add_argument("--n-frames", type=int, default=24)
    parser.add_argument("--plan-dt", type=float, default=0.02)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--start-offset-m", type=float, default=0.30)
    parser.add_argument("--swing-clearance-m", type=float, default=0.25)
    parser.add_argument("--semantic-touchdown-margin-m", type=float, default=0.0)
    parser.add_argument("--candidate-radius-m", type=float, default=0.35)
    parser.add_argument("--commands", type=str, default=None, help="Semicolon-separated vx,vy,vyaw triples.")
    args = parser.parse_args()
    commands = _commands_from_arg(args.commands)
    rows = run_probe(args, commands)
    final_cycle = max(0, int(args.cycles) - 1)
    final_rows = [row for row in rows if int(row["cycle"]) == final_cycle]
    ok = len(final_rows) == len(commands) and all(
        bool(row["valid_env"])
        and not bool(row["standstill"])
        and min(int(v) for v in row["per_leg_valid"]) > 0
        and int(row["root_crosses_small"]) == 1
        for row in final_rows
    )
    print(
        json.dumps(
            {
                "type": "summary",
                "ok": bool(ok),
                "final_cycle": int(final_cycle),
                "final_rows": int(len(final_rows)),
                "command_count": int(len(commands)),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
