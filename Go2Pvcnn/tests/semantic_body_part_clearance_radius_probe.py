from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


def _cuda_sync_if_needed(device: str | torch.device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _reward_params_for_radius(base_params: dict[str, Any], *, radius: float, margin_scale: float) -> dict[str, Any]:
    params = dict(base_params)
    params["foot_query_radius_m"] = float(radius)
    params["calf_query_radius_m"] = float(radius)
    params["thigh_query_radius_m"] = float(radius)
    params["base_query_radius_m"] = float(radius)
    params["foot_margin_m"] = float(base_params["foot_margin_m"]) * float(margin_scale)
    params["calf_margin_m"] = float(base_params["calf_margin_m"]) * float(margin_scale)
    params["thigh_margin_m"] = float(base_params["thigh_margin_m"]) * float(margin_scale)
    params["base_margin_m"] = float(base_params["base_margin_m"]) * float(margin_scale)
    return params


def _summarize_reward(reward: torch.Tensor, *, elapsed_s: float, radius: float, step: int) -> dict[str, Any]:
    rew = torch.as_tensor(reward, dtype=torch.float32).detach()
    nonzero = rew < 0.0
    return {
        "step": int(step),
        "radius_m": float(radius),
        "elapsed_ms": float(elapsed_s * 1000.0),
        "nonzero_env_count": int(nonzero.sum().item()),
        "nonzero_env_rate": float(nonzero.to(dtype=torch.float32).mean().item()),
        "mean_reward": float(rew.mean().item()),
        "min_reward": float(rew.min().item()),
        "max_reward": float(rew.max().item()),
    }


def _scanner_small_cell_count(env) -> int:
    scanner = env.scene["semantic_height_scanner"]
    semantic_map = torch.as_tensor(scanner.data.semantic_map)
    return int((semantic_map == 1).sum().item())


def _diagnose_body_query_hits(env, params: dict[str, Any]) -> dict[str, Any]:
    from extension.batch_mpc_planner.terrain import TerrainQueryCache, height_at, semantic_at
    from extension.mdp.semantic_body_part_clearance import (
        _body_geometry_query_points,
        _current_body_part_sample_points,
        _current_scanner_terrain,
        _semantic_id_mask,
    )

    robot = env.scene["robot"]
    scanner = env.scene["semantic_height_scanner"]
    root = getattr(env, "unwrapped", env)
    body_ids = getattr(root, "_semantic_body_part_clearance_body_ids", None)
    if body_ids is None:
        foot_ids, _ = robot.find_bodies(".*_foot")
        calf_ids, _ = robot.find_bodies(".*_calf")
        thigh_ids, _ = robot.find_bodies(".*_thigh")
        body_ids = {"foot": foot_ids, "calf": calf_ids, "thigh": thigh_ids}
        root._semantic_body_part_clearance_body_ids = body_ids

    terrain = _current_scanner_terrain(scanner, device=torch.as_tensor(robot.data.body_pos_w).device)
    points = _current_body_part_sample_points(
        robot,
        body_ids=body_ids,
        calf_sections=int(params["calf_sections"]),
        thigh_sections=int(params["thigh_sections"]),
    )
    cache = TerrainQueryCache()
    groups = {
        "foot": (
            points["foot"].reshape(env.num_envs, -1, 3),
            float(params["foot_sphere_radius_m"]),
            float(params["foot_query_radius_m"]),
            float(params["foot_margin_m"]),
        ),
        "calf": (
            points["calf"].reshape(env.num_envs, -1, 3),
            float(params["calf_capsule_radius_m"]),
            float(params["calf_query_radius_m"]),
            float(params["calf_margin_m"]),
        ),
        "thigh": (
            points["thigh"].reshape(env.num_envs, -1, 3),
            float(params["thigh_capsule_radius_m"]),
            float(params["thigh_query_radius_m"]),
            float(params["thigh_margin_m"]),
        ),
    }
    out: dict[str, Any] = {}
    total_small_hits = 0
    total_small_positive = 0
    total_queries = 0
    for name, (centers, body_radius, query_radius, margin) in groups.items():
        surface_z = centers[..., 2] - body_radius
        query_xy, query_surface_z = _body_geometry_query_points(
            centers=centers,
            surface_z=surface_z,
            query_radius_m=query_radius,
            terrain=terrain,
        )
        terrain_z = height_at(terrain, query_xy, cache=cache)
        semantic_id = semantic_at(terrain, query_xy, cache=cache)
        small_mask = _semantic_id_mask(semantic_id.to(dtype=torch.long), params["small_semantic_ids"])
        deficit = torch.relu(terrain_z + margin - query_surface_z)
        small_positive = small_mask & (deficit > 0.0)
        total_small_hits += int(small_mask.sum().item())
        total_small_positive += int(small_positive.sum().item())
        total_queries += int(query_xy.numel() // 2)
        out[f"{name}_query_count"] = int(query_xy.numel() // 2)
        out[f"{name}_small_hit_count"] = int(small_mask.sum().item())
        out[f"{name}_small_positive_deficit_count"] = int(small_positive.sum().item())
        out[f"{name}_max_deficit_m"] = float(deficit.max().item())
    out["body_query_count"] = total_queries
    out["body_small_hit_count"] = total_small_hits
    out["body_small_positive_deficit_count"] = total_small_positive
    return out


def _append_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_probe(
    *,
    num_envs: int,
    steps: int,
    radii: tuple[float, ...],
    margin_scale: float,
    output_jsonl: Path | None,
) -> dict[str, Any]:
    import gymnasium as gym
    import go2_pvcnn.tasks  # noqa: F401
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
    )

    cfg = TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size = min(int(num_envs), 64)
    cfg.planner_owned_reference_cache = False
    cfg.use_batched_reference_trajectory = False
    cfg.rewards.reference_foot_pos = None
    term = cfg.rewards.semantic_body_part_clearance
    if term is None:
        raise RuntimeError("flat-small cfg does not mount semantic_body_part_clearance")

    env = None
    rows: list[dict[str, Any]] = []
    try:
        env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-v0", cfg=cfg)
        root = env.unwrapped
        env.reset()
        actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=root.device)

        scanner = root.scene["semantic_height_scanner"]
        semantic_map = torch.as_tensor(scanner.data.semantic_map)
        result_header = {
            "type": "header",
            "num_envs": int(num_envs),
            "steps": int(steps),
            "radii": [float(v) for v in radii],
            "margin_scale": float(margin_scale),
            "scanner_shape": [int(v) for v in semantic_map.shape],
            "small_cell_count": int((semantic_map == 1).sum().item()),
        }
        rows.append(result_header)
        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            output_jsonl.write_text("", encoding="utf-8")
        _append_jsonl(output_jsonl, result_header)

        for step in range(int(steps)):
            _append_jsonl(output_jsonl, {"type": "trace", "event": "before_step", "step": int(step)})
            env.step(actions)
            _append_jsonl(
                output_jsonl,
                {
                    "type": "trace",
                    "event": "after_step",
                    "step": int(step),
                    "small_cell_count": _scanner_small_cell_count(root),
                },
            )
            for radius in radii:
                params = _reward_params_for_radius(term.params, radius=float(radius), margin_scale=float(margin_scale))
                _append_jsonl(
                    output_jsonl,
                    {"type": "trace", "event": "before_reward", "step": int(step), "radius_m": float(radius)},
                )
                diagnostics = _diagnose_body_query_hits(root, params)
                _cuda_sync_if_needed(root.device)
                start = time.perf_counter()
                reward = term.func(root, **params)
                _cuda_sync_if_needed(root.device)
                row = _summarize_reward(reward, elapsed_s=time.perf_counter() - start, radius=float(radius), step=step)
                row.update(diagnostics)
                rows.append(row)
                _append_jsonl(output_jsonl, row)

        data_rows = [row for row in rows if row.get("type") != "header"]
        summary = {
            "type": "summary",
            "num_envs": int(num_envs),
            "steps": int(steps),
            "radii": [float(v) for v in radii],
            "max_nonzero_env_count": max((int(row["nonzero_env_count"]) for row in data_rows), default=0),
            "max_nonzero_env_rate": max((float(row["nonzero_env_rate"]) for row in data_rows), default=0.0),
            "min_reward": min((float(row["min_reward"]) for row in data_rows), default=0.0),
            "mean_elapsed_ms": sum(float(row["elapsed_ms"]) for row in data_rows) / max(len(data_rows), 1),
        }
        rows.append(summary)
        _append_jsonl(output_jsonl, summary)
        return summary
    except BaseException as exc:
        _append_jsonl(
            output_jsonl,
            {
                "type": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--radii", type=str, default="0.04,0.08,0.12,0.16")
    parser.add_argument("--margin-scale", type=float, default=1.0)
    parser.add_argument("--output-jsonl", type=Path, default=Path("Go2Pvcnn/tests/artifacts/semantic_body_part_clearance_radius_probe.jsonl"))
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        radii = tuple(float(item) for item in args.radii.split(",") if item.strip())
        summary = run_probe(
            num_envs=args.num_envs,
            steps=args.steps,
            radii=radii,
            margin_scale=args.margin_scale,
            output_jsonl=args.output_jsonl,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
