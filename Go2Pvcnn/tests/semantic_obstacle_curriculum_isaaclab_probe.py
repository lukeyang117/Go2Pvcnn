from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


def _semantic_leaf_paths(root: str) -> list[str]:
    import isaaclab.sim as sim_utils
    from go2_pvcnn.sensor.semantic_contacter.semantic_global_contact_sensor import filter_semantic_leaf_obstacle_paths

    return filter_semantic_leaf_obstacle_paths(sim_utils.find_matching_prim_paths(f"{root}/.*/.*/.*"), root)


def _counts_by_row_col(paths: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in paths:
        parts = path.split("/")
        row = next(part for part in parts if part.startswith("row_"))
        col = next(part for part in parts if part.startswith("col_"))
        key = f"{row}/{col}"
        out[key] = out.get(key, 0) + 1
    return out


def _terrain_names(cfg) -> tuple[str, ...]:
    sub = cfg.scene.terrain.terrain_generator.sub_terrains
    return tuple(str(name) for name in sub.keys())


def run_probe(
    *,
    num_envs: int,
    row: int,
    force_low_collision_steps: int,
    output_json: Path | None = None,
    trace_json: Path | None = None,
) -> dict[str, object]:
    import gymnasium as gym
    import go2_pvcnn.tasks  # noqa: F401
    from extension.semantic_course import SEMANTIC_COURSE_LARGE_ROOT, SEMANTIC_COURSE_SMALL_ROOT
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        SEMANTIC_CONTACT_BODY_NAMES,
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
    )

    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size = min(int(num_envs), 64)
    trace: list[str] = []

    def _trace(label: str) -> None:
        trace.append(label)
        if trace_json is not None:
            trace_json.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")

    env = None
    try:
        _trace("before_gym_make")
        env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
        _trace("after_gym_make")
        root = env.unwrapped
        env.reset()
        _trace("after_env_reset")

        terrain_names = _terrain_names(cfg)
        flat_col = terrain_names.index("flat")
        curriculum_cfg = cfg.semantic_obstacle_curriculum
        row_idx = max(0, min(int(row), len(curriculum_cfg.plane_counts) - 1))
        nonflat_row_idx = max(0, min(int(row), len(curriculum_cfg.non_plane_counts) - 1))
        flat_key = f"row_{row_idx:02d}/col_{flat_col:02d}"
        nonflat_col = 1 if flat_col == 0 else 0
        nonflat_key = f"row_{nonflat_row_idx:02d}/col_{nonflat_col:02d}"

        curriculum_outputs = []
        for _ in range(int(force_low_collision_steps)):
            _trace("before_curriculum_compute")
            out = root.curriculum_manager.compute()
            _trace("after_curriculum_compute")
            # IsaacLab compute stores state internally and returns None.
            state = getattr(root, "_semantic_obstacle_curriculum_state", None)
            episode_collision_flags = None
            if state is not None and state.episode_had_small_collision is not None:
                episode_collision_flags = int(state.episode_had_small_collision.sum().item())
            curriculum_outputs.append(
                {
                    "episode_had_small_collision_count": episode_collision_flags,
                    "has_runtime_level": hasattr(root, "_semantic_obstacle_curriculum_level")
                    or hasattr(root.scene.terrain.cfg, "semantic_obstacle_curriculum_level"),
                }
            )
        state = getattr(root, "_semantic_obstacle_curriculum_state", None)
        small_paths = _semantic_leaf_paths(SEMANTIC_COURSE_SMALL_ROOT)
        _trace("after_small_paths")
        large_paths = _semantic_leaf_paths(SEMANTIC_COURSE_LARGE_ROOT)
        _trace("after_large_paths")
        small_counts = _counts_by_row_col(small_paths)
        large_counts = _counts_by_row_col(large_paths)

        small = root.scene.sensors["semantic_contact_small"]
        large = root.scene.sensors["semantic_contact_large"]
        body_count = len(SEMANTIC_CONTACT_BODY_NAMES)
        small_matrix = small.data.force_matrix_w
        large_matrix = large.data.force_matrix_w

        result = {
            "num_envs": int(num_envs),
            "terrain_names": terrain_names,
            "row": int(row),
            "flat_col": flat_col,
            "flat_key": flat_key,
            "nonflat_key": nonflat_key,
            "expected_flat_small": int(curriculum_cfg.plane_counts[row_idx].small),
            "expected_flat_large": int(curriculum_cfg.plane_counts[row_idx].large),
            "expected_nonflat_small": int(curriculum_cfg.non_plane_counts[nonflat_row_idx].small),
            "expected_nonflat_large": int(curriculum_cfg.non_plane_counts[nonflat_row_idx].large),
            "actual_flat_small": int(small_counts.get(flat_key, 0)),
            "actual_flat_large": int(large_counts.get(flat_key, 0)),
            "actual_nonflat_small": int(small_counts.get(nonflat_key, 0)),
            "actual_nonflat_large": int(large_counts.get(nonflat_key, 0)),
            "total_small": len(small_paths),
            "total_large": len(large_paths),
            "small_force_shape": tuple(int(v) for v in small_matrix.shape),
            "large_force_shape": tuple(int(v) for v in large_matrix.shape),
            "expected_small_shape": (int(num_envs), body_count, len(small_paths), 3),
            "expected_large_shape": (int(num_envs), body_count, len(large_paths), 3),
            "small_finite": bool(torch.isfinite(small_matrix).all().item()),
            "large_finite": bool(torch.isfinite(large_matrix).all().item()),
            "small_has_filters": bool(small.has_semantic_filters),
            "large_has_filters": bool(large.has_semantic_filters),
            "small_filter_count": int(small.contact_physx_view.filter_count) if small.has_semantic_filters else 0,
            "large_filter_count": int(large.contact_physx_view.filter_count) if large.has_semantic_filters else 0,
            "has_runtime_semantic_level": hasattr(root, "_semantic_obstacle_curriculum_level")
            or hasattr(root.scene.terrain.cfg, "semantic_obstacle_curriculum_level"),
            "curriculum_outputs": curriculum_outputs,
        }
        if output_json is not None:
            output_json.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            _trace("after_output_json")
        return result
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--row", type=int, default=9)
    parser.add_argument("--force-low-collision-steps", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--trace-json", type=Path, default=None)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        result = run_probe(
            num_envs=args.num_envs,
            row=args.row,
            force_low_collision_steps=args.force_low_collision_steps,
            output_json=args.output_json,
            trace_json=args.trace_json,
        )
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
        print(payload, flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
