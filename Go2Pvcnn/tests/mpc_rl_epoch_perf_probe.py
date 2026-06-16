from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--mpc-num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--require-replan", action="store_true")
    parser.add_argument("--print-cuda-memory", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--optimize-steps", type=int, default=None)
    return parser.parse_args()


def _write_summary(path: Path | None, output: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def _cuda_memory(torch_module) -> dict[str, int]:
    if not torch_module.cuda.is_available():
        return {}
    torch_module.cuda.synchronize()
    return {
        "cuda_max_memory_allocated": int(torch_module.cuda.max_memory_allocated()),
        "cuda_max_memory_reserved": int(torch_module.cuda.max_memory_reserved()),
    }


def _manager_counters(manager) -> dict:
    if manager is not None and hasattr(manager, "runtime_counters"):
        return manager.runtime_counters()
    return {}


def _base_output(args: argparse.Namespace, cfg, *, phase: str, completed_steps: int) -> dict:
    return {
        "phase": phase,
        "num_envs": int(args.num_envs),
        "mpc_num_envs": int(args.mpc_num_envs),
        "parallel_plan_batch_size": int(cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size),
        "horizon_steps": int(cfg.mpc_planner_cfg.runtime.horizon_steps),
        "replan_interval_steps": int(cfg.mpc_planner_cfg.runtime.replan_interval_steps),
        "steps": int(args.steps),
        "completed_steps": int(completed_steps),
    }


def main() -> None:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch
        import go2_pvcnn.tasks  # noqa: F401
        from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled
        from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
            TeacherElevationTrajectoryMpcSemanticEnvCfg,
        )

        cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
        cfg.scene.num_envs = int(args.num_envs)
        cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size = int(args.mpc_num_envs)
        cfg.mpc_planner_cfg.runtime.horizon_steps = 25
        cfg.mpc_planner_cfg.runtime.replan_interval_steps = 25
        if args.optimize_steps is not None:
            cfg.mpc_planner_cfg.runtime.optimize_steps = int(args.optimize_steps)
        cfg.mpc_planner_cfg.diagnostics.emit_runtime_counters = True
        cfg.mpc_planner_cfg.diagnostics.profile_cuda_sync = False
        env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
        root = env.unwrapped
        manager = attach_trajectory_manager_if_enabled(
            root,
            cfg,
            experiment_name="teacher_elevation_trajectory_mpc_semantic",
            device=root.device,
        )
        env.reset()
        reset_output = _base_output(args, cfg, phase="after_reset", completed_steps=0)
        reset_output["runtime_counters"] = _manager_counters(manager)
        if bool(args.print_cuda_memory) and torch.cuda.is_available():
            reset_output.update(_cuda_memory(torch))
        _write_summary(args.summary_path, reset_output)
        action_shape = env.action_space.shape
        action = torch.zeros(action_shape, dtype=torch.float32, device=root.device)
        start = time.perf_counter()
        completed_steps = 0
        max_sampled_plan_count_seen = int(reset_output["runtime_counters"].get("sampled_plan_count", 0) or 0)
        replan_event_count = 1 if max_sampled_plan_count_seen > 0 else 0
        try:
            target_steps = int(args.steps)
            for step_index in range(target_steps):
                env.step(action)
                completed_steps = step_index + 1
                counters = _manager_counters(manager)
                sampled_plan_count = int(counters.get("sampled_plan_count", 0) or 0)
                max_sampled_plan_count_seen = max(max_sampled_plan_count_seen, sampled_plan_count)
                if sampled_plan_count > 0:
                    replan_event_count += 1
                phase = "complete" if completed_steps >= target_steps else "stepping"
                progress_output = _base_output(args, cfg, phase=phase, completed_steps=completed_steps)
                progress_output["epoch_seconds"] = time.perf_counter() - start
                progress_output["runtime_counters"] = counters
                progress_output["max_sampled_plan_count_seen"] = int(max_sampled_plan_count_seen)
                progress_output["replan_event_count"] = int(replan_event_count)
                if bool(args.print_cuda_memory) and torch.cuda.is_available():
                    progress_output.update(_cuda_memory(torch))
                _write_summary(args.summary_path, progress_output)
        except BaseException as exc:
            error_output = _base_output(args, cfg, phase="error", completed_steps=completed_steps)
            error_output["error_type"] = type(exc).__name__
            error_output["error"] = str(exc)
            error_output["runtime_counters"] = _manager_counters(manager)
            error_output["max_sampled_plan_count_seen"] = int(max_sampled_plan_count_seen)
            error_output["replan_event_count"] = int(replan_event_count)
            if bool(args.print_cuda_memory) and torch.cuda.is_available():
                error_output.update(_cuda_memory(torch))
            _write_summary(args.summary_path, error_output)
            raise
        elapsed = time.perf_counter() - start
        counters = _manager_counters(manager)
        sampled_plan_count = int(counters.get("sampled_plan_count", 0) or 0)
        max_sampled_plan_count_seen = max(max_sampled_plan_count_seen, sampled_plan_count)
        if bool(args.require_replan) and max_sampled_plan_count_seen <= 0:
            raise RuntimeError(f"required at least one MPC replan, got counters={counters}")
        output = _base_output(args, cfg, phase="complete", completed_steps=completed_steps)
        output["epoch_seconds"] = elapsed
        output["runtime_counters"] = counters
        output["max_sampled_plan_count_seen"] = int(max_sampled_plan_count_seen)
        output["replan_event_count"] = int(replan_event_count)
        if bool(args.print_cuda_memory) and torch.cuda.is_available():
            output.update(_cuda_memory(torch))
        _write_summary(args.summary_path, output)
        print(json.dumps(output), flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
