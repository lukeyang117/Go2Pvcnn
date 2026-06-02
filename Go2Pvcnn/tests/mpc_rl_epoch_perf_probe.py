from __future__ import annotations

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


def main() -> None:
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
        cfg.scene.num_envs = 1024
        cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size = 64
        cfg.mpc_planner_cfg.runtime.horizon_steps = 25
        cfg.mpc_planner_cfg.runtime.replan_interval_steps = 25
        cfg.mpc_planner_cfg.diagnostics.emit_runtime_counters = False
        cfg.mpc_planner_cfg.diagnostics.profile_cuda_sync = False
        env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
        root = env.unwrapped
        attach_trajectory_manager_if_enabled(
            root,
            cfg,
            experiment_name="teacher_elevation_trajectory_mpc_semantic",
            device=root.device,
        )
        env.reset()
        action_shape = env.action_space.shape
        action = torch.zeros(action_shape, dtype=torch.float32, device=root.device)
        start = time.perf_counter()
        for _ in range(25):
            env.step(action)
        elapsed = time.perf_counter() - start
        print(json.dumps({"num_envs": 1024, "selected_mpc_envs": 64, "epoch_seconds": elapsed}), flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
