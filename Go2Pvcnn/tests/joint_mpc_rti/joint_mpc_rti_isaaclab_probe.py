from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real IsaacLab smoke probe for joint MPC RTI.")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def run_probe(*, num_envs: int, steps: int) -> dict[str, object]:
    import gymnasium as gym
    import torch

    import go2_pvcnn.tasks  # noqa: F401
    from extension.joint_mpc_rti.integration.isaaclab_adapter import state_from_env
    from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
    )

    cfg = TeacherElevationTrajectoryMpcSemanticEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.planner_backend = "joint_mpc_rti"
    cfg.joint_mpc_rti_cfg.runtime.horizon_steps = 16
    cfg.joint_mpc_rti_cfg.runtime.dt = 0.02
    cfg.rewards.reference_foot_pos = None
    env = None
    try:
        env = gym.make("Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0", cfg=cfg)
        root = env.unwrapped
        manager = attach_trajectory_manager_if_enabled(
            root,
            cfg,
            experiment_name="teacher_elevation_trajectory_mpc_semantic",
            device=root.device,
        )
        if manager is None:
            raise RuntimeError("joint MPC RTI trajectory manager was not attached")
        env.reset()
        action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=root.device)
        planner_ms: list[float] = []
        x0_error_max = 0.0
        reference_finite = True
        target_step = -1
        completed_steps = 0
        for step_index in range(int(steps)):
            env.step(action)
            measured = state_from_env(root, device=root.device)
            if torch.cuda.is_available() and torch.device(root.device).type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                manager.refresh_from_env(root, force=True)
                end_event.record()
                end_event.synchronize()
                planner_ms.append(float(start_event.elapsed_time(end_event)))
            else:
                start = time.perf_counter()
                manager.refresh_from_env(root, force=True)
                planner_ms.append(1000.0 * (time.perf_counter() - start))
            trajectory = manager.latest_trajectory()
            target_step = int(manager._buffer.reference.target_step)
            x0_error = torch.abs(trajectory.state[:, 0] - measured.as_vector()).amax()
            x0_error_max = max(x0_error_max, float(x0_error.item()))
            reference_finite = reference_finite and bool(
                torch.isfinite(trajectory.state).all().item()
                and torch.isfinite(trajectory.control).all().item()
                and torch.isfinite(trajectory.foot_pos_w).all().item()
            )
            completed_steps = step_index + 1
        if manager._field_sync is None:
            raise RuntimeError("RayCaster field sync was not initialized")
        field = manager._field_sync.latest_field()
        output = {
            "num_envs": int(num_envs),
            "steps": int(steps),
            "completed_steps": int(completed_steps),
            "planner_backend": manager.planner_backend,
            "target_step": int(target_step),
            "field_version_min": int(field.version.min().item()),
            "field_version_max": int(field.version.max().item()),
            "field_ready_count": int(manager._field_sync.ready.sum().item()),
            "reference_finite": bool(reference_finite),
            "x0_error_max": float(x0_error_max),
            "planner_ms_mean": float(sum(planner_ms) / max(len(planner_ms), 1)),
            "planner_ms_max": float(max(planner_ms, default=0.0)),
            "planner_ms_first": float(planner_ms[0] if planner_ms else 0.0),
            "planner_ms_last": float(planner_ms[-1] if planner_ms else 0.0),
        }
        if output["target_step"] != 1:
            raise RuntimeError(f"expected pending reference target_step=1, got {output['target_step']}")
        if not output["reference_finite"]:
            raise RuntimeError("joint MPC RTI emitted a non-finite reference")
        if output["field_ready_count"] != int(num_envs):
            raise RuntimeError(f"not all field rows are ready: {output}")
        if output["x0_error_max"] > 1.0e-6:
            raise RuntimeError(f"planner x0 does not match measured state: {output}")
        return output
    finally:
        if env is not None:
            env.close()


def main() -> None:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(args.headless))
    simulation_app = app_launcher.app
    try:
        output = run_probe(num_envs=args.num_envs, steps=args.steps)
        payload = json.dumps(output, ensure_ascii=False, sort_keys=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(payload, flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
