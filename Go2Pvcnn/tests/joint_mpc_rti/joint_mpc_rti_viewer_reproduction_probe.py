from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from Go2Pvcnn.tests.fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "max": float(tensor.max().item()),
    }


def _ground_gap(terrain, foot_pos_w: torch.Tensor, contact: torch.Tensor) -> tuple[float, float]:
    from extension.batch_mpc_planner.terrain import height_at

    terrain_z = height_at(terrain, foot_pos_w[..., :2]).to(foot_pos_w)
    gap = torch.abs(foot_pos_w[..., 2] - terrain_z - 0.022)
    stance = gap[contact]
    swing = gap[~contact]
    return (
        float(stance.max().item()) if stance.numel() else 0.0,
        float(swing.max().item()) if swing.numel() else 0.0,
    )


def _run_case(runtime, *, name: str, command: tuple[float, float, float], cycles: int) -> dict[str, object]:
    from extension.joint_mpc_rti.integration.isaaclab_adapter import state_from_env
    from extension.batch_mpc_planner.terrain import height_at

    viewer = runtime._viewer
    base_env = runtime.base_env
    robot = runtime.robot
    manager = base_env._trajectory_manager
    terrain = runtime._single_env_terrain()
    command_tensor = torch.tensor(command, dtype=torch.float32, device=base_env.device).view(1, 3)

    runtime.reset()
    manager._solver_state = None
    manager._last_result = None
    manager._graph_runner = None
    previous_joint = viewer._joint_pos_robot_to_planner(robot, robot.data.joint_pos[:1]).clone()
    previous_foot = robot.data.body_pos_w[:1].index_select(1, runtime.foot_ids).clone()
    initial_state = state_from_env(base_env, device=base_env.device)
    previous_contact = None

    adapter_order_error: list[float] = []
    adapter_velocity_order_error: list[float] = []
    joint_step_max: list[float] = []
    foot_step_max: list[float] = []
    stance_xy_step_max: list[float] = []
    stance_gap_max: list[float] = []
    swing_gap_max: list[float] = []
    actual_plan_joint_error: list[float] = []
    actual_plan_foot_error: list[float] = []
    cycle_rows: list[dict[str, object]] = []

    for cycle in range(cycles):
        raw_state = state_from_env(base_env, device=base_env.device)
        reordered_joint = viewer._joint_pos_robot_to_planner(robot, robot.data.joint_pos[:1])
        reordered_velocity = viewer._joint_pos_robot_to_planner(robot, robot.data.joint_vel[:1])
        adapter_order_error.append(float(torch.abs(raw_state.joint_pos - reordered_joint).max().item()))
        adapter_velocity_order_error.append(float(torch.abs(raw_state.joint_vel - reordered_velocity).max().item()))

        result = viewer._plan_joint_viewer_trajectory(
            manager=manager,
            env=base_env,
            command=command_tensor,
        )
        viewer._viewer_direct_playback_step(base_env, result, frame_idx=1)

        actual_joint = viewer._joint_pos_robot_to_planner(robot, robot.data.joint_pos[:1]).clone()
        actual_foot = robot.data.body_pos_w[:1].index_select(1, runtime.foot_ids).clone()
        planned_joint = result.joint_angles[:, 1]
        planned_foot = result.foot_pos_w[:, 1]
        contact = result.contact_state[:, 1].to(dtype=torch.bool)

        joint_step_max.append(float(torch.abs(actual_joint - previous_joint).max().item()))
        foot_step_max.append(
            float(torch.linalg.vector_norm(actual_foot - previous_foot, dim=-1).max().item())
        )
        stance_gap, swing_gap = _ground_gap(terrain, actual_foot, contact)
        from extension.joint_mpc_rti.terrain.query import query_world

        field_gap = actual_foot[..., 2] - query_world(manager._field_sync.latest_field(), actual_foot).height_w
        stance_gap_max.append(stance_gap)
        swing_gap_max.append(swing_gap)
        actual_plan_joint_error.append(float(torch.abs(actual_joint - planned_joint).max().item()))
        actual_plan_foot_error.append(
            float(torch.linalg.vector_norm(actual_foot - planned_foot, dim=-1).max().item())
        )
        if previous_contact is not None:
            consecutive_stance = torch.logical_and(contact, previous_contact)
            xy_step = torch.linalg.vector_norm(actual_foot[..., :2] - previous_foot[..., :2], dim=-1)
            stance_xy_step_max.append(
                float(torch.where(consecutive_stance, xy_step, torch.zeros_like(xy_step)).max().item())
            )
        previous_joint = actual_joint
        previous_foot = actual_foot
        previous_contact = contact
        cycle_rows.append(
            {
                "cycle": cycle,
                "contact": contact[0].tolist(),
                "viewer_ground_gap_m": [float(value) for value in (
                    actual_foot[..., 2] - height_at(terrain, actual_foot[..., :2])
                )[0].tolist()],
                "planner_field_gap_m": [float(value) for value in field_gap[0].tolist()],
            }
        )

    final_state = state_from_env(base_env, device=base_env.device)
    root_xy_drift = torch.linalg.vector_norm(final_state.root_pos_w[:, :2] - initial_state.root_pos_w[:, :2], dim=-1)
    root_yaw_drift = torch.abs(final_state.root_rpy_w[:, 2] - initial_state.root_rpy_w[:, 2])

    return {
        "name": name,
        "command": command,
        "cycles": cycles,
        "adapter_joint_order_error_rad": _summary(adapter_order_error),
        "adapter_joint_velocity_order_error_rad_s": _summary(adapter_velocity_order_error),
        "joint_step_max_rad": _summary(joint_step_max),
        "foot_step_max_m": _summary(foot_step_max),
        "stance_xy_step_max_m": _summary(stance_xy_step_max or [0.0]),
        "stance_ground_gap_max_m": _summary(stance_gap_max),
        "swing_ground_gap_max_m": _summary(swing_gap_max),
        "actual_vs_planner_joint_max_rad": _summary(actual_plan_joint_error),
        "actual_vs_planner_foot_max_m": _summary(actual_plan_foot_error),
        "root_xy_drift_m": float(root_xy_drift.max().item()),
        "root_yaw_drift_rad": float(root_yaw_drift.max().item()),
        "initial_root_pos_w": [float(value) for value in initial_state.root_pos_w[0].tolist()],
        "initial_joint_pos": [float(value) for value in initial_state.joint_pos[0].tolist()],
        "cycle_rows": cycle_rows,
    }


def main() -> int:
    output_path = Path(os.environ.get("JOINT_MPC_VIEWER_REPRO_OUTPUT", "/tmp/joint_mpc_viewer_repro.json"))
    device = os.environ.get("MPC_TEST_DEVICE", "cuda:0")
    cycles = int(os.environ.get("JOINT_MPC_VIEWER_REPRO_CYCLES", "16"))
    runtime = None
    try:
        runtime = RealViewerRuntimeFixture(
            num_envs=1,
            planner_backend="joint_mpc_rti",
            requested_n_frames=16,
            device=device,
            task_id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
        )
        case_specs = (
            ("standstill", (0.0, 0.0, 0.0)),
            ("forward_slow", (0.1, 0.0, 0.0)),
            ("forward_fast", (0.4, 0.0, 0.0)),
            ("backward", (-0.25, 0.0, 0.0)),
            ("lateral_left", (0.0, 0.25, 0.0)),
            ("lateral_right", (0.0, -0.25, 0.0)),
            ("yaw_left", (0.0, 0.0, 0.5)),
            ("mixed", (0.2, 0.15, 0.3)),
            ("mixed_reverse", (0.35, -0.2, -0.35)),
        )
        requested = {
            name.strip()
            for name in os.environ.get("JOINT_MPC_VIEWER_REPRO_CASES", "").split(",")
            if name.strip()
        }
        selected_specs = case_specs if not requested else tuple(spec for spec in case_specs if spec[0] in requested)
        cases = [_run_case(runtime, name=name, command=command, cycles=cycles) for name, command in selected_specs]
        standstill = next((case for case in cases if case["name"] == "standstill"), None)
        acceptance = {
            "adapter_joint_order_error_rad": max(
                float(case["adapter_joint_order_error_rad"]["max"]) for case in cases
            ) <= 1.0e-6,
            "stance_ground_gap_max_m": max(
                float(case["stance_ground_gap_max_m"]["max"]) for case in cases
            ) <= 0.012,
            "joint_step_max_rad": max(float(case["joint_step_max_rad"]["max"]) for case in cases) <= 0.35,
            "viewer_plan_foot_error_m": max(
                float(case["actual_vs_planner_foot_max_m"]["max"]) for case in cases
            ) <= 1.0e-4,
            "standstill_root_xy_drift_m": standstill is None or float(standstill["root_xy_drift_m"]) <= 1.0e-5,
            "standstill_root_yaw_drift_rad": standstill is None or float(standstill["root_yaw_drift_rad"]) <= 1.0e-5,
        }
        result = {
            "device": device,
            "robot_joint_names": tuple(runtime.robot.joint_names),
            "planner_joint_names": tuple(runtime._viewer.PLANNER_JOINT_ORDER),
            "cases": cases,
            "acceptance": acceptance,
            "passed": all(acceptance.values()),
        }
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 2
    except BaseException as exc:  # noqa: BLE001
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_path.write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 1
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
