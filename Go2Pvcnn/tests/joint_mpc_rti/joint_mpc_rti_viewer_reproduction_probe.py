from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict, replace
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


def _root_tracking_solver_layers(
    *,
    state: torch.Tensor,
    step_diagnostics_rows: list[object],
    command_body: torch.Tensor,
    line_alpha: torch.Tensor,
    warm_start: torch.Tensor,
    dt: float,
    root_position_trust: float,
) -> dict[str, object]:
    actual = torch.as_tensor(state)
    nominal = torch.stack([row.nominal_state for row in step_diagnostics_rows], dim=1)
    direction = torch.stack([row.qp_direction for row in step_diagnostics_rows], dim=1)
    actual_edges = torch.stack((actual[:, :-1], actual[:, 1:]), dim=2)
    full_qp = nominal + direction
    selected = nominal + torch.as_tensor(line_alpha, dtype=actual.dtype, device=actual.device)[
        ..., None, None
    ] * direction

    def body_velocity(edges: torch.Tensor) -> torch.Tensor:
        delta = (edges[:, :, 1, :2] - edges[:, :, 0, :2]) / float(dt)
        yaw = edges[:, :, 0, 5]
        return torch.stack(
            (
                torch.cos(yaw) * delta[..., 0] + torch.sin(yaw) * delta[..., 1],
                -torch.sin(yaw) * delta[..., 0] + torch.cos(yaw) * delta[..., 1],
            ),
            dim=-1,
        )

    velocities = {
        "actual": body_velocity(actual_edges),
        "nominal": body_velocity(nominal),
        "full_qp": body_velocity(full_qp),
        "selected": body_velocity(selected),
    }
    command_xy = torch.as_tensor(command_body, dtype=actual.dtype, device=actual.device)[
        :, None, :2
    ]
    errors = {
        name: torch.linalg.vector_norm(velocity - command_xy, dim=-1)
        for name, velocity in velocities.items()
    }
    full_root_xy_deviation = torch.linalg.vector_norm(
        full_qp[:, :, 1, :2] - nominal[:, :, 1, :2], dim=-1
    )
    selected_root_xy_deviation = torch.linalg.vector_norm(
        selected[:, :, 1, :2] - nominal[:, :, 1, :2], dim=-1
    )
    warm = torch.as_tensor(warm_start, dtype=torch.bool, device=actual.device)
    trust = direction[:, :, 1, :2].abs().amax(dim=-1) / float(root_position_trust)

    def warm_mean(value: torch.Tensor) -> float:
        selected_value = value[warm]
        return float(selected_value.mean().item()) if selected_value.numel() else float("nan")

    cycles = []
    for cycle in range(int(actual.shape[1]) - 1):
        cycles.append(
            {
                "cycle": cycle,
                "warm_start": bool(warm[0, cycle].item()),
                "line_alpha": float(line_alpha[0, cycle].item()),
                "root_xy_trust_utilization": float(trust[0, cycle].item()),
                "published_root_xy_deviation_m": {
                    "full_qp": float(full_root_xy_deviation[0, cycle].item()),
                    "selected": float(selected_root_xy_deviation[0, cycle].item()),
                },
                "velocity_body_mps": {
                    name: [float(value) for value in velocity[0, cycle].tolist()]
                    for name, velocity in velocities.items()
                },
                "error_mps": {
                    name: float(value[0, cycle].item()) for name, value in errors.items()
                },
            }
        )
    return {
        "mean_error_mps": {
            name: float(value.mean().item()) for name, value in errors.items()
        },
        "warm_mean_error_mps": {
            name: warm_mean(value) for name, value in errors.items()
        },
        "root_xy_trust_utilization": {
            "mean": float(trust.mean().item()),
            "max": float(trust.max().item()),
            "saturated_cycle_count": int((trust >= 1.0 - 1.0e-5).sum().item()),
        },
        "published_root_xy_deviation_m": {
            "full_qp_max": float(full_root_xy_deviation.max().item()),
            "selected_max": float(selected_root_xy_deviation.max().item()),
        },
        "published_root_xy_violation_count": {
            "full_qp": int((full_root_xy_deviation > 1.0e-7).sum().item()),
            "selected": int((selected_root_xy_deviation > 1.0e-7).sum().item()),
        },
        "cycles": cycles,
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


def _ground_initial_stance(runtime) -> None:
    viewer = runtime._viewer
    stance_leg_indices = torch.tensor((1, 2), dtype=torch.long, device=runtime.base_env.device)
    stance_foot_ids = runtime.foot_ids.index_select(0, stance_leg_indices)
    viewer._refresh_viewer_scanner(runtime.base_env, runtime.scanner, minimum_steps=1)
    viewer._viewer_ground_robot_from_scanner(
        runtime.base_env,
        runtime.scanner,
        stance_foot_ids,
        foot_contact_offset=float(
            runtime.base_env._trajectory_manager._cfg.gait.foot_contact_offset
        ),
    )
    viewer._refresh_viewer_scanner(runtime.base_env, runtime.scanner, minimum_steps=1)


def _actual_foot_pos_w(runtime) -> torch.Tensor:
    actual = runtime._viewer._read_actual_kinematic_state(
        runtime.base_env,
        runtime.foot_ids.tolist(),
    )
    return torch.as_tensor(
        actual["foot_pos_w"],
        dtype=torch.float32,
        device=runtime.base_env.device,
    )


def _small_start_xy(
    obstacle_xy: tuple[float, float],
    command: tuple[float, float, float],
    *,
    distance_m: float,
    device: torch.device,
) -> tuple[float, float]:
    command_xy = torch.tensor(command[:2], dtype=torch.float32, device=device)
    axis = command_xy / torch.linalg.vector_norm(command_xy).clamp_min(1.0e-6)
    obstacle = torch.tensor(obstacle_xy, dtype=torch.float32, device=device)
    start = obstacle - float(distance_m) * axis
    return float(start[0].item()), float(start[1].item())


def _select_small_anchor(anchors):
    candidates = tuple(
        anchor
        for anchor in anchors
        if getattr(getattr(anchor, "stage", None), "value", getattr(anchor, "stage", None)) == "S4"
        and getattr(anchor, "semantic_class", None) == "small"
    )
    for shape in ("sphere", "cuboid"):
        selected = tuple(anchor for anchor in candidates if anchor.shape_kind == shape)
        if selected:
            return min(selected, key=lambda anchor: int(getattr(anchor, "slot_index", 0)))
    if candidates:
        return min(candidates, key=lambda anchor: int(getattr(anchor, "slot_index", 0)))
    raise RuntimeError("viewer small reproduction requires an S4 small obstacle anchor")


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
    _ground_initial_stance(runtime)
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


def _run_small_case(
    runtime,
    *,
    name: str,
    command: tuple[float, float, float],
    cycles: int,
) -> dict[str, object]:
    from extension.joint_mpc_rti.integration.isaaclab_adapter import state_from_env
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from Go2Pvcnn.tests.joint_mpc_rti.joint_metrics import JointMetricTrace, evaluate_trace
    from Go2Pvcnn.tests.joint_mpc_rti.run_joint_acceptance import (
        _small_detector_row,
        strict_crossing_event,
    )

    viewer = runtime._viewer
    base_env = runtime.base_env
    manager = base_env._trajectory_manager
    runtime.s4_semantic_course_anchor("small")
    anchor = _select_small_anchor(runtime._semantic_course_anchors())
    command_tensor = torch.tensor(command, dtype=torch.float32, device=base_env.device).view(1, 3)
    start_xy = _small_start_xy(
        anchor.world_xy,
        command,
        distance_m=0.40,
        device=base_env.device,
    )

    runtime.reset()
    runtime._write_env0_root_xy(start_xy)
    runtime._sync_targeted_scan_pose()
    _ground_initial_stance(runtime)
    manager._solver_state = None
    manager._last_result = None
    manager._graph_runner = None

    states: list[torch.Tensor] = []
    feet: list[torch.Tensor] = []
    contacts: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    fallbacks: list[torch.Tensor] = []
    statuses: list[torch.Tensor] = []
    timestamps: list[torch.Tensor] = []
    foot_heights: list[torch.Tensor] = []
    foot_distances: list[torch.Tensor] = []
    map_valids: list[torch.Tensor] = []
    x0_errors: list[torch.Tensor] = []
    x1_errors: list[torch.Tensor] = []
    cold_rows: list[torch.Tensor] = []
    warm_rows: list[torch.Tensor] = []
    collision_rows = {part: [] for part in ("foot", "knee", "calf", "thigh", "base")}
    penetration_rows = {part: [] for part in collision_rows}
    touchdown_rows: list[torch.Tensor] = []
    stance_rows: list[torch.Tensor] = []
    airborne_rows: list[torch.Tensor] = []
    actual_plan_foot_errors: list[float] = []
    actual_plan_joint_errors: list[float] = []
    loss_breakdown_rows: list[dict[str, float]] = []
    step_diagnostics_rows: list[object] = []

    def append_actual_row(
        *,
        measured,
        actual_foot: torch.Tensor,
        contact: torch.Tensor,
        previous_contact: torch.Tensor,
        raw_trajectory,
        node: int,
        timestamp: float,
        x0_error: torch.Tensor,
        x1_error: torch.Tensor,
        cold_start: torch.Tensor,
        warm_start: torch.Tensor,
    ) -> None:
        geometry = go2_fk(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos)
        geometry = replace(geometry, foot_pos_w=actual_foot)
        detector = _small_detector_row(
            manager._field_sync.latest_field(),
            geometry,
            contact,
            previous_contact,
        )
        states.append(measured.as_vector())
        feet.append(actual_foot)
        contacts.append(contact)
        phases.append(manager._solver_state.gait_phase.clone())
        alphas.append(raw_trajectory.line_search_alpha if node else torch.ones_like(raw_trajectory.line_search_alpha))
        valids.append(raw_trajectory.valid if node else torch.ones_like(raw_trajectory.valid))
        fallbacks.append(raw_trajectory.fallback if node else torch.zeros_like(raw_trajectory.fallback))
        statuses.append(raw_trajectory.status if node else torch.zeros_like(raw_trajectory.status))
        timestamps.append(torch.full((1,), timestamp, dtype=torch.float32, device=base_env.device))
        foot_heights.append(detector["foot_height"])
        foot_distances.append(detector["foot_distance"])
        map_valids.append(detector["map_valid"])
        x0_errors.append(x0_error)
        x1_errors.append(x1_error)
        cold_rows.append(cold_start)
        warm_rows.append(warm_start)
        for part in collision_rows:
            collision_rows[part].append(detector["collision"][part])
            penetration_rows[part].append(detector["penetration"][part])
        touchdown_rows.append(detector["touchdown_on_small"])
        stance_rows.append(detector["stance_on_small"])
        airborne_rows.append(detector["airborne_touchdown"])

    previous_contact = None
    first_cross_cycle = None
    post_cross_cycles = 24
    command_axis = command_tensor[:, :2] / torch.linalg.vector_norm(
        command_tensor[:, :2], dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    obstacle_center = torch.tensor([anchor.world_xy], dtype=torch.float32, device=base_env.device)
    obstacle_radius = 0.5 * float(anchor.target_diameter)
    for cycle in range(cycles):
        measured_before = state_from_env(base_env, device=base_env.device)
        measured_before_vector = measured_before.as_vector().clone()
        actual_foot_before = _actual_foot_pos_w(runtime)
        result = viewer._plan_joint_viewer_trajectory(
            manager=manager,
            env=base_env,
            command=command_tensor,
        )
        raw_trajectory = manager.latest_trajectory()
        step_diagnostics_rows.append(manager._last_result.diagnostics)
        loss_breakdown_rows.append(
            {
                key: float(torch.as_tensor(value).reshape(-1)[0].item())
                for key, value in raw_trajectory.loss_breakdown.items()
            }
        )
        contact0 = raw_trajectory.contact_state[:, 0].to(dtype=torch.bool)
        if previous_contact is None:
            zero = torch.zeros(1, dtype=torch.float32, device=base_env.device)
            false = torch.zeros(1, dtype=torch.bool, device=base_env.device)
            append_actual_row(
                measured=measured_before,
                actual_foot=actual_foot_before,
                contact=contact0,
                previous_contact=contact0,
                raw_trajectory=raw_trajectory,
                node=0,
                timestamp=0.0,
                x0_error=zero,
                x1_error=zero,
                cold_start=false,
                warm_start=false,
            )
            previous_contact = contact0

        viewer._viewer_direct_playback_step(base_env, result, frame_idx=1)
        measured_after = state_from_env(base_env, device=base_env.device)
        actual_foot_after = _actual_foot_pos_w(runtime)
        contact1 = raw_trajectory.contact_state[:, 1].to(dtype=torch.bool)
        planned_x1 = raw_trajectory.state[:, 1]
        actual_plan_foot_errors.append(
            float(
                torch.linalg.vector_norm(
                    actual_foot_after - raw_trajectory.foot_pos_w[:, 1],
                    dim=-1,
                ).max().item()
            )
        )
        actual_plan_joint_errors.append(
            float((measured_after.joint_pos - raw_trajectory.state[:, 1, 6:]).abs().max().item())
        )
        append_actual_row(
            measured=measured_after,
            actual_foot=actual_foot_after,
            contact=contact1,
            previous_contact=previous_contact,
            raw_trajectory=raw_trajectory,
            node=1,
            timestamp=(cycle + 1) * float(manager._cfg.runtime.dt),
            x0_error=(raw_trajectory.state[:, 0] - measured_before_vector).abs().amax(dim=-1),
            x1_error=(planned_x1 - measured_after.as_vector()).abs().amax(dim=-1),
            cold_start=raw_trajectory.cold_start,
            warm_start=raw_trajectory.warm_start,
        )
        previous_contact = contact1
        progress = (
            (measured_after.root_pos_w[:, :2] - obstacle_center) * command_axis
        ).sum(dim=-1)
        if first_cross_cycle is None and bool((progress > obstacle_radius).all().item()):
            first_cross_cycle = cycle + 1
        if first_cross_cycle is not None and cycle + 1 >= first_cross_cycle + post_cross_cycles:
            break

    state = torch.stack(states, dim=1)
    foot = torch.stack(feet, dim=1)
    contact = torch.stack(contacts, dim=1)
    stance_anchor = foot.clone()
    for node in range(1, int(foot.shape[1])):
        continuing = contact[:, node] & contact[:, node - 1]
        stance_anchor[:, node] = torch.where(
            continuing[..., None],
            stance_anchor[:, node - 1],
            foot[:, node],
        )
    crossing = strict_crossing_event(
        state[..., :2],
        command_tensor[:, :2],
        obstacle_center,
        radius_m=obstacle_radius,
    ).success.to(dtype=state.dtype)
    trace = JointMetricTrace(
        root_pos_w=state[..., :3],
        root_rpy_w=state[..., 3:6],
        joint_pos=state[..., 6:],
        foot_pos_w=foot,
        contact_state=contact,
        command_body=command_tensor[:, None].expand(-1, state.shape[1], -1),
        gait_phase=torch.stack(phases, dim=1),
        foot_height_w=torch.stack(foot_heights, dim=1),
        foot_small_distance_m=torch.stack(foot_distances, dim=1),
        part_collision={part: torch.stack(rows, dim=1) for part, rows in collision_rows.items()},
        line_alpha=torch.stack(alphas, dim=1),
        nominal_root_pos_w=state[..., :3],
        nominal_root_rpy_w=state[..., 3:6],
        valid=torch.stack(valids, dim=1),
        map_valid=torch.stack(map_valids, dim=1),
        timestamps=torch.stack(timestamps, dim=1),
        dt=float(manager._cfg.runtime.dt),
        stance_anchor_w=stance_anchor,
        strict_cross_success=crossing,
        touchdown_on_small=torch.stack(touchdown_rows, dim=1),
        stance_on_small=torch.stack(stance_rows, dim=1),
        airborne_touchdown=torch.stack(airborne_rows, dim=1),
        part_penetration_m={part: torch.stack(rows, dim=1) for part, rows in penetration_rows.items()},
        x0_injection_error=torch.stack(x0_errors, dim=1),
        published_x1_error=torch.stack(x1_errors, dim=1),
        cold_start=torch.stack(cold_rows, dim=1),
        warm_start=torch.stack(warm_rows, dim=1),
        warm_cache_invariant_fault=torch.zeros_like(torch.stack(valids, dim=1)),
    )
    metric_report = evaluate_trace(trace, scenario="small", key=("viewer", name))
    stacked_phases = torch.stack(phases, dim=1)
    stacked_alphas = torch.stack(alphas, dim=1)
    stacked_valids = torch.stack(valids, dim=1)
    stacked_fallbacks = torch.stack(fallbacks, dim=1)
    stacked_statuses = torch.stack(statuses, dim=1)
    invalid_cycles = [
        {
            "node": int(node),
            "phase": int(stacked_phases[0, node].item()),
            "line_alpha": float(stacked_alphas[0, node].item()),
            "fallback": bool(stacked_fallbacks[0, node].item()),
            "status": int(stacked_statuses[0, node].item()),
        }
        for node in torch.where(~stacked_valids[0])[0].tolist()
    ]
    filter_names = ("finite", "joint_position", "joint_velocity", "published_kinematics")
    for row in invalid_cycles:
        node = int(row["node"])
        if node <= 0:
            continue
        diagnostics = step_diagnostics_rows[node - 1]
        row["candidate_loss"] = [
            float(value) for value in diagnostics.candidate_loss[0].tolist()
        ]
        row["candidate_filter_valid"] = {
            name: [
                bool(value)
                for value in diagnostics.candidate_filter_valid[0, :, index].tolist()
            ]
            for index, name in enumerate(filter_names)
        }
        row["support_target_m"] = [
            float(value) for value in diagnostics.support_target[0].tolist()
        ]
        row["qp_direction_x1"] = [
            float(value) for value in diagnostics.qp_direction[0, 1].tolist()
        ]
    continuing_stance = contact[:, 1:] & contact[:, :-1]
    stance_step = torch.linalg.vector_norm(
        foot[:, 1:, :, :2] - foot[:, :-1, :, :2],
        dim=-1,
    )
    masked_stance_step = torch.where(
        continuing_stance,
        stance_step,
        torch.full_like(stance_step, -1.0),
    )
    worst_flat_index = int(masked_stance_step.reshape(-1).argmax().item())
    edge_index = (worst_flat_index // 4) % int(stance_step.shape[1])
    leg_index = worst_flat_index % 4
    ending_node = edge_index + 1
    foot_delta = foot[0, ending_node, leg_index] - foot[0, edge_index, leg_index]
    root_delta = state[0, ending_node, :3] - state[0, edge_index, :3]
    surface_error_before = (
        foot[0, edge_index, leg_index, 2]
        - torch.stack(foot_heights, dim=1)[0, edge_index, leg_index]
        - 0.022
    )
    surface_error_after = (
        foot[0, ending_node, leg_index, 2]
        - torch.stack(foot_heights, dim=1)[0, ending_node, leg_index]
        - 0.022
    )
    step_diagnostics = step_diagnostics_rows[edge_index]
    nominal_edge = step_diagnostics.nominal_state
    full_step_edge = nominal_edge + step_diagnostics.qp_direction
    nominal_geometry = go2_fk(
        nominal_edge[..., :3], nominal_edge[..., 3:6], nominal_edge[..., 6:]
    )
    full_step_geometry = go2_fk(
        full_step_edge[..., :3], full_step_edge[..., 3:6], full_step_edge[..., 6:]
    )
    solver_anchor = step_diagnostics.stance_anchor_w[0, leg_index]

    def anchor_xy_error(point: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(point[:2] - solver_anchor[:2]).item())

    worst_stance_event = {
        "edge_index": edge_index,
        "ending_node": ending_node,
        "leg_index": leg_index,
        "phase_before": int(torch.stack(phases, dim=1)[0, edge_index].item()),
        "phase_after": int(torch.stack(phases, dim=1)[0, ending_node].item()),
        "foot_step_m": float(stance_step[0, edge_index, leg_index].item()),
        "foot_delta_w_m": [float(value) for value in foot_delta.tolist()],
        "root_delta_w_m": [float(value) for value in root_delta.tolist()],
        "anchor_residual_m": float(
            torch.linalg.vector_norm(
                foot[0, ending_node, leg_index] - stance_anchor[0, ending_node, leg_index]
            ).item()
        ),
        "surface_error_before_m": float(surface_error_before.item()),
        "surface_error_after_m": float(surface_error_after.item()),
        "small_distance_before_m": float(
            torch.stack(foot_distances, dim=1)[0, edge_index, leg_index].item()
        ),
        "small_distance_after_m": float(
            torch.stack(foot_distances, dim=1)[0, ending_node, leg_index].item()
        ),
        "line_alpha": float(torch.stack(alphas, dim=1)[0, ending_node].item()),
        "loss_breakdown": loss_breakdown_rows[edge_index],
        "solver_layers": {
            "anchor_w_m": [float(value) for value in solver_anchor.tolist()],
            "measured_node0_anchor_xy_error_m": anchor_xy_error(foot[0, edge_index, leg_index]),
            "nominal_node0_anchor_xy_error_m": anchor_xy_error(
                nominal_geometry.foot_pos_w[0, 0, leg_index]
            ),
            "nominal_x1_anchor_xy_error_m": anchor_xy_error(
                nominal_geometry.foot_pos_w[0, 1, leg_index]
            ),
            "qp_full_step_x1_anchor_xy_error_m": anchor_xy_error(
                full_step_geometry.foot_pos_w[0, 1, leg_index]
            ),
            "selected_x1_anchor_xy_error_m": anchor_xy_error(
                foot[0, ending_node, leg_index]
            ),
            "nominal_root_x1_delta_m": [
                float(value)
                for value in (nominal_edge[0, 1, :3] - nominal_edge[0, 0, :3]).tolist()
            ],
            "qp_direction_x1": [
                float(value) for value in step_diagnostics.qp_direction[0, 1].tolist()
            ],
        },
    }
    joint_delta = state[:, 1:, 6:] - state[:, :-1, 6:]
    worst_joint_flat_index = int(joint_delta.abs().reshape(-1).argmax().item())
    joint_index = worst_joint_flat_index % int(joint_delta.shape[-1])
    joint_edge_index = (worst_joint_flat_index // int(joint_delta.shape[-1])) % int(
        joint_delta.shape[1]
    )
    joint_ending_node = joint_edge_index + 1
    joint_diagnostics = step_diagnostics_rows[joint_edge_index]
    joint_nominal = joint_diagnostics.nominal_state
    joint_leg_index = joint_index // 3
    joint_nominal_geometry = go2_fk(
        joint_nominal[..., :3], joint_nominal[..., 3:6], joint_nominal[..., 6:]
    )
    joint_nominal_foot = joint_nominal_geometry.foot_pos_w[0, :, joint_leg_index]
    joint_touchdown_target = joint_diagnostics.touchdown_reference_w[0, :, joint_leg_index]
    stacked_foot_heights = torch.stack(foot_heights, dim=1)
    joint_node_loss = {
        name: {
            "weighted": [float(value) for value in energy[0].tolist()],
        }
        for name, energy in joint_diagnostics.node_loss_breakdown.items()
    }
    worst_joint_event = {
        "edge_index": joint_edge_index,
        "ending_node": joint_ending_node,
        "joint_index": joint_index,
        "phase_before": int(torch.stack(phases, dim=1)[0, joint_edge_index].item()),
        "phase_after": int(torch.stack(phases, dim=1)[0, joint_ending_node].item()),
        "joint_before_rad": float(state[0, joint_edge_index, 6 + joint_index].item()),
        "joint_after_rad": float(state[0, joint_ending_node, 6 + joint_index].item()),
        "joint_delta_rad": float(joint_delta[0, joint_edge_index, joint_index].item()),
        "line_alpha": float(torch.stack(alphas, dim=1)[0, joint_ending_node].item()),
        "nominal_joint_delta_rad": float(
            (
                joint_nominal[0, 1, 6 + joint_index]
                - joint_nominal[0, 0, 6 + joint_index]
            ).item()
        ),
        "qp_direction_x1_rad": float(
            joint_diagnostics.qp_direction[0, 1, 6 + joint_index].item()
        ),
        "node_loss_energy": joint_node_loss,
        "nominal_foot_w_m": [
            [float(value) for value in row.tolist()] for row in joint_nominal_foot
        ],
        "touchdown_target_w_m": [
            [float(value) for value in row.tolist()] for row in joint_touchdown_target
        ],
        "nominal_foot_delta_m": [
            float(value) for value in (joint_nominal_foot[1] - joint_nominal_foot[0]).tolist()
        ],
        "touchdown_target_delta_m": [
            float(value)
            for value in (joint_touchdown_target[1] - joint_touchdown_target[0]).tolist()
        ],
        "nominal_foot_target_error_m": [
            float(torch.linalg.vector_norm(joint_nominal_foot[node] - joint_touchdown_target[node]).item())
            for node in range(2)
        ],
        "actual_surface_error_m": [
            float(
                foot[0, node, joint_leg_index, 2]
                - stacked_foot_heights[0, node, joint_leg_index]
                - 0.022
            )
            for node in (joint_edge_index, joint_ending_node)
        ],
    }
    swing_clearance = foot[..., 2] - stacked_foot_heights - 0.022
    swing_mask = (
        ~contact
        & stacked_valids[..., None]
        & (torch.stack(timestamps, dim=1) > 0.0)[..., None]
    )
    masked_swing_clearance = torch.where(
        swing_mask,
        swing_clearance,
        torch.full_like(swing_clearance, torch.inf),
    )
    worst_swing_flat_index = int(masked_swing_clearance.reshape(-1).argmin().item())
    swing_leg_index = worst_swing_flat_index % 4
    swing_node = (worst_swing_flat_index // 4) % int(swing_clearance.shape[1])
    worst_swing_event = {
        "node": swing_node,
        "leg_index": swing_leg_index,
        "phase": int(torch.stack(phases, dim=1)[0, swing_node].item()),
        "clearance_m": float(swing_clearance[0, swing_node, swing_leg_index].item()),
        "foot_w_m": [
            float(value) for value in foot[0, swing_node, swing_leg_index].tolist()
        ],
        "terrain_height_m": float(
            stacked_foot_heights[0, swing_node, swing_leg_index].item()
        ),
        "small_distance_m": float(
            torch.stack(foot_distances, dim=1)[0, swing_node, swing_leg_index].item()
        ),
    }
    if swing_node > 0:
        swing_diagnostics = step_diagnostics_rows[swing_node - 1]
        swing_nominal = swing_diagnostics.nominal_state
        swing_full = swing_nominal + swing_diagnostics.qp_direction
        swing_nominal_foot = go2_fk(
            swing_nominal[..., :3], swing_nominal[..., 3:6], swing_nominal[..., 6:]
        ).foot_pos_w[0, 1, swing_leg_index]
        swing_full_foot = go2_fk(
            swing_full[..., :3], swing_full[..., 3:6], swing_full[..., 6:]
        ).foot_pos_w[0, 1, swing_leg_index]
        candidate_safe_z = swing_diagnostics.candidate_swing_safe_z[
            0, :, swing_leg_index
        ]
        selected_alpha = float(stacked_alphas[0, swing_node].item())
        alpha_values = candidate_safe_z.new_tensor((1.0, 0.5, 0.25, 0.125, 0.0))
        selected_candidate = int(
            torch.abs(alpha_values - selected_alpha).argmin().item()
        )
        nominal_safe_z = candidate_safe_z[4]
        full_safe_z = candidate_safe_z[0]
        selected_safe_z = candidate_safe_z[selected_candidate]
        selected_foot_z = foot[0, swing_node, swing_leg_index, 2]
        worst_swing_event["solver_layers"] = {
            "nominal_x1_foot_w_m": [float(value) for value in swing_nominal_foot.tolist()],
            "qp_full_x1_foot_w_m": [float(value) for value in swing_full_foot.tolist()],
            "selected_x1_foot_w_m": [
                float(value) for value in foot[0, swing_node, swing_leg_index].tolist()
            ],
            "safe_floor_z_m": {
                "nominal": float(nominal_safe_z.item()),
                "full": float(full_safe_z.item()),
                "selected": float(selected_safe_z.item()),
            },
            "safe_floor_deficit_m": {
                "nominal": float((nominal_safe_z - swing_nominal_foot[2]).item()),
                "full": float((full_safe_z - swing_full_foot[2]).item()),
                "selected": float((selected_safe_z - selected_foot_z).item()),
            },
            "candidate_loss": [
                float(value) for value in swing_diagnostics.candidate_loss[0].tolist()
            ],
            "candidate_filter_valid": {
                name: [
                    bool(value)
                    for value in swing_diagnostics.candidate_filter_valid[0, :, index].tolist()
                ]
                for index, name in enumerate(filter_names)
            },
        }
    stance_surface_error = foot[..., 2] - stacked_foot_heights - 0.022
    masked_stance_gap = torch.where(
        contact,
        stance_surface_error,
        torch.full_like(stance_surface_error, -torch.inf),
    )
    gap_flat_index = int(masked_stance_gap.reshape(-1).argmax().item())
    gap_leg = gap_flat_index % 4
    gap_node = (gap_flat_index // 4) % int(stance_surface_error.shape[1])
    gap_onset = bool(
        gap_node > 0
        and contact[0, gap_node, gap_leg]
        and not contact[0, gap_node - 1, gap_leg]
    )
    worst_stance_gap_event: dict[str, object] = {
        "node": gap_node,
        "leg_index": gap_leg,
        "phase": int(stacked_phases[0, gap_node].item()),
        "surface_error_m": float(stance_surface_error[0, gap_node, gap_leg].item()),
        "onset": gap_onset,
        "line_alpha": float(stacked_alphas[0, gap_node].item()),
        "valid": bool(stacked_valids[0, gap_node].item()),
        "warm_start": bool(torch.stack(warm_rows, dim=1)[0, gap_node].item()),
    }
    if gap_node > 0:
        gap_diagnostics = step_diagnostics_rows[gap_node - 1]
        gap_nominal = gap_diagnostics.nominal_state
        gap_full = gap_nominal + gap_diagnostics.qp_direction
        gap_nominal_foot = go2_fk(
            gap_nominal[..., :3], gap_nominal[..., 3:6], gap_nominal[..., 6:]
        ).foot_pos_w[0, 1, gap_leg]
        gap_full_foot = go2_fk(
            gap_full[..., :3], gap_full[..., 3:6], gap_full[..., 6:]
        ).foot_pos_w[0, 1, gap_leg]
        worst_stance_gap_event["solver_layers"] = {
            "nominal_x1_foot_w_m": [float(value) for value in gap_nominal_foot.tolist()],
            "qp_full_x1_foot_w_m": [float(value) for value in gap_full_foot.tolist()],
            "selected_x1_foot_w_m": [
                float(value) for value in foot[0, gap_node, gap_leg].tolist()
            ],
            "candidate_loss": [
                float(value) for value in gap_diagnostics.candidate_loss[0].tolist()
            ],
            "candidate_filter_valid": {
                name: [
                    bool(value)
                    for value in gap_diagnostics.candidate_filter_valid[0, :, index].tolist()
                ]
                for index, name in enumerate(filter_names)
            },
        }
    masked_stance_surface = torch.where(
        contact,
        stance_surface_error,
        torch.full_like(stance_surface_error, torch.inf),
    )
    penetration_flat_index = int(masked_stance_surface.reshape(-1).argmin().item())
    penetration_leg = penetration_flat_index % 4
    penetration_node = (penetration_flat_index // 4) % int(stance_surface_error.shape[1])
    penetration_onset = bool(
        penetration_node > 0
        and contact[0, penetration_node, penetration_leg]
        and not contact[0, penetration_node - 1, penetration_leg]
    )
    worst_stance_penetration_event = {
        "node": penetration_node,
        "leg_index": penetration_leg,
        "phase": int(stacked_phases[0, penetration_node].item()),
        "surface_error_m": float(
            stance_surface_error[0, penetration_node, penetration_leg].item()
        ),
        "onset": penetration_onset,
        "line_alpha": float(stacked_alphas[0, penetration_node].item()),
        "valid": bool(stacked_valids[0, penetration_node].item()),
        "foot_z_m": float(foot[0, penetration_node, penetration_leg, 2].item()),
        "terrain_height_m": float(
            stacked_foot_heights[0, penetration_node, penetration_leg].item()
        ),
    }
    root_tracking_solver_layers = _root_tracking_solver_layers(
        state=state,
        step_diagnostics_rows=step_diagnostics_rows,
        command_body=command_tensor,
        line_alpha=stacked_alphas[:, 1:],
        warm_start=torch.stack(warm_rows, dim=1)[:, 1:],
        dt=float(manager._cfg.runtime.dt),
        root_position_trust=float(manager._cfg.solver.root_position_trust),
    )
    return {
        "name": name,
        "command": command,
        "max_cycles": cycles,
        "executed_cycles": len(actual_plan_foot_errors),
        "obstacle": {
            "world_xy": anchor.world_xy,
            "shape": anchor.shape_kind,
            "diameter_m": float(anchor.target_diameter),
            "height_m": float(anchor.target_height),
        },
        "actual_state_samples": int(state.shape[1]),
        "actual_vs_planner_foot_max_m": _summary(actual_plan_foot_errors),
        "actual_vs_planner_joint_max_rad": _summary(actual_plan_joint_errors),
        "worst_stance_event": worst_stance_event,
        "worst_stance_gap_event": worst_stance_gap_event,
        "worst_stance_penetration_event": worst_stance_penetration_event,
        "worst_joint_event": worst_joint_event,
        "worst_swing_event": worst_swing_event,
        "root_tracking_solver_layers": root_tracking_solver_layers,
        "invalid_cycles": invalid_cycles,
        "metrics": {metric_name: asdict(metric) for metric_name, metric in metric_report.metrics.items()},
        "passed": metric_report.passed,
    }


def main() -> int:
    output_path = Path(os.environ.get("JOINT_MPC_VIEWER_REPRO_OUTPUT", "/tmp/joint_mpc_viewer_repro.json"))
    device = os.environ.get("MPC_TEST_DEVICE", "cuda:0")
    scenario = os.environ.get("JOINT_MPC_VIEWER_REPRO_SCENARIO", "flat")
    if scenario not in {"flat", "small"}:
        raise ValueError("JOINT_MPC_VIEWER_REPRO_SCENARIO must be flat or small")
    default_cycles = "160" if scenario == "small" else "16"
    cycles = int(os.environ.get("JOINT_MPC_VIEWER_REPRO_CYCLES", default_cycles))
    warmup_steps = int(os.environ.get("JOINT_MPC_VIEWER_REPRO_WARMUP_STEPS", "0"))
    runtime = None
    try:
        runtime = RealViewerRuntimeFixture(
            num_envs=1,
            planner_backend="joint_mpc_rti",
            requested_n_frames=30,
            warmup_steps=warmup_steps,
            device=device,
            task_id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
        )
        manager = runtime.base_env._trajectory_manager
        manager._cfg.solver.use_cuda_graph = False
        flat_case_specs = (
            ("standstill", (0.0, 0.0, 0.0)),
            ("forward_slow", (0.1, 0.0, 0.0)),
            ("forward_fast", (0.4, 0.0, 0.0)),
            ("backward", (-0.25, 0.0, 0.0)),
            ("lateral_left", (0.0, 0.25, 0.0)),
            ("lateral_right", (0.0, -0.25, 0.0)),
            ("yaw_left", (0.0, 0.0, 0.5)),
            ("yaw_right", (0.0, 0.0, -0.5)),
        )
        small_case_specs = (
            ("small_forward", (1.0, 0.0, 0.0)),
            ("small_backward", (-1.0, 0.0, 0.0)),
            ("small_lateral_left", (0.0, 0.5, 0.0)),
            ("small_lateral_right", (0.0, -0.5, 0.0)),
        )
        case_specs = small_case_specs if scenario == "small" else flat_case_specs
        requested = {
            name.strip()
            for name in os.environ.get("JOINT_MPC_VIEWER_REPRO_CASES", "").split(",")
            if name.strip()
        }
        selected_specs = case_specs if not requested else tuple(spec for spec in case_specs if spec[0] in requested)
        case_runner = _run_small_case if scenario == "small" else _run_case
        cases = [case_runner(runtime, name=name, command=command, cycles=cycles) for name, command in selected_specs]
        if not cases:
            raise ValueError("JOINT_MPC_VIEWER_REPRO_CASES selected no known cases")
        if scenario == "small":
            acceptance = {
                "shared_small_metrics": all(bool(case["passed"]) for case in cases),
                "actual_state_samples": all(
                    int(case["actual_state_samples"]) == int(case["executed_cycles"]) + 1
                    and int(case["executed_cycles"]) <= cycles
                    for case in cases
                ),
            }
            result = {
                "scenario": scenario,
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
            "scenario": scenario,
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
