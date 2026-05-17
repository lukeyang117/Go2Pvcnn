from __future__ import annotations

import os

import pytest
import torch

from Go2Pvcnn.tests.fixtures import viewer_runtime_diagnostics as viewer_diag


def _device() -> str:
    return os.environ.get("MPC_TEST_DEVICE", "cuda:0")


def _enable_collision_headless() -> bool:
    return os.environ.get("MPC_T302_HEADLESS", "0") == "1"


pytestmark = pytest.mark.skipif(
    not _enable_collision_headless(),
    reason="Set MPC_T302_HEADLESS=1 to run T302 IsaacLab headless collision acceptance.",
)


def _height_at(terrain, points_xy: torch.Tensor) -> torch.Tensor:
    from extension.batch_mpc_planner.terrain import height_at

    return height_at(terrain, points_xy).to(dtype=points_xy.dtype, device=points_xy.device)


def _semantic_at(terrain, points_xy: torch.Tensor) -> torch.Tensor:
    from extension.batch_mpc_planner.terrain import semantic_at

    return semantic_at(terrain, points_xy)


def _stance_semantic_count(result, terrain) -> int:
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32)
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool)
    sem = _semantic_at(terrain, foot[..., :2])
    obstacle = torch.logical_and(contact, sem > 0)
    return int(obstacle.to(torch.int64).sum().item())


def _stance_semantic_ratio(result, terrain) -> float:
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32)
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool)
    sem = _semantic_at(terrain, foot[..., :2])
    obstacle = torch.logical_and(contact, sem > 0)
    contact_mass = torch.clamp(contact.to(dtype=torch.float32).sum(), min=1.0)
    return float(obstacle.to(dtype=torch.float32).sum().item() / float(contact_mass.item()))


def _obstacle_risk_scale(result, name: str) -> float:
    breakdown = getattr(result, "loss_breakdown", None)
    if not breakdown or name not in breakdown:
        return 1.0
    return float(torch.as_tensor(breakdown[name]).reshape(-1)[0].item())


def _planned_collision_metrics(result, terrain, viewer_module) -> dict[str, float]:
    from extension.batch_mpc_planner.kinematics import fk_leg_points_from_joint_angles

    root = torch.as_tensor(result.root_pos_w, dtype=torch.float32).contiguous()
    quat = torch.as_tensor(result.root_quat_w, dtype=torch.float32, device=root.device).contiguous()
    rpy = viewer_module._quat_wxyz_to_rpy(quat.reshape(-1, 4)).to(dtype=root.dtype, device=root.device).reshape_as(root)
    joints = torch.as_tensor(result.joint_angles, dtype=torch.float32, device=root.device).contiguous()
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32, device=root.device).contiguous()
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=root.device)
    swing = torch.logical_not(contact)
    leg_points = fk_leg_points_from_joint_angles(root, rpy, joints, shank_sample_count=2)
    foot_clearance = foot[..., 2] - _height_at(terrain, foot[..., :2])
    knee_clearance = leg_points.knee_pos_world[..., 2] - _height_at(terrain, leg_points.knee_pos_world[..., :2])
    shank_clearance = leg_points.shank_sample_world[..., 2] - _height_at(terrain, leg_points.shank_sample_world[..., :2])
    root_bottom = root.clone()
    root_bottom[..., 2] = root_bottom[..., 2] - 0.18
    root_bottom_clearance = root_bottom[..., 2] - _height_at(terrain, root_bottom[..., :2])
    swing_mass = torch.clamp(swing.to(dtype=torch.float32).sum(), min=1.0)
    root_mass = torch.clamp(torch.ones_like(root_bottom_clearance).sum(), min=1.0)
    return {
        "root_bottom_min_clearance": float(root_bottom_clearance.min().item()),
        "swing_foot_min_clearance": float(foot_clearance[swing].min().item()) if bool(swing.any().item()) else 1.0,
        "knee_min_clearance": float(knee_clearance.min().item()),
        "shank_min_clearance": float(shank_clearance.min().item()),
        "root_bottom_collision_ratio": float((root_bottom_clearance < 0.0).to(torch.float32).sum().item() / float(root_mass.item())),
        "swing_foot_collision_ratio": float(((foot_clearance < 0.0) & swing).to(torch.float32).sum().item() / float(swing_mass.item())),
        "knee_collision_ratio": float((knee_clearance < 0.0).to(torch.float32).mean().item()),
        "shank_collision_ratio": float((shank_clearance < 0.0).to(torch.float32).mean().item()),
        "joint_finite": float(torch.isfinite(joints).all().item()),
    }


def _command_direction(command: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    cmd_xy = torch.as_tensor(command, dtype=torch.float32, device=device).reshape(-1, 3)[0, :2]
    norm = torch.linalg.vector_norm(cmd_xy)
    if float(norm.item()) <= 1.0e-6:
        return torch.tensor((1.0, 0.0), dtype=torch.float32, device=device)
    return cmd_xy / norm


def _crosses_obstacle_along_command(root: torch.Tensor, obstacle_xy: torch.Tensor, command: torch.Tensor) -> tuple[bool, float]:
    direction = _command_direction(command, device=root.device)
    rel = root[0, :, :2] - obstacle_xy
    along = (rel * direction).sum(dim=-1)
    lateral = rel[:, 0] * (-direction[1]) + rel[:, 1] * direction[0]
    crossed = (along[0] * along[-1]) < 0.0
    return bool(crossed.item()), float(torch.abs(lateral).min().item())


def _min_root_distance_to_obstacle(root: torch.Tensor, obstacle_xy: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(root[0, :, :2] - obstacle_xy, dim=-1).min().item())


def _assert_collision_metrics_safe(metrics: dict[str, float]) -> None:
    assert metrics["root_bottom_collision_ratio"] <= 0.02
    assert metrics["swing_foot_collision_ratio"] <= 0.02
    assert metrics["knee_collision_ratio"] <= 0.02
    assert metrics["shank_collision_ratio"] <= 0.02
    assert metrics["root_bottom_min_clearance"] > -0.02
    assert metrics["joint_finite"] == 1.0


def test_t302_cobblestone_mpc_headless_collision_metrics() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        terrain="cobblestone",
        device=_device(),
        warmup_steps=6,
    )
    try:
        commands = (
            "forward",
            "backward",
            "lateral_left",
            "lateral_right",
            "yaw_left",
            "yaw_right",
            "forward_yaw_left",
            "forward_yaw_right",
            "diagonal_forward_left",
            "diagonal_forward_right",
        )
        rows: list[dict[str, float]] = []
        for name in commands:
            plan = runtime.plan_case(name)
            terrain = runtime._single_env_terrain()
            rows.append(_planned_collision_metrics(plan.result, terrain, runtime._viewer))

        assert rows
        assert max(row["swing_foot_collision_ratio"] for row in rows) <= 0.02
        assert max(row["knee_collision_ratio"] for row in rows) <= 0.02
        assert max(row["shank_collision_ratio"] for row in rows) <= 0.02
        assert max(row["root_bottom_collision_ratio"] for row in rows) <= 0.02
        assert min(row["root_bottom_min_clearance"] for row in rows) > -0.02
        assert min(row["joint_finite"] for row in rows) == 1.0
    finally:
        runtime.close()


def test_t302_low_small_obstacle_crosses_without_stance_on_obstacle() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=_device(),
        warmup_steps=6,
        semantic_small_height_m=0.16,
    )
    try:
        plan = runtime.plan_case_near_s4_anchor_command_relative(
            "small",
            command_name="forward",
            longitudinal_offset_m=-0.35,
            lateral_offset_m=0.0,
            z_clearance=0.65,
        )
        terrain = runtime._single_env_terrain()
        root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
        anchor = runtime.s4_semantic_course_anchor("small")
        obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
        crossed, min_lateral = _crosses_obstacle_along_command(root, obstacle_xy, plan.command)
        metrics = _planned_collision_metrics(plan.result, terrain, runtime._viewer)

        assert crossed
        assert min_lateral < 0.20
        assert _stance_semantic_count(plan.result, terrain) == 0
        _assert_collision_metrics_safe(metrics)
    finally:
        runtime.close()


def test_t302_low_small_obstacle_crosses_for_command_directions_without_collisions() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=_device(),
        warmup_steps=6,
        semantic_small_height_m=0.16,
    )
    try:
        commands = ("forward", "backward", "lateral_left", "lateral_right")
        rows: list[tuple[str, bool, float, int, float, dict[str, float]]] = []
        for command_name in commands:
            plan = runtime.plan_case_near_s4_anchor_command_relative(
                "small",
                command_name=command_name,
                longitudinal_offset_m=-0.35,
                lateral_offset_m=0.0,
                z_clearance=0.65,
            )
            terrain = runtime._single_env_terrain()
            root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
            anchor = runtime.s4_semantic_course_anchor("small")
            obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
            crossed, min_lateral = _crosses_obstacle_along_command(root, obstacle_xy, plan.command)
            rows.append(
                (
                    command_name,
                    crossed,
                    min_lateral,
                    _stance_semantic_count(plan.result, terrain),
                    _stance_semantic_ratio(plan.result, terrain),
                    _planned_collision_metrics(plan.result, terrain, runtime._viewer),
                )
            )

        assert len(rows) == len(commands)
        assert all(crossed for _name, crossed, _lat, _count, _ratio, _metrics in rows), rows
        assert max(min_lateral for _name, _crossed, min_lateral, _count, _ratio, _metrics in rows) < 0.20
        assert max(count for _name, _crossed, _lat, count, _ratio, _metrics in rows) == 0
        assert max(ratio for _name, _crossed, _lat, _count, ratio, _metrics in rows) == pytest.approx(0.0)
        for _name, _crossed, _lat, _count, _ratio, metrics in rows:
            _assert_collision_metrics_safe(metrics)
    finally:
        runtime.close()


def test_t302_large_obstacle_avoids_or_scales_tracking_near_yaw() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=_device(),
        warmup_steps=6,
    )
    try:
        plan = runtime.plan_case_near_s4_anchor_command_relative(
            "large",
            command_name="yaw_left",
            longitudinal_offset_m=-0.15,
            lateral_offset_m=0.0,
            z_clearance=0.65,
        )
        terrain = runtime._single_env_terrain()
        assert _stance_semantic_count(plan.result, terrain) == 0
        assert _obstacle_risk_scale(plan.result, "obstacle_risk_yaw_scale") <= 0.5
    finally:
        runtime.close()


def test_t302_large_obstacle_in_command_direction_is_avoided_or_deweighted() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=_device(),
        warmup_steps=6,
    )
    try:
        plan = runtime.plan_case_near_s4_anchor_command_relative(
            "large",
            command_name="forward",
            longitudinal_offset_m=-0.35,
            lateral_offset_m=0.0,
            z_clearance=0.65,
        )
        terrain = runtime._single_env_terrain()
        root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
        anchor = runtime.s4_semantic_course_anchor("large")
        obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
        min_dist = _min_root_distance_to_obstacle(root, obstacle_xy)
        metrics = _planned_collision_metrics(plan.result, terrain, runtime._viewer)

        assert _stance_semantic_count(plan.result, terrain) == 0
        assert _obstacle_risk_scale(plan.result, "obstacle_risk_linear_scale") <= 0.5
        assert min_dist > 0.5 * float(anchor.target_diameter) + 0.08
        _assert_collision_metrics_safe(metrics)
    finally:
        runtime.close()


def test_t302_high_small_obstacle_in_command_direction_is_not_crossed_like_low_small() -> None:
    runtime = viewer_diag.make_real_runtime_fixture(
        num_envs=2,
        planner_backend="mpc",
        device=_device(),
        warmup_steps=6,
        semantic_small_height_m=0.46,
    )
    try:
        plan = runtime.plan_case_near_s4_anchor_command_relative(
            "small",
            command_name="forward",
            longitudinal_offset_m=-0.35,
            lateral_offset_m=0.0,
            z_clearance=0.65,
        )
        terrain = runtime._single_env_terrain()
        root = torch.as_tensor(plan.result.root_pos_w, dtype=torch.float32)
        anchor = runtime.s4_semantic_course_anchor("small")
        obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=root.device)
        crossed, min_lateral = _crosses_obstacle_along_command(root, obstacle_xy, plan.command)
        min_dist = _min_root_distance_to_obstacle(root, obstacle_xy)
        metrics = _planned_collision_metrics(plan.result, terrain, runtime._viewer)

        assert _obstacle_risk_scale(plan.result, "obstacle_risk_linear_scale") <= 0.5
        assert (not crossed) or min_lateral > 0.22
        assert min_dist > 0.5 * float(anchor.target_diameter) + 0.08
        assert _stance_semantic_count(plan.result, terrain) == 0
        _assert_collision_metrics_safe(metrics)
    finally:
        runtime.close()
