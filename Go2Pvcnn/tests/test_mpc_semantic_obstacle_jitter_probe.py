from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT, GO2PVCNN_ROOT / "tests"):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from mpc_semantic_obstacle_jitter_probe import (
    _StructuralSemanticWeights,
    _LowSmallStraightWeights,
    _LossOnlySemanticWeights,
    _candidate_variants_for_variant,
    _crossing_metrics,
    _jitter_metrics,
    _patched_structural_loss_for_variant,
    _post_blend_result_for_variant,
    _selector_jitter_sort_key,
    _selector_clearance_jitter_sort_key,
    _selector_metric_task_jitter_sort_key,
    _selector_large_smooth_metric_sort_key,
    _selector_priority_jitter_sort_key,
    _selector_sort_key,
    _selector_task_jitter_sort_key,
    _semantic_task_metrics,
    _semantic_policy_metrics,
    _semantic_policy_margin_metrics,
    _semantic_clearance_policy_metrics,
    _semantic_command_shape_for_variant,
    _low_small_foot_over_metrics,
    _rolling_segment_playback_error_metrics,
    _rolling_segment_terminal_trace_rows,
    _terminal_foot_anomaly_metrics,
    _semantic_probe_seed,
    _terrain_grid_world_xy_for_probe,
    _effective_planning_variant_for_semantic,
    _command_path_metrics,
    _semantic_collision_metrics,
    _loss_only_continuity_anchor_extra_loss,
    _loss_only_low_small_foot_over_extra_loss,
    _loss_only_weights_for_variant,
    _loss_only_high_large_avoid_extra_loss,
    _loss_only_low_small_crossing_extra_loss,
    _semantic_low_small_straight_extra_loss,
    _semantic_structural_extra_loss,
    _variant_cfg,
)
from mpc_swing_trajectory_quality_probe import _trajectory_summary
from extension.batch_mpc_planner.config import MpcPlannerCfg
from extension.batch_mpc_planner.types import MpcPlannerTerrain
from fixtures.viewer_runtime_diagnostics import _apply_semantic_small_height_override


def _terrain() -> MpcPlannerTerrain:
    height_map = torch.zeros((1, 3, 3), dtype=torch.float32)
    semantic_map = torch.zeros((1, 3, 3), dtype=torch.long)
    semantic_map[0, 1, 1] = 1
    semantic_map[0, 1, 2] = 2
    return MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )


def _decoded_for_structural_loss(*, root_xy: tuple[float, float], foot_xy: tuple[float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        root_pos=torch.tensor([[[root_xy[0], root_xy[1], 0.30], [root_xy[0], root_xy[1], 0.30]]], dtype=torch.float32),
        root_rpy=torch.zeros((1, 2, 3), dtype=torch.float32),
        foot_pos=torch.tensor(
            [
                [
                    [[foot_xy[0], foot_xy[1], 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
                    [[foot_xy[0], foot_xy[1], 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        contact_prob=torch.tensor([[[0.95, 0.0, 0.0, 0.0], [0.95, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        swing_prob=torch.tensor([[[0.05, 1.0, 1.0, 1.0], [0.05, 1.0, 1.0, 1.0]]], dtype=torch.float32),
        swing_center=torch.full((1, 4), 0.25, dtype=torch.float32),
        swing_width=torch.full((1, 4), 0.30, dtype=torch.float32),
    )


def test_structural_loss_penalizes_low_small_foot_without_body_avoidance() -> None:
    height_map = torch.zeros((1, 3, 3), dtype=torch.float32)
    semantic_map = torch.zeros((1, 3, 3), dtype=torch.long)
    semantic_map[0, 1, 1] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    decoded = _decoded_for_structural_loss(root_xy=(0.0, 0.0), foot_xy=(0.0, 0.0))

    per_env, breakdown = _semantic_structural_extra_loss(decoded, terrain)

    assert per_env.item() > 0.0
    assert breakdown["test_structural_low_small_foot"].item() > 0.0
    assert breakdown["test_structural_high_body"].item() == pytest.approx(0.0)


def test_low_small_straight_extra_loss_penalizes_detour_but_not_straight_overpass() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    root_rpy = torch.zeros((1, 5, 3), dtype=torch.float32)
    straight = torch.tensor(
        [[[-0.5, 0.0, 0.30], [-0.15, 0.0, 0.30], [0.20, 0.0, 0.30], [0.55, 0.0, 0.30], [0.90, 0.0, 0.30]]],
        dtype=torch.float32,
    )
    detour = straight.clone()
    detour[:, 1:4, 1] = 0.35

    straight_loss, straight_breakdown = _semantic_low_small_straight_extra_loss(
        straight,
        root_rpy,
        command,
        terrain,
    )
    detour_loss, detour_breakdown = _semantic_low_small_straight_extra_loss(
        detour,
        root_rpy,
        command,
        terrain,
    )

    assert straight_loss.item() == pytest.approx(0.0)
    assert straight_breakdown["test_low_small_straight_lateral"].item() == pytest.approx(0.0)
    assert detour_loss.item() > straight_loss.item() + 0.01
    assert detour_breakdown["test_low_small_straight_lateral"].item() > 0.0


def test_low_small_path_tube_loss_accepts_body_frame_mixed_curve_and_penalizes_detour() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.50, 0.25, 1.00]], dtype=torch.float32)
    dt = 0.02
    horizon = 80
    yaw = torch.arange(horizon, dtype=torch.float32) * dt * command[0, 2]
    body_step = command[0, :2] * dt
    cy = torch.cos(yaw[:-1])
    sy = torch.sin(yaw[:-1])
    world_step = torch.stack(
        (
            cy * body_step[0] - sy * body_step[1],
            sy * body_step[0] + cy * body_step[1],
        ),
        dim=-1,
    )
    root_xy = torch.zeros((horizon, 2), dtype=torch.float32)
    root_xy[:, 0] = -0.30
    root_xy[1:] = root_xy[:1] + torch.cumsum(world_step, dim=0)
    root = torch.cat((root_xy, torch.full((horizon, 1), 0.30)), dim=-1).unsqueeze(0)
    root_rpy = torch.zeros((1, horizon, 3), dtype=torch.float32)
    root_rpy[0, :, 2] = yaw
    weights = _LowSmallStraightWeights(
        lateral_weight=0.0,
        lateral_worst_weight=0.0,
        reverse_weight=0.0,
        progress_weight=0.0,
        root_clearance_weight=0.0,
        root_clearance_worst_weight=0.0,
        use_body_yaw_path=True,
        path_tube_weight=10.0,
        path_tube_worst_weight=10.0,
    )

    on_path_loss, on_path_breakdown = _semantic_low_small_straight_extra_loss(
        root,
        root_rpy,
        command,
        terrain,
        weights=weights,
    )
    detour = root.clone()
    detour[:, 20:45, 1] += 0.50
    detour_loss, detour_breakdown = _semantic_low_small_straight_extra_loss(
        detour,
        root_rpy,
        command,
        terrain,
        weights=weights,
    )

    assert on_path_loss.item() == pytest.approx(0.0)
    assert on_path_breakdown["test_low_small_path_tube"].item() == pytest.approx(0.0)
    assert detour_loss.item() > on_path_loss.item() + 0.01
    assert detour_breakdown["test_low_small_path_tube"].item() > 0.0


def test_loss_only_low_small_crossing_penalizes_detour_more_than_local_overpass() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    root_rpy = torch.zeros((1, 5, 3), dtype=torch.float32)
    root = torch.tensor(
        [[[-0.50, 0.0, 0.34], [-0.20, 0.0, 0.34], [0.10, 0.0, 0.34], [0.40, 0.0, 0.34], [0.70, 0.0, 0.34]]],
        dtype=torch.float32,
    )
    foot = torch.full((1, 5, 4, 3), -0.75, dtype=torch.float32)
    foot[..., 2] = 0.0
    decoded_overpass = SimpleNamespace(
        root_pos=root,
        root_rpy=root_rpy,
        foot_pos=foot,
        contact_prob=torch.zeros((1, 5, 4), dtype=torch.float32),
        swing_prob=torch.ones((1, 5, 4), dtype=torch.float32),
        swing_center=torch.full((1, 4), 0.25, dtype=torch.float32),
        swing_width=torch.full((1, 4), 0.30, dtype=torch.float32),
    )
    detour = SimpleNamespace(**{**decoded_overpass.__dict__, "root_pos": root.clone()})
    detour.root_pos[:, 1:4, 1] = 0.45

    overpass_loss, overpass_breakdown = _loss_only_low_small_crossing_extra_loss(
        decoded_overpass,
        command,
        terrain,
    )
    detour_loss, detour_breakdown = _loss_only_low_small_crossing_extra_loss(detour, command, terrain)

    assert overpass_breakdown["loss_only_low_small_path"].item() == pytest.approx(0.0)
    assert detour_breakdown["loss_only_low_small_path"].item() > 0.0
    assert detour_loss.item() > overpass_loss.item() + 0.01


def test_loss_only_low_small_foot_over_loss_penalizes_side_foot_detour() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    height_map[0, 2, 2] = 0.16
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    root = torch.tensor(
        [[[-0.50, 0.0, 0.34], [-0.20, 0.0, 0.34], [0.10, 0.0, 0.34], [0.40, 0.0, 0.34], [0.70, 0.0, 0.34]]],
        dtype=torch.float32,
    )
    root_rpy = torch.zeros((1, 5, 3), dtype=torch.float32)
    base_foot = torch.zeros((1, 5, 4, 3), dtype=torch.float32)
    base_foot[..., 0, :] = torch.tensor([0.0, 0.0, 0.24], dtype=torch.float32)
    base_foot[..., 1:, :] = torch.tensor([[-0.4, 0.30, 0.0], [-0.4, -0.30, 0.0], [-0.2, 0.25, 0.0]])
    over_foot = base_foot.clone()
    over_foot[:, 2, 0, :] = torch.tensor([0.0, 0.0, 0.26], dtype=torch.float32)
    side_foot = base_foot.clone()
    side_foot[:, :, 0, 1] = 0.18
    side_foot[:, :, 0, 2] = 0.26
    contact = torch.ones((1, 5, 4), dtype=torch.float32)
    contact[..., 0] = 0.0
    swing = 1.0 - contact
    common = dict(
        root_pos=root,
        root_rpy=root_rpy,
        contact_prob=contact,
        swing_prob=swing,
        swing_center=torch.full((1, 4), 0.25, dtype=torch.float32),
        swing_width=torch.full((1, 4), 0.30, dtype=torch.float32),
    )

    over_loss, over_breakdown = _loss_only_low_small_foot_over_extra_loss(
        SimpleNamespace(**common, foot_pos=over_foot),
        command,
        terrain,
    )
    side_loss, side_breakdown = _loss_only_low_small_foot_over_extra_loss(
        SimpleNamespace(**common, foot_pos=side_foot),
        command,
        terrain,
    )

    assert over_breakdown["loss_only_low_small_foot_over_xy"].item() < 0.01
    assert side_breakdown["loss_only_low_small_foot_over_xy"].item() > over_breakdown[
        "loss_only_low_small_foot_over_xy"
    ].item()
    assert side_loss.item() > over_loss.item() + 0.01


def test_loss_only_low_small_foot_over_window_penalizes_jump_into_target_window() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    height_map[0, 2, 2] = 0.16
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    root = torch.tensor(
        [[[-0.35, 0.0, 0.34], [-0.10, 0.0, 0.34], [0.10, 0.0, 0.34], [0.35, 0.0, 0.34]]],
        dtype=torch.float32,
    )
    common = dict(
        root_pos=root,
        root_rpy=torch.zeros((1, 4, 3), dtype=torch.float32),
        contact_prob=torch.zeros((1, 4, 4), dtype=torch.float32),
        swing_prob=torch.ones((1, 4, 4), dtype=torch.float32),
        swing_center=torch.full((1, 4), 0.25, dtype=torch.float32),
        swing_width=torch.full((1, 4), 0.30, dtype=torch.float32),
    )
    smooth_foot = torch.full((1, 4, 4, 3), -0.75, dtype=torch.float32)
    smooth_foot[..., 2] = 0.0
    smooth_foot[:, :, 0, :] = torch.tensor(
        [[[-0.04, 0.0, 0.24], [-0.01, 0.0, 0.25], [0.02, 0.0, 0.25], [0.05, 0.0, 0.24]]],
        dtype=torch.float32,
    )
    jump_foot = smooth_foot.clone()
    jump_foot[:, 0, 0, :] = torch.tensor([0.42, 0.0, 0.24], dtype=torch.float32)
    weights = _LossOnlySemanticWeights(
        low_small_foot_over=True,
        low_small_foot_over_xy_weight=0.0,
        low_small_foot_over_direct_xy_weight=0.0,
        low_small_foot_over_z_weight=0.0,
        low_small_foot_over_leg_weight=0.0,
        low_small_foot_over_ineligible_penalty=0.0,
        low_small_foot_over_time_gate_penalty=0.0,
        low_small_foot_over_window_weight=1.0,
        low_small_foot_over_window_min_count=1.0,
        low_small_foot_over_window_step_weight=1000.0,
        low_small_foot_over_window_step_cap_m=0.06,
        low_small_foot_over_window_accel_weight=0.0,
    )

    smooth_loss, _ = _loss_only_low_small_foot_over_extra_loss(
        SimpleNamespace(**common, foot_pos=smooth_foot),
        command,
        terrain,
        weights=weights,
    )
    jump_loss, _ = _loss_only_low_small_foot_over_extra_loss(
        SimpleNamespace(**common, foot_pos=jump_foot),
        command,
        terrain,
        weights=weights,
    )

    assert jump_loss.item() > smooth_loss.item() + 0.5


def test_loss_only_low_small_foot_over_path_curve_prefers_smooth_swing_across_obstacle() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    height_map[0, 2, 2] = 0.16
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    root = torch.tensor(
        [[[-0.35, 0.0, 0.34], [-0.10, 0.0, 0.34], [0.10, 0.0, 0.34], [0.35, 0.0, 0.34]]],
        dtype=torch.float32,
    )
    common = dict(
        root_pos=root,
        root_rpy=torch.zeros((1, 4, 3), dtype=torch.float32),
        contact_prob=torch.zeros((1, 4, 4), dtype=torch.float32),
        swing_prob=torch.ones((1, 4, 4), dtype=torch.float32),
        swing_center=torch.full((1, 4), 0.25, dtype=torch.float32),
        swing_width=torch.full((1, 4), 0.30, dtype=torch.float32),
    )
    smooth_foot = torch.full((1, 4, 4, 3), -0.75, dtype=torch.float32)
    smooth_foot[..., 2] = 0.0
    smooth_foot[:, :, 0, :] = torch.tensor(
        [[[-0.24, 0.0, 0.21], [-0.08, 0.0, 0.27], [0.08, 0.0, 0.27], [0.24, 0.0, 0.21]]],
        dtype=torch.float32,
    )
    center_jump_foot = smooth_foot.clone()
    center_jump_foot[:, :, 0, :] = torch.tensor(
        [[[-0.42, 0.0, 0.21], [0.00, 0.0, 0.27], [0.00, 0.0, 0.27], [0.42, 0.0, 0.21]]],
        dtype=torch.float32,
    )
    weights = _LossOnlySemanticWeights(
        low_small_foot_over=True,
        low_small_foot_over_xy_weight=0.0,
        low_small_foot_over_direct_xy_weight=0.0,
        low_small_foot_over_z_weight=0.0,
        low_small_foot_over_leg_weight=0.0,
        low_small_foot_over_ineligible_penalty=0.0,
        low_small_foot_over_time_gate_penalty=0.0,
        low_small_foot_over_path_curve_weight=100.0,
        low_small_foot_over_path_curve_z_weight=100.0,
        low_small_foot_over_path_curve_window_m=0.30,
    )

    smooth_loss, _ = _loss_only_low_small_foot_over_extra_loss(
        SimpleNamespace(**common, foot_pos=smooth_foot),
        command,
        terrain,
        weights=weights,
    )
    jump_loss, _ = _loss_only_low_small_foot_over_extra_loss(
        SimpleNamespace(**common, foot_pos=center_jump_foot),
        command,
        terrain,
        weights=weights,
    )

    assert jump_loss.item() > smooth_loss.item() + 0.5


def test_loss_only_low_small_foot_over_path_curve_body_yaw_matches_turning_swing() -> None:
    height_map = torch.zeros((1, 7, 7), dtype=torch.float32)
    height_map[0, 3, 3] = 0.16
    semantic_map = torch.zeros((1, 7, 7), dtype=torch.long)
    semantic_map[0, 3, 3] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-0.6, 0.6),
        world_y_range=(-0.6, 0.6),
    )
    command = torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32)
    yaw = torch.tensor([-0.45, -0.15, 0.15, 0.45], dtype=torch.float32)
    root_xy = torch.tensor([[-0.30, -0.10], [-0.10, -0.02], [0.10, 0.02], [0.30, 0.10]], dtype=torch.float32)
    root = torch.cat((root_xy, torch.full((4, 1), 0.34)), dim=-1).unsqueeze(0)
    root_rpy = torch.zeros((1, 4, 3), dtype=torch.float32)
    root_rpy[0, :, 2] = yaw
    foot = torch.full((1, 4, 4, 3), -0.75, dtype=torch.float32)
    foot[..., 2] = 0.0
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    curve_phase = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=torch.float32)
    foot_xy = curve_phase[:, None] * 0.30 * heading
    foot[:, :, 0, :2] = foot_xy
    foot[:, :, 0, 2] = torch.tensor([0.21, 0.27, 0.27, 0.21], dtype=torch.float32)
    common = dict(
        root_pos=root,
        root_rpy=root_rpy,
        foot_pos=foot,
        contact_prob=torch.zeros((1, 4, 4), dtype=torch.float32),
        swing_prob=torch.ones((1, 4, 4), dtype=torch.float32),
        swing_center=torch.full((1, 4), 0.25, dtype=torch.float32),
        swing_width=torch.full((1, 4), 0.30, dtype=torch.float32),
    )
    base_weights = _LossOnlySemanticWeights(
        low_small_foot_over=True,
        low_small_foot_over_xy_weight=0.0,
        low_small_foot_over_direct_xy_weight=0.0,
        low_small_foot_over_z_weight=0.0,
        low_small_foot_over_leg_weight=0.0,
        low_small_foot_over_ineligible_penalty=0.0,
        low_small_foot_over_time_gate_penalty=0.0,
        low_small_foot_over_path_curve_weight=100.0,
        low_small_foot_over_path_curve_z_weight=0.0,
        low_small_foot_over_path_curve_window_m=0.30,
    )

    fixed_loss, _ = _loss_only_low_small_foot_over_extra_loss(
        SimpleNamespace(**common),
        command,
        terrain,
        weights=base_weights,
    )
    yaw_loss, _ = _loss_only_low_small_foot_over_extra_loss(
        SimpleNamespace(**common),
        command,
        terrain,
        weights=replace(base_weights, low_small_foot_over_path_curve_body_yaw=True),
    )

    assert yaw_loss.item() < fixed_loss.item()


def test_probe_terrain_grid_world_xy_respects_sensor_pose() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 3, 3), dtype=torch.float32),
        semantic_map=torch.zeros((1, 3, 3), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
        sensor_pos_w=torch.tensor([[10.0, 20.0, 0.0]], dtype=torch.float32),
        sensor_yaw=torch.tensor([torch.pi / 2.0], dtype=torch.float32),
    )

    grid = _terrain_grid_world_xy_for_probe(terrain, dtype=torch.float32, device=torch.device("cpu")).reshape(1, 3, 3, 2)

    assert grid[0, 1, 1].tolist() == pytest.approx([10.0, 20.0])
    assert grid[0, 1, 2].tolist() == pytest.approx([10.0, 21.0])


def test_loss_only_high_large_avoid_penalizes_direct_path_more_than_bypass() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 2
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    root_rpy = torch.zeros((1, 5, 3), dtype=torch.float32)
    direct_root = torch.tensor(
        [[[-0.50, 0.0, 0.30], [-0.20, 0.0, 0.30], [0.10, 0.0, 0.30], [0.40, 0.0, 0.30], [0.70, 0.0, 0.30]]],
        dtype=torch.float32,
    )
    bypass_root = direct_root.clone()
    bypass_root[..., 1] = 0.55
    foot = torch.zeros((1, 5, 4, 3), dtype=torch.float32)
    decoded_direct = SimpleNamespace(
        root_pos=direct_root,
        root_rpy=root_rpy,
        foot_pos=foot,
        contact_prob=torch.zeros((1, 5, 4), dtype=torch.float32),
        swing_prob=torch.ones((1, 5, 4), dtype=torch.float32),
    )
    decoded_bypass = SimpleNamespace(**{**decoded_direct.__dict__, "root_pos": bypass_root})

    direct_loss, direct_breakdown = _loss_only_high_large_avoid_extra_loss(decoded_direct, command, terrain)
    bypass_loss, bypass_breakdown = _loss_only_high_large_avoid_extra_loss(decoded_bypass, command, terrain)

    assert direct_breakdown["loss_only_high_large_body"].item() > bypass_breakdown[
        "loss_only_high_large_body"
    ].item()
    assert direct_breakdown["loss_only_high_large_corridor"].item() > 0.0
    assert direct_loss.item() > bypass_loss.item() + 0.01


def test_loss_only_high_large_avoid_distance_margin_penalizes_near_bypass() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 2
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    root_rpy = torch.zeros((1, 5, 3), dtype=torch.float32)
    near_root = torch.tensor(
        [[[-0.50, 0.20, 0.30], [-0.20, 0.20, 0.30], [0.10, 0.20, 0.30], [0.40, 0.20, 0.30], [0.70, 0.20, 0.30]]],
        dtype=torch.float32,
    )
    far_root = near_root.clone()
    far_root[..., 1] = 0.55
    base = dict(
        root_rpy=root_rpy,
        foot_pos=torch.zeros((1, 5, 4, 3), dtype=torch.float32),
        contact_prob=torch.zeros((1, 5, 4), dtype=torch.float32),
        swing_prob=torch.ones((1, 5, 4), dtype=torch.float32),
    )
    weights = _LossOnlySemanticWeights(
        high_large_avoid=True,
        high_large_body_weight=0.0,
        high_large_body_worst_weight=0.0,
        high_large_corridor_weight=0.0,
        high_large_corridor_worst_weight=0.0,
        high_large_root_semantic_weight=0.0,
        high_large_root_semantic_worst_weight=0.0,
        high_large_lateral_escape_weight=0.0,
        high_large_distance_weight=20.0,
        high_large_distance_worst_weight=40.0,
        high_large_distance_margin_m=0.32,
    )

    near_loss, near_breakdown = _loss_only_high_large_avoid_extra_loss(
        SimpleNamespace(**base, root_pos=near_root),
        command,
        terrain,
        weights=weights,
    )
    far_loss, far_breakdown = _loss_only_high_large_avoid_extra_loss(
        SimpleNamespace(**base, root_pos=far_root),
        command,
        terrain,
        weights=weights,
    )

    assert near_breakdown["loss_only_high_large_distance_margin"].item() > 0.0
    assert far_breakdown["loss_only_high_large_distance_margin"].item() == pytest.approx(0.0)
    assert near_loss.item() > far_loss.item() + 0.01


def test_loss_only_high_large_scurve_accepts_either_side_bypass() -> None:
    height_map = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic_map = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic_map[0, 2, 2] = 2
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    x = torch.tensor([-0.50, -0.20, 0.10, 0.40, 0.70], dtype=torch.float32)
    direct = torch.stack((x, torch.zeros_like(x), torch.full_like(x, 0.30)), dim=-1).unsqueeze(0)
    left = direct.clone()
    left[..., 1] = 0.42
    right = direct.clone()
    right[..., 1] = -0.42
    base = dict(
        root_rpy=torch.zeros((1, 5, 3), dtype=torch.float32),
        foot_pos=torch.zeros((1, 5, 4, 3), dtype=torch.float32),
        contact_prob=torch.zeros((1, 5, 4), dtype=torch.float32),
        swing_prob=torch.ones((1, 5, 4), dtype=torch.float32),
    )
    weights = _LossOnlySemanticWeights(
        high_large_avoid=True,
        high_large_body_weight=0.0,
        high_large_body_worst_weight=0.0,
        high_large_corridor_weight=0.0,
        high_large_corridor_worst_weight=0.0,
        high_large_root_semantic_weight=0.0,
        high_large_root_semantic_worst_weight=0.0,
        high_large_lateral_escape_weight=0.0,
        high_large_scurve_weight=20.0,
        high_large_scurve_worst_weight=40.0,
        high_large_scurve_lateral_m=0.42,
    )

    direct_loss, direct_breakdown = _loss_only_high_large_avoid_extra_loss(
        SimpleNamespace(**base, root_pos=direct),
        command,
        terrain,
        weights=weights,
    )
    left_loss, left_breakdown = _loss_only_high_large_avoid_extra_loss(
        SimpleNamespace(**base, root_pos=left),
        command,
        terrain,
        weights=weights,
    )
    right_loss, right_breakdown = _loss_only_high_large_avoid_extra_loss(
        SimpleNamespace(**base, root_pos=right),
        command,
        terrain,
        weights=weights,
    )

    assert direct_breakdown["loss_only_high_large_scurve"].item() > 0.0
    assert left_breakdown["loss_only_high_large_scurve"].item() == pytest.approx(0.0)
    assert right_breakdown["loss_only_high_large_scurve"].item() == pytest.approx(0.0)
    assert direct_loss.item() > left_loss.item() + 0.01
    assert direct_loss.item() > right_loss.item() + 0.01


def test_nominal_command_shape_keeps_low_small_command_unchanged() -> None:
    terrain = _terrain()

    shaped, diagnostics = _semantic_command_shape_for_variant(
        "nominal_cmd_shape_a_v1",
        semantic_class="small",
        semantic_target_height=0.16,
        terrain=terrain,
        obstacle_xy=torch.tensor((0.0, 0.0), dtype=torch.float32),
        command=(0.50, 0.25, 1.00),
    )

    assert shaped == pytest.approx((0.50, 0.25, 1.00))
    assert diagnostics["command_shaped"] == 0
    assert diagnostics["command_shape_reason"] == "low_small_cross"


def test_nominal_command_shape_reduces_forward_and_adds_lateral_for_high_or_large() -> None:
    terrain = _terrain()

    high_small, high_small_diag = _semantic_command_shape_for_variant(
        "nominal_cmd_shape_a_v1",
        semantic_class="small",
        semantic_target_height=0.46,
        terrain=terrain,
        obstacle_xy=torch.tensor((0.0, 0.0), dtype=torch.float32),
        command=(0.50, 0.00, 0.00),
    )
    large, large_diag = _semantic_command_shape_for_variant(
        "nominal_cmd_shape_a_v1",
        semantic_class="large",
        semantic_target_height=0.55,
        terrain=terrain,
        obstacle_xy=torch.tensor((0.0, 0.0), dtype=torch.float32),
        command=(0.50, 0.00, 0.00),
    )

    assert high_small[0] < 0.50
    assert abs(high_small[1]) >= 0.25
    assert high_small_diag["command_shaped"] == 1
    assert high_small_diag["command_shape_reason"] == "avoid_high_or_large"
    assert large[0] < 0.50
    assert abs(large[1]) >= 0.25
    assert large_diag["command_shaped"] == 1


def test_nominal_command_shape_conservative_variant_uses_smaller_lateral_escape() -> None:
    terrain = _terrain()

    default, _ = _semantic_command_shape_for_variant(
        "nominal_cmd_shape_a_v1",
        semantic_class="large",
        semantic_target_height=0.55,
        terrain=terrain,
        obstacle_xy=torch.tensor((0.0, 0.0), dtype=torch.float32),
        command=(0.50, 0.00, 0.00),
    )
    conservative, conservative_diag = _semantic_command_shape_for_variant(
        "nominal_cmd_shape_a_conservative_v4",
        semantic_class="large",
        semantic_target_height=0.55,
        terrain=terrain,
        obstacle_xy=torch.tensor((0.0, 0.0), dtype=torch.float32),
        command=(0.50, 0.00, 0.00),
    )

    assert conservative_diag["command_shaped"] == 1
    assert conservative[0] > default[0]
    assert abs(conservative[1]) < abs(default[1])
    assert abs(conservative[1]) >= 0.20


def test_nominal_command_shape_low_exact_variant_reuses_exact_low_small_loss_cfg() -> None:
    base = MpcPlannerCfg()

    exact = _variant_cfg(base, "nominal_cmd_shape_a_low_exact_v4")
    low_small = _variant_cfg(base, "loss_low_small_cont_v2")

    assert exact.runtime.optimize_steps == low_small.runtime.optimize_steps
    assert exact.runtime.lr == low_small.runtime.lr
    assert exact.losses.foot_trajectory_regularization.accel_weight == low_small.losses.foot_trajectory_regularization.accel_weight


def test_nominal_command_shape_low_accel_variants_strengthen_continuity_cfg() -> None:
    base = MpcPlannerCfg()

    exact = _variant_cfg(base, "nominal_cmd_shape_a_low_exact_v4")
    accel = _variant_cfg(base, "nominal_cmd_shape_a_low_accel_v5")
    anchor = _variant_cfg(base, "nominal_cmd_shape_a_low_accel_anchor_v5")

    assert accel.runtime.optimize_steps > exact.runtime.optimize_steps
    assert accel.runtime.lr < exact.runtime.lr
    assert accel.losses.foot_trajectory_regularization.accel_weight > exact.losses.foot_trajectory_regularization.accel_weight
    assert anchor.runtime.optimize_steps >= accel.runtime.optimize_steps
    assert anchor.losses.foot_trajectory_regularization.boundary_weight >= accel.losses.foot_trajectory_regularization.boundary_weight


def test_nominal_command_shape_combined_v6_uses_anchor_for_low_small_and_conservative_escape() -> None:
    base = MpcPlannerCfg()
    terrain = _terrain()

    combined_cfg = _variant_cfg(base, "nominal_cmd_shape_a_combined_v6")
    anchor_cfg = _variant_cfg(base, "nominal_cmd_shape_a_low_accel_anchor_v5")
    shaped, diagnostics = _semantic_command_shape_for_variant(
        "nominal_cmd_shape_a_combined_v6",
        semantic_class="large",
        semantic_target_height=0.55,
        terrain=terrain,
        obstacle_xy=torch.tensor((0.0, 0.0), dtype=torch.float32),
        command=(0.50, 0.00, 0.00),
    )

    assert combined_cfg.runtime.optimize_steps == anchor_cfg.runtime.optimize_steps
    assert diagnostics["command_shaped"] == 1
    assert shaped[0] > 0.25
    assert 0.20 <= abs(shaped[1]) < 0.30


def test_nominal_command_shape_combined_v7_routes_low_small_loss_only_to_low_small() -> None:
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v7",
        semantic_class="small",
        semantic_target_height=0.16,
    ) == "nominal_cmd_shape_a_low_accel_anchor_v5"
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v7",
        semantic_class="small",
        semantic_target_height=0.46,
    ) == "nominal_cmd_shape_a_conservative_v4"
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v7",
        semantic_class="large",
        semantic_target_height=0.55,
    ) == "nominal_cmd_shape_a_conservative_v4"


def test_nominal_command_shape_combined_v8_routes_low_small_to_proven_v2_loss() -> None:
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v8",
        semantic_class="small",
        semantic_target_height=0.16,
    ) == "loss_low_small_cont_v2"
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v8",
        semantic_class="large",
        semantic_target_height=0.55,
    ) == "nominal_cmd_shape_a_conservative_v4"


def test_nominal_command_shape_combined_v9_routes_low_small_to_stepcap_and_large_to_escape() -> None:
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v9",
        semantic_class="small",
        semantic_target_height=0.16,
    ) == "loss_low_small_stepcap_v4"
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v9",
        semantic_class="small",
        semantic_target_height=0.46,
    ) == "nominal_cmd_shape_a_conservative_v4"
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v9",
        semantic_class="large",
        semantic_target_height=0.55,
    ) == "nominal_cmd_shape_a_conservative_v4"


def test_nominal_command_shape_combined_v10_routes_low_small_by_command_type() -> None:
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v10",
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.50, 0.00, 0.00),
    ) == "struct_lowfoot_cross_hard"
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v10",
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.50, 0.25, 1.00),
    ) == "loss_low_small_stepcap_v4"
    assert _effective_planning_variant_for_semantic(
        "nominal_cmd_shape_a_combined_v10",
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.50, 0.00, 0.00),
    ) == "nominal_cmd_shape_a_conservative_v4"


def test_nominal_command_shape_picks_side_with_less_semantic_occupancy() -> None:
    height_map = torch.zeros((1, 7, 7), dtype=torch.float32)
    semantic_map = torch.zeros((1, 7, 7), dtype=torch.long)
    semantic_map[0, 4:6, 3:5] = 2
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )

    shaped, diagnostics = _semantic_command_shape_for_variant(
        "nominal_cmd_shape_a_v1",
        semantic_class="large",
        semantic_target_height=0.55,
        terrain=terrain,
        obstacle_xy=torch.tensor((0.0, 0.0), dtype=torch.float32),
        command=(0.50, 0.00, 0.00),
    )

    assert diagnostics["command_shape_side"] == -1
    assert shaped[1] < 0.0
    assert diagnostics["command_shape_left_score"] > diagnostics["command_shape_right_score"]


def test_semantic_probe_seed_ignores_display_variant_and_uses_effective_candidate() -> None:
    direct = _semantic_probe_seed(
        semantic_class="small",
        command_name="forward_yaw_v050_vy025_yaw100",
        cycle=0,
        effective_candidate="loss_low_small_cont_v2",
    )
    routed = _semantic_probe_seed(
        semantic_class="small",
        command_name="forward_yaw_v050_vy025_yaw100",
        cycle=0,
        effective_candidate=_effective_planning_variant_for_semantic(
            "nominal_cmd_shape_a_combined_v8",
            semantic_class="small",
            semantic_target_height=0.16,
        ),
    )
    different_candidate = _semantic_probe_seed(
        semantic_class="small",
        command_name="forward_yaw_v050_vy025_yaw100",
        cycle=0,
        effective_candidate="nominal_cmd_shape_a_low_accel_anchor_v5",
    )

    assert direct == routed
    assert different_candidate != direct


def test_loss_only_continuity_anchor_penalizes_early_foot_jump() -> None:
    state = SimpleNamespace(foot_pos=torch.zeros((1, 4, 3), dtype=torch.float32))
    root = torch.zeros((1, 6, 3), dtype=torch.float32)
    smooth_foot = torch.zeros((1, 6, 4, 3), dtype=torch.float32)
    jump_foot = smooth_foot.clone()
    jump_foot[:, 1:3, :, 0] = 0.40
    decoded_smooth = SimpleNamespace(root_pos=root, foot_pos=smooth_foot, contact_prob=torch.ones((1, 6, 4)))
    decoded_jump = SimpleNamespace(root_pos=root, foot_pos=jump_foot, contact_prob=torch.ones((1, 6, 4)))

    smooth_loss, smooth_breakdown = _loss_only_continuity_anchor_extra_loss(decoded_smooth, state)
    jump_loss, jump_breakdown = _loss_only_continuity_anchor_extra_loss(decoded_jump, state)

    assert smooth_breakdown["loss_only_first_foot_anchor"].item() == pytest.approx(0.0)
    assert jump_breakdown["loss_only_first_foot_anchor"].item() > 0.0
    assert jump_loss.item() > smooth_loss.item() + 0.01


def test_low_small_stepcap_variants_increase_worst_step_and_accel_penalties() -> None:
    v2 = _variant_cfg(MpcPlannerCfg(), "loss_low_small_cont_v2")
    v3 = _variant_cfg(MpcPlannerCfg(), "loss_low_small_stepcap_v3")
    v4 = _variant_cfg(MpcPlannerCfg(), "loss_low_small_stepcap_v4")

    assert v3.runtime.lr < v2.runtime.lr
    assert v3.runtime.optimize_steps > v2.runtime.optimize_steps
    assert v3.losses.foot_trajectory_regularization.boundary_weight > v2.losses.foot_trajectory_regularization.boundary_weight
    assert v4.losses.foot_trajectory_regularization.boundary_weight > v3.losses.foot_trajectory_regularization.boundary_weight

    weights_v2 = _loss_only_weights_for_variant("loss_low_small_cont_v2")
    weights_v3 = _loss_only_weights_for_variant("loss_low_small_stepcap_v3")
    weights_v4 = _loss_only_weights_for_variant("loss_low_small_stepcap_v4")
    assert weights_v2 is not None and weights_v3 is not None and weights_v4 is not None
    assert weights_v3.foot_step_worst_weight > weights_v2.foot_step_worst_weight
    assert weights_v4.foot_step_worst_weight > weights_v3.foot_step_worst_weight
    assert weights_v4.first_foot_anchor_weight < weights_v2.first_foot_anchor_weight


def test_command_path_metrics_accept_body_frame_mixed_curve_and_reject_detour() -> None:
    command = (0.50, 0.25, 1.00)
    dt = 0.02
    horizon = 50
    yaw = torch.arange(horizon, dtype=torch.float32) * dt * command[2]
    cy = torch.cos(yaw[:-1])
    sy = torch.sin(yaw[:-1])
    body_step = torch.tensor(command[:2], dtype=torch.float32) * dt
    world_step = torch.stack(
        (
            cy * body_step[0] - sy * body_step[1],
            sy * body_step[0] + cy * body_step[1],
        ),
        dim=-1,
    )
    root_xy = torch.zeros((horizon, 2), dtype=torch.float32)
    root_xy[1:] = torch.cumsum(world_step, dim=0)
    root = torch.cat((root_xy, torch.full((horizon, 1), 0.30)), dim=-1).unsqueeze(0)
    root_rpy = torch.zeros((1, horizon, 3), dtype=torch.float32)
    root_rpy[0, :, 2] = yaw
    on_path = _command_path_metrics(root, root_rpy, command, dt=dt)

    detour = root.clone()
    detour[:, 10:30, 1] += 0.60
    off_path = _command_path_metrics(detour, root_rpy, command, dt=dt)

    assert on_path["command_path_lateral_error_max"] < 1.0e-5
    assert off_path["command_path_lateral_error_max"] > 0.50
    assert off_path["command_path_lateral_error_max"] > on_path["command_path_lateral_error_max"]


def test_command_path_metrics_are_relative_to_nonzero_start() -> None:
    command = (0.50, 0.00, 0.00)
    dt = 0.02
    horizon = 20
    root_xy = torch.zeros((horizon, 2), dtype=torch.float32)
    root_xy[:, 0] = 4.0 + torch.arange(horizon, dtype=torch.float32) * command[0] * dt
    root_xy[:, 1] = -3.0
    root = torch.cat((root_xy, torch.full((horizon, 1), 0.30)), dim=-1).unsqueeze(0)
    root_rpy = torch.zeros((1, horizon, 3), dtype=torch.float32)

    metrics = _command_path_metrics(root, root_rpy, command, dt=dt)

    assert metrics["command_path_lateral_error_max"] < 1.0e-5
    assert metrics["command_path_lateral_error_mean"] < 1.0e-5


def test_crossing_metrics_detect_ever_crossed_even_if_long_horizon_returns_front() -> None:
    obstacle_xy = torch.tensor((0.0, 0.0), dtype=torch.float32)
    root = torch.tensor(
        [[[-0.30, 0.0, 0.30], [0.05, 0.0, 0.30], [0.20, 0.0, 0.30], [-0.05, 0.0, 0.30]]],
        dtype=torch.float32,
    )

    metrics = _crossing_metrics(root, obstacle_xy, (0.50, 0.00, 0.00))

    assert metrics["crossed_obstacle_along_command"] == 0
    assert metrics["ever_crossed_obstacle_along_command"] == 1
    assert metrics["max_along_obstacle"] > 0.0


def test_structural_loss_penalizes_high_small_and_large_body_field() -> None:
    height_map = torch.zeros((1, 3, 3), dtype=torch.float32)
    height_map[0, 1, 1] = 0.46
    semantic_map = torch.zeros((1, 3, 3), dtype=torch.long)
    semantic_map[0, 1, 1] = 1
    semantic_map[0, 1, 2] = 2
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    decoded = _decoded_for_structural_loss(root_xy=(0.0, 0.0), foot_xy=(-1.0, -1.0))

    per_env, breakdown = _semantic_structural_extra_loss(decoded, terrain)

    assert per_env.item() > 0.0
    assert breakdown["test_structural_high_body"].item() > 0.0
    assert breakdown["test_structural_low_small_foot"].item() == pytest.approx(0.0)


def test_structural_large_only_weights_skip_high_small_body_field() -> None:
    height_map = torch.zeros((1, 3, 3), dtype=torch.float32)
    height_map[0, 1, 1] = 0.46
    semantic_map = torch.zeros((1, 3, 3), dtype=torch.long)
    semantic_map[0, 1, 1] = 1
    terrain = MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=semantic_map,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    decoded = _decoded_for_structural_loss(root_xy=(0.0, 0.0), foot_xy=(-1.0, -1.0))

    _, breakdown = _semantic_structural_extra_loss(
        decoded,
        terrain,
        weights=_StructuralSemanticWeights(include_high_small_body=False, include_large_body=True),
    )

    assert breakdown["test_structural_high_body"].item() == pytest.approx(0.0)




def test_semantic_policy_metrics_encode_crossing_requirement_by_class_and_height() -> None:
    low_small = _semantic_policy_metrics(
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=1,
    )
    assert low_small["desired_crossing"] == 1
    assert low_small["semantic_policy_violation"] == 0

    missed_low_small = _semantic_policy_metrics(
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=0,
    )
    assert missed_low_small["semantic_policy_violation"] == 1

    large_crossed = _semantic_policy_metrics(
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
        crossed=1,
    )
    assert large_crossed["desired_crossing"] == 0
    assert large_crossed["semantic_policy_violation"] == 1

    yaw_only = _semantic_policy_metrics(
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.0, 0.0, 1.0),
        crossed=0,
    )
    assert yaw_only["desired_crossing"] == 0
    assert yaw_only["semantic_policy_violation"] == 0


def test_selector_variant_expands_to_baseline_and_structural_candidates() -> None:
    assert _candidate_variants_for_variant("select_baseline_gentle_smooth") == (
        "baseline",
        "struct_lowfoot_largebody_gentle_smooth",
    )
    assert _candidate_variants_for_variant("select_policy_pool") == (
        "baseline",
        "struct_lowfoot_largebody_gentle_smooth",
        "struct_lowfoot_highbody",
    )
    assert "body_stance_crossing" in _candidate_variants_for_variant(
        "select_policy_class_wide_margin",
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
    )
    assert "body_stance_crossing" not in _candidate_variants_for_variant(
        "select_policy_class_wide_margin",
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
    )
    assert "struct_lowfoot_cross_hard" in _candidate_variants_for_variant(
        "select_policy_class_hardcross_margin",
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
    )
    assert "struct_lowfoot_cross_hard" not in _candidate_variants_for_variant(
        "select_policy_class_hardcross_margin",
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
    )
    assert _candidate_variants_for_variant(
        "select_policy_class_jitter_margin",
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
    ) == _candidate_variants_for_variant(
        "select_policy_class_wide_margin",
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
    )
    assert "risk_strong" in _candidate_variants_for_variant(
        "select_policy_class_risk_jitter_margin",
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
    )
    assert "risk_strong" not in _candidate_variants_for_variant(
        "select_policy_class_risk_jitter_margin",
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
    )
    assert "risk_strong" in _candidate_variants_for_variant(
        "select_policy_class_priority_jitter_margin",
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
    )
    assert "risk_strong" in _candidate_variants_for_variant(
        "select_policy_class_clearance_jitter_margin",
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
    )
    assert _candidate_variants_for_variant("baseline") == ("baseline",)


def test_policy_margin_metrics_penalize_near_large_or_high_small_avoidance() -> None:
    near_large = _semantic_policy_margin_metrics(
        semantic_class="large",
        semantic_target_diameter=0.45,
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
        crossed=0,
        min_root_distance=0.10,
    )
    assert near_large["semantic_policy_margin_deficit"] > 0.0

    far_large = _semantic_policy_margin_metrics(
        semantic_class="large",
        semantic_target_diameter=0.45,
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
        crossed=0,
        min_root_distance=0.35,
    )
    assert far_large["semantic_policy_margin_deficit"] == pytest.approx(0.0)

    low_small_cross = _semantic_policy_margin_metrics(
        semantic_class="small",
        semantic_target_diameter=0.12,
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=1,
        min_root_distance=0.02,
    )
    assert low_small_cross["semantic_policy_margin_deficit"] == pytest.approx(0.0)


def test_clearance_policy_allows_large_bypass_but_rejects_low_small_miss_and_near_avoidance() -> None:
    large_clear_bypass = _semantic_clearance_policy_metrics(
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
        crossed=1,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
    )
    assert large_clear_bypass["semantic_clearance_policy_violation"] == 0

    large_near = _semantic_clearance_policy_metrics(
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
        crossed=0,
        semantic_policy_margin_deficit=0.05,
        stance_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
    )
    assert large_near["semantic_clearance_policy_violation"] == 1

    large_tiny_swing_penetration = _semantic_clearance_policy_metrics(
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
        crossed=0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.000833,
    )
    assert large_tiny_swing_penetration["semantic_clearance_policy_violation"] == 0

    large_repeated_penetration = _semantic_clearance_policy_metrics(
        semantic_class="large",
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
        crossed=0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.01,
    )
    assert large_repeated_penetration["semantic_clearance_policy_violation"] == 1

    low_small_missed = _semantic_clearance_policy_metrics(
        semantic_class="small",
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
    )
    assert low_small_missed["semantic_clearance_policy_violation"] == 1


def test_semantic_task_metrics_require_low_small_straight_overpass_not_just_reaching_back_side() -> None:
    straight_overpass = _semantic_task_metrics(
        semantic_class="small",
        semantic_target_diameter=0.12,
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=1,
        max_abs_lateral_to_obstacle=0.03,
        min_abs_lateral_to_obstacle=0.03,
        root_lateral_deviation_from_start_max=0.03,
        root_along_reverse_rate=0.0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        touchdown_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
        foot_accel_max_to_mean=8.0,
        root_accel_max_to_mean=6.0,
        worst_max_to_median_step=5.0,
        worst_boundary_to_median_step=3.0,
    )
    assert straight_overpass["small_overpass_success"] == 1
    assert straight_overpass["semantic_task_violation"] == 0

    reached_back_by_detour = _semantic_task_metrics(
        semantic_class="small",
        semantic_target_diameter=0.12,
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=1,
        max_abs_lateral_to_obstacle=0.31,
        min_abs_lateral_to_obstacle=0.31,
        root_lateral_deviation_from_start_max=0.31,
        root_along_reverse_rate=0.0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        touchdown_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
        foot_accel_max_to_mean=8.0,
        root_accel_max_to_mean=6.0,
        worst_max_to_median_step=5.0,
        worst_boundary_to_median_step=3.0,
    )
    assert reached_back_by_detour["small_overpass_success"] == 0
    assert reached_back_by_detour["semantic_task_violation"] == 1


def test_semantic_task_metrics_require_foot_to_pass_over_low_small_not_around_it() -> None:
    no_foot_over = _semantic_task_metrics(
        semantic_class="small",
        semantic_target_diameter=0.12,
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=1,
        max_abs_lateral_to_obstacle=0.03,
        min_abs_lateral_to_obstacle=0.03,
        root_lateral_deviation_from_start_max=0.03,
        root_along_reverse_rate=0.0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        touchdown_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
        foot_accel_max_to_mean=8.0,
        root_accel_max_to_mean=6.0,
        worst_max_to_median_step=5.0,
        worst_boundary_to_median_step=3.0,
        foot_over_low_small_success=0,
    )
    assert no_foot_over["small_overpass_success"] == 0
    assert no_foot_over["semantic_task_violation"] == 1

    foot_over = dict(no_foot_over)
    foot_over.update(
        _semantic_task_metrics(
            semantic_class="small",
            semantic_target_diameter=0.12,
            semantic_target_height=0.16,
            command=(0.5, 0.0, 0.0),
            crossed=1,
            max_abs_lateral_to_obstacle=0.03,
            min_abs_lateral_to_obstacle=0.03,
            root_lateral_deviation_from_start_max=0.03,
            root_along_reverse_rate=0.0,
            semantic_policy_margin_deficit=0.0,
            stance_on_semantic_rate=0.0,
            touchdown_on_semantic_rate=0.0,
            root_on_semantic_rate=0.0,
            foot_semantic_penetration_rate=0.0,
            foot_accel_max_to_mean=8.0,
            root_accel_max_to_mean=6.0,
            worst_max_to_median_step=5.0,
            worst_boundary_to_median_step=3.0,
            foot_over_low_small_success=1,
        )
    )
    assert foot_over["small_overpass_success"] == 1
    assert foot_over["semantic_task_violation"] == 0

    jumpy_metrics = _semantic_task_metrics(
        semantic_class="small",
        semantic_target_diameter=0.12,
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=1,
        max_abs_lateral_to_obstacle=0.03,
        min_abs_lateral_to_obstacle=0.03,
        root_lateral_deviation_from_start_max=0.03,
        root_along_reverse_rate=0.0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        touchdown_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
        foot_accel_max_to_mean=8.0,
        root_accel_max_to_mean=6.0,
        worst_max_to_median_step=5.0,
        worst_boundary_to_median_step=3.0,
        foot_over_low_small_success=1,
        foot_step_anomaly_flag=1,
    )
    assert jumpy_metrics["small_overpass_success"] == 0
    assert jumpy_metrics["semantic_task_continuity_violation"] == 1
    assert jumpy_metrics["semantic_task_violation"] == 1


def test_low_small_foot_over_metrics_distinguish_overpass_from_side_detour() -> None:
    class Result:
        pass

    terrain = MpcPlannerTerrain(
        height_map=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.16, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32),
        semantic_map=torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=torch.long),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    obstacle_xy = torch.tensor((0.0, 0.0), dtype=torch.float32)

    over = Result()
    over.foot_pos_w = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.24], [0.3, 0.3, 0.0], [0.3, -0.3, 0.0], [-0.3, 0.3, 0.0]],
                [[0.05, 0.0, 0.25], [0.3, 0.3, 0.0], [0.3, -0.3, 0.0], [-0.3, 0.3, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    over.contact_state = torch.tensor([[[False, True, True, True], [False, True, True, True]]])

    side = Result()
    side.foot_pos_w = over.foot_pos_w.clone()
    side.foot_pos_w[..., 0, 1] = 0.25
    side.contact_state = over.contact_state

    over_metrics = _low_small_foot_over_metrics(over, terrain, obstacle_xy, semantic_target_height=0.16)
    side_metrics = _low_small_foot_over_metrics(side, terrain, obstacle_xy, semantic_target_height=0.16)

    assert over_metrics["foot_over_low_small_success"] == 1
    assert over_metrics["foot_over_low_small_frame_count"] == 2
    assert side_metrics["foot_over_low_small_success"] == 0
    assert side_metrics["foot_over_low_small_frame_count"] == 0


def test_terminal_foot_anomaly_metrics_flag_last_frame_jump_and_airborne_stance() -> None:
    class Result:
        pass

    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 3, 3), dtype=torch.float32),
        semantic_map=torch.zeros((1, 3, 3), dtype=torch.long),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    result = Result()
    foot = torch.zeros((1, 5, 4, 3), dtype=torch.float32)
    foot[:, :, :, 2] = 0.0
    foot[0, 1:4, 0, 0] = torch.tensor([0.02, 0.04, 0.06])
    foot[0, 4, 0, 0] = 0.42
    foot[0, 4, 0, 2] = 0.18
    result.foot_pos_w = foot
    result.contact_state = torch.ones((1, 5, 4), dtype=torch.bool)

    metrics = _terminal_foot_anomaly_metrics(result, terrain)

    assert metrics["terminal_foot_step_max"] == pytest.approx(0.4025, abs=1.0e-3)
    assert metrics["terminal_foot_step_leg"] == 0
    assert metrics["terminal_foot_step_to_median"] > 10.0
    assert metrics["any_foot_step_max"] == pytest.approx(metrics["terminal_foot_step_max"])
    assert metrics["any_foot_step_frame"] == 3
    assert metrics["any_foot_step_leg"] == 0
    assert metrics["any_foot_step_to_median"] > 10.0
    assert metrics["terminal_stance_airborne_max"] == pytest.approx(0.18)
    assert metrics["terminal_stance_airborne_leg"] == 0
    assert metrics["foot_step_anomaly_flag"] == 1
    assert metrics["terminal_foot_anomaly_flag"] == 1


def test_terminal_foot_anomaly_metrics_flag_mid_trajectory_jump() -> None:
    class Result:
        pass

    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 3, 3), dtype=torch.float32),
        semantic_map=torch.zeros((1, 3, 3), dtype=torch.long),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    result = Result()
    foot = torch.zeros((1, 5, 4, 3), dtype=torch.float32)
    foot[0, 1, 0, 0] = 0.02
    foot[0, 2, 0, 0] = 0.40
    foot[0, 3, 0, 0] = 0.42
    foot[0, 4, 0, 0] = 0.44
    result.foot_pos_w = foot
    result.contact_state = torch.ones((1, 5, 4), dtype=torch.bool)

    metrics = _terminal_foot_anomaly_metrics(result, terrain)

    assert metrics["any_foot_step_frame"] == 1
    assert metrics["any_foot_step_leg"] == 0
    assert metrics["any_foot_step_max"] == pytest.approx(0.38, abs=1.0e-6)
    assert metrics["foot_step_anomaly_flag"] == 1
    assert metrics["terminal_foot_anomaly_flag"] == 0


def test_terminal_foot_anomaly_metrics_flag_replan_boundary_jump() -> None:
    class Result:
        pass

    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 3, 3), dtype=torch.float32),
        semantic_map=torch.zeros((1, 3, 3), dtype=torch.long),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    result = Result()
    foot = torch.zeros((1, 7, 4, 3), dtype=torch.float32)
    foot[0, 1:4, 0, 0] = torch.tensor([0.02, 0.04, 0.06])
    foot[0, 4, 0, 0] = 0.42
    foot[0, 5, 0, 0] = 0.44
    foot[0, 6, 0, 0] = 0.46
    result.foot_pos_w = foot
    result.contact_state = torch.ones((1, 7, 4), dtype=torch.bool)

    metrics = _terminal_foot_anomaly_metrics(result, terrain, replan_interval_steps=4)

    assert metrics["replan_boundary_foot_step_frame"] == 3
    assert metrics["replan_boundary_foot_step_leg"] == 0
    assert metrics["replan_boundary_foot_step_max"] == pytest.approx(0.36, abs=1.0e-6)
    assert metrics["replan_boundary_foot_step_to_median"] > 10.0
    assert metrics["replan_boundary_foot_anomaly_flag"] == 1


def test_rolling_segment_playback_error_metrics_report_worst_segment_and_leg() -> None:
    result = SimpleNamespace(
        rolling_segment_terminal_foot_error=torch.tensor(
            [[0.01, 0.02, 0.03, 0.04], [0.08, 0.25, 0.03, 0.02]], dtype=torch.float32
        ),
        rolling_segment_terminal_root_error=torch.tensor([0.001, 0.004], dtype=torch.float32),
    )

    metrics = _rolling_segment_playback_error_metrics(result)

    assert metrics["rolling_segment_terminal_foot_error_max"] == pytest.approx(0.25)
    assert metrics["rolling_segment_terminal_foot_error_segment"] == 1
    assert metrics["rolling_segment_terminal_foot_error_leg"] == 1
    assert metrics["rolling_segment_terminal_root_error_max"] == pytest.approx(0.004)


def test_rolling_segment_terminal_trace_rows_include_worst_leg_positions() -> None:
    traces = (
        {
            "segment": 0,
            "planned_foot": torch.tensor([[[0.0, 0.0, 0.0], [0.20, 0.0, 0.0]]], dtype=torch.float32),
            "actual_foot": torch.tensor([[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]], dtype=torch.float32),
            "planned_root": torch.tensor([[1.0, 2.0, 0.3]], dtype=torch.float32),
            "actual_root": torch.tensor([[1.0, 2.0, 0.3]], dtype=torch.float32),
            "planned_joint": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            "actual_joint": torch.tensor([[0.1, 0.25]], dtype=torch.float32),
            "internal_fk_foot": torch.tensor([[[0.0, 0.0, 0.0], [0.24, 0.0, 0.0]]], dtype=torch.float32),
            "planned_touchdown": torch.tensor([[[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]]], dtype=torch.float32),
            "contact_state": torch.tensor([[True, False]]),
            "frame": 24,
        },
    )

    rows = _rolling_segment_terminal_trace_rows(traces)

    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "rolling_terminal_trace"
    assert row["segment"] == 0
    assert row["worst_leg"] == 1
    assert row["foot_error_norm"] == pytest.approx(0.05)
    assert row["joint_error_max_abs"] == pytest.approx(0.05)
    assert row["planned_foot_xyz"] == pytest.approx([0.20, 0.0, 0.0])
    assert row["actual_foot_xyz"] == pytest.approx([0.25, 0.0, 0.0])
    assert row["internal_fk_foot_xyz"] == pytest.approx([0.24, 0.0, 0.0])
    assert row["internal_fk_error_norm"] == pytest.approx(0.04)
    assert row["actual_vs_internal_fk_error_norm"] == pytest.approx(0.01)
    assert row["planned_touchdown_xyz"] == pytest.approx([0.4, 0.0, 0.0])
    assert row["contact_state"] == 0


def test_semantic_task_metrics_allow_large_bypass_and_reject_discontinuous_small_overpass() -> None:
    large_bypass = _semantic_task_metrics(
        semantic_class="large",
        semantic_target_diameter=0.45,
        semantic_target_height=0.55,
        command=(0.5, 0.0, 0.0),
        crossed=1,
        max_abs_lateral_to_obstacle=0.35,
        min_abs_lateral_to_obstacle=0.35,
        root_lateral_deviation_from_start_max=0.35,
        root_along_reverse_rate=0.0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        touchdown_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
        foot_accel_max_to_mean=8.0,
        root_accel_max_to_mean=6.0,
        worst_max_to_median_step=5.0,
        worst_boundary_to_median_step=3.0,
    )
    assert large_bypass["large_avoid_success"] == 1
    assert large_bypass["semantic_task_violation"] == 0

    jittery_small = _semantic_task_metrics(
        semantic_class="small",
        semantic_target_diameter=0.12,
        semantic_target_height=0.16,
        command=(0.5, 0.0, 0.0),
        crossed=1,
        max_abs_lateral_to_obstacle=0.03,
        min_abs_lateral_to_obstacle=0.03,
        root_lateral_deviation_from_start_max=0.03,
        root_along_reverse_rate=0.0,
        semantic_policy_margin_deficit=0.0,
        stance_on_semantic_rate=0.0,
        touchdown_on_semantic_rate=0.0,
        root_on_semantic_rate=0.0,
        foot_semantic_penetration_rate=0.0,
        foot_accel_max_to_mean=45.0,
        root_accel_max_to_mean=6.0,
        worst_max_to_median_step=5.0,
        worst_boundary_to_median_step=3.0,
    )
    assert jittery_small["small_overpass_success"] == 0
    assert jittery_small["semantic_task_violation"] == 1


def test_selector_sort_key_prefers_policy_then_contact_then_margin() -> None:
    base = {
        "score": 10.0,
        "semantic_policy_violation": 0,
        "stance_on_semantic_rate": 0.0,
        "touchdown_on_semantic_rate": 0.0,
        "root_on_semantic_rate": 0.0,
        "foot_semantic_penetration_rate": 0.0,
        "semantic_policy_margin_deficit": 0.0,
    }
    policy_bad = {**base, "score": 1.0, "semantic_policy_violation": 1}
    margin_bad = {**base, "score": 1.0, "semantic_policy_margin_deficit": 0.2}
    contact_bad = {**base, "score": 1.0, "stance_on_semantic_rate": 0.01}

    assert _selector_sort_key(base) < _selector_sort_key(policy_bad)
    assert _selector_sort_key(base) < _selector_sort_key(contact_bad)
    assert _selector_sort_key(base) < _selector_sort_key(margin_bad)


def test_jitter_selector_sort_key_prefers_smoother_policy_clean_candidates() -> None:
    base = {
        "score": 10.0,
        "semantic_policy_violation": 0,
        "stance_on_semantic_rate": 0.0,
        "touchdown_on_semantic_rate": 0.0,
        "root_on_semantic_rate": 0.0,
        "foot_semantic_penetration_rate": 0.0,
        "semantic_policy_margin_deficit": 0.0,
        "foot_accel_max_to_mean": 6.0,
        "root_accel_max_to_mean": 5.0,
        "worst_max_to_median_step": 4.0,
        "worst_boundary_to_median_step": 3.0,
        "min_z_quadratic_r2": 0.8,
    }
    jitter_bad = {
        **base,
        "score": 1.0,
        "foot_accel_max_to_mean": 40.0,
        "root_accel_max_to_mean": 25.0,
        "worst_max_to_median_step": 30.0,
    }
    policy_bad = {**base, "score": 1.0, "semantic_policy_violation": 1}

    assert _selector_jitter_sort_key(base) < _selector_jitter_sort_key(jitter_bad)
    assert _selector_jitter_sort_key(base) < _selector_jitter_sort_key(policy_bad)


def test_post_blend_result_smooths_initial_foot_handoff_without_moving_later_frames() -> None:
    root = torch.zeros((1, 20, 3), dtype=torch.float32)
    foot = torch.zeros((1, 20, 4, 3), dtype=torch.float32)
    foot[:, 1:, 0, 0] = 1.0
    result = SimpleNamespace(root_pos_w=root, foot_pos_w=foot, foot_pos_root=foot.clone())

    blended = _post_blend_result_for_variant(result, "post_blend_body_hard_contact_only", blend_frames=5)
    blended_foot = torch.as_tensor(blended.foot_pos_w)

    assert blended_foot[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert 0.0 < blended_foot[0, 1, 0, 0].item() < 1.0
    assert blended_foot[0, 4, 0, 0].item() == pytest.approx(1.0)
    assert blended_foot[0, 5, 0, 0].item() == pytest.approx(1.0)


def test_priority_jitter_selector_uses_candidate_priority_after_margin_and_jitter() -> None:
    base = {
        "score": 10.0,
        "semantic_policy_violation": 0,
        "stance_on_semantic_rate": 0.0,
        "touchdown_on_semantic_rate": 0.0,
        "root_on_semantic_rate": 0.0,
        "foot_semantic_penetration_rate": 0.0,
        "semantic_policy_margin_deficit": 0.0,
        "foot_accel_max_to_mean": 6.0,
        "root_accel_max_to_mean": 5.0,
        "worst_max_to_median_step": 4.0,
        "worst_boundary_to_median_step": 3.0,
        "min_z_quadratic_r2": 0.8,
        "selector_candidate_priority": 0.0,
    }
    low_priority = {**base, "selector_candidate_priority": 4.0}
    high_margin = {**base, "semantic_policy_margin_deficit": 0.05, "selector_candidate_priority": 0.0}
    high_jitter = {**base, "foot_accel_max_to_mean": 40.0, "selector_candidate_priority": 0.0}

    assert _selector_priority_jitter_sort_key(base) < _selector_priority_jitter_sort_key(low_priority)
    assert _selector_priority_jitter_sort_key(base) < _selector_priority_jitter_sort_key(high_margin)
    assert _selector_priority_jitter_sort_key(base) < _selector_priority_jitter_sort_key(high_jitter)


def test_clearance_jitter_selector_uses_clearance_policy_before_legacy_crossing_policy() -> None:
    clear_bypass = {
        "score": 10.0,
        "semantic_policy_violation": 1,
        "semantic_clearance_policy_violation": 0,
        "stance_on_semantic_rate": 0.0,
        "touchdown_on_semantic_rate": 0.0,
        "root_on_semantic_rate": 0.0,
        "foot_semantic_penetration_rate": 0.0,
        "semantic_policy_margin_deficit": 0.0,
        "foot_accel_max_to_mean": 6.0,
        "root_accel_max_to_mean": 5.0,
        "worst_max_to_median_step": 4.0,
        "worst_boundary_to_median_step": 3.0,
        "min_z_quadratic_r2": 0.8,
    }
    legacy_clean_but_near = {
        **clear_bypass,
        "semantic_policy_violation": 0,
        "semantic_clearance_policy_violation": 1,
        "semantic_policy_margin_deficit": 0.05,
        "score": 1.0,
    }

    assert _selector_clearance_jitter_sort_key(clear_bypass) < _selector_clearance_jitter_sort_key(legacy_clean_but_near)


def test_task_jitter_selector_uses_corrected_semantic_task_before_score() -> None:
    straight_small_overpass = {
        "score": 10.0,
        "semantic_task_violation": 0,
        "semantic_task_continuity_violation": 0,
        "semantic_task_contact_violation": 0,
        "semantic_clearance_policy_violation": 0,
        "stance_on_semantic_rate": 0.0,
        "touchdown_on_semantic_rate": 0.0,
        "root_on_semantic_rate": 0.0,
        "foot_semantic_penetration_rate": 0.0,
        "semantic_policy_margin_deficit": 0.0,
        "foot_accel_max_to_mean": 6.0,
        "root_accel_max_to_mean": 5.0,
        "worst_max_to_median_step": 4.0,
        "worst_boundary_to_median_step": 3.0,
        "root_lateral_deviation_from_start_max": 0.04,
        "max_abs_lateral_to_obstacle": 0.08,
        "min_z_quadratic_r2": 0.8,
    }
    cheap_detour_that_only_reaches_backside = {
        **straight_small_overpass,
        "score": 1.0,
        "semantic_task_violation": 1,
        "root_lateral_deviation_from_start_max": 0.45,
        "max_abs_lateral_to_obstacle": 0.45,
    }
    discontinuous_overpass = {
        **straight_small_overpass,
        "score": 1.0,
        "semantic_task_violation": 1,
        "semantic_task_continuity_violation": 1,
        "foot_accel_max_to_mean": 45.0,
    }

    assert _selector_task_jitter_sort_key(straight_small_overpass) < _selector_task_jitter_sort_key(
        cheap_detour_that_only_reaches_backside
    )
    assert _selector_task_jitter_sort_key(straight_small_overpass) < _selector_task_jitter_sort_key(discontinuous_overpass)


def test_metric_task_selector_prefers_clean_ever_crossed_path_before_raw_score() -> None:
    base = {
        "score": 10.0,
        "semantic_task_contact_violation": 0,
        "semantic_task_continuity_violation": 0,
        "ever_crossed_obstacle_along_command": 1,
        "stance_on_semantic_rate": 0.0,
        "touchdown_on_semantic_rate": 0.0,
        "root_on_semantic_rate": 0.0,
        "foot_semantic_penetration_rate": 0.0,
        "command_path_lateral_error_max": 0.20,
        "command_path_progress_error_final": 0.10,
        "foot_accel_max_to_mean": 8.0,
        "root_accel_max_to_mean": 8.0,
        "worst_max_to_median_step": 3.0,
        "worst_boundary_to_median_step": 3.0,
        "min_z_quadratic_r2": 0.9,
    }
    no_cross_low_score = {**base, "score": 1.0, "ever_crossed_obstacle_along_command": 0}
    detour_low_score = {**base, "score": 1.0, "command_path_lateral_error_max": 0.60}
    contact_low_score = {**base, "score": 1.0, "semantic_task_contact_violation": 1}
    clean_higher_score = {**base, "score": 20.0}

    selected = min(
        [no_cross_low_score, detour_low_score, contact_low_score, clean_higher_score],
        key=_selector_metric_task_jitter_sort_key,
    )

    assert selected is clean_higher_score


def test_large_smooth_metric_selector_prefers_margin_then_lower_jump() -> None:
    base = {
        "score": 10.0,
        "semantic_task_contact_violation": 0,
        "stance_on_semantic_rate": 0.0,
        "touchdown_on_semantic_rate": 0.0,
        "root_on_semantic_rate": 0.0,
        "foot_semantic_penetration_rate": 0.0,
        "semantic_policy_margin_deficit": 0.0,
        "worst_max_to_median_step": 20.0,
        "worst_boundary_to_median_step": 3.0,
        "foot_accel_max_to_mean": 10.0,
        "root_accel_max_to_mean": 10.0,
        "min_z_quadratic_r2": 0.9,
    }
    margin_bad_low_jump = {**base, "semantic_policy_margin_deficit": 0.02, "worst_max_to_median_step": 2.0}
    jump_bad_low_score = {**base, "score": 1.0, "worst_max_to_median_step": 45.0}
    clean_smooth_higher_score = {**base, "score": 20.0}

    selected = min(
        [margin_bad_low_jump, jump_bad_low_score, clean_smooth_higher_score],
        key=_selector_large_smooth_metric_sort_key,
    )

    assert selected is clean_smooth_higher_score


def test_semantic_collision_metrics_count_stance_and_class_rates() -> None:
    result = SimpleNamespace(
        root_pos_w=torch.tensor([[[0.0, 0.0, 0.30], [1.0, 0.0, 0.30]]], dtype=torch.float32),
        foot_pos_w=torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        contact_state=torch.tensor([[[True, False, False, False], [True, True, False, False]]]),
        planned_touchdown_w=torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        ),
    )

    metrics = _semantic_collision_metrics(result, _terrain())

    assert metrics["stance_on_small_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["stance_on_large_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["foot_on_semantic_rate"] == pytest.approx(4.0 / 8.0)
    assert metrics["touchdown_on_semantic_rate"] == pytest.approx(4.0 / 8.0)
    assert metrics["root_on_large_rate"] == pytest.approx(0.5)


def test_jitter_metrics_report_root_and_foot_acceleration_spikes() -> None:
    result = SimpleNamespace(
        root_pos_w=torch.tensor([[[0.0, 0.0, 0.3], [0.1, 0.0, 0.3], [0.6, 0.0, 0.3], [1.2, 0.0, 0.3]]]),
        foot_pos_w=torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0]] * 4,
                    [[0.1, 0.0, 0.0]] * 4,
                    [[0.6, 0.0, 0.0]] * 4,
                    [[1.2, 0.0, 0.0]] * 4,
                ]
            ],
            dtype=torch.float32,
        ),
    )

    metrics = _jitter_metrics(result)

    assert metrics["root_max_to_median_step"] > 1.0
    assert metrics["root_accel_max"] > 0.1
    assert metrics["foot_accel_max"] > 0.1
    assert metrics["worst_root_accel_frame"] == 1.0
    assert metrics["worst_root_accel_value"] == pytest.approx(metrics["root_accel_max"])
    assert metrics["worst_foot_accel_frame"] == 1.0
    assert metrics["worst_foot_accel_leg"] == 0.0
    assert metrics["worst_foot_accel_value"] == pytest.approx(metrics["foot_accel_max"])
    assert metrics["foot_accel_mean_for_ratio"] == pytest.approx(metrics["foot_accel_mean"])


def test_variant_cfg_changes_only_test_time_loss_weights() -> None:
    base = MpcPlannerCfg()
    base.runtime.randomize_replan_phase = True

    parametric = _variant_cfg(base, "parametric_v1")
    assert parametric.runtime.optimize_steps >= 40

    phase_fixed = _variant_cfg(base, "phase_fixed_probe")
    assert base.runtime.randomize_replan_phase is True
    assert phase_fixed.runtime.randomize_replan_phase is False

    semantic = _variant_cfg(base, "semantic_strong")
    assert semantic.losses.semantic_contact_avoid.weight > base.losses.semantic_contact_avoid.weight
    assert semantic.losses.semantic_obstacle.large_weight > base.losses.semantic_obstacle.large_weight
    assert semantic.losses.touchdown_semantic.large_weight > base.losses.touchdown_semantic.large_weight

    risk = _variant_cfg(base, "risk_strong")
    assert risk.losses.high_obstacle_avoidance.weight > base.losses.high_obstacle_avoidance.weight
    assert risk.losses.obstacle_risk.linear_scale_when_blocked < base.losses.obstacle_risk.linear_scale_when_blocked

    smooth = _variant_cfg(base, "smooth_strong")
    assert smooth.losses.foot_trajectory_regularization.accel_weight > base.losses.foot_trajectory_regularization.accel_weight
    assert smooth.losses.foot_trajectory_regularization.boundary_weight > base.losses.foot_trajectory_regularization.boundary_weight

    contact = _variant_cfg(base, "contact_only_semantic")
    assert contact.losses.stance_semantic.weight > base.losses.stance_semantic.weight
    assert contact.losses.semantic_obstacle.body_soft_field_weight == base.losses.semantic_obstacle.body_soft_field_weight

    stance = _variant_cfg(base, "stance_only_semantic")
    assert stance.losses.stance_semantic.weight > base.losses.stance_semantic.weight
    assert stance.losses.semantic_contact_avoid.weight == base.losses.semantic_contact_avoid.weight

    body = _variant_cfg(base, "high_body_margin")
    assert body.losses.high_obstacle_avoidance.weight > base.losses.high_obstacle_avoidance.weight
    assert body.losses.semantic_obstacle.body_soft_field_weight > base.losses.semantic_obstacle.body_soft_field_weight

    risk_stance = _variant_cfg(base, "risk_stance_crossing")
    assert risk_stance.losses.stance_semantic.weight > base.losses.stance_semantic.weight
    assert risk_stance.losses.obstacle_risk.linear_scale_when_blocked < base.losses.obstacle_risk.linear_scale_when_blocked

    body_stance_smooth = _variant_cfg(base, "body_stance_crossing_smooth")
    assert body_stance_smooth.losses.stance_semantic.weight > base.losses.stance_semantic.weight
    assert body_stance_smooth.losses.foot_trajectory_regularization.accel_weight > base.losses.foot_trajectory_regularization.accel_weight

    crossing_light = _variant_cfg(base, "crossing_contact_light")
    assert crossing_light.losses.low_small_crossing.weight > base.losses.low_small_crossing.weight
    assert crossing_light.losses.low_small_crossing.weight < _variant_cfg(base, "crossing_strong").losses.low_small_crossing.weight

    body_light = _variant_cfg(base, "body_light")
    assert body_light.losses.high_obstacle_avoidance.weight > base.losses.high_obstacle_avoidance.weight
    assert body_light.losses.high_obstacle_avoidance.weight < body.losses.high_obstacle_avoidance.weight

    body_light_touchdown = _variant_cfg(base, "body_light_touchdown_crossing")
    assert body_light_touchdown.losses.touchdown_semantic.weight > base.losses.touchdown_semantic.weight
    assert body_light_touchdown.losses.stance_semantic.weight == base.losses.stance_semantic.weight

    hard_contact = _variant_cfg(base, "hard_contact_crossing_light")
    assert hard_contact.losses.semantic_contact_avoid.activation_margin > base.losses.semantic_contact_avoid.activation_margin
    assert hard_contact.losses.semantic_contact_avoid.soft_field_weight == pytest.approx(0.0)

    body_hard_contact = _variant_cfg(base, "body_hard_contact_only")
    assert body_hard_contact.losses.high_obstacle_avoidance.weight > base.losses.high_obstacle_avoidance.weight
    assert body_hard_contact.losses.semantic_contact_avoid.soft_worst_field_weight == pytest.approx(0.0)

    progress_only = _variant_cfg(base, "crossing_progress_only")
    assert progress_only.losses.low_small_crossing.weight > base.losses.low_small_crossing.weight
    assert progress_only.losses.swing_clearance_terrain.weight == base.losses.swing_clearance_terrain.weight

    body_hard_progress = _variant_cfg(base, "body_hard_contact_crossing_progress")
    assert body_hard_progress.losses.low_small_crossing.weight > base.losses.low_small_crossing.weight
    assert body_hard_progress.losses.semantic_contact_avoid.soft_field_weight == pytest.approx(0.0)

    long_swing = _variant_cfg(base, "long_swing_crossing")
    assert long_swing.runtime.swing_window_max_width > base.runtime.swing_window_max_width
    assert long_swing.runtime.nominal_swing_height_m > base.runtime.nominal_swing_height_m

    body_long_swing = _variant_cfg(base, "body_long_swing_hard_contact")
    assert body_long_swing.losses.high_obstacle_avoidance.weight > base.losses.high_obstacle_avoidance.weight
    assert body_long_swing.runtime.swing_window_max_width > base.runtime.swing_window_max_width

    opt40 = _variant_cfg(base, "opt40_body_crossing_progress")
    assert opt40.runtime.optimize_steps >= 40
    assert opt40.runtime.lr < base.runtime.lr

    opt40_risk = _variant_cfg(base, "opt40_body_hard_contact_risk_progress")
    assert opt40_risk.runtime.optimize_steps >= 40
    assert opt40_risk.losses.obstacle_risk.linear_scale_when_blocked < base.losses.obstacle_risk.linear_scale_when_blocked

    opt40_highbody = _variant_cfg(base, "opt40_body_hard_contact_highbody_progress")
    assert opt40_highbody.runtime.optimize_steps >= 40
    assert opt40_highbody.losses.high_obstacle_avoidance.lateral_clearance_m > body_hard_progress.losses.high_obstacle_avoidance.lateral_clearance_m

    foot_soft = _variant_cfg(base, "foot_soft_cross_progress")
    assert foot_soft.losses.semantic_contact_avoid.soft_field_weight > base.losses.semantic_contact_avoid.soft_field_weight
    assert foot_soft.losses.low_small_crossing.weight > base.losses.low_small_crossing.weight

    opt40_foot_soft = _variant_cfg(base, "opt40_body_foot_soft_cross_progress")
    assert opt40_foot_soft.runtime.optimize_steps >= 40
    assert opt40_foot_soft.losses.semantic_contact_avoid.soft_worst_field_weight > base.losses.semantic_contact_avoid.soft_worst_field_weight

    support_touchdown = _variant_cfg(base, "support_touchdown_cross_progress")
    assert support_touchdown.losses.touchdown_surface.support_search_radius_m > base.losses.touchdown_surface.support_search_radius_m
    assert support_touchdown.losses.touchdown_semantic.weight > base.losses.touchdown_semantic.weight

    opt40_support = _variant_cfg(base, "opt40_body_support_touchdown_cross_progress")
    assert opt40_support.runtime.optimize_steps >= 40
    assert opt40_support.losses.touchdown_surface.invalid_support_weight > base.losses.touchdown_surface.invalid_support_weight

    split_gentle_smooth = _variant_cfg(base, "struct_lowfoot_largebody_gentle_smooth")
    assert split_gentle_smooth.losses.foot_trajectory_regularization.accel_weight > base.losses.foot_trajectory_regularization.accel_weight
    assert split_gentle_smooth.losses.semantic_obstacle.foot_soft_field_weight < base.losses.semantic_obstacle.foot_soft_field_weight

    hard_cross = _variant_cfg(base, "struct_lowfoot_cross_hard")
    assert hard_cross.losses.low_small_crossing.weight > _variant_cfg(base, "crossing_strong").losses.low_small_crossing.weight
    assert hard_cross.losses.low_small_crossing.pass_margin_m >= 0.18

    opt40_hard_cross = _variant_cfg(base, "opt40_struct_lowfoot_cross_hard")
    assert opt40_hard_cross.runtime.optimize_steps >= 40
    assert opt40_hard_cross.losses.low_small_crossing.obstacle_depth_m >= 0.45

    loss_all = _variant_cfg(base, "loss_semantic_all_v1")
    assert loss_all.runtime.optimize_steps >= base.runtime.optimize_steps
    assert _candidate_variants_for_variant("loss_semantic_all_v1") == ("loss_semantic_all_v1",)


def test_semantic_small_height_override_supports_importer_terrain_cfg() -> None:
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(generate_semantic_course=None),
        scene=SimpleNamespace(terrain=SimpleNamespace()),
    )

    _apply_semantic_small_height_override(env_cfg, 0.46)

    assert env_cfg.scene.terrain.semantic_course_scale_profile_overrides["small"][1] == pytest.approx(0.46)


def test_trajectory_summary_handles_zero_median_swing_steps() -> None:
    result = SimpleNamespace(
        root_pos_w=torch.zeros((1, 5, 3), dtype=torch.float32),
        foot_pos_w=torch.zeros((1, 5, 4, 3), dtype=torch.float32),
        contact_state=torch.tensor([[[False, True, True, True]] * 5]),
    )

    summary = _trajectory_summary(command_name="zero_swing", cycle=0, result=result)[0]

    assert summary["swing_run_count"] == 1
    assert torch.isnan(torch.tensor(float(summary["worst_max_to_median_step"])))
