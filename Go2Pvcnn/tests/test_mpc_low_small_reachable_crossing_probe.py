from __future__ import annotations

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

from mpc_low_small_reachable_crossing_probe import (  # noqa: E402
    reachable_extra_loss,
    reachable_cfg_for_variant,
    reachable_command_direction_metrics,
    reachable_command_frame_endpoint_metrics,
    reachable_distance_window_weights,
    reachable_touchdown_chain_trace,
    reachable_foot_height_relative_to_root_metrics,
    reachable_foot_over_arc_metrics,
    reachable_ik_fk_consistency_metrics,
    reachable_swing_continuity_metrics,
)
from extension.batch_mpc_planner.config import MpcPlannerCfg  # noqa: E402
from extension.batch_mpc_planner.types import MpcPlannerTerrain, MpcRobotState  # noqa: E402
from extension.batch_mpc_planner.debug_variants import mpc_debug_extra_loss  # noqa: E402
from extension.batch_mpc_planner.config import MpcPlannerCfg  # noqa: E402
from fixtures.viewer_runtime_diagnostics import _apply_semantic_small_profile_override  # noqa: E402


def test_semantic_small_profile_override_can_change_diameter_and_height() -> None:
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(generate_semantic_course=SimpleNamespace(params={})),
    )

    _apply_semantic_small_profile_override(
        env_cfg,
        semantic_small_diameter_m=0.08,
        semantic_small_height_m=0.14,
    )

    assert env_cfg.events.generate_semantic_course.params["scale_profile_overrides"] == {"small": (0.08, 0.14)}


def test_reachable_small_variant_tightens_small_obstacle_loss_window_without_mutating_base_cfg() -> None:
    base = MpcPlannerCfg()

    tuned = reachable_cfg_for_variant(base, "reachable_loss_small_v1")

    assert tuned is not base
    assert tuned.losses.low_small_foot_over.radius_m < base.losses.low_small_foot_over.radius_m
    assert tuned.losses.low_small_foot_over.along_window_m < base.losses.low_small_foot_over.along_window_m
    assert tuned.losses.semantic_obstacle.soft_margin_m < base.losses.semantic_obstacle.soft_margin_m
    assert tuned.losses.tracking.vel_weight < base.losses.tracking.vel_weight


def test_reachable_small_v2_adds_continuity_to_small_window_variant() -> None:
    base = MpcPlannerCfg()

    v1 = reachable_cfg_for_variant(base, "reachable_loss_small_v1")
    v2 = reachable_cfg_for_variant(base, "reachable_loss_small_v2")

    assert v2.losses.low_small_foot_over.radius_m == pytest.approx(v1.losses.low_small_foot_over.radius_m)
    assert v2.losses.low_small_stepcap.foot_accel_weight > v1.losses.low_small_stepcap.foot_accel_weight
    assert v2.losses.low_small_stepcap.foot_accel_worst_weight > v1.losses.low_small_stepcap.foot_accel_worst_weight
    assert v2.losses.foot_trajectory_regularization.accel_weight > v1.losses.foot_trajectory_regularization.accel_weight
    assert v2.runtime.optimize_steps >= v1.runtime.optimize_steps


def test_parametric_v1_enables_parametric_trajectory_cfg() -> None:
    cfg = reachable_cfg_for_variant(MpcPlannerCfg(), "parametric_v1")

    assert cfg.runtime.optimize_steps >= 40


def test_reachable_fk_cross_v1_uses_continuous_approach_then_cross_window() -> None:
    root_xy = torch.tensor(
        [
            [[0.00, 0.00], [0.05, 0.00]],  # too far: approach, not cross
            [[0.18, 0.00], [0.24, 0.00]],  # good distance: cross
            [[0.34, 0.00], [0.36, 0.00]],  # too close: neither force long cross nor back up hard
        ],
        dtype=torch.float32,
    )
    obstacle_xy = torch.tensor([[0.40, 0.00], [0.40, 0.00], [0.40, 0.00]], dtype=torch.float32)

    weights = reachable_distance_window_weights(
        root_xy,
        obstacle_xy,
        command=torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32),
        min_cross_distance_m=0.14,
        max_cross_distance_m=0.28,
        sigma_m=0.04,
    )

    assert weights["approach_weight"][0] > weights["cross_weight"][0]
    assert weights["cross_weight"][1] > weights["approach_weight"][1]
    assert weights["cross_weight"][1] > weights["cross_weight"][0]
    assert weights["cross_weight"][1] > weights["cross_weight"][2]
    assert torch.all(weights["cross_weight"] >= 0.0)
    assert torch.all(weights["cross_weight"] <= 1.0)


def test_reachable_fk_cross_v1_extra_loss_reports_fk_approach_cross_terms() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.30], [0.05, 0.00, 0.30], [0.10, 0.00, 0.30]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.20, 0.10, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.28, 0.10, 0.02], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.36, 0.10, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32),
            "swing_prob": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        },
    )()

    extra, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v1",
        command=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert extra.shape == (1,)
    assert extra.item() > 0.0
    assert breakdown["reachable_fk_cross_window"].item() > 0.0
    assert breakdown["reachable_fk_cross_over"].item() > 0.0
    assert breakdown["reachable_fk_approach_distance"].item() >= 0.0


def test_reachable_fk_cross_v2_gates_cross_credit_when_base_is_low_or_drifting() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.09], [0.05, 0.18, 0.09], [0.10, 0.24, 0.09]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.18, 0.02, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.30, 0.02, 0.20], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.44, 0.02, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32),
            "swing_prob": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        },
    )()

    extra, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v2",
        command=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert extra.shape == (1,)
    assert breakdown["reachable_fk_base_height_guard"].item() > 0.0
    assert breakdown["reachable_fk_direction_lateral"].item() > 0.0
    assert breakdown["reachable_fk_cross_window"].item() > 0.0


def test_reachable_fk_cross_v3_penalizes_full_path_lateral_drift() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.15], [0.05, 0.12, 0.15], [0.10, 0.22, 0.15]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.18, 0.02, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.30, 0.02, 0.20], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.44, 0.02, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32),
            "swing_prob": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        },
    )()

    _, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v3",
        command=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert breakdown["reachable_fk_lateral_path_guard"].item() > 0.0
    assert breakdown["reachable_fk_direction_lateral"].item() > 0.0


def test_reachable_fk_cross_v4_bypasses_crossing_terms_for_pure_yaw() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.20], [0.00, 0.00, 0.20], [0.00, 0.00, 0.20]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [[[[0.20, 0.10, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]]] * 3],
                dtype=torch.float32,
            ),
            "contact_prob": torch.ones((1, 3, 4), dtype=torch.float32),
            "swing_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
        },
    )()

    extra, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v4",
        command=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.20, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert extra.shape == (1,)
    assert "reachable_fk_cross_over" not in breakdown
    assert "reachable_fk_cross_window" not in breakdown
    assert breakdown["reachable_yaw_only_reachability"].item() >= 0.0


def test_reachable_fk_cross_v4_mixed_yaw_uses_translation_frame_for_lateral_path() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.20], [0.10, 0.14, 0.20], [0.20, 0.20, 0.20]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [[[[0.20, 0.02, 0.20], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]]] * 3],
                dtype=torch.float32,
            ),
            "contact_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
            "swing_prob": torch.ones((1, 3, 4), dtype=torch.float32),
        },
    )()

    _, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v4",
        command=torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert breakdown["reachable_fk_lateral_path_guard"].item() > 0.0
    assert breakdown["reachable_fk_direction_lateral"].item() > 0.0


def test_reachable_fk_cross_v4_penalizes_fk_small_contact_inside_lane() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.20], [0.10, 0.00, 0.20], [0.20, 0.00, 0.20]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.30, 0.01, 0.10], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.30, 0.01, 0.10], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.30, 0.01, 0.10], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            "swing_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
        },
    )()

    _, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v4",
        command=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert breakdown["reachable_fk_small_contact_guard"].item() > 0.0


def test_reachable_fk_cross_v5_command_specific_cfg_splits_pure_yaw_and_mixed_yaw() -> None:
    base = MpcPlannerCfg()

    pure_yaw = reachable_cfg_for_variant(base, "reachable_fk_cross_v5", command=(0.0, 0.0, 1.0))
    mixed_yaw = reachable_cfg_for_variant(base, "reachable_fk_cross_v5", command=(0.5, 0.25, 1.0))
    forward = reachable_cfg_for_variant(base, "reachable_fk_cross_v5", command=(0.5, 0.0, 0.0))

    assert pure_yaw.losses.low_small_crossing.weight == pytest.approx(0.0)
    assert pure_yaw.losses.low_small_foot_over.weight == pytest.approx(0.0)
    assert pure_yaw.losses.low_small_stepcap.foot_accel_weight >= base.losses.low_small_stepcap.foot_accel_weight
    assert mixed_yaw.losses.low_small_crossing.weight < forward.losses.low_small_crossing.weight
    assert mixed_yaw.losses.low_small_foot_over.weight < forward.losses.low_small_foot_over.weight
    assert mixed_yaw.losses.progress.weight >= forward.losses.progress.weight


def test_reachable_fk_cross_v6_pure_yaw_uses_baseline_like_cfg_without_extra_crossing() -> None:
    base = MpcPlannerCfg()

    pure_yaw = reachable_cfg_for_variant(base, "reachable_fk_cross_v6", command=(0.0, 0.0, 1.0))
    mixed_yaw = reachable_cfg_for_variant(base, "reachable_fk_cross_v6", command=(0.5, 0.25, 1.0))

    assert pure_yaw.losses.low_small_crossing.weight == pytest.approx(0.0)
    assert pure_yaw.losses.low_small_foot_over.weight == pytest.approx(0.0)
    assert pure_yaw.losses.tracking.vel_weight == pytest.approx(base.losses.tracking.vel_weight)
    assert pure_yaw.runtime.optimize_steps == base.runtime.optimize_steps
    assert mixed_yaw.losses.tracking.vel_weight > reachable_cfg_for_variant(base, "reachable_fk_cross_v5", command=(0.5, 0.25, 1.0)).losses.tracking.vel_weight
    assert mixed_yaw.losses.progress.weight > base.losses.progress.weight


def test_reachable_fk_cross_v7_penalizes_mixed_yaw_low_direction_cosine() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.20], [0.02, 0.20, 0.20], [0.04, 0.40, 0.20]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [[[[0.20, 0.02, 0.20], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]]] * 3],
                dtype=torch.float32,
            ),
            "contact_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
            "swing_prob": torch.ones((1, 3, 4), dtype=torch.float32),
        },
    )()

    _, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v7",
        command=torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert breakdown["reachable_fk_command_direction_cosine"].item() > 0.0
    assert breakdown["reachable_fk_command_progress"].item() > 0.0


def test_reachable_fk_cross_v8_penalizes_mixed_yaw_low_base_with_direction_terms() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.09], [0.20, 0.00, 0.09], [0.40, 0.00, 0.09]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [[[[0.20, 0.02, 0.20], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]]] * 3],
                dtype=torch.float32,
            ),
            "contact_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
            "swing_prob": torch.ones((1, 3, 4), dtype=torch.float32),
        },
    )()

    _, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v8",
        command=torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert breakdown["reachable_fk_mixed_base_height_guard"].item() > 0.0
    assert "reachable_fk_command_direction_cosine" in breakdown


def test_reachable_fk_cross_v9_keeps_mixed_yaw_reachability_barrier() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.16], [0.20, 0.00, 0.16], [0.40, 0.00, 0.16]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.65, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.80, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.95, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
            "swing_prob": torch.ones((1, 3, 4), dtype=torch.float32),
        },
    )()

    _, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v9",
        command=torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert breakdown["reachable_fk_mixed_reachability_barrier"].item() > 0.0
    assert breakdown["reachable_fk_command_direction_cosine"].item() >= 0.0


def test_production_debug_variant_v9_matches_probe_extra_loss_terms() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.16], [0.20, 0.00, 0.16], [0.40, 0.00, 0.16]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.65, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.80, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.95, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
            "swing_prob": torch.ones((1, 3, 4), dtype=torch.float32),
        },
    )()
    command = torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32)
    obstacle_xy = torch.tensor([[0.30, 0.00]], dtype=torch.float32)
    obstacle_height = torch.tensor([0.12], dtype=torch.float32)

    probe_extra, probe_breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v9",
        command=command,
        obstacle_xy=obstacle_xy,
        obstacle_height=obstacle_height,
    )
    prod_extra, prod_breakdown = mpc_debug_extra_loss(
        decoded,
        variant="reachable_fk_cross_v9",
        command=command,
        obstacle_xy=obstacle_xy,
        obstacle_height=obstacle_height,
    )

    torch.testing.assert_close(prod_extra, probe_extra)
    assert set(prod_breakdown) == set(probe_breakdown)
    torch.testing.assert_close(
        prod_breakdown["reachable_fk_mixed_reachability_barrier"],
        probe_breakdown["reachable_fk_mixed_reachability_barrier"],
    )


def test_reachable_fk_cross_v10_has_soft_mixed_yaw_combo_terms() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.12], [0.08, 0.20, 0.12], [0.16, 0.40, 0.12]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.65, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.80, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.95, 0.20, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
            "swing_prob": torch.ones((1, 3, 4), dtype=torch.float32),
        },
    )()

    _, breakdown = reachable_extra_loss(
        decoded,
        variant="reachable_fk_cross_v10",
        command=torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert breakdown["reachable_fk_command_direction_cosine"].item() > 0.0
    assert breakdown["reachable_fk_mixed_base_height_guard"].item() > 0.0
    assert breakdown["reachable_fk_mixed_reachability_barrier"].item() > 0.0
    assert breakdown["reachable_fk_mixed_soft_balance"].item() >= 0.0


def test_reachable_fk_cross_v11_adds_endpoint_touchdown_reachability_and_height_terms() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.18], [0.10, 0.00, 0.18], [0.20, 0.00, 0.18]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.10, 0.08, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.55, 0.08, 0.36], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.12, 0.08, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32),
            "swing_prob": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            "swing_center": torch.tensor([[0.25, 0.20, 0.20, 0.20]], dtype=torch.float32),
            "swing_width": torch.tensor([[0.90, 0.20, 0.20, 0.20]], dtype=torch.float32),
        },
    )()

    _, breakdown = mpc_debug_extra_loss(
        decoded,
        variant="reachable_fk_cross_v11",
        command=torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32),
        obstacle_xy=torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        obstacle_height=torch.tensor([0.12], dtype=torch.float32),
    )

    assert breakdown["reachable_fk_touchdown_endpoint_consistency"].item() > 0.0
    assert breakdown["reachable_fk_sampled_touchdown_reachability"].item() > 0.0
    assert breakdown["reachable_fk_foot_above_root_guard"].item() > 0.0


def test_reachable_fk_cross_v12_prioritizes_endpoint_term_over_reachability_and_height() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.00, 0.00, 0.18], [0.10, 0.00, 0.18], [0.20, 0.00, 0.18]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.10, 0.08, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.55, 0.08, 0.36], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                        [[0.12, 0.08, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32),
            "swing_prob": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            "swing_center": torch.tensor([[0.25, 0.20, 0.20, 0.20]], dtype=torch.float32),
            "swing_width": torch.tensor([[0.90, 0.20, 0.20, 0.20]], dtype=torch.float32),
        },
    )()
    kwargs = {
        "command": torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.float32),
        "obstacle_xy": torch.tensor([[0.30, 0.00]], dtype=torch.float32),
        "obstacle_height": torch.tensor([0.12], dtype=torch.float32),
    }

    _, v11 = mpc_debug_extra_loss(decoded, variant="reachable_fk_cross_v11", **kwargs)
    _, v12 = mpc_debug_extra_loss(decoded, variant="reachable_fk_cross_v12", **kwargs)

    assert v12["reachable_fk_touchdown_endpoint_consistency"].item() > v11["reachable_fk_touchdown_endpoint_consistency"].item()
    assert v12["reachable_fk_sampled_touchdown_reachability"].item() < v11["reachable_fk_sampled_touchdown_reachability"].item()
    assert v12["reachable_fk_foot_above_root_guard"].item() < v11["reachable_fk_foot_above_root_guard"].item()


def test_command_direction_metrics_use_command_frame_for_lateral_motion() -> None:
    root = torch.tensor(
        [
            [
                [1.0, -0.60, 0.30],
                [1.0, -0.30, 0.30],
                [1.0, 0.00, 0.30],
                [1.0, 0.30, 0.30],
            ]
        ],
        dtype=torch.float32,
    )

    metrics = reachable_command_direction_metrics(root, command=(0.0, 0.5, 0.0))

    assert metrics["translation_command_active"] == 1
    assert metrics["command_direction_cosine"] == pytest.approx(1.0)
    assert metrics["along_progress_m"] == pytest.approx(0.90)
    assert metrics["lateral_drift_m"] == pytest.approx(0.0)
    assert metrics["speed_magnitude_tracking_error"] > 0.0


def test_reachable_loss_variant_strengthens_feasibility_and_continuity_without_mutating_base_cfg() -> None:
    base = MpcPlannerCfg()

    tuned = reachable_cfg_for_variant(base, "reachable_loss_v1")

    assert tuned is not base
    assert base.losses.ik_fk_residual.weight == pytest.approx(8.0)
    assert tuned.losses.ik_fk_residual.weight > base.losses.ik_fk_residual.weight
    assert tuned.losses.kinematics.weight > base.losses.kinematics.weight
    assert tuned.losses.foot_trajectory_regularization.boundary_weight > base.losses.foot_trajectory_regularization.boundary_weight
    assert tuned.losses.foot_trajectory_regularization.accel_weight > base.losses.foot_trajectory_regularization.accel_weight
    assert tuned.runtime.optimize_steps > base.runtime.optimize_steps


def test_reachable_loss_v2_relaxes_speed_tracking_and_strengthens_stepcap_more_than_v1() -> None:
    base = MpcPlannerCfg()

    v1 = reachable_cfg_for_variant(base, "reachable_loss_v1")
    v2 = reachable_cfg_for_variant(base, "reachable_loss_v2")

    assert v2.losses.tracking.vel_weight < base.losses.tracking.vel_weight
    assert v2.losses.low_small_crossing.weight < base.losses.low_small_crossing.weight
    assert v2.losses.progress.weight >= base.losses.progress.weight
    assert v2.losses.low_small_stepcap.foot_accel_weight > v1.losses.low_small_stepcap.foot_accel_weight
    assert v2.losses.ik_fk_residual.weight > v1.losses.ik_fk_residual.weight
    assert v2.runtime.optimize_steps >= v1.runtime.optimize_steps


def test_reachable_extra_loss_penalizes_fk_residual_and_fk_swing_jumps() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.0, 0.0, 0.30], [0.0, 0.0, 0.30], [0.0, 0.0, 0.30]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.40, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                        [[0.80, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                        [[0.40, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.zeros((1, 3, 4), dtype=torch.float32),
        },
    )()

    extra, breakdown = reachable_extra_loss(decoded, variant="reachable_struct_v1")

    assert extra.shape == (1,)
    assert extra.item() > 0.0
    assert breakdown["reachable_fk_residual"].item() > 0.0
    assert breakdown["reachable_fk_step"].item() > 0.0
    assert breakdown["reachable_fk_accel"].item() > 0.0


def test_reachable_struct_v2_extra_loss_penalizes_decoded_and_touchdown_feasibility() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.0, 0.0, 0.30], [0.0, 0.0, 0.30], [0.0, 0.0, 0.30]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.65, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                        [[0.75, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                        [[0.85, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32),
            "swing_center": torch.tensor([[0.35, 0.10, 0.10, 0.10]], dtype=torch.float32),
            "swing_width": torch.tensor([[0.35, 0.20, 0.20, 0.20]], dtype=torch.float32),
        },
    )()

    extra, breakdown = reachable_extra_loss(decoded, variant="reachable_struct_v2")

    assert extra.shape == (1,)
    assert extra.item() > 0.0
    assert breakdown["reachable_fk_residual"].item() > 0.0
    assert breakdown["reachable_touchdown_fk_residual"].item() > 0.0
    assert breakdown["reachable_fk_step"].item() > 0.0


def test_reachable_struct_v3_extra_loss_adds_full_horizon_limit_barrier() -> None:
    decoded = type(
        "Decoded",
        (),
        {
            "root_pos": torch.tensor([[[0.0, 0.0, 0.30], [0.0, 0.0, 0.30], [0.0, 0.0, 0.30]]], dtype=torch.float32),
            "root_rpy": torch.zeros((1, 3, 3), dtype=torch.float32),
            "foot_pos": torch.tensor(
                [
                    [
                        [[0.65, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                        [[0.75, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                        [[0.95, 0.20, 0.00], [0.20, -0.20, 0.00], [-0.20, 0.20, 0.00], [-0.20, -0.20, 0.00]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "contact_prob": torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32),
            "swing_center": torch.tensor([[0.35, 0.10, 0.10, 0.10]], dtype=torch.float32),
            "swing_width": torch.tensor([[0.35, 0.20, 0.20, 0.20]], dtype=torch.float32),
        },
    )()

    extra, breakdown = reachable_extra_loss(decoded, variant="reachable_struct_v3")

    assert extra.shape == (1,)
    assert extra.item() > 0.0
    assert breakdown["reachable_fk_worst_residual"].item() > 0.0
    assert breakdown["reachable_raw_joint_limit_excess"].item() > 0.0
    assert breakdown["reachable_touchdown_fk_residual"].item() > 0.0


def test_swing_continuity_metrics_report_step_accel_and_boundary_ratios() -> None:
    fk_foot = torch.zeros((1, 6, 4, 3), dtype=torch.float32)
    fk_foot[0, :, 0, 0] = torch.tensor([0.0, 0.02, 0.04, 0.06, 0.50, 0.52], dtype=torch.float32)
    contact_state = torch.zeros((1, 6, 4), dtype=torch.bool)
    contact_state[0, :, 1:] = True

    metrics = reachable_swing_continuity_metrics(
        fk_foot,
        contact_state,
        replan_interval_steps=4,
    )

    assert metrics["fk_swing_foot_step_max_to_median"] > 10.0
    assert metrics["fk_swing_foot_accel_max_to_mean"] > 1.0
    assert metrics["replan_boundary_fk_foot_step_to_median"] > 10.0


def test_command_frame_endpoint_metrics_report_forward_swing_then_rear_touchdown() -> None:
    obstacle_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
    planned_foot = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    fk_foot = planned_foot.clone()
    planned_foot[0, :, 0, 0] = torch.tensor([-0.10, 0.10, 0.30, 0.45], dtype=torch.float32)
    fk_foot[0, :, 0, 0] = torch.tensor([-0.10, 0.08, 0.25, 0.35], dtype=torch.float32)
    touchdown = planned_foot.clone()
    touchdown[0, :, 0, 0] = torch.tensor([-0.10, -0.05, -0.08, -0.12], dtype=torch.float32)
    contact_state = torch.zeros((1, 4, 4), dtype=torch.bool)
    contact_state[0, :, 1:] = True

    metrics = reachable_command_frame_endpoint_metrics(
        planned_foot,
        fk_foot,
        touchdown,
        contact_state,
        obstacle_xy,
        command=(0.5, 0.0, 0.0),
        replan_interval_steps=2,
    )

    assert metrics["planned_swing_along_forward_step_max_m"] == pytest.approx(0.20)
    assert metrics["touchdown_behind_swing_foot_along_max_m"] == pytest.approx(0.57)
    assert metrics["touchdown_behind_fk_foot_along_max_m"] == pytest.approx(0.47)
    assert metrics["planned_vs_fk_along_error_max_m"] == pytest.approx(0.10)


def test_touchdown_chain_trace_reports_exported_and_fk_layers() -> None:
    cfg = MpcPlannerCfg()
    cfg.runtime.horizon_steps = 5
    cfg.runtime.optimize_steps = 0
    cfg.runtime.touchdown_event_cap = 1
    state = MpcRobotState(
        root_pos=torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32),
        root_rpy=torch.zeros((1, 3), dtype=torch.float32),
        foot_pos=torch.tensor([[[0.20, 0.10, 0.00], [0.20, -0.10, 0.00], [-0.20, 0.10, 0.00], [-0.20, -0.10, 0.00]]], dtype=torch.float32),
        joint_angles=torch.zeros((1, 12), dtype=torch.float32),
    )
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )

    row = reachable_touchdown_chain_trace(
        terrain,
        state,
        torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
        cfg,
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        (0.5, 0.0, 0.0),
    )

    assert row["type"] == "reachable_touchdown_chain_trace"
    assert len(row["optimized_export"]["along_m"]) == 4
    assert len(row["fk_from_clamped_ik"]["along_m"]) == 4
    assert "delta_optimized_export_to_fk" in row


def test_foot_height_relative_to_root_metrics_report_foot_above_root() -> None:
    root = torch.zeros((1, 3, 3), dtype=torch.float32)
    root[..., 2] = 0.20
    planned_foot = torch.zeros((1, 3, 4, 3), dtype=torch.float32)
    fk_foot = planned_foot.clone()
    planned_foot[0, 1, 0, 2] = 0.35
    fk_foot[0, 1, 0, 2] = 0.28
    contact_state = torch.ones((1, 3, 4), dtype=torch.bool)
    contact_state[0, 1, 0] = False

    metrics = reachable_foot_height_relative_to_root_metrics(planned_foot, fk_foot, root, contact_state)

    assert metrics["planned_swing_foot_above_root_z_max_m"] == pytest.approx(0.15)
    assert metrics["fk_swing_foot_above_root_z_max_m"] == pytest.approx(0.08)


def test_foot_over_arc_requires_fk_lift_then_land_inside_lane() -> None:
    obstacle_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
    fk_foot = torch.zeros((1, 5, 4, 3), dtype=torch.float32)
    fk_foot[:, :, :, 2] = 0.0
    fk_foot[0, :, 0, :] = torch.tensor(
        [
            [-0.24, 0.01, 0.00],
            [-0.12, 0.01, 0.18],
            [0.00, 0.00, 0.24],
            [0.12, 0.01, 0.16],
            [0.24, 0.01, 0.00],
        ],
        dtype=torch.float32,
    )
    contact_state = torch.zeros((1, 5, 4), dtype=torch.bool)
    contact_state[0, 4, 0] = True

    metrics = reachable_foot_over_arc_metrics(
        fk_foot,
        contact_state,
        obstacle_xy,
        command=(0.5, 0.0, 0.0),
        obstacle_height=0.12,
        clearance=0.05,
        lane_half_width=0.08,
    )

    assert metrics["fk_foot_over_low_small_success"] == 1
    assert metrics["fk_foot_over_low_small_lift_then_land"] == 1
    assert metrics["fk_foot_over_low_small_touchdown_after"] == 1
    assert metrics["fk_foot_over_low_small_clearance_max"] == pytest.approx(0.12)
    assert metrics["fk_foot_over_low_small_min_lateral"] <= 0.01


def test_foot_over_arc_rejects_side_bypass() -> None:
    obstacle_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
    fk_foot = torch.zeros((1, 5, 4, 3), dtype=torch.float32)
    fk_foot[0, :, 0, :] = torch.tensor(
        [
            [-0.24, 0.20, 0.00],
            [-0.12, 0.20, 0.18],
            [0.00, 0.20, 0.24],
            [0.12, 0.20, 0.16],
            [0.24, 0.20, 0.00],
        ],
        dtype=torch.float32,
    )
    contact_state = torch.zeros((1, 5, 4), dtype=torch.bool)
    contact_state[0, 4, 0] = True

    metrics = reachable_foot_over_arc_metrics(
        fk_foot,
        contact_state,
        obstacle_xy,
        command=(0.5, 0.0, 0.0),
        obstacle_height=0.12,
        clearance=0.05,
        lane_half_width=0.08,
    )

    assert metrics["fk_foot_over_low_small_success"] == 0
    assert metrics["fk_foot_over_low_small_lift_then_land"] == 0
    assert metrics["fk_foot_over_low_small_touchdown_after"] == 0


def test_low_small_crossing_acceptance_combines_endpoint_foot_over_and_root_height_guard() -> None:
    obstacle_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
    root = torch.zeros((1, 5, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    planned_foot = torch.zeros((1, 5, 4, 3), dtype=torch.float32)
    fk_foot = planned_foot.clone()
    arc = torch.tensor(
        [
            [-0.24, 0.01, 0.00],
            [-0.12, 0.01, 0.18],
            [0.00, 0.00, 0.24],
            [0.12, 0.01, 0.16],
            [0.24, 0.01, 0.00],
        ],
        dtype=torch.float32,
    )
    planned_foot[0, :, 0, :] = arc
    fk_foot[0, :, 0, :] = arc
    touchdown = planned_foot.clone()
    touchdown[0, :, 0, 0] = 0.24
    contact_state = torch.ones((1, 5, 4), dtype=torch.bool)
    contact_state[0, :4, 0] = False

    endpoint = reachable_command_frame_endpoint_metrics(
        planned_foot,
        fk_foot,
        touchdown,
        contact_state,
        obstacle_xy,
        command=(0.5, 0.0, 0.0),
        replan_interval_steps=25,
    )
    foot_over = reachable_foot_over_arc_metrics(
        fk_foot,
        contact_state,
        obstacle_xy,
        command=(0.5, 0.0, 0.0),
        obstacle_height=0.12,
        clearance=0.05,
        lane_half_width=0.08,
    )
    height = reachable_foot_height_relative_to_root_metrics(planned_foot, fk_foot, root, contact_state)

    assert endpoint["touchdown_behind_swing_foot_along_max_m"] == pytest.approx(0.0)
    assert foot_over["fk_foot_over_low_small_success"] == 1
    assert foot_over["fk_foot_over_low_small_lift_then_land"] == 1
    assert foot_over["fk_foot_over_low_small_touchdown_after"] == 1
    assert height["planned_swing_foot_above_root_z_max_m"] <= 0.0
    assert height["fk_swing_foot_above_root_z_max_m"] <= 0.0


def test_low_small_crossing_acceptance_rejects_foot_over_arc_above_root() -> None:
    root = torch.zeros((1, 5, 3), dtype=torch.float32)
    root[..., 2] = 0.20
    planned_foot = torch.zeros((1, 5, 4, 3), dtype=torch.float32)
    fk_foot = planned_foot.clone()
    planned_foot[0, :, 0, :] = torch.tensor(
        [
            [-0.24, 0.01, 0.00],
            [-0.12, 0.01, 0.18],
            [0.00, 0.00, 0.24],
            [0.12, 0.01, 0.16],
            [0.24, 0.01, 0.00],
        ],
        dtype=torch.float32,
    )
    fk_foot.copy_(planned_foot)
    contact_state = torch.ones((1, 5, 4), dtype=torch.bool)
    contact_state[0, :4, 0] = False

    height = reachable_foot_height_relative_to_root_metrics(planned_foot, fk_foot, root, contact_state)

    assert height["planned_swing_foot_above_root_z_max_m"] == pytest.approx(0.04)
    assert height["fk_swing_foot_above_root_z_max_m"] == pytest.approx(0.04)


def test_ik_fk_consistency_reports_unreachable_planned_feet_and_touchdowns() -> None:
    planned_foot = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    fk_foot = planned_foot.clone()
    fk_foot[0, 1, 0] = torch.tensor([0.20, 0.00, 0.10], dtype=torch.float32)
    planned_touchdown = planned_foot.clone()
    fk_touchdown = planned_touchdown.clone()
    fk_touchdown[0, 1, 0] = torch.tensor([0.20, 0.00, 0.10], dtype=torch.float32)
    raw_joint = torch.zeros((1, 2, 12), dtype=torch.float32)
    clamped_joint = raw_joint.clone()
    clamped_joint[0, 1, 2] = -0.8378

    metrics = reachable_ik_fk_consistency_metrics(
        planned_foot,
        fk_foot,
        planned_touchdown,
        fk_touchdown,
        raw_joint,
        clamped_joint,
    )

    assert metrics["terminal_planned_vs_fk_foot_error_max"] == pytest.approx((0.20**2 + 0.10**2) ** 0.5)
    assert metrics["touchdown_ik_fk_error_max"] == pytest.approx((0.20**2 + 0.10**2) ** 0.5)
    assert metrics["calf_upper_saturation_max"] == pytest.approx(0.8378)
