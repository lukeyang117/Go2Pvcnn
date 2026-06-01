from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT, GO2PVCNN_ROOT / "tests"):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture, refresh_targeted_scanner_pose  # noqa: E402
from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles, solve_joint_angles_from_trajectory  # noqa: E402
from mpc_swing_trajectory_quality_probe import (  # noqa: E402
    _parse_command,
    _summary_score,
    _trajectory_summary,
)


SEMANTIC_SMALL_ID = 1
SEMANTIC_LARGE_ID = 2

DEFAULT_COMMANDS = (
    "forward_v050:0.50 0.00 0.00",
    "forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00",
    "yaw100:0.00 0.00 1.00",
)

KNOWN_VARIANTS = (
    "parametric_v1",
    "baseline",
    "semantic_strong",
    "stance_only_semantic",
    "contact_only_semantic",
    "risk_strong",
    "crossing_strong",
    "risk_crossing",
    "risk_contact_crossing",
    "high_body_margin",
    "risk_stance_crossing",
    "body_stance_crossing",
    "body_stance_crossing_smooth",
    "crossing_contact_light",
    "body_light",
    "body_light_crossing_light",
    "body_light_touchdown_crossing",
    "hard_contact_crossing_light",
    "body_hard_contact_crossing_light",
    "body_hard_contact_only",
    "crossing_progress_only",
    "body_crossing_progress_only",
    "body_hard_contact_crossing_progress",
    "long_swing_crossing",
    "body_long_swing_crossing",
    "body_long_swing_hard_contact",
    "opt40_body_crossing_progress",
    "opt40_body_hard_contact_progress",
    "opt40_body_hard_contact_risk_progress",
    "opt40_body_hard_contact_highbody_progress",
    "foot_soft_cross_progress",
    "body_foot_soft_cross_progress",
    "opt40_body_foot_soft_cross_progress",
    "support_touchdown_cross_progress",
    "body_support_touchdown_cross_progress",
    "opt40_body_support_touchdown_cross_progress",
    "struct_lowfoot_highbody",
    "struct_lowfoot_highbody_strong",
    "opt40_struct_lowfoot_highbody",
    "struct_lowfoot_only",
    "struct_lowfoot_largebody",
    "struct_lowfoot_largebody_gentle",
    "struct_lowfoot_largebody_gentle_smooth",
    "struct_lowfoot_cross_hard",
    "opt40_struct_lowfoot_cross_hard",
    "select_baseline_gentle_smooth",
    "select_policy_pool",
    "select_policy_class_wide_margin",
    "select_policy_class_hardcross_margin",
    "select_policy_class_jitter_margin",
    "select_policy_class_risk_jitter_margin",
    "select_policy_class_priority_jitter_margin",
    "select_policy_class_clearance_jitter_margin",
    "select_policy_class_task_jitter_margin",
    "select_policy_class_straight_task_jitter_margin",
    "select_policy_class_path_task_jitter_margin",
    "select_policy_class_metric_task_jitter_margin",
    "select_policy_class_large_smooth_metric_margin",
    "straight_low_small_task",
    "straight_smooth_low_small_task",
    "path_tube_low_small_task",
    "path_tube_smooth_low_small_task",
    "loss_low_small_cross_v1",
    "loss_high_large_avoid_v1",
    "loss_continuity_anchor_v1",
    "loss_semantic_all_v1",
    "loss_low_small_cont_v2",
    "loss_low_small_stepcap_v3",
    "loss_low_small_stepcap_v4",
    "loss_low_small_footover_v1",
    "loss_low_small_footover_clear_v2",
    "loss_low_small_footover_cont_v3",
    "loss_low_small_footover_gate_v4",
    "loss_low_small_footover_gate_cont_v5",
    "loss_low_small_footover_gate_strong_v6",
    "loss_low_small_footover_leg_v7",
    "loss_low_small_footover_leg_cont_v8",
    "loss_low_small_footover_wide_v9",
    "loss_low_small_footover_wide_cont_v10",
    "loss_low_small_footover_cap_v11",
    "loss_low_small_footover_cap_strong_v12",
    "loss_low_small_footover_window_v13",
    "loss_low_small_footover_window_cont_v14",
    "loss_low_small_footover_coupled_v15",
    "loss_low_small_footover_coupled_cont_v16",
    "loss_low_small_footover_path_v17",
    "loss_low_small_footover_path_cont_v18",
    "loss_low_small_footover_pathweak_v19",
    "loss_low_small_footover_pathweak_cap_v20",
    "loss_high_large_escape_v2",
    "loss_semantic_all_v2",
    "loss_high_large_smooth_v3",
    "loss_semantic_all_v3",
    "loss_high_large_margin_v4",
    "loss_semantic_all_v4",
    "loss_high_large_balanced_v5",
    "loss_semantic_all_v5",
    "loss_high_large_scurve_v6",
    "loss_semantic_all_v6",
    "loss_high_large_handoff_v7",
    "loss_high_large_handoff_v8",
    "loss_high_large_handoff_v9",
    "loss_high_large_handoff_v10",
    "loss_high_large_ikfk_v11",
    "loss_high_large_ikfk_v12",
    "fk_output_probe",
    "phase_fixed_probe",
    "loss_high_large_fksemantic_v13",
    "fk_output_semantic_v14",
    "fk_output_smooth_v1",
    "fk_output_smooth_v2",
    "nominal_cmd_shape_a_v1",
    "nominal_cmd_shape_a_smooth_v2",
    "nominal_cmd_shape_a_low_small_guard_v3",
    "nominal_cmd_shape_a_conservative_v4",
    "nominal_cmd_shape_a_low_exact_v4",
    "nominal_cmd_shape_a_low_accel_v5",
    "nominal_cmd_shape_a_low_accel_anchor_v5",
    "nominal_cmd_shape_a_combined_v6",
    "nominal_cmd_shape_a_combined_v7",
    "nominal_cmd_shape_a_combined_v8",
    "nominal_cmd_shape_a_combined_v9",
    "nominal_cmd_shape_a_combined_v10",
    "smooth_strong",
    "body_hard_contact_only_smooth",
    "post_blend_body_hard_contact_only",
    "combined",
)

STRUCTURAL_VARIANTS = {
    "struct_lowfoot_highbody",
    "struct_lowfoot_highbody_strong",
    "opt40_struct_lowfoot_highbody",
    "struct_lowfoot_only",
    "struct_lowfoot_largebody",
    "struct_lowfoot_largebody_gentle",
    "struct_lowfoot_largebody_gentle_smooth",
    "struct_lowfoot_cross_hard",
    "opt40_struct_lowfoot_cross_hard",
}

SELECTOR_VARIANTS = {
    "select_baseline_gentle_smooth": (
        "baseline",
        "struct_lowfoot_largebody_gentle_smooth",
    ),
    "select_policy_pool": (
        "baseline",
        "struct_lowfoot_largebody_gentle_smooth",
        "struct_lowfoot_highbody",
    ),
}

LOW_SMALL_CROSSING_POOL = (
    "struct_lowfoot_highbody",
    "body_stance_crossing",
    "body_hard_contact_crossing_progress",
    "struct_lowfoot_largebody_gentle_smooth",
    "baseline",
)

HIGH_LARGE_AVOIDANCE_POOL = (
    "struct_lowfoot_largebody_gentle_smooth",
    "baseline",
    "body_light",
    "body_hard_contact_only",
    "opt40_body_hard_contact_progress",
)

RISK_HIGH_LARGE_AVOIDANCE_POOL = (
    "risk_strong",
    "opt40_body_hard_contact_risk_progress",
    "opt40_body_hard_contact_highbody_progress",
    "body_hard_contact_only",
    "struct_lowfoot_largebody_gentle_smooth",
    "body_light",
    "baseline",
)

SMOOTH_HIGH_LARGE_AVOIDANCE_POOL = (
    "post_blend_body_hard_contact_only",
    "body_hard_contact_only",
    "struct_lowfoot_largebody_gentle_smooth",
    "opt40_body_hard_contact_progress",
    "body_light",
    "baseline",
)

HARD_LOW_SMALL_CROSSING_POOL = (
    "struct_lowfoot_cross_hard",
    "opt40_struct_lowfoot_cross_hard",
    "body_stance_crossing",
    "struct_lowfoot_highbody",
    "body_hard_contact_crossing_progress",
    "baseline",
)

STRAIGHT_LOW_SMALL_CROSSING_POOL = (
    "path_tube_smooth_low_small_task",
    "path_tube_low_small_task",
    "straight_smooth_low_small_task",
    "straight_low_small_task",
    "struct_lowfoot_cross_hard",
    "opt40_struct_lowfoot_cross_hard",
    "struct_lowfoot_highbody",
    "body_stance_crossing",
    "body_hard_contact_crossing_progress",
    "struct_lowfoot_largebody_gentle_smooth",
    "baseline",
)


@dataclass(frozen=True)
class _StructuralSemanticWeights:
    low_small_foot_weight: float = 35.0
    low_small_foot_worst_weight: float = 12.0
    low_small_touchdown_weight: float = 18.0
    low_small_touchdown_worst_weight: float = 8.0
    high_body_weight: float = 14.0
    high_body_worst_weight: float = 28.0
    low_small_soft_margin_m: float = 0.24
    high_body_soft_margin_m: float = 0.30
    high_small_relative_height_m: float = 0.30
    body_stencil_radius_m: float = 0.18
    include_high_small_body: bool = True
    include_large_body: bool = True


@dataclass(frozen=True)
class _LowSmallStraightWeights:
    lateral_weight: float = 50.0
    lateral_worst_weight: float = 80.0
    reverse_weight: float = 25.0
    progress_weight: float = 30.0
    root_clearance_weight: float = 160.0
    root_clearance_worst_weight: float = 240.0
    lane_margin_m: float = 0.14
    obstacle_depth_m: float = 0.22
    pass_margin_m: float = 0.08
    corridor_width_m: float = 0.28
    forward_distance_m: float = 1.0
    high_small_relative_height_m: float = 0.30
    root_bottom_offset_z_m: float = -0.18
    root_clearance_margin_m: float = 0.04
    linear_speed_eps: float = 1.0e-4
    use_body_yaw_path: bool = False
    path_tube_weight: float = 0.0
    path_tube_worst_weight: float = 0.0


@dataclass(frozen=True)
class _LossOnlySemanticWeights:
    low_small_cross: bool = False
    high_large_avoid: bool = False
    continuity_anchor: bool = False
    low_small_path_weight: float = 1.0
    low_small_foot_weight: float = 0.75
    low_small_swing_clearance_weight: float = 120.0
    low_small_swing_clearance_worst_weight: float = 180.0
    low_small_swing_clearance_m: float = 0.16
    low_small_foot_over: bool = False
    low_small_foot_over_xy_weight: float = 180.0
    low_small_foot_over_direct_xy_weight: float = 0.0
    low_small_foot_over_leg_weight: float = 0.0
    low_small_foot_over_z_weight: float = 260.0
    low_small_foot_over_radius_m: float = 0.08
    low_small_foot_over_clearance_m: float = 0.04
    low_small_foot_over_ineligible_penalty: float = 0.25
    low_small_foot_over_time_gate_penalty: float = 0.0
    low_small_foot_over_along_window_m: float = 0.24
    low_small_foot_over_corridor_width_m: float = 0.30
    low_small_foot_over_forward_distance_m: float = 1.0
    low_small_foot_over_window_weight: float = 0.0
    low_small_foot_over_window_min_count: float = 3.0
    low_small_foot_over_window_sigma_m: float = 0.08
    low_small_foot_over_window_z_temp_m: float = 0.025
    low_small_foot_over_window_step_weight: float = 0.0
    low_small_foot_over_window_step_cap_m: float = 0.055
    low_small_foot_over_window_accel_weight: float = 0.0
    low_small_foot_over_window_accel_cap_m: float = 0.065
    low_small_foot_over_window_coupled: bool = False
    low_small_foot_over_path_curve_weight: float = 0.0
    low_small_foot_over_path_curve_z_weight: float = 0.0
    low_small_foot_over_path_curve_window_m: float = 0.28
    low_small_foot_over_path_curve_body_yaw: bool = False
    high_large_body_weight: float = 24.0
    high_large_body_worst_weight: float = 48.0
    high_large_corridor_weight: float = 80.0
    high_large_corridor_worst_weight: float = 140.0
    high_large_root_semantic_weight: float = 90.0
    high_large_root_semantic_worst_weight: float = 220.0
    high_large_lateral_escape_weight: float = 60.0
    high_large_distance_weight: float = 0.0
    high_large_distance_worst_weight: float = 0.0
    high_large_distance_margin_m: float = 0.30
    high_large_scurve_weight: float = 0.0
    high_large_scurve_worst_weight: float = 0.0
    high_large_scurve_lateral_m: float = 0.40
    high_large_scurve_temperature: float = 20.0
    high_large_lateral_clearance_m: float = 0.34
    high_large_corridor_width_m: float = 0.55
    high_large_forward_distance_m: float = 1.0
    high_large_longitudinal_influence_m: float = 0.55
    semantic_soft_margin_m: float = 0.28
    high_small_relative_height_m: float = 0.30
    body_stencil_radius_m: float = 0.20
    foot_step_worst_weight: float = 0.0
    foot_boundary_weight: float = 3.0
    foot_accel_worst_weight: float = 0.0
    foot_accel_weight: float = 18.0
    foot_step_cap_weight: float = 0.0
    foot_step_cap_m: float = 0.08
    foot_accel_cap_weight: float = 0.0
    foot_accel_cap_m: float = 0.10
    foot_jerk_weight: float = 6.0
    root_step_worst_weight: float = 0.0
    root_accel_worst_weight: float = 0.0
    root_accel_weight: float = 10.0
    first_foot_anchor_weight: float = 35.0
    first_foot_anchor_frames: int = 4
    high_large_handoff: bool = False
    high_large_fk_semantic: bool = False
    handoff_frames: int = 4
    handoff_foot_anchor_weight: float = 0.0
    handoff_foot_step_weight: float = 0.0
    handoff_foot_step_worst_weight: float = 0.0
    handoff_foot_accel_weight: float = 0.0
    handoff_foot_accel_worst_weight: float = 0.0
    handoff_root_step_weight: float = 0.0
    handoff_root_step_worst_weight: float = 0.0
    fk_stance_weight: float = 0.0
    fk_contact_weight: float = 0.0
    fk_touchdown_weight: float = 0.0
    fk_large_weight: float = 50.0
    fk_small_weight: float = 50.0
    fk_contact_activation_margin: float = 0.08
    fk_contact_worst_weight: float = 10.0


def _finite_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1.0e-9:
        return float("nan")
    return num / den


def _semantic_probe_seed(
    *,
    semantic_class: str,
    command_name: str,
    cycle: int,
    effective_candidate: str,
) -> int:
    key = f"{semantic_class}|{command_name}|{int(cycle)}|{effective_candidate}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little") & 0x7FFFFFFF


def _norm_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if int(values.numel()) == 0:
        return {
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_max_to_mean": float("nan"),
            f"{prefix}_max_to_median": float("nan"),
        }
    max_v = float(values.max().item())
    mean_v = float(values.mean().item())
    median_v = float(values.median().item())
    return {
        f"{prefix}_max": max_v,
        f"{prefix}_mean": mean_v,
        f"{prefix}_median": median_v,
        f"{prefix}_max_to_mean": _finite_ratio(max_v, mean_v),
        f"{prefix}_max_to_median": _finite_ratio(max_v, median_v),
    }


def _positive_median(values: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    values = torch.as_tensor(values)
    positive = values[values > float(eps)]
    if int(positive.numel()) > 0:
        return torch.median(positive).clamp_min(float(eps))
    return torch.median(values.reshape(-1)).clamp_min(float(eps))


def _jitter_metrics(result) -> dict[str, float]:
    root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
    root_step = torch.linalg.vector_norm(root[:, 1:] - root[:, :-1], dim=-1) if root.shape[1] > 1 else torch.empty(0)
    foot_step = (
        torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
        if foot.shape[1] > 1
        else torch.empty(0)
    )
    root_accel = (
        torch.linalg.vector_norm(root[:, 2:] - 2.0 * root[:, 1:-1] + root[:, :-2], dim=-1)
        if root.shape[1] > 2
        else torch.empty(0)
    )
    foot_accel = (
        torch.linalg.vector_norm(foot[:, 2:] - 2.0 * foot[:, 1:-1] + foot[:, :-2], dim=-1)
        if foot.shape[1] > 2
        else torch.empty(0)
    )
    out: dict[str, float] = {}
    out.update(_norm_stats(root_step, "root_step"))
    out.update(_norm_stats(root_accel, "root_accel"))
    out.update(_norm_stats(foot_step, "foot_step"))
    out.update(_norm_stats(foot_accel, "foot_accel"))
    out["root_accel_mean_for_ratio"] = out["root_accel_mean"]
    out["foot_accel_mean_for_ratio"] = out["foot_accel_mean"]
    out["root_max_to_median_step"] = out["root_step_max_to_median"]
    out["foot_max_to_median_step"] = out["foot_step_max_to_median"]
    if root_accel.numel() > 0:
        flat_idx = int(torch.argmax(root_accel.reshape(-1)).item())
        _, frame_idx = torch.unravel_index(torch.tensor(flat_idx, device=root_accel.device), root_accel.shape)
        out["worst_root_accel_frame"] = float(int(frame_idx.item()) + 1)
        out["worst_root_accel_value"] = float(root_accel[0, int(frame_idx.item())].item())
    else:
        out["worst_root_accel_frame"] = -1.0
        out["worst_root_accel_value"] = 0.0
    if foot_accel.numel() > 0:
        flat_idx = int(torch.argmax(foot_accel.reshape(-1)).item())
        _, frame_idx, leg_idx = torch.unravel_index(torch.tensor(flat_idx, device=foot_accel.device), foot_accel.shape)
        out["worst_foot_accel_frame"] = float(int(frame_idx.item()) + 1)
        out["worst_foot_accel_leg"] = float(int(leg_idx.item()))
        out["worst_foot_accel_value"] = float(foot_accel[0, int(frame_idx.item()), int(leg_idx.item())].item())
    else:
        out["worst_foot_accel_frame"] = -1.0
        out["worst_foot_accel_leg"] = -1.0
        out["worst_foot_accel_value"] = 0.0
    if foot.shape[1] > 1:
        step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
        median = torch.median(step.reshape(-1)).clamp_min(1.0e-9)
        flat_idx = int(torch.argmax((step / median).reshape(-1)).item())
        _, frame_idx, leg_idx = torch.unravel_index(torch.tensor(flat_idx, device=step.device), step.shape)
        out["worst_foot_step_frame"] = float(int(frame_idx.item()))
        out["worst_foot_step_leg"] = float(int(leg_idx.item()))
        out["worst_foot_step_value"] = float(step[0, int(frame_idx.item()), int(leg_idx.item())].item())
    else:
        out["worst_foot_step_frame"] = -1.0
        out["worst_foot_step_leg"] = -1.0
        out["worst_foot_step_value"] = 0.0
    return out


def _mask_rate(mask: torch.Tensor, denom_mask: torch.Tensor | None = None) -> float:
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if denom_mask is None:
        denom = int(mask.numel())
        num = int(torch.count_nonzero(mask).item())
    else:
        denom_mask = torch.as_tensor(denom_mask, dtype=torch.bool, device=mask.device)
        denom = int(torch.count_nonzero(denom_mask).item())
        num = int(torch.count_nonzero(mask & denom_mask).item())
    if denom <= 0:
        return 0.0
    return float(num) / float(denom)


def _semantic_collision_metrics(result, terrain) -> dict[str, float]:
    from extension.batch_mpc_planner.terrain import height_at, semantic_at

    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32)
    root = torch.as_tensor(result.root_pos_w, dtype=torch.float32)
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
    swing = torch.logical_not(contact)
    foot_sem = semantic_at(terrain, foot[..., :2])
    foot_height = height_at(terrain, foot[..., :2]).to(dtype=foot.dtype, device=foot.device)
    foot_penetrates = foot[..., 2] < foot_height
    foot_obstacle = foot_sem > 0
    foot_small = foot_sem == SEMANTIC_SMALL_ID
    foot_large = foot_sem == SEMANTIC_LARGE_ID

    root_sem = semantic_at(terrain, root[..., :2])
    root_height = height_at(terrain, root[..., :2]).to(dtype=root.dtype, device=root.device)
    root_bottom = root[..., 2] - 0.18
    root_penetrates = root_bottom < root_height
    root_small = root_sem == SEMANTIC_SMALL_ID
    root_large = root_sem == SEMANTIC_LARGE_ID

    touchdown = getattr(result, "planned_touchdown_w", None)
    if touchdown is None:
        touchdown_on_semantic_rate = 0.0
        touchdown_on_small_rate = 0.0
        touchdown_on_large_rate = 0.0
    else:
        touchdown = torch.as_tensor(touchdown, dtype=torch.float32, device=foot.device)
        touchdown_sem = semantic_at(terrain, touchdown[..., :2])
        touchdown_on_semantic_rate = _mask_rate(touchdown_sem > 0)
        touchdown_on_small_rate = _mask_rate(touchdown_sem == SEMANTIC_SMALL_ID)
        touchdown_on_large_rate = _mask_rate(touchdown_sem == SEMANTIC_LARGE_ID)

    return {
        "foot_on_semantic_rate": _mask_rate(foot_obstacle),
        "foot_on_small_rate": _mask_rate(foot_small),
        "foot_on_large_rate": _mask_rate(foot_large),
        "stance_on_semantic_rate": _mask_rate(foot_obstacle, contact),
        "stance_on_small_rate": _mask_rate(foot_small, contact),
        "stance_on_large_rate": _mask_rate(foot_large, contact),
        "swing_over_semantic_rate": _mask_rate(foot_obstacle, swing),
        "swing_over_small_rate": _mask_rate(foot_small, swing),
        "swing_over_large_rate": _mask_rate(foot_large, swing),
        "foot_semantic_penetration_rate": _mask_rate(foot_obstacle & foot_penetrates),
        "small_penetration_rate": _mask_rate(foot_small & foot_penetrates),
        "large_penetration_rate": _mask_rate(foot_large & foot_penetrates),
        "touchdown_on_semantic_rate": touchdown_on_semantic_rate,
        "touchdown_on_small_rate": touchdown_on_small_rate,
        "touchdown_on_large_rate": touchdown_on_large_rate,
        "root_on_semantic_rate": _mask_rate(root_sem > 0),
        "root_on_small_rate": _mask_rate(root_small),
        "root_on_large_rate": _mask_rate(root_large),
        "root_semantic_penetration_rate": _mask_rate((root_sem > 0) & root_penetrates),
    }


def _low_small_foot_over_metrics(
    result,
    terrain,
    obstacle_xy: torch.Tensor,
    *,
    semantic_target_height: float,
    lateral_margin_m: float = 0.08,
    min_clearance_m: float = 0.03,
) -> dict[str, float | int]:
    """Check whether a swing foot actually passes over the low-small footprint."""
    from extension.batch_mpc_planner.terrain import height_at, semantic_at

    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32)
    if foot.ndim != 4 or int(foot.shape[0]) <= 0:
        return {
            "foot_over_low_small_success": 0,
            "foot_over_low_small_rate": 0.0,
            "foot_over_low_small_frame_count": 0,
            "foot_over_low_small_max_clearance": 0.0,
            "foot_over_low_small_min_lateral": float("inf"),
        }
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
    swing = torch.logical_not(contact)
    obstacle_xy = torch.as_tensor(obstacle_xy, dtype=foot.dtype, device=foot.device)
    rel = foot[0, ..., :2] - obstacle_xy.view(1, 1, 2)
    lateral_dist = torch.linalg.vector_norm(rel, dim=-1)
    terrain_z = height_at(terrain, foot[..., :2])[0].to(dtype=foot.dtype, device=foot.device)
    clearance = foot[0, ..., 2] - terrain_z
    object_top_clearance = foot[0, ..., 2] - float(semantic_target_height)
    over_mask = (
        swing[0]
        & (lateral_dist <= float(lateral_margin_m))
        & (clearance >= float(min_clearance_m))
        & (object_top_clearance >= float(min_clearance_m))
    )
    frame_over = torch.any(over_mask, dim=-1)
    frame_count = int(torch.count_nonzero(frame_over).item())
    over_count = int(torch.count_nonzero(over_mask).item())
    swing_count = int(torch.count_nonzero(swing[0]).item())
    if over_count > 0:
        max_clearance = float(torch.where(over_mask, object_top_clearance, torch.zeros_like(object_top_clearance)).amax().item())
        min_lateral = float(torch.where(over_mask, lateral_dist, torch.full_like(lateral_dist, float("inf"))).amin().item())
    else:
        max_clearance = 0.0
        min_lateral = float(torch.amin(lateral_dist).item()) if int(lateral_dist.numel()) > 0 else float("inf")
    return {
        "foot_over_low_small_success": int(frame_count > 0),
        "foot_over_low_small_rate": float(over_count) / float(max(swing_count, 1)),
        "foot_over_low_small_frame_count": int(frame_count),
        "foot_over_low_small_max_clearance": max_clearance,
        "foot_over_low_small_min_lateral": min_lateral,
    }


def _terminal_foot_anomaly_metrics(
    result,
    terrain,
    *,
    replan_interval_steps: int = 25,
    step_ratio_limit: float = 8.0,
    stance_airborne_limit_m: float = 0.04,
) -> dict[str, float | int]:
    """Quantify the user-visible final-frame foot jump and airborne stance symptom."""
    from extension.batch_mpc_planner.terrain import height_at

    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float32)
    if foot.ndim != 4 or int(foot.shape[1]) <= 0:
        return {
            "any_foot_step_max": 0.0,
            "any_foot_step_frame": -1,
            "any_foot_step_leg": -1,
            "any_foot_step_to_median": 0.0,
            "terminal_foot_step_max": 0.0,
            "terminal_foot_step_leg": -1,
            "terminal_foot_step_to_median": 0.0,
            "replan_boundary_foot_step_max": 0.0,
            "replan_boundary_foot_step_frame": -1,
            "replan_boundary_foot_step_leg": -1,
            "replan_boundary_foot_step_to_median": 0.0,
            "replan_boundary_foot_anomaly_flag": 0,
            "terminal_stance_airborne_max": 0.0,
            "terminal_stance_airborne_leg": -1,
            "foot_step_anomaly_flag": 0,
            "terminal_foot_anomaly_flag": 0,
        }
    device = foot.device
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=device)
    if foot.shape[1] > 1:
        step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
        step_median = _positive_median(step)
        any_step_max_tensor, any_flat_idx = torch.max(step.reshape(-1), dim=0)
        _, any_frame_idx, any_leg_idx = torch.unravel_index(any_flat_idx, step.shape)
        any_step_max = float(any_step_max_tensor.item())
        any_step_to_median = _finite_ratio(any_step_max, float(step_median.item()))
        terminal_step = step[:, -1]
        reference_step = step[:, :-1] if int(step.shape[1]) > 1 else step
        median = _positive_median(reference_step)
        terminal_max, terminal_flat_leg = torch.max(terminal_step.reshape(-1), dim=0)
        terminal_leg = int(terminal_flat_leg.item()) % int(foot.shape[2])
        terminal_step_max = float(terminal_max.item())
        terminal_step_to_median = _finite_ratio(terminal_step_max, float(median.item()))
        interval = max(1, int(replan_interval_steps))
        boundary_start = interval - 1
        boundary_frames = (
            torch.arange(boundary_start, int(step.shape[1]), interval, device=device)
            if boundary_start < int(step.shape[1])
            else torch.empty((0,), dtype=torch.long, device=device)
        )
        if int(boundary_frames.numel()) > 0:
            boundary_step = step.index_select(1, boundary_frames)
            boundary_max_tensor, boundary_flat_idx = torch.max(boundary_step.reshape(-1), dim=0)
            _, boundary_local_frame_idx, boundary_leg_idx = torch.unravel_index(boundary_flat_idx, boundary_step.shape)
            boundary_frame = int(boundary_frames[int(boundary_local_frame_idx.item())].item())
            boundary_step_max = float(boundary_max_tensor.item())
            boundary_step_to_median = _finite_ratio(boundary_step_max, float(step_median.item()))
            boundary_leg = int(boundary_leg_idx.item())
        else:
            boundary_frame = -1
            boundary_leg = -1
            boundary_step_max = 0.0
            boundary_step_to_median = 0.0
    else:
        any_frame_idx = torch.tensor(-1, device=device)
        any_leg_idx = torch.tensor(-1, device=device)
        any_step_max = 0.0
        any_step_to_median = 0.0
        terminal_leg = -1
        terminal_step_max = 0.0
        terminal_step_to_median = 0.0
        boundary_frame = -1
        boundary_leg = -1
        boundary_step_max = 0.0
        boundary_step_to_median = 0.0

    final_foot = foot[:, -1]
    terrain_z = height_at(terrain, final_foot[..., :2]).to(dtype=foot.dtype, device=device)
    final_contact = contact[:, -1] if contact.ndim == 3 and int(contact.shape[1]) > 0 else torch.zeros_like(final_foot[..., 0], dtype=torch.bool)
    airborne = torch.relu(final_foot[..., 2] - terrain_z)
    stance_airborne = torch.where(final_contact, airborne, torch.zeros_like(airborne))
    stance_airborne_max_tensor, stance_flat_leg = torch.max(stance_airborne.reshape(-1), dim=0)
    stance_leg = int(stance_flat_leg.item()) % int(foot.shape[2])
    stance_airborne_max = float(stance_airborne_max_tensor.item())
    terminal_anomaly = int(
        terminal_step_to_median > float(step_ratio_limit)
        or stance_airborne_max > float(stance_airborne_limit_m)
    )
    step_anomaly = int(any_step_to_median > float(step_ratio_limit))
    boundary_anomaly = int(boundary_step_to_median > float(step_ratio_limit))
    return {
        "any_foot_step_max": float(any_step_max),
        "any_foot_step_frame": int(any_frame_idx.item()),
        "any_foot_step_leg": int(any_leg_idx.item()),
        "any_foot_step_to_median": float(any_step_to_median),
        "terminal_foot_step_max": float(terminal_step_max),
        "terminal_foot_step_leg": int(terminal_leg),
        "terminal_foot_step_to_median": float(terminal_step_to_median),
        "replan_boundary_foot_step_max": float(boundary_step_max),
        "replan_boundary_foot_step_frame": int(boundary_frame),
        "replan_boundary_foot_step_leg": int(boundary_leg),
        "replan_boundary_foot_step_to_median": float(boundary_step_to_median),
        "replan_boundary_foot_anomaly_flag": int(boundary_anomaly),
        "terminal_stance_airborne_max": float(stance_airborne_max),
        "terminal_stance_airborne_leg": int(stance_leg),
        "foot_step_anomaly_flag": int(step_anomaly),
        "terminal_foot_anomaly_flag": int(terminal_anomaly),
    }


def _rolling_segment_playback_error_metrics(result) -> dict[str, float | int]:
    foot_error = getattr(result, "rolling_segment_terminal_foot_error", None)
    root_error = getattr(result, "rolling_segment_terminal_root_error", None)
    if foot_error is None:
        return {
            "rolling_segment_terminal_foot_error_max": 0.0,
            "rolling_segment_terminal_foot_error_segment": -1,
            "rolling_segment_terminal_foot_error_leg": -1,
            "rolling_segment_terminal_root_error_max": 0.0,
        }
    foot_error_t = torch.as_tensor(foot_error, dtype=torch.float32)
    if int(foot_error_t.numel()) <= 0:
        return {
            "rolling_segment_terminal_foot_error_max": 0.0,
            "rolling_segment_terminal_foot_error_segment": -1,
            "rolling_segment_terminal_foot_error_leg": -1,
            "rolling_segment_terminal_root_error_max": 0.0,
        }
    flat_idx = int(torch.argmax(foot_error_t.reshape(-1)).item())
    segment_idx, leg_idx = torch.unravel_index(torch.tensor(flat_idx, device=foot_error_t.device), foot_error_t.shape)
    root_error_t = torch.as_tensor(root_error, dtype=torch.float32) if root_error is not None else torch.zeros((), dtype=torch.float32)
    return {
        "rolling_segment_terminal_foot_error_max": float(foot_error_t.reshape(-1)[flat_idx].item()),
        "rolling_segment_terminal_foot_error_segment": int(segment_idx.item()),
        "rolling_segment_terminal_foot_error_leg": int(leg_idx.item()),
        "rolling_segment_terminal_root_error_max": float(root_error_t.reshape(-1).max().item())
        if int(root_error_t.numel()) > 0
        else 0.0,
    }


def _tensor_xyz_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in torch.as_tensor(value, dtype=torch.float64).reshape(-1).tolist()]


_GO2_JOINT_LIMITS = torch.tensor(
    (
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
    ),
    dtype=torch.float64,
)
_GO2_JOINT_NAMES = (
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
)


def _rolling_segment_terminal_trace_rows(traces) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trace in traces or ():
        planned_foot = torch.as_tensor(trace["planned_foot"], dtype=torch.float64)
        actual_foot = torch.as_tensor(trace["actual_foot"], dtype=torch.float64)
        if planned_foot.numel() == 0 or actual_foot.numel() == 0:
            continue
        foot_error = actual_foot - planned_foot
        foot_error_norm = torch.linalg.vector_norm(foot_error, dim=-1)
        flat_idx = int(torch.argmax(foot_error_norm.reshape(-1)).item())
        batch_idx, leg_idx = torch.unravel_index(torch.tensor(flat_idx, device=foot_error_norm.device), foot_error_norm.shape)
        batch = int(batch_idx.item())
        leg = int(leg_idx.item())
        planned_joint = torch.as_tensor(trace.get("planned_joint", torch.empty(0)), dtype=torch.float64)
        actual_joint = torch.as_tensor(trace.get("actual_joint", torch.empty(0)), dtype=torch.float64)
        joint_error_max = 0.0
        if planned_joint.numel() > 0 and actual_joint.shape == planned_joint.shape:
            joint_error_max = float((actual_joint - planned_joint).abs().max().item())
        raw_joint = torch.as_tensor(trace.get("raw_ik_joint", torch.empty(0)), dtype=torch.float64)
        clamped_joint = torch.as_tensor(trace.get("clamped_ik_joint", torch.empty(0)), dtype=torch.float64)
        clamp_delta = torch.empty(0, dtype=torch.float64)
        clamp_delta_max = None
        clamp_delta_worst_joint = None
        clamp_delta_worst_joint_name = None
        raw_worst_joint_value = None
        clamped_worst_joint_value = None
        joint_limit_lower = None
        joint_limit_upper = None
        if raw_joint.numel() > 0 and clamped_joint.shape == raw_joint.shape:
            clamp_delta = clamped_joint - raw_joint
            abs_delta = clamp_delta.abs()
            clamp_delta_max = float(abs_delta.reshape(-1).max().item())
            worst_joint_flat = int(torch.argmax(abs_delta[batch].reshape(-1)).item())
            clamp_delta_worst_joint = worst_joint_flat
            if 0 <= worst_joint_flat < len(_GO2_JOINT_NAMES):
                clamp_delta_worst_joint_name = _GO2_JOINT_NAMES[worst_joint_flat]
                limits = _GO2_JOINT_LIMITS[worst_joint_flat]
                joint_limit_lower = float(limits[0].item())
                joint_limit_upper = float(limits[1].item())
            raw_worst_joint_value = float(raw_joint[batch, worst_joint_flat].item())
            clamped_worst_joint_value = float(clamped_joint[batch, worst_joint_flat].item())
        internal_fk = torch.as_tensor(trace.get("internal_fk_foot", torch.empty(0)), dtype=torch.float64)
        internal_fk_xyz = None
        internal_fk_error_norm = None
        actual_vs_internal_fk_error_norm = None
        if internal_fk.numel() > 0 and internal_fk.shape == planned_foot.shape:
            internal_fk_xyz = _tensor_xyz_list(internal_fk[batch, leg])
            internal_fk_error_norm = float(torch.linalg.vector_norm(internal_fk[batch, leg] - planned_foot[batch, leg]).item())
            actual_vs_internal_fk_error_norm = float(
                torch.linalg.vector_norm(actual_foot[batch, leg] - internal_fk[batch, leg]).item()
            )
        planned_root = torch.as_tensor(trace.get("planned_root", torch.empty(0)), dtype=torch.float64)
        actual_root = torch.as_tensor(trace.get("actual_root", torch.empty(0)), dtype=torch.float64)
        root_error_norm = 0.0
        if planned_root.numel() > 0 and actual_root.shape == planned_root.shape:
            root_error_norm = float(torch.linalg.vector_norm(actual_root - planned_root, dim=-1).reshape(-1).max().item())
        touchdown_xyz = None
        touchdown = trace.get("planned_touchdown", None)
        if touchdown is not None:
            touchdown_t = torch.as_tensor(touchdown, dtype=torch.float64)
            if touchdown_t.ndim == 3 and touchdown_t.shape[-2] > leg:
                touchdown_xyz = _tensor_xyz_list(touchdown_t[batch, leg])
            elif touchdown_t.ndim == 4 and touchdown_t.shape[-2] > leg:
                touchdown_xyz = _tensor_xyz_list(touchdown_t[batch, -1, leg])
        contact_value = -1
        contact_state = trace.get("contact_state", None)
        if contact_state is not None:
            contact_t = torch.as_tensor(contact_state)
            if contact_t.ndim >= 2 and contact_t.shape[-1] > leg:
                contact_value = int(bool(contact_t[batch, leg]))
        rows.append(
            {
                "type": "rolling_terminal_trace",
                "segment": int(trace.get("segment", -1)),
                "frame": int(trace.get("frame", -1)),
                "batch": batch,
                "worst_leg": leg,
                "foot_error_norm": float(foot_error_norm[batch, leg].item()),
                "foot_error_xyz": _tensor_xyz_list(foot_error[batch, leg]),
                "planned_foot_xyz": _tensor_xyz_list(planned_foot[batch, leg]),
                "actual_foot_xyz": _tensor_xyz_list(actual_foot[batch, leg]),
                "internal_fk_foot_xyz": internal_fk_xyz,
                "internal_fk_error_norm": internal_fk_error_norm,
                "actual_vs_internal_fk_error_norm": actual_vs_internal_fk_error_norm,
                "planned_root_xyz": _tensor_xyz_list(planned_root[batch]) if planned_root.numel() > 0 else None,
                "actual_root_xyz": _tensor_xyz_list(actual_root[batch]) if actual_root.numel() > 0 else None,
                "root_error_norm": root_error_norm,
                "joint_error_max_abs": joint_error_max,
                "ik_clamp_delta_max_abs": clamp_delta_max,
                "ik_clamp_worst_joint": clamp_delta_worst_joint,
                "ik_clamp_worst_joint_name": clamp_delta_worst_joint_name,
                "ik_raw_worst_joint_value": raw_worst_joint_value,
                "ik_clamped_worst_joint_value": clamped_worst_joint_value,
                "ik_worst_joint_limit_lower": joint_limit_lower,
                "ik_worst_joint_limit_upper": joint_limit_upper,
                "raw_ik_joint": _tensor_xyz_list(raw_joint[batch]) if raw_joint.numel() > 0 else None,
                "clamped_ik_joint": _tensor_xyz_list(clamped_joint[batch]) if clamped_joint.numel() > 0 else None,
                "ik_clamp_delta": _tensor_xyz_list(clamp_delta[batch]) if clamp_delta.numel() > 0 else None,
                "planned_joint": _tensor_xyz_list(planned_joint[batch]) if planned_joint.numel() > 0 else None,
                "actual_joint": _tensor_xyz_list(actual_joint[batch]) if actual_joint.numel() > 0 else None,
                "planned_touchdown_xyz": touchdown_xyz,
                "contact_state": contact_value,
            }
        )
    return rows


def _structural_weights_for_variant(variant: str) -> _StructuralSemanticWeights | None:
    if variant == "struct_lowfoot_highbody":
        return _StructuralSemanticWeights()
    if variant == "struct_lowfoot_highbody_strong":
        return _StructuralSemanticWeights(
            low_small_foot_weight=50.0,
            low_small_foot_worst_weight=18.0,
            low_small_touchdown_weight=24.0,
            low_small_touchdown_worst_weight=12.0,
            high_body_weight=20.0,
            high_body_worst_weight=42.0,
            low_small_soft_margin_m=0.28,
            high_body_soft_margin_m=0.34,
            body_stencil_radius_m=0.20,
        )
    if variant == "opt40_struct_lowfoot_highbody":
        return _StructuralSemanticWeights(
            low_small_foot_weight=45.0,
            low_small_foot_worst_weight=16.0,
            low_small_touchdown_weight=22.0,
            low_small_touchdown_worst_weight=10.0,
            high_body_weight=18.0,
            high_body_worst_weight=36.0,
            low_small_soft_margin_m=0.26,
            high_body_soft_margin_m=0.32,
            body_stencil_radius_m=0.20,
        )
    if variant == "struct_lowfoot_only":
        return _StructuralSemanticWeights(
            high_body_weight=0.0,
            high_body_worst_weight=0.0,
            include_high_small_body=False,
            include_large_body=False,
        )
    if variant == "struct_lowfoot_largebody":
        return _StructuralSemanticWeights(
            low_small_foot_weight=40.0,
            low_small_foot_worst_weight=14.0,
            low_small_touchdown_weight=20.0,
            low_small_touchdown_worst_weight=9.0,
            high_body_weight=12.0,
            high_body_worst_weight=24.0,
            low_small_soft_margin_m=0.25,
            high_body_soft_margin_m=0.28,
            body_stencil_radius_m=0.18,
            include_high_small_body=False,
            include_large_body=True,
        )
    if variant == "struct_lowfoot_largebody_gentle":
        return _StructuralSemanticWeights(
            low_small_foot_weight=36.0,
            low_small_foot_worst_weight=12.0,
            low_small_touchdown_weight=18.0,
            low_small_touchdown_worst_weight=8.0,
            high_body_weight=7.0,
            high_body_worst_weight=14.0,
            low_small_soft_margin_m=0.24,
            high_body_soft_margin_m=0.24,
            body_stencil_radius_m=0.16,
            include_high_small_body=False,
            include_large_body=True,
        )
    if variant == "struct_lowfoot_largebody_gentle_smooth":
        return _structural_weights_for_variant("struct_lowfoot_largebody_gentle")
    if variant == "struct_lowfoot_cross_hard":
        return _StructuralSemanticWeights(
            low_small_foot_weight=58.0,
            low_small_foot_worst_weight=22.0,
            low_small_touchdown_weight=30.0,
            low_small_touchdown_worst_weight=14.0,
            high_body_weight=8.0,
            high_body_worst_weight=16.0,
            low_small_soft_margin_m=0.30,
            high_body_soft_margin_m=0.24,
            body_stencil_radius_m=0.16,
            include_high_small_body=False,
            include_large_body=True,
        )
    if variant == "opt40_struct_lowfoot_cross_hard":
        return _structural_weights_for_variant("struct_lowfoot_cross_hard")
    return None


def _straight_weights_for_variant(variant: str) -> _LowSmallStraightWeights | None:
    if variant in {"straight_low_small_task", "straight_smooth_low_small_task"}:
        return _LowSmallStraightWeights()
    if variant in {"path_tube_low_small_task", "path_tube_smooth_low_small_task"}:
        return _LowSmallStraightWeights(
            lateral_weight=15.0,
            lateral_worst_weight=20.0,
            progress_weight=45.0,
            lane_margin_m=0.18,
            use_body_yaw_path=True,
            path_tube_weight=180.0,
            path_tube_worst_weight=260.0,
        )
    return None


def _loss_only_weights_for_variant(variant: str) -> _LossOnlySemanticWeights | None:
    if variant == "loss_low_small_cross_v1":
        return _LossOnlySemanticWeights(low_small_cross=True)
    if variant == "loss_high_large_avoid_v1":
        return _LossOnlySemanticWeights(high_large_avoid=True)
    if variant == "loss_continuity_anchor_v1":
        return _LossOnlySemanticWeights(continuity_anchor=True)
    if variant == "loss_semantic_all_v1":
        return _LossOnlySemanticWeights(
            low_small_cross=True,
            high_large_avoid=True,
            continuity_anchor=True,
            foot_boundary_weight=4.0,
            foot_accel_weight=22.0,
            foot_jerk_weight=8.0,
            root_accel_weight=12.0,
            first_foot_anchor_weight=45.0,
        )
    if variant == "loss_low_small_cont_v2":
        return _LossOnlySemanticWeights(
            low_small_cross=True,
            continuity_anchor=True,
            low_small_path_weight=1.15,
            low_small_foot_weight=0.75,
            foot_step_worst_weight=45.0,
            foot_accel_weight=26.0,
            foot_accel_worst_weight=80.0,
            foot_jerk_weight=12.0,
            root_accel_weight=16.0,
            root_accel_worst_weight=55.0,
            first_foot_anchor_weight=80.0,
            first_foot_anchor_frames=8,
        )
    if variant == "loss_low_small_stepcap_v3":
        base = _loss_only_weights_for_variant("loss_low_small_cont_v2")
        return replace(
            base,
            foot_step_worst_weight=180.0,
            foot_boundary_weight=12.0,
            foot_accel_weight=40.0,
            foot_accel_worst_weight=220.0,
            foot_jerk_weight=28.0,
            root_accel_weight=18.0,
            root_accel_worst_weight=70.0,
            first_foot_anchor_weight=45.0,
            first_foot_anchor_frames=6,
        )
    if variant == "loss_low_small_stepcap_v4":
        base = _loss_only_weights_for_variant("loss_low_small_cont_v2")
        return replace(
            base,
            foot_step_worst_weight=260.0,
            foot_boundary_weight=16.0,
            foot_accel_weight=46.0,
            foot_accel_worst_weight=300.0,
            foot_jerk_weight=36.0,
            root_step_worst_weight=80.0,
            root_accel_weight=20.0,
            root_accel_worst_weight=90.0,
            first_foot_anchor_weight=35.0,
            first_foot_anchor_frames=4,
        )
    if variant == "loss_low_small_footover_v1":
        base = _loss_only_weights_for_variant("loss_low_small_cont_v2")
        return replace(
            base,
            low_small_foot_over=True,
            low_small_foot_over_xy_weight=180.0,
            low_small_foot_over_z_weight=180.0,
            low_small_foot_over_radius_m=0.08,
            low_small_foot_over_clearance_m=0.035,
            low_small_swing_clearance_weight=100.0,
            low_small_swing_clearance_worst_weight=150.0,
        )
    if variant == "loss_low_small_footover_clear_v2":
        base = _loss_only_weights_for_variant("loss_low_small_footover_v1")
        return replace(
            base,
            low_small_foot_over_xy_weight=240.0,
            low_small_foot_over_z_weight=360.0,
            low_small_foot_over_clearance_m=0.055,
            low_small_swing_clearance_m=0.18,
            low_small_swing_clearance_weight=140.0,
            low_small_swing_clearance_worst_weight=220.0,
        )
    if variant == "loss_low_small_footover_cont_v3":
        base = _loss_only_weights_for_variant("loss_low_small_footover_clear_v2")
        return replace(
            base,
            foot_step_worst_weight=220.0,
            foot_boundary_weight=14.0,
            foot_accel_weight=42.0,
            foot_accel_worst_weight=260.0,
            foot_jerk_weight=34.0,
            root_step_worst_weight=70.0,
            root_accel_weight=20.0,
            root_accel_worst_weight=90.0,
            first_foot_anchor_weight=40.0,
            first_foot_anchor_frames=4,
        )
    if variant == "loss_low_small_footover_gate_v4":
        base = _loss_only_weights_for_variant("loss_low_small_cont_v2")
        return replace(
            base,
            low_small_foot_over=True,
            low_small_foot_over_xy_weight=220.0,
            low_small_foot_over_direct_xy_weight=260.0,
            low_small_foot_over_z_weight=320.0,
            low_small_foot_over_radius_m=0.08,
            low_small_foot_over_clearance_m=0.045,
            low_small_foot_over_ineligible_penalty=1.50,
            low_small_foot_over_time_gate_penalty=4.0,
            low_small_foot_over_along_window_m=0.26,
            low_small_swing_clearance_m=0.17,
            low_small_swing_clearance_weight=120.0,
            low_small_swing_clearance_worst_weight=180.0,
        )
    if variant == "loss_low_small_footover_gate_cont_v5":
        base = _loss_only_weights_for_variant("loss_low_small_footover_gate_v4")
        return replace(
            base,
            foot_step_worst_weight=240.0,
            foot_boundary_weight=14.0,
            foot_accel_weight=44.0,
            foot_accel_worst_weight=280.0,
            foot_jerk_weight=36.0,
            root_step_worst_weight=80.0,
            root_accel_weight=20.0,
            root_accel_worst_weight=90.0,
            first_foot_anchor_weight=35.0,
            first_foot_anchor_frames=4,
        )
    if variant == "loss_low_small_footover_gate_strong_v6":
        base = _loss_only_weights_for_variant("loss_low_small_footover_gate_cont_v5")
        return replace(
            base,
            low_small_foot_over_xy_weight=340.0,
            low_small_foot_over_direct_xy_weight=420.0,
            low_small_foot_over_z_weight=520.0,
            low_small_foot_over_clearance_m=0.06,
            low_small_foot_over_ineligible_penalty=2.5,
            low_small_foot_over_time_gate_penalty=8.0,
            low_small_swing_clearance_m=0.19,
        )
    if variant == "loss_low_small_footover_leg_v7":
        base = _loss_only_weights_for_variant("loss_low_small_footover_gate_v4")
        return replace(
            base,
            low_small_foot_over_xy_weight=90.0,
            low_small_foot_over_direct_xy_weight=70.0,
            low_small_foot_over_leg_weight=220.0,
            low_small_foot_over_z_weight=220.0,
            low_small_foot_over_ineligible_penalty=2.0,
            low_small_foot_over_time_gate_penalty=6.0,
            low_small_foot_over_clearance_m=0.045,
            foot_step_worst_weight=180.0,
            foot_boundary_weight=16.0,
            foot_accel_weight=58.0,
            foot_accel_worst_weight=420.0,
            foot_jerk_weight=52.0,
            root_step_worst_weight=80.0,
            root_accel_weight=24.0,
            root_accel_worst_weight=110.0,
        )
    if variant == "loss_low_small_footover_leg_cont_v8":
        base = _loss_only_weights_for_variant("loss_low_small_footover_leg_v7")
        return replace(
            base,
            low_small_foot_over_xy_weight=70.0,
            low_small_foot_over_direct_xy_weight=50.0,
            low_small_foot_over_leg_weight=180.0,
            low_small_foot_over_z_weight=180.0,
            foot_step_worst_weight=280.0,
            foot_boundary_weight=20.0,
            foot_accel_weight=72.0,
            foot_accel_worst_weight=620.0,
            foot_jerk_weight=76.0,
            root_step_worst_weight=100.0,
            root_accel_weight=28.0,
            root_accel_worst_weight=140.0,
        )
    if variant == "loss_low_small_footover_wide_v9":
        base = _loss_only_weights_for_variant("loss_low_small_cont_v2")
        return replace(
            base,
            low_small_foot_over=True,
            low_small_foot_over_xy_weight=20.0,
            low_small_foot_over_direct_xy_weight=420.0,
            low_small_foot_over_leg_weight=0.0,
            low_small_foot_over_z_weight=300.0,
            low_small_foot_over_radius_m=0.08,
            low_small_foot_over_clearance_m=0.045,
            low_small_foot_over_ineligible_penalty=4.0,
            low_small_foot_over_time_gate_penalty=10.0,
            low_small_foot_over_along_window_m=0.55,
            low_small_swing_clearance_m=0.17,
            foot_step_worst_weight=220.0,
            foot_boundary_weight=18.0,
            foot_accel_weight=64.0,
            foot_accel_worst_weight=520.0,
            foot_jerk_weight=70.0,
            root_step_worst_weight=90.0,
            root_accel_weight=24.0,
            root_accel_worst_weight=110.0,
            first_foot_anchor_weight=30.0,
            first_foot_anchor_frames=4,
        )
    if variant == "loss_low_small_footover_wide_cont_v10":
        base = _loss_only_weights_for_variant("loss_low_small_footover_wide_v9")
        return replace(
            base,
            low_small_foot_over_direct_xy_weight=300.0,
            low_small_foot_over_z_weight=240.0,
            low_small_foot_over_ineligible_penalty=6.0,
            low_small_foot_over_time_gate_penalty=14.0,
            low_small_foot_over_along_window_m=0.70,
            foot_step_worst_weight=360.0,
            foot_boundary_weight=24.0,
            foot_accel_weight=88.0,
            foot_accel_worst_weight=820.0,
            foot_jerk_weight=100.0,
            root_step_worst_weight=110.0,
            root_accel_weight=30.0,
            root_accel_worst_weight=150.0,
        )
    if variant == "loss_low_small_footover_cap_v11":
        base = _loss_only_weights_for_variant("loss_low_small_footover_gate_v4")
        return replace(
            base,
            low_small_foot_over_xy_weight=180.0,
            low_small_foot_over_direct_xy_weight=180.0,
            low_small_foot_over_z_weight=260.0,
            foot_step_worst_weight=260.0,
            foot_step_cap_weight=1800.0,
            foot_step_cap_m=0.075,
            foot_boundary_weight=18.0,
            foot_accel_weight=58.0,
            foot_accel_worst_weight=520.0,
            foot_accel_cap_weight=2200.0,
            foot_accel_cap_m=0.10,
            foot_jerk_weight=70.0,
            root_step_worst_weight=90.0,
            root_accel_weight=24.0,
            root_accel_worst_weight=110.0,
        )
    if variant == "loss_low_small_footover_cap_strong_v12":
        base = _loss_only_weights_for_variant("loss_low_small_footover_cap_v11")
        return replace(
            base,
            low_small_foot_over_xy_weight=140.0,
            low_small_foot_over_direct_xy_weight=130.0,
            low_small_foot_over_z_weight=220.0,
            foot_step_worst_weight=420.0,
            foot_step_cap_weight=3600.0,
            foot_step_cap_m=0.065,
            foot_accel_worst_weight=900.0,
            foot_accel_cap_weight=4800.0,
            foot_accel_cap_m=0.075,
            foot_jerk_weight=110.0,
            root_step_worst_weight=120.0,
            root_accel_weight=32.0,
            root_accel_worst_weight=160.0,
        )
    if variant == "loss_low_small_footover_window_v13":
        base = _loss_only_weights_for_variant("loss_low_small_cont_v2")
        return replace(
            base,
            low_small_foot_over=True,
            low_small_foot_over_xy_weight=80.0,
            low_small_foot_over_direct_xy_weight=120.0,
            low_small_foot_over_z_weight=180.0,
            low_small_foot_over_radius_m=0.08,
            low_small_foot_over_clearance_m=0.045,
            low_small_foot_over_ineligible_penalty=1.5,
            low_small_foot_over_time_gate_penalty=4.0,
            low_small_foot_over_along_window_m=0.30,
            low_small_foot_over_window_weight=420.0,
            low_small_foot_over_window_min_count=4.0,
            low_small_foot_over_window_sigma_m=0.08,
            low_small_foot_over_window_z_temp_m=0.025,
            low_small_foot_over_window_step_weight=900.0,
            low_small_foot_over_window_step_cap_m=0.055,
            low_small_foot_over_window_accel_weight=1200.0,
            low_small_foot_over_window_accel_cap_m=0.065,
            low_small_swing_clearance_m=0.17,
            foot_step_worst_weight=180.0,
            foot_boundary_weight=16.0,
            foot_accel_weight=52.0,
            foot_accel_worst_weight=420.0,
            foot_jerk_weight=60.0,
            root_step_worst_weight=80.0,
            root_accel_weight=22.0,
            root_accel_worst_weight=100.0,
            first_foot_anchor_weight=30.0,
            first_foot_anchor_frames=4,
        )
    if variant == "loss_low_small_footover_window_cont_v14":
        base = _loss_only_weights_for_variant("loss_low_small_footover_window_v13")
        return replace(
            base,
            low_small_foot_over_xy_weight=60.0,
            low_small_foot_over_direct_xy_weight=90.0,
            low_small_foot_over_z_weight=150.0,
            low_small_foot_over_window_weight=520.0,
            low_small_foot_over_window_min_count=5.0,
            low_small_foot_over_window_step_weight=1500.0,
            low_small_foot_over_window_step_cap_m=0.045,
            low_small_foot_over_window_accel_weight=2200.0,
            low_small_foot_over_window_accel_cap_m=0.050,
            foot_step_worst_weight=260.0,
            foot_boundary_weight=20.0,
            foot_accel_weight=68.0,
            foot_accel_worst_weight=680.0,
            foot_jerk_weight=86.0,
            root_step_worst_weight=100.0,
            root_accel_weight=28.0,
            root_accel_worst_weight=140.0,
        )
    if variant == "loss_low_small_footover_coupled_v15":
        base = _loss_only_weights_for_variant("loss_low_small_footover_window_v13")
        return replace(
            base,
            low_small_foot_over_window_coupled=True,
        )
    if variant == "loss_low_small_footover_coupled_cont_v16":
        base = _loss_only_weights_for_variant("loss_low_small_footover_window_cont_v14")
        return replace(
            base,
            low_small_foot_over_window_coupled=True,
        )
    if variant == "loss_low_small_footover_path_v17":
        base = _loss_only_weights_for_variant("loss_low_small_footover_gate_v4")
        return replace(
            base,
            low_small_foot_over_xy_weight=110.0,
            low_small_foot_over_direct_xy_weight=160.0,
            low_small_foot_over_z_weight=240.0,
            low_small_foot_over_path_curve_weight=520.0,
            low_small_foot_over_path_curve_z_weight=180.0,
            low_small_foot_over_path_curve_window_m=0.30,
            foot_step_worst_weight=220.0,
            foot_boundary_weight=16.0,
            foot_accel_weight=58.0,
            foot_accel_worst_weight=520.0,
            foot_jerk_weight=70.0,
            root_step_worst_weight=90.0,
            root_accel_weight=24.0,
            root_accel_worst_weight=110.0,
        )
    if variant == "loss_low_small_footover_path_cont_v18":
        base = _loss_only_weights_for_variant("loss_low_small_footover_path_v17")
        return replace(
            base,
            low_small_foot_over_xy_weight=90.0,
            low_small_foot_over_direct_xy_weight=120.0,
            low_small_foot_over_z_weight=220.0,
            low_small_foot_over_path_curve_weight=720.0,
            low_small_foot_over_path_curve_z_weight=220.0,
            foot_step_worst_weight=360.0,
            foot_boundary_weight=22.0,
            foot_accel_weight=82.0,
            foot_accel_worst_weight=820.0,
            foot_jerk_weight=100.0,
            root_step_worst_weight=120.0,
            root_accel_weight=30.0,
            root_accel_worst_weight=150.0,
        )
    if variant == "loss_low_small_footover_pathweak_v19":
        base = _loss_only_weights_for_variant("loss_low_small_footover_gate_v4")
        return replace(
            base,
            low_small_foot_over_path_curve_weight=120.0,
            low_small_foot_over_path_curve_z_weight=60.0,
            low_small_foot_over_path_curve_window_m=0.30,
            foot_step_worst_weight=220.0,
            foot_boundary_weight=16.0,
            foot_accel_weight=56.0,
            foot_accel_worst_weight=460.0,
            foot_jerk_weight=64.0,
            root_step_worst_weight=90.0,
            root_accel_weight=24.0,
            root_accel_worst_weight=110.0,
        )
    if variant == "loss_low_small_footover_pathweak_cap_v20":
        base = _loss_only_weights_for_variant("loss_low_small_footover_pathweak_v19")
        return replace(
            base,
            low_small_foot_over_path_curve_weight=90.0,
            low_small_foot_over_path_curve_z_weight=45.0,
            foot_step_worst_weight=300.0,
            foot_boundary_weight=20.0,
            foot_accel_weight=70.0,
            foot_accel_worst_weight=650.0,
            foot_step_cap_weight=1800.0,
            foot_step_cap_m=0.085,
            foot_accel_cap_weight=1800.0,
            foot_accel_cap_m=0.11,
            foot_jerk_weight=90.0,
            root_step_worst_weight=110.0,
            root_accel_weight=28.0,
            root_accel_worst_weight=130.0,
        )
    if variant == "loss_high_large_escape_v2":
        return _LossOnlySemanticWeights(
            high_large_avoid=True,
            continuity_anchor=True,
            high_large_body_weight=34.0,
            high_large_body_worst_weight=78.0,
            high_large_corridor_weight=150.0,
            high_large_corridor_worst_weight=280.0,
            high_large_root_semantic_weight=180.0,
            high_large_root_semantic_worst_weight=420.0,
            high_large_lateral_escape_weight=180.0,
            high_large_lateral_clearance_m=0.46,
            high_large_corridor_width_m=0.70,
            high_large_forward_distance_m=1.20,
            high_large_longitudinal_influence_m=0.75,
            semantic_soft_margin_m=0.36,
            body_stencil_radius_m=0.24,
            foot_step_worst_weight=35.0,
            foot_accel_weight=18.0,
            foot_accel_worst_weight=60.0,
            root_accel_weight=16.0,
            root_accel_worst_weight=60.0,
            first_foot_anchor_weight=70.0,
            first_foot_anchor_frames=8,
        )
    if variant == "loss_semantic_all_v2":
        return _LossOnlySemanticWeights(
            low_small_cross=True,
            high_large_avoid=True,
            continuity_anchor=True,
            low_small_path_weight=1.10,
            low_small_foot_weight=0.70,
            high_large_body_weight=32.0,
            high_large_body_worst_weight=72.0,
            high_large_corridor_weight=135.0,
            high_large_corridor_worst_weight=240.0,
            high_large_root_semantic_weight=150.0,
            high_large_root_semantic_worst_weight=360.0,
            high_large_lateral_escape_weight=150.0,
            high_large_lateral_clearance_m=0.44,
            high_large_corridor_width_m=0.65,
            high_large_forward_distance_m=1.20,
            high_large_longitudinal_influence_m=0.70,
            semantic_soft_margin_m=0.34,
            body_stencil_radius_m=0.22,
            foot_step_worst_weight=45.0,
            foot_boundary_weight=4.0,
            foot_accel_weight=24.0,
            foot_accel_worst_weight=80.0,
            foot_jerk_weight=12.0,
            root_accel_weight=16.0,
            root_accel_worst_weight=60.0,
            first_foot_anchor_weight=90.0,
            first_foot_anchor_frames=8,
        )
    if variant == "loss_high_large_smooth_v3":
        return _LossOnlySemanticWeights(
            high_large_avoid=True,
            continuity_anchor=True,
            high_large_body_weight=24.0,
            high_large_body_worst_weight=56.0,
            high_large_corridor_weight=90.0,
            high_large_corridor_worst_weight=150.0,
            high_large_root_semantic_weight=110.0,
            high_large_root_semantic_worst_weight=260.0,
            high_large_lateral_escape_weight=70.0,
            high_large_lateral_clearance_m=0.36,
            high_large_corridor_width_m=0.55,
            high_large_forward_distance_m=1.10,
            high_large_longitudinal_influence_m=0.60,
            semantic_soft_margin_m=0.30,
            body_stencil_radius_m=0.20,
            foot_step_worst_weight=120.0,
            foot_boundary_weight=10.0,
            foot_accel_weight=30.0,
            foot_accel_worst_weight=180.0,
            foot_jerk_weight=24.0,
            root_step_worst_weight=70.0,
            root_accel_weight=24.0,
            root_accel_worst_weight=120.0,
            first_foot_anchor_weight=130.0,
            first_foot_anchor_frames=12,
        )
    if variant == "loss_semantic_all_v3":
        return _LossOnlySemanticWeights(
            low_small_cross=True,
            high_large_avoid=True,
            continuity_anchor=True,
            low_small_path_weight=1.10,
            low_small_foot_weight=0.70,
            high_large_body_weight=24.0,
            high_large_body_worst_weight=56.0,
            high_large_corridor_weight=90.0,
            high_large_corridor_worst_weight=150.0,
            high_large_root_semantic_weight=110.0,
            high_large_root_semantic_worst_weight=260.0,
            high_large_lateral_escape_weight=70.0,
            high_large_lateral_clearance_m=0.36,
            high_large_corridor_width_m=0.55,
            high_large_forward_distance_m=1.10,
            high_large_longitudinal_influence_m=0.60,
            semantic_soft_margin_m=0.30,
            body_stencil_radius_m=0.20,
            foot_step_worst_weight=120.0,
            foot_boundary_weight=10.0,
            foot_accel_weight=30.0,
            foot_accel_worst_weight=180.0,
            foot_jerk_weight=24.0,
            root_step_worst_weight=70.0,
            root_accel_weight=24.0,
            root_accel_worst_weight=120.0,
            first_foot_anchor_weight=130.0,
            first_foot_anchor_frames=12,
        )
    if variant == "loss_high_large_margin_v4":
        return _LossOnlySemanticWeights(
            high_large_avoid=True,
            continuity_anchor=True,
            high_large_body_weight=24.0,
            high_large_body_worst_weight=56.0,
            high_large_corridor_weight=75.0,
            high_large_corridor_worst_weight=120.0,
            high_large_root_semantic_weight=90.0,
            high_large_root_semantic_worst_weight=220.0,
            high_large_lateral_escape_weight=45.0,
            high_large_distance_weight=140.0,
            high_large_distance_worst_weight=300.0,
            high_large_distance_margin_m=0.32,
            high_large_lateral_clearance_m=0.34,
            high_large_corridor_width_m=0.55,
            high_large_forward_distance_m=1.10,
            high_large_longitudinal_influence_m=0.58,
            semantic_soft_margin_m=0.30,
            body_stencil_radius_m=0.20,
            foot_step_worst_weight=140.0,
            foot_boundary_weight=10.0,
            foot_accel_weight=32.0,
            foot_accel_worst_weight=200.0,
            foot_jerk_weight=26.0,
            root_step_worst_weight=90.0,
            root_accel_weight=26.0,
            root_accel_worst_weight=140.0,
            first_foot_anchor_weight=150.0,
            first_foot_anchor_frames=14,
        )
    if variant == "loss_semantic_all_v4":
        base = _loss_only_weights_for_variant("loss_high_large_margin_v4")
        return replace(
            base,
            low_small_cross=True,
            low_small_path_weight=1.10,
            low_small_foot_weight=0.70,
        )
    if variant == "loss_high_large_balanced_v5":
        return _LossOnlySemanticWeights(
            high_large_avoid=True,
            continuity_anchor=True,
            high_large_body_weight=30.0,
            high_large_body_worst_weight=70.0,
            high_large_corridor_weight=130.0,
            high_large_corridor_worst_weight=240.0,
            high_large_root_semantic_weight=150.0,
            high_large_root_semantic_worst_weight=360.0,
            high_large_lateral_escape_weight=125.0,
            high_large_distance_weight=80.0,
            high_large_distance_worst_weight=180.0,
            high_large_distance_margin_m=0.28,
            high_large_lateral_clearance_m=0.42,
            high_large_corridor_width_m=0.66,
            high_large_forward_distance_m=1.20,
            high_large_longitudinal_influence_m=0.70,
            semantic_soft_margin_m=0.34,
            body_stencil_radius_m=0.22,
            foot_step_worst_weight=170.0,
            foot_boundary_weight=14.0,
            foot_accel_weight=38.0,
            foot_accel_worst_weight=260.0,
            foot_jerk_weight=34.0,
            root_step_worst_weight=120.0,
            root_accel_weight=34.0,
            root_accel_worst_weight=190.0,
            first_foot_anchor_weight=180.0,
            first_foot_anchor_frames=16,
        )
    if variant == "loss_semantic_all_v5":
        base = _loss_only_weights_for_variant("loss_high_large_balanced_v5")
        return replace(
            base,
            low_small_cross=True,
            low_small_path_weight=1.10,
            low_small_foot_weight=0.70,
        )
    if variant == "loss_high_large_scurve_v6":
        return _LossOnlySemanticWeights(
            high_large_avoid=True,
            continuity_anchor=True,
            high_large_body_weight=22.0,
            high_large_body_worst_weight=50.0,
            high_large_corridor_weight=45.0,
            high_large_corridor_worst_weight=80.0,
            high_large_root_semantic_weight=80.0,
            high_large_root_semantic_worst_weight=180.0,
            high_large_lateral_escape_weight=20.0,
            high_large_distance_weight=80.0,
            high_large_distance_worst_weight=180.0,
            high_large_distance_margin_m=0.30,
            high_large_scurve_weight=260.0,
            high_large_scurve_worst_weight=520.0,
            high_large_scurve_lateral_m=0.42,
            high_large_lateral_clearance_m=0.34,
            high_large_corridor_width_m=0.62,
            high_large_forward_distance_m=1.20,
            high_large_longitudinal_influence_m=0.70,
            semantic_soft_margin_m=0.32,
            body_stencil_radius_m=0.20,
            foot_step_worst_weight=180.0,
            foot_boundary_weight=14.0,
            foot_accel_weight=40.0,
            foot_accel_worst_weight=280.0,
            foot_jerk_weight=38.0,
            root_step_worst_weight=120.0,
            root_accel_weight=34.0,
            root_accel_worst_weight=190.0,
            first_foot_anchor_weight=180.0,
            first_foot_anchor_frames=16,
        )
    if variant == "loss_semantic_all_v6":
        base = _loss_only_weights_for_variant("loss_high_large_scurve_v6")
        return replace(
            base,
            low_small_cross=True,
            low_small_path_weight=1.10,
            low_small_foot_weight=0.70,
        )
    if variant == "loss_high_large_handoff_v7":
        return _LossOnlySemanticWeights(
            continuity_anchor=True,
            foot_step_worst_weight=320.0,
            foot_boundary_weight=12.0,
            foot_accel_weight=24.0,
            foot_accel_worst_weight=140.0,
            foot_jerk_weight=22.0,
            root_step_worst_weight=80.0,
            root_accel_weight=14.0,
            root_accel_worst_weight=60.0,
            first_foot_anchor_weight=260.0,
            first_foot_anchor_frames=6,
        )
    if variant == "loss_high_large_handoff_v8":
        return _LossOnlySemanticWeights(
            continuity_anchor=True,
            foot_step_worst_weight=520.0,
            foot_boundary_weight=18.0,
            foot_accel_weight=34.0,
            foot_accel_worst_weight=220.0,
            foot_jerk_weight=34.0,
            root_step_worst_weight=120.0,
            root_accel_weight=18.0,
            root_accel_worst_weight=90.0,
            first_foot_anchor_weight=420.0,
            first_foot_anchor_frames=8,
        )
    if variant == "loss_high_large_handoff_v9":
        return _LossOnlySemanticWeights(
            high_large_handoff=True,
            handoff_frames=4,
            handoff_foot_anchor_weight=520.0,
            handoff_foot_step_weight=90.0,
            handoff_foot_step_worst_weight=380.0,
            handoff_foot_accel_weight=80.0,
            handoff_foot_accel_worst_weight=260.0,
            handoff_root_step_weight=35.0,
            handoff_root_step_worst_weight=120.0,
        )
    if variant == "loss_high_large_handoff_v10":
        return _LossOnlySemanticWeights(
            high_large_handoff=True,
            handoff_frames=6,
            handoff_foot_anchor_weight=760.0,
            handoff_foot_step_weight=130.0,
            handoff_foot_step_worst_weight=560.0,
            handoff_foot_accel_weight=120.0,
            handoff_foot_accel_worst_weight=420.0,
            handoff_root_step_weight=50.0,
            handoff_root_step_worst_weight=180.0,
        )
    if variant in {"loss_high_large_fksemantic_v13", "fk_output_semantic_v14"}:
        return _LossOnlySemanticWeights(
            high_large_fk_semantic=True,
            fk_stance_weight=120.0,
            fk_contact_weight=180.0,
            fk_touchdown_weight=80.0,
            fk_large_weight=80.0,
            fk_small_weight=80.0,
            fk_contact_activation_margin=0.10,
            fk_contact_worst_weight=16.0,
        )
    if variant == "nominal_cmd_shape_a_low_small_guard_v3":
        return _loss_only_weights_for_variant("loss_low_small_cont_v2")
    if variant == "nominal_cmd_shape_a_low_exact_v4":
        return _loss_only_weights_for_variant("loss_low_small_cont_v2")
    if variant == "nominal_cmd_shape_a_low_accel_v5":
        base = _loss_only_weights_for_variant("loss_low_small_cont_v2")
        return replace(
            base,
            foot_boundary_weight=6.0,
            foot_accel_weight=46.0,
            foot_accel_worst_weight=180.0,
            foot_jerk_weight=24.0,
            root_accel_weight=18.0,
            root_accel_worst_weight=70.0,
            first_foot_anchor_weight=65.0,
        )
    if variant == "nominal_cmd_shape_a_low_accel_anchor_v5":
        base = _loss_only_weights_for_variant("nominal_cmd_shape_a_low_accel_v5")
        return replace(
            base,
            foot_boundary_weight=8.0,
            foot_step_worst_weight=70.0,
            foot_accel_weight=52.0,
            foot_accel_worst_weight=220.0,
            foot_jerk_weight=30.0,
            first_foot_anchor_weight=120.0,
            first_foot_anchor_frames=12,
        )
    if variant == "nominal_cmd_shape_a_combined_v6":
        return _loss_only_weights_for_variant("nominal_cmd_shape_a_low_accel_anchor_v5")
    return None


def _semantic_height_class_masks(
    terrain,
    root_pos: torch.Tensor,
    *,
    high_small_relative_height_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    from extension.batch_mpc_planner.terrain import height_at
    from extension.batch_mpc_planner.losses.terrain_clearance import _nearby_height_for_sparse_semantic

    if terrain.semantic_map is None:
        return None
    batch = int(root_pos.shape[0])
    dtype = root_pos.dtype
    device = root_pos.device
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    if int(height.shape[0]) == 1 and batch > 1:
        height = height.expand(batch, -1, -1)
    if int(semantic.shape[0]) != batch or int(height.shape[0]) != batch:
        return None
    nearby_height = _nearby_height_for_sparse_semantic(
        terrain,
        height,
        dtype=dtype,
        device=device,
    ).reshape(batch, int(height.shape[-2]), int(height.shape[-1]))
    root0_ground = height_at(terrain, root_pos[:, :1, :2]).reshape(batch, 1, 1).to(dtype=dtype, device=device)
    small = semantic == SEMANTIC_SMALL_ID
    high_small = torch.logical_and(small, (nearby_height - root0_ground) > float(high_small_relative_height_m))
    low_small = torch.logical_and(small, torch.logical_not(high_small))
    large = semantic == SEMANTIC_LARGE_ID
    return low_small, high_small, large


def _terrain_grid_world_xy_for_probe(terrain, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        batch = 1
        rows, cols = int(height.shape[-2]), int(height.shape[-1])
    else:
        batch = int(height.shape[0])
        rows, cols = int(height.shape[-2]), int(height.shape[-1])
    xs = torch.linspace(float(terrain.world_x_range[0]), float(terrain.world_x_range[1]), cols, dtype=dtype, device=device)
    ys = torch.linspace(float(terrain.world_y_range[0]), float(terrain.world_y_range[1]), rows, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    local_xy = torch.stack((xx, yy), dim=-1).reshape(1, rows * cols, 2).expand(batch, -1, -1)
    if terrain.sensor_pos_w is None:
        return local_xy
    sensor_pos = torch.as_tensor(terrain.sensor_pos_w, dtype=dtype, device=device)
    if sensor_pos.ndim == 1:
        sensor_pos = sensor_pos.view(1, -1).expand(batch, -1)
    elif int(sensor_pos.shape[0]) == 1 and batch > 1:
        sensor_pos = sensor_pos.expand(batch, -1)
    if terrain.sensor_yaw is None:
        yaw = torch.zeros((batch,), dtype=dtype, device=device)
    else:
        yaw = torch.as_tensor(terrain.sensor_yaw, dtype=dtype, device=device).reshape(-1)
        if int(yaw.shape[0]) == 1 and batch > 1:
            yaw = yaw.expand(batch)
    cy = torch.cos(yaw).view(batch, 1)
    sy = torch.sin(yaw).view(batch, 1)
    world_xy = torch.stack(
        (
            cy * local_xy[..., 0] - sy * local_xy[..., 1],
            sy * local_xy[..., 0] + cy * local_xy[..., 1],
        ),
        dim=-1,
    )
    return world_xy + sensor_pos[:, None, :2]


def _semantic_low_small_straight_extra_loss(
    root_pos: torch.Tensor,
    root_rpy: torch.Tensor,
    command: torch.Tensor,
    terrain,
    *,
    weights: _LowSmallStraightWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from extension.batch_mpc_planner.terrain import height_at
    from extension.batch_mpc_planner.losses.terrain_clearance import _nearby_height_for_sparse_semantic

    weights = weights or _LowSmallStraightWeights()
    root_pos = torch.as_tensor(root_pos)
    root_rpy = torch.as_tensor(root_rpy, dtype=root_pos.dtype, device=root_pos.device)
    batch = int(root_pos.shape[0])
    dtype = root_pos.dtype
    device = root_pos.device
    zero = torch.zeros((batch,), dtype=dtype, device=device)
    if terrain.semantic_map is None or int(root_pos.shape[1]) < 2:
        return zero, {
            "test_low_small_straight_lateral": zero,
            "test_low_small_straight_reverse": zero,
            "test_low_small_straight_progress": zero,
        }

    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    if int(height.shape[0]) == 1 and batch > 1:
        height = height.expand(batch, -1, -1)
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    if int(height.shape[0]) != batch or int(semantic.shape[0]) != batch:
        return zero, {
            "test_low_small_straight_lateral": zero,
            "test_low_small_straight_reverse": zero,
            "test_low_small_straight_progress": zero,
        }

    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    cmd_xy = cmd[:, :2]
    cmd_speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
    active = cmd_speed > float(weights.linear_speed_eps)
    heading = cmd_xy / cmd_speed.clamp_min(1.0e-6).unsqueeze(-1)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)

    grid_xy = _terrain_grid_world_xy_for_probe(terrain, dtype=dtype, device=device)
    if int(grid_xy.shape[0]) == 1 and batch > 1:
        grid_xy = grid_xy.expand(batch, -1, -1)
    grid_sem = semantic.reshape(batch, -1)
    nearby_grid_z = _nearby_height_for_sparse_semantic(terrain, height, dtype=dtype, device=device)
    root0 = root_pos[:, 0]
    root_ground_z = height_at(terrain, root0[:, None, :2]).reshape(batch).to(dtype=dtype, device=device)
    low_small = torch.logical_and(
        grid_sem == SEMANTIC_SMALL_ID,
        (nearby_grid_z - root_ground_z[:, None]) <= float(weights.high_small_relative_height_m),
    )
    delta = grid_xy - root0[:, None, :2]
    obstacle_forward = (delta * heading[:, None, :]).sum(dim=-1)
    obstacle_lateral = (delta * left[:, None, :]).sum(dim=-1)
    candidate = torch.logical_and(
        low_small,
        torch.logical_and(
            torch.logical_and(obstacle_forward >= 0.0, obstacle_forward <= float(weights.forward_distance_m)),
            torch.abs(obstacle_lateral) <= float(weights.corridor_width_m),
        ),
    )
    candidate = torch.logical_and(candidate, active[:, None])
    candidate_f = candidate.to(dtype=dtype, device=device)
    count = candidate_f.sum(dim=-1)
    obstacle_lateral_center = (obstacle_lateral * candidate_f).sum(dim=-1) / count.clamp_min(1.0)
    obstacle_forward_max = torch.where(candidate, obstacle_forward, torch.zeros_like(obstacle_forward)).amax(dim=-1)
    required_progress = obstacle_forward_max + float(weights.obstacle_depth_m) + float(weights.pass_margin_m)

    rel_path = root_pos[..., :2] - root0[:, None, :2]
    along = (rel_path * heading[:, None, :]).sum(dim=-1)
    lateral = (rel_path * left[:, None, :]).sum(dim=-1) - obstacle_lateral_center[:, None]
    active_path = count > 0.0
    path_tube_loss = zero
    path_tube_worst = zero
    if bool(weights.use_body_yaw_path):
        yaw = root_rpy[:, :-1, 2]
        body_step = cmd[:, None, :2] * 0.02
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        expected_step = torch.stack(
            (
                cy * body_step[..., 0] - sy * body_step[..., 1],
                sy * body_step[..., 0] + cy * body_step[..., 1],
            ),
            dim=-1,
        )
        expected_xy = torch.zeros_like(root_pos[..., :2])
        expected_xy[:, :1] = root_pos[:, :1, :2]
        expected_xy[:, 1:] = root_pos[:, :1, :2] + torch.cumsum(expected_step, dim=1)
        path_err = torch.linalg.vector_norm(root_pos[..., :2] - expected_xy, dim=-1)
        path_deficit = torch.relu(path_err - float(weights.lane_margin_m))
        path_tube_loss = torch.where(active_path, path_deficit.square().mean(dim=1), zero)
        path_tube_worst = torch.where(active_path, path_deficit.amax(dim=1).square(), zero)
    lateral_deficit = torch.relu(torch.abs(lateral) - float(weights.lane_margin_m))
    lateral_loss = torch.where(
        active_path,
        lateral_deficit.square().mean(dim=1) + lateral_deficit.amax(dim=1).square(),
        zero,
    )
    along_step = along[:, 1:] - along[:, :-1]
    reverse_loss = torch.where(active_path, torch.relu(-along_step).square().mean(dim=1), zero)
    progress_loss = torch.where(active_path, torch.relu(required_progress - along[:, -1]).square(), zero)
    root_sem = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if root_sem.ndim == 2:
        root_sem = root_sem.unsqueeze(0)
    root_sem_at_path = torch.zeros((batch, int(root_pos.shape[1])), dtype=torch.bool, device=device)
    try:
        from extension.batch_mpc_planner.terrain import semantic_at

        root_sem_at_path = semantic_at(terrain, root_pos[..., :2]) == SEMANTIC_SMALL_ID
    except Exception:
        root_sem_at_path = torch.zeros((batch, int(root_pos.shape[1])), dtype=torch.bool, device=device)
    root_height = height_at(terrain, root_pos[..., :2]).to(dtype=dtype, device=device)
    root_bottom = root_pos[..., 2] + float(weights.root_bottom_offset_z_m)
    root_clearance_deficit = torch.relu(root_height + float(weights.root_clearance_margin_m) - root_bottom)
    root_clearance_mask = torch.logical_and(root_sem_at_path, active_path[:, None]).to(dtype=dtype, device=device)
    root_clearance_loss = (root_clearance_deficit.square() * root_clearance_mask).sum(dim=1) / root_clearance_mask.sum(
        dim=1
    ).clamp_min(1.0)
    root_clearance_worst = torch.where(
        root_clearance_mask > 0.0,
        root_clearance_deficit,
        torch.zeros_like(root_clearance_deficit),
    ).amax(dim=1).square()
    out = (
        float(weights.lateral_weight) * lateral_loss
        + float(weights.lateral_worst_weight) * lateral_deficit.amax(dim=1).square()
        + float(weights.reverse_weight) * reverse_loss
        + float(weights.progress_weight) * progress_loss
        + float(weights.root_clearance_weight) * root_clearance_loss
        + float(weights.root_clearance_worst_weight) * root_clearance_worst
        + float(weights.path_tube_weight) * path_tube_loss
        + float(weights.path_tube_worst_weight) * path_tube_worst
    )
    breakdown = {
        "test_low_small_straight_lateral": float(weights.lateral_weight) * lateral_loss
        + float(weights.lateral_worst_weight) * lateral_deficit.amax(dim=1).square(),
        "test_low_small_straight_reverse": float(weights.reverse_weight) * reverse_loss,
        "test_low_small_straight_progress": float(weights.progress_weight) * progress_loss,
        "test_low_small_straight_root_clearance": float(weights.root_clearance_weight) * root_clearance_loss
        + float(weights.root_clearance_worst_weight) * root_clearance_worst,
        "test_low_small_path_tube": float(weights.path_tube_weight) * path_tube_loss
        + float(weights.path_tube_worst_weight) * path_tube_worst,
    }
    return out, breakdown


def _body_stencil_xy(root_pos: torch.Tensor, root_rpy: torch.Tensor, radius_m: float) -> torch.Tensor:
    radius = float(radius_m)
    offsets = torch.tensor(
        [
            [0.0, 0.0],
            [radius, 0.0],
            [-radius, 0.0],
            [0.0, radius],
            [0.0, -radius],
            [radius, radius],
            [radius, -radius],
            [-radius, radius],
            [-radius, -radius],
        ],
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    yaw = root_rpy[..., 2]
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    ox = offsets[:, 0].view(1, 1, -1)
    oy = offsets[:, 1].view(1, 1, -1)
    return torch.stack((cy * ox - sy * oy, sy * ox + cy * oy), dim=-1) + root_pos[..., None, :2]


def _semantic_structural_extra_loss(
    decoded,
    terrain,
    *,
    weights: _StructuralSemanticWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from extension.batch_mpc_planner.losses.terrain_clearance import (
        _sample_obstacle_field,
        _semantic_obstacle_field,
        finite_horizon_touchdown_phase,
        sample_time,
    )

    weights = weights or _StructuralSemanticWeights()
    root_pos = torch.as_tensor(decoded.root_pos)
    root_rpy = torch.as_tensor(decoded.root_rpy, dtype=root_pos.dtype, device=root_pos.device)
    foot_pos = torch.as_tensor(decoded.foot_pos, dtype=root_pos.dtype, device=root_pos.device)
    contact_prob = torch.as_tensor(decoded.contact_prob, dtype=root_pos.dtype, device=root_pos.device)
    masks = _semantic_height_class_masks(
        terrain,
        root_pos,
        high_small_relative_height_m=weights.high_small_relative_height_m,
    )
    zero = torch.zeros((int(root_pos.shape[0]),), dtype=root_pos.dtype, device=root_pos.device)
    if masks is None:
        return zero, {
            "test_structural_low_small_foot": zero,
            "test_structural_low_small_touchdown": zero,
            "test_structural_high_body": zero,
        }
    low_small, high_small, large = masks
    empty = torch.zeros_like(large)

    low_field = _semantic_obstacle_field(
        terrain,
        dtype=root_pos.dtype,
        device=root_pos.device,
        small_ids=(SEMANTIC_SMALL_ID,),
        large_ids=(SEMANTIC_LARGE_ID,),
        small_weight=1.0,
        large_weight=0.0,
        soft_margin_m=weights.low_small_soft_margin_m,
        small_mask_override=low_small,
        large_mask_override=empty,
    )
    if low_field is None:
        low_foot = zero
        low_touchdown = zero
    else:
        foot_soft = _sample_obstacle_field(terrain, low_field, foot_pos[..., :2]).to(
            dtype=root_pos.dtype,
            device=root_pos.device,
        )
        contact_sq = contact_prob.square()
        low_foot_mean = (foot_soft * contact_sq).mean(dim=(1, 2))
        low_foot_worst = (foot_soft * contact_sq).amax(dim=(1, 2))
        low_foot = (
            float(weights.low_small_foot_weight) * low_foot_mean
            + float(weights.low_small_foot_worst_weight) * low_foot_worst
        )
        touchdown_phase = finite_horizon_touchdown_phase(decoded.swing_center, decoded.swing_width)
        touchdown_w = sample_time(foot_pos, touchdown_phase, cyclic=False)
        touchdown_soft = _sample_obstacle_field(terrain, low_field, touchdown_w[..., :2]).to(
            dtype=root_pos.dtype,
            device=root_pos.device,
        )
        low_touchdown = (
            float(weights.low_small_touchdown_weight) * touchdown_soft.mean(dim=1)
            + float(weights.low_small_touchdown_worst_weight) * touchdown_soft.amax(dim=1)
        )

    if (
        (not weights.include_high_small_body and not weights.include_large_body)
        or (float(weights.high_body_weight) <= 0.0 and float(weights.high_body_worst_weight) <= 0.0)
    ):
        high_body = zero
    else:
        high_field = _semantic_obstacle_field(
            terrain,
            dtype=root_pos.dtype,
            device=root_pos.device,
            small_ids=(SEMANTIC_SMALL_ID,),
            large_ids=(SEMANTIC_LARGE_ID,),
            small_weight=1.0,
            large_weight=1.0,
            soft_margin_m=weights.high_body_soft_margin_m,
            small_mask_override=high_small if weights.include_high_small_body else empty,
            large_mask_override=large if weights.include_large_body else empty,
        )
        if high_field is None:
            high_body = zero
        else:
            body_xy = _body_stencil_xy(root_pos, root_rpy, weights.body_stencil_radius_m)
            body_soft = _sample_obstacle_field(terrain, high_field, body_xy).to(
                dtype=root_pos.dtype,
                device=root_pos.device,
            )
            high_body = (
                float(weights.high_body_weight) * body_soft.mean(dim=(1, 2))
                + float(weights.high_body_worst_weight) * body_soft.amax(dim=(1, 2))
            )

    breakdown = {
        "test_structural_low_small_foot": low_foot,
        "test_structural_low_small_touchdown": low_touchdown,
        "test_structural_high_body": high_body,
    }
    return low_foot + low_touchdown + high_body, breakdown


def _loss_only_low_small_crossing_extra_loss(
    decoded,
    command: torch.Tensor,
    terrain,
    *,
    weights: _LossOnlySemanticWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or _LossOnlySemanticWeights(low_small_cross=True)
    straight_weights = _LowSmallStraightWeights(
        lateral_weight=30.0 * float(weights.low_small_path_weight),
        lateral_worst_weight=45.0 * float(weights.low_small_path_weight),
        reverse_weight=12.0 * float(weights.low_small_path_weight),
        progress_weight=50.0 * float(weights.low_small_path_weight),
        root_clearance_weight=0.0,
        root_clearance_worst_weight=0.0,
        lane_margin_m=0.14,
        obstacle_depth_m=0.30,
        pass_margin_m=0.10,
        use_body_yaw_path=False,
        path_tube_weight=0.0,
        path_tube_worst_weight=0.0,
    )
    path_loss, path_breakdown = _semantic_low_small_straight_extra_loss(
        decoded.root_pos,
        decoded.root_rpy,
        command,
        terrain,
        weights=straight_weights,
    )
    structural_weights = _StructuralSemanticWeights(
        low_small_foot_weight=45.0 * float(weights.low_small_foot_weight),
        low_small_foot_worst_weight=16.0 * float(weights.low_small_foot_weight),
        low_small_touchdown_weight=24.0 * float(weights.low_small_foot_weight),
        low_small_touchdown_worst_weight=10.0 * float(weights.low_small_foot_weight),
        high_body_weight=0.0,
        high_body_worst_weight=0.0,
        low_small_soft_margin_m=float(weights.semantic_soft_margin_m),
        high_small_relative_height_m=float(weights.high_small_relative_height_m),
        include_high_small_body=False,
        include_large_body=False,
    )
    foot_loss, foot_breakdown = _semantic_structural_extra_loss(decoded, terrain, weights=structural_weights)

    from extension.batch_mpc_planner.terrain import height_at

    foot_pos = torch.as_tensor(decoded.foot_pos)
    swing_prob = torch.as_tensor(decoded.swing_prob, dtype=foot_pos.dtype, device=foot_pos.device)
    masks = _semantic_height_class_masks(
        terrain,
        torch.as_tensor(decoded.root_pos, dtype=foot_pos.dtype, device=foot_pos.device),
        high_small_relative_height_m=float(weights.high_small_relative_height_m),
    )
    if masks is None:
        clearance = torch.zeros_like(path_loss)
    else:
        low_small, _, _ = masks
        from extension.batch_mpc_planner.losses.terrain_clearance import _sample_obstacle_field, _semantic_obstacle_field

        empty = torch.zeros_like(low_small)
        field = _semantic_obstacle_field(
            terrain,
            dtype=foot_pos.dtype,
            device=foot_pos.device,
            small_ids=(SEMANTIC_SMALL_ID,),
            large_ids=(SEMANTIC_LARGE_ID,),
            small_weight=1.0,
            large_weight=0.0,
            soft_margin_m=float(weights.semantic_soft_margin_m),
            small_mask_override=low_small,
            large_mask_override=empty,
        )
        if field is None:
            clearance = torch.zeros_like(path_loss)
        else:
            soft = _sample_obstacle_field(terrain, field, foot_pos[..., :2]).to(dtype=foot_pos.dtype, device=foot_pos.device)
            terrain_z = height_at(terrain, foot_pos[..., :2]).to(dtype=foot_pos.dtype, device=foot_pos.device)
            deficit = torch.relu(terrain_z + float(weights.low_small_swing_clearance_m) - foot_pos[..., 2])
            active = swing_prob * soft
            clearance_mean = (active * deficit.square()).sum(dim=(1, 2)) / active.sum(dim=(1, 2)).clamp_min(1.0)
            clearance_worst = torch.where(active > 0.0, deficit, torch.zeros_like(deficit)).amax(dim=(1, 2)).square()
            clearance = (
                float(weights.low_small_swing_clearance_weight) * clearance_mean
                + float(weights.low_small_swing_clearance_worst_weight) * clearance_worst
            )
    breakdown = {
        "loss_only_low_small_path": path_breakdown["test_low_small_straight_lateral"]
        + path_breakdown["test_low_small_straight_reverse"]
        + path_breakdown["test_low_small_straight_progress"]
        + path_breakdown["test_low_small_path_tube"],
        "loss_only_low_small_foot_semantic": foot_breakdown["test_structural_low_small_foot"]
        + foot_breakdown["test_structural_low_small_touchdown"],
        "loss_only_low_small_swing_clearance": clearance,
    }
    return path_loss + foot_loss + clearance, breakdown


def _loss_only_low_small_foot_over_extra_loss(
    decoded,
    command: torch.Tensor,
    terrain,
    *,
    weights: _LossOnlySemanticWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from extension.batch_mpc_planner.terrain import height_at
    from extension.batch_mpc_planner.losses.terrain_clearance import _nearby_height_for_sparse_semantic

    weights = weights or _LossOnlySemanticWeights(low_small_foot_over=True)
    foot_pos = torch.as_tensor(decoded.foot_pos)
    root_pos = torch.as_tensor(decoded.root_pos, dtype=foot_pos.dtype, device=foot_pos.device)
    batch = int(foot_pos.shape[0])
    dtype = foot_pos.dtype
    device = foot_pos.device
    zero = torch.zeros((batch,), dtype=dtype, device=device)
    if terrain.semantic_map is None or int(foot_pos.shape[1]) < 1:
        return zero, {
            "loss_only_low_small_foot_over_xy": zero,
            "loss_only_low_small_foot_over_z": zero,
        }

    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    if int(height.shape[0]) == 1 and batch > 1:
        height = height.expand(batch, -1, -1)
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    if int(height.shape[0]) != batch or int(semantic.shape[0]) != batch:
        return zero, {
            "loss_only_low_small_foot_over_xy": zero,
            "loss_only_low_small_foot_over_z": zero,
        }

    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    cmd_xy = cmd[:, :2]
    speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
    heading = cmd_xy / speed.clamp_min(1.0e-6).unsqueeze(-1)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)

    grid_xy = _terrain_grid_world_xy_for_probe(terrain, dtype=dtype, device=device)
    if int(grid_xy.shape[0]) == 1 and batch > 1:
        grid_xy = grid_xy.expand(batch, -1, -1)
    grid_sem = semantic.reshape(batch, -1)
    grid_z = height.reshape(batch, -1)
    nearby_grid_z = _nearby_height_for_sparse_semantic(terrain, height, dtype=dtype, device=device)
    root0 = root_pos[:, 0]
    root_ground_z = height_at(terrain, root0[:, None, :2]).reshape(batch).to(dtype=dtype, device=device)
    low_small = torch.logical_and(
        grid_sem == SEMANTIC_SMALL_ID,
        (nearby_grid_z - root_ground_z[:, None]) <= float(weights.high_small_relative_height_m),
    )
    delta = grid_xy - root0[:, None, :2]
    obs_forward = (delta * heading[:, None, :]).sum(dim=-1)
    obs_lateral = (delta * left[:, None, :]).sum(dim=-1)
    candidate = torch.logical_and(
        low_small,
        torch.logical_and(
            torch.logical_and(obs_forward >= 0.0, obs_forward <= float(weights.low_small_foot_over_forward_distance_m)),
            torch.abs(obs_lateral) <= float(weights.low_small_foot_over_corridor_width_m),
        ),
    )
    active = torch.logical_and(candidate.any(dim=-1), speed > 1.0e-4)
    candidate_f = candidate.to(dtype=dtype, device=device)
    count = candidate_f.sum(dim=-1).clamp_min(1.0)
    target_xy = (grid_xy * candidate_f[..., None]).sum(dim=1) / count[:, None]
    target_forward = (obs_forward * candidate_f).sum(dim=-1) / count
    target_z = (nearby_grid_z * candidate_f).sum(dim=-1) / count
    target_z = torch.maximum(target_z, (grid_z * candidate_f).sum(dim=-1) / count)

    swing_prob = torch.as_tensor(decoded.swing_prob, dtype=dtype, device=device)
    contact_prob = torch.as_tensor(decoded.contact_prob, dtype=dtype, device=device)
    swing_weight = swing_prob * (1.0 - contact_prob).clamp_min(0.0)
    rel = foot_pos[..., :2] - target_xy[:, None, None, :]
    dist = torch.linalg.vector_norm(rel, dim=-1)
    root_rel = root_pos[..., :2] - root0[:, None, :2]
    root_along = (root_rel * heading[:, None, :]).sum(dim=-1)
    root_gate = torch.relu(
        float(weights.low_small_foot_over_along_window_m) - torch.abs(root_along - target_forward[:, None])
    ) / max(float(weights.low_small_foot_over_along_window_m), 1.0e-6)
    clearance_target = target_z[:, None, None] + float(weights.low_small_foot_over_clearance_m)
    z_deficit = torch.relu(clearance_target - foot_pos[..., 2])
    xy_deficit = torch.relu(dist - float(weights.low_small_foot_over_radius_m))
    candidate_cost = (
        float(weights.low_small_foot_over_xy_weight) * xy_deficit.square()
        + float(weights.low_small_foot_over_z_weight) * z_deficit.square()
    )
    ineligible = float(weights.low_small_foot_over_ineligible_penalty) * (1.0 - swing_weight).clamp_min(0.0)
    candidate_cost = candidate_cost + ineligible
    best = torch.amin(candidate_cost.reshape(batch, -1), dim=-1)
    best_xy = torch.amin(xy_deficit.reshape(batch, -1).square(), dim=-1)
    best_z = torch.amin(z_deficit.reshape(batch, -1).square(), dim=-1)
    gated = root_gate[..., None] * swing_weight
    gated_denom = gated.sum(dim=(1, 2)).clamp_min(1.0)
    direct_xy = (gated * xy_deficit.square()).sum(dim=(1, 2)) / gated_denom
    direct_z = (gated * z_deficit.square()).sum(dim=(1, 2)) / gated_denom
    per_leg_gate = gated.sum(dim=1).clamp_min(1.0)
    per_leg_xy = (gated * xy_deficit.square()).sum(dim=1) / per_leg_gate
    per_leg_z = (gated * z_deficit.square()).sum(dim=1) / per_leg_gate
    per_leg_missing = torch.relu(0.25 - gated.sum(dim=1)).square()
    leg_cost = per_leg_xy + per_leg_z + per_leg_missing
    best_leg = torch.amin(leg_cost, dim=1)
    missing_gate = torch.relu(0.25 - gated.sum(dim=(1, 2))).square()
    if float(weights.low_small_foot_over_window_weight) > 0.0:
        sigma = max(float(weights.low_small_foot_over_window_sigma_m), 1.0e-6)
        z_temp = max(float(weights.low_small_foot_over_window_z_temp_m), 1.0e-6)
        xy_score = torch.exp(-0.5 * (dist / sigma).square())
        z_score = torch.sigmoid((foot_pos[..., 2] - clearance_target) / z_temp)
        window_score = root_gate[..., None] * swing_weight * xy_score * z_score
        per_leg_score = window_score.sum(dim=1)
        window_loss = torch.relu(float(weights.low_small_foot_over_window_min_count) - per_leg_score).amin(dim=1).square()
        window_step_loss = zero
        if int(foot_pos.shape[1]) >= 2 and float(weights.low_small_foot_over_window_step_weight) > 0.0:
            step = torch.linalg.vector_norm(foot_pos[:, 1:] - foot_pos[:, :-1], dim=-1)
            step_gate = torch.maximum(window_score[:, 1:], window_score[:, :-1])
            step_deficit = torch.relu(step - float(weights.low_small_foot_over_window_step_cap_m)).square()
            per_leg_step = (step_gate * step_deficit).sum(dim=1) / step_gate.sum(dim=1).clamp_min(1.0)
            per_leg_step = per_leg_step + torch.relu(0.25 - step_gate.sum(dim=1)).square()
            if bool(weights.low_small_foot_over_window_coupled):
                best_leg_idx = torch.argmin(
                    torch.relu(float(weights.low_small_foot_over_window_min_count) - per_leg_score),
                    dim=1,
                )
                window_step_loss = per_leg_step.gather(1, best_leg_idx[:, None]).squeeze(1)
            else:
                window_step_loss = per_leg_step.amin(dim=1)
        window_accel_loss = zero
        if int(foot_pos.shape[1]) >= 3 and float(weights.low_small_foot_over_window_accel_weight) > 0.0:
            accel = torch.linalg.vector_norm(foot_pos[:, 2:] - 2.0 * foot_pos[:, 1:-1] + foot_pos[:, :-2], dim=-1)
            accel_gate = torch.maximum(
                torch.maximum(window_score[:, 2:], window_score[:, 1:-1]),
                window_score[:, :-2],
            )
            accel_deficit = torch.relu(accel - float(weights.low_small_foot_over_window_accel_cap_m)).square()
            per_leg_accel = (accel_gate * accel_deficit).sum(dim=1) / accel_gate.sum(dim=1).clamp_min(1.0)
            per_leg_accel = per_leg_accel + torch.relu(0.25 - accel_gate.sum(dim=1)).square()
            if bool(weights.low_small_foot_over_window_coupled):
                best_leg_idx = torch.argmin(
                    torch.relu(float(weights.low_small_foot_over_window_min_count) - per_leg_score),
                    dim=1,
                )
                window_accel_loss = per_leg_accel.gather(1, best_leg_idx[:, None]).squeeze(1)
            else:
                window_accel_loss = per_leg_accel.amin(dim=1)
    else:
        window_loss = zero
        window_step_loss = zero
        window_accel_loss = zero
    if (
        float(weights.low_small_foot_over_path_curve_weight) > 0.0
        or float(weights.low_small_foot_over_path_curve_z_weight) > 0.0
    ):
        curve_window = max(float(weights.low_small_foot_over_path_curve_window_m), 1.0e-6)
        curve_phase = ((root_along - target_forward[:, None]) / curve_window).clamp(-1.0, 1.0)
        if bool(weights.low_small_foot_over_path_curve_body_yaw):
            root_rpy = torch.as_tensor(decoded.root_rpy, dtype=dtype, device=device)
            local_heading = torch.stack((torch.cos(root_rpy[..., 2]), torch.sin(root_rpy[..., 2])), dim=-1)
            local_left = torch.stack((-local_heading[..., 1], local_heading[..., 0]), dim=-1)
        else:
            local_heading = heading[:, None, :].expand(batch, int(foot_pos.shape[1]), 2)
            local_left = left[:, None, :].expand(batch, int(foot_pos.shape[1]), 2)
        curve_target_xy = target_xy[:, None, None, :] + curve_phase[:, :, None, None] * local_heading[:, :, None, :] * curve_window
        curve_lateral = ((foot_pos[..., :2] - curve_target_xy) * local_left[:, :, None, :]).sum(dim=-1)
        curve_along = ((foot_pos[..., :2] - curve_target_xy) * local_heading[:, :, None, :]).sum(dim=-1)
        curve_gate = root_gate[..., None] * swing_weight
        curve_xy_err = curve_along.square() + curve_lateral.square()
        per_leg_curve = (curve_gate * curve_xy_err).sum(dim=1) / curve_gate.sum(dim=1).clamp_min(1.0)
        per_leg_curve = per_leg_curve + torch.relu(0.25 - curve_gate.sum(dim=1)).square()
        curve_xy_loss = per_leg_curve.amin(dim=1)
        arch = 4.0 * (0.5 * (curve_phase + 1.0)) * (1.0 - 0.5 * (curve_phase + 1.0))
        curve_z_target = clearance_target[:, :, 0] + 0.04 * arch
        curve_z_err = torch.relu(curve_z_target[:, :, None] - foot_pos[..., 2]).square()
        per_leg_curve_z = (curve_gate * curve_z_err).sum(dim=1) / curve_gate.sum(dim=1).clamp_min(1.0)
        per_leg_curve_z = per_leg_curve_z + torch.relu(0.25 - curve_gate.sum(dim=1)).square()
        curve_z_loss = per_leg_curve_z.amin(dim=1)
    else:
        curve_xy_loss = zero
        curve_z_loss = zero
    xy_loss = torch.where(
        active,
        float(weights.low_small_foot_over_xy_weight) * best_xy
        + float(weights.low_small_foot_over_direct_xy_weight) * direct_xy,
        zero,
    )
    z_loss = torch.where(active, float(weights.low_small_foot_over_z_weight) * (best_z + direct_z), zero)
    total = torch.where(
        active,
        best
        + xy_loss
        + z_loss
        + float(weights.low_small_foot_over_leg_weight) * best_leg
        + float(weights.low_small_foot_over_time_gate_penalty) * missing_gate,
        zero,
    )
    total = total + torch.where(
        active,
        float(weights.low_small_foot_over_window_weight) * window_loss
        + float(weights.low_small_foot_over_window_step_weight) * window_step_loss
        + float(weights.low_small_foot_over_window_accel_weight) * window_accel_loss,
        zero,
    )
    total = total + torch.where(
        active,
        float(weights.low_small_foot_over_path_curve_weight) * curve_xy_loss
        + float(weights.low_small_foot_over_path_curve_z_weight) * curve_z_loss,
        zero,
    )
    breakdown = {
        "loss_only_low_small_foot_over_xy": xy_loss,
        "loss_only_low_small_foot_over_z": z_loss,
        "loss_only_low_small_foot_over_window": torch.where(
            active,
            float(weights.low_small_foot_over_window_weight) * window_loss,
            zero,
        ),
        "loss_only_low_small_foot_over_path_curve": torch.where(
            active,
            float(weights.low_small_foot_over_path_curve_weight) * curve_xy_loss,
            zero,
        ),
    }
    return total, breakdown


def _loss_only_high_large_avoid_extra_loss(
    decoded,
    command: torch.Tensor,
    terrain,
    *,
    weights: _LossOnlySemanticWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or _LossOnlySemanticWeights(high_large_avoid=True)
    root_pos = torch.as_tensor(decoded.root_pos)
    root_rpy = torch.as_tensor(decoded.root_rpy, dtype=root_pos.dtype, device=root_pos.device)
    batch = int(root_pos.shape[0])
    dtype = root_pos.dtype
    device = root_pos.device
    zero = torch.zeros((batch,), dtype=dtype, device=device)
    masks = _semantic_height_class_masks(
        terrain,
        root_pos,
        high_small_relative_height_m=float(weights.high_small_relative_height_m),
    )
    if masks is None:
        return zero, {
            "loss_only_high_large_body": zero,
            "loss_only_high_large_corridor": zero,
            "loss_only_high_large_lateral_escape": zero,
        }
    _, high_small, large = masks
    obstacle = torch.logical_or(high_small, large)

    from extension.batch_mpc_planner.losses.terrain_clearance import _sample_obstacle_field, _semantic_obstacle_field

    empty = torch.zeros_like(large)
    field = _semantic_obstacle_field(
        terrain,
        dtype=dtype,
        device=device,
        small_ids=(SEMANTIC_SMALL_ID,),
        large_ids=(SEMANTIC_LARGE_ID,),
        small_weight=1.0,
        large_weight=1.0,
        soft_margin_m=float(weights.semantic_soft_margin_m),
        small_mask_override=high_small,
        large_mask_override=large,
    )
    if field is None:
        body_loss = zero
        root_semantic_loss = zero
    else:
        body_xy = _body_stencil_xy(root_pos, root_rpy, float(weights.body_stencil_radius_m))
        body_soft = _sample_obstacle_field(terrain, field, body_xy).to(dtype=dtype, device=device)
        body_loss = (
            float(weights.high_large_body_weight) * body_soft.mean(dim=(1, 2))
            + float(weights.high_large_body_worst_weight) * body_soft.amax(dim=(1, 2))
        )
        root_soft = _sample_obstacle_field(terrain, field, root_pos[..., :2]).to(dtype=dtype, device=device)
        root_semantic_loss = (
            float(weights.high_large_root_semantic_weight) * root_soft.mean(dim=1)
            + float(weights.high_large_root_semantic_worst_weight) * root_soft.amax(dim=1)
        )

    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        cmd = torch.cat((cmd, torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)), dim=-1)
    cmd_xy = cmd[:, :2]
    speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
    heading = cmd_xy / speed.clamp_min(1.0e-6).unsqueeze(-1)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    grid_xy = _terrain_grid_world_xy_for_probe(terrain, dtype=dtype, device=device)
    if int(grid_xy.shape[0]) == 1 and batch > 1:
        grid_xy = grid_xy.expand(batch, -1, -1)
    obstacle_flat = obstacle.reshape(batch, -1)
    root0 = root_pos[:, 0, :2]
    delta = grid_xy - root0[:, None, :]
    obs_forward = (delta * heading[:, None, :]).sum(dim=-1)
    obs_lateral = (delta * left[:, None, :]).sum(dim=-1)
    corridor_obstacle = torch.logical_and(
        obstacle_flat,
        torch.logical_and(
            torch.logical_and(obs_forward >= 0.0, obs_forward <= float(weights.high_large_forward_distance_m)),
            torch.abs(obs_lateral) <= float(weights.high_large_corridor_width_m),
        ),
    )
    active = torch.logical_and(corridor_obstacle.any(dim=-1), speed > 1.0e-4)
    obstacle_f = corridor_obstacle.to(dtype=dtype)
    count = obstacle_f.sum(dim=-1).clamp_min(1.0)
    obs_forward_center = (obs_forward * obstacle_f).sum(dim=-1) / count
    obs_lateral_center = (obs_lateral * obstacle_f).sum(dim=-1) / count
    rel_path = root_pos[..., :2] - root0[:, None, :]
    along = (rel_path * heading[:, None, :]).sum(dim=-1)
    lateral = (rel_path * left[:, None, :]).sum(dim=-1) - obs_lateral_center[:, None]
    longitudinal_gate = torch.relu(
        float(weights.high_large_longitudinal_influence_m) - torch.abs(along - obs_forward_center[:, None])
    ) / max(float(weights.high_large_longitudinal_influence_m), 1.0e-6)
    lateral_deficit = torch.relu(float(weights.high_large_lateral_clearance_m) - torch.abs(lateral))
    corridor_cost = longitudinal_gate * lateral_deficit.square()
    corridor_mean = torch.where(active, corridor_cost.mean(dim=1), zero)
    corridor_worst = torch.where(active, corridor_cost.amax(dim=1), zero)
    corridor_loss = (
        float(weights.high_large_corridor_weight) * corridor_mean
        + float(weights.high_large_corridor_worst_weight) * corridor_worst
    )
    final_lateral = torch.abs(lateral[:, -1])
    lateral_escape = torch.where(
        active,
        torch.relu(float(weights.high_large_lateral_clearance_m) - final_lateral).square(),
        zero,
    )
    obstacle_any = obstacle.reshape(batch, -1)
    obstacle_count = obstacle_any.to(dtype=dtype).sum(dim=-1)
    if float(weights.high_large_distance_weight) <= 0.0 and float(weights.high_large_distance_worst_weight) <= 0.0:
        distance_loss = zero
    else:
        grid_delta_path = root_pos[..., None, :2] - grid_xy[:, None, :, :]
        grid_dist = torch.linalg.vector_norm(grid_delta_path, dim=-1)
        masked_dist = torch.where(obstacle_any[:, None, :], grid_dist, torch.full_like(grid_dist, 1.0e6))
        min_dist_t = masked_dist.amin(dim=-1)
        distance_deficit = torch.relu(float(weights.high_large_distance_margin_m) - min_dist_t)
        active_distance = obstacle_count > 0.0
        distance_loss = torch.where(
            active_distance,
            float(weights.high_large_distance_weight) * distance_deficit.square().mean(dim=1)
            + float(weights.high_large_distance_worst_weight) * distance_deficit.amax(dim=1).square(),
            zero,
        )
    if float(weights.high_large_scurve_weight) <= 0.0 and float(weights.high_large_scurve_worst_weight) <= 0.0:
        scurve_loss = zero
    else:
        along_norm = torch.clamp(
            (along - obs_forward_center[:, None]) / max(float(weights.high_large_longitudinal_influence_m), 1.0e-6),
            min=-1.0,
            max=1.0,
        )
        lateral_profile = float(weights.high_large_scurve_lateral_m) * (1.0 + torch.cos(math.pi * along_norm)) * 0.5
        scurve_cost = longitudinal_gate * torch.relu(lateral_profile - torch.abs(lateral)).square()
        scurve_loss = torch.where(
            active,
            float(weights.high_large_scurve_weight) * scurve_cost.mean(dim=1)
            + float(weights.high_large_scurve_worst_weight) * scurve_cost.amax(dim=1),
            zero,
        )
    breakdown = {
        "loss_only_high_large_body": body_loss,
        "loss_only_high_large_root_semantic": root_semantic_loss,
        "loss_only_high_large_corridor": corridor_loss,
        "loss_only_high_large_lateral_escape": lateral_escape,
        "loss_only_high_large_distance_margin": distance_loss,
        "loss_only_high_large_scurve": scurve_loss,
    }
    return (
        body_loss
        + root_semantic_loss
        + corridor_loss
        + float(weights.high_large_lateral_escape_weight) * lateral_escape
        + distance_loss
        + scurve_loss
    ), breakdown


def _loss_only_continuity_anchor_extra_loss(
    decoded,
    state,
    *,
    weights: _LossOnlySemanticWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or _LossOnlySemanticWeights(continuity_anchor=True)
    foot = torch.as_tensor(decoded.foot_pos)
    root = torch.as_tensor(decoded.root_pos, dtype=foot.dtype, device=foot.device)
    batch = int(foot.shape[0])
    zero = torch.zeros((batch,), dtype=foot.dtype, device=foot.device)
    if int(foot.shape[1]) < 2:
        return zero, {
            "loss_only_foot_boundary": zero,
            "loss_only_foot_accel": zero,
            "loss_only_foot_jerk": zero,
            "loss_only_root_accel": zero,
            "loss_only_first_foot_anchor": zero,
        }
    step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
    decoded_contact = getattr(decoded, "contact_prob", None)
    if decoded_contact is None:
        contact = torch.ones_like(foot[..., 0])
    else:
        contact = torch.as_tensor(decoded_contact, dtype=foot.dtype, device=foot.device)
    boundary_weight = 0.25 + 0.75 * contact[:, 1:]
    foot_boundary = (step.square() * boundary_weight).mean(dim=(1, 2))
    foot_step_worst = step.amax(dim=(1, 2)).square()
    foot_step_cap = torch.relu(step - float(weights.foot_step_cap_m)).square().mean(dim=(1, 2))
    foot_step_cap_worst = torch.relu(step - float(weights.foot_step_cap_m)).amax(dim=(1, 2)).square()
    if int(foot.shape[1]) >= 3:
        foot_accel_raw = torch.linalg.vector_norm(foot[:, 2:] - 2.0 * foot[:, 1:-1] + foot[:, :-2], dim=-1)
        root_accel_raw = torch.linalg.vector_norm(root[:, 2:] - 2.0 * root[:, 1:-1] + root[:, :-2], dim=-1)
        foot_accel = foot_accel_raw.square().mean(dim=(1, 2))
        foot_accel_worst = foot_accel_raw.amax(dim=(1, 2)).square()
        foot_accel_cap = torch.relu(foot_accel_raw - float(weights.foot_accel_cap_m)).square().mean(dim=(1, 2))
        foot_accel_cap_worst = torch.relu(foot_accel_raw - float(weights.foot_accel_cap_m)).amax(dim=(1, 2)).square()
        root_accel = root_accel_raw.square().mean(dim=1)
        root_accel_worst = root_accel_raw.amax(dim=1).square()
    else:
        foot_accel = zero
        foot_accel_worst = zero
        foot_accel_cap = zero
        foot_accel_cap_worst = zero
        root_accel = zero
        root_accel_worst = zero
    root_step = torch.linalg.vector_norm(root[:, 1:] - root[:, :-1], dim=-1)
    root_step_worst = root_step.amax(dim=1).square()
    if int(foot.shape[1]) >= 4:
        foot_jerk = torch.linalg.vector_norm(
            foot[:, 3:] - 3.0 * foot[:, 2:-1] + 3.0 * foot[:, 1:-2] - foot[:, :-3],
            dim=-1,
        ).square().mean(dim=(1, 2))
    else:
        foot_jerk = zero
    state_foot = torch.as_tensor(state.foot_pos, dtype=foot.dtype, device=foot.device)
    frames = min(max(int(weights.first_foot_anchor_frames), 1), int(foot.shape[1]))
    frame_weight = torch.linspace(1.0, 0.25, frames, dtype=foot.dtype, device=foot.device).view(1, frames, 1, 1)
    first_anchor = ((foot[:, :frames] - state_foot[:, None]) ** 2 * frame_weight).mean(dim=(1, 2, 3))
    breakdown = {
        "loss_only_foot_boundary": float(weights.foot_boundary_weight) * foot_boundary,
        "loss_only_foot_step_worst": float(weights.foot_step_worst_weight) * foot_step_worst,
        "loss_only_foot_step_cap": float(weights.foot_step_cap_weight) * (foot_step_cap + foot_step_cap_worst),
        "loss_only_foot_accel": float(weights.foot_accel_weight) * foot_accel
        + float(weights.foot_accel_worst_weight) * foot_accel_worst,
        "loss_only_foot_accel_cap": float(weights.foot_accel_cap_weight) * (foot_accel_cap + foot_accel_cap_worst),
        "loss_only_foot_jerk": float(weights.foot_jerk_weight) * foot_jerk,
        "loss_only_root_step_worst": float(weights.root_step_worst_weight) * root_step_worst,
        "loss_only_root_accel": float(weights.root_accel_weight) * root_accel
        + float(weights.root_accel_worst_weight) * root_accel_worst,
        "loss_only_first_foot_anchor": float(weights.first_foot_anchor_weight) * first_anchor,
    }
    out = sum(breakdown.values())
    return out, breakdown


def _loss_only_high_large_handoff_extra_loss(
    decoded,
    state,
    terrain,
    *,
    weights: _LossOnlySemanticWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or _LossOnlySemanticWeights(high_large_handoff=True)
    foot = torch.as_tensor(decoded.foot_pos)
    root = torch.as_tensor(decoded.root_pos, dtype=foot.dtype, device=foot.device)
    batch = int(foot.shape[0])
    zero = torch.zeros((batch,), dtype=foot.dtype, device=foot.device)
    if int(foot.shape[1]) < 2:
        return zero, {
            "loss_only_high_large_handoff_anchor": zero,
            "loss_only_high_large_handoff_step": zero,
            "loss_only_high_large_handoff_accel": zero,
            "loss_only_high_large_handoff_root_step": zero,
        }
    masks = _semantic_height_class_masks(
        terrain,
        root,
        high_small_relative_height_m=float(weights.high_small_relative_height_m),
    )
    if masks is None:
        gate = zero
    else:
        _, high_small, large = masks
        gate = torch.logical_or(high_small, large).reshape(batch, -1).any(dim=-1).to(dtype=foot.dtype, device=foot.device)
    frames = min(max(int(weights.handoff_frames), 2), int(foot.shape[1]))
    state_foot = torch.as_tensor(state.foot_pos, dtype=foot.dtype, device=foot.device)
    anchor_weight = torch.linspace(1.0, 0.20, frames, dtype=foot.dtype, device=foot.device).view(1, frames, 1, 1)
    anchor = ((foot[:, :frames] - state_foot[:, None]) ** 2 * anchor_weight).mean(dim=(1, 2, 3))
    foot_step_raw = torch.linalg.vector_norm(foot[:, 1:frames] - foot[:, : frames - 1], dim=-1)
    foot_step = foot_step_raw.square().mean(dim=(1, 2))
    foot_step_worst = foot_step_raw.amax(dim=(1, 2)).square()
    if frames >= 3:
        foot_accel_raw = torch.linalg.vector_norm(foot[:, 2:frames] - 2.0 * foot[:, 1 : frames - 1] + foot[:, : frames - 2], dim=-1)
        foot_accel = foot_accel_raw.square().mean(dim=(1, 2))
        foot_accel_worst = foot_accel_raw.amax(dim=(1, 2)).square()
    else:
        foot_accel = zero
        foot_accel_worst = zero
    root_step_raw = torch.linalg.vector_norm(root[:, 1:frames] - root[:, : frames - 1], dim=-1)
    root_step = root_step_raw.square().mean(dim=1)
    root_step_worst = root_step_raw.amax(dim=1).square()
    breakdown = {
        "loss_only_high_large_handoff_anchor": float(weights.handoff_foot_anchor_weight) * anchor,
        "loss_only_high_large_handoff_step": float(weights.handoff_foot_step_weight) * foot_step
        + float(weights.handoff_foot_step_worst_weight) * foot_step_worst,
        "loss_only_high_large_handoff_accel": float(weights.handoff_foot_accel_weight) * foot_accel
        + float(weights.handoff_foot_accel_worst_weight) * foot_accel_worst,
        "loss_only_high_large_handoff_root_step": float(weights.handoff_root_step_weight) * root_step
        + float(weights.handoff_root_step_worst_weight) * root_step_worst,
    }
    return gate * sum(breakdown.values()), {name: gate * value for name, value in breakdown.items()}


def _loss_only_high_large_fk_semantic_extra_loss(
    decoded,
    terrain,
    *,
    weights: _LossOnlySemanticWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles, solve_joint_angles_from_trajectory
    from extension.batch_mpc_planner.losses.terrain_clearance import semantic_contact_avoidance_loss, stance_semantic_obstacle_loss

    weights = weights or _LossOnlySemanticWeights(high_large_fk_semantic=True)
    root = torch.as_tensor(decoded.root_pos)
    rpy = torch.as_tensor(decoded.root_rpy, dtype=root.dtype, device=root.device)
    foot = torch.as_tensor(decoded.foot_pos, dtype=root.dtype, device=root.device)
    batch = int(root.shape[0])
    zero = torch.zeros((batch,), dtype=root.dtype, device=root.device)
    masks = _semantic_height_class_masks(
        terrain,
        root,
        high_small_relative_height_m=float(weights.high_small_relative_height_m),
    )
    if masks is None:
        return zero, {
            "loss_only_high_large_fk_stance_semantic": zero,
            "loss_only_high_large_fk_contact_semantic": zero,
        }
    _, high_small, large = masks
    gate = torch.logical_or(high_small, large).reshape(batch, -1).any(dim=-1).to(dtype=root.dtype, device=root.device)
    joint = solve_joint_angles_from_trajectory(root, rpy, foot, clamp_to_limits=True)
    fk_foot = fk_feet_from_joint_angles(root, rpy, joint)
    contact_prob = torch.as_tensor(decoded.contact_prob, dtype=root.dtype, device=root.device)
    stance = stance_semantic_obstacle_loss(
        terrain,
        fk_foot,
        contact_prob,
        ground_ids=(0,),
        small_ids=(SEMANTIC_SMALL_ID,),
        large_ids=(SEMANTIC_LARGE_ID,),
        small_weight=float(weights.fk_small_weight),
        large_weight=float(weights.fk_large_weight),
        min_contact_prob=0.40,
    )
    contact = semantic_contact_avoidance_loss(
        terrain,
        fk_foot,
        contact_prob,
        ground_ids=(0,),
        small_ids=(SEMANTIC_SMALL_ID,),
        large_ids=(SEMANTIC_LARGE_ID,),
        small_weight=float(weights.fk_small_weight),
        large_weight=float(weights.fk_large_weight),
        activation_margin=float(weights.fk_contact_activation_margin),
        worst_contact_weight=float(weights.fk_contact_worst_weight),
        soft_margin_m=0.18,
        soft_field_weight=0.0,
        soft_worst_field_weight=0.0,
    )
    breakdown = {
        "loss_only_high_large_fk_stance_semantic": float(weights.fk_stance_weight) * stance,
        "loss_only_high_large_fk_contact_semantic": float(weights.fk_contact_weight) * contact,
    }
    return gate * sum(breakdown.values()), {name: gate * value for name, value in breakdown.items()}


def _loss_only_semantic_extra_loss(
    decoded,
    state,
    command: torch.Tensor,
    terrain,
    weights: _LossOnlySemanticWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    root_pos = torch.as_tensor(decoded.root_pos)
    total = torch.zeros((int(root_pos.shape[0]),), dtype=root_pos.dtype, device=root_pos.device)
    breakdown: dict[str, torch.Tensor] = {}
    if bool(weights.low_small_cross):
        extra, parts = _loss_only_low_small_crossing_extra_loss(decoded, command, terrain, weights=weights)
        total = total + extra
        breakdown.update(parts)
    if bool(weights.low_small_foot_over):
        extra, parts = _loss_only_low_small_foot_over_extra_loss(decoded, command, terrain, weights=weights)
        total = total + extra
        breakdown.update(parts)
    if bool(weights.high_large_avoid):
        extra, parts = _loss_only_high_large_avoid_extra_loss(decoded, command, terrain, weights=weights)
        total = total + extra
        breakdown.update(parts)
    if bool(weights.continuity_anchor):
        extra, parts = _loss_only_continuity_anchor_extra_loss(decoded, state, weights=weights)
        total = total + extra
        breakdown.update(parts)
    if bool(weights.high_large_handoff):
        extra, parts = _loss_only_high_large_handoff_extra_loss(decoded, state, terrain, weights=weights)
        total = total + extra
        breakdown.update(parts)
    if bool(weights.high_large_fk_semantic):
        extra, parts = _loss_only_high_large_fk_semantic_extra_loss(decoded, terrain, weights=weights)
        total = total + extra
        breakdown.update(parts)
    return total, breakdown


@contextmanager
def _patched_structural_loss_for_variant(variant: str):
    weights = _structural_weights_for_variant(variant)
    straight_weights = _straight_weights_for_variant(variant)
    loss_only_weights = _loss_only_weights_for_variant(variant)
    if weights is None and straight_weights is None and loss_only_weights is None:
        yield
        return
    yield


def _variant_cfg(base_cfg, name: str):
    cfg = copy.deepcopy(base_cfg)
    variant = str(name)
    if variant == "parametric_v1":
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-2)
        return cfg
    if variant in {
        "nominal_cmd_shape_a_combined_v7",
        "nominal_cmd_shape_a_combined_v8",
        "nominal_cmd_shape_a_combined_v9",
        "nominal_cmd_shape_a_combined_v10",
    }:
        return cfg
    if variant in {"baseline", "fk_output_probe"}:
        return cfg
    if variant == "phase_fixed_probe":
        cfg.runtime.randomize_replan_phase = False
        return cfg
    if variant == "semantic_strong":
        cfg.losses.stance_semantic.weight *= 3.0
        cfg.losses.stance_semantic.small_weight *= 2.0
        cfg.losses.stance_semantic.large_weight *= 2.0
        cfg.losses.touchdown_semantic.weight *= 3.0
        cfg.losses.touchdown_semantic.small_weight *= 2.0
        cfg.losses.touchdown_semantic.large_weight *= 2.0
        cfg.losses.semantic_contact_avoid.weight *= 2.0
        cfg.losses.semantic_contact_avoid.worst_contact_weight *= 1.5
        cfg.losses.semantic_contact_avoid.soft_field_weight *= 2.0
        cfg.losses.semantic_contact_avoid.soft_worst_field_weight *= 2.0
        cfg.losses.semantic_obstacle.weight *= 2.0
        cfg.losses.semantic_obstacle.large_weight *= 2.0
        cfg.losses.semantic_obstacle.body_soft_field_weight *= 2.0
        cfg.losses.semantic_obstacle.body_soft_worst_field_weight *= 2.0
        cfg.losses.semantic_obstacle.foot_soft_field_weight *= 2.0
        cfg.losses.semantic_obstacle.foot_soft_worst_field_weight *= 2.0
        return cfg
    if variant == "contact_only_semantic":
        cfg = _variant_cfg(cfg, "stance_only_semantic")
        cfg.losses.semantic_contact_avoid.weight *= 2.0
        cfg.losses.semantic_contact_avoid.worst_contact_weight *= 1.5
        cfg.losses.semantic_contact_avoid.soft_field_weight *= 1.5
        cfg.losses.semantic_contact_avoid.soft_worst_field_weight *= 1.5
        return cfg
    if variant == "hard_contact_crossing_light":
        cfg.losses.low_small_crossing.weight *= 1.5
        cfg.losses.low_small_crossing.pass_margin_m = max(float(cfg.losses.low_small_crossing.pass_margin_m), 0.09)
        cfg.losses.low_small_crossing.obstacle_depth_m = max(float(cfg.losses.low_small_crossing.obstacle_depth_m), 0.28)
        cfg.losses.swing_clearance_terrain.weight *= 1.25
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(float(cfg.losses.swing_clearance_terrain.min_clearance_m), 0.14)
        cfg.losses.semantic_contact_avoid.weight *= 1.5
        cfg.losses.semantic_contact_avoid.small_weight *= 2.0
        cfg.losses.semantic_contact_avoid.large_weight *= 2.0
        cfg.losses.semantic_contact_avoid.activation_margin = max(
            float(cfg.losses.semantic_contact_avoid.activation_margin),
            0.12,
        )
        cfg.losses.semantic_contact_avoid.worst_contact_weight *= 2.0
        cfg.losses.semantic_contact_avoid.soft_field_weight = 0.0
        cfg.losses.semantic_contact_avoid.soft_worst_field_weight = 0.0
        cfg.losses.touchdown_semantic.weight *= 2.0
        return cfg
    if variant == "crossing_progress_only":
        cfg.losses.low_small_crossing.weight *= 1.5
        cfg.losses.low_small_crossing.pass_margin_m = max(float(cfg.losses.low_small_crossing.pass_margin_m), 0.09)
        cfg.losses.low_small_crossing.obstacle_depth_m = max(float(cfg.losses.low_small_crossing.obstacle_depth_m), 0.28)
        return cfg
    if variant == "foot_soft_cross_progress":
        cfg = _variant_cfg(cfg, "crossing_progress_only")
        cfg.losses.semantic_contact_avoid.weight *= 2.0
        cfg.losses.semantic_contact_avoid.activation_margin = max(
            float(cfg.losses.semantic_contact_avoid.activation_margin),
            0.10,
        )
        cfg.losses.semantic_contact_avoid.soft_margin_m = max(
            float(cfg.losses.semantic_contact_avoid.soft_margin_m),
            0.24,
        )
        cfg.losses.semantic_contact_avoid.soft_field_weight *= 3.0
        cfg.losses.semantic_contact_avoid.soft_worst_field_weight *= 3.0
        cfg.losses.touchdown_semantic.weight *= 2.0
        return cfg
    if variant == "support_touchdown_cross_progress":
        cfg = _variant_cfg(cfg, "crossing_progress_only")
        cfg.losses.touchdown_semantic.weight *= 4.0
        cfg.losses.touchdown_semantic.small_weight *= 3.0
        cfg.losses.touchdown_semantic.large_weight *= 3.0
        cfg.losses.touchdown_surface.support_search_radius_m = max(
            float(cfg.losses.touchdown_surface.support_search_radius_m),
            0.20,
        )
        cfg.losses.touchdown_surface.support_height_tolerance_m = min(
            float(cfg.losses.touchdown_surface.support_height_tolerance_m),
            0.02,
        )
        cfg.losses.touchdown_surface.support_distance_weight *= 2.0
        cfg.losses.touchdown_surface.support_height_weight *= 2.0
        cfg.losses.touchdown_surface.invalid_support_weight *= 2.0
        return cfg
    if variant == "long_swing_crossing":
        cfg = _variant_cfg(cfg, "crossing_progress_only")
        cfg.runtime.nominal_swing_height_m = max(float(cfg.runtime.nominal_swing_height_m), 0.18)
        cfg.runtime.swing_window_min_width = max(float(cfg.runtime.swing_window_min_width), 0.40)
        cfg.runtime.swing_window_max_width = max(float(cfg.runtime.swing_window_max_width), 0.85)
        cfg.losses.swing_clearance_terrain.weight *= 1.5
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(float(cfg.losses.swing_clearance_terrain.min_clearance_m), 0.18)
        cfg.losses.swing_clearance_terrain.worst_deficit_weight *= 1.5
        cfg.losses.swing_window.width_prior_weight *= 0.5
        return cfg
    if variant == "stance_only_semantic":
        cfg.losses.stance_semantic.weight *= 3.0
        cfg.losses.stance_semantic.small_weight *= 2.0
        cfg.losses.stance_semantic.large_weight *= 2.0
        cfg.losses.touchdown_semantic.weight *= 3.0
        cfg.losses.touchdown_semantic.small_weight *= 2.0
        cfg.losses.touchdown_semantic.large_weight *= 2.0
        return cfg
    if variant == "risk_strong":
        cfg.losses.high_obstacle_avoidance.weight *= 2.0
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.55,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.50,
        )
        cfg.losses.obstacle_risk.linear_scale_when_blocked = min(
            float(cfg.losses.obstacle_risk.linear_scale_when_blocked),
            0.25,
        )
        cfg.losses.obstacle_risk.yaw_scale_when_blocked = min(
            float(cfg.losses.obstacle_risk.yaw_scale_when_blocked),
            0.25,
        )
        cfg.losses.obstacle_risk.linear_corridor_width_m = max(float(cfg.losses.obstacle_risk.linear_corridor_width_m), 0.50)
        cfg.losses.obstacle_risk.yaw_swept_radius_m = max(float(cfg.losses.obstacle_risk.yaw_swept_radius_m), 0.70)
        return cfg
    if variant == "crossing_strong":
        cfg.losses.low_small_crossing.weight *= 2.0
        cfg.losses.low_small_crossing.pass_margin_m = max(float(cfg.losses.low_small_crossing.pass_margin_m), 0.12)
        cfg.losses.low_small_crossing.obstacle_depth_m = max(float(cfg.losses.low_small_crossing.obstacle_depth_m), 0.32)
        cfg.losses.swing_clearance_terrain.weight *= 1.5
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(float(cfg.losses.swing_clearance_terrain.min_clearance_m), 0.16)
        cfg.losses.swing_clearance_terrain.worst_deficit_weight *= 1.5
        return cfg
    if variant == "crossing_contact_light":
        cfg.losses.low_small_crossing.weight *= 1.5
        cfg.losses.low_small_crossing.pass_margin_m = max(float(cfg.losses.low_small_crossing.pass_margin_m), 0.09)
        cfg.losses.low_small_crossing.obstacle_depth_m = max(float(cfg.losses.low_small_crossing.obstacle_depth_m), 0.28)
        cfg.losses.swing_clearance_terrain.weight *= 1.25
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(float(cfg.losses.swing_clearance_terrain.min_clearance_m), 0.14)
        cfg.losses.swing_clearance_terrain.worst_deficit_weight *= 1.25
        cfg.losses.stance_semantic.weight *= 1.5
        cfg.losses.touchdown_semantic.weight *= 2.0
        cfg.losses.touchdown_semantic.small_weight *= 1.5
        cfg.losses.touchdown_semantic.large_weight *= 1.5
        return cfg
    if variant == "smooth_strong":
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.0
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        return cfg
    if variant == "high_body_margin":
        cfg.losses.high_obstacle_avoidance.weight *= 2.0
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.55,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.50,
        )
        cfg.losses.semantic_obstacle.weight *= 1.5
        cfg.losses.semantic_obstacle.large_weight *= 2.0
        cfg.losses.semantic_obstacle.body_weight *= 1.5
        cfg.losses.semantic_obstacle.foot_weight *= 0.75
        cfg.losses.semantic_obstacle.body_soft_field_weight *= 2.0
        cfg.losses.semantic_obstacle.body_soft_worst_field_weight *= 2.0
        return cfg
    if variant == "body_light":
        cfg.losses.high_obstacle_avoidance.weight *= 1.5
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.50,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.45,
        )
        cfg.losses.semantic_obstacle.weight *= 1.25
        cfg.losses.semantic_obstacle.large_weight *= 1.5
        cfg.losses.semantic_obstacle.body_weight *= 1.25
        cfg.losses.semantic_obstacle.body_soft_field_weight *= 1.5
        cfg.losses.semantic_obstacle.body_soft_worst_field_weight *= 1.5
        return cfg
    if variant == "combined":
        cfg = _variant_cfg(cfg, "semantic_strong")
        cfg = _variant_cfg(cfg, "risk_strong")
        cfg = _variant_cfg(cfg, "crossing_strong")
        cfg = _variant_cfg(cfg, "smooth_strong")
        return cfg
    if variant == "risk_crossing":
        cfg = _variant_cfg(cfg, "risk_strong")
        cfg = _variant_cfg(cfg, "crossing_strong")
        return cfg
    if variant == "risk_contact_crossing":
        cfg = _variant_cfg(cfg, "risk_strong")
        cfg = _variant_cfg(cfg, "crossing_strong")
        cfg = _variant_cfg(cfg, "contact_only_semantic")
        return cfg
    if variant == "risk_stance_crossing":
        cfg = _variant_cfg(cfg, "risk_strong")
        cfg = _variant_cfg(cfg, "crossing_strong")
        cfg = _variant_cfg(cfg, "stance_only_semantic")
        return cfg
    if variant == "body_stance_crossing":
        cfg = _variant_cfg(cfg, "high_body_margin")
        cfg = _variant_cfg(cfg, "crossing_strong")
        cfg = _variant_cfg(cfg, "stance_only_semantic")
        return cfg
    if variant == "body_stance_crossing_smooth":
        cfg = _variant_cfg(cfg, "body_stance_crossing")
        cfg = _variant_cfg(cfg, "smooth_strong")
        return cfg
    if variant == "body_light_crossing_light":
        cfg = _variant_cfg(cfg, "body_light")
        cfg = _variant_cfg(cfg, "crossing_contact_light")
        return cfg
    if variant == "body_light_touchdown_crossing":
        cfg = _variant_cfg(cfg, "body_light")
        cfg.losses.low_small_crossing.weight *= 1.5
        cfg.losses.low_small_crossing.pass_margin_m = max(float(cfg.losses.low_small_crossing.pass_margin_m), 0.09)
        cfg.losses.low_small_crossing.obstacle_depth_m = max(float(cfg.losses.low_small_crossing.obstacle_depth_m), 0.28)
        cfg.losses.swing_clearance_terrain.weight *= 1.25
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(float(cfg.losses.swing_clearance_terrain.min_clearance_m), 0.14)
        cfg.losses.touchdown_semantic.weight *= 3.0
        cfg.losses.touchdown_semantic.small_weight *= 2.0
        cfg.losses.touchdown_semantic.large_weight *= 2.0
        return cfg
    if variant == "body_hard_contact_crossing_light":
        cfg = _variant_cfg(cfg, "body_light")
        cfg = _variant_cfg(cfg, "hard_contact_crossing_light")
        return cfg
    if variant == "body_hard_contact_only":
        cfg = _variant_cfg(cfg, "body_light")
        cfg.losses.semantic_contact_avoid.weight *= 1.5
        cfg.losses.semantic_contact_avoid.small_weight *= 2.0
        cfg.losses.semantic_contact_avoid.large_weight *= 2.0
        cfg.losses.semantic_contact_avoid.activation_margin = max(
            float(cfg.losses.semantic_contact_avoid.activation_margin),
            0.12,
        )
        cfg.losses.semantic_contact_avoid.worst_contact_weight *= 2.0
        cfg.losses.semantic_contact_avoid.soft_field_weight = 0.0
        cfg.losses.semantic_contact_avoid.soft_worst_field_weight = 0.0
        cfg.losses.touchdown_semantic.weight *= 2.0
        return cfg
    if variant == "body_hard_contact_only_smooth":
        cfg = _variant_cfg(cfg, "body_hard_contact_only")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 3.0
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.0e-2)
        return cfg
    if variant == "post_blend_body_hard_contact_only":
        return _variant_cfg(cfg, "body_hard_contact_only")
    if variant == "body_crossing_progress_only":
        cfg = _variant_cfg(cfg, "body_light")
        cfg = _variant_cfg(cfg, "crossing_progress_only")
        return cfg
    if variant == "body_hard_contact_crossing_progress":
        cfg = _variant_cfg(cfg, "body_hard_contact_only")
        cfg = _variant_cfg(cfg, "crossing_progress_only")
        return cfg
    if variant == "body_long_swing_crossing":
        cfg = _variant_cfg(cfg, "body_light")
        cfg = _variant_cfg(cfg, "long_swing_crossing")
        return cfg
    if variant == "body_long_swing_hard_contact":
        cfg = _variant_cfg(cfg, "body_hard_contact_only")
        cfg = _variant_cfg(cfg, "long_swing_crossing")
        return cfg
    if variant == "opt40_body_crossing_progress":
        cfg = _variant_cfg(cfg, "body_crossing_progress_only")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "opt40_body_hard_contact_progress":
        cfg = _variant_cfg(cfg, "body_hard_contact_crossing_progress")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "opt40_body_hard_contact_risk_progress":
        cfg = _variant_cfg(cfg, "body_hard_contact_crossing_progress")
        cfg = _variant_cfg(cfg, "risk_strong")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "opt40_body_hard_contact_highbody_progress":
        cfg = _variant_cfg(cfg, "body_hard_contact_crossing_progress")
        cfg.losses.high_obstacle_avoidance.weight *= 1.5
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.60,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.52,
        )
        cfg.losses.semantic_obstacle.body_soft_field_weight *= 1.5
        cfg.losses.semantic_obstacle.body_soft_worst_field_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "body_foot_soft_cross_progress":
        cfg = _variant_cfg(cfg, "body_light")
        cfg = _variant_cfg(cfg, "foot_soft_cross_progress")
        return cfg
    if variant == "opt40_body_foot_soft_cross_progress":
        cfg = _variant_cfg(cfg, "body_foot_soft_cross_progress")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "body_support_touchdown_cross_progress":
        cfg = _variant_cfg(cfg, "body_light")
        cfg = _variant_cfg(cfg, "support_touchdown_cross_progress")
        return cfg
    if variant == "opt40_body_support_touchdown_cross_progress":
        cfg = _variant_cfg(cfg, "body_support_touchdown_cross_progress")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "struct_lowfoot_highbody":
        cfg = _variant_cfg(cfg, "crossing_progress_only")
        cfg.losses.semantic_obstacle.foot_soft_field_weight *= 0.5
        cfg.losses.semantic_obstacle.foot_soft_worst_field_weight *= 0.5
        return cfg
    if variant == "struct_lowfoot_highbody_strong":
        cfg = _variant_cfg(cfg, "crossing_strong")
        cfg.losses.semantic_obstacle.foot_soft_field_weight *= 0.5
        cfg.losses.semantic_obstacle.foot_soft_worst_field_weight *= 0.5
        return cfg
    if variant == "opt40_struct_lowfoot_highbody":
        cfg = _variant_cfg(cfg, "struct_lowfoot_highbody")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "struct_lowfoot_only":
        cfg = _variant_cfg(cfg, "crossing_progress_only")
        cfg.losses.semantic_obstacle.foot_soft_field_weight *= 0.5
        cfg.losses.semantic_obstacle.foot_soft_worst_field_weight *= 0.5
        return cfg
    if variant == "struct_lowfoot_largebody":
        cfg = _variant_cfg(cfg, "struct_lowfoot_only")
        return cfg
    if variant == "struct_lowfoot_largebody_gentle":
        cfg = _variant_cfg(cfg, "struct_lowfoot_only")
        return cfg
    if variant == "struct_lowfoot_largebody_gentle_smooth":
        cfg = _variant_cfg(cfg, "struct_lowfoot_largebody_gentle")
        cfg = _variant_cfg(cfg, "smooth_strong")
        return cfg
    if variant == "struct_lowfoot_cross_hard":
        cfg = _variant_cfg(cfg, "struct_lowfoot_only")
        cfg.losses.low_small_crossing.weight = max(
            float(cfg.losses.low_small_crossing.weight),
            float(_variant_cfg(base_cfg, "crossing_strong").losses.low_small_crossing.weight) * 1.25,
        )
        cfg.losses.low_small_crossing.pass_margin_m = max(float(cfg.losses.low_small_crossing.pass_margin_m), 0.18)
        cfg.losses.low_small_crossing.obstacle_depth_m = max(float(cfg.losses.low_small_crossing.obstacle_depth_m), 0.45)
        cfg.losses.swing_clearance_terrain.weight *= 1.5
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(float(cfg.losses.swing_clearance_terrain.min_clearance_m), 0.18)
        cfg.losses.touchdown_semantic.weight *= 2.0
        cfg.losses.semantic_contact_avoid.weight *= 1.5
        cfg.losses.semantic_contact_avoid.activation_margin = max(
            float(cfg.losses.semantic_contact_avoid.activation_margin),
            0.12,
        )
        return cfg
    if variant == "opt40_struct_lowfoot_cross_hard":
        cfg = _variant_cfg(cfg, "struct_lowfoot_cross_hard")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant in SELECTOR_VARIANTS:
        return cfg
    if variant == "select_policy_class_wide_margin":
        return cfg
    if variant == "select_policy_class_hardcross_margin":
        return cfg
    if variant == "select_policy_class_jitter_margin":
        return cfg
    if variant == "select_policy_class_risk_jitter_margin":
        return cfg
    if variant == "select_policy_class_priority_jitter_margin":
        return cfg
    if variant == "select_policy_class_clearance_jitter_margin":
        return cfg
    if variant == "select_policy_class_task_jitter_margin":
        return cfg
    if variant == "select_policy_class_straight_task_jitter_margin":
        return cfg
    if variant == "select_policy_class_path_task_jitter_margin":
        return cfg
    if variant == "select_policy_class_metric_task_jitter_margin":
        return cfg
    if variant == "select_policy_class_large_smooth_metric_margin":
        return cfg
    if variant == "straight_low_small_task":
        cfg = _variant_cfg(cfg, "struct_lowfoot_cross_hard")
        cfg.losses.tracking.weight *= 2.0
        cfg.losses.tracking.vel_weight *= 2.0
        cfg.losses.progress.weight *= 2.0
        cfg.losses.progress.min_progress_m = max(float(cfg.losses.progress.min_progress_m), 0.10)
        cfg.losses.smoothness.root_weight *= 1.5
        cfg.losses.low_small_crossing.weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "straight_smooth_low_small_task":
        cfg = _variant_cfg(cfg, "straight_low_small_task")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 3.0
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 48)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.0e-2)
        return cfg
    if variant == "path_tube_low_small_task":
        cfg = _variant_cfg(cfg, "struct_lowfoot_cross_hard")
        cfg.losses.tracking.weight *= 1.5
        cfg.losses.tracking.vel_weight *= 1.5
        cfg.losses.progress.weight *= 2.0
        cfg.losses.progress.min_progress_m = max(float(cfg.losses.progress.min_progress_m), 0.10)
        cfg.losses.low_small_crossing.weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "path_tube_smooth_low_small_task":
        cfg = _variant_cfg(cfg, "path_tube_low_small_task")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.0
        cfg.losses.smoothness.foot_weight *= 1.25
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 48)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.0e-2)
        return cfg
    if variant == "loss_low_small_cross_v1":
        cfg.losses.low_small_crossing.weight *= 1.25
        cfg.losses.low_small_crossing.pass_margin_m = max(float(cfg.losses.low_small_crossing.pass_margin_m), 0.10)
        cfg.losses.low_small_crossing.obstacle_depth_m = max(float(cfg.losses.low_small_crossing.obstacle_depth_m), 0.30)
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(
            float(cfg.losses.swing_clearance_terrain.min_clearance_m),
            0.16,
        )
        return cfg
    if variant == "loss_high_large_avoid_v1":
        cfg.losses.high_obstacle_avoidance.weight *= 1.25
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.50,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.50,
        )
        return cfg
    if variant == "loss_continuity_anchor_v1":
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.0
        cfg.losses.smoothness.foot_weight *= 1.25
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.5e-2)
        return cfg
    if variant == "loss_semantic_all_v1":
        cfg = _variant_cfg(cfg, "loss_low_small_cross_v1")
        cfg = _variant_cfg(cfg, "loss_high_large_avoid_v1")
        cfg = _variant_cfg(cfg, "loss_continuity_anchor_v1")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 36)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.2e-2)
        return cfg
    if variant == "loss_low_small_cont_v2":
        cfg = _variant_cfg(cfg, "loss_low_small_cross_v1")
        cfg = _variant_cfg(cfg, "loss_continuity_anchor_v1")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 1.0e-2)
        return cfg
    if variant == "loss_low_small_stepcap_v3":
        cfg = _variant_cfg(cfg, "loss_low_small_cont_v2")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 3.0
        cfg.losses.smoothness.foot_weight *= 2.25
        cfg.losses.smoothness.root_weight *= 1.35
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if variant == "loss_low_small_stepcap_v4":
        cfg = _variant_cfg(cfg, "loss_low_small_stepcap_v3")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.35
        cfg.losses.smoothness.foot_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if variant == "loss_low_small_footover_v1":
        cfg = _variant_cfg(cfg, "loss_low_small_cont_v2")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 48)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 8.0e-3)
        return cfg
    if variant == "loss_low_small_footover_clear_v2":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_v1")
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(
            float(cfg.losses.swing_clearance_terrain.min_clearance_m),
            0.18,
        )
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if variant == "loss_low_small_footover_cont_v3":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_clear_v2")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.0
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if variant == "loss_low_small_footover_gate_v4":
        cfg = _variant_cfg(cfg, "loss_low_small_cont_v2")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if variant == "loss_low_small_footover_gate_cont_v5":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_gate_v4")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.0
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if variant == "loss_low_small_footover_gate_strong_v6":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_gate_cont_v5")
        cfg.losses.swing_clearance_terrain.min_clearance_m = max(
            float(cfg.losses.swing_clearance_terrain.min_clearance_m),
            0.19,
        )
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 72)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 3.0e-3)
        return cfg
    if variant == "loss_low_small_footover_leg_v7":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_gate_v4")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 3.5
        cfg.losses.smoothness.foot_weight *= 2.0
        cfg.losses.smoothness.root_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 72)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 3.0e-3)
        return cfg
    if variant == "loss_low_small_footover_leg_cont_v8":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_leg_v7")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.5
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 80)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    if variant == "loss_low_small_footover_wide_v9":
        cfg = _variant_cfg(cfg, "loss_low_small_cont_v2")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 3.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 4.0
        cfg.losses.smoothness.foot_weight *= 2.5
        cfg.losses.smoothness.root_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 80)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    if variant == "loss_low_small_footover_wide_cont_v10":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_wide_v9")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.5
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 96)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.0e-3)
        return cfg
    if variant == "loss_low_small_footover_cap_v11":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_gate_v4")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 3.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 4.0
        cfg.losses.smoothness.foot_weight *= 2.5
        cfg.losses.smoothness.root_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 80)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    if variant == "loss_low_small_footover_cap_strong_v12":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_cap_v11")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.75
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.75
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 96)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.0e-3)
        return cfg
    if variant == "loss_low_small_footover_window_v13":
        cfg = _variant_cfg(cfg, "loss_low_small_cont_v2")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 3.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 4.0
        cfg.losses.smoothness.foot_weight *= 2.25
        cfg.losses.smoothness.root_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 88)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    if variant == "loss_low_small_footover_window_cont_v14":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_window_v13")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.5
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 104)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.0e-3)
        return cfg
    if variant == "loss_low_small_footover_coupled_v15":
        return _variant_cfg(cfg, "loss_low_small_footover_window_v13")
    if variant == "loss_low_small_footover_coupled_cont_v16":
        return _variant_cfg(cfg, "loss_low_small_footover_window_cont_v14")
    if variant == "loss_low_small_footover_path_v17":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_gate_v4")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 3.5
        cfg.losses.smoothness.foot_weight *= 2.0
        cfg.losses.smoothness.root_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 80)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    if variant == "loss_low_small_footover_path_cont_v18":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_path_v17")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.5
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 104)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.0e-3)
        return cfg
    if variant == "loss_low_small_footover_pathweak_v19":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_gate_v4")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.5
        cfg.losses.smoothness.foot_weight *= 1.75
        cfg.losses.smoothness.root_weight *= 1.35
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 72)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 3.0e-3)
        return cfg
    if variant == "loss_low_small_footover_pathweak_cap_v20":
        cfg = _variant_cfg(cfg, "loss_low_small_footover_pathweak_v19")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.25
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.25
        cfg.losses.smoothness.foot_weight *= 1.25
        cfg.losses.smoothness.root_weight *= 1.15
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 88)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    if variant == "loss_high_large_escape_v2":
        cfg = _variant_cfg(cfg, "loss_high_large_avoid_v1")
        cfg = _variant_cfg(cfg, "loss_continuity_anchor_v1")
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.60,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.60,
        )
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 48)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 8.0e-3)
        return cfg
    if variant == "loss_semantic_all_v2":
        cfg = _variant_cfg(cfg, "loss_low_small_cross_v1")
        cfg = _variant_cfg(cfg, "loss_high_large_escape_v2")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 48)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 8.0e-3)
        return cfg
    if variant == "loss_high_large_smooth_v3":
        cfg = _variant_cfg(cfg, "loss_high_large_avoid_v1")
        cfg = _variant_cfg(cfg, "loss_continuity_anchor_v1")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 3.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 4.0
        cfg.losses.smoothness.foot_weight *= 2.0
        cfg.losses.smoothness.root_weight *= 2.0
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.50,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.55,
        )
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if variant == "loss_semantic_all_v3":
        cfg = _variant_cfg(cfg, "loss_low_small_cross_v1")
        cfg = _variant_cfg(cfg, "loss_high_large_smooth_v3")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if variant == "loss_high_large_margin_v4":
        cfg = _variant_cfg(cfg, "loss_high_large_smooth_v3")
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.52,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.56,
        )
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if variant == "loss_semantic_all_v4":
        cfg = _variant_cfg(cfg, "loss_low_small_cross_v1")
        cfg = _variant_cfg(cfg, "loss_high_large_margin_v4")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if variant == "loss_high_large_balanced_v5":
        cfg = _variant_cfg(cfg, "loss_high_large_escape_v2")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 4.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 5.0
        cfg.losses.smoothness.foot_weight *= 2.5
        cfg.losses.smoothness.root_weight *= 2.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 72)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 3.0e-3)
        return cfg
    if variant == "loss_semantic_all_v5":
        cfg = _variant_cfg(cfg, "loss_low_small_cross_v1")
        cfg = _variant_cfg(cfg, "loss_high_large_balanced_v5")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 72)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 3.0e-3)
        return cfg
    if variant == "loss_high_large_scurve_v6":
        cfg = _variant_cfg(cfg, "loss_high_large_balanced_v5")
        cfg.losses.high_obstacle_avoidance.lateral_clearance_m = max(
            float(cfg.losses.high_obstacle_avoidance.lateral_clearance_m),
            0.58,
        )
        cfg.losses.high_obstacle_avoidance.corridor_width_m = max(
            float(cfg.losses.high_obstacle_avoidance.corridor_width_m),
            0.62,
        )
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 80)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    if variant == "loss_semantic_all_v6":
        cfg = _variant_cfg(cfg, "loss_low_small_cross_v1")
        cfg = _variant_cfg(cfg, "loss_high_large_scurve_v6")
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 80)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    if variant == "loss_high_large_handoff_v7":
        cfg = _variant_cfg(cfg, "loss_continuity_anchor_v1")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 8.0e-3)
        return cfg
    if variant == "loss_high_large_handoff_v8":
        cfg = _variant_cfg(cfg, "loss_high_large_handoff_v7")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.5
        cfg.losses.smoothness.foot_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if variant == "loss_high_large_handoff_v9":
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.25
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 8.0e-3)
        return cfg
    if variant == "loss_high_large_handoff_v10":
        cfg = _variant_cfg(cfg, "loss_high_large_handoff_v9")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.25
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.25
        cfg.losses.smoothness.foot_weight *= 1.15
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 48)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 6.0e-3)
        return cfg
    if variant == "loss_high_large_ikfk_v11":
        cfg.losses.ik_fk_residual.weight *= 3.0
        cfg.losses.ik_fk_residual.contact_weight *= 2.0
        cfg.losses.kinematics.weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 8.0e-3)
        return cfg
    if variant == "loss_high_large_ikfk_v12":
        cfg = _variant_cfg(cfg, "loss_high_large_ikfk_v11")
        cfg.losses.ik_fk_residual.weight *= 2.0
        cfg.losses.ik_fk_residual.contact_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.25
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if variant == "loss_high_large_fksemantic_v13":
        cfg.losses.semantic_contact_avoid.weight *= 1.25
        cfg.losses.stance_semantic.weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 8.0e-3)
        return cfg
    if variant == "fk_output_semantic_v14":
        return _variant_cfg(cfg, "loss_high_large_fksemantic_v13")
    if variant == "fk_output_smooth_v1":
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 3.0
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.ik_fk_residual.weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 6.0e-3)
        return cfg
    if variant == "fk_output_smooth_v2":
        cfg = _variant_cfg(cfg, "fk_output_smooth_v1")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.5
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.ik_fk_residual.weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if variant == "nominal_cmd_shape_a_v1":
        return cfg
    if variant == "nominal_cmd_shape_a_smooth_v2":
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.0
        cfg.losses.smoothness.foot_weight *= 1.25
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 8.0e-3)
        return cfg
    if variant == "nominal_cmd_shape_a_low_small_guard_v3":
        cfg = _variant_cfg(cfg, "nominal_cmd_shape_a_smooth_v2")
        cfg = _variant_cfg(cfg, "loss_low_small_cont_v2")
        return cfg
    if variant == "nominal_cmd_shape_a_conservative_v4":
        return cfg
    if variant == "nominal_cmd_shape_a_low_exact_v4":
        return _variant_cfg(cfg, "loss_low_small_cont_v2")
    if variant == "nominal_cmd_shape_a_low_accel_v5":
        cfg = _variant_cfg(cfg, "loss_low_small_cont_v2")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.5
        cfg.losses.smoothness.foot_weight *= 1.75
        cfg.losses.smoothness.root_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if variant == "nominal_cmd_shape_a_low_accel_anchor_v5":
        cfg = _variant_cfg(cfg, "nominal_cmd_shape_a_low_accel_v5")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.5
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.5
        cfg.losses.smoothness.foot_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if variant == "nominal_cmd_shape_a_combined_v6":
        return _variant_cfg(cfg, "nominal_cmd_shape_a_low_accel_anchor_v5")
    raise ValueError(f"Unknown variant {name!r}; known={KNOWN_VARIANTS}")


def _effective_planning_variant_for_semantic(
    variant: str,
    *,
    semantic_class: str,
    semantic_target_height: float,
    command: tuple[float, float, float] | None = None,
    high_small_height_m: float = 0.30,
) -> str:
    if str(variant) not in {
        "nominal_cmd_shape_a_combined_v7",
        "nominal_cmd_shape_a_combined_v8",
        "nominal_cmd_shape_a_combined_v9",
        "nominal_cmd_shape_a_combined_v10",
    }:
        return str(variant)
    is_low_small = str(semantic_class) == "small" and float(semantic_target_height) <= float(high_small_height_m)
    linear_speed = math.sqrt(float(command[0]) ** 2 + float(command[1]) ** 2) if command is not None else 0.0
    lateral_or_yaw = bool(command is not None and (abs(float(command[1])) > 1.0e-4 or abs(float(command[2])) > 1.0e-4))
    if str(variant) == "nominal_cmd_shape_a_combined_v10":
        if is_low_small and linear_speed > 1.0e-4:
            return "loss_low_small_stepcap_v4" if lateral_or_yaw else "struct_lowfoot_cross_hard"
        return "nominal_cmd_shape_a_conservative_v4"
    if str(variant) == "nominal_cmd_shape_a_combined_v9":
        return "loss_low_small_stepcap_v4" if is_low_small else "nominal_cmd_shape_a_conservative_v4"
    if str(variant) == "nominal_cmd_shape_a_combined_v8":
        return "loss_low_small_cont_v2" if is_low_small else "nominal_cmd_shape_a_conservative_v4"
    return "nominal_cmd_shape_a_low_accel_anchor_v5" if is_low_small else "nominal_cmd_shape_a_conservative_v4"


def _semantic_command_shape_for_variant(
    variant: str,
    *,
    semantic_class: str,
    semantic_target_height: float,
    terrain,
    obstacle_xy: torch.Tensor,
    command: tuple[float, float, float],
    high_small_height_m: float = 0.30,
    min_lateral_speed_mps: float = 0.30,
    vx_scale: float = 0.45,
    yaw_scale: float = 0.75,
) -> tuple[tuple[float, float, float], dict[str, float | int | str]]:
    shape_variants = {
        "nominal_cmd_shape_a_v1",
        "nominal_cmd_shape_a_smooth_v2",
        "nominal_cmd_shape_a_low_small_guard_v3",
        "nominal_cmd_shape_a_conservative_v4",
        "nominal_cmd_shape_a_low_exact_v4",
        "nominal_cmd_shape_a_low_accel_v5",
        "nominal_cmd_shape_a_low_accel_anchor_v5",
        "nominal_cmd_shape_a_combined_v6",
    }
    vx, vy, yaw = (float(command[0]), float(command[1]), float(command[2]))
    diagnostics: dict[str, float | int | str] = {
        "requested_vx": vx,
        "requested_vy": vy,
        "requested_yaw": yaw,
        "nominal_vx": vx,
        "nominal_vy": vy,
        "nominal_yaw": yaw,
        "command_shaped": 0,
        "command_shape_side": 0,
        "command_shape_left_score": 0.0,
        "command_shape_right_score": 0.0,
        "command_shape_reason": "not_shape_variant",
    }
    if str(variant) not in shape_variants:
        return (vx, vy, yaw), diagnostics
    if str(variant) in {"nominal_cmd_shape_a_conservative_v4", "nominal_cmd_shape_a_combined_v6"}:
        min_lateral_speed_mps = 0.22
        vx_scale = 0.58
        yaw_scale = 0.90

    is_low_small = str(semantic_class) == "small" and float(semantic_target_height) <= float(high_small_height_m)
    linear_speed = math.sqrt(vx * vx + vy * vy)
    if is_low_small:
        diagnostics["command_shape_reason"] = "low_small_cross"
        return (vx, vy, yaw), diagnostics
    if linear_speed <= 1.0e-6:
        diagnostics["command_shape_reason"] = "no_linear_command"
        return (vx, vy, yaw), diagnostics

    from extension.batch_mpc_planner.terrain import semantic_at

    device = torch.as_tensor(terrain.height_map).device
    dtype = torch.float32
    cmd_xy = torch.tensor((vx, vy), dtype=dtype, device=device)
    direction = cmd_xy / torch.linalg.vector_norm(cmd_xy).clamp_min(1.0e-6)
    left = torch.stack((-direction[1], direction[0]))
    obstacle = torch.as_tensor(obstacle_xy, dtype=dtype, device=device)
    forward_offsets = torch.tensor((-0.10, 0.15, 0.40, 0.65), dtype=dtype, device=device)
    lateral_offsets = torch.tensor((0.24, 0.36, 0.50), dtype=dtype, device=device)

    def side_score(side: float) -> float:
        points = (
            obstacle
            + forward_offsets[:, None, None] * direction.view(1, 1, 2)
            + float(side) * lateral_offsets[None, :, None] * left.view(1, 1, 2)
        ).reshape(-1, 2)
        sem = semantic_at(terrain, points)
        sem_f = (torch.as_tensor(sem, dtype=dtype, device=device) > 0).to(dtype=dtype)
        near_weight = torch.linspace(1.0, 0.4, int(points.shape[0]), dtype=dtype, device=device)
        return float((sem_f.reshape(-1) * near_weight).sum().item())

    left_score = side_score(1.0)
    right_score = side_score(-1.0)
    side = -1.0 if left_score > right_score else 1.0
    current_lateral = vy / max(abs(vx), 1.0e-6)
    shaped_vx = vx * float(vx_scale)
    shaped_vy = side * max(abs(vy), float(min_lateral_speed_mps), abs(vx) * 0.55)
    if abs(current_lateral) > abs(shaped_vy / max(abs(shaped_vx), 1.0e-6)):
        shaped_vy = vy
    shaped_yaw = yaw * float(yaw_scale)
    shaped = (float(shaped_vx), float(shaped_vy), float(shaped_yaw))
    diagnostics.update(
        {
            "nominal_vx": shaped[0],
            "nominal_vy": shaped[1],
            "nominal_yaw": shaped[2],
            "command_shaped": 1,
            "command_shape_side": int(side),
            "command_shape_left_score": float(left_score),
            "command_shape_right_score": float(right_score),
            "command_shape_reason": "avoid_high_or_large",
        }
    )
    return shaped, diagnostics


def _command_relative_xy(
    origin_xy: tuple[float, float],
    command: tuple[float, float, float],
    *,
    longitudinal_offset_m: float,
    lateral_offset_m: float,
    device: torch.device,
) -> tuple[float, float]:
    command_xy = torch.tensor(command[:2], dtype=torch.float64, device=device)
    norm = torch.linalg.vector_norm(command_xy)
    if float(norm.item()) <= 1.0e-6:
        forward = torch.tensor((1.0, 0.0), dtype=torch.float64, device=device)
    else:
        forward = command_xy / norm
    left = torch.stack((-forward[1], forward[0]))
    origin = torch.tensor(origin_xy, dtype=torch.float64, device=device)
    xy = origin + forward * float(longitudinal_offset_m) + left * float(lateral_offset_m)
    return (float(xy[0].item()), float(xy[1].item()))


def _command_heading_yaw(command: tuple[float, float, float]) -> float | None:
    linear_speed = math.sqrt(float(command[0]) ** 2 + float(command[1]) ** 2)
    if linear_speed <= 1.0e-6:
        return None
    return 0.0


def _set_env0_yaw(runtime: RealViewerRuntimeFixture, yaw: float | None) -> None:
    if yaw is None:
        return
    root_pose = torch.cat(
        [
            torch.as_tensor(runtime.robot.data.root_pos_w, device=runtime.base_env.device, dtype=torch.float32),
            torch.as_tensor(runtime.robot.data.root_quat_w, device=runtime.base_env.device, dtype=torch.float32),
        ],
        dim=-1,
    ).clone()
    half = 0.5 * float(yaw)
    root_pose[0, 3] = math.cos(half)
    root_pose[0, 4] = 0.0
    root_pose[0, 5] = 0.0
    root_pose[0, 6] = math.sin(half)
    env_ids = torch.tensor([0], dtype=torch.long, device=runtime.base_env.device)
    runtime.robot.write_root_pose_to_sim(root_pose[:1], env_ids=env_ids)
    if hasattr(runtime.robot, "write_root_velocity_to_sim"):
        zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=runtime.base_env.device)
        runtime.robot.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    runtime.base_env.scene.write_data_to_sim()


def _crossing_metrics(root: torch.Tensor, obstacle_xy: torch.Tensor, command: tuple[float, float, float]) -> dict[str, float | int]:
    root = torch.as_tensor(root, dtype=torch.float32)
    obstacle_xy = torch.as_tensor(obstacle_xy, dtype=torch.float32, device=root.device)
    cmd_xy = torch.tensor(command[:2], dtype=torch.float32, device=root.device)
    norm = torch.linalg.vector_norm(cmd_xy)
    direction = torch.tensor((1.0, 0.0), dtype=torch.float32, device=root.device) if float(norm.item()) <= 1.0e-6 else cmd_xy / norm
    rel = root[0, :, :2] - obstacle_xy
    along = (rel * direction).sum(dim=-1)
    lateral = rel[:, 0] * (-direction[1]) + rel[:, 1] * direction[0]
    lateral_from_start = lateral - lateral[0]
    along_step = along[1:] - along[:-1] if int(along.numel()) > 1 else torch.empty(0, dtype=along.dtype, device=along.device)
    reverse_rate = _mask_rate(along_step < -1.0e-4) if int(along_step.numel()) > 0 else 0.0
    ever_crossed = bool((along[0] < 0.0 and torch.any(along[1:] > 0.0)).item()) if int(along.numel()) > 1 else False
    return {
        "min_root_distance_to_obstacle": float(torch.linalg.vector_norm(rel, dim=-1).min().item()),
        "min_abs_lateral_to_obstacle": float(torch.abs(lateral).min().item()),
        "max_abs_lateral_to_obstacle": float(torch.abs(lateral).max().item()),
        "root_lateral_deviation_from_start_max": float(torch.abs(lateral_from_start).max().item()),
        "root_along_reverse_rate": float(reverse_rate),
        "crossed_obstacle_along_command": int(bool(((along[0] * along[-1]) < 0.0).item())),
        "ever_crossed_obstacle_along_command": int(ever_crossed),
        "min_along_obstacle": float(along.min().item()),
        "max_along_obstacle": float(along.max().item()),
        "start_along_obstacle": float(along[0].item()),
        "end_along_obstacle": float(along[-1].item()),
    }


def _command_path_metrics(
    root: torch.Tensor,
    root_rpy: torch.Tensor,
    command: tuple[float, float, float],
    *,
    dt: float,
) -> dict[str, float]:
    root = torch.as_tensor(root, dtype=torch.float32)
    root_rpy = torch.as_tensor(root_rpy, dtype=torch.float32, device=root.device)
    if root.ndim != 3 or int(root.shape[1]) < 2:
        return {
            "command_path_lateral_error_max": 0.0,
            "command_path_lateral_error_mean": 0.0,
            "command_path_progress_error_final": 0.0,
        }
    cmd = torch.tensor(command[:3], dtype=torch.float32, device=root.device)
    yaw = root_rpy[:, :-1, 2]
    body_step = cmd[:2] * float(dt)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    world_step = torch.stack(
        (
            cy * body_step[0] - sy * body_step[1],
            sy * body_step[0] + cy * body_step[1],
        ),
        dim=-1,
    )
    expected = torch.zeros_like(root[..., :2])
    expected[:, :1] = root[:, :1, :2]
    expected[:, 1:] = root[:, :1, :2] + torch.cumsum(world_step, dim=1)
    path_delta = root[..., :2] - expected
    lateral_err = torch.linalg.vector_norm(path_delta, dim=-1)
    cmd_norm = torch.linalg.vector_norm(cmd[:2]).clamp_min(1.0e-6)
    heading0 = cmd[:2] / cmd_norm
    progress_error = ((root[:, -1, :2] - expected[:, -1]) * heading0).sum(dim=-1)
    return {
        "command_path_lateral_error_max": float(lateral_err.max().item()),
        "command_path_lateral_error_mean": float(lateral_err.mean().item()),
        "command_path_progress_error_final": float(torch.abs(progress_error).max().item()),
    }


def _root_rpy_from_viewer_result(result: object) -> torch.Tensor:
    root_pos = torch.as_tensor(result.root_pos_w, dtype=torch.float32)
    root_rpy = getattr(result, "root_rpy", None)
    if root_rpy is not None:
        return torch.as_tensor(root_rpy, dtype=torch.float32, device=root_pos.device)
    root_quat = torch.as_tensor(result.root_quat_w, dtype=torch.float32, device=root_pos.device)
    w, x, y, z = root_quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    rpy = torch.zeros_like(root_pos)
    rpy[..., 2] = yaw
    return rpy


def _replace_result_foot_positions(result: object, foot: torch.Tensor) -> object:
    foot_root = foot - torch.as_tensor(result.root_pos_w, dtype=foot.dtype, device=foot.device).unsqueeze(2)
    if hasattr(result, "__dataclass_fields__"):
        return replace(result, foot_pos_w=foot, foot_pos_root=foot_root)
    data = dict(result.__dict__)
    data["foot_pos_w"] = foot
    data["foot_pos_root"] = foot_root
    return type(result)(**data)


def _post_blend_result_for_variant(result: object, variant: str, *, blend_frames: int = 12):
    if str(variant) in {"fk_output_probe", "fk_output_semantic_v14", "fk_output_smooth_v1", "fk_output_smooth_v2"}:
        from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles

        root = torch.as_tensor(result.root_pos_w)
        rpy = _root_rpy_from_viewer_result(result).to(dtype=root.dtype, device=root.device)
        joints = torch.as_tensor(result.joint_angles, dtype=root.dtype, device=root.device)
        fk_foot = fk_feet_from_joint_angles(root, rpy, joints)
        return _replace_result_foot_positions(result, fk_foot)
    if str(variant) != "post_blend_body_hard_contact_only":
        return result
    foot = torch.as_tensor(result.foot_pos_w).clone()
    horizon = int(foot.shape[1])
    frames = min(int(blend_frames), horizon)
    if frames <= 1:
        return result
    start = foot[:, :1].clone()
    target = foot[:, frames - 1 : frames].clone()
    alpha = torch.linspace(0.0, 1.0, frames, dtype=foot.dtype, device=foot.device).view(1, frames, 1, 1)
    foot[:, :frames] = start + alpha * (target - start)
    return _replace_result_foot_positions(result, foot)


def _semantic_policy_metrics(
    *,
    semantic_class: str,
    semantic_target_height: float,
    command: tuple[float, float, float],
    crossed: int,
    high_small_height_m: float = 0.30,
    linear_speed_eps: float = 1.0e-4,
) -> dict[str, int]:
    linear_speed = math.sqrt(float(command[0]) ** 2 + float(command[1]) ** 2)
    is_linear = linear_speed > float(linear_speed_eps)
    is_low_small = str(semantic_class) == "small" and float(semantic_target_height) <= float(high_small_height_m)
    desired_crossing = int(bool(is_linear and is_low_small))
    crossed_int = int(bool(crossed))
    return {
        "desired_crossing": desired_crossing,
        "semantic_policy_violation": int(crossed_int != desired_crossing),
    }


def _semantic_policy_margin_metrics(
    *,
    semantic_class: str,
    semantic_target_diameter: float,
    semantic_target_height: float,
    command: tuple[float, float, float],
    crossed: int,
    min_root_distance: float,
    high_small_height_m: float = 0.30,
    avoid_extra_margin_m: float = 0.08,
) -> dict[str, float]:
    policy = _semantic_policy_metrics(
        semantic_class=semantic_class,
        semantic_target_height=semantic_target_height,
        command=command,
        crossed=crossed,
        high_small_height_m=high_small_height_m,
    )
    if int(policy["desired_crossing"]) == 1:
        return {"semantic_policy_margin_deficit": 0.0}
    linear_speed = math.sqrt(float(command[0]) ** 2 + float(command[1]) ** 2)
    if linear_speed <= 1.0e-4 and str(semantic_class) == "small":
        return {"semantic_policy_margin_deficit": 0.0}
    desired_distance = 0.5 * float(semantic_target_diameter) + float(avoid_extra_margin_m)
    deficit = max(0.0, desired_distance - float(min_root_distance))
    return {"semantic_policy_margin_deficit": float(deficit)}


def _semantic_clearance_policy_metrics(
    *,
    semantic_class: str,
    semantic_target_height: float,
    command: tuple[float, float, float],
    crossed: int,
    semantic_policy_margin_deficit: float,
    stance_on_semantic_rate: float,
    root_on_semantic_rate: float,
    foot_semantic_penetration_rate: float,
    high_small_height_m: float = 0.30,
    foot_penetration_tolerance: float = 0.002,
    linear_speed_eps: float = 1.0e-4,
) -> dict[str, int]:
    linear_speed = math.sqrt(float(command[0]) ** 2 + float(command[1]) ** 2)
    is_linear = linear_speed > float(linear_speed_eps)
    is_low_small = str(semantic_class) == "small" and float(semantic_target_height) <= float(high_small_height_m)
    if is_linear and is_low_small:
        violation = int(not bool(crossed))
    else:
        violation = int(
            float(semantic_policy_margin_deficit) > 1.0e-6
            or float(stance_on_semantic_rate) > 1.0e-6
            or float(root_on_semantic_rate) > 1.0e-6
            or float(foot_semantic_penetration_rate) > float(foot_penetration_tolerance)
        )
    return {"semantic_clearance_policy_violation": violation}


def _semantic_task_metrics(
    *,
    semantic_class: str,
    semantic_target_diameter: float,
    semantic_target_height: float,
    command: tuple[float, float, float],
    crossed: int,
    max_abs_lateral_to_obstacle: float,
    min_abs_lateral_to_obstacle: float,
    root_lateral_deviation_from_start_max: float,
    root_along_reverse_rate: float,
    command_path_lateral_error_max: float = 0.0,
    semantic_policy_margin_deficit: float,
    stance_on_semantic_rate: float,
    touchdown_on_semantic_rate: float,
    root_on_semantic_rate: float,
    foot_semantic_penetration_rate: float,
    foot_accel_max_to_mean: float,
    root_accel_max_to_mean: float,
    worst_max_to_median_step: float,
    worst_boundary_to_median_step: float,
    foot_over_low_small_success: int = 1,
    foot_step_anomaly_flag: int = 0,
    high_small_height_m: float = 0.30,
    crossing_lane_margin_m: float = 0.06,
    root_lateral_drift_margin_m: float = 0.10,
    root_reverse_rate_tolerance: float = 0.02,
    foot_penetration_tolerance: float = 0.002,
    foot_accel_limit: float = 30.0,
    root_accel_limit: float = 30.0,
    jump_limit: float = 30.0,
    boundary_limit: float = 12.0,
    linear_speed_eps: float = 1.0e-4,
) -> dict[str, float | int]:
    linear_speed = math.sqrt(float(command[0]) ** 2 + float(command[1]) ** 2)
    is_linear = linear_speed > float(linear_speed_eps)
    is_low_small = str(semantic_class) == "small" and float(semantic_target_height) <= float(high_small_height_m)
    semantic_clean = (
        float(stance_on_semantic_rate) <= 1.0e-6
        and float(touchdown_on_semantic_rate) <= 1.0e-6
        and float(foot_semantic_penetration_rate) <= float(foot_penetration_tolerance)
    )
    root_clean_for_avoid = float(root_on_semantic_rate) <= 1.0e-6
    continuous = (
        float(foot_accel_max_to_mean) <= float(foot_accel_limit)
        and float(root_accel_max_to_mean) <= float(root_accel_limit)
        and float(worst_max_to_median_step) <= float(jump_limit)
        and float(worst_boundary_to_median_step) <= float(boundary_limit)
        and int(foot_step_anomaly_flag) == 0
    )
    lane_limit = 0.5 * float(semantic_target_diameter) + float(crossing_lane_margin_m)
    drift_limit = lane_limit + float(root_lateral_drift_margin_m)
    path_drift = (
        float(command_path_lateral_error_max)
        if float(command_path_lateral_error_max) > 0.0
        else float(root_lateral_deviation_from_start_max)
    )
    local_overpass = float(min_abs_lateral_to_obstacle) <= lane_limit
    foot_over_small = int(foot_over_low_small_success) > 0
    small_overpass_success = int(
        bool(is_linear and is_low_small and crossed and local_overpass and foot_over_small and semantic_clean and continuous)
    )
    large_avoid_success = int(
        bool(
            (not is_low_small)
            and semantic_clean
            and root_clean_for_avoid
            and continuous
            and float(semantic_policy_margin_deficit) <= 1.0e-6
        )
    )
    if is_linear and is_low_small:
        violation = int(not bool(small_overpass_success))
    elif str(semantic_class) == "small" and not is_linear:
        violation = int(not bool(semantic_clean and continuous))
    else:
        violation = int(not bool(large_avoid_success))
    return {
        "small_overpass_success": small_overpass_success,
        "large_avoid_success": large_avoid_success,
        "semantic_task_violation": violation,
        "semantic_task_continuity_violation": int(not bool(continuous)),
        "semantic_task_contact_violation": int(not bool(semantic_clean)),
        "small_overpass_lane_limit": float(lane_limit),
        "small_overpass_drift_limit": float(drift_limit),
        "small_overpass_path_drift": float(path_drift),
        "small_overpass_local_lateral": float(min_abs_lateral_to_obstacle),
    }


def _candidate_variants_for_variant(
    variant: str,
    *,
    semantic_class: str | None = None,
    semantic_target_height: float | None = None,
    command: tuple[float, float, float] | None = None,
) -> tuple[str, ...]:
    if str(variant) in {"select_policy_class_wide_margin", "select_policy_class_jitter_margin"}:
        if semantic_class is None or semantic_target_height is None or command is None:
            return SELECTOR_VARIANTS["select_policy_pool"]
        desired = _semantic_policy_metrics(
            semantic_class=semantic_class,
            semantic_target_height=float(semantic_target_height),
            command=command,
            crossed=1,
        )["desired_crossing"]
        return LOW_SMALL_CROSSING_POOL if int(desired) == 1 else HIGH_LARGE_AVOIDANCE_POOL
    if str(variant) == "select_policy_class_hardcross_margin":
        if semantic_class is None or semantic_target_height is None or command is None:
            return SELECTOR_VARIANTS["select_policy_pool"]
        desired = _semantic_policy_metrics(
            semantic_class=semantic_class,
            semantic_target_height=float(semantic_target_height),
            command=command,
            crossed=1,
        )["desired_crossing"]
        return HARD_LOW_SMALL_CROSSING_POOL if int(desired) == 1 else HIGH_LARGE_AVOIDANCE_POOL
    if str(variant) in {
        "select_policy_class_risk_jitter_margin",
        "select_policy_class_priority_jitter_margin",
        "select_policy_class_clearance_jitter_margin",
        "select_policy_class_task_jitter_margin",
    }:
        if semantic_class is None or semantic_target_height is None or command is None:
            return SELECTOR_VARIANTS["select_policy_pool"]
        desired = _semantic_policy_metrics(
            semantic_class=semantic_class,
            semantic_target_height=float(semantic_target_height),
            command=command,
            crossed=1,
        )["desired_crossing"]
        return LOW_SMALL_CROSSING_POOL if int(desired) == 1 else RISK_HIGH_LARGE_AVOIDANCE_POOL
    if str(variant) == "select_policy_class_straight_task_jitter_margin":
        if semantic_class is None or semantic_target_height is None or command is None:
            return SELECTOR_VARIANTS["select_policy_pool"]
        desired = _semantic_policy_metrics(
            semantic_class=semantic_class,
            semantic_target_height=float(semantic_target_height),
            command=command,
            crossed=1,
        )["desired_crossing"]
        return STRAIGHT_LOW_SMALL_CROSSING_POOL if int(desired) == 1 else RISK_HIGH_LARGE_AVOIDANCE_POOL
    if str(variant) == "select_policy_class_path_task_jitter_margin":
        if semantic_class is None or semantic_target_height is None or command is None:
            return SELECTOR_VARIANTS["select_policy_pool"]
        desired = _semantic_policy_metrics(
            semantic_class=semantic_class,
            semantic_target_height=float(semantic_target_height),
            command=command,
            crossed=1,
        )["desired_crossing"]
        return STRAIGHT_LOW_SMALL_CROSSING_POOL if int(desired) == 1 else RISK_HIGH_LARGE_AVOIDANCE_POOL
    if str(variant) == "select_policy_class_metric_task_jitter_margin":
        if semantic_class is None or semantic_target_height is None or command is None:
            return SELECTOR_VARIANTS["select_policy_pool"]
        desired = _semantic_policy_metrics(
            semantic_class=semantic_class,
            semantic_target_height=float(semantic_target_height),
            command=command,
            crossed=1,
        )["desired_crossing"]
        return LOW_SMALL_CROSSING_POOL if int(desired) == 1 else RISK_HIGH_LARGE_AVOIDANCE_POOL
    if str(variant) == "select_policy_class_large_smooth_metric_margin":
        if semantic_class is None or semantic_target_height is None or command is None:
            return SELECTOR_VARIANTS["select_policy_pool"]
        desired = _semantic_policy_metrics(
            semantic_class=semantic_class,
            semantic_target_height=float(semantic_target_height),
            command=command,
            crossed=1,
        )["desired_crossing"]
        return LOW_SMALL_CROSSING_POOL if int(desired) == 1 else SMOOTH_HIGH_LARGE_AVOIDANCE_POOL
    return SELECTOR_VARIANTS.get(str(variant), (str(variant),))


def _selector_sort_key(row: dict[str, float | int | str]) -> tuple[float, ...]:
    contact = (
        float(row.get("stance_on_semantic_rate", 0.0))
        + float(row.get("touchdown_on_semantic_rate", 0.0))
        + float(row.get("root_on_semantic_rate", 0.0))
        + float(row.get("foot_semantic_penetration_rate", 0.0))
    )
    return (
        float(row.get("semantic_policy_violation", 0.0)),
        contact,
        float(row.get("semantic_policy_margin_deficit", 0.0)),
        float(row["score"]),
    )


def _selector_jitter_sort_key(row: dict[str, float | int | str]) -> tuple[float, ...]:
    contact = (
        float(row.get("stance_on_semantic_rate", 0.0))
        + float(row.get("touchdown_on_semantic_rate", 0.0))
        + float(row.get("root_on_semantic_rate", 0.0))
        + float(row.get("foot_semantic_penetration_rate", 0.0))
    )
    return (
        float(row.get("semantic_policy_violation", 0.0)),
        contact,
        float(row.get("semantic_policy_margin_deficit", 0.0)),
        float(row.get("foot_accel_max_to_mean", 0.0)),
        float(row.get("root_accel_max_to_mean", 0.0)),
        float(row.get("worst_max_to_median_step", 0.0)),
        float(row.get("worst_boundary_to_median_step", 0.0)),
        -float(row.get("min_z_quadratic_r2", 0.0)),
        float(row["score"]),
    )


def _selector_priority_jitter_sort_key(row: dict[str, float | int | str]) -> tuple[float, ...]:
    base_key = _selector_jitter_sort_key(row)
    return (*base_key[:-1], float(row.get("selector_candidate_priority", 99.0)), base_key[-1])


def _selector_clearance_jitter_sort_key(row: dict[str, float | int | str]) -> tuple[float, ...]:
    contact = (
        float(row.get("stance_on_semantic_rate", 0.0))
        + float(row.get("touchdown_on_semantic_rate", 0.0))
        + float(row.get("root_on_semantic_rate", 0.0))
        + float(row.get("foot_semantic_penetration_rate", 0.0))
    )
    return (
        float(row.get("semantic_clearance_policy_violation", row.get("semantic_policy_violation", 0.0))),
        contact,
        float(row.get("semantic_policy_margin_deficit", 0.0)),
        float(row.get("foot_accel_max_to_mean", 0.0)),
        float(row.get("root_accel_max_to_mean", 0.0)),
        float(row.get("worst_max_to_median_step", 0.0)),
        float(row.get("worst_boundary_to_median_step", 0.0)),
        -float(row.get("min_z_quadratic_r2", 0.0)),
        float(row.get("semantic_policy_violation", 0.0)),
        float(row["score"]),
    )


def _selector_task_jitter_sort_key(row: dict[str, float | int | str]) -> tuple[float, ...]:
    contact = (
        float(row.get("stance_on_semantic_rate", 0.0))
        + float(row.get("touchdown_on_semantic_rate", 0.0))
        + float(row.get("root_on_semantic_rate", 0.0))
        + float(row.get("foot_semantic_penetration_rate", 0.0))
    )
    return (
        float(row.get("semantic_task_violation", row.get("semantic_clearance_policy_violation", 0.0))),
        float(row.get("semantic_task_continuity_violation", 0.0)),
        float(row.get("semantic_task_contact_violation", 0.0)),
        contact,
        float(row.get("semantic_policy_margin_deficit", 0.0)),
        float(row.get("foot_accel_max_to_mean", 0.0)),
        float(row.get("root_accel_max_to_mean", 0.0)),
        float(row.get("worst_max_to_median_step", 0.0)),
        float(row.get("worst_boundary_to_median_step", 0.0)),
        float(row.get("command_path_lateral_error_max", row.get("root_lateral_deviation_from_start_max", 0.0))),
        float(row.get("root_lateral_deviation_from_start_max", 0.0)),
        float(row.get("max_abs_lateral_to_obstacle", 0.0)),
        -float(row.get("min_z_quadratic_r2", 0.0)),
        float(row["score"]),
    )


def _selector_metric_task_jitter_sort_key(row: dict[str, float | int | str]) -> tuple[float, ...]:
    contact = (
        float(row.get("stance_on_semantic_rate", 0.0))
        + float(row.get("touchdown_on_semantic_rate", 0.0))
        + float(row.get("root_on_semantic_rate", 0.0))
        + float(row.get("foot_semantic_penetration_rate", 0.0))
    )
    return (
        float(row.get("semantic_task_contact_violation", 0.0)),
        float(row.get("semantic_task_continuity_violation", 0.0)),
        -float(row.get("ever_crossed_obstacle_along_command", row.get("crossed_obstacle_along_command", 0.0))),
        contact,
        float(row.get("command_path_lateral_error_max", row.get("root_lateral_deviation_from_start_max", 0.0))),
        float(row.get("command_path_progress_error_final", 0.0)),
        float(row.get("foot_accel_max_to_mean", 0.0)),
        float(row.get("root_accel_max_to_mean", 0.0)),
        float(row.get("worst_max_to_median_step", 0.0)),
        float(row.get("worst_boundary_to_median_step", 0.0)),
        -float(row.get("min_z_quadratic_r2", 0.0)),
        float(row["score"]),
    )


def _selector_large_smooth_metric_sort_key(row: dict[str, float | int | str]) -> tuple[float, ...]:
    contact = (
        float(row.get("stance_on_semantic_rate", 0.0))
        + float(row.get("touchdown_on_semantic_rate", 0.0))
        + float(row.get("root_on_semantic_rate", 0.0))
        + float(row.get("foot_semantic_penetration_rate", 0.0))
    )
    return (
        float(row.get("semantic_task_contact_violation", 0.0)),
        contact,
        float(row.get("semantic_policy_margin_deficit", 0.0)),
        float(row.get("worst_max_to_median_step", 0.0)),
        float(row.get("worst_boundary_to_median_step", 0.0)),
        float(row.get("foot_accel_max_to_mean", 0.0)),
        float(row.get("root_accel_max_to_mean", 0.0)),
        -float(row.get("min_z_quadratic_r2", 0.0)),
        float(row["score"]),
    )


def _semantic_score(row: dict[str, float | int | str]) -> float:
    trajectory = _summary_score(row)
    return (
        trajectory
        + 250.0 * float(row.get("semantic_task_violation", 0.0))
        + 80.0 * float(row.get("semantic_task_continuity_violation", 0.0))
        + 120.0 * float(row.get("semantic_task_contact_violation", 0.0))
        + 50.0 * float(row["stance_on_semantic_rate"])
        + 20.0 * float(row["touchdown_on_semantic_rate"])
        + 20.0 * float(row["root_on_semantic_rate"])
        + 10.0 * max(0.0, float(row["foot_accel_max_to_mean"]) - 1.0)
        + 10.0 * max(0.0, float(row["root_accel_max_to_mean"]) - 1.0)
        + 150.0 * float(row.get("semantic_policy_violation", 0.0))
        + 80.0 * float(row.get("semantic_policy_margin_deficit", 0.0))
    )


def _aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["variant"]), str(row["semantic_class"]), str(row["command"])), []).append(row)
    summaries: list[dict[str, float | int | str]] = []
    for (variant, semantic_class, command), values in grouped.items():
        scores = [_semantic_score(row) for row in values]
        summaries.append(
            {
                "type": "semantic_obstacle_summary",
                "variant": variant,
                "semantic_class": semantic_class,
                "command": command,
                "cycle_count": len(values),
                "score_mean": sum(scores) / max(len(scores), 1),
                "score_max": max(scores) if scores else float("nan"),
                "max_worst_max_to_median_step": max(float(row["worst_max_to_median_step"]) for row in values),
                "max_worst_boundary_to_median_step": max(float(row["worst_boundary_to_median_step"]) for row in values),
                "min_z_quadratic_r2": min(float(row["min_z_quadratic_r2"]) for row in values),
                "max_root_accel_max_to_mean": max(float(row["root_accel_max_to_mean"]) for row in values),
                "max_foot_accel_max_to_mean": max(float(row["foot_accel_max_to_mean"]) for row in values),
                "max_any_foot_step_to_median": max(float(row.get("any_foot_step_to_median", 0.0)) for row in values),
                "max_terminal_foot_step_to_median": max(
                    float(row.get("terminal_foot_step_to_median", 0.0)) for row in values
                ),
                "max_replan_boundary_foot_step_to_median": max(
                    float(row.get("replan_boundary_foot_step_to_median", 0.0)) for row in values
                ),
                "max_rolling_segment_terminal_foot_error": max(
                    float(row.get("rolling_segment_terminal_foot_error_max", 0.0)) for row in values
                ),
                "max_rolling_segment_terminal_root_error": max(
                    float(row.get("rolling_segment_terminal_root_error_max", 0.0)) for row in values
                ),
                "max_terminal_stance_airborne": max(float(row.get("terminal_stance_airborne_max", 0.0)) for row in values),
                "foot_step_anomaly_count": sum(int(row.get("foot_step_anomaly_flag", 0)) for row in values),
                "replan_boundary_foot_anomaly_count": sum(
                    int(row.get("replan_boundary_foot_anomaly_flag", 0)) for row in values
                ),
                "terminal_foot_anomaly_count": sum(int(row.get("terminal_foot_anomaly_flag", 0)) for row in values),
                "max_stance_on_semantic_rate": max(float(row["stance_on_semantic_rate"]) for row in values),
                "max_touchdown_on_semantic_rate": max(float(row["touchdown_on_semantic_rate"]) for row in values),
                "max_root_on_semantic_rate": max(float(row["root_on_semantic_rate"]) for row in values),
                "min_root_distance_to_obstacle": min(float(row["min_root_distance_to_obstacle"]) for row in values),
                "max_command_path_lateral_error": max(
                    float(row.get("command_path_lateral_error_max", 0.0)) for row in values
                ),
                "max_command_path_progress_error": max(
                    float(row.get("command_path_progress_error_final", 0.0)) for row in values
                ),
                "max_semantic_policy_violation": max(int(row.get("semantic_policy_violation", 0)) for row in values),
                "semantic_task_violation_count": sum(int(row.get("semantic_task_violation", 0)) for row in values),
                "small_overpass_success_count": sum(int(row.get("small_overpass_success", 0)) for row in values),
                "large_avoid_success_count": sum(int(row.get("large_avoid_success", 0)) for row in values),
                "max_semantic_task_continuity_violation": max(
                    int(row.get("semantic_task_continuity_violation", 0)) for row in values
                ),
                "max_semantic_task_contact_violation": max(
                    int(row.get("semantic_task_contact_violation", 0)) for row in values
                ),
                "max_semantic_policy_margin_deficit": max(float(row.get("semantic_policy_margin_deficit", 0.0)) for row in values),
            }
        )
    summaries.sort(key=lambda row: (float(row["score_mean"]), str(row["semantic_class"]), str(row["command"])))
    return summaries


def _aggregate_variants(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    summaries: list[dict[str, float | int | str]] = []
    for variant, values in grouped.items():
        scores = [_semantic_score(row) for row in values]
        summaries.append(
            {
                "type": "variant_summary",
                "variant": variant,
                "cycle_count": len(values),
                "score_mean": sum(scores) / max(len(scores), 1),
                "score_max": max(scores) if scores else float("nan"),
                "max_worst_max_to_median_step": max(float(row["worst_max_to_median_step"]) for row in values),
                "max_worst_boundary_to_median_step": max(float(row["worst_boundary_to_median_step"]) for row in values),
                "min_z_quadratic_r2": min(float(row["min_z_quadratic_r2"]) for row in values),
                "max_root_accel_max_to_mean": max(float(row["root_accel_max_to_mean"]) for row in values),
                "max_foot_accel_max_to_mean": max(float(row["foot_accel_max_to_mean"]) for row in values),
                "max_any_foot_step_to_median": max(float(row.get("any_foot_step_to_median", 0.0)) for row in values),
                "max_terminal_foot_step_to_median": max(
                    float(row.get("terminal_foot_step_to_median", 0.0)) for row in values
                ),
                "max_replan_boundary_foot_step_to_median": max(
                    float(row.get("replan_boundary_foot_step_to_median", 0.0)) for row in values
                ),
                "max_rolling_segment_terminal_foot_error": max(
                    float(row.get("rolling_segment_terminal_foot_error_max", 0.0)) for row in values
                ),
                "max_rolling_segment_terminal_root_error": max(
                    float(row.get("rolling_segment_terminal_root_error_max", 0.0)) for row in values
                ),
                "max_terminal_stance_airborne": max(float(row.get("terminal_stance_airborne_max", 0.0)) for row in values),
                "foot_step_anomaly_count": sum(int(row.get("foot_step_anomaly_flag", 0)) for row in values),
                "replan_boundary_foot_anomaly_count": sum(
                    int(row.get("replan_boundary_foot_anomaly_flag", 0)) for row in values
                ),
                "terminal_foot_anomaly_count": sum(int(row.get("terminal_foot_anomaly_flag", 0)) for row in values),
                "max_stance_on_semantic_rate": max(float(row["stance_on_semantic_rate"]) for row in values),
                "max_touchdown_on_semantic_rate": max(float(row["touchdown_on_semantic_rate"]) for row in values),
                "max_root_on_semantic_rate": max(float(row["root_on_semantic_rate"]) for row in values),
                "max_semantic_penetration_rate": max(float(row["foot_semantic_penetration_rate"]) for row in values),
                "min_root_distance_to_obstacle": min(float(row["min_root_distance_to_obstacle"]) for row in values),
                "max_command_path_lateral_error": max(
                    float(row.get("command_path_lateral_error_max", 0.0)) for row in values
                ),
                "max_command_path_progress_error": max(
                    float(row.get("command_path_progress_error_final", 0.0)) for row in values
                ),
                "policy_violation_count": sum(int(row.get("semantic_policy_violation", 0)) for row in values),
                "clearance_policy_violation_count": sum(
                    int(row.get("semantic_clearance_policy_violation", 0)) for row in values
                ),
                "semantic_task_violation_count": sum(int(row.get("semantic_task_violation", 0)) for row in values),
                "small_overpass_success_count": sum(int(row.get("small_overpass_success", 0)) for row in values),
                "large_avoid_success_count": sum(int(row.get("large_avoid_success", 0)) for row in values),
                "continuity_violation_count": sum(
                    int(row.get("semantic_task_continuity_violation", 0)) for row in values
                ),
                "contact_violation_count": sum(int(row.get("semantic_task_contact_violation", 0)) for row in values),
                "max_semantic_policy_margin_deficit": max(float(row.get("semantic_policy_margin_deficit", 0.0)) for row in values),
            }
        )
    summaries.sort(key=lambda row: (float(row["score_mean"]), float(row["score_max"]), str(row["variant"])))
    return summaries


def _concat_viewer_results(parts: list[object]) -> object:
    if not parts:
        raise ValueError("cannot concatenate an empty result list")
    first = parts[0]
    tensor_fields = (
        "root_pos_w",
        "root_quat_w",
        "joint_angles",
        "foot_pos_w",
        "foot_pos_root",
        "contact_state",
    )
    data: dict[str, object] = {}
    for field in tensor_fields:
        values = [torch.as_tensor(getattr(part, field)) for part in parts]
        data[field] = torch.cat(values, dim=1)
    data["num_frames"] = int(torch.as_tensor(data["root_pos_w"]).shape[1])
    if getattr(first, "planned_touchdown_w", None) is not None:
        touchdown_values = [torch.as_tensor(getattr(part, "planned_touchdown_w")) for part in parts]
        if touchdown_values[0].ndim == 4:
            data["planned_touchdown_w"] = torch.cat(touchdown_values, dim=1)
        else:
            expanded = [
                value[:, None].expand(value.shape[0], int(torch.as_tensor(getattr(part, "root_pos_w")).shape[1]), *value.shape[1:])
                for part, value in zip(parts, touchdown_values)
            ]
            data["planned_touchdown_w"] = torch.cat(expanded, dim=1)
    else:
        data["planned_touchdown_w"] = None
    optional_fields = (
        "touchdown_seq",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "status",
        "feasible",
        "safe_fallback",
        "hard_reason_mask",
        "hard_reason_names",
        "status_names",
        "loss_breakdown",
    )
    for field in optional_fields:
        data[field] = getattr(first, field, None)
    return SimpleNamespace(**data)


def _clone_mpc_terrain(terrain):
    if terrain is None:
        return None
    values = {
        "height_map": torch.as_tensor(terrain.height_map).clone(),
        "world_x_range": tuple(terrain.world_x_range),
        "world_y_range": tuple(terrain.world_y_range),
        "semantic_map": None if getattr(terrain, "semantic_map", None) is None else torch.as_tensor(terrain.semantic_map).clone(),
        "sensor_pos_w": None if getattr(terrain, "sensor_pos_w", None) is None else torch.as_tensor(terrain.sensor_pos_w).clone(),
        "sensor_yaw": None if getattr(terrain, "sensor_yaw", None) is None else torch.as_tensor(terrain.sensor_yaw).clone(),
        "is_plane_terrain": None
        if getattr(terrain, "is_plane_terrain", None) is None
        else torch.as_tensor(terrain.is_plane_terrain).clone(),
    }
    return type(terrain)(**values)


def _plan_rolling_viewer_trajectory(
    runtime: RealViewerRuntimeFixture,
    *,
    terrain,
    state,
    command: torch.Tensor,
    total_frames: int,
    candidate_cfg,
    effective_candidate_variant: str,
    trace_terminal: bool = False,
) -> object:
    candidate_cfg = copy.deepcopy(candidate_cfg)
    candidate_cfg.runtime.horizon_steps = 25
    candidate_cfg.runtime.replan_interval_steps = 25
    horizon = max(2, int(candidate_cfg.runtime.horizon_steps))
    frames_left = int(total_frames)
    segment_state = state
    segment_terrain = terrain
    parts: list[object] = []
    initial_foot_errors: list[torch.Tensor] = []
    initial_touchdown_errors: list[torch.Tensor] = []
    terminal_foot_errors: list[torch.Tensor] = []
    terminal_root_errors: list[torch.Tensor] = []
    terminal_traces: list[dict[str, object]] = []
    segment_terrains: list[object] = []
    segment_lengths: list[int] = []
    segment_loss_breakdowns: list[object] = []
    while frames_left > 0:
        segment_index = len(parts)
        with _patched_structural_loss_for_variant(effective_candidate_variant):
            segment = runtime._viewer._plan_viewer_trajectory(
                terrain=segment_terrain,
                state=segment_state,
                command=command,
                mpc_cfg=candidate_cfg,
            )
        segment = _post_blend_result_for_variant(segment, effective_candidate_variant)
        finite_fields = {
            "root_pos_w": torch.as_tensor(segment.root_pos_w),
            "foot_pos_w": torch.as_tensor(segment.foot_pos_w),
            "joint_angles": torch.as_tensor(segment.joint_angles),
        }
        if not all(bool(torch.isfinite(value).all().item()) for value in finite_fields.values()):
            print(
                json.dumps(
                    {
                        "type": "rolling_segment_nonfinite",
                        "segment": int(segment_index),
                        **{name: int(torch.count_nonzero(~torch.isfinite(value)).item()) for name, value in finite_fields.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        play_count = min(frames_left, int(segment.num_frames))
        if play_count <= 0:
            break
        segment_terrains.append(_clone_mpc_terrain(segment_terrain))
        segment_lengths.append(int(play_count))
        segment_loss_breakdowns.append(getattr(segment, "loss_breakdown", None))
        segment_initial_foot = torch.as_tensor(
            segment.foot_pos_w[:, 0],
            dtype=torch.float64,
            device=runtime.base_env.device,
        )
        current_initial_foot = torch.as_tensor(
            segment_state.foot_pos,
            dtype=torch.float64,
            device=runtime.base_env.device,
        )
        initial_foot_errors.append(torch.linalg.vector_norm(segment_initial_foot - current_initial_foot, dim=-1).detach().cpu())
        planned_touchdown_initial = getattr(segment, "planned_touchdown_w", None)
        if planned_touchdown_initial is not None:
            planned_touchdown_initial_t = torch.as_tensor(
                planned_touchdown_initial,
                dtype=torch.float64,
                device=runtime.base_env.device,
            )
            if planned_touchdown_initial_t.ndim == 4:
                planned_touchdown_initial_t = planned_touchdown_initial_t[:, 0]
            initial_touchdown_errors.append(
                torch.linalg.vector_norm(planned_touchdown_initial_t - current_initial_foot, dim=-1).detach().cpu()
            )
        parts.append(
            SimpleNamespace(
                num_frames=play_count,
                root_pos_w=torch.as_tensor(segment.root_pos_w[:, :play_count]).clone(),
                root_quat_w=torch.as_tensor(segment.root_quat_w[:, :play_count]).clone(),
                joint_angles=torch.as_tensor(segment.joint_angles[:, :play_count]).clone(),
                foot_pos_w=torch.as_tensor(segment.foot_pos_w[:, :play_count]).clone(),
                foot_pos_root=torch.as_tensor(segment.foot_pos_root[:, :play_count]).clone(),
                contact_state=torch.as_tensor(segment.contact_state[:, :play_count]).clone(),
                planned_touchdown_w=(
                    None
                    if getattr(segment, "planned_touchdown_w", None) is None
                    else torch.as_tensor(segment.planned_touchdown_w[:, :play_count]).clone()
                    if torch.as_tensor(segment.planned_touchdown_w).ndim == 4
                    else torch.as_tensor(segment.planned_touchdown_w).clone()
                ),
                touchdown_seq=getattr(segment, "touchdown_seq", None),
                root_lin_vel_w=getattr(segment, "root_lin_vel_w", None),
                root_ang_vel_w=getattr(segment, "root_ang_vel_w", None),
                status=getattr(segment, "status", None),
                feasible=getattr(segment, "feasible", None),
                safe_fallback=getattr(segment, "safe_fallback", None),
                hard_reason_mask=getattr(segment, "hard_reason_mask", None),
                hard_reason_names=getattr(segment, "hard_reason_names", None),
                status_names=getattr(segment, "status_names", None),
                loss_breakdown=getattr(segment, "loss_breakdown", None),
            )
        )
        runtime._viewer._viewer_direct_playback_step(runtime.base_env, segment, frame_idx=play_count - 1)
        robot = runtime.base_env.scene["robot"]
        if hasattr(robot, "write_root_velocity_to_sim"):
            root_vel = torch.zeros((1, 6), dtype=torch.float32, device=runtime.base_env.device)
            robot.write_root_velocity_to_sim(root_vel)
        if hasattr(robot, "write_joint_state_to_sim"):
            joint_pos = torch.as_tensor(robot.data.joint_pos[:1], dtype=torch.float32, device=runtime.base_env.device)
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
        if hasattr(runtime.base_env.scene, "write_data_to_sim"):
            runtime.base_env.scene.write_data_to_sim()
        refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
        segment_state = runtime._single_env_state()
        planned_terminal_foot = torch.as_tensor(
            segment.foot_pos_w[:, play_count - 1],
            dtype=torch.float64,
            device=runtime.base_env.device,
        )
        planned_terminal_root = torch.as_tensor(
            segment.root_pos_w[:, play_count - 1],
            dtype=torch.float64,
            device=runtime.base_env.device,
        )
        actual_terminal_foot = torch.as_tensor(
            segment_state.foot_pos,
            dtype=torch.float64,
            device=runtime.base_env.device,
        )
        actual_terminal_root = torch.as_tensor(
            segment_state.root_pos,
            dtype=torch.float64,
            device=runtime.base_env.device,
        )
        terminal_foot_errors.append(torch.linalg.vector_norm(actual_terminal_foot - planned_terminal_foot, dim=-1).detach().cpu())
        terminal_root_errors.append(torch.linalg.vector_norm(actual_terminal_root - planned_terminal_root, dim=-1).detach().cpu())
        if trace_terminal:
            planned_joint = torch.as_tensor(
                segment.joint_angles[:, play_count - 1],
                dtype=torch.float64,
                device=runtime.base_env.device,
            )
            planned_rpy = torch.as_tensor(
                segment.root_quat_w[:, play_count - 1],
                dtype=torch.float64,
                device=runtime.base_env.device,
            )
            planned_rpy = runtime._viewer._quat_wxyz_to_rpy(planned_rpy)
            raw_ik_joint = solve_joint_angles_from_trajectory(
                planned_terminal_root[:, None, :],
                planned_rpy[:, None, :],
                planned_terminal_foot[:, None, :, :],
                clamp_to_limits=False,
            )[:, 0]
            clamped_ik_joint = solve_joint_angles_from_trajectory(
                planned_terminal_root[:, None, :],
                planned_rpy[:, None, :],
                planned_terminal_foot[:, None, :, :],
                clamp_to_limits=True,
            )[:, 0]
            internal_fk_foot = fk_feet_from_joint_angles(
                planned_terminal_root[:, None, :],
                planned_rpy[:, None, :],
                planned_joint[:, None, :],
            )[:, 0]
            actual_joint = torch.as_tensor(
                segment_state.joint_angles,
                dtype=torch.float64,
                device=runtime.base_env.device,
            )
            planned_touchdown = getattr(segment, "planned_touchdown_w", None)
            if planned_touchdown is not None:
                planned_touchdown_t = torch.as_tensor(planned_touchdown, dtype=torch.float64, device=runtime.base_env.device)
                if planned_touchdown_t.ndim == 4:
                    planned_touchdown_t = planned_touchdown_t[:, min(play_count - 1, planned_touchdown_t.shape[1] - 1)]
            else:
                planned_touchdown_t = None
            contact_state = getattr(segment, "contact_state", None)
            if contact_state is not None:
                contact_state_t = torch.as_tensor(contact_state, device=runtime.base_env.device)
                contact_state_t = contact_state_t[:, play_count - 1]
            else:
                contact_state_t = None
            terminal_traces.append(
                {
                    "segment": int(segment_index),
                    "frame": int(play_count - 1),
                    "planned_foot": planned_terminal_foot.detach().cpu(),
                    "actual_foot": actual_terminal_foot.detach().cpu(),
                    "planned_root": planned_terminal_root.detach().cpu(),
                    "actual_root": actual_terminal_root.detach().cpu(),
                    "planned_joint": planned_joint.detach().cpu(),
                    "actual_joint": actual_joint.detach().cpu(),
                    "raw_ik_joint": raw_ik_joint.detach().cpu(),
                    "clamped_ik_joint": clamped_ik_joint.detach().cpu(),
                    "internal_fk_foot": internal_fk_foot.detach().cpu(),
                    "planned_touchdown": None if planned_touchdown_t is None else planned_touchdown_t.detach().cpu(),
                    "contact_state": None if contact_state_t is None else contact_state_t.detach().cpu(),
                }
            )
        state_fields = {
            "root_pos": torch.as_tensor(segment_state.root_pos),
            "foot_pos": torch.as_tensor(segment_state.foot_pos),
            "joint_angles": torch.as_tensor(segment_state.joint_angles),
        }
        if not all(bool(torch.isfinite(value).all().item()) for value in state_fields.values()):
            print(
                json.dumps(
                    {
                        "type": "rolling_state_nonfinite",
                        "segment": int(segment_index),
                        **{name: int(torch.count_nonzero(~torch.isfinite(value)).item()) for name, value in state_fields.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        segment_terrain = runtime._single_env_terrain()
        frames_left -= play_count
    result = _concat_viewer_results(parts)
    result.rolling_segment_terminal_foot_error = (
        torch.cat(terminal_foot_errors, dim=0) if terminal_foot_errors else torch.empty((0, 4), dtype=torch.float64)
    )
    result.rolling_segment_initial_foot_error = (
        torch.cat(initial_foot_errors, dim=0) if initial_foot_errors else torch.empty((0, 4), dtype=torch.float64)
    )
    result.rolling_segment_initial_touchdown_error = (
        torch.cat(initial_touchdown_errors, dim=0) if initial_touchdown_errors else torch.empty((0, 4), dtype=torch.float64)
    )
    result.rolling_segment_terminal_root_error = (
        torch.cat(terminal_root_errors, dim=0) if terminal_root_errors else torch.empty((0,), dtype=torch.float64)
    )
    result.rolling_segment_terminal_traces = tuple(terminal_traces)
    result.rolling_segment_terrains = tuple(segment_terrains)
    result.rolling_segment_lengths = tuple(segment_lengths)
    result.rolling_segment_loss_breakdowns = tuple(segment_loss_breakdowns)
    return result


def run_probe(
    *,
    device: str,
    cases: tuple[str, ...],
    commands: tuple[str, ...],
    variants: tuple[str, ...],
    cycles: int,
    requested_n_frames: int,
    playback_frame: int,
    warmup_steps: int,
    longitudinal_offset_m: float,
    lateral_offset_m: float,
    z_clearance: float,
    semantic_small_height_m: float | None,
    trace_foot_mismatch: bool = False,
) -> int:
    runtime = RealViewerRuntimeFixture(
        num_envs=1,
        device=device,
        planner_backend="mpc",
        requested_n_frames=requested_n_frames,
        warmup_steps=warmup_steps,
        task_id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
        env_cfg_entry_point=(
            "go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg:"
            "TeacherElevationTrajectoryMpcSemanticEnvCfg"
        ),
        semantic_small_height_m=semantic_small_height_m,
    )
    rows: list[dict[str, float | int | str]] = []
    try:
        print(
            json.dumps(
                {
                    "type": "probe_header",
                    "device": device,
                    "cases": list(cases),
                    "commands": list(commands),
                    "variants": list(variants),
                    "cycles": int(cycles),
                    "requested_n_frames": int(requested_n_frames),
                    "playback_frame": int(playback_frame),
                    "warmup_steps": int(warmup_steps),
                    "longitudinal_offset_m": float(longitudinal_offset_m),
                    "lateral_offset_m": float(lateral_offset_m),
                    "semantic_small_height_m": semantic_small_height_m,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for semantic_class in cases:
            if semantic_class not in {"small", "large"}:
                raise ValueError(f"case must be 'small' or 'large', got {semantic_class!r}")
            anchor = runtime.s4_semantic_course_anchor(semantic_class)
            obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=runtime.base_env.device)
            for command_text in commands:
                command_name, command_tuple = _parse_command(command_text)
                for variant in variants:
                    runtime.reset()
                    start_xy = _command_relative_xy(
                        anchor.world_xy,
                        command_tuple,
                        longitudinal_offset_m=longitudinal_offset_m,
                        lateral_offset_m=lateral_offset_m,
                        device=runtime.base_env.device,
                    )
                    runtime._write_env0_root_xy(start_xy, z_clearance=z_clearance)
                    _set_env0_yaw(runtime, _command_heading_yaw(command_tuple))
                    runtime._sync_targeted_scan_pose()
                    state = runtime._single_env_state()
                    for cycle in range(int(cycles)):
                        terrain = runtime._single_env_terrain()
                        candidate_rows: list[tuple[dict[str, float | int | str], object]] = []
                        for candidate_variant in _candidate_variants_for_variant(
                            variant,
                            semantic_class=semantic_class,
                            semantic_target_height=float(anchor.target_height),
                            command=command_tuple,
                        ):
                            effective_candidate_variant = _effective_planning_variant_for_semantic(
                                candidate_variant,
                                semantic_class=semantic_class,
                                semantic_target_height=float(anchor.target_height),
                                command=command_tuple,
                            )
                            shaped_command_tuple, command_shape_diagnostics = _semantic_command_shape_for_variant(
                                effective_candidate_variant,
                                semantic_class=semantic_class,
                                semantic_target_height=float(anchor.target_height),
                                terrain=terrain,
                                obstacle_xy=obstacle_xy,
                                command=command_tuple,
                            )
                            planning_command = torch.tensor(
                                [shaped_command_tuple],
                                dtype=torch.float64,
                                device=runtime.base_env.device,
                            )
                            candidate_cfg = _variant_cfg(runtime.mpc_planner_cfg, effective_candidate_variant)
                            probe_seed = _semantic_probe_seed(
                                semantic_class=semantic_class,
                                command_name=command_name,
                                cycle=cycle,
                                effective_candidate=effective_candidate_variant,
                            )
                            torch.manual_seed(probe_seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(probe_seed)
                            candidate_result = _plan_rolling_viewer_trajectory(
                                runtime,
                                terrain=terrain,
                                state=state,
                                command=planning_command,
                                total_frames=runtime.requested_n_frames,
                                candidate_cfg=candidate_cfg,
                                effective_candidate_variant=effective_candidate_variant,
                                trace_terminal=bool(trace_foot_mismatch),
                            )
                            summary = _trajectory_summary(
                                command_name=command_name,
                                cycle=cycle,
                                result=candidate_result,
                                layer="result",
                                variant=variant,
                                terrain_case=f"semantic_{semantic_class}",
                            )[0]
                            row = {
                                **summary,
                                **_jitter_metrics(candidate_result),
                                **_semantic_collision_metrics(candidate_result, terrain),
                                **_terminal_foot_anomaly_metrics(
                                    candidate_result,
                                    terrain,
                                    replan_interval_steps=25,
                                ),
                                **_rolling_segment_playback_error_metrics(candidate_result),
                                **_low_small_foot_over_metrics(
                                    candidate_result,
                                    terrain,
                                    obstacle_xy,
                                    semantic_target_height=float(anchor.target_height),
                                ),
                                **_crossing_metrics(torch.as_tensor(candidate_result.root_pos_w), obstacle_xy, command_tuple),
                                **_command_path_metrics(
                                    torch.as_tensor(candidate_result.root_pos_w),
                                    _root_rpy_from_viewer_result(candidate_result),
                                    command_tuple,
                                    dt=runtime.plan_dt,
                                ),
                                "type": "semantic_obstacle_cycle",
                                "variant": variant,
                                "selected_candidate": candidate_variant,
                                "effective_candidate": effective_candidate_variant,
                                "semantic_class": semantic_class,
                                "semantic_target_diameter": float(anchor.target_diameter),
                                "semantic_target_height": float(anchor.target_height),
                                "semantic_anchor_x": float(anchor.world_xy[0]),
                                "semantic_anchor_y": float(anchor.world_xy[1]),
                                "semantic_probe_seed": int(probe_seed),
                                "score": 0.0,
                            }
                            row.update(command_shape_diagnostics)
                            row["selector_candidate_priority"] = float(
                                _candidate_variants_for_variant(
                                    variant,
                                    semantic_class=semantic_class,
                                    semantic_target_height=float(anchor.target_height),
                                    command=command_tuple,
                                ).index(candidate_variant)
                            )
                            row.update(
                                _semantic_policy_metrics(
                                    semantic_class=semantic_class,
                                    semantic_target_height=float(anchor.target_height),
                                    command=command_tuple,
                                    crossed=int(row["ever_crossed_obstacle_along_command"]),
                                )
                            )
                            row.update(
                                _semantic_policy_margin_metrics(
                                    semantic_class=semantic_class,
                                    semantic_target_diameter=float(anchor.target_diameter),
                                    semantic_target_height=float(anchor.target_height),
                                    command=command_tuple,
                                    crossed=int(row["ever_crossed_obstacle_along_command"]),
                                    min_root_distance=float(row["min_root_distance_to_obstacle"]),
                                )
                            )
                            row.update(
                                _semantic_clearance_policy_metrics(
                                    semantic_class=semantic_class,
                                    semantic_target_height=float(anchor.target_height),
                                    command=command_tuple,
                                    crossed=int(row["ever_crossed_obstacle_along_command"]),
                                    semantic_policy_margin_deficit=float(row["semantic_policy_margin_deficit"]),
                                    stance_on_semantic_rate=float(row["stance_on_semantic_rate"]),
                                    root_on_semantic_rate=float(row["root_on_semantic_rate"]),
                                    foot_semantic_penetration_rate=float(row["foot_semantic_penetration_rate"]),
                                )
                            )
                            row.update(
                                _semantic_task_metrics(
                                    semantic_class=semantic_class,
                                    semantic_target_diameter=float(anchor.target_diameter),
                                    semantic_target_height=float(anchor.target_height),
                                    command=command_tuple,
                                    crossed=int(row["ever_crossed_obstacle_along_command"]),
                                    max_abs_lateral_to_obstacle=float(row["max_abs_lateral_to_obstacle"]),
                                    min_abs_lateral_to_obstacle=float(row["min_abs_lateral_to_obstacle"]),
                                    root_lateral_deviation_from_start_max=float(
                                        row["root_lateral_deviation_from_start_max"]
                                    ),
                                    root_along_reverse_rate=float(row["root_along_reverse_rate"]),
                                    command_path_lateral_error_max=float(row["command_path_lateral_error_max"]),
                                    semantic_policy_margin_deficit=float(row["semantic_policy_margin_deficit"]),
                                    stance_on_semantic_rate=float(row["stance_on_semantic_rate"]),
                                    touchdown_on_semantic_rate=float(row["touchdown_on_semantic_rate"]),
                                    root_on_semantic_rate=float(row["root_on_semantic_rate"]),
                                    foot_semantic_penetration_rate=float(row["foot_semantic_penetration_rate"]),
                                    foot_accel_max_to_mean=float(row["foot_accel_max_to_mean"]),
                                    root_accel_max_to_mean=float(row["root_accel_max_to_mean"]),
                                    worst_max_to_median_step=float(row["worst_max_to_median_step"]),
                                    worst_boundary_to_median_step=float(row["worst_boundary_to_median_step"]),
                                    foot_over_low_small_success=int(row["foot_over_low_small_success"]),
                                    foot_step_anomaly_flag=int(row.get("foot_step_anomaly_flag", 0)),
                                )
                            )
                            row["score"] = _semantic_score(row)
                            candidate_rows.append((row, candidate_result))
                        selector_key = (
                            _selector_jitter_sort_key
                            if str(variant) in {"select_policy_class_jitter_margin", "select_policy_class_risk_jitter_margin"}
                            else _selector_priority_jitter_sort_key
                            if str(variant) == "select_policy_class_priority_jitter_margin"
                            else _selector_clearance_jitter_sort_key
                            if str(variant) == "select_policy_class_clearance_jitter_margin"
                            else _selector_task_jitter_sort_key
                            if str(variant)
                            in {
                                "select_policy_class_task_jitter_margin",
                                "select_policy_class_straight_task_jitter_margin",
                                "select_policy_class_path_task_jitter_margin",
                            }
                            else _selector_metric_task_jitter_sort_key
                            if str(variant) == "select_policy_class_metric_task_jitter_margin"
                            else _selector_large_smooth_metric_sort_key
                            if str(variant) == "select_policy_class_large_smooth_metric_margin"
                            else _selector_sort_key
                        )
                        row, result = min(candidate_rows, key=lambda item: selector_key(item[0]))
                        if trace_foot_mismatch:
                            for trace_row in _rolling_segment_terminal_trace_rows(
                                getattr(result, "rolling_segment_terminal_traces", ())
                            ):
                                trace_row.update(
                                    {
                                        "command": command_name,
                                        "cycle": int(cycle),
                                        "variant": variant,
                                        "selected_candidate": str(row["selected_candidate"]),
                                        "effective_candidate": str(row["effective_candidate"]),
                                        "semantic_class": semantic_class,
                                    }
                                )
                                print(json.dumps(trace_row, sort_keys=True), flush=True)
                        frame_idx = min(int(playback_frame), int(result.num_frames) - 1)
                        planned_root = torch.as_tensor(result.root_pos_w[:, frame_idx], dtype=torch.float64, device=runtime.base_env.device)
                        planned_foot = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64, device=runtime.base_env.device)
                        runtime._viewer._viewer_direct_playback_step(runtime.base_env, result, frame_idx=frame_idx)
                        playback_state = runtime._single_env_state()
                        actual_root = torch.as_tensor(playback_state.root_pos, dtype=torch.float64, device=runtime.base_env.device)
                        actual_foot = torch.as_tensor(playback_state.foot_pos, dtype=torch.float64, device=runtime.base_env.device)
                        root_playback_error = torch.linalg.vector_norm(actual_root - planned_root, dim=-1)
                        foot_playback_error = torch.linalg.vector_norm(actual_foot - planned_foot, dim=-1)
                        row.update(
                            {
                                "playback_frame": int(frame_idx),
                                "playback_root_error_max": float(root_playback_error.max().item()),
                                "playback_root_error_mean": float(root_playback_error.mean().item()),
                                "playback_foot_error_max": float(foot_playback_error.max().item()),
                                "playback_foot_error_mean": float(foot_playback_error.mean().item()),
                                "playback_terminal_root_error_max": float(root_playback_error.max().item()),
                                "playback_terminal_foot_error_max": float(foot_playback_error.max().item())
                                if frame_idx == int(result.num_frames) - 1
                                else 0.0,
                            }
                        )
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
                        refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
                        state = runtime._single_env_state()
        summaries = _aggregate(rows)
        for summary in summaries:
            print(json.dumps(summary, sort_keys=True), flush=True)
        variant_summaries = _aggregate_variants(rows)
        for summary in variant_summaries:
            print(json.dumps(summary, sort_keys=True), flush=True)
        print(
            json.dumps(
                {
                    "type": "probe_footer",
                    "cycle_count": len(rows),
                    "summary_count": len(summaries),
                    "best_variant": str(variant_summaries[0]["variant"]) if variant_summaries else "",
                    "best_variant_score_mean": float(variant_summaries[0]["score_mean"]) if variant_summaries else float("nan"),
                    "worst_score": max((_semantic_score(row) for row in rows), default=float("nan")),
                    "max_stance_on_semantic_rate": max((float(row["stance_on_semantic_rate"]) for row in rows), default=0.0),
                    "max_touchdown_on_semantic_rate": max(
                        (float(row["touchdown_on_semantic_rate"]) for row in rows),
                        default=0.0,
                    ),
                    "max_root_on_semantic_rate": max((float(row["root_on_semantic_rate"]) for row in rows), default=0.0),
                    "max_root_accel_max_to_mean": max((float(row["root_accel_max_to_mean"]) for row in rows), default=0.0),
                    "max_foot_accel_max_to_mean": max((float(row["foot_accel_max_to_mean"]) for row in rows), default=0.0),
                    "semantic_task_violation_count": sum(int(row.get("semantic_task_violation", 0)) for row in rows),
                    "small_overpass_success_count": sum(int(row.get("small_overpass_success", 0)) for row in rows),
                    "large_avoid_success_count": sum(int(row.get("large_avoid_success", 0)) for row in rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        runtime.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", default="small,large")
    parser.add_argument("--commands", default=",".join(DEFAULT_COMMANDS))
    parser.add_argument(
        "--variants",
        default="baseline",
        help="Comma-separated test-only cfg variants: " + ",".join(KNOWN_VARIANTS),
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--requested-n-frames", type=int, default=300)
    parser.add_argument("--playback-frame", type=int, default=299)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--longitudinal-offset-m", type=float, default=-0.35)
    parser.add_argument("--lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--z-clearance", type=float, default=0.65)
    parser.add_argument("--semantic-small-height-m", type=float, default=None)
    parser.add_argument("--trace-foot-mismatch", action="store_true", default=False)
    args = parser.parse_args()
    cases = tuple(item.strip() for item in str(args.cases).split(",") if item.strip())
    commands = tuple(item.strip() for item in str(args.commands).split(",") if item.strip())
    variants = tuple(item.strip() for item in str(args.variants).split(",") if item.strip())
    return run_probe(
        device=str(args.device),
        cases=cases,
        commands=commands,
        variants=variants,
        cycles=int(args.cycles),
        requested_n_frames=int(args.requested_n_frames),
        playback_frame=int(args.playback_frame),
        warmup_steps=int(args.warmup_steps),
        longitudinal_offset_m=float(args.longitudinal_offset_m),
        lateral_offset_m=float(args.lateral_offset_m),
        z_clearance=float(args.z_clearance),
        semantic_small_height_m=args.semantic_small_height_m,
        trace_foot_mismatch=bool(args.trace_foot_mismatch),
    )


if __name__ == "__main__":
    raise SystemExit(main())
