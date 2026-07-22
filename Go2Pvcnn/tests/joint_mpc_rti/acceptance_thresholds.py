from __future__ import annotations

from dataclasses import dataclass


THRESHOLDS: dict[str, tuple[str, float]] = {
    "stance_xy_slip_max_m": ("le", 0.0005),
    "stance_xy_slip_mean_m": ("le", 0.0002),
    "stance_stationary_ratio": ("ge", 1.0),
    "stance_root_carry_ratio_abs": ("le", 0.10),
    "swing_active_motion_ratio": ("ge", 0.50),
    "foot_root_lead_time_min_ms": ("ge", 20.0),
    "foot_root_lead_time_max_ms": ("le", 80.0),
    "root_leak_before_foot_m": ("le", 0.0005),
    "root_roll_error_abs_max_deg": ("le", 6.0),
    "root_pitch_error_abs_max_deg": ("le", 6.0),
    "root_lateral_offset_from_nominal_m": ("le", 0.06),
    "root_lateral_velocity_error_mps": ("le", 0.20),
    "root_yaw_error_from_nominal_deg": ("le", 10.0),
    "root_roll_pitch_rate_max_rps": ("le", 0.6),
    "root_yaw_rate_assist_error_rps": ("le", 0.8),
    "airborne_touchdown_count": ("le", 0.0),
    "touchdown_on_small_count": ("le", 0.0),
    "unsafe_stance_anchor_count": ("le", 0.0),
    "forced_touchdown_after_extension_count": ("le", 0.0),
    "liftoff_guard_violation_count": ("le", 0.0),
    "cross_success_rate": ("ge", 0.95),
    "foot_collision_frame_rate": ("le", 0.0),
    "knee_collision_frame_rate": ("le", 0.0),
    "calf_collision_frame_rate": ("le", 0.0),
    "thigh_collision_frame_rate": ("le", 0.0),
    "base_collision_frame_rate": ("le", 0.0),
    "line_search_alpha_0_rate": ("le", 0.10),
    "line_search_alpha_0_max_run": ("le", 2.0),
}


@dataclass(frozen=True)
class MetricCellResult:
    key: tuple[str, ...]
    values: dict[str, float | int | None]
    passed: bool
    failures: tuple[str, ...]


def evaluate_metric_cell(
    key: tuple[str, ...],
    values: dict[str, float | int | None],
) -> MetricCellResult:
    failures: list[str] = []
    for name, value in values.items():
        if name not in THRESHOLDS or value is None:
            continue
        operator, threshold = THRESHOLDS[name]
        scalar = float(value)
        passed = scalar <= threshold if operator == "le" else scalar >= threshold
        if not passed:
            failures.append(name)
    return MetricCellResult(key=key, values=values, passed=not failures, failures=tuple(failures))


__all__ = ["MetricCellResult", "THRESHOLDS", "evaluate_metric_cell"]
