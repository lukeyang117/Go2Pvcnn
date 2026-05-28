"""Config contracts for the batch MPC backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class MpcRuntimeCfg:
    horizon_steps: int = 25
    dt: float = 0.02
    optimize_steps: int = 24
    lr: float = 2e-2
    optimizer: str = "adam"
    grad_clip_norm: float = 10.0
    contact_threshold: float = 0.40
    replan_interval_steps: int = 50
    max_stale_steps: int = 100
    warm_start_from_previous_plan: bool = True
    detach_warm_start: bool = True
    detach_cache_on_write: bool = True
    heavy_loss_stride: int = 2
    heavy_loss_enable_from_iter: int = 8
    parallel_plan_batch_size: int = 4096
    randomize_replan_phase: bool = False
    randomize_command_phase: bool = True
    command_hard_lin_delta: float = 0.25
    command_hard_yaw_delta: float = 0.35
    command_soft_lin_delta: float = 0.05
    command_soft_yaw_delta: float = 0.10
    command_blend_steps: int = 8
    terrain_subset_before_build: bool = True
    step_local_reference_cache: bool = True
    train_dtype: str = "float32"
    amp_enabled: bool = False
    optimizer_unroll_graph: bool = False
    profile_4096_required: bool = True
    step_freq: float = 2.0
    duty_factor: float = 0.5
    leg_phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
    touchdown_event_cap: int = 2
    nominal_stride_scale: float = 0.5
    nominal_swing_height_m: float = 0.12
    nominal_yaw_stride_scale: float = 0.5
    swing_window_min_width: float = 0.30
    swing_window_max_width: float = 0.70
    swing_window_center_scale: float = 0.60
    swing_window_temperature: float = 40.0
    swing_center_urgency_temperature: float = 0.10
    swing_center_reachability_weight: float = 0.25
    swing_center_touchdown_proxy_weight: float = 0.25


@dataclass
class MpcDiagnosticsCfg:
    enabled: bool = False
    strict_failure_mask: bool = True
    emit_viewer_fields: bool = True
    emit_runtime_counters: bool = False
    profile_cuda_sync: bool = False


@dataclass
class MpcLossTermCfg:
    enabled: bool = True
    weight: float = 1.0


@dataclass
class MpcTrackingLossCfg(MpcLossTermCfg):
    vel_weight: float = 1.0
    yaw_weight: float = 0.5


@dataclass
class MpcSmoothnessLossCfg(MpcLossTermCfg):
    root_weight: float = 24.0
    foot_weight: float = 24.0


@dataclass
class MpcFootTrajectoryRegularizationLossCfg(MpcLossTermCfg):
    boundary_weight: float = 8.0
    accel_weight: float = 8.0


@dataclass
class MpcContactRegularizationLossCfg(MpcLossTermCfg):
    binary_weight: float = 1.0
    transition_weight: float = 0.5
    min_support_legs: int = 2


@dataclass
class MpcSwingWindowLossCfg(MpcLossTermCfg):
    width_prior_weight: float = 0.20
    phase_prior_weight: float = 0.10


@dataclass
class MpcDiagonalPairLossCfg(MpcLossTermCfg):
    pair_center_weight: float = 1.0
    half_cycle_weight: float = 1.0
    width_match_weight: float = 0.25


@dataclass
class MpcSwingCenterUrgencyLossCfg(MpcLossTermCfg):
    pass


@dataclass
class MpcClearanceLossCfg(MpcLossTermCfg):
    min_clearance_m: float = 0.12
    worst_deficit_weight: float = 12.0
    boundary_min_swing_prob: float = 0.40
    boundary_weight: float = 0.50


@dataclass
class MpcTouchdownSurfaceLossCfg(MpcLossTermCfg):
    ground_weight: float = 1.0
    slope_weight: float = 1.0
    support_distance_weight: float = 1.0
    support_height_weight: float = 1.0
    support_slope_weight: float = 1.0
    invalid_support_weight: float = 10.0
    max_slope: float = 0.60
    slope_sample_step_m: float = 0.03
    support_search_radius_m: float = 0.12
    support_search_step_m: float = 0.03
    support_height_tolerance_m: float = 0.03
    max_support_slope: float = 0.60


@dataclass
class MpcTouchdownSemanticLossCfg(MpcLossTermCfg):
    small_weight: float = 10.0
    large_weight: float = 50.0
    ground_ids: tuple[int, ...] = (0,)
    small_ids: tuple[int, ...] = (1,)
    large_ids: tuple[int, ...] = (2,)


@dataclass
class MpcStanceSemanticLossCfg(MpcTouchdownSemanticLossCfg):
    pass


@dataclass
class MpcSemanticContactAvoidLossCfg(MpcTouchdownSemanticLossCfg):
    activation_margin: float = 0.05
    worst_contact_weight: float = 8.0
    soft_margin_m: float = 0.18
    soft_field_weight: float = 2.0
    soft_worst_field_weight: float = 8.0


@dataclass
class MpcSemanticObstacleLossCfg(MpcLossTermCfg):
    small_weight: float = 1.0
    large_weight: float = 5.0
    body_weight: float = 1.0
    foot_weight: float = 1.0
    high_small_relative_height_m: float = 0.30
    body_stencil_radius_m: float = 0.16
    soft_margin_m: float = 0.22
    body_soft_field_weight: float = 4.0
    body_soft_worst_field_weight: float = 10.0
    foot_soft_field_weight: float = 2.0
    foot_soft_worst_field_weight: float = 8.0


@dataclass
class MpcObstacleRiskCfg(MpcLossTermCfg):
    high_small_relative_height_m: float = 0.30
    linear_corridor_width_m: float = 0.40
    linear_forward_distance_m: float = 1.0
    yaw_swept_radius_m: float = 0.60
    linear_scale_when_blocked: float = 0.5
    yaw_scale_when_blocked: float = 0.5
    linear_speed_eps: float = 1.0e-4
    yaw_speed_eps: float = 1.0e-4


@dataclass
class MpcLowSmallCrossingLossCfg(MpcLossTermCfg):
    high_small_relative_height_m: float = 0.30
    corridor_width_m: float = 0.28
    forward_distance_m: float = 1.0
    pass_margin_m: float = 0.06
    obstacle_depth_m: float = 0.24
    linear_speed_eps: float = 1.0e-4


@dataclass
class MpcTouchdownKeepoutLossCfg(MpcLossTermCfg):
    touchdown_keepout_radius_extra_m: float = 0.05
    low_small_circle_max_components: int = 8


@dataclass
class MpcSwingFootClearanceLossCfg(MpcLossTermCfg):
    swing_foot_clearance_margin_m: float = 0.02


@dataclass
class MpcHighObstacleAvoidanceLossCfg(MpcLossTermCfg):
    high_small_relative_height_m: float = 0.30
    corridor_width_m: float = 0.40
    forward_distance_m: float = 1.0
    lateral_clearance_m: float = 0.45
    longitudinal_influence_m: float = 0.55
    linear_speed_eps: float = 1.0e-4


@dataclass
class MpcLowSmallFootCrossingLossCfg(MpcLossTermCfg):
    high_small_relative_height_m: float = 0.30
    soft_margin_m: float = 0.30
    foot_weight: float = 58.0
    foot_worst_weight: float = 22.0
    touchdown_weight: float = 30.0
    touchdown_worst_weight: float = 14.0


@dataclass
class MpcLowSmallFootOverLossCfg(MpcLossTermCfg):
    high_small_relative_height_m: float = 0.30
    corridor_width_m: float = 0.30
    forward_distance_m: float = 1.0
    along_window_m: float = 0.26
    radius_m: float = 0.08
    clearance_m: float = 0.065
    xy_weight: float = 220.0
    direct_xy_weight: float = 260.0
    z_weight: float = 420.0
    ineligible_penalty: float = 1.5
    time_gate_penalty: float = 4.0
    path_curve_weight: float = 120.0
    path_curve_z_weight: float = 90.0
    path_curve_window_m: float = 0.30
    path_curve_body_yaw: bool = True
    window_weight: float = 0.0
    window_min_count: float = 3.0
    window_sigma_m: float = 0.08
    window_z_temp_m: float = 0.025
    window_step_weight: float = 0.0
    window_step_cap_m: float = 0.055
    window_accel_weight: float = 0.0
    window_accel_cap_m: float = 0.065
    window_coupled: bool = False
    linear_speed_eps: float = 1.0e-4


@dataclass
class MpcLowSmallStepcapLossCfg(MpcLossTermCfg):
    foot_boundary_weight: float = 16.0
    foot_step_worst_weight: float = 260.0
    foot_accel_weight: float = 46.0
    foot_accel_worst_weight: float = 300.0
    foot_jerk_weight: float = 36.0
    root_step_worst_weight: float = 80.0
    root_accel_weight: float = 20.0
    root_accel_worst_weight: float = 90.0
    first_foot_anchor_weight: float = 35.0
    first_foot_anchor_frames: int = 4
    high_small_relative_height_m: float = 0.30
    lateral_or_yaw_eps: float = 1.0e-4


@dataclass
class MpcHighLargeStepcapLossCfg(MpcLossTermCfg):
    foot_boundary_weight: float = 22.0
    foot_step_worst_weight: float = 520.0
    foot_accel_weight: float = 72.0
    foot_accel_worst_weight: float = 700.0
    foot_jerk_weight: float = 90.0
    root_step_worst_weight: float = 140.0
    root_accel_weight: float = 32.0
    root_accel_worst_weight: float = 180.0
    high_small_relative_height_m: float = 0.30
    lateral_or_yaw_eps: float = 1.0e-4


@dataclass
class MpcBodyCollisionLossCfg(MpcLossTermCfg):
    bottom_offset_z_m: float = -0.18
    margin_m: float = 0.04
    stencil_xy_m: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.18, 0.0),
        (-0.18, 0.0),
        (0.0, 0.10),
        (0.0, -0.10),
    )


@dataclass
class MpcLegCollisionLossCfg(MpcLossTermCfg):
    knee_margin_m: float = 0.06
    shank_margin_m: float = 0.06
    shank_sample_count: int = 2
    worst_deficit_weight: float = 16.0


@dataclass
class MpcSwingDirectionLossCfg(MpcLossTermCfg):
    pass


@dataclass
class MpcRootFootCenterLossCfg(MpcLossTermCfg):
    pass


@dataclass
class MpcRootHeightLossCfg(MpcLossTermCfg):
    pass


@dataclass
class MpcSupportPlaneLossCfg(MpcLossTermCfg):
    swing_weight: float = 0.20


@dataclass
class MpcProgressLossCfg(MpcLossTermCfg):
    min_progress_m: float = 0.02


@dataclass
class MpcKinematicsLossCfg(MpcLossTermCfg):
    joint_limit_margin_rad: float = 0.10


@dataclass
class MpcIkFkResidualLossCfg(MpcLossTermCfg):
    contact_weight: float = 2.0


@dataclass
class MpcLossesCfg:
    tracking: MpcTrackingLossCfg = field(default_factory=MpcTrackingLossCfg)
    smoothness: MpcSmoothnessLossCfg = field(default_factory=MpcSmoothnessLossCfg)
    foot_trajectory_regularization: MpcFootTrajectoryRegularizationLossCfg = field(
        default_factory=MpcFootTrajectoryRegularizationLossCfg
    )
    contact_regularization: MpcContactRegularizationLossCfg = field(default_factory=MpcContactRegularizationLossCfg)
    swing_window: MpcSwingWindowLossCfg = field(default_factory=lambda: MpcSwingWindowLossCfg(enabled=True, weight=0.8))
    diagonal_pair: MpcDiagonalPairLossCfg = field(default_factory=lambda: MpcDiagonalPairLossCfg(enabled=True, weight=1.0))
    swing_center_urgency: MpcSwingCenterUrgencyLossCfg = field(default_factory=lambda: MpcSwingCenterUrgencyLossCfg(enabled=True, weight=1.5))
    stance_ground: MpcLossTermCfg = field(default_factory=lambda: MpcLossTermCfg(enabled=True, weight=3.0))
    swing_clearance_terrain: MpcClearanceLossCfg = field(default_factory=lambda: MpcClearanceLossCfg(enabled=True, weight=12.0))
    touchdown_surface: MpcTouchdownSurfaceLossCfg = field(default_factory=lambda: MpcTouchdownSurfaceLossCfg(enabled=True, weight=2.0))
    touchdown_semantic: MpcTouchdownSemanticLossCfg = field(default_factory=lambda: MpcTouchdownSemanticLossCfg(enabled=True, weight=2.0))
    stance_semantic: MpcStanceSemanticLossCfg = field(default_factory=lambda: MpcStanceSemanticLossCfg(enabled=True, weight=2.0))
    semantic_contact_avoid: MpcSemanticContactAvoidLossCfg = field(default_factory=lambda: MpcSemanticContactAvoidLossCfg(enabled=True, weight=20.0))
    semantic_obstacle: MpcSemanticObstacleLossCfg = field(default_factory=lambda: MpcSemanticObstacleLossCfg(enabled=True, weight=1.0))
    obstacle_risk: MpcObstacleRiskCfg = field(default_factory=lambda: MpcObstacleRiskCfg(enabled=True, weight=1.0))
    low_small_crossing: MpcLowSmallCrossingLossCfg = field(default_factory=lambda: MpcLowSmallCrossingLossCfg(enabled=True, weight=8.0))
    touchdown_keepout: MpcTouchdownKeepoutLossCfg = field(default_factory=lambda: MpcTouchdownKeepoutLossCfg(enabled=True, weight=8.0))
    swing_foot_clearance: MpcSwingFootClearanceLossCfg = field(default_factory=lambda: MpcSwingFootClearanceLossCfg(enabled=True, weight=12.0))
    low_small_foot_crossing: MpcLowSmallFootCrossingLossCfg = field(
        default_factory=lambda: MpcLowSmallFootCrossingLossCfg(enabled=True, weight=1.0)
    )
    low_small_foot_over: MpcLowSmallFootOverLossCfg = field(
        default_factory=lambda: MpcLowSmallFootOverLossCfg(enabled=True, weight=1.0)
    )
    low_small_stepcap: MpcLowSmallStepcapLossCfg = field(
        default_factory=lambda: MpcLowSmallStepcapLossCfg(enabled=True, weight=1.0)
    )
    high_large_stepcap: MpcHighLargeStepcapLossCfg = field(
        default_factory=lambda: MpcHighLargeStepcapLossCfg(enabled=True, weight=1.0)
    )
    high_obstacle_avoidance: MpcHighObstacleAvoidanceLossCfg = field(
        default_factory=lambda: MpcHighObstacleAvoidanceLossCfg(enabled=True, weight=250.0)
    )
    body_collision: MpcBodyCollisionLossCfg = field(default_factory=lambda: MpcBodyCollisionLossCfg(enabled=True, weight=2.0))
    leg_collision: MpcLegCollisionLossCfg = field(default_factory=lambda: MpcLegCollisionLossCfg(enabled=True, weight=16.0))
    swing_direction: MpcSwingDirectionLossCfg = field(default_factory=lambda: MpcSwingDirectionLossCfg(enabled=True, weight=1.0))
    root_foot_center: MpcRootFootCenterLossCfg = field(default_factory=lambda: MpcRootFootCenterLossCfg(enabled=True, weight=1.0))
    root_height: MpcRootHeightLossCfg = field(default_factory=lambda: MpcRootHeightLossCfg(enabled=True, weight=3.0))
    support_plane_rp: MpcSupportPlaneLossCfg = field(default_factory=lambda: MpcSupportPlaneLossCfg(enabled=True, weight=1.0))
    kinematics: MpcKinematicsLossCfg = field(default_factory=MpcKinematicsLossCfg)
    ik_fk_residual: MpcIkFkResidualLossCfg = field(default_factory=lambda: MpcIkFkResidualLossCfg(enabled=True, weight=8.0))
    progress: MpcProgressLossCfg = field(default_factory=MpcProgressLossCfg)


@dataclass
class MpcPlannerCfg:
    runtime: MpcRuntimeCfg = field(default_factory=MpcRuntimeCfg)
    diagnostics: MpcDiagnosticsCfg = field(default_factory=MpcDiagnosticsCfg)
    losses: MpcLossesCfg = field(default_factory=MpcLossesCfg)
    profile_name: str = "train_4096"
    debug_loss_variant: str | None = None
    debug_loss_variant_cfg_applied: bool = False


def _copy_if_has(cfg, attr: str, cast, default):
    value = getattr(cfg, attr, None)
    if value is None:
        return default
    return cast(value)


def _set_if_has(cfg, attr: str, cast, target, target_attr: str) -> None:
    value = getattr(cfg, attr, None)
    if value is None:
        return
    setattr(target, target_attr, cast(value))


def _override_loss_term(task_cfg, *, prefix: str, loss_term) -> None:
    _set_if_has(task_cfg, f"{prefix}_enabled", bool, loss_term, "enabled")
    _set_if_has(task_cfg, f"{prefix}_weight", float, loss_term, "weight")


def _tuple_ints_if_has(cfg, attr: str, target, target_attr: str) -> None:
    value = getattr(cfg, attr, None)
    if value is not None:
        setattr(target, target_attr, tuple(int(v) for v in value))


def planner_cfg_from_task_cfg(task_cfg) -> MpcPlannerCfg:
    """Build planner cfg from task cfg while preserving MPC defaults."""
    cfg_obj = getattr(task_cfg, "mpc_planner_cfg", None)
    if isinstance(cfg_obj, MpcPlannerCfg):
        out = copy.deepcopy(cfg_obj)
    else:
        out = MpcPlannerCfg()
    runtime = out.runtime
    runtime.horizon_steps = _copy_if_has(task_cfg, "reference_trajectory_horizon", int, runtime.horizon_steps)
    runtime.dt = _copy_if_has(task_cfg, "plan_dt", float, runtime.dt)
    runtime.replan_interval_steps = _copy_if_has(task_cfg, "reference_replan_interval_steps", int, runtime.replan_interval_steps)
    runtime.max_stale_steps = _copy_if_has(task_cfg, "mpc_max_stale_steps", int, runtime.max_stale_steps)
    runtime.parallel_plan_batch_size = _copy_if_has(
        task_cfg, "mpc_parallel_plan_batch_size", int, runtime.parallel_plan_batch_size
    )
    runtime.optimize_steps = _copy_if_has(task_cfg, "mpc_optimize_steps", int, runtime.optimize_steps)
    runtime.lr = _copy_if_has(task_cfg, "mpc_lr", float, runtime.lr)
    runtime.contact_threshold = _copy_if_has(task_cfg, "mpc_contact_threshold", float, runtime.contact_threshold)
    runtime.randomize_replan_phase = _copy_if_has(
        task_cfg,
        "mpc_randomize_replan_phase",
        bool,
        runtime.randomize_replan_phase,
    )
    runtime.command_hard_lin_delta = _copy_if_has(task_cfg, "mpc_command_hard_lin_delta", float, runtime.command_hard_lin_delta)
    runtime.command_hard_yaw_delta = _copy_if_has(task_cfg, "mpc_command_hard_yaw_delta", float, runtime.command_hard_yaw_delta)
    runtime.command_soft_lin_delta = _copy_if_has(task_cfg, "mpc_command_soft_lin_delta", float, runtime.command_soft_lin_delta)
    runtime.command_soft_yaw_delta = _copy_if_has(task_cfg, "mpc_command_soft_yaw_delta", float, runtime.command_soft_yaw_delta)
    runtime.step_freq = _copy_if_has(task_cfg, "mpc_step_freq", float, runtime.step_freq)
    runtime.duty_factor = _copy_if_has(task_cfg, "mpc_duty_factor", float, runtime.duty_factor)
    runtime.touchdown_event_cap = _copy_if_has(task_cfg, "mpc_touchdown_event_cap", int, runtime.touchdown_event_cap)
    runtime.nominal_stride_scale = _copy_if_has(task_cfg, "mpc_nominal_stride_scale", float, runtime.nominal_stride_scale)
    runtime.nominal_swing_height_m = _copy_if_has(task_cfg, "mpc_nominal_swing_height_m", float, runtime.nominal_swing_height_m)
    runtime.nominal_yaw_stride_scale = _copy_if_has(task_cfg, "mpc_nominal_yaw_stride_scale", float, runtime.nominal_yaw_stride_scale)
    runtime.swing_window_min_width = _copy_if_has(task_cfg, "mpc_swing_window_min_width", float, runtime.swing_window_min_width)
    runtime.swing_window_max_width = _copy_if_has(task_cfg, "mpc_swing_window_max_width", float, runtime.swing_window_max_width)
    runtime.swing_window_center_scale = _copy_if_has(task_cfg, "mpc_swing_window_center_scale", float, runtime.swing_window_center_scale)
    runtime.swing_window_temperature = _copy_if_has(task_cfg, "mpc_swing_window_temperature", float, runtime.swing_window_temperature)
    runtime.swing_center_urgency_temperature = _copy_if_has(
        task_cfg,
        "mpc_swing_center_urgency_temperature",
        float,
        runtime.swing_center_urgency_temperature,
    )
    leg_phase = getattr(task_cfg, "mpc_leg_phase_offsets", None)
    if leg_phase is not None:
        runtime.leg_phase_offsets = tuple(float(v) for v in leg_phase)
    out.profile_name = str(getattr(task_cfg, "mpc_profile_name", out.profile_name))
    debug_variant = getattr(task_cfg, "mpc_debug_loss_variant", out.debug_loss_variant)
    out.debug_loss_variant = None if debug_variant in (None, "", "baseline") else str(debug_variant)
    out.diagnostics.enabled = bool(getattr(task_cfg, "mpc_diagnostics_enabled", out.diagnostics.enabled))
    _set_if_has(task_cfg, "mpc_diagnostics_strict_failure_mask", bool, out.diagnostics, "strict_failure_mask")
    _set_if_has(task_cfg, "mpc_diagnostics_emit_viewer_fields", bool, out.diagnostics, "emit_viewer_fields")
    _set_if_has(task_cfg, "mpc_diagnostics_emit_runtime_counters", bool, out.diagnostics, "emit_runtime_counters")
    _set_if_has(task_cfg, "mpc_diagnostics_profile_cuda_sync", bool, out.diagnostics, "profile_cuda_sync")

    losses = out.losses
    _override_loss_term(task_cfg, prefix="mpc_loss_tracking", loss_term=losses.tracking)
    _set_if_has(task_cfg, "mpc_loss_tracking_vel_weight", float, losses.tracking, "vel_weight")
    _set_if_has(task_cfg, "mpc_loss_tracking_yaw_weight", float, losses.tracking, "yaw_weight")
    _override_loss_term(task_cfg, prefix="mpc_loss_smoothness", loss_term=losses.smoothness)
    _set_if_has(task_cfg, "mpc_loss_smoothness_root_weight", float, losses.smoothness, "root_weight")
    _set_if_has(task_cfg, "mpc_loss_smoothness_foot_weight", float, losses.smoothness, "foot_weight")
    _override_loss_term(
        task_cfg,
        prefix="mpc_loss_foot_trajectory_regularization",
        loss_term=losses.foot_trajectory_regularization,
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_foot_trajectory_regularization_boundary_weight",
        float,
        losses.foot_trajectory_regularization,
        "boundary_weight",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_foot_trajectory_regularization_accel_weight",
        float,
        losses.foot_trajectory_regularization,
        "accel_weight",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_contact_regularization", loss_term=losses.contact_regularization)
    _set_if_has(task_cfg, "mpc_loss_contact_binary_weight", float, losses.contact_regularization, "binary_weight")
    _set_if_has(task_cfg, "mpc_loss_contact_transition_weight", float, losses.contact_regularization, "transition_weight")
    _set_if_has(task_cfg, "mpc_loss_contact_min_support_legs", int, losses.contact_regularization, "min_support_legs")
    _override_loss_term(task_cfg, prefix="mpc_loss_swing_window", loss_term=losses.swing_window)
    _override_loss_term(task_cfg, prefix="mpc_loss_diagonal_pair", loss_term=losses.diagonal_pair)
    _override_loss_term(task_cfg, prefix="mpc_loss_swing_center_urgency", loss_term=losses.swing_center_urgency)
    _override_loss_term(task_cfg, prefix="mpc_loss_stance_ground", loss_term=losses.stance_ground)
    _override_loss_term(task_cfg, prefix="mpc_loss_swing_clearance_terrain", loss_term=losses.swing_clearance_terrain)
    _set_if_has(
        task_cfg,
        "mpc_loss_swing_clearance_terrain_min_clearance_m",
        float,
        losses.swing_clearance_terrain,
        "min_clearance_m",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_swing_clearance_terrain_worst_deficit_weight",
        float,
        losses.swing_clearance_terrain,
        "worst_deficit_weight",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_swing_clearance_terrain_boundary_min_swing_prob",
        float,
        losses.swing_clearance_terrain,
        "boundary_min_swing_prob",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_swing_clearance_terrain_boundary_weight",
        float,
        losses.swing_clearance_terrain,
        "boundary_weight",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_touchdown_surface", loss_term=losses.touchdown_surface)
    _set_if_has(task_cfg, "mpc_loss_touchdown_surface_max_slope", float, losses.touchdown_surface, "max_slope")
    _override_loss_term(task_cfg, prefix="mpc_loss_touchdown_semantic", loss_term=losses.touchdown_semantic)
    _set_if_has(task_cfg, "mpc_loss_touchdown_semantic_small_weight", float, losses.touchdown_semantic, "small_weight")
    _set_if_has(task_cfg, "mpc_loss_touchdown_semantic_large_weight", float, losses.touchdown_semantic, "large_weight")
    _tuple_ints_if_has(task_cfg, "mpc_loss_touchdown_semantic_ground_ids", losses.touchdown_semantic, "ground_ids")
    _tuple_ints_if_has(task_cfg, "mpc_loss_touchdown_semantic_small_ids", losses.touchdown_semantic, "small_ids")
    _tuple_ints_if_has(task_cfg, "mpc_loss_touchdown_semantic_large_ids", losses.touchdown_semantic, "large_ids")
    _override_loss_term(task_cfg, prefix="mpc_loss_stance_semantic", loss_term=losses.stance_semantic)
    _set_if_has(task_cfg, "mpc_loss_stance_semantic_small_weight", float, losses.stance_semantic, "small_weight")
    _set_if_has(task_cfg, "mpc_loss_stance_semantic_large_weight", float, losses.stance_semantic, "large_weight")
    _tuple_ints_if_has(task_cfg, "mpc_loss_stance_semantic_ground_ids", losses.stance_semantic, "ground_ids")
    _tuple_ints_if_has(task_cfg, "mpc_loss_stance_semantic_small_ids", losses.stance_semantic, "small_ids")
    _tuple_ints_if_has(task_cfg, "mpc_loss_stance_semantic_large_ids", losses.stance_semantic, "large_ids")
    _override_loss_term(task_cfg, prefix="mpc_loss_semantic_contact_avoid", loss_term=losses.semantic_contact_avoid)
    _set_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_small_weight", float, losses.semantic_contact_avoid, "small_weight")
    _set_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_large_weight", float, losses.semantic_contact_avoid, "large_weight")
    _set_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_activation_margin", float, losses.semantic_contact_avoid, "activation_margin")
    _set_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_worst_contact_weight", float, losses.semantic_contact_avoid, "worst_contact_weight")
    _set_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_soft_margin_m", float, losses.semantic_contact_avoid, "soft_margin_m")
    _set_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_soft_field_weight", float, losses.semantic_contact_avoid, "soft_field_weight")
    _set_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_soft_worst_field_weight", float, losses.semantic_contact_avoid, "soft_worst_field_weight")
    _tuple_ints_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_ground_ids", losses.semantic_contact_avoid, "ground_ids")
    _tuple_ints_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_small_ids", losses.semantic_contact_avoid, "small_ids")
    _tuple_ints_if_has(task_cfg, "mpc_loss_semantic_contact_avoid_large_ids", losses.semantic_contact_avoid, "large_ids")
    _override_loss_term(task_cfg, prefix="mpc_loss_semantic_obstacle", loss_term=losses.semantic_obstacle)
    _set_if_has(task_cfg, "mpc_loss_semantic_obstacle_small_weight", float, losses.semantic_obstacle, "small_weight")
    _set_if_has(task_cfg, "mpc_loss_semantic_obstacle_large_weight", float, losses.semantic_obstacle, "large_weight")
    _set_if_has(task_cfg, "mpc_loss_semantic_obstacle_body_weight", float, losses.semantic_obstacle, "body_weight")
    _set_if_has(task_cfg, "mpc_loss_semantic_obstacle_foot_weight", float, losses.semantic_obstacle, "foot_weight")
    _set_if_has(
        task_cfg,
        "mpc_loss_semantic_obstacle_high_small_relative_height_m",
        float,
        losses.semantic_obstacle,
        "high_small_relative_height_m",
    )
    _set_if_has(task_cfg, "mpc_loss_semantic_obstacle_body_stencil_radius_m", float, losses.semantic_obstacle, "body_stencil_radius_m")
    _set_if_has(task_cfg, "mpc_loss_semantic_obstacle_soft_margin_m", float, losses.semantic_obstacle, "soft_margin_m")
    _set_if_has(
        task_cfg,
        "mpc_loss_semantic_obstacle_body_soft_field_weight",
        float,
        losses.semantic_obstacle,
        "body_soft_field_weight",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_semantic_obstacle_body_soft_worst_field_weight",
        float,
        losses.semantic_obstacle,
        "body_soft_worst_field_weight",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_semantic_obstacle_foot_soft_field_weight",
        float,
        losses.semantic_obstacle,
        "foot_soft_field_weight",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_semantic_obstacle_foot_soft_worst_field_weight",
        float,
        losses.semantic_obstacle,
        "foot_soft_worst_field_weight",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_obstacle_risk", loss_term=losses.obstacle_risk)
    _set_if_has(task_cfg, "mpc_loss_obstacle_risk_high_small_relative_height_m", float, losses.obstacle_risk, "high_small_relative_height_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_risk_linear_corridor_width_m", float, losses.obstacle_risk, "linear_corridor_width_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_risk_linear_forward_distance_m", float, losses.obstacle_risk, "linear_forward_distance_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_risk_yaw_swept_radius_m", float, losses.obstacle_risk, "yaw_swept_radius_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_risk_linear_scale_when_blocked", float, losses.obstacle_risk, "linear_scale_when_blocked")
    _set_if_has(task_cfg, "mpc_loss_obstacle_risk_yaw_scale_when_blocked", float, losses.obstacle_risk, "yaw_scale_when_blocked")
    _set_if_has(task_cfg, "mpc_loss_obstacle_risk_linear_speed_eps", float, losses.obstacle_risk, "linear_speed_eps")
    _set_if_has(task_cfg, "mpc_loss_obstacle_risk_yaw_speed_eps", float, losses.obstacle_risk, "yaw_speed_eps")
    _override_loss_term(task_cfg, prefix="mpc_loss_low_small_crossing", loss_term=losses.low_small_crossing)
    _set_if_has(task_cfg, "mpc_loss_low_small_crossing_high_small_relative_height_m", float, losses.low_small_crossing, "high_small_relative_height_m")
    _set_if_has(task_cfg, "mpc_loss_low_small_crossing_corridor_width_m", float, losses.low_small_crossing, "corridor_width_m")
    _set_if_has(task_cfg, "mpc_loss_low_small_crossing_forward_distance_m", float, losses.low_small_crossing, "forward_distance_m")
    _set_if_has(task_cfg, "mpc_loss_low_small_crossing_pass_margin_m", float, losses.low_small_crossing, "pass_margin_m")
    _set_if_has(task_cfg, "mpc_loss_low_small_crossing_obstacle_depth_m", float, losses.low_small_crossing, "obstacle_depth_m")
    _set_if_has(task_cfg, "mpc_loss_low_small_crossing_linear_speed_eps", float, losses.low_small_crossing, "linear_speed_eps")
    _override_loss_term(task_cfg, prefix="mpc_loss_touchdown_keepout", loss_term=losses.touchdown_keepout)
    _set_if_has(
        task_cfg,
        "mpc_loss_touchdown_keepout_radius_extra_m",
        float,
        losses.touchdown_keepout,
        "touchdown_keepout_radius_extra_m",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_touchdown_keepout_low_small_circle_max_components",
        int,
        losses.touchdown_keepout,
        "low_small_circle_max_components",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_swing_foot_clearance", loss_term=losses.swing_foot_clearance)
    _set_if_has(
        task_cfg,
        "mpc_loss_swing_foot_clearance_margin_m",
        float,
        losses.swing_foot_clearance,
        "swing_foot_clearance_margin_m",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_low_small_foot_crossing", loss_term=losses.low_small_foot_crossing)
    _set_if_has(
        task_cfg,
        "mpc_loss_low_small_foot_crossing_high_small_relative_height_m",
        float,
        losses.low_small_foot_crossing,
        "high_small_relative_height_m",
    )
    _set_if_has(task_cfg, "mpc_loss_low_small_foot_crossing_soft_margin_m", float, losses.low_small_foot_crossing, "soft_margin_m")
    _set_if_has(task_cfg, "mpc_loss_low_small_foot_crossing_foot_weight", float, losses.low_small_foot_crossing, "foot_weight")
    _set_if_has(
        task_cfg,
        "mpc_loss_low_small_foot_crossing_foot_worst_weight",
        float,
        losses.low_small_foot_crossing,
        "foot_worst_weight",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_low_small_foot_crossing_touchdown_weight",
        float,
        losses.low_small_foot_crossing,
        "touchdown_weight",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_low_small_foot_crossing_touchdown_worst_weight",
        float,
        losses.low_small_foot_crossing,
        "touchdown_worst_weight",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_low_small_stepcap", loss_term=losses.low_small_stepcap)
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_foot_boundary_weight", float, losses.low_small_stepcap, "foot_boundary_weight")
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_foot_step_worst_weight", float, losses.low_small_stepcap, "foot_step_worst_weight")
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_foot_accel_weight", float, losses.low_small_stepcap, "foot_accel_weight")
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_foot_accel_worst_weight", float, losses.low_small_stepcap, "foot_accel_worst_weight")
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_foot_jerk_weight", float, losses.low_small_stepcap, "foot_jerk_weight")
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_root_step_worst_weight", float, losses.low_small_stepcap, "root_step_worst_weight")
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_root_accel_weight", float, losses.low_small_stepcap, "root_accel_weight")
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_root_accel_worst_weight", float, losses.low_small_stepcap, "root_accel_worst_weight")
    _set_if_has(
        task_cfg,
        "mpc_loss_low_small_stepcap_first_foot_anchor_weight",
        float,
        losses.low_small_stepcap,
        "first_foot_anchor_weight",
    )
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_first_foot_anchor_frames", int, losses.low_small_stepcap, "first_foot_anchor_frames")
    _set_if_has(
        task_cfg,
        "mpc_loss_low_small_stepcap_high_small_relative_height_m",
        float,
        losses.low_small_stepcap,
        "high_small_relative_height_m",
    )
    _set_if_has(task_cfg, "mpc_loss_low_small_stepcap_lateral_or_yaw_eps", float, losses.low_small_stepcap, "lateral_or_yaw_eps")
    _override_loss_term(task_cfg, prefix="mpc_loss_high_obstacle_avoidance", loss_term=losses.high_obstacle_avoidance)
    _set_if_has(
        task_cfg,
        "mpc_loss_high_obstacle_avoidance_high_small_relative_height_m",
        float,
        losses.high_obstacle_avoidance,
        "high_small_relative_height_m",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_high_obstacle_avoidance_corridor_width_m",
        float,
        losses.high_obstacle_avoidance,
        "corridor_width_m",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_high_obstacle_avoidance_forward_distance_m",
        float,
        losses.high_obstacle_avoidance,
        "forward_distance_m",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_high_obstacle_avoidance_lateral_clearance_m",
        float,
        losses.high_obstacle_avoidance,
        "lateral_clearance_m",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_high_obstacle_avoidance_longitudinal_influence_m",
        float,
        losses.high_obstacle_avoidance,
        "longitudinal_influence_m",
    )
    _set_if_has(
        task_cfg,
        "mpc_loss_high_obstacle_avoidance_linear_speed_eps",
        float,
        losses.high_obstacle_avoidance,
        "linear_speed_eps",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_body_collision", loss_term=losses.body_collision)
    _set_if_has(task_cfg, "mpc_loss_body_collision_bottom_offset_z_m", float, losses.body_collision, "bottom_offset_z_m")
    _set_if_has(task_cfg, "mpc_loss_body_collision_margin_m", float, losses.body_collision, "margin_m")
    _override_loss_term(task_cfg, prefix="mpc_loss_leg_collision", loss_term=losses.leg_collision)
    _set_if_has(task_cfg, "mpc_loss_leg_collision_knee_margin_m", float, losses.leg_collision, "knee_margin_m")
    _set_if_has(task_cfg, "mpc_loss_leg_collision_shank_margin_m", float, losses.leg_collision, "shank_margin_m")
    _set_if_has(task_cfg, "mpc_loss_leg_collision_shank_sample_count", int, losses.leg_collision, "shank_sample_count")
    _set_if_has(task_cfg, "mpc_loss_leg_collision_worst_deficit_weight", float, losses.leg_collision, "worst_deficit_weight")
    _override_loss_term(task_cfg, prefix="mpc_loss_swing_direction", loss_term=losses.swing_direction)
    _override_loss_term(task_cfg, prefix="mpc_loss_root_foot_center", loss_term=losses.root_foot_center)
    _override_loss_term(task_cfg, prefix="mpc_loss_root_height", loss_term=losses.root_height)
    _override_loss_term(task_cfg, prefix="mpc_loss_support_plane_rp", loss_term=losses.support_plane_rp)
    _set_if_has(task_cfg, "mpc_loss_support_plane_rp_swing_weight", float, losses.support_plane_rp, "swing_weight")
    _override_loss_term(task_cfg, prefix="mpc_loss_kinematics", loss_term=losses.kinematics)
    _set_if_has(task_cfg, "mpc_loss_kinematics_joint_limit_margin_rad", float, losses.kinematics, "joint_limit_margin_rad")
    _override_loss_term(task_cfg, prefix="mpc_loss_ik_fk_residual", loss_term=losses.ik_fk_residual)
    _set_if_has(task_cfg, "mpc_loss_ik_fk_residual_contact_weight", float, losses.ik_fk_residual, "contact_weight")
    _override_loss_term(task_cfg, prefix="mpc_loss_progress", loss_term=losses.progress)
    _set_if_has(task_cfg, "mpc_loss_progress_min_progress_m", float, losses.progress, "min_progress_m")
    return out


def validate_mpc_config(cfg: MpcPlannerCfg) -> None:
    if cfg.runtime.horizon_steps <= 1:
        raise ValueError("runtime.horizon_steps must be > 1")
    if cfg.runtime.dt <= 0.0:
        raise ValueError("runtime.dt must be positive")
    if cfg.runtime.optimize_steps < 0:
        raise ValueError("runtime.optimize_steps must be >= 0")
    if cfg.runtime.parallel_plan_batch_size <= 0:
        raise ValueError("runtime.parallel_plan_batch_size must be positive")
    if cfg.runtime.touchdown_event_cap <= 0:
        raise ValueError("runtime.touchdown_event_cap must be positive")
    if len(cfg.runtime.leg_phase_offsets) != 4:
        raise ValueError("runtime.leg_phase_offsets must contain 4 phase offsets")
    if not 0.0 < cfg.runtime.duty_factor < 1.0:
        raise ValueError("runtime.duty_factor must be in (0, 1)")
    if cfg.runtime.swing_window_min_width <= 0.0:
        raise ValueError("runtime.swing_window_min_width must be positive")
    if cfg.runtime.swing_window_max_width <= cfg.runtime.swing_window_min_width:
        raise ValueError("runtime.swing_window_max_width must exceed swing_window_min_width")
    if cfg.runtime.swing_window_center_scale <= 0.0:
        raise ValueError("runtime.swing_window_center_scale must be positive")
    if cfg.runtime.swing_window_temperature <= 0.0:
        raise ValueError("runtime.swing_window_temperature must be positive")
    if cfg.runtime.swing_center_urgency_temperature <= 0.0:
        raise ValueError("runtime.swing_center_urgency_temperature must be positive")


__all__ = [
    "MpcBodyCollisionLossCfg",
    "MpcDiagnosticsCfg",
    "MpcLegCollisionLossCfg",
    "MpcLossesCfg",
    "MpcLowSmallCrossingLossCfg",
    "MpcLowSmallFootCrossingLossCfg",
    "MpcLowSmallStepcapLossCfg",
    "MpcObstacleRiskCfg",
    "MpcPlannerCfg",
    "MpcRuntimeCfg",
    "MpcSemanticContactAvoidLossCfg",
    "MpcStanceSemanticLossCfg",
    "MpcSwingFootClearanceLossCfg",
    "MpcTouchdownKeepoutLossCfg",
    "planner_cfg_from_task_cfg",
    "validate_mpc_config",
]
