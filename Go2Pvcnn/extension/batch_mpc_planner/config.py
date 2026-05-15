"""Config contracts for the batch MPC backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class MpcRuntimeCfg:
    horizon_steps: int = 25
    dt: float = 0.02
    optimize_steps: int = 24
    lr: float = 1e-2
    optimizer: str = "adam"
    grad_clip_norm: float = 10.0
    contact_threshold: float = 0.55
    replan_interval_steps: int = 50
    max_stale_steps: int = 100
    warm_start_from_previous_plan: bool = True
    detach_warm_start: bool = True
    detach_cache_on_write: bool = True
    heavy_loss_stride: int = 2
    heavy_loss_enable_from_iter: int = 8
    selection_mode: str = "fixed_topk_priority"
    max_dirty_envs_per_step: int = 256
    target_dirty_ratio: float = 0.05
    randomize_replan_phase: bool = True
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
    nominal_swing_height_m: float = 0.10
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
    root_weight: float = 1.0
    foot_weight: float = 1.0


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
    min_clearance_m: float = 0.05


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


@dataclass
class MpcSemanticObstacleLossCfg(MpcLossTermCfg):
    small_weight: float = 1.0
    large_weight: float = 5.0
    body_weight: float = 1.0
    foot_weight: float = 1.0
    body_stencil_radius_m: float = 0.16


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
    contact_regularization: MpcContactRegularizationLossCfg = field(default_factory=MpcContactRegularizationLossCfg)
    swing_window: MpcSwingWindowLossCfg = field(default_factory=lambda: MpcSwingWindowLossCfg(enabled=True, weight=0.8))
    diagonal_pair: MpcDiagonalPairLossCfg = field(default_factory=lambda: MpcDiagonalPairLossCfg(enabled=True, weight=1.0))
    swing_center_urgency: MpcSwingCenterUrgencyLossCfg = field(default_factory=lambda: MpcSwingCenterUrgencyLossCfg(enabled=True, weight=1.0))
    stance_ground: MpcLossTermCfg = field(default_factory=lambda: MpcLossTermCfg(enabled=True, weight=3.0))
    swing_clearance_terrain: MpcClearanceLossCfg = field(default_factory=lambda: MpcClearanceLossCfg(enabled=True, weight=2.0))
    touchdown_surface: MpcTouchdownSurfaceLossCfg = field(default_factory=lambda: MpcTouchdownSurfaceLossCfg(enabled=True, weight=2.0))
    touchdown_semantic: MpcTouchdownSemanticLossCfg = field(default_factory=lambda: MpcTouchdownSemanticLossCfg(enabled=True, weight=2.0))
    semantic_obstacle: MpcSemanticObstacleLossCfg = field(default_factory=lambda: MpcSemanticObstacleLossCfg(enabled=True, weight=1.0))
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
    runtime.max_dirty_envs_per_step = _copy_if_has(task_cfg, "mpc_max_dirty_envs_per_step", int, runtime.max_dirty_envs_per_step)
    runtime.optimize_steps = _copy_if_has(task_cfg, "mpc_optimize_steps", int, runtime.optimize_steps)
    runtime.lr = _copy_if_has(task_cfg, "mpc_lr", float, runtime.lr)
    runtime.contact_threshold = _copy_if_has(task_cfg, "mpc_contact_threshold", float, runtime.contact_threshold)
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
    _override_loss_term(task_cfg, prefix="mpc_loss_touchdown_surface", loss_term=losses.touchdown_surface)
    _set_if_has(task_cfg, "mpc_loss_touchdown_surface_max_slope", float, losses.touchdown_surface, "max_slope")
    _override_loss_term(task_cfg, prefix="mpc_loss_touchdown_semantic", loss_term=losses.touchdown_semantic)
    _set_if_has(task_cfg, "mpc_loss_touchdown_semantic_small_weight", float, losses.touchdown_semantic, "small_weight")
    _set_if_has(task_cfg, "mpc_loss_touchdown_semantic_large_weight", float, losses.touchdown_semantic, "large_weight")
    _override_loss_term(task_cfg, prefix="mpc_loss_semantic_obstacle", loss_term=losses.semantic_obstacle)
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
    if cfg.runtime.max_dirty_envs_per_step <= 0:
        raise ValueError("runtime.max_dirty_envs_per_step must be positive")
    if cfg.runtime.selection_mode not in ("fixed_topk_priority",):
        raise ValueError("runtime.selection_mode must be 'fixed_topk_priority'")
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
    "MpcDiagnosticsCfg",
    "MpcLossesCfg",
    "MpcPlannerCfg",
    "MpcRuntimeCfg",
    "planner_cfg_from_task_cfg",
    "validate_mpc_config",
]
