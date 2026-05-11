"""Config contracts for the batch MPC backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class MpcRuntimeCfg:
    horizon_steps: int = 80
    dt: float = 0.02
    optimize_steps: int = 24
    lr: float = 1e-2
    optimizer: str = "adam"
    grad_clip_norm: float = 10.0
    contact_temperature: float = 0.20
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
    duty_factor: float = 0.55
    leg_phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
    touchdown_event_cap: int = 2
    nominal_stride_scale: float = 0.9
    nominal_max_stride_m: float = 0.18
    nominal_swing_height_m: float = 0.045
    nominal_yaw_stride_scale: float = 1.0
    nominal_backward_stride_scale: float = 0.70
    nominal_yaw_stride_atten: float = 0.35


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
class MpcStanceSlipLossCfg(MpcLossTermCfg):
    slip_tolerance_m_per_step: float = 0.004


@dataclass
class MpcSwingStrideLossCfg(MpcLossTermCfg):
    min_swing_span_m: float = 0.045
    command_speed_deadzone_mps: float = 0.08


@dataclass
class MpcRootFrameDriftLossCfg(MpcLossTermCfg):
    min_rel_m: float = 0.18
    max_rel_m: float = 0.68


@dataclass
class MpcRootFrameFollowLossCfg(MpcLossTermCfg):
    rel_change_tolerance_m_per_step: float = 0.025


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
    min_support_legs: int = 1
    max_airborne_steps: int = 10


@dataclass
class MpcContactScheduleLossCfg(MpcLossTermCfg):
    min_support_prob: float = 0.35


@dataclass
class MpcClearanceLossCfg(MpcLossTermCfg):
    min_clearance_m: float = 0.04


@dataclass
class MpcObstacleLossCfg(MpcLossTermCfg):
    body_margin_m: float = 0.04
    foot_margin_m: float = 0.04
    repulsion_radius_m: float = 0.20
    sample_count: int = 9


@dataclass
class MpcProgressLossCfg(MpcLossTermCfg):
    min_progress_m: float = 0.02


@dataclass
class MpcKinematicsLossCfg(MpcLossTermCfg):
    joint_limit_rad: float = 2.6
    joint_limit_margin_rad: float = 0.10


@dataclass
class MpcLossesCfg:
    tracking: MpcTrackingLossCfg = field(default_factory=MpcTrackingLossCfg)
    smoothness: MpcSmoothnessLossCfg = field(default_factory=MpcSmoothnessLossCfg)
    contact_regularization: MpcContactRegularizationLossCfg = field(default_factory=MpcContactRegularizationLossCfg)
    contact_schedule: MpcContactScheduleLossCfg = field(default_factory=lambda: MpcContactScheduleLossCfg(enabled=True, weight=0.8))
    stance_slip: MpcStanceSlipLossCfg = field(default_factory=lambda: MpcStanceSlipLossCfg(enabled=True, weight=0.9))
    swing_stride: MpcSwingStrideLossCfg = field(default_factory=lambda: MpcSwingStrideLossCfg(enabled=True, weight=1.1))
    root_frame_drift: MpcRootFrameDriftLossCfg = field(default_factory=lambda: MpcRootFrameDriftLossCfg(enabled=True, weight=1.4))
    root_frame_follow: MpcRootFrameFollowLossCfg = field(default_factory=lambda: MpcRootFrameFollowLossCfg(enabled=True, weight=0.8))
    swing_clearance: MpcClearanceLossCfg = field(default_factory=MpcClearanceLossCfg)
    terrain_clearance: MpcClearanceLossCfg = field(default_factory=lambda: MpcClearanceLossCfg(enabled=True, weight=0.8))
    obstacle_small: MpcObstacleLossCfg = field(default_factory=MpcObstacleLossCfg)
    obstacle_large: MpcObstacleLossCfg = field(default_factory=lambda: MpcObstacleLossCfg(enabled=True, weight=1.2))
    touchdown_support: MpcLossTermCfg = field(default_factory=lambda: MpcLossTermCfg(enabled=True, weight=0.25))
    kinematics: MpcKinematicsLossCfg = field(default_factory=MpcKinematicsLossCfg)
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
    runtime.replan_interval_steps = _copy_if_has(
        task_cfg,
        "reference_replan_interval_steps",
        int,
        runtime.replan_interval_steps,
    )
    runtime.max_stale_steps = _copy_if_has(task_cfg, "mpc_max_stale_steps", int, runtime.max_stale_steps)
    runtime.max_dirty_envs_per_step = _copy_if_has(task_cfg, "mpc_max_dirty_envs_per_step", int, runtime.max_dirty_envs_per_step)
    runtime.optimize_steps = _copy_if_has(task_cfg, "mpc_optimize_steps", int, runtime.optimize_steps)
    runtime.lr = _copy_if_has(task_cfg, "mpc_lr", float, runtime.lr)
    runtime.contact_threshold = _copy_if_has(task_cfg, "mpc_contact_threshold", float, runtime.contact_threshold)
    runtime.contact_temperature = _copy_if_has(task_cfg, "mpc_contact_temperature", float, runtime.contact_temperature)
    runtime.command_hard_lin_delta = _copy_if_has(task_cfg, "mpc_command_hard_lin_delta", float, runtime.command_hard_lin_delta)
    runtime.command_hard_yaw_delta = _copy_if_has(task_cfg, "mpc_command_hard_yaw_delta", float, runtime.command_hard_yaw_delta)
    runtime.command_soft_lin_delta = _copy_if_has(task_cfg, "mpc_command_soft_lin_delta", float, runtime.command_soft_lin_delta)
    runtime.command_soft_yaw_delta = _copy_if_has(task_cfg, "mpc_command_soft_yaw_delta", float, runtime.command_soft_yaw_delta)
    runtime.step_freq = _copy_if_has(task_cfg, "mpc_step_freq", float, runtime.step_freq)
    runtime.duty_factor = _copy_if_has(task_cfg, "mpc_duty_factor", float, runtime.duty_factor)
    runtime.touchdown_event_cap = _copy_if_has(task_cfg, "mpc_touchdown_event_cap", int, runtime.touchdown_event_cap)
    runtime.nominal_stride_scale = _copy_if_has(task_cfg, "mpc_nominal_stride_scale", float, runtime.nominal_stride_scale)
    runtime.nominal_max_stride_m = _copy_if_has(task_cfg, "mpc_nominal_max_stride_m", float, runtime.nominal_max_stride_m)
    runtime.nominal_swing_height_m = _copy_if_has(task_cfg, "mpc_nominal_swing_height_m", float, runtime.nominal_swing_height_m)
    runtime.nominal_yaw_stride_scale = _copy_if_has(
        task_cfg,
        "mpc_nominal_yaw_stride_scale",
        float,
        runtime.nominal_yaw_stride_scale,
    )
    runtime.nominal_backward_stride_scale = _copy_if_has(
        task_cfg,
        "mpc_nominal_backward_stride_scale",
        float,
        runtime.nominal_backward_stride_scale,
    )
    runtime.nominal_yaw_stride_atten = _copy_if_has(
        task_cfg,
        "mpc_nominal_yaw_stride_atten",
        float,
        runtime.nominal_yaw_stride_atten,
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
    _set_if_has(task_cfg, "mpc_loss_contact_max_airborne_steps", int, losses.contact_regularization, "max_airborne_steps")
    _override_loss_term(task_cfg, prefix="mpc_loss_contact_schedule", loss_term=losses.contact_schedule)
    _set_if_has(task_cfg, "mpc_loss_contact_schedule_min_support_prob", float, losses.contact_schedule, "min_support_prob")

    _override_loss_term(task_cfg, prefix="mpc_loss_stance_slip", loss_term=losses.stance_slip)
    _set_if_has(task_cfg, "mpc_loss_stance_slip_tolerance_m_per_step", float, losses.stance_slip, "slip_tolerance_m_per_step")
    _override_loss_term(task_cfg, prefix="mpc_loss_swing_stride", loss_term=losses.swing_stride)
    _set_if_has(task_cfg, "mpc_loss_swing_stride_min_swing_span_m", float, losses.swing_stride, "min_swing_span_m")
    _set_if_has(
        task_cfg,
        "mpc_loss_swing_stride_command_speed_deadzone_mps",
        float,
        losses.swing_stride,
        "command_speed_deadzone_mps",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_root_frame_drift", loss_term=losses.root_frame_drift)
    _set_if_has(task_cfg, "mpc_loss_root_frame_drift_min_rel_m", float, losses.root_frame_drift, "min_rel_m")
    _set_if_has(task_cfg, "mpc_loss_root_frame_drift_max_rel_m", float, losses.root_frame_drift, "max_rel_m")
    _override_loss_term(task_cfg, prefix="mpc_loss_root_frame_follow", loss_term=losses.root_frame_follow)
    _set_if_has(
        task_cfg,
        "mpc_loss_root_frame_follow_rel_change_tolerance_m_per_step",
        float,
        losses.root_frame_follow,
        "rel_change_tolerance_m_per_step",
    )
    _override_loss_term(task_cfg, prefix="mpc_loss_swing_clearance", loss_term=losses.swing_clearance)
    _set_if_has(task_cfg, "mpc_loss_swing_clearance_min_clearance_m", float, losses.swing_clearance, "min_clearance_m")
    _override_loss_term(task_cfg, prefix="mpc_loss_terrain_clearance", loss_term=losses.terrain_clearance)
    _set_if_has(task_cfg, "mpc_loss_terrain_clearance_min_clearance_m", float, losses.terrain_clearance, "min_clearance_m")

    _override_loss_term(task_cfg, prefix="mpc_loss_obstacle_small", loss_term=losses.obstacle_small)
    _set_if_has(task_cfg, "mpc_loss_obstacle_small_body_margin_m", float, losses.obstacle_small, "body_margin_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_small_foot_margin_m", float, losses.obstacle_small, "foot_margin_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_small_repulsion_radius_m", float, losses.obstacle_small, "repulsion_radius_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_small_sample_count", int, losses.obstacle_small, "sample_count")

    _override_loss_term(task_cfg, prefix="mpc_loss_obstacle_large", loss_term=losses.obstacle_large)
    _set_if_has(task_cfg, "mpc_loss_obstacle_large_body_margin_m", float, losses.obstacle_large, "body_margin_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_large_foot_margin_m", float, losses.obstacle_large, "foot_margin_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_large_repulsion_radius_m", float, losses.obstacle_large, "repulsion_radius_m")
    _set_if_has(task_cfg, "mpc_loss_obstacle_large_sample_count", int, losses.obstacle_large, "sample_count")

    _override_loss_term(task_cfg, prefix="mpc_loss_touchdown_support", loss_term=losses.touchdown_support)
    _override_loss_term(task_cfg, prefix="mpc_loss_kinematics", loss_term=losses.kinematics)
    _set_if_has(task_cfg, "mpc_loss_kinematics_joint_limit_rad", float, losses.kinematics, "joint_limit_rad")
    _set_if_has(task_cfg, "mpc_loss_kinematics_joint_limit_margin_rad", float, losses.kinematics, "joint_limit_margin_rad")
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
    if cfg.runtime.contact_temperature <= 0.0:
        raise ValueError("runtime.contact_temperature must be positive")
    if cfg.runtime.max_dirty_envs_per_step <= 0:
        raise ValueError("runtime.max_dirty_envs_per_step must be positive")
    if cfg.runtime.selection_mode not in ("fixed_topk_priority",):
        raise ValueError("runtime.selection_mode must be 'fixed_topk_priority'")
    if cfg.runtime.touchdown_event_cap <= 0:
        raise ValueError("runtime.touchdown_event_cap must be positive")
    if len(cfg.runtime.leg_phase_offsets) != 4:
        raise ValueError("runtime.leg_phase_offsets must contain 4 phase offsets")


__all__ = [
    "MpcDiagnosticsCfg",
    "MpcLossesCfg",
    "MpcPlannerCfg",
    "MpcRuntimeCfg",
    "planner_cfg_from_task_cfg",
    "validate_mpc_config",
]
