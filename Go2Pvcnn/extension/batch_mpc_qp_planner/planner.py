"""Initial MPC-QP planner backend.

The first version keeps the backend isolated and cache-compatible while the
full constrained QP solver is developed behind the new package boundary.
"""

from __future__ import annotations

import copy
import time

import torch
from torch import Tensor

from extension.batch_mpc_planner.planner import plan_segment
from extension.batch_mpc_planner.types import MpcPlannerResult, MpcPlannerTerrain, MpcRobotState

from .config import MpcQpPlannerCfg, validate_mpc_qp_config
from .constraints import repair_touchdown_semantic_keepout, safety_diagnostics
from .continuous import build_controls_from_nominal, decode_controls_to_result
from .gait import alternating_diagonal_gait_masks
from .losses import continuous_loss_diagnostics
from .solver import continuous_qp_update
from .qp import (
    apply_fk_body_leg_root_lift,
    apply_fk_body_leg_xy_repair,
    apply_fk_shank_clearance_lift,
    apply_safety_qp_step,
    suppress_fk_low_small_contact_state,
)


def _diagnostic_vector(batch: int, value: int | float, *, like: Tensor) -> Tensor:
    return torch.full((batch,), float(value), dtype=like.dtype, device=like.device)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def plan_segment_qp(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    *,
    cfg: MpcQpPlannerCfg,
) -> MpcPlannerResult:
    """Plan one QP backend horizon for a batch of environments.

    This bootstraps ``mpc_qp`` as an opt-in backend with explicit QP iteration
    diagnostics. It uses a zero-Adam nominal pass as the safe initial trajectory
    until the constrained QP solve is filled in.
    """

    total_t0 = time.perf_counter()
    validate_mpc_qp_config(cfg)
    base_cfg = copy.deepcopy(cfg)
    base_cfg.runtime.optimize_steps = 0
    nominal_t0 = time.perf_counter()
    result = plan_segment(terrain, state, command, cfg=base_cfg)
    nominal_ms = _elapsed_ms(nominal_t0)
    batch = int(result.root_pos.shape[0])
    configured = int(cfg.runtime.qp_iterations)
    qp_diagnostics: dict[str, Tensor] = {}
    executed = 0
    solve_t0 = time.perf_counter()
    continuous_enabled = bool(getattr(cfg.runtime, "continuous_trajectory_enabled", True))
    if continuous_enabled:
        fixed_gait = alternating_diagonal_gait_masks(
            batch=batch,
            horizon=int(result.root_pos.shape[1]),
            device=result.root_pos.device,
        )
        controls = build_controls_from_nominal(
            result,
            terrain,
            start_tangent_scale=float(getattr(cfg.runtime, "continuous_start_tangent_scale", 1.0)),
        )
        sample_count = int(getattr(cfg.runtime, "continuous_bezier_sample_count", 0))
        if sample_count <= 0:
            sample_count = int(result.root_pos.shape[1])
        for _ in range(configured):
            controls, step_diagnostics = continuous_qp_update(
                controls,
                terrain,
                cfg,
                command=command,
                root_pos=result.root_pos,
                root_rpy=result.root_rpy,
                contact_state=fixed_gait.stance_mask,
            )
            for name, value in step_diagnostics.items():
                if name.endswith("_count"):
                    qp_diagnostics[name] = qp_diagnostics.get(name, torch.zeros_like(value)) + value
                elif name.endswith("_after_max"):
                    qp_diagnostics[name] = value
                elif name.endswith("_scale_min"):
                    qp_diagnostics[name] = torch.minimum(qp_diagnostics.get(name, value), value)
                elif name.endswith("_reduces_progress"):
                    qp_diagnostics[name] = torch.maximum(qp_diagnostics.get(name, value), value)
                elif name.endswith("_max") or name.endswith("_before_max"):
                    qp_diagnostics[name] = torch.maximum(qp_diagnostics.get(name, value), value)
                else:
                    qp_diagnostics[name] = value
            result = decode_controls_to_result(
                result,
                terrain,
                controls,
                sample_count=sample_count,
                contact_state=fixed_gait.stance_mask,
            )
            executed += 1
    else:
        for _ in range(configured):
            result, step_diagnostics = apply_safety_qp_step(result, terrain, state, command, cfg)
            qp_diagnostics.update(step_diagnostics)
            executed += 1
    solve_ms = _elapsed_ms(solve_t0)
    repair_t0 = time.perf_counter()
    final_xy_diagnostics: dict[str, Tensor] = {}
    final_root_diagnostics: dict[str, Tensor] = {}
    final_fk_diagnostics: dict[str, Tensor] = {}
    final_contact_diagnostics: dict[str, Tensor] = {}
    if not continuous_enabled:
        result = repair_touchdown_semantic_keepout(result, terrain, state)
        result, final_xy_diagnostics = apply_fk_body_leg_xy_repair(result, terrain, cfg)
        result, final_root_diagnostics = apply_fk_body_leg_root_lift(result, terrain, cfg)
        result, final_fk_diagnostics = apply_fk_shank_clearance_lift(result, terrain)
        result, final_contact_diagnostics = suppress_fk_low_small_contact_state(result, terrain)
    repair_ms = _elapsed_ms(repair_t0)
    diagnostics_t0 = time.perf_counter()
    safety = safety_diagnostics(result, terrain)
    continuous_diag = {}
    if continuous_enabled:
        continuous_diag = continuous_loss_diagnostics(
            result,
            terrain,
            footprint_radius_m=float(getattr(cfg.runtime, "continuous_footprint_radius_m", 0.04)),
            low_small_clearance_m=float(getattr(cfg.runtime, "low_small_swing_clearance_m", 0.0)),
        )
    diagnostics_ms = _elapsed_ms(diagnostics_t0)
    total_ms = _elapsed_ms(total_t0)
    diagnostics = {
        "qp_iterations_configured": _diagnostic_vector(batch, configured, like=result.root_pos),
        "qp_iterations_executed": _diagnostic_vector(batch, executed, like=result.root_pos),
        "qp_continuous_enabled": _diagnostic_vector(batch, 1 if continuous_enabled else 0, like=result.root_pos),
        "qp_fixed_gait_active": _diagnostic_vector(batch, 1 if continuous_enabled else 0, like=result.root_pos),
        "qp_repair_main_path_active": _diagnostic_vector(batch, 0 if continuous_enabled else 1, like=result.root_pos),
        "qp_fallback_reason": _diagnostic_vector(batch, 0, like=result.root_pos),
        "qp_nominal_ms": _diagnostic_vector(batch, nominal_ms, like=result.root_pos),
        "qp_solve_ms": _diagnostic_vector(batch, solve_ms, like=result.root_pos),
        "qp_repair_ms": _diagnostic_vector(batch, repair_ms, like=result.root_pos),
        "qp_diagnostics_ms": _diagnostic_vector(batch, diagnostics_ms, like=result.root_pos),
        "qp_total_ms": _diagnostic_vector(batch, total_ms, like=result.root_pos),
    }
    diagnostics.update(qp_diagnostics)
    diagnostics.update(final_xy_diagnostics)
    diagnostics.update(final_root_diagnostics)
    diagnostics.update(final_fk_diagnostics)
    diagnostics.update(final_contact_diagnostics)
    diagnostics.update(safety)
    diagnostics.update(continuous_diag)
    diagnostics.setdefault("qp_step_cap_violation_count", _diagnostic_vector(batch, 0, like=result.root_pos))
    diagnostics.setdefault("qp_terrain_risk_reduces_target_progress", _diagnostic_vector(batch, 0, like=result.root_pos))
    diagnostics.setdefault("qp_semantic_repaired_touchdown_count", _diagnostic_vector(batch, 0, like=result.root_pos))
    diagnostics.setdefault("qp_touchdown_semantic_fallback_count", _diagnostic_vector(batch, 0, like=result.root_pos))
    diagnostics.setdefault("qp_fk_body_leg_xy_repair_count", _diagnostic_vector(batch, 0, like=result.root_pos))
    diagnostics.setdefault("qp_fk_body_leg_root_lift_count", _diagnostic_vector(batch, 0, like=result.root_pos))
    diagnostics.setdefault("qp_low_small_crossing_root_lift_count", _diagnostic_vector(batch, 0, like=result.root_pos))
    diagnostics.setdefault("qp_low_small_contact_over_repair_count", _diagnostic_vector(batch, 0, like=result.root_pos))
    diagnostics.setdefault("qp_fk_low_small_contact_suppressed_count", _diagnostic_vector(batch, 0, like=result.root_pos))
    loss_breakdown = dict(result.loss_breakdown or {})
    loss_breakdown.update(diagnostics)
    cost_breakdown = dict(result.cost_breakdown)
    cost_breakdown.update(diagnostics)
    return MpcPlannerResult(
        root_pos=result.root_pos,
        root_rpy=result.root_rpy,
        foot_pos=result.foot_pos,
        joint_angles=result.joint_angles,
        contact_state=result.contact_state,
        touchdown_seq=result.touchdown_seq,
        planned_touchdown_w=result.planned_touchdown_w,
        cost_total=result.cost_total,
        cost_breakdown=cost_breakdown,
        status=result.status,
        feasible=result.feasible,
        safe_fallback=result.safe_fallback,
        loss_breakdown=loss_breakdown,
        hard_reason_mask=result.hard_reason_mask,
    )


__all__ = ["plan_segment_qp"]
