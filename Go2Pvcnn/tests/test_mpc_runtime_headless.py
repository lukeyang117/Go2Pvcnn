from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from tests.fixtures import viewer_runtime_diagnostics as viewer_diag


def _make_real_runtime_fixture(**kwargs):
    assert hasattr(viewer_diag, "make_real_runtime_fixture")
    return viewer_diag.make_real_runtime_fixture(**kwargs)


def _enable_4096_runtime_test() -> bool:
    return os.environ.get("MPC_RUNTIME_4096", "0").strip() == "1"


def _enable_long_drift_test() -> bool:
    return os.environ.get("MPC_RUNTIME_LONG_DRIFT", "0").strip() == "1"


def _enable_long_drift_sweep_test() -> bool:
    return os.environ.get("MPC_RUNTIME_LONG_DRIFT_SWEEP", "0").strip() == "1"


def _enable_long_drift_sequence_sweep_test() -> bool:
    return os.environ.get("MPC_RUNTIME_LONG_DRIFT_SEQUENCE_SWEEP", "0").strip() == "1"


def _runtime_device_override() -> str | None:
    value = os.environ.get("MPC_TEST_DEVICE", "").strip()
    return value or None


@dataclass(frozen=True, slots=True)
class _LongDriftVariant:
    name: str
    direction: str


def _long_drift_variants() -> tuple[_LongDriftVariant, ...]:
    requested = os.environ.get(
        "MPC_LONG_DRIFT_VARIANTS",
        "baseline,dir1_stance_anchor_proxy,dir2_phase_continuity,"
        "dir3_anchor_nominal_proxy,dir4_stronger_stance_loss,dir5_diagnostics_only",
    )
    mapping = {
        "baseline": "current planner behavior",
        "dir1_stance_anchor_proxy": "planner input clamps previous-contact feet to persistent anchors",
        "dir2_phase_continuity": "nominal gait phase advances across replans",
        "dir3_anchor_nominal_proxy": "nominal stance frames use persistent anchors instead of current root-relative feet",
        "dir4_stronger_stance_loss": "existing stance/root-frame losses are strengthened",
        "dir5_diagnostics_only": "diagnostic metrics only, no behavior change",
        "dir6_yaw_anchor_nominal_proxy": "anchor nominal replacement only when yaw dominates the command",
        "dir7_yaw_anchor_blend_proxy": "anchor nominal blend only when yaw dominates the command",
        "dir8_moderate_stance_loss": "moderately strengthen existing stance/root-frame losses",
        "dir9_linear_body_seed_proxy": "linear commands seed nominal feet from persistent body-frame footprint",
        "dir10_yaw_anchor_linear_seed_proxy": "yaw anchor nominal plus linear persistent body-frame footprint seed",
        "dir11_running_linear_body_seed_proxy": "linear commands use slowly updated body-frame footprint memory",
        "dir12_stance_only_yaw_anchor_linear_seed_proxy": "dir10 but yaw anchor only applies to prior-contact stance legs",
        "dir13_strict_gate_yaw_anchor_linear_seed_proxy": "dir10 with stricter command-regime gates",
        "dir14_soft_gate_yaw_anchor_linear_seed_proxy": "dir10 with continuous soft command-regime gates",
        "dir15_soft_gate_z_anchor_proxy": "dir14 plus contact-gated touchdown/world-z anchor replacement",
        "dir16_soft_gate_z_anchor_low_swing_proxy": "dir15 plus reduced nominal swing height",
        "dir17_soft_gate_z_anchor_touchdown_ramp_proxy": "dir15 plus touchdown-nearby z settle ramp",
        "dir18_soft_gate_z_anchor_disp_cap_proxy": "dir15 plus anchor displacement cap to suppress long-horizon yaw overshoot",
        "dir19_soft_gate_z_anchor_yaw_entry_ramp_proxy": "dir15 plus multi-replan yaw-entry ramp for mixed-to-yaw transitions",
        "dir20_soft_gate_z_anchor_disp_cap_yaw_entry_ramp_proxy": "dir15 plus displacement cap and yaw-entry ramp",
        "yawfix1_horizon_anchor_blend": "yaw stance anchor influence ramps within the horizon to reduce abrupt foot jumps",
        "yawfix2_foot_spike_loss": "yaw-dominant extra foot velocity/acceleration spike penalty",
        "yawfix3_touchdown_continuity_loss": "yaw-dominant touchdown frames are penalized for jumping away from the previous anchor",
        "yawfix4_body_relative_yaw_anchor": "yaw target mixes world stance anchor with root-yaw-rotated body footprint",
        "yawfix5_early_stance_guard": "yaw anchor influence is reduced on touchdown/early-stance frames",
        "yawfix4a_yaw_gate_body_anchor": "yawfix4 with a stricter yaw-dominant gate to protect linear/lateral commands",
        "yawfix4b_touchdown_jump_limiter": "yawfix4 with body-anchor displacement capped around touchdown distances",
        "yawfix4c_early_stance_hold": "yawfix4 with touchdown and early-stance anchor influence reduced",
        "yawfix4d_command_ramp": "yawfix4 with multi-replan yaw entry ramp for command switches",
        "yawfix4e_near_touchdown_mask": "yawfix4 applied mainly near touchdown/stance instead of every contact frame",
        "yawfix4f_full_guarded_combo": "yawfix4 with yaw gate, command ramp, touchdown cap, and early-stance hold",
    }
    variants: list[_LongDriftVariant] = []
    for raw_name in requested.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in mapping:
            raise ValueError(f"Unknown MPC_LONG_DRIFT_VARIANTS entry {name!r}; known={sorted(mapping)}")
        variants.append(_LongDriftVariant(name=name, direction=mapping[name]))
    return tuple(variants)


def _long_drift_command_names() -> tuple[str, ...]:
    default = "forward,backward,lateral_left,lateral_right,yaw_left,yaw_right"
    requested = os.environ.get("MPC_LONG_DRIFT_COMMANDS", default)
    known = {
        "forward",
        "backward",
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
    }
    out: list[str] = []
    for raw_name in requested.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in known:
            raise ValueError(f"Unknown MPC_LONG_DRIFT_COMMANDS entry {name!r}; known={sorted(known)}")
        out.append(name)
    return tuple(out)


def _mixed_long_drift_command_specs() -> tuple[tuple[str, tuple[float, float, float]], ...]:
    default = (
        "mix_forward_yaw_left:0.20,0.00,0.10;"
        "mix_forward_yaw_right:0.20,0.00,-0.10;"
        "mix_diag_yaw_left:0.10,0.10,0.10;"
        "mix_diag_yaw_right:0.10,-0.10,-0.10;"
        "mix_boundary_equal_left:0.15,0.00,0.15;"
        "mix_boundary_equal_right:0.15,0.00,-0.15"
    )
    requested = os.environ.get("MPC_LONG_DRIFT_MIXED_COMMANDS", default)
    out: list[tuple[str, tuple[float, float, float]]] = []
    for chunk in requested.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                "Invalid MPC_LONG_DRIFT_MIXED_COMMANDS entry "
                f"{chunk!r}; expected name:vx,vy,wz"
            )
        name, raw_values = chunk.split(":", 1)
        values = tuple(float(v.strip()) for v in raw_values.split(","))
        if len(values) != 3:
            raise ValueError(
                "Invalid MPC_LONG_DRIFT_MIXED_COMMANDS entry "
                f"{chunk!r}; expected exactly 3 floats"
            )
        out.append((name.strip(), (values[0], values[1], values[2])))
    return tuple(out)


def _sequence_long_drift_specs() -> tuple[tuple[str, tuple[str, ...]], ...]:
    default = (
        "forward_yaw_left_forward:forward,yaw_left,forward;"
        "forward_stop_backward:forward,standstill,backward;"
        "lateral_left_yaw_right_lateral_left:lateral_left,yaw_right,lateral_left;"
        "yaw_left_forward_yaw_right:yaw_left,forward,yaw_right;"
        "diag_mix_to_yaw:mix_diag_yaw_left,yaw_left,mix_diag_yaw_right"
    )
    requested = os.environ.get("MPC_LONG_DRIFT_SEQUENCES", default)
    out: list[tuple[str, tuple[str, ...]]] = []
    for chunk in requested.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid MPC_LONG_DRIFT_SEQUENCES entry {chunk!r}; expected name:a,b,c")
        name, raw_segments = chunk.split(":", 1)
        segments = tuple(seg.strip() for seg in raw_segments.split(",") if seg.strip())
        if not segments:
            raise ValueError(f"Invalid MPC_LONG_DRIFT_SEQUENCES entry {chunk!r}; no segments")
        out.append((name.strip(), segments))
    return tuple(out)


def _command_tensor_from_spec(runtime, spec: tuple[float, float, float]) -> torch.Tensor:
    return (
        torch.tensor(spec, dtype=torch.float32, device=runtime.base_env.device)
        .unsqueeze(0)
        .expand(runtime.num_envs, -1)
        .clone()
        .to(dtype=torch.float64)
    )


def _resolve_long_drift_commands(runtime) -> dict[str, torch.Tensor]:
    commands = {name: runtime._command_tensor(name) for name in runtime.command_cases}
    for name, spec in _mixed_long_drift_command_specs():
        commands[name] = _command_tensor_from_spec(runtime, spec)
    return commands


def _segment_metrics_summary(
    *,
    variant_name: str,
    seq_name: str,
    segment_name: str,
    cycles: int,
    rel_radius_series: list[float],
    foot_err_series: list[float],
    foot_step_series: list[float],
    root_dx_series: list[float],
    root_dy_series: list[float],
    root_dyaw_series: list[float],
    stance_anchor_err_series: list[float],
    touchdown_jump_series: list[float],
    touchdown_ground_gap_series: list[float],
    touchdown_airborne_ratio_series: list[float],
    touchdown_airborne_max_gap_series: list[float],
    phase_discontinuity_series: list[float],
    transition_foot_err_series: list[float],
    transition_anchor_err_series: list[float],
    contact_flip_count: int,
) -> dict[str, float | str]:
    drift = rel_radius_series[-1] - rel_radius_series[0]
    return {
        "variant": variant_name,
        "seq": seq_name,
        "segment": segment_name,
        "cycles": float(cycles),
        "rel_start": rel_radius_series[0],
        "rel_end": rel_radius_series[-1],
        "drift": drift,
        "abs_drift": abs(drift),
        "foot_err_mean": sum(foot_err_series) / len(foot_err_series),
        "foot_step_mean": sum(foot_step_series) / len(foot_step_series),
        "dx_mean": sum(root_dx_series) / len(root_dx_series),
        "dy_mean": sum(root_dy_series) / len(root_dy_series),
        "dyaw_mean": sum(root_dyaw_series) / len(root_dyaw_series),
        "stance_anchor_error": sum(stance_anchor_err_series) / len(stance_anchor_err_series),
        "touchdown_jump_distance": sum(touchdown_jump_series) / len(touchdown_jump_series),
        "touchdown_ground_gap_mean": sum(touchdown_ground_gap_series) / len(touchdown_ground_gap_series),
        "touchdown_airborne_ratio": sum(touchdown_airborne_ratio_series) / len(touchdown_airborne_ratio_series),
        "touchdown_airborne_max_gap": sum(touchdown_airborne_max_gap_series) / len(touchdown_airborne_max_gap_series),
        "phase_discontinuity": (
            sum(phase_discontinuity_series) / len(phase_discontinuity_series)
            if phase_discontinuity_series
            else 0.0
        ),
        "transition_foot_err_mean": (
            sum(transition_foot_err_series) / len(transition_foot_err_series)
            if transition_foot_err_series
            else 0.0
        ),
        "transition_anchor_error_mean": (
            sum(transition_anchor_err_series) / len(transition_anchor_err_series)
            if transition_anchor_err_series
            else 0.0
        ),
        "contact_flip_count": float(contact_flip_count),
    }


def _planned_touchdown_ground_metrics(terrain, result, *, tol_m: float = 0.02) -> tuple[float, float, float]:
    touchdown_w = torch.as_tensor(result.planned_touchdown_w, dtype=torch.float64)
    if touchdown_w.ndim == 2:
        touchdown_xy = touchdown_w[None, :, :2]
        touchdown_z = touchdown_w[None, :, 2]
    elif touchdown_w.ndim == 3:
        touchdown_xy = touchdown_w[:, :, :2]
        touchdown_z = touchdown_w[:, :, 2]
    elif touchdown_w.ndim == 4:
        touchdown_xy = touchdown_w[:, 0, :, :2]
        touchdown_z = touchdown_w[:, 0, :, 2]
    else:
        raise ValueError(f"Unsupported planned_touchdown_w shape {tuple(touchdown_w.shape)}")
    try:
        terrain_height_raw = _terrain_height_at(terrain, touchdown_xy.to(dtype=torch.float32))
        terrain_z = torch.as_tensor(terrain_height_raw, dtype=torch.float64, device=touchdown_w.device)
        if terrain_z.shape != touchdown_z.shape:
            terrain_z = terrain_z.reshape(touchdown_z.shape)
        gap = touchdown_z - terrain_z
        airborne = gap > float(tol_m)
        return (
            float(gap.mean().item()),
            float(airborne.to(dtype=torch.float64).mean().item()),
            float(torch.clamp(gap, min=0.0).max().item()),
        )
    except Exception as exc:
        raise RuntimeError(
            "touchdown ground metric failed: "
            f"touchdown_xy_shape={tuple(touchdown_xy.shape)} "
            f"touchdown_z_shape={tuple(touchdown_z.shape)} "
            f"terrain_type={type(terrain).__name__}"
        ) from exc


def _planned_touchdown_event_ground_metrics(
    terrain,
    result,
    *,
    tol_m: float = 0.02,
) -> tuple[float, float, float, float]:
    contact_state = torch.as_tensor(result.contact_state, dtype=torch.bool)
    if contact_state.ndim != 3:
        raise ValueError(f"contact_state must be [B,T,4], got {tuple(contact_state.shape)}")
    prev = torch.cat((contact_state[:, :1], contact_state[:, :-1]), dim=1)
    rises = torch.logical_and(contact_state, torch.logical_not(prev))
    valid = rises.any(dim=1)
    valid_count = int(valid.sum().item())
    total_count = int(valid.numel())
    if valid_count == 0:
        return 0.0, 0.0, 0.0, 0.0

    touchdown_seq = torch.as_tensor(result.touchdown_seq, dtype=torch.float64)
    if touchdown_seq.ndim != 4 or int(touchdown_seq.shape[-1]) != 3:
        raise ValueError(f"touchdown_seq must be [B,4,E,3], got {tuple(touchdown_seq.shape)}")
    first_touchdown = touchdown_seq[:, :, 0, :]
    touchdown_xy = first_touchdown[..., :2]
    touchdown_z = first_touchdown[..., 2]
    terrain_z = torch.as_tensor(
        _terrain_height_at(terrain, touchdown_xy.to(dtype=torch.float32)),
        dtype=torch.float64,
        device=touchdown_z.device,
    ).reshape(touchdown_z.shape)
    gap = touchdown_z - terrain_z
    gap_valid = gap[valid]
    airborne = gap_valid > float(tol_m)
    return (
        float(gap_valid.mean().item()),
        float(airborne.to(dtype=torch.float64).mean().item()),
        float(torch.clamp(gap_valid, min=0.0).max().item()),
        float(valid_count / max(total_count, 1)),
    )


def _planned_stance_ground_metrics(
    terrain,
    result,
    *,
    tol_m: float = 0.02,
) -> tuple[float, float, float]:
    foot_pos_w = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
    contact_state = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot_pos_w.device)
    if foot_pos_w.ndim != 4 or int(foot_pos_w.shape[-2]) != 4 or int(foot_pos_w.shape[-1]) != 3:
        raise ValueError(f"foot_pos_w must be [B,T,4,3], got {tuple(foot_pos_w.shape)}")
    if contact_state.shape != foot_pos_w.shape[:3]:
        raise ValueError(
            f"contact_state shape {tuple(contact_state.shape)} does not match foot_pos_w {tuple(foot_pos_w.shape)}"
        )
    terrain_z = torch.as_tensor(
        _terrain_height_at(terrain, foot_pos_w[..., :2].reshape(int(foot_pos_w.shape[0]), -1, 2).to(dtype=torch.float32)),
        dtype=torch.float64,
        device=foot_pos_w.device,
    ).reshape(foot_pos_w.shape[:3])
    gap = foot_pos_w[..., 2] - terrain_z
    valid = contact_state
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return 0.0, 0.0, 0.0
    gap_valid = gap[valid]
    airborne = gap_valid > float(tol_m)
    return (
        float(gap_valid.mean().item()),
        float(airborne.to(dtype=torch.float64).mean().item()),
        float(torch.clamp(gap_valid, min=0.0).max().item()),
    )


def _terrain_height_at(terrain, points_xy: torch.Tensor) -> torch.Tensor:
    if hasattr(terrain, "height_at"):
        return terrain.height_at(points_xy)
    if hasattr(terrain, "height_map") and hasattr(terrain, "world_x_range") and hasattr(terrain, "world_y_range"):
        return _sample_mpc_height_map(terrain, points_xy)
    raise TypeError(f"Unsupported terrain type for touchdown metric: {type(terrain).__name__}")


def _sample_mpc_height_map(terrain, points_xy: torch.Tensor) -> torch.Tensor:
    height_map = torch.as_tensor(terrain.height_map, dtype=torch.float32, device=points_xy.device)
    if height_map.ndim == 2:
        height_map = height_map.unsqueeze(0)
    if height_map.ndim != 3:
        raise ValueError(f"terrain.height_map must be [B,H,W] or [H,W], got {tuple(height_map.shape)}")

    batch = int(height_map.shape[0])
    points_xy = torch.as_tensor(points_xy, dtype=torch.float32, device=height_map.device)
    if points_xy.ndim == 2:
        points_xy = points_xy.unsqueeze(0)
    if points_xy.ndim != 3 or int(points_xy.shape[-1]) != 2:
        raise ValueError(f"points_xy must be [B,P,2] or [P,2], got {tuple(points_xy.shape)}")
    if int(points_xy.shape[0]) == 1 and batch > 1:
        points_xy = points_xy.expand(batch, -1, -1)
    if int(points_xy.shape[0]) != batch:
        raise ValueError(
            f"points_xy batch {int(points_xy.shape[0])} does not match terrain batch {batch}"
        )

    x0, x1 = terrain.world_x_range
    y0, y1 = terrain.world_y_range
    xs = points_xy[..., 0].clamp(float(x0), float(x1))
    ys = points_xy[..., 1].clamp(float(y0), float(y1))
    x_norm = (xs - float(x0)) / max(float(x1) - float(x0), 1.0e-6) * 2.0 - 1.0
    y_norm = (float(y1) - ys) / max(float(y1) - float(y0), 1.0e-6) * 2.0 - 1.0
    sample_grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        height_map.unsqueeze(1),
        sample_grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return sampled[:, 0, :, 0]


def _yaw_dominance(command: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    cmd = torch.as_tensor(command, dtype=like.dtype, device=like.device)
    lin = torch.linalg.vector_norm(cmd[:, :2], dim=-1, keepdim=True)
    yaw = torch.abs(cmd[:, 2:3])
    return yaw / torch.clamp(lin + yaw, min=1.0e-6)


def _linear_dominance(command: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    return 1.0 - _yaw_dominance(command, like=like)


def _initial_foot_rel_body(state) -> torch.Tensor:
    root = torch.as_tensor(state.root_pos, dtype=torch.float32, device=state.foot_pos.device)
    rpy = torch.as_tensor(state.root_rpy, dtype=torch.float32, device=state.foot_pos.device)
    foot = torch.as_tensor(state.foot_pos, dtype=torch.float32, device=state.foot_pos.device)
    rel = foot - root.unsqueeze(1)
    yaw = rpy[:, 2]
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    rel_body_xy = torch.stack(
        (
            cy * rel[..., 0] + sy * rel[..., 1],
            -sy * rel[..., 0] + cy * rel[..., 1],
        ),
        dim=-1,
    )
    return torch.cat((rel_body_xy, rel[..., 2:3]), dim=-1)


def _foot_pos_from_body_rel(state, rel_body: torch.Tensor) -> torch.Tensor:
    root = torch.as_tensor(state.root_pos, dtype=torch.float32, device=state.foot_pos.device)
    rpy = torch.as_tensor(state.root_rpy, dtype=torch.float32, device=state.foot_pos.device)
    rel_body = torch.as_tensor(rel_body, dtype=torch.float32, device=state.foot_pos.device)
    yaw = rpy[:, 2]
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    rel_world_xy = torch.stack(
        (
            cy * rel_body[..., 0] - sy * rel_body[..., 1],
            sy * rel_body[..., 0] + cy * rel_body[..., 1],
        ),
        dim=-1,
    )
    rel_world = torch.cat((rel_world_xy, rel_body[..., 2:3]), dim=-1)
    return root.unsqueeze(1) + rel_world


def _clone_mpc_state(state, *, foot_pos: torch.Tensor):
    from extension.batch_mpc_planner.types import MpcRobotState

    return MpcRobotState(
        root_pos=state.root_pos,
        root_rpy=state.root_rpy,
        joint_angles=state.joint_angles,
        foot_pos=foot_pos.to(dtype=state.foot_pos.dtype, device=state.foot_pos.device),
        foot_vel=state.foot_vel,
    )


def _binary_gate(weight: torch.Tensor, *, threshold: float) -> torch.Tensor:
    return (weight > float(threshold)).to(dtype=weight.dtype, device=weight.device)



def _yawfix_anchor_weight(nominal: dict[str, torch.Tensor], command: torch.Tensor, *, yaw_threshold_start: float = 0.35) -> torch.Tensor:
    yaw_dom = _yaw_dominance(command, like=nominal["foot_pos"])
    return torch.clamp((yaw_dom - float(yaw_threshold_start)) / 0.45, min=0.0, max=1.0).to(
        dtype=nominal["foot_pos"].dtype,
        device=nominal["foot_pos"].device,
    )


_YAWFIX4_DERIVED_VARIANTS = {
    "yawfix4_body_relative_yaw_anchor",
    "yawfix4a_yaw_gate_body_anchor",
    "yawfix4b_touchdown_jump_limiter",
    "yawfix4c_early_stance_hold",
    "yawfix4d_command_ramp",
    "yawfix4e_near_touchdown_mask",
    "yawfix4f_full_guarded_combo",
}


def _with_yawfix_anchor_nominal(
    original_builder,
    state,
    command,
    terrain,
    runtime_cfg,
    memory,
    shared: SimpleNamespace,
    variant_name: str,
):
    nominal = original_builder(state, command, terrain, runtime_cfg)
    anchor = getattr(shared, "stance_anchor_w", None)
    if anchor is None:
        return nominal
    contact = nominal["contact_prior"] > 0.5
    yaw_weight = _yawfix_anchor_weight(nominal, command)
    anchor_t = anchor.to(dtype=nominal["foot_pos"].dtype, device=nominal["foot_pos"].device).unsqueeze(1)
    effective = yaw_weight[:, None, None, :]

    if variant_name == "yawfix1_horizon_anchor_blend":
        horizon = int(nominal["foot_pos"].shape[1])
        time_ramp = torch.linspace(
            0.15,
            1.0,
            horizon,
            dtype=nominal["foot_pos"].dtype,
            device=nominal["foot_pos"].device,
        ).view(1, horizon, 1, 1)
        effective = effective * time_ramp
    elif variant_name in _YAWFIX4_DERIVED_VARIANTS:
        rel_body = getattr(shared, "running_foot_rel_body", None)
        if rel_body is not None:
            body_anchor = _foot_pos_from_body_rel(state, rel_body).to(
                dtype=nominal["foot_pos"].dtype,
                device=nominal["foot_pos"].device,
            ).unsqueeze(1)
            body_blend = 0.45
            if variant_name in {"yawfix4a_yaw_gate_body_anchor", "yawfix4f_full_guarded_combo"}:
                yaw_weight = _yawfix_anchor_weight(nominal, command, yaw_threshold_start=0.50)
                linear_dom = _linear_dominance(command, like=nominal["foot_pos"]).to(
                    dtype=nominal["foot_pos"].dtype,
                    device=nominal["foot_pos"].device,
                )
                linear_guard = torch.clamp((0.70 - linear_dom) / 0.35, min=0.0, max=1.0)
                yaw_weight = yaw_weight * linear_guard
                effective = yaw_weight[:, None, None, :]
                body_blend = 0.38
            if variant_name in {"yawfix4d_command_ramp", "yawfix4f_full_guarded_combo"}:
                yaw_scalar = float(yaw_weight.mean().item())
                prev_yaw_scalar = float(getattr(shared, "yawfix4_prev_yaw_weight", 0.0))
                if yaw_scalar > 0.10:
                    if prev_yaw_scalar <= 0.10:
                        yaw_entry_steps = 0
                    else:
                        yaw_entry_steps = int(getattr(shared, "yawfix4_yaw_entry_steps", 0)) + 1
                    ramp_scalar = min(1.0, float(yaw_entry_steps + 1) / 5.0)
                else:
                    yaw_entry_steps = 999
                    ramp_scalar = 1.0
                shared.yawfix4_prev_yaw_weight = yaw_scalar
                shared.yawfix4_yaw_entry_steps = yaw_entry_steps
                effective = effective * torch.full_like(effective, ramp_scalar)
            anchor_t = torch.lerp(anchor_t, body_anchor, body_blend)
            if variant_name in {"yawfix4b_touchdown_jump_limiter", "yawfix4f_full_guarded_combo"}:
                delta = anchor_t - nominal["foot_pos"]
                dist = torch.linalg.vector_norm(delta[..., :2], dim=-1, keepdim=True)
                cap = torch.clamp(0.095 / torch.clamp(dist, min=1.0e-6), max=1.0)
                capped_xy = nominal["foot_pos"][..., :2] + delta[..., :2] * cap
                anchor_z = anchor_t[..., 2:3].expand_as(nominal["foot_pos"][..., 2:3])
                anchor_t = torch.cat((capped_xy, anchor_z), dim=-1)
            if variant_name in {"yawfix4c_early_stance_hold", "yawfix4f_full_guarded_combo"}:
                prev_contact = torch.cat((contact[:, :1], contact[:, :-1]), dim=1)
                touchdown = torch.logical_and(contact, torch.logical_not(prev_contact))
                early = torch.logical_or(touchdown, torch.cat((touchdown[:, :-1], torch.zeros_like(touchdown[:, :1])), dim=1))
                effective = torch.where(early.unsqueeze(-1), effective * 0.20, effective)
            if variant_name == "yawfix4e_near_touchdown_mask":
                prev_contact = torch.cat((contact[:, :1], contact[:, :-1]), dim=1)
                next_contact = torch.cat((contact[:, 1:], contact[:, -1:]), dim=1)
                touchdown = torch.logical_and(contact, torch.logical_not(prev_contact))
                pre_touchdown = torch.logical_and(next_contact, torch.logical_not(contact))
                near_touchdown = torch.logical_or(touchdown, pre_touchdown)
                near_touchdown = torch.logical_or(
                    near_touchdown,
                    torch.cat((touchdown[:, :-1], torch.zeros_like(touchdown[:, :1])), dim=1),
                )
                mask = torch.where(
                    near_touchdown.unsqueeze(-1),
                    torch.ones_like(effective),
                    torch.full_like(effective, 0.35),
                )
                effective = effective * mask
    elif variant_name == "yawfix5_early_stance_guard":
        prev_contact = torch.cat((contact[:, :1], contact[:, :-1]), dim=1)
        touchdown = torch.logical_and(contact, torch.logical_not(prev_contact))
        early = torch.logical_or(touchdown, torch.cat((touchdown[:, :-1], torch.zeros_like(touchdown[:, :1])), dim=1))
        effective = torch.where(early.unsqueeze(-1), effective * 0.25, effective)

    replacement = torch.lerp(nominal["foot_pos"], anchor_t, effective)
    nominal["foot_pos"] = torch.where(contact.unsqueeze(-1), replacement, nominal["foot_pos"])
    nominal["foot_pos"][..., 2:3] = torch.where(
        contact.unsqueeze(-1),
        torch.lerp(nominal["foot_pos"][..., 2:3], anchor_t[..., 2:3], effective),
        nominal["foot_pos"][..., 2:3],
    )
    return nominal


def _yawfix_foot_spike_extra_loss(decoded, nominal, state, command, terrain, cfg):
    del nominal, state, terrain, cfg
    if int(decoded.foot_pos.shape[1]) < 3:
        return torch.zeros(decoded.foot_pos.shape[0], dtype=decoded.foot_pos.dtype, device=decoded.foot_pos.device)
    yaw = _yaw_dominance(command, like=decoded.foot_pos).view(-1)
    dfoot = decoded.foot_pos[:, 1:] - decoded.foot_pos[:, :-1]
    dnorm = torch.linalg.vector_norm(dfoot, dim=-1)
    accel = dfoot[:, 1:] - dfoot[:, :-1]
    anorm = torch.linalg.vector_norm(accel, dim=-1)
    spike = torch.relu(dnorm - 0.055).square().mean(dim=(1, 2))
    acc = torch.relu(anorm - 0.060).square().mean(dim=(1, 2))
    return yaw * (14.0 * spike + 8.0 * acc)


def _yawfix_touchdown_continuity_extra_loss(decoded, nominal, state, command, cfg, shared: SimpleNamespace):
    del nominal, state, cfg
    anchor = getattr(shared, "stance_anchor_w", None)
    if anchor is None or int(decoded.foot_pos.shape[1]) < 2:
        return torch.zeros(decoded.foot_pos.shape[0], dtype=decoded.foot_pos.dtype, device=decoded.foot_pos.device)
    yaw = _yaw_dominance(command, like=decoded.foot_pos).view(-1)
    contact = decoded.contact_prob > 0.5
    touchdown = torch.logical_and(contact[:, 1:], torch.logical_not(contact[:, :-1]))
    anchor_t = anchor.to(dtype=decoded.foot_pos.dtype, device=decoded.foot_pos.device).unsqueeze(1)
    jump = torch.linalg.vector_norm(decoded.foot_pos[:, 1:] - anchor_t, dim=-1)
    denom = torch.clamp(touchdown.to(dtype=decoded.foot_pos.dtype).sum(dim=(1, 2)), min=1.0)
    excess = torch.relu(jump - 0.075).square() * touchdown.to(dtype=decoded.foot_pos.dtype)
    return yaw * 20.0 * (excess.sum(dim=(1, 2)) / denom)


@contextmanager
def _long_drift_variant_context(runtime, variant_name: str, shared: SimpleNamespace):
    cfg = runtime.mpc_planner_cfg
    losses = cfg.losses
    old_values = {
        "diagnostics_enabled": bool(cfg.diagnostics.enabled),
        "stance_ground_weight": float(losses.stance_ground.weight),
        "root_center_weight": float(losses.root_foot_center.weight),
        "support_plane_weight": float(losses.support_plane_rp.weight),
        "touchdown_surface_weight": float(losses.touchdown_surface.weight),
    }

    try:
        if variant_name == "dir4_stronger_stance_loss":
            losses.stance_ground.weight = 4.0
            losses.root_foot_center.weight = 3.5
            losses.support_plane_rp.weight = 2.5
            losses.touchdown_surface.weight = 1.25
        if variant_name == "dir8_moderate_stance_loss":
            losses.stance_ground.weight = 1.8
            losses.root_foot_center.weight = 2.2
            losses.support_plane_rp.weight = 1.4
            losses.touchdown_surface.weight = 0.65
        if variant_name == "dir5_diagnostics_only":
            cfg.diagnostics.enabled = True

        retired_dense_variants = {
            "dir2_phase_continuity",
            "dir3_anchor_nominal_proxy",
            "dir6_yaw_anchor_nominal_proxy",
            "dir7_yaw_anchor_blend_proxy",
            "dir9_linear_body_seed_proxy",
            "dir10_yaw_anchor_linear_seed_proxy",
            "dir11_running_linear_body_seed_proxy",
            "dir12_stance_only_yaw_anchor_linear_seed_proxy",
            "dir13_strict_gate_yaw_anchor_linear_seed_proxy",
            "dir14_soft_gate_yaw_anchor_linear_seed_proxy",
            "dir15_soft_gate_z_anchor_proxy",
            "dir16_soft_gate_z_anchor_low_swing_proxy",
            "dir17_soft_gate_z_anchor_touchdown_ramp_proxy",
            "dir18_soft_gate_z_anchor_disp_cap_proxy",
            "dir19_soft_gate_z_anchor_yaw_entry_ramp_proxy",
            "dir20_soft_gate_z_anchor_disp_cap_yaw_entry_ramp_proxy",
            "yawfix1_horizon_anchor_blend",
            "yawfix2_foot_spike_loss",
            "yawfix3_touchdown_continuity_loss",
            "yawfix4_body_relative_yaw_anchor",
            "yawfix5_early_stance_guard",
            "yawfix4a_yaw_gate_body_anchor",
            "yawfix4b_touchdown_jump_limiter",
            "yawfix4c_early_stance_hold",
            "yawfix4d_command_ramp",
            "yawfix4e_near_touchdown_mask",
            "yawfix4f_full_guarded_combo",
        }
        if variant_name in retired_dense_variants:
            yield
        else:
            yield
    finally:
        cfg.diagnostics.enabled = old_values["diagnostics_enabled"]
        losses.stance_ground.weight = old_values["stance_ground_weight"]
        losses.root_foot_center.weight = old_values["root_center_weight"]
        losses.support_plane_rp.weight = old_values["support_plane_weight"]
        losses.touchdown_surface.weight = old_values["touchdown_surface_weight"]


@pytest.fixture(scope="module")
def real_semantic_mpc_runtime():
    kwargs = {"num_envs": 2, "planner_backend": "mpc"}
    device = _runtime_device_override()
    if device is not None:
        kwargs["device"] = device
    runtime = _make_real_runtime_fixture(**kwargs)
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture(scope="module")
def real_semantic_mpc_runtime_4096():
    if not _enable_4096_runtime_test():
        pytest.skip("Set MPC_RUNTIME_4096=1 to run 4096-env IsaacLab headless runtime acceptance.")
    kwargs = {
        "num_envs": 4096,
        "planner_backend": "mpc",
        "warmup_steps": 2,
        "task_id": "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
        "env_cfg_entry_point": (
            "go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg:"
            "TeacherElevationTrajectoryMpcSemanticEnvCfg"
        ),
    }
    device = _runtime_device_override()
    if device is not None:
        kwargs["device"] = device
    runtime = _make_real_runtime_fixture(**kwargs)
    try:
        yield runtime
    finally:
        runtime.close()


def test_mpc_runtime_fixture_attaches_mpc_backend(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime

    assert runtime.planner_backend == "mpc"
    assert runtime.scanner_name == "semantic_height_scanner"
    manager = runtime.base_env._trajectory_manager
    assert manager is not None
    assert getattr(manager, "planner_backend", None) == "mpc"


def test_mpc_runtime_plan_case_headless_smoke(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    standstill = runtime.plan_case("standstill")
    forward = runtime.plan_case("forward")

    assert standstill.result.num_frames == runtime.requested_n_frames
    assert standstill.result.contact_state.dtype == torch.bool
    assert standstill.grounded_crossing is None
    assert standstill.summary["standstill"] is True
    assert torch.isfinite(standstill.result.root_pos_w).all()
    assert torch.isfinite(standstill.result.foot_pos_w).all()

    assert forward.summary["standstill"] is False
    assert forward.summary["dx"] > 0.03
    assert abs(forward.summary["dx"]) > abs(forward.summary["dy"]) + 0.01
    assert torch.isfinite(forward.result.root_pos_w).all()
    assert torch.isfinite(forward.result.foot_pos_w).all()


def test_mpc_runtime_forward_plan_has_time_varying_joint_angles(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    forward = runtime.plan_case("forward")

    root = torch.as_tensor(forward.result.root_pos_w, dtype=torch.float64)
    joints = torch.as_tensor(forward.result.joint_angles, dtype=torch.float64)
    feet = torch.as_tensor(forward.result.foot_pos_w, dtype=torch.float64)

    root_dx = torch.abs(root[:, -1, 0] - root[:, 0, 0])
    joint_tspan = torch.abs(joints.amax(dim=1) - joints.amin(dim=1))
    foot_tspan = torch.linalg.vector_norm(feet.amax(dim=1) - feet.amin(dim=1), dim=-1)

    assert float(root_dx.max().item()) > 0.05
    assert float(foot_tspan.max().item()) > 0.01
    # Regression guardrail: moving command should not keep all joint trajectories
    # exactly constant over the full MPC horizon.
    assert float(joint_tspan.max().item()) > 1.0e-3


def test_mpc_runtime_viewer_style_replan_keeps_feet_moving(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    runtime.reset()
    viewer = runtime._viewer
    terrain = runtime._single_env_terrain()
    command = runtime._command_tensor("forward")[:1]
    state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
    foot_step_means: list[float] = []

    for _ in range(8):
        result = viewer._plan_viewer_trajectory(
                terrain=terrain,
                state=state,
                command=command,
                mpc_cfg=runtime.mpc_planner_cfg,
        )
        foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
        foot_step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
        foot_step_means.append(float(foot_step.mean().item()))
        frame_idx = result.num_frames - 1
        viewer._apply_direct_playback_to_robot(runtime.robot, result, frame_idx=frame_idx)
        runtime.base_env.scene.write_data_to_sim()
        runtime.base_env.sim.render()
        runtime.base_env.scene.update(float(runtime.base_env.physics_dt))
        state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())

    assert min(foot_step_means) > 1.0e-4
    assert foot_step_means[-1] > 0.25 * foot_step_means[0]


def test_mpc_runtime_yaw_playback_wxyz_rpy_matches_plan(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    yaw_left = runtime.plan_case("yaw_left")
    frame_idx = min(25, yaw_left.result.num_frames - 1)

    runtime._viewer._apply_direct_playback_to_robot(runtime.robot, yaw_left.result, frame_idx=frame_idx)
    runtime.base_env.scene.write_data_to_sim()
    runtime.base_env.sim.render()
    runtime.base_env.scene.update(float(runtime.base_env.physics_dt))

    actual = runtime._viewer._read_actual_base_state(runtime.base_env)
    plan = runtime._viewer._planner_state_from_reference_result(yaw_left.result, frame_idx=frame_idx)
    plan_rpy = runtime._viewer._quat_wxyz_to_rpy(plan.root_quat)

    torch.testing.assert_close(actual["rpy_if_wxyz"], plan_rpy, atol=2.0e-4, rtol=2.0e-4)
    # T300e intentionally lets MPC estimate roll/pitch from the support plane;
    # this guard verifies playback preserves that planned orientation without
    # allowing runaway tilt.
    assert float(actual["rpy_if_wxyz"][0, :2].abs().max().item()) < 2.0e-2


def test_mpc_runtime_command_matrix_tracks_motion_and_limits_drift(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    viewer = runtime._viewer
    terrain = runtime._single_env_terrain()

    command_names = _long_drift_command_names()

    for name in command_names:
        runtime.reset()
        command = runtime._command_tensor(name)[:1]
        state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
        rel_series: list[float] = []
        foot_step_series: list[float] = []
        foot_err_series: list[float] = []
        dx_series: list[float] = []
        dy_series: list[float] = []
        dyaw_series: list[float] = []

        for _ in range(8):
            result = viewer._plan_viewer_trajectory(
                terrain=terrain,
                state=state,
                command=command,
                mpc_cfg=runtime.mpc_planner_cfg,
            )
            root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
            foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
            quat = torch.as_tensor(result.root_quat_w, dtype=torch.float64)
            rpy = viewer._quat_wxyz_to_rpy(quat)
            rel = foot - root.unsqueeze(2)
            rel_series.append(float(torch.linalg.vector_norm(rel, dim=-1).max().item()))
            foot_step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
            foot_step_series.append(float(foot_step.mean().item()))
            dx_series.append(float((root[0, -1, 0] - root[0, 0, 0]).item()))
            dy_series.append(float((root[0, -1, 1] - root[0, 0, 1]).item()))
            dyaw_series.append(float((rpy[0, -1, 2] - rpy[0, 0, 2]).item()))

            frame_idx = result.num_frames - 1
            viewer._apply_direct_playback_to_robot(runtime.robot, result, frame_idx=frame_idx)
            runtime.base_env.scene.write_data_to_sim()
            runtime.base_env.sim.render()
            runtime.base_env.scene.update(float(runtime.base_env.physics_dt))
            actual_kin = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
            plan_foot_last = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64)
            actual_foot_last = torch.as_tensor(actual_kin["foot_pos_w"], dtype=torch.float64)
            foot_err = torch.linalg.vector_norm(actual_foot_last - plan_foot_last, dim=-1)
            foot_err_series.append(float(foot_err.mean().item()))
            state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())

        rel_growth = rel_series[-1] - rel_series[0]
        foot_step_mean = sum(foot_step_series) / len(foot_step_series)
        foot_err_mean = sum(foot_err_series) / len(foot_err_series)
        dx_mean = sum(dx_series) / len(dx_series)
        dy_mean = sum(dy_series) / len(dy_series)
        dyaw_mean = sum(dyaw_series) / len(dyaw_series)

        assert rel_growth < 0.25, (name, rel_growth, rel_series)
        assert rel_series[-1] < 0.85, (name, rel_series[-1], rel_series)
        assert foot_err_mean < 0.18, (name, foot_err_mean, foot_err_series)
        min_foot_step = 0.004 if name.startswith("yaw_") else 0.005
        assert foot_step_mean > min_foot_step, (name, foot_step_mean, foot_step_series, min_foot_step)

        if name == "forward":
            assert dx_mean > 0.10, (name, dx_mean)
        elif name == "backward":
            assert dx_mean < -0.10, (name, dx_mean)
        elif name == "lateral_left":
            assert dy_mean > 0.08, (name, dy_mean)
        elif name == "lateral_right":
            assert dy_mean < -0.08, (name, dy_mean)
        elif name == "yaw_left":
            assert dyaw_mean > 0.15, (name, dyaw_mean)
        elif name == "yaw_right":
            assert dyaw_mean < -0.15, (name, dyaw_mean)


@pytest.mark.skipif(
    not _enable_long_drift_test(),
    reason="Set MPC_RUNTIME_LONG_DRIFT=1 to run long IsaacLab MPC foot-drift reproduction.",
)
def test_mpc_runtime_long_replan_foot_drift_reproduction(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    viewer = runtime._viewer
    terrain = runtime._single_env_terrain()
    command_names = _long_drift_command_names()
    cycles = int(os.environ.get("MPC_LONG_DRIFT_CYCLES", "120"))
    drift_threshold_m = float(os.environ.get("MPC_LONG_DRIFT_EXPECT_MIN_M", "0.045"))
    reports: list[dict[str, float | str]] = []

    for name in command_names:
        runtime.reset()
        command = runtime._command_tensor(name)[:1]
        state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
        rel_radius_series: list[float] = []
        foot_err_series: list[float] = []
        foot_step_series: list[float] = []
        root_dx_series: list[float] = []
        root_dy_series: list[float] = []
        root_dyaw_series: list[float] = []

        for _ in range(cycles):
            result = viewer._plan_viewer_trajectory(
                terrain=terrain,
                state=state,
                command=command,
                mpc_cfg=runtime.mpc_planner_cfg,
            )
            root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
            foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
            rpy = viewer._quat_wxyz_to_rpy(torch.as_tensor(result.root_quat_w, dtype=torch.float64))
            rel = foot - root.unsqueeze(2)
            rel_radius_series.append(float(torch.linalg.vector_norm(rel[:, -1], dim=-1).mean().item()))
            foot_step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
            foot_step_series.append(float(foot_step.mean().item()))
            root_dx_series.append(float((root[0, -1, 0] - root[0, 0, 0]).item()))
            root_dy_series.append(float((root[0, -1, 1] - root[0, 0, 1]).item()))
            root_dyaw_series.append(float((rpy[0, -1, 2] - rpy[0, 0, 2]).item()))

            frame_idx = result.num_frames - 1
            viewer._apply_direct_playback_to_robot(runtime.robot, result, frame_idx=frame_idx)
            runtime.base_env.scene.write_data_to_sim()
            runtime.base_env.sim.render()
            runtime.base_env.scene.update(float(runtime.base_env.physics_dt))
            actual_kin = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
            plan_foot_last = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64)
            actual_foot_last = torch.as_tensor(actual_kin["foot_pos_w"], dtype=torch.float64)
            foot_err = torch.linalg.vector_norm(actual_foot_last - plan_foot_last, dim=-1)
            foot_err_series.append(float(foot_err.mean().item()))
            state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())

        drift = rel_radius_series[-1] - rel_radius_series[0]
        report = {
            "name": name,
            "rel_start": rel_radius_series[0],
            "rel_end": rel_radius_series[-1],
            "drift": drift,
            "abs_drift": abs(drift),
            "foot_err_mean": sum(foot_err_series) / len(foot_err_series),
            "foot_err_last": foot_err_series[-1],
            "foot_step_mean": sum(foot_step_series) / len(foot_step_series),
            "dx_mean": sum(root_dx_series) / len(root_dx_series),
            "dy_mean": sum(root_dy_series) / len(root_dy_series),
            "dyaw_mean": sum(root_dyaw_series) / len(root_dyaw_series),
        }
        reports.append(report)
        print(
            "MPC_LONG_DRIFT "
            f"name={name} "
            f"cycles={cycles} "
            f"rel_start={report['rel_start']:.4f} "
            f"rel_end={report['rel_end']:.4f} "
            f"drift={report['drift']:+.4f} "
            f"abs_drift={report['abs_drift']:.4f} "
            f"foot_err_mean={report['foot_err_mean']:.4f} "
            f"foot_err_last={report['foot_err_last']:.4f} "
            f"foot_step_mean={report['foot_step_mean']:.4f} "
            f"dx_mean={report['dx_mean']:+.4f} "
            f"dy_mean={report['dy_mean']:+.4f} "
            f"dyaw_mean={report['dyaw_mean']:+.4f}",
            flush=True,
        )

    max_report = max(reports, key=lambda item: float(item["abs_drift"]))
    mean_abs_drift = sum(float(item["abs_drift"]) for item in reports) / len(reports)
    print(
        "MPC_LONG_DRIFT_SUMMARY "
        f"cycles={cycles} "
        f"mean_abs_drift={mean_abs_drift:.4f} "
        f"max_name={max_report['name']} "
        f"max_abs_drift={float(max_report['abs_drift']):.4f}",
        flush=True,
    )

    assert float(max_report["abs_drift"]) >= drift_threshold_m, (
        "long replan drift did not reproduce above threshold",
        drift_threshold_m,
        reports,
    )


@pytest.mark.skipif(
    not _enable_long_drift_sweep_test(),
    reason="Set MPC_RUNTIME_LONG_DRIFT_SWEEP=1 to run long IsaacLab MPC variant sweep.",
)
def test_mpc_runtime_long_replan_variant_sweep(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    viewer = runtime._viewer
    terrain = runtime._single_env_terrain()
    command_names = _long_drift_command_names()
    cycles = int(os.environ.get("MPC_LONG_DRIFT_CYCLES", "120"))
    variants = _long_drift_variants()
    baseline_by_name: dict[str, dict[str, float | str]] = {}
    all_reports: list[dict[str, float | str]] = []

    for variant in variants:
        variant_reports: list[dict[str, float | str]] = []
        print(
            "MPC_LONG_DRIFT_VARIANT_INFO "
            f"variant={variant.name} "
            f"direction={variant.direction}",
            flush=True,
        )
        for name in command_names:
            runtime.reset()
            command = runtime._command_tensor(name)[:1]
            state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
            shared = SimpleNamespace(
                stance_anchor_w=torch.as_tensor(state.foot_pos, dtype=torch.float32).clone(),
                prev_contact_state=torch.ones((1, 4), dtype=torch.bool, device=state.foot_pos.device),
                phase_shift=0.0,
                last_touchdown_w=torch.as_tensor(state.foot_pos, dtype=torch.float32).clone(),
                initial_foot_rel_body=_initial_foot_rel_body(state),
                running_foot_rel_body=_initial_foot_rel_body(state),
                prev_touchdown_w=None,
                prev_contact_first=None,
            )
            rel_radius_series: list[float] = []
            foot_err_series: list[float] = []
            foot_step_series: list[float] = []
            root_dx_series: list[float] = []
            root_dy_series: list[float] = []
            root_dyaw_series: list[float] = []
            stance_anchor_err_series: list[float] = []
            touchdown_jump_series: list[float] = []
            touchdown_ground_gap_series: list[float] = []
            touchdown_airborne_ratio_series: list[float] = []
            touchdown_airborne_max_gap_series: list[float] = []
            phase_discontinuity_series: list[float] = []
            contact_flip_count = 0

            with _long_drift_variant_context(runtime, variant.name, shared):
                for _ in range(cycles):
                    plan_state = state
                    if variant.name == "dir1_stance_anchor_proxy":
                        anchored_feet = torch.where(
                            shared.prev_contact_state.unsqueeze(-1).to(device=state.foot_pos.device),
                            shared.stance_anchor_w.to(dtype=state.foot_pos.dtype, device=state.foot_pos.device),
                            torch.as_tensor(state.foot_pos, dtype=state.foot_pos.dtype, device=state.foot_pos.device),
                        )
                        plan_state = _clone_mpc_state(state, foot_pos=anchored_feet)

                    result = viewer._plan_viewer_trajectory(
                        terrain=terrain,
                        state=plan_state,
                        command=command,
                        mpc_cfg=runtime.mpc_planner_cfg,
                    )
                    root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
                    foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
                    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
                    rpy = viewer._quat_wxyz_to_rpy(torch.as_tensor(result.root_quat_w, dtype=torch.float64))
                    rel = foot - root.unsqueeze(2)
                    rel_radius_series.append(float(torch.linalg.vector_norm(rel[:, -1], dim=-1).mean().item()))
                    foot_step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
                    foot_step_series.append(float(foot_step.mean().item()))
                    root_dx_series.append(float((root[0, -1, 0] - root[0, 0, 0]).item()))
                    root_dy_series.append(float((root[0, -1, 1] - root[0, 0, 1]).item()))
                    root_dyaw_series.append(float((rpy[0, -1, 2] - rpy[0, 0, 2]).item()))

                    anchor = shared.stance_anchor_w.to(dtype=foot.dtype, device=foot.device)
                    contact_prob = contact.to(dtype=foot.dtype)
                    anchor_err = torch.linalg.vector_norm(foot - anchor.unsqueeze(1), dim=-1)
                    denom = torch.clamp(contact_prob.sum(), min=1.0)
                    stance_anchor_err_series.append(float((anchor_err * contact_prob).sum().item() / denom.item()))
                    touchdown = torch.logical_and(contact[:, 1:], torch.logical_not(contact[:, :-1]))
                    if bool(touchdown.any().item()):
                        td_delta = torch.linalg.vector_norm(foot[:, 1:] - anchor.unsqueeze(1), dim=-1)
                        touchdown_jump_series.append(float(td_delta[touchdown].mean().item()))
                    else:
                        touchdown_jump_series.append(0.0)
                    td_gap_mean, td_airborne_ratio, td_airborne_max_gap = _planned_touchdown_ground_metrics(
                        terrain,
                        result,
                    )
                    touchdown_ground_gap_series.append(td_gap_mean)
                    touchdown_airborne_ratio_series.append(td_airborne_ratio)
                    touchdown_airborne_max_gap_series.append(td_airborne_max_gap)
                    first_contact = contact[:, 0]
                    if shared.prev_contact_first is not None:
                        phase_discontinuity_series.append(float(torch.logical_xor(first_contact, shared.prev_contact_first).float().mean().item()))
                    contact_flip_count += int(torch.count_nonzero(contact[:, 1:] != contact[:, :-1]).item())
                    shared.prev_contact_first = first_contact.detach().clone()

                    frame_idx = result.num_frames - 1
                    viewer._apply_direct_playback_to_robot(runtime.robot, result, frame_idx=frame_idx)
                    runtime.base_env.scene.write_data_to_sim()
                    runtime.base_env.sim.render()
                    runtime.base_env.scene.update(float(runtime.base_env.physics_dt))
                    actual_kin = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
                    plan_foot_last = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64)
                    actual_foot_last = torch.as_tensor(actual_kin["foot_pos_w"], dtype=torch.float64)
                    foot_err = torch.linalg.vector_norm(actual_foot_last - plan_foot_last, dim=-1)
                    foot_err_series.append(float(foot_err.mean().item()))

                    state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
                    last_contact = contact[:, frame_idx].to(dtype=torch.bool, device=state.foot_pos.device)
                    last_foot = torch.as_tensor(state.foot_pos, dtype=torch.float32, device=state.foot_pos.device)
                    touchdown_last = torch.logical_and(last_contact, torch.logical_not(shared.prev_contact_state.to(device=last_contact.device)))
                    update_anchor = torch.logical_or(last_contact, touchdown_last).unsqueeze(-1)
                    shared.stance_anchor_w = torch.where(update_anchor, last_foot, shared.stance_anchor_w.to(device=last_foot.device))
                    shared.prev_contact_state = last_contact.detach().clone()
                    current_rel_body = _initial_foot_rel_body(state)
                    running_rel_body = shared.running_foot_rel_body.to(device=current_rel_body.device)
                    touchdown_mask = touchdown_last.unsqueeze(-1).to(dtype=current_rel_body.dtype, device=current_rel_body.device)
                    contact_mask = last_contact.unsqueeze(-1).to(dtype=current_rel_body.dtype, device=current_rel_body.device)
                    touch_alpha = 0.35
                    contact_alpha = 0.10
                    blended_rel_body = torch.lerp(running_rel_body, current_rel_body, touch_alpha * touchdown_mask)
                    blended_rel_body = torch.lerp(blended_rel_body, current_rel_body, contact_alpha * contact_mask)
                    shared.running_foot_rel_body = blended_rel_body.detach().clone()
                    horizon_phase_advance = (
                        float(runtime.plan_dt)
                        * float(runtime.requested_n_frames)
                        * float(runtime.mpc_planner_cfg.runtime.step_freq)
                    )
                    shared.phase_shift = (float(shared.phase_shift) + horizon_phase_advance) % 1.0

            drift = rel_radius_series[-1] - rel_radius_series[0]
            report = {
                "variant": variant.name,
                "name": name,
                "rel_start": rel_radius_series[0],
                "rel_end": rel_radius_series[-1],
                "drift": drift,
                "abs_drift": abs(drift),
                "foot_err_mean": sum(foot_err_series) / len(foot_err_series),
                "foot_step_mean": sum(foot_step_series) / len(foot_step_series),
                "dx_mean": sum(root_dx_series) / len(root_dx_series),
                "dy_mean": sum(root_dy_series) / len(root_dy_series),
                "dyaw_mean": sum(root_dyaw_series) / len(root_dyaw_series),
                "stance_anchor_error": sum(stance_anchor_err_series) / len(stance_anchor_err_series),
                "touchdown_jump_distance": sum(touchdown_jump_series) / len(touchdown_jump_series),
                "touchdown_ground_gap_mean": sum(touchdown_ground_gap_series) / len(touchdown_ground_gap_series),
                "touchdown_airborne_ratio": sum(touchdown_airborne_ratio_series) / len(touchdown_airborne_ratio_series),
                "touchdown_airborne_max_gap": sum(touchdown_airborne_max_gap_series) / len(touchdown_airborne_max_gap_series),
                "phase_discontinuity": (
                    sum(phase_discontinuity_series) / len(phase_discontinuity_series)
                    if phase_discontinuity_series
                    else 0.0
                ),
                "contact_flip_count": float(contact_flip_count),
            }
            if variant.name == "baseline":
                baseline_by_name[name] = report
                delta_abs = 0.0
            else:
                base = baseline_by_name.get(name)
                delta_abs = float(report["abs_drift"]) - float(base["abs_drift"]) if base is not None else 0.0
            report["delta_abs_drift_vs_baseline"] = delta_abs
            variant_reports.append(report)
            all_reports.append(report)
            print(
                "MPC_LONG_DRIFT_VARIANT "
                f"variant={variant.name} "
                f"name={name} "
                f"cycles={cycles} "
                f"rel_start={report['rel_start']:.4f} "
                f"rel_end={report['rel_end']:.4f} "
                f"drift={report['drift']:+.4f} "
                f"abs_drift={report['abs_drift']:.4f} "
                f"delta_abs_vs_baseline={delta_abs:+.4f} "
                f"foot_err_mean={report['foot_err_mean']:.4f} "
                f"foot_step_mean={report['foot_step_mean']:.4f} "
                f"stance_anchor_error={report['stance_anchor_error']:.4f} "
                f"touchdown_jump_distance={report['touchdown_jump_distance']:.4f} "
                f"touchdown_ground_gap_mean={report['touchdown_ground_gap_mean']:+.4f} "
                f"touchdown_airborne_ratio={report['touchdown_airborne_ratio']:.4f} "
                f"touchdown_airborne_max_gap={report['touchdown_airborne_max_gap']:.4f} "
                f"phase_discontinuity={report['phase_discontinuity']:.4f} "
                f"contact_flip_count={int(report['contact_flip_count'])} "
                f"dx_mean={report['dx_mean']:+.4f} "
                f"dy_mean={report['dy_mean']:+.4f} "
                f"dyaw_mean={report['dyaw_mean']:+.4f}",
                flush=True,
            )

        max_report = max(variant_reports, key=lambda item: float(item["abs_drift"]))
        mean_abs_drift = sum(float(item["abs_drift"]) for item in variant_reports) / len(variant_reports)
        mean_delta = sum(float(item["delta_abs_drift_vs_baseline"]) for item in variant_reports) / len(variant_reports)
        print(
            "MPC_LONG_DRIFT_VARIANT_SUMMARY "
            f"variant={variant.name} "
            f"cycles={cycles} "
            f"mean_abs_drift={mean_abs_drift:.4f} "
            f"mean_delta_abs_vs_baseline={mean_delta:+.4f} "
            f"max_name={max_report['name']} "
            f"max_abs_drift={float(max_report['abs_drift']):.4f}",
            flush=True,
        )

    assert all_reports


@pytest.mark.skipif(
    not _enable_long_drift_sequence_sweep_test(),
    reason="Set MPC_RUNTIME_LONG_DRIFT_SEQUENCE_SWEEP=1 to run mixed-command and sequence long-horizon IsaacLab MPC sweep.",
)
def test_mpc_runtime_long_replan_variant_sequence_sweep(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    viewer = runtime._viewer
    terrain = runtime._single_env_terrain()
    variants = _long_drift_variants()
    commands = _resolve_long_drift_commands(runtime)
    sequences = _sequence_long_drift_specs()
    cycles = int(os.environ.get("MPC_LONG_DRIFT_SEQUENCE_CYCLES", os.environ.get("MPC_LONG_DRIFT_CYCLES", "60")))
    transition_window = int(os.environ.get("MPC_LONG_DRIFT_TRANSITION_WINDOW", "5"))
    baseline_by_segment: dict[tuple[str, str], dict[str, float | str]] = {}
    all_reports: list[dict[str, float | str]] = []

    for variant in variants:
        print(
            "MPC_LONG_DRIFT_SEQUENCE_VARIANT_INFO "
            f"variant={variant.name} "
            f"direction={variant.direction}",
            flush=True,
        )
        for seq_name, segment_names in sequences:
            runtime.reset()
            state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
            shared = SimpleNamespace(
                stance_anchor_w=torch.as_tensor(state.foot_pos, dtype=torch.float32).clone(),
                prev_contact_state=torch.ones((1, 4), dtype=torch.bool, device=state.foot_pos.device),
                phase_shift=0.0,
                last_touchdown_w=torch.as_tensor(state.foot_pos, dtype=torch.float32).clone(),
                initial_foot_rel_body=_initial_foot_rel_body(state),
                running_foot_rel_body=_initial_foot_rel_body(state),
                prev_touchdown_w=None,
                prev_contact_first=None,
            )
            seq_reports: list[dict[str, float | str]] = []

            with _long_drift_variant_context(runtime, variant.name, shared):
                for segment_idx, segment_name in enumerate(segment_names):
                    if segment_name not in commands:
                        raise ValueError(
                            f"Unknown segment command {segment_name!r}; "
                            f"known={sorted(commands)}"
                        )
                    command = commands[segment_name][:1]
                    rel_radius_series: list[float] = []
                    foot_err_series: list[float] = []
                    foot_step_series: list[float] = []
                    root_dx_series: list[float] = []
                    root_dy_series: list[float] = []
                    root_dyaw_series: list[float] = []
                    stance_anchor_err_series: list[float] = []
                    touchdown_jump_series: list[float] = []
                    touchdown_ground_gap_series: list[float] = []
                    touchdown_airborne_ratio_series: list[float] = []
                    touchdown_airborne_max_gap_series: list[float] = []
                    phase_discontinuity_series: list[float] = []
                    transition_foot_err_series: list[float] = []
                    transition_anchor_err_series: list[float] = []
                    contact_flip_count = 0

                    for cycle_idx in range(cycles):
                        plan_state = state
                        if variant.name == "dir1_stance_anchor_proxy":
                            anchored_feet = torch.where(
                                shared.prev_contact_state.unsqueeze(-1).to(device=state.foot_pos.device),
                                shared.stance_anchor_w.to(dtype=state.foot_pos.dtype, device=state.foot_pos.device),
                                torch.as_tensor(state.foot_pos, dtype=state.foot_pos.dtype, device=state.foot_pos.device),
                            )
                            plan_state = _clone_mpc_state(state, foot_pos=anchored_feet)

                        result = viewer._plan_viewer_trajectory(
                        terrain=terrain,
                        state=plan_state,
                        command=command,
                        mpc_cfg=runtime.mpc_planner_cfg,
                        )
                        root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
                        foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
                        contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
                        rpy = viewer._quat_wxyz_to_rpy(torch.as_tensor(result.root_quat_w, dtype=torch.float64))
                        rel = foot - root.unsqueeze(2)
                        rel_radius_series.append(float(torch.linalg.vector_norm(rel[:, -1], dim=-1).mean().item()))
                        foot_step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
                        foot_step_series.append(float(foot_step.mean().item()))
                        root_dx_series.append(float((root[0, -1, 0] - root[0, 0, 0]).item()))
                        root_dy_series.append(float((root[0, -1, 1] - root[0, 0, 1]).item()))
                        root_dyaw_series.append(float((rpy[0, -1, 2] - rpy[0, 0, 2]).item()))

                        anchor = shared.stance_anchor_w.to(dtype=foot.dtype, device=foot.device)
                        contact_prob = contact.to(dtype=foot.dtype)
                        anchor_err = torch.linalg.vector_norm(foot - anchor.unsqueeze(1), dim=-1)
                        denom = torch.clamp(contact_prob.sum(), min=1.0)
                        anchor_err_value = float((anchor_err * contact_prob).sum().item() / denom.item())
                        stance_anchor_err_series.append(anchor_err_value)
                        touchdown = torch.logical_and(contact[:, 1:], torch.logical_not(contact[:, :-1]))
                        if bool(touchdown.any().item()):
                            td_delta = torch.linalg.vector_norm(foot[:, 1:] - anchor.unsqueeze(1), dim=-1)
                            touchdown_jump_series.append(float(td_delta[touchdown].mean().item()))
                        else:
                            touchdown_jump_series.append(0.0)
                        td_gap_mean, td_airborne_ratio, td_airborne_max_gap = _planned_touchdown_ground_metrics(
                            terrain,
                            result,
                        )
                        touchdown_ground_gap_series.append(td_gap_mean)
                        touchdown_airborne_ratio_series.append(td_airborne_ratio)
                        touchdown_airborne_max_gap_series.append(td_airborne_max_gap)
                        first_contact = contact[:, 0]
                        if shared.prev_contact_first is not None:
                            phase_discontinuity_series.append(
                                float(torch.logical_xor(first_contact, shared.prev_contact_first).float().mean().item())
                            )
                        contact_flip_count += int(torch.count_nonzero(contact[:, 1:] != contact[:, :-1]).item())
                        shared.prev_contact_first = first_contact.detach().clone()

                        frame_idx = result.num_frames - 1
                        viewer._apply_direct_playback_to_robot(runtime.robot, result, frame_idx=frame_idx)
                        runtime.base_env.scene.write_data_to_sim()
                        runtime.base_env.sim.render()
                        runtime.base_env.scene.update(float(runtime.base_env.physics_dt))
                        actual_kin = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
                        plan_foot_last = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64)
                        actual_foot_last = torch.as_tensor(actual_kin["foot_pos_w"], dtype=torch.float64)
                        foot_err = torch.linalg.vector_norm(actual_foot_last - plan_foot_last, dim=-1)
                        foot_err_value = float(foot_err.mean().item())
                        foot_err_series.append(foot_err_value)
                        if cycle_idx < transition_window:
                            transition_foot_err_series.append(foot_err_value)
                            transition_anchor_err_series.append(anchor_err_value)

                        state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
                        last_contact = contact[:, frame_idx].to(dtype=torch.bool, device=state.foot_pos.device)
                        last_foot = torch.as_tensor(state.foot_pos, dtype=torch.float32, device=state.foot_pos.device)
                        touchdown_last = torch.logical_and(
                            last_contact,
                            torch.logical_not(shared.prev_contact_state.to(device=last_contact.device)),
                        )
                        update_anchor = torch.logical_or(last_contact, touchdown_last).unsqueeze(-1)
                        shared.stance_anchor_w = torch.where(
                            update_anchor,
                            last_foot,
                            shared.stance_anchor_w.to(device=last_foot.device),
                        )
                        shared.prev_contact_state = last_contact.detach().clone()
                        current_rel_body = _initial_foot_rel_body(state)
                        running_rel_body = shared.running_foot_rel_body.to(device=current_rel_body.device)
                        touchdown_mask = touchdown_last.unsqueeze(-1).to(
                            dtype=current_rel_body.dtype,
                            device=current_rel_body.device,
                        )
                        contact_mask = last_contact.unsqueeze(-1).to(
                            dtype=current_rel_body.dtype,
                            device=current_rel_body.device,
                        )
                        touch_alpha = 0.35
                        contact_alpha = 0.10
                        blended_rel_body = torch.lerp(running_rel_body, current_rel_body, touch_alpha * touchdown_mask)
                        blended_rel_body = torch.lerp(blended_rel_body, current_rel_body, contact_alpha * contact_mask)
                        shared.running_foot_rel_body = blended_rel_body.detach().clone()
                        horizon_phase_advance = (
                            float(runtime.plan_dt)
                            * float(runtime.requested_n_frames)
                            * float(runtime.mpc_planner_cfg.runtime.step_freq)
                        )
                        shared.phase_shift = (float(shared.phase_shift) + horizon_phase_advance) % 1.0

                    report = _segment_metrics_summary(
                        variant_name=variant.name,
                        seq_name=seq_name,
                        segment_name=segment_name,
                        cycles=cycles,
                        rel_radius_series=rel_radius_series,
                        foot_err_series=foot_err_series,
                        foot_step_series=foot_step_series,
                        root_dx_series=root_dx_series,
                        root_dy_series=root_dy_series,
                        root_dyaw_series=root_dyaw_series,
                        stance_anchor_err_series=stance_anchor_err_series,
                        touchdown_jump_series=touchdown_jump_series,
                        touchdown_ground_gap_series=touchdown_ground_gap_series,
                        touchdown_airborne_ratio_series=touchdown_airborne_ratio_series,
                        touchdown_airborne_max_gap_series=touchdown_airborne_max_gap_series,
                        phase_discontinuity_series=phase_discontinuity_series,
                        transition_foot_err_series=transition_foot_err_series,
                        transition_anchor_err_series=transition_anchor_err_series,
                        contact_flip_count=contact_flip_count,
                    )
                    baseline_key = (seq_name, segment_name)
                    if variant.name == "baseline":
                        baseline_by_segment[baseline_key] = report
                        delta_abs = 0.0
                        delta_transition = 0.0
                    else:
                        base = baseline_by_segment.get(baseline_key)
                        delta_abs = float(report["abs_drift"]) - float(base["abs_drift"]) if base is not None else 0.0
                        delta_transition = (
                            float(report["transition_foot_err_mean"]) - float(base["transition_foot_err_mean"])
                            if base is not None
                            else 0.0
                        )
                    report["delta_abs_drift_vs_baseline"] = delta_abs
                    report["delta_transition_foot_err_vs_baseline"] = delta_transition
                    seq_reports.append(report)
                    all_reports.append(report)
                    print(
                        "MPC_LONG_DRIFT_SEQUENCE "
                        f"variant={variant.name} "
                        f"seq={seq_name} "
                        f"segment={segment_name} "
                        f"cycles={cycles} "
                        f"rel_start={report['rel_start']:.4f} "
                        f"rel_end={report['rel_end']:.4f} "
                        f"drift={report['drift']:+.4f} "
                        f"abs_drift={report['abs_drift']:.4f} "
                        f"delta_abs_vs_baseline={delta_abs:+.4f} "
                        f"foot_err_mean={report['foot_err_mean']:.4f} "
                        f"transition_foot_err_mean={report['transition_foot_err_mean']:.4f} "
                        f"delta_transition_foot_err_vs_baseline={delta_transition:+.4f} "
                        f"foot_step_mean={report['foot_step_mean']:.4f} "
                        f"stance_anchor_error={report['stance_anchor_error']:.4f} "
                        f"transition_anchor_error_mean={report['transition_anchor_error_mean']:.4f} "
                        f"touchdown_jump_distance={report['touchdown_jump_distance']:.4f} "
                        f"touchdown_ground_gap_mean={report['touchdown_ground_gap_mean']:+.4f} "
                        f"touchdown_airborne_ratio={report['touchdown_airborne_ratio']:.4f} "
                        f"touchdown_airborne_max_gap={report['touchdown_airborne_max_gap']:.4f} "
                        f"phase_discontinuity={report['phase_discontinuity']:.4f} "
                        f"contact_flip_count={int(report['contact_flip_count'])} "
                        f"dx_mean={report['dx_mean']:+.4f} "
                        f"dy_mean={report['dy_mean']:+.4f} "
                        f"dyaw_mean={report['dyaw_mean']:+.4f}",
                        flush=True,
                    )

            seq_max = max(seq_reports, key=lambda item: float(item["abs_drift"]))
            seq_mean_abs = sum(float(item["abs_drift"]) for item in seq_reports) / len(seq_reports)
            seq_mean_transition = sum(float(item["transition_foot_err_mean"]) for item in seq_reports) / len(seq_reports)
            print(
                "MPC_LONG_DRIFT_SEQUENCE_SUMMARY "
                f"variant={variant.name} "
                f"seq={seq_name} "
                f"cycles={cycles} "
                f"mean_abs_drift={seq_mean_abs:.4f} "
                f"mean_transition_foot_err={seq_mean_transition:.4f} "
                f"max_segment={seq_max['segment']} "
                f"max_abs_drift={float(seq_max['abs_drift']):.4f}",
                flush=True,
            )

    assert all_reports


def test_mpc_runtime_viewer_playback_kinematics_consistency(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    forward = runtime.plan_case("forward")
    frame_idx = min(7, forward.result.num_frames - 1)
    readback = runtime.playback_sync_authoritative_readback(forward.result, frame_idx=frame_idx)

    torch.testing.assert_close(
        readback.root_pos_w,
        torch.as_tensor(forward.result.root_pos_w[:, frame_idx], dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        readback.joint_pos,
        torch.as_tensor(forward.result.joint_angles[:, frame_idx], dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )

    foot_ids = runtime.foot_ids[:4]
    foot_actual = torch.as_tensor(runtime.robot.data.body_pos_w[:1, foot_ids, :], dtype=torch.float64).clone()
    foot_plan = torch.as_tensor(forward.result.foot_pos_w[:, frame_idx], dtype=torch.float64).clone()
    foot_err_norm = torch.linalg.vector_norm(foot_actual - foot_plan, dim=-1)

    # Regression guardrail: this catches the joint-order mismatch bug that can
    # produce decimeter-level "flying feet" during viewer playback.
    assert float(foot_err_norm.max().item()) < 0.12
    assert float(foot_err_norm.mean().item()) < 0.08


def test_mpc_runtime_diagnostics_layer_emits_hard_mask_when_enabled(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    original_enabled = bool(runtime.mpc_planner_cfg.diagnostics.enabled)
    runtime.mpc_planner_cfg.diagnostics.enabled = True
    try:
        forward = runtime.plan_case("forward")
    finally:
        runtime.mpc_planner_cfg.diagnostics.enabled = original_enabled

    assert forward.result.hard_reason_mask is not None
    assert forward.result.hard_reason_mask.dtype == torch.bool
    assert tuple(forward.result.hard_reason_mask.shape[:1]) == (1,)
    assert forward.result.status is not None
    assert torch.as_tensor(forward.result.status).numel() == 1


def test_mpc_runtime_4096_headless_global_sync_sample_counters(real_semantic_mpc_runtime_4096):
    runtime = real_semantic_mpc_runtime_4096
    manager = runtime.base_env._trajectory_manager

    runtime.mpc_planner_cfg.diagnostics.emit_runtime_counters = True
    runtime.mpc_planner_cfg.diagnostics.profile_cuda_sync = False
    runtime.mpc_planner_cfg.runtime.optimize_steps = 0
    runtime.mpc_planner_cfg.runtime.parallel_plan_batch_size = 64
    runtime.mpc_planner_cfg.runtime.max_stale_steps = 100

    runtime.base_env.common_step_counter = int(getattr(runtime.base_env, "common_step_counter", 0)) + 1
    manager.refresh_from_env(runtime.base_env)
    first = manager.runtime_counters()
    print("MPC_4096_COUNTERS_FIRST", first, flush=True)

    assert first["num_envs"] == 4096
    assert first["global_due"] is True
    assert first["global_due_count"] == 4096
    assert first["sampled_plan_count"] == 64
    assert first["max_stale_observed"] >= 0
    assert first["planner_ms"] >= 0.0
    assert first["cache_ms"] >= 0.0

    command_dirty_mask = torch.zeros((4096,), dtype=torch.bool, device=runtime.base_env.device)
    command_dirty_mask[:128] = True
    manager.mark_command_changed(command_dirty_mask)
    assert not bool(torch.any(manager.reference_reward_mask()[:128]).item())
    runtime.base_env.common_step_counter = int(getattr(runtime.base_env, "common_step_counter", 0)) + 1
    manager.refresh_from_env(runtime.base_env)
    second = manager.runtime_counters()
    print("MPC_4096_COUNTERS_SECOND", second, flush=True)

    assert second["global_due"] is False
    assert second["sampled_plan_count"] == 0
    assert second["global_due_count"] == 0


def test_mpc_semantic_runtime_4096_collect_data_under_10s(real_semantic_mpc_runtime_4096):
    runtime = real_semantic_mpc_runtime_4096
    manager = runtime.base_env._trajectory_manager

    assert runtime.task_id == "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0"
    assert runtime.env_cfg.__class__.__name__ == "TeacherElevationTrajectoryMpcSemanticEnvCfg"
    assert runtime.planner_backend == "mpc"
    assert runtime.scanner_name == "semantic_height_scanner"
    assert runtime.base_env.num_envs == 4096

    runtime.mpc_planner_cfg.diagnostics.emit_runtime_counters = True
    runtime.mpc_planner_cfg.diagnostics.profile_cuda_sync = False
    runtime.mpc_planner_cfg.runtime.parallel_plan_batch_size = 64

    actions = torch.zeros(
        (runtime.base_env.num_envs, runtime.base_env.action_manager.total_action_dim),
        dtype=torch.float32,
        device=runtime.base_env.device,
    )
    if runtime.base_env.device.type == "cuda":
        torch.cuda.synchronize(runtime.base_env.device)
    start = torch.cuda.Event(enable_timing=True) if runtime.base_env.device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if runtime.base_env.device.type == "cuda" else None
    import time

    wall_start = time.perf_counter()
    if start is not None:
        start.record()
    for _ in range(24):
        runtime.base_env.step(actions)
    if end is not None:
        end.record()
        torch.cuda.synchronize(runtime.base_env.device)
        collect_time_s = float(start.elapsed_time(end)) / 1000.0
    else:
        collect_time_s = time.perf_counter() - wall_start
    counters = manager.runtime_counters()
    metrics = {"collect_time_s": collect_time_s, **counters}
    metrics_path = os.environ.get("T302G_4096_METRICS_JSON", "").strip()
    if metrics_path:
        import json

        Path(metrics_path).write_text(json.dumps(metrics, sort_keys=True, indent=2))
    print("MPC_SEMANTIC_4096_COLLECT_DATA", metrics, flush=True)

    assert collect_time_s < 10.0
    assert counters.get("sampled_plan_count", 0) <= runtime.mpc_planner_cfg.runtime.parallel_plan_batch_size
