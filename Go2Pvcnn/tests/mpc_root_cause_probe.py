from __future__ import annotations

import json
import math
import os
import sys
import traceback
import copy
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from extension.batch_mpc_planner.config import MpcRuntimeCfg
from extension.batch_mpc_planner.kinematics import _JOINT_LIMITS
from extension.batch_mpc_planner.nominal import build_nominal_trajectory
from extension.batched_planner.ik import batch_forward_kinematics
from extension.convention import euler_to_quat_batch
from Go2Pvcnn.tests.fixtures.viewer_runtime_diagnostics import make_real_runtime_fixture
from Go2Pvcnn.tests.mpc_yaw_gait_failure_probe import (
    CUSTOM_COMMANDS,
    _body_relative_foot,
    _command_tensor,
    _pair_left_right_alternation_stats,
    _sample_ground_height,
    _stance_ground_metrics_from_frame,
)


def _command_sequences() -> tuple[tuple[str, tuple[str, ...]], ...]:
    default = (
        "forward:forward;"
        "backward:backward;"
        "forward_speeds:forward_slow,forward,forward_fast;"
        "backward_speeds:backward_slow,backward,backward_fast;"
        "yaw_speeds:yaw_left_slow,yaw_left,yaw_left_fast,yaw_right_slow,yaw_right,yaw_right_fast;"
        "yaw_switch:yaw_left,yaw_right,yaw_left;"
        "mixed_yaw:forward_yaw_left,forward_yaw_right,lateral_left_yaw_right,lateral_right_yaw_left"
    )
    requested = os.environ.get("MPC_ROOT_CAUSE_SEQUENCES", default)
    out: list[tuple[str, tuple[str, ...]]] = []
    for raw in requested.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        name, sep, segments = raw.partition(":")
        if not sep:
            raise ValueError(f"bad sequence {raw!r}; expected name:a,b,c")
        out.append((name.strip(), tuple(part.strip() for part in segments.split(",") if part.strip())))
    return tuple(out)


def _variants() -> tuple[tuple[str, float | None], ...]:
    requested = os.environ.get("MPC_ROOT_CAUSE_VARIANTS", "baseline:none;ikfk:4.0")
    out: list[tuple[str, float | None]] = []
    for raw in requested.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        name, sep, value = raw.partition(":")
        if not sep or value.strip().lower() in {"none", "off", "baseline"}:
            out.append((name.strip(), None))
        else:
            out.append((name.strip(), float(value)))
    return tuple(out)


def _apply_variant(runtime, weight: float | None) -> None:
    cfg = copy.deepcopy(runtime.mpc_planner_cfg)
    cfg.losses.ik_fk_residual.enabled = weight is not None
    if weight is not None:
        cfg.losses.ik_fk_residual.weight = float(weight)
    runtime.mpc_planner_cfg = cfg


def _joint_limit_stats(joint: torch.Tensor) -> dict[str, float]:
    q = torch.as_tensor(joint, dtype=torch.float64)
    limits = _JOINT_LIMITS.to(dtype=q.dtype, device=q.device)
    lower = limits[:, 0].view(1, 1, 12)
    upper = limits[:, 1].view(1, 1, 12)
    span = upper - lower
    margin = torch.minimum(q - lower, upper - q)
    near = margin < 0.025
    sat = torch.logical_or(torch.isclose(q, lower, atol=1.0e-5, rtol=0.0), torch.isclose(q, upper, atol=1.0e-5, rtol=0.0))
    norm_margin = margin / torch.clamp(span, min=1.0e-6)
    return {
        "joint_limit_near_ratio": float(near.to(dtype=torch.float64).mean().item()),
        "joint_limit_saturation_ratio": float(sat.to(dtype=torch.float64).mean().item()),
        "joint_limit_margin_min": float(margin.min().item()),
        "joint_limit_norm_margin_mean": float(norm_margin.mean().item()),
    }


def _fk_foot_from_result(result) -> torch.Tensor:
    root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
    quat = torch.as_tensor(result.root_quat_w, dtype=torch.float64)
    joint = torch.as_tensor(result.joint_angles, dtype=torch.float64)
    batch, horizon = int(root.shape[0]), int(root.shape[1])
    fk_body = batch_forward_kinematics(
        root.reshape(batch * horizon, 3),
        quat.reshape(batch * horizon, 4),
        joint.reshape(batch * horizon, 12),
    ).reshape(batch, horizon, 12, 3)
    return fk_body[:, :, 8:12, :]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row if isinstance(row.get(key), (int, float))})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_max"] = _max(vals)
    return out


def _nominal_metrics(state, command: torch.Tensor, terrain, runtime_cfg: MpcRuntimeCfg) -> dict[str, float]:
    nominal = build_nominal_trajectory(state, command, terrain, runtime_cfg)
    root = torch.as_tensor(nominal["root_pos"], dtype=torch.float64)
    rpy = torch.as_tensor(nominal["root_rpy"], dtype=torch.float64)
    quat = euler_to_quat_batch(rpy[..., 0], rpy[..., 1], rpy[..., 2]).to(dtype=torch.float64)
    foot = torch.as_tensor(nominal["foot_pos"], dtype=torch.float64)
    contact = torch.as_tensor(nominal["contact_prior"] > 0.5, dtype=torch.bool)
    rel = _body_relative_foot(root, quat, foot)
    pair = _pair_left_right_alternation_stats(rel, contact, front_eps=0.01)
    cmd = torch.as_tensor(command, dtype=torch.float64)
    lin_speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    yaw_abs = torch.abs(cmd[:, 2])
    swing_stride_active = (lin_speed > 0.08).to(dtype=torch.float64)
    yaw_dom = yaw_abs / torch.clamp(lin_speed + yaw_abs, min=1.0e-6)
    pair.update(
        {
            "nominal_yaw_dom": float(yaw_dom.mean().item()),
            "nominal_lin_speed": float(lin_speed.mean().item()),
            "nominal_yaw_abs": float(yaw_abs.mean().item()),
            "nominal_swing_stride_active": float(swing_stride_active.mean().item()),
        }
    )
    return pair


def _plan_with_memory_exposed(runtime, terrain, state, command, memory):
    viewer = runtime._viewer
    result = viewer._plan_viewer_trajectory(
        backend=runtime.planner_backend,
        terrain=terrain,
        state=state,
        command=command,
        requested_n_frames=runtime.requested_n_frames,
        dt=runtime.plan_dt,
        legacy_cfg=runtime.planner_cfg,
        together_cfg=runtime.together_planner_cfg,
        mpc_cfg=runtime.mpc_planner_cfg,
    )
    return result, memory, None


def _runtime_metrics(runtime, terrain, state_before, command, memory, result, plan_memory) -> dict[str, float]:
    frame_idx = int(result.num_frames) - 1
    plan_root = torch.as_tensor(result.root_pos_w[:, frame_idx], dtype=torch.float64)
    plan_quat = torch.as_tensor(result.root_quat_w[:, frame_idx], dtype=torch.float64)
    plan_joint = torch.as_tensor(result.joint_angles[:, frame_idx], dtype=torch.float64)
    plan_foot = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64)
    contact = torch.as_tensor(result.contact_state[:, frame_idx], dtype=torch.bool)

    fk_foot = _fk_foot_from_result(result)
    fk_last = fk_foot[:, frame_idx].to(dtype=torch.float64)
    ik_fk_err = torch.linalg.vector_norm(fk_last - plan_foot, dim=-1)

    viewer = runtime._viewer
    viewer._apply_direct_playback_to_robot(runtime.robot, result, frame_idx=frame_idx)
    runtime.base_env.scene.write_data_to_sim()
    runtime.base_env.sim.render()
    runtime.base_env.scene.update(float(runtime.base_env.physics_dt))
    actual_kin = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
    actual_base = viewer._read_actual_base_state(runtime.base_env)
    actual_root = torch.as_tensor(actual_base["root_pos_w"], dtype=torch.float64)
    actual_joint = torch.as_tensor(actual_kin["joint_pos_planner"], dtype=torch.float64)
    actual_foot = torch.as_tensor(actual_kin["foot_pos_w"], dtype=torch.float64)

    out = {
        "ik_fk_foot_err_mean": float(ik_fk_err.mean().item()),
        "ik_fk_foot_err_contact_mean": float(ik_fk_err[contact].mean().item()) if bool(contact.any().item()) else 0.0,
        "actual_plan_foot_err_mean": float(torch.linalg.vector_norm(actual_foot - plan_foot, dim=-1).mean().item()),
        "actual_plan_foot_err_contact_mean": float(torch.linalg.vector_norm(actual_foot - plan_foot, dim=-1)[contact].mean().item())
        if bool(contact.any().item())
        else 0.0,
        "actual_fk_foot_err_mean": float(torch.linalg.vector_norm(actual_foot - fk_last, dim=-1).mean().item()),
        "actual_plan_root_err": float(torch.linalg.vector_norm(actual_root - plan_root, dim=-1).mean().item()),
        "actual_plan_joint_err_mean": float(torch.abs(actual_joint - plan_joint).mean().item()),
        "actual_plan_joint_err_max": float(torch.abs(actual_joint - plan_joint).max().item()),
    }
    out.update(_joint_limit_stats(torch.as_tensor(result.joint_angles, dtype=torch.float64)))
    out.update(
        _stance_ground_metrics_from_frame(
            terrain,
            actual_foot,
            contact,
            tol=0.02,
            prefix="actual_last",
        )
    )
    out.update(
        _stance_ground_metrics_from_frame(
            terrain,
            plan_foot,
            contact,
            tol=0.02,
            prefix="plan_last",
        )
    )
    return out


def main() -> int:
    output_path = Path(os.environ.get("MPC_ROOT_CAUSE_OUTPUT", "/tmp/mpc_root_cause_probe.jsonl"))
    cycles = int(os.environ.get("MPC_ROOT_CAUSE_CYCLES", "12"))
    device = os.environ.get("MPC_TEST_DEVICE", "cuda:2")
    runtime = None
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            runtime = make_real_runtime_fixture(num_envs=2, planner_backend="mpc", device=device)
            viewer = runtime._viewer
            terrain = runtime._single_env_terrain()
            sequences = _command_sequences()
            variants = _variants()
            handle.write(json.dumps({"kind": "startup", "cycles": cycles, "device": device, "sequences": sequences, "variants": variants}) + "\n")
            for variant_name, ikfk_weight in variants:
                _apply_variant(runtime, ikfk_weight)
                for seq_name, segments in sequences:
                    runtime.reset()
                    state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
                    memory = None
                    for segment in segments:
                        command = _command_tensor(runtime, segment)
                        nominal_rows: list[dict[str, float]] = []
                        runtime_rows: list[dict[str, float]] = []
                        for cycle_idx in range(cycles):
                            nominal_metrics = _nominal_metrics(state, command, terrain, runtime.mpc_planner_cfg.runtime)
                            nominal_rows.append(nominal_metrics)
                            result, memory, plan_memory = _plan_with_memory_exposed(runtime, terrain, state, command, memory)
                            metrics = _runtime_metrics(runtime, terrain, state, command, memory, result, plan_memory)
                            runtime_rows.append(metrics)
                            handle.write(json.dumps({"kind": "cycle", "variant": variant_name, "seq": seq_name, "segment": segment, "cycle": cycle_idx, **nominal_metrics, **metrics}, ensure_ascii=False) + "\n")
                            state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
                        handle.write(json.dumps({"kind": "nominal_segment", "variant": variant_name, "seq": seq_name, "segment": segment, "cycles": cycles, **_summary(nominal_rows)}, ensure_ascii=False) + "\n")
                        handle.write(json.dumps({"kind": "runtime_segment", "variant": variant_name, "seq": seq_name, "segment": segment, "cycles": cycles, **_summary(runtime_rows)}, ensure_ascii=False) + "\n")
                        handle.flush()
        print(output_path)
        return 0
    except Exception as exc:  # noqa: BLE001
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "kind": "exception",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False) + "\n")
        print(output_path)
        return 1
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
