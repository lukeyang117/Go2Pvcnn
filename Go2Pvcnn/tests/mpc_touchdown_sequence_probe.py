from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from Go2Pvcnn.tests.test_mpc_runtime_headless import (
    _initial_foot_rel_body,
    _long_drift_variant_context,
    _long_drift_variants,
    _planned_stance_ground_metrics,
    _make_real_runtime_fixture,
    _planned_touchdown_event_ground_metrics,
    _planned_touchdown_ground_metrics,
    _resolve_long_drift_commands,
    _sequence_long_drift_specs,
)


def _viewer_plan_with_memory(runtime, viewer, terrain, state, command, memory):
    result = viewer._plan_viewer_trajectory(
                terrain=terrain,
                state=state,
                command=command,
                mpc_cfg=runtime.mpc_planner_cfg,
    )
    return result, memory


def _contact_pattern_metrics(contact: torch.Tensor) -> dict[str, float]:
    contact = torch.as_tensor(contact, dtype=torch.bool)
    swing = torch.logical_not(contact)
    swing_count = swing.sum(dim=-1).to(dtype=torch.float32)
    two_swing = swing_count == 2
    if bool(two_swing.any().item()):
        fl, fr, rl, rr = swing[..., 0], swing[..., 1], swing[..., 2], swing[..., 3]
        diag = torch.logical_or(fl & rr, fr & rl) & two_swing
        lateral = torch.logical_or(fl & rl, fr & rr) & two_swing
        front_hind = torch.logical_or(fl & fr, rl & rr) & two_swing
        denom = torch.clamp(two_swing.to(dtype=torch.float32).sum(), min=1.0)
        diag_ratio = float(diag.to(dtype=torch.float32).sum().item() / denom.item())
        lateral_ratio = float(lateral.to(dtype=torch.float32).sum().item() / denom.item())
        front_hind_ratio = float(front_hind.to(dtype=torch.float32).sum().item() / denom.item())
    else:
        diag_ratio = 0.0
        lateral_ratio = 0.0
        front_hind_ratio = 0.0
    total = float(swing_count.numel())
    return {
        "swing_count_mean": float(swing_count.mean().item()),
        "swing_count_max": float(swing_count.max().item()),
        "single_swing_ratio": float((swing_count == 1).to(dtype=torch.float32).sum().item() / total),
        "two_swing_ratio": float((swing_count == 2).to(dtype=torch.float32).sum().item() / total),
        "triple_or_more_swing_ratio": float((swing_count >= 3).to(dtype=torch.float32).sum().item() / total),
        "all_stance_ratio": float((swing_count == 0).to(dtype=torch.float32).sum().item() / total),
        "diagonal_swing_pair_ratio": diag_ratio,
        "lateral_swing_pair_ratio": lateral_ratio,
        "front_hind_swing_pair_ratio": front_hind_ratio,
    }


def main() -> int:
    output_path = Path("/tmp/mpc_joint_metrics.jsonl")
    runtime = None
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "startup", "stage": "file_opened"}, ensure_ascii=False) + "\n")
            handle.flush()
            runtime = _make_real_runtime_fixture(num_envs=2, planner_backend="mpc", device="cuda:2")
            handle.write(json.dumps({"kind": "startup", "stage": "runtime_created"}, ensure_ascii=False) + "\n")
            handle.flush()

            viewer = runtime._viewer
            terrain = runtime._single_env_terrain()
            commands = _resolve_long_drift_commands(runtime)
            sequences = _sequence_long_drift_specs()
            requested_variants = [variant.name for variant in _long_drift_variants()]
            cycles = int(os.environ.get("MPC_PROBE_CYCLES", os.environ.get("MPC_LONG_DRIFT_SEQUENCE_CYCLES", "20")))
            transition_window = int(os.environ.get("MPC_PROBE_TRANSITION_WINDOW", os.environ.get("MPC_LONG_DRIFT_TRANSITION_WINDOW", "5")))
            max_sequences = int(os.environ.get("MPC_PROBE_MAX_SEQUENCES", "0"))
            if max_sequences > 0:
                sequences = sequences[:max_sequences]

            handle.write(json.dumps({
                "kind": "startup",
                "stage": "config_ready",
                "variants": requested_variants,
                "sequences": [name for name, _ in sequences],
                "cycles": cycles,
                "transition_window": transition_window,
            }, ensure_ascii=False) + "\n")
            handle.flush()

            for variant_name in requested_variants:
                handle.write(json.dumps({"kind": "variant", "variant": variant_name}, ensure_ascii=False) + "\n")
                handle.flush()
                for seq_name, segment_names in sequences:
                    handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "stage": "reset_start"}, ensure_ascii=False) + "\n")
                    handle.flush()
                    runtime.reset()
                    handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "stage": "reset_done"}, ensure_ascii=False) + "\n")
                    handle.flush()
                    handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "stage": "state_read_start"}, ensure_ascii=False) + "\n")
                    handle.flush()
                    state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
                    handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "stage": "state_read_done"}, ensure_ascii=False) + "\n")
                    handle.flush()
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
                    mpc_foothold_memory = None
                    seq_reports = []
                    with _long_drift_variant_context(runtime, variant_name, shared):
                        for segment_name in segment_names:
                            handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "stage": "segment_start"}, ensure_ascii=False) + "\n")
                            handle.flush()
                            command = commands[segment_name][:1]
                            rel_radius_series = []
                            foot_err_series = []
                            root_err_series = []
                            root_rel_foot_err_series = []
                            foot_step_series = []
                            root_dx_series = []
                            root_dy_series = []
                            root_dyaw_series = []
                            stance_anchor_err_series = []
                            touchdown_jump_series = []
                            td_gap_series = []
                            td_airborne_ratio_series = []
                            td_airborne_max_gap_series = []
                            td_event_gap_series = []
                            td_event_airborne_ratio_series = []
                            td_event_airborne_max_gap_series = []
                            td_event_valid_ratio_series = []
                            stance_gap_series = []
                            stance_airborne_ratio_series = []
                            stance_airborne_max_gap_series = []
                            phase_discontinuity_series = []
                            contact_flip_count = 0
                            transition_foot_err_series = []
                            foot_step_max_series = []
                            touchdown_jump_max_series = []
                            contact_pattern_series = []

                            for cycle_idx in range(cycles):
                                handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "cycle": cycle_idx, "stage": "plan_start"}, ensure_ascii=False) + "\n")
                                handle.flush()
                                result, mpc_foothold_memory = _viewer_plan_with_memory(
                                    runtime,
                                    viewer,
                                    terrain,
                                    state,
                                    command,
                                    mpc_foothold_memory,
                                )
                                handle.write(json.dumps({
                                    "kind": "progress",
                                    "variant": variant_name,
                                    "seq": seq_name,
                                    "segment": segment_name,
                                    "cycle": cycle_idx,
                                    "stage": "plan_done",
                                    "planned_touchdown_shape": tuple(result.planned_touchdown_w.shape),
                                    "num_frames": int(result.num_frames),
                                }, ensure_ascii=False) + "\n")
                                handle.flush()
                                root = torch.as_tensor(result.root_pos_w, dtype=torch.float64)
                                foot = torch.as_tensor(result.foot_pos_w, dtype=torch.float64)
                                contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=foot.device)
                                rpy = viewer._quat_wxyz_to_rpy(torch.as_tensor(result.root_quat_w, dtype=torch.float64))
                                rel = foot - root.unsqueeze(2)
                                rel_radius_series.append(float(torch.linalg.vector_norm(rel[:, -1], dim=-1).mean().item()))
                                foot_step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
                                foot_step_series.append(float(foot_step.mean().item()))
                                foot_step_max_series.append(float(foot_step.max().item()))
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
                                    touchdown_jump_max_series.append(float(td_delta[touchdown].max().item()))
                                else:
                                    touchdown_jump_series.append(0.0)
                                    touchdown_jump_max_series.append(0.0)
                                contact_pattern_series.append(_contact_pattern_metrics(contact))

                                handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "cycle": cycle_idx, "stage": "touchdown_metric_start"}, ensure_ascii=False) + "\n")
                                handle.flush()
                                td_gap_mean, td_air_ratio, td_air_max = _planned_touchdown_ground_metrics(terrain, result)
                                handle.write(json.dumps({
                                    "kind": "progress",
                                    "variant": variant_name,
                                    "seq": seq_name,
                                    "segment": segment_name,
                                    "cycle": cycle_idx,
                                    "stage": "touchdown_metric_done",
                                    "touchdown_ground_gap_mean": td_gap_mean,
                                    "touchdown_airborne_ratio": td_air_ratio,
                                    "touchdown_airborne_max_gap": td_air_max,
                                }, ensure_ascii=False) + "\n")
                                handle.flush()
                                td_gap_series.append(td_gap_mean)
                                td_airborne_ratio_series.append(td_air_ratio)
                                td_airborne_max_gap_series.append(td_air_max)
                                td_event_gap_mean, td_event_air_ratio, td_event_air_max, td_event_valid_ratio = (
                                    _planned_touchdown_event_ground_metrics(terrain, result)
                                )
                                td_event_gap_series.append(td_event_gap_mean)
                                td_event_airborne_ratio_series.append(td_event_air_ratio)
                                td_event_airborne_max_gap_series.append(td_event_air_max)
                                td_event_valid_ratio_series.append(td_event_valid_ratio)
                                stance_gap_mean, stance_air_ratio, stance_air_max = _planned_stance_ground_metrics(
                                    terrain,
                                    result,
                                )
                                stance_gap_series.append(stance_gap_mean)
                                stance_airborne_ratio_series.append(stance_air_ratio)
                                stance_airborne_max_gap_series.append(stance_air_max)
                                first_contact = contact[:, 0]
                                if shared.prev_contact_first is not None:
                                    phase_discontinuity_series.append(
                                        float(torch.logical_xor(first_contact, shared.prev_contact_first).float().mean().item())
                                    )
                                contact_flip_count += int(torch.count_nonzero(contact[:, 1:] != contact[:, :-1]).item())
                                shared.prev_contact_first = first_contact.detach().clone()

                                frame_idx = result.num_frames - 1
                                handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "cycle": cycle_idx, "stage": "playback_start", "frame_idx": int(frame_idx)}, ensure_ascii=False) + "\n")
                                handle.flush()
                                viewer._apply_direct_playback_to_robot(runtime.robot, result, frame_idx=frame_idx)
                                runtime.base_env.scene.write_data_to_sim()
                                runtime.base_env.sim.render()
                                runtime.base_env.scene.update(float(runtime.base_env.physics_dt))
                                handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "cycle": cycle_idx, "stage": "playback_done"}, ensure_ascii=False) + "\n")
                                handle.flush()
                                handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "cycle": cycle_idx, "stage": "actual_read_start"}, ensure_ascii=False) + "\n")
                                handle.flush()
                                actual_kin = viewer._read_actual_kinematic_state(runtime.base_env, runtime.foot_ids.tolist())
                                handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "cycle": cycle_idx, "stage": "actual_read_done"}, ensure_ascii=False) + "\n")
                                handle.flush()
                                actual_base = viewer._read_actual_base_state(runtime.base_env)
                                plan_root_last = torch.as_tensor(result.root_pos_w[:, frame_idx], dtype=torch.float64)
                                actual_root_last = torch.as_tensor(actual_base["root_pos_w"], dtype=torch.float64)
                                root_err = float(torch.linalg.vector_norm(actual_root_last - plan_root_last, dim=-1).mean().item())
                                root_err_series.append(root_err)
                                plan_foot_last = torch.as_tensor(result.foot_pos_w[:, frame_idx], dtype=torch.float64)
                                actual_foot_last = torch.as_tensor(actual_kin["foot_pos_w"], dtype=torch.float64)
                                foot_err = float(torch.linalg.vector_norm(actual_foot_last - plan_foot_last, dim=-1).mean().item())
                                foot_err_series.append(foot_err)
                                plan_rel_last = plan_foot_last - plan_root_last.unsqueeze(1)
                                actual_rel_last = actual_foot_last - actual_root_last.unsqueeze(1)
                                root_rel_foot_err = float(torch.linalg.vector_norm(actual_rel_last - plan_rel_last, dim=-1).mean().item())
                                root_rel_foot_err_series.append(root_rel_foot_err)
                                if cycle_idx < transition_window:
                                    transition_foot_err_series.append(foot_err)

                                handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "cycle": cycle_idx, "stage": "state_refresh_start"}, ensure_ascii=False) + "\n")
                                handle.flush()
                                state = viewer._mpc_state_from_env(runtime.base_env, runtime.foot_ids.tolist())
                                handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "cycle": cycle_idx, "stage": "state_refresh_done"}, ensure_ascii=False) + "\n")
                                handle.flush()
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
                                touchdown_mask = touchdown_last.unsqueeze(-1).to(dtype=current_rel_body.dtype, device=current_rel_body.device)
                                contact_mask = last_contact.unsqueeze(-1).to(dtype=current_rel_body.dtype, device=current_rel_body.device)
                                blended_rel_body = torch.lerp(running_rel_body, current_rel_body, 0.35 * touchdown_mask)
                                blended_rel_body = torch.lerp(blended_rel_body, current_rel_body, 0.10 * contact_mask)
                                shared.running_foot_rel_body = blended_rel_body.detach().clone()
                                shared.phase_shift = (
                                    float(shared.phase_shift)
                                    + float(runtime.plan_dt) * float(runtime.requested_n_frames) * float(runtime.mpc_planner_cfg.runtime.step_freq)
                                ) % 1.0

                            report = {
                                "kind": "segment",
                                "variant": variant_name,
                                "seq": seq_name,
                                "segment": segment_name,
                                "abs_drift": abs(rel_radius_series[-1] - rel_radius_series[0]),
                                "root_err_mean": sum(root_err_series) / len(root_err_series),
                                "foot_err_mean": sum(foot_err_series) / len(foot_err_series),
                                "root_rel_foot_err_mean": sum(root_rel_foot_err_series) / len(root_rel_foot_err_series),
                                "transition_foot_err_mean": sum(transition_foot_err_series) / len(transition_foot_err_series),
                                "foot_step_mean": sum(foot_step_series) / len(foot_step_series),
                                "foot_step_max": max(foot_step_max_series),
                                "dx_mean": sum(root_dx_series) / len(root_dx_series),
                                "dy_mean": sum(root_dy_series) / len(root_dy_series),
                                "dyaw_mean": sum(root_dyaw_series) / len(root_dyaw_series),
                                "stance_anchor_error": sum(stance_anchor_err_series) / len(stance_anchor_err_series),
                                "touchdown_jump_distance": sum(touchdown_jump_series) / len(touchdown_jump_series),
                                "touchdown_jump_max": max(touchdown_jump_max_series),
                                "touchdown_ground_gap_mean": sum(td_gap_series) / len(td_gap_series),
                                "touchdown_airborne_ratio": sum(td_airborne_ratio_series) / len(td_airborne_ratio_series),
                                "touchdown_airborne_max_gap": sum(td_airborne_max_gap_series) / len(td_airborne_max_gap_series),
                                "touchdown_event_ground_gap_mean": sum(td_event_gap_series) / len(td_event_gap_series),
                                "touchdown_event_airborne_ratio": sum(td_event_airborne_ratio_series) / len(td_event_airborne_ratio_series),
                                "touchdown_event_airborne_max_gap": sum(td_event_airborne_max_gap_series) / len(td_event_airborne_max_gap_series),
                                "touchdown_event_valid_ratio": sum(td_event_valid_ratio_series) / len(td_event_valid_ratio_series),
                                "stance_ground_gap_mean": sum(stance_gap_series) / len(stance_gap_series),
                                "stance_airborne_ratio": sum(stance_airborne_ratio_series) / len(stance_airborne_ratio_series),
                                "stance_airborne_max_gap": sum(stance_airborne_max_gap_series) / len(stance_airborne_max_gap_series),
                                "touchdown_gap_trend": float(td_gap_series[-1] - td_gap_series[0]),
                                "touchdown_event_gap_trend": float(td_event_gap_series[-1] - td_event_gap_series[0]),
                                "stance_gap_trend": float(stance_gap_series[-1] - stance_gap_series[0]),
                                "phase_discontinuity": (
                                    sum(phase_discontinuity_series) / len(phase_discontinuity_series)
                                    if phase_discontinuity_series
                                    else 0.0
                                ),
                                "contact_flip_count": float(contact_flip_count),
                            }
                            if contact_pattern_series:
                                for key in contact_pattern_series[0]:
                                    report[key] = sum(m[key] for m in contact_pattern_series) / len(contact_pattern_series)
                            seq_reports.append(report)
                            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
                            handle.flush()
                            handle.write(json.dumps({"kind": "progress", "variant": variant_name, "seq": seq_name, "segment": segment_name, "stage": "segment_done"}, ensure_ascii=False) + "\n")
                            handle.flush()

                    summary = {
                        "kind": "summary",
                        "variant": variant_name,
                        "seq": seq_name,
                        "mean_abs_drift": sum(r["abs_drift"] for r in seq_reports) / len(seq_reports),
                        "mean_root_err": sum(r["root_err_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_transition_foot_err": sum(r["transition_foot_err_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_root_rel_foot_err": sum(r["root_rel_foot_err_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_foot_step": sum(r["foot_step_mean"] for r in seq_reports) / len(seq_reports),
                        "max_foot_step": max(r["foot_step_max"] for r in seq_reports),
                        "mean_dx": sum(r["dx_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_dy": sum(r["dy_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_dyaw": sum(r["dyaw_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_phase_discontinuity": sum(r["phase_discontinuity"] for r in seq_reports) / len(seq_reports),
                        "mean_contact_flip_count": sum(r["contact_flip_count"] for r in seq_reports) / len(seq_reports),
                        "mean_stance_anchor_error": sum(r["stance_anchor_error"] for r in seq_reports) / len(seq_reports),
                        "mean_touchdown_jump_distance": sum(r["touchdown_jump_distance"] for r in seq_reports) / len(seq_reports),
                        "max_touchdown_jump": max(r["touchdown_jump_max"] for r in seq_reports),
                        "mean_touchdown_ground_gap": sum(r["touchdown_ground_gap_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_touchdown_event_ground_gap": sum(r["touchdown_event_ground_gap_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_stance_ground_gap": sum(r["stance_ground_gap_mean"] for r in seq_reports) / len(seq_reports),
                        "mean_stance_airborne_ratio": sum(r["stance_airborne_ratio"] for r in seq_reports) / len(seq_reports),
                        "max_stance_airborne_max_gap": max(r["stance_airborne_max_gap"] for r in seq_reports),
                        "mean_touchdown_gap_trend": sum(r["touchdown_gap_trend"] for r in seq_reports) / len(seq_reports),
                        "mean_touchdown_event_gap_trend": sum(r["touchdown_event_gap_trend"] for r in seq_reports) / len(seq_reports),
                        "mean_stance_gap_trend": sum(r["stance_gap_trend"] for r in seq_reports) / len(seq_reports),
                        "mean_touchdown_airborne_ratio": sum(r["touchdown_airborne_ratio"] for r in seq_reports) / len(seq_reports),
                        "mean_touchdown_event_airborne_ratio": sum(r["touchdown_event_airborne_ratio"] for r in seq_reports) / len(seq_reports),
                        "max_touchdown_airborne_max_gap": max(r["touchdown_airborne_max_gap"] for r in seq_reports),
                        "max_touchdown_event_airborne_max_gap": max(r["touchdown_event_airborne_max_gap"] for r in seq_reports),
                        "mean_diagonal_swing_pair_ratio": sum(r["diagonal_swing_pair_ratio"] for r in seq_reports) / len(seq_reports),
                        "mean_lateral_swing_pair_ratio": sum(r["lateral_swing_pair_ratio"] for r in seq_reports) / len(seq_reports),
                        "mean_front_hind_swing_pair_ratio": sum(r["front_hind_swing_pair_ratio"] for r in seq_reports) / len(seq_reports),
                        "mean_single_swing_ratio": sum(r["single_swing_ratio"] for r in seq_reports) / len(seq_reports),
                        "mean_two_swing_ratio": sum(r["two_swing_ratio"] for r in seq_reports) / len(seq_reports),
                        "mean_triple_or_more_swing_ratio": sum(r["triple_or_more_swing_ratio"] for r in seq_reports) / len(seq_reports),
                    }
                    handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                    handle.flush()
        print(output_path)
        return 0
    except Exception as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "kind": "exception",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False) + "\n")
            handle.flush()
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
