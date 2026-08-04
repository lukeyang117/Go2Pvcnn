"""Script to play a trained teacher policy."""

from __future__ import annotations

import argparse
import atexit
import os
import select
import signal
import sys
import termios
import threading
import tty
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch


PARALLELISM_TERMINATION_NAMES = (
    "time_out",
    "base_contact",
    "bad_orientation",
    "parallelism_ref_root_z_too_far",
    "parallelism_ref_projected_gravity_too_far",
    "parallelism_ref_foot_z_too_far",
    "parallelism_ref_joint_pos_too_far",
)


@dataclass
class ParallelismPlayPanelState:
    """UI-owned command and diagnostic-only termination controls."""

    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    last_applied_command: tuple[float, float, float] | None = None
    suppress_termination: dict[str, bool] = field(
        default_factory=lambda: dict.fromkeys(PARALLELISM_TERMINATION_NAMES, True)
    )


def _parallelism_debug_command_from_env() -> tuple[float, float, float] | None:
    """Read an optional fixed command used for non-interactive play diagnosis."""

    value = os.environ.get("PARALLELISM_PLAY_DEBUG_COMMAND", "").strip()
    if not value:
        return None
    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "PARALLELISM_PLAY_DEBUG_COMMAND must be three comma-separated numbers, "
            "for example 0.5,0,0"
        ) from exc
    if len(parts) != 3:
        raise ValueError(
            "PARALLELISM_PLAY_DEBUG_COMMAND must be three comma-separated numbers, "
            "for example 0.5,0,0"
        )
    return parts


def _panel_command_tensor(state: ParallelismPlayPanelState, command: torch.Tensor) -> torch.Tensor:
    """Return one clamped body-frame command in the command tensor's dtype/device."""

    values = command.new_tensor(((state.vx, state.vy, state.vyaw),))
    lower = command.new_tensor((-1.0, -0.5, -1.0))
    upper = command.new_tensor((1.0, 0.5, 1.0))
    return torch.maximum(torch.minimum(values, upper), lower)


def _filter_termination_masks(
    raw_masks: dict[str, torch.Tensor],
    suppress_termination: dict[str, bool],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep raw diagnostics while OR-ing only termination terms allowed to reset."""

    if not raw_masks:
        raise ValueError("raw_masks must contain at least one termination term")
    first = next(iter(raw_masks.values()))
    effective_done = torch.zeros_like(torch.as_tensor(first), dtype=torch.bool)
    diagnostics: dict[str, torch.Tensor] = {}
    for name, value in raw_masks.items():
        raw = torch.as_tensor(value, dtype=torch.bool, device=effective_done.device)
        diagnostics[name] = raw
        if not bool(suppress_termination.get(name, False)):
            effective_done |= raw
    return effective_done, diagnostics


def _set_panel_termination_checkbox(
    state: ParallelismPlayPanelState,
    name: str,
    checked: bool,
) -> None:
    """Map the UI checkbox to the termination filter.

    An unchecked box means the play viewer must keep running. Checking a term
    explicitly opts into allowing that term to reset the environment.
    """

    state.suppress_termination[name] = not bool(checked)


@dataclass(frozen=True)
class ParallelismVisualFrame:
    root_pos_w: torch.Tensor
    root_rpy_w: torch.Tensor
    joint_pos: torch.Tensor
    foot_pos_w: torch.Tensor
    contact_state: torch.Tensor
    future_root_pos_w: torch.Tensor
    future_foot_pos_w: torch.Tensor
    future_contact_state: torch.Tensor


@dataclass
class ParallelismTerminationDiagnostics:
    raw_masks: dict[str, torch.Tensor] = field(default_factory=dict)


def _parallelism_contact_debug_data(base_env, *, env_id: int = 0) -> dict[str, object]:
    """Collect resolved contact sensor data for diagnosing play reset terms."""

    result: dict[str, object] = {}
    try:
        sensor = base_env.scene["contact_forces"]
    except Exception as exc:  # noqa: BLE001 - debug-only best effort.
        return {"error": f"missing contact_forces sensor: {exc}"}

    body_names = tuple(getattr(sensor, "body_names", ()) or ())
    result["sensor_body_names"] = body_names
    forces = torch.as_tensor(sensor.data.net_forces_w[env_id]).detach()
    force_norm = torch.linalg.vector_norm(forces, dim=-1)
    result["force_norm"] = force_norm.detach().cpu().tolist()
    history = torch.as_tensor(sensor.data.net_forces_w_history[env_id]).detach()
    history_norm_max = torch.max(torch.linalg.vector_norm(history, dim=-1), dim=0).values
    result["history_force_norm_max"] = history_norm_max.detach().cpu().tolist()
    if body_names and int(force_norm.numel()) == len(body_names):
        max_index = int(torch.argmax(force_norm).item())
        result["max_force_body"] = body_names[max_index]
        result["max_force_norm"] = float(force_norm[max_index].item())
        result["base_force_norms"] = [
            (name, float(force_norm[index].item()))
            for index, name in enumerate(body_names)
            if "base" in name
        ]
        result["foot_force_norms"] = [
            (name, float(force_norm[index].item()))
            for index, name in enumerate(body_names)
            if "foot" in name
        ]

    for name, term_cfg in zip(base_env.termination_manager._term_names, base_env.termination_manager._term_cfgs):
        if name != "base_contact":
            continue
        sensor_cfg = getattr(term_cfg, "params", {}).get("sensor_cfg")
        result["base_contact_body_names_cfg"] = getattr(sensor_cfg, "body_names", None)
        result["base_contact_body_ids"] = list(getattr(sensor_cfg, "body_ids", []) or [])
        body_ids = getattr(sensor_cfg, "body_ids", None)
        if body_ids is not None:
            ids_tensor = torch.as_tensor(body_ids, dtype=torch.long, device=force_norm.device)
            selected = force_norm.index_select(0, ids_tensor)
            selected_history = history_norm_max.index_select(0, ids_tensor)
            result["base_contact_selected_force_norm"] = selected.detach().cpu().tolist()
            result["base_contact_selected_history_force_norm_max"] = selected_history.detach().cpu().tolist()
            if body_names:
                result["base_contact_selected_body_names"] = [
                    body_names[int(index)] for index in ids_tensor.detach().cpu().tolist()
                ]
        break
    return result


def _parallelism_debug_snapshot(base_env, manager, diagnostics, dones, *, timestep: int) -> None:
    """Print the state boundaries needed to diagnose play-only alignment/reset issues."""

    phase = int(torch.as_tensor(manager.phase)[0].item())
    plan_count = int(torch.as_tensor(manager.plan_count)[0].item())
    if timestep > 5 and phase != 0 and not bool(torch.as_tensor(dones).any().item()) and timestep % 5 != 0:
        return
    frame = _parallelism_visual_frame(manager)
    policy_robot = base_env.scene["robot"]
    reference_robot = base_env.scene["reference_robot"]
    policy_root = torch.as_tensor(policy_robot.data.root_pos_w[0]).detach().cpu()
    manager_root = torch.as_tensor(frame.root_pos_w).detach().cpu()
    reference_root = torch.as_tensor(reference_robot.data.root_pos_w[0]).detach().cpu()
    origin = getattr(getattr(base_env.scene, "env_origins", None), "__getitem__", lambda _: None)(0)
    if origin is None:
        origin = torch.zeros(3)
    else:
        origin = torch.as_tensor(origin).detach().cpu()
    raw = {
        name: bool(torch.as_tensor(value)[0].item())
        for name, value in (diagnostics.raw_masks.items() if diagnostics is not None else ())
    }
    effective_terminated = bool(torch.as_tensor(base_env.termination_manager._terminated_buf)[0].item())
    effective_truncated = bool(torch.as_tensor(base_env.termination_manager._truncated_buf)[0].item())
    episode_length = int(torch.as_tensor(base_env.episode_length_buf)[0].item())
    command_manager = getattr(base_env, "command_manager", None)
    command = None
    if command_manager is not None and hasattr(command_manager, "get_command"):
        command_value = command_manager.get_command("base_velocity")
        if command_value is not None:
            command = torch.as_tensor(command_value)[0].detach().cpu().tolist()
    manager_start = torch.as_tensor(manager.root_pos_w[0, 0]).detach().cpu()
    manager_delta = manager_root - manager_start
    next_phase = min(phase + 1, int(manager.root_pos_w.shape[1]) - 1)
    manager_step_delta = (
        torch.as_tensor(manager.root_pos_w[0, next_phase]).detach().cpu()
        - manager_root
    )
    plan_valid = bool(torch.as_tensor(manager.plan_valid)[0].item())
    standstill_latched = bool(torch.as_tensor(manager.standstill_latched)[0].item())
    plan_valid_count = int(torch.as_tensor(manager.plan_valid_count)[0].item())
    plan_reject_counts = torch.as_tensor(manager.plan_reject_counts)[0].detach().cpu().tolist()
    plan_collision_count = int(torch.as_tensor(manager.plan_collision_counts)[0].item())
    plan_per_leg_valid_count = torch.as_tensor(manager.plan_per_leg_valid_count)[0].detach().cpu().tolist()
    plan_per_leg_collision_count = torch.as_tensor(manager.plan_per_leg_collision_count)[0].detach().cpu().tolist()
    print(
        "[Parallelism][debug] "
        f"step={timestep} phase={phase} plan_count={plan_count} "
        f"episode_length={episode_length} "
        f"command={command} "
        f"policy_root={policy_root.tolist()} manager_ref={manager_root.tolist()} "
        f"reference_data={reference_root.tolist()} env_origin={origin.tolist()} "
        f"manager_delta_from_frame0={manager_delta.tolist()} "
        f"manager_next_delta={manager_step_delta.tolist()} "
        f"plan_valid={plan_valid} standstill_latched={standstill_latched} valid_candidates={plan_valid_count}/200 "
        f"reject(valid_map,joint,landing,collision,candidate_semantic,fk_semantic)={plan_reject_counts} "
        f"collision_candidates={plan_collision_count} "
        f"per_leg_valid={plan_per_leg_valid_count} "
        f"per_leg_collision={plan_per_leg_collision_count} "
        f"policy-ref={torch.linalg.vector_norm(policy_root - manager_root).item():.6f} "
        f"reference-ref={torch.linalg.vector_norm(reference_root - manager_root).item():.6f} "
        f"dones={torch.as_tensor(dones).detach().cpu().tolist()} "
        f"effective(terminated={effective_terminated},truncated={effective_truncated}) "
        f"raw={raw}",
        flush=True,
    )
    actual_foot = manager._measured_foot_pos_w(
        policy_robot,
        torch.tensor([0], dtype=torch.long, device=manager.device),
    )
    if actual_foot is not None:
        actual_foot = torch.as_tensor(actual_foot[0]).detach().cpu()
        reference_foot = torch.as_tensor(frame.foot_pos_w).detach().cpu()
        foot_error = torch.linalg.vector_norm(actual_foot - reference_foot, dim=-1)
        trajectory_foot = torch.as_tensor(manager.foot_pos_w[0]).detach().cpu()
        trajectory_lift = trajectory_foot[..., 2].amax(dim=0) - trajectory_foot[0, :, 2]
        contact = torch.as_tensor(frame.contact_state).detach().cpu()
        joint_data = _parallelism_joint_error_data(base_env, frame)
        joint_error_by_leg: dict[str, list[float]] = {}
        if joint_data is not None:
            joint_names, _, _, joint_error = joint_data
            for leg_name in ("FL", "FR", "RL", "RR"):
                indices = [index for index, name in enumerate(joint_names) if str(name).startswith(f"{leg_name}_")]
                joint_error_by_leg[leg_name] = (
                    torch.abs(joint_error[indices]).detach().cpu().tolist() if indices else []
                )
        print(
            "[Parallelism][leg-debug] "
            f"step={timestep} phase={phase} order={['FL', 'FR', 'RL', 'RR']} "
            f"contact={contact.tolist()} "
            f"reference_z={reference_foot[:, 2].tolist()} "
            f"actual_z={actual_foot[:, 2].tolist()} "
            f"foot_error={foot_error.tolist()} "
            f"trajectory_apex_lift={trajectory_lift.tolist()} "
            f"joint_abs_error_by_leg={joint_error_by_leg}",
            flush=True,
        )
    if timestep <= 5 or bool(raw.get("base_contact", False)):
        print(
            "[Parallelism][contact-debug] "
            f"step={timestep} {_parallelism_contact_debug_data(base_env)}",
            flush=True,
        )


def _parallelism_joint_error_data(base_env, frame: ParallelismVisualFrame, *, env_id: int = 0):
    """Return policy/reference joint values in the robot articulation order."""

    robot = base_env.scene["robot"]
    actual = torch.as_tensor(robot.data.joint_pos[env_id], dtype=frame.joint_pos.dtype, device=frame.joint_pos.device)
    reference = torch.as_tensor(frame.joint_pos, dtype=actual.dtype, device=actual.device)
    if actual.shape != reference.shape:
        return None
    names = tuple(getattr(robot, "joint_names", ()) or ())
    if len(names) != int(actual.numel()):
        names = tuple(f"joint_{index}" for index in range(int(actual.numel())))
    error = actual - reference
    return names, reference, actual, error


def _parallelism_visual_frame(manager, *, env_id: int = 0) -> ParallelismVisualFrame:
    """Extract one manager phase consistently for the reference robot and markers."""

    phase = int(torch.as_tensor(manager.phase)[env_id].item())
    return ParallelismVisualFrame(
        root_pos_w=manager.root_pos_w[env_id, phase],
        root_rpy_w=manager.root_rpy_w[env_id, phase],
        joint_pos=manager.joint_pos[env_id, phase],
        foot_pos_w=manager.foot_pos_w[env_id, phase],
        contact_state=manager.contact_state[env_id, phase],
        future_root_pos_w=manager.root_pos_w[env_id, phase:],
        future_foot_pos_w=manager.foot_pos_w[env_id, phase:],
        future_contact_state=manager.contact_state[env_id, phase:],
    )


def _install_parallelism_termination_filter(termination_manager, state: ParallelismPlayPanelState):
    """Mask selected reset terms before ManagerBasedRLEnv reads termination buffers."""

    existing = getattr(termination_manager, "_parallelism_play_diagnostics", None)
    if existing is not None:
        return existing

    diagnostics = ParallelismTerminationDiagnostics()
    original_compute = termination_manager.compute

    def compute():
        original_compute()
        raw_masks = {
            name: value.clone()
            for name, value in getattr(termination_manager, "_term_dones", {}).items()
        }
        diagnostics.raw_masks = raw_masks
        if not raw_masks:
            return termination_manager._truncated_buf | termination_manager._terminated_buf

        termination_manager._truncated_buf.zero_()
        termination_manager._terminated_buf.zero_()
        for name, term_cfg in zip(termination_manager._term_names, termination_manager._term_cfgs):
            value = raw_masks[name]
            if bool(state.suppress_termination.get(name, False)):
                continue
            if bool(getattr(term_cfg, "time_out", False)):
                termination_manager._truncated_buf |= value
            else:
                termination_manager._terminated_buf |= value
        return termination_manager._truncated_buf | termination_manager._terminated_buf

    termination_manager.compute = compute
    termination_manager._parallelism_play_diagnostics = diagnostics
    return diagnostics


def _write_parallelism_reference_robot(reference_robot, frame: ParallelismVisualFrame) -> None:
    """Write one reference frame into the play-only, collision-free Go2."""

    from extension.convention import euler_to_quat_batch

    with torch.inference_mode():
        root_pos_w = frame.root_pos_w.reshape(1, 3)
        root_rpy_w = frame.root_rpy_w.reshape(1, 3)
        root_quat_w = euler_to_quat_batch(root_rpy_w[:, 0], root_rpy_w[:, 1], root_rpy_w[:, 2])
        root_pose = torch.cat((root_pos_w, root_quat_w), dim=-1)
        joint_pos = frame.joint_pos.reshape(1, -1)
        reference_robot.write_root_pose_to_sim(root_pose)
        reference_robot.write_root_velocity_to_sim(torch.zeros(1, 6, dtype=root_pose.dtype, device=root_pose.device))
        reference_robot.write_joint_state_to_sim(
            joint_pos,
            torch.zeros_like(joint_pos),
        )


def _write_parallelism_reference_root(reference_robot, frame: ParallelismVisualFrame) -> None:
    """Synchronize only the root after stepping, avoiding a second joint write."""

    from extension.convention import euler_to_quat_batch

    with torch.inference_mode():
        root_pos_w = frame.root_pos_w.reshape(1, 3)
        root_rpy_w = frame.root_rpy_w.reshape(1, 3)
        root_quat_w = euler_to_quat_batch(root_rpy_w[:, 0], root_rpy_w[:, 1], root_rpy_w[:, 2])
        root_pose = torch.cat((root_pos_w, root_quat_w), dim=-1)
        reference_robot.write_root_pose_to_sim(root_pose)
        reference_robot.write_root_velocity_to_sim(torch.zeros(1, 6, dtype=root_pose.dtype, device=root_pose.device))


def _sync_parallelism_reference_visual_state(base_env) -> None:
    """Flush a planner-written reference frame using the viewer playback path."""

    with torch.inference_mode():
        scene = getattr(base_env, "scene", None)
        reference_robot = None
        if scene is not None:
            try:
                reference_robot = scene["reference_robot"]
            except (KeyError, TypeError):
                reference_robot = getattr(scene, "reference_robot", None)
        if reference_robot is not None and hasattr(reference_robot, "write_data_to_sim"):
            reference_robot.write_data_to_sim()
        sim = getattr(base_env, "sim", None)
        if sim is not None and hasattr(sim, "render"):
            sim.render()
        update_fn = getattr(scene, "update", None) if scene is not None else None
        if callable(update_fn) and not isinstance(scene, dict):
            update_fn(float(getattr(base_env, "physics_dt", 0.0)))


class _ParallelismPlayPanel:
    """Small in-app control panel for the physical parallelism rollout."""

    def __init__(self, state: ParallelismPlayPanelState) -> None:
        self.state = state
        self._status_models = {}
        self._models = {}
        try:
            import omni.ui as ui
        except ModuleNotFoundError:
            self.window = None
            return
        self._ui = ui
        self.window = ui.Window("Parallelism Policy Debug", width=440, height=760)
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("速度命令 (root 坐标系)", height=20)
                self._add_speed_slider("vx", -1.0, 1.0)
                self._add_speed_slider("vy", -0.5, 0.5)
                self._add_speed_slider("vyaw", -1.0, 1.0)
                ui.Spacer(height=8)
                ui.Label("Termination / Reset (勾选后允许)", height=20)
                all_model = ui.SimpleBoolModel(False)
                all_model.add_value_changed_fn(self._set_all_suppressed)
                self._models["all"] = all_model
                with ui.HStack(height=22):
                    ui.CheckBox(model=all_model, width=20)
                    ui.Label("允许全部 termination / reset")
                for name in PARALLELISM_TERMINATION_NAMES:
                    model = ui.SimpleBoolModel(False)
                    model.add_value_changed_fn(lambda value, key=name: self._set_suppressed(key, value))
                    self._models[name] = model
                    with ui.HStack(height=22):
                        ui.CheckBox(model=model, width=20)
                        ui.Label(name, width=245)
                        self._status_models[name] = ui.Label("未启用", width=55)
                ui.Spacer(height=8)
                ui.Label("规划关节 / policy 实际关节 / 差值 (rad)", height=20)
                self._joint_summary = ui.Label("等待关节数据...", height=0, word_wrap=True)

    def _add_speed_slider(self, name: str, lower: float, upper: float) -> None:
        ui = self._ui
        model = ui.SimpleFloatModel(getattr(self.state, name))
        model.add_value_changed_fn(
            lambda value, key=name: setattr(self.state, key, float(value.get_value_as_float()))
        )
        self._models[name] = model
        with ui.HStack(height=26):
            ui.Label(name, width=42)
            ui.FloatSlider(model=model, min=lower, max=upper, width=220)
            ui.FloatField(model=model, width=65)

    def _set_suppressed(self, name: str, model) -> None:
        _set_panel_termination_checkbox(self.state, name, model.get_value_as_bool())
        self._sync_all_model()

    def _set_all_suppressed(self, model) -> None:
        enabled = bool(model.get_value_as_bool())
        for name in PARALLELISM_TERMINATION_NAMES:
            self.state.suppress_termination[name] = not enabled
            item = self._models.get(name)
            if item is not None and bool(item.get_value_as_bool()) != enabled:
                item.set_value(enabled)

    def _sync_all_model(self) -> None:
        model = self._models.get("all")
        if model is None:
            return
        enabled = all(not self.state.suppress_termination.get(name, True) for name in PARALLELISM_TERMINATION_NAMES)
        if bool(model.get_value_as_bool()) != enabled:
            model.set_value(enabled)

    def update_diagnostics(self, diagnostics: ParallelismTerminationDiagnostics | None) -> None:
        if diagnostics is None:
            return
        for name, label in self._status_models.items():
            raw = diagnostics.raw_masks.get(name)
            triggered = raw is not None and bool(torch.as_tensor(raw)[0].item())
            enabled = not self.state.suppress_termination.get(name, True)
            label.text = "触发" if triggered and enabled else ("已启用" if enabled else "未启用")

    def update_joint_error(self, error_data) -> None:
        if error_data is None or not hasattr(self, "_joint_summary"):
            return
        names, reference, actual, error = error_data
        lines = [
            f"max |e| = {float(torch.max(torch.abs(error)).item()):+.4f}   "
            f"RMS = {float(torch.sqrt(torch.mean(error.square())).item()):+.4f}"
        ]
        for name, ref, act, diff in zip(names, reference, actual, error):
            lines.append(
                f"{name:<18} ref {float(ref):+7.3f}  "
                f"act {float(act):+7.3f}  e {float(diff):+7.3f}"
            )
        self._joint_summary.text = "\n".join(lines)


class _ParallelismPlayVisualizer:
    """Shows the reference robot and plan with Kit debug-draw overlays."""

    _LEG_COLORS = ((0.0, 0.9, 1.0, 1.0), (1.0, 0.35, 0.0, 1.0), (0.25, 1.0, 0.25, 1.0), (1.0, 0.15, 0.8, 1.0))
    _ROOT_COLOR = (1.0, 0.8, 0.1, 1.0)
    _REFERENCE_BODY_COLOR = (0.1, 0.45, 1.0, 1.0)
    _POLICY_BODY_COLOR = (1.0, 0.75, 0.05, 1.0)
    _ERROR_COLOR = (1.0, 0.05, 0.05, 1.0)
    _POINT_SIZE = 6
    _LINE_SIZE = 3

    def __init__(self) -> None:
        from isaacsim.util.debug_draw import _debug_draw

        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._last_marker_plan_count: int | None = None

    def _draw_path(self, positions: torch.Tensor, color: tuple[float, float, float, float]) -> None:
        points = torch.as_tensor(positions).detach().to(device="cpu", dtype=torch.float32).tolist()
        if len(points) < 2:
            return
        start_points = [tuple(point) for point in points[:-1]]
        end_points = [tuple(point) for point in points[1:]]
        self._draw.draw_lines(
            start_points,
            end_points,
            [color] * len(start_points),
            [self._LINE_SIZE] * len(start_points),
        )

    def _draw_point(self, position: torch.Tensor, color: tuple[float, float, float, float]) -> None:
        point = torch.as_tensor(position).detach().to(device="cpu", dtype=torch.float32).tolist()
        self._draw.draw_points([tuple(point)], [color], [self._POINT_SIZE])

    def _draw_error_lines(self, base_env) -> None:
        actual_robot = base_env.scene["robot"]
        reference_robot = base_env.scene["reference_robot"]
        actual = getattr(getattr(actual_robot, "data", None), "body_pos_w", None)
        reference = getattr(getattr(reference_robot, "data", None), "body_pos_w", None)
        if actual is None or reference is None:
            return
        actual = torch.as_tensor(actual[0], dtype=torch.float32)
        reference = torch.as_tensor(reference[0], dtype=torch.float32, device=actual.device)
        if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[-1] != 3:
            return
        start_points = [tuple(point) for point in actual.detach().cpu().tolist()]
        end_points = [tuple(point) for point in reference.detach().cpu().tolist()]
        self._draw.draw_lines(
            start_points,
            end_points,
            [self._ERROR_COLOR] * len(start_points),
            [self._LINE_SIZE] * len(start_points),
        )
        self._draw.draw_points(
            end_points,
            [self._REFERENCE_BODY_COLOR] * len(end_points),
            [self._POINT_SIZE] * len(end_points),
        )
        self._draw.draw_points(
            start_points,
            [self._POLICY_BODY_COLOR] * len(start_points),
            [self._POINT_SIZE] * len(start_points),
        )

    def write_reference(self, base_env, manager) -> None:
        # Isaac Lab performs internal resets inside env.step(). Refresh here so
        # the first post-reset frame is rebuilt from the live policy state.
        manager.refresh()
        frame = _parallelism_visual_frame(manager)
        _write_parallelism_reference_robot(base_env.scene["reference_robot"], frame)
        _sync_parallelism_reference_visual_state(base_env)

    def update(self, base_env, manager) -> None:
        frame = _parallelism_visual_frame(manager)

        plan_count = int(torch.as_tensor(getattr(manager, "plan_count", torch.zeros(1)))[0].item())
        self._draw.clear_lines()
        self._draw_error_lines(base_env)
        self._draw_path(frame.future_root_pos_w, self._ROOT_COLOR)
        for leg_idx in range(4):
            future = frame.future_foot_pos_w[:, leg_idx].to(dtype=torch.float32)
            self._draw_path(future, self._LEG_COLORS[leg_idx])
        if self._last_marker_plan_count != plan_count:
            self._last_marker_plan_count = plan_count
            self._draw.clear_points()
            for leg_idx in range(4):
                future = frame.future_foot_pos_w[:, leg_idx].to(dtype=torch.float32)
                contact = frame.future_contact_state[:, leg_idx]
                transition = torch.nonzero((~contact[:-1]) & contact[1:], as_tuple=False).flatten()
                touchdown_idx = int(transition[0].item() + 1) if int(transition.numel()) else int(future.shape[0] - 1)
                self._draw_point(future[touchdown_idx], self._LEG_COLORS[leg_idx])


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for _path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play a trained teacher policy")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during play")
    parser.add_argument("--video_length", type=int, default=2000000, help="Length of recorded video (steps)")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (steps)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
    parser.add_argument("--checkpoint", type=str, default="model_1600.pt", help="Checkpoint file name")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory name")
    parser.add_argument(
        "--experiment",
        type=str,
        default="teacher_elevation_trajectory_mpc_semantic",
        choices=[
            "teacher_elevation_trajectory_mpc_semantic",
            "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance",
            "parallelism_tracking_flat",
        ],
        help="Experiment/task to play.",
    )
    parser.add_argument("--sample", action="store_true", default=False, help="Sample actions with std instead of using policy")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many play steps; 0 means run until the app exits.")
    parser.add_argument(
        "--use-raw-reference-trajectory",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug-livestream",
        action="store_true",
        default=False,
        help="Print startup and loop timing diagnostics for WebRTC livestream bottlenecks.",
    )
    parser.add_argument(
        "--step-mode",
        action="store_true",
        default=False,
        help="Pause play loop and advance exactly one env/render step for each Space key press.",
    )
    parser.add_argument(
        "--keyboard-control",
        action="store_true",
        default=False,
        help="Use terminal hold-to-move keyboard velocity commands.",
    )
    parser.add_argument("--keyboard-linear-speed", type=float, default=0.5, help="Keyboard forward/backward speed.")
    parser.add_argument("--keyboard-lateral-speed", type=float, default=0.25, help="Keyboard left/right speed.")
    parser.add_argument("--keyboard-yaw-speed", type=float, default=0.5, help="Keyboard yaw-rate speed.")
    parser.add_argument("--keyboard-speed-step", type=float, default=0.1, help="Keyboard +/- speed increment.")
    parser.add_argument("--terrain-row", type=int, default=None, help="Initial terrain row for env0; omit for default.")
    parser.add_argument("--terrain-col", type=int, default=None, help="Initial terrain column for env0; omit for default.")
    parser.add_argument(
        "--planner-backend",
        type=str,
        default="mpc",
        choices=["mpc", "parallelism"],
        help="Trajectory planner backend.",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _prepare_runtime_args(args_cli: argparse.Namespace) -> argparse.Namespace:
    if getattr(args_cli, "livestream", -1) in (1, 2) and not args_cli.enable_cameras:
        args_cli.enable_cameras = True
        print(
            "[INFO][play.py] livestream: enabled AppLauncher --enable_cameras so the simulator "
            "uses a rendering experience (works without X11; WebRTC client on another machine).",
            flush=True,
        )
    return args_cli


def _launch_app(args_cli: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    return app_launcher, app_launcher.app


def _resolve_render_mode(args_cli: argparse.Namespace) -> str | None:
    if args_cli.video or getattr(args_cli, "livestream", -1) in (1, 2):
        return "rgb_array"
    return None


def _livestream_camera_update_interval(livestream: int) -> int:
    return 4 if livestream in (1, 2) else 1


def _should_update_follow_camera(*, timestep: int, num_envs: int, livestream: int, interval: int) -> bool:
    if num_envs != 1:
        return False
    if livestream in (1, 2):
        return timestep % max(1, interval) == 0
    return True


def _play_loop_should_continue(simulation_app, *, timestep: int, max_steps: int) -> bool:
    """Keep an interactive play session alive until an explicit limit is reached.

    Isaac Sim can report ``is_running() == False`` transiently when the GUI is
    attached through VNC/VirtualGL.  Treating that value as the play-loop
    lifetime closes the viewer after the first environment step.
    """

    if max_steps > 0:
        return timestep < max_steps
    return True


@dataclass
class _KeyboardVelocityController:
    enabled: bool
    linear_speed: float
    lateral_speed: float
    yaw_speed: float
    speed_step: float
    max_linear_speed: float = 1.0
    max_lateral_speed: float = 0.5
    max_yaw_speed: float = 1.0
    hold_timeout_s: float = 0.15
    poll_interval_s: float = 0.02

    def __post_init__(self) -> None:
        self._pressed: set[str] = set()
        self._last_pressed_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._terminal_fd: int | None = None
        self._terminal_original_attrs = None

    def press(self, key: str) -> None:
        key = str(key).lower()
        with self._lock:
            if key in {"+", "="}:
                self.linear_speed = min(self.max_linear_speed, self.linear_speed + self.speed_step)
                self.lateral_speed = min(self.max_lateral_speed, self.lateral_speed + self.speed_step)
                self.yaw_speed = min(self.max_yaw_speed, self.yaw_speed + self.speed_step)
                return
            if key in {"-", "_"}:
                self.linear_speed = max(0.0, self.linear_speed - self.speed_step)
                self.lateral_speed = max(0.0, self.lateral_speed - self.speed_step)
                self.yaw_speed = max(0.0, self.yaw_speed - self.speed_step)
                return
            if key in {" ", "space", "x"}:
                self._pressed.clear()
                self._last_pressed_at.clear()
                return
            self._pressed.add(key)
            self._last_pressed_at[key] = perf_counter()

    def release(self, key: str) -> None:
        key = str(key).lower()
        with self._lock:
            self._pressed.discard(key)

    def command_values(self) -> tuple[float, float, float]:
        with self._lock:
            self._expire_stale_keys_locked(perf_counter())
            keys = set(self._pressed)
            linear_speed = float(self.linear_speed)
            lateral_speed = float(self.lateral_speed)
            yaw_speed = float(self.yaw_speed)

        vx = (1.0 if "w" in keys else 0.0) + (-1.0 if "s" in keys else 0.0)
        vy = (1.0 if "a" in keys else 0.0) + (-1.0 if "d" in keys else 0.0)
        yaw = (1.0 if "q" in keys else 0.0) + (-1.0 if "e" in keys else 0.0)
        return (
            float(np.clip(vx * linear_speed, -self.max_linear_speed, self.max_linear_speed)),
            float(np.clip(vy * lateral_speed, -self.max_lateral_speed, self.max_lateral_speed)),
            float(np.clip(yaw * yaw_speed, -self.max_yaw_speed, self.max_yaw_speed)),
        )

    def command_tensor(self, *, device, dtype, num_envs: int) -> torch.Tensor:
        values = self.command_values()
        command = torch.tensor(values, device=device, dtype=dtype).view(1, 3)
        return command.repeat(int(num_envs), 1)

    @staticmethod
    def _key_to_name(key: str) -> str | None:
        if key == "\x1b":
            return "esc"
        if key in {" ", "\r", "\n"}:
            return "space"
        text = str(key).lower()
        if text == "key.space":
            return "space"
        if text == "key.esc":
            return "esc"
        if len(text) == 1:
            return text
        return None

    def _expire_stale_keys_locked(self, now_s: float) -> None:
        stale = [
            key
            for key in self._pressed
            if key in {"w", "s", "a", "d", "q", "e"}
            and now_s - self._last_pressed_at.get(key, 0.0) > self.hold_timeout_s
        ]
        for key in stale:
            self._pressed.discard(key)
            self._last_pressed_at.pop(key, None)

    def _terminal_read_loop(self) -> None:
        assert self._terminal_fd is not None
        while not self._stop_event.is_set():
            readable, _, _ = select.select([self._terminal_fd], [], [], self.poll_interval_s)
            if not readable:
                continue
            try:
                char = os.read(self._terminal_fd, 1).decode(errors="ignore")
            except OSError:
                break
            name = self._key_to_name(char)
            if name == "esc":
                self._stop_event.set()
                break
            if name is not None:
                self.press(name)

    def __enter__(self) -> "_KeyboardVelocityController":
        if not self.enabled:
            return self
        try:
            if not sys.stdin.isatty():
                raise RuntimeError("stdin is not a TTY")
            self._terminal_fd = sys.stdin.fileno()
            self._terminal_original_attrs = termios.tcgetattr(self._terminal_fd)
            tty.setcbreak(self._terminal_fd)
            self._reader_thread = threading.Thread(target=self._terminal_read_loop, name="play-terminal-keyboard", daemon=True)
            self._reader_thread.start()
            print(
                "[play.py] Terminal keyboard control enabled: hold W/S/A/D/Q/E, +/- speed, Space or X stop, Esc stop.",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - terminal availability varies in headless launchers.
            print(f"[WARN][play.py] --keyboard-control disabled: failed to start terminal keyboard reader: {exc}", flush=True)
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.5)
            self._reader_thread = None
        if self._terminal_fd is not None and self._terminal_original_attrs is not None:
            try:
                termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_original_attrs)
            except Exception:
                pass
        self._terminal_fd = None
        self._terminal_original_attrs = None


def _apply_keyboard_velocity_command(base_env, controller: _KeyboardVelocityController) -> torch.Tensor | None:
    if not controller.enabled:
        return None
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_command"):
        return None
    command = command_manager.get_command("base_velocity")
    if command is None:
        return None
    target = controller.command_tensor(device=command.device, dtype=command.dtype, num_envs=int(command.shape[0]))
    command[:, :3] = target
    return command


def _apply_panel_velocity_command(base_env, state: ParallelismPlayPanelState) -> torch.Tensor | None:
    """Apply the visual panel command to env0 without changing other environments."""

    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_command"):
        return None
    command = command_manager.get_command("base_velocity")
    if command is None or int(command.shape[0]) == 0:
        return None
    target = _panel_command_tensor(state, command)[0]
    command[0, :3] = target
    target_tuple = tuple(float(value) for value in target.detach().cpu().tolist())
    if state.last_applied_command != target_tuple:
        state.last_applied_command = target_tuple
        manager = getattr(base_env, "parallelism_reference_manager", None)
        mark_command_changed = getattr(manager, "mark_command_changed", None)
        if callable(mark_command_changed):
            env_mask = torch.zeros(int(command.shape[0]), dtype=torch.bool, device=command.device)
            env_mask[0] = True
            mark_command_changed(env_mask)
    return command


def _apply_initial_terrain_selection(base_env, *, terrain_row: int | None, terrain_col: int | None, env_id: int = 0) -> None:
    if terrain_row is None and terrain_col is None:
        return
    terrain = getattr(getattr(base_env, "scene", None), "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain is None or terrain_origins is None:
        raise RuntimeError("--terrain-row/--terrain-col require curriculum terrain_origins.")

    origins = torch.as_tensor(terrain_origins)
    if origins.ndim != 3 or int(origins.shape[-1]) != 3:
        raise RuntimeError(f"Expected terrain_origins shape [rows, cols, 3], got {tuple(origins.shape)}")
    num_rows, num_cols = int(origins.shape[0]), int(origins.shape[1])
    row = int(terrain_row) if terrain_row is not None else int(getattr(terrain, "terrain_levels")[env_id])
    col = int(terrain_col) if terrain_col is not None else int(getattr(terrain, "terrain_types")[env_id])
    if not (0 <= row < num_rows):
        raise ValueError(f"--terrain-row must be in [0, {num_rows - 1}], got {row}")
    if not (0 <= col < num_cols):
        raise ValueError(f"--terrain-col must be in [0, {num_cols - 1}], got {col}")

    selected_origin = origins[row, col]
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is not None:
        terrain_levels[env_id] = row
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is not None:
        terrain_types[env_id] = col
    env_origins = getattr(terrain, "env_origins", None)
    if env_origins is not None:
        env_origins[env_id] = selected_origin.to(device=env_origins.device, dtype=env_origins.dtype)
    scene_env_origins = getattr(base_env.scene, "env_origins", None)
    if scene_env_origins is not None:
        scene_env_origins[env_id] = selected_origin.to(device=scene_env_origins.device, dtype=scene_env_origins.dtype)
    print(f"[play.py] Initial terrain env{env_id}: row={row}, col={col}", flush=True)


@dataclass
class _TerminalStepGate:
    enabled: bool

    def __post_init__(self) -> None:
        self._stdin_fd = None
        self._old_termios = None
        self._old_flags = None
        self._raw_enabled = False
        self._old_signal_handlers: dict[int, object] = {}
        self._atexit_registered = False

    def __enter__(self) -> "_TerminalStepGate":
        if not self.enabled:
            return self
        if not sys.stdin.isatty():
            print("[WARN][play.py] stdin is not a TTY; --step-mode cannot receive Space key presses.", flush=True)
            return self
        import fcntl
        import termios
        import tty

        self._stdin_fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._stdin_fd)
        self._old_flags = fcntl.fcntl(self._stdin_fd, fcntl.F_GETFL)
        tty.setcbreak(self._stdin_fd)
        fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, self._old_flags | os.O_NONBLOCK)
        self._raw_enabled = True
        self._install_cleanup_guards()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._remove_cleanup_guards()
        self._restore_terminal_state()

    def wait_for_step(self) -> bool:
        if not self.enabled:
            return True
        if not self._raw_enabled:
            sleep(0.05)
            return False
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                return False
            char = sys.stdin.read(1)
            if not char:
                return False
            if char == "\x03":
                raise KeyboardInterrupt
            if char == " ":
                return True

    def _restore_terminal_state(self) -> None:
        if not self._raw_enabled:
            return
        import fcntl
        import termios

        assert self._stdin_fd is not None
        self._raw_enabled = False
        if self._old_termios is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_termios)
        if self._old_flags is not None:
            fcntl.fcntl(self._stdin_fd, fcntl.F_SETFL, self._old_flags)

    def _install_cleanup_guards(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._restore_terminal_state)
            self._atexit_registered = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._old_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _remove_cleanup_guards(self) -> None:
        if self._atexit_registered:
            try:
                atexit.unregister(self._restore_terminal_state)
            except Exception:
                pass
            self._atexit_registered = False
        for signum, handler in self._old_signal_handlers.items():
            signal.signal(signum, handler)
        self._old_signal_handlers.clear()

    def _handle_signal(self, signum, frame) -> None:
        self._restore_terminal_state()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))


def _compute_follow_camera_pose(robot_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_direction = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    camera_position = robot_pos - camera_direction + np.array([0.0, 0.0, 1.5], dtype=np.float64)
    return camera_position, robot_pos


def _collect_runtime_debug_snapshot(args_cli: argparse.Namespace, *, argv: list[str] | None = None) -> dict[str, object]:
    return {
        "argv": list(sys.argv if argv is None else argv),
        "env": {
            "LIVESTREAM": os.environ.get("LIVESTREAM"),
            "HEADLESS": os.environ.get("HEADLESS"),
            "ENABLE_CAMERAS": os.environ.get("ENABLE_CAMERAS"),
        },
        "args": {
            "livestream": getattr(args_cli, "livestream", None),
            "headless": getattr(args_cli, "headless", None),
            "enable_cameras": getattr(args_cli, "enable_cameras", None),
            "device": getattr(args_cli, "device", None),
            "debug_livestream": getattr(args_cli, "debug_livestream", None),
        },
    }


def _print_runtime_debug_snapshot(args_cli: argparse.Namespace) -> None:
    snapshot = _collect_runtime_debug_snapshot(args_cli)
    print("[debug-livestream] runtime launch snapshot:", flush=True)
    print(f"[debug-livestream]   argv={snapshot['argv']}", flush=True)
    print(f"[debug-livestream]   env={snapshot['env']}", flush=True)
    print(f"[debug-livestream]   args={snapshot['args']}", flush=True)
    if snapshot["args"]["livestream"] == 0 and snapshot["args"]["headless"]:
        print(
            "[debug-livestream] warning: effective livestream=0 while headless=True; "
            "WebRTC is not actually enabled in this run.",
            flush=True,
        )


@dataclass(slots=True)
class _LivestreamDebug:
    enabled: bool
    startup_marks: list[tuple[str, float]] = field(default_factory=list)
    loop_samples: list[dict[str, float]] = field(default_factory=list)
    _startup_last: float = field(default_factory=perf_counter)

    def mark_startup(self, label: str) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        self.startup_marks.append((label, now - self._startup_last))
        self._startup_last = now

    def add_loop_sample(
        self,
        *,
        policy_s: float,
        env_step_s: float,
        camera_s: float,
        total_s: float,
        timestep: int,
        step_probe: dict[str, float] | None = None,
    ) -> None:
        if not self.enabled:
            return
        sample = {
            "policy_s": policy_s,
            "env_step_s": env_step_s,
            "camera_s": camera_s,
            "total_s": total_s,
            "timestep": float(timestep),
        }
        if step_probe is not None:
            sample.update(step_probe)
        self.loop_samples.append(sample)
        if len(self.loop_samples) in {1, 10, 30}:
            self.print_loop_summary(prefix=f"[debug-livestream][sample={len(self.loop_samples)}]")

    def print_startup_summary(self) -> None:
        if not self.enabled or not self.startup_marks:
            return
        print("[debug-livestream] startup timing summary:", flush=True)
        for label, dt_s in self.startup_marks:
            print(f"[debug-livestream]   {label:<24} {dt_s * 1000.0:8.1f} ms", flush=True)

    def print_loop_summary(self, *, prefix: str = "[debug-livestream]") -> None:
        if not self.enabled or not self.loop_samples:
            return
        count = len(self.loop_samples)
        totals = {"policy_s": 0.0, "env_step_s": 0.0, "camera_s": 0.0, "total_s": 0.0}
        for sample in self.loop_samples:
            for key in totals:
                totals[key] += sample[key]
        mean_total_ms = totals["total_s"] * 1000.0 / count
        fps = 1.0 / (totals["total_s"] / count) if totals["total_s"] > 0.0 else float("inf")
        print(
            f"{prefix} mean step={mean_total_ms:0.2f} ms "
            f"(policy={totals['policy_s'] * 1000.0 / count:0.2f} ms, "
            f"env={totals['env_step_s'] * 1000.0 / count:0.2f} ms, "
            f"camera={totals['camera_s'] * 1000.0 / count:0.2f} ms) "
            f"approx_fps={fps:0.2f}",
            flush=True,
        )
        detail_keys = [
            "action_process_s",
            "action_apply_s",
            "sim_step_s",
            "sim_render_s",
            "scene_update_s",
            "obs_compute_s",
            "reward_compute_s",
            "termination_compute_s",
            "command_compute_s",
        ]
        detail_parts = []
        for key in detail_keys:
            if key in self.loop_samples[0]:
                value_ms = sum(sample.get(key, 0.0) for sample in self.loop_samples) * 1000.0 / count
                detail_parts.append(f"{key.removesuffix('_s')}={value_ms:0.2f} ms")
        if detail_parts:
            print(f"{prefix} env breakdown: " + ", ".join(detail_parts), flush=True)


@dataclass(slots=True)
class _StepProbe:
    enabled: bool
    accumulators: dict[str, float] = field(
        default_factory=lambda: {
            "action_process_s": 0.0,
            "action_apply_s": 0.0,
            "sim_step_s": 0.0,
            "sim_render_s": 0.0,
            "scene_update_s": 0.0,
            "obs_compute_s": 0.0,
            "reward_compute_s": 0.0,
            "termination_compute_s": 0.0,
            "command_compute_s": 0.0,
        }
    )

    def wrap_method(self, obj, attr_name: str, metric_key: str) -> None:
        if not self.enabled or not hasattr(obj, attr_name):
            return
        original = getattr(obj, attr_name)
        if not callable(original):
            return

        def wrapped(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.accumulators[metric_key] += perf_counter() - start

        setattr(obj, attr_name, wrapped)

    def snapshot_and_reset(self) -> dict[str, float]:
        snapshot = dict(self.accumulators)
        for key in self.accumulators:
            self.accumulators[key] = 0.0
        return snapshot


def _install_env_step_probes(base_env, *, enabled: bool) -> _StepProbe:
    probe = _StepProbe(enabled=enabled)
    if not enabled:
        return probe

    probe.wrap_method(base_env.action_manager, "process_action", "action_process_s")
    probe.wrap_method(base_env.action_manager, "apply_action", "action_apply_s")
    probe.wrap_method(base_env.sim, "step", "sim_step_s")
    probe.wrap_method(base_env.sim, "render", "sim_render_s")
    probe.wrap_method(base_env.scene, "update", "scene_update_s")
    probe.wrap_method(base_env.observation_manager, "compute", "obs_compute_s")
    probe.wrap_method(base_env.reward_manager, "compute", "reward_compute_s")
    probe.wrap_method(base_env.termination_manager, "compute", "termination_compute_s")
    probe.wrap_method(base_env.command_manager, "compute", "command_compute_s")
    return probe


def _pump_play_paused_window(base_env, *, sleep_s: float = 0.01) -> None:
    base_env.sim.render()
    if hasattr(base_env.scene, "update"):
        base_env.scene.update(float(base_env.physics_dt))
    if sleep_s > 0.0:
        sleep(float(sleep_s))


def _make_env_wrapper(env, *, gym_module, vec_env_cls, tensor_dict_cls, clip_actions: float | None = None):
    class SimpleRslRlEnvWrapper(vec_env_cls):
        """Simple wrapper for RSL-RL without PVCNN."""

        def __init__(self, env, clip_actions: float | None = None):
            self.env = env
            self.clip_actions = clip_actions
            self.num_envs = env.num_envs
            self.device = env.device
            self.max_episode_length = env.max_episode_length

            if hasattr(env, "action_manager"):
                self.num_actions = env.action_manager.total_action_dim
            else:
                self.num_actions = gym_module.spaces.flatdim(env.single_action_space)

            if clip_actions is not None:
                self.env.action_space = gym_module.spaces.Box(
                    low=-clip_actions,
                    high=clip_actions,
                    shape=(self.num_actions,),
                    dtype=env.action_space.dtype,
                )

            obs_dict, _ = self.env.reset()
            self._initial_observations = self._format_observations(obs_dict)

        @property
        def unwrapped(self):
            return self.env.unwrapped

        @property
        def cfg(self):
            return self.env.unwrapped.cfg

        @property
        def episode_length_buf(self):
            return self.env.unwrapped.episode_length_buf

        @episode_length_buf.setter
        def episode_length_buf(self, value):
            self.env.unwrapped.episode_length_buf = value

        @property
        def observation_space(self):
            return self.env.observation_space

        @property
        def action_space(self):
            return self.env.action_space

        def _flatten_group(self, obs_dict, group_names: list[str]) -> torch.Tensor:
            values = []
            for name in group_names:
                value = obs_dict[name]
                values.append(value.reshape(value.shape[0], -1))
            return torch.cat(values, dim=-1)

        def _format_observations(self, obs_dict) -> tuple[torch.Tensor, dict]:
            policy_obs = self._flatten_group(obs_dict, ["policy_elevation_semantic_map", "policy_state"])
            critic_obs = self._flatten_group(obs_dict, ["critic_elevation_semantic_map", "critic_state"])
            return policy_obs, {"observations": {"critic": critic_obs}}

        def get_observations(self):
            obs_dict = self.env.unwrapped.observation_manager.compute()
            return self._format_observations(obs_dict)

        def reset(self):
            obs_dict, _ = self.env.reset()
            return self._format_observations(obs_dict)

        def consume_initial_observations(self):
            observations = self._initial_observations
            self._initial_observations = None
            if observations is not None:
                return observations
            return self.get_observations()

        def step(self, actions):
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

            obs_dict, rewards, dones, truncated, extras = self.env.step(actions)
            dones = dones | truncated

            obs, obs_extras = self._format_observations(obs_dict)
            extras.update(obs_extras)
            return obs, rewards, dones, extras

    return SimpleRslRlEnvWrapper(env, clip_actions=clip_actions)


def _configure_reference_trajectory(env_cfg, *, use_raw_reference_trajectory: bool) -> None:
    if hasattr(env_cfg, "use_batched_reference_trajectory"):
        env_cfg.use_batched_reference_trajectory = False
        if hasattr(env_cfg, "planner_owned_reference_cache"):
            env_cfg.planner_owned_reference_cache = False
        if use_raw_reference_trajectory:
            print(
                "[play.py] Warning: --use-raw-reference-trajectory is legacy-only and is ignored; "
                "policy playback runs without MPC reference trajectory.",
                flush=True,
            )
        return

    if hasattr(env_cfg, "use_raw_reference_trajectory"):
        env_cfg.use_raw_reference_trajectory = bool(use_raw_reference_trajectory)


def _attach_reference_manager_if_enabled(env, env_cfg, experiment_name: str) -> None:
    if not getattr(env_cfg, "planner_owned_reference_cache", False):
        return

    from extension.trajectory_manager_factory import attach_trajectory_manager_if_enabled

    manager_device = getattr(env, "device", env_cfg.sim.device)
    manager = attach_trajectory_manager_if_enabled(
        env,
        env_cfg,
        experiment_name=experiment_name,
        device=manager_device,
    )
    if manager is not None:
        print(
            f"[Planner] Attached {getattr(manager, 'planner_backend', 'mpc')} trajectory manager "
            f"for {experiment_name}",
            flush=True,
        )


def main() -> int:
    args_cli = _prepare_runtime_args(_parse_args())
    debug = _LivestreamDebug(enabled=bool(args_cli.debug_livestream))
    if args_cli.debug_livestream:
        _print_runtime_debug_snapshot(args_cli)

    _, simulation_app = _launch_app(args_cli)
    debug.mark_startup("app launch")

    import gymnasium as gym

    from agent import get_train_cfg
    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY,
        TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
    )
    from tracking.parallelism_tracking_env_cfg import ParallelismTrackingFlatEnvCfg_PLAY
    import go2_pvcnn.tasks.register_envs  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.dict import print_dict
    from rsl_rl.env import VecEnv
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    debug.mark_startup("python imports")

    experiment_play_map = {
        "teacher_elevation_trajectory_mpc_semantic": (
            TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
            "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0",
        ),
        "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance": (
            TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY,
            "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-Play-v0",
        ),
        "parallelism_tracking_flat": (
            ParallelismTrackingFlatEnvCfg_PLAY,
            "Isaac-Go2-Parallelism-Tracking-Flat-v0",
        ),
    }

    experiment_name = args_cli.experiment
    env_cfg_cls, task_id = experiment_play_map[experiment_name]

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    log_dir = os.path.join(log_root_path, args_cli.run_dir)
    checkpoint_path = os.path.join(log_dir, args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n{'=' * 80}")
    print(f"Playing - {experiment_name}")
    print(f"{'=' * 80}")
    print(f"Task: {task_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Livestream mode: {getattr(args_cli, 'livestream', 0)}")
    print(f"Debug livestream: {args_cli.debug_livestream}")
    print(f"Step mode: {args_cli.step_mode}")
    print(f"Keyboard control: {args_cli.keyboard_control}")
    print(f"Initial terrain row/col: {args_cli.terrain_row}/{args_cli.terrain_col}")
    print(f"{'=' * 80}\n")

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    _configure_reference_trajectory(
        env_cfg,
        use_raw_reference_trajectory=bool(args_cli.use_raw_reference_trajectory),
    )
    env_cfg.planner_backend = (
        "parallelism" if experiment_name == "parallelism_tracking_flat" else str(args_cli.planner_backend)
    )

    render_mode = _resolve_render_mode(args_cli)
    if render_mode is not None:
        env_cfg.sim.enable_cameras = True
    if args_cli.video:
        print(f"[Video] Recording enabled (length={args_cli.video_length})", flush=True)
    debug.mark_startup("env cfg setup")

    print(f"[INFO][play.py] gym.make({task_id!r}) ... (scene build can take several minutes)", flush=True)
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    print("[INFO][play.py] gym.make done.", flush=True)
    debug.mark_startup("gym.make")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{args_cli.checkpoint.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording video during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        debug.mark_startup("video wrapper")

    assert isinstance(env.unwrapped, ManagerBasedRLEnv)
    base_env = env.unwrapped
    _apply_initial_terrain_selection(
        base_env,
        terrain_row=args_cli.terrain_row,
        terrain_col=args_cli.terrain_col,
        env_id=0,
    )
    _attach_reference_manager_if_enabled(base_env, env_cfg, experiment_name)
    step_probe = _install_env_step_probes(base_env, enabled=bool(args_cli.debug_livestream))

    is_parallelism_play = experiment_name == "parallelism_tracking_flat"
    parallelism_panel_state = None
    parallelism_manager = None
    parallelism_diagnostics = None
    parallelism_panel = None
    parallelism_visualizer = None
    if is_parallelism_play:
        from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager

        parallelism_panel_state = ParallelismPlayPanelState()
        debug_command = _parallelism_debug_command_from_env()
        if debug_command is not None:
            parallelism_panel_state.vx, parallelism_panel_state.vy, parallelism_panel_state.vyaw = debug_command
            print(
                "[Parallelism][debug] fixed panel command enabled: "
                f"vx={parallelism_panel_state.vx:+.3f} "
                f"vy={parallelism_panel_state.vy:+.3f} "
                f"vyaw={parallelism_panel_state.vyaw:+.3f}",
                flush=True,
            )
        if os.environ.get("PARALLELISM_PLAY_ALLOW_BASE_CONTACT_RESET", "").strip() == "1":
            parallelism_panel_state.suppress_termination["base_contact"] = False
        parallelism_manager = get_parallelism_reference_manager(base_env)
        parallelism_diagnostics = _install_parallelism_termination_filter(
            base_env.termination_manager,
            parallelism_panel_state,
        )
        if not bool(getattr(args_cli, "headless", False)):
            parallelism_panel = _ParallelismPlayPanel(parallelism_panel_state)
            parallelism_visualizer = _ParallelismPlayVisualizer()
        print("[Parallelism] Policy play visualization enabled.", flush=True)

    print("\n[Wrapper] Creating RSL-RL environment wrapper...", flush=True)
    wrapped_env = _make_env_wrapper(
        base_env,
        gym_module=gym,
        vec_env_cls=VecEnv,
        tensor_dict_cls=TensorDict,
        clip_actions=100.0,
    )
    debug.mark_startup("wrapper init")

    print("\n[Environment] Created successfully", flush=True)
    print(f"  - Observation space: {wrapped_env.observation_space}", flush=True)
    print(f"  - Action space: {wrapped_env.action_space}", flush=True)
    print(f"  - Device: {wrapped_env.device}", flush=True)
    print(f"  - Render mode: {render_mode}", flush=True)
    print(f"  - Render interval: {base_env.cfg.sim.render_interval}", flush=True)

    train_cfg = get_train_cfg(experiment_name)
    print("\n[Runner] Creating OnPolicyRunner...", flush=True)
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=env_cfg.sim.device)
    debug.mark_startup("runner init")

    print(f"\n[Checkpoint] Loading model from: {checkpoint_path}", flush=True)
    runner.load(checkpoint_path, load_optimizer=False)
    print("[Policy] Loaded successfully", flush=True)
    debug.mark_startup("checkpoint load")

    if args_cli.sample:
        policy = runner.alg.policy.act
    else:
        policy = runner.get_inference_policy(device=wrapped_env.device)
    print(f"[Policy] Using {'sampling' if args_cli.sample else 'inference'} mode", flush=True)

    obs, _ = wrapped_env.consume_initial_observations()
    if parallelism_visualizer is not None and parallelism_manager is not None:
        parallelism_visualizer.write_reference(base_env, parallelism_manager)
        parallelism_visualizer.update(base_env, parallelism_manager)
        if parallelism_panel is not None:
            parallelism_panel.update_joint_error(
                _parallelism_joint_error_data(base_env, _parallelism_visual_frame(parallelism_manager))
            )
    timestep = 0
    camera_interval = _livestream_camera_update_interval(getattr(args_cli, "livestream", 0))
    debug.mark_startup("first observations")
    debug.print_startup_summary()
    if args_cli.debug_livestream:
        print(
            f"[debug-livestream] camera follow interval={camera_interval} "
            f"(livestream={getattr(args_cli, 'livestream', 0)}, num_envs={args_cli.num_envs})",
            flush=True,
        )

    print(f"\n{'=' * 80}", flush=True)
    print("Starting Play Loop", flush=True)
    if args_cli.step_mode:
        print("Step mode enabled: press Space to advance one env/render step.", flush=True)
    print(f"{'=' * 80}\n", flush=True)
    print(
        "[Play][trace] loop_init "
        f"max_steps={args_cli.max_steps} "
        f"simulation_app.is_running={bool(simulation_app.is_running())}",
        flush=True,
    )

    keyboard_controller = _KeyboardVelocityController(
        enabled=bool(args_cli.keyboard_control) and not is_parallelism_play,
        linear_speed=float(args_cli.keyboard_linear_speed),
        lateral_speed=float(args_cli.keyboard_lateral_speed),
        yaw_speed=float(args_cli.keyboard_yaw_speed),
        speed_step=float(args_cli.keyboard_speed_step),
    )

    try:
        with _TerminalStepGate(enabled=bool(args_cli.step_mode)) as step_gate, keyboard_controller:
            while True:
                should_continue = _play_loop_should_continue(
                    simulation_app,
                    timestep=timestep,
                    max_steps=args_cli.max_steps,
                )
                print(
                    "[Play][trace] loop_check "
                    f"timestep={timestep} continue={should_continue} "
                    f"simulation_app.is_running={bool(simulation_app.is_running())}",
                    flush=True,
                )
                if not should_continue:
                    break
                print(f"[Play][trace] step_begin timestep={timestep + 1}", flush=True)
                if not step_gate.wait_for_step():
                    if args_cli.step_mode:
                        _pump_play_paused_window(base_env)
                    continue
                step_start = perf_counter()
                with torch.inference_mode():
                    if parallelism_panel_state is not None:
                        _apply_panel_velocity_command(base_env, parallelism_panel_state)
                    else:
                        _apply_keyboard_velocity_command(base_env, keyboard_controller)
                    print(f"[Play][trace] post_step_command_done timestep={timestep + 1}", flush=True)
                    obs, _ = wrapped_env.get_observations()
                    print(f"[Play][trace] observations_ready timestep={timestep + 1}", flush=True)
                    policy_start = perf_counter()
                    actions = policy(obs)
                    policy_s = perf_counter() - policy_start
                    print(f"[Play][trace] policy_ready timestep={timestep + 1}", flush=True)

                    if parallelism_panel_state is not None:
                        _apply_panel_velocity_command(base_env, parallelism_panel_state)
                    else:
                        _apply_keyboard_velocity_command(base_env, keyboard_controller)
                    if parallelism_visualizer is not None and parallelism_manager is not None:
                        parallelism_visualizer.write_reference(base_env, parallelism_manager)
                    env_start = perf_counter()
                    obs, rewards, dones, extras = wrapped_env.step(actions)
                    env_step_s = perf_counter() - env_start
                    print(
                        "[Play][trace] env_step_done "
                        f"timestep={timestep + 1} dones={torch.as_tensor(dones).detach().cpu().tolist()} "
                        f"elapsed={env_step_s:.4f}s",
                        flush=True,
                    )
                    if parallelism_panel_state is not None:
                        _apply_panel_velocity_command(base_env, parallelism_panel_state)
                    else:
                        _apply_keyboard_velocity_command(base_env, keyboard_controller)

                timestep += 1
                if parallelism_visualizer is not None and parallelism_manager is not None:
                    if bool(torch.as_tensor(dones).any().item()):
                        # ManagerBasedRLEnv resets inside env.step(). Rebuild the
                        # reference cache from the reset policy state, but defer
                        # writing the reference articulation until the next loop.
                        # Isaac Sim can close when an articulation is written in
                        # the same frame as its internal reset.
                        done_mask = torch.as_tensor(
                            dones,
                            dtype=torch.bool,
                            device=parallelism_manager.device,
                        )
                        parallelism_manager.reset(done_mask)
                    else:
                        print(f"[Play][trace] manager_refresh_begin timestep={timestep}", flush=True)
                        parallelism_manager.refresh()
                        print(f"[Play][trace] manager_refresh_done timestep={timestep}", flush=True)
                        print(f"[Play][trace] reference_write_begin timestep={timestep}", flush=True)
                        _write_parallelism_reference_robot(
                            base_env.scene["reference_robot"],
                            _parallelism_visual_frame(parallelism_manager),
                        )
                        print(f"[Play][trace] reference_write_done timestep={timestep}", flush=True)
                        print(f"[Play][trace] visual_sync_begin timestep={timestep}", flush=True)
                        _sync_parallelism_reference_visual_state(base_env)
                        print(f"[Play][trace] visual_sync_done timestep={timestep}", flush=True)
                    print(f"[Play][trace] reference_refresh_done timestep={timestep}", flush=True)
                if args_cli.debug_livestream and parallelism_manager is not None:
                    _parallelism_debug_snapshot(
                        base_env,
                        parallelism_manager,
                        parallelism_diagnostics,
                        dones,
                        timestep=timestep,
                    )

                if parallelism_visualizer is not None and parallelism_manager is not None:
                    parallelism_visualizer.update(base_env, parallelism_manager)
                if parallelism_panel is not None:
                    parallelism_panel.update_diagnostics(parallelism_diagnostics)
                    if parallelism_manager is not None:
                        parallelism_panel.update_joint_error(
                            _parallelism_joint_error_data(base_env, _parallelism_visual_frame(parallelism_manager))
                        )

                camera_s = 0.0
                if _should_update_follow_camera(
                    timestep=timestep,
                    num_envs=args_cli.num_envs,
                    livestream=getattr(args_cli, "livestream", 0),
                    interval=camera_interval,
                ):
                    camera_start = perf_counter()
                    robot_pos = base_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                    camera_position, target_position = _compute_follow_camera_pose(robot_pos)
                    base_env.sim.set_camera_view(camera_position, target_position)
                    camera_s = perf_counter() - camera_start

                total_s = perf_counter() - step_start
                debug.add_loop_sample(
                    policy_s=policy_s,
                    env_step_s=env_step_s,
                    camera_s=camera_s,
                    total_s=total_s,
                    timestep=timestep,
                    step_probe=step_probe.snapshot_and_reset() if args_cli.debug_livestream else None,
                )
                print(f"[Play][trace] step_complete timestep={timestep}", flush=True)

                if args_cli.video and timestep == args_cli.video_length:
                    break
                if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
                    break

    except KeyboardInterrupt:
        print("\n[Play] Interrupted by user")
    except BaseException as exc:
        import traceback

        print(
            f"\n[Play][trace] unexpected_exit type={type(exc).__name__} "
            f"message={exc!s} timestep={timestep}",
            flush=True,
        )
        traceback.print_exc()
        raise

    finally:
        print(f"\n{'=' * 80}")
        print(f"Play Complete - Timesteps: {timestep}")
        print(f"{'=' * 80}\n")
        debug.print_loop_summary(prefix="[debug-livestream][final]")
        wrapped_env.env.close()
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
