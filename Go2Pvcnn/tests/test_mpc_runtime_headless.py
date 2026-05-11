from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch


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


def _runtime_device_override() -> str | None:
    value = os.environ.get("MPC_TEST_DEVICE", "").strip()
    return value or None


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
    kwargs = {"num_envs": 4096, "planner_backend": "mpc", "warmup_steps": 2}
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
    assert float(actual["rpy_if_wxyz"][0, :2].abs().max().item()) < 1.0e-3


def test_mpc_runtime_command_matrix_tracks_motion_and_limits_drift(real_semantic_mpc_runtime):
    runtime = real_semantic_mpc_runtime
    viewer = runtime._viewer
    terrain = runtime._single_env_terrain()

    command_names = (
        "forward",
        "backward",
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
    )

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


def test_mpc_runtime_4096_headless_dirty_budget_counters(real_semantic_mpc_runtime_4096):
    runtime = real_semantic_mpc_runtime_4096
    manager = runtime.base_env._trajectory_manager

    runtime.mpc_planner_cfg.diagnostics.emit_runtime_counters = True
    runtime.mpc_planner_cfg.diagnostics.profile_cuda_sync = False
    runtime.mpc_planner_cfg.runtime.optimize_steps = 0
    runtime.mpc_planner_cfg.runtime.max_dirty_envs_per_step = 64
    runtime.mpc_planner_cfg.runtime.max_stale_steps = 100

    runtime.base_env.common_step_counter = int(getattr(runtime.base_env, "common_step_counter", 0)) + 1
    manager.refresh_from_env(runtime.base_env)
    first = manager.runtime_counters()
    print("MPC_4096_COUNTERS_FIRST", first, flush=True)

    assert first["num_envs"] == 4096
    assert first["dirty_count"] >= first["selected_dirty_count"] >= 0
    assert first["selected_dirty_count"] <= 64
    assert first["dirty_backlog"] == first["dirty_count"] - first["selected_dirty_count"]
    assert first["max_stale_observed"] >= 0
    assert first["planner_ms"] >= 0.0
    assert first["cache_ms"] >= 0.0

    command_dirty_mask = torch.zeros((4096,), dtype=torch.bool, device=runtime.base_env.device)
    command_dirty_mask[:128] = True
    manager.mark_command_changed(command_dirty_mask)
    runtime.base_env.common_step_counter = int(getattr(runtime.base_env, "common_step_counter", 0)) + 1
    manager.refresh_from_env(runtime.base_env)
    second = manager.runtime_counters()
    print("MPC_4096_COUNTERS_SECOND", second, flush=True)

    assert second["selected_dirty_count"] <= 64
    assert second["dirty_count"] >= second["selected_dirty_count"]
    assert second["dirty_backlog"] == second["dirty_count"] - second["selected_dirty_count"]
