from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


def test_final_planner_has_no_post_qp_root_direction_repair() -> None:
    source = Path("Go2Pvcnn/extension/joint_mpc_rti/planner.py").read_text()

    assert "_root_direction_limits" not in source
    assert "recover_control_direction" not in source
    assert "restore_candidate" not in source


def test_state_from_env_reorders_robot_joint_position_and_velocity_into_planner_order() -> None:
    from extension.joint_mpc_rti.integration.isaaclab_adapter import state_from_env

    robot_order = (
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
    )
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    robot = SimpleNamespace(
        joint_names=robot_order,
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.32]]),
            root_quat_w=root_quat,
            root_lin_vel_b=torch.zeros(1, 3),
            root_ang_vel_b=torch.zeros(1, 3),
            joint_pos=torch.arange(12, dtype=torch.float32).view(1, 12),
            joint_vel=(100.0 + torch.arange(12, dtype=torch.float32)).view(1, 12),
        ),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": robot}))

    state = state_from_env(env, device="cpu")

    expected_indices = torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
    torch.testing.assert_close(state.joint_pos, robot.data.joint_pos.index_select(-1, expected_indices))
    torch.testing.assert_close(state.joint_vel, robot.data.joint_vel.index_select(-1, expected_indices))


def test_task_cfg_declares_joint_backend_config_without_changing_default() -> None:
    source = Path("Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py").read_text()

    assert 'planner_backend: str = "mpc"' in source
    assert "joint_mpc_rti_cfg" in source
    assert "JointMpcRtiCfg" in source


def test_task_joint_backend_uses_verified_realtime_solver_profile() -> None:
    source = Path("Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py").read_text()

    assert "self.joint_mpc_rti_cfg.solver.compile_kernels = True" in source
    assert "self.joint_mpc_rti_cfg.solver.line_search_alphas = (1.0, 0.5, 0.25, 0.125, 0.0)" in source
    assert "self.joint_mpc_rti_cfg.solver.use_cuda_graph = True" in source
    assert "emit_loss_breakdown" not in source
    assert "diagonal_state_riccati" not in source


def test_viewer_reproduction_uses_only_axis_isolated_commands() -> None:
    source = Path(
        "Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_viewer_reproduction_probe.py"
    ).read_text()

    assert '("mixed",' not in source
    assert '("mixed_reverse",' not in source
    assert '("yaw_left", (0.0, 0.0, 0.5))' in source
    assert '("yaw_right", (0.0, 0.0, -0.5))' in source
    assert '("small_forward", (1.0, 0.0, 0.0))' in source
    assert '("small_backward", (-1.0, 0.0, 0.0))' in source
    assert '("small_lateral_left", (0.0, 0.5, 0.0))' in source
    assert '("small_lateral_right", (0.0, -0.5, 0.0))' in source


def test_viewer_reproduction_preserves_fixed_h30_solver_contract() -> None:
    source = Path(
        "Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_viewer_reproduction_probe.py"
    ).read_text()

    assert "requested_n_frames=30" in source
    assert 'os.environ.get("JOINT_MPC_VIEWER_REPRO_WARMUP_STEPS", "0")' in source
    assert "_viewer_ground_robot_from_scanner" in source
    assert "runtime.foot_ids.index_select(0, stance_leg_indices)" in source
    assert "manager._cfg.solver.use_cuda_graph = False" in source


def test_viewer_small_reproduction_uses_real_anchor_and_shared_acceptance_metrics() -> None:
    source = Path(
        "Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_viewer_reproduction_probe.py"
    ).read_text()

    assert 'os.environ.get("JOINT_MPC_VIEWER_REPRO_SCENARIO", "flat")' in source
    assert 's4_semantic_course_anchor("small")' in source
    assert "manager.latest_trajectory()" in source
    assert "measured_before_vector = measured_before.as_vector().clone()" in source
    assert "_small_detector_row" in source
    assert "strict_crossing_event" in source
    assert "evaluate_trace" in source
    assert 'scenario="small"' in source
    assert "post_cross_cycles = 24" in source
    assert '"worst_stance_event"' in source
    assert '"loss_breakdown"' in source
    assert '"solver_layers"' in source


def test_root_tracking_solver_layers_separate_nominal_full_selected_and_actual() -> None:
    from .joint_mpc_rti_viewer_reproduction_probe import _root_tracking_solver_layers

    state = torch.zeros(1, 3, 18)
    state[0, 1, 0] = 0.016
    state[0, 2, 0] = 0.036
    nominal = torch.zeros(1, 2, 18)
    nominal[0, 1, 0] = 0.009
    direction = torch.zeros_like(nominal)
    direction[0, 1, 0] = 0.010
    diagnostics = [
        SimpleNamespace(nominal_state=nominal.clone(), qp_direction=direction.clone()),
        SimpleNamespace(nominal_state=nominal.clone(), qp_direction=direction.clone()),
    ]

    layers = _root_tracking_solver_layers(
        state=state,
        step_diagnostics_rows=diagnostics,
        command_body=torch.tensor([[1.0, 0.0, 0.0]]),
        line_alpha=torch.tensor([[0.5, 1.0]]),
        warm_start=torch.tensor([[False, True]]),
        dt=0.02,
        root_position_trust=0.01,
    )

    assert layers["mean_error_mps"] == pytest.approx(
        {"actual": 0.1, "nominal": 0.55, "full_qp": 0.05, "selected": 0.175}
    )
    assert layers["warm_mean_error_mps"]["selected"] == pytest.approx(0.05)
    assert layers["root_xy_trust_utilization"]["max"] == pytest.approx(1.0)
    assert layers["root_xy_trust_utilization"]["saturated_cycle_count"] == 2
    assert layers["published_root_xy_deviation_m"] == pytest.approx(
        {"full_qp_max": 0.01, "selected_max": 0.01}
    )
    assert layers["published_root_xy_violation_count"] == {"full_qp": 2, "selected": 2}
    assert layers["cycles"][0]["velocity_body_mps"]["selected"] == pytest.approx([0.7, 0.0])


def test_viewer_small_reproduction_prefers_representative_sphere_then_cuboid() -> None:
    from .joint_mpc_rti_viewer_reproduction_probe import _select_small_anchor

    anchors = (
        SimpleNamespace(stage="S4", semantic_class="small", shape_kind="capsule", world_xy=(-3.0, 0.0)),
        SimpleNamespace(stage="S4", semantic_class="small", shape_kind="cuboid", world_xy=(1.0, 0.0)),
        SimpleNamespace(stage="S4", semantic_class="small", shape_kind="sphere", world_xy=(2.0, 0.0)),
    )

    assert _select_small_anchor(anchors).shape_kind == "sphere"
    assert _select_small_anchor(anchors[0:2]).shape_kind == "cuboid"


def test_factory_creates_joint_mpc_rti_manager() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.trajectory_manager_factory import create_trajectory_manager

    cfg = SimpleNamespace(planner_backend="joint_mpc_rti", joint_mpc_rti_cfg=JointMpcRtiCfg())
    manager = create_trajectory_manager(cfg, device="cpu")

    assert manager.planner_backend == "joint_mpc_rti"
    assert manager.horizon_steps() == 30


@pytest.mark.parametrize("num_envs", (1, 40, 512, 1024))
def test_factory_builds_joint_mpc_for_requested_environment_count(num_envs: int) -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.trajectory_manager_factory import create_trajectory_manager

    cfg = SimpleNamespace(planner_backend="joint_mpc_rti", joint_mpc_rti_cfg=JointMpcRtiCfg())
    manager = create_trajectory_manager(cfg, device="cpu", num_envs=num_envs)

    assert manager._num_envs == num_envs
    assert manager.pending_valid.shape == (num_envs,)


def test_attach_factory_infers_environment_count_from_env() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.trajectory_manager_factory import attach_trajectory_manager

    cfg = SimpleNamespace(
        planner_backend="joint_mpc_rti",
        joint_mpc_rti_cfg=JointMpcRtiCfg(),
        reference_command_name="base_velocity",
    )
    root = SimpleNamespace(num_envs=512, device=torch.device("cpu"), command_manager=None)
    env = SimpleNamespace(unwrapped=root, device=torch.device("cpu"))

    manager = attach_trajectory_manager(env, cfg)

    assert manager._num_envs == 512
    assert manager.pending_valid.shape == (512,)


def test_reference_adapter_preserves_full_horizon_and_future_frame_one() -> None:
    from extension.joint_mpc_rti.integration.reference_adapter import trajectory_to_reference_cache
    from extension.joint_mpc_rti.types import JointMpcRtiTrajectory

    state = torch.zeros(2, 31, 18)
    state[:, 1, 0] = torch.tensor([0.1, -0.2])
    state[:, 1, 6:] = 0.3
    foot = torch.zeros(2, 31, 4, 3)
    contact = torch.ones(2, 31, 4, dtype=torch.bool)
    trajectory = JointMpcRtiTrajectory(
        state=state,
        derived_velocity=torch.zeros(2, 30, 18),
        foot_pos_w=foot,
        contact_state=contact,
        valid=torch.ones(2, dtype=torch.bool),
        fallback=torch.zeros(2, dtype=torch.bool),
        status=torch.zeros(2, dtype=torch.long),
        line_search_alpha=torch.zeros(2),
    )

    cache = trajectory_to_reference_cache(trajectory)

    assert cache.horizon_length() == 31
    torch.testing.assert_close(cache.root_pos_w[:, 1], state[:, 1, :3])
    torch.testing.assert_close(cache.joint_angles[:, 1], state[:, 1, 6:])
    assert torch.equal(cache.phase_index[:, 1], torch.ones(2, dtype=torch.long))


def test_reward_frame_selection_uses_manager_frame_one_for_joint_backend() -> None:
    from extension.joint_mpc_rti.integration.reference_adapter import trajectory_to_reference_cache
    from extension.joint_mpc_rti.types import JointMpcRtiTrajectory
    from extension.mdp.rewards_reference import _select_reference_frame

    state = torch.zeros(2, 31, 18)
    cache = trajectory_to_reference_cache(
        JointMpcRtiTrajectory(
            state=state,
            derived_velocity=torch.zeros(2, 30, 18),
            foot_pos_w=torch.zeros(2, 31, 4, 3),
            contact_state=torch.ones(2, 31, 4, dtype=torch.bool),
            valid=torch.ones(2, dtype=torch.bool),
            fallback=torch.zeros(2, dtype=torch.bool),
            status=torch.zeros(2, dtype=torch.long),
            line_search_alpha=torch.zeros(2),
        )
    )
    manager = SimpleNamespace(
        refresh_from_env=lambda env: cache,
        current_frame_ids=lambda: torch.ones(2, dtype=torch.long),
    )
    root = SimpleNamespace(_trajectory_manager=manager, _trajectory_reference_cache=None)
    env = SimpleNamespace(
        unwrapped=root,
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.zeros(2, dtype=torch.long),
    )

    selected_cache, frame_ids = _select_reference_frame(env)

    assert selected_cache is cache
    assert torch.equal(frame_ids, torch.ones(2, dtype=torch.long))


def test_manager_refresh_from_env_builds_ready_reference_cache() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.runtime.manager import JointMpcRtiManager

    batch = 2
    root_quat = torch.zeros(batch, 4)
    root_quat[:, 0] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.32], [0.2, 0.0, 0.32]]),
            root_quat_w=root_quat,
            root_lin_vel_b=torch.zeros(batch, 3),
            root_ang_vel_b=torch.zeros(batch, 3),
            joint_pos=torch.tensor([[0.0, 0.8, -1.5] * 4] * batch),
            joint_vel=torch.zeros(batch, 12),
        )
    )
    ray_hits = torch.zeros(batch, 151 * 151, 3)
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=ray_hits,
            semantic_map=torch.zeros(batch, 151, 151, dtype=torch.long),
            pos_w=torch.zeros(batch, 3),
            quat_w=root_quat,
        )
    )
    scene = {"robot": robot, "semantic_height_scanner": scanner}
    root = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[0.2, 0.0, 0.0]] * batch)),
        num_envs=batch,
        device=torch.device("cpu"),
    )
    env = SimpleNamespace(unwrapped=root, num_envs=batch, device=torch.device("cpu"))
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=batch, device="cpu")

    cache = manager.refresh_from_env(env)

    assert cache.is_ready()
    assert cache.horizon_length() == 31
    assert torch.equal(manager.current_frame_ids(), torch.ones(batch, dtype=torch.long))
    assert manager._field_sync is not None
    assert scanner._joint_mpc_field_observer is manager._field_sync
    assert torch.equal(manager._field_sync.latest_field().version, torch.zeros(batch, dtype=torch.long))
    assert torch.equal(
        manager._field_sync.latest_perceptive_field().refresh_id,
        torch.zeros(batch, dtype=torch.long),
    )
    assert manager.latest_trajectory().state.shape == (batch, 31, 18)


def test_viewer_cli_accepts_joint_mpc_rti() -> None:
    path = Path("Go2Pvcnn/extension/viz/go2_foostep_planner.py")
    spec = importlib.util.spec_from_file_location("go2_footstep_viewer_for_joint_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    args = module._parse_args(["--planner-backend", "joint_mpc_rti"])

    assert args.planner_backend == "joint_mpc_rti"
    assert module._viewer_playback_frame_index("joint_mpc_rti", playback_frame=9) == 1
    assert module._viewer_playback_frame_index("mpc", playback_frame=9) == 9

    source = path.read_text()
    assert "env_cfg.planner_backend = str(args_cli.planner_backend)" in source
    assert "command_body=command" in source
    assert "force=True" in source


def test_joint_viewer_applies_only_first_future_frame() -> None:
    from extension.joint_mpc_rti.integration.viewer_adapter import JointMpcRtiViewerAdapter

    trajectory = SimpleNamespace(state=torch.zeros(1, 31, 18))
    adapter = JointMpcRtiViewerAdapter.for_test(trajectory)

    assert adapter.next_playback_frame().frame_index == 1
    assert adapter.next_playback_frame().frame_index == 1
