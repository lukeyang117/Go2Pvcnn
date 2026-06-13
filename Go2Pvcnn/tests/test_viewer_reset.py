from __future__ import annotations

import sys
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.batch_mpc_planner.types import MpcPlannerTerrain
from extension.batch_mpc_planner.config import MpcPlannerCfg
from extension.viz import go2_foostep_planner as viewer
from scripts import play


class _FakeRobot:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            root_lin_vel_w=torch.tensor([[0.2, 0.0, 0.1]], dtype=torch.float32),
            root_ang_vel_w=torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32),
            joint_pos=torch.full((1, 12), 0.25, dtype=torch.float32),
            joint_vel=torch.full((1, 12), 0.4, dtype=torch.float32),
            body_pos_w=torch.tensor(
                [
                    [
                        [0.2, 0.1, 0.2],
                        [0.2, -0.1, 0.2],
                        [-0.2, 0.1, 0.2],
                        [-0.2, -0.1, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ),
        )
        self.last_root_pose = None
        self.last_root_vel = None
        self.last_joint_pos = None
        self.last_joint_vel = None

    def write_root_pose_to_sim(self, root_pose):
        root_pose = torch.as_tensor(root_pose, dtype=torch.float32).clone()
        self.last_root_pose = root_pose
        self.data.root_pos_w = root_pose[:, :3]
        self.data.root_quat_w = root_pose[:, 3:7]

    def write_root_velocity_to_sim(self, root_vel):
        root_vel = torch.as_tensor(root_vel, dtype=torch.float32).clone()
        self.last_root_vel = root_vel
        self.data.root_lin_vel_w = root_vel[:, :3]
        self.data.root_ang_vel_w = root_vel[:, 3:6]

    def write_joint_state_to_sim(self, joint_pos, joint_vel):
        self.last_joint_pos = torch.as_tensor(joint_pos, dtype=torch.float32).clone()
        self.last_joint_vel = torch.as_tensor(joint_vel, dtype=torch.float32).clone()
        self.data.joint_pos = self.last_joint_pos
        self.data.joint_vel = self.last_joint_vel


class _FakeScene:
    def __init__(self, robot: _FakeRobot) -> None:
        self.robot = robot
        self.write_count = 0
        self.update_count = 0

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot

    def write_data_to_sim(self):
        self.write_count += 1

    def update(self, _dt: float):
        self.update_count += 1


class _FakeSim:
    def __init__(self) -> None:
        self.render_count = 0

    def render(self):
        self.render_count += 1


def _fake_base_env(command: torch.Tensor | None = None):
    robot = _FakeRobot()
    scene = _FakeScene(robot)
    sim = _FakeSim()
    command_manager = None
    if command is not None:
        command_manager = SimpleNamespace(get_command=lambda _name: command)
    return SimpleNamespace(scene=scene, sim=sim, physics_dt=0.02, command_manager=command_manager), robot, scene, sim


def test_viewer_zero_base_command_clears_command_tensor() -> None:
    command = torch.tensor([[0.3, -0.1, 0.4]], dtype=torch.float32)
    base_env, _, _, _ = _fake_base_env(command)

    viewer._viewer_zero_base_command(base_env)

    torch.testing.assert_close(command, torch.zeros_like(command))


def test_viewer_apply_reset_snapshot_restores_root_and_joint_state() -> None:
    base_env, robot, scene, sim = _fake_base_env()
    snapshot = viewer.ViewerResetSnapshot(
        joint_pos=torch.zeros((1, 12), dtype=torch.float32),
        joint_vel=torch.zeros((1, 12), dtype=torch.float32),
    )
    root_pos = torch.tensor([[1.0, 2.0, 0.6]], dtype=torch.float32)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    viewer._viewer_apply_joint_reset_snapshot(base_env, snapshot, root_pos_w=root_pos, root_quat_w=root_quat)

    torch.testing.assert_close(robot.data.root_pos_w, root_pos)
    torch.testing.assert_close(robot.data.joint_pos, snapshot.joint_pos)
    torch.testing.assert_close(robot.data.root_lin_vel_w, torch.zeros_like(robot.data.root_lin_vel_w))
    assert scene.write_count == 1
    assert scene.update_count == 1
    assert sim.render_count == 1


def test_viewer_main_builds_only_selected_backend_planner_cfgs(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(viewer, "_parse_args", lambda: SimpleNamespace(planner_backend="mpc", plan_dt=0.02, n_frames=25, livestream=-1))
    monkeypatch.setattr(viewer, "_prepare_runtime_args", lambda args: args)
    monkeypatch.setattr(viewer, "_launch_app", lambda _args: (None, SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(viewer, "_build_env_cfg", lambda _args: SimpleNamespace())
    monkeypatch.setattr(viewer, "_build_mpc_planner_cfg", lambda _env_cfg, args_cli=None: calls.append("mpc") or SimpleNamespace())

    class _Stop(Exception):
        pass

    def _stop_make(*_args, **_kwargs):
        raise _Stop()

    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=_stop_make))
    monkeypatch.setitem(sys.modules, "go2_pvcnn.tasks.register_envs", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.envs",
        SimpleNamespace(ManagerBasedRLEnv=object),
    )

    with pytest.raises(_Stop):
        viewer.main()

    assert calls == ["mpc"]


def test_viewer_build_env_cfg_uses_viewer_cfg_name() -> None:
    source = (GO2PVCNN_ROOT / "extension/viz/go2_foostep_planner.py").read_text(encoding="utf-8")
    build_env_cfg_source = source[source.index("def _build_env_cfg") : source.index("def _build_mpc_planner_cfg")]

    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER" in build_env_cfg_source
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY" not in build_env_cfg_source


def test_play_configure_reference_trajectory_disables_mpc() -> None:
    env_cfg = SimpleNamespace(
        use_batched_reference_trajectory=True,
        planner_owned_reference_cache=True,
    )

    play._configure_reference_trajectory(env_cfg, use_raw_reference_trajectory=False)

    assert env_cfg.use_batched_reference_trajectory is False
    assert env_cfg.planner_owned_reference_cache is False


def test_play_attach_reference_manager_skips_when_cache_disabled(monkeypatch) -> None:
    calls: list[str] = []

    def _unexpected_attach(*_args, **_kwargs):
        calls.append("attach")
        raise AssertionError("play.py should not attach MPC when planner_owned_reference_cache is false")

    monkeypatch.setattr(
        "extension.trajectory_manager_factory.attach_trajectory_manager_if_enabled",
        _unexpected_attach,
    )
    env_cfg = SimpleNamespace(planner_owned_reference_cache=False, sim=SimpleNamespace(device="cpu"))

    play._attach_reference_manager_if_enabled(SimpleNamespace(device="cpu"), env_cfg, "teacher_elevation_trajectory_mpc_semantic")

    assert calls == []


def test_play_wrapper_get_observations_returns_rsl_rl_tuple() -> None:
    compute_count = 0

    class _FakeObservationManager:
        def compute(self):
            nonlocal compute_count
            compute_count += 1
            return {
                "policy_elevation_semantic_map": torch.ones((2, 2, 2, 2), dtype=torch.float32),
                "policy_state": torch.ones((2, 3), dtype=torch.float32) * 2.0,
                "critic_elevation_semantic_map": torch.ones((2, 2, 2, 2), dtype=torch.float32) * 3.0,
                "critic_state": torch.ones((2, 4), dtype=torch.float32) * 4.0,
            }

    class _FakeEnv:
        num_envs = 2
        device = "cpu"
        max_episode_length = 10
        action_space = SimpleNamespace(dtype="float32")
        action_manager = SimpleNamespace(total_action_dim=12)
        unwrapped = SimpleNamespace(num_envs=2, cfg=SimpleNamespace(), observation_manager=_FakeObservationManager())
        episode_length_buf = torch.zeros(2, dtype=torch.long)

        def reset(self):
            return self.unwrapped.observation_manager.compute(), {}

    class _FakeVecEnv:
        pass

    class _FakeBox:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    gym_module = SimpleNamespace(spaces=SimpleNamespace(Box=_FakeBox, flatdim=lambda _space: 12))
    wrapped = play._make_env_wrapper(
        _FakeEnv(),
        gym_module=gym_module,
        vec_env_cls=_FakeVecEnv,
        tensor_dict_cls=dict,
        clip_actions=None,
    )

    obs, extras = wrapped.get_observations()

    assert obs.shape == (2, 11)
    assert extras["observations"]["critic"].shape == (2, 12)
    assert compute_count == 2


def test_play_wrapper_consumes_cached_initial_observations_without_second_reset() -> None:
    compute_count = 0

    class _FakeObservationManager:
        def compute(self):
            nonlocal compute_count
            compute_count += 1
            return {
                "policy_elevation_semantic_map": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                "policy_state": torch.ones((1, 3), dtype=torch.float32),
                "critic_elevation_semantic_map": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                "critic_state": torch.ones((1, 4), dtype=torch.float32),
            }

    class _FakeEnv:
        num_envs = 1
        device = "cpu"
        max_episode_length = 10
        action_space = SimpleNamespace(dtype="float32")
        action_manager = SimpleNamespace(total_action_dim=12)
        unwrapped = SimpleNamespace(num_envs=1, cfg=SimpleNamespace(), observation_manager=_FakeObservationManager())
        episode_length_buf = torch.zeros(1, dtype=torch.long)

        def reset(self):
            return self.unwrapped.observation_manager.compute(), {}

    class _FakeVecEnv:
        pass

    gym_module = SimpleNamespace(spaces=SimpleNamespace(Box=lambda **kwargs: kwargs, flatdim=lambda _space: 12))
    wrapped = play._make_env_wrapper(
        _FakeEnv(),
        gym_module=gym_module,
        vec_env_cls=_FakeVecEnv,
        tensor_dict_cls=dict,
        clip_actions=None,
    )

    obs, extras = wrapped.consume_initial_observations()

    assert obs.shape == (1, 11)
    assert extras["observations"]["critic"].shape == (1, 12)
    assert compute_count == 1



def test_play_main_unpacks_get_observations_result_for_policy_obs() -> None:
    source = (GO2PVCNN_ROOT / "scripts/play.py").read_text(encoding="utf-8")

    assert "obs, _ = wrapped_env.consume_initial_observations()" in source
    assert "obs, _ = wrapped_env.get_observations(), None" not in source


def test_play_loop_applies_keyboard_command_before_policy_and_step() -> None:
    source = (GO2PVCNN_ROOT / "scripts/play.py").read_text(encoding="utf-8")
    loop_source = source[source.index("while _play_loop_should_continue") :]

    first_apply = loop_source.index("_apply_keyboard_velocity_command(base_env, keyboard_controller)")
    policy_call = loop_source.index("actions = policy(obs)")
    second_apply = loop_source.index("_apply_keyboard_velocity_command(base_env, keyboard_controller)", first_apply + 1)
    env_step = loop_source.index("wrapped_env.step(actions)")

    assert first_apply < policy_call
    assert policy_call < second_apply < env_step


def test_play_cli_has_optional_max_steps_exit_for_headless_smoke() -> None:
    source = (GO2PVCNN_ROOT / "scripts/play.py").read_text(encoding="utf-8")

    assert '"--max-steps"' in source
    assert "while _play_loop_should_continue(simulation_app, timestep=timestep, max_steps=args_cli.max_steps):" in source
    assert "if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:" in source


def test_play_cli_has_keyboard_control_and_terrain_selection_flags() -> None:
    source = (GO2PVCNN_ROOT / "scripts/play.py").read_text(encoding="utf-8")

    for flag in (
        '"--keyboard-control"',
        '"--keyboard-linear-speed"',
        '"--keyboard-lateral-speed"',
        '"--keyboard-yaw-speed"',
        '"--keyboard-speed-step"',
        '"--terrain-row"',
        '"--terrain-col"',
    ):
        assert flag in source
    assert "pynput" not in source
    assert "importlib.import_module" not in source
    assert "termios.tcgetattr" in source
    assert "tty.setcbreak" in source
    assert "_KeyboardVelocityController" in source


def test_keyboard_velocity_controller_uses_terminal_reader_thread_not_pynput() -> None:
    source = (GO2PVCNN_ROOT / "scripts/play.py").read_text(encoding="utf-8")

    assert "threading.Thread(" in source
    assert "_terminal_read_loop" in source
    assert "select.select" in source
    assert "--keyboard-backend" not in source


def test_flat_small_play_cfg_disables_training_curriculum_without_semantic_contact_sensors() -> None:
    source = (GO2PVCNN_ROOT / "go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py").read_text(
        encoding="utf-8"
    )
    class_source = source[
        source.index("class TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY") :
    ]

    assert "self.curriculum.terrain_levels = None" in class_source
    assert "self.scene.semantic_contact_small = None" in class_source
    assert "self.scene.semantic_contact_large = None" in class_source
    assert 'hasattr(self.commands.base_velocity, "limit_ranges")' in class_source


def test_play_cfgs_disable_timeout_refresh_for_visualization() -> None:
    source = (GO2PVCNN_ROOT / "go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py").read_text(
        encoding="utf-8"
    )
    base_play_source = source[
        source.index("class TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY") :
        source.index("class TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER")
    ]
    flat_play_source = source[source.index("class TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY") :]

    assert "self.terminations.time_out = None" in base_play_source
    assert "self.terminations.time_out = None" in flat_play_source


def test_keyboard_velocity_controller_maps_pressed_keys_to_body_command() -> None:
    controller = play._KeyboardVelocityController(
        enabled=True,
        linear_speed=0.5,
        lateral_speed=0.25,
        yaw_speed=0.4,
        speed_step=0.1,
    )

    controller.press("w")
    controller.press("a")
    controller.press("q")
    command = controller.command_tensor(device="cpu", dtype=torch.float32, num_envs=2)

    expected = torch.tensor([[0.5, 0.25, 0.4], [0.5, 0.25, 0.4]], dtype=torch.float32)
    torch.testing.assert_close(command, expected)

    controller.release("w")
    controller.release("a")
    controller.release("q")
    torch.testing.assert_close(
        controller.command_tensor(device="cpu", dtype=torch.float32, num_envs=1),
        torch.zeros((1, 3), dtype=torch.float32),
    )


def test_keyboard_velocity_controller_speed_step_scales_and_clamps() -> None:
    controller = play._KeyboardVelocityController(
        enabled=True,
        linear_speed=0.95,
        lateral_speed=0.45,
        yaw_speed=0.95,
        speed_step=0.2,
    )

    controller.press("+")
    controller.press("w")
    controller.press("d")
    controller.press("e")

    torch.testing.assert_close(
        controller.command_tensor(device="cpu", dtype=torch.float32, num_envs=1),
        torch.tensor([[1.0, -0.5, -1.0]], dtype=torch.float32),
    )


def test_apply_keyboard_command_overwrites_base_velocity_tensor() -> None:
    command = torch.zeros((2, 3), dtype=torch.float32)
    base_env, _, _, _ = _fake_base_env(command)
    controller = play._KeyboardVelocityController(
        enabled=True,
        linear_speed=0.4,
        lateral_speed=0.2,
        yaw_speed=0.3,
        speed_step=0.1,
    )
    controller.press("s")
    controller.press("d")

    play._apply_keyboard_velocity_command(base_env, controller)

    torch.testing.assert_close(command, torch.tensor([[-0.4, -0.2, 0.0], [-0.4, -0.2, 0.0]]))


def test_apply_initial_terrain_selection_syncs_env0_row_col_and_origins() -> None:
    terrain_origins = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3)
    terrain = SimpleNamespace(
        terrain_origins=terrain_origins,
        terrain_levels=torch.zeros(2, dtype=torch.long),
        terrain_types=torch.zeros(2, dtype=torch.long),
        env_origins=torch.zeros((2, 3), dtype=torch.float32),
    )
    scene = SimpleNamespace(terrain=terrain, env_origins=torch.zeros((2, 3), dtype=torch.float32))
    base_env = SimpleNamespace(scene=scene, device="cpu", num_envs=2)

    play._apply_initial_terrain_selection(base_env, terrain_row=3, terrain_col=4, env_id=0)

    assert int(terrain.terrain_levels[0]) == 3
    assert int(terrain.terrain_types[0]) == 4
    torch.testing.assert_close(terrain.env_origins[0], terrain_origins[3, 4])
    torch.testing.assert_close(scene.env_origins[0], terrain_origins[3, 4])


def test_play_loop_should_continue_uses_max_steps_without_app_running() -> None:
    app = SimpleNamespace(is_running=lambda: False)

    assert play._play_loop_should_continue(app, timestep=0, max_steps=1)
    assert not play._play_loop_should_continue(app, timestep=1, max_steps=1)
    assert not play._play_loop_should_continue(app, timestep=0, max_steps=0)


def test_actor_critic_cnn_uses_group_cnn_cfg_for_saved_policy_shape() -> None:
    from rsl_rl.modules.actor_critic_cnn import ActorCriticCNN

    cfg = {
        "output_channels": [32, 64],
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": "zeros",
        "max_pool": [True, True],
        "activation": "elu",
        "flatten": True,
    }

    model = ActorCriticCNN(
        num_actor_obs=557,
        num_critic_obs=560,
        num_actions=12,
        actor_cnn_cfg=cfg,
        critic_cnn_cfg=cfg,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    state_keys = set(model.state_dict())
    assert "actor_cnns.policy_elevation_semantic_map.0.weight" in state_keys
    assert "critic_cnns.critic_elevation_semantic_map.3.weight" in state_keys
    assert model.actor[0].weight.shape == (256, 1069)
    assert model.critic[0].weight.shape == (256, 1072)


def test_viewer_ground_robot_from_scanner_shifts_root_z_to_match_ground(monkeypatch) -> None:
    base_env, robot, scene, sim = _fake_base_env()
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    monkeypatch.setattr(viewer, "_compute_mpc_local_terrain", lambda scanner, env_id=0: (terrain, None))

    z_shift = viewer._viewer_ground_robot_from_scanner(base_env, object(), [0, 1, 2, 3])

    assert z_shift == pytest.approx(-0.2)
    torch.testing.assert_close(robot.data.root_pos_w[:, 2], torch.tensor([0.3], dtype=torch.float32))
    torch.testing.assert_close(robot.data.root_lin_vel_w, torch.zeros_like(robot.data.root_lin_vel_w))
    assert scene.write_count == 1
    assert scene.update_count == 1
    assert sim.render_count == 1


def test_viewer_step_mode_defers_command_replan_until_current_trajectory_finishes() -> None:
    result = SimpleNamespace(num_frames=5)
    previous = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
    changed = torch.tensor([[0.0, 0.3, 0.0]], dtype=torch.float64)

    assert not viewer._viewer_loop_need_replan(
        result=result,
        playback_frame=3,
        reset_requested=False,
        teleop_values=changed,
        last_cmd=previous,
        defer_command_replan_until_trajectory_end=True,
    )
    assert viewer._viewer_loop_need_replan(
        result=result,
        playback_frame=5,
        reset_requested=False,
        teleop_values=changed,
        last_cmd=previous,
        defer_command_replan_until_trajectory_end=True,
    )


def test_viewer_step_gate_requires_space_for_each_frame() -> None:
    gate = viewer.ViewerStepGate(enabled=True)

    assert not gate.consume_frame_permission(step_requested=False)
    assert gate.consume_frame_permission(step_requested=True)
    assert not gate.consume_frame_permission(step_requested=False)


def test_viewer_step_gate_can_toggle_runtime_mode() -> None:
    gate = viewer.ViewerStepGate(enabled=False)

    assert gate.consume_frame_permission(step_requested=False)
    assert gate.toggle_enabled()
    assert not gate.consume_frame_permission(step_requested=False)
    assert gate.consume_frame_permission(step_requested=True)
    assert not gate.toggle_enabled()
    assert gate.consume_frame_permission(step_requested=False)


def test_viewer_teleop_signal_handler_removes_guards_before_interrupt(monkeypatch) -> None:
    teleop = viewer.TerminalTeleop(
        device=torch.device("cpu"),
        vx_scale=1.0,
        vy_scale=1.0,
        yaw_scale=1.0,
        timeout_s=0.1,
    )
    calls: list[str] = []
    monkeypatch.setattr(teleop, "_remove_cleanup_guards", lambda: calls.append("remove"))
    monkeypatch.setattr(teleop, "_restore_terminal_state", lambda: calls.append("restore"))

    with pytest.raises(KeyboardInterrupt):
        teleop._handle_signal(signal.SIGINT, None)

    assert calls == ["remove", "restore"]


def test_viewer_step_mode_defers_command_replan_only_while_enabled() -> None:
    result = SimpleNamespace(num_frames=5)
    previous = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
    changed = torch.tensor([[0.0, 0.3, 0.0]], dtype=torch.float64)

    assert viewer._viewer_loop_need_replan(
        result=result,
        playback_frame=3,
        reset_requested=False,
        teleop_values=changed,
        last_cmd=previous,
        defer_command_replan_until_trajectory_end=False,
    )


def test_viewer_selects_latched_command_while_step_mode_enabled() -> None:
    live = torch.zeros((1, 3), dtype=torch.float64)
    latched = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)

    selected = viewer._viewer_select_active_teleop_values(
        live_values=live,
        latched_values=latched,
        step_mode_enabled=True,
    )

    torch.testing.assert_close(selected, latched)


def test_viewer_mpc_planning_keeps_body_frame_command() -> None:
    source = Path(viewer.__file__).read_text(encoding="utf-8")
    function_start = source.index("def _plan_viewer_trajectory(")
    next_function = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:next_function]

    assert "_viewer_mpc_world_command_from_root_frame" not in function_source
    assert "plan_segment(" in function_source
    assert "command," in function_source


def test_viewer_mpc_cfg_uses_nested_planner_runtime() -> None:
    mpc_cfg = MpcPlannerCfg()
    mpc_cfg.runtime.horizon_steps = 300
    mpc_cfg.runtime.replan_interval_steps = 300
    mpc_cfg.runtime.dt = 0.03
    env_cfg = SimpleNamespace(mpc_planner_cfg=mpc_cfg)

    cfg = viewer._build_mpc_planner_cfg(env_cfg)

    assert cfg.runtime.horizon_steps == 300
    assert cfg.runtime.replan_interval_steps == 300
    assert cfg.runtime.dt == pytest.approx(0.03)


def test_play_step_gate_disabled_does_not_block() -> None:
    gate = play._TerminalStepGate(enabled=False)

    assert gate.wait_for_step()


def test_viewer_step_mode_paused_loop_keeps_rendering_window() -> None:
    base_env, _, scene, sim = _fake_base_env()

    viewer._viewer_pump_paused_window(base_env, sleep_s=0.0)

    assert sim.render_count == 1
    assert scene.update_count == 1


def test_viewer_step_mode_updates_visualizer_only_when_frame_is_permitted() -> None:
    calls = []

    def record_update() -> None:
        calls.append("update")

    viewer._viewer_update_visualizer_when_permitted(frame_permitted=False, update_fn=record_update)
    viewer._viewer_update_visualizer_when_permitted(frame_permitted=True, update_fn=record_update)

    assert calls == ["update"]
