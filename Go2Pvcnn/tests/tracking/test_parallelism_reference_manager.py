from types import SimpleNamespace

import torch

from tracking.managers.parallelism_reference_manager import ParallelismReferenceManager


class _Scene(dict):
    pass


def _fake_env(num_envs: int = 2):
    device = "cpu"
    body_pos_w = torch.zeros(num_envs, 8, 3)
    body_pos_w[:, -4:, 2] = torch.tensor([0.11, 0.12, 0.13, 0.14])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros(num_envs, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
            joint_pos=torch.zeros(num_envs, 12),
            joint_vel=torch.zeros(num_envs, 12),
            body_pos_w=body_pos_w,
        )
    )
    scene = _Scene(robot=robot)
    command_manager = SimpleNamespace(get_command=lambda name: torch.zeros(num_envs, 3))

    def reset(*args, **kwargs):
        return args, kwargs

    return SimpleNamespace(num_envs=num_envs, device=device, scene=scene, command_manager=command_manager, reset=reset)


def _fake_env_with_scanner(num_envs: int = 2):
    env = _fake_env(num_envs)
    side = 151
    elevation = torch.arange(num_envs * side * side, dtype=torch.float32).reshape(num_envs, side, side) * 0.001
    ray_hits = torch.zeros(num_envs, side * side, 3, dtype=torch.float32)
    grid_x = torch.arange(side, dtype=torch.float32) * 0.01
    grid_y = torch.arange(side, dtype=torch.float32) * 0.01
    xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")
    ray_hits[..., 0] = xx.reshape(1, -1)
    ray_hits[..., 1] = yy.reshape(1, -1)
    ray_hits[..., 2] = elevation.reshape(num_envs, -1) + 1.0
    semantic = torch.zeros(num_envs, side, side, dtype=torch.long)
    semantic[0, 10, 20] = 1
    semantic[1, 30, 40] = 2
    valid = torch.ones(num_envs, side, side, dtype=torch.bool)
    valid[1, 0, 0] = False
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            elevation_map=elevation,
            ray_hits_w=ray_hits,
            semantic_map=semantic,
            valid_mask=valid,
        )
    )
    env.scene["semantic_height_scanner"] = scanner
    env.scene.sensors = {"semantic_height_scanner": scanner}
    return env, ray_hits[..., 2].reshape(num_envs, side, side), semantic, valid


def test_manager_replans_after_the_phase_22_to_23_transition(monkeypatch):
    env = _fake_env()
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1
        for env_id in env_ids.tolist():
            manager.joint_pos[env_id] = torch.arange(manager.horizon).view(-1, 1).repeat(1, 12)

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset(torch.tensor([0, 1]))
    assert torch.all(manager.phase == 0)
    first_plan_count = manager.plan_count.clone()
    for _ in range(22):
        manager.step()
    assert torch.equal(manager.plan_count, first_plan_count)
    manager.step()
    assert torch.all(manager.phase == 0)
    assert torch.all(manager.plan_count == first_plan_count + 1)


def test_manager_timer_is_independent_of_episode_length_buffer(monkeypatch):
    env = _fake_env()
    env.episode_length_buf = torch.tensor([100, 100], dtype=torch.long)
    manager = ParallelismReferenceManager(env, autostart=False)
    first_plan_count = torch.zeros(2, dtype=torch.long)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    first_plan_count.copy_(manager.plan_count)

    for _ in range(22):
        manager.step()
    assert torch.equal(manager.plan_count, first_plan_count)

    manager.step()
    assert torch.all(manager.plan_count == first_plan_count + 1)
    assert torch.all(manager.parallelism_step_count == 0)


def test_manager_command_change_and_timer_are_deduplicated(monkeypatch):
    env = _fake_env(num_envs=1)
    manager = ParallelismReferenceManager(env, autostart=False)
    plan_count = []

    def fake_plan(env_ids, cycle):
        plan_count.append(env_ids.tolist())
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    plan_count.clear()
    manager.parallelism_step_count[:] = 23
    manager.mark_command_changed(torch.tensor([True]))
    manager.refresh()

    assert plan_count == [[0]]
    assert manager.plan_count.tolist() == [2]
    assert manager.parallelism_step_count.tolist() == [0]


def test_manager_detects_direct_command_value_change(monkeypatch):
    env = _fake_env(num_envs=1)
    command = torch.zeros(1, 3)
    env.command_manager = SimpleNamespace(get_command=lambda _name: command)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.refresh()
    command[:, 0] = 0.5
    manager.refresh()

    assert manager.plan_count.tolist() == [2]
    assert manager.parallelism_step_count.tolist() == [0]


def test_prepare_step_reference_replans_after_23_completed_controls(monkeypatch):
    env = _fake_env(num_envs=1)
    manager = ParallelismReferenceManager(env, autostart=False)
    first_plan_count = torch.zeros(1, dtype=torch.long)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    first_plan_count.copy_(manager.plan_count)

    for _ in range(23):
        manager.prepare_step_reference()

    assert torch.equal(manager.plan_count, first_plan_count)
    assert manager.parallelism_step_count.tolist() == [23]

    manager.prepare_step_reference()
    assert manager.plan_count.tolist() == [2]
    assert manager.parallelism_step_count.tolist() == [1]


def test_manager_timer_is_per_environment(monkeypatch):
    env = _fake_env()
    manager = ParallelismReferenceManager(env, autostart=False)
    planned_env_ids = []

    def fake_plan(env_ids, cycle):
        planned_env_ids.append(env_ids.tolist())
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    planned_env_ids.clear()
    manager.parallelism_step_count[:] = torch.tensor([23, 7])
    manager.refresh()

    assert planned_env_ids == [[0]]
    assert manager.parallelism_step_count.tolist() == [0, 7]


def test_current_joint_velocity_uses_finite_difference(monkeypatch):
    env = _fake_env()
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1
        values = torch.arange(manager.horizon, dtype=torch.float32).view(-1, 1).repeat(1, 12)
        manager.joint_pos[env_ids] = values

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    vel = manager.current_joint_vel
    assert torch.allclose(vel, torch.full((2, 12), 1.0 / manager.dt))


def test_next_joint_position_is_the_target_for_the_current_action(monkeypatch):
    env = _fake_env(num_envs=1)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1
        manager.joint_pos[env_ids] = torch.arange(manager.horizon, dtype=torch.float32).view(1, -1, 1).repeat(
            int(env_ids.numel()), 1, 12
        )

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()

    assert torch.allclose(manager.current_joint_pos, torch.zeros(1, 12))
    assert torch.allclose(manager.next_joint_pos, torch.ones(1, 12))

    manager.parallelism_step_count[:] = 5
    manager.refresh()
    assert torch.allclose(manager.next_joint_pos, torch.full((1, 12), 6.0))


def test_phase_22_snapshot_uses_frame_23_then_refreshes_to_new_phase_zero(monkeypatch):
    env = _fake_env(num_envs=1)
    manager = ParallelismReferenceManager(env, autostart=False)
    plan_cycles = []

    def fake_plan(env_ids, cycle):
        plan_cycles.append(int(cycle[0]))
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1
        values = torch.arange(manager.horizon, dtype=torch.float32).view(1, -1, 1).repeat(
            int(env_ids.numel()), 1, 12
        )
        manager.joint_pos[env_ids] = values

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    plan_cycles.clear()

    manager.parallelism_step_count[:] = 22
    manager.refresh()
    manager.prepare_step_reference()
    assert manager.phase.item() == 22
    assert torch.allclose(manager.step_joint_pos, torch.full((1, 12), 23.0))
    assert torch.allclose(manager.step_joint_vel, torch.full((1, 12), 1.0 / manager.dt))

    manager.parallelism_step_count[:] = 23
    manager.refresh()
    assert manager.phase.item() == 0
    assert plan_cycles == [1]


def test_reference_root_velocity_uses_live_policy_root_frame(monkeypatch):
    env = _fake_env(num_envs=1)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1
        manager.root_pos_w[env_ids] = torch.zeros(1, manager.horizon, 3)
        manager.root_rpy_w[env_ids] = torch.zeros(1, manager.horizon, 3)
        manager.root_pos_w[env_ids, 1, 0] = manager.dt

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()

    env.scene["robot"].data.root_quat_w[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    forward = manager.current_root_lin_vel_b_policy
    assert torch.allclose(forward, torch.tensor([[1.0, 0.0, 0.0]]), atol=1.0e-5)

    half = 0.5**0.5
    env.scene["robot"].data.root_quat_w[:] = torch.tensor([[half, 0.0, 0.0, half]])
    rotated = manager.current_root_lin_vel_b_policy
    assert torch.allclose(rotated, torch.tensor([[0.0, -1.0, 0.0]]), atol=1.0e-5)


def test_reference_root_pose_uses_next_frame_in_live_policy_frame(monkeypatch):
    env = _fake_env(num_envs=1)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1
        manager.root_pos_w[env_ids] = 0.0
        manager.root_rpy_w[env_ids] = 0.0
        manager.root_pos_w[env_ids, 1, 0] = 1.0
        manager.root_rpy_w[env_ids, 1, 2] = torch.pi / 2.0

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()

    pos_b = manager.current_root_pos_b_policy
    rot_b = manager.current_root_rot_b_policy
    assert torch.allclose(pos_b, torch.tensor([[1.0, 0.0, 0.0]]), atol=1.0e-5)
    assert torch.allclose(rot_b[:, 2], torch.tensor([torch.pi / 2.0]), atol=1.0e-5)


def test_reference_root_pose_clamps_to_last_frame(monkeypatch):
    env = _fake_env(num_envs=1)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.root_pos_w[env_ids] = 0.0
        manager.root_pos_w[env_ids, -1, 1] = 0.4

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.reset()
    manager.parallelism_step_count[:] = 22
    manager.refresh()
    assert torch.allclose(manager.current_root_pos_b_policy, torch.tensor([[0.0, 0.4, 0.0]]), atol=1.0e-5)


def test_repeated_refresh_at_reset_does_not_replan(monkeypatch):
    env = _fake_env()
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.refresh()
    manager.refresh()
    manager.refresh()
    assert manager.plan_count.tolist() == [1, 1]


def test_mark_command_changed_forces_replan_on_next_refresh(monkeypatch):
    env = _fake_env()
    env.episode_length_buf = torch.tensor([7, 7], dtype=torch.long)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.refresh()

    manager.mark_command_changed(torch.tensor([True, False]))
    manager.refresh()

    assert manager.plan_count.tolist() == [2, 1]


def test_panel_speed_replan_changes_reference_root(monkeypatch):
    env = _fake_env(num_envs=1)
    command = torch.zeros(1, 3)
    env.command_manager = SimpleNamespace(get_command=lambda _name: command)
    manager = ParallelismReferenceManager(env, autostart=False)

    def fake_plan(env_ids, cycle):
        manager._cached_cycle[env_ids] = cycle
        manager._initialized[env_ids] = True
        manager.plan_count[env_ids] += 1
        command_now = manager._command(env_ids)
        manager.root_pos_w[env_ids] = 0.0
        manager.root_pos_w[env_ids, :, 0] = command_now[:, 0].unsqueeze(-1)

    monkeypatch.setattr(manager, "_plan", fake_plan)
    manager.refresh()
    assert torch.allclose(manager.root_pos_w[:, :, 0], torch.zeros(1, manager.horizon))

    command[:, 0] = 0.5
    manager.mark_command_changed(torch.tensor([True]))
    manager.refresh()

    assert torch.allclose(manager.root_pos_w[:, :, 0], torch.full((1, manager.horizon), 0.5))


def test_command_to_planner_contract_reads_latest_vx_vy_yaw():
    env = _fake_env(num_envs=2)
    command = torch.tensor([[0.35, -0.2, 0.8], [-0.4, 0.15, -0.6]])
    env.command_manager = SimpleNamespace(get_command=lambda _name: command)
    manager = ParallelismReferenceManager(env, autostart=False)

    result = manager._command(torch.tensor([1, 0]))

    assert result.shape == (2, 3)
    assert torch.allclose(result, command[[1, 0]])
    assert result.is_contiguous()


def test_command_to_planner_contract_truncates_extra_channels():
    env = _fake_env(num_envs=1)
    command = torch.tensor([[0.2, 0.1, -0.7, 99.0]])
    env.command_manager = SimpleNamespace(get_command=lambda _name: command)
    manager = ParallelismReferenceManager(env, autostart=False)

    result = manager._command(torch.tensor([0]))

    assert result.shape == (1, 3)
    assert torch.allclose(result, command[:, :3])


def test_terrain_reads_semantic_height_scanner_maps():
    env, elevation, semantic, valid = _fake_env_with_scanner()
    env.scene["robot"].data.root_pos_w[:, 0] = torch.tensor([1.0, -2.0])
    env.scene["robot"].data.root_pos_w[:, 1] = torch.tensor([0.5, 3.0])
    manager = ParallelismReferenceManager(env, autostart=False)

    terrain = manager._terrain(env.scene["robot"].data.root_pos_w)

    assert torch.equal(terrain.height_w, elevation)
    assert torch.equal(terrain.semantic_id, semantic)
    assert torch.equal(terrain.valid_mask, valid)
    assert terrain.resolution == 0.01
    assert torch.allclose(terrain.origin_w[:, 0], torch.zeros(2))
    assert torch.allclose(terrain.origin_w[:, 1], torch.zeros(2))


def test_state_uses_measured_foot_body_positions():
    env = _fake_env()
    manager = ParallelismReferenceManager(env, autostart=False)

    state = manager._state(torch.tensor([0, 1]))

    assert state.foot_pos_w is not None
    assert torch.equal(state.foot_pos_w, env.scene["robot"].data.body_pos_w[:, -4:])


def test_state_reorders_robot_joints_into_parallelism_leg_order():
    env = _fake_env()
    robot = env.scene["robot"]
    robot.joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    robot.data.joint_pos[:] = torch.arange(12, dtype=torch.float32)
    manager = ParallelismReferenceManager(env, autostart=False)

    state = manager._state(torch.tensor([0, 1]))

    expected = torch.tensor(
        [
            3.0,
            4.0,
            5.0,
            0.0,
            1.0,
            2.0,
            9.0,
            10.0,
            11.0,
            6.0,
            7.0,
            8.0,
        ]
    ).expand(2, -1)
    assert torch.equal(state.joint_pos, expected)


def test_env_reset_hook_replans_after_reset():
    env = _fake_env()
    manager = ParallelismReferenceManager(env, autostart=False)

    env.reset()

    assert int(manager.plan_count.sum().item()) == env.num_envs


def test_standstill_count_tracks_only_consecutive_failed_replans():
    env = _fake_env()
    manager = ParallelismReferenceManager(env, autostart=False)
    ids = torch.tensor([0, 1])

    manager._update_standstill_count(ids, torch.tensor([False, True]))
    assert manager.standstill_count.tolist() == [1, 0]

    manager._update_standstill_count(ids, torch.tensor([False, False]))
    assert manager.standstill_count.tolist() == [2, 1]

    manager._update_standstill_count(ids, torch.tensor([True, False]))
    assert manager.standstill_count.tolist() == [0, 2]


def test_reset_clears_standstill_count_but_command_change_does_not():
    env = _fake_env()
    manager = ParallelismReferenceManager(env, autostart=False)
    manager.standstill_count[:] = torch.tensor([2, 1])

    manager.mark_command_changed(torch.tensor([True, False]))
    assert manager.standstill_count.tolist() == [2, 1]

    manager.reset(torch.tensor([0]))
    assert manager.standstill_count.tolist() == [0, 1]


def test_internal_environment_reset_invalidates_reference_without_planning():
    env = _fake_env()
    manager = ParallelismReferenceManager(env, autostart=False)
    manager.standstill_count[:] = torch.tensor([2, 1])
    manager.standstill_latched[:] = True
    manager._initialized[:] = True
    manager._cached_cycle[:] = 3

    manager.on_environment_reset(torch.tensor([0]))

    assert manager.standstill_count.tolist() == [0, 1]
    assert manager.standstill_latched.tolist() == [False, True]
    assert manager._initialized.tolist() == [False, True]
    assert manager._cached_cycle.tolist() == [-1, 3]


def test_terrain_following_mask_uses_non_flat_terrain_names():
    env = _fake_env(num_envs=3)
    env.scene.terrain = SimpleNamespace(
        terrain_types=torch.tensor([0, 1, 2], dtype=torch.long),
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                sub_terrains={
                    "flat": object(),
                    "random_rough": object(),
                    "pyramid_stairs": object(),
                }
            )
        ),
    )
    manager = ParallelismReferenceManager(env, autostart=False)

    mask = manager._terrain_following_mask(torch.tensor([0, 1, 2]))

    assert mask.tolist() == [False, True, True]


def test_terrain_following_mask_defaults_to_flat_without_metadata():
    env = _fake_env(num_envs=2)
    manager = ParallelismReferenceManager(env, autostart=False)

    mask = manager._terrain_following_mask(torch.tensor([0, 1]))

    assert mask.tolist() == [False, False]
