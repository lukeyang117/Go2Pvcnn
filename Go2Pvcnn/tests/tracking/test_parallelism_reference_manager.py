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

    env.episode_length_buf = torch.tensor([5])
    assert torch.allclose(manager.next_joint_pos, torch.full((1, 12), 6.0))


def test_phase_22_snapshot_uses_frame_23_then_refreshes_to_new_phase_zero(monkeypatch):
    env = _fake_env(num_envs=1)
    env.episode_length_buf = torch.tensor([22])
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

    manager.prepare_step_reference()
    assert manager.phase.item() == 22
    assert torch.allclose(manager.step_joint_pos, torch.full((1, 12), 23.0))
    assert torch.allclose(manager.step_joint_vel, torch.full((1, 12), 1.0 / manager.dt))

    env.episode_length_buf[:] = 23
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
