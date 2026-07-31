from types import SimpleNamespace

import torch

from tracking.managers.parallelism_reference_manager import ParallelismReferenceManager


class _Scene(dict):
    pass


def _fake_env(num_envs: int = 2):
    device = "cpu"
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros(num_envs, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
            joint_pos=torch.zeros(num_envs, 12),
            joint_vel=torch.zeros(num_envs, 12),
        )
    )
    scene = _Scene(robot=robot)
    command_manager = SimpleNamespace(get_command=lambda name: torch.zeros(num_envs, 3))
    return SimpleNamespace(num_envs=num_envs, device=device, scene=scene, command_manager=command_manager)


def test_manager_replans_on_reset_and_after_horizon(monkeypatch):
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
    for _ in range(23):
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
