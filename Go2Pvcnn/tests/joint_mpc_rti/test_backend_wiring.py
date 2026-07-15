from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch


def test_task_cfg_declares_joint_backend_config_without_changing_default() -> None:
    source = Path("Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py").read_text()

    assert 'planner_backend: str = "mpc"' in source
    assert "joint_mpc_rti_cfg" in source
    assert "JointMpcRtiCfg" in source


def test_factory_creates_joint_mpc_rti_manager() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.trajectory_manager_factory import create_trajectory_manager

    cfg = SimpleNamespace(planner_backend="joint_mpc_rti", joint_mpc_rti_cfg=JointMpcRtiCfg())
    manager = create_trajectory_manager(cfg, device="cpu")

    assert manager.planner_backend == "joint_mpc_rti"
    assert manager.horizon_steps() == 16


def test_reference_adapter_preserves_full_horizon_and_future_frame_one() -> None:
    from extension.joint_mpc_rti.integration.reference_adapter import trajectory_to_reference_cache
    from extension.joint_mpc_rti.types import JointMpcRtiTrajectory

    state = torch.zeros(2, 17, 18)
    state[:, 1, 0] = torch.tensor([0.1, -0.2])
    state[:, 1, 6:] = 0.3
    foot = torch.zeros(2, 17, 4, 3)
    contact = torch.ones(2, 17, 4, dtype=torch.bool)
    trajectory = JointMpcRtiTrajectory(
        state=state,
        control=torch.zeros(2, 16, 18),
        foot_pos_w=foot,
        contact_state=contact,
        valid=torch.ones(2, dtype=torch.bool),
        fallback=torch.zeros(2, dtype=torch.bool),
        status=torch.zeros(2, dtype=torch.long),
    )

    cache = trajectory_to_reference_cache(trajectory)

    assert cache.horizon_length() == 17
    torch.testing.assert_close(cache.root_pos_w[:, 1], state[:, 1, :3])
    torch.testing.assert_close(cache.joint_angles[:, 1], state[:, 1, 6:])
    assert torch.equal(cache.phase_index[:, 1], torch.ones(2, dtype=torch.long))


def test_reward_frame_selection_uses_manager_frame_one_for_joint_backend() -> None:
    from extension.joint_mpc_rti.integration.reference_adapter import trajectory_to_reference_cache
    from extension.joint_mpc_rti.types import JointMpcRtiTrajectory
    from extension.mdp.rewards_reference import _select_reference_frame

    state = torch.zeros(2, 17, 18)
    cache = trajectory_to_reference_cache(
        JointMpcRtiTrajectory(
            state=state,
            control=torch.zeros(2, 16, 18),
            foot_pos_w=torch.zeros(2, 17, 4, 3),
            contact_state=torch.ones(2, 17, 4, dtype=torch.bool),
            valid=torch.ones(2, dtype=torch.bool),
            fallback=torch.zeros(2, dtype=torch.bool),
            status=torch.zeros(2, dtype=torch.long),
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
    assert cache.horizon_length() == 17
    assert torch.equal(manager.current_frame_ids(), torch.ones(batch, dtype=torch.long))
