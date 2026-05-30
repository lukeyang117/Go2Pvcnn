from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.mdp.rewards_reference import reference_foot_pos_reward


class _FakeManager:
    planner_backend = "mpc"

    def __init__(self, cache, mask: torch.Tensor, frame_ids: torch.Tensor) -> None:
        self._cache = cache
        self._mask = mask
        self._frame_ids = frame_ids
        self.refresh_count = 0

    def refresh_from_env(self, env):
        del env
        self.refresh_count += 1
        return self._cache

    def reference_reward_mask(self) -> torch.Tensor:
        return self._mask

    def current_frame_ids(self) -> torch.Tensor:
        return self._frame_ids


def test_reference_foot_pos_reward_uses_world_feet_and_manager_phase() -> None:
    current = torch.zeros((2, 4, 3), dtype=torch.float32)
    ref = current.clone()
    ref[1] += 1.0
    cache = SimpleNamespace(
        foot_pos_w=torch.stack((ref, ref + 10.0), dim=1),
        root_pos_w=torch.zeros((2, 2, 3), dtype=torch.float32),
        is_ready=lambda: True,
        horizon_length=lambda: 2,
    )
    manager = _FakeManager(cache, torch.tensor([1.0, 0.0]), torch.tensor([0, 0]))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=current,
            root_pos_w=torch.zeros((2, 3), dtype=torch.float32),
            root_quat_w=torch.zeros((2, 4), dtype=torch.float32),
        )
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(_trajectory_manager=manager),
        scene={"robot": robot},
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.tensor([1, 1]),
        cfg=SimpleNamespace(reference_trajectory_horizon=2),
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3])

    reward = reference_foot_pos_reward(env, sigma=0.5, asset_cfg=asset_cfg)

    torch.testing.assert_close(reward[0], torch.tensor(1.0))
    torch.testing.assert_close(reward[1], torch.tensor(0.0))
