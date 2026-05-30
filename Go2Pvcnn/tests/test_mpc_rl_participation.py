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
from extension.batch_mpc_planner.participation import (
    MpcReferenceParticipationCfg,
    MpcTerrainDifficultyPair,
    select_mpc_reference_envs,
)


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


def test_participation_exclude_pair_is_terrain_and_row_logic() -> None:
    terrain_types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    terrain_levels = torch.tensor([0, 3, 3, 7, 7], dtype=torch.long)
    cfg = MpcReferenceParticipationCfg(
        enabled=True,
        exclude_pairs=(MpcTerrainDifficultyPair(terrain_cols=(1,), terrain_rows=(7,)),),
        selection_mode="round_robin",
    )

    selected, next_cursor, eligible = select_mpc_reference_envs(
        num_envs=5,
        device=torch.device("cpu"),
        terrain_types=terrain_types,
        terrain_levels=terrain_levels,
        terrain_names=["flat", "stairs", "rough"],
        cfg=cfg,
        sample_count=5,
        cursor=0,
        return_eligible=True,
    )

    assert eligible.tolist() == [True, True, True, False, True]
    assert selected.tolist() == [True, True, True, False, True]
    assert next_cursor == 0


def test_participation_round_robin_wraps_inside_eligible_ids() -> None:
    terrain_types = torch.zeros(6, dtype=torch.long)
    terrain_levels = torch.zeros(6, dtype=torch.long)
    cfg = MpcReferenceParticipationCfg(enabled=True, selection_mode="round_robin")

    selected, next_cursor, eligible = select_mpc_reference_envs(
        num_envs=6,
        device=torch.device("cpu"),
        terrain_types=terrain_types,
        terrain_levels=terrain_levels,
        terrain_names=["flat"],
        cfg=cfg,
        sample_count=4,
        cursor=4,
        return_eligible=True,
    )

    assert eligible.tolist() == [True, True, True, True, True, True]
    assert selected.tolist() == [True, True, False, False, True, True]
    assert next_cursor == 2
