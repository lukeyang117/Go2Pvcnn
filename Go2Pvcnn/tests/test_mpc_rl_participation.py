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
from extension.batch_mpc_planner.manager import MpcTrajectoryManager
from extension.batch_mpc_planner.config import MpcPlannerCfg


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


def _make_simple_mpc_result(*, batch: int, horizon: int, offset: float = 0.0):
    root = torch.zeros((batch, horizon, 3), dtype=torch.float32)
    root[..., 0] = torch.arange(horizon, dtype=torch.float32).view(1, horizon) + float(offset)
    foot = root[:, :, None, :].expand(batch, horizon, 4, 3).clone()
    foot[..., 0] += torch.tensor([0.2, 0.2, -0.2, -0.2], dtype=torch.float32).view(1, 1, 4)
    foot[..., 1] += torch.tensor([0.1, -0.1, 0.1, -0.1], dtype=torch.float32).view(1, 1, 4)
    return SimpleNamespace(
        root_pos=root,
        root_rpy=torch.zeros((batch, horizon, 3), dtype=torch.float32),
        foot_pos=foot,
        joint_angles=torch.zeros((batch, horizon, 12), dtype=torch.float32),
        contact_state=torch.ones((batch, horizon, 4), dtype=torch.bool),
        planned_touchdown_w=foot,
        cost_total=torch.zeros(batch, dtype=torch.float32),
    )


class _FakeRobot:
    def __init__(self, *, num_envs: int) -> None:
        root_pos = torch.zeros((num_envs, 3), dtype=torch.float32)
        root_pos[:, 2] = 0.30
        foot_offsets = torch.tensor(
            [[0.2, 0.1, -0.3], [0.2, -0.1, -0.3], [-0.2, 0.1, -0.3], [-0.2, -0.1, -0.3]],
            dtype=torch.float32,
        )
        root_pos[:, 0] = torch.arange(num_envs, dtype=torch.float32)
        self.data = SimpleNamespace(
            root_pos_w=root_pos,
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(num_envs, -1),
            joint_pos=torch.zeros((num_envs, 12), dtype=torch.float32),
            body_pos_w=root_pos[:, None, :] + foot_offsets[None, :, :],
        )

    def find_bodies(self, pattern: str):
        assert pattern == ".*_foot"
        return [0, 1, 2, 3], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class _FakeScene(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _make_fake_mpc_env(*, num_envs: int, terrain_types, terrain_levels):
    ray_hits = torch.zeros((num_envs, 3, 3, 3), dtype=torch.float32)
    scanner_data = SimpleNamespace(
        ray_hits_w=ray_hits,
        semantic_map=torch.zeros((num_envs, 3, 3), dtype=torch.long),
        pos_w=torch.zeros((num_envs, 3), dtype=torch.float32),
        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(num_envs, -1),
    )
    scanner = SimpleNamespace(
        data=scanner_data,
        cfg=SimpleNamespace(pattern_cfg=SimpleNamespace(size=(1.0, 1.0))),
    )
    terrain = SimpleNamespace(
        terrain_types=torch.as_tensor(terrain_types, dtype=torch.long),
        terrain_levels=torch.as_tensor(terrain_levels, dtype=torch.long),
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains={"flat": object(), "stairs": object(), "rough": object()})),
    )
    scene = _FakeScene(
        robot=_FakeRobot(num_envs=num_envs),
        terrain=terrain,
        sensors={"semantic_height_scanner": scanner},
    )
    planner_cfg = MpcPlannerCfg()
    planner_cfg.runtime.horizon_steps = 25
    planner_cfg.runtime.replan_interval_steps = 25
    planner_cfg.runtime.dt = 0.02
    planner_cfg.runtime.max_stale_steps = 25
    planner_cfg.runtime.parallel_plan_batch_size = 2
    planner_cfg.runtime.optimize_steps = 0
    cfg = SimpleNamespace(
        planner_backend="mpc",
        reference_command_name="base_velocity",
        reference_height_scanner_name="semantic_height_scanner",
        reference_trajectory_horizon=25,
        reference_replan_interval_steps=25,
        plan_dt=0.02,
        mpc_max_stale_steps=25,
        mpc_parallel_plan_batch_size=2,
        mpc_optimize_steps=0,
        mpc_diagnostics_enabled=False,
        mpc_planner_cfg=planner_cfg,
    )
    env = SimpleNamespace(
        scene=scene,
        cfg=cfg,
        num_envs=num_envs,
        device=torch.device("cpu"),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        command_manager=SimpleNamespace(get_command=lambda _name: torch.zeros((num_envs, 3), dtype=torch.float32)),
        common_step_counter=0,
    )
    env.unwrapped = env
    return env


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


def test_mpc_manager_selects_only_participating_envs(monkeypatch) -> None:
    planned_batches: list[int] = []

    def fake_plan_segment(terrain, states, command, cfg):
        del terrain, command, cfg
        planned_batches.append(int(states.root_pos.shape[0]))
        return _make_simple_mpc_result(batch=int(states.root_pos.shape[0]), horizon=25)

    monkeypatch.setattr("extension.batch_mpc_planner.manager.plan_segment", fake_plan_segment)
    env = _make_fake_mpc_env(
        num_envs=8,
        terrain_types=[0, 0, 1, 1, 1, 2, 2, 2],
        terrain_levels=[0, 1, 7, 7, 3, 7, 2, 1],
    )
    env.cfg.mpc_planner_cfg.reference_participation.exclude_pairs = (
        MpcTerrainDifficultyPair(terrain_cols=(0,), terrain_rows=(0, 1)),
        MpcTerrainDifficultyPair(terrain_cols=(1,), terrain_rows=(7,)),
        MpcTerrainDifficultyPair(terrain_cols=(2,), terrain_rows=(7, 2, 1)),
    )
    manager = MpcTrajectoryManager(env.cfg, device=torch.device("cpu"))

    cache = manager.refresh_from_env(env)

    assert planned_batches == [1]
    assert cache.root_pos_w.shape[1] == 25
    assert int(manager.reference_reward_mask().sum().item()) == 1
