from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.mdp.semantic_contact_rewards import filtered_contact_penalty_from_force_matrix


def test_filtered_contact_penalty_threshold_monotonic_clip_and_finite() -> None:
    force = torch.zeros((1, 1, 4, 3), dtype=torch.float32)
    force[..., 0] = torch.tensor([0.5, 2.0, 6.0, 100.0]).view(1, 1, 4)

    penalty = filtered_contact_penalty_from_force_matrix(
        force,
        force_threshold=1.0,
        force_scale=5.0,
        force_clip=1.0,
    )

    assert torch.isfinite(penalty).all()
    assert penalty.shape == (1,)
    assert penalty.item() == 1.0


def test_filtered_contact_penalty_zero_below_threshold() -> None:
    force = torch.zeros((2, 1, 3, 3), dtype=torch.float32)
    force[..., 0] = 0.5

    penalty = filtered_contact_penalty_from_force_matrix(
        force,
        force_threshold=1.0,
        force_scale=5.0,
        force_clip=1.0,
    )

    torch.testing.assert_close(penalty, torch.zeros(2))


def test_global_semantic_contact_penalty_shapes_and_body_weights() -> None:
    from extension.mdp.semantic_contact_rewards import global_semantic_contact_penalty_from_matrices

    small = torch.zeros((2, 3, 4, 3), dtype=torch.float32)
    large = torch.zeros((2, 3, 2, 3), dtype=torch.float32)
    small[0, 1, 2, 0] = 6.0
    large[1, 2, 1, 1] = 11.0

    penalty = global_semantic_contact_penalty_from_matrices(
        small,
        large,
        body_weights=(1.0, 2.0, 5.0),
        force_threshold=1.0,
        force_scale=10.0,
        force_clip=10.0,
        small_weight=1.0,
        large_weight=2.0,
    )

    torch.testing.assert_close(penalty, torch.tensor([1.0, 10.0]))


def test_semantic_global_contact_collision_reward_reads_scene_entity_cfgs() -> None:
    from extension.mdp.semantic_contact_rewards import semantic_global_contact_collision_reward

    class _Sensor:
        def __init__(self, matrix: torch.Tensor):
            self.data = SimpleNamespace(force_matrix_w=matrix)
            self.body_names = ["a", "b"]

    env = SimpleNamespace(device="cpu", num_envs=1)
    env.scene = {
        "semantic_contact_small": _Sensor(torch.zeros((1, 2, 1, 3), dtype=torch.float32)),
        "semantic_contact_large": _Sensor(torch.ones((1, 2, 1, 3), dtype=torch.float32) * 3.0),
    }

    reward = semantic_global_contact_collision_reward(
        env,
        small_sensor_cfg=SimpleNamespace(name="semantic_contact_small"),
        large_sensor_cfg=SimpleNamespace(name="semantic_contact_large"),
        body_names=("a", "b"),
        body_weights=(1.0, 1.0),
        force_threshold=1.0,
        force_scale=10.0,
        force_clip=10.0,
        small_weight=1.0,
        large_weight=2.0,
    )

    assert reward.shape == (1,)
    assert reward.item() < 0.0
