from __future__ import annotations

import sys
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
