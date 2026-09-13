from __future__ import annotations

import torch

from evaluation.metrics import EpisodeAccumulator


def test_episode_accumulator_counts_each_collision_once() -> None:
    acc = EpisodeAccumulator(2, torch.device("cpu"))
    root = torch.zeros(2, 3)
    command = torch.zeros(2, 3)
    acc.update(root, command, torch.tensor([True, False]), torch.tensor([False, True]), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    records = acc.finish(torch.tensor([0, 1]))
    assert records[0]["large_collision_episode"] is True
    assert records[1]["small_collision_episode"] is True


def test_episode_accumulator_tracks_path_and_progress() -> None:
    acc = EpisodeAccumulator(1, torch.device("cpu"))
    command = torch.tensor([[1.0, 0.0, 0.0]])
    acc.update(torch.tensor([[0.0, 0.0, 0.0]]), command, torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    acc.update(torch.tensor([[1.0, 1.0, 0.0]]), command, torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    record = acc.finish(torch.tensor([0]))[0]
    assert record["traversed_path_m"] == torch.sqrt(torch.tensor(2.0)).item()
    assert record["command_progress_m"] == 1.0


def test_episode_accumulator_contact_rate_uses_valid_steps() -> None:
    acc = EpisodeAccumulator(1, torch.device("cpu"))
    root = torch.zeros(1, 3)
    command = torch.tensor([[1.0, 0.0, 0.0]])
    acc.update(root, command, torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    acc.update(root, command, torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
    acc.update(root, command, torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))

    record = acc.finish(torch.tensor([0]))[0]
    assert record["contact_step_rate"] == 0.0
