import torch

from tracking.managers.parallelism_amp_manager import ParallelismAmpManager


def test_invalid_clears_only_selected_history_and_recovery_needs_24_frames():
    manager = ParallelismAmpManager(2, "cpu")
    valid = torch.ones(2, dtype=torch.bool)
    for _ in range(23):
        payload = manager.push_transition(torch.zeros(2, 39), torch.ones(2, 39), torch.zeros(2, 39), torch.ones(2, 39), valid)
    assert payload.amp_active.tolist() == [True, True]
    payload = manager.push_transition(torch.zeros(2, 39), torch.ones(2, 39), torch.zeros(2, 39), torch.ones(2, 39), torch.tensor([False, True]))
    assert manager.valid_count.tolist() == [0, 24]
    assert payload.amp_active.tolist() == [False, True]
    for _ in range(22):
        payload = manager.push_transition(torch.zeros(2, 39), torch.ones(2, 39), torch.zeros(2, 39), torch.ones(2, 39), valid)
    assert payload.amp_active.tolist() == [False, True]
    payload = manager.push_transition(torch.zeros(2, 39), torch.ones(2, 39), torch.zeros(2, 39), torch.ones(2, 39), valid)
    assert payload.amp_active.tolist() == [True, True]

