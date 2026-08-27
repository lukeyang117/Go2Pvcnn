import torch

from tracking.managers.parallelism_amp_manager import ParallelismAmpManager


def _state(value: float, batch: int = 3) -> torch.Tensor:
    return torch.full((batch, 39), value, dtype=torch.float32)


def test_push_is_batched_and_wraps_after_23_transitions():
    manager = ParallelismAmpManager(3, "cpu")
    valid = torch.ones(3, dtype=torch.bool)
    for step in range(50):
        payload = manager.push_transition(_state(float(step)), _state(float(step + 1)), _state(float(step)), _state(float(step + 1)), valid)
    assert payload.agent_window.shape == (3, 24, 39)
    assert torch.equal(manager.valid_count, torch.full((3,), 24, dtype=torch.long))
    assert torch.allclose(manager.agent_terminal, _state(50.0))


def test_reset_clears_only_selected_rows():
    manager = ParallelismAmpManager(2, "cpu")
    valid = torch.ones(2, dtype=torch.bool)
    for _ in range(23):
        manager.push_transition(_state(0.0, 2), _state(1.0, 2), _state(0.0, 2), _state(1.0, 2), valid)
    manager.reset(torch.tensor([True, False]))
    assert manager.valid_count.tolist() == [0, 24]
