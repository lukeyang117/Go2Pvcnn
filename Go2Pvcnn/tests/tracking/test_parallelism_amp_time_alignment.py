import torch

from tracking.managers.parallelism_amp_manager import ParallelismAmpManager


def test_replan_boundary_uses_new_plan_start():
    manager = ParallelismAmpManager(1, "cpu")
    zeros = torch.zeros(1, 39)
    ones = torch.ones(1, 39)
    valid = torch.ones(1, dtype=torch.bool)
    for _ in range(22):
        manager.push_transition(zeros, ones, zeros, ones, valid)
    a23 = torch.full((1, 39), 9.0)
    b0 = torch.full((1, 39), 100.0)
    b1 = torch.full((1, 39), 101.0)
    payload = manager.push_transition(a23, b1, b0, b1, valid)
    assert torch.allclose(manager.expert_terminal, b1)
    assert torch.allclose(manager.last_expert_delta, torch.ones(1, 39))


def test_successful_replan_does_not_clear_history():
    manager = ParallelismAmpManager(1, "cpu")
    valid = torch.ones(1, dtype=torch.bool)
    for _ in range(23):
        manager.push_transition(torch.zeros(1, 39), torch.ones(1, 39), torch.zeros(1, 39), torch.ones(1, 39), valid)
    before = manager.valid_count.clone()
    manager.push_transition(torch.ones(1, 39), torch.full((1, 39), 2.0), torch.ones(1, 39), torch.full((1, 39), 2.0), valid)
    assert before.item() == 24
    assert manager.valid_count.item() == 24
